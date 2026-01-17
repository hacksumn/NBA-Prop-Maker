"""
NBA PLAYER PROPS - COMPLETE PIPELINE (LEAKAGE-SAFE)
Downloads data, builds features, trains models with STRICT leakage prevention

Usage:
    python full_pipeline.py
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


# =============================================================================
# STRICT COLUMN DEFINITIONS - SINGLE SOURCE OF TRUTH
# =============================================================================

# Columns that identify the row (never features)
ID_COLS = {
    'player', 'tm', 'opp', 'game_date', 'game_id', 'matchup_id', 'matchup_raw', 'res'
}

# Current game stats - THESE ARE TARGETS OR LEAKY (never features)
# If it happens DURING the game being predicted, it's leaky
CURRENT_GAME_COLS = {
    # Box score stats
    'pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf',
    'mp',  # minutes played
    'fg', 'fga', 'fg_pct',  # field goals
    '3p', '3pa', '3p_pct',  # three pointers
    'ft', 'fta', 'ft_pct',  # free throws
    'orb', 'drb',  # offensive/defensive rebounds
    'gmsc',  # game score
    'won', 'is_home',  # game outcome/context
    # Combo targets
    'pra', 'pr', 'pa',
    # NBA API columns that leak current game info
    'FANTASY_PTS', 'fantasy_pts', 'PLUS_MINUS', 'plus_minus',
    'MIN', 'min',  # minutes in different formats
}

# Line proxies - used for betting simulation, NOT as features
# Including these as features would make the task trivially easy
LINE_PROXY_COLS = {
    'pts_l10', 'trb_l10', 'ast_l10', 'stl_l10', 'blk_l10', 'tov_l10',
    'pra_l10', 'pr_l10', 'pa_l10',
}

# All columns that should NEVER be features
EXCLUDED_FROM_FEATURES = ID_COLS | CURRENT_GAME_COLS | LINE_PROXY_COLS


def get_valid_features(df: pd.DataFrame) -> list:
    """
    Get ONLY valid pre-tip-off features.
    This is the SINGLE source of truth for feature selection.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    valid = []
    for col in numeric_cols:
        # Skip if explicitly excluded
        if col in EXCLUDED_FROM_FEATURES:
            continue
        # Skip if it's a line proxy we might have missed
        if col.endswith('_l10') and col.replace('_l10', '') in CURRENT_GAME_COLS:
            continue
        valid.append(col)
    
    return sorted(valid)


def validate_no_leakage(feature_cols: list, target: str) -> bool:
    """
    STRICT validation that features don't leak target information.
    Returns True if safe, raises error if leakage detected.
    """
    issues = []
    
    # Check for current game stats
    for col in feature_cols:
        if col in CURRENT_GAME_COLS:
            issues.append(f"LEAKAGE: Current game stat '{col}' in features")
        if col in ID_COLS:
            issues.append(f"LEAKAGE: Identifier '{col}' in features")
        # Check for direct target proxy
        if col == f'{target}_l10':
            issues.append(f"LEAKAGE: Direct target L10 '{col}' in features")
    
    if issues:
        print("\n*** LEAKAGE DETECTED ***")
        for issue in issues:
            print(f"  {issue}")
        raise ValueError("Leakage detected - fix before training!")
    
    return True


def validate_predictions_reasonable(y_true, y_pred, target: str) -> bool:
    """
    Check if predictions are reasonable (not suspiciously good).
    """
    r2 = r2_score(y_true, y_pred)
    
    # Expected R2 ranges by prop type
    max_r2 = {
        'pts': 0.70, 'trb': 0.65, 'ast': 0.65,
        'stl': 0.45, 'blk': 0.50, 'tov': 0.55,
        'pra': 0.75, 'pr': 0.70, 'pa': 0.70,
    }.get(target, 0.70)
    
    if r2 > max_r2:
        print(f"\n*** WARNING: R2={r2:.3f} for {target} exceeds expected max {max_r2} ***")
        print(f"    This may indicate data leakage!")
        return False
    
    return True


# =============================================================================
# DATA PIPELINE
# =============================================================================

def step1_download_data():
    """Download 2025-26 season data from NBA.com"""
    print("\n" + "="*60)
    print("STEP 1: DOWNLOADING 2025-26 DATA")
    print("="*60)
    
    from nba_api.stats.endpoints import leaguegamelog
    
    print("Fetching from NBA.com (30-60 seconds)...")
    game_log = leaguegamelog.LeagueGameLog(
        season='2025-26',
        season_type_all_star='Regular Season',
        player_or_team_abbreviation='P',
        timeout=120
    )
    df = game_log.get_data_frames()[0]
    print(f"  Fetched {len(df):,} records")
    
    # Drop unnecessary columns from NBA API
    drop_cols = ['SEASON_ID', 'PLAYER_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 
                 'VIDEO_AVAILABLE', 'MIN_SEC', 'NBA_FANTASY_PTS', 
                 'DD2', 'TD3', 'WNBA_FANTASY_PTS',
                 # CRITICAL: These are current-game stats that cause leakage
                 'FANTASY_PTS', 'PLUS_MINUS']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Transform columns
    column_map = {
        'PLAYER_NAME': 'player', 'TEAM_ABBREVIATION': 'tm', 'MATCHUP': 'matchup_raw',
        'GAME_DATE': 'game_date', 'WL': 'res', 'MIN': 'mp',
        'FGM': 'fg', 'FGA': 'fga', 'FG_PCT': 'fg_pct',
        'FG3M': '3p', 'FG3A': '3pa', 'FG3_PCT': '3p_pct',
        'FTM': 'ft', 'FTA': 'fta', 'FT_PCT': 'ft_pct',
        'OREB': 'orb', 'DREB': 'drb', 'REB': 'trb',
        'AST': 'ast', 'STL': 'stl', 'BLK': 'blk', 'TOV': 'tov', 'PF': 'pf', 'PTS': 'pts',
    }
    df = df.rename(columns=column_map)
    
    # Parse opponent
    def parse_opp(m):
        if pd.isna(m): return None
        parts = m.replace('.', '').split()
        return parts[2] if len(parts) >= 3 and parts[1] in ['vs', '@'] else None
    
    df['opp'] = df['matchup_raw'].apply(parse_opp)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['won'] = (df['res'] == 'W').astype(int)
    
    # Game IDs
    df['game_id'] = df['game_date'].dt.strftime('%Y%m%d') + '_' + df['player'].str.replace(' ', '_') + '_' + df['tm']
    df['matchup_id'] = df.apply(lambda x: x['game_date'].strftime('%Y%m%d') + '_' + '_'.join(sorted([x['tm'], str(x['opp'])])), axis=1)
    
    # Ensure numeric
    for col in ['pts', 'fg', 'fga', 'ft', 'fta', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', '3p', '3pa', 'mp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate GmSc
    df['gmsc'] = (df['pts'] + 0.4*df['fg'] - 0.7*df['fga'] - 0.4*(df['fta']-df['ft']) + 
                  0.7*df['orb'] + 0.3*df['drb'] + df['stl'] + 0.7*df['ast'] + 0.7*df['blk'] - 
                  0.4*df['pf'] - df['tov']).round(1)
    
    df = df.sort_values(['game_date', 'player']).reset_index(drop=True)
    
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"  Players: {df['player'].nunique()}")
    
    return df


def step2_rolling_features(df):
    """Create rolling average features using SHIFTED data (no leakage)"""
    print("\n" + "="*60)
    print("STEP 2: CREATING ROLLING FEATURES")
    print("="*60)
    
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    grouped = df.groupby('player')
    
    # Games played counter
    df['games_played'] = grouped.cumcount()
    
    # Stats to create rolling features for
    stats = ['pts', 'trb', 'ast', 'mp', 'fga', 'fg_pct', '3pa', '3p_pct', 'fta', 'ft_pct', 'tov', 'stl', 'blk']
    windows = [5, 10, 20]
    
    # CRITICAL: Shift first to prevent leakage
    for stat in stats:
        df[f'{stat}_shifted'] = grouped[stat].shift(1)
    
    # Rolling means on SHIFTED data
    for window in windows:
        for stat in stats:
            df[f'{stat}_l{window}'] = grouped[f'{stat}_shifted'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        print(f"  L{window} features created")
    
    # Volatility (std dev) - only for L5, L20 to avoid using L10
    for stat in ['pts', 'trb', 'ast', 'mp']:
        df[f'{stat}_std_l5'] = grouped[f'{stat}_shifted'].transform(
            lambda x: x.rolling(5, min_periods=3).std()
        )
        df[f'{stat}_std_l20'] = grouped[f'{stat}_shifted'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
    
    # Trends (L5/L20 ratio - doesn't use L10)
    for stat in ['pts', 'trb', 'ast']:
        df[f'{stat}_trend'] = df[f'{stat}_l5'] / df[f'{stat}_l20'].replace(0, np.nan)
    
    # Drop shifted columns
    df = df.drop(columns=[f'{s}_shifted' for s in stats])
    
    print(f"  Total columns: {len(df.columns)}")
    return df


def step3_opponent_features(df):
    """Add opponent defensive features"""
    print("\n" + "="*60)
    print("STEP 3: ADDING OPPONENT FEATURES")
    print("="*60)
    
    # Team game stats
    team_games = df.groupby(['tm', 'game_date', 'opp']).agg({
        'pts': 'sum', 'trb': 'sum', 'ast': 'sum', 'fga': 'sum'
    }).reset_index()
    team_games.columns = ['tm', 'game_date', 'opp', 'tm_pts', 'tm_trb', 'tm_ast', 'tm_fga']
    
    # What each team allowed (opponent's stats = what this team allowed)
    opp_stats = team_games[['tm', 'game_date', 'tm_pts', 'tm_trb', 'tm_ast', 'tm_fga']].copy()
    opp_stats = opp_stats.rename(columns={
        'tm': 'opp', 'tm_pts': 'pts_allowed', 'tm_trb': 'trb_allowed', 
        'tm_ast': 'ast_allowed', 'tm_fga': 'fga_allowed'
    })
    team_games = team_games.merge(opp_stats, on=['opp', 'game_date'], how='left')
    
    # Rolling defensive stats (using L5 instead of L10 to leave L10 for line proxy)
    team_games = team_games.sort_values(['tm', 'game_date']).reset_index(drop=True)
    grouped = team_games.groupby('tm')
    
    for stat in ['pts_allowed', 'trb_allowed', 'ast_allowed', 'fga_allowed']:
        team_games[f'{stat}_shifted'] = grouped[stat].shift(1)
        team_games[f'{stat}_l5'] = grouped[f'{stat}_shifted'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
    
    # Defensive ranks
    for stat in ['pts_allowed', 'trb_allowed', 'ast_allowed']:
        team_games[f'{stat}_rank'] = team_games.groupby('game_date')[f'{stat}_l5'].rank(method='average')
    
    # Merge to players
    opp_cols = ['pts_allowed_l5', 'trb_allowed_l5', 'ast_allowed_l5', 'fga_allowed_l5',
                'pts_allowed_rank', 'trb_allowed_rank', 'ast_allowed_rank']
    opp_df = team_games[['tm', 'game_date'] + opp_cols].copy()
    opp_df = opp_df.rename(columns={'tm': 'opp'})
    opp_df.columns = ['opp', 'game_date'] + [f'opp_{c}' for c in opp_cols]
    
    df = df.merge(opp_df, on=['opp', 'game_date'], how='left')
    
    print(f"  Added {len(opp_cols)} opponent features")
    return df


def step4_schedule_features(df):
    """Add schedule and fatigue features"""
    print("\n" + "="*60)
    print("STEP 4: ADDING SCHEDULE FEATURES")
    print("="*60)
    
    # Team schedule
    team_games = df.groupby(['tm', 'game_date']).first().reset_index()[['tm', 'game_date']]
    team_games = team_games.sort_values(['tm', 'game_date']).reset_index(drop=True)
    grouped = team_games.groupby('tm')
    
    team_games['prev_game'] = grouped['game_date'].shift(1)
    team_games['days_rest'] = (team_games['game_date'] - team_games['prev_game']).dt.days
    team_games['is_b2b'] = (team_games['days_rest'] <= 1).astype(int)
    team_games['is_b2b_second'] = (team_games['days_rest'] == 1).astype(int)
    
    # Games in windows
    for tm in team_games['tm'].unique():
        mask = team_games['tm'] == tm
        tm_df = team_games[mask].set_index('game_date').sort_index()
        team_games.loc[mask, 'games_last_7d'] = tm_df.index.to_series().rolling('7D', closed='left').count().values
        team_games.loc[mask, 'games_last_14d'] = tm_df.index.to_series().rolling('14D', closed='left').count().values
    
    # Merge to players
    sched_cols = ['days_rest', 'is_b2b', 'is_b2b_second', 'games_last_7d', 'games_last_14d']
    df = df.merge(team_games[['tm', 'game_date'] + sched_cols], on=['tm', 'game_date'], how='left')
    
    # Player fatigue (use L3 and L5, not L10)
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    grouped = df.groupby('player')
    
    df['mp_shifted'] = grouped['mp'].shift(1)
    df['mp_l3'] = grouped['mp_shifted'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df['mp_avg_l3'] = grouped['mp_shifted'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['high_min_prev'] = (grouped['mp'].shift(1) >= 36).astype(float)
    df['games_streak'] = grouped.cumcount() + 1
    df['player_days_rest'] = (df['game_date'] - grouped['game_date'].shift(1)).dt.days
    
    df = df.drop(columns=['mp_shifted', 'prev_game'], errors='ignore')
    
    print(f"  Added schedule features")
    print(f"  Total columns: {len(df.columns)}")
    return df


def step5_train_models(df):
    """Train all prediction models with STRICT leakage prevention"""
    print("\n" + "="*60)
    print("STEP 5: TRAINING MODELS (LEAKAGE-SAFE)")
    print("="*60)
    
    # Filter to players with enough games
    df = df[df['games_played'] >= 10].copy()
    print(f"  Training data: {len(df):,} rows")
    
    # Create combo targets (current game values - will be EXCLUDED from features)
    df['pra'] = df['pts'] + df['trb'] + df['ast']
    df['pr'] = df['pts'] + df['trb']
    df['pa'] = df['pts'] + df['ast']
    
    # Create line proxies (for betting simulation - will be EXCLUDED from features)
    df['pra_l10'] = df['pts_l10'] + df['trb_l10'] + df['ast_l10']
    df['pr_l10'] = df['pts_l10'] + df['trb_l10']
    df['pa_l10'] = df['pts_l10'] + df['ast_l10']
    
    # Get valid features (STRICT filtering)
    feature_cols = get_valid_features(df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Sample features: {feature_cols[:10]}")
    
    # EXPLICIT CHECK: Print any columns that look suspicious
    suspicious = [c for c in feature_cols if any(x in c.upper() for x in ['FANTASY', 'PLUS_MINUS', 'PLUSMINUS'])]
    if suspicious:
        print(f"  *** SUSPICIOUS COLUMNS FOUND: {suspicious} ***")
        feature_cols = [c for c in feature_cols if c not in suspicious]
        print(f"  Removed them. New feature count: {len(feature_cols)}")
    
    # Verify no leakage in feature list
    for col in feature_cols:
        if col in EXCLUDED_FROM_FEATURES:
            raise ValueError(f"LEAKAGE: {col} should not be in features!")
    
    # Temporal split - 14 days for test
    max_date = df['game_date'].max()
    cutoff = max_date - pd.Timedelta(days=14)
    train = df[df['game_date'] < cutoff].copy()
    test = df[df['game_date'] >= cutoff].copy()
    print(f"  Train: {len(train):,}, Test: {len(test):,}")
    print(f"  Train period: {train['game_date'].min().date()} to {train['game_date'].max().date()}")
    print(f"  Test period: {test['game_date'].min().date()} to {test['game_date'].max().date()}")
    
    # Train each prop
    targets = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa']
    results = {}
    
    print("\n  Training models...")
    
    for target in targets:
        # Validate no leakage for this specific target
        validate_no_leakage(feature_cols, target)
        
        # Line column for betting simulation
        line_col = f'{target}_l10'
        
        # Prepare data
        train_clean = train.dropna(subset=[target])
        test_clean = test.dropna(subset=[target])
        
        X_train = train_clean[feature_cols].copy()
        y_train = train_clean[target].values
        X_test = test_clean[feature_cols].copy()
        y_test = test_clean[target].values
        
        # Fill NaN with medians
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        
        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, 
            min_samples_leaf=20, random_state=42
        )
        model.fit(X_train_sc, y_train)
        
        # Predict
        pred = model.predict(X_test_sc)
        pred = np.maximum(0, pred)  # Floor at 0
        
        # Evaluate
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        
        # Check for suspicious accuracy
        validate_predictions_reasonable(y_test, pred, target)
        
        # Betting simulation using L10 as line
        if line_col in test_clean.columns:
            lines = test_clean[line_col].values
            valid_mask = ~np.isnan(lines)
            
            if valid_mask.sum() > 0:
                pred_over = pred[valid_mask] > lines[valid_mask]
                actual_over = y_test[valid_mask] > lines[valid_mask]
                accuracy = (pred_over == actual_over).mean()
            else:
                accuracy = 0.5
        else:
            accuracy = 0.5
        
        status = "OK" if accuracy < 0.65 else "SUSPICIOUS"
        print(f"  {target.upper():4} - R2: {r2:.3f}, MAE: {mae:.2f}, Bet Acc: {accuracy:.1%} [{status}]")
        
        # Save model
        results[target] = {
            'model': model, 'scaler': scaler, 'medians': medians, 
            'features': feature_cols, 'r2': r2, 'mae': mae, 'accuracy': accuracy
        }
        
        with open(f'model_{target}.pkl', 'wb') as f:
            pickle.dump({
                'model': model, 'scaler': scaler, 'medians': medians, 
                'features': feature_cols, 'target': target,
                'metrics': {'r2': r2, 'mae': mae, 'accuracy': accuracy}
            }, f)
    
    return results, df


def main():
    print("="*60)
    print("NBA PLAYER PROPS - LEAKAGE-SAFE PIPELINE")
    print(f"Started: {datetime.now()}")
    print("="*60)
    
    # Run pipeline
    df = step1_download_data()
    df = step2_rolling_features(df)
    df = step3_opponent_features(df)
    df = step4_schedule_features(df)
    
    # Save processed data
    df.to_csv('player_games_schedule.csv', index=False)
    print(f"\n  Saved: player_games_schedule.csv ({len(df):,} rows)")
    
    results, df = step5_train_models(df)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    print(f"Total rows: {len(df):,}")
    print(f"Models saved: {len(results)}")
    
    # Realistic expectations check
    suspicious = [t for t, r in results.items() if r['accuracy'] > 0.60]
    profitable = [t for t, r in results.items() if 0.5238 < r['accuracy'] <= 0.60]
    
    if suspicious:
        print(f"\nWARNING - Suspiciously high accuracy (may have leakage): {', '.join(suspicious)}")
    
    if profitable:
        print(f"\nPotentially profitable props (52.4%-60%): {', '.join(profitable)}")
    else:
        print(f"\nNo props in profitable range - model may need tuning")
    
    print("\nEXPECTED RANGES (if no leakage):")
    print("  R2: 0.35-0.55 (lower for rare events like STL/BLK)")
    print("  Betting accuracy: 52-58% (55%+ is excellent)")
    print("  Anything above 65% is almost certainly leakage")
    
    print("\nNext steps:")
    print("  1. Get API key from https://the-odds-api.com/ (free)")
    print("  2. python get_real_lines.py --api-key YOUR_KEY")
    print("  3. python task8_betting_lines.py --input real_lines.csv")


if __name__ == "__main__":
    main()
