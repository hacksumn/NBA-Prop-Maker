"""
MINUTES DEVIATION MODEL
Predicts when a player will play MORE or LESS minutes than their recent average.

The insight: Vegas knows player averages. We don't beat them by predicting averages.
We beat them by identifying WHEN a player will deviate from average.

Target: minutes_deviation = actual_minutes - L10_minutes_avg
Positive = played more than usual
Negative = played less than usual

Key factors that cause deviation:
- Back-to-back games (fatigue -> less minutes)
- Blowouts (starters sit -> less minutes)
- Teammate injuries (more usage -> more minutes)
- Foul trouble history
- Pace of opponent
- Days rest
- Recent minutes trend (coach adjusting role?)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
except ImportError:
    print("Installing required packages...")
    import os
    os.system('pip install xgboost scikit-learn --break-system-packages -q')
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error


CONFIG = {
    'data_dir': Path('./data'),
    'models_dir': Path('./models'),
    'output_dir': Path('./output'),
    'min_games': 15,  # Need enough history
}

for d in CONFIG.values():
    if isinstance(d, Path):
        d.mkdir(parents=True, exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    """Load and standardize data."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    # Standardize column names
    col_map = {
        'PLAYER_NAME': 'player',
        'GAME_DATE': 'game_date',
        'MATCHUP': 'matchup',
        'MIN': 'mp',
        'PTS': 'pts',
        'REB': 'trb',
        'AST': 'ast',
        'STL': 'stl',
        'BLK': 'blk',
        'TOV': 'tov',
        'FGM': 'fg',
        'FGA': 'fga',
        'FG3A': '3pa',
        'FTM': 'ft',
        'FTA': 'fta',
        'WL': 'result',
        'PLUS_MINUS': 'plus_minus',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Parse dates
    if 'game_date' in df.columns:
        try:
            df['game_date'] = pd.to_datetime(df['game_date'], format='%b %d, %Y')
        except:
            df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Extract team/opponent from matchup
    if 'matchup' in df.columns:
        df['is_home'] = df['matchup'].apply(lambda x: 0 if '@' in str(x) else 1)
        df['team'] = df['matchup'].apply(lambda x: str(x).split()[0] if pd.notna(x) else None)
        df['opp'] = df['matchup'].apply(lambda x: str(x).split()[-1] if pd.notna(x) else None)
    
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    print(f"  {len(df):,} games, {df['player'].nunique()} players")
    print(f"  Date range: {df['game_date'].min().strftime('%Y-%m-%d')} to {df['game_date'].max().strftime('%Y-%m-%d')}")
    
    return df


def create_minutes_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features specifically for predicting minutes deviation.
    
    Focus on factors that cause players to play MORE or LESS than usual.
    """
    print("\nCreating minutes deviation features...")
    df = df.copy()
    
    # Sort by player and date
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    
    # Games played counter
    df['games_played'] = df.groupby('player').cumcount() + 1
    
    # ==========================================================================
    # BASELINE MINUTES (what we're deviating FROM)
    # ==========================================================================
    print("  Computing baselines...")
    
    for stat in ['mp']:
        # Shifted to avoid leakage
        df[f'{stat}_shifted'] = df.groupby('player')[stat].shift(1)
        
        # Rolling averages (L5, L10, L20)
        for window in [5, 10, 20]:
            df[f'{stat}_l{window}'] = df.groupby('player')[f'{stat}_shifted'].transform(
                lambda x: x.rolling(window, min_periods=3).mean()
            )
        
        # EWMA
        df[f'{stat}_ewma5'] = df.groupby('player')[f'{stat}_shifted'].transform(
            lambda x: x.ewm(span=5, min_periods=3).mean()
        )
        df[f'{stat}_ewma3'] = df.groupby('player')[f'{stat}_shifted'].transform(
            lambda x: x.ewm(span=3, min_periods=2).mean()
        )
        
        # Std dev (consistency)
        df[f'{stat}_std_l10'] = df.groupby('player')[f'{stat}_shifted'].transform(
            lambda x: x.rolling(10, min_periods=5).std()
        )
    
    # ==========================================================================
    # TARGET: Minutes deviation from L10
    # ==========================================================================
    df['mp_deviation'] = df['mp'] - df['mp_l10']
    df['mp_deviation_pct'] = df['mp_deviation'] / df['mp_l10'].clip(lower=1)
    
    # ==========================================================================
    # SCHEDULE FACTORS (big impact on minutes)
    # ==========================================================================
    print("  Schedule factors...")
    
    # Days since last game
    df['prev_game_date'] = df.groupby('player')['game_date'].shift(1)
    df['days_rest'] = (df['game_date'] - df['prev_game_date']).dt.days.fillna(3)
    
    # Back-to-back detection
    df['is_b2b'] = (df['days_rest'] <= 1).astype(int)
    
    # 3 games in 4 nights (simplified: if B2B, assume heavy schedule)
    df['is_heavy_schedule'] = df['is_b2b']  # Simplified proxy
    
    # Previous game minutes (fatigue indicator)
    df['prev_mp'] = df.groupby('player')['mp'].shift(1)
    df['prev_mp_high'] = (df['prev_mp'] > 38).astype(int)  # Played heavy last game
    df['prev_mp_low'] = (df['prev_mp'] < 20).astype(int)   # Played light last game
    
    # ==========================================================================
    # MINUTES TREND (is coach changing role?)
    # ==========================================================================
    print("  Minutes trend...")
    
    # L3 vs L10 (recent trend)
    df['mp_l3'] = df.groupby('player')['mp_shifted'].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )
    df['mp_trend_l3_l10'] = df['mp_l3'] - df['mp_l10']
    df['mp_trend_pct'] = df['mp_trend_l3_l10'] / df['mp_l10'].clip(lower=1)
    
    # Is role expanding or shrinking?
    df['role_expanding'] = (df['mp_trend_l3_l10'] > 2).astype(int)
    df['role_shrinking'] = (df['mp_trend_l3_l10'] < -2).astype(int)
    
    # Minutes volatility (inconsistent minutes = harder to predict)
    df['mp_cv'] = df['mp_std_l10'] / df['mp_l10'].clip(lower=1)  # Coefficient of variation
    
    # ==========================================================================
    # ROLE INDICATORS
    # ==========================================================================
    print("  Role indicators...")
    
    # Star/starter/bench classification based on recent minutes
    df['is_star'] = (df['mp_l10'] >= 32).astype(int)
    df['is_starter'] = (df['mp_l10'] >= 24).astype(int)
    df['is_rotation'] = ((df['mp_l10'] >= 15) & (df['mp_l10'] < 24)).astype(int)
    df['is_bench'] = (df['mp_l10'] < 15).astype(int)
    
    # ==========================================================================
    # GAME CONTEXT (will this be a blowout?)
    # ==========================================================================
    print("  Game context...")
    
    # Previous game margin (blowouts = less minutes for starters)
    df['prev_margin'] = df.groupby('player')['plus_minus'].shift(1).abs()
    df['prev_blowout'] = (df['prev_margin'] > 15).astype(int)
    
    # Win/loss trend (team struggling = starters play more?)
    df['prev_win'] = (df.groupby('player')['result'].shift(1) == 'W').astype(int)
    
    # ==========================================================================
    # OPPONENT PACE (fast teams = more possessions = more minutes opportunity)
    # ==========================================================================
    print("  Opponent context...")
    
    # Estimate opponent pace - using expanding mean (only past data)
    # Sort by date first to ensure proper expanding calculation
    df = df.sort_values('game_date').reset_index(drop=True)
    
    if 'fga' in df.columns:
        # Use expanding mean per opponent (only includes games before current)
        df['opp_fga_expanding'] = df.groupby('opp')['fga'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        league_avg_fga = df['fga'].mean()
        df['opp_pace_factor'] = (df['opp_fga_expanding'] / league_avg_fga).fillna(1.0)
    else:
        df['opp_pace_factor'] = 1.0
    
    # Re-sort by player and date for remaining calculations
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    
    # ==========================================================================
    # HOME/AWAY EFFECT
    # ==========================================================================
    print("  Home/away splits...")
    
    # Home/away splits - use expanding mean to avoid leakage
    # For each game, compute avg minutes at home/away using only PRIOR games
    
    # Initialize columns
    df['mp_home_avg'] = np.nan
    df['mp_away_avg'] = np.nan
    
    # Compute expanding average for home games per player
    home_mask = df['is_home'] == 1
    away_mask = df['is_home'] == 0
    
    # Group by player and compute expanding mean on shifted values (prior games only)
    for player in df['player'].unique():
        player_mask = df['player'] == player
        
        # Home games expanding average
        home_player = df[player_mask & home_mask]['mp_shifted']
        if len(home_player) > 0:
            home_expanding = home_player.expanding(min_periods=1).mean()
            # Shift so we don't include current game
            home_expanding = home_expanding.shift(1)
            df.loc[player_mask & home_mask, 'mp_home_running'] = home_expanding.values
        
        # Away games expanding average  
        away_player = df[player_mask & away_mask]['mp_shifted']
        if len(away_player) > 0:
            away_expanding = away_player.expanding(min_periods=1).mean()
            away_expanding = away_expanding.shift(1)
            df.loc[player_mask & away_mask, 'mp_away_running'] = away_expanding.values
    
    # Forward fill to carry averages to all games
    df['mp_home_avg'] = df.groupby('player')['mp_home_running'].transform(lambda x: x.ffill())
    df['mp_away_avg'] = df.groupby('player')['mp_away_running'].transform(lambda x: x.ffill())
    
    # Fill any remaining NaN with L10
    df['mp_home_avg'] = df['mp_home_avg'].fillna(df['mp_l10'])
    df['mp_away_avg'] = df['mp_away_avg'].fillna(df['mp_l10'])
    
    # Drop temp columns
    df = df.drop(columns=['mp_home_running', 'mp_away_running'], errors='ignore')
    
    # Expected minutes for this game's location
    df['mp_location_expected'] = np.where(df['is_home'] == 1, df['mp_home_avg'], df['mp_away_avg'])
    df['mp_location_diff'] = df['mp_location_expected'] - df['mp_l10']
    
    # ==========================================================================
    # COMPOSITE FEATURES
    # ==========================================================================
    print("  Composite features...")
    
    # Fatigue score (higher = more tired)
    df['fatigue_score'] = (
        df['is_b2b'] * 2 +
        df['prev_mp_high'] * 1 +
        df['is_heavy_schedule'] * 1 +
        (df['days_rest'] == 0).astype(int) * 2
    )
    
    # Expected minutes adjustment from baseline
    df['mp_expected_adj'] = (
        df['mp_trend_l3_l10'] +  # Recent trend
        df['mp_location_diff'].fillna(0) +  # Home/away effect
        -df['fatigue_score'] * 1.5  # Fatigue reduction
    )
    
    # Fill NaN
    df = df.fillna(0)
    
    print(f"  Created {len([c for c in df.columns if c not in ['player', 'game_date', 'team', 'opp', 'matchup', 'result']])} features")
    
    return df


def get_feature_columns() -> list:
    """Get feature columns for minutes deviation model."""
    return [
        # Baseline minutes
        'mp_l5', 'mp_l10', 'mp_l20', 'mp_ewma3', 'mp_ewma5',
        'mp_std_l10', 'mp_cv',
        
        # Schedule
        'days_rest', 'is_b2b', 'is_heavy_schedule',
        'prev_mp', 'prev_mp_high', 'prev_mp_low',
        
        # Trend
        'mp_l3', 'mp_trend_l3_l10', 'mp_trend_pct',
        'role_expanding', 'role_shrinking',
        
        # Role
        'is_star', 'is_starter', 'is_rotation', 'is_bench',
        'games_played',
        
        # Game context
        'prev_blowout', 'prev_win',
        
        # Opponent
        'opp_pace_factor',
        
        # Home/away
        'is_home', 'mp_location_diff',
        
        # Composites
        'fatigue_score', 'mp_expected_adj',
    ]


def train_minutes_model(df: pd.DataFrame) -> dict:
    """Train model to predict minutes deviation."""
    print("\n" + "="*60)
    print("TRAINING MINUTES DEVIATION MODEL")
    print("="*60)
    
    # Filter to players with enough history
    df = df[df['games_played'] >= CONFIG['min_games']].copy()
    
    # Remove rows with no target
    df = df[df['mp_l10'].notna() & (df['mp_l10'] > 0)].copy()
    
    print(f"\nTraining data: {len(df):,} games")
    
    # Features and target
    feature_cols = get_feature_columns()
    
    # Only use features that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Using {len(feature_cols)} features")
    
    X = df[feature_cols].fillna(0)
    y = df['mp_deviation']  # Target: actual - L10
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Model parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 150,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }
    
    cv_results = []
    
    print("\nCross-validation:")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, y_pred)
        
        # Direction accuracy: did we predict +/- correctly?
        pred_direction = np.sign(y_pred)
        actual_direction = np.sign(y_val)
        direction_acc = (pred_direction == actual_direction).mean()
        
        # Baseline: always predict 0 (no deviation)
        baseline_mae = mean_absolute_error(y_val, np.zeros_like(y_val))
        
        cv_results.append({
            'fold': fold,
            'mae': mae,
            'baseline_mae': baseline_mae,
            'improvement': baseline_mae - mae,
            'direction_acc': direction_acc,
        })
        
        print(f"  Fold {fold}: MAE={mae:.2f} (baseline={baseline_mae:.2f}), Direction={direction_acc:.1%}")
    
    # Summary
    avg_mae = np.mean([r['mae'] for r in cv_results])
    avg_baseline = np.mean([r['baseline_mae'] for r in cv_results])
    avg_direction = np.mean([r['direction_acc'] for r in cv_results])
    
    print(f"\nAverage:")
    print(f"  MAE: {avg_mae:.2f} minutes (baseline: {avg_baseline:.2f})")
    print(f"  Improvement: {avg_baseline - avg_mae:.2f} minutes")
    print(f"  Direction Accuracy: {avg_direction:.1%}")
    
    # Train final model
    print("\nTraining final model...")
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X, y, verbose=False)
    
    # Feature importance
    importance = sorted(zip(feature_cols, final_model.feature_importances_), key=lambda x: -x[1])
    print("\nTop 10 features:")
    for feat, imp in importance[:10]:
        print(f"  {imp*100:5.1f}%  {feat}")
    
    # Save model
    model_path = CONFIG['models_dir'] / 'minutes_deviation_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_cols': feature_cols,
            'cv_results': cv_results,
        }, f)
    print(f"\nModel saved to {model_path}")
    
    return {
        'model': final_model,
        'feature_cols': feature_cols,
        'mae': avg_mae,
        'direction_accuracy': avg_direction,
    }


def predict_minutes_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Predict minutes deviation for each player."""
    
    model_path = CONFIG['models_dir'] / 'minutes_deviation_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError("Train model first!")
    
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    
    model = saved['model']
    feature_cols = saved['feature_cols']
    
    # Get latest data for each player
    df = df.sort_values('game_date')
    latest = df.groupby('player').last().reset_index()
    latest = latest[latest['games_played'] >= CONFIG['min_games']]
    
    print(f"\nPredicting for {len(latest)} players...")
    
    # Ensure features exist
    for c in feature_cols:
        if c not in latest.columns:
            latest[c] = 0
    
    X = latest[feature_cols].fillna(0)
    
    # Predict deviation
    latest['mp_deviation_pred'] = model.predict(X)
    
    # Expected minutes = L10 + predicted deviation
    latest['mp_expected'] = latest['mp_l10'] + latest['mp_deviation_pred']
    
    # Confidence based on player consistency
    latest['mp_confidence'] = 1 - latest['mp_cv'].clip(0, 1)
    
    # Sort by absolute deviation (biggest predicted changes)
    latest['mp_deviation_abs'] = latest['mp_deviation_pred'].abs()
    latest = latest.sort_values('mp_deviation_abs', ascending=False)
    
    return latest


def analyze_deviation_factors(df: pd.DataFrame):
    """Analyze what factors actually drive minutes deviation."""
    print("\n" + "="*60)
    print("MINUTES DEVIATION ANALYSIS")
    print("="*60)
    
    # Need enough data
    df = df[df['games_played'] >= 15].copy()
    df = df[df['mp_l10'].notna()].copy()
    
    print(f"\nAnalyzing {len(df):,} games...")
    
    # Overall deviation stats
    print(f"\nDeviation Statistics:")
    print(f"  Mean: {df['mp_deviation'].mean():.2f} minutes")
    print(f"  Std: {df['mp_deviation'].std():.2f} minutes")
    print(f"  Median: {df['mp_deviation'].median():.2f} minutes")
    
    # Factor analysis
    factors = [
        ('is_b2b', 'Back-to-back games'),
        ('prev_mp_high', 'Heavy minutes prev game (>38)'),
        ('is_heavy_schedule', '3+ games in 4 days'),
        ('is_home', 'Home games'),
        ('prev_blowout', 'Previous game blowout'),
        ('role_expanding', 'Role expanding (L3 > L10 + 2)'),
        ('role_shrinking', 'Role shrinking (L3 < L10 - 2)'),
    ]
    
    print("\nFactor Impact on Minutes:")
    print("-" * 50)
    
    for factor, desc in factors:
        if factor not in df.columns:
            continue
        
        with_factor = df[df[factor] == 1]['mp_deviation'].mean()
        without_factor = df[df[factor] == 0]['mp_deviation'].mean()
        diff = with_factor - without_factor
        n_with = (df[factor] == 1).sum()
        
        print(f"  {desc}:")
        print(f"    With: {with_factor:+.1f} min (n={n_with:,})")
        print(f"    Without: {without_factor:+.1f} min")
        print(f"    Effect: {diff:+.1f} min")
        print()
    
    # By role
    print("\nDeviation by Player Role:")
    for role, label in [('is_star', 'Stars (32+ mpg)'), 
                        ('is_starter', 'Starters (24+ mpg)'),
                        ('is_rotation', 'Rotation (15-24 mpg)'),
                        ('is_bench', 'Bench (<15 mpg)')]:
        if role in df.columns:
            role_dev = df[df[role] == 1]['mp_deviation'].std()
            role_n = (df[role] == 1).sum()
            print(f"  {label}: std={role_dev:.1f} min (n={role_n:,})")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python minutes_deviation.py <command> [data_file]")
        print("Commands: train, predict, analyze")
        return
    
    command = sys.argv[1].lower()
    data_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not data_file:
        print("Please provide a data file path")
        return
    
    # Load and prepare data
    df = load_data(data_file)
    df = create_minutes_features(df)
    
    if command == 'train':
        train_minutes_model(df)
        
    elif command == 'predict':
        predictions = predict_minutes_deviation(df)
        
        print("\n=== BIGGEST PREDICTED DEVIATIONS ===")
        print("\nPlayers expected to play MORE than usual:")
        up = predictions[predictions['mp_deviation_pred'] > 2].head(10)
        print(up[['player', 'mp_l10', 'mp_deviation_pred', 'mp_expected', 'is_b2b', 'fatigue_score']].to_string(index=False))
        
        print("\nPlayers expected to play LESS than usual:")
        down = predictions[predictions['mp_deviation_pred'] < -2].head(10)
        print(down[['player', 'mp_l10', 'mp_deviation_pred', 'mp_expected', 'is_b2b', 'fatigue_score']].to_string(index=False))
        
        # Save
        out_path = CONFIG['output_dir'] / f"minutes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        predictions.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
        
    elif command == 'analyze':
        analyze_deviation_factors(df)
        
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
