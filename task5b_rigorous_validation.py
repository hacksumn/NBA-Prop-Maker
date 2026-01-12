"""
Task 5B: Rigorous Data Validation
NBA Player Props Betting System

This script MANUALLY verifies that our feature engineering is correct
by tracing through specific players and calculating expected values by hand.

NO ASSUMPTIONS - we verify everything against raw data.
"""

import pandas as pd
import numpy as np


def load_all_datasets():
    """Load raw and processed datasets for comparison."""
    print("Loading all datasets for comparison...")
    
    datasets = {}
    
    # Raw data
    datasets['raw'] = pd.read_csv('database_24_25.csv')
    datasets['raw']['Data'] = pd.to_datetime(datasets['raw']['Data'])
    print(f"  Raw: {len(datasets['raw']):,} rows")
    
    # Clean data
    datasets['clean'] = pd.read_csv('player_games_clean.csv', parse_dates=['game_date'])
    print(f"  Clean: {len(datasets['clean']):,} rows")
    
    # Features data (with opponent features)
    datasets['features'] = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
    print(f"  Features: {len(datasets['features']):,} rows")
    
    # Final model data
    datasets['model'] = pd.read_csv('model_data_filtered.csv', parse_dates=['game_date'])
    print(f"  Model (filtered): {len(datasets['model']):,} rows")
    
    return datasets


def verify_player_rolling_average(datasets: dict, player_name: str):
    """
    Manually verify rolling average calculation for a specific player.
    
    This is the CRITICAL test - we calculate what the L5/L10 should be
    by hand and compare to what our pipeline produced.
    """
    print(f"\n{'='*60}")
    print(f"MANUAL VERIFICATION: {player_name}")
    print(f"{'='*60}")
    
    # Get raw data for this player
    raw = datasets['raw']
    raw_player = raw[raw['Player'] == player_name].sort_values('Data').reset_index(drop=True)
    
    if len(raw_player) == 0:
        print(f"  ✗ Player '{player_name}' not found in raw data")
        return False
    
    print(f"\nRaw data: {len(raw_player)} games")
    print(raw_player[['Data', 'PTS', 'TRB', 'AST', 'MP']].head(15).to_string())
    
    # Get processed data for this player
    features = datasets['features']
    feat_player = features[features['player'] == player_name].sort_values('game_date').reset_index(drop=True)
    
    print(f"\nProcessed data: {len(feat_player)} games")
    
    # MANUAL CALCULATION: What should pts_l5 be for game 6?
    # Game 6 should use games 1-5 (shifted by 1, then rolling 5)
    print(f"\n[MANUAL VERIFICATION - pts_l5 for game index 5 (6th game)]")
    
    if len(raw_player) >= 6:
        # Games 0-4 (first 5 games) should be used for game 5's pts_l5
        manual_pts_l5 = raw_player.iloc[0:5]['PTS'].mean()
        pipeline_pts_l5 = feat_player.iloc[5]['pts_l5'] if len(feat_player) > 5 else None
        
        print(f"  Games 1-5 PTS: {raw_player.iloc[0:5]['PTS'].tolist()}")
        print(f"  Manual calculation: mean = {manual_pts_l5:.4f}")
        print(f"  Pipeline pts_l5: {pipeline_pts_l5}")
        
        if pipeline_pts_l5 is not None and abs(manual_pts_l5 - pipeline_pts_l5) < 0.01:
            print(f"  ✓ MATCH - Rolling average is correct")
        else:
            print(f"  ✗ MISMATCH - Pipeline calculation is wrong!")
            return False
    
    # MANUAL CALCULATION: What should pts_l10 be for game 11?
    print(f"\n[MANUAL VERIFICATION - pts_l10 for game index 10 (11th game)]")
    
    if len(raw_player) >= 11:
        # Games 0-9 (first 10 games) should be used for game 10's pts_l10
        manual_pts_l10 = raw_player.iloc[0:10]['PTS'].mean()
        pipeline_pts_l10 = feat_player.iloc[10]['pts_l10'] if len(feat_player) > 10 else None
        
        print(f"  Games 1-10 PTS: {raw_player.iloc[0:10]['PTS'].tolist()}")
        print(f"  Manual calculation: mean = {manual_pts_l10:.4f}")
        print(f"  Pipeline pts_l10: {pipeline_pts_l10}")
        
        if pipeline_pts_l10 is not None and abs(manual_pts_l10 - pipeline_pts_l10) < 0.01:
            print(f"  ✓ MATCH - Rolling average is correct")
        else:
            print(f"  ✗ MISMATCH - Pipeline calculation is wrong!")
            return False
    
    # LEAKAGE TEST: Verify game N's features don't include game N's stats
    print(f"\n[LEAKAGE TEST - Does game N use game N's data?]")
    
    if len(raw_player) >= 3:
        # For game 2 (index 1), pts_l5 should ONLY be game 1's points
        game1_pts = raw_player.iloc[0]['PTS']
        game2_pts_l5 = feat_player.iloc[1]['pts_l5'] if len(feat_player) > 1 else None
        
        print(f"  Game 1 PTS: {game1_pts}")
        print(f"  Game 2 pts_l5 (should equal game 1 only): {game2_pts_l5}")
        
        if game2_pts_l5 is not None and abs(game1_pts - game2_pts_l5) < 0.01:
            print(f"  ✓ PASS - No leakage (game 2 only uses game 1)")
        else:
            print(f"  ✗ FAIL - Possible leakage!")
            return False
    
    # Verify game 1 has NaN (no prior data)
    print(f"\n[FIRST GAME TEST - Should have NaN features]")
    game1_pts_l5 = feat_player.iloc[0]['pts_l5'] if len(feat_player) > 0 else None
    print(f"  Game 1 pts_l5: {game1_pts_l5}")
    
    if pd.isna(game1_pts_l5):
        print(f"  ✓ PASS - First game has NaN (no prior data)")
    else:
        print(f"  ✗ FAIL - First game should have NaN!")
        return False
    
    return True


def verify_opponent_features(datasets: dict, player_name: str, game_idx: int = 10):
    """
    Verify opponent defensive features are calculated correctly.
    """
    print(f"\n{'='*60}")
    print(f"OPPONENT FEATURE VERIFICATION: {player_name}, Game {game_idx+1}")
    print(f"{'='*60}")
    
    raw = datasets['raw']
    features = datasets['features']
    
    # Get player's game
    feat_player = features[features['player'] == player_name].sort_values('game_date').reset_index(drop=True)
    
    if len(feat_player) <= game_idx:
        print(f"  Player doesn't have {game_idx+1} games")
        return False
    
    game = feat_player.iloc[game_idx]
    opponent = game['opp']
    game_date = game['game_date']
    
    print(f"\nPlayer game: {player_name} vs {opponent} on {game_date.date()}")
    print(f"Pipeline opp_pts_allowed_l10: {game['opp_pts_allowed_l10']}")
    
    # Manually calculate: what did this opponent allow in their last 10 games BEFORE this date?
    # Get all games where this team was playing (they are 'tm')
    opp_games = raw[raw['Tm'] == opponent].copy()
    opp_games = opp_games.sort_values('Data')
    
    # Get unique game dates for opponent
    opp_game_dates = opp_games.groupby('Data').first().reset_index()[['Data', 'Opp']]
    opp_game_dates = opp_game_dates[opp_game_dates['Data'] < game_date].tail(10)
    
    print(f"\nOpponent's last 10 games before {game_date.date()}:")
    
    if len(opp_game_dates) < 10:
        print(f"  Only {len(opp_game_dates)} prior games available")
    
    # For each of opponent's games, find what the OTHER team scored (that's what opponent allowed)
    pts_allowed_list = []
    for _, opp_game in opp_game_dates.iterrows():
        opp_date = opp_game['Data']
        other_team = opp_game['Opp']
        
        # Get other team's total points in that game
        other_team_pts = raw[(raw['Tm'] == other_team) & (raw['Data'] == opp_date)]['PTS'].sum()
        pts_allowed_list.append(other_team_pts)
        print(f"  {opp_date.date()}: vs {other_team} - allowed {other_team_pts} pts")
    
    if pts_allowed_list:
        manual_opp_pts_allowed = np.mean(pts_allowed_list)
        print(f"\nManual calculation: mean pts allowed = {manual_opp_pts_allowed:.1f}")
        print(f"Pipeline opp_pts_allowed_l10: {game['opp_pts_allowed_l10']:.1f}")
        
        # Allow some tolerance due to different window handling at season start
        if abs(manual_opp_pts_allowed - game['opp_pts_allowed_l10']) < 2.0:
            print(f"  ✓ CLOSE ENOUGH - Values within 2 points")
            return True
        else:
            print(f"  ⚠ DISCREPANCY - Investigate further")
            return False
    
    return True


def verify_schedule_features(datasets: dict, player_name: str):
    """
    Verify schedule features (days rest, B2B) are correct.
    """
    print(f"\n{'='*60}")
    print(f"SCHEDULE FEATURE VERIFICATION: {player_name}")
    print(f"{'='*60}")
    
    raw = datasets['raw']
    
    # Find player's team
    player_team = raw[raw['Player'] == player_name]['Tm'].mode().iloc[0]
    print(f"Player's team: {player_team}")
    
    # Get team's game dates from raw data
    team_games = raw[raw['Tm'] == player_team][['Data']].drop_duplicates().sort_values('Data').reset_index(drop=True)
    team_games['days_since_last'] = team_games['Data'].diff().dt.days
    
    print(f"\nTeam's first 10 games and rest days:")
    print(team_games.head(10).to_string())
    
    # Check against processed data
    # Load schedule data
    schedule = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
    sched_player = schedule[schedule['player'] == player_name].sort_values('game_date').reset_index(drop=True)
    
    print(f"\nPipeline schedule features for {player_name}:")
    print(sched_player[['game_date', 'days_rest', 'is_b2b_second']].head(10).to_string())
    
    # Verify a few specific values
    all_match = True
    for i in range(1, min(5, len(team_games))):
        raw_days = team_games.iloc[i]['days_since_last']
        
        # Find corresponding row in schedule data
        game_date = team_games.iloc[i]['Data']
        sched_row = sched_player[sched_player['game_date'] == game_date]
        
        if len(sched_row) > 0:
            pipeline_days = sched_row.iloc[0]['days_rest']
            
            if pd.notna(raw_days) and pd.notna(pipeline_days):
                if abs(raw_days - pipeline_days) < 0.01:
                    print(f"  Game {i+1} ({game_date.date()}): Raw={raw_days:.0f}, Pipeline={pipeline_days:.0f} ✓")
                else:
                    print(f"  Game {i+1} ({game_date.date()}): Raw={raw_days:.0f}, Pipeline={pipeline_days:.0f} ✗ MISMATCH")
                    all_match = False
    
    return all_match


def check_target_leakage_correlation(datasets: dict):
    """
    Final check: verify features don't have impossibly high correlation with targets.
    """
    print(f"\n{'='*60}")
    print("CORRELATION SANITY CHECK")
    print(f"{'='*60}")
    
    model = datasets['model']
    
    # Get feature columns (not identifiers or targets)
    id_cols = ['player', 'tm', 'opp', 'game_date', 'game_id', 'matchup_id']
    target_cols = ['pts', 'trb', 'ast']
    
    feature_cols = [c for c in model.columns if c not in id_cols and c not in target_cols]
    
    print(f"\nTop 10 correlations with PTS:")
    pts_corr = model[feature_cols + ['pts']].corr()['pts'].drop('pts').abs().sort_values(ascending=False)
    print(pts_corr.head(10).to_string())
    
    print(f"\nTop 10 correlations with TRB:")
    trb_corr = model[feature_cols + ['trb']].corr()['trb'].drop('trb').abs().sort_values(ascending=False)
    print(trb_corr.head(10).to_string())
    
    print(f"\nTop 10 correlations with AST:")
    ast_corr = model[feature_cols + ['ast']].corr()['ast'].drop('ast').abs().sort_values(ascending=False)
    print(ast_corr.head(10).to_string())
    
    # Check for suspicious correlations
    max_corr = max(pts_corr.max(), trb_corr.max(), ast_corr.max())
    print(f"\nMaximum correlation found: {max_corr:.4f}")
    
    if max_corr > 0.85:
        print("  ✗ SUSPICIOUS - Correlation too high, possible leakage!")
        return False
    elif max_corr > 0.75:
        print("  ⚠ WARNING - High correlation, verify this is expected")
        return True
    else:
        print("  ✓ PASS - Correlations are in realistic range")
        return True


def run_prediction_sanity_check(datasets: dict):
    """
    Quick sanity check: if we predict pts using pts_l10, what R² do we get?
    This should be 0.35-0.60 for a leak-free dataset.
    """
    print(f"\n{'='*60}")
    print("QUICK MODEL SANITY CHECK")
    print(f"{'='*60}")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error
    
    model_data = datasets['model'].dropna(subset=['pts', 'pts_l10', 'pts_l5'])
    
    X = model_data[['pts_l10']].values
    y = model_data['pts'].values
    
    # Simple train/test split (last 20% of data)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Fit simple model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nSimple model: PTS ~ pts_l10")
    print(f"  R² score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} points")
    
    if r2 > 0.80:
        print(f"  ✗ FAIL - R² too high ({r2:.2f}), LEAKAGE DETECTED!")
        return False
    elif r2 > 0.60:
        print(f"  ⚠ WARNING - R² higher than typical ({r2:.2f}), verify")
        return True
    elif r2 > 0.30:
        print(f"  ✓ PASS - R² in expected range (0.30-0.60)")
        return True
    else:
        print(f"  ⚠ NOTE - R² lower than expected, features may be weak")
        return True


def main():
    print("Task 5B: Rigorous Data Validation")
    print("="*60)
    print("This script MANUALLY verifies calculations against raw data.")
    print("="*60)
    
    # Load all datasets
    datasets = load_all_datasets()
    
    # Find a high-volume player to test (someone with lots of games)
    raw = datasets['raw']
    player_counts = raw['Player'].value_counts()
    test_players = player_counts.head(5).index.tolist()
    
    print(f"\nTest players (most games): {test_players}")
    
    all_passed = True
    
    # Test 1: Verify rolling averages for multiple players
    for player in test_players[:3]:
        if not verify_player_rolling_average(datasets, player):
            all_passed = False
    
    # Test 2: Verify opponent features
    for player in test_players[:2]:
        if not verify_opponent_features(datasets, player, game_idx=15):
            all_passed = False
    
    # Test 3: Verify schedule features
    if not verify_schedule_features(datasets, test_players[0]):
        all_passed = False
    
    # Test 4: Correlation sanity check
    if not check_target_leakage_correlation(datasets):
        all_passed = False
    
    # Test 5: Quick model sanity check
    if not run_prediction_sanity_check(datasets):
        all_passed = False
    
    # Final verdict
    print(f"\n{'='*60}")
    print("FINAL VALIDATION VERDICT")
    print(f"{'='*60}")
    
    if all_passed:
        print("\n✓ ALL RIGOROUS CHECKS PASSED")
        print("  Data pipeline is verified against ground truth.")
        print("  Safe to proceed to modeling.")
    else:
        print("\n✗ SOME CHECKS FAILED")
        print("  Review the failures above before proceeding.")
        print("  DO NOT train models on potentially corrupted data.")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
