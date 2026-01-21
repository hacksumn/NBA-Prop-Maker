"""
Task 2: Historical Rolling Features
NBA Player Props Betting System

Creates rolling averages using ONLY past games.
Critical: .shift(1) prevents data leakage.
"""

import pandas as pd
import numpy as np

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling average features for each player.
    
    CRITICAL: All rolling calculations use .shift(1) to exclude current game.
    This means for game N, we only use data from games 1 to N-1.
    
    Returns:
        DataFrame with rolling features added
    """
    # Sort by player and date to ensure correct temporal order
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    
    # Stats to calculate rolling averages for
    rolling_stats = ['pts', 'trb', 'ast', 'mp', 'fga', 'fg_pct', 
                     '3pa', '3p_pct', 'fta', 'ft_pct', 'tov', 'stl', 'blk']
    
    # Windows for rolling calculations
    windows = [5, 10, 20]
    
    # Stats to calculate volatility (std dev) for - just the main props
    volatility_stats = ['pts', 'trb', 'ast', 'mp']
    
    print("Creating rolling features...")
    print("This may take a minute for 16k+ rows...")
    
    # Group by player for all calculations
    grouped = df.groupby('player')
    
    # 1. Create games played counter (before current game)
    df['games_played'] = grouped.cumcount()  # 0-indexed, so first game = 0
    print(f"  ✓ Games played counter")
    
    # 2. Create shifted versions of stats (THIS PREVENTS LEAKAGE)
    # We shift first, then calculate rolling stats on the shifted values
    for stat in rolling_stats:
        df[f'{stat}_shifted'] = grouped[stat].shift(1)
    print(f"  ✓ Shifted stats (leakage prevention)")
    
    # 3. Calculate rolling means on shifted data
    for window in windows:
        for stat in rolling_stats:
            col_name = f'{stat}_l{window}'
            # Rolling mean on the SHIFTED column
            df[col_name] = grouped[f'{stat}_shifted'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        print(f"  ✓ L{window} rolling means")
    
    # 4. Calculate rolling standard deviations (volatility) on shifted data
    for stat in volatility_stats:
        for window in [10, 20]:  # Only L10 and L20 for volatility
            col_name = f'{stat}_std_l{window}'
            df[col_name] = grouped[f'{stat}_shifted'].transform(
                lambda x: x.rolling(window, min_periods=3).std()
            )
    print(f"  ✓ Volatility features (std dev)")
    
    # 5. Calculate trend features (L5 vs L20 ratio) - shows hot/cold streaks
    for stat in ['pts', 'trb', 'ast']:
        l5_col = f'{stat}_l5'
        l20_col = f'{stat}_l20'
        trend_col = f'{stat}_trend'
        # Ratio > 1 means player is trending up, < 1 trending down
        df[trend_col] = df[l5_col] / df[l20_col].replace(0, np.nan)
    print(f"  ✓ Trend features (L5/L20 ratio)")
    
    # 6. Drop the intermediate shifted columns (we don't need them in final output)
    shifted_cols = [f'{stat}_shifted' for stat in rolling_stats]
    df = df.drop(columns=shifted_cols)
    
    return df


def validate_no_leakage(df: pd.DataFrame) -> bool:
    """
    Validate that rolling features don't include current game data.
    
    Test: For each player's first game, rolling features should be NaN.
    Test: For second game, L5 average should equal first game's actual value.
    """
    print("\n" + "="*60)
    print("LEAKAGE VALIDATION")
    print("="*60)
    
    all_passed = True
    
    # Test 1: First game should have NaN for rolling features (no history)
    first_games = df[df['games_played'] == 0]
    first_game_pts_l5_nulls = first_games['pts_l5'].isna().sum()
    first_game_total = len(first_games)
    
    # Note: min_periods=1 means we get values even with 1 game, but shifted
    # So first game (games_played=0) should have NaN because shift(1) on first game = NaN
    print(f"\n[Test 1] First games have NaN rolling features:")
    print(f"  First games with NaN pts_l5: {first_game_pts_l5_nulls}/{first_game_total}")
    if first_game_pts_l5_nulls == first_game_total:
        print(f"  ✓ PASS - No leakage on first games")
    else:
        print(f"  ✗ FAIL - Some first games have rolling data (LEAKAGE!)")
        all_passed = False
    
    # Test 2: Verify second game's L5 equals first game's actual value
    # Pick a player with multiple games to verify
    test_player = df.groupby('player').size().idxmax()  # Player with most games
    player_df = df[df['player'] == test_player].sort_values('game_date').head(5)
    
    print(f"\n[Test 2] Manual inspection for '{test_player}':")
    print(player_df[['game_date', 'pts', 'games_played', 'pts_l5', 'pts_l10']].to_string())
    
    # Check: game 2's pts_l5 should equal game 1's pts
    if len(player_df) >= 2:
        game1_pts = player_df.iloc[0]['pts']
        game2_pts_l5 = player_df.iloc[1]['pts_l5']
        
        print(f"\n  Game 1 actual PTS: {game1_pts}")
        print(f"  Game 2 pts_l5: {game2_pts_l5}")
        
        if abs(game1_pts - game2_pts_l5) < 0.01:
            print(f"  ✓ PASS - L5 correctly uses only past games")
        else:
            print(f"  ✗ FAIL - L5 doesn't match expected value")
            all_passed = False
    
    # Test 3: Correlation check - pts_l10 should NOT have r > 0.9 with pts
    # High correlation suggests leakage
    corr = df[['pts', 'pts_l10']].dropna().corr().iloc[0, 1]
    print(f"\n[Test 3] Correlation between actual PTS and pts_l10:")
    print(f"  Correlation: {corr:.4f}")
    if corr < 0.85:
        print(f"  ✓ PASS - Correlation is reasonable (no obvious leakage)")
    else:
        print(f"  ⚠ WARNING - High correlation may indicate leakage")
        # Not failing here because high correlation is possible for consistent players
    
    print("\n" + "="*60)
    if all_passed:
        print("STATUS: LEAKAGE VALIDATION PASSED ✓")
    else:
        print("STATUS: POTENTIAL LEAKAGE DETECTED - INVESTIGATE!")
    print("="*60)
    
    return all_passed


def summarize_features(df: pd.DataFrame):
    """Print summary of created features."""
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    
    # Count feature types
    rolling_mean_cols = [c for c in df.columns if '_l5' in c or '_l10' in c or '_l20' in c]
    rolling_mean_cols = [c for c in rolling_mean_cols if '_std_' not in c]
    std_cols = [c for c in df.columns if '_std_' in c]
    trend_cols = [c for c in df.columns if '_trend' in c]
    
    print(f"\nRolling Mean Features: {len(rolling_mean_cols)}")
    print(f"  {rolling_mean_cols[:5]}...")
    
    print(f"\nVolatility Features: {len(std_cols)}")
    print(f"  {std_cols}")
    
    print(f"\nTrend Features: {len(trend_cols)}")
    print(f"  {trend_cols}")
    
    print(f"\nTotal New Features: {len(rolling_mean_cols) + len(std_cols) + len(trend_cols) + 1}")
    
    # Show sample of key features
    print("\n[SAMPLE - Key features for first 5 rows with sufficient history]:")
    sample = df[df['games_played'] >= 10].head(5)
    display_cols = ['player', 'game_date', 'pts', 'pts_l5', 'pts_l10', 'pts_l20', 
                    'pts_std_l10', 'pts_trend', 'games_played']
    print(sample[display_cols].to_string())
    
    # Show null counts
    print("\n[NULL COUNTS - Expected for early games]:")
    null_counts = df[rolling_mean_cols[:5]].isna().sum()
    print(null_counts.to_string())


def main():
    # File paths
    input_path = 'player_games_clean.csv'
    output_path = 'player_games_features.csv'
    
    print("Task 2: Historical Rolling Features")
    print("-" * 40)
    
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv(input_path, parse_dates=['game_date'])
    print(f"Loaded {len(df):,} rows")
    
    # Create rolling features
    df = create_rolling_features(df)
    
    # Validate no leakage
    passed = validate_no_leakage(df)
    
    # Summarize
    summarize_features(df)
    
    if passed:
        # Save
        df.to_csv(output_path, index=False)
        print(f"\n✓ Features saved to: {output_path}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Total rows: {len(df):,}")
    else:
        print("\n✗ Fix leakage issues before saving!")


if __name__ == "__main__":
    main()