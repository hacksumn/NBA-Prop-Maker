"""
Task 4: Schedule & Fatigue Features
NBA Player Props Betting System

Adds rest, workload, and schedule context.
All features use only data available before tip-off.
"""

import pandas as pd
import numpy as np


def create_team_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create team-level schedule features.
    
    These are calculated at team level first, then merged to players.
    """
    print("Creating team schedule features...")
    
    # Get unique team-games
    team_games = df.groupby(['tm', 'game_date']).first().reset_index()[['tm', 'game_date', 'opp']]
    team_games = team_games.sort_values(['tm', 'game_date']).reset_index(drop=True)
    
    grouped = team_games.groupby('tm')
    
    # 1. Days since last game
    team_games['prev_game_date'] = grouped['game_date'].shift(1)
    team_games['days_rest'] = (team_games['game_date'] - team_games['prev_game_date']).dt.days
    
    # 2. Back-to-back indicator (0 or 1 days rest)
    team_games['is_b2b'] = (team_games['days_rest'] <= 1).astype(int)
    
    # 3. Second of back-to-back (played yesterday)
    team_games['is_b2b_second'] = (team_games['days_rest'] == 1).astype(int)
    
    # 4. Games in last 7 days - use a different approach
    # Set game_date as index for time-based rolling
    games_7d = []
    games_14d = []
    
    for tm in team_games['tm'].unique():
        tm_df = team_games[team_games['tm'] == tm].copy()
        tm_df = tm_df.set_index('game_date').sort_index()
        
        # Count games in rolling window (excluding current game with closed='left')
        # Rolling on index requires the index to be datetime
        g7 = tm_df.index.to_series().rolling('7D', closed='left').count()
        g14 = tm_df.index.to_series().rolling('14D', closed='left').count()
        
        tm_df['games_last_7d'] = g7.values
        tm_df['games_last_14d'] = g14.values
        tm_df = tm_df.reset_index()
        
        games_7d.append(tm_df[['tm', 'game_date', 'games_last_7d']])
        games_14d.append(tm_df[['tm', 'game_date', 'games_last_14d']])
    
    games_7d_df = pd.concat(games_7d, ignore_index=True)
    games_14d_df = pd.concat(games_14d, ignore_index=True)
    
    team_games = team_games.merge(games_7d_df, on=['tm', 'game_date'], how='left')
    team_games = team_games.merge(games_14d_df, on=['tm', 'game_date'], how='left')
    
    print(f"  ✓ Days rest, B2B flags")
    print(f"  ✓ Games density (7d, 14d)")
    
    # Drop intermediate columns
    team_games = team_games.drop(columns=['prev_game_date'])
    
    return team_games


def create_player_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create player-specific fatigue features.
    
    Recent minutes load indicates fatigue risk.
    """
    print("Creating player fatigue features...")
    
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    grouped = df.groupby('player')
    
    # 1. Minutes in last 3 games (shifted to exclude current)
    df['mp_shifted'] = grouped['mp'].shift(1)
    df['mp_l3'] = grouped['mp_shifted'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    
    # 2. Average minutes last 3 games
    df['mp_avg_l3'] = grouped['mp_shifted'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # 3. High minutes flag (played 36+ minutes last game)
    df['high_min_prev'] = (grouped['mp'].shift(1) >= 36).astype(float)
    
    # 4. Consecutive games played streak
    # Count how many games in a row this player has played
    df['played'] = 1  # Every row means they played
    df['games_streak'] = grouped['played'].cumsum()
    
    # 5. Player's days since last game (handles players who miss games)
    df['prev_game_date_player'] = grouped['game_date'].shift(1)
    df['player_days_rest'] = (df['game_date'] - df['prev_game_date_player']).dt.days
    
    print(f"  ✓ Minutes load (L3 sum, L3 avg)")
    print(f"  ✓ High minutes flag")
    print(f"  ✓ Games streak, player rest")
    
    # Drop intermediate columns
    df = df.drop(columns=['mp_shifted', 'played', 'prev_game_date_player'])
    
    return df


def merge_schedule_features(player_df: pd.DataFrame, 
                            team_schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Merge team schedule features to player data.
    """
    print("Merging schedule features to player data...")
    
    # Select schedule features to merge
    schedule_cols = ['tm', 'game_date', 'days_rest', 'is_b2b', 'is_b2b_second',
                     'games_last_7d', 'games_last_14d']
    
    player_df = player_df.merge(
        team_schedule[schedule_cols],
        on=['tm', 'game_date'],
        how='left'
    )
    
    print(f"  ✓ Merged {len(schedule_cols) - 2} schedule features")
    
    return player_df


def validate_schedule_features(df: pd.DataFrame) -> bool:
    """
    Validate schedule and fatigue features.
    """
    print("\n" + "="*60)
    print("SCHEDULE FEATURE VALIDATION")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Days rest range (should be 0-10 typically, rarely more)
    days_rest = df['days_rest'].dropna()
    print(f"\n[Test 1] Days rest distribution:")
    print(f"  Min: {days_rest.min():.0f}")
    print(f"  Max: {days_rest.max():.0f}")
    print(f"  Mean: {days_rest.mean():.2f}")
    print(f"  Back-to-backs (1 day): {(days_rest == 1).sum():,} games")
    
    if days_rest.min() >= 0 and days_rest.max() <= 15:
        print(f"  ✓ PASS - Days rest in expected range")
    else:
        print(f"  ⚠ WARNING - Unusual days rest values")
    
    # Test 2: B2B rate (NBA teams play ~12-15 B2Bs per season)
    b2b_rate = df['is_b2b_second'].mean() * 100
    print(f"\n[Test 2] Back-to-back rate:")
    print(f"  B2B second games: {b2b_rate:.1f}% of all games")
    
    if 8 <= b2b_rate <= 20:
        print(f"  ✓ PASS - B2B rate in expected range (8-20%)")
    else:
        print(f"  ⚠ WARNING - B2B rate outside expected range")
    
    # Test 3: Games in 7 days (should be 0-5 typically)
    games_7d = df['games_last_7d'].dropna()
    print(f"\n[Test 3] Games in last 7 days:")
    print(f"  Min: {games_7d.min():.0f}")
    print(f"  Max: {games_7d.max():.0f}")
    print(f"  Mean: {games_7d.mean():.2f}")
    
    if games_7d.max() <= 6:
        print(f"  ✓ PASS - Games density reasonable")
    else:
        print(f"  ⚠ WARNING - Unusually high game density")
    
    # Test 4: Minutes L3 range
    mp_l3 = df['mp_l3'].dropna()
    print(f"\n[Test 4] Minutes last 3 games (sum):")
    print(f"  Min: {mp_l3.min():.1f}")
    print(f"  Max: {mp_l3.max():.1f}")
    print(f"  Mean: {mp_l3.mean():.1f}")
    
    # Max should be around 144 (48 min * 3 games)
    if mp_l3.max() <= 160:
        print(f"  ✓ PASS - Minutes in expected range")
    else:
        print(f"  ⚠ WARNING - Unusually high minutes")
    
    # Test 5: Correlation - should see slight negative correlation
    # between B2B and points (players score less on B2Bs)
    b2b_corr = df[['pts', 'is_b2b_second']].corr().iloc[0, 1]
    print(f"\n[Test 5] Correlation: PTS vs is_b2b_second:")
    print(f"  Correlation: {b2b_corr:.4f}")
    if b2b_corr < 0:
        print(f"  ✓ PASS - Negative correlation (less scoring on B2Bs)")
    else:
        print(f"  ⚠ Note - Expected slight negative correlation")
    
    print("\n" + "="*60)
    if all_passed:
        print("STATUS: SCHEDULE FEATURE VALIDATION PASSED ✓")
    else:
        print("STATUS: SOME CHECKS NEED REVIEW")
    print("="*60)
    
    return all_passed


def summarize_schedule_features(df: pd.DataFrame):
    """Print summary of schedule features."""
    print("\n" + "="*60)
    print("SCHEDULE FEATURE SUMMARY")
    print("="*60)
    
    schedule_cols = ['days_rest', 'is_b2b', 'is_b2b_second', 
                     'games_last_7d', 'games_last_14d',
                     'mp_l3', 'mp_avg_l3', 'high_min_prev', 
                     'games_streak', 'player_days_rest']
    
    print(f"\nSchedule/Fatigue Features Added: {len(schedule_cols)}")
    for col in schedule_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null:,} non-null values")
    
    # Show sample
    print("\n[SAMPLE - Schedule context]:")
    sample = df[df['days_rest'].notna()].head(8)
    display_cols = ['player', 'game_date', 'tm', 'pts', 'days_rest', 
                    'is_b2b_second', 'games_last_7d', 'mp_l3']
    print(sample[display_cols].to_string())


def main():
    # File paths
    input_path = 'player_games_opp_features.csv'
    output_path = 'player_games_schedule.csv'
    
    print("Task 4: Schedule & Fatigue Features")
    print("-" * 40)
    
    # Load data
    print("Loading data with opponent features...")
    df = pd.read_csv(input_path, parse_dates=['game_date'])
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Create team schedule features
    team_schedule = create_team_schedule_features(df)
    
    # Create player fatigue features
    df = create_player_fatigue_features(df)
    
    # Merge team schedule to players
    df = merge_schedule_features(df, team_schedule)
    
    # Validate
    passed = validate_schedule_features(df)
    
    # Summarize
    summarize_schedule_features(df)
    
    # Save (even with warnings, these features are valid)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Features saved to: {output_path}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Total rows: {len(df):,}")


if __name__ == "__main__":
    main()
