"""
Task 3: Opponent Defensive Features
NBA Player Props Betting System

Adds opponent context: how many points/rebounds/assists does this
opponent typically ALLOW? Uses only data available before tip-off.
"""

import pandas as pd
import numpy as np


def create_team_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player stats to team-game level.
    
    For each team-game, calculate total points scored, allowed, pace indicators.
    """
    print("Creating team-game aggregations...")
    
    # Group by team and game date to get team totals
    team_games = df.groupby(['tm', 'game_date', 'opp']).agg({
        'pts': 'sum',      # Team total points scored
        'trb': 'sum',      # Team total rebounds
        'ast': 'sum',      # Team total assists
        'fga': 'sum',      # Team field goal attempts (pace indicator)
        '3pa': 'sum',      # Team 3-point attempts
        'tov': 'sum',      # Team turnovers
        'mp': 'sum',       # Total minutes (should be ~240 for regulation)
        'player': 'count'  # Players who played (rotation size)
    }).reset_index()
    
    team_games = team_games.rename(columns={
        'pts': 'tm_pts',
        'trb': 'tm_trb', 
        'ast': 'tm_ast',
        'fga': 'tm_fga',
        '3pa': 'tm_3pa',
        'tov': 'tm_tov',
        'mp': 'tm_mp',
        'player': 'tm_players_used'
    })
    
    print(f"  ✓ Created {len(team_games):,} team-game records")
    
    return team_games


def create_opponent_defensive_stats(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate what each team ALLOWS (opponent's offensive stats against them).
    
    This requires matching each game with the opponent's stats from that same game.
    """
    print("Calculating opponent defensive stats...")
    
    # Create a copy with opponent perspective
    # For each game, find what the opponent scored (which is what this team allowed)
    opp_stats = team_games[['tm', 'game_date', 'tm_pts', 'tm_trb', 'tm_ast', 'tm_fga']].copy()
    opp_stats = opp_stats.rename(columns={
        'tm': 'opp',  # This team becomes the opponent
        'tm_pts': 'pts_allowed',
        'tm_trb': 'trb_allowed',
        'tm_ast': 'ast_allowed',
        'tm_fga': 'fga_allowed'  # Pace indicator
    })
    
    # Merge back: for each team-game, get what they allowed (opponent's stats)
    team_games = team_games.merge(
        opp_stats,
        on=['opp', 'game_date'],
        how='left'
    )
    
    print(f"  ✓ Merged opponent stats")
    
    return team_games


def create_rolling_defensive_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling defensive averages for each team.
    
    CRITICAL: Use .shift(1) to only include PAST games, not current game.
    """
    print("Creating rolling defensive features...")
    
    # Sort by team and date
    team_games = team_games.sort_values(['tm', 'game_date']).reset_index(drop=True)
    
    # Group by team
    grouped = team_games.groupby('tm')
    
    # Defensive stats to roll (what they allow)
    def_stats = ['pts_allowed', 'trb_allowed', 'ast_allowed', 'fga_allowed']
    
    # Offensive stats to roll (what they score - useful for total predictions)
    off_stats = ['tm_pts', 'tm_trb', 'tm_ast', 'tm_fga']
    
    all_stats = def_stats + off_stats
    
    # Shift first (CRITICAL for no leakage)
    for stat in all_stats:
        team_games[f'{stat}_shifted'] = grouped[stat].shift(1)
    
    # Calculate L10 rolling averages
    window = 10
    for stat in all_stats:
        col_name = f'{stat}_l{window}'
        team_games[col_name] = grouped[f'{stat}_shifted'].transform(
            lambda x: x.rolling(window, min_periods=3).mean()
        )
    
    print(f"  ✓ L10 rolling defensive averages")
    
    # Calculate defensive rank features (relative to league average)
    # This helps normalize across different points in season
    for stat in def_stats:
        l10_col = f'{stat}_l10'
        rank_col = f'{stat}_rank'
        # Lower rank = better defense (allows fewer points)
        team_games[rank_col] = team_games.groupby('game_date')[l10_col].rank(method='average')
    
    print(f"  ✓ Defensive rankings")
    
    # Drop shifted columns
    shifted_cols = [f'{stat}_shifted' for stat in all_stats]
    team_games = team_games.drop(columns=shifted_cols)
    
    return team_games


def merge_opponent_features_to_players(player_df: pd.DataFrame, 
                                        team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Merge opponent defensive features back to player-game level.
    
    For each player-game, we add the opponent's defensive tendencies
    ENTERING that game (not including that game's results).
    """
    print("Merging opponent features to player data...")
    
    # Select only the features we want to add
    opp_features = [
        'pts_allowed_l10', 'trb_allowed_l10', 'ast_allowed_l10', 'fga_allowed_l10',
        'pts_allowed_rank', 'trb_allowed_rank', 'ast_allowed_rank',
        'tm_pts_l10', 'tm_fga_l10'  # Team's own offensive tendencies
    ]
    
    # Create opponent feature set (keyed by opponent and date)
    opp_feature_df = team_games[['tm', 'game_date'] + opp_features].copy()
    
    # Rename 'tm' to 'opp' for merging (these become opponent features)
    opp_feature_df = opp_feature_df.rename(columns={'tm': 'opp'})
    
    # Add 'opp_' prefix to feature names
    rename_dict = {col: f'opp_{col}' for col in opp_features}
    opp_feature_df = opp_feature_df.rename(columns=rename_dict)
    
    # Merge to player data
    player_df = player_df.merge(
        opp_feature_df,
        on=['opp', 'game_date'],
        how='left'
    )
    
    print(f"  ✓ Added {len(opp_features)} opponent features")
    
    return player_df


def validate_opponent_features(df: pd.DataFrame) -> bool:
    """
    Validate opponent features are properly constructed.
    """
    print("\n" + "="*60)
    print("OPPONENT FEATURE VALIDATION")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Check for reasonable ranges
    pts_allowed = df['opp_pts_allowed_l10'].dropna()
    print(f"\n[Test 1] Points allowed L10 range:")
    print(f"  Min: {pts_allowed.min():.1f}")
    print(f"  Max: {pts_allowed.max():.1f}")
    print(f"  Mean: {pts_allowed.mean():.1f}")
    
    # NBA teams score roughly 100-130 points per game
    if pts_allowed.min() >= 90 and pts_allowed.max() <= 140:
        print(f"  ✓ PASS - Values in expected NBA range")
    else:
        print(f"  ⚠ WARNING - Values outside typical range")
    
    # Test 2: Check defensive rank range (should be 1-30)
    rank_col = df['opp_pts_allowed_rank'].dropna()
    print(f"\n[Test 2] Defensive rank range:")
    print(f"  Min: {rank_col.min():.1f}")
    print(f"  Max: {rank_col.max():.1f}")
    
    if rank_col.min() >= 1 and rank_col.max() <= 30:
        print(f"  ✓ PASS - Ranks in expected range (1-30)")
    else:
        print(f"  ✗ FAIL - Ranks outside expected range")
        all_passed = False
    
    # Test 3: Check correlation between opp defense and player scoring
    # Players SHOULD score more against bad defenses (higher pts_allowed)
    corr_df = df[['pts', 'opp_pts_allowed_l10']].dropna()
    corr = corr_df.corr().iloc[0, 1]
    print(f"\n[Test 3] Correlation: player PTS vs opponent pts_allowed_l10:")
    print(f"  Correlation: {corr:.4f}")
    if corr > 0:
        print(f"  ✓ PASS - Positive correlation (players score more vs bad D)")
    else:
        print(f"  ⚠ WARNING - Expected positive correlation")
    
    # Test 4: Null counts (early games won't have opponent history)
    null_count = df['opp_pts_allowed_l10'].isna().sum()
    null_pct = null_count / len(df) * 100
    print(f"\n[Test 4] Null opponent features:")
    print(f"  Nulls: {null_count:,} ({null_pct:.1f}%)")
    if null_pct < 20:
        print(f"  ✓ PASS - Acceptable null rate for early-season games")
    else:
        print(f"  ⚠ WARNING - High null rate")
    
    print("\n" + "="*60)
    if all_passed:
        print("STATUS: OPPONENT FEATURE VALIDATION PASSED ✓")
    else:
        print("STATUS: SOME CHECKS FAILED - INVESTIGATE")
    print("="*60)
    
    return all_passed


def summarize_opponent_features(df: pd.DataFrame):
    """Print summary of opponent features."""
    print("\n" + "="*60)
    print("OPPONENT FEATURE SUMMARY")
    print("="*60)
    
    opp_cols = [c for c in df.columns if c.startswith('opp_')]
    print(f"\nOpponent Features Added: {len(opp_cols)}")
    for col in opp_cols:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,} non-null values")
    
    # Show sample with opponent features
    print("\n[SAMPLE - Player vs Opponent Defense]:")
    sample = df[df['opp_pts_allowed_l10'].notna()].head(5)
    display_cols = ['player', 'game_date', 'opp', 'pts', 'pts_l10', 
                    'opp_pts_allowed_l10', 'opp_pts_allowed_rank']
    print(sample[display_cols].to_string())


def main():
    # File paths
    input_path = 'player_games_features.csv'
    output_path = 'player_games_opp_features.csv'
    
    print("Task 3: Opponent Defensive Features")
    print("-" * 40)
    
    # Load player features
    print("Loading player features...")
    df = pd.read_csv(input_path, parse_dates=['game_date'])
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Create team-game aggregations
    team_games = create_team_game_stats(df)
    
    # Add opponent stats (what each team allowed)
    team_games = create_opponent_defensive_stats(team_games)
    
    # Create rolling defensive features
    team_games = create_rolling_defensive_features(team_games)
    
    # Merge back to player level
    df = merge_opponent_features_to_players(df, team_games)
    
    # Validate
    passed = validate_opponent_features(df)
    
    # Summarize
    summarize_opponent_features(df)
    
    if passed:
        # Save
        df.to_csv(output_path, index=False)
        print(f"\n✓ Features saved to: {output_path}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Total rows: {len(df):,}")
    else:
        print("\n⚠ Validation warnings - review before proceeding")
        df.to_csv(output_path, index=False)
        print(f"  Saved anyway: {output_path}")


if __name__ == "__main__":
    main()
