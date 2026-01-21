"""
Task 1: Data Foundation & Cleaning
NBA Player Props Betting System

This script cleans and standardizes the raw NBA data.
NO feature engineering here - just clean foundation.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data and perform cleaning operations.
    
    Returns:
        Clean DataFrame with standardized schema
    """
    # Load raw data
    df = pd.read_csv(filepath)
    
    # 1. Standardize column names (lowercase, underscores)
    df.columns = df.columns.str.lower().str.replace('%', '_pct').str.replace(' ', '_')
    
    # Rename 'data' to 'game_date' (more descriptive)
    df = df.rename(columns={'data': 'game_date'})
    
    # 2. Parse dates properly
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # 3. Create unique player-game identifier
    # Format: YYYYMMDD_PLAYER_TEAM_OPP
    df['game_id'] = (
        df['game_date'].dt.strftime('%Y%m%d') + '_' +
        df['player'].str.replace(' ', '_') + '_' +
        df['tm'] + '_' +
        df['opp']
    )
    
    # 4. Derive home/away indicator
    # Logic: In NBA data, home team is typically listed second in matchup
    # We need to infer from the data structure
    # Since each game has both teams' players, we can identify home/away
    # by checking which team appears as 'opp' for the other team's players
    
    # Create a game identifier (date + teams involved, sorted alphabetically)
    df['matchup_id'] = df.apply(
        lambda x: x['game_date'].strftime('%Y%m%d') + '_' + '_'.join(sorted([x['tm'], x['opp']])),
        axis=1
    )
    
    # For now, we'll mark home/away as unknown - we'd need schedule data to confirm
    # This is HONEST - we don't fabricate what we don't know
    df['is_home'] = np.nan  # Will be filled in Task 2 if we get schedule data
    
    # 5. Convert result to binary
    df['won'] = (df['res'] == 'W').astype(int)
    
    # 6. Ensure numeric columns are proper types
    numeric_cols = ['mp', 'fg', 'fga', 'fg_pct', '3p', '3pa', '3p_pct', 
                    'ft', 'fta', 'ft_pct', 'orb', 'drb', 'trb', 'ast', 
                    'stl', 'blk', 'tov', 'pf', 'pts', 'gmsc']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 7. Sort by date and player for consistent ordering
    df = df.sort_values(['game_date', 'player']).reset_index(drop=True)
    
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Run validation checks on cleaned data.
    
    Returns:
        Dictionary of validation results
    """
    checks = {}
    
    # Check 1: No duplicate player-games
    checks['duplicate_games'] = df['game_id'].duplicated().sum()
    
    # Check 2: All dates parsed correctly
    checks['null_dates'] = df['game_date'].isnull().sum()
    
    # Check 3: Points should be non-negative
    checks['negative_pts'] = (df['pts'] < 0).sum()
    
    # Check 4: Minutes should be 0-60 (with some OT allowance)
    checks['invalid_minutes'] = ((df['mp'] < 0) | (df['mp'] > 70)).sum()
    
    # Check 5: FG% should be 0-1 or NaN (for 0 attempts)
    checks['invalid_fg_pct'] = ((df['fg_pct'] < 0) | (df['fg_pct'] > 1)).sum()
    
    # Check 6: Expected number of teams
    checks['team_count'] = df['tm'].nunique()
    
    # Check 7: Date range
    checks['date_min'] = df['game_date'].min()
    checks['date_max'] = df['game_date'].max()
    checks['total_games'] = len(df)
    checks['unique_players'] = df['player'].nunique()
    
    return checks


def print_validation_report(checks: dict):
    """Print formatted validation report."""
    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60)
    
    # Critical checks (should all be 0)
    critical_checks = ['duplicate_games', 'null_dates', 'negative_pts', 
                       'invalid_minutes', 'invalid_fg_pct']
    
    print("\n[CRITICAL CHECKS] - Should all be 0:")
    all_passed = True
    for check in critical_checks:
        status = "✓ PASS" if checks[check] == 0 else "✗ FAIL"
        if checks[check] != 0:
            all_passed = False
        print(f"  {check}: {checks[check]} {status}")
    
    print("\n[DATA SUMMARY]:")
    print(f"  Total player-games: {checks['total_games']:,}")
    print(f"  Unique players: {checks['unique_players']}")
    print(f"  Teams: {checks['team_count']}")
    print(f"  Date range: {checks['date_min'].date()} to {checks['date_max'].date()}")
    
    print("\n" + "="*60)
    if all_passed:
        print("STATUS: ALL CRITICAL CHECKS PASSED ✓")
    else:
        print("STATUS: SOME CHECKS FAILED - INVESTIGATE BEFORE PROCEEDING")
    print("="*60 + "\n")
    
    return all_passed


def main():
    # File paths
    input_path = 'database_24_25.csv'
    output_path = 'player_games_clean.csv'
    
    print("Task 1: Data Foundation & Cleaning")
    print("-" * 40)
    
    # Load and clean
    print("Loading and cleaning data...")
    df = load_and_clean_data(input_path)
    
    # Validate
    print("Running validation checks...")
    checks = validate_data(df)
    passed = print_validation_report(checks)
    
    if passed:
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        # Show sample of output
        print("\n[SAMPLE OUTPUT - First 5 rows, key columns]:")
        print(df[['game_date', 'player', 'tm', 'opp', 'won', 'mp', 'pts', 'trb', 'ast', 'game_id']].head())
        
        print("\n[COLUMN SCHEMA]:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
    else:
        print("Fix validation errors before saving.")
        

if __name__ == "__main__":
    main()