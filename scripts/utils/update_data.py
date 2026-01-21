"""
NBA Data Updater
Automatically fetches player game logs from NBA.com using nba_api (FREE)

Installation:
    pip install nba_api pandas

Usage:
    python update_data.py                    # Update with recent games
    python update_data.py --full             # Full season download
    python update_data.py --days 7           # Last 7 days only
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import argparse
from pathlib import Path

# Install nba_api if not present
try:
    from nba_api.stats.endpoints import leaguegamelog, playergamelog
    from nba_api.stats.static import players, teams
except ImportError:
    print("Installing nba_api...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nba_api'])
    from nba_api.stats.endpoints import leaguegamelog, playergamelog
    from nba_api.stats.static import players, teams


def get_all_player_game_logs(season: str = '2024-25', 
                              season_type: str = 'Regular Season') -> pd.DataFrame:
    """
    Fetch all player game logs for a season.
    
    Args:
        season: NBA season (e.g., '2024-25')
        season_type: 'Regular Season', 'Playoffs', 'Pre Season'
    
    Returns:
        DataFrame with all player box scores
    """
    print(f"Fetching {season} {season_type} game logs...")
    print("(This may take 30-60 seconds due to NBA.com rate limits)")
    
    try:
        # LeagueGameLog gets all player stats for all games
        game_log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation='P',  # P = Players
            timeout=60
        )
        
        df = game_log.get_data_frames()[0]
        print(f"  ✓ Fetched {len(df):,} player-game records")
        return df
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def transform_to_project_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform NBA.com format to match our project's database format.
    
    NBA.com columns -> Our columns
    """
    print("Transforming to project format...")
    
    # Create mapping
    column_map = {
        'PLAYER_NAME': 'Player',
        'TEAM_ABBREVIATION': 'Tm',
        'MATCHUP': 'matchup_raw',  # Will parse this
        'GAME_DATE': 'Data',
        'WL': 'Res',
        'MIN': 'MP',
        'FGM': 'FG',
        'FGA': 'FGA',
        'FG_PCT': 'FG%',
        'FG3M': '3P',
        'FG3A': '3PA',
        'FG3_PCT': '3P%',
        'FTM': 'FT',
        'FTA': 'FTA',
        'FT_PCT': 'FT%',
        'OREB': 'ORB',
        'DREB': 'DRB',
        'REB': 'TRB',
        'AST': 'AST',
        'STL': 'STL',
        'BLK': 'BLK',
        'TOV': 'TOV',
        'PF': 'PF',
        'PTS': 'PTS',
        'PLUS_MINUS': 'plus_minus'
    }
    
    # Rename columns
    df = df.rename(columns=column_map)
    
    # Parse opponent from matchup (e.g., "LAL vs. GSW" or "LAL @ GSW")
    def parse_opponent(matchup, team):
        if pd.isna(matchup):
            return None
        parts = matchup.replace('.', '').split()
        if len(parts) >= 3:
            # "LAL vs GSW" or "LAL @ GSW"
            if parts[1] in ['vs', '@']:
                return parts[2]
        return None
    
    df['Opp'] = df.apply(lambda x: parse_opponent(x.get('matchup_raw'), x.get('Tm')), axis=1)
    
    # Convert result (W/L)
    df['Res'] = df['Res'].apply(lambda x: 'W' if x == 'W' else 'L' if x == 'L' else x)
    
    # Calculate game score (GmSc = PTS + 0.4*FG - 0.7*FGA - 0.4*(FTA-FT) + 0.7*ORB + 0.3*DRB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV)
    df['GmSc'] = (
        df['PTS'] + 0.4 * df['FG'] - 0.7 * df['FGA'] 
        - 0.4 * (df['FTA'] - df['FT']) + 0.7 * df['ORB'] + 0.3 * df['DRB']
        + df['STL'] + 0.7 * df['AST'] + 0.7 * df['BLK'] 
        - 0.4 * df['PF'] - df['TOV']
    ).round(1)
    
    # Select and order columns to match project format
    output_cols = [
        'Player', 'Tm', 'Opp', 'Res', 'MP', 
        'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
        'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', 'Data'
    ]
    
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols].copy()
    
    # Sort by date and player
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(['Data', 'Player']).reset_index(drop=True)
    
    print(f"  ✓ Transformed {len(df):,} records")
    print(f"  ✓ Date range: {df['Data'].min().date()} to {df['Data'].max().date()}")
    print(f"  ✓ Players: {df['Player'].nunique()}")
    print(f"  ✓ Teams: {df['Tm'].nunique()}")
    
    return df


def update_existing_data(existing_path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new data with existing data, avoiding duplicates.
    """
    print(f"Merging with existing data from {existing_path}...")
    
    if not Path(existing_path).exists():
        print("  No existing file, using new data only")
        return new_df
    
    existing = pd.read_csv(existing_path)
    existing['Data'] = pd.to_datetime(existing['Data'])
    
    print(f"  Existing: {len(existing):,} rows ({existing['Data'].min().date()} to {existing['Data'].max().date()})")
    
    # Combine and remove duplicates
    combined = pd.concat([existing, new_df], ignore_index=True)
    
    # Remove duplicates based on Player + Date + Team
    combined = combined.drop_duplicates(
        subset=['Player', 'Data', 'Tm'], 
        keep='last'  # Keep newer data if duplicate
    )
    
    combined = combined.sort_values(['Data', 'Player']).reset_index(drop=True)
    
    print(f"  Combined: {len(combined):,} rows ({combined['Data'].min().date()} to {combined['Data'].max().date()})")
    
    return combined


def run_feature_pipeline():
    """Run the feature engineering pipeline after data update."""
    print("\n" + "="*60)
    print("Running feature engineering pipeline...")
    print("="*60)
    
    import subprocess
    
    scripts = [
        'task1_data_foundation.py',
        'task2_rolling_features.py', 
        'task3_opponent_features.py',
        'task4_schedule_features.py'
    ]
    
    for script in scripts:
        if Path(script).exists():
            print(f"\nRunning {script}...")
            result = subprocess.run(['python', script], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {script} completed")
            else:
                print(f"  ✗ {script} failed:")
                print(result.stderr[:500])
                return False
        else:
            print(f"  ⚠ {script} not found, skipping")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Update NBA player game log data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_data.py                # Update current season
  python update_data.py --full         # Full season re-download
  python update_data.py --season 2023-24  # Different season
  python update_data.py --no-pipeline  # Skip feature engineering
        """
    )
    
    parser.add_argument('--season', default='2024-25', help='NBA season (e.g., 2024-25)')
    parser.add_argument('--full', action='store_true', help='Full download (replace existing)')
    parser.add_argument('--output', default='database_24_25.csv', help='Output file path')
    parser.add_argument('--no-pipeline', action='store_true', help='Skip feature engineering')
    
    args = parser.parse_args()
    
    print("="*60)
    print("NBA Data Updater")
    print("="*60)
    print(f"Season: {args.season}")
    print(f"Output: {args.output}")
    print(f"Mode: {'Full download' if args.full else 'Update existing'}")
    
    # Fetch data from NBA.com
    raw_df = get_all_player_game_logs(args.season)
    
    if raw_df is None or len(raw_df) == 0:
        print("\n✗ Failed to fetch data")
        return
    
    # Transform to project format
    new_df = transform_to_project_format(raw_df)
    
    # Merge or replace
    if args.full or not Path(args.output).exists():
        final_df = new_df
    else:
        final_df = update_existing_data(args.output, new_df)
    
    # Save
    final_df.to_csv(args.output, index=False)
    print(f"\n✓ Saved to {args.output}")
    print(f"  Total rows: {len(final_df):,}")
    print(f"  Date range: {final_df['Data'].min().date()} to {final_df['Data'].max().date()}")
    
    # Run feature pipeline
    if not args.no_pipeline:
        success = run_feature_pipeline()
        if success:
            print("\n" + "="*60)
            print("✓ DATA UPDATE COMPLETE")
            print("="*60)
            print("You can now run: python task10_deployment.py")
        else:
            print("\n⚠ Feature pipeline had errors")
    else:
        print("\n✓ Data saved (skipped feature pipeline)")
        print("Run these manually:")
        print("  python task1_data_foundation.py")
        print("  python task2_rolling_features.py")
        print("  python task3_opponent_features.py")
        print("  python task4_schedule_features.py")


if __name__ == "__main__":
    main()
