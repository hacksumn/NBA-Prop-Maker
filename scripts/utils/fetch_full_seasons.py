"""
Fetch Full Season Data (2023-24 + 2024-25)
NBA Player Props System

This script fetches complete historical data for multiple seasons
to dramatically increase training data size.

Current: 90 days (~12k games)
Target: 2+ seasons (~50k+ games)

Expected Impact: +5-10% accuracy improvement
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs
from nba_api.stats.static import teams
import time
from datetime import datetime

def fetch_season_games(season='2024-25'):
    """
    Fetch all games for a specific season.
    
    Args:
        season: Season string like '2024-25' or '2023-24'
    
    Returns:
        DataFrame with all player game logs
    """
    print(f"\n{'='*60}")
    print(f"FETCHING {season} SEASON DATA")
    print(f"{'='*60}")
    
    try:
        # Use PlayerGameLogs endpoint to get all player games
        print(f"Requesting data from NBA API...")
        
        gamelog = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable='Regular Season'
        )
        
        df = gamelog.get_data_frames()[0]
        
        print(f"  ✓ Fetched {len(df):,} player-games")
        print(f"  ✓ Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
        print(f"  ✓ Unique players: {df['PLAYER_NAME'].nunique()}")
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'PLAYER_NAME': 'player',
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup_raw',
            'PTS': 'pts',
            'REB': 'trb',
            'AST': 'ast',
            'STL': 'stl',
            'BLK': 'blk',
            'TOV': 'tov',
            'MIN': 'mp',
            'FGM': 'fg',
            'FGA': 'fga',
            'FG_PCT': 'fg_pct',
            'FG3M': '3p',
            'FG3A': '3pa',
            'FG3_PCT': '3p_pct',
            'FTM': 'ft',
            'FTA': 'fta',
            'FT_PCT': 'ft_pct',
            'PLUS_MINUS': 'plus_minus'
        })
        
        # Parse game_date
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Extract opponent from matchup (e.g., "LAL vs. BOS" -> "BOS")
        df['is_home'] = df['matchup_raw'].str.contains('vs.').astype(int)
        df['opp'] = df['matchup_raw'].str.extract(r'(vs\.|@)\s+([A-Z]{3})')[1]
        
        # Convert minutes from "MM:SS" to decimal
        if df['mp'].dtype == 'object':
            def convert_minutes(mp_str):
                if pd.isna(mp_str) or mp_str == '':
                    return 0.0
                try:
                    if ':' in str(mp_str):
                        mins, secs = str(mp_str).split(':')
                        return float(mins) + float(secs) / 60
                    else:
                        return float(mp_str)
                except:
                    return 0.0
            
            df['mp'] = df['mp'].apply(convert_minutes)
        
        # Fill NaN values
        df['fg_pct'] = df['fg_pct'].fillna(0)
        df['3p_pct'] = df['3p_pct'].fillna(0)
        df['ft_pct'] = df['ft_pct'].fillna(0)
        
        # Sort by date
        df = df.sort_values('game_date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error fetching {season}: {e}")
        return None


def combine_seasons(seasons=['2023-24', '2024-25']):
    """
    Fetch and combine multiple seasons.
    
    Args:
        seasons: List of season strings
    
    Returns:
        Combined DataFrame with all seasons
    """
    print(f"\n{'='*60}")
    print(f"FETCHING MULTIPLE SEASONS")
    print(f"{'='*60}")
    print(f"Seasons: {', '.join(seasons)}")
    
    all_data = []
    
    for season in seasons:
        df = fetch_season_games(season)
        
        if df is not None:
            df['season'] = season
            all_data.append(df)
            print(f"  ✓ {season}: {len(df):,} games")
            
            # Rate limiting - NBA API has limits
            print(f"  Waiting 2 seconds (rate limiting)...")
            time.sleep(2)
        else:
            print(f"  ✗ {season}: Failed to fetch")
    
    if not all_data:
        print(f"\n✗ No data fetched!")
        return None
    
    # Combine all seasons
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"COMBINED DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total games: {len(combined):,}")
    print(f"Date range: {combined['game_date'].min()} to {combined['game_date'].max()}")
    print(f"Unique players: {combined['player'].nunique()}")
    print(f"Seasons: {combined['season'].nunique()}")
    
    return combined


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("NBA PROPS - FETCH FULL SEASON DATA")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch 2023-24 and 2024-25 seasons
    seasons = ['2023-24', '2024-25']
    
    df = combine_seasons(seasons)
    
    if df is None:
        print("\n✗ Failed to fetch data")
        return
    
    # Save raw data
    output_file = 'data/database_full_seasons.csv'
    
    print(f"\n{'='*60}")
    print(f"SAVING DATA")
    print(f"{'='*60}")
    print(f"Output: {output_file}")
    
    df.to_csv(output_file, index=False)
    
    print(f"  ✓ Saved {len(df):,} rows")
    
    # Summary by season
    print(f"\n{'='*60}")
    print(f"BREAKDOWN BY SEASON")
    print(f"{'='*60}")
    
    for season in df['season'].unique():
        season_data = df[df['season'] == season]
        print(f"\n{season}:")
        print(f"  Games: {len(season_data):,}")
        print(f"  Players: {season_data['player'].nunique()}")
        print(f"  Date range: {season_data['game_date'].min().strftime('%Y-%m-%d')} to {season_data['game_date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Run full pipeline to process this data")
    print(f"  2. Add matchup features")
    print(f"  3. Retrain models")
    print(f"\nExpected improvement: +5-10% accuracy")


if __name__ == "__main__":
    main()
