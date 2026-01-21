"""
FRESH DATA DOWNLOADER - 2025-26 Season
NBA Player Props Betting System

Downloads current season box scores from NBA.com
with CURRENT rosters (trades included)

Usage:
    pip install nba_api pandas
    python get_fresh_data.py
"""

import pandas as pd
import time
from datetime import datetime, timedelta
import os

try:
    from nba_api.stats.endpoints import playergamelogs, leaguegamefinder
    from nba_api.stats.static import teams, players
except ImportError:
    print("Installing nba_api...")
    os.system('pip install nba_api')
    from nba_api.stats.endpoints import playergamelogs, leaguegamefinder
    from nba_api.stats.static import teams, players


def get_current_season():
    """Determine current NBA season string (e.g., '2025-26')"""
    today = datetime.now()
    year = today.year
    month = today.month
    
    # NBA season starts in October
    if month >= 10:
        return f"{year}-{str(year+1)[2:]}"
    else:
        return f"{year-1}-{str(year)[2:]}"


def download_player_box_scores(season='2025-26'):
    """
    Download all player box scores for the season.
    Returns DataFrame in same format as your original database_24_25.csv
    """
    print(f"\n{'='*60}")
    print(f"Downloading {season} Player Box Scores")
    print(f"{'='*60}")
    
    print("\nFetching from NBA.com (this takes ~30 seconds)...")
    
    # Get player game logs for entire season
    # Season type: Regular Season
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable='Regular Season'
    )
    
    # Small delay to respect rate limits
    time.sleep(1)
    
    df = logs.get_data_frames()[0]
    
    print(f"  ✓ Downloaded {len(df):,} player-game records")
    print(f"  ✓ {df['PLAYER_NAME'].nunique()} unique players")
    print(f"  ✓ Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    return df


def format_to_original_schema(df):
    """
    Convert NBA API format to match your original database_24_25.csv schema
    
    Original columns:
    Player, Tm, Opp, Res, MP, FG, FGA, FG%, 3P, 3PA, 3P%, FT, FTA, FT%,
    ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, GmSc, Data
    """
    print("\nConverting to original schema...")
    
    # Create copy
    out = pd.DataFrame()
    
    # Map columns
    out['Player'] = df['PLAYER_NAME']
    out['Tm'] = df['TEAM_ABBREVIATION']
    
    # Parse matchup to get opponent and home/away
    # Format: "DET vs. BOS" or "DET @ BOS"
    def parse_matchup(matchup, team):
        if ' vs. ' in matchup:
            parts = matchup.split(' vs. ')
            opp = parts[1] if parts[0] == team else parts[0]
            return opp, 'H'
        elif ' @ ' in matchup:
            parts = matchup.split(' @ ')
            opp = parts[1] if parts[0] == team else parts[0]
            return opp, 'A'
        return 'UNK', 'U'
    
    matchup_data = df.apply(lambda x: parse_matchup(x['MATCHUP'], x['TEAM_ABBREVIATION']), axis=1)
    out['Opp'] = [m[0] for m in matchup_data]
    
    # Win/Loss
    out['Res'] = df['WL']
    
    # Stats
    out['MP'] = df['MIN']
    out['FG'] = df['FGM']
    out['FGA'] = df['FGA']
    out['FG%'] = df['FG_PCT']
    out['3P'] = df['FG3M']
    out['3PA'] = df['FG3A']
    out['3P%'] = df['FG3_PCT']
    out['FT'] = df['FTM']
    out['FTA'] = df['FTA']
    out['FT%'] = df['FT_PCT']
    out['ORB'] = df['OREB']
    out['DRB'] = df['DREB']
    out['TRB'] = df['REB']
    out['AST'] = df['AST']
    out['STL'] = df['STL']
    out['BLK'] = df['BLK']
    out['TOV'] = df['TOV']
    out['PF'] = df['PF']
    out['PTS'] = df['PTS']
    
    # Game Score (simplified formula)
    # GmSc = PTS + 0.4*FG - 0.7*FGA - 0.4*(FTA-FT) + 0.7*ORB + 0.3*DRB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
    out['GmSc'] = (df['PTS'] + 0.4*df['FGM'] - 0.7*df['FGA'] - 0.4*(df['FTA']-df['FTM']) + 
                   0.7*df['OREB'] + 0.3*df['DREB'] + df['STL'] + 0.7*df['AST'] + 
                   0.7*df['BLK'] - 0.4*df['PF'] - df['TOV'])
    
    # Date
    out['Data'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')
    
    # Sort by date
    out = out.sort_values('Data').reset_index(drop=True)
    
    print(f"  ✓ Formatted {len(out):,} records")
    
    return out


def validate_data(df):
    """Quick validation of downloaded data"""
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    print(f"\nTotal records: {len(df):,}")
    print(f"Players: {df['Player'].nunique()}")
    print(f"Teams: {df['Tm'].nunique()}")
    print(f"Date range: {df['Data'].min()} to {df['Data'].max()}")
    
    # Check for any missing critical columns
    critical = ['Player', 'Tm', 'PTS', 'TRB', 'AST', 'Data']
    missing = [c for c in critical if df[c].isna().sum() > 0]
    if missing:
        print(f"\n⚠ Columns with missing values: {missing}")
    else:
        print(f"\n✓ All critical columns complete")
    
    # Show sample
    print("\nSample of data:")
    print(df[['Player', 'Tm', 'Opp', 'PTS', 'TRB', 'AST', 'Data']].head(10).to_string())
    
    return True


def main():
    print("="*60)
    print("NBA FRESH DATA DOWNLOADER")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine season
    season = get_current_season()
    print(f"\nDetected season: {season}")
    
    # Download
    raw_df = download_player_box_scores(season)
    
    # Format
    formatted_df = format_to_original_schema(raw_df)
    
    # Validate
    validate_data(formatted_df)
    
    # Save
    output_file = f'database_{season.replace("-", "_")}.csv'
    formatted_df.to_csv(output_file, index=False)
    print(f"\n✓ SAVED: {output_file}")
    print(f"  {len(formatted_df):,} player-game records")
    
    # Also save as the standard filename for pipeline compatibility
    formatted_df.to_csv('database_fresh.csv', index=False)
    print(f"✓ SAVED: database_fresh.csv (for pipeline)")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Rename database_fresh.csv to database_24_25.csv (backup old one first)")
    print("2. Run the full pipeline:")
    print("   python task1_data_foundation.py")
    print("   python task2_rolling_features.py")
    print("   python task3_opponent_features.py")
    print("   python task4_schedule_features.py")
    print("   python task5_final_dataset.py")
    print("3. Retrain models with fresh data:")
    print("   python task6_train_models.py")
    print("="*60)


if __name__ == "__main__":
    main()
