"""
COMPLETE RETRAIN PIPELINE
Downloads 2025-26 data, rebuilds features, retrains all models

Usage:
    python retrain_2526.py
"""

import subprocess
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and check for errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    print("="*60)
    print("COMPLETE 2025-26 SEASON RETRAIN")
    print(f"Started: {datetime.now()}")
    print("="*60)
    
    # Step 1: Download 2025-26 data
    print("\n" + "="*60)
    print("STEP 1: Downloading 2025-26 season data from NBA.com")
    print("="*60)
    
    from nba_api.stats.endpoints import leaguegamelog
    import time
    
    print("Fetching data (this takes 30-60 seconds)...")
    try:
        game_log = leaguegamelog.LeagueGameLog(
            season='2025-26',
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='P',
            timeout=120
        )
        df = game_log.get_data_frames()[0]
        print(f"✓ Fetched {len(df):,} player-game records")
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        print("\nTrying 2024-25 season as fallback...")
        try:
            game_log = leaguegamelog.LeagueGameLog(
                season='2024-25',
                season_type_all_star='Regular Season',
                player_or_team_abbreviation='P',
                timeout=120
            )
            df = game_log.get_data_frames()[0]
            print(f"✓ Fetched {len(df):,} player-game records (2024-25 season)")
        except Exception as e2:
            print(f"✗ Error: {e2}")
            return
    
    # Transform to our format
    print("\nTransforming data...")
    column_map = {
        'PLAYER_NAME': 'Player',
        'TEAM_ABBREVIATION': 'Tm',
        'MATCHUP': 'matchup_raw',
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
    }
    
    df = df.rename(columns=column_map)
    
    # Parse opponent
    def parse_opponent(matchup):
        if pd.isna(matchup):
            return None
        parts = matchup.replace('.', '').split()
        if len(parts) >= 3 and parts[1] in ['vs', '@']:
            return parts[2]
        return None
    
    df['Opp'] = df['matchup_raw'].apply(parse_opponent)
    
    # Calculate GmSc
    numeric_cols = ['PTS', 'FG', 'FGA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 
                    'AST', 'STL', 'BLK', 'TOV', 'PF', '3P', '3PA']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['GmSc'] = (
        df['PTS'] + 0.4 * df['FG'] - 0.7 * df['FGA'] 
        - 0.4 * (df['FTA'] - df['FT']) + 0.7 * df['ORB'] + 0.3 * df['DRB']
        + df['STL'] + 0.7 * df['AST'] + 0.7 * df['BLK'] 
        - 0.4 * df['PF'] - df['TOV']
    ).round(1)
    
    # Select columns
    output_cols = [
        'Player', 'Tm', 'Opp', 'Res', 'MP',
        'FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
        'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', 'Data'
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols]
    
    # Sort
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(['Data', 'Player']).reset_index(drop=True)
    
    # Save
    df.to_csv('database_24_25.csv', index=False)
    print(f"✓ Saved {len(df):,} rows to database_24_25.csv")
    print(f"  Date range: {df['Data'].min().date()} to {df['Data'].max().date()}")
    print(f"  Players: {df['Player'].nunique()}")
    
    # Step 2-5: Run feature engineering pipeline
    scripts = [
        ('task1_data_foundation.py', 'Cleaning & standardizing data'),
        ('task2_rolling_features.py', 'Creating rolling features'),
        ('task3_opponent_features.py', 'Adding opponent features'),
        ('task4_schedule_features.py', 'Adding schedule features'),
    ]
    
    for script, desc in scripts:
        if Path(script).exists():
            print(f"\n{'='*60}")
            print(f"STEP: {desc}")
            print(f"{'='*60}")
            result = subprocess.run([sys.executable, script], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: {script} had issues")
                print(result.stderr[:500] if result.stderr else "")
            else:
                # Print last few lines
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    print(line)
    
    # Step 6: Retrain models
    print(f"\n{'='*60}")
    print("STEP: Retraining all models")
    print(f"{'='*60}")
    
    if Path('task6_train_models.py').exists():
        result = subprocess.run([sys.executable, 'task6_train_models.py'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        for line in lines[-30:]:
            print(line)
    
    # Summary
    print(f"\n{'='*60}")
    print("RETRAIN COMPLETE")
    print(f"{'='*60}")
    print(f"Finished: {datetime.now()}")
    
    # Check what we have
    if Path('player_games_schedule.csv').exists():
        final = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
        print(f"\nFinal dataset:")
        print(f"  Rows: {len(final):,}")
        print(f"  Date range: {final['game_date'].min().date()} to {final['game_date'].max().date()}")
        print(f"  Players: {final['player'].nunique()}")
    
    model_count = len(list(Path('.').glob('model_*.pkl')))
    print(f"\nModels trained: {model_count}")
    
    print(f"\nNow run:")
    print(f"  python task10_deployment.py --skip-update")


if __name__ == "__main__":
    main()
