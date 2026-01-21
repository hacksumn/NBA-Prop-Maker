"""
Process Full Season Data Through Pipeline
NBA Player Props System

This script runs the complete data pipeline on the full season data.
"""

import subprocess
import pandas as pd
from datetime import datetime

def run_pipeline_step(script_path, description, input_file=None):
    """Run a pipeline script."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    # If input file specified, temporarily replace the main data file
    if input_file:
        print(f"Using input: {input_file}")
        # Create a backup and swap files
        subprocess.run(['cp', 'data/database_24_25.csv', 'data/database_24_25_backup.csv'], check=False)
        subprocess.run(['cp', input_file, 'data/database_24_25.csv'], check=True)
    
    try:
        result = subprocess.run(
            ['python3.11', script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False
    finally:
        # Restore backup if we swapped files
        if input_file:
            subprocess.run(['mv', 'data/database_24_25_backup.csv', 'data/database_24_25.csv'], check=False)


def main():
    """Process full season data."""
    
    print("\n" + "="*60)
    print("PROCESS FULL SEASON DATA THROUGH PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if full season data exists
    try:
        df = pd.read_csv('data/database_full_seasons.csv')
        print(f"\n✓ Found full season data: {len(df):,} rows")
    except:
        print(f"\n✗ Full season data not found!")
        print(f"Run: python scripts/utils/fetch_full_seasons.py")
        return
    
    # Replace the main database with full seasons
    print(f"\nReplacing database_24_25.csv with full season data...")
    subprocess.run(['cp', 'data/database_24_25.csv', 'data/database_24_25_original.csv'], check=False)
    subprocess.run(['cp', 'data/database_full_seasons.csv', 'data/database_24_25.csv'], check=True)
    print(f"  ✓ Database replaced")
    
    # Run pipeline steps
    steps = [
        ('scripts/pipeline/task1_data_foundation.py', 'Step 1: Clean and prepare data'),
        ('scripts/pipeline/task2_rolling_features.py', 'Step 2: Calculate rolling features'),
        ('scripts/pipeline/task3_opponent_features.py', 'Step 3: Add opponent features'),
        ('scripts/pipeline/task4_schedule_features.py', 'Step 4: Add schedule features'),
        ('scripts/pipeline/task5_final_dataset.py', 'Step 5: Create final dataset'),
    ]
    
    for script, description in steps:
        success = run_pipeline_step(script, description)
        if not success:
            print(f"\n✗ Pipeline failed at: {description}")
            return
    
    # Check result
    try:
        df_final = pd.read_csv('data/player_games_schedule.csv', parse_dates=['game_date'])
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Final dataset: {len(df_final):,} rows")
        print(f"Date range: {df_final['game_date'].min()} to {df_final['game_date'].max()}")
        print(f"Players: {df_final['player'].nunique()}")
        print(f"Columns: {len(df_final.columns)}")
        
        print(f"\n{'='*60}")
        print("NEXT STEP")
        print("="*60)
        print(f"Add matchup features:")
        print(f"  python scripts/utils/add_matchup_features.py")
        
    except Exception as e:
        print(f"\n✗ Error loading final dataset: {e}")


if __name__ == "__main__":
    main()
