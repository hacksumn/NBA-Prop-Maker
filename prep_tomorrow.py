"""
PREP TOMORROW'S PICKS
Run this AFTER tonight's games finish

Does everything:
1. Downloads fresh data (with tonight's results)
2. Rebuilds feature pipeline
3. Generates predictions
4. Fetches Vegas lines
5. Locks in picks

Usage:
    python prep_tomorrow.py
"""

import subprocess
import sys
from datetime import datetime

def run_step(step_name, command):
    """Run a step and report status"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"⚠ WARNING: {step_name} may have had issues")
        return False
    
    print(f"✓ {step_name} complete")
    return True


def main():
    print("="*60)
    print("🏀 PREP TOMORROW'S PICKS")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will take 2-3 minutes...\n")
    
    steps = [
        ("Download fresh data", "python get_fresh_data.py"),
        ("Task 1: Clean data", "python task1_data_foundation.py"),
        ("Task 2: Rolling features", "python task2_rolling_features.py"),
        ("Task 3: Opponent features", "python task3_opponent_features.py"),
        ("Task 4: Schedule features", "python task4_schedule_features.py"),
        ("Task 5: Final dataset", "python task5_final_dataset.py"),
        ("Generate predictions", "python task7_daily_predictions.py"),
        ("Fetch Vegas lines", "python get_vegas_lines.py"),
        ("Lock in picks", "python grade_bets.py lock"),
    ]
    
    for step_name, command in steps:
        success = run_step(step_name, command)
        if not success:
            response = input(f"\nContinue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return
    
    print("\n" + "="*60)
    print("🎯 TOMORROW'S PICKS ARE LOCKED!")
    print("="*60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nFiles created:")
    print("  - tracking_latest.csv (your picks)")
    print("  - model_vs_vegas.csv (full comparison)")
    print("\nGrade results with: python grade_bets.py grade")
    print("="*60)


if __name__ == "__main__":
    main()
