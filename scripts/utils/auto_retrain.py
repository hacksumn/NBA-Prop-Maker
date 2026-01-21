"""
Automated Retraining Pipeline
NBA Player Props System

This script automates the complete retraining workflow:
1. Fetch fresh NBA data
2. Rebuild all features
3. Add matchup history
4. Retrain classification models

Run this weekly to keep models current.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {command}\n")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e}")
        return False

def main():
    """Run the complete retraining pipeline."""
    
    print("\n" + "="*60)
    print("NBA PROPS - AUTOMATED RETRAINING PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to project root
    project_root = Path(__file__).parent.parent.parent
    print(f"\nProject root: {project_root}")
    
    # Steps in the pipeline
    steps = [
        {
            'command': 'python scripts/utils/get_fresh_data.py',
            'description': 'Step 1: Fetch fresh NBA data',
            'required': True
        },
        {
            'command': 'python scripts/pipeline/task1_data_foundation.py',
            'description': 'Step 2: Clean and prepare data',
            'required': True
        },
        {
            'command': 'python scripts/pipeline/task2_rolling_features.py',
            'description': 'Step 3: Calculate rolling averages',
            'required': True
        },
        {
            'command': 'python scripts/pipeline/task3_opponent_features.py',
            'description': 'Step 4: Add opponent defense features',
            'required': True
        },
        {
            'command': 'python scripts/pipeline/task4_schedule_features.py',
            'description': 'Step 5: Add schedule/fatigue features',
            'required': True
        },
        {
            'command': 'python scripts/pipeline/task5_final_dataset.py',
            'description': 'Step 6: Create final dataset',
            'required': True
        },
        {
            'command': 'python scripts/utils/add_matchup_features.py',
            'description': 'Step 7: Add matchup history features',
            'required': True
        },
        {
            'command': 'python scripts/pipeline/task7_daily_predictions.py',
            'description': 'Step 8: Train classification models',
            'required': True
        }
    ]
    
    # Execute pipeline
    failed_steps = []
    
    for i, step in enumerate(steps, 1):
        success = run_command(step['command'], step['description'])
        
        if not success:
            failed_steps.append(step['description'])
            if step['required']:
                print(f"\n✗ PIPELINE FAILED at {step['description']}")
                print(f"Cannot continue - this step is required")
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_steps:
        print(f"\n✗ FAILED - {len(failed_steps)} step(s) failed:")
        for step in failed_steps:
            print(f"  - {step}")
        sys.exit(1)
    else:
        print(f"\n✓ SUCCESS - All {len(steps)} steps completed")
        print(f"\nModels are now up-to-date and ready to use:")
        print(f"  python generate_picks_v2.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
