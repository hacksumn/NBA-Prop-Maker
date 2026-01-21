# Repository Structure

This document explains the organization of the NBA Prop Maker repository.

## Directory Layout

```
NBA-Prop-Maker/
├── README.md                    # Main project documentation
├── PROJECT_STATE.md             # Detailed project state
├── QUICK_START_GUIDE.md         # Quick start guide
├── CHANGELOG.md                 # Version history
├── STRUCTURE.md                 # This file
│
├── full_pipeline.py             # All-in-one pipeline script
├── generate_picks_v2.py         # 🆕 Production picks generator (classification model)
├── feature_cols.json            # Feature column definitions
│
├── data/                        # Data files
│   ├── database_24_25.csv       # Raw NBA game data (2024-25 season)
│   ├── player_games_schedule.csv # Final feature-engineered dataset
│   └── intermediate/            # Intermediate processing files
│       ├── player_games_clean.csv
│       ├── player_games_features.csv
│       ├── player_games_opp_features.csv
│       ├── model_data_full.csv
│       └── model_data_filtered.csv
│
├── models/                      # Trained models
│   ├── regression/              # Old regression models
│   │   ├── model_pts.pkl
│   │   ├── model_trb.pkl
│   │   ├── model_ast.pkl
│   │   └── ... (6 more)
│   └── classifiers/             # 🆕 New classification models (to be generated)
│       ├── classifier_pts.pkl
│       └── ... (future)
│
├── results/                     # Predictions and betting results
│   ├── real_lines.csv           # Latest Vegas lines
│   ├── vegas_lines_consolidated.csv # Consolidated lines
│   └── archive/                 # Historical results
│       ├── predictions_*.csv
│       ├── bets_locked_*.csv
│       ├── bets_summary_*.txt
│       └── tracking_*.csv
│
├── images/                      # Visualizations
│   ├── backtest_cumulative_profit.png
│   └── backtest_roi_by_edge.png
│
└── scripts/                     # All Python scripts
    ├── pipeline/                # Main pipeline tasks
    │   ├── task1_data_foundation.py      # Data cleaning
    │   ├── task2_rolling_features.py     # Rolling averages
    │   ├── task3_opponent_features.py    # Opponent defense
    │   ├── task4_schedule_features.py    # Schedule/fatigue
    │   ├── task5_final_dataset.py        # Final dataset
    │   ├── task6_train_models.py         # Model training (regression)
    │   ├── task7_daily_predictions.py    # 🆕 Classification model training
    │   ├── task8_betting_lines.py        # Line comparison
    │   ├── task9_backtesting.py          # Backtesting
    │   └── task10_deployment.py          # Deployment
    │
    └── utils/                   # Utility scripts
        ├── get_fresh_data.py    # Fetch latest NBA data
        ├── update_data.py       # Update existing data
        ├── grade_bets.py        # Grade bet results
        └── minutes_model.py     # Minutes prediction model
```

## Key Files

### Production Scripts (Use These)

1. **`generate_picks_v2.py`** 🆕
   - Main production script for generating betting picks
   - Uses classification models to predict OVER/UNDER
   - Fetches real Vegas lines from The Odds API
   - Only recommends bets with 60%+ confidence
   - **This is what you run daily to get picks**

2. **`full_pipeline.py`**
   - All-in-one pipeline from data → models → predictions
   - Good for retraining everything from scratch
   - Run when you want to rebuild the entire system

3. **`scripts/pipeline/task7_daily_predictions.py`** 🆕
   - Trains classification models (new approach)
   - Predicts probability of OVER/UNDER directly
   - Uses advanced features including matchup history
   - Run this to train new classification models

### Data Flow

```
Raw Data (database_24_25.csv)
    ↓
task1_data_foundation.py → player_games_clean.csv
    ↓
task2_rolling_features.py → player_games_features.csv
    ↓
task3_opponent_features.py → player_games_opp_features.csv
    ↓
task4_schedule_features.py → player_games_schedule.csv
    ↓
task5_final_dataset.py → model_data_full.csv
    ↓
task6_train_models.py → models/regression/*.pkl (old)
    OR
task7_daily_predictions.py → models/classifiers/*.pkl (new) 🆕
    ↓
generate_picks_v2.py → picks_v2_latest.csv 🆕
```

## Model Versions

### Old Approach (Regression)
- **Location**: `models/regression/`
- **Method**: Predict raw stats (e.g., "player will score 25.3 points")
- **Issue**: Doesn't account for Vegas line intelligence
- **Status**: Deprecated but kept for reference

### New Approach (Classification) 🆕
- **Location**: `models/classifiers/` (to be created)
- **Method**: Predict probability of OVER/UNDER directly
- **Advantage**: Uses Vegas line as a feature, only bets when confident
- **Status**: Current production approach

## Quick Start

### To Generate Today's Picks:

```bash
# 1. Train classification models (if not already done)
python scripts/pipeline/task7_daily_predictions.py

# 2. Generate picks using real Vegas lines
python generate_picks_v2.py
```

### To Retrain Everything:

```bash
# Run full pipeline
python full_pipeline.py

# Then train classification models
python scripts/pipeline/task7_daily_predictions.py
```

### To Update Data:

```bash
# Fetch fresh NBA data
python scripts/utils/get_fresh_data.py

# Rebuild features
python scripts/pipeline/task2_rolling_features.py
python scripts/pipeline/task3_opponent_features.py
python scripts/pipeline/task4_schedule_features.py
python scripts/pipeline/task5_final_dataset.py
```

## File Paths in Code

**Important**: Scripts may need path updates after reorganization.

If you see import errors or file not found errors:
- Update paths to use `data/`, `models/`, `results/` directories
- Or run scripts from the repository root directory

Example path updates:
```python
# Old
df = pd.read_csv('database_24_25.csv')

# New
df = pd.read_csv('data/database_24_25.csv')
```

## What Was Removed

The following files were removed to reduce clutter:
- Debug scripts (debug_leakage.py, etc.)
- Old validation scripts
- Duplicate data files
- Outdated betting line scripts (replaced by generate_picks_v2.py)
- Test files

All removed files can be regenerated if needed.

## Next Steps

1. **Update script paths** if any scripts fail to find files
2. **Train classification models** using task7_daily_predictions.py
3. **Test generate_picks_v2.py** to ensure it works with new structure
4. **Update documentation** with any additional changes

---

**Last Updated**: January 21, 2026
**Version**: 2.0 (Classification Model Release)
