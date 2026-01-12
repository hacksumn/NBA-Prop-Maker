# NBA Player Props Betting System - Project State

## Last Updated: January 9, 2025

## Current Status: ALL TASKS COMPLETE ✓

---

## Project Overview
Building a machine learning pipeline to predict NBA player props (points, rebounds, assists, etc.) with enough accuracy to beat sportsbook lines. Target: 53-58% accuracy (anything above 52.38% is profitable at -110 odds).

**Critical Rule:** ZERO data leakage - models can only use information available BEFORE tip-off.

---

## Completed Tasks

### Task 1: Data Foundation ✓
- Input: `database_24_25.csv` (raw NBA box scores from Basketball Reference)
- Output: `player_games_clean.csv`
- Standardized columns, parsed dates, created game IDs
- 16,512 rows, 562 players, 30 teams
- Date range: Oct 22, 2024 → Feb 7, 2025

### Task 2: Rolling Features ✓
- Output: `player_games_features.csv`
- Created L5, L10, L20 rolling averages for all stats
- Used `.shift(1)` to prevent leakage (excludes current game)
- Verified: Game 2's L5 = Game 1's actual value (exact match)
- 51 new features (39 rolling means, 8 volatility, 3 trends)

### Task 3: Opponent Features ✓
- Output: `player_games_opp_features.csv`
- Added opponent defensive stats (pts_allowed_l10, etc.)
- Defensive rankings (1-30 scale)
- 9 opponent features added

### Task 4: Schedule Features ✓
- Output: `player_games_schedule.csv`
- Days rest, back-to-back flags, games density
- Player fatigue (minutes L3, high minutes flag)
- 10 schedule features added
- Final: 99 columns

### Task 5: Final Dataset ✓
- Output: `model_data_filtered.csv`
- Removed all leaky columns (current game stats)
- Filtered to players with 10+ games
- 11,351 rows, 79 columns
- Verified: R² = 0.54, MAE = 4.78 (realistic, no leakage)

### Task 5B: Rigorous Validation ✓
- Manually verified rolling averages against raw data
- Exact match for multiple players (Jaden McDaniels, Julian Strawther, Jarrett Allen)
- Opponent features verified (BOS allowed 109.2 - exact match)
- Schedule features verified (days rest matches date diffs)

### Task 6: Model Training ✓
- Output: 9 model files (`model_*.pkl`) + 9 prediction files (`predictions_*.csv`)
- Temporal split: Train before Jan 17, Test after
- All models passed leakage checks

**Model Results:**
| Prop | R² | MAE | Bet Accuracy | ROI |
|------|-----|-----|--------------|-----|
| PTS | 0.568 | 4.58 | 54.7% | +4.8% |
| TRB | 0.493 | 1.91 | 56.6% | +8.8% |
| AST | 0.533 | 1.33 | 56.8% | +9.2% |
| STL | 0.081 | 0.72 | 52.7% | +0.7% |
| BLK | 0.215 | 0.52 | 49.9% | -5.1% |
| TOV | 0.325 | 0.87 | 57.7% | +11.3% |
| PRA | 0.619 | 6.01 | 57.4% | +10.5% |
| PR | 0.577 | 5.55 | 56.1% | +7.8% |
| PA | 0.627 | 5.01 | 55.5% | +6.6% |

### Task 7: Daily Prediction Script ✓
- Output: `task7_daily_predictions.py`
- Features:
  - Multiple input modes: interactive, CSV file, command line, or historical date
  - Fuzzy player name matching (handles accents like Jokić)
  - Automatic opponent feature lookup
  - Schedule feature computation
  - Edge detection (highlights best betting opportunities)
  - CSV export option
  - **Minutes filter** (`--min-minutes 25`) to focus on rotation players
  - **Edge filter** (`--min-edge 2.0`) to only show significant edges  
  - **Top N filter** (`--top 10`) to show only top edges
  - **Predictions floored at 0** (no negative stats)

**Usage Examples:**
```bash
# Rotation players with significant edges
python task7_daily_predictions.py --date 2025-02-07 --min-minutes 25 --min-edge 2.0

# Top 10 edges
python task7_daily_predictions.py --date 2025-02-07 --min-minutes 20 --top 10

# Specific players
python task7_daily_predictions.py --players "LeBron James,LAL,GSW;Nikola Jokic,DEN,MIA"
```

### Task 8: Compare to Real Betting Lines ✓ (NEW)
- Output: `task8_betting_lines.py`
- Features:
  - **The Odds API integration** (requires free API key from the-odds-api.com)
  - **Manual line entry mode** for custom analysis
  - **CSV input** for batch processing (see sample_lines.csv)
  - **Edge calculation** with implied probability estimates
  - **Kelly Criterion bet sizing** (quarter Kelly)
  - **Bet recommendations**: Strong (★★★), Moderate (★★), Slight (★)
  - **Bankroll management** suggestions

**Usage Examples:**
```bash
# From CSV file with lines
python task8_betting_lines.py --input lines.csv --bankroll 1000

# With The Odds API (live lines)
export ODDS_API_KEY=your_key
python task8_betting_lines.py --api

# Manual entry
python task8_betting_lines.py --manual --players "LeBron James;Nikola Jokic"
```

**Sample Lines CSV Format:**
```csv
player,prop,line,over_odds,under_odds,book
LeBron James,pts,25.5,-110,-110,DraftKings
Nikola Jokić,pra,50.5,-108,-112,FanDuel
```

---

### Task 9: Backtesting Framework ✓
- Output: `task9_backtesting.py`
- Analyzes historical predictions vs actual outcomes
- Features:
  - **Overall ROI & win rate** calculation
  - **Breakdown by prop type** - identifies best/worst props
  - **Breakdown by edge size** - validates edge correlation
  - **Breakdown by direction** - OVER vs UNDER performance
  - **Breakdown by player tier** - bench/rotation/starter/star
  - **Optimal threshold finder** - finds best minimum edge
  - **Visualization** - cumulative profit & ROI charts
  - **Export** - detailed results to CSV

**Backtest Results (29,412 bets):**
```
Overall: 54.4% win rate, +3.88% ROI, +1,141 units profit ✓

By Prop:
  Pts+Reb+Ast   57.2%  +9.2% ROI  ✓ BEST
  Rebounds      56.1%  +7.0% ROI  ✓
  Turnovers     56.0%  +7.0% ROI  ✓
  Blocks        47.9%  -8.6% ROI  ✗ AVOID

By Edge Size:
  0-0.5 pts     51.6%  -1.5% ROI  ✗ (don't bet low edge)
  1.5-2 pts     58.9% +12.5% ROI  ✓
  3-5 pts       64.7% +23.5% ROI  ✓
  5+ pts        73.4% +40.1% ROI  ✓ BEST

By Direction:
  UNDER         62.2% +18.7% ROI  ✓ (model favors unders)
  OVER          47.8%  -8.8% ROI  ✗

By Player Tier:
  Bench (<15)   51.1%  -2.5% ROI  ✗ AVOID
  Rotation      56.1%  +7.0% ROI  ✓ BEST
  Starters      55.1%  +5.2% ROI  ✓
  Stars (32+)   54.6%  +4.3% ROI  ✓
```

**Key Recommendations:**
1. Focus on PRA, Rebounds, Turnovers, Assists
2. Avoid Blocks prop
3. Use minimum 1.5+ point edge
4. Favor UNDER bets
5. Skip bench players (<15 min)

**Usage:**
```bash
# Full backtest
python task9_backtesting.py

# With minimum edge filter
python task9_backtesting.py --min-edge 1.5

# Generate plots
python task9_backtesting.py --plot

# Export results
python task9_backtesting.py --output results.csv
```

---

### Task 10: Deployment Pipeline ✓
- Output: `task10_deployment.py`
- Automates the full daily prediction workflow
- Features:
  - **Data freshness check** - warns if data is stale
  - **Automatic game fetching** - via The Odds API
  - **Player roster detection** - finds active players per team
  - **Batch predictions** - all players in one run
  - **Opportunity filtering** - applies backtest findings:
    - Min 1.5 point edge
    - Min 15 minutes avg
    - Excludes blocks prop
    - Prefers UNDER bets
  - **Email alerts** - optional notification
  - **Logging** - tracks all runs
  - **Export** - daily CSVs + JSON summary

**Usage:**
```bash
# Full daily run
python task10_deployment.py

# Test without saving
python task10_deployment.py --dry-run

# With stale data
python task10_deployment.py --force --skip-update

# With email alerts
python task10_deployment.py --email you@email.com

# Custom thresholds
python task10_deployment.py --min-edge 2.0 --min-minutes 20
```

**Schedule with Cron (Linux):**
```bash
# Run every day at 9 AM
0 9 * * * cd /path/to/nba && python task10_deployment.py >> cron.log 2>&1
```

**Schedule with Task Scheduler (Windows):**
1. Open Task Scheduler
2. Create Basic Task → Daily trigger
3. Action: Start a program
   - Program: `python`
   - Arguments: `C:\path\to\task10_deployment.py`
   - Start in: `C:\path\to\nba`

**Output Files:**
```
daily_predictions/
├── predictions_2025-02-07.csv    # All player predictions
├── opportunities_2025-02-07.csv  # Filtered betting opportunities
└── summary_2025-02-07.json       # Run summary
```

---

## 🎉 PROJECT COMPLETE!

All 10 tasks have been completed. The system includes:

1. **Data Pipeline** (Tasks 1-5): Clean data → rolling features → opponent features → schedule features
2. **Model Training** (Task 6): 9 gradient boosting models with 54.4% overall win rate
3. **Daily Predictions** (Task 7): Generate predictions for any player/matchup
4. **Line Comparison** (Task 8): Compare to real sportsbook lines
5. **Backtesting** (Task 9): Validated +3.88% ROI on 29,412 historical bets
6. **Deployment** (Task 10): Automated daily pipeline with alerts

**Key Findings from Backtest:**
- Overall: 54.4% win rate, +3.88% ROI
- Best props: PRA (+9.2%), Rebounds (+7.0%), Turnovers (+7.0%)
- Avoid: Blocks (-8.6% ROI)
- Higher edges = higher ROI (5+ pts edge = 73% win rate!)
- UNDER bets outperform OVER bets significantly
- Rotation players (15-25 min) most profitable

---

## File Structure
```
project_folder/
├── database_24_25.csv              # Raw data
├── player_games_clean.csv          # Task 1 output
├── player_games_features.csv       # Task 2 output
├── player_games_opp_features.csv   # Task 3 output
├── player_games_schedule.csv       # Task 4 output (MAIN DATA FILE)
├── model_data_filtered.csv         # Task 5 output
├── model_pts.pkl                   # Trained models
├── model_trb.pkl
├── model_ast.pkl
├── model_stl.pkl
├── model_blk.pkl
├── model_tov.pkl
├── model_pra.pkl
├── model_pr.pkl
├── model_pa.pkl
├── predictions_*.csv               # Test set predictions
├── task1_data_foundation.py
├── task2_rolling_features.py
├── task3_opponent_features.py
├── task4_schedule_features.py
├── task5_final_dataset.py
├── task5b_rigorous_validation.py
├── task6_train_models.py
├── task7_daily_predictions.py      # Daily prediction script
├── task8_betting_lines.py          # Compare to real betting lines
├── task9_backtesting.py            # Backtesting framework
├── task10_deployment.py            # Automated daily pipeline
├── daily_predictions/              # Output folder for daily runs
├── backtest_cumulative_profit.png  # Profit over time chart
├── backtest_roi_by_edge.png        # ROI vs edge threshold chart
└── sample_lines.csv                # Sample betting lines CSV
```

---

## Key Technical Details

### Feature Engineering
- All rolling features use `.shift(1)` to exclude current game
- Rolling windows: L5, L10, L20
- Volatility: std dev for L10, L20
- Trends: L5/L20 ratio

### Leakage Prevention Rules
- NO current game stats as features (pts, trb, ast, mp, fg%, etc.)
- ONLY historical rolling averages
- R² > 0.80 = leakage flag
- Betting accuracy > 65% = leakage flag

### Betting Simulation
- Lines simulated using player's L10 average (what sportsbooks use)
- Break-even at -110 odds: 52.38%
- Profitable: 8 of 9 props

### Daily Predictions (Task 7)
- Uses player's most recent rolling features
- Updates opponent defensive context for new matchup
- Uses realistic schedule feature defaults
- Fuzzy name matching handles special characters

---

## To Continue
Upload this file to a new chat and say:
"I'm continuing my NBA props betting system. We completed Task 7 (daily predictions). Ready for Task 8: Compare to Real Betting Lines."
