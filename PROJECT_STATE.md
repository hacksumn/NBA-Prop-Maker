# Project State — NBA Prop Maker

**Last Updated:** March 25, 2026
**Status:** Active — running daily via Windows Task Scheduler

---

## Current System Architecture

The project runs as a two-step daily pipeline on Windows:

1. `run_daily.py` — Fetches last night's box scores and today's PrizePicks lines, then auto-grades all ungraded picks and betslips.
2. `nba_props.py predict` — Generates today's picks, betslips, and HTML dashboard.

Both steps are chained in `run_morning.bat` and scheduled via Windows Task Scheduler at ~9am daily.

---

## Live Performance (as of 2026-03-25)

| Metric | Value |
|---|---|
| Overall Win Rate | **56.4%** (474 W / 366 L) |
| Total Graded Picks | 840 |
| Betslip Win Rate | 20.0% (3 W / 12 L) |
| Data Rows | 75,799 player-game rows |
| Unique Players | 796 |
| Seasons Covered | 2023-24, 2024-25, 2025-26 |
| PrizePicks Lines Tracked | 21,474 rows |
| Last Data Refresh | 2026-03-23 |

---

## Model Metrics (Current)

| Prop | MAE | O/U Accuracy |
|---|---|---|
| PTS | 4.14 | 78.9% |
| TRB | 1.84 | 73.3% |
| AST | 1.32 | 77.3% |
| TOV | 0.87 | 70.2% |
| STL | 0.70 | 49.5% |
| BLK | 0.51 | 32.0% |

---

## Bias Correction Values (Current)

| Prop | Bias |
|---|---|
| PTS | -0.4743 |
| TRB | -0.2250 |
| AST | -0.1924 |
| STL | -0.1125 |
| BLK | -0.0743 |
| TOV | -0.1175 |

---

## Pipeline Components

All 10 task scripts are complete and archived in `scripts/pipeline/`. The active daily workflow no longer calls these directly — they are superseded by `nba_props.py` and `run_daily.py`. They remain in the repo as reference implementations.

| Script | Purpose | Status |
|---|---|---|
| `task1_data_foundation.py` | Raw data cleaning | Archived |
| `task2_rolling_features.py` | Rolling averages | Archived |
| `task3_opponent_features.py` | Opponent defense context | Archived |
| `task4_schedule_features.py` | Schedule/fatigue features | Archived |
| `task5_final_dataset.py` | Final feature set assembly | Archived |
| `task5b_rigorous_validation.py` | Leakage validation | Archived |
| `task6_train_models.py` | Model training | Archived |
| `task6_train_models_optimized.py` | Optimized training | Archived |
| `task7_daily_predictions.py` | Daily predictions | Superseded by nba_props.py |
| `task8_betting_lines.py` | Betting line comparison | Superseded by nba_props.py |
| `task9_backtesting.py` | Backtesting framework | Reference |
| `task10_deployment.py` | Deployment pipeline | Superseded by run_daily.py |

---

## Known Issues

- STL and BLK models are weak and excluded from picks by default.
- Betslip win rate (20%) is low — parlays are hard by nature.
- No live injury feed; minutes prediction is used as a proxy.
- `IMPROVEMENTS_SUMMARY.md` references a stale Ubuntu path — doc is outdated but harmless.
- Two `data/` directories exist locally (`data/` and `files/data/`) — `run_daily.py` writes to `files/data/`, `nba_props.py` reads from `files/data/`. These should be consolidated.

---

## Immediate Priorities

1. Consolidate the two `data/` directories into one canonical location.
2. Archive or remove the legacy `files/scripts/pipeline/` task scripts.
3. Improve STL/BLK models or permanently remove them from the pipeline.
4. Add a real injury feed (e.g., ESPN or Rotowire API) to replace minutes-based proxy filtering.
