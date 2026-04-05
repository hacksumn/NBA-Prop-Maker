# Project State - Fresh Start NBA

**Last Updated:** April 5, 2026  
**Status:** Active - daily orchestrator, advanced model bundle, and Streamlit GUI are all present in the local repo.

---

## Current Operating Shape

The project now has two main operator entry points:

1. `run_morning.bat` -> `run_daily.py`
   - This is the scheduled production path.
   - `run_daily.py` now owns the full morning flow: game logs, auto-grading, PrizePicks lines, five modeling layers, projection merge/archive, optional weekly retrain, and live pick generation via `nba_props.py predict`.

2. `run_gui.bat` -> `vegas_gui.py`
   - This launches the local Streamlit "Vegas Intelligence Console".
   - The GUI reads from `data/`, `models/`, and `output/` and exposes picks, performance, model-room, and command controls.

The repo-root `data/` directory is now the canonical live data location. The older note about `files/data/` being the active path is no longer true for the current code.

---

## Live Snapshot (as of 2026-04-05)

| Metric | Value |
|---|---|
| Overall Pick Win Rate | **54.1%** (693 W / 588 L) |
| Total Graded Picks | 1,281 |
| Pending Picks | 92 |
| Betslip Win Rate | **26.1%** (12 W / 34 L) |
| Total Graded Betslips | 46 |
| Pending Betslips | 5 |
| Data Rows | 77,890 player-game rows |
| Unique Players | 800 |
| Seasons Covered | 2023-24, 2024-25, 2025-26 |
| Historical Line Rows | 29,986 |
| Historical Line Players | 364 |
| Last Box Score Date | 2026-04-04 |
| Last Line Snapshot Date | 2026-04-05 |
| Detected Absence Rows | 173 |

Today's live output currently contains 6 picks across `BLK`, `STL`, and `TRB`, with average confidence 77.5%. `output/betslips_latest.csv` currently contains 5 slips.

---

## Recent Structural Changes

- `run_daily.py` is now the single morning orchestrator instead of a simple fetch-only step.
- `vegas_gui.py` and `run_gui.bat` were added for a local interactive control room.
- Advanced model artifacts were refreshed on 2026-04-05, including:
  - `models/training_results.json`
  - `models/bias_advanced.json`
  - `models/selected_features_advanced.json`
  - `models/feature_importance_advanced.json`
  - `models/market_feature_sets_advanced.json`
  - `models/confidence_calibrator.pkl`

---

## Model Metrics (Current Regression Snapshot)

Source: `models/results.json`

| Prop | MAE | O/U Accuracy |
|---|---:|---:|
| PTS | 4.08 | 71.4% |
| TRB | 1.75 | 64.4% |
| AST | 1.31 | 60.6% |
| STL | 0.69 | 53.4% |
| BLK | 0.49 | 33.0% |
| TOV | 0.87 | 56.0% |

---

## Advanced Training Snapshot

Source: `models/training_results.json`

| Prop | Real-Line Hit Rate | Market Sign Accuracy | Meta AUC |
|---|---:|---:|---:|
| PTS | 50.5% | 49.6% | 0.488 |
| TRB | 55.6% | 49.1% | 0.527 |
| AST | 58.0% | 51.4% | 0.558 |
| STL | 71.8% | 64.9% | 0.548 |
| BLK | 74.3% | 74.0% | 0.846 |
| TOV | 56.3% | 50.9% | 0.476 |

Important note: the old blanket statement that STL/BLK are excluded from live consideration is no longer an accurate description of the current repo state. The current live board can surface STL/BLK picks through market-model scoring even though structural STL/BLK fallback picks remain disabled by config.

---

## Advanced Bias Correction Values

Source: `models/bias_advanced.json`

| Prop | Bias |
|---|---:|
| PTS | 0.2784 |
| TRB | 0.0305 |
| AST | 0.0150 |
| PRA | 0.1518 |
| PR | 0.1880 |
| PA | 0.2089 |
| STL | -0.0009 |
| BLK | 0.0228 |
| TOV | 0.0189 |

---

## Active Outputs

- `data/player_projections_today.csv` - current merged projection file consumed by the picker
- `data/projection_archive/` - archived projection snapshots
- `output/predictions_latest.csv` - full prediction board
- `output/picks_latest.csv` - filtered live picks
- `output/betslips_latest.csv` - current slips
- `output/dashboard_latest.html` - HTML dashboard
- `output/archive/` - dated output snapshots

---

## Known Issues / Reality Check

- `README.md` and older docs still describe the March 25, 2026 state and need to be brought in line with the current orchestrator + GUI setup.
- There is still no true real-time injury/status feed; the system relies on detected absences plus usage/minutes modeling.
- The live prop gating logic is more complex than the old docs suggest, especially for STL/BLK and under-only behavior.
- The repo currently mixes source files with many generated model and output artifacts, so commit hygiene and retention policy are still worth tightening.

---

## Immediate Priorities

1. Refresh `README.md` and any stale docs so they match the April 2026 workflow.
2. Decide whether the Streamlit GUI should be treated as the primary operator interface.
3. Revisit live eligibility rules for STL/BLK and other micro-props now that advanced validation is stronger.
4. Add a real injury/status feed instead of relying mostly on proxy signals.
5. Define a cleaner policy for archiving or ignoring generated artifacts under `models/` and `output/`.
