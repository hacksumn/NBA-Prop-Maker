# NBA Prop Maker 🏀

> A machine learning system for predicting NBA player prop bets — built with Python, scikit-learn, and real NBA data from `nba_api` + PrizePicks. Fully automated daily pipeline: fetch data, generate picks, auto-grade results.

---

## Live Performance (as of March 25, 2026)

| Metric | Value |
|---|---|
| **Overall Win Rate** | **56.4%** (474 W / 366 L — 840 graded picks) |
| **Betslip Win Rate** | 20.0% (3 W / 12 L — parlay slips) |
| **Data Coverage** | 75,799 player-game rows · 796 unique players · 3 seasons |
| **PrizePicks Lines Tracked** | 21,474 rows · 343 players |
| **Last Data Refresh** | 2026-03-23 |

### Win Rate by Prop Type

| Prop | W | L | Win % |
|---|---|---|---|
| **AST** | 19 | 7 | **73.1%** |
| **PA** (Pts+Asts) | 121 | 75 | **61.7%** |
| **TRB** | 29 | 18 | **61.7%** |
| **PR** (Pts+Rebs) | 114 | 87 | 56.7% |
| **PTS** | 90 | 84 | 51.7% |
| **PRA** (Pts+Rebs+Asts) | 101 | 95 | 51.5% |

### Win Rate by Confidence Tier

| Confidence | W | L | Win % |
|---|---|---|---|
| Below 60% | 24 | 14 | **63.2%** |
| 60–69% | 265 | 208 | 56.0% |
| 70–79% | 135 | 101 | 57.2% |
| 80%+ | 50 | 43 | 53.8% |

### Model Accuracy (Regression — MAE / Over-Under Accuracy)

| Prop | MAE | O/U Accuracy |
|---|---|---|
| **PTS** | 4.14 | 78.9% |
| **TRB** | 1.84 | 73.3% |
| **AST** | 1.32 | 77.3% |
| **TOV** | 0.87 | 70.2% |
| **STL** | 0.70 | 49.5% |
| **BLK** | 0.51 | 32.0% |

> **Note:** STL and BLK accuracy is low — these are high-variance stats. The system currently avoids generating picks for them.

---

## What It Does

This system is a **two-part automated pipeline**:

**Part 1 — Data Fetcher (`run_daily.py`)** runs every morning and:
- Pulls last night's NBA box scores via `nba_api` → `data/nba_data.csv`
- Pulls today's PrizePicks prop lines → `data/historical_lines.csv`
- Auto-grades all ungraded picks and betslips against real box scores

**Part 2 — Prediction Engine (`nba_props.py`)** generates today's picks:
- Builds rolling features (L3/L5/L10/L20 averages, EWMA, std dev)
- Opponent matchup context and schedule fatigue features
- Minutes prediction model to filter injured/limited players
- Bias correction per prop type (`models/bias.json`)
- Outputs picks, betslips, and a sortable HTML dashboard

---

## Quick Start

### 1. Install Requirements

```bash
pip install pandas numpy scikit-learn requests nba_api playwright
playwright install chromium
```

### 2. Configure PrizePicks (One-Time Setup)

PrizePicks uses Cloudflare bot detection. The scraper tries a **direct API call first** (works most days). If that fails, it falls back to your Chrome profile.

Edit `prizepicks_scraper.py` and set your Chrome profile path:

```python
# Windows:
CHROME_PROFILE = r"C:\Users\<YourUsername>\AppData\Local\Google\Chrome\User Data"
# macOS:
CHROME_PROFILE = "/Users/<YourUsername>/Library/Application Support/Google/Chrome"
# Linux:
CHROME_PROFILE = "/home/<YourUsername>/.config/google-chrome"
```

Or set the environment variable:
```bash
# Windows
set PRIZEPICKS_CHROME_PROFILE=C:\Users\<YourUsername>\AppData\Local\Google\Chrome\User Data
# macOS/Linux
export PRIZEPICKS_CHROME_PROFILE="/Users/<YourUsername>/Library/Application Support/Google/Chrome"
```

### 3. Run Daily

```bash
# Recommended: run every morning (incremental update + today's lines)
python run_daily.py

# Full refresh (re-pull all 3 seasons — use once or after a long gap)
python run_daily.py --full-refresh

# Only update prop lines
python run_daily.py --lines-only

# Only update game logs
python run_daily.py --logs-only
```

### 4. Generate Predictions

```bash
python nba_props.py predict
```

Outputs:
- `output/picks_latest.csv` — today's top picks
- `output/dashboard_latest.html` — visual sortable dashboard
- `output/betslips_history.csv` — 2/3/4-pick Power Play slips (auto-graded next morning)

### 5. Retrain Models

```bash
python nba_props.py train
```

Rebuilds all features, runs time-series cross-validation, saves OOF biases to `models/bias.json`, and updates classifiers.

### 6. Automate (Windows Task Scheduler)

Use `run_morning.bat` via Windows Task Scheduler each morning at ~9am:

```bat
@echo off
cd /d "C:\Users\jakep\Downloads\Fresh_Start_NBA"
python run_daily.py >> logs\morning_task.log 2>&1
python nba_props.py predict >> logs\morning_task.log 2>&1
```

---

## Project Structure

```
NBA-Prop-Maker/
│
├── run_daily.py              # Step 1 — Morning data fetcher + auto-grader
├── nba_props.py              # Step 2 — Prediction engine (train/predict/backtest)
├── nba_scraper.py            # NBA.com game log scraper (via nba_api)
├── prizepicks_scraper.py     # PrizePicks prop line scraper
├── run_morning.bat           # Windows batch launcher for Task Scheduler
├── show_betslip.py           # Utility: print today's betslip as JSON
├── full_pipeline.py          # Full end-to-end research pipeline
├── feature_pipeline.py       # Modular feature engineering class
├── advanced_features.py      # 159-feature advanced engineering module
├── train_advanced_models.py  # Advanced model training script
├── generate_picks_v2.py      # Legacy classifier-based pick generator
├── minutes_model.py          # Minutes prediction subsystem
│
├── data/
│   ├── nba_data.csv          # Historical player game logs (75k+ rows)
│   ├── historical_lines.csv  # PrizePicks prop lines history (21k+ rows)
│   └── player_positions.csv  # Player position lookup
│
├── models/
│   ├── models.pkl            # Trained regression models
│   ├── bias.json             # Per-prop bias correction values
│   ├── results.json          # Model MAE + accuracy metrics
│   ├── ou_classifiers.pkl    # O/U classifier bundle
│   ├── classifiers/          # Per-prop O/U classifier .pkl files
│   └── regression/           # Per-prop regression .pkl files
│
├── output/
│   ├── picks_latest.csv      # Today's picks
│   ├── picks_history.csv     # All-time picks log (graded)
│   ├── betslips_history.csv  # All-time betslip log (graded)
│   └── dashboard_latest.html # Visual dashboard
│
├── scripts/
│   ├── pipeline/             # task1–task10: full ML pipeline scripts
│   └── utils/                # Helper utilities (grading, data building, etc.)
│
├── results/                  # Archived prediction/bet CSVs + Vegas lines
├── logs/                     # Daily run logs
└── feature_cols.json         # Selected feature column names for models
```

---

## Key Design Decisions

**Why `nba_api` instead of a paid Odds API?**
Free, reliable, and covers 3 full seasons of player-level box scores. No API key required.

**Why PrizePicks?**
It's the most popular DFS prop platform and offers a free public API endpoint that works without authentication most days.

**Why bias correction?**
The regression models consistently underpredict certain stats. `bias.json` stores the average residual per prop and adds it back at prediction time.

**Why filter by minutes prediction?**
A player projected for 15 minutes is a bad pick regardless of their rolling averages. The minutes model filters out players likely to be limited or DNP.

**Why auto-grade betslips?**
Tracking real P&L against actual box scores is the only honest way to know if the model is working.

---

## Known Limitations

- **STL and BLK** models are weak (32–50% accuracy). These props are excluded from picks by default.
- **Betslip win rate (20%)** is low — parlays are hard. Individual pick win rate (56.4%) is the more meaningful number.
- The PrizePicks scraper may break if PrizePicks changes their API structure.
- No live injury feed — the system uses minutes prediction as a proxy for availability.
- The `IMPROVEMENTS_SUMMARY.md` references an Ubuntu path (`/home/ubuntu/NBA-Prop-Maker`) — this is a stale doc from a previous dev environment, not a bug.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| ML Models | scikit-learn (GradientBoosting, RandomForest, Ridge, Isotonic Calibration) |
| Data Source | `nba_api`, PrizePicks public API |
| Automation | Windows Task Scheduler + `.bat` launcher |
| Output | CSV + HTML dashboard |

---

## Disclaimer

This project is for **educational and research purposes only**. Sports betting involves risk, and past performance does not guarantee future results. Always gamble responsibly and within your means.

---

*Built by Jake — Fresh Start NBA · Last updated March 25, 2026*
