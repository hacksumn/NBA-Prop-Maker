<div align="center">
  <h1>🏀 Fresh Start NBA Prop Predictor</h1>
  <p><strong>A First-Principles Machine Learning System for NBA Player Props</strong></p>

  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/ML-scikit--learn-orange.svg" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Picks_Graded-1%2C281-informational.svg" alt="Picks Graded">
  <img src="https://img.shields.io/badge/Win_Rate-54.1%25-success.svg" alt="Win Rate">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg" alt="Streamlit">
</div>

---

> **Fresh Start NBA** is a fully automated, first-principles NBA player prop prediction engine. It runs every morning on Windows via Task Scheduler — fetching last night's box scores, scraping today's PrizePicks lines, grading yesterday's picks, and generating a ranked list of today's best prop bets. No manual input required.

---

## 📊 Live Performance (as of April 5, 2026)

| Metric | Value |
|---|---|
| **Overall Pick Win Rate** | **54.1%** (693 W / 588 L) |
| **Total Graded Picks** | 1,281 |
| **Betslip Win Rate** | **26.1%** (12 W / 34 L) |
| **Total Graded Betslips** | 46 |
| **Data Coverage** | 77,890 player-game rows · 800 unique players · 3 seasons |
| **PrizePicks Lines Tracked** | 29,986 rows · 364 players |
| **Last Box Score Date** | 2026-04-04 |

### Model Accuracy (Regression — MAE / Over-Under Accuracy)

| Prop | MAE | O/U Accuracy |
|---|---:|---:|
| **PTS** | 4.08 | 71.4% |
| **TRB** | 1.75 | 64.4% |
| **AST** | 1.31 | 60.6% |
| **STL** | 0.69 | 53.4% |
| **TOV** | 0.87 | 56.0% |
| **BLK** | 0.49 | 33.0% |

### Advanced Model Snapshot (Market-Scoring AUC)

| Prop | Real-Line Hit Rate | Market Sign Accuracy | Meta AUC |
|---|---:|---:|---:|
| **BLK** | 74.3% | 74.0% | **0.846** |
| **STL** | 71.8% | 64.9% | 0.548 |
| **AST** | 58.0% | 51.4% | 0.558 |
| **TRB** | 55.6% | 49.1% | 0.527 |
| **TOV** | 56.3% | 50.9% | 0.476 |
| **PTS** | 50.5% | 49.6% | 0.488 |

---

## ⚙️ How It Works — The 5-Layer Engine

The prediction engine is built on five stacked analytical layers. Each one removes a specific source of noise from the raw data.

| Layer | Module | What It Does |
|---|---|---|
| **L1 — PBP Foundation** | `pbp_fetcher.py` | Fetches and caches possession-level play-by-play data from the NBA API |
| **L2 — PPP Engine** | `ppp_engine.py` | Converts PBP into Pace-adjusted PPP using Ridge Regression to isolate true talent from opponent quality |
| **L3 — Usage & Injury** | `usage_injury_model.py` | Redistributes usage/minutes when stars sit out; applies efficiency penalties for forced shots |
| **L4 — Luck Regression** | `luck_regression_model.py` | Regresses opponent FT% and 3PT% to league average; uses expected eFG% instead of hot/cold streaks |
| **L5 — Blowout Discount** | `blowout_discount_model.py` | Filters garbage time from training data (≥25pt Q3, ≥15pt Q4, ≥10pt final 3 min) |

### The Daily Pipeline (`run_daily.py`)

```
NBA API (box scores)  →  nba_data.csv
                              ↓
                       pbp_fetcher.py  →  hist_cache/pbp_dates/
                              ↓
                       ppp_engine.py  (Ridge Regression)
                              ↓
               usage_injury_model.py + minutes_model.py
                              ↓
                    luck_regression_model.py
                              ↓
                  blowout_discount_model.py
                              ↓
              feature_pipeline.py  →  trained models (models/)
                              ↓
                     nba_props.py predict
                              ↓
         picks_latest.csv + betslips_latest.csv + dashboard.html
```

---

## 🖥️ Operator Entry Points

There are two ways to run the system:

**1. Automated Morning Pipeline (Production)**
```bat
run_morning.bat  →  run_daily.py
```
Scheduled via Windows Task Scheduler at ~9am. Runs the full pipeline: game logs, auto-grading, PrizePicks lines, all 5 modeling layers, projection merge, optional weekly retrain, and live pick generation.

**2. Interactive Dashboard (Manual Control)**
```bat
run_gui.bat  →  vegas_gui.py  (Streamlit)
```
Launches the local "Vegas Intelligence Console" — a Streamlit dashboard that reads from `data/`, `models/`, and `output/` and exposes picks, performance metrics, model room, and command controls.

---

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install pandas numpy scikit-learn requests nba_api playwright streamlit openpyxl
playwright install chromium
```

### 2. Configure PrizePicks (One-Time Setup)
PrizePicks uses Cloudflare bot detection. The scraper tries a **direct API call first** (works most days). If that fails, it falls back to your Chrome profile.

```bash
# Windows — set your Chrome profile path
set PRIZEPICKS_CHROME_PROFILE=C:\Users\<YourUsername>\AppData\Local\Google\Chrome\User Data
```

### 3. Run Daily Operations

```bash
# Full morning run (recommended)
python run_daily.py

# Full refresh — re-pull all seasons from scratch
python run_daily.py --full-refresh

# Only update prop lines
python run_daily.py --lines-only

# Only update game logs
python run_daily.py --logs-only

# Skip the 5-layer modeling (data fetch only)
python run_daily.py --skip-layers

# Generate predictions only
python nba_props.py predict

# Retrain all models
python nba_props.py train
```

### 4. Launch the Dashboard
```bash
python -m streamlit run vegas_gui.py
# or double-click run_gui.bat
```

### 5. Automate via Windows Task Scheduler
Double-click or schedule `run_morning.bat` at ~9am daily. All output is appended to `logs/morning_task.log`.

---

## 📂 Project Structure

```text
Fresh_Start_NBA/
│
├── run_daily.py              # Master daily orchestrator — runs everything in order
├── nba_props.py              # Prediction engine (train / predict / backtest)
├── vegas_gui.py              # Streamlit "Vegas Intelligence Console" dashboard
│
├── ppp_engine.py             # Layer 2: Pace/PPP opponent-adjusted ratings
├── pbp_fetcher.py            # Layer 1: Fetches and caches PBP data
├── usage_injury_model.py     # Layer 3: Usage rate and injury replacement model
├── luck_regression_model.py  # Layer 4: Luck regression / variance removal
├── blowout_discount_model.py # Layer 5: Garbage time filtering
│
├── feature_pipeline.py       # Modular feature engineering class
├── train_advanced_models.py  # Advanced model training script
├── calibrate_confidence.py   # Confidence score calibration
├── prizepicks_scraper.py     # PrizePicks prop line scraper
├── nba_scraper.py            # NBA.com game log scraper (via nba_api)
├── minutes_model.py          # Minutes projection model
├── minutes_deviation.py      # Minutes deviation tracker
│
├── run_morning.bat           # Windows batch launcher for Task Scheduler
├── run_gui.bat               # Windows batch launcher for Streamlit dashboard
│
├── data/
│   ├── nba_data.csv                  # Master box score dataset (77k+ rows)
│   ├── historical_lines.csv          # All PrizePicks lines ever scraped (30k+ rows)
│   ├── player_projections_today.csv  # Today's merged projection file (consumed by picker)
│   ├── projection_archive/           # Archived daily projection snapshots
│   └── hist_cache/pbp_dates/         # Cached PBP CSVs by date
│
├── models/
│   ├── bias.json                     # Per-prop bias correction values
│   ├── bias_advanced.json            # Advanced model bias corrections
│   ├── results.json                  # Regression MAE + O/U accuracy metrics
│   ├── training_results.json         # Advanced training metrics + AUC scores
│   ├── confidence_calibrator.pkl     # Isotonic regression confidence calibrator
│   └── [per-prop XGB + meta .pkl files]
│
├── output/
│   ├── picks_latest.csv              # Today's filtered live picks
│   ├── picks_history.csv             # All historical picks (graded WIN/LOSS/PUSH)
│   ├── betslips_latest.csv           # Current betslips
│   ├── betslips_history.csv          # All betslips with graded results
│   ├── dashboard_latest.html         # Auto-generated HTML dashboard
│   └── archive/                      # Dated output snapshots
│
├── colab_notebooks/                  # Jupyter notebooks for EDA and model training
├── logs/                             # Daily execution logs
├── .streamlit/config.toml            # Streamlit theme configuration
│
├── CLAUDE.md                         # AI assistant context — read first
├── ARCHITECTURE.md                   # Deep-dive first-principles model architecture
├── PROJECT_STATE.md                  # Current performance metrics and known issues
├── CHANGELOG.md                      # Version history
├── feature_cols.json                 # Active feature column list
└── feature_cols_advanced.json        # Advanced feature column list
```

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **ML Models** | scikit-learn (GradientBoosting, RandomForest, Ridge, Isotonic Calibration) |
| **Advanced Models** | XGBoost (per-prop edge models + meta classifiers) |
| **Data Sources** | `nba_api`, PrizePicks public API |
| **Dashboard** | Streamlit, HTML |
| **Automation** | Windows Task Scheduler + `.bat` launchers |
| **Output** | CSV, Excel, HTML dashboard |

---

## 📋 Known Issues

- No real-time injury/status feed — the system relies on detected absences and usage/minutes modeling as a proxy. Live injury feed integration is the highest-priority improvement.
- BLK model has strong AUC (0.846) but low raw O/U accuracy (33%) — high variance stat.
- Betslip win rate reflects parlay math; individual pick win rate is the more meaningful signal.
- **PrizePicks 403/429:** PrizePicks blocks direct API access via Cloudflare. The scraper falls back through three strategies: (1) direct API, (2) session cookie, (3) Playwright with real Chrome profile. When all fail, picks are generated against Odds API lines — a `LINE SOURCE WARNING` is printed to the log. Always check this warning before betting.
- **Step 1.6 betslip grading gap:** Betslip grading has no live NBA API fallback. If Step 1 missed a game date, betslips from that date stay UNGRADED until the next successful data fetch.

## 📊 picks_history.csv Column Reference

| Column | Description |
|---|---|
| `game_date` | Date picks were generated |
| `player`, `prop`, `line` | Player, stat type, and line used |
| `prediction`, `direction` | Model's projected value and OVER/UNDER call |
| `confidence` | Model confidence (0–100) |
| `actual`, `result` | Box score result and WIN/LOSS/PUSH grade |
| `pick_source` | `'market_model'`, `'structural_micro'`, or `'volume_fill'` |
| `line_source` | `'prizepicks'` or `'odds_api'` — track this for grading validity |

**`volume_fill` picks** are generated by a relaxed second pass when the primary filter produces fewer than 10 candidates. They are confidence-capped at 63% and should not be used as betslip legs.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Sports betting involves risk, and past performance does not guarantee future results. Always gamble responsibly and within your means.

---

<div align="center">
  <i>Built by Jake — Fresh Start NBA · Last updated April 5, 2026</i>
</div>
