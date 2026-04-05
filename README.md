<div align="center">
  <h1>🏀 Fresh Start NBA Prop Predictor</h1>
  <p><strong>A First-Principles Machine Learning System for NBA Player Props</strong></p>

  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/ML-scikit--learn-orange.svg" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Win_Rate-56.4%25-success.svg" alt="Win Rate">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</div>

---

> **Overview:** An automated, 5-layer prediction engine built with Python, scikit-learn, and real NBA data from `nba_api` + PrizePicks. It features a fully automated daily pipeline: fetch data, generate picks, and auto-grade results.

## 📊 Live Performance (as of March 2026)

| Metric | Value |
|---|---|
| **Overall Win Rate** | **56.4%** (474 W / 366 L — 840 graded picks) |
| **Betslip Win Rate** | 20.0% (3 W / 12 L — parlay slips) |
| **Data Coverage** | 75,799 player-game rows · 796 unique players · 3 seasons |
| **PrizePicks Lines Tracked** | 21,474 rows · 343 players |

### Win Rate by Prop Type
| Prop | W | L | Win % |
|---|---|---|---|
| **AST** | 19 | 7 | **73.1%** |
| **PA** (Pts+Asts) | 121 | 75 | **61.7%** |
| **TRB** | 29 | 18 | **61.7%** |
| **PR** (Pts+Rebs) | 114 | 87 | 56.7% |
| **PTS** | 90 | 84 | 51.7% |
| **PRA** (Pts+Rebs+Asts) | 101 | 95 | 51.5% |

> **Note:** STL and BLK accuracy is low due to high variance. The system currently avoids generating picks for them.

---

## ⚙️ How It Works (The 5-Layer Engine)

This system operates on a **two-part automated pipeline** powered by a rigorous 5-layer analytical architecture:

1. **Layer 1 — Play-by-Play (PBP) Foundation:** Raw possession-level data fetched from the NBA API.
2. **Layer 2 — PPP Engine:** Converts raw PBP data into Pace-adjusted Points Per Possession (PPP) using Ridge Regression to isolate true offensive talent from defensive quality.
3. **Layer 3 — Usage & Injury Model:** Redistributes usage and minutes when key players sit out, applying efficiency penalties where necessary.
4. **Layer 4 — Luck Regression:** Removes variance by regressing opponent FT% and 3PT% toward the league average, relying on expected eFG% rather than hot streaks.
5. **Layer 5 — Blowout Discount:** Filters out garbage time from training data to maintain projection purity.

### The Daily Pipeline
- **Morning Fetch (`run_daily.py`):** Pulls last night's box scores, scrapes today's PrizePicks lines, and auto-grades all ungraded picks and betslips.
- **Prediction Engine (`nba_props.py`):** Generates today's picks, applies bias corrections, and outputs sortable dashboards and betslips.

---

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install pandas numpy scikit-learn requests nba_api playwright streamlit
playwright install chromium
```

### 2. Configure PrizePicks (One-Time Setup)
PrizePicks uses Cloudflare bot detection. The scraper tries a **direct API call first**. If that fails, it falls back to your Chrome profile.

Set the environment variable for your Chrome profile:
```bash
# Windows
set PRIZEPICKS_CHROME_PROFILE=C:\Users\<YourUsername>\AppData\Local\Google\Chrome\User Data
```

### 3. Run Daily Operations
```bash
# Recommended: run every morning (incremental update + today's lines)
python run_daily.py

# Generate Predictions
python nba_props.py predict

# Launch the Visual Dashboard
python -m streamlit run vegas_gui.py
```

### 4. Automate via Task Scheduler
Use `run_morning.bat` via Windows Task Scheduler each morning at ~9am to automate the entire process.

---

## 📂 Project Structure

```text
Fresh_Start_NBA/
├── run_daily.py              # Step 1 — Morning data fetcher + auto-grader
├── nba_props.py              # Step 2 — Prediction engine (train/predict/backtest)
├── vegas_gui.py              # Interactive Streamlit operations dashboard
├── ppp_engine.py             # Layer 2: Pace/PPP opponent-adjusted ratings
├── pbp_fetcher.py            # Layer 1: Fetches cached PBP data
├── usage_injury_model.py     # Layer 3: Usage rate and injury replacement
├── luck_regression_model.py  # Layer 4: Luck regression / variance removal
├── blowout_discount_model.py # Layer 5: Garbage time filtering
├── prizepicks_scraper.py     # PrizePicks prop line scraper
├── nba_scraper.py            # NBA.com game log scraper (via nba_api)
├── run_morning.bat           # Windows batch launcher for Task Scheduler
├── run_gui.bat               # Windows batch launcher for the Streamlit dashboard
├── data/                     # Historical logs, cached lines, and daily projections
├── models/                   # Trained ML model files (.pkl, .json)
├── output/                   # Generated picks, betslips, and HTML reports
└── logs/                     # Daily execution logs
```

---

## 🛠 Tech Stack

- **Language:** Python 3.11+
- **ML Models:** scikit-learn (GradientBoosting, RandomForest, Ridge, Isotonic Calibration)
- **Data Sources:** `nba_api`, PrizePicks public API
- **UI/Dashboard:** Streamlit, HTML
- **Automation:** Windows Task Scheduler + `.bat` launchers

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Sports betting involves risk, and past performance does not guarantee future results. Always gamble responsibly and within your means.

---
<div align="center">
  <i>Built by Jake — Fresh Start NBA</i>
</div>
