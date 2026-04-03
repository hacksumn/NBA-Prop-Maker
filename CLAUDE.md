# CLAUDE.md — Fresh Start NBA Prop Predictor

> This file is the single source of truth for any AI assistant working on this project.
> Read this file completely before touching any code, suggesting any changes, or answering any questions.

---

## What This Project Is

**Fresh Start NBA** is a fully automated, first-principles NBA player prop prediction system built and owned by Jake. It runs every morning on a Windows machine via Task Scheduler, fetches last night's box scores and today's PrizePicks lines, grades yesterday's picks, and generates a ranked list of today's best prop bets — all without any manual input.

This is not a hobby script. This is a live, money-on-the-line prediction engine that Jake uses daily on PrizePicks. Every change you make has real financial consequences. Treat it accordingly.

---

## The Core Goal

**Identify NBA player props where the model's projection meaningfully disagrees with the PrizePicks line — and bet the ones where the edge is largest.**

The system targets PrizePicks Power Play slips (parlays of 2, 3, or 4 picks). The model must:

1. Project each player's stat line for tonight's game with maximum accuracy.
2. Compare that projection to the PrizePicks line.
3. Rank picks by confidence and edge size.
4. Automatically build and log 2-pick, 3-pick, and 4-pick betslips.
5. Grade all picks and slips the next morning using real box score data.
6. Track everything in clean, readable CSV and Excel files.

---

## The Model Architecture (5 Layers)

The prediction engine is built on five stacked analytical layers. Each one removes a specific source of noise from the raw data.

### Layer 1 — Play-by-Play (PBP) Foundation
Raw possession-level data fetched from the NBA API via `pbp_fetcher.py`. This is the ground truth. Every other layer builds on top of it. PBP data is cached in `data/hist_cache/pbp_dates/` by date to avoid redundant API calls.

### Layer 2 — PPP Engine (`ppp_engine.py`)
Converts raw PBP data into **Pace-adjusted Points Per Possession (PPP)** ratings. This is the core insight: a player's raw stats are meaningless without knowing how many possessions occurred and how hard the opponent's defense was. The PPP engine uses Ridge Regression to simultaneously solve for every team's true offensive and defensive strength, isolating player talent from opponent quality.

**The math:**
- Possessions = `FGA + 0.44 × FTA − ORB + TOV`
- Offensive Rating = `(Points / Possessions) × 100`
- Adjusted PPP = `Player_Base_PPP + Opponent_Defensive_Adjustment`

### Layer 3 — Usage & Injury Model (`usage_injury_model.py`, `minutes_model.py`, `minutes_deviation.py`)
When a star player sits out, their usage and minutes must be redistributed. This layer:
- Tracks each player's Usage Rate (USG%) and how it changes when teammates are absent.
- Applies an efficiency penalty to players absorbing extra usage (forced shots = lower PPP).
- Uses minutes deviation as a proxy for injury impact when no live injury feed is available.

**Known gap:** No live injury feed. Minutes prediction is used as a proxy. A real feed (ESPN/Rotowire API) is a top priority.

### Layer 4 — Luck Regression (`luck_regression_model.py`)
Separates skill from variance. The three pillars:
1. **Opponent FT%** — Regressed to league average (defenses don't control FT makes).
2. **Opponent 3PT%** — Regressed toward league average (defenses control attempts, not makes).
3. **Shooting variance** — Uses expected eFG% from shot location data, not raw hot/cold streaks.

### Layer 5 — Blowout Discount (`blowout_discount_model.py`)
Filters out garbage time from training data and adjusts projections for games likely to become blowouts. Garbage time thresholds:
- Score differential ≥ 25 in Q3
- Score differential ≥ 15 in Q4
- Score differential ≥ 10 in final 3 minutes

---

## File & Directory Structure

```
Fresh_Start_NBA/
│
├── run_daily.py              # Master daily pipeline — runs everything in order
├── run_morning.bat           # Windows batch file that chains run_daily.py + nba_props.py
├── nba_props.py              # Prediction engine — generates picks, betslips, HTML dashboard
│
├── ppp_engine.py             # Layer 2: Pace/PPP opponent-adjusted ratings
├── pbp_fetcher.py            # Fetches and caches play-by-play data from NBA API
├── usage_injury_model.py     # Layer 3: Usage rate and injury replacement model
├── minutes_model.py          # Minutes projection model
├── minutes_deviation.py      # Minutes deviation tracker
├── luck_regression_model.py  # Layer 4: Luck regression / variance removal
├── blowout_discount_model.py # Layer 5: Garbage time filtering
├── feature_pipeline.py       # Feature engineering pipeline
├── prizepicks_scraper.py     # Scrapes today's PrizePicks NBA prop lines
├── nba_scraper.py            # Fetches NBA box score data
├── train_advanced_models.py  # Model training script
├── calibrate_confidence.py   # Confidence score calibration
├── build_historical_dataset.py # Historical dataset builder
├── outlier_heatmap.py        # Outlier detection visualization
│
├── feature_cols.json         # Active feature column list
├── feature_cols_advanced.json # Advanced feature column list
│
├── data/
│   ├── nba_data.csv          # Master box score dataset (75,000+ player-game rows)
│   ├── historical_lines.csv  # All PrizePicks lines ever scraped (26,000+ rows)
│   ├── hist_cache/
│   │   └── pbp_dates/        # Cached PBP CSVs by date (pbp_YYYY-MM-DD.csv)
│   └── [other data files]
│
├── models/                   # Trained ML model files (.pkl, .joblib)
│
├── output/
│   ├── picks_latest.csv      # Today's generated picks (ranked by confidence)
│   ├── picks_history.csv     # All historical picks with WIN/LOSS/PUSH grades
│   ├── betslips_history.csv  # All betslips with individual pick columns + results
│   ├── betslips_history.xlsx # Clean Excel version with formatting and summary tabs
│   └── dashboard.html        # Daily HTML dashboard (auto-generated)
│
├── logs/
│   └── run_YYYYMMDD.log      # Daily execution log
│
├── CLAUDE.md                 # This file — read first, always
├── ARCHITECTURE.md           # Deep-dive model architecture reference
├── PROJECT_STATE.md          # Current performance metrics and known issues
├── CHANGELOG.md              # Version history
└── README.md                 # Public-facing project overview
```

---

## The Daily Pipeline (`run_daily.py`)

The pipeline runs in strict sequential order. Each step must succeed before the next begins.

| Step | Function | What It Does |
|------|----------|--------------|
| Step 1 | `step1_update_nba_data()` | Fetches last night's box scores into `nba_data.csv` |
| Step 1.5 | `step1_5_grade_picks()` | Grades all ungraded rows in `picks_history.csv` against box scores. Has a **live NBA API fallback** if a date is missing from `nba_data.csv`. |
| Step 1.6 | `step1_6_grade_betslips()` | Grades all ungraded betslips in `betslips_history.csv`. **Currently lacks the live fallback** — known bug, fix in progress. |
| Step 2 | `step2_prizepicks_lines()` | Scrapes today's PrizePicks lines into `historical_lines.csv` |
| Step 3+ | `nba_props.py predict` | Runs the full prediction engine, generates picks and betslips |

**Key rule:** Step 1.5 can fetch missing dates live from the NBA API. Step 1.6 cannot — it only grades if the date is already in `nba_data.csv`. This is the root cause of the "No betslips could be graded" bug.

---

## Data Flow

```
NBA API (box scores)
        ↓
  nba_data.csv  ←──────────────────────────────────────┐
        ↓                                               │
  pbp_fetcher.py → hist_cache/pbp_dates/               │
        ↓                                               │
  ppp_engine.py (Ridge Regression)                      │
        ↓                                               │
  usage_injury_model.py + minutes_model.py              │
        ↓                                               │
  luck_regression_model.py                              │
        ↓                                               │
  blowout_discount_model.py                             │
        ↓                                               │
  feature_pipeline.py → trained models (models/)        │
        ↓                                               │
  nba_props.py predict                                  │
        ↓                                               │
  picks_latest.csv + betslips_history.csv               │
        ↓                                               │
  PrizePicks lines (prizepicks_scraper.py) ─────────────┘
        ↓
  dashboard.html (daily output)
```

---

## Current Performance (as of March 2026)

| Metric | Value |
|--------|-------|
| Individual Pick Win Rate | **53–57%** (varies by day) |
| Best Prop Categories | AST (72%), TRB (62%), PA (57%) |
| Weakest Prop Categories | PTS (49%), STL (excluded), BLK (excluded) |
| Betslip Win Rate | ~14–20% (parlay math — expected) |
| Data Coverage | 75,000+ player-game rows, 3 seasons |
| PrizePicks Lines Tracked | 26,000+ rows |

**Key insight:** The model has real edge at the individual pick level. The betslip win rate is low because parlays mathematically compound losses. The 2-pick slip is the most efficient format; 4-pick slips are negative EV at PrizePicks payout rates.

---

## Known Issues & Priorities

| Issue | Priority | Status |
|-------|----------|--------|
| Step 1.6 lacks live NBA API fallback for betslip grading | **HIGH** | Fix in progress |
| No live injury feed — using minutes proxy | **HIGH** | Planned |
| STL and BLK models are weak (excluded from picks) | Medium | Acknowledged |
| Two `data/` directories exist — needs consolidation | Medium | Planned |
| Betslip history had picks crammed into one cell | Fixed | Done |

---

## Optimization Targets (What We're Building Toward)

These are the next-level improvements that will meaningfully increase edge:

1. **Live Injury Feed** — Replace the minutes-proxy with a real-time injury/lineup source (ESPN API, Rotowire, or RotoGrinders). This is the single highest-leverage improvement available.

2. **Confidence Threshold Filter** — Only build betslips from picks with model confidence ≥ 70%. Current threshold is too low, diluting slip quality.

3. **Prop-Type Filtering** — Restrict betslip picks to AST, TRB, and PA props only. PTS props are near coin-flip and drag down slip win rates.

4. **2-Pick-Only Strategy** — Run a parallel tracking mode with 2-pick slips only to validate whether this format is profitable at current accuracy levels.

5. **Backtesting Module** — Simulate different slip strategies (2-pick only, prop-type filtered, confidence-filtered) against the full `picks_history.csv` dataset to find the optimal configuration before betting real money.

6. **Line Movement Integration** — Track how PrizePicks lines move throughout the day. Sharp money moves lines. If the model agrees with the direction of movement, confidence increases.

7. **OVER/UNDER Bias Analysis** — The model may have a systematic directional bias (e.g., consistently better at UNDER predictions). Identifying and exploiting this bias is free edge.

8. **Web Dashboard** — Real-time performance dashboard accessible from any device, not just the local machine.

---

## Rules for AI Assistants Working on This Project

These are non-negotiable. Follow them every single time.

### 1. Debug Before Delivering
**Never present code, a script, or a patch without first verifying it is syntactically correct and logically sound.** Run the code mentally or actually execute it. Check for:
- Import errors and missing dependencies
- Variable name mismatches between functions
- Off-by-one errors in date logic
- Silent failures (functions that return `None` instead of raising)
- CSV column name assumptions that may not match the actual file headers

Jake has lost time repeatedly working backwards from errors that should have been caught before delivery. This is the #1 rule.

### 2. Read the Actual Files First
Before modifying any script, read the current version of that script. Do not assume you know what it contains based on memory or prior context. Column names, function signatures, and logic change. Always verify against the live file.

### 3. Patch Surgically
When fixing a bug, change only what needs to change. Do not refactor unrelated code, rename variables, or "clean up" things that weren't broken. Every unnecessary change is a new opportunity to introduce a bug.

### 4. Preserve the Data Contract
The following files are the source of truth for performance tracking. Never change their column structure without explicitly updating all code that reads them:
- `output/picks_history.csv`
- `output/betslips_history.csv`
- `data/nba_data.csv`
- `data/historical_lines.csv`

### 5. Log Everything
Every pipeline step must log its outcome. If a step produces no output (e.g., "no betslips graded"), the log must explain **why** — not just report the null result. Silent failures are unacceptable.

### 6. Never Break the Morning Run
`run_morning.bat` runs automatically at ~9am daily. Any change that could cause it to crash or hang must be tested in isolation first. If a change is risky, add a try/except with a graceful fallback — never let one step's failure kill the entire pipeline.

### 7. Money Is on the Line
Jake bets real money based on this system's output. Do not introduce changes that alter pick rankings, confidence scores, or betslip construction without clearly explaining what changed and why. Always flag if a change could affect today's picks.

### 8. Think Strategically, Not Just Tactically
When Jake asks for a fix, also ask: is there a better underlying approach? Is the fix treating a symptom or the root cause? Propose the better solution, explain the tradeoff, and let Jake decide. Do not just execute blindly.

---

## Glossary

| Term | Definition |
|------|-----------|
| **PPP** | Points Per Possession — the core efficiency metric |
| **USG%** | Usage Rate — % of team possessions a player finishes while on the floor |
| **PBP** | Play-by-Play data — possession-level game data from NBA API |
| **PA** | Points + Assists combined stat (PrizePicks prop type) |
| **PR** | Points + Rebounds combined stat |
| **PRA** | Points + Rebounds + Assists combined stat |
| **Power Play** | PrizePicks parlay format — all picks must win, no insurance |
| **Garbage Time** | Late-game minutes when outcome is decided; excluded from training data |
| **Luck Regression** | Mathematical process of separating skill from variance in stats |
| **Edge** | The difference between the model's projection and the PrizePicks line |
| **Confidence Score** | Model's self-reported certainty in a pick (0–100%) |
| **Betslip** | A single PrizePicks entry containing 2–4 picks |

---

*Last updated: April 2026. Maintained by Jake. AI assistants: read this file before every session.*
