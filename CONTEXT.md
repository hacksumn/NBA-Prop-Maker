# NBA Props Betting System — Claude Context File

Drop this file into a new Claude session and say: **"Read CONTEXT.md and continue where we left off."**

---

## Project Location

```
C:\Users\jakep\Downloads\Fresh_Start_NBA\
├── files\
│   ├── nba_props.py          ← THE ONLY FILE THAT MATTERS (3,540 lines)
│   ├── models\
│   │   ├── models.pkl              — 6 XGBoost regression models (pts/trb/ast/stl/blk/tov), 232 features
│   │   ├── ou_classifiers.pkl      — 6 binary OVER/UNDER classifiers (same stats)
│   │   ├── minutes_deviation_model.pkl
│   │   ├── bias.json
│   │   ├── classifiers\            — per-stat classifier pkls
│   │   └── regression\             — per-stat regression pkls
│   ├── output\
│   │   ├── picks_*.csv / dashboard_*.html   — daily outputs
│   │   ├── picks_history.csv
│   │   └── betslips_history.csv
│   └── data\                       ← NOTE: data_dir is one level UP (see below)
└── data\                           ← CONFIG['data_dir'] points HERE
    ├── nba_data.csv                — 76,097 game rows, 2023-10-24 to 2026-03-24
    └── historical_lines.csv        — 24,427 real Vegas prop lines, 2026-02-16 to 2026-03-28
```

**Critical path note:** `CONFIG['data_dir']` is `_HERE.parent / 'data'` = `Fresh_Start_NBA/data/`, NOT `files/data/`. There is a stale `files/data/historical_lines.csv` (only 1,326 rows — ignore it). Always use `Fresh_Start_NBA/data/`.

---

## How to Run

```bash
cd C:\Users\jakep\Downloads\Fresh_Start_NBA\files

python nba_props.py update          # fetch latest NBA game data
python nba_props.py train           # train regression + O/U classifiers
python nba_props.py predict         # generate today's picks + save lines snapshot
python nba_props.py backtest 30     # backtest last 30 days (uses real lines where available)
python nba_props.py backfill 30     # pull 30 days of historical lines from Odds API
python nba_props.py all             # full pipeline
```

---

## Odds API

- **Key:** `8269e5e566cd8d8f8c294b85244ffab7`
- **Credits:** ~20,000 paid credits loaded
- **Historical access:** YES (confirmed working — used to backfill 30 days)
- **Key in CONFIG:** already hardcoded at line 44 in nba_props.py

---

## Architecture: nba_props.py Function Map

```
Line 66   fetch_nba_data()               — NBA API / local CSV loader
Line 230  fetch_todays_games()           — upcoming game matchups
Line 265  _normalize_name()              — strip accents/punctuation for name matching
Line 273  apply_first_principles_adjustments() — rule-based prediction tweaks
Line 345  fetch_prizepicks_lines()       — PrizePicks scraper
Line 419  fetch_vegas_lines()            — Odds API live lines
Line 474  save_lines_snapshot()          — append today's lines to historical_lines.csv
Line 499  backfill_historical_lines()    — pull N days from Odds API historical endpoint
Line 577  fetch_game_totals()            — game O/U totals
Line 639  fetch_player_positions()       — player position data
Line 676  fetch_injury_data()            — ESPN injury API (live)
Line 703  apply_injury_adjustments()     — boost predictions for teammates-out
Line 751  create_features()             — 500+ lines of feature engineering (238 features)
Line 1530 get_feature_columns()         — feature exclusion list
Line 1582 _train_lgbm()                 — optional LightGBM training
Line 1601 train_models()               — XGBoost regression training w/ time-series CV
Line 1804 train_ou_classifiers()        — binary OVER/UNDER XGBoost classifiers
Line 1898 predict_ou_proba()            — add P(OVER) columns to pred_df
Line 1938 _predict()                    — ensemble predict helper
Line 1948 generate_predictions()        — full prediction pipeline per player
Line 2186 merge_vegas_lines()           — join Vegas lines onto pred_df
Line 2221 filter_best_picks()           — grade A/B+/B/C, confidence %, edge filter
Line 2333 backtest()                    — train/test split evaluation w/ real lines
Line 2524 _calc_payout()               — parlay payout math
Line 2529 log_betslips()               — append to betslips_history.csv
Line 2586 sanity_check_picks()          — rule-based filters (injury, usage, etc.)
Line 2764 save_picks()                  — CSV + HTML dashboard output
Line 2810 generate_html_report()        — full dashboard HTML generation
Line 3363 main()                        — CLI entry point
```

---

## CONFIG (line 37)

```python
CONFIG = {
    'data_dir':    _HERE.parent / 'data',     # → Fresh_Start_NBA/data/
    'models_dir':  _HERE / 'models',          # → files/models/
    'output_dir':  _HERE / 'output',          # → files/output/
    'odds_api_key': '8269e5e566cd8d8f8c294b85244ffab7',
    'min_games':    10,
    'ewma_span':    5,
    'confidence_threshold': 0.6,
    'edge_threshold':   1.5,      # min absolute edge for picks
    'edge_z_threshold': 0.5,      # min edge / player_std
    'targets': ['pts', 'trb', 'ast', 'pra', 'pr', 'pa'],
}
```

---

## Models: Current State

### Regression (models.pkl)
- **6 models:** pts, trb, ast, stl, blk, tov
- **232 features** (XGBoost residual mode)
- **Backtest results (30 days, real Vegas lines ~50% coverage):**

| Stat | Line Source | MAE | High-Edge Accuracy | Bets |
|------|-------------|-----|-------------------|------|
| PTS  | 52% real / 48% L10 | 4.06 | **75.9%** | 1,421 |
| TRB  | 51% real / 49% L10 | 1.81 | **71.2%** | 337 |
| AST  | 38% real / 62% L10 | 1.30 | **74.4%** | 82 |
| STL  | 12% real           | 0.69 | — | — |
| BLK  | 6% real            | 0.51 | — | — |
| TOV  | 0% real (L10 only) | 0.86 | 87.5% | 8 |

### O/U Binary Classifiers (ou_classifiers.pkl)
- **6 classifiers:** pts, trb, ast, stl, blk, tov
- Trained on historical_lines.csv matched to actual outcomes
- Features: same 232 as regression **+ Vegas line as input feature**
- Output: `P(OVER)` probability — used to adjust confidence in `filter_best_picks()`

### Pick Grading
- **Grade A:** model + L10 consensus AND |edge| >= 2.0
- **Grade B+:** consensus AND edge < 2.0
- **Grade B:** L10 neutral
- **Grade C:** model contradicts L10 trend
- O/U prob ≥ 0.62 → +5% confidence; ≤ 0.40 → -8% confidence

---

## Data Files

### nba_data.csv (`Fresh_Start_NBA/data/`)
- 76,097 rows of player game logs
- Columns: player, game_date, team, opp, matchup, result, season, mp, fg, fga, fg_pct, 3p, 3pa, 3p_pct, ft, fta, ft_pct, orb, drb, trb, ast, stl, blk, tov, pf, pts, plus_minus, is_home
- Date range: 2023-10-24 → 2026-03-24

### historical_lines.csv (`Fresh_Start_NBA/data/`)
- 24,427 rows of real Vegas prop lines
- Columns: game_date, player, player_norm, prop, line, num_books, stat
- Date range: 2026-02-16 → 2026-03-28
- Props: player_points, player_rebounds, player_assists, player_points_rebounds_assists, player_points_rebounds, player_points_assists, player_steals, player_blocks, player_turnovers
- **Grows every day** when `predict` runs (auto-snapshots today's lines)

---

## What Was Done In Recent Sessions

### Session 1 (earlier, in sandbox — some work may be lost):
- Built the entire nba_props.py pipeline from scratch
- Feature engineering: 238 features across 14 sections
- Regression models w/ time-series CV
- Prediction + grading pipeline
- HTML dashboard output

### Session 2 (this session — all saved to disk):
1. **Historical Prop Lines** — implemented `save_lines_snapshot()`, `backfill_historical_lines()`
2. **Fixed Odds API parsing** — historical endpoint wraps response in `['data']` key (not a bare list)
3. **Backfilled 30 days** of real Vegas lines (14,782 → 24,427 rows after daily snapshots)
4. **Updated `backtest()`** — uses real lines where available, L10 fallback otherwise, prints "X% real lines" per stat
5. **Ran predict** — 89 picks generated, 207 lines saved for 2026-03-15 (note: today is 2026-03-28, re-run `predict` to get fresh picks)
6. **Injury data** (`fetch_injury_data`, `apply_injury_adjustments`) — already implemented at lines 676–750
7. **O/U binary classifiers** (`train_ou_classifiers`, `predict_ou_proba`) — already implemented at lines 1804–1937

### Session 4 (2026-03-28 — saved to disk):
1. **Fixed `train_advanced_models.py`** — hardcoded Linux paths (`/home/claude/`) replaced with Windows paths
2. **Fixed `feature_pipeline.py`** — same path issue fixed, now outputs `data/model_data_advanced.csv`
3. **Ran full advanced training pipeline** — `feature_pipeline.py` → `train_advanced_models.py` → 9 models saved as `xgb_{stat}_advanced.pkl`
4. **Wired advanced models into `nba_props.py`** — added `_build_advanced_features()` and `_load_advanced_models()`, modified `generate_predictions()` to use advanced models (117 features) when available, fallback to standard models
5. **Removed STL/BLK from picks** — models too weak (61%/66%), excluded from `filter_best_picks` props list
6. **Raised UNDER thresholds** — AST: 0.75→1.5, TRB: 1.0→2.0, PTS: 1.5→3.0, PRA/PR/PA: 1.5→2.5-3.0
7. **Disabled most OVER picks** — PTS/TRB/AST/PRA/PR OVERs set to 99.0 threshold (never bet); only PA OVER allowed on 8+ gap. Result: 66 picks → 33 cleaner picks, all UNDERs.

### Session 3 (2026-03-28 — saved to disk):
1. **Manually graded 3/25 picks** — nba_api was lagging (only through 3/24), fetched box scores directly via `leaguegamelog` with explicit date range
2. **Updated picks_history.csv** — 172 picks from 3/25 graded: 73W/87L (45.6%)
3. **Overall record dropped:** 56.4% (840 picks) → 54.7% (1,000 picks)
4. **Diagnosed OVER bias problem** — see critical finding below
5. **Pipeline did not run on 3/26** — no log file exists for that date, unknown cause. Check Task Scheduler.
6. **picks_history.csv write fails if Excel has it open** — PermissionError on `to_csv()`. Close Excel first.

---

## CRITICAL FINDING: OVER Bias (discovered 2026-03-28)

**The model over-projects player stats. OVER picks are losing money.**

Analysis across all 1,000 graded picks:

| Prop | OVER % | UNDER % |
|---|---|---|
| PTS | **44.5%** ❌ | 55.9% ✅ |
| TRB | 44.4% ❌ | **65.8%** ✅ |
| PRA | 48.7% ❌ | 54.5% ✅ |
| PR  | 50.0%     | 58.6% ✅ |
| PA  | 53.3%     | 61.7% ✅ |
| AST UNDER | — | **74.1%** ✅ |

**Overall: OVERs = 48.9% (losing), UNDERs = 59.2% (winning)**

### Confidence/Edge scores are broken:
- `<75` confidence: **55.7%** win rate
- `85-90` confidence: **51.5%** win rate
- `6-8` edge: **47.6%** win rate (worse than low edge picks)
- The ranking logic is NOT predictive. Do not use confidence or edge to filter top picks.

### Bias corrections in model are not aggressive enough:
Current corrections (PTS: -0.4743, TRB: -0.2250, etc.) are not enough to fix the systematic over-projection. The bias corrections need to be increased or the model needs retraining with stronger regularization.

### Win rate trend (declining):
| Date | % |
|---|---|
| 03/17 | 64.6% |
| 03/18 | 57.6% |
| 03/19 | 64.8% |
| 03/20 | 50.8% |
| 03/21 | 48.5% |
| 03/25 | 45.6% |

### What to fix:
1. **Increase bias corrections** — especially PTS and PRA which are below 50% on OVERs
2. **Filter to UNDER-only or UNDER-weighted** until OVER bias is fixed
3. **Recalibrate confidence scoring** — it's currently inversely correlated with success
4. **Re-examine edge calculation** — high edge ≠ high win rate

---

## Session 5 (2026-03-30/31 — all saved to disk)

### Bug Fixes
1. **Layer 2 & 4 merging 0 players (CRITICAL)** — `player_adj_ppp.csv` uses abbreviated names ("N. Jokić"), proj uses full names. Fixed by merging on `player_id` instead. `player_luck_scores.csv` uses lowercase `player_name` but code checked uppercase `PLAYER_NAME`. Fixed column name. Now: 236 adj_ppp + 462 luck scores merging correctly every run.
2. **Step 1.6 betslip grading `_log` NameError** — logger is named `logger`, not `_log`. Fixed with global sed replace.
3. **Betslip date normalization** — betslips stored as `3/28/2026`, box scores use `2026-03-28`. Added `pd.to_datetime` normalization on load.
4. **Betslip nan-string player check** — `dtype=str` reads empty Pick columns as string `"nan"`. `if not str(player).strip()` passed `"nan"` as truthy, counting 4 picks for a 2-pick slip → `graded < n_picks` → slip skipped. Fixed by also checking `str(player).lower() == "nan"`.
5. **Betslip blank game_date rows (6 rows)** — historical betslips logged without dates. Patched to `2026-03-29`. Unverifiable Quentin Grimes set marked LOSS. Mathurin 3/30 slip (IND didn't play) marked LOSS.
6. **DNP handling in betslip grader** — if a player doesn't appear in box scores, grader was skipping the whole slip. Fixed: None results = DNP = PUSH leg per PrizePicks policy. Slip grades at reduced tier (4-pick with 1 DNP grades as 3-pick, etc.).
7. **Step 1.6 live-fetch fallback** — added same nba_api live fetch as Step 1.5 so betslip grading never blocks on missing dates.
8. **Atomic CSV writes** — `picks_history.csv` and `betslips_history.csv` now use `_atomic_csv()` (write to .tmp then `os.replace()`) to survive Excel file locks.
9. **`proj` missing `player_id`** — Step 8 now carries `player_id` from nba_data.csv through to projection DataFrame so Layer 2 can merge on it.

### Strategy Changes
10. **UNDER-only** — PA OVER threshold raised to 99.0. All OVER bets now disabled. Data: OVERs 48.9% (losing), UNDERs 59.2% (winning).
11. **Top-10 cap** — `filter_best_picks` now returns max 10 picks per day (was 30-60).
12. **Confidence calibration** — `calibrate_confidence.py` trains isotonic regression on graded picks. Raw confidence formula (fake 80-90% outputs) replaced with calibrated probability. Picks ranked by: empirical stat win-rate × 0.7 + calibrated_conf × 0.3.
13. **Stat priority ranking** — AST UNDER (74.3%) > TRB UNDER (65.2%) > PA UNDER (59.8%) > PR UNDER (57.1%) > PRA UNDER (56.1%) > PTS UNDER (54.9%).
14. **Weekly calibration rebuild** — `calibrate_confidence.py` added to `run_morning.bat` Sunday rebuild alongside `feature_pipeline.py` and `train_advanced_models.py`.

### New Files
- `calibrate_confidence.py` — trains isotonic regression confidence calibrator, saves `models/confidence_calibrator.pkl`
- `outlier_heatmap.py` — generates HTML heatmap of training data outliers (flag low/high minutes, z-score outliers)

### Daily Results (3/26–3/31)
| Date | W | L | P | % | Notes |
|------|---|---|---|---|-------|
| 3/27 | 70 | 86 | 0 | 45% | 156 picks, OVERs everywhere, pre-fix |
| 3/28 | 20 | 7 | 0 | 74% | UNDER-only kicks in, tight thresholds |
| 3/29 | 38 | 28 | 4 | 58% | Solid UNDER day |
| 3/30 | 1 | 0 | 0 | 100% | Only 1 pick (Caris LeVert AST UNDER) |

### Betslip Results (3/26–3/31)
| Date | Type | Result | Payout |
|------|------|--------|--------|
| 3/27 | 2-pick | LOSS | $0 |
| 3/27 | 3-pick | LOSS | $0 |
| 3/27 | 4-pick | LOSS | $0 |
| 3/28 | 2-pick | WIN | $60 |
| 3/28 | 3-pick | LOSS | $0 |
| 3/28 | 4-pick | LOSS | $0 |
| 3/29 | 2-pick | WIN | $40 (DNP discount) |
| 3/29 | 3-pick | WIN | $60 (DNP discount) |
| 3/29 | 4-pick | WIN | $100 (DNP discount) |

### Current Architecture (run_daily.py pipeline)
```
STEP 1   — nba_scraper → nba_data.csv
STEP 1.5 — Auto-grade picks (live-fetch fallback if date missing from CSV)
STEP 1.6 — Auto-grade betslips (live-fetch fallback + DNP handling)
STEP 2   — prizepicks_scraper → historical_lines.csv
STEP 3   — blowout_discount_model → game_blowout_index.csv
STEP 4   — pbp_fetcher → pbp_possessions_master.csv
STEP 5   — ppp_engine → team_ratings.csv, player_adj_ppp.csv (236 players)
STEP 6   — usage_injury_model → player_profiles.csv (486 players)
STEP 7   — luck_regression_model → player_luck_scores.csv (462 players)
STEP 8   — Merge all layers → player_projections_today.csv
           Layer 2: 236 players have adj_ppp (merge on player_id)
           Layer 3: 486 players have usage profiles
           Layer 4: 462 players have luck scores
           Layer 5: 462 players have blowout-clean baselines
```

### Key Commands
```bash
cd C:\Users\jakep\Downloads\Fresh_Start_NBA
python run_daily.py              # full morning pipeline
python nba_props.py predict      # generate picks (run AFTER run_daily.py)
python calibrate_confidence.py   # rebuild confidence calibrator (auto on Sundays)
python outlier_heatmap.py        # generate outlier report → output/outlier_heatmap.html
```

---

## What's Next (Not Yet Done)

The plan that was drafted but NOT yet implemented:

### Remaining from "high impact" list:
These two were planned but need verification that they're wired up end-to-end:

1. **Injury adjustments called in generate_predictions()?**
   Check line ~2100 in `generate_predictions()` — does it call `fetch_injury_data()` and `apply_injury_adjustments()`? If not, wire it in.

2. **O/U proba called in generate_predictions()?**
   Check that `predict_ou_proba(pred_df, df)` is called after `merge_vegas_lines()` in `generate_predictions()`.

3. **O/U prob used in filter_best_picks()?**
   Check that `filter_best_picks()` reads `{stat}_ou_prob` columns and adjusts confidence accordingly.

4. **`train_ou_classifiers` called from `train` command in main()?**
   Check main() around line 3400 — after `train_models(df)`, is `train_ou_classifiers(df)` called?

### Medium priority:
- **Separate models by player tier** (stars vs rotation vs bench have very different variance)
- **Current-season weighting** (2023 games shouldn't count the same as 2026 games)
- **Calibrated confidence** via isotonic regression on backtest predictions

---

## Quick Verification Commands

```bash
# Check everything is wired up
python nba_props.py predict

# Re-run backtest to see current accuracy
python nba_props.py backtest 30

# Re-train with O/U classifiers
python nba_props.py train
```

---

## Key Decisions / Preferences (remember these)

- **Single file:** All logic stays in `nba_props.py`. No splitting into modules.
- **No breaking changes:** New features are additive — graceful fallback if data/model is missing.
- **Real lines over L10:** Backtest uses real Vegas lines where available; L10 is only fallback.
- **Injury at predict-time only:** Don't add injury features to training — historical game logs already reflect real outcomes when teammates were out.
- **O/U classifier includes line as feature:** The Vegas line itself is the most informative single feature for the binary classifier.
- **Windows terminal:** Avoid Unicode arrows (→) in print() — use `->` instead (cp1252 encoding issue).
