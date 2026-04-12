# ARCHITECTURE.md

## System Overview

Fresh Start NBA is a fully automated NBA player prop prediction pipeline that runs daily via Windows Task Scheduler. It ingests box scores, scrapes prop lines, runs a 5-layer analytical engine, generates ranked picks and betslips, and grades results the next morning.

- **Language:** Python 3.11+
- **ML:** scikit-learn (GradientBoosting, Ridge, Isotonic) + XGBoost (per-prop edge + meta classifiers)
- **Automation:** `run_morning.bat` → `run_daily.py` → `nba_props.py predict`
- **Output:** `picks_latest.csv`, `betslips_latest.csv`, `dashboard_latest.html`
- **Production data:** `data/nba_data.csv` (77k+ rows), `data/historical_lines.csv` (30k+ rows)

---

## Prediction Targets

| Prop | Status | Notes |
|---|---|---|
| TRB | Active (UNDER only) | Live quality policy: UNDER enabled |
| AST | Active (UNDER only) | Live quality policy: UNDER enabled |
| STL | Active (UNDER only) | Restored by post-retrain quality policy; eligible for under-only betslips when `confidence >= 70` |
| BLK | Active (UNDER only) | Restored by post-retrain quality policy; eligible for under-only betslips when `confidence >= 70` |
| PTS | Disabled | O/U accuracy ~50%, quality policy rejects |
| PRA / PR / PA | Volume-fill UNDER only | Blended hit rate near 50%, primary pass disabled |

Quality policy is loaded dynamically at prediction time from `models/training_results.json` and `models/training_edge_analysis.json`. Do not hard-code thresholds — read from these files.

---

## Data Sources

### 1. NBA API (`nba_api` library)
- **Raw format:** JSON → DataFrame → CSV
- **Frequency:** Daily (yesterday's games fetched each morning)
- **Location:** `data/nba_data.csv`
- **Coverage:** 3 seasons, 77k+ player-game rows, 800+ unique players
- **Key fields:** `game_date`, `player_id`, `player_name`, `team_abbr`, `pts`, `trb`, `ast`, `stl`, `blk`, `min`, `fga`, `fta`, `orb`, `tov`
- **Known issues:** Some dates may fail to fetch; Step 1.5 has a live fallback retry. Step 1.6 does not.

### 2. PrizePicks API (`prizepicks_scraper.py`)
- **Raw format:** JSON (public API)
- **Frequency:** Daily (today's lines scraped at pipeline start)
- **Location:** `data/historical_lines.csv` (append-only, `source` field tracks origin)
- **Coverage:** 30k+ rows since 2024
- **Key fields:** `game_date`, `player`, `prop`, `line`, `num_books`, `source`
- **Known issues:** Cloudflare 403/429 blocks direct API most days. Fallback chain: direct → cookie → Playwright. When all fail, Odds API lines are used and `line_source` is set to `'odds_api'`.

### 3. Odds API (fallback)
- **Raw format:** JSON
- **Frequency:** On-demand when PrizePicks fails
- **Key fields:** same as PrizePicks, but lines can differ
- **Known issues:** Credit-limited (50 credits threshold gate). Lines are not identical to PrizePicks.

### 4. Play-by-Play data (`pbp_fetcher.py`)
- **Location:** `data/hist_cache/pbp_dates/pbp_YYYY-MM-DD.csv`
- **Frequency:** Cached by date; only fetched if not already cached
- **Key fields:** `PCTIMESTRING`, `SCORE`, `PLAYER1_ID`, `EVENTMSGTYPE`, possession counts

---

## Pipeline Stages (run_daily.py)

| Step | Function | Output | Fatal? |
|---|---|---|---|
| 1 | `step1_update_nba_data()` | `data/nba_data.csv` updated | Yes |
| 1.5 | `step1_5_grade_picks()` | `picks_history.csv` graded rows | No (live API fallback) |
| 1.6 | `step1_6_grade_betslips()` | `betslips_history.csv` graded rows | No (but no live fallback — known bug) |
| 2 | `step2_prizepicks_lines()` | `data/historical_lines.csv` updated | No (falls back to Odds API) |
| 3 | `step3_fetch_pbp()` | `data/hist_cache/pbp_dates/*.csv` | No |
| 4 | `step4_ppp_engine()` | PPP ratings in memory | No |
| 5 | `step5_usage_injury()` | Usage/minutes adjustments | No |
| 6 | `step6_detect_absences()` | `data/detected_absences.csv`, `data/live_injury_status.csv`, `data/live_injury_status.json` | No |
| 7 | `step7_luck_regression()` | Luck-adjusted projections | No |
| 8 | `step8_merge_projections()` | `data/player_projections_today.csv` | No |
| 8 (display) | Top-10 scorer list | Log output only | No |
| 9 | `nba_props.py predict` | `picks_latest.csv`, `betslips_latest.csv`, dashboard | No |

### Step 6 → Step 8 Data Flow
Step 6 now writes two availability artifacts:

1. `data/detected_absences.csv`
   Columns:
   `player_id, player_name, team_abbr, recent_gp, latest_played_team, latest_game_date`

2. `data/live_injury_status.csv` / `data/live_injury_status.json`
   Flattened / nested live status feed from:
   - official NBA injury report
   - ESPN injury API fallback

Step 8 reads both artifacts:
- merges `recent_gp` as `absence_recent_gp`
- merges live injury columns such as `live_injury_bucket`, `live_injury_status`, `live_injury_source`, and `live_team_status_pending`

Step 8 then:
- suppresses official `OUT` / `DOUBTFUL` players from the top-10 display
- tags softer live statuses like `QUESTIONABLE`, `DAY TO DAY`, and `PROBABLE`
- suppresses luck-driven fields for players with either:
  - `absence_recent_gp <= 2`
  - live injury bucket in `out`, `doubtful`, `questionable`, `day_to_day`

---

## Time Semantics

**Sacred rule:** Any feature used at prediction time must reflect information available before game tip-off.

| Information type | Availability cutoff |
|---|---|
| Box scores | Previous night's games only |
| PrizePicks lines | Scraped morning of game day |
| Injury status | Official NBA injury report + ESPN fallback captured during the morning run; recent-games proxy remains fallback context |
| Rolling stats | Must be shifted (exclude current game) |
| PBP data | Previous games only (cached) |

No same-day box score data may appear in any feature. Rolling windows must exclude the game being predicted.

---

## Leakage Guardrails

| Risk | Prevention |
|---|---|
| Rolling stats including current game | All rolling features shifted before use (see DEC-002) |
| Opponent same-game stats in features | Features use L20 rolling averages, not same-game opponent stats |
| Train/val/test contamination | Chronological split only (see DEC-001) |
| Injury status from after game start | Live injury feed captured pregame during Step 6; recent-games proxy remains fallback only |
| PrizePicks line from after game start | Lines scraped before morning pipeline, not intra-game |
| Grading using predicted-game data | Grading only runs for prior-day games via Step 1.5/1.6 |

The `leakage-auditor` subagent (`.claude/agents/leakage-auditor.md`) can be invoked to audit any file in the pipeline.

---

## Data Contracts (Column Schemas)

### `output/picks_history.csv` (source of truth for performance tracking)
| Column | Type | Description |
|---|---|---|
| `game_date` | string YYYY-MM-DD | Date picks were generated |
| `player` | string | Player name |
| `prop` | string | Prop type (TRB, AST, BLK, PTS, etc.) |
| `line` | float | The line the pick was generated against |
| `prediction` | float | Model's projected value |
| `direction` | string | OVER or UNDER |
| `confidence` | float | Model confidence 0–100 |
| `actual` | float | Actual stat from box score |
| `result` | string | WIN / LOSS / PUSH / UNGRADED |
| `pick_source` | string | `'market_model'`, `'structural_micro'`, or `'volume_fill'` |
| `line_source` | string | `'prizepicks'` or `'odds_api'` |

**Never change this schema without updating all code that reads it.**

### `output/betslips_history.csv`
| Column | Type | Description |
|---|---|---|
| `slip_date` | string YYYY-MM-DD | Date betslip was built |
| `slip_id` | string | Unique identifier for this slip |
| `slip_size` | int | Number of legs (2, 3, or 4) |
| `pick_1_player` ... `pick_4_player` | string | Player names per leg |
| `pick_1_prop` ... `pick_4_prop` | string | Prop types per leg |
| `pick_1_direction` ... `pick_4_direction` | string | OVER/UNDER per leg |
| `pick_1_line` ... `pick_4_line` | float | Lines per leg |
| `pick_1_result` ... `pick_4_result` | string | WIN/LOSS/PUSH/UNGRADED per leg |
| `slip_result` | string | WIN / LOSS / UNGRADED |

### `data/nba_data.csv`
Master box score table. Do not modify schema. Key columns: `game_date`, `player_id`, `player_name`, `team_abbr`, all stat columns.

### `data/historical_lines.csv`
Append-only prop lines table. Key columns: `game_date`, `player`, `prop`, `line`, `num_books`, `source`.

---

## pick_source and line_source Semantics

### pick_source
Describes how a pick was selected by `filter_best_picks()`:
- `'market_model'` — passed all primary filter gates (edge, meta, regime, market edge checks)
- `'structural_micro'` — passed structural micro-filter (e.g., mismatch-based picks)
- `'volume_fill'` — filled by second pass with relaxed thresholds (65% of min_edge, dir_prob ≥ 0.53, confidence capped at 63%)

**Rule:** `volume_fill` picks must not be used as betslip legs. They are informational only.

### line_source
Describes where the pick's line came from:
- `'prizepicks'` — line sourced from PrizePicks API (correct market for grading)
- `'odds_api'` — line sourced from Odds API (fallback; may differ from PrizePicks line)

When `line_source != 'prizepicks'`, a `LINE SOURCE WARNING` is printed to the log. The operator must verify each pick's line on PrizePicks before placing a bet.

---

## Model Stack

### Layer 1 — PBP Foundation (`pbp_fetcher.py`)
Raw possession-level data. Ground truth for efficiency calculations.

### Layer 2 — PPP Engine (`ppp_engine.py`)
Ridge Regression on possession data to produce Pace-adjusted PPP opponent-adjusted ratings.
Formula: `Possessions = FGA + 0.44 × FTA − ORB + TOV`

### Layer 3 — Usage & Injury (`usage_injury_model.py`, `minutes_model.py`)
Redistributes usage when teammates are absent. Applies efficiency penalties for forced shots.
Step 6 now persists a live injury feed alongside the absence proxy.
**Known gap:** pending official team reports plus missing ESPN player entries can still leave some players unresolved.

### Layer 4 — Luck Regression (`luck_regression_model.py`)
Regresses opponent FT% and 3PT% to league average. Uses expected eFG% from shot location, not hot/cold streaks.

### Layer 5 — Blowout Discount (`blowout_discount_model.py`)
Filters garbage time from training data. Thresholds: ≥25pt diff in Q3, ≥15pt in Q4, ≥10pt in final 3 min.

### Pick Filter (`nba_props.py filter_best_picks()`)
Two-pass filter:
1. **Primary pass:** Full gate stack (min_edge, meta_prob, regime, market edge, market sign checks). Tags picks as `market_model` or `structural_micro`.
2. **Volume-fill pass:** Relaxed gates (65% min_edge, dir_prob ≥ 0.53). Tags picks as `volume_fill`, caps confidence at 63%.

### Betslip Builder (`nba_props.py log_betslips()`)
- excludes `volume_fill`
- excludes any pick with `confidence < 70`
- under-only pool now allows `AST`, `TRB`, `STL`, and `BLK`

---

## Evaluation Design

- **Split strategy:** Chronological (most recent season held out)
- **Primary metrics:** MAE (regression), O/U accuracy, Meta AUC
- **Business metrics:** pick win rate, betslip win rate, edge size
- **Walk-forward:** Preferred for parameter selection; not yet fully automated
- **Holdout policy:** Never use test-set results to tune thresholds

---

## Runtime / Daily Prediction Flow

```
~9:00am: run_morning.bat
  → run_daily.py
      Step 1: fetch yesterday's box scores → nba_data.csv
      Step 1.5: grade yesterday's picks (live API fallback if date missing)
      Step 1.6: grade yesterday's betslips (NO live fallback — known bug)
      Step 2: scrape today's PrizePicks lines (direct → cookie → Playwright → Odds API)
      Steps 3-8: run 5-layer engine, merge projections, detect absences
      Step 8 display: show top-10 active projected scorers
  → nba_props.py predict
      Load today's lines (from historical_lines.csv or fresh fetch)
      Generate picks via two-pass filter
      Build betslips from primary picks (not volume_fill)
      Save picks_latest.csv, betslips_latest.csv, dashboard_latest.html
      Append to picks_history.csv and betslips_history.csv
```

---

## Open Design Questions

1. **Pending-report injury gap** — some players can still slip through when the official report is pending and ESPN also lacks the player entry.
2. **Betslip grading fallback** — Step 1.6 needs same live API retry as Step 1.5. Not yet implemented.
3. **Shared injury-feed module** — Step 6 currently imports the live injury helpers from `nba_props.py`; this should be centralized into a dedicated module.
4. **Blowout baseline duplicate root cause** — Step 8 dedup is a merge-time fix; source CSV may have duplicate rows that should be cleaned upstream.
5. **`pra`/`pr`/`pa` model rehabilitation** — These are disabled because blended hit rate is near 50%. Could be improved with better feature engineering or market-model calibration.
