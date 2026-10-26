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

### 3. Odds API (reference market and disabled fallback)
- **Raw format:** JSON
- **Frequency:** Paid only for explicitly capped research acquisition; implicit live-card fallback is disabled
- **Key fields:** sportsbook line, bookmaker, over/under price, event/tip, requested and actual snapshot times
- **Known issues:** Sportsbook lines are not identical to PrizePicks and cannot substitute for the wagered line. The remaining 13,436 credits are assigned only to paired T-60 source-transfer observations.

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
| Opponent same-game stats in features | Point-in-time Ridge ratings fit only games with `game_date < target_date`; corrected cheap context uses an as-of merge that refuses exact-date matches |
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

The 2025-26 source currently has blank `opp` and `is_home` values. Defense
builders must reconstruct them from reciprocal `(GAME_ID, team)` rows and the
`matchup` string, then reject disagreements. They may not treat these blanks as
neutral matchup context.

All 2025-26 player rows also have null `orb`/`drb`. Team-defense possession
math must not aggregate this source for that season or coerce those nulls to
zero. Use `data/nba_team_game_logs.csv` instead.

### `data/nba_team_game_logs.csv`

Exact NBA team-game box scores for regular season, `PlayIn`, and playoffs.
Primary key is `(GAME_ID, TEAM_ABBREVIATION)`. `PTS`, `FGA`, `FTA`, `OREB`, and
`TOV` are the authoritative possession/rating inputs. Partial GAME_ID coverage,
duplicate keys, required stat nulls, or a game without exactly two teams are
fatal for defense feature generation.

### `data/pointintime_team_defense_ratings.csv`

One strict-pregame row per `(GAME_ID, team)`. Core audit columns are
`game_date`, `team`, `opponent`, `is_home`, `rating_asof_date`,
`n_prior_games`, `opp_n_prior_games`, and both rating-source fields. Model
candidates are `opp_def_rtg_pt`, `opp_pace_pt`, and `opp_def_tier_pt`.
`rating_asof_date` must always be strictly earlier than `game_date`; duplicate
keys or required nulls are fatal.

### `data/historical_lines.csv`
Lossless append-only prop snapshot table. Physical key:
`(game_date, player_norm, prop, line, source, snapshot_ts)`. `line` is part of
the physical key because a bookmaker can return multiple alternate lines for
one player/market/snapshot. Core columns are
`game_date`, `player`, `player_norm`, `prop`, `line`, `num_books`, `source`,
`snapshot_ts`, `projection_type`, and `is_promo`. Odds API snapshots may also
populate `bookmaker`, `event_id`, `commence_time`, `requested_snapshot_ts`,
`over_price`, and `under_price`.

The physical store may contain multiple books and multiple intraday snapshots
for the same player/prop/day. Model training, backtests, and live cached-line
selection must use the deterministic preferred-line view implemented by
`_select_preferred_line_rows()`; joining the raw store directly will fan out
player-game rows. The view excludes known post-tip observations, ranks the
configured sportsbook above PrizePicks above generic Odds API sources, and then
uses the latest eligible snapshot within that source.

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

#### Point-in-time team-defense adapter (DEC-051)

`scripts/build_pointintime_team_defense.py` reuses
`ppp_engine.fit_team_ratings()` once per historical target date. Each fit uses
current-season games strictly before that date. A team needs five prior games
before its current-season rating is trusted; otherwise the adapter uses the
previous season's final rating, then league average only when no prior season
exists. Exact NBA team game logs are the source because player-level 2025-26
ORB is entirely null and the local PBP master does not provide a reconciled
replacement. The source fetch must include regular season, `PlayIn`, and
playoffs.

`scripts/build_prop_defense_rating_training_data.py` performs the audited
many-to-one `(GAME_ID, team)` prop join and
`train_advanced_models.py --require-defense-rating-features` enforces the
contract in an isolated model directory. At live inference, `nba_props.py`
fetches the requested slate schedule, scores only its scheduled matchups from
completed prior games, and fails closed if a model requires these fields but the
schedule/rating join is unavailable.

DEC-052 activates this path in production. Current selected models consume
`opp_def_rtg_pt` for STL and `opp_pace_pt` for BLK. The remaining props did not
retain a rigorous field during selection; no missing value is filled with a
legacy neutral default when an active model requires one.

The old Step 8 `ppp_scale = adj_ppp / 1.15` is a player historical-efficiency
scale, not a scheduled-opponent defense adjustment. It must not be described as
today's matchup rating. The full-season enriched `adj_ppp` /
`opp_def_rtg_avg` training fields remain disabled because they are hindsight
aggregates.

### Layer 3 — Usage & Injury (`usage_injury_model.py`, `minutes_model.py`)
Redistributes usage when teammates are absent. Applies efficiency penalties for forced shots.
Step 6 now persists a live injury feed alongside the absence proxy.
**Known gap:** pending official team reports plus missing ESPN player entries can still leave some players unresolved.

#### Validated availability/minutes hurdle (`scripts/train_availability_minutes_model.py`)

The legacy `minutes_model.py` trains only on appearance rows and cannot learn DNP/non-appearance risk. DEC-048 adds a separate research/deployment package with two outputs:

1. `availability_probability = P(player appears)`
2. `conditional_minutes_prediction = minutes | player appears`

Historical candidates are reconstructed before each team game using only prior same-team appearances. Eligibility requires two prior team appearances, recent rotation minutes of at least eight, and no more than 45 days since the last appearance. The feature state is updated only after the target game is labeled. Player identity, current-game minutes, and current-game availability are excluded from the model feature matrix.

- Candidate table: `data/availability_minutes_candidates.csv`
- Leakage-safe prop-training features: `data/availability_minutes_oof_predictions.csv`
- Full-fit future scorer: `models/availability_minutes_model.pkl`
- Policy: `models/availability_minutes_policy.json`
- Validation/audit: `output/availability_minutes_validation_*`
- Reproducible notebook: `notebooks/availability_minutes_validation.ipynb`

The OOF table is the only permitted source for historical prop-model joins. The full-fit pickle is for future scoring and remains `enabled_for_live=false` until the retrained prop stack passes a separate chronological test.

#### OOF-to-prop training adapter (`scripts/build_prop_availability_training_data.py`)

DEC-049 materializes `data/model_data_availability_advanced.csv` by joining the advanced appearance table to the strict OOF table on `(GAME_ID, PLAYER_ID, team)`. The adapter validates one-to-one keys, reconciles game dates and actual minutes, requires `model_train_end < game_date`, rejects source dates newer than the OOF table, and fails below 90% coverage inside the OOF window. It retains matched appearance rows only because player props void a non-appearance; availability risk remains available to the conditional stat models as a pregame feature rather than a realized zero target.

The registered candidate features are `availability_probability`, `conditional_minutes_prediction`, and `expected_minutes_prediction`. `train_advanced_models.py --require-availability-features` is the mandatory fail-closed flag for candidate retraining. It rejects missing/null features, invalid ranges, product inconsistencies, or chronology failures and writes `availability_feature_contract.json` beside the candidate models. Candidate artifacts must use an isolated model directory; production models remain unchanged until the downstream chronological comparison passes and the future-game feature adapter exists.

### Layer 4 — Luck Regression (`luck_regression_model.py`)
Regresses opponent FT% and 3PT% to league average. Uses expected eFG% from shot location, not hot/cold streaks.

### Layer 5 — Blowout Discount (`blowout_discount_model.py`)
Filters garbage time from training data. Thresholds: ≥25pt diff in Q3, ≥15pt in Q4, ≥10pt in final 3 min.

### Pick Filter (`nba_props.py filter_best_picks()`)
Two-pass filter:
1. **Primary pass:** Full gate stack (min_edge, meta_prob, regime, market edge, market sign checks). Tags picks as `market_model` or `structural_micro`.
2. **Volume-fill pass:** Relaxed gates (65% min_edge, dir_prob ≥ 0.53). Tags picks as `volume_fill`, caps confidence at 63%.

#### Probability bundle and final-card contract (DEC-056)

Probability sidecars, OOF predictions, isotonic calibration, and diagnostics
are one versioned bundle. Training fits calibration on earlier OOF dates while
reserving the latest 20 dates, then records SHA-256 hashes for every sidecar and
the OOF artifact. Live inference fails closed if calibration is missing, lacks
an active target, or its hashes do not match the active sidecars. It may not
silently use a stale calibrator or fall back to raw probability.

The executable dynamic policy currently permits AST/STL/BLK UNDER only. There
is no separate STL/BLK exclusion overriding that policy. Final-card selection
permits at most one pick per prop; PTS/TRB/PRA/PR/PA/TOV remain disabled until
their own policy evidence changes.

### Betslip Builder (`nba_props.py log_betslips()`)
- excludes `volume_fill`
- excludes any pick with `confidence < 70`
- under-only pool derives from the live dynamic policy (currently `AST`, `STL`, and `BLK`)

---

## Evaluation Design

- **Split strategy:** Chronological (most recent season held out)
- **Primary metrics:** MAE (regression), O/U accuracy, Meta AUC
- **Business metrics:** pick win rate, betslip win rate, edge size
- **Walk-forward:** Automated by `scripts/evaluate_clv_walk_forward.py`; expanding date folds select feature groups before an untouched future holdout
- **Holdout policy:** Never use test-set results to tune thresholds
- **Availability/minutes validation:** model family selection uses four expanding date folds across the first two seasons; the latest season is excluded from fitting and selection. Candidate rows are grouped by calendar date so a date cannot be split across train/validation. Acceptance requires availability, conditional minutes, hard-gated minutes, and soft expected-minutes gains in at least three folds and the latest-season holdout. DEC-048 cleared all four folds and the holdout.

### Historical CLV acquisition manifest

- Generator: `scripts/build_clv_snapshot_manifest.py`
- Instruction table: `data/clv_snapshot_manifest_2026_live.csv`
- QA summary: `output/clv_snapshot_manifest_summary.json`
- Grain: one row per `(nba_game_id, snapshot_kind)`, where `snapshot_kind` is `decision`, `t_minus_60`, or `t_minus_5`
- Tip contract: scheduled UTC tip from ESPN's scoreboard feed, reconciled one-to-one to the local `nba_data.csv` date/home/away/game ID
- Decision contract: first immutable timestamped pick archive whose bytes equal the final daily card; local filename time is interpreted in `America/Chicago`
- Safety contract: a missing decision archive or a timestamp at/after tip is `blocked`; it is never replaced with a guessed time
- Paid-call contract: manifest generation sends zero Odds API requests. Execution must first resolve `odds_api_event_id`, honor `execution_status`, checkpoint `manifest_key`, and stay within the explicit credit cap.
- Executor: `scripts/execute_clv_snapshot_manifest.py`; raw JSON is saved under `data/odds_api_clv_raw/`, parsed event checkpoints under `data/odds_api_clv_parsed/`, and the resumable ledger/checkpoint under `data/clv_snapshot_execution_*`.
- Corrected paid physical grain: `(game_date, player_norm, prop, line, source, snapshot_ts)`. Alternate lines are retained; `_select_preferred_line_rows()` selects the most balanced paired-price line for each book/snapshot before source priority.
- Feature materialization: `scripts/build_clv_features.py` writes `data/clv_features_2026_live.csv` at one row per game/player/prop. Pretip fields include consensus line/no-vig probability, book/price dispersion, PP-minus-consensus gap, T-60/T-5 movement, decision/T-5 movement, and market-withdrawal flags. `actual`, `mp`, and hit labels are outcome-only columns and must never enter pre-tip feature matrices.
- QA: `scripts/audit_clv_snapshot_panel.py` verifies raw/master reconciliation, timing, price completeness, duplicates, dates, and box-score joins. `scripts/analyze_clv_features.py` produces the dose-response, movement, card coverage, and regime-correction artifacts.
- Frozen eligible-pool evaluator: `scripts/evaluate_clv_walk_forward.py` reconstructs the archived card's exact standard/non-promo PrizePicks pool, tests each market feature group with expanding folds, compares selections with date/prop-matched random and always-UNDER controls, and writes disabled shadow artifacts unless every validation and holdout gate passes.
- Paired close capture: `scripts/capture_prizepicks_closing_snapshots.py` captures the PrizePicks payload in T-60 and T-5 windows. Raw PrizePicks payloads live under `data/prizepicks_clv_raw/`, per-event parsed files under `data/prizepicks_clv_snapshots/`, and completion state in `data/prizepicks_clv_capture_checkpoint.json`. Rows use sources `prizepicks_t_minus_60` / `prizepicks_t_minus_5` and retain event ID, matchup description, scheduled tip, actual capture time, and requested target time.
- T-60 sportsbook pairing: after a direct PrizePicks T-60 game is captured, the free current-events endpoint must resolve exactly one matching game before the current event-odds endpoint is called. Sportsbook raw JSON lives under `data/odds_api_prizepicks_t60_raw/`, parsed rows under `data/odds_api_prizepicks_t60_parsed/`, and lifetime spend/completion state in `data/prizepicks_t60_pair_checkpoint.json`.
- Credit boundary: `CONFIG['allow_live_odds_api_spend']` remains false for line fallback. The paired collector is a separate, explicit research path with a `13,436` lifetime ceiling, a nine-credit preflight ceiling per T-60 game, actual cost from response headers, no T-5 paid calls, and terminal-4xx retry blocking. The Odds API key is read only from `ODDS_API_KEY`; historical executors still require explicit caps.
- Expansion panel: `data/clv_snapshot_manifest_2026_expansion.csv` covers 264 games from `2026-02-23` through `2026-03-29` at T-60/T-5. Its isolated raw, parsed, checkpoint, and ledger paths use the `*_expansion*` naming contract; `scripts/build_clv_features.py` and `scripts/audit_clv_snapshot_panel.py` accept explicit paths so isolated panels can be materialized and audited without replacing the original panel.
- Combined sportsbook evaluation: `scripts/evaluate_sportsbook_expansion.py` joins the original and expansion feature tables, selects features on 54 development dates with 24 expanding validation dates, and reserves the latest 20 archived-prediction dates for exactly 100 top-five holdout selections. Artifacts use `sportsbook_expansion_*`; models remain source-scoped shadow artifacts.
- Paid-executor concurrency: the executor creates a PID lock beside its checkpoint and rejects an overlapping process. Atomic file replacement retries transient Windows sharing failures. This prevents command-runner timeouts from producing concurrent paid requests.

---

## Runtime / Daily Prediction Flow

```
~9:00am: run_morning.bat
  → run_daily.py
      Step 1: fetch yesterday's box scores → nba_data.csv
      Step 1.5: grade yesterday's picks (live API fallback if date missing)
      Step 1.6: grade yesterday's betslips (NO live fallback — known bug)
      Step 2: scrape today's PrizePicks lines (direct → cookie → Playwright; no implicit paid fallback)
      Step 2.5: launch PrizePicks T-60/T-5 watcher; pair sportsbook only at T-60 under the 13,436-credit lifetime cap
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
