# TASKS.md

# Active Work Queue

## NOW
- [x] **[CRITICAL] PTS OVER permanently disabled; PTS UNDER auto-gate added (2026-04-15)** — Investigation found: dynamic policy was already blocking PTS (model data: under_hit=0.537, market_hit=0.491, clv_corr=0.054). Sigma NaN is historical-only (PTS stopped generating picks before sigma was added). +2.8 pt bias is selection-induced, not systematic. Changes: (1) fallback `allow_over=False`, `min_edge_over=9.9`, (2) hard dynamic `allow_over=False` for PTS regardless of eval data, (3) `pts_under_tracking_allowed` gate added — auto-enables PTS UNDER when `under_hit ≥ 0.54` AND `market_hit ≥ 0.50`. Currently not triggered (market_hit=0.4915). See DECISIONS.md 2026-04-15.
- [ ] **[CRITICAL] Accumulate true-labeled market history** - keep running the live PrizePicks save path so `historical_lines.csv` contains enough real `standard/goblin/demon` rows to support a meaningful post-gate audit. Current verified live snapshot is `655` rows (`602 demon`, `35 standard`, `18 goblin`) across `2026-04-14` / `2026-04-15` / `2026-04-18`.
- [ ] **[CRITICAL] OVER edge audit on standard lines only** - recompute hit rates on `projection_type='standard'` and `is_promo=False` lines only once enough true-labeled rows are graded. The newly verified `2026-04-12` `5/5` card does not qualify as clean evidence because that slate was still legacy-standard cached history.
- [ ] **Monitor first live `team_pending` slate** - on the next day with a pending official report, rerun the normal pipeline and confirm `data/player_projections_today.csv` contains explicit `team_pending` rows instead of blank live status fields.
- [ ] **Power Play EV calibration audit** - compare realized slip outcomes against the current heuristic correlation weights (`0.70` same player, `0.35` same team, `0.15` same game, `0.00` otherwise) once more graded slips accumulate.
- [ ] **Prop-type monitoring for betslips** - monitor live STL/BLK slip behavior and decide whether further per-prop safeguards are needed beyond the hard confidence gate. Current first truly labeled active slate (`2026-04-14`) produced a `5`-pick card that is `100% demon` lines.

## NEXT

- [ ] **Backfill or consciously leave legacy line history** - decide whether to migrate older `historical_lines.csv` rows to explicit `projection_type='standard'` / `is_promo=False` columns or keep them as legacy data until new rows accumulate.
- [ ] **2-pick-only parallel tracking** - add a separate tracking mode that logs only 2-pick betslips to validate whether that format is more profitable at current accuracy levels.
- [ ] **Backtesting module expansion** - extend the current under-only backtest helper so it can compare 2-pick only, prop-filtered, and confidence-filtered slip strategies against full `picks_history.csv`.
- [ ] **Line movement integration** - track intraday line movement and feed agreement/disagreement into confidence.
- [ ] **OVER/UNDER bias analysis** - audit `picks_history.csv` for systematic directional bias after standard-line gating is materialized in new live saves.

## LATER
- [ ] **PrizePicks 403 long-term fix** - automate session cookie refresh or investigate an authenticated Playwright session with `PRIZEPICKS_CHROME_PROFILE`.
- [ ] **Consolidate duplicate `data/` directories** - audit and collapse any duplicate data roots.
- [ ] **Audit blowout baseline CSVs for duplicate rows** - Step 8 dedup is still a merge-time fix; source CSV cleanup is still open.

## BLOCKED
- [ ] **PrizePicks line accuracy analysis** - blocked by ongoing `403/429`; cannot reliably compare Odds API lines vs. PrizePicks lines on the same days.

## DONE (April 2026 sessions)
- [x] **Terminal log output reformatted (2026-04-15)** — Replaced 26-column `DataFrame.to_string()` dumps with clean fixed-width pick tables in `nba_props.py` (TODAY'S PICKS block) and `run_daily.py` (`_summarise_live_pick_outputs`). Root cause: `run_daily.py` pipes all subprocess stdout through `logger.info("  {line}")`, making wide DataFrames unreadable. New format: PLAYER / PROP / DIR / LINE / PRED / EDGE / CONF / HIT% / TIER with optional mkt/σ/INJ extras. Syntax verified.
- [x] **Archive-gap backfill + stale-date DNP auto-grade (2026-04-15)** — `run_daily.py` `step1_5_grade_picks` now (1) scans `output/archive/` on every morning run and auto-restores any dated picks folder missing from `picks_history.csv`, and (2) marks picks as DNP when a slate date is >3 days old with no box scores from either `nba_data.csv` or the live nba_api. April 11 (no games, all DNP) and April 14 (4-0) were manually backfilled; history is now at 1,415 rows with no archive gaps. Syntax verified.
- [x] **Graded April 14 picks (4-0)** — Toumani Camara TRB UNDER (actual 4), Moussa Diabaté AST UNDER (actual 1), Bam Adebayo STL UNDER (actual 1), Andrew Wiggins BLK UNDER (actual 0). All WIN. Root cause: NBA API lag kept `nba_data.csv` at Apr 12 cutoff, so the predict step never ran → picks not appended to history.
- [x] **Verified `2026-04-12` card result** - `output/picks_history.csv` now confirms the full `2026-04-12` card graded `5W / 0L / 0P`.
- [x] **First truly labeled active slate** - after the 2026-04-13 morning PrizePicks pull, `_select_active_slate_lines()` now resolves to a genuinely labeled `2026-04-14` board (`257` rows: `234 demon`, `14 standard`, `9 goblin`).
- [x] **Materialize new market/EV schema on a live save** - `python prizepicks_scraper.py` succeeded on 2026-04-12 and rewrote `data/historical_lines.csv` with `projection_type` / `is_promo`; `python nba_props.py predict` then rewrote `output/picks_latest.csv` / `output/picks_history.csv` with `projection_type`, `is_promo`, `break_even_prob`, and `exceeds_ev_threshold`.
- [x] **PrizePicks fetch stabilization** - `_fetch_direct()` now tries the lighter `league_id` request variants that succeeded live, and the Playwright profile fallback now warns and returns `None` instead of crashing the entire line save on `TargetClosedError`.
- [x] **PrizePicks market-type fix** - `prizepicks_scraper.py` now uses `odds_type` as the placeable `standard/goblin/demon` contract, preserves `is_promo`, and logs both `projection_type` and `odds_type` for diagnostics. `nba_props.py` carries that metadata end-to-end and blocks unplaceable pick directions.
- [x] **Live API field verification** - direct PrizePicks API smoke on 2026-04-12 confirmed `projection_type='Single Stat'`, `odds_type='demon'`, and `is_promo=false`, resolving the old field-name ambiguity.
- [x] **Betslip vig accounting + true EV + correlation-aware selection** - `filter_best_picks()` now emits `break_even_prob` and `exceeds_ev_threshold`; `log_betslips()` now writes slip EV fields and ranks Power Play candidates by correlation-adjusted EV.
- [x] **Backtest new under-only slip mix** - `backtest_under_only_slips()` and `scripts/backtest_under_only_slips.py` generated fresh artifacts. Current summary: 2-pick UNDER-only `9` slips, `0.4444` ROI; 3-pick UNDER-only `9` slips, `1.2222` ROI.
- [x] **Extract shared injury-feed module** - official NBA + ESPN injury helpers now live in `injury_feed.py`; both `run_daily.py` and `nba_props.py` import the same contract.
- [x] **Pending-report injury gap** - Step 8 now stamps unresolved `live_team_status_pending=True` rows as `team_pending`, writes explicit live status/comment/source fields, suppresses their luck fields, and removes them from the active top-10 display when ESPN has no player-level entry.
- [x] **Betslip confidence hard gate** - `log_betslips()` now enforces `confidence >= 70` before any leg can enter a real-money slip
- [x] **STL/BLK under-only betslip restore** - under-only betslips now allow `STL` and `BLK` in addition to `AST` and `TRB`, gated by the live quality policy plus the hard 70-confidence floor
- [x] **Step 6 live injury persistence** - `run_daily.py` now fetches the combined official NBA + ESPN injury feed and writes `data/live_injury_status.json` and `data/live_injury_status.csv`
- [x] **Step 8 live injury merge** - `run_daily.py` now merges the live injury feed into `data/player_projections_today.csv` and uses it for luck suppression plus top-10 display filtering/tagging
- [x] **Step 9 live-prediction crash** - fixed the `run_daily.py` failure from `logs/run_20260411.log`; `nba_props.py:_predict()` now accepts FP blend inputs as Series, lists, or NumPy arrays
- [x] **Full pipeline runtime confirmation** - reran `python run_daily.py` on 2026-04-11; full wall clock is now verified at `486.6s`
- [x] **Problem 13** - Step 6 absence detection now skips ghost roster players whose `player_profiles.csv` team does not match their latest local game-log team; verified Anthony Davis and D'Angelo Russell no longer contaminate WAS absences
- [x] **Step 3 runtime reduction** - `blowout_discount_model.py` now uses local `data/nba_data.csv` as the primary per-player game-log source for clean baselines and only falls back to `PlayerGameLog` when local rows are missing
- [x] **Problem 11** - `market_edge_pred` confidence contribution fixed: now directional (amplifying -> add, dampening -> zero, disagree -> subtract); extreme dampening gate added
- [x] **volume_fill betslip exclusion** - `log_betslips()` now filters `pick_source != 'volume_fill'`
- [x] **Problem 9** - replaced the hard-coded live first-principles/XGB `35/65` blend with adaptive per-player weights driven by sample depth, current-season share, minutes stability, form drift, and clean-baseline coverage
- [x] **Problem 8** - Step 8 now clears all luck-derived fields for absence-risk players (`recent_gp <= 2`) before saving `data/player_projections_today.csv`
- [x] **Problem 6** - `_merge_layer_features()` name-based fallback added (diacritic-normalized); unmatched players now logged by name
- [x] **Problem 5** - Layer 4 luck scores 24h staleness skip removed; luck scores now always rebuild with fresh Step 1 data
- [x] **STATUS.md, TASKS.md, DECISIONS.md, ARCHITECTURE.md rewrite** - repo memory files were rebuilt with current project state

---

## Task Template

### Task
Short name:

### Objective
Exact intended outcome:

### Why it matters
How this affects model quality, reproducibility, or revenue:

### Inputs / files
Files, datasets, scripts, and configs involved:

### Risks
Leakage, schema breakage, backfill issues, runtime issues:

### Validation
Exact commands or checks required before marking complete:

### Done when
Concrete acceptance criteria:
