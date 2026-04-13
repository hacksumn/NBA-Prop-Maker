# TASKS.md

# Active Work Queue

## NOW
- [ ] **[CRITICAL] Accumulate true-labeled market history** - keep running the live PrizePicks save path so `historical_lines.csv` contains enough real `standard/goblin/demon` rows to support a meaningful post-gate audit. Current verified live snapshot is only `77` rows (`76 demon`, `1 standard`) across `2026-04-14` / `2026-04-15`.
- [ ] **[CRITICAL] OVER edge audit on standard lines only** - recompute hit rates on `projection_type='standard'` and `is_promo=False` lines only once enough true-labeled rows are graded. The old `48.7%` OVER rate is blended across ungated history, and most backfilled legacy rows now default to `standard`.
- [ ] **Monitor first live `team_pending` slate** - on the next day with a pending official report, rerun the normal pipeline and confirm `data/player_projections_today.csv` contains explicit `team_pending` rows instead of blank live status fields.
- [ ] **Power Play EV calibration audit** - compare realized slip outcomes against the current heuristic correlation weights (`0.70` same player, `0.35` same team, `0.15` same game, `0.00` otherwise) once more graded slips accumulate.
- [ ] **Prop-type monitoring for betslips** - monitor live STL/BLK slip behavior and decide whether further per-prop safeguards are needed beyond the hard confidence gate.

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
