# TASKS.md

# Active Work Queue

## NOW
- [x] **[DISTRIBUTION] Extract σ from quantile spread** - `{target}_sigma = (q75-q25)/1.35` now computed in `_predict_all_targets()` and exposed as a column. `filter_best_picks()` now prefers `{prop}_sigma` over static `{prop}_std`. Context-conditioned via quantile sidecar features. (`nba_props.py`)
- [x] **[DISTRIBUTION] σ-context calibration analysis** - `compute_sigma_context_analysis()` added to `probability_utils.py`; called from `calibrate_confidence.py`; output saved to `models/sigma_context_analysis.json`. Shows Brier/ECE/hit rate bucketed by σ quartile per target.
- [x] **[DISTRIBUTION] Routing verified + pra/pr/pa quantile sidecars** - TRB/AST confirmed correctly on quantile regression (Poisson is wrong here — quantile is nonparametric). `PROBABILITY_TARGETS` in `probability_utils.py` expanded to include `pra`, `pr`, `pa` so they get quantile sidecars on next retrain.
- [x] **[DISTRIBUTION] σ-tier confidence/priority bonus** - `_load_sigma_p25_thresholds()` added; `filter_best_picks()` sets `low_sigma_under=True` when `{prop}_sigma < p25_threshold`, applies `base_conf += 0.04` (+4 conf_pct) and `selection_priority += 0.50`. `sigma` and `low_sigma_under` columns added to all pick output paths. Thresholds: AST=1.68, TRB=2.10, PTS=5.58.
- [ ] **Pending-report injury gap** - tighten Step 8 handling for players on teams with pending official reports when ESPN also lacks a player entry. Current example: Kawhi can still show as active because he is absent from both the absence proxy and the live status map.
- [ ] **Backtest new under-only slip mix** - measure AST/TRB/STL/BLK under-only betslip performance under the new hard `confidence >= 70` gate before treating the new pool as settled bankroll policy.

## NEXT
- [ ] **Extract shared injury-feed module** - move official NBA + ESPN injury helpers out of `nba_props.py` so both `run_daily.py` and `nba_props.py` read from one dedicated contract instead of a cross-import.
- [ ] **Prop-type monitoring for betslips** - monitor live STL/BLK slip behavior and decide whether further per-prop safeguards are needed beyond the hard confidence gate.
- [ ] **2-pick-only parallel tracking** - add a separate tracking mode that logs only 2-pick betslips to validate whether that format is more profitable at current accuracy levels.

## LATER
- [ ] **Backtesting module** - simulate different slip strategies (2-pick only, prop-type filtered, confidence-filtered) against full `picks_history.csv` before changing bankroll allocation.
- [ ] **Line movement integration** - track intraday line movement and feed agreement/disagreement into confidence.
- [ ] **OVER/UNDER bias analysis** - audit `picks_history.csv` for systematic directional bias.
- [ ] **PrizePicks 403 long-term fix** - automate session cookie refresh or investigate an authenticated Playwright session with `PRIZEPICKS_CHROME_PROFILE`.
- [ ] **Consolidate duplicate `data/` directories** - audit and collapse any duplicate data roots.
- [ ] **Audit blowout baseline CSVs for duplicate rows** - Step 8 dedup is still a merge-time fix; source CSV cleanup is still open.

## BLOCKED
- [ ] **PrizePicks line accuracy analysis** - blocked by ongoing `403/429`; cannot reliably compare Odds API lines vs. PrizePicks lines on the same days.

## DONE (April 2026 sessions)
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
