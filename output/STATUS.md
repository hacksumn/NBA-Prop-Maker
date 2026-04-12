# STATUS.md

## Current Objective
The three immediate follow-ups from the last session are complete:
1. betslips now enforce `confidence >= 70`
2. the under-only betslip pool now allows `AST`, `TRB`, `STL`, and `BLK`
3. Step 6 and Step 8 now consume a live injury feed instead of relying only on the recent-games absence proxy

**[2026-04-12] Distribution modeling upgrade — COMPLETE.**
All three upgrades shipped:
1. **σ extraction**: `{target}_sigma = (q75 - q25) / 1.35` now computed in `_predict_all_targets()` and exposed as a column. `filter_best_picks()` now prefers `{prop}_sigma` over rolling `{prop}_std` for z-score computation. Context-conditioned because quantile models are trained on minutes, usage, pace, and matchup features.
2. **σ-context analysis**: `compute_sigma_context_analysis()` added to `probability_utils.py`; called from `calibrate_confidence.py`; results saved to `models/sigma_context_analysis.json`. Shows Brier/ECE/hit rate bucketed by σ quartile per target.
3. **Routing verified + pra/pr/pa added**: TRB/AST confirmed correct on quantile regression (not Poisson — quantile is nonparametric and handles correlated high-range stats better). `PROBABILITY_TARGETS` expanded from `{pts,trb,ast}` to `{pts,trb,ast,pra,pr,pa}` so composite props get quantile sidecars on next retrain.

**[2026-04-12] σ-tier confidence/priority bonus — COMPLETE.**
`filter_best_picks()` now rewards low-σ UNDER picks:
- `_load_sigma_p25_thresholds()` reads `models/sigma_context_analysis.json` once per call. Loaded thresholds: AST=1.68, TRB=2.10, PTS=5.58.
- `low_sigma_under` flag fires when `direction == 'UNDER'` and `{prop}_sigma < p25_threshold`.
- `base_conf += 0.04` (+4 conf_pct) applied for all low-σ UNDER picks.
- `selection_priority += 0.50` stacks with `priority_under_lane` (which gives 1.00 for AST, 0.70 for TRB), putting low-σ UNDER picks at the top of the card.
- `sigma` and `low_sigma_under` columns added to all pick output paths and to `picks_latest.csv`.

Next action: monitor `picks_latest.csv` for `low_sigma_under=True` flags and confirm σ values are flowing through in the next daily run.

## Current Branch / Working Scope
- Branch: `master`
- Scope: full pipeline, daily prediction, grading, and betslip construction
- Owner: Jake
- Last updated: 2026-04-11

## What Is Working
- Full daily pipeline runs end-to-end (`run_daily.py` -> `nba_props.py predict`)
- Box score ingestion (Step 1) and pick grading (Step 1.5) with live NBA API fallback
- Betslip construction and grading (Steps 1.6-9)
- Step 3 clean baselines use local `data/nba_data.csv` as the primary per-player game-log source and only fall back to `PlayerGameLog` when local rows are missing
- Step 6 absence detection excludes ghost roster players when `player_profiles.csv` team assignments disagree with the player's latest local game-log team
- Step 6 now also persists a live injury feed to:
  - `data/live_injury_status.json`
  - `data/live_injury_status.csv`
- Latest live injury feed snapshot contains `251` player-status rows across `36` teams, with `5` pending official team reports
- Step 8 now merges both:
  - `data/detected_absences.csv`
  - `data/live_injury_status.csv`
- `data/player_projections_today.csv` now carries live availability columns:
  - `live_injury_bucket`
  - `live_injury_status`
  - `live_injury_source`
  - `live_team_status_pending`
- Step 8 luck suppression is now driven by either:
  - absence proxy risk (`absence_recent_gp <= 2`)
  - live injury buckets (`out`, `doubtful`, `questionable`, `day_to_day`)
- Top-10 projection display now suppresses official `OUT` / `DOUBTFUL` players and tags softer live statuses such as `QUESTIONABLE`, `DAY TO DAY`, and `PROBABLE`
- Live first-principles blending is player-adaptive instead of a hard-coded `35% FP / 65% XGB`
- Pick probability outputs are explicit and internally consistent: `dir_prob` and legacy `ou_prob` both represent the chosen-side probability, while `p_over_raw`, `p_over_cal`, `p_under_raw`, `p_under_cal`, and `prob_source` expose side-specific debug values
- `nba_props.py:_predict()` handles first-principles blend inputs as Series, lists, or NumPy arrays without crashing in the primitive ridge-blend path
- `log_betslips()` now excludes:
  - `volume_fill` picks
  - any pick with `confidence < 70`
- The under-only betslip pool now includes `AST`, `TRB`, `STL`, and `BLK`
- Latest synthetic betslip test confirmed:
  - `69`-confidence picks are excluded
  - `volume_fill` picks are excluded
  - `STL` and `BLK` can enter eligible betslip pools when they clear the live gate
- Latest full pipeline verification finished successfully at `2026-04-11 17:51:26`
- Latest full pipeline wall clock: `486.6s`
- Latest generated betslips have minimum leg confidence `75.2`

## What Is Not Working / At Risk
- Pending official team reports plus incomplete ESPN coverage can still leave some players unresolved in Step 8; Kawhi is the current example
- The live injury feed is now present, but it is reused by importing `fetch_injury_data()` from `nba_props.py` inside `run_daily.py`; this should be centralized into a shared module
- PrizePicks `403/429` remains active most days
- `pra`, `pr`, and `pa` composite models remain disabled due to weak blended performance
- Step 8 dedup is still a merge-time fix; duplicate-row root cause in blowout baseline sources is not fixed upstream

## Known Good Facts
- Canonical box score dataset: `data/nba_data.csv`
- Canonical lines dataset: `data/historical_lines.csv` with `source` tracking
- Canonical live injury artifacts:
  - `data/live_injury_status.json`
  - `data/live_injury_status.csv`
- Current primary live stack: XGBoost per-prop edge models + meta classifiers, all retrained 2026-04-11
- Evaluation remains chronological / walk-forward with the most recent season held out
- Active pick targets after retrain: `AST UNDER`, `TRB UNDER`, `STL UNDER`, `BLK UNDER`
- Blend leadership: Bayesian Ridge dominates AST (`57%`) and TRB (`72%`); ExtraTrees dominates PTS (`64%`)
- Cross-target UNDER edge hit rate: `62.3%` (`n=16,436`)
- Cross-target OVER edge hit rate: `48.7%` (`n=9,929`)
- Retrained under-only evidence now driving betslip policy:
  - `AST UNDER`: `65.4%` hit, `n=2172`
  - `TRB UNDER`: `63.1%` hit, `n=2278`
  - `STL UNDER`: `74.0%` hit, `n=2034`
  - `BLK UNDER`: `79.2%` hit, `n=1756`

## Active Risks
- PrizePicks scraping can silently fall back to Odds API lines; always check the `line_source` warning in logs
- `volume_fill` picks are capped at `63%` confidence and excluded from betslips, but they still appear in `picks_latest.csv`
- `player_profiles.csv` can still carry stale official team assignments after trades; the local-team mismatch filter prevents those rows from contaminating `detected_absences.csv`, but it is not a substitute for a live roster or injury feed
- The new Step 6 live-injury path depends on external official/ESPN availability; if both sources fail, the pipeline falls back to the absence proxy only
- Step 8 can now suppress official `OUT` / `DOUBTFUL` players, but team-pending gaps still require follow-up handling and monitoring

## Recent Changes
- **[2026-04-11] Betslip confidence hard gate** - `log_betslips()` now enforces `confidence >= 70` before any leg can enter a real-money slip
- **[2026-04-11] STL/BLK under-only betslip restore** - the under-only betslip pool now includes `STL` and `BLK` in addition to `AST` and `TRB`, gated by the live quality policy plus the new hard confidence floor
- **[2026-04-11] Step 6 live injury persistence** - `run_daily.py` now fetches the combined official NBA + ESPN injury feed and writes `data/live_injury_status.json` and `data/live_injury_status.csv`
- **[2026-04-11] Step 8 live injury merge** - `run_daily.py` now merges the persisted live injury feed into `data/player_projections_today.csv` and uses it for luck suppression plus top-10 display filtering/tagging
- **[2026-04-11] Step 9 live-prediction crash fix** - `nba_props.py:_predict()` now coerces FP blend inputs safely for pandas Series, lists, and NumPy arrays
- **[2026-04-11] Full runtime re-measured** - reran `python run_daily.py`; full wall clock now verified at `486.6s`

## Current Blockers
- No hard runtime blockers
- Primary operational risk is the remaining live injury coverage gap on pending official reports / missing ESPN entries
- PrizePicks `403/429`

## Next 3 Highest-Priority Steps
1. **Pending-report injury gap** - tighten handling for players on teams with pending official reports when ESPN also lacks the player entry
2. **Backtest new under-only slip mix** - measure STL/BLK under-only slip performance under the hard 70-confidence gate before treating this as settled bankroll policy
3. **Extract shared injury-feed module** - move the official/ESPN feed helpers out of `nba_props.py` so `run_daily.py` and `nba_props.py` read from one dedicated contract without a cross-import

## Validation Snapshot
Last verification performed: 2026-04-11
- `python -c "import ast; ast.parse(open('nba_props.py', encoding='utf-8').read()); ast.parse(open('run_daily.py', encoding='utf-8').read())"` -> OK
- synthetic `log_betslips()` test with temp output dir -> OK; `69`-confidence and `volume_fill` legs excluded, `STL`/`BLK` eligible when confidence clears `70`
- `from nba_props import fetch_injury_data; from run_daily import _flatten_live_injury_status` smoke test -> OK; flattened feed returned `251` player rows across `36` teams
- `python run_daily.py` -> OK; `logs/run_20260411.log` ends with `Pipeline complete in 486.6s` and `All steps completed successfully`
- projection artifact check -> OK; `data/player_projections_today.csv` contains `live_injury_bucket`, `live_injury_status`, `live_injury_source`, and `live_team_status_pending`
- betslip artifact check -> OK; latest `output/betslips_latest.csv` has minimum leg confidence `75.2`

---

## Session Handoff
- Last completed task: implemented the three requested follow-ups: hard betslip confidence gate, STL/BLK under-only pool restore, and Step 6/Step 8 live injury feed integration
- Current objective: close the remaining pending-report / missing-ESPN injury gap and validate the new STL/BLK under-only betslip policy with real historical evidence
- Current branch: `master`
- Files changed: `nba_props.py`, `run_daily.py`, `data/live_injury_status.json`, `data/live_injury_status.csv`, `data/player_projections_today.csv`, `output/betslips_latest.csv`, `output/picks_latest.csv`, `output/STATUS.md`, `output/TASKS.md`, `output/DECISIONS.md`, `output/ARCHITECTURE.md`
- Verification run:
  - syntax parse for `nba_props.py` and `run_daily.py`
  - synthetic `log_betslips()` policy test
  - injury-feed flatten smoke test
  - full `python run_daily.py`
- Known issues: Kawhi-style gaps can remain when the official report is pending and ESPN does not list the player; the injury feed is shared via cross-import instead of a dedicated module; PrizePicks availability remains unstable
- Immediate next step: investigate the pending-report injury gap before making any further pick-policy changes

## Resume Here
When starting a new session:
1. Read `CLAUDE.md`, `output/STATUS.md`, `output/TASKS.md`, and `output/DECISIONS.md` in that order.
2. Confirm current objective and blockers from repo state, not chat memory.
3. Keep changes scoped and verify exact commands before marking work complete.
4. Update `output/STATUS.md`, `output/TASKS.md`, and `output/DECISIONS.md` before stopping.
