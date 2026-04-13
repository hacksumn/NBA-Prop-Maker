# STATUS.md

## Current Objective

**[2026-04-12] Build enough true-labeled market history to audit standard-line edge cleanly.**

Shipped today:

1. PrizePicks parsing now uses `odds_type` as the placeable market contract. The live API check on 2026-04-12 showed `projection_type="Single Stat"`, `odds_type="demon"`, and `is_promo=false`, so the old `projection_type` assumption was wrong.
2. `prizepicks_scraper.py` and `nba_props.py` now preserve normalized `projection_type` (`standard` / `goblin` / `demon`) plus `is_promo` through the line and pick paths.
3. `filter_best_picks()` now blocks `UNDER` on `goblin` or promo lines and blocks `OVER` on `demon` lines, including the fill paths.
4. `filter_best_picks()` now emits `break_even_prob` and `exceeds_ev_threshold` for each pick.
5. `log_betslips()` now ranks Power Play slips by correlation-adjusted EV and writes `joint_prob_raw`, `estimated_correlation`, `joint_prob_adj`, `break_even_joint_prob`, `ev_per_unit`, and `ev_dollar`.
6. `backtest_under_only_slips()` plus `scripts/backtest_under_only_slips.py` generated fresh under-only backtest artifacts in `output/`.
7. `injury_feed.py` is now the shared official NBA + ESPN injury contract for both `run_daily.py` and `nba_props.py`.
8. Step 8 still includes the pending-team fallback: unresolved rows on pending-report teams are stamped as `team_pending`, have luck fields suppressed, and are removed from the active top-10 display.

Materialization status:

- `python prizepicks_scraper.py` succeeded on 2026-04-12 at `20:15:39` and rewrote `data/historical_lines.csv` with `projection_type` and `is_promo`.
- `python nba_props.py predict` then rewrote `output/picks_latest.csv` and `output/picks_history.csv` with `projection_type`, `is_promo`, `break_even_prob`, and `exceeds_ev_threshold`.
- The remaining limitation is not schema materialization anymore; it is label depth. The fresh true-labeled PrizePicks snapshot is only `77` rows across `2026-04-14` and `2026-04-15`, while the active `2026-04-12` cached slate remains legacy-standard rows from earlier history.

## Current Branch / Working Scope
- Branch: `master`
- Scope: full pipeline, daily prediction, grading, and betslip construction
- Owner: Jake
- Last updated: 2026-04-12

## What Is Working
- Full daily pipeline was last confirmed end-to-end on `2026-04-11`; today's changes were verified with targeted tests, backtests, and live API smokes rather than a full rerun
- Box score ingestion (Step 1) and pick grading (Step 1.5) with live NBA API fallback
- Betslip construction and grading (Steps 1.6-9)
- Step 3 clean baselines use local `data/nba_data.csv` as the primary per-player game-log source and only fall back to `PlayerGameLog` when local rows are missing
- Step 6 absence detection excludes ghost roster players when `player_profiles.csv` team assignments disagree with the player's latest local game-log team
- Step 6 now also persists a live injury feed to:
  - `data/live_injury_status.json`
  - `data/live_injury_status.csv`
- Step 6 and `nba_props.py` now both import the same shared injury contract from `injury_feed.py`
- Direct `fetch_injury_data()` smoke on `2026-04-12` returned `36` teams, `271` player-status entries, and `0` pending teams
- Step 8 now merges both:
  - `data/detected_absences.csv`
  - `data/live_injury_status.csv`
- `data/player_projections_today.csv` now carries live availability columns:
  - `live_injury_bucket`
  - `live_injury_status`
  - `live_injury_source`
  - `live_team_status_pending`
- Step 8 now stamps unresolved pending-team rows as `live_injury_bucket='team_pending'`, suppresses their luck fields, and keeps them out of the active top-10 projection display
- Step 8 luck suppression is now driven by either:
  - absence proxy risk (`absence_recent_gp <= 2`)
  - live injury buckets (`out`, `doubtful`, `questionable`, `day_to_day`, `team_pending`)
- Top-10 projection display now suppresses official `OUT` / `DOUBTFUL` players and tags softer live statuses such as `QUESTIONABLE`, `DAY TO DAY`, and `PROBABLE`
- Live first-principles blending is player-adaptive instead of a hard-coded `35% FP / 65% XGB`
- Pick probability outputs are explicit and internally consistent: `dir_prob` and legacy `ou_prob` both represent the chosen-side probability, while `p_over_raw`, `p_over_cal`, `p_under_raw`, `p_under_cal`, and `prob_source` expose side-specific debug values
- `nba_props.py:_predict()` handles first-principles blend inputs as Series, lists, or NumPy arrays without crashing in the primitive ridge-blend path
- PrizePicks parsing now logs both `projection_type` and `odds_type` observations and uses `odds_type` as the market-side field for gating
- PrizePicks direct fetch was hardened: the live-success path now includes a minimal `league_id` request variant, and the Playwright profile fallback now fails closed instead of crashing the whole fetch
- `merge_vegas_lines()` and the predict paths now carry `projection_type` and `is_promo`
- Pick filtering now carries `projection_type`, `is_promo`, `break_even_prob`, and `exceeds_ev_threshold`
- `log_betslips()` now excludes:
  - `volume_fill` picks
  - any pick with `confidence < 70`
- `log_betslips()` now searches candidate combinations by correlation-adjusted EV instead of plain confidence order
- The under-only betslip pool now includes `AST`, `TRB`, `STL`, and `BLK`
- `data/historical_lines.csv` now materially contains `projection_type` and `is_promo`
- `output/picks_latest.csv` and `output/picks_history.csv` now materially contain `projection_type`, `is_promo`, `break_even_prob`, and `exceeds_ev_threshold`
- Latest successful PrizePicks snapshot summary:
  - snapshot timestamp: `2026-04-12 20:15:39`
  - rows: `77`
  - slate dates: `2026-04-14`, `2026-04-15`
  - market mix: `76 demon`, `1 standard`, `0 promo`
- Under-only slip backtest artifacts now exist:
  - `output/under_only_slip_backtest_detail.csv`
  - `output/under_only_slip_backtest_summary.json`
- Current backtest summary:
  - `2-pick UNDER-Only Backtest`: `9` slips, `5` wins, `4` losses, `0.4444` ROI
  - `3-pick UNDER-Only Backtest`: `9` slips, `5` wins, `4` losses, `1.2222` ROI
- Unit tests now cover:
  - slate-date market parsing
  - PrizePicks promo/goblin/demon gating
  - injury-feed merge precedence
  - EV-ranked betslip selection
  - Step 8 `team_pending` fallback

## What Is Not Working / At Risk
- End-to-end live verification of the new `team_pending` fallback is still outstanding on a real pending-team slate; today's saved projection artifact produced `0` fallback rows
- Standard-line-only edge analysis is still open; most historical rows were backfilled to `projection_type='standard'`, so today's fresh live materialization is not enough by itself for a clean hit-rate audit
- The active `2026-04-12` cached slate remains legacy-standard rows even though the newest true-labeled PrizePicks snapshot is for `2026-04-14` / `2026-04-15`
- PrizePicks `403/429` remains active most days
- Correlation weights are still heuristic (`0.70` same player, `0.35` same team, `0.15` same game, `0.00` otherwise); they are not yet learned from graded slip history
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
- PrizePicks field semantics were verified directly on `2026-04-12`: `odds_type` is the market contract, `projection_type` is a content label, `is_promo` is present as a boolean
- Blend leadership: Bayesian Ridge dominates AST (`57%`) and TRB (`72%`); ExtraTrees dominates PTS (`64%`)
- Cross-target UNDER edge hit rate: `62.3%` (`n=16,436`)
- Cross-target OVER edge hit rate: `48.7%` (`n=9,929`)
- Retrained under-only evidence now driving betslip policy:
  - `AST UNDER`: `65.4%` hit, `n=2172`
  - `TRB UNDER`: `63.1%` hit, `n=2278`
  - `STL UNDER`: `74.0%` hit, `n=2034`
  - `BLK UNDER`: `79.2%` hit, `n=1756`
- Latest under-only slip backtest summary on current logic:
  - `2-pick`: `9` slips, `55.56%` win rate, `0.4444` ROI
  - `3-pick`: `9` slips, `55.56%` win rate, `1.2222` ROI

## Active Risks
- PrizePicks scraping can silently fall back to Odds API lines; always check the `line_source` warning in logs
- Odds API fallback and legacy line history default to `projection_type='standard'`, so the market gate is most trustworthy on fresh PrizePicks-backed saves
- `volume_fill` picks are capped at `63%` confidence and excluded from betslips, but they still appear in `picks_latest.csv`
- `player_profiles.csv` can still carry stale official team assignments after trades; the local-team mismatch filter prevents those rows from contaminating `detected_absences.csv`, but it is not a substitute for a live roster or injury feed
- The new Step 6 live-injury path depends on external official/ESPN availability; if both sources fail, the pipeline falls back to the absence proxy only
- Step 8 now marks unresolved pending-team rows explicitly, but it still needs live-slate monitoring on the next day with a pending official report
- Standard-line OVER edge may look materially different once old ungated history is separated from new gated market metadata

## Recent Changes

- **[2026-04-12] PrizePicks fetch stabilization** - `prizepicks_scraper.py` now tries more direct API header/param variants, including the minimal `league_id` request shape that succeeded live today, and the Playwright profile fallback now returns a warning instead of crashing on `TargetClosedError`.
- **[2026-04-12] PrizePicks market-type fix** - `prizepicks_scraper.py` now captures `is_promo` and uses `odds_type` (not raw `projection_type`) as the placeable market-side field, while `nba_props.py` carries that metadata through line normalization, vegas merge, pick filtering, and fill paths.
- **[2026-04-12] Power Play EV-ranked betslips** - `filter_best_picks()` now emits `break_even_prob` and `exceeds_ev_threshold`; `log_betslips()` now writes true EV and correlation fields and ranks slips by correlation-adjusted EV instead of plain confidence order.
- **[2026-04-12] Under-only slip backtest** - `backtest_under_only_slips()` plus `scripts/backtest_under_only_slips.py` generated `output/under_only_slip_backtest_detail.csv` and `output/under_only_slip_backtest_summary.json`.
- **[2026-04-12] Shared injury-feed module** - official NBA + ESPN injury helpers were extracted into `injury_feed.py`; both `run_daily.py` and `nba_props.py` now import the same feed contract.
- **[2026-04-12] Step 8 pending-team fallback** - `run_daily.py` now stamps unresolved rows on `live_team_status_pending=True` teams as `team_pending`, writes explicit live status/comment/source fields, treats that bucket as luck-suppression risk, and suppresses those rows from the active top-10 display. Targeted unit test added in `tests/test_pending_team_status_fallback.py`.
- **[2026-04-12] Live materialization completed** - `python prizepicks_scraper.py` rewrote `data/historical_lines.csv` with market metadata, and `python nba_props.py predict` rewrote `output/picks_latest.csv` / `output/picks_history.csv` with pick EV fields from a PrizePicks-backed active slate.
- **[2026-04-12] σ distribution upgrades tested and debugged** - 8 automated tests run across DEC-022/023/024. Bug found and fixed in `compute_sigma_forward_validation()`: direct column access on `edge`/`confidence` crashed with `KeyError` when those columns were absent from older history rows; fixed with `.columns` guards falling back to `np.nan`. All 8 tests pass: loader correctness, edge math, column persistence, gate boundary filtering, confidence/priority bonus, CSV write, forward validation bucketing, and all 4 graceful-skip paths.
- **[2026-04-12] σ-tiered dynamic `min_edge_under`** - `filter_best_picks()` now lowers the edge gate by 25% (floor 0.50) for low-σ UNDER picks (σ < p25) and raises it 25% for high-σ UNDER picks (σ > p75). `_load_sigma_thresholds()` returns both p25 and p75 dicts from `models/sigma_context_analysis.json`. Effective bars: AST low=0.675/high=1.125, TRB low=0.900/high=1.500, PTS low=1.500/high=2.500.
- **[2026-04-11] Betslip confidence hard gate** - `log_betslips()` now enforces `confidence >= 70` before any leg can enter a real-money slip
- **[2026-04-11] STL/BLK under-only betslip restore** - the under-only betslip pool now includes `STL` and `BLK` in addition to `AST` and `TRB`, gated by the live quality policy plus the new hard confidence floor
- **[2026-04-11] Step 6 live injury persistence** - `run_daily.py` now fetches the combined official NBA + ESPN injury feed and writes `data/live_injury_status.json` and `data/live_injury_status.csv`
- **[2026-04-11] Step 8 live injury merge** - `run_daily.py` now merges the persisted live injury feed into `data/player_projections_today.csv` and uses it for luck suppression plus top-10 display filtering/tagging
- **[2026-04-11] Step 9 live-prediction crash fix** - `nba_props.py:_predict()` now coerces FP blend inputs safely for pandas Series, lists, and NumPy arrays
- **[2026-04-11] Full runtime re-measured** - reran `python run_daily.py`; full wall clock now verified at `486.6s`

## Current Blockers
- **[CRITICAL] Standard-line strategy audit** - the plumbing fix is shipped and artifacts are now materialized, but the true OVER/UNDER edge on `projection_type='standard'` and `is_promo=False` lines is still unknown because almost all older rows were backfilled to standard.
- **[CRITICAL] True-labeled sample depth** - the newest verified PrizePicks snapshot has only `77` true-labeled rows across `2026-04-14` / `2026-04-15`; that is not enough graded history to trust a full directional edge audit yet.
- PrizePicks `403/429`

## Next 3 Highest-Priority Steps

1. **Accumulate more true-labeled PrizePicks history** - keep running the live save path so `historical_lines.csv` contains enough real `standard/goblin/demon` rows to support a meaningful edge audit.
2. **Audit standard-line OVER/UNDER edge** - once enough true-labeled rows are graded, recompute hit rates on `projection_type='standard'` and `is_promo=False` only.
3. **Monitor the next pending-team slate** - confirm `data/player_projections_today.csv` shows explicit `team_pending` rows on a real day with a pending official report.

## Validation Snapshot
Last verification performed: 2026-04-12
- `python -c "import ast, pathlib; files=['prizepicks_scraper.py','injury_feed.py','run_daily.py','nba_props.py','scripts/backtest_under_only_slips.py','tests/test_slate_date_handling.py','tests/test_prizepicks_market_gating.py','tests/test_injury_feed_merge.py','tests/test_betslip_ev_selection.py','tests/test_pending_team_status_fallback.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8')) for f in files]; print('syntax ok')"` -> OK
- `python -m unittest tests.test_slate_date_handling tests.test_prizepicks_market_gating tests.test_injury_feed_merge tests.test_betslip_ev_selection tests.test_pending_team_status_fallback` -> OK (`8` tests)
- `python scripts/backtest_under_only_slips.py` -> OK; wrote `output/under_only_slip_backtest_detail.csv` and `output/under_only_slip_backtest_summary.json`
- Direct PrizePicks API smoke on `2026-04-12` -> intermittent. A successful response earlier in the session confirmed keys include `projection_type`, `odds_type`, and `is_promo`, with observed sample `projection_type='Single Stat'`, `odds_type='demon'`, `is_promo=False`; later retries also hit `403` and `429`
- `python -c "from injury_feed import fetch_injury_data; data=fetch_injury_data(); print('teams', len(data)); print('pending_teams', sum(1 for v in data.values() if v.get('not_yet_submitted'))); print('player_statuses', sum(len(v.get('status_map', {})) for v in data.values()))"` -> OK (`teams 36`, `pending_teams 0`, `player_statuses 271`)
- `python prizepicks_scraper.py` -> OK; direct API succeeded on `minimal-league-only / variant 0`, parsed `77` validated NBA lines across `2026-04-14` / `2026-04-15`, and rewrote `data/historical_lines.csv`
- `python nba_props.py predict` -> OK; used `1015` cached PrizePicks lines on active slate `2026-04-12`, generated `5` live picks, and rewrote `output/picks_latest.csv`, `output/picks_history.csv`, `output/betslips_latest.csv`, and `output/betslips_history.csv`
- `python -c "import pandas as pd; from pathlib import Path; hist=pd.read_csv(Path('data/historical_lines.csv')); picks=pd.read_csv(Path('output/picks_latest.csv')); print('historical_has', 'projection_type' in hist.columns, 'is_promo' in hist.columns); print('picks_has', [c for c in ['projection_type','is_promo','break_even_prob','exceeds_ev_threshold'] if c in picks.columns])"` -> OK (`historical_has True True`; `picks_has ['projection_type', 'is_promo', 'break_even_prob', 'exceeds_ev_threshold']`)
- `python -c "import pandas as pd; from nba_props import _select_active_slate_lines; hist=pd.read_csv('data/historical_lines.csv'); active, slate=_select_active_slate_lines(hist); print('active_slate', slate); print(active['projection_type'].fillna('missing').value_counts().to_dict()); print(active['is_promo'].fillna(False).astype(bool).value_counts().to_dict())"` -> OK (`active_slate 2026-04-12`; `{'standard': 1015}`; `{False: 1015}`)
- `python -c "import pandas as pd; hist=pd.read_csv('data/historical_lines.csv'); latest=hist[hist['source'].astype(str).eq('prizepicks_scraper')].copy(); latest['snapshot_ts']=pd.to_datetime(latest['snapshot_ts'], errors='coerce'); max_ts=latest['snapshot_ts'].max(); snap=latest[latest['snapshot_ts'].eq(max_ts)].copy(); print('projection_type_counts', snap['projection_type'].fillna('missing').value_counts().to_dict()); print('promo_counts', snap['is_promo'].fillna(False).astype(bool).value_counts().to_dict())"` -> OK (`{'demon': 76, 'standard': 1}`; `{False: 77}`)

---

## Session Handoff
- Last completed task: stabilized the PrizePicks fetch path, successfully materialized market metadata into `historical_lines.csv`, and rewrote the picks artifacts under the new EV-aware schema
- Current objective: accumulate enough true-labeled market history to run a trustworthy standard-line edge audit
- Current branch: `master`
- Files changed: `prizepicks_scraper.py`, `injury_feed.py`, `nba_props.py`, `run_daily.py`, `scripts/backtest_under_only_slips.py`, `tests/test_slate_date_handling.py`, `tests/test_prizepicks_market_gating.py`, `tests/test_injury_feed_merge.py`, `tests/test_betslip_ev_selection.py`, `tests/test_pending_team_status_fallback.py`, `output/STATUS.md`, `output/TASKS.md`, `output/DECISIONS.md`
- Verification run:
  - syntax parse for the updated market, injury-feed, betslip, and test files
  - `python -m unittest tests.test_slate_date_handling tests.test_prizepicks_market_gating tests.test_injury_feed_merge tests.test_betslip_ev_selection tests.test_pending_team_status_fallback`
  - `python scripts/backtest_under_only_slips.py`
  - direct PrizePicks API smoke
  - direct `fetch_injury_data()` smoke
  - `python prizepicks_scraper.py`
  - `python nba_props.py predict`
  - on-disk artifact schema and slate-mix checks for `historical_lines.csv` and `picks_latest.csv`
- Known issues: PrizePicks availability remains unstable; the active slate is still legacy-standard while the newest true-labeled snapshot is future-dated; the `team_pending` fallback still needs confirmation on the next real pending-team slate
- Immediate next step: keep collecting true-labeled PrizePicks rows, then run the standard-line-only edge audit once enough of those rows are graded

## Resume Here
When starting a new session:
1. Read `CLAUDE.md`, `output/STATUS.md`, `output/TASKS.md`, and `output/DECISIONS.md` in that order.
2. Confirm current objective and blockers from repo state, not chat memory.
3. Keep changes scoped and verify exact commands before marking work complete.
4. Update `output/STATUS.md`, `output/TASKS.md`, and `output/DECISIONS.md` before stopping.
