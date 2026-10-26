# Changelog — NBA Prop Maker

---

## [2.1.2] — 2026-10-25

### Fixed (early-season crash in step 8 found on the first 2026-27 run)
- **Root cause:** before any player has 5 games, step 3 wrote an empty `player_clean_baselines.csv` and step 6 wrote an empty `player_profiles.csv`; step 8 then crashed reading the empty baselines file (`No columns to parse from file`), leaving `player_projections_today.csv` stale.
- `_csv_has_rows()` added. Steps 3 and 6 no longer overwrite an existing file with an empty result, and an empty file is no longer treated as "fresh" (so it is rebuilt on the next run).
- Step 8 skips Layer 5 when its file is empty (same fallback path as "file not found": projections use raw L20).
- Step 9 refuses to generate picks if step 8 failed in the same run AND `player_projections_today.csv` is older than 48h (or missing). If step 8 fails but the file is recent, picks still run with a warning.
- No columns added, removed or renamed. Backup: `run_daily.py.bak_before_empty_layer_fix`.

### Added
- `rebuild_profiles_2025_26.py` (one-off): rebuilds the emptied `data/player_profiles.csv` from 2025-26 final stats.

### Known limits
- `player_clean_baselines.csv` (last season's) was emptied before this fix and cannot be restored; it rebuilds automatically once players reach 5 GP in 2026-27. Until then projections use raw L20 (no `clean_*` columns).
- `nba_props.py predict` (step 9) has not been run against projections without the Layer 5 columns.

---

## [2.1.1] — 2026-10-25

### Changed
- **`run_daily.py`**: new `_apply_current_roster_team()` called in step 8. `team_abbr` in the base projections now comes from `data/reference/official_rosters/official_roster_<season>.csv` (matched on player_id, then normalized name) instead of the player's last logged game, which is stale for anyone traded or signed since. Players not on the roster file keep their previous team. Fails soft (missing/invalid roster file leaves projections unchanged). No columns added, removed or renamed.
- Offline check against `nba_data.csv` (through 2026-05-18): 120 of 802 players get a new team; 447/802 matched to the 2026-27 roster.
- Backup of the prior file: `run_daily.py.bak_before_roster_fix`.

### Added
- `data/reference/official_rosters/official_roster_2026-27.csv` (521 players, 30 teams; `player_pool_guard.py` overwrites it on its normal 18-hour refresh).
- `data/reference/official_rosters/roster_team_changes_2026-27.csv` (last logged team vs 2026-27 roster team, per player).

### Known limits
- 74 roster players (mostly rookies/new signings) have no rows in `nba_data.csv`, so they get no projection until they log a 2026-27 game.

---

## [2.1.0] — 2026-03-25

### Updated
- **README** fully rewritten with real live performance stats (56.4% win rate, 840 graded picks)
- **models/results.json** updated with latest regression MAE and O/U accuracy metrics
- **models/bias.json** updated with latest per-prop bias correction values
- All pipeline scripts (`task1–task10`) synced from local working copy
- All utility scripts (`scripts/utils/`) synced from local working copy
- `nba_props.py`, `nba_scraper.py`, `prizepicks_scraper.py`, `run_daily.py` all updated to latest versions
- `full_pipeline.py`, `feature_pipeline.py`, `advanced_features.py`, `train_advanced_models.py` added/updated
- `generate_picks_v2.py`, `minutes_model.py`, `minutes_deviation.py` synced
- `feature_cols.json`, `feature_cols_advanced.json` updated
- `results/real_lines.csv` and `results/vegas_lines_consolidated.csv` updated

### Live Performance as of 2026-03-25
- **Overall Win Rate: 56.4%** (474 W / 366 L — 840 graded picks)
- AST: 73.1% | PA: 61.7% | TRB: 61.7% | PR: 56.7% | PTS: 51.7% | PRA: 51.5%
- Model MAE: PTS 4.14 | TRB 1.84 | AST 1.32 | TOV 0.87 | STL 0.70 | BLK 0.51
- Data: 75,799 player-game rows, 796 players, 3 seasons (2023-24 through 2025-26)
- PrizePicks lines tracked: 21,474 rows

---

## [2.0.0] — 2026-03

### Added
- `run_daily.py` — unified morning runner (data fetch + auto-grading)
- `nba_scraper.py` — incremental NBA API game log fetcher
- `prizepicks_scraper.py` — PrizePicks line fetcher with Cloudflare bypass (direct API first, Playwright fallback)
- Auto-grading of individual picks and betslips against real box scores each morning
- Betslip generation (2/3/4-pick Power Plays) with auto-grading next day
- `run_morning.bat` for Windows Task Scheduler automation
- `show_betslip.py` utility for daily betslip display
- Minutes prediction model to filter limited/DNP players
- Starter/foul/blowout features in minutes model
- Injury OUT/DOUBTFUL player filtering via live injury reports
- Deduplication: 1 pick per player in display and betslips
- Role-change sanity check to filter bad picks before saving
- Output paths unified: betslips_history + picks_history → `output/`

### Changed
- Model architecture shifted from old task-based pipeline to unified `nba_props.py`
- Data paths standardized to `files/data/` (local) and `data/` (repo)

---

## [1.1.0] — 2026-01-17 — CRITICAL FIX

### Bug Fix: Data Leakage Removed

**Issue:** Models were using current-game statistics (`FANTASY_PTS`, `PLUS_MINUS`) as features, causing unrealistically high prediction edges (15–18 points).

**Fix:**
- Removed `FANTASY_PTS` and `PLUS_MINUS` from all feature sets
- Retrained all models with clean, leakage-free features
- Added `task5b_rigorous_validation.py` for ongoing leakage detection

**Results After Fix:**
- Prediction edges: 2–5 points (realistic)
- Betting accuracy: 52–58% range
- All 6 validation tests passing

### Model Performance (After Leakage Fix)

| Prop | Win Rate | ROI |
|---|---|---|
| TRB | 57.5% | +10.7% |
| TOV | 56.7% | +9.1% |
| PA | 56.3% | +8.2% |
| PR | 56.0% | +7.6% |
| PRA | 55.4% | +6.3% |
| PTS | 55.4% | +6.3% |
| AST | 55.3% | +6.1% |
| STL | 52.7% | +0.7% |
| BLK | 48.2% | -8.7% ❌ Avoid |

### Added
- `task6_train_models_optimized.py` — enhanced training with hyperparameter tuning
- `task5b_rigorous_validation.py` — comprehensive leakage validation
- `scripts/utils/auto_retrain.py` — automated retraining script
- Classifier models per prop (`models/classifiers/`)
- Regression models per prop (`models/regression/`)
- Matchup history features (opponent positional defense)

---

## [1.0.0] — 2025-12 — Initial Release

- Basic rolling average features (L5, L10, L20)
- GradientBoosting regression models for PTS, TRB, AST, STL, BLK, TOV
- PrizePicks line scraping
- Manual pick grading via `scripts/utils/grade_bets.py`
- Results archived in `results/archive/`

> **Note:** This version had data leakage issues. Use v1.1.0 or later.

---

**Current Version:** 2.1.0
**Status:** Active — running daily
**Last Updated:** March 25, 2026
