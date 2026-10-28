# STATUS.md

## Current Objective

**[2026-10-27, DEC-061] DAILY COLLECTION, LEAKAGE-SAFE FULL-LAYER TRAINING, AND A REAL FUTURE SHADOW TEST ARE NOW OPERATING.** The PBP index is normalized to canonical 10-character NBA game IDs (223 distinct games; all 43 current-season games preserved), live scoring prefers the current-season `player_projections_today.csv` layer artifact, and availability/minutes inputs are reconstructed at prediction time rather than silently zero-filled. The completed training refresh produced 75,208 chronological availability predictions with zero chronology violations. Model acceptance is now based on the latest complete season: 2025-26 contributes 35,444 untouched candidates across 193 dates; the two 2026-27 dates are forward-scored only. The availability model beat its baselines on all four walk-forward folds and the complete-season holdout. Its join covers 89.4% of all eligible rows and 95.3% of the rotation-player proxy, with explicit 85%/95% fail-closed gates.

The unified table has 52,975 rows through 2026-10-25. Point-in-time defense joins 52,975/52,975 rows with zero duplicates, nulls, or chronology violations. An isolated nine-prop candidate was trained under `models/weekly_candidates/full_layers_20261027_150127`; production remained byte-hash unchanged. On 49,301 identical non-push real-line rows, no prop achieved a positive-confidence improvement over both production and always-UNDER, so nothing was promoted. The candidate is frozen for a separate future holdout beginning 2026-10-28, requiring at least 15 dates and 100 rows before manual review.

Four Windows tasks are installed: protected daily FanDuel scoring at 08:30, training-data refresh at 10:30, FanDuel T-60/T-5 capture at 12:00, and isolated weekly candidate training Sunday afternoon after the ordinary refresh. The October 27 watcher is alive with four games queued; before the first window it made zero paid calls and the provider checkpoint still reported 366 credits remaining / 134 used. Every paid event payload will be raw-checkpointed before parsing and merged at exact FanDuel player/market/line/price grain. Full verification is `160 passed, 330 warnings, 1 subtest passed`. See DEC-061.

**[2026-10-26, DEC-060] TODAY'S FANDUEL CARD IS LIVE AND THE DAILY CONTRACT NOW MATCHES THE OPERATOR'S ACTUAL BOOK.** FanDuel is the sole authoritative live source; PrizePicks is neither fetched nor used as fallback. The successful October 26 ingest saved 565 exact FanDuel contracts across 113 players, eight offered markets, and nine games. All 565 rows retain paired Over/Under prices, and the physical line store has zero duplicate keys. Exact alternate lines are retained instead of medianed. The provider reports 366 credits remaining and 134 used on the active key. A prior 62-credit interrupted attempt produced no data because successful responses were held in memory behind an unbounded event request; event calls are now time-bounded and every successful paid payload is atomically checkpointed before the next request.

The live path also now removes exact duplicate NBA schedule rows, preserves/normalizes leading-zero game IDs, refreshes exact team-game logs every day, and aligns canonical feature keys consistently. The exact team source contains 8,004 rows / 4,002 games through October 25 with zero duplicate keys or required nulls. The protected live run published two picks—Brandin Podziemski AST UNDER 3.5 and Jarrett Allen BLK UNDER 1.5—and created no slip because only two picks cleared the gates. The wrapper also refreshes the persisted user-level Odds API key before launching Python, preventing old terminals from reusing a canceled process-level key. All 56 production model hashes remained unchanged. Full verification is `151 passed, 1 subtest passed`; the 330 warnings are existing dataframe-fragmentation warnings. See DEC-060.

**[2026-10-25, DEC-059] ROUTINE OPERATION IS NOW ONE GUARDED COMMAND, THE LIVE LAYERS ARE REFRESHED, AND STALE CROSS-SEASON LUCK IS DISARMED.** `scripts/project_operator.ps1` provides explicit `Preflight`, `RefreshLayers`, `CapturePlan`, `StartCapture`, `DailySafe`, and `ReleaseManifest` actions. Every mutating daily/layer action hashes the complete 56-artifact production bundle before and after execution and fails if a non-promotion operation changes it. Odds API spending remains off by default; paid T-60 pairing can only be requested through `DailySafe -PairSportsbookT60`, which still requires the independent Python configuration gate. The protected layer refresh completed in 147.3 seconds: 36/36 PBP game fetches succeeded, the PBP master reached 138,289 player-game rows, Ridge ratings covered 30 teams and 999 players, the live injury feed returned 84 statuses, and the merged projection file contains 845 players. All 56 protected model hashes were unchanged.

The refresh exposed an early-season Layer 4 defect: no 2026-27 player yet met the 15-game luck minimum, so an empty frame raised `KeyError: 'total_luck_score'` and the orchestrator reused a 3,824-hour-old 2025-26 cache. The builder now returns a schema-valid empty frame; daily operation explicitly disables the luck layer until current-season eligibility exists; successful luck outputs are season-versioned; and Step 8 accepts only a nonempty same-season snapshot no more than 168 hours old. The repaired October projection run neutralized luck rather than importing last season's adjustments and completed with zero errors. `output/release_manifest_20261025.json` binds 100 source files, 11 canonical data files, 56 production artifacts, calibration lineage, policies, and evidence by SHA-256. Full verification is `139 passed, 1 subtest passed`; remaining warnings are the existing pandas fragmentation warnings. See DEC-059.

**[2026-10-25, DEC-058] THE USEFUL ZIP DATA IS RECOVERED; INVALID/OBSOLETE ARTIFACTS WERE LEFT ALONE.** The SHA-256-verified `Fresh_Start_NBA.zip` recovery restored 506 unique 2025-26 player usage profiles and 550 unique clean/blowout-adjusted player baselines as active early-season fallbacks and immutable season snapshots. The position cache is now a conflict-free union: 587 archived players plus one current-only player, 588 unique normalized names total, with zero blank keys, duplicate keys, or position conflicts. The 1,225-game 2025-26 blowout index is preserved under `data/season_snapshots/`; the active 36-game 2026-27 index was not overwritten and is byte-identical to its pre-recovery backup. `historical_lines.csv` and `nba_data.csv` were already supersets of the ZIP and were intentionally not touched. Source, model, OOF, pick, slip, live-injury, and projection artifacts from the ZIP/worktree were deliberately excluded.

The recovery is reproducible and fail-closed in `scripts/recover_zip_assets.py`: it verifies the known ZIP hash, schemas, minimum row floors, unique keys, allowed position values, and the 2025-26 date range before its first write; backs up every active cache; writes atomically; and emits `output/zip_recovery_20261025_manifest.json`. Daily usage-profile fallback now requires an actual data row rather than file existence, blowout/profile/baseline refreshes save season-specific snapshots, empty blowout refreshes cannot overwrite the current index, and player-position refreshes use the rolling NBA season plus union-preserving/partial-response guards. All restored hashes match the manifest, all 56 production model hashes remain unchanged, and the full suite passes: `134 passed, 1 subtest passed`. See DEC-058.

**[2026-10-25, DEC-057] THE OCTOBER PRODUCTION-BUNDLE INCIDENT IS CONTAINED AND THE RELEASE PATH IS FAIL-CLOSED.** The October 25 Sunday run continued after its feature rebuild failed, trained directly into `models/` from the stale 73,531-row August table, removed the required STL/BLK point-in-time defense selections, and then overwrote the trainer's chronological calibrator with an all-row isotonic fit. Production has been recovered from the hash-recorded DEC-052 exact-defense artifacts, with a fresh DEC-056-style calibrator fitted on 44,787 earlier real-line OOF rows and the latest 5,391 rows / 20 dates left untouched. The recovered active bundle again uses `opp_def_rtg_pt` for STL and `opp_pace_pt` for BLK; all nine sidecars/calibrators load under one hash contract. The broken October bundle is recoverable from `models/backups/dec057_pre_recovery_20261025/`, and the recovery is recorded in `output/dec057_production_recovery_manifest.json`.

Sunday maintenance no longer writes production. It now refreshes exact team logs, rebuilds the canonical feature table, rebuilds point-in-time defense, performs the one-to-one defense join, trains all nine props into a timestamped `models/weekly_candidates/` directory, runs the same-row OOF audit, and verifies every top-level production hash is unchanged. Calibration is performed once inside training with a chronological 20-date holdout; the standalone script was repaired to use the same split and is no longer called by Sunday maintenance. Promotion now validates defense chronology, retained STL/BLK rigorous fields, sidecar/OOF hashes, and a nonempty chronological calibration holdout before the first write; all production files are backed up before replacement and a failed activation rolls back.

Current audited data: 7,990 exact team-game rating rows / 3,995 games through `2026-10-24`, zero duplicate keys, required nulls, or chronology violations; the rigorous-defense prop join is `74,254/74,254` with zero feature nulls. Training/live parity passes on all 99 selected fields: 97 exact local matches and two scheduled external defense attachments. A current-data isolated candidate completed in `models/weekly_candidates/20261025_recovery_validation`; zero props cleared the promotion gate, so it was not promoted, and all 56 production hashes remained unchanged. PrizePicks still returned no current lines on October 25; that external availability issue is no longer logged as success, leaves `historical_lines.csv` untouched, and makes the pipeline exit nonzero. Full verification: `128 passed, 1 subtest passed`. See DEC-057.

**[2026-08-23, DEC-056] THE CALIBRATION/MODEL-BUNDLE DEFECT AND THE DISABLED-LANE DEFECT ARE FIXED.** The August probability sidecars can no longer load with the stale June calibrator: training now reserves the latest 20 OOF dates from calibration, persists calibrator/diagnostics with SHA-256 hashes of all sidecars and OOF data, and live inference fails closed on any generation mismatch. The active calibrator was rebuilt from 44,787 earlier rows / 64 dates, leaving 5,391 rows / 20 dates (`2026-04-27..2026-05-18`) outside its fit. STL/BLK are no longer silently removed after the dynamic policy approves them, the dedicated UNDER-only betslip path now consumes that same policy, and the card is capped at one pick per prop. Current executable policy is AST/STL/BLK UNDER-only; PTS/TRB/PRA/PR/PA/TOV remain disabled. On the reserved calibration period, the highest eligible AST/STL/BLK UNDER per prop/date went `51/60` (85.0%) versus 66.1% expected for a date/prop-matched random UNDER, paired lift +18.9pp (95% CI +9.9 to +28.0). This supports the ranking/abstention repair but is not a new-season or PrizePicks-standard-only claim. No regression/classifier/sidecar model changed. Full suite: `120 passed, 1 subtest passed`. See DEC-056 and `output/probability_under_selection_summary.json`.

**[2026-08-23, DEC-055] FIVE ACTIVE DATA/OPERATIONS DEFECTS ARE FIXED AND REGRESSION-LOCKED.** PrizePicks standard/goblin/demon/promo contracts are no longer medianed into synthetic lines; the exact 20.5/17.5 reproduction now preserves both offers and selects the real standard 20.5 line. Every daily modeling layer now receives one rolling active-season contract, paid sportsbook T-60 capture is PrizePicks-only by default and requires both an explicit CLI request plus the configuration spend gate, and unknown snapshot timestamps are rejected whenever a historical cutoff/known-time contract applies. The production audit now uses the real `result`/`direction` schema, excludes PUSH/DNP/pending rows, and fails closed instead of inferring direction from edge. The repaired audit ran on 1,528 persisted picks / 1,422 binary graded rows: PA beat same-row always-UNDER by +2.0pp; PTS, PR, and PRA lost; AST/TRB tied; STL/BLK were below the 50-pick floor. This is production pick-history evidence, not a newly frozen training holdout. Full suite: `114 passed, 1 subtest passed`. See DEC-055.

**[2026-08-14, DEC-054] ACTIVE PROP TRAINING/LIVE FEATURE CONTRACT IS NOW EXACT; LIVE STATE IS NO LONGER ONE GAME STALE.** A repository-wide selected-feature audit found that `nba_props.py` rebuilt 85 of the 99 active model inputs differently from the `FeaturePipeline` table used for training. The active advanced scorer now keeps the legacy builder only for the separately trained minutes model, then fail-closed overwrites every production-selected prop input with the exact shared `FeaturePipeline` implementation. On the complete real table, all `97/97` locally computed active fields match training across `73,531` aligned player-games at `atol=rtol=1e-9`; the remaining two selected fields (`opp_def_rtg_pt`, `opp_pace_pt`) are intentionally attached from the scheduled point-in-time matchup adapter. Zero drifted or missing selected fields remain.

The live frame also now appends an outcome-empty scheduled target row before shifted/rolling features are computed. That makes T+0 inference consume the latest completed game; the former `groupby.last()` path stopped on the last completed row whose `shift(1)` features excluded that same game. Cache identity now includes slate date, exact team-log coverage ignores only the synthetic unfinished row, and the 2025-26 player-feed ORB nulls cannot suppress the authoritative exact-team-log season. `scripts/audit_training_live_feature_parity.py` is a fail-closed reusable contract, and `scripts/retrain_updated_models.ps1` provides isolated retrain, OOF comparison, and explicit backed-up promotion. No model artifact was retrained or activated by this correctness change. Full suite: `103 passed`. See DEC-054.

**[2026-08-13, DEC-053] LIVE/TRAINING MISMATCH IN `{stat}_best_estimate`/`edge_vs_avg` FIXED (PTS/TRB/AST); NO MODEL RETRAINED.** The production points/rebounds/assists models were trained on one composite-feature formula (`feature_pipeline.py`) and scored live every day on a different, independently-written one (`nba_props.py`) — a train/serve mismatch on the #1 most important points feature. Added `nba_props._align_composite_features_to_training()`, called after `_build_advanced_features()` at both call sites, to overwrite the live values with the exact training formula once its real ingredients exist. On 81,065 real historical rows the raw feature diff was material for points (mean 0.376, 7.5% of rows off by >1.0) and small for rebounds/assists. Fed through the actual loaded production models, the same-row accuracy delta (OLD vs NEW formula) was PTS +0.07pp, TRB +0.22pp, AST −0.10pp — **none clear a 95% CI**, so this is recorded as a correctness fix, not a proven accuracy gain. New gap found and left open: `nba_props.py`'s own live `{stat}_defense_adj` (one of best_estimate's five inputs) still isn't pace-adjusted, unlike the training-side version DEC-051 already repaired — same root cause (two independently maintained feature pipelines), third instance found. No production model file was modified. Full suite `98 passed`. See DEC-053.

**[2026-08-13, DEC-052] RIGOROUS POINT-IN-TIME TEAM DEFENSE IS ACTIVE; THE MISSING-ORB BUILD WAS REPLACED.** Jake explicitly directed that the stronger opponent- and pace-adjusted defense path remain active, superseding DEC-051's no-promotion deployment state. Verification found that every one of the `28,358` 2025-26 player rows had null ORB/DRB, so the first build's player aggregation had inflated pace. Exact NBA team game logs now cover regular season, Play-In, and playoffs: `7,918` team rows / `3,959` games through `2026-06-13`, zero duplicate keys and zero required nulls. The rebuilt point-in-time table has zero chronology violations; its prop join is `73,531/73,531` with zero feature nulls. Pace means are credible at `101.41`, `101.81`, and `102.36` across the three seasons.

All nine exact-source bundles were retrained in `models/defense_rating_retrain_exact/` and 54 artifacts were atomically activated. Production feature selection now uses the rigorous `opp_def_rtg_pt` for STL and `opp_pace_pt` for BLK; no rigorous field survived selection for the other seven props, and none was forcibly injected. Previous active artifacts are recoverable from `models/backups/dec052_pre_exact_orb_activation/`; hashes are recorded in `output/defense_rating_exact_activation_manifest.json`. The same-row OOF audit remains a disclosure, not a deployment veto: no prop cleared the untouched-performance evidence standard, so no future-return claim is made. Live scheduled-opponent smoke passed, focused tests are `10/10`, and the full suite is `96/96`. See DEC-052.

**[2026-08-13, DEC-051] TEAM-DEFENSE DATA CONTRACT AND LIVE MATCHUP PLUMBING FIXED; CANDIDATES DID NOT CLEAR PROMOTION.** The rigorous `ppp_engine.fit_team_ratings()` math is now materialized point-in-time: one expanding current-season fit per target date, strictly prior games only, five-game trust floor, previous-season-final fallback, and league-average fallback only when no prior season exists. `data/pointintime_team_defense_ratings.csv` contains `7,532` team-game rows / `3,766` games through `2026-05-18`, with zero duplicate keys, required nulls, or chronology violations. The fail-closed prop adapter joined `73,531/73,531` rows and repaired the `27,329` newest-season rows whose source `opp` and `is_home` fields were blank.

The crude comparison path is also repaired in an isolated regenerated table: possessions now include ORB, allowed production is normalized per 100 possessions, all 30 teams are ranked before merging to players, rank 1 means best defense, and 2025-26 matchup identity is reconstructed from `GAME_ID`/`matchup`. Corrected opponent coverage is `100%`, pace-factor mean is `0.9999` (not `0.487`), ranks stay in `1..30`, and no opponent/date has conflicting ranks. `nba_props.py` now attaches a scheduled opponent's point-in-time features for any future model that requires them and fails closed instead of filling missing matchup context with zero. Production models do not require these fields and remain unchanged.

All nine candidates trained in `models/defense_rating_retrain`. `opp_def_tier_pt` survived only for STL; `opp_pace_pt` plus `opp_def_tier_pt` survived only for BLK; all three rigorous fields were rejected for the other seven props. The reproducible q50 same-row audit covers `36,009` identical non-push real-line OOF rows across up to `80` dates. No candidate has a positive 95% interval over production and always-UNDER. STL/BLK made exactly the same all-UNDER decisions as the baseline on these rows despite selecting the features. **No candidate promoted.** The production artifact digest remained `98945f18...51042`; full verification is `95 passed`. See DEC-051 and `output/defense_rating_same_row_oof_summary.json`.

**[2026-08-13, DEC-050] AVAILABILITY/MINUTES PROP RETRAIN DID NOT CLEAR PROMOTION FOR ANY PROP; PRODUCTION UNCHANGED.** All nine candidates were retrained in the isolated `models/availability_retrain/` directory per the DEC-049 authorization. Availability/minutes features survived per-prop feature selection only for STL (conditional + expected minutes) and PRA (conditional minutes only); all other props retained none. On a same-row OOF comparison against production, no candidate-vs-production change had a positive 95% confidence interval (largest moves: TOV −0.90pp, PTS +0.59pp), and 7 of 9 candidates lost to the same-row naive always-UNDER baseline. Verified by mtime and content: `models/*_advanced.pkl` and `models/selected_features_advanced.json` (production, last changed 2026-06-07) are untouched — none of the three availability/minutes fields appear there. **The experiment is closed for PTS, TRB, AST, PR, PA, BLK, TOV.** STL is the only follow-up: it beat both production (+0.15pp) and always-UNDER on this comparison, but the margin is too small to promote without an isolated STL-only ablation (with vs. without the two minutes features, identical rows/folds) plus a reserved future holdout the candidate never touched. See DEC-050.

**[2026-08-13, DEC-049] OOF AVAILABILITY/MINUTES FEATURES ARE NOW WIRED INTO PROP RETRAINING.** `scripts/build_prop_availability_training_data.py` performs a fail-closed one-to-one join on `(GAME_ID, PLAYER_ID, team)`, checks date/minutes reconciliation and `model_train_end < game_date`, and materializes `data/model_data_availability_advanced.csv`. The artifact has `52,775` matched appearance rows from `2024-03-08` through `2026-05-18`, zero duplicate keys, zero feature nulls, and zero chronology failures. OOF-window coverage is `52,822/58,435` (`90.39%`) overall and `44,032/45,685` (`96.38%`) for the `mp_l10 >= 15` likely-prop proxy. The adapter explicitly excludes `15,096` pre-OOF rows, `5,613` unmatched fringe/roster rows, and `47` zero-minute non-appearances.

`train_advanced_models.py` now registers `availability_probability`, `conditional_minutes_prediction`, and `expected_minutes_prediction` as candidate features. `--require-availability-features` makes availability-aware retraining fail if the merged artifact, chronology metadata, ranges, or product identity are invalid. An isolated STL end-to-end smoke retrain completed on all `52,775` rows with the contract verified; conditional and expected minutes survived its 40-feature cap, while standalone availability probability did not. This is a pipeline smoke result, not a model-promotion result. Production models were not overwritten and live use remains disabled.

**[2026-08-13, DEC-048] AVAILABILITY/MINUTES MODEL BUILT AND VALIDATED; READY FOR PROP RETRAINING, NOT LIVE USE.** The new hurdle system reconstructs `98,217` pregame player/team/game candidates from prior same-team appearances, including `26,704` candidate non-appearances that the old appeared-player-only minutes model could not learn. It uses 29 strictly pregame features to predict `P(play)` and minutes conditional on playing. Four expanding development folds selected histogram gradient boosting for both components; every acceptance metric improved in all `4/4` folds. On the latest-season model holdout (`2025-26`, `35,444` candidates / `193` dates), availability AUC is `0.886` and Brier improves `0.1462 → 0.1115`; conditional-minutes MAE improves `5.028 → 4.788`; fixed-0.50-gated all-candidate MAE improves `6.157 → 5.903`; soft expected-minutes RMSE improves `9.878 → 8.728`. All four paired daily gains have positive 95% date-clustered bootstrap intervals. The likely-prop-pool proxy (`n=19,398`) improves on all four measures as well.

**The next model task is now authorized and technically safe:** retrain the prop models using `data/availability_minutes_oof_predictions.csv`. It contains `74,965` predictions from March 8, 2024 through May 18, 2026 with zero cases where `model_train_end >= game_date`; using the full-fit model on historical training rows would be leakage and is forbidden. `models/availability_minutes_model.pkl` is trained for future scoring but `enabled_for_live=false` until the surrounding prop stack clears its own chronological comparison. The historical candidate contract cannot identify players before their second appearance for a team or after more than 45 days out, and it lacks a transaction feed; live injury/roster data remains necessary.

**[2026-08-13, DEC-047] THE 13,436-CREDIT RESERVE IS NOW ASSIGNED TO PAIRED PRIZEPICKS/SPORTSBOOK T-60 CAPTURE.** The closing watcher still saves PrizePicks directly at T-60 and T-5, but now the daily runner enables a paid sportsbook call only after a real PrizePicks T-60 event is present and uniquely matched. Current event lookup is free; the live event-odds call has a conservative ceiling of nine credits (one per requested market in `us`), not the 90-credit historical ceiling. The paired path has its own raw/parsed stores and durable lifetime checkpoint, saves both prices before completion, blocks ambiguous event matches, never spends at T-5, and refuses the next call if it could exceed `13,436`. A free authenticated check returned HTTP 200, `x-requests-last=0`, and `x-requests-remaining=13,436`. ESPN shows no NBA slate on August 13; the Odds API currently lists 34 future events beginning October 20, so no paid observation exists to retrieve today.

**This does not mean the entire model is useless.** The current model-edge rank went `56/100` on the untouched sportsbook holdout, above matched random (`51.94%`) but below the accepted T-60 no-vig overlay (`61/100`). The result says the existing model is not the best selector by itself and that consensus line, dispersion, T-5 probability, movement, and availability did not add incremental selection value in this test. It does not erase the model's projection/minutes/eligibility functions. Production remains unchanged until the accepted sportsbook feature transfers on paired PrizePicks lines.

**[2026-08-13, DEC-046] THE DATA-SCALE BLOCKER IS RESOLVED FOR SPORTSBOOK FEATURES.** A new paid expansion covers `35` completed dates (`2026-02-23` through `2026-03-29`), `264` games, and all `528/528` requested T-60/T-5 event snapshots. It added `314,162` sportsbook rows; `314,054` have paired prices (`99.97%`), physical-key duplicates are zero, date mismatches are zero, post-request/post-tip snapshots are zero, and box-score join coverage is `94.81%`. The expansion feature table has `25,801` rows, `23,622` T-60-to-T-5 movements, `434` withdrawals, and `24,462` joined outcomes. Combined with the original panel, the frozen sportsbook table now has `42,790` rows across `80` dates.

**T-60 no-vig probability is the first feature to clear a properly powered chronological test.** On `54` development dates, it improved top-five validation hit rate by `+21.9pp`, was positive in `3/4` expanding folds, improved mean AUC by `+0.0374`, and covered `24` validation dates / `119` selections. On the untouched `20`-date / `100`-selection future holdout, the T-60 overlay went `61/100` versus current model-edge ranking `56/100`, date/prop-matched random `51.94%`, and the full eligible always-UNDER pool `52.43%`; the one-sided matched-random probability is `3.45%`. The first ten holdout dates were `58%` and the last ten were `64%`. T-5 no-vig passed development selection but failed the holdout (`52/100` versus current `55/100`), so it is not accepted. Movement initially screened positively at T-5 but added nothing after no-vig and was rejected; consensus line, dispersion, and availability also failed the final feature-selection sequence.

**Source boundary:** the accepted T-60 result predicts UNDER against sportsbook consensus, while the live card is placed at PrizePicks. The T-60 sportsbook shadow artifact therefore remains `enabled_for_production=false` until a paired forward PrizePicks/book test confirms transfer to the actual line being bet. This is no longer a sample-size failure: it is one precisely defined source-transfer test. Direct PrizePicks T-60 capture is already installed.

Expansion checkpoint spend is `42,129`; the Odds API reports `13,436` credits remaining. This is `8,555` lower than `64,120 - 42,129`. Two timed-out shell invocations may have left overlapping executor processes and caused uncheckpointed duplicate requests; that cause is strongly suggested by the later file contention but remains a hypothesis without the API request dashboard. A cross-platform PID lock and Windows atomic-replace retry are now installed so paid executors cannot overlap again. No expansion row was lost. Final verification: Python compilation passed, the 39 focused acquisition/evaluation tests passed, and the complete suite passed `72/72` tests.

**[2026-08-12, DEC-045] WALK-FORWARD TEST COMPLETE; NO CLV FEATURE PROMOTED.** The frozen table was evaluated chronologically on the same eligible PrizePicks pool used by the current card: PTS/TRB/AST, UNDER, `projection_type=standard`, non-promo, archived model edge at least 5%, and graded non-push outcomes. Exact archived projection metadata leaves only `889` eligible T-60 rows across `8` dates and `902` eligible T-5 rows across `9` dates. The six requested feature groups were tested one at a time with expanding walk-forward folds; none cleared the predeclared evidence gate (`12` validation dates and `60` validation card rows, plus positive lift/AUC conditions). The apparent T-60 PP-gap lift is hypothesis-generating only: `+26.7pp` across just `3` validation dates / `15` picks.

Untouched future comparison did not justify replacing the current model. At T-60, the research-only base shadow overlay went `5/9` (`55.6%`) versus current model-edge ranking `4/9` (`44.4%`) and matched random `51.0%` / `48.3%`; the two-date always-UNDER eligible pool was `54/82` (`65.9%`). At T-5, both shadow and current ranking went `4/10` (`40.0%`) versus matched random `46.5%` / `45.0%`; the three-date pool was again `54/82` (`65.9%`). Both overlay artifacts are explicitly disabled for production. Reproduce with `python scripts/evaluate_clv_walk_forward.py`.

**Forward collection is now the active improvement path.** `scripts/capture_prizepicks_closing_snapshots.py` saves direct PrizePicks raw payloads and per-event parsed snapshots in T-60 (`54-66` minutes) and T-5 (`1-9` minutes) windows, checkpoints each event/window, and appends source-tagged observations to `historical_lines.csv`. `run_daily.py` launches the watcher after the morning PrizePicks snapshot. It makes zero Odds API calls. The live card is restored to PrizePicks, source mismatches force an empty official card, implicit live Odds fallback is disabled, the API key is environment-only, and all `64,120` remaining Odds API credits stay reserved until a named missing observation emerges from testing. Final verification: Python compilation passed and the complete suite passed `68/68` tests.

**[2026-08-12, DEC-043/044] PAID CLV PANEL ACQUIRED, REPAIRED, AUDITED, AND MATERIALIZED INTO FEATURES.**

The line store is lossless at `(game_date, player_norm, prop, line, source, snapshot_ts)`. The paid canary proved `line` must be in the physical key because books return alternate lines; without it, 248 of 634 canary rows would have been discarded. The master now has `412,328` rows and zero corrected-key duplicates.

Acquisition completed all `474` evidence-safe manifest rows under the hard cap: `469` returned data and `5` are genuine empty-market observations. Actual spend was `35,880`; the API reports `64,120` remaining. The paid slice contains `253,725` rows, `253,680` with paired over/under prices (`99.98%`), zero post-request/post-tip snapshots, zero date mismatches after UTC-date repair, and `95.28%` box-score join coverage at unique player/prop/date grain. TOV was never returned; blind retries are not warranted.

`data/clv_features_2026_live.csv` now carries `16,989` player/prop/game rows, including `15,391` T-60-to-T-5 line movements, `7,920` decision-to-close movements, no-vig consensus, book dispersion, PrizePicks-minus-consensus gaps, and withdrawal flags. The gap dose-response survives at scale (`n=10,943`): UNDER is `49.5%` when PP is below consensus and `67.1%` when PP is at least one unit higher.

**DEC-040 market-wide correction:** at 30+ minutes, paid T-5 sportsbook consensus UNDER is only `51.9%` for STL (`n=621`) and `58.7%` for BLK (`n=965`). STL softness is not market-wide at scale; BLK is much weaker than the prior `n=40` estimate. The large PrizePicks rates are primarily source-specific shading.

**[2026-08-12, DEC-042] THE INCOMPLETE AUDIT ALIAS MAP IS FIXED.** The corrected nine-market audit has `37,935` graded lines, not `21,113`. All-prop always-UNDER is `58.51%`; at 30+ minutes it is `47.74%`. In the live window, the naive baseline is `61.67%` (`n=17,129`) versus the model's `68.21%` (`n=173`), a `+6.53pp` point estimate; the baseline still lies inside the model's 95% interval, so selection edge remains unestablished. The earlier aggregate figures excluded PA/PR/PRA/TOV. Per-stat findings for PTS/TRB/AST/STL/BLK are unchanged.

**[2026-08-12, DEC-041] THE THREE-SNAPSHOT REQUEST MANIFEST IS BUILT AND AUDITED.** `data/clv_snapshot_manifest_2026_live.csv` contains exactly `561` rows: 187 games x decision/T-60/T-5. Scheduled tips reconcile one-to-one for all 187 local games. All `374` T-60/T-5 rows are ready. Of the 187 decision rows, `100` have a provable pre-tip archived final-card timestamp; `46` have no final-card archive and `41` resolve only at/after that game's tip, so those `87` rows are explicitly blocked rather than guessed. The currently executable ceiling is `42,780` credits (`42,660` event odds + `120` grouped event-list lookups); the theoretical complete panel remains `50,625`. No Odds API request was sent while building the manifest. Reproduce: `python scripts/build_clv_snapshot_manifest.py`.

**[2026-08-12, DEC-040] Per-stat edge map now resolves into THREE regimes, not one edge.** Reproduce: `python scripts/market_edge_map.py`.

**[RESOLVED BY DEC-043]** Prices are populated on `253,725` paid rows, with `99.98%` paired-price completeness. Older DEC-040 text below is retained as the pre-panel finding and is corrected by DEC-044 where the larger sample disagrees.

**PrizePicks shades exactly three stats** (median gap vs the book, `n=973` overlapping keys, `2026-04-17`→`2026-05-18`): AST **+1.00** (higher in 84.0%), TRB **+0.75** (77.8%), BLK **+0.50** (65.5%). PA/PR/PRA/PTS/STL have median gap **0.00** — PrizePicks prices those at the book, and matches it *exactly* on STL in `62.2%` of keys.

**The PrizePicks-minus-book gap predicts UNDER outcomes monotonically** — the most actionable signal found to date, computable pre-tip and independent of minutes forecasting:

| Gap bucket | n | UNDER win rate |
|---|---|---|
| PP below book | 98 | 53.1% |
| equal | 317 | 59.0% |
| PP +0 to +0.5 | 152 | 68.4% |
| PP +0.5 to +1 | 239 | 66.5% |
| PP +1 or more | 120 | **73.3%** |

**Three regimes** (naive always-UNDER, 30+ minutes, by book):

| Stat | all | 30+ min | PP 30+ | Book 30+ | Regime |
|---|---|---|---|---|---|
| BLK | 75.9% | **71.3%** | 72.9% | 75.0% (`n=40`) | **market-wide** |
| STL | 70.4% | **64.6%** | 67.1% | 50.0% (`n=30`) | **market-wide** |
| AST | 63.2% | 53.3% | 59.4% | 53.2% | PrizePicks shading only |
| TRB | 60.0% | 49.1% | 53.7% | 44.4% | PrizePicks shading only |
| TOV | 52.1% | 47.9% | 47.9% | — | dead |
| PA | 54.2% | 42.0% | 45.0% | 43.0% | **inverted** |
| PTS | 53.1% | 40.5% | 42.9% | 38.6% | **inverted** |
| PR | 53.8% | 39.9% | 43.8% | 45.8% | **inverted** |
| PRA | 53.6% | 38.0% | 41.4% | 41.8% | **inverted** |

**This corrects DEC-038.** DEC-038 said the edge lives *entirely* inside PrizePicks' pricing. That is wrong for STL/BLK — PrizePicks does not shade steals at all, yet STL UNDER still returns `64.6%` at full minutes, and BLK runs `71–75%` at **both** books. Those two are a market-wide mispricing of low-count defensive stats and are the most durable component. DEC-038 remains correct for AST/TRB. ⚠️ Book-side samples are small (`BLK n=40`, `STL n=30`) — suggestive, not established.

**Archived line movement is not usable**: across `32,485` multi-snapshot keys, lines moving *down* gave `55.8%` unders vs *up* `60.9%` vs unchanged `61.6%` — backwards. The archive holds repeated morning captures, not real pre-tip movement.

---

**[2026-08-11] STRATEGY DECISION PENDING — the ML layer does not beat a naive always-UNDER baseline. See DEC-036.**

The system is dormant (last artifacts `2026-05-19`; season over). A full retrospective audit on 2026-08-11 verified the machinery and invalidated the headline metric. Nothing was changed in the pipeline; the strategy response is Jake's call.

Audit outcome in four lines:

1. **Grading and forward-logging are trustworthy.** 180/180 live-era pick results independently recomputed from `data/nba_data.csv` matched what the pipeline logged. DNPs correctly excluded. `output/archive/` holds real pre-game timestamped snapshots. Do not rebuild this part.
2. **The documented "53–57%" win rate is not the live system.** `picks_history.csv` blends a *backfilled* era (2026-03-17→03-29, `n=1,249`, 54.0%) with the *real live* era (2026-03-30→05-18, `n=173`, **68.2%**, 95% CI [60.9%, 74.7%]). March rows have null `pick_source`/`line_source`/`prob_source`, null `graded_at`, and archive folders containing only backfilled betslip files.
3. **The models still have no established selection edge over "bet UNDER on everything"** *(aggregate reading corrected by DEC-042 and source interpretation revised by DEC-038)*. Corrected nine-market naive always-UNDER over the live window: **61.67%** (`n=17,129`) vs the model's 68.21% (`n=173`). The `+6.53pp` point estimate is larger than DEC-036 reported, but the baseline remains inside the model's 95% CI. Per-prop, `avg_bet_accuracy` is *below* naive for PTS (−2.0), TRB (−0.9), and AST (−1.5); that training-time result is unchanged.
4. **Most of the UNDER edge is a minutes bet.** UNDER win rate falls monotonically with minutes played: 95.8% (<10 min) → 50.2% (35+ min). Controlling for 30+ minutes drops the all-prop UNDER rate to **53.2%**. Only **STL (64.0%)** and **BLK (71.0%)** survive minutes control — that structural right-skew mispricing is the system's one durable edge.

**[2026-08-11, DEC-038] The May collapse was a market switch, not model decay.** The live card changed from PrizePicks lines to FanDuel lines around `2026-05-11` (`CONFIG['live_sportsbook_key'] = 'fanduel'`). Win rate steps at exactly that seam: 82.1% (week of 05-04, PrizePicks) → **44.4%** (05-11, FanDuel) → **40.0%** (05-18, FanDuel). Cause: the edge is a PrizePicks pricing artifact. Naive always-UNDER is **67.1%** on PrizePicks lines but **52.0%** on FanDuel lines (44.3% at 30+ minutes, i.e. FanDuel favours OVER). An UNDER-only engine (DEC-034) aimed at an efficiently-priced book runs backwards.

**[2026-08-11, DEC-038] Split by `line_source`, the live record reads differently than the DEC-036 aggregate:**

| Subset | Picks | Win rate | 95% CI | Naive baseline | Verdict |
|---|---|---|---|---|---|
| On PrizePicks lines | 95 | **80.0%** | [70.9%, 86.8%] | 67.1% | **beats baseline** |
| On FanDuel lines | 23 | 43.5% | [25.6%, 63.2%] | 52.0% | not distinguishable |

⚠️ Do **not** promote the 80% to a headline figure: it is a **post-hoc subgroup** (split chosen after seeing outcomes) and compares a confidence-filtered selection against an unfiltered baseline. It raises the priority of a proper selection-vs-random test on the PrizePicks pool; it does not substitute for one. DEC-036's training-time findings are unaffected.

Realized live-era money (verified, real): 124 slips, `$2,480` wagered → `$5,740` returned, **`+$3,260` / +131% ROI**. Caveat: 7 weeks, postseason only, 173 picks. Per DEC-038 the profit was earned **entirely on PrizePicks lines** — `betslips_history.csv` ends `2026-05-10`, the day before the FanDuel era began, so the losing stretch was logged but never wagered.

Open decision for Jake (DEC-036 item 5): (a) narrow to STL/BLK structural unders + an explicit minutes/availability model, dropping the composite stat models; (b) keep current behavior and treat the ML layer as a redundant filter; (c) rebuild around CLV measurement first. Highest-value unanswered question is **CLV** — `clv_corr` is `0.074`, so there is still no evidence any pick beats a closing line.

Reproduce the whole audit: `python scripts/audit_naive_under_baseline.py`

---

**[SUPERSEDED 2026-04-13] Build enough true-labeled market history to audit standard-line edge cleanly.**

Shipped that day:

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
- `python prizepicks_scraper.py` succeeded again on 2026-04-13 at `09:02:26` and appended a much larger true-labeled PrizePicks snapshot: `655` validated rows across `2026-04-14`, `2026-04-15`, and `2026-04-18`.
- The active slate is no longer legacy data: `_select_active_slate_lines()` now resolves to `2026-04-14` with `257` true-labeled rows (`234 demon`, `14 standard`, `9 goblin`, `0 promo`).
- The remaining limitation is not schema materialization anymore; it is graded sample depth for a clean standard-line hit-rate audit.

## Current Branch / Working Scope
- Branch: `master`
- Scope: full pipeline, daily prediction, grading, and betslip construction
- Owner: Jake
- Last updated: 2026-04-15

## What Is Working

- Full daily pipeline last confirmed end-to-end on `2026-04-15` (log output above); three critical Step 6/8 bugs fixed in this session (DEC-032, DEC-033)
- **UNDER-only strategy locked (DEC-034)**: fallback policy now hard-blocks OVER for pts/pra/pr/pa/stl/blk; dynamic policy adds matching `_over_hardblocked` gate; `allow_structural_micro_props=True` re-enables STL/BLK structural UNDER picks (73%/78% training hit rate). Only TRB/AST/TOV OVER remain mathematically eligible.
- **Absence threshold (DEC-032)**: `detect_recent_absences()` now requires missed most-recent game before flagging; load-managed stars (Curry, Turner) no longer contaminate absence list or trigger luck suppression
- **Step 8 dedup + name normalization (DEC-033)**: all layer sources deduped before merge; Layers 3/4/5 join on `player_norm` (NFKD ASCII); diacritic players (Jokić, Dončić, etc.) now get correct features instead of zeros
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
  - snapshot timestamp: `2026-04-13 09:02:26`
  - rows: `655`
  - slate dates: `2026-04-14`, `2026-04-15`, `2026-04-18`
  - market mix: `602 demon`, `35 standard`, `18 goblin`, `0 promo`
- First truly labeled active slate summary:
  - active slate date: `2026-04-14`
  - active rows: `257`
  - market mix: `234 demon`, `14 standard`, `9 goblin`, `0 promo`
- `output/picks_history.csv` now verifies the `2026-04-12` card went `5W / 0L / 0P`
- All-time graded pick record is now `712W / 597L / 21P` (`53.53%` win rate on graded picks)
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
- **[Corrected 2026-08-12, DEC-042] The ML layer is not distinguishable from the complete nine-market naive always-UNDER rule.** Live-era card 68.21% (`n=173`) vs naive always-UNDER 61.67% (`n=17,129`) on the same window; the baseline remains inside the model's 95% CI. Per-prop `avg_bet_accuracy` is *below* naive for PTS (−2.0), TRB (−0.9), AST (−1.5).
- **[2026-08-11, DEC-036] Most of the UNDER edge is a minutes/availability effect, not stat projection.** UNDER win rate 95.8% (<10 min) → 50.2% (35+ min); at 30+ minutes the all-prop UNDER rate is 53.2%. Only STL (64.0%) and BLK (71.0%) survive minutes control
- **[2026-08-11, DEC-036] PTS UNDER at 30+ minutes hits 40.3%** — directionally backwards, not merely weak. Keep PTS disabled
- **[SUPERSEDED 2026-08-13 by DEC-046] No CLV evidence exists.** T-60 sportsbook no-vig now cleared chronological validation and a 100-selection holdout. Transfer to the actual PrizePicks line remains open and is the purpose of the paired forward collector.
- **[SUPERSEDED 2026-08-12 by DEC-039] DEC-037 line-store defect.** The physical store is now lossless and the historical readers materialize a one-row preferred view at read time.
- **[SUPERSEDED 2026-08-12 by DEC-039] DEC-037 uncapped-backfill defect.** The paid path now requires an explicit cap and checkpoints each event atomically; the legacy implementation raises immediately.
- **[2026-08-11, DEC-037] ~19,300 Odds API credits unaccounted for** (`19,364` remaining `2026-05-13 13:09` → `63` remaining `2026-05-16 16:32`). Not from the daily pipeline: no `run_20260514.log`, `morning_task.log` skips `2026-05-13`→`2026-05-18`, `run_20260515.log` made zero Odds API calls. `source='odds_api_backfill'` appears nowhere in `data/` or `output/`. Cause unproven — settle it from the Odds API usage dashboard
- **[2026-08-11] System is dormant** — last pipeline artifacts are `2026-05-19`; season over. Nothing has run since
- End-to-end live verification of the new `team_pending` fallback is still outstanding on a real pending-team slate; today's saved projection artifact produced `0` fallback rows
- Standard-line-only edge analysis is still open; most older historical rows were backfilled to `projection_type='standard'`, so the newly verified `5/5` on `2026-04-12` is still not eligible as clean standard-line evidence
- The first truly labeled active slate (`2026-04-14`) is heavily demon-skewed (`234/257` rows), so standard-line sample growth is still slow
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
- **Verified 2026-08-11 (DEC-036) — grading integrity is sound:** 180/180 live-era pick results independently recomputed from `data/nba_data.csv` matched the logged results; DNPs (`n=80`) correctly excluded from denominators; `output/archive/` holds genuine intraday pre-game snapshots. One cosmetic mislabel: Donovan Clingan PTS OVER 14.0 / actual 14.0 logged `LOSS`, should be `PUSH` (2026-03-25)
- **Verified 2026-08-11 (DEC-036) — true pick record splits into two eras.** Backfilled 2026-03-17→03-29: `n=1,249`, 54.0%. Live 2026-03-30→05-18: `n=173`, **68.2%**, 95% CI [60.9%, 74.7%]. The old "53–57%" figure blended the two and describes neither
- **Verified 2026-08-11 (DEC-036) — realized live-era money:** 124 slips, `$2,480` wagered → `$5,740` returned, **`+$3,260` / +131% ROI**. Backfilled-era slips lost `−$140` (`n=31`), consistent with them not being real
- **Corrected 2026-08-12 (DEC-042) — complete naive always-UNDER benchmark** (`n=37,935` graded lines): PTS 52.8%, TRB 59.6%, AST 63.1%, STL 70.1%, BLK 75.9%, PA 54.1%, PR 53.5%, PRA 53.4%, TOV 52.1%. Minutes-controlled (30+ min): PTS 40.2%, TRB 48.7%, AST 53.1%, STL 64.1%, BLK 71.1%, PA 41.8%, PR 39.4%, PRA 37.8%, TOV 47.9%. **Any future edge claim must clear its same-stat benchmark, not 50%.**
- Cross-target UNDER edge hit rate: `62.3%` (`n=16,436`) — ⚠️ DEC-036: never compared against naive always-UNDER; not evidence of model edge on its own
- Cross-target OVER edge hit rate: `48.7%` (`n=9,929`)
- Retrained under-only evidence that drove DEC-019 / DEC-034 betslip policy — ⚠️ DEC-036: each figure is ~equal to the naive always-UNDER rate for that prop, so it cannot distinguish a model from a constant. The resulting policy is still directionally right, but the justification was not valid evidence:
  - `AST UNDER`: `65.4%` hit, `n=2172` (naive: `62.9%`)
  - `TRB UNDER`: `63.1%` hit, `n=2278` (naive: `59.4%`)
  - `STL UNDER`: `74.0%` hit, `n=2034` (naive: `70.0%`)
  - `BLK UNDER`: `79.2%` hit, `n=1756` (naive: `75.8%`)
- Latest under-only slip backtest summary on current logic:
  - `2-pick`: `9` slips, `55.56%` win rate, `0.4444` ROI
  - `3-pick`: `9` slips, `55.56%` win rate, `1.2222` ROI

## Active Risks
- **[DEC-036] Do not extrapolate the `+$3,260` live result.** It is 7 weeks, 173 picks, postseason only, and mostly reproducible by a baseline requiring no model
- **[DEC-036] Do not cite `avg_bet_accuracy` from `models/training_results.json` in isolation** as evidence a model works — for these props it tracks the naive always-UNDER rate
- **[DEC-036] The surviving edge is narrow: STL and BLK structural unders.** It rests on right-skew mispricing (mean BLK line `0.975` vs mean actual `0.698`; mean STL line `1.493` vs mean actual `1.228`). If PrizePicks reprices those two markets, the edge is gone
- **[RESOLVED BY DEC-045] `CONFIG['live_sportsbook_key']` is live-money-critical.** It is restored to `prizepicks`, source mismatches now force an empty official card, and the implicit paid fallback is disabled.
- **[DEC-038] Never run the UNDER-only strategy against sportsbook lines.** Sportsbook prices are a fair-value *reference* for CLV, not a betting target. FanDuel lines favour OVER at 30+ minutes
- **[DEC-038] Report the live record split by `line_source`.** The blended figure misleads in both directions — it understated the PrizePicks-era result and hid the FanDuel-era collapse
- **[DEC-039] Never run a paid backfill without an explicit cap and a snapshot manifest.** The generic CLI is safe-by-default, but the planned CLV pull still needs exact decision/60-minute/close timestamps and a `55,000`-credit ceiling.
- **[DEC-039] Never join raw `historical_lines.csv` directly to player-games.** It intentionally contains multiple sources and snapshots; use `_select_preferred_line_rows()` or a purpose-specific as-of view.
- **[DEC-039] Do not spend the remaining ~45k credits on dense 5-minute polling.** Three targeted snapshots over the 187-game live window answer the CLV question for about 50.5k credits; dense polling would exceed the tier without a commensurate modeling benefit.
- PrizePicks scraping can silently fall back to Odds API lines; always check the `line_source` warning in logs
- Odds API fallback and legacy line history default to `projection_type='standard'`, so the market gate is most trustworthy on fresh PrizePicks-backed saves
- `volume_fill` picks are capped at `63%` confidence and excluded from betslips, but they still appear in `picks_latest.csv`
- `player_profiles.csv` can still carry stale official team assignments after trades; the local-team mismatch filter prevents those rows from contaminating `detected_absences.csv`, but it is not a substitute for a live roster or injury feed
- The new Step 6 live-injury path depends on external official/ESPN availability; if both sources fail, the pipeline falls back to the absence proxy only
- Step 8 now marks unresolved pending-team rows explicitly, but it still needs live-slate monitoring on the next day with a pending official report
- Standard-line OVER edge may look materially different once old ungated history is separated from new gated market metadata

## Recent Changes

- **[2026-08-13] Availability/minutes OOF-to-prop adapter completed (DEC-049)** - added the one-to-one historical adapter, registered all three pregame features, added a fail-closed trainer contract, materialized and hashed the 52,775-row training table, added five integration tests, and completed an isolated STL smoke retrain without touching production models. Full repository verification: `86 passed`.
- **[2026-08-13] Availability/minutes hurdle model validated (DEC-048)** - added the pregame candidate builder, two-stage training pipeline, four expanding date folds, latest-season holdout, strong rolling/last-status baselines, 5,000-draw date-clustered bootstrap audit, strict OOF materialization, deployment scorer, five leakage/contract tests, an executed notebook, and a durable report. Audit result: `ready_for_prop_retraining`; live remains disabled. Full repository verification: `81 passed`.
- **[2026-08-13] Paired forward T-60 collector enabled (DEC-047)** - `run_daily.py` now launches the PrizePicks watcher with paired sportsbook T-60 capture under a durable `13,436`-credit lifetime ceiling. The collector retains the PrizePicks game description for exact event matching, saves free event lookup/raw sportsbook/parsed sportsbook artifacts separately, records actual header-reported cost, blocks terminal 4xx retries, and makes no paid request at T-5. Free account verification: HTTP 200, 34 future NBA events, 0-credit call, 13,436 remaining. Focused tests: 6/6 pass; complete suite: 76/76 pass.
- **[2026-08-12] Lossless line-store and safe-backfill repair (DEC-039)** - changed the physical key to `(game_date, player_norm, prop, source, snapshot_ts)`, moved source priority to read time, hardened every historical-line consumer against join fanout, retained bookmaker/event/commence/over-price/under-price metadata, added mandatory credit caps plus per-event atomic checkpoints, and recovered all usable local sources. Materialized store: `158,603` raw rows, `46,824` preferred rows, `3,357` Odds/FanDuel rows, `907` Odds/FanDuel keys joining to box scores. No paid request was sent. Targeted tests: 16/16 pass.
- **[2026-08-11] Retrospective audit (DEC-036)** - full-directory examination of the dormant system. Verified grading integrity (180/180 against raw box scores), split `picks_history.csv` into its backfilled vs live eras, and computed the naive always-UNDER baseline for the first time. Result: the ML layer is not distinguishable from always-UNDER, and most of the observed edge is a minutes/availability effect. Added `scripts/audit_naive_under_baseline.py` (read-only, reproduces every figure). **No pipeline, model, gate, or policy code was changed** — the strategy response is an open decision for Jake.
- **[2026-08-11] May decline diagnosed as a market switch (DEC-038)** - splitting the live era by `line_source` showed the collapse is coincident with `CONFIG['live_sportsbook_key'] = 'fanduel'` taking over the live card around `2026-05-11`, not with any model or feature change. On PrizePicks lines the card went `80.0%` (`n=95`, 95% CI [70.9%, 86.8%]) against a `67.1%` naive baseline; on FanDuel lines `43.5%` (`n=23`) against `52.0%`. Revises DEC-036's aggregate reading; DEC-036's training-time findings stand. `betslips_history.csv` ends `2026-05-10`, so the losing stretch was never wagered. **No code or config changed.**
- **[SUPERSEDED BY DEC-042] [2026-08-11] `scripts/audit_naive_under_baseline.py` line selection hardened** - source-priority selection itself remains valid, but the quoted aggregates covered only five markets because the alias/formula contract silently omitted PA/PR/PRA/TOV. DEC-042 carries the corrected nine-market figures.
- **[2026-08-11] Line-store and backfill defects traced (DEC-037)** - follow-up to the audit, asking why only `259` `odds_api` rows survived in `historical_lines.csv`. Found (1) `_merge_historical_lines()` dedups to one row per `(game_date, player_norm, prop)` with the sportsbook ranked below PrizePicks, so ~`1,200` keys carry the PrizePicks number where an Odds API price was overwritten; (2) `backfill_historical_lines()` has no per-day checkpoint and unguarded JSON access, so a mid-loop failure discards every credit spent; (3) ~`19,300` credits were consumed `2026-05-13`→`2026-05-16` outside the daily pipeline with nothing written to disk (cause unproven — needs the Odds API dashboard). **No code changed**; schema fix is an open decision.
- **[2026-04-13] Verified 5/5 card** - `output/picks_history.csv` now shows the full `2026-04-12` card graded `5W / 0L / 0P`: Max Christie `AST UNDER 3.0`, Julian Champagnie `TRB UNDER 7.5`, Max Christie `TRB UNDER 4.0`, Luguentz Dort `TRB UNDER 4.0`, and Davion Mitchell `AST UNDER 7.5`.
- **[2026-04-13] First truly labeled active slate** - `python prizepicks_scraper.py` pulled `655` true-labeled rows across `2026-04-14` / `2026-04-15` / `2026-04-18`; `_select_active_slate_lines()` now resolves to a genuinely labeled `2026-04-14` slate (`234 demon`, `14 standard`, `9 goblin`).
- **[2026-04-13] New live card from labeled board** - `python nba_props.py predict` used the labeled `2026-04-14` PrizePicks slate and wrote a `5`-pick card; all published picks are `demon` UNDERs and both logged betslips are 2-pick slips.
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
- **[CRITICAL] True-labeled sample depth** - the newest verified PrizePicks snapshot now has `655` true-labeled rows, but only `35` of them are `standard` and none of those rows are graded yet. That is still not enough to trust a full directional standard-line audit.
- PrizePicks `403/429`

## Next 3 Highest-Priority Steps

1. **Accumulate more true-labeled PrizePicks history** - keep running the live save path so `historical_lines.csv` contains enough real `standard/goblin/demon` rows to support a meaningful edge audit.
2. **Audit standard-line OVER/UNDER edge** - once enough true-labeled rows are graded, recompute hit rates on `projection_type='standard'` and `is_promo=False` only.
3. **Monitor demon-heavy board behavior** - the first labeled active slate is overwhelmingly `demon`; track whether the current UNDER-heavy policy remains robust when the live board is mostly less-only lines.

## Validation Snapshot
Last verification performed: 2026-04-13
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
- `python -c "import pandas as pd; df=pd.read_csv('output/picks_history.csv'); sub=df[df['game_date'].astype(str).eq('2026-04-12')]; print(sub[['player','prop','direction','line','projection_type','is_promo','result','actual','line_source']].to_string(index=False)); print((sub['result']=='WIN').sum(), len(sub))"` -> OK (`5` wins out of `5`)
- `python prizepicks_scraper.py` -> OK; direct API succeeded on `minimal-league-only / variant 0`, parsed `664` validated NBA lines across `2026-04-14`, `2026-04-15`, and `2026-04-18`, and rewrote `data/historical_lines.csv`
- `python -c "import pandas as pd; from nba_props import _select_active_slate_lines; hist=pd.read_csv('data/historical_lines.csv'); active, slate=_select_active_slate_lines(hist); print('active_slate', slate); print('active_rows', len(active)); print(active['projection_type'].value_counts().to_dict()); print(active['is_promo'].fillna(False).astype(bool).value_counts().to_dict())"` -> OK (`active_slate 2026-04-14`; `active_rows 257`; `{'demon': 234, 'standard': 14, 'goblin': 9}`; `{False: 257}`)
- `python nba_props.py predict` -> OK; used `257` cached PrizePicks lines on active slate `2026-04-14`, generated `5` live picks, and rewrote `output/picks_latest.csv`, `output/picks_history.csv`, `output/betslips_latest.csv`, and `output/betslips_history.csv`
- `python -c "import pandas as pd; picks=pd.read_csv('output/picks_latest.csv'); print(picks[['player','prop','direction','projection_type','is_promo','confidence','pick_source','line_source']].to_string(index=False)); print(picks['projection_type'].value_counts().to_dict())"` -> OK; all `5` current picks are `demon`

---

## Session Handoff
- Last completed task: verified the `2026-04-12` card went `5/5`, pulled the first truly labeled active slate, and generated a new live card from that labeled board
- Current objective: accumulate enough graded true-labeled market history to run a trustworthy standard-line edge audit
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
  - `python -c ...` verification of the `2026-04-12` `5/5` result set
- Known issues: PrizePicks availability remains unstable; standard-line graded sample depth is still too small for a clean audit; the `team_pending` fallback still needs confirmation on the next real pending-team slate
- Immediate next step: keep collecting and grading true-labeled PrizePicks rows, especially `standard` rows, then run the standard-line-only edge audit

## Resume Here
When starting a new session:
1. Read `CLAUDE.md`, `output/STATUS.md`, `output/TASKS.md`, and `output/DECISIONS.md` in that order.
2. Confirm current objective and blockers from repo state, not chat memory.
3. Keep changes scoped and verify exact commands before marking work complete.
4. Update `output/STATUS.md`, `output/TASKS.md`, and `output/DECISIONS.md` before stopping.

---

## 2026-04-15 Update

- **April 14 picks manually backfilled** — 4 picks (Camara TRB, Diabaté AST, Adebayo STL, Wiggins BLK) were archived but missing from `picks_history.csv` due to NBA API lag (latest game_date still showing 2026-04-12 on April 15 morning run). Actuals fetched via `nba_api` player game logs; all 4 graded WIN (4-0).
- **picks_history.csv**: 1,405 rows total
- **Known gap**: no auto-heal when archive picks exist but history rows are absent — see DECISIONS.md for recommended fix
- **Current branch**: `master`

---

## 2026-04-15 — PTS Model Policy Hardened

- **PTS OVER permanently blocked** in both fallback and dynamic policy (`allow_over=False` hard-set in code regardless of eval data)
- **PTS UNDER auto-gate added** (`pts_under_tracking_allowed`): will enable when `under_hit ≥ 0.54` AND `market_hit ≥ 0.50` — currently NOT triggered (market_hit=0.491)
- **Sigma NaN confirmed non-issue**: PTS sidecar produces valid sigma; NaN in history is because PTS stopped generating picks before sigma was added (Apr 2 vs Apr 12)
- **Bias diagnosis**: +2.8 pt OVER bias and -2.6 pt UNDER bias are selection-induced (model trained unbiased at +0.13 overall); no change to bias_advanced.json needed
- **Files changed**: `nba_props.py` (fallback policy, dynamic policy, policy dict)
- **Syntax verified**: `ast.parse` clean
- **Current branch**: `master`

---

## 2026-04-15 — Terminal Log Output Reformatted

- **Problem**: `run_daily.py` pipes all nba_props.py stdout through `logger.info("  {line}")`, so the 26-column `DataFrame.to_string()` block produced unreadable wrapped garbage in the log
- **Fix**: Replaced both wide-DataFrame prints with clean fixed-width tables
  - `nba_props.py` "TODAY'S PICKS" block: shows PLAYER / PROP / DIR / LINE / PRED / EDGE / CONF / HIT% / TIER + optional mkt/σ/INJ extras per row
  - `run_daily.py` `_summarise_live_pick_outputs`: shows same compact format via logger
- **Files changed**: `nba_props.py`, `run_daily.py`
- **Syntax verified**: both files `ast.parse` clean
