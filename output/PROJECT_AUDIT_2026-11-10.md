# Fresh Start NBA project audit — November 10, 2026

**Assessment: Needs revision for strong accuracy/edge claims. Repairs verified; model improvement remains unproven.**

The strongest immediate improvements are trustworthy line selection, fully separated validation dates, and frozen forward predictions. These are more valuable than another layer of features while evaluation inputs can change materially.

## Coverage and limits

Inventoried the workspace outside `.git` (about 13,350 files, including generated artifacts). Parsed all 108 active Python files under the root, `scripts`, and `tests`; no syntax errors in the final audit. Read canonical project memory and traced selected high-impact ingestion, feature, scoring, selection, calibration, grading, and candidate-evaluation paths. Profiled the full canonical box-score, line, pick, and forward-holdout CSVs. Ran the complete test suite.

This is a broad, targeted review, **not a semantic inspection of every cached payload, serialized estimator, notebook cell, dashboard view, or external data source**. Scheduler execution, live APIs, paid collection, browser rendering, and a full candidate retrain were not exercised. Existing uncommitted changes were preserved. All data/model artifacts were left in place.

Reproducible evidence: `project_audit_evidence_2026-11-10_verified.json`. Its source SHA-256 hashes match the first audit for the five canonical CSVs; the audit did not alter them. Inventory sizes/counts are snapshots and grow as new evidence is added.

## Highest-impact findings and repairs

### 1. P1 — Valid timestamps silently became missing in mixed-format columns: FIXED

`nba_props._select_preferred_line_rows`, advanced training's `_load_lines_lookup`, daily snapshot freshness, live price gating, candidate/tracking history filters, and the naive-baseline audit used pandas timestamp inference on mixed formats. The first legacy timestamp could make later valid ISO timestamps parse as `NaT`. Consequences included lost pregame rows, incorrect latest-line selection, and ineffective known-time checks.

Across 742,967 stored line rows, **584,364 valid timestamps were rejected by the default parser**. Explicit mixed-format parsing leaves 77,679 genuinely unparseable snapshots; 158,603 rows lack a parseable tip time. There are 373 known snapshots at or after tip. These categories overlap and must not be added together.

On the same raw CSV, executing the old selector from Git HEAD versus the repaired selector produced:

| Measurement | Before | After |
|---|---:|---:|
| Preferred contracts selected | 46,824 | 65,835 |
| Matching contracts whose line changed | — | 17,159 |
| Matching contracts whose snapshot changed | — | 27,087 |

The comparison joined on `game_date, player_norm, prop`; 46,824 contracts matched. This is a selection correction, **not evidence of improved win rate**. Historical training/evaluation metrics must be rerun in isolated candidate directories. Production selection can also change when it encounters mixed formats; saved cards were not regenerated.

**Remaining:** legacy timestamps without timezone offsets are still treated as UTC by existing policy. Their actual timezone is not established. Some research readers still accept unknown snapshot/tip times. Add explicit provenance/eligibility cohorts rather than deleting those rows or calling them verified pregame evidence.

### 2. P1 — Row-based chronological folds share entire game dates: FIXED for future advanced/minutes training

Sorting by date then applying `TimeSeriesSplit` to player rows does not separate slates. On the 84,544-row master box-score dataset, **all five default fold boundaries shared the same date**: January 25, 2024; November 1, 2024; February 4, 2025; November 12, 2025; February 12, 2026. Thus outcomes from a game/slate could train a model evaluated on another row from that slate, which does not match morning prediction conditions.

Added `chronological_validation.date_grouped_splits`: expanding folds over distinct dates with strict `max(train_date) < min(validation_date)`. Applied it to nine advanced regression/sidecar/blender/market/meta CV paths and the minutes-model CV path. Missing/unsorted dates or too few distinct dates now fail rather than quietly splitting rows.

**Remaining:** frozen production weights and old OOF artifacts retain their original training procedure. They were not rewritten or retrained. Refresh isolated candidates and calibrators together, compare on common untouched future rows, and retain original artifacts as historical evidence. This change does not certify every other training script's split policy.

### 3. P1 — The future holdout was not protected against late capture or rerun replacement: FIXED

`scripts/capture_candidate_future_holdout.py` previously selected pregame *line snapshots* without requiring prediction *capture* before tip. A cached morning line could therefore receive an after-tip prediction. Repeated capture used `drop_duplicates(..., keep='last')`, replacing the first prediction and even its stored actual with a new blank actual.

Capture now requires a known snapshot observed by capture time and a known future tip, both before scoring and again after scoring. Writes use the existing append-only history helper. Evaluation retains only the first provably pregame prediction per candidate/date/player/prop/exact-line contract; repeated snapshots remain physically stored but do not inflate the result count.

**Current data:** all 1,777 stored captures pass timestamp eligibility; no duplicate exact evaluation contracts were found. This is a latent-defect repair, not a finding that the existing holdout was contaminated. Four captured dates include a still-ungraded slate; the saved summary has three graded dates.

**Remaining:** different lines for the same player/stat remain different contracts and are dependent. Slate-cluster resampling addresses within-slate dependence but cannot make three dates a reliable sample. Model/data/config fingerprints per captured run would strengthen provenance further.

### 4. P1 — Holdout confidence intervals targeted a different average: FIXED

The summary reported row-weighted accuracy differences, while `_date_bootstrap` averaged each slate equally. With different slate sizes, the interval described a different quantity from the displayed point estimate. Bootstrap draws now resample whole dates but calculate summed correct-count differences divided by summed observations, matching the displayed row-weighted metric.

Existing summary files were preserved; the next grading run will compute the revised intervals. The changed interpretation is tested with deliberately unequal date sizes.

## What current performance actually supports

Raw pick history has 873 WIN, 669 LOSS, 98 DNP, 27 PUSH, and seven pending rows. Those span multiple model/policy/source eras. A pooled hit rate is not a current-system validation result.

For FanDuel-tagged binary results from October 26 through November 9, 2026:

| Prop | All graded W/N | Exactly priced W/N | Flat one-unit profit | Priced ROI |
|---|---:|---:|---:|---:|
| AST | 34/54 | 30/48 | +7.73u | +16.1% |
| BLK | 6/10 | 4/7 | −0.32u | −4.5% |
| STL | 5/6 | 1/1 | +0.68u | +68.5% |

Prices are usable only when finite American odds have absolute value at least 100. Missing-price wins **and losses** are excluded from ROI. These are suggested picks, not verified executed wagers; returns assume one unit per recorded pick and do not include stake sizing or slippage. There are only 56 exactly priced binary picks overall. One priced STL outcome provides almost no evidence about repeatable performance.

The existing `clv_summary.json` reports 34 rows with a later close, only 28 with an exact-line close, and **mean EV versus the close of −3.86%**. Its exact-close beat rate is 25%, while the mean no-vig probability movement is positive. These are different quantities: movement toward the selection need not overcome the offered vig. Verify exact-price capture and extend the panel before treating short-run positive returns as established edge.

The saved shadow summary reports candidate accuracy 51.06%, production 51.68%, and always-UNDER 55.44% over 1,277 binary rows and three dates. It is insufficient for promotion; the candidate's nominal result is below both comparators. Whole-board direction accuracy is also a different population from the filtered live card.

## Further high-priority work

1. **P1 — Freeze gate experiments prospectively.** DEC-071/073 loosened gates using only eight/four slates. This is exploratory selection on observed outcomes, not independent confirmation. Record a policy version/hash with every prediction and candidate; predeclare a future review period, dates/rows, metric, and stopping rule. Keep the current thresholds unchanged for a defined window. Evaluate old/new policies on the same subsequently captured offers, with actual prices and date-cluster uncertainty.

2. **P1 — Treat market probability as the baseline.** Always-UNDER is useful, but it does not measure price-relative skill. Report model Brier/log loss and calibration against paired-price no-vig probabilities on untouched FanDuel rows. Separate model probability from ranking/confidence/meta scores. Evaluate calibration in the actual selected region, especially near the current 1% minimum estimated EV, where small probability errors can reverse the sign. Use cross-fitted calibration within development and reserve the final future period.

3. **P1 — Resolve current-roster versus historical-profile team labels.** DEC-067 documents stale absence-team mappings and cross-season L20 windows in Step 8. This can misattribute missing teammates and blend adjustments after trades. Join on stable NBA IDs with effective-date team membership and a prediction-time availability snapshot. Produce a player-level before/after mapping and card diff in shadow mode. Do not use today's roster to rewrite historical team assignments.

4. **P1 — Add strict evaluation eligibility without destroying legacy data.** Separate verified pre-tip snapshots, unknown-time legacy records, current sportsbook rows, proxy lines, and recovered picks. Require known `snapshot < tip` for headline historical market results. A reconstructed archive can aid debugging without proving what was available when the bet could have been placed. Confirm training and live selectors share the same main-line rule when alternate lines are offered; advanced lookup currently does not use the live balanced-price main-line selection.

5. **P1 — Audit partial-slate grading.** `run_daily.step1_6_grade_betslips` interprets any missing player box row as DNP once that date has available results. A date with partial games/data is not proof that another player was inactive. Add event-final and participation evidence, retaining unresolved legs until their event is final. Validate refund/push reductions separately by book; do not infer all refund rules from one payout helper. This risk was traced in code, not reproduced against a confirmed wrongly graded historical slip; remediation needs a precise event-final contract.

6. **P2 — Remove shadowed definitions in a separately verified refactor.** `nba_props.py` defines `backtest` at two locations and `sanity_check_picks` twice; `vegas_gui.py` similarly duplicates `configure_page` and `render_sidebar`. Python uses the later definition. Editing the earlier function can silently do nothing. Preserve current behavior while extracting the active versions and testing callers; no deletion/refactor was performed here.

7. **P2 — Establish one project environment and a run lock.** The executed tests use the Hermes-agent Python environment, consistent with the existing operator warning. No project lockfile is present. Capture exact runtime versions, build a project-owned environment, and check it in preflight. Add a process lock before scheduled/manual pipeline mutations; append-only merging alone does not prevent concurrent lost updates. Exercise interrupted atomic writes and overlapping runs in temporary directories.

8. **P2 — Fix misleading project guidance.** README/PROJECT_STATE still present April figures and PrizePicks-first behavior although current production is FanDuel. ARCHITECTURE calls the approach profitable without current supporting evidence. Replace static badges with dated, source/policy-specific results or clearly label them historical. `outlier_heatmap.py` uses a machine-specific absolute path and full-history player z-scores; it is descriptive EDA, not a causal pregame feature. Zero points or overtime is not automatically a corrupt row, and flagged games should not be removed simply because they are unusual.

## Improvement sequence and acceptance criteria

| Order | Work | Evidence required before promotion |
|---|---|---|
| 1 | Rebuild an isolated candidate with repaired line parsing and date-grouped folds | New input/config/code fingerprints; no same-date overlap; explicit known-time coverage; no production overwrite |
| 2 | Refit its bound calibrator and repeat same-contract comparisons | Untouched chronological period; Brier/log loss, calibration, priced EV/ROI, matched market baseline; date-cluster uncertainty |
| 3 | Continue frozen live shadow collection | At least the registry's declared 15 dates/100 rows; report per-prop sample size and no automatic promotion; extend if evidence remains weak |
| 4 | Shadow effective-date rosters and minutes/availability changes | Mapping diffs, pregame-only features, live/training parity, and a same-offer future comparison |
| 5 | Test additional features individually | Ablation against the repaired baseline; prioritize minutes uncertainty, role changes, rest, and opponent context only when causally available |

Do not optimize hit rate alone: a higher hit rate at worse prices can lower returns. More picks, more complex models, and more permissive gates are not themselves evidence of improvement. No numerical improvement target is justified by this audit.

## Verification and reproduction

`python -m pytest -q tests --disable-warnings` → **238 passed, 330 warnings, 1 subtest passed** (224 tests at entry; 14 added). Warnings are principally DataFrame fragmentation in feature creation and remain an optimization issue.

Added tests cover mixed-format line selection and cutoffs, training lookup consistency, strict holdout eligibility, first-capture preservation including existing actuals, repeated-snapshot weighting, bootstrap weighting, exact-price ROI exclusions, and date-isolated folds.

`python scripts/audit_project_integrity.py --output output/project_audit_evidence_2026-11-10_verified.json` → complete, 108 active Python sources parsed, zero syntax errors. Use a new output filename for subsequent runs; the script intentionally refuses to overwrite evidence.

The old/new selector comparison compiled only the historical selector function from `git show HEAD:nba_props.py` into a copy of the current module namespace, ran both functions on the same CSV, and compared exact contract keys. Neither call wrote files.

No model retrain, promotion, network fetch, paid API use, live-card regeneration, scheduler modification, or bankroll change was performed. This audit repairs evaluation integrity and identifies a concrete path to test better picks; it does not establish improved betting accuracy.
