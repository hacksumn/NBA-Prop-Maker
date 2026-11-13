# Live pick gate review — November 12, 2026

**Needs revision. Review coverage is partial.** Three demonstrated execution defects were repaired. Current historical evidence cannot establish optimal thresholds for all gates. This review covers the active CLI, source/price boundary, candidate generation, sanity checks, final card, candidate reporting, and relevant historical evaluation paths. It includes static code traces, targeted regressions, and all 149 settled candidate outcomes. It does not certify every possible input combination, injury feed response, or model-training artifact.

The current production lanes are AST, STL and BLK UNDER. Other props and all OVER lanes are disabled by the loaded evidence policy. The 15–3 rejected-candidate result is accurate, but covers six dates in AST UNDER. Earlier generation and sanity rejections are absent from that population.

## Verified repairs

1. **P1 — `all` bypassed publication checks.** Its separate scoring branch did not attach or filter exact FanDuel prices, did not apply the same source/date policy as `predict`, and omitted candidate logging. After its existing update/train stages, `all` now delegates to the canonical `predict` branch with fresh feature generation. Mocked command-entry tests verify delegation without invoking training or providers. Source policy, causal history cutoff and exact-price filtering are now shared.
2. **P1 — Invalid numerical inputs could pass the price gate.** Infinite directional probability was clipped to 99%, producing approximately +89% modeled EV at -110. Infinite American prices were also accepted by the conversion helper. These inputs now fail validation. NaN resulting from invalid model probability is explicitly rejected before EV comparison. Regression checks cover infinity, out-of-range probability and infinite prices.
3. **P1 — The run-time snapshot boundary was incomplete.** Cached selection previously omitted its supported `as_of` argument. A snapshot could be pre-tip yet later than the prediction run. `predict` now supplies the current UTC time. The price gate requires a known snapshot timestamp at or before that time, alongside a future start time. Regression checks cover unknown/future snapshots. A maximum snapshot-age policy remains an open operating choice.

Code and regression evidence are in [nba_props.py](../nba_props.py) and [test_live_gate_entry_points.py](../tests/test_live_gate_entry_points.py).

## Gate inventory and assessment

| Stage | Active rule or behavior | Assessment and action |
|---|---|---|
| Provider availability | Daily pipeline skips live picks when authoritative fetch fails | Consistent with the requested source policy. Transient network errors currently abandon this stage. Add bounded transient retries with sanitized logging and credit-aware per-event checkpoint reuse. |
| Source and slate | FanDuel, exact local-date slate, standard contracts | Shared by `predict` and `all` after repair. Alternate markets remain disabled. |
| Snapshot/time eligibility | Known snapshot at/before run, game start after run | Repaired. Same-day old quotes still qualify. Choose an age policy using capture cadence and observed line movement. |
| Input/history eligibility | Minimum-games contract, roster/schedule eligibility, history strictly before slate in `predict` | Causal cutoff shared with `all`. Historical roster/absence mapping and cross-season windows remain documented risks. Earlier rejections lack a structured outcome ledger. |
| Prop/direction evidence | Loaded training results and edge analysis determine allowed lanes | AST/STL/BLK UNDER only today. Missing/corrupt training results fall back to permissive defaults. Recommend an explicit policy-unavailable state for production publication. |
| Primary directional edge | Combined edge is 45% model residual and 55% market-edge prediction when available. FanDuel residual minimum 0.05 and side probability minimum 50% | Multiple agreement/regime tests follow. These thresholds need prospective ablation. Edge and probability can carry related information. |
| Low/high sigma edge | High sigma raises edge requirement 25%. Low sigma claims to reduce it 25% | Low-sigma expression `max(min_edge, min_edge * 0.75)` is a no-op for positive floors. Resolve desired policy before changing its effective threshold. |
| Agreement/regime | L10/L5/anchor disagreement can require extra edge. High regime score plus disagreement rejects | Earlier rejected rows are not retained, preventing a reliable full replay from candidate history alone. |
| Injury/availability | OUT, DOUBTFUL, QUESTIONABLE, and unresolved DAY-TO-DAY removed. Pending-team rule treats unknown OVER separately | Static trace verified. No comprehensive retained first-decision ledger exists to estimate false rejections or status-at-run accuracy. Preserve run-time status evidence. |
| Minutes/role sanity | Expected-minute inconsistency, last-game spikes/crashes, role instability | Static trace verified. Text for implied-minute checks is confusing. Validate conditional rates using prediction-time minutes/status snapshots before tuning. |
| Historical volatility/sanity | L10 CV >1.00, outlier spike, line/model mismatch. Micro-stat exceptions use recent strict-under rates | This is separate from the final sigma ceiling. Audit them jointly and independently to measure incremental value. |
| Exact price | Paired exact-line odds, finite valid probability, known pregame timing, favorite at least -200, model EV >=1% | Integrity strengthened. Calibration uncertainty around a 1% EV margin remains consequential. Current cap rejections lost money despite a 66.7% hit rate. |
| Final meta | AST/STL/BLK floor 42%. AST is labeled soft in prop policy but final card still enforces a hard floor | Behavior is explicit in final accuracy gates. Rename the stage-specific mode or document that soft only describes primary generation. |
| Final volatility | AST <=3.506, STL/BLK <=1.50 | Newly permitted AST band has no settled published observations so far. No empirical basis to declare the new ceiling optimal. |
| Estimated hit rate | >=54.5%, learned from historical published picks | Lowest recorded candidate value is 57.7%. Zero recorded failures, so this dataset cannot test the floor. History pools market/model eras. The comment referring to PrizePicks break-even is inappropriate for variable sportsbook prices. |
| Source exclusions | `volume_fill` and `emergency_fill` always removed by card builder | Generates guaranteed exclusions and mislabeled diagnostics. Separate tracking generation from publishable candidate generation. |
| Final card diversity | One pick per player, no total/per-stat cap | Correlation control is intentional. Rank by expected utility/EV after calibration if selecting between multiple props on one player. Evaluate on held-out offers. |
| Betslip selection | Additional eligibility, probability and dependence rules | Relevant existing slip regressions passed. No new claim about real joint calibration or executed parlay returns. |
| Evaluation entry points | Historical daily-card backtest runs `filter_best_picks` without complete live sanity/price/card chain | This backtest cannot validate the full production gate stack. Build a common pure selection pipeline with historical as-of inputs and no network calls. |

## Recomputed analytics

Evidence was read on November 12. Candidate history contains 168 rows dated November 5–12. There are 149 binary outcomes, six DNPs and 13 pending rows. The pending slate is excluded from performance. All 149 W/L classifications match current stored box scores. Exact candidate keys have zero duplicates. All stored candidate snapshots and log times precede tip, and all timestamp fields needed for that check are known.

| First-candidate bucket, including sticky promotions | W–L | Profit per 1-unit stake | ROI |
|---|---|---|---|
| Published | 36–20 | +9.02u | +16.11% |
| Price blocked | 53–22 | +6.24u | +8.32% |
| Other blocked | 15–3 | +10.61u | +58.96% |
| Price blocked by favorite cap | 22–11 | -1.71u | -5.19% |
| Price blocked by EV minimum | 31–11 | +7.95u | +18.92% |

Profits are recomputed from recorded decimal odds at flat unit stakes, with DNPs/pending excluded. They describe suggested offers and do not establish actual executed returns. Values can differ slightly from the pipeline's sum of individually rounded profits.

The rejected group includes overlapping failures. Mutually exclusive decomposition is meta only 6–2 (+4.19u), volatility only 3–0 (+2.97u), both 1–0 (+0.66u), volume-fill exclusions 4–1 (+2.19u), and missing-volatility/emergency-fill 1–0 (+0.60u). There is one additional meta-rejected DNP.

### Changes already made on November 10 have little future evidence

The AST meta floor changed from 46% to 42%, and the volatility ceiling from about 2.92 to 3.506. The published 42–46% meta band has two settled observations, both November 10 losses (-2u). Published candidates at or above 46% went 36–18 (+11.02u), but that is an observational comparison. No settled published current-season FanDuel AST pick falls within the newly opened volatility band. These observations are insufficient to validate or reverse either setting.

Removing the recorded blockers using today's limits would clear seven previously blocked settled rows that went 6–1 (+4.83u). That calculation does not rerun historical models or guarantee that all other production gates would choose those same rows.

### Probability and closing-market checks

| Selected cohort | Model Brier | Market no-vig Brier | Model log loss | Market log loss |
|---|---|---|---|---|
| All 149 settled candidates | 0.22083 | 0.23284 | 0.63500 | 0.65835 |
| 56 settled published candidates | 0.22998 | 0.24580 | 0.65280 | 0.68464 |

Lower is better for these proper scores. Published mean model probability is 62.54%, with observed hit rate 64.29%. This is selected-cohort evidence over seven dates. It does not establish board-wide calibration or justify a 1% EV threshold.

The saved CLV summary contains 90 published picks, only 28 with a later exact-line close. Mean EV versus that close is -3.86%, with a 25% exact-close beat rate. The broader lean panel has 1,221 exact closes and mean EV versus close -5.88%. Positive short-run realized returns and these negative price-relative results both need to remain visible. Closing coverage is incomplete and the populations differ from the candidate table.

Exploratory whole-date bootstrap intervals are retained in the JSON receipt. Seven or fewer dates, dependent offers and outcome-driven policy tuning prevent treating those intervals as independent confirmation of an edge.

## Remaining errors and measurement limitations

### P1 — First candidate and publication decisions are mixed

Sticky promotion changes only `published` and `gate_bucket`, retaining the first candidate's probabilities, price and failure reason. Three published rows retain failed gates. This is a provenance problem rather than evidence that the current price gate was bypassed.

Zach LaVine AST UNDER 2.5 on November 6 was initially rejected at -158 with EV -0.0317. Later publication used -138 with EV +0.0227. Hartenstein on November 10 retains +118 from the first candidate while publication used +116. Bane and Hartenstein also retain the old meta<46 failures after promotion. The price-divergent observations both lost, so the current total profit happens to agree. A promoted winner could produce different returns in the two files.

**Proposed fix.** Keep the first-candidate ledger and add immutable decision events keyed by run, offer, policy and player. Store publication-time prices/probabilities and explicit rejection reasons. Calculate actual publication performance from publication events. Retain this distinction in every gate audit.

### P1 — Grading does not reconcile later box-score changes

Wembanyama BLK UNDER 3.5 has stored actual 1 versus current box 2 on November 7, and stored 9 versus current box 8 on November 9, in both pick and candidate histories. Present W/L and profit are unchanged. The source of the disagreement is unverified. These are confirmed local discrepancies, and the grading code only processes previously ungraded rows.

**Proposed fix.** Add reconciliation events for changed box scores and derived outcomes, preserving original grades and provenance. A date having some box scores also does not establish that every missing player's event is final. Require event-final/participation evidence before assigning DNP. This review verified all 149 settled W/L results but did not independently certify all six DNP decisions.

### P1 — Current data cannot evaluate every gate

Candidate history starts after generation and sanity checks. It omits disabled props, disallowed directions, low edge/probability, injury removals and role/sanity failures. Stored history also lacks a full policy/model fingerprint and complete thresholds. The 15–3 population therefore cannot answer whether those omitted gates are efficient.

**Proposed fix.** Capture structured rejection events before each stage, including all simultaneous failures, prediction-time status, odds, thresholds, policy/model hash and scoring time. Evaluate gate removals on identical pregame offers. Separate marginal removals from overlapping rejections.

### P1 — Threshold evidence is outcome-selected and mixes eras

Earlier gate changes used these same short observed windows. Broad pick history additionally mixes sources, models, unknown-time records and restored archives. Of 1,689 records, 1,422 lack source tags. There are 118 FanDuel-tagged rows, including 23 from May and 95 from October/November. Thirty-eight FanDuel-tagged rows lack known snapshot/start timing.

**Proposed fix.** Freeze a named policy prospectively. Use completed future slates, actual paired-price offers, known pregame timestamps, and date-cluster uncertainty. Evaluate incumbent and alternative policies on the same offers. Keep historical exploration and the final evaluation window distinct.

### P2 — Code clarity and wasted work

- The low-sigma reduction is a demonstrated no-op. Choose the intended minimum first, then implement an explicit floor such as a source-specific minimum combined with the reduction. Do not silently lower the live threshold during an audit.
- Fallback fills are built toward a 500-row candidate-pool request and then excluded. Generate them only for explicit tracking when the current card policy cannot publish them. Label their exclusion directly.
- Duplicate `sanity_check_picks` and `backtest` definitions leave obsolete logic in shadowed functions. Consolidate the active definitions in a separate behavior-preserving change.
- Final accuracy helpers repeatedly reload policy/sigma JSON and relearn history summaries. Build one immutable per-run selection context, then reuse it across candidate diagnostics and card construction. Benchmark before claiming a runtime gain.
- The price probability helper falls back to composite confidence or 50% when directional probability is absent. Normal live candidates supply that probability, but the pricing boundary should eventually require explicit probability and provenance. Keep legacy display compatibility separate.
- Logs currently expose the Odds API key inside exception URLs. Redact request credentials in future logging and regenerate affected logs only through an explicit evidence-preservation procedure.

## Recommended threshold decisions

| Gate | Current decision supported by this review | Evidence required for a change |
|---|---|---|
| Exact source/paired price/pregame/finite inputs | Enforce consistently | Boundary regressions and actual provider-format checks |
| -200 favorite cap | Retain during frozen evaluation | Prospective net return/calibration by price band |
| 1% EV minimum | Retain as incumbent, withhold claims it is optimal | Selected-region calibration and held-out threshold sweep versus market baseline |
| Meta 42 | Retain as incumbent experiment | Same-offer future comparison against 46 and meta-as-ranking alternatives |
| AST volatility 3.506 | Retain as incumbent experiment | New-band observations plus independent/marginal gate ablation |
| Estimated hit 54.5 | Treat as unvalidated auxiliary filter | Rejections and true prospective performance by estimated-hit band |
| Fill-source exclusion | Retain current publication policy | Adequate independent fill sample with calibration and price-relative metrics |
| Injury/minutes/role gates | Preserve safety policy while instrumenting | Run-time feed snapshots, event outcomes and per-rule false-rejection analysis |

The existing candidate holdout plan specifies 15 dates and 100 graded rows for model evaluation. Gate experiments need their own predeclared window and power/precision target. Merely meeting an arbitrary row count does not demonstrate a profitable gate. Prefer changes supported by stable price-relative performance across dates and calibration uncertainty smaller than the proposed EV margin.

## Verification and evidence

The reproducible [audit script](../scripts/audit_live_gates.py) writes [a source-hashed JSON receipt](live_gate_audit_2026-11-12.json). Its assertions verify zero settled outcome mismatches and unchanged input hashes during calculation. It reports the two actual-value discrepancies without rewriting raw evidence.

The full test result is recorded in [the test log](live_gate_tests_2026-11-12.txt). The review's source-specific calculations used local snapshots. Training, provider availability and actual execution of bets were outside the verification performed.

### Review usefulness and clarity

| Category | Observed defects | Assessment |
|---|---|---|
| Usefulness/completeness | 1 / unknown | Complete upstream rejection evidence is missing, so empirical all-gate certification remains incomplete. |
| Analytical clarity | 1 / unknown | Generic selection labels and sticky promotions obscure actual reasons. |
| Visual/interaction consistency | N/A | Repository code/evidence review, with no dashboard changes. |

### Analytical correctness and robustness

| Category | Observed defects | Assessment |
|---|---|---|
| Source authority/confidence | 1 / unknown | Two actuals disagree across stored sources. Live revision provenance unverified. |
| Value accuracy | 1 / unknown | Current binary outcomes all agree. Candidate publication-price provenance remains inconsistent. |
| Within-chart agreement | N/A | No chart was revised. |
| Complete source details | 1 / unknown | Early decisions and policy versions are not fully retained. |
| Cross-artifact consistency | 1 / unknown | First-candidate and publication prices differ for two promoted rows. |
| Data-quality controls | 1 / unknown | Completed grading has no revision reconciliation. Execution defects described above were repaired and tested. |
| Conclusion support | 1 / unknown | Current observations cannot establish optimal thresholds or independent forward profitability. |

Counts refer to scoped issue units. The complete denominator of all possible gate states is unknown. Categories overlap and must not be summed into an accuracy percentage.


## Implementation follow-through

Following approval to proceed, primary generation and sanity rejection capture, immutable publication receipts, policy/model/code fingerprints, paired market and injury-report evidence, explicit fallback diagnostics, and daily future-comparison/grade-reconciliation summaries were implemented. The low-sigma no-op was removed while retaining the effective configured minimum.

The future experiment specification is `output/gate_experiment.json`. Its first eligible date is November 13, 2026, with 30 completed dates and 300 binary contracts as the minimum review point. It measures incumbent versus stricter meta46/volatility1.25 final-candidate eligibility on the same first pregame offers and excludes policy/model/code drift. Actual publication is reported separately. Alternate full-card reranking and upstream-gate ablations remain outside this initial evaluator. First-failure generation coverage applies to the primary path and disabled props. Fallback paths are retained at the final-candidate stage.

Reconciliation across 1,733 history rows with available boxes found five file-level conflicts, including one March 25 Clingan PTS OVER14 record stored as LOSS despite actual14. Current grading convention implies PUSH. This conflict needs source/convention review, alongside the two Wembanyama actual differences. Events are in `output/grade_reconciliation/` and the current summary is `output/grade_reconciliation_summary.json`.

Full suite passed 252 tests and 10 subtests. Synthetic capture/evaluation tests passed. First real scheduled receipt and future settled observations remain pending.
