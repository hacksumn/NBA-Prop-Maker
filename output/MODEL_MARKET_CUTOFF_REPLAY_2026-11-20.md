# DEC-105: model-versus-market cutoff replay

The proposed cutoff improves the historical published-card ROI and saves 1.05 units at flat one-unit stakes. It is an exploratory exposure reduction, not a demonstrated improvement in forecasting accuracy. The broader candidate population loses 2.91 units of historical profit when the same cutoff is applied.

## Rule and results

For a FanDuel UNDER, exclude a pick when `model_side_prob - no_vig_side_prob > 0.10`. These are probabilities in [0, 1], using the paired no-vig market price, not the vig-inclusive break-even price, meta probability, confidence score or raw projection edge. Exactly +0.10 remains eligible. Negative gaps are not rejected by this rule. Existing price, EV, timing and accuracy requirements still apply.

Flat one-unit singles, WIN/LOSS only:

| Population | Policy | Picks | Wins | Hit rate | Profit | ROI |
|---|---|---:|---:|---:|---:|---:|
| Published, Nov 1–19 | Before | 134 | 79 | 59.0% | +8.82u | +6.6% |
| Published | Keep gap <= 0.10 | 66 | 43 | 65.2% | +9.87u | +14.9% |
| Published | Removed | 68 | 36 | 52.9% | -1.05u | -1.5% |
| Candidates, Nov 5–19 | Before | 275 | 175 | 63.6% | +18.49u | +6.7% |
| Candidates | Keep gap <= 0.10 | 203 | 134 | 66.0% | +15.57u | +7.7% |
| Candidates | Removed | 72 | 41 | 56.9% | +2.91u | +4.0% |

The benefit is concentrated in published AST: retained 59 picks return +7.42u; removed 53 return -3.11u. Removed BLK (13 picks, +0.83u) and STL (2 picks, +1.24u) were profitable. This does not establish a universal losing high-gap lane. The broad cutoff implements the requested A without fitting additional prop-specific thresholds to these same outcomes.

Exploratory paired date-cluster bootstrap, 20,000 draws, seed 104: published ROI change +8.37 percentage points, 95% interval [-5.85, +27.45] points (16 dates). Candidate ROI change +0.95 points, interval [-5.70, +7.64] points (14 dates). Both intervals include no improvement. These intervals do not correct for discovering the threshold on the evaluated data.

## Coverage and limits

All 1,752 published-history rows and 317 candidate-history rows were inspected. Only 152 published rows have both logged model and market probabilities: 134 binary settlements, 8 DNPs, 10 pending. The remaining 1,600 published rows include 1,485 WIN/LOSS rows that cannot assess this rule from the recorded probabilities. They are not represented as retained or as evidence for improvement. All 317 candidate rows are assessable: 275 binary, 19 DNP, 23 pending. Pending includes the November 20 slate; no outcome is inferred.

The November 20 published card has 10 pending picks. The rule would exclude Onyeka Okongwu AST UNDER 2.5 (gap +0.1212), leaving nine. This is a snapshot illustration, not advice based on a refreshed offer. No saved card was rewritten or regenerated.

This is a removal-only replay on recorded offers. It does not regenerate past candidates, replace a removed player's pick, rerank a slate, reconstruct missing historical probabilities with today's model, simulate parlays, or model actual stakes. Candidates and published picks overlap and are not independent replications. Candidate promotion retains first-observed offer fields, so candidate results describe logged offers; published history is authoritative for published-card P&L. Prices were recomputed independently from American odds, and every assessed binary grade agrees with its logged actual and line.

## Implementation and monitoring

`nba_props.CONFIG['live_max_under_model_market_gap'] = 0.10` adds the cutoff to the existing exact-contract FanDuel price gate before card selection. `None` rolls it back. The same four-decimal probabilities used by the replay are compared in production. Failures use the existing `price_gate_reason` and remain in candidate history and gate receipts; no schema change. This affects future picks and downstream slips; it does not alter the probability model. The existing FanDuel price gate must remain enabled for this policy to operate.

Freeze the 0.10 threshold. Starting with full slates on November 21, compare retained and rejected final candidates at their recorded prices, separately from actual published returns. Review after at least 30 newly completed dates and 300 new binary candidates; those counts are review minima, not proof of an edge. Include hit rate, units, ROI, date-cluster uncertainty, prop mix and empty-card frequency. Do not tune the cutoff or exempt props using this same discovery sample. Candidate grading already captures rejected rows; no new scheduled task is required. The replay command can be rerun into a new directory and the row outputs filtered to the prospective period.

## Reproduction

Command: `python scripts/replay_model_market_cutoff.py`.

Evidence directory: `output/model_market_cutoff_20261120_180052_390511/` contains `summary.json` (input hashes, script hash, Git revision, accounting and intervals), full annotated source rows, and date/prop breakdowns. Each run writes to a new directory and refuses to overwrite an existing destination. The input histories' SHA-256 values were unchanged after replay and implementation verification.

Verification: 46 focused tests and 16 subtests passed for pricing, replay boundary/accounting, canonical live entry points, candidate retention and card selection. The production cutoff was independently exercised against all 152 priced published offers, agreeing exactly with replay decisions. Full suite: 389 passed, 4 dashboard-upload harness failures, 85 subtests passed. Those same four failures reproduce using only unchanged Git HEAD files in an isolated temporary folder; no A code is involved. Second replay reproduced all population metrics, source hashes and bootstrap intervals exactly.

B (market-anchored rebuild), C (shadow lanes), and D (ESPN totals and logging) remain separate queued tasks.
