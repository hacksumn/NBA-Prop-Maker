# Prospective shadow lanes (DEC-108)

These hypotheses are paper-only. About 40 combinations were inspected in the discovery audit, so their historical results may reflect selection noise. Collect new observations before considering either lane for betting. There is no automatic promotion, and no strategy, model or live gate was enabled by this change.

| Lane identifier | Offer observed | Meta band, fraction units |
|---|---|---|
| `pra_under_meta_gt_070` | FanDuel main-line PRA UNDER | `0.70 < meta_prob <= 1.00` |
| `pts_under_meta_050_070` | FanDuel main-line PTS UNDER | `0.50 <= meta_prob <= 0.70` |

The direction is always UNDER, including when the calibrated model favors OVER. Meta probabilities are taken from the scored prediction columns, not the live pick-row percentages. Freeze these bands; do not tune them using accumulating outcomes. The existing PTS and PRA publication policy remains disabled.

## Collection contract

The next FanDuel prediction run starts collection in `output/shadow_lanes_history.csv`. The logger observes all scored offers in the two bands before live-card gate decisions affect selection; it receives the full prediction frame, never the published card. The new rows never enter `picks_history.csv`, cards, betslips, candidate-history performance, or model training.

Require a finite calibrated probability and exact line, paired valid FanDuel prices, a known snapshot at or before capture, and a known tip after capture on the active Chicago slate date. Unpriced/late/unknown-time offers are skipped with counts. Live EV, favorite-price, accuracy and model-market-gap gates are not eligibility filters: their recorded diagnostics are retained, but rejected offers can still be observed. Invalid quote timing is never accepted just to fill the sample.

The append-only key is `(game_date, player, prop, direction)`. Only the first eligible offer per player/prop/slate is retained, including its exact line, probability and price. A rerun with a different line, meta score, or price cannot overwrite that offer or add another outcome for the same player/prop/date. Existing result cells change only through grading. No historical backfill is included; test fixtures and the discovery audit are not counted as prospective observations.

New-file schema includes: game date, player/team/opponent, prop/direction/line, prediction, `meta_prob` (fraction), calibrated over probability, direction probability (percentage), lane/version, run ID, event ID, snapshot/tip/capture timestamps, paired American prices, decimal side odds, break-even and no-vig market probabilities, model side probability and EV, price-gate diagnostics, `tracking_only=True`, `published=False`, and result/actual/paper profit/grading time. The run ID links to the existing `output/gate_runs/` receipt with model/source/config fingerprints. Existing history schemas are unchanged.

## Daily grading and review

`run_daily.py` Step 1.9 grades the separate history after ordinary candidate grading, including in logs-only runs. It reuses the existing box-score preparation and candidate grading conventions for PTS and PRA. Missing box-score dates remain pending. Pushes return zero paper units; DNPs have no paper return. Existing grades are preserved. A file lock prevents the append-only logger from racing the grading rewrite. Logging/grading failures are warnings and do not block live picks.

Daily logs show each lane's binary outcomes toward a review minimum of 300, distinct settled dates, hit rate, one-unit paper profit and ROI at recorded prices, plus pushes, DNPs and pending rows. ROI and the 300-outcome counter exclude pushes/DNPs; the counter is per lane, not pooled. At 300 the log only marks the review minimum as reached. That is not proof of an edge or authorization to bet.

Before any decision, inspect date coverage and within-date/player correlation, calibration against the no-vig market, uncertainty clustered by date, price/line availability and any model-version changes. Report the fixed prospective cohorts separately from the discovery audit and from actual bets. Do not combine the two lanes as independent evidence when they involve the same player/game.

## Verification

`tests/test_shadow_lanes.py` verifies exact meta boundaries and fraction units, forced UNDER direction, paired pre-tip quotes on the correct slate, immutable first capture across line/price/meta changes, input/config preservation, separate-file writes, nonfatal failures and daily grading (PRA sums, PTS, American-price payout, push, DNP, missing-date pending and idempotency). Live card and slip files are checked unchanged by the grader. The focused suite passed 50 tests and 24 subtests. Broader regression: 385 tests and 89 subtests passed with the previously failing dashboard-upload harness file explicitly excluded.
