# Market-anchored probability rebuild (DEC-109)

The new model starts with FanDuel's paired no-vig probability and learns only a regularized correction. It is implemented, trained, chronologically evaluated, and installed as a separate shadow scorer. It does not replace the live probability sidecars or alter the current card policy.

The historical results are modestly better than the market, but the corrections do not overcome the offered vig at the existing +1% UNDER EV threshold. Activating this candidate as the betting probability would have produced zero qualifying UNDERs in the evaluated sample. This is a probability-design repair with promising retrospective results, not a demonstrated profitable replacement strategy.

## Model

For each exact FanDuel main line, compute `q_over = implied_over / (implied_over + implied_under)`. Predict:

`p_over = sigmoid(logit(q_over) + correction)`

The market logit has a fixed coefficient of one. Forty coefficients model a five-term basis separately for eight props: intercept, log line level, last-10-games gap to the line, last-5 versus last-10 momentum, and recent minutes. Fit Bernoulli likelihood with a fixed L2 penalty of 0.02. Clip the residual correction to +/-0.50 logit units. This specification was fixed before evaluating outcomes; no threshold or hyperparameter grid was searched afterward. It does not use the old model's probability as a predictor.

Training and live inference use the same feature builder over raw played box scores. Rolling features require at least 10 prior appearances and exclude games on or after the earlier of offer date and capture date. Unusable history falls back to the market probability; missing/invalid paired prices produce no score. No same-game outcome, current-game minutes, closing quote after capture, ESPN total scaling, or revised production model prediction enters the correction.

## Data and timing

The final build inspected 79,741 priced FanDuel history rows. Timestamp, source, valid paired prices, local slate date and pre-tip checks left 72,735 eligible snapshot rows, reduced to 37,455 unique player/prop/date offers. Of these, 1,105 lacked a played actual and were excluded; 36,350 had graded non-push outcomes. There were no pushes in this selected data. Residual features were available for 36,049 rows; other assessable offers use the market fallback at inference.

The prior-season foundation contains 28,433 offers over 79 dates. Much of the February/March data was recorded late in the day: these use the first recorded pre-tip offer, not invented morning prices. A strict morning-only trial stopped before model evaluation because only 19 prior-season dates survived. Current-season evaluation uses the actual first archived morning run when present, otherwise the fixed 10am Chicago boundary. Quotes must be available at capture, before tip and within 24 hours of tip; morning captures also require quote age <=24 hours. This training/evaluation capture-time difference is a limitation.

Features and targets are joined with validated unique player/date box scores and a shared player normalizer. Repeated snapshots and line moves never become multiple game labels. Historical master-box revisions may differ from precisely what was available at the time; the retrospective as-of joins cannot remove that limitation. Input snapshots and code hashes are recorded.

## Chronological results

Two predeclared tests score the same 7,917 current-season offers over 18 dates, ending November 19:

| Test | Market log loss | Anchored log loss | Difference | Date-cluster 95% interval |
|---|---:|---:|---:|---|
| Frozen prior-season fit | 0.688367 | 0.687649 | -0.000718 | [-0.001374, -0.000072] |
| Expanding, prior whole dates only | 0.688367 | 0.687455 | -0.000912 | [-0.001873, -0.000012] |

Bootstrap: 10,000 draws, seed 109, resampling whole dates with row-weighted paired differences. This accounts for related outcomes within a night. The intervals narrowly exclude zero overall; individual prop intervals all cross zero. No prop was cherry-picked for promotion. Brier score also improves, from 0.247637 for the market to 0.247184 in walk-forward evaluation. The mean absolute probability correction is 1.14 percentage points.

For the 6,527 exact-line rows with archived incumbent calibrated probabilities, all three methods can be compared on identical offers:

| Method | Log loss |
|---|---:|
| FanDuel no-vig price | 0.687891 |
| Incumbent calibrated probability | 0.689231 |
| Market-anchored walk-forward probability | 0.686924 |

This table uses the incumbent **after calibration**. It is not the raw-probability comparison in DEC-107. The row count differs from DEC-104 because this rebuild validates quote age/tip/local date and selects one main offer per player/prop/date rather than conditioning on every archived line match.

Both anchored tests produce **zero** UNDERs at the existing +1% offered-price EV threshold. ROI is undefined for zero stakes, not 0% ROI. No profitable betting claim follows from the small log-loss gain. The diagnostic does not replay incumbent meta gates, per-player selection, parlays or staking. The November outcomes had already been inspected in earlier audits, so this is chronological retrospective evidence, not a new untouched prospective holdout.

## Installed behavior

`output/market_probability_shadow.json` holds the final research model trained through November 19: 36,049 fit rows across 97 dates. Existing production artifacts remain intact: all **74 existing files in the production model directory** match their recorded SHA-256 values. CONFIG values were not changed, including the provisional +10pp cutoff. No stat projections, probability sidecars, calibrator, meta model, betslips or card were replaced.

On the next FanDuel prediction, the new scorer reads its separate artifact and logs to `output/market_probability_history.csv`. It scores the paired market offers directly and records market/anchored/available incumbent probabilities, feature history dates, exact prices, quote/tip/capture timestamps, training-through date, model SHA and source run ID. It rejects unknown/future/post-tip/over-24h quotes and models trained on the offer date or later. The first offer per player/prop/slate is immutable across reruns. Its output never enters pick generation, price gating, cards, slips or the C shadow-lane definitions.

Daily Step 1.9 grades the separate history under the same settlement rules as existing candidate tracking and prints paired log-loss differences against the market. Logged UNDER paper returns across all observed offers are a reference outcome record, not selected model bets. Missing actual dates remain pending, already-settled rows are preserved, and logger/grader failures are nonfatal.

Collect at least 30 new completed dates and 300 new binary offers before a prospective review, separately from retrospective data. Examine date-cluster uncertainty, same-line market/incumbent comparisons, calibration, quote age, model-version changes and returns at executable prices. Minimum counts never trigger automatic promotion. The current candidate is explicitly marked `shadow_only` and retrospective `promotion_eligible` is false pending prospective evaluation and an economic case for changing live selection.

## Reproduction and verification

Final command:

```powershell
python scripts/train_market_probability.py --output-dir output/market_probability_<new_run> --data-end 2026-11-19 --install-shadow
```

The output directory must be new. `--data-end` must precede today and can advance for an explicitly requested future rebuild. Without `--install-shadow`, no active research artifact is changed. Even with it, only the separate output shadow artifact is installed; production models and CONFIG are never promoted. Model/hash/source/dataset receipts and per-offer frozen/walk-forward predictions are under `output/market_probability_20261120_final/`. Earlier build attempts remain preserved separately.

Independent rebuilds reproduced both reported log-loss values exactly. The temporary-directory live smoke scored 531 still-pre-tip offers from the saved slate and left the incumbent input frame unchanged. Those cached quotes were 11–13 hours old, so this was a wiring/score-isolation check, not an executable-price validation. Its receipt is `output/market_probability_20261120_v3/live_shadow_smoke.json`. No prospective history was seeded and no live prediction run was triggered.

The focused suite passed 61 tests and 29 subtests. Tests cover strict as-of features, future-label perturbation, whole-date walk-forward separation, snapshot selection, price validity, market-offset arithmetic, bounded corrections, history fallback, schema validation, training-date guards, immutable isolated logging, comparable-row loss accounting and grading. The broader suite passed 396 tests/94 subtests with the previously failing dashboard-upload harness file excluded. Final repeat results are recorded in DEC-109.
