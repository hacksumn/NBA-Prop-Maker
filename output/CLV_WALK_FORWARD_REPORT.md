# Frozen CLV Walk-Forward Decision Report

Date: 2026-08-12  
Decision: Keep the production model unchanged; collect direct PrizePicks T-60/T-5 observations; reserve 64,120 Odds API credits.

## Executive summary

The requested chronological eligible-pool test is complete. None of the six
candidate market feature groups has enough walk-forward evidence to enter the
production model. The most promising signal, PrizePicks-minus-consensus gap,
showed an apparent `+26.7pp` T-60 top-five lift, but that result covers only
three validation dates and fifteen selections. Treating it as validated would
be overfitting.

The untouched future comparison is also underpowered and mixed. The T-60 base
shadow overlay beat the current model-edge ranking (`5/9` versus `4/9`) but did
not beat the eligible always-UNDER pool (`54/82`) and covers only two dates. At
T-5, both approaches went `4/10` across three dates. No production artifact was
enabled or replaced.

The pipeline now captures PrizePicks itself near T-60 and T-5, which is the
missing observation needed for true same-book CLV. It uses zero Odds API
credits. The live card is PrizePicks-only, source mismatches force an empty
official card, and implicit Odds API fallback is disabled.

## Test contract

- Frozen input: `data/clv_features_2026_live.csv` (`16,989` rows).
- Eligible pool: archived PrizePicks PTS/TRB/AST UNDER candidates, standard,
  non-promo, model under-edge at least 5%, graded and non-push.
- Selection: expanding chronological folds on development dates only.
- Candidate groups: PP-minus-consensus gap, sportsbook consensus line, no-vig
  probability, book dispersion, T-60-to-T-5 movement, and market
  withdrawal/availability.
- Leakage rule: movement and withdrawal are forbidden at T-60; they become
  eligible at T-5.
- Controls: same-date/prop matched random selection with 2,000 repetitions,
  current model-edge ranking, and the same-window always-UNDER eligible pool.
- Holdout: newest dates are untouched until feature selection is frozen.

## Feature-selection result

No group survived. The gate required mean top-five lift of at least `+1pp`,
positive lift in at least three folds, non-negative mean AUC lift, at least
twelve validation dates, and at least sixty validation card rows.

| Feature group | T-60 result | T-5 result | Decision |
|---|---:|---:|---|
| PP-minus-consensus gap | +26.7pp; 3/3 positive | +20.0pp; 2/3 positive | Do not promote; only 3 dates / 15 picks |
| Sportsbook consensus line | +20.0pp; 2/3 positive | +6.7pp; 2/3 positive | Do not promote |
| No-vig probability | +6.7pp; 1/3 positive | +6.7pp; 1/3 positive | Do not promote |
| Book dispersion | +13.3pp; 1/3 positive; negative AUC lift | 0.0pp; 1/3 positive; negative AUC lift | Do not promote |
| T-60-to-T-5 movement | Not knowable at T-60 | 0.0pp; 0/3 positive; negative AUC lift | Do not promote |
| Withdrawal/availability | Withdrawal not knowable at T-60 | +6.7pp; 1/3 positive; negative AUC lift | Do not promote |

## Untouched future comparison

| Snapshot | Method | Record / rate | Matched random | Dates |
|---|---|---:|---:|---:|
| T-60 | Base shadow overlay | 5/9 (55.6%) | 51.0% | 2 |
| T-60 | Current model-edge rank | 4/9 (44.4%) | 48.3% | 2 |
| T-60 | Always-UNDER eligible pool | 54/82 (65.9%) | n/a | 2 |
| T-5 | Base shadow overlay | 4/10 (40.0%) | 46.5% | 3 |
| T-5 | Current model-edge rank | 4/10 (40.0%) | 45.0% | 3 |
| T-5 | Always-UNDER eligible pool | 54/82 (65.9%) | n/a | 3 |

The shadow overlay contains only the existing model features because no market
feature survived. Its files are useful for reproducibility, but
`models/clv_selection_policy.json` explicitly disables both snapshots.

## What happens next

1. Let the daily runner collect direct PrizePicks T-60 and T-5 snapshots when
   NBA games resume.
2. Verify the first live raw payload, per-event snapshot, checkpoint, and master
   line-store rows.
3. Rerun the frozen test only after the evidence floor is met: at least 12
   validation dates / 60 validation selections and a separate future holdout of
   at least 15 dates / 100 picks, including at least 20 standard picks.
4. Release Odds API credits only if that test identifies a concrete missing
   sportsbook observation. Do not spend them to create more generic density.

## Caveats

- Archived prediction files reproduce the model output used on each date; they
  do not represent one immutable model binary across the full window.
- Historical PrizePicks references are not true closes, which is why the direct
  capture path is now required.
- The eligible sample is small because most recoverable archived PrizePicks
  lines are demon lines, while the production test must remain standard and
  non-promo.
- Hit rate is shown for diagnostic comparison, not as a profitability claim.

Sources: `output/clv_walk_forward_summary.json`,
`output/clv_walk_forward_feature_selection.csv`,
`output/clv_holdout_comparison.csv`, and
`models/clv_selection_policy.json`.
