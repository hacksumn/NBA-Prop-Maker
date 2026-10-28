# Availability and Minutes Model Validation

## Executive Summary

- **The model is ready to become an input to prop-model retraining.** It passed all four expanding development folds and the latest-season holdout at the candidate player/team/game grain.
- **Availability is the larger gain.** Holdout Brier score improved from `0.1462` to `0.1118`, with AUC `0.885`.
- **Conditional minutes also improved.** Holdout MAE fell from `5.028` to `4.827` minutes.
- **The combined hurdle clears the strongest simple baseline.** Fixed 0.50-gated minutes MAE fell from `6.157` to `5.964`; soft expected-minutes RMSE fell from `9.878` to `8.789`.

## Dataset and Grain

- Source: `data/nba_data.csv`, SHA-256 `913b70f6c51666c2d4b731a179a12ef2feff148ed54fbd42ee2b30f10a8ede60`.
- Source coverage: `82,034` appearance rows from 2023-10-24 through 2026-10-25.
- Candidate table: `98,460` unique player/team/game rows across `510` dates and `778` players.
- Holdout: `2025-26`, `35,444` candidates from 2025-10-24 through 2026-05-18.
- OOF feature table: `75,208` rows; chronology violations `0`.

## Validation Results

| Metric | Model | Baseline | Improvement |
|---|---:|---:|---:|
| Availability Brier (lower is better) | 0.1118 | 0.1462 | 0.0344 |
| Conditional minutes MAE | 4.827 | 5.028 | 0.200 min |
| Hard-gated all-candidate MAE | 5.964 | 6.157 | 0.193 min |
| Soft expected-minutes RMSE | 8.789 | 9.878 | 1.089 min |

All four gains have positive 95% date-clustered bootstrap intervals. The availability calibration deciles have a maximum absolute predicted-versus-observed gap of `0.009`.

## Prop-Pool Proxy

Among `19,398` holdout candidates with prior L10 minutes at least 15 and prior availability at least 80%, availability Brier improved by `0.0152`, conditional minutes MAE improved by `0.137` minutes, hard-gated MAE improved by `0.250` minutes, and soft RMSE improved by `1.041` minutes. This is the closest historical proxy to players likely to receive a prop line, but it is not a true archived PrizePicks board.

## Decision and Next Step

Use `data/availability_minutes_oof_predictions.csv`—not the full-fit deployment model—to join availability probability and minutes predictions into historical prop training. Keep `models/availability_minutes_model.pkl` disabled for live use until the retrained prop stack beats its current comparator and same-prop baselines on a separate chronological test.

## Caveats

- Historical candidates are inferred from prior same-team appearances because the repository has no historical transaction/roster feed.
- Returns after more than 45 days and players before their second appearance for a team are outside the candidate contract.
- The fixed 0.50 hard gate was not optimized on the holdout.
- The model improves prediction components; it does not by itself prove better prop selection or betting performance.
