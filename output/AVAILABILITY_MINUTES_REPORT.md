# Availability and Minutes Model Validation

## Executive Summary

- **The model is ready to become an input to prop-model retraining.** It passed all four expanding development folds and the latest-season holdout at the candidate player/team/game grain.
- **Availability is the larger gain.** Holdout Brier score improved from `0.1462` to `0.1115`, with AUC `0.886`.
- **Conditional minutes also improved.** Holdout MAE fell from `5.028` to `4.788` minutes.
- **The combined hurdle clears the strongest simple baseline.** Fixed 0.50-gated minutes MAE fell from `6.157` to `5.903`; soft expected-minutes RMSE fell from `9.878` to `8.728`.

## Dataset and Grain

- Source: `data/nba_data.csv`, SHA-256 `ea6dded12deb3b693ebde7c2338f1c887d8ab2db6302cabb618b8860735f3ebd`.
- Source coverage: `81,065` appearance rows from 2023-10-24 through 2026-05-18.
- Candidate table: `98,217` unique player/team/game rows across `508` dates and `762` players.
- Holdout: `2025-26`, `35,444` candidates from 2025-10-24 through 2026-05-18.
- OOF feature table: `74,965` rows; chronology violations `0`.

## Validation Results

| Metric | Model | Baseline | Improvement |
|---|---:|---:|---:|
| Availability Brier (lower is better) | 0.1115 | 0.1462 | 0.0347 |
| Conditional minutes MAE | 4.788 | 5.028 | 0.240 min |
| Hard-gated all-candidate MAE | 5.903 | 6.157 | 0.255 min |
| Soft expected-minutes RMSE | 8.728 | 9.878 | 1.150 min |

All four gains have positive 95% date-clustered bootstrap intervals. The availability calibration deciles have a maximum absolute predicted-versus-observed gap of `0.023`.

## Prop-Pool Proxy

Among `19,398` holdout candidates with prior L10 minutes at least 15 and prior availability at least 80%, availability Brier improved by `0.0158`, conditional minutes MAE improved by `0.159` minutes, hard-gated MAE improved by `0.283` minutes, and soft RMSE improved by `1.111` minutes. This is the closest historical proxy to players likely to receive a prop line, but it is not a true archived PrizePicks board.

## Decision and Next Step

Use `data/availability_minutes_oof_predictions.csv`—not the full-fit deployment model—to join availability probability and minutes predictions into historical prop training. Keep `models/availability_minutes_model.pkl` disabled for live use until the retrained prop stack beats its current comparator and same-prop baselines on a separate chronological test.

## Caveats

- Historical candidates are inferred from prior same-team appearances because the repository has no historical transaction/roster feed.
- Returns after more than 45 days and players before their second appearance for a team are outside the candidate contract.
- The fixed 0.50 hard gate was not optimized on the holdout.
- The model improves prediction components; it does not by itself prove better prop selection or betting performance.
