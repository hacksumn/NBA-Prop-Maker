# Sportsbook CLV Expansion and Walk-Forward Report

Date: August 13, 2026

## Executive Summary

- **Accept T-60 no-vig as a sportsbook signal.** It went `61/100` on the untouched holdout versus `56/100` for current model-edge ranking and `51.94%` for matched random.
- **Do not turn it on for PrizePicks yet.** The winning test graded the sportsbook consensus line, not the PrizePicks line being wagered; paired source transfer remains the only production blocker.
- **Use the remaining `13,436` credits only at T-60.** The live request costs at most nine credits per game, supporting at least `1,492` paired game captures. Paid T-5 collection is excluded because T-5 failed the holdout; free direct PrizePicks T-5 capture continues for audit value.
- **Do not call the entire model useless.** Other market fields failed as incremental selectors in this test, but that does not test away the projection, minutes, injury, or eligibility components.

## The panel is large and clean enough for the sportsbook decision

- Window: 2026-02-23 through 2026-03-29.
- Games: 264 across 35 completed dates.
- Requested snapshots: 528 T-60/T-5 event snapshots.
- Completed with data: 528/528.
- Parsed sportsbook rows: 314,162.
- Paired over/under prices: 314,054 (99.97%).
- Box-score join: 24,462 / 25,801 player/prop/date keys (94.81%).
- Physical duplicates, date mismatches, post-request snapshots, post-tip snapshots: zero.
- Expansion spend recorded by checkpoint: 42,129 credits.
- API-reported balance: 13,436 credits.

## The comparison preserves chronology and an untouched future period

- Combined panel: 42,790 rows across 80 chronological dates.
- Eligible rows: PTS/TRB/AST consensus UNDER, at least two books, at least 90% paired-price completeness, graded non-push outcome.
- Development: 54 dates.
- Expanding validation: 24 dates / 119 top-five selections.
- Untouched holdout: latest 20 dates carrying archived model predictions / exactly 100 top-five selections.
- Random control: 2,000 date/prop-matched random cards.
- Current comparator: archived model prediction ranked by consensus-line minus model prediction.

## T-60 no-vig wins; T-5 and the remaining market fields do not

| Snapshot | Method | Holdout | Matched random | Always-UNDER pool | Decision |
|---|---:|---:|---:|---:|---|
| T-60 | No-vig overlay | 61/100 (61%) | 51.94% | 52.43% | Accept as sportsbook signal |
| T-60 | Current model edge | 56/100 (56%) | 53.51% | 52.43% | Comparator |
| T-5 | No-vig overlay | 52/100 (52%) | 50.77% | 52.60% | Reject |
| T-5 | Current model edge | 55/100 (55%) | 54.72% | 52.60% | Comparator |

T-60 no-vig delivered `+9.07pp` over its matched-random mean with a one-sided simulation probability of `3.45%`. It beat the current model on 9 dates, tied on 6, and lost on 5. Holdout stability improved from 58% in the first ten dates to 64% in the last ten.

Validation evidence for T-60 no-vig: `+21.9pp` mean top-five lift, positive in `3/4` folds, `+0.0374` mean AUC lift, 24 dates, and 119 selections. It was the only group retained. T-5 movement appeared positive before no-vig was included but added `-2.47pp` afterward, so it was rejected. Consensus line, book dispersion, and availability also failed the sequential gate.

## PrizePicks transfer is the remaining boundary

This accepted label is UNDER against the sportsbook consensus line. The live card is bet at PrizePicks, whose line can differ. Therefore the result is strong enough to guide the next test but not sufficient to silently turn on the feature for PrizePicks production.

The required remaining observation is now exact: direct PrizePicks T-60 standard/non-promo lines paired with T-60 sportsbook no-vig consensus. The direct PrizePicks watcher is installed. The remaining 13,436 Odds API credits should be used only for the matching T-60 book panel; T-5 has already failed and does not need more density.

## The collector now protects the remaining balance

The API balance is 8,555 credits below the arithmetic expectation from the prior recorded balance and the expansion checkpoint. The strongest current hypothesis is overlapping Python executors left alive by two shell-command timeouts; the later atomic-file contention supports that explanation. Request-level dashboard history is required to prove it. The executor now uses a PID lock and retries atomic Windows replacements, preventing a recurrence.

## Recommended next steps

1. Let the daily watcher collect direct PrizePicks T-60/T-5 and the paired sportsbook panel at T-60 when NBA games resume.
2. Freeze standard, non-promo PrizePicks eligibility before outcomes and retain every eligible row rather than only eventual selections.
3. After the forward sample reaches 20 dates and 100 selections, compare the T-60 overlay with current model ranking, matched random, and always-UNDER on the actual PrizePicks line.
4. Enable the sportsbook shadow feature in production only if the paired PrizePicks holdout passes the same chronological gate.

## Further questions

- Does the T-60 no-vig ordering retain its lift when PrizePicks and sportsbook lines disagree?
- Is transfer stable by PTS, TRB, and AST, or concentrated in one market?
- Does PrizePicks standard/non-promo availability supply five eligible selections per date often enough to reach the holdout target on schedule?

## Caveats and assumptions

- The `61/100` result applies to sportsbook-consensus UNDER outcomes across 20 historical dates; it is not yet a same-line PrizePicks result.
- The current model's `56/100` is a meaningful comparator but not proof that every internal model component adds value independently.
- The nine-credit live ceiling assumes one region and nine requested markets; actual cost is lower when markets are unavailable and is recorded from API response headers.
- The unexplained 8,555-credit gap remains a hypothesis until reconciled against request-level account history.

Sources: `output/clv_panel_expansion_audit.json`, `output/clv_features_expansion_summary.json`, `output/sportsbook_expansion_summary.json`, `output/sportsbook_expansion_feature_selection.csv`, `output/sportsbook_expansion_holdout_comparison.csv`, and `output/sportsbook_expansion_holdout_selections.csv`.
