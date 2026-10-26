# Project Status — NBA Prop Predictor (Aug 2026)

**Last Updated:** 2026-08-23 — Diagnosis: Plumbing fixed, edge not yet proven.

---

## The Core Insight

We have **successfully repaired the data pipeline and validation discipline**. The remaining constraint is **proving incremental selection edge against the correct baseline (always-UNDER) and demonstrating that edge transfers to live PrizePicks markets**.

### System Health Summary

| Category | Status | Evidence |
|----------|--------|----------|
| **Data Ingestion** | ✅ Fixed | 300k+ sportsbook observations, proper alternate-line keys, high price-pair completeness |
| **Train/Live Consistency** | ✅ Fixed | All 97 active features verified across 73,531 historical rows; feature parity audit passed |
| **Feature Engineering** | ✅ Solid | Team defense opponent-adjusted & pace-neutral; availability model AUC=0.886 on holdout |
| **Validation Discipline** | ✅ Rigorous | Chronological folds, always-UNDER benchmarks, confidence intervals, untouched holdouts, 103 tests passing |
| **Incremental Model Edge** | ❌ Unproven | No availability/defense candidate showed confident improvement vs. production; 7/9 candidates lost to always-UNDER |
| **Selection vs. Prediction** | ⚠️ Weak Link | Model predicts stats reasonably (e.g., PTS MAE=4.08, TRB MAE=1.75) but struggles to select bets; sportsbook holdout: 56/100 vs. T−60 market 61/100 |
| **Market Signal Transfer** | ❌ Unproven | Best verified signal is T−60 sportsbook, but no paired PrizePicks T−60 data yet |
| **Availability Downstream** | ⚠️ Limited | Standalone AUC=0.886 but only 2/9 props retained availability fields; resulting improvements were too small to confirm |
| **Code Duplication** | ⚠️ Risk | Duplicate training/inference paths; stale live state; legacy documentation |

---

## What We've Actually Learned

### What Works
1. **Box score ingestion & team logs:** Live box scores, robust fallback to NBA API, complete team rosters and game logs.
2. **Possess-based features:** PPP, Usage %, Pace—all correctly calculated, opponent-adjusted, and shift-lagged.
3. **Minutes forecasting:** Availability model beats rolling baselines; reliably predicts who plays and for how long.
4. **Garbage-time filtering:** Blowout discounts and floor-time cutoffs remove bench garbage and preserve signal.
5. **Always-UNDER as a benchmark:** It's a powerful, simple baseline. Any model that doesn't beat it on the same hold-out set isn't adding value.

### What Doesn't (Yet)
1. **Feature selection for props:** Availability and opponent-adjusted defense mostly didn't survive selection. Only STL/BLK retained a few features, and even then duplicated always-UNDER decisions.
2. **Projection → Selection gap:** Predicting a stat well ≠ knowing when the offered line is wrong. The model's predictions are reasonable, but its bet rankings miss the T−60 market signal.
3. **Sportsbook → PrizePicks transfer:** The T−60 sportsbook no-vig probability clears chronological holdout. But we don't yet have paired historical PrizePicks T−60 lines. Until we do, the signal is unproven on the actual betting surface.
4. **Stat-bias entanglement:** STL/BLK lines are biased UNDER historically. A model selecting UNDERs can appear "right" while adding zero value. Every evaluation must isolate signal from bias.

---

## Open Questions

1. **Why did availability and defense fail downstream?**
   - Are the features themselves weak?
   - Or is the model architecture (current calibration, feature interaction) unable to leverage them?
   - Or are the training/holdout samples too small or misaligned?

2. **Can T−60 sportsbook edge transfer to PrizePicks?**
   - PrizePicks lines are not identical to sportsbook no-vig probabilities.
   - Until we have paired historical data, we cannot calibrate that transfer.

3. **Is there hidden leakage or selection bias in the current pipeline?**
   - All 103 tests pass, but edge still marginal.
   - Possible: silent time-ordering issues, train/holdout overlap, or undetected stale state.

4. **Should we retire some props?** 
   - PTS is 50.5% on real lines—below always-UNDER (~52%). Disable it?
   - BLK has high accuracy (74%) but mostly on UNDER picks (bias). Is there true edge?

---

## Critical Path Forward

### Near-term (Production Bet Confidence)
The real bottleneck is **paired forward PrizePicks + sportsbook T−60 data**. Without it, any edge claim remains hypothetical.

1. **Build paired forward panel:**
   - Capture PrizePicks line + sportsbook no-vig equivalent at T−60 for 100+ days.
   - Compare model selection against both.
   - Isolate PrizePicks-specific edge (if any).

2. **Always-UNDER gate:** 
   - Any prop that loses to always-UNDER on holdout should be disabled or heavily de-ranked.

3. **STL availability ablation:**
   - Isolate why availability didn't help STL prop despite AUC=0.886.
   - Is it a feature interaction? Calibration? Or is the edge simply too small?

### Offseason (Structural Cleanup)
1. **Remove duplicate systems:** Consolidate all training/inference into the canonical pipeline.
2. **Stale live-state audit:** Ensure every prop input is fresh (not cached from yesterday).
3. **Feature path cleanup:** Delete legacy prop inputs; document the canonical feature set.

---

## Metrics That Matter

- **Holdout Win Rate** (chronological fold, real-line, same-row vs. always-UNDER)
- **Confidence Interval Width** (narrow = high confidence)
- **Paired Forward Return** (PrizePicks actual 2-week ROI after accounting for hold)
- **Edge Attribution** (how much of win rate is the model vs. stat bias?)

---

## Open Risks

1. **Silent leakage:** All tests pass, but edge is marginal. Possible: undetected time-ordering, train/holdout contamination, or stale features.
2. **Market drift:** What works in April 2026 may not work in September 2026. Forward testing is essential.
3. **Selection vs. prediction gap:** The model can forecast stats but struggles to rank bets. This may be structural—predictions ≠ valuations.
4. **Props to retire:** PTS (50.5%), TOV (56.3%), and others below 52% should be disabled to avoid negative ROI.

---

## Success Criteria (End of Offseason)

- [ ] Paired forward PrizePicks + sportsbook T−60 panel running (100+ days).
- [ ] STL availability ablation complete; decision on feature retention made.
- [ ] Always-UNDER gates in place for all props below 52% on holdout.
- [ ] All duplicate training/inference paths removed; stale state audit complete.
- [ ] 6-week forward test showing >52% win rate and >0% ROI before accounting for hold.
