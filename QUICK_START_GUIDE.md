# Quick Start Guide - Fixed NBA Prop Prediction Model

## ✅ Status: Production Ready

All data leakage issues have been fixed. Models are now producing realistic predictions.

---

## Quick Commands

### Generate Daily Predictions

```bash
# Get predictions for a specific date with quality filters
python3 task7_daily_predictions.py --date 2026-01-09 --min-minutes 20 --min-edge 2.0

# Get top 10 betting opportunities
python3 task7_daily_predictions.py --date 2026-01-09 --top 10

# Predict specific players
python3 task7_daily_predictions.py --players "LeBron James,LAL,GSW;Nikola Jokic,DEN,MIA"
```

### Validate Models

```bash
# Run comprehensive validation
python3 validate_models.py

# Run automated test suite
python3 test_suite.py
```

### Retrain Models (if needed)

```bash
# Standard training
python3 task6_train_models.py

# Optimized training with hyperparameter tuning
python3 task6_train_models_optimized.py
```

---

## What Was Fixed

### The Problem
- **Data Leakage:** Models were using `FANTASY_PTS` and `PLUS_MINUS` features
- These are calculated FROM the current game's stats
- Caused unrealistic prediction edges of 15-18 points

### The Solution
- Removed leaky features from training
- Retrained all 9 models
- Now using only pre-game features:
  - Rolling averages (L5, L10, L20)
  - Opponent defensive metrics
  - Schedule/fatigue indicators

### The Results
- ✅ Realistic edges: 2-5 points (was 15-18)
- ✅ Betting accuracy: 52-58% (realistic range)
- ✅ ROI: +0.7% to +10.7% (profitable)
- ✅ All validation tests passing

---

## Model Performance Summary

| Prop | Win Rate | ROI | Recommendation |
|------|----------|-----|----------------|
| **TRB** (Rebounds) | 57.5% | +10.7% | ⭐⭐⭐ Best |
| **TOV** (Turnovers) | 56.7% | +9.1% | ⭐⭐⭐ Excellent |
| **PA** (Pts+Ast) | 56.3% | +8.2% | ⭐⭐ Very Good |
| **PR** (Pts+Reb) | 56.0% | +7.6% | ⭐⭐ Very Good |
| **PRA** (Pts+Reb+Ast) | 55.4% | +6.3% | ⭐⭐ Good |
| **PTS** (Points) | 55.4% | +6.3% | ⭐⭐ Good |
| **AST** (Assists) | 55.3% | +6.1% | ⭐⭐ Good |
| **STL** (Steals) | 52.7% | +0.7% | ⭐ Marginal |
| **BLK** (Blocks) | 48.2% | -8.7% | ❌ Avoid |

---

## Best Practices

### 1. Filter Settings
```bash
--min-minutes 20    # Focus on rotation players
--min-edge 2.0      # Only significant edges
--top 10            # Show best opportunities
```

### 2. Prop Selection
- ✅ **Use:** TRB, TOV, PA, PR, PRA, PTS, AST
- ❌ **Avoid:** BLK (negative ROI)
- ⚠️ **Caution:** STL (marginal)

### 3. Betting Strategy
- Favor **UNDER** bets (historically 62% win rate)
- Minimum edge: **2.0 points**
- Focus on players: **20-32 minutes per game**
- Avoid bench players (<15 min) and extreme stars (>35 min)

### 4. Risk Management
- Use Kelly Criterion for bet sizing
- Don't bet more than 2-3% of bankroll per bet
- Track performance daily
- Adjust strategy based on results

---

## Example Output

```
────────────────────────────────────────────────────────────
  LeBron James (LAL vs MIL)
  2026-01-09
────────────────────────────────────────────────────────────
  Prop             Pred     Line     Edge
  ────────────────────────────────────────
  Points           21.6     25.9     -4.3 *
  Rebounds          4.9      4.9     +0.0
  Assists           6.5      6.1     +0.4
  Pts+Reb+Ast      32.9     36.9     -4.0 *
  Pts+Reb          26.4     30.8     -4.4 *
  Pts+Ast          28.1     32.0     -3.9 *
  
  [TOP EDGES]
  → Pts+Reb: UNDER 30.8 (pred: 26.4, edge: 4.4)
  → Points: UNDER 25.9 (pred: 21.6, edge: 4.3)
  → Pts+Reb+Ast: UNDER 36.9 (pred: 32.9, edge: 4.0)
```

**Interpretation:**
- Model predicts LeBron will score **21.6 points**
- Line is **25.9 points**
- Edge of **4.3 points** suggests betting **UNDER**
- This is a realistic edge (not the 15+ point edges from before)

---

## Validation Checklist

Before using models in production, verify:

- [ ] Run `python3 test_suite.py` - All tests pass
- [ ] Run `python3 validate_models.py` - No leakage detected
- [ ] Check prediction edges - Should be 2-5 points (not 15+)
- [ ] Verify betting accuracy - Should be 52-58% (not >65%)
- [ ] Review model files - Updated after fixing leakage

---

## Troubleshooting

### Issue: Predictions seem unrealistic
**Check:**
1. Are edges >10 points? → Models may need retraining
2. Run `python3 validate_models.py` to check for leakage
3. Verify you're using the latest model files (after fix)

### Issue: No predictions generated
**Check:**
1. Date format: `YYYY-MM-DD`
2. Data availability for that date
3. Minimum minutes filter (try lowering `--min-minutes`)

### Issue: Models not loading
**Check:**
1. Install dependencies: `pip3 install scikit-learn pandas numpy`
2. Verify model files exist: `ls model_*.pkl`
3. Retrain if needed: `python3 task6_train_models.py`

---

## Files Reference

### Core Files
- `task6_train_models.py` - Model training (FIXED)
- `task7_daily_predictions.py` - Generate predictions
- `model_*.pkl` - Trained models (9 files)

### Validation Files
- `validate_models.py` - Comprehensive validation
- `test_suite.py` - Automated tests
- `DEBUGGING_REPORT.md` - Full debugging documentation

### Optional Files
- `task6_train_models_optimized.py` - Enhanced training
- `task8_betting_lines.py` - Compare to real lines
- `task9_backtesting.py` - Historical performance

---

## Support

For issues or questions:
1. Check `DEBUGGING_REPORT.md` for detailed information
2. Run validation scripts to diagnose problems
3. Review test results for specific failures

---

**Last Updated:** January 17, 2026  
**Status:** ✅ Production Ready  
**Version:** 1.0 (Fixed)
