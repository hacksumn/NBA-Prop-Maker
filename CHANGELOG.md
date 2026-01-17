# Changelog - NBA Prop Prediction Model

## [1.1.0] - 2026-01-17 - CRITICAL FIX

### 🔴 Critical Bug Fix: Data Leakage

**Issue:** Models were using current-game statistics (FANTASY_PTS, PLUS_MINUS) as features, causing unrealistic predictions.

**Impact:**
- Prediction edges: 15-18 points (unrealistic)
- Model essentially "seeing" the outcome it was predicting
- Not usable for real betting scenarios

**Fix:**
- Removed FANTASY_PTS and PLUS_MINUS from feature set
- Retrained all 9 models with clean features
- Added comprehensive validation suite

**Results After Fix:**
- Prediction edges: 2-5 points (realistic)
- Betting accuracy: 52-58% (profitable range)
- ROI: +0.7% to +10.7%
- All validation tests passing ✅

### Changed Files

#### Modified
- `task6_train_models.py` - Fixed feature exclusion list

#### Added
- `validate_models.py` - Comprehensive model validation
- `test_suite.py` - Automated test suite (6 tests)
- `task6_train_models_optimized.py` - Enhanced training with hyperparameter tuning
- `debug_leakage.py` - Diagnostic script for identifying leakage
- `debug_leakage2.py` - Additional diagnostic script
- `DEBUGGING_REPORT.md` - Full documentation of debugging process
- `QUICK_START_GUIDE.md` - User guide for fixed models
- `CHANGELOG.md` - This file

#### Updated
- All `model_*.pkl` files - Retrained without leaky features
- All `predictions_*.csv` files - New predictions generated

### Validation Results

```
✅ TEST 1: No Leaky Features - PASSED
✅ TEST 2: Rolling Features Shifted - PASSED
✅ TEST 3: Realistic Predictions - PASSED
✅ TEST 4: Edge Distribution - PASSED
✅ TEST 5: Model Performance - PASSED
✅ TEST 6: Prediction Consistency - PASSED
```

### Model Performance (After Fix)

| Prop | Before | After | Status |
|------|--------|-------|--------|
| TRB | N/A | 57.5% win, +10.7% ROI | ✅ Excellent |
| TOV | N/A | 56.7% win, +9.1% ROI | ✅ Excellent |
| PA | N/A | 56.3% win, +8.2% ROI | ✅ Very Good |
| PR | N/A | 56.0% win, +7.6% ROI | ✅ Very Good |
| PRA | N/A | 55.4% win, +6.3% ROI | ✅ Good |
| PTS | N/A | 55.4% win, +6.3% ROI | ✅ Good |
| AST | N/A | 55.3% win, +6.1% ROI | ✅ Good |
| STL | N/A | 52.7% win, +0.7% ROI | ⚠️ Marginal |
| BLK | N/A | 48.2% win, -8.7% ROI | ❌ Avoid |

### Breaking Changes

None - API and usage remain the same.

### Migration Guide

If you have old model files:
1. Delete old `model_*.pkl` files
2. Run `python3 task6_train_models.py` to retrain
3. Verify with `python3 test_suite.py`

### Recommendations

**Use these props:**
- TRB, TOV, PA, PR, PRA, PTS, AST

**Avoid these props:**
- BLK (negative ROI)

**Best practices:**
- Minimum edge: 2.0 points
- Minimum minutes: 20 per game
- Favor UNDER bets

---

## [1.0.0] - 2025-02-07 - Initial Release

Initial release with 9 trained models and prediction pipeline.

**Note:** This version had data leakage issues. Use v1.1.0 or later.

---

**Current Version:** 1.1.0  
**Status:** ✅ Production Ready  
**Last Updated:** January 17, 2026
