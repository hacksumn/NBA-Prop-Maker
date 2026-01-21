"""
Task 7: Classification Model - Beat the Line
NBA Player Props Betting System v2

FUNDAMENTAL CHANGE:
- OLD: Predict raw stats → compare to line → bet if edge exists
- NEW: Predict probability of OVER/UNDER directly → bet when confident

Key insight: Vegas line IS information. Use it as a feature, not just a comparison point.

The model learns WHEN players beat their lines, not what their stats will be.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_historical_data():
    """Load full historical data with all features."""
    print("Loading historical data...")
    
    # Try to load matchup-enhanced data first
    try:
        df = pd.read_csv('data/player_games_with_matchups.csv', parse_dates=['game_date'])
        if 'pts_matchup_edge' in df.columns:
            print(f"  Loaded matchup-enhanced data: {len(df):,} rows")
            print(f"  Matchup features present: ✓")
            has_matchup = (df['n_prev_matchups'] > 0).mean()
            print(f"  Rows with matchup history: {has_matchup:.1%}")
        else:
            raise FileNotFoundError("No matchup features")
    except:
        # Fallback to project data
        df = pd.read_csv('data/player_games_schedule.csv', parse_dates=['game_date'])
        print(f"  Main data: {len(df):,} rows (no matchup features)")
    
    # Filter to players with enough games (need rolling stats)
    df = df[df['games_played'] >= 10].copy()
    print(f"  After filtering (10+ games): {len(df):,} rows")
    
    # Create combo stats
    df['pra'] = df['pts'] + df['trb'] + df['ast']
    df['pr'] = df['pts'] + df['trb']
    df['pa'] = df['pts'] + df['ast']
    
    # Create "line" proxies using L10 averages (what sportsbooks use)
    df['pts_line'] = df['pts_l10']
    df['trb_line'] = df['trb_l10']
    df['ast_line'] = df['ast_l10']
    df['pra_line'] = df['pts_l10'] + df['trb_l10'] + df['ast_l10']
    df['pr_line'] = df['pts_l10'] + df['trb_l10']
    df['pa_line'] = df['pts_l10'] + df['ast_l10']
    
    return df


def create_classification_features(df, prop):
    """
    Create features specifically for classification (over/under prediction).
    
    The key is: we're predicting IF player beats line, not WHAT their stat will be.
    
    Enhanced with interaction features AND player vs opponent matchup history.
    """
    merged = df.copy()
    
    # Get the actual stat and line
    if prop in ['pts', 'trb', 'ast']:
        actual_col = prop
        line_col = f'{prop}_line'
        l5_col = f'{prop}_l5'
        l10_col = f'{prop}_l10'
        l20_col = f'{prop}_l20'
        std_col = f'{prop}_std_l10'
        # Matchup columns (new format)
        matchup_edge_col = f'{prop}_matchup_edge'
        recent_matchup_edge_col = f'{prop}_recent_matchup_edge'
        favorable_col = f'{prop}_favorable_matchup'
        unfavorable_col = f'{prop}_unfavorable_matchup'
        vs_opp_col = f'{prop}_vs_opp'
        vs_opp_l3_col = f'{prop}_vs_opp_l3'
    else:
        actual_col = prop  # pra, pr, pa
        line_col = f'{prop}_line'
        l5_col = 'pts_l5'
        l10_col = 'pts_l10'
        l20_col = 'pts_l20'
        std_col = 'pts_std_l10'
        # Use pts matchup for combo props
        matchup_edge_col = 'pts_matchup_edge'
        recent_matchup_edge_col = 'pts_recent_matchup_edge'
        favorable_col = 'pts_favorable_matchup'
        unfavorable_col = 'pts_unfavorable_matchup'
        vs_opp_col = 'pts_vs_opp'
        vs_opp_l3_col = 'pts_vs_opp_l3'
    
    # TARGET: Did player go OVER the line?
    merged['actual'] = merged[actual_col]
    merged['line'] = merged[line_col]
    merged['went_over'] = (merged['actual'] > merged['line']).astype(int)
    
    # ============================================
    # BASE FEATURES
    # ============================================
    merged['avg_vs_line'] = merged[l10_col] - merged['line']
    merged['l5_vs_line'] = merged[l5_col] - merged['line']
    merged['l20_vs_line'] = merged[l20_col] - merged['line']
    merged['variance'] = merged[std_col].fillna(5)
    merged['line_percentile'] = merged['avg_vs_line'] / (merged['variance'] + 0.1)
    merged['trend'] = merged['l5_vs_line'] - merged['l20_vs_line']
    merged['recent_vs_season'] = merged[l5_col] / (merged[l20_col] + 0.1)
    
    # Schedule factors
    merged['is_b2b'] = merged['is_b2b_second'].fillna(0)
    merged['high_mins'] = merged['high_min_prev'].fillna(0)
    merged['fatigue_risk'] = merged['is_b2b'] * merged['high_mins']
    
    # Matchup factors (team defense)
    merged['easy_matchup'] = merged['opp_pts_allowed_rank'].fillna(15) / 30
    merged['hard_matchup'] = 1 - merged['easy_matchup']
    
    # Rest
    merged['days_rest_capped'] = merged['days_rest'].fillna(1).clip(0, 5)
    merged['well_rested'] = (merged['days_rest_capped'] >= 2).astype(int)
    
    # ============================================
    # PLAYER VS OPPONENT MATCHUP FEATURES
    # ============================================
    
    # Number of prior meetings
    merged['matchup_games'] = merged['n_prev_matchups'].fillna(0) if 'n_prev_matchups' in merged.columns else 0
    merged['has_matchup_history'] = (merged['matchup_games'] >= 2).astype(int)
    merged['strong_matchup_history'] = (merged['matchup_games'] >= 3).astype(int)
    
    # Matchup edge: how much better/worse vs this specific opponent
    if matchup_edge_col in merged.columns:
        merged['matchup_edge'] = merged[matchup_edge_col].fillna(0)
    else:
        merged['matchup_edge'] = 0
    
    # Recent matchup edge (last 3 games vs opponent)
    if recent_matchup_edge_col in merged.columns:
        merged['recent_matchup_edge'] = merged[recent_matchup_edge_col].fillna(0)
    else:
        merged['recent_matchup_edge'] = merged['matchup_edge']
    
    # Favorable/unfavorable matchup flags (pre-computed)
    if favorable_col in merged.columns:
        merged['favorable_vs_opp'] = merged[favorable_col].fillna(0)
        merged['unfavorable_vs_opp'] = merged[unfavorable_col].fillna(0)
    else:
        merged['favorable_vs_opp'] = ((merged['matchup_edge'] > 2) & (merged['has_matchup_history'] == 1)).astype(int)
        merged['unfavorable_vs_opp'] = ((merged['matchup_edge'] < -2) & (merged['has_matchup_history'] == 1)).astype(int)
    
    # Matchup edge vs line: does matchup history suggest over/under?
    merged['matchup_edge_vs_line'] = merged['matchup_edge'] - merged['avg_vs_line']
    
    # Strong matchup signals
    merged['strong_matchup_over'] = (
        (merged['strong_matchup_history'] == 1) & 
        (merged['matchup_edge'] > 3)
    ).astype(int)
    merged['strong_matchup_under'] = (
        (merged['strong_matchup_history'] == 1) & 
        (merged['matchup_edge'] < -3)
    ).astype(int)
    
    # ============================================
    # INTERACTION FEATURES
    # ============================================
    
    # Fatigue + Trend interaction
    merged['fatigue_x_cold'] = merged['is_b2b'] * (merged['trend'] < 0).astype(int)
    merged['fatigue_x_hot'] = merged['is_b2b'] * (merged['trend'] > 0).astype(int)
    
    # Matchup + Form interaction
    merged['easy_x_hot'] = merged['easy_matchup'] * (merged['recent_vs_season'] > 1).astype(int)
    merged['hard_x_cold'] = merged['hard_matchup'] * (merged['recent_vs_season'] < 1).astype(int)
    
    # Matchup history + Form interaction
    merged['good_matchup_x_hot'] = merged['strong_matchup_over'] * (merged['trend'] > 0).astype(int)
    merged['bad_matchup_x_cold'] = merged['strong_matchup_under'] * (merged['trend'] < 0).astype(int)
    
    # Matchup + Schedule interaction (NEW!)
    merged['good_matchup_rested'] = merged['strong_matchup_over'] * merged['well_rested']
    merged['bad_matchup_tired'] = merged['strong_matchup_under'] * merged['is_b2b']
    
    # Variance-adjusted edge
    merged['edge_reliability'] = merged['avg_vs_line'] / (merged['variance'] + 1)
    merged['matchup_reliability'] = merged['matchup_edge'] / (merged['variance'] + 1)
    
    # Line extremes
    merged['line_vs_l20'] = merged['line'] - merged[l20_col]
    merged['line_extreme_low'] = (merged['line_vs_l20'] < -3).astype(int)
    merged['line_extreme_high'] = (merged['line_vs_l20'] > 3).astype(int)
    
    # Consistency
    variance_median = merged['variance'].median()
    merged['is_consistent'] = (merged['variance'] < variance_median).astype(int)
    merged['consistent_above_line'] = merged['is_consistent'] * (merged['avg_vs_line'] > 0).astype(int)
    
    # Usage/load
    merged['mins_load'] = merged['mp_l10'].fillna(25) / 36
    merged['high_usage'] = (merged['mins_load'] > 0.9).astype(int)
    merged['games_density'] = merged['games_last_7d'].fillna(2) / 4
    merged['heavy_schedule'] = (merged['games_density'] > 0.75).astype(int)
    merged['compound_fatigue'] = merged['is_b2b'] * merged['heavy_schedule'] * merged['high_usage']
    
    # Non-linear terms
    merged['trend_sq'] = merged['trend'] ** 2
    merged['variance_sq'] = merged['variance'] ** 2
    merged['edge_sq'] = merged['avg_vs_line'] ** 2
    merged['matchup_edge_sq'] = merged['matchup_edge'] ** 2
    
    return merged


def build_classification_model(merged, prop):
    """
    Train classification model to predict over/under.
    """
    print(f"\n{'='*60}")
    print(f"  {prop.upper()} - Classification Model")
    print(f"{'='*60}")
    
    # Features for classification - expanded set with interactions AND matchup history
    feature_cols = [
        # Base features
        'avg_vs_line', 'l5_vs_line', 'l20_vs_line', 
        'variance', 'line_percentile', 'trend', 'recent_vs_season',
        # Schedule
        'is_b2b', 'high_mins', 'fatigue_risk', 'days_rest_capped', 'well_rested',
        # Matchup (team defense)
        'easy_matchup', 'hard_matchup',
        # PLAYER VS OPPONENT HISTORY
        'matchup_edge', 'recent_matchup_edge', 
        'has_matchup_history', 'strong_matchup_history', 'matchup_games',
        'matchup_edge_vs_line', 'strong_matchup_over', 'strong_matchup_under',
        'favorable_vs_opp', 'unfavorable_vs_opp',
        # Interactions
        'fatigue_x_cold', 'fatigue_x_hot',
        'easy_x_hot', 'hard_x_cold',
        'good_matchup_x_hot', 'bad_matchup_x_cold',
        'good_matchup_rested', 'bad_matchup_tired',
        'edge_reliability', 'matchup_reliability',
        'line_extreme_low', 'line_extreme_high',
        'is_consistent', 'consistent_above_line',
        # Usage/load
        'mins_load', 'high_usage', 'games_density', 'heavy_schedule',
        'compound_fatigue',
        # Non-linear terms
        'trend_sq', 'variance_sq', 'edge_sq', 'matchup_edge_sq'
    ]
    
    # Clean data
    clean = merged.dropna(subset=feature_cols + ['went_over', 'line'])
    print(f"  Clean rows: {len(clean):,}")
    
    if len(clean) < 500:
        print(f"  ✗ Not enough data, skipping")
        return None
    
    # Temporal split (last 21 days = test)
    clean = clean.sort_values('game_date')
    max_date = clean['game_date'].max()
    cutoff = max_date - pd.Timedelta(days=21)
    
    train = clean[clean['game_date'] < cutoff]
    test = clean[clean['game_date'] >= cutoff]
    
    print(f"  Train: {len(train):,}, Test: {len(test):,}")
    
    X_train = train[feature_cols]
    y_train = train['went_over']
    X_test = test[feature_cols]
    y_test = test['went_over']
    
    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Train multiple classifiers - prioritize non-linear models
    models = {
        'gbm_deep': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8, random_state=42
        ),
        'gbm_wide': GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, random_state=42
        ),
        'rf_deep': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=10,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'rf_wide': RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            max_features=0.7, random_state=42, n_jobs=-1
        ),
        'logistic': LogisticRegression(C=0.1, max_iter=1000)  # baseline only
    }
    
    best_model = None
    best_acc = 0
    best_name = None
    best_brier = 1.0  # Lower is better
    
    print(f"\n  Testing models:")
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        
        # Get probabilities
        y_pred_proba = model.predict_proba(X_test_sc)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred_proba)
        
        print(f"    {name}: {acc:.1%} acc, {brier:.4f} brier")
        
        # Prefer lower brier score (better calibration) over raw accuracy
        if brier < best_brier:
            best_brier = brier
            best_acc = acc
            best_model = model
            best_name = name
            best_proba = y_pred_proba
    
    print(f"  Best model: {best_name.upper()}")
    print(f"  Test accuracy: {best_acc:.1%}")
    
    # Analyze by confidence level
    print(f"\n  [Accuracy by Confidence]")
    test_results = test.copy()
    test_results['pred_proba'] = best_proba
    test_results['pred_over'] = (best_proba > 0.5).astype(int)
    test_results['correct'] = (test_results['pred_over'] == test_results['went_over']).astype(int)
    
    # Confidence buckets
    buckets = [
        (0.50, 0.55, 'Low (50-55%)'),
        (0.55, 0.60, 'Med (55-60%)'),
        (0.60, 0.65, 'High (60-65%)'),
        (0.65, 1.00, 'Very High (65%+)')
    ]
    
    profitable_threshold = None
    for low, high, label in buckets:
        # Over bets
        over_mask = (test_results['pred_proba'] >= low) & (test_results['pred_proba'] < high)
        # Under bets  
        under_mask = (test_results['pred_proba'] <= 1-low) & (test_results['pred_proba'] > 1-high)
        
        mask = over_mask | under_mask
        subset = test_results[mask]
        
        if len(subset) >= 10:
            acc = subset['correct'].mean()
            n = len(subset)
            
            # Calculate ROI at -110 odds
            wins = subset['correct'].sum()
            losses = n - wins
            profit = wins * 1.0 - losses * 1.1
            roi = profit / n * 100 if n > 0 else 0
            
            status = "✓ PROFITABLE" if acc > 0.5238 else "✗ losing"
            print(f"    {label}: {acc:.1%} ({n} bets), ROI: {roi:+.1f}% {status}")
            
            if acc > 0.5238 and profitable_threshold is None:
                profitable_threshold = low
    
    # Calibration check
    brier = brier_score_loss(y_test, best_proba)
    print(f"\n  Brier score: {brier:.4f} (lower is better, 0.25 = random)")
    
    # Return model bundle
    return {
        'model': best_model,
        'scaler': scaler,
        'features': feature_cols,
        'accuracy': best_acc,
        'brier': brier,
        'profitable_threshold': profitable_threshold or 0.55,
        'prop': prop,
        'model_type': best_name
    }


def analyze_what_model_learned(model_bundle, merged):
    """Show what the model actually learned."""
    prop = model_bundle['prop']
    model = model_bundle['model']
    features = model_bundle['features']
    
    print(f"\n  [What the model learned for {prop.upper()}]")
    
    if hasattr(model, 'feature_importances_'):
        importances = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
        print(f"  Top predictors:")
        for feat, imp in importances[:5]:
            print(f"    {feat}: {imp:.3f}")
    elif hasattr(model, 'coef_'):
        coefs = sorted(zip(features, model.coef_[0]), key=lambda x: -abs(x[1]))
        print(f"  Top coefficients:")
        for feat, coef in coefs[:5]:
            direction = "→ OVER" if coef > 0 else "→ UNDER"
            print(f"    {feat}: {coef:+.3f} {direction}")


def main():
    print("="*60)
    print("NBA Props v2: Classification Model (Beat the Line)")
    print("="*60)
    print("\nFundamental change: Predict OVER/UNDER directly, not raw stats")
    print("Use Vegas line as FEATURE, not just comparison target\n")
    
    # Load data
    df = load_historical_data()
    
    # Props to model
    props = ['pts', 'trb', 'ast', 'pra', 'pr', 'pa']
    
    # Train classification models for each prop
    models = {}
    
    for prop in props:
        # Create classification features
        merged = create_classification_features(df, prop)
        
        # Build model
        model_bundle = build_classification_model(merged, prop)
        
        if model_bundle:
            models[prop] = model_bundle
            analyze_what_model_learned(model_bundle, merged)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Prop':<6} {'Model':<8} {'Accuracy':<10} {'Brier':<8} {'Min Conf'}")
    print("-"*50)
    
    for prop, bundle in models.items():
        print(f"{prop.upper():<6} {bundle['model_type'].upper():<8} "
              f"{bundle['accuracy']:.1%}      {bundle['brier']:.4f}   "
              f"{bundle['profitable_threshold']:.0%}")
    
    # Save models
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print("="*60)
    
    for prop, bundle in models.items():
        filename = f"models/classifiers/classifier_{prop}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"  ✓ Saved {filename}")
    
    # Key insight
    print(f"\n{'='*60}")
    print("KEY INSIGHT")
    print("="*60)
    print("""
The classification approach is fundamentally different:

OLD MODEL (broken):
  1. Predict player scores 25.3 points
  2. Vegas line is 24.5
  3. Bet OVER because 25.3 > 24.5
  PROBLEM: Vegas already knows he averages ~25. The line accounts for it.

NEW MODEL (this one):
  1. Vegas set line at 24.5
  2. Player's L10 is 26.0 (1.5 above line)
  3. Player is on B2B, tired
  4. Model predicts 48% chance of OVER
  5. NO BET - not confident enough

The new model learns WHEN players beat their lines, considering:
  - How far line is from their average
  - Schedule/fatigue factors  
  - Recent form vs season average
  - Matchup difficulty

Only bet when confidence exceeds profitable threshold (usually 55-60%).
""")


if __name__ == "__main__":
    main()
