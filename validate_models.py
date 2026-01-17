"""
Model Validation and Analysis Script
Comprehensive validation of trained models to ensure no leakage and good performance
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_models():
    """Load all trained models."""
    models = {}
    props = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa']
    
    for prop in props:
        model_path = f'model_{prop}.pkl'
        if Path(model_path).exists():
            with open(model_path, 'rb') as f:
                models[prop] = pickle.load(f)
    
    return models


def check_feature_leakage(models):
    """Check if any models have leaky features."""
    print("="*60)
    print("FEATURE LEAKAGE CHECK")
    print("="*60)
    
    leaky_features = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'mp', 'fg', 'fga', 
                      'fg_pct', '3p', '3pa', '3p_pct', 'ft', 'fta', 'ft_pct',
                      'FANTASY_PTS', 'PLUS_MINUS', 'gmsc', 'orb', 'drb']
    
    all_clean = True
    for prop, model_data in models.items():
        features = model_data['features']
        found_leaky = [f for f in features if f in leaky_features]
        
        if found_leaky:
            print(f"\n⚠️ {prop.upper()}: Found leaky features!")
            for f in found_leaky:
                print(f"  - {f}")
            all_clean = False
    
    if all_clean:
        print("\n✓ All models clean - no leaky features detected!")
    
    return all_clean


def analyze_predictions():
    """Analyze prediction files for patterns."""
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    props = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa']
    
    summary = []
    
    for prop in props:
        pred_file = f'predictions_{prop}.csv'
        if not Path(pred_file).exists():
            continue
        
        df = pd.read_csv(pred_file)
        
        # Calculate metrics
        errors = df['actual'] - df['predicted']
        abs_errors = np.abs(errors)
        
        # Check if predictions are too close to actuals (sign of leakage)
        correlation = df['actual'].corr(df['predicted'])
        
        # Betting accuracy (did we predict over/under correctly?)
        pred_over = df['predicted'] > df['line']
        actual_over = df['actual'] > df['line']
        betting_acc = (pred_over == actual_over).mean()
        
        summary.append({
            'prop': prop.upper(),
            'n_predictions': len(df),
            'mae': abs_errors.mean(),
            'rmse': np.sqrt((errors**2).mean()),
            'correlation': correlation,
            'betting_accuracy': betting_acc,
            'mean_actual': df['actual'].mean(),
            'mean_predicted': df['predicted'].mean(),
            'mean_line': df['line'].mean()
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.to_string(index=False))
    
    # Check for suspicious patterns
    print("\n" + "="*60)
    print("LEAKAGE INDICATORS")
    print("="*60)
    
    suspicious = []
    for _, row in summary_df.iterrows():
        if row['correlation'] > 0.85:
            suspicious.append(f"{row['prop']}: Very high correlation ({row['correlation']:.3f})")
        if row['betting_accuracy'] > 0.65:
            suspicious.append(f"{row['prop']}: Unrealistic betting accuracy ({row['betting_accuracy']:.1%})")
    
    if suspicious:
        print("\n⚠️ Potential issues found:")
        for s in suspicious:
            print(f"  - {s}")
    else:
        print("\n✓ No suspicious patterns detected!")
    
    return summary_df


def analyze_feature_importance(models):
    """Analyze which features are most important across models."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    all_importance = {}
    
    for prop, model_data in models.items():
        if 'feature_importance' in model_data and model_data['feature_importance']:
            print(f"\n{prop.upper()} - Top 5 features:")
            for feat, imp in model_data['feature_importance'][:5]:
                print(f"  {feat}: {imp:.4f}")
                
                # Aggregate across all models
                if feat not in all_importance:
                    all_importance[feat] = []
                all_importance[feat].append(imp)
    
    # Find most commonly important features
    print("\n" + "="*60)
    print("MOST IMPORTANT FEATURES ACROSS ALL MODELS")
    print("="*60)
    
    avg_importance = {feat: np.mean(imps) for feat, imps in all_importance.items()}
    top_features = sorted(avg_importance.items(), key=lambda x: -x[1])[:15]
    
    print("\nTop 15 features by average importance:")
    for i, (feat, imp) in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat}: {imp:.4f}")


def check_temporal_consistency():
    """Check if predictions are consistent over time."""
    print("\n" + "="*60)
    print("TEMPORAL CONSISTENCY CHECK")
    print("="*60)
    
    # Load predictions for PTS (as example)
    df = pd.read_csv('predictions_pts.csv', parse_dates=['game_date'])
    df = df.sort_values('game_date')
    
    # Group by date and calculate daily metrics
    daily_metrics = df.groupby('game_date').apply(lambda x: pd.Series({
        'mae': np.abs(x['actual'] - x['predicted']).mean(),
        'betting_acc': ((x['predicted'] > x['line']) == (x['actual'] > x['line'])).mean(),
        'n_games': len(x)
    })).reset_index()
    
    print(f"\nDaily MAE statistics:")
    print(f"  Mean: {daily_metrics['mae'].mean():.2f}")
    print(f"  Std: {daily_metrics['mae'].std():.2f}")
    print(f"  Min: {daily_metrics['mae'].min():.2f}")
    print(f"  Max: {daily_metrics['mae'].max():.2f}")
    
    print(f"\nDaily Betting Accuracy statistics:")
    print(f"  Mean: {daily_metrics['betting_acc'].mean():.1%}")
    print(f"  Std: {daily_metrics['betting_acc'].std():.3f}")
    print(f"  Min: {daily_metrics['betting_acc'].min():.1%}")
    print(f"  Max: {daily_metrics['betting_acc'].max():.1%}")
    
    # Check for drift
    if daily_metrics['mae'].std() > 2.0:
        print("\n⚠️ High variance in daily MAE - model may be unstable")
    else:
        print("\n✓ Stable performance over time")


def generate_validation_report():
    """Generate a comprehensive validation report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL VALIDATION REPORT")
    print("="*80)
    
    models = load_all_models()
    
    if not models:
        print("❌ No models found! Run task6_train_models.py first.")
        return
    
    print(f"\nLoaded {len(models)} models")
    
    # Run all checks
    leakage_clean = check_feature_leakage(models)
    summary_df = analyze_predictions()
    analyze_feature_importance(models)
    check_temporal_consistency()
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if leakage_clean:
        print("✓ No feature leakage detected")
    else:
        print("❌ Feature leakage detected - fix required!")
    
    # Check if betting accuracies are realistic
    realistic_betting = all(summary_df['betting_accuracy'] < 0.65)
    if realistic_betting:
        print("✓ Betting accuracies are realistic (52-65%)")
    else:
        print("❌ Some betting accuracies are unrealistically high!")
    
    # Check if correlations are reasonable
    reasonable_corr = all(summary_df['correlation'] < 0.85)
    if reasonable_corr:
        print("✓ Prediction correlations are reasonable")
    else:
        print("⚠️ Some correlations are very high - possible overfitting")
    
    if leakage_clean and realistic_betting:
        print("\n" + "="*80)
        print("🎉 ALL CHECKS PASSED - MODELS ARE READY FOR PRODUCTION!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️ SOME ISSUES DETECTED - REVIEW REQUIRED")
        print("="*80)


if __name__ == "__main__":
    generate_validation_report()
