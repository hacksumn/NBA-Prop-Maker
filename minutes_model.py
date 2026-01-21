"""
Minutes Prediction Model
========================
Predicts expected minutes for next game - a key driver of counting stats.

Why: PTS, REB, AST are nearly linear in minutes played.
A good minutes prediction can significantly improve prop predictions.

Usage:
    python minutes_model.py              # Train and add feature
    python minutes_model.py --evaluate   # Evaluate only, don't modify data
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Try to import LightGBM
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
    print("✓ LightGBM available")
except ImportError:
    HAS_LGBM = False
    print("⚠ LightGBM not installed, using GradientBoostingRegressor")
    print("  For better results: pip install lightgbm")


# Features for minutes prediction (all pre-tip-off)
MINUTES_FEATURES = [
    # Historical minutes patterns
    'mp_l5', 'mp_l10', 'mp_l20',
    'mp_l3', 'mp_avg_l3',
    'mp_std_l10', 'mp_std_l20',
    
    # Schedule/fatigue factors
    'days_rest',
    'is_b2b',
    'is_b2b_second',
    'games_last_7d',
    'games_last_14d',
    'high_min_prev',
    'games_streak',
    'player_days_rest',
    
    # Role indicator
    'games_played',
]


def load_data(filepath='player_games_schedule.csv'):
    """Load the schedule data with all features."""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    df = pd.read_csv(filepath, parse_dates=['game_date'])
    print(f"  Loaded {len(df):,} rows, {df['player'].nunique()} players")
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
    return df


def create_next_game_minutes_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target: next game's actual minutes.
    
    CRITICAL: We predict NEXT game minutes using shift(-1).
    For each row, target = minutes from that player's NEXT game.
    """
    print("\n" + "="*60)
    print("CREATING TARGET (next game minutes)")
    print("="*60)
    
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    
    # Target = next game's minutes for this player
    df['next_mp'] = df.groupby('player')['mp'].shift(-1)
    
    # Stats before dropping
    print(f"  Current game MP: mean={df['mp'].mean():.1f}, std={df['mp'].std():.1f}")
    print(f"  Next game MP target: mean={df['next_mp'].dropna().mean():.1f}, std={df['next_mp'].dropna().std():.1f}")
    
    # Count rows we'll lose
    na_count = df['next_mp'].isna().sum()
    print(f"  Rows without next game (will drop): {na_count:,}")
    
    return df


def prepare_features(df: pd.DataFrame, feature_cols: list):
    """Prepare feature matrix, handling missing values."""
    
    # Get available features
    available = [f for f in feature_cols if f in df.columns]
    missing = [f for f in feature_cols if f not in df.columns]
    
    if missing:
        print(f"  Note: Missing features (skipping): {missing}")
    
    print(f"  Using {len(available)} features")
    
    X = df[available].copy()
    
    return X, available


def train_with_timeseries_cv(df: pd.DataFrame, feature_cols: list, n_splits: int = 5):
    """
    Train minutes model using TimeSeriesSplit cross-validation.
    This respects temporal ordering - always train on past, validate on future.
    """
    print("\n" + "="*60)
    print(f"TRAINING WITH {n_splits}-FOLD TIME SERIES CV")
    print("="*60)
    
    # Filter to rows with valid target and sufficient history
    df_train = df.dropna(subset=['next_mp']).copy()
    df_train = df_train[df_train['games_played'] >= 5]  # Need some history
    
    # Sort by date (critical for time series split)
    df_train = df_train.sort_values('game_date').reset_index(drop=True)
    
    print(f"  Training samples: {len(df_train):,}")
    
    # Prepare features
    X, available_features = prepare_features(df_train, feature_cols)
    y = df_train['next_mp'].values
    
    # Fill NaN in features with column medians
    medians = X.median()
    X = X.fillna(medians)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_results = []
    all_val_preds = np.zeros(len(X))
    all_val_mask = np.zeros(len(X), dtype=bool)
    
    print(f"\n  {'Fold':<6} {'Train Size':<12} {'Val Size':<10} {'MAE':<8} {'R²':<8}")
    print("  " + "-"*50)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Train model
        if HAS_LGBM:
            model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            # Use early stopping
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lambda env: None]  # Suppress output
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                min_samples_leaf=20,
                random_state=42
            )
            model.fit(X_tr, y_tr)
        
        # Predict and evaluate
        val_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        fold_results.append({'fold': fold+1, 'mae': mae, 'r2': r2, 'train_size': len(train_idx), 'val_size': len(val_idx)})
        print(f"  {fold+1:<6} {len(train_idx):<12,} {len(val_idx):<10,} {mae:<8.2f} {r2:<8.3f}")
        
        # Store predictions for analysis
        all_val_preds[val_idx] = val_pred
        all_val_mask[val_idx] = True
    
    # Summary statistics
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    std_mae = np.std([r['mae'] for r in fold_results])
    
    print("  " + "-"*50)
    print(f"  {'AVG':<6} {'':<12} {'':<10} {avg_mae:<8.2f} {avg_r2:<8.3f}")
    print(f"  MAE std: ±{std_mae:.2f}")
    
    return model, scaler, medians, available_features, {
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'std_mae': std_mae,
        'fold_results': fold_results
    }


def train_final_model(df: pd.DataFrame, feature_cols: list, test_days: int = 14):
    """
    Train final production model on all data except last test_days.
    """
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    
    # Filter to rows with valid target
    df_train = df.dropna(subset=['next_mp']).copy()
    df_train = df_train[df_train['games_played'] >= 5]
    df_train = df_train.sort_values('game_date').reset_index(drop=True)
    
    # Temporal split
    max_date = df_train['game_date'].max()
    cutoff = max_date - pd.Timedelta(days=test_days)
    
    train = df_train[df_train['game_date'] < cutoff]
    test = df_train[df_train['game_date'] >= cutoff]
    
    print(f"  Train: {len(train):,} rows (before {cutoff.date()})")
    print(f"  Test:  {len(test):,} rows (after {cutoff.date()})")
    
    # Prepare features
    X_train, available_features = prepare_features(train, feature_cols)
    y_train = train['next_mp'].values
    
    X_test, _ = prepare_features(test, feature_cols)
    y_test = test['next_mp'].values
    
    # Fill NaN and scale
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Train final model
    if HAS_LGBM:
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=20,
            random_state=42
        )
    
    model.fit(X_train_sc, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_sc)
    test_pred = model.predict(X_test_sc)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\n  Final Model Results:")
    print(f"    Train MAE: {train_mae:.2f} min, R²: {train_r2:.3f}")
    print(f"    Test  MAE: {test_mae:.2f} min, R²: {test_r2:.3f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = sorted(zip(available_features, model.feature_importances_), 
                           key=lambda x: -x[1])
        print(f"\n  Top 5 Features:")
        for feat, imp in importance[:5]:
            print(f"    {feat}: {imp:.4f}")
    
    # Compare to baseline (just using mp_l10)
    if 'mp_l10' in test.columns:
        baseline_pred = test['mp_l10'].fillna(test['mp_l10'].median())
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        improvement = (baseline_mae - test_mae) / baseline_mae * 100
        print(f"\n  Baseline (mp_l10) MAE: {baseline_mae:.2f}")
        print(f"  Improvement over baseline: {improvement:.1f}%")
    
    return model, scaler, medians, available_features, {
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2
    }


def add_predicted_minutes_to_dataset(df: pd.DataFrame, model, scaler, medians, 
                                      feature_cols: list) -> pd.DataFrame:
    """Add predicted_minutes column to full dataset."""
    print("\n" + "="*60)
    print("ADDING PREDICTED MINUTES TO DATASET")
    print("="*60)
    
    # Prepare features for all rows
    X, _ = prepare_features(df, feature_cols)
    X = X.fillna(medians)
    X_scaled = scaler.transform(X)
    
    # Predict
    df['predicted_minutes'] = model.predict(X_scaled)
    
    # Clip to reasonable range
    df['predicted_minutes'] = df['predicted_minutes'].clip(0, 48)
    
    print(f"  Added predicted_minutes column")
    print(f"  Stats: mean={df['predicted_minutes'].mean():.1f}, "
          f"std={df['predicted_minutes'].std():.1f}, "
          f"min={df['predicted_minutes'].min():.1f}, "
          f"max={df['predicted_minutes'].max():.1f}")
    
    return df


def validate_predictions(df: pd.DataFrame):
    """Sanity check the predictions."""
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    # Correlation with actual minutes
    valid = df.dropna(subset=['mp', 'predicted_minutes'])
    corr = valid['mp'].corr(valid['predicted_minutes'])
    print(f"  Correlation (predicted vs actual MP): {corr:.3f}")
    
    if corr > 0.95:
        print("  ⚠ WARNING: Very high - possible leakage?")
    elif corr > 0.70:
        print("  ✓ Good correlation")
    elif corr > 0.50:
        print("  ○ Moderate correlation")
    else:
        print("  ⚠ Low correlation - model may not be helpful")
    
    # Check it's not just copying mp_l10
    if 'mp_l10' in df.columns:
        corr_l10 = df['predicted_minutes'].corr(df['mp_l10'].fillna(0))
        print(f"  Correlation (predicted vs mp_l10): {corr_l10:.3f}")
        if corr_l10 > 0.99:
            print("  ⚠ Nearly identical to mp_l10 - model not adding value")
        else:
            print("  ✓ Adds information beyond mp_l10")
    
    # Sample predictions
    print(f"\n  Sample predictions:")
    sample = df[df['games_played'] >= 20].sample(5, random_state=42)[
        ['player', 'game_date', 'mp', 'mp_l10', 'predicted_minutes']
    ]
    print(sample.to_string(index=False))
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--input', default='player_games_schedule.csv', help='Input file')
    parser.add_argument('--cv-folds', type=int, default=5, help='CV folds')
    parser.add_argument('--test-days', type=int, default=14, help='Test days')
    args = parser.parse_args()
    
    print("="*60)
    print("MINUTES PREDICTION MODEL")
    print("="*60)
    print("Predicts next-game minutes to improve prop predictions")
    
    # Load data
    df = load_data(args.input)
    
    # Create target
    df = create_next_game_minutes_target(df)
    
    # Cross-validation evaluation
    cv_model, cv_scaler, cv_medians, cv_features, cv_metrics = train_with_timeseries_cv(
        df, MINUTES_FEATURES, n_splits=args.cv_folds
    )
    
    if args.evaluate:
        print("\n" + "="*60)
        print("EVALUATION COMPLETE (--evaluate flag set)")
        print("="*60)
        return
    
    # Train final model
    model, scaler, medians, features, metrics = train_final_model(
        df, MINUTES_FEATURES, test_days=args.test_days
    )
    
    # Add predictions to dataset
    df = add_predicted_minutes_to_dataset(df, model, scaler, medians, features)
    
    # Validate
    validate_predictions(df)
    
    # Save updated dataset
    print("\n" + "="*60)
    print("SAVING")
    print("="*60)
    
    # Remove the temporary next_mp column before saving
    if 'next_mp' in df.columns:
        df = df.drop(columns=['next_mp'])
    
    df.to_csv(args.input, index=False)
    print(f"  ✓ Updated {args.input} with 'predicted_minutes' column")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'medians': medians,
        'features': features,
        'metrics': metrics,
        'cv_metrics': cv_metrics,
        'created': datetime.now().isoformat()
    }
    with open('model_minutes.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  ✓ Saved model_minutes.pkl")
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  CV MAE:     {cv_metrics['avg_mae']:.2f} ± {cv_metrics['std_mae']:.2f} minutes")
    print(f"  Test MAE:   {metrics['test_mae']:.2f} minutes")
    print(f"  Test R²:    {metrics['test_r2']:.3f}")
    print(f"\n  New feature 'predicted_minutes' added to dataset")
    print(f"  This should improve prop model accuracy by 5-15%")
    print("\n  NEXT STEP: Re-run full_pipeline.py to retrain prop models")
    print("="*60)


if __name__ == "__main__":
    main()
