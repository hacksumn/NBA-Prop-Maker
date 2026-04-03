"""
NBA Minutes Prediction Model
=============================
Predicts player minutes for upcoming games.

Minutes are the foundation - if you know minutes, you can scale all other predictions.

Usage:
    python minutes_model.py train [data_file]
    python minutes_model.py predict [data_file]
    python minutes_model.py evaluate [data_file]
"""

import pandas as pd
import numpy as np
import pickle
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'models_dir': Path('./models'),
    'min_games': 10,
    'min_minutes': 12,  # Only predict for rotation players
}

CONFIG['models_dir'].mkdir(parents=True, exist_ok=True)


# =============================================================================
# FEATURE ENGINEERING FOR MINUTES
# =============================================================================

def create_minutes_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features specifically optimized for minutes prediction.
    
    Key predictors:
    - Recent minutes (L3, L5, L10, L20)
    - Minutes consistency (low variance = predictable role)
    - Role indicators (starter vs bench)
    - Schedule factors (B2B, rest)
    - Team context (pace, depth)
    - Trend (expanding/shrinking role)
    """
    print("Creating minutes features...")
    
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    grouped = df.groupby('player')
    
    # Games played
    df['games_played'] = grouped.cumcount()
    
    # -------------------------------------------------------------------------
    # 1. Core Minutes Rolling Averages
    # -------------------------------------------------------------------------
    print("  Rolling averages...")
    
    for window in [3, 5, 10, 20]:
        df[f'mp_l{window}'] = grouped['mp'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    
    # EWMA - recent games weighted more
    df['mp_ewma3'] = grouped['mp'].transform(
        lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
    )
    df['mp_ewma5'] = grouped['mp'].transform(
        lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()
    )
    
    # -------------------------------------------------------------------------
    # 2. Minutes Consistency/Variance
    # -------------------------------------------------------------------------
    print("  Consistency metrics...")
    
    # Standard deviation
    df['mp_std_l10'] = grouped['mp'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).std()
    )
    df['mp_std_l20'] = grouped['mp'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).std()
    )
    
    # Coefficient of variation (lower = more consistent)
    df['mp_cv'] = df['mp_std_l10'] / df['mp_l10'].replace(0, np.nan)
    df['mp_cv'] = df['mp_cv'].fillna(0.3).clip(0, 1)
    
    # Min/Max range
    df['mp_min_l10'] = grouped['mp'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).min()
    )
    df['mp_max_l10'] = grouped['mp'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).max()
    )
    df['mp_range_l10'] = df['mp_max_l10'] - df['mp_min_l10']
    
    # -------------------------------------------------------------------------
    # 3. Role Indicators
    # -------------------------------------------------------------------------
    print("  Role indicators...")
    
    # Starter indicator (typically 28+ minutes)
    df['is_starter'] = (df['mp_l10'] >= 28).astype(int)
    
    # Star player indicator (32+ minutes)
    df['is_star'] = (df['mp_l10'] >= 32).astype(int)
    
    # Deep bench indicator (under 15 minutes)
    df['is_bench'] = (df['mp_l10'] < 15).astype(int)
    
    # Role tier (1=star, 2=starter, 3=rotation, 4=deep bench)
    df['role_tier'] = pd.cut(
        df['mp_l10'].fillna(0),
        bins=[-1, 12, 20, 28, 100],
        labels=[4, 3, 2, 1]
    ).astype(float).fillna(3)
    
    # -------------------------------------------------------------------------
    # 4. Trend Features
    # -------------------------------------------------------------------------
    print("  Trend features...")
    
    # Short vs long term (role expanding or shrinking?)
    df['mp_trend_3v10'] = df['mp_l3'] / df['mp_l10'].replace(0, np.nan)
    df['mp_trend_3v10'] = df['mp_trend_3v10'].fillna(1.0)
    
    df['mp_trend_5v20'] = df['mp_l5'] / df['mp_l20'].replace(0, np.nan)
    df['mp_trend_5v20'] = df['mp_trend_5v20'].fillna(1.0)
    
    # EWMA trend
    df['mp_ewma_trend'] = df['mp_ewma3'] / df['mp_ewma5'].replace(0, np.nan)
    df['mp_ewma_trend'] = df['mp_ewma_trend'].fillna(1.0)
    
    # Role change detection
    df['role_expanding'] = (df['mp_trend_3v10'] > 1.1).astype(int)
    df['role_shrinking'] = (df['mp_trend_3v10'] < 0.9).astype(int)
    
    # Linear trend (simplified: L3 - L10 gives direction)
    df['mp_slope'] = (df['mp_l3'] - df['mp_l10']).fillna(0).clip(-5, 5)
    
    # -------------------------------------------------------------------------
    # 5. Schedule Features
    # -------------------------------------------------------------------------
    print("  Schedule features...")
    
    # Days rest
    df['prev_date'] = grouped['game_date'].shift(1)
    df['days_rest'] = (df['game_date'] - df['prev_date']).dt.days
    df['days_rest'] = df['days_rest'].fillna(3).clip(0, 7)
    
    # Back-to-back
    df['is_b2b'] = (df['days_rest'] <= 1).astype(int)
    
    # Games in last 7 days (use days_rest as proxy)
    # If days_rest is 1 or 2, likely playing frequently
    df['games_l7d'] = np.where(df['days_rest'] <= 1, 3, 
                       np.where(df['days_rest'] <= 2, 2, 1))
    
    # Previous game minutes (fatigue from specific game)
    df['prev_mp'] = grouped['mp'].shift(1)
    df['prev_mp'] = df['prev_mp'].fillna(df['mp_l10'])
    
    # High minutes in prev game (might rest today)
    df['high_mp_prev'] = (df['prev_mp'] > 38).astype(int)
    
    # Very high minutes (OT or heavy load)
    df['very_high_mp_prev'] = (df['prev_mp'] > 42).astype(int)
    
    # -------------------------------------------------------------------------
    # 6. Team Context
    # -------------------------------------------------------------------------
    print("  Team context...")
    
    if 'team' in df.columns:
        # Team's average minutes for starters (proxy for pace/style)
        team_mp = df.groupby(['team', 'game_date']).agg({
            'mp': ['mean', 'std', 'max']
        }).reset_index()
        team_mp.columns = ['team', 'game_date', 'team_mp_avg', 'team_mp_std', 'team_mp_max']
        
        # Rolling team averages
        team_mp = team_mp.sort_values(['team', 'game_date'])
        team_grouped = team_mp.groupby('team')
        
        team_mp['team_mp_avg_l10'] = team_grouped['team_mp_avg'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        
        df = df.merge(
            team_mp[['team', 'game_date', 'team_mp_avg_l10']],
            on=['team', 'game_date'],
            how='left'
        )
        df['team_mp_avg_l10'] = df['team_mp_avg_l10'].fillna(22.0)  # League avg
    else:
        df['team_mp_avg_l10'] = 22.0
    
    # -------------------------------------------------------------------------
    # 7. Game Context (Opponent)
    # -------------------------------------------------------------------------
    print("  Opponent context...")
    
    # Opponent pace (fast teams = more possessions)
    if 'fga' in df.columns and 'opp' in df.columns:
        opp_pace = df.groupby(['opp', 'game_date']).agg({
            'fga': 'sum',
            'fta': 'sum' if 'fta' in df.columns else 'count',
            'tov': 'sum' if 'tov' in df.columns else 'count',
        }).reset_index()
        
        # Possessions estimate
        opp_pace['opp_poss'] = opp_pace['fga'] + 0.44 * opp_pace.get('fta', 0) + opp_pace.get('tov', 0)
        opp_pace = opp_pace.sort_values(['opp', 'game_date'])
        
        opp_grouped = opp_pace.groupby('opp')
        opp_pace['opp_pace_l10'] = opp_grouped['opp_poss'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        
        df = df.merge(
            opp_pace[['opp', 'game_date', 'opp_pace_l10']],
            on=['opp', 'game_date'],
            how='left'
        )
        
        # Normalize pace
        pace_mean = df['opp_pace_l10'].mean()
        if pace_mean > 0:
            df['opp_pace_factor'] = df['opp_pace_l10'] / pace_mean
        else:
            df['opp_pace_factor'] = 1.0
        df['opp_pace_factor'] = df['opp_pace_factor'].fillna(1.0)
    else:
        df['opp_pace_factor'] = 1.0
        df['opp_pace_l10'] = 100.0  # Default
    
    # -------------------------------------------------------------------------
    # 8. Home/Away
    # -------------------------------------------------------------------------
    if 'is_home' in df.columns:
        # Need to regroup after adding columns
        grouped = df.groupby('player')
        
        # Home minutes average
        df['mp_home_temp'] = np.where(df['is_home'] == 1, df['mp'], np.nan)
        df['mp_home_avg'] = grouped['mp_home_temp'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        
        # Away minutes average
        df['mp_away_temp'] = np.where(df['is_home'] == 0, df['mp'], np.nan)
        df['mp_away_avg'] = grouped['mp_away_temp'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        
        # Home/away differential
        df['mp_home_diff'] = df['mp_home_avg'] - df['mp_away_avg']
        df['mp_home_diff'] = df['mp_home_diff'].fillna(0)
        
        # Situational average
        df['mp_sit_avg'] = np.where(
            df['is_home'] == 1,
            df['mp_home_avg'],
            df['mp_away_avg']
        )
        df['mp_sit_avg'] = df['mp_sit_avg'].fillna(df['mp_l10'])
    
    # -------------------------------------------------------------------------
    # 9. Composite Predictions
    # -------------------------------------------------------------------------
    print("  Composite features...")
    
    # Best estimate combining multiple signals
    df['mp_best_est'] = (
        df['mp_ewma5'] * 0.35 +
        df['mp_l10'] * 0.30 +
        df.get('mp_sit_avg', df['mp_l10']) * 0.20 +
        df['mp_l20'].fillna(df['mp_l10']) * 0.15
    )
    
    # Confidence score (inverse of CV, scaled by games played)
    games_factor = (df['games_played'] / 15).clip(0, 1)
    df['mp_confidence'] = ((1 - df['mp_cv'].clip(0, 0.5)) * games_factor).clip(0, 1)
    
    # Adjustment for B2B
    df['mp_b2b_adj'] = df['mp_best_est'] * np.where(df['is_b2b'] == 1, 0.97, 1.0)
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    temp_cols = [c for c in df.columns if '_temp' in c]
    df = df.drop(columns=temp_cols + ['prev_date'], errors='ignore')
    
    # Fill NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    feature_count = len([c for c in df.columns if c.startswith('mp_') or c.startswith('is_') or c.startswith('role_') or c in ['days_rest', 'games_played', 'opp_pace_factor', 'team_mp_avg_l10']])
    print(f"  Created {feature_count} minutes-specific features")
    
    return df


def get_minutes_features(df: pd.DataFrame) -> List[str]:
    """Get feature columns for minutes model."""
    
    # Include only minutes-relevant features
    include_prefixes = [
        'mp_l', 'mp_ewma', 'mp_std', 'mp_cv', 'mp_min', 'mp_max', 'mp_range',
        'mp_trend', 'mp_slope', 'mp_home', 'mp_away', 'mp_sit', 'mp_best',
        'mp_confidence', 'mp_b2b',
        'is_starter', 'is_star', 'is_bench', 'is_b2b', 'is_home',
        'role_', 'days_rest', 'games_l7d', 'prev_mp', 'high_mp', 'very_high',
        'team_mp', 'opp_pace', 'games_played',
    ]
    
    exclude = ['mp', 'mp_home_temp', 'mp_away_temp']
    
    features = []
    for col in df.columns:
        if col in exclude:
            continue
        if any(col.startswith(p) for p in include_prefixes):
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                features.append(col)
    
    return features


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_minutes_model(df: pd.DataFrame) -> Dict:
    """Train XGBoost model specifically for minutes prediction."""
    
    try:
        import xgboost as xgb
    except ImportError:
        import os
        os.system('pip install xgboost --break-system-packages -q')
        import xgboost as xgb
    
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    print("\nTraining minutes model...")
    
    # Filter to rotation players with enough games
    df = df[
        (df['games_played'] >= CONFIG['min_games']) &
        (df['mp_l10'] >= CONFIG['min_minutes'])
    ].copy()
    
    print(f"  Training samples: {len(df):,}")
    
    # Sort by date
    df = df.sort_values('game_date').reset_index(drop=True)
    
    # Get features
    feature_cols = get_minutes_features(df)
    print(f"  Features: {len(feature_cols)}")
    
    X = df[feature_cols].fillna(0)
    y = df['mp']
    
    # XGBoost parameters (tuned for minutes)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,  # Shallower to prevent overfitting
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 10,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Baseline comparison (just use L10)
        l10_idx = feature_cols.index('mp_l10') if 'mp_l10' in feature_cols else None
        if l10_idx is not None:
            baseline_pred = X_val.iloc[:, l10_idx]
            baseline_mae = mean_absolute_error(y_val, baseline_pred)
        else:
            baseline_mae = mae
        
        # Accuracy within 3 minutes
        within_3 = (np.abs(y_pred - y_val) <= 3).mean()
        
        # Accuracy within 5 minutes
        within_5 = (np.abs(y_pred - y_val) <= 5).mean()
        
        cv_results.append({
            'fold': fold,
            'mae': mae,
            'rmse': rmse,
            'baseline_mae': baseline_mae,
            'improvement': baseline_mae - mae,
            'within_3': within_3,
            'within_5': within_5,
        })
        
        print(f"    Fold {fold}: MAE={mae:.2f} (baseline={baseline_mae:.2f}, -{baseline_mae-mae:.2f}), ±3min={within_3:.1%}")
    
    # Summary
    avg_mae = np.mean([r['mae'] for r in cv_results])
    avg_baseline = np.mean([r['baseline_mae'] for r in cv_results])
    avg_within_3 = np.mean([r['within_3'] for r in cv_results])
    avg_within_5 = np.mean([r['within_5'] for r in cv_results])
    
    print(f"\n  Average MAE: {avg_mae:.2f} (baseline: {avg_baseline:.2f})")
    print(f"  Improvement over baseline: {avg_baseline - avg_mae:.2f} minutes")
    print(f"  Within 3 min: {avg_within_3:.1%}")
    print(f"  Within 5 min: {avg_within_5:.1%}")
    
    # Train final model
    print("\n  Training final model...")
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X, y, verbose=False)
    
    # Feature importance
    importance = dict(zip(feature_cols, final_model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print("\n  Top 15 features:")
    for feat, imp in top_features:
        print(f"    {feat}: {imp:.4f}")
    
    # Save model
    model_path = CONFIG['models_dir'] / 'minutes_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_cols': feature_cols,
            'params': params,
        }, f)
    
    print(f"\n  Model saved to {model_path}")
    
    # Save results
    results = {
        'mae': avg_mae,
        'baseline_mae': avg_baseline,
        'improvement': avg_baseline - avg_mae,
        'within_3_min': avg_within_3,
        'within_5_min': avg_within_5,
        'top_features': top_features,
        'n_samples': len(df),
        'n_features': len(feature_cols),
    }
    
    results_path = CONFIG['models_dir'] / 'minutes_results.json'
    
    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    return results


# =============================================================================
# PREDICTION
# =============================================================================

def predict_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Generate minutes predictions for players."""
    
    model_path = CONFIG['models_dir'] / 'minutes_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError("No trained model. Run 'train' first.")
    
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    
    model = saved['model']
    feature_cols = saved['feature_cols']
    
    print("\nGenerating minutes predictions...")
    
    # Get latest data per player
    df = df.sort_values('game_date')
    latest = df.groupby('player').last().reset_index()
    
    # Filter to rotation players
    latest = latest[
        (latest['games_played'] >= CONFIG['min_games']) &
        (latest['mp_l10'] >= CONFIG['min_minutes'])
    ]
    
    print(f"  {len(latest)} players")
    
    # Predict
    X = latest[feature_cols].fillna(0)
    predictions = model.predict(X)
    
    # Build output
    results = []
    for i, (_, row) in enumerate(latest.iterrows()):
        pred = predictions[i]
        l10 = row['mp_l10']
        
        results.append({
            'player': row['player'],
            'team': row.get('team', ''),
            'mp_pred': round(pred, 1),
            'mp_l10': round(l10, 1),
            'mp_l5': round(row.get('mp_l5', l10), 1),
            'mp_diff': round(pred - l10, 1),
            'is_starter': int(row.get('is_starter', 0)),
            'is_b2b': int(row.get('is_b2b', 0)),
            'confidence': round(row.get('mp_confidence', 0.5), 2),
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('mp_pred', ascending=False)
    
    return result_df


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(df: pd.DataFrame, test_days: int = 30):
    """Evaluate model on recent games."""
    
    model_path = CONFIG['models_dir'] / 'minutes_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError("No trained model. Run 'train' first.")
    
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    
    model = saved['model']
    feature_cols = saved['feature_cols']
    
    print(f"\nEvaluating on last {test_days} days...")
    
    # Split data
    df = df.sort_values('game_date')
    max_date = df['game_date'].max()
    test_start = max_date - pd.Timedelta(days=test_days)
    
    test_df = df[
        (df['game_date'] >= test_start) &
        (df['games_played'] >= CONFIG['min_games']) &
        (df['mp_l10'] >= CONFIG['min_minutes'])
    ].copy()
    
    print(f"  Test samples: {len(test_df):,}")
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['mp']
    
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = np.mean(np.abs(y_test - y_pred))
    baseline_mae = np.mean(np.abs(y_test - test_df['mp_l10']))
    
    within_3 = (np.abs(y_pred - y_test) <= 3).mean()
    within_5 = (np.abs(y_pred - y_test) <= 5).mean()
    
    print(f"\n  Results:")
    print(f"    MAE: {mae:.2f} (baseline: {baseline_mae:.2f})")
    print(f"    Improvement: {baseline_mae - mae:.2f} minutes")
    print(f"    Within 3 min: {within_3:.1%}")
    print(f"    Within 5 min: {within_5:.1%}")
    
    # By role
    print(f"\n  By role:")
    for role, label in [(1, 'Star (32+)'), (2, 'Starter (28+)'), (3, 'Rotation (20+)'), (4, 'Bench')]:
        mask = test_df['role_tier'] == role
        if mask.sum() > 0:
            role_mae = np.mean(np.abs(y_test[mask] - y_pred[mask]))
            print(f"    {label}: MAE={role_mae:.2f} ({mask.sum()} games)")
    
    return {
        'mae': mae,
        'baseline_mae': baseline_mae,
        'within_3': within_3,
        'within_5': within_5,
    }


# =============================================================================
# MAIN
# =============================================================================

def load_data(path: str = None) -> pd.DataFrame:
    """Load and prepare data."""
    
    if path:
        df = pd.read_csv(path, parse_dates=['game_date'])
    else:
        # Try default locations
        for loc in ['data/nba_data.csv', '../data/nba_data.csv', 'nba_data.csv']:
            if Path(loc).exists():
                df = pd.read_csv(loc, parse_dates=['game_date'])
                break
        else:
            raise FileNotFoundError("No data file found. Provide path as argument.")
    
    print(f"Loaded {len(df):,} rows")
    
    # Standardize columns if needed
    col_map = {
        'PLAYER_NAME': 'player',
        'TEAM_ABBREVIATION': 'team',
        'MIN': 'mp',
        'REB': 'trb',
        'FGA': 'fga',
        'FTA': 'fta',
        'TOV': 'tov',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Ensure is_home exists
    if 'is_home' not in df.columns:
        if 'matchup_raw' in df.columns:
            df['is_home'] = df['matchup_raw'].apply(lambda x: 0 if '@' in str(x) else 1)
        elif 'matchup' in df.columns:
            df['is_home'] = df['matchup'].apply(lambda x: 0 if '@' in str(x) else 1)
        else:
            df['is_home'] = 1
    
    # Ensure opp exists
    if 'opp' not in df.columns:
        if 'matchup_raw' in df.columns:
            df['opp'] = df['matchup_raw'].apply(lambda x: str(x).split()[-1])
        elif 'matchup' in df.columns:
            df['opp'] = df['matchup'].apply(lambda x: str(x).split()[-1])
    
    return df


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands: train, predict, evaluate")
        return
    
    command = sys.argv[1].lower()
    data_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("\n" + "="*60)
    print(f"MINUTES MODEL - {command.upper()}")
    print("="*60)
    
    # Load data
    df = load_data(data_path)
    
    # Create features
    df = create_minutes_features(df)
    
    if command == 'train':
        train_minutes_model(df)
        
    elif command == 'predict':
        preds = predict_minutes(df)
        print("\nTop 20 predicted minutes:")
        print(preds.head(20).to_string(index=False))
        
        # Save
        output_path = Path('./output/minutes_predictions.csv')
        output_path.parent.mkdir(exist_ok=True)
        preds.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
        
    elif command == 'evaluate':
        test_days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        evaluate_model(df, test_days)
        
    else:
        print(f"Unknown command: {command}")
        print("Commands: train, predict, evaluate")


if __name__ == "__main__":
    main()
