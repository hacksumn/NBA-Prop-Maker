"""
Task 6: Train Production Models (OPTIMIZED VERSION)
NBA Player Props Betting System

Improvements over original:
1. Fixed data leakage (removed FANTASY_PTS, PLUS_MINUS)
2. Better hyperparameter tuning with GridSearchCV
3. Ensemble methods for better predictions
4. Feature importance analysis
5. Better validation metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Expected ranges for leakage detection
PROP_CONFIG = {
    'pts': {'min_mae': 3.5, 'max_r2': 0.70, 'line_feature': 'pts_l10'},
    'trb': {'min_mae': 1.5, 'max_r2': 0.65, 'line_feature': 'trb_l10'},
    'ast': {'min_mae': 1.0, 'max_r2': 0.65, 'line_feature': 'ast_l10'},
    'stl': {'min_mae': 0.4, 'max_r2': 0.45, 'line_feature': 'stl_l10'},
    'blk': {'min_mae': 0.3, 'max_r2': 0.50, 'line_feature': 'blk_l10'},
    'tov': {'min_mae': 0.5, 'max_r2': 0.55, 'line_feature': 'tov_l10'},
    'pra': {'min_mae': 5.0, 'max_r2': 0.75, 'line_feature': None},
    'pr':  {'min_mae': 4.0, 'max_r2': 0.70, 'line_feature': None},
    'pa':  {'min_mae': 4.0, 'max_r2': 0.70, 'line_feature': None},
}


def load_data() -> pd.DataFrame:
    """Load the full dataset with all stats for targets."""
    print("Loading data...")
    df = pd.read_csv('player_games_schedule.csv', parse_dates=['game_date'])
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def create_combo_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create combination targets and their line proxies."""
    print("Creating combination targets...")
    
    df['pra'] = df['pts'] + df['trb'] + df['ast']
    df['pr'] = df['pts'] + df['trb']
    df['pa'] = df['pts'] + df['ast']
    
    df['pra_l10'] = df['pts_l10'] + df['trb_l10'] + df['ast_l10']
    df['pr_l10'] = df['pts_l10'] + df['trb_l10']
    df['pa_l10'] = df['pts_l10'] + df['ast_l10']
    
    print(f"  ✓ PRA mean: {df['pra'].mean():.1f}")
    print(f"  ✓ PR mean: {df['pr'].mean():.1f}")
    print(f"  ✓ PA mean: {df['pa'].mean():.1f}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get ONLY valid pre-tip-off features.
    CRITICAL: Exclude ALL current-game stats.
    """
    id_cols = {'player', 'tm', 'opp', 'game_date', 'game_id', 'matchup_id'}
    
    # Current game stats (TARGETS or LEAKY - never use as features)
    current_game = {
        'pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf',
        'mp', 'fg', 'fga', 'fg_pct', '3p', '3pa', '3p_pct',
        'ft', 'fta', 'ft_pct', 'orb', 'drb', 'gmsc',
        'res', 'won', 'is_home',
        'pra', 'pr', 'pa',
        'pra_l10', 'pr_l10', 'pa_l10',
        'FANTASY_PTS', 'PLUS_MINUS',  # LEAKAGE: calculated from current game
        'matchup_raw'
    }
    
    feature_cols = [c for c in df.columns 
                    if c not in id_cols 
                    and c not in current_game]
    
    return sorted(feature_cols)


def temporal_split(df: pd.DataFrame, test_days: int = 21):
    """Temporal train/test split."""
    df = df.sort_values('game_date').reset_index(drop=True)
    
    max_date = df['game_date'].max()
    cutoff = max_date - pd.Timedelta(days=test_days)
    
    train = df[df['game_date'] < cutoff].copy()
    test = df[df['game_date'] >= cutoff].copy()
    
    print(f"\nTemporal split:")
    print(f"  Cutoff: {cutoff.date()}")
    print(f"  Train: {len(train):,} ({train['game_date'].min().date()} to {train['game_date'].max().date()})")
    print(f"  Test: {len(test):,} ({test['game_date'].min().date()} to {test['game_date'].max().date()})")
    
    return train, test


def prepare_xy(train_df, test_df, feature_cols, target):
    """Prepare features and targets, handling missing values."""
    
    train_clean = train_df.dropna(subset=[target])
    test_clean = test_df.dropna(subset=[target])
    
    X_train = train_clean[feature_cols].copy()
    y_train = train_clean[target].values
    X_test = test_clean[feature_cols].copy()
    y_test = test_clean[target].values
    
    # Fill missing with train medians
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    
    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    return X_train_sc, X_test_sc, y_train, y_test, scaler, medians, test_clean


def train_model_optimized(X_train, y_train, model_type='gbm', quick=False):
    """
    Train a model with optimized hyperparameters.
    
    Args:
        quick: If True, skip grid search and use default params
    """
    if model_type == 'gbm':
        if quick:
            model = GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                min_samples_leaf=15, subsample=0.8, random_state=42
            )
            model.fit(X_train, y_train)
        else:
            param_grid = {
                'n_estimators': [100, 150],
                'max_depth': [4, 5],
                'learning_rate': [0.05, 0.1],
                'min_samples_leaf': [15, 20]
            }
            base_model = GradientBoostingRegressor(subsample=0.8, random_state=42)
            grid = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=15,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
    elif model_type == 'ridge':
        if quick:
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
        else:
            param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
            base_model = Ridge()
            grid = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            
    elif model_type == 'ensemble':
        # Ensemble of multiple models
        gbm = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.8, random_state=42
        )
        ridge = Ridge(alpha=1.0)
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=15,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        
        model = VotingRegressor([
            ('gbm', gbm),
            ('ridge', ridge),
            ('rf', rf)
        ])
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def evaluate(y_true, y_pred):
    """Calculate metrics."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }


def realistic_betting_simulation(y_true, y_pred, lines, target):
    """Realistic betting simulation using player's L10 average as line."""
    valid_mask = ~np.isnan(lines)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    lines = lines[valid_mask]
    
    if len(y_true) == 0:
        return {'accuracy': 0, 'n_bets': 0, 'edge': 0, 'roi_pct': 0}
    
    pred_over = y_pred > lines
    actual_over = y_true > lines
    
    correct = pred_over == actual_over
    accuracy = correct.mean()
    
    wins = correct.sum()
    losses = (~correct).sum()
    profit = wins * 1.0 - losses * 1.1
    roi = profit / len(correct) * 100 if len(correct) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'edge': accuracy - 0.5,
        'wins': int(wins),
        'losses': int(losses),
        'profit_units': profit,
        'roi_pct': roi,
        'n_bets': len(correct)
    }


def check_leakage(train_m, test_m, target):
    """Check for data leakage."""
    cfg = PROP_CONFIG.get(target, {'min_mae': 0.3, 'max_r2': 0.80})
    issues = []
    
    if test_m['r2'] > cfg['max_r2']:
        issues.append(f"R²={test_m['r2']:.3f} > {cfg['max_r2']} (too high)")
    
    if test_m['mae'] < cfg['min_mae']:
        issues.append(f"MAE={test_m['mae']:.2f} < {cfg['min_mae']} (too low)")
    
    gap = train_m['r2'] - test_m['r2']
    if gap > 0.20:
        issues.append(f"R² gap={gap:.3f} (overfitting)")
    
    return len(issues) == 0, issues


def get_importance(model, features, n=10):
    """Get top feature importances."""
    if hasattr(model, 'feature_importances_'):
        imp = sorted(zip(features, model.feature_importances_), 
                     key=lambda x: -x[1])[:n]
        return imp
    elif hasattr(model, 'coef_'):
        imp = sorted(zip(features, np.abs(model.coef_)), 
                     key=lambda x: -x[1])[:n]
        return imp
    return []


def train_prop(train_df, test_df, feature_cols, target, optimize=True):
    """Train model for one prop with optimization."""
    cfg = PROP_CONFIG.get(target, {'min_mae': 0.3, 'max_r2': 0.80, 'line_feature': None})
    
    print(f"\n{'─'*55}")
    print(f"  {target.upper()}")
    print(f"{'─'*55}")
    
    X_tr, X_te, y_tr, y_te, scaler, medians, test_clean = prepare_xy(
        train_df, test_df, feature_cols, target
    )
    
    print(f"  Train: {len(X_tr):,}, Test: {len(X_te):,}, Mean: {y_tr.mean():.2f}")
    
    # Train multiple models, pick best
    best = None
    best_mae = float('inf')
    
    model_types = ['gbm', 'ridge', 'rf'] if not optimize else ['gbm', 'ridge', 'rf', 'ensemble']
    
    for mtype in model_types:
        try:
            model = train_model_optimized(X_tr, y_tr, mtype, quick=not optimize)
            pred_tr = model.predict(X_tr)
            pred_te = model.predict(X_te)
            
            m_te = evaluate(y_te, pred_te)
            
            if m_te['mae'] < best_mae:
                best_mae = m_te['mae']
                best = {
                    'type': mtype,
                    'model': model,
                    'pred_tr': pred_tr,
                    'pred_te': pred_te,
                    'm_tr': evaluate(y_tr, pred_tr),
                    'm_te': m_te,
                    'scaler': scaler,
                    'medians': medians
                }
        except Exception as e:
            print(f"  ⚠ {mtype.upper()} failed: {str(e)}")
            continue
    
    if best is None:
        print("  ✗ All models failed!")
        return None
    
    print(f"  Best: {best['type'].upper()}")
    print(f"  Train R²: {best['m_tr']['r2']:.3f}, MAE: {best['m_tr']['mae']:.2f}")
    print(f"  Test R²:  {best['m_te']['r2']:.3f}, MAE: {best['m_te']['mae']:.2f}")
    
    # Leakage check
    passed, issues = check_leakage(best['m_tr'], best['m_te'], target)
    if issues:
        for iss in issues:
            print(f"  ⚠ LEAKAGE: {iss}")
    else:
        print(f"  ✓ No leakage")
    
    # Betting simulation
    line_feat = cfg.get('line_feature')
    if line_feat is None:
        line_feat = f"{target}_l10"
    
    if line_feat in test_clean.columns:
        lines = test_clean[line_feat].values
    else:
        lines = np.full(len(y_te), y_tr.mean())
    
    betting = realistic_betting_simulation(y_te, best['pred_te'], lines, target)
    print(f"  Betting: {betting['accuracy']:.1%} ({betting['wins']}-{betting['losses']}), "
          f"ROI: {betting['roi_pct']:+.1f}%")
    
    if betting['accuracy'] > 0.62:
        print(f"  ⚠ Accuracy > 62% is suspicious - verify no leakage!")
        passed = False
    elif betting['accuracy'] > 0.5238:
        print(f"  ✓ Profitable (>52.38%)")
    else:
        print(f"  ○ Below profitability threshold")
    
    # Feature importance
    top_feats = get_importance(best['model'], feature_cols, 10)
    if top_feats:
        print(f"  Top features:")
        for feat, imp in top_feats[:5]:
            print(f"    - {feat}: {imp:.4f}")
    
    return {
        'target': target,
        'model_type': best['type'],
        'model': best['model'],
        'scaler': best['scaler'],
        'medians': best['medians'],
        'features': feature_cols,
        'm_tr': best['m_tr'],
        'm_te': best['m_te'],
        'betting': betting,
        'passed': passed,
        'feature_importance': top_feats,
        'predictions': pd.DataFrame({
            'player': test_clean['player'].values,
            'game_date': test_clean['game_date'].values,
            'actual': y_te,
            'predicted': best['pred_te'],
            'line': lines
        })
    }


def main():
    print("="*60)
    print("Task 6: Train ALL Player Prop Models (OPTIMIZED)")
    print("="*60)
    print("\nOptimizations:")
    print("  - Fixed data leakage (removed FANTASY_PTS, PLUS_MINUS)")
    print("  - Better hyperparameter tuning")
    print("  - Ensemble methods")
    print("  - Feature importance analysis")
    
    df = load_data()
    df = df[df['games_played'] >= 10].copy()
    print(f"Filtered to {len(df):,} rows (10+ games played)")
    
    df = create_combo_targets(df)
    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)}")
    
    train_df, test_df = temporal_split(df, test_days=21)
    
    targets = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pra', 'pr', 'pa']
    
    print(f"\n{'='*60}")
    print("TRAINING MODELS")
    print("="*60)
    
    results = {}
    for t in targets:
        if t in df.columns:
            result = train_prop(train_df, test_df, feature_cols, t, optimize=True)
            if result:
                results[t] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Prop':<5} {'Type':<8} {'R²':<6} {'MAE':<6} {'Bet%':<6} {'ROI':<7} {'OK'}")
    print("─"*55)
    
    all_ok = True
    for t, r in results.items():
        ok = "✓" if r['passed'] else "⚠"
        if not r['passed']:
            all_ok = False
        print(f"{t.upper():<5} {r['model_type'].upper():<8} "
              f"{r['m_te']['r2']:.3f}  {r['m_te']['mae']:.2f}   "
              f"{r['betting']['accuracy']:.1%}  {r['betting']['roi_pct']:+.1f}%   {ok}")
    
    # Save
    print(f"\n{'='*60}")
    print("SAVING")
    print("="*60)
    
    for t, r in results.items():
        with open(f"model_{t}.pkl", 'wb') as f:
            pickle.dump({
                'model': r['model'],
                'scaler': r['scaler'],
                'medians': r['medians'],
                'features': r['features'],
                'target': t,
                'type': r['model_type'],
                'metrics': r['m_te'],
                'feature_importance': r['feature_importance'],
                'date': datetime.now().isoformat()
            }, f)
        
        r['predictions'].to_csv(f"predictions_{t}.csv", index=False)
    
    print(f"  ✓ {len(results)} models saved (model_*.pkl)")
    print(f"  ✓ {len(results)} predictions saved (predictions_*.csv)")
    
    print(f"\n{'='*60}")
    if all_ok:
        print("STATUS: ALL MODELS PASSED ✓")
    else:
        print("STATUS: SOME WARNINGS ⚠ (review above)")
    print("="*60)
    
    print("\n[PROFITABLE PROPS (>52.38%)]")
    profitable = [(t,r) for t,r in results.items() if r['betting']['accuracy'] > 0.5238]
    for t, r in sorted(profitable, key=lambda x: -x[1]['betting']['accuracy']):
        print(f"  {t.upper()}: {r['betting']['accuracy']:.1%}, {r['betting']['roi_pct']:+.1f}% ROI")


if __name__ == "__main__":
    main()
