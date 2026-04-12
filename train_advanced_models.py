"""
Train XGBoost Models with Advanced Features
============================================
Uses all advanced features from the feature pipeline for improved predictions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import re
import unicodedata
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_pinball_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from probability_utils import (
    COUNT_TARGETS,
    PROBABILITY_TARGETS,
    QUANTILE_LEVELS,
    apply_isotonic_calibrators,
    compute_binary_probability_metrics,
    ensure_monotonic_quantiles,
    ensure_monotonic_tail_probs,
    fit_isotonic_calibrators,
    ordinal_over_probability,
    over_probability_from_quantiles,
    poisson_over_probability,
    quantile_tag,
    quantile_tags,
    tail_threshold_tag,
)
warnings.filterwarnings('ignore')

# ── Per-target regression hyperparameters ─────────────────────────────────────
# Tuned for each stat's distribution: range, skew, and predictability.
REGRESSION_PARAMS: Dict[str, Dict] = {
    # High range, high variance — deeper trees, more rounds
    'pts': dict(max_depth=7, learning_rate=0.04, n_estimators=400,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.05, reg_lambda=1.0),
    # Moderate range, right-skewed
    'trb': dict(max_depth=6, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                reg_alpha=0.10, reg_lambda=1.0),
    # Lower range, heavy zero-inflation at tails
    'ast': dict(max_depth=5, learning_rate=0.05, n_estimators=300,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.10, reg_lambda=1.5),
    # Largest range (combo), benefits from more depth
    'pra': dict(max_depth=7, learning_rate=0.04, n_estimators=400,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.05, reg_lambda=1.0),
    'pr':  dict(max_depth=7, learning_rate=0.04, n_estimators=350,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.10, reg_lambda=1.0),
    'pa':  dict(max_depth=6, learning_rate=0.05, n_estimators=350,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.10, reg_lambda=1.0),
    # Low range, right-skewed — lighter model prevents overfitting
    'tov': dict(max_depth=5, learning_rate=0.05, n_estimators=250,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.20, reg_lambda=2.0),
    # Very sparse, high noise — shallow + strong regularization
    'stl': dict(max_depth=4, learning_rate=0.05, n_estimators=200,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.25, reg_lambda=2.0),
    'blk': dict(max_depth=4, learning_rate=0.05, n_estimators=200,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.30, reg_lambda=2.0),
}

# ── Per-target classifier hyperparameters ─────────────────────────────────────
CLASSIFIER_PARAMS: Dict[str, Dict] = {
    'pts': dict(max_depth=5, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.10, reg_lambda=1.0),
    'trb': dict(max_depth=4, learning_rate=0.05, n_estimators=200,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.20, reg_lambda=1.5),
    'ast': dict(max_depth=4, learning_rate=0.05, n_estimators=200,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.20, reg_lambda=1.5),
    'pra': dict(max_depth=5, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.10, reg_lambda=1.0),
    'pr':  dict(max_depth=5, learning_rate=0.05, n_estimators=250,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.10, reg_lambda=1.0),
    'pa':  dict(max_depth=5, learning_rate=0.05, n_estimators=250,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.10, reg_lambda=1.0),
    'tov': dict(max_depth=4, learning_rate=0.05, n_estimators=200,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.20, reg_lambda=2.0),
    'stl': dict(max_depth=3, learning_rate=0.05, n_estimators=150,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.30, reg_lambda=2.0),
    'blk': dict(max_depth=3, learning_rate=0.05, n_estimators=150,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.30, reg_lambda=2.0),
}

EDGE_REGRESSION_PARAMS: Dict[str, Dict] = {
    target: dict(
        max_depth=max(3, params.get('max_depth', 5) - 1),
        learning_rate=params.get('learning_rate', 0.05),
        n_estimators=max(180, int(params.get('n_estimators', 250) * 0.8)),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        min_child_weight=max(3, params.get('min_child_weight', 3)),
        reg_alpha=max(0.1, params.get('reg_alpha', 0.1)),
        reg_lambda=max(1.0, params.get('reg_lambda', 1.0)),
    )
    for target, params in REGRESSION_PARAMS.items()
}

META_CLASSIFIER_PARAMS: Dict[str, Dict] = {
    target: dict(
        max_depth=min(4, params.get('max_depth', 4)),
        learning_rate=params.get('learning_rate', 0.05),
        n_estimators=max(150, int(params.get('n_estimators', 200) * 0.8)),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        min_child_weight=max(3, params.get('min_child_weight', 3)),
        reg_alpha=max(0.15, params.get('reg_alpha', 0.1)),
        reg_lambda=max(1.0, params.get('reg_lambda', 1.0)),
    )
    for target, params in CLASSIFIER_PARAMS.items()
}

BLEND_REGRESSION_TARGETS = {'pts', 'trb', 'ast'}
BLEND_BASE_MODEL_ORDER = ['xgb', 'etr', 'bayes', 'fp']

# ── Season recency weights ─────────────────────────────────────────────────────
SEASON_WEIGHTS = {
    '2025-26': (pd.Timestamp('2025-10-01'), None,                      3.0),
    '2024-25': (pd.Timestamp('2024-10-01'), pd.Timestamp('2025-10-01'), 2.0),
    '2023-24': (None,                       pd.Timestamp('2024-10-01'), 1.0),
}


def _get_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Return per-row training weights based on game season (recency bias)."""
    dates = pd.to_datetime(df['game_date'])
    w = np.ones(len(df))
    for season, (lo, hi, weight) in SEASON_WEIGHTS.items():
        mask = np.ones(len(df), dtype=bool)
        if lo is not None:
            mask &= (dates >= lo).values
        if hi is not None:
            mask &= (dates < hi).values
        w[mask] = weight
    return w


def _print_calibration(records: list, target: str, label: str = "real lines") -> None:
    """Print hit rate by edge bucket and CLV-style correlation."""
    summary = _summarize_calibration(records)
    if not summary:
        return
    bucket_rows = summary.get('bucket_summary', [])
    clv_corr = summary.get('clv_corr')
    clv_str = f"{clv_corr:+.3f}" if clv_corr is not None and not np.isnan(clv_corr) else "n/a"
    print(f"\n  Edge calibration for {target.upper()} [{label}]  (CLV corr: {clv_str})")
    print(f"  {'Bucket':<12}  {'N':>6}  {'Hit%':>7}  {'Avg Edge':>9}")
    print("  " + "-" * 42)
    if not bucket_rows:
        return
    any_printed = False
    for row in bucket_rows:
        if row['n'] < 10:
            continue
        any_printed = True
        print(f"  {row['label']:<12}  {row['n']:>6}  {row['hit_rate']:>7.1%}  {row['avg_edge']:>+9.2f}")
    if not any_printed:
        edges = pd.DataFrame(records)['edge'].values
        print(f"  (edge range {edges.min():+.2f} to {edges.max():+.2f}, "
              f"median {np.median(edges):+.2f} -- all rows in 0-1 band)")


def _summarize_calibration(records: list) -> Optional[dict]:
    """Return structured real-line diagnostics by edge bucket and direction."""
    if not records:
        return None
    df = pd.DataFrame(records)  # columns: edge, actual_beat_line, actual, line, pred
    df['hit'] = ((df['edge'] > 0) == df['actual_beat_line'].astype(bool)).astype(int)

    # CLV-style: correlation between pred-line and actual-line
    act_dev = df['actual'] - df['line']
    clv_corr = df['edge'].corr(act_dev)

    buckets = [
        ('UNDER >=3',  None,  -3.0),
        ('UNDER 2-3',  -3.0,  -2.0),
        ('UNDER 1-2',  -2.0,  -1.0),
        ('UNDER 0-1',  -1.0,   0.0),
        ('OVER  0-1',   0.0,   1.0),
        ('OVER  1-2',   1.0,   2.0),
        ('OVER  2-3',   2.0,   3.0),
        ('OVER  >=3',   3.0,  None),
    ]

    bucket_rows = []
    edges = df['edge'].values
    for label, lo, hi in buckets:
        mask = np.ones(len(df), dtype=bool)
        if lo is not None:
            mask &= edges >= lo
        if hi is not None:
            mask &= edges < hi
        sub = df[mask]
        bucket_rows.append({
            'label': label,
            'lower': lo,
            'upper': hi,
            'n': int(len(sub)),
            'hit_rate': float(sub['hit'].mean()) if len(sub) else None,
            'avg_edge': float(sub['edge'].mean()) if len(sub) else None,
            'avg_actual_minus_line': float((sub['actual'] - sub['line']).mean()) if len(sub) else None,
        })

    direction_rows = []
    for label, mask in [
        ('under_edges', df['edge'] < 0),
        ('over_edges', df['edge'] > 0),
        ('strong_edges_abs_ge_2', df['edge'].abs() >= 2.0),
        ('strong_edges_abs_ge_3', df['edge'].abs() >= 3.0),
    ]:
        sub = df[mask]
        direction_rows.append({
            'label': label,
            'n': int(len(sub)),
            'hit_rate': float(sub['hit'].mean()) if len(sub) else None,
            'avg_edge': float(sub['edge'].mean()) if len(sub) else None,
            'avg_actual_minus_line': float((sub['actual'] - sub['line']).mean()) if len(sub) else None,
        })

    return {
        'n_records': int(len(df)),
        'hit_rate': float(df['hit'].mean()) if len(df) else None,
        'avg_edge': float(df['edge'].mean()) if len(df) else None,
        'avg_abs_edge': float(df['edge'].abs().mean()) if len(df) else None,
        'avg_actual_minus_line': float((df['actual'] - df['line']).mean()) if len(df) else None,
        'clv_corr': float(clv_corr) if pd.notna(clv_corr) else None,
        'bucket_summary': bucket_rows,
        'direction_summary': direction_rows,
    }

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

# Prop name mapping from historical_lines.csv to model target names
PROP_TO_TARGET = {
    'player_points':                   'pts',
    'player_rebounds':                  'trb',
    'player_assists':                   'ast',
    'player_points_rebounds_assists':   'pra',
    'player_points_rebounds':           'pr',
    'player_points_assists':            'pa',
    'player_steals':                    'stl',
    'player_blocks':                    'blk',
    'player_turnovers':                 'tov',
}

# Fallback proxy line column when richer recency anchors are missing.
PROXY_LINE_COL = {
    'pts': 'pts_l10', 'trb': 'trb_l10', 'ast': 'ast_l10',
    'pra': None,       'pr':  None,       'pa':  None,
    'stl': 'stl_l10', 'blk': 'blk_l10', 'tov': 'tov_l10',
}

CURRENT_ROW_LEAK_FEATURES = [
    'usage_proxy',
]

SEASON_CONTEXT_FEATURES = [
    'clean_pts', 'clean_reb', 'clean_ast', 'clean_fg3m', 'clean_min',
    'pts_delta', 'n_exclude', 'n_heavy', 'n_partial',
    'possessions', 'raw_ppp', 'garbage_time_flag', 'competitive_poss',
    'adj_ppp', 'raw_ppp_season', 'opp_def_rtg_avg', 'total_possessions',
    'efg_pct', 'ts_pct',
    'usg_pct', 'pts_per_poss', 'ast_per_poss', 'reb_per_poss',
    'min_pg', 'poss_pg', 'off_rtg', 'net_rtg', 'role_weight',
    'total_luck_score', 'efg_luck_score', 'fg3_luck_score', 'ft_luck_score',
    'pts_luck_adj', 'fg3_regressed', 'ft_regressed',
]

DYNAMIC_CONTEXT_SOURCE_FEATURES = ['possessions', 'raw_ppp', 'competitive_poss', 'garbage_time_flag']


def _seasonalize_feature_block(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Align enriched context features to the same season-level granularity used at
    inference. This prevents same-game layer values from leaking into training.
    """
    avail = [c for c in cols if c in df.columns]
    if not avail or 'season' not in df.columns:
        return df

    group_keys = None
    if 'player_id' in df.columns and 'player' in df.columns:
        group_keys = ['player_id', 'player', 'season']
    elif 'player_id' in df.columns:
        group_keys = ['player_id', 'season']
    elif 'player' in df.columns:
        group_keys = ['player', 'season']
    if group_keys is None:
        return df

    out = df.copy()
    out[avail] = out.groupby(group_keys, dropna=False)[avail].transform('mean')
    return out


def _build_dynamic_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leak-safe rolling context features from per-game PBP layer fields."""
    required = [c for c in DYNAMIC_CONTEXT_SOURCE_FEATURES if c in df.columns]
    group_col = 'player_id' if 'player_id' in df.columns else ('PLAYER_ID' if 'PLAYER_ID' in df.columns else ('player' if 'player' in df.columns else None))
    if len(required) < 2 or group_col is None or 'game_date' not in df.columns:
        return df

    out = df.copy()
    out['game_date'] = pd.to_datetime(out['game_date'], errors='coerce')
    out = out.sort_values([group_col, 'game_date']).reset_index(drop=True)

    for col in required:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    g = out.groupby(group_col, dropna=False)
    if 'possessions' in out.columns:
        out['possessions_l5_ctx'] = g['possessions'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        out['possessions_l10_ctx'] = g['possessions'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    if 'competitive_poss' in out.columns:
        out['competitive_poss_l5_ctx'] = g['competitive_poss'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        out['competitive_poss_l10_ctx'] = g['competitive_poss'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    if 'raw_ppp' in out.columns:
        out['raw_ppp_l5_ctx'] = g['raw_ppp'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        out['raw_ppp_l10_ctx'] = g['raw_ppp'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    if 'garbage_time_flag' in out.columns:
        out['garbage_rate_l10_ctx'] = g['garbage_time_flag'].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    if {'competitive_poss', 'possessions'}.issubset(out.columns):
        share = (out['competitive_poss'] / out['possessions'].replace(0, np.nan)).clip(lower=0, upper=1)
        out['competitive_share_l10_ctx'] = share.groupby(out[group_col]).transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )

    return out


def _load_lines_lookup() -> dict:
    """Return {(player_norm, game_date, target): line} from historical_lines.csv."""
    import unicodedata, re
    lines_path = DATA_DIR / 'historical_lines.csv'
    if not lines_path.exists():
        return {}

    def _norm(name):
        n = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
        return re.sub(r'[^a-z0-9 ]', '', n.lower()).strip()

    df = pd.read_csv(lines_path)
    df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
    if 'player_norm' not in df.columns:
        df['player_norm'] = df['player'].apply(_norm)
    df['target'] = df['prop'].map(PROP_TO_TARGET)
    df = df.dropna(subset=['target'])

    lookup = {}
    for _, row in df.iterrows():
        key = (row['player_norm'], row['game_date'], row['target'])
        lookup[key] = float(row['line'])
    return lookup


def _numeric_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return pd.Series(default, index=df.index, dtype=float)


def _target_component_cols(target: str, suffix: str) -> List[str]:
    combo_map = {
        'pra': ['pts', 'trb', 'ast'],
        'pr': ['pts', 'trb'],
        'pa': ['pts', 'ast'],
    }
    if target in combo_map:
        return [f'{part}_{suffix}' for part in combo_map[target]]
    return [f'{target}_{suffix}']


def _sum_target_feature(df: pd.DataFrame, target: str, suffix: str, default=np.nan) -> pd.Series:
    cols = [c for c in _target_component_cols(target, suffix) if c in df.columns]
    if not cols:
        return pd.Series(default, index=df.index, dtype=float)
    vals = [pd.to_numeric(df[c], errors='coerce') for c in cols]
    if len(vals) == 1:
        return vals[0]
    stacked = pd.concat(vals, axis=1)
    return stacked.sum(axis=1, min_count=len(cols))


def _avg_target_feature(df: pd.DataFrame, target: str, suffix: str, default=np.nan) -> pd.Series:
    cols = [c for c in _target_component_cols(target, suffix) if c in df.columns]
    if not cols:
        return pd.Series(default, index=df.index, dtype=float)
    stacked = pd.concat([pd.to_numeric(df[c], errors='coerce') for c in cols], axis=1)
    return stacked.mean(axis=1)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def _build_market_feature_block(df: pd.DataFrame, target: str, line: pd.Series,
                                raw_pred: Optional[pd.Series] = None) -> pd.DataFrame:
    line = pd.to_numeric(line, errors='coerce')
    l10 = _sum_target_feature(df, target, 'l10')
    l5 = _sum_target_feature(df, target, 'l5')
    best = _sum_target_feature(df, target, 'best_estimate')
    anchor = _sum_target_feature(df, target, 'recency_anchor')
    season_avg = _sum_target_feature(df, target, 'season_avg')
    std = _sum_target_feature(df, target, 'std_l10')
    std = std.fillna(_sum_target_feature(df, target, 'std'))
    std = std.clip(lower=0.5)
    conf = _avg_target_feature(df, target, 'confidence', default=0.5).fillna(0.5)

    clean_map = {
        'pts': ['clean_pts'],
        'trb': ['clean_reb'],
        'ast': ['clean_ast'],
        'pra': ['clean_pts', 'clean_reb', 'clean_ast'],
        'pr': ['clean_pts', 'clean_reb'],
        'pa': ['clean_pts', 'clean_ast'],
    }
    clean_cols = [c for c in clean_map.get(target, []) if c in df.columns]
    if clean_cols:
        clean = pd.concat([pd.to_numeric(df[c], errors='coerce') for c in clean_cols], axis=1).sum(axis=1, min_count=len(clean_cols))
    else:
        clean = pd.Series(np.nan, index=df.index, dtype=float)

    mp_l5 = _numeric_series(df, 'mp_l5')
    mp_l10 = _numeric_series(df, 'mp_l10')
    mp_l20 = _numeric_series(df, 'mp_l20')
    usage_l5 = _numeric_series(df, 'usage_l5')
    usage_l10 = _numeric_series(df, 'usage_l10')
    fga_l5 = _numeric_series(df, 'fga_l5')
    fga_l20 = _numeric_series(df, 'fga_l20')
    role_expanding = _numeric_series(df, 'role_expanding', default=0.0).fillna(0.0)
    role_shrinking = _numeric_series(df, 'role_shrinking', default=0.0).fillna(0.0)
    days_rest = _numeric_series(df, 'days_rest', default=2.0).fillna(2.0)
    is_b2b_second = _numeric_series(df, 'is_b2b_second', default=0.0).fillna(0.0)
    games_played = _numeric_series(df, 'games_played', default=0.0).fillna(0.0)
    team_spread = _numeric_series(df, 'team_spread', default=0.0).fillna(0.0)
    game_total = _numeric_series(df, 'game_total')
    game_env = _numeric_series(df, 'game_environment_score')
    line_move = _numeric_series(df, f'{target}_line_move', default=0.0).fillna(0.0)
    risk_factor = _numeric_series(df, 'risk_factor', default=1.0).fillna(1.0)
    total_luck = _numeric_series(df, 'total_luck_score', default=0.0).fillna(0.0)
    comp_share = _numeric_series(df, 'competitive_share_l10_ctx', default=1.0).fillna(1.0)
    garbage_rate = _numeric_series(df, 'garbage_rate_l10_ctx', default=0.0).fillna(0.0)
    season_reliability = _numeric_series(df, 'season_sample_reliability', default=0.0).fillna(0.0)

    minutes_shock = _safe_ratio((mp_l5 - mp_l20).abs(), mp_l20.abs().clip(lower=8))
    usage_shock = _safe_ratio((usage_l5 - usage_l10).abs(), usage_l10.abs().clip(lower=0.05))
    fga_shock = _safe_ratio((fga_l5 - fga_l20).abs(), fga_l20.abs().clip(lower=1.0))
    role_shock = (
        minutes_shock.fillna(0) * 0.45 +
        usage_shock.fillna(0) * 0.30 +
        fga_shock.fillna(0) * 0.15 +
        role_expanding.abs().fillna(0) * 0.05 +
        role_shrinking.abs().fillna(0) * 0.05
    )

    block = pd.DataFrame(index=df.index)
    block['market_line'] = line
    block['market_abs_line'] = line.abs()
    block['market_line_vs_l10'] = line - l10
    block['market_line_vs_l5'] = line - l5
    block['market_line_vs_best'] = line - best
    block['market_line_vs_anchor'] = line - anchor
    block['market_line_vs_clean'] = line - clean
    block['market_l5_vs_l10'] = l5 - l10
    block['market_best_vs_l10'] = best - l10
    block['market_anchor_vs_l10'] = anchor - l10
    block['market_anchor_vs_season'] = anchor - season_avg
    block['market_std'] = std
    block['market_line_z_l10'] = _safe_ratio(line - l10, std)
    block['market_line_z_anchor'] = _safe_ratio(line - anchor, std)
    block['market_best_z_line'] = _safe_ratio(best - line, std)
    block['market_confidence_mean'] = conf
    block['market_risk_factor'] = risk_factor
    block['market_days_rest'] = days_rest
    block['market_is_b2b_second'] = is_b2b_second
    block['market_games_played'] = games_played
    block['market_team_spread'] = team_spread
    block['market_game_total'] = game_total.fillna(game_total.median() if game_total.notna().any() else 0.0)
    block['market_environment_score'] = game_env.fillna(0.0)
    block['market_line_move'] = line_move
    block['market_minutes_shock'] = minutes_shock.fillna(0.0)
    block['market_usage_shock'] = usage_shock.fillna(0.0)
    block['market_fga_shock'] = fga_shock.fillna(0.0)
    block['market_role_shock'] = role_shock.fillna(0.0)
    block['market_role_delta'] = (role_expanding - role_shrinking).fillna(0.0)
    block['market_total_luck'] = total_luck
    block['market_competitive_share'] = comp_share.clip(lower=0, upper=1)
    block['market_garbage_rate'] = garbage_rate.clip(lower=0, upper=1)
    block['market_season_reliability'] = season_reliability.clip(lower=0, upper=1)
    block['market_blowout_risk'] = (_safe_ratio(team_spread.abs(), pd.Series(12.0, index=df.index)) + garbage_rate.fillna(0.0)).fillna(0.0)

    if raw_pred is not None:
        raw_pred = pd.to_numeric(raw_pred, errors='coerce')
        block['signal_raw_edge'] = raw_pred - line
        block['signal_abs_raw_edge'] = (raw_pred - line).abs()
        block['signal_raw_vs_best'] = raw_pred - best
        block['signal_raw_vs_anchor'] = raw_pred - anchor
        block['signal_raw_vs_l10'] = raw_pred - l10
    return block.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_meta_feature_block(market_block: pd.DataFrame, raw_edge: pd.Series,
                              market_edge_pred: pd.Series) -> pd.DataFrame:
    meta = market_block.copy()
    raw_edge = pd.to_numeric(raw_edge, errors='coerce').fillna(0.0)
    market_edge_pred = pd.to_numeric(market_edge_pred, errors='coerce').fillna(0.0)
    meta['signal_raw_edge'] = raw_edge
    meta['signal_abs_raw_edge'] = raw_edge.abs()
    meta['signal_market_edge_pred'] = market_edge_pred
    meta['signal_abs_market_edge_pred'] = market_edge_pred.abs()
    meta['signal_edge_disagreement'] = (market_edge_pred - raw_edge).abs()
    meta['signal_direction_match'] = (np.sign(market_edge_pred) == np.sign(raw_edge)).astype(float)
    meta['signal_shock_penalty'] = meta['market_role_shock'] * meta['signal_abs_market_edge_pred']
    return meta.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_player_name(name: object) -> str:
    text = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def _build_fp_projection_proxy(df: pd.DataFrame, target: str) -> pd.Series:
    """Return the closest historical first-principles proxy available in the frame."""
    if target == 'pts':
        if 'proj_pts_final' in df.columns:
            return pd.to_numeric(df['proj_pts_final'], errors='coerce')
        pts_base = _numeric_series(df, 'pts_luck_adj')
        if pts_base.isna().all():
            pts_base = _numeric_series(df, 'clean_pts')
        if pts_base.isna().all():
            pts_base = _sum_target_feature(df, 'pts', 'best_estimate')
        if 'adj_ppp' in df.columns:
            scale = (_numeric_series(df, 'adj_ppp') / 1.15).clip(lower=0.7, upper=1.3)
            return pts_base.fillna(_sum_target_feature(df, 'pts', 'best_estimate')).fillna(_sum_target_feature(df, 'pts', 'recency_anchor')) * scale.fillna(1.0)
        return pts_base.fillna(_sum_target_feature(df, 'pts', 'best_estimate')).fillna(_sum_target_feature(df, 'pts', 'recency_anchor'))
    if target == 'trb':
        series = _numeric_series(df, 'proj_reb')
        if series.isna().all():
            series = _numeric_series(df, 'clean_reb')
        return series.fillna(_sum_target_feature(df, 'trb', 'best_estimate')).fillna(_sum_target_feature(df, 'trb', 'recency_anchor'))
    if target == 'ast':
        series = _numeric_series(df, 'proj_ast')
        if series.isna().all():
            series = _numeric_series(df, 'clean_ast')
        return series.fillna(_sum_target_feature(df, 'ast', 'best_estimate')).fillna(_sum_target_feature(df, 'ast', 'recency_anchor'))
    return pd.Series(np.nan, index=df.index, dtype=float)


def _predict_regression_model(model, X: pd.DataFrame, fp_input: Optional[pd.Series] = None) -> np.ndarray:
    """Predict from either a plain regressor or a saved blended-regression bundle."""
    if isinstance(model, dict) and model.get('kind') in {'ridge_blend_regressor', 'positive_ridge_blend_regressor'}:
        base_preds = []
        for base_name in model.get('base_order', []):
            if base_name == 'fp':
                if fp_input is None:
                    base_preds.append(np.zeros(len(X), dtype=float))
                else:
                    base_preds.append(np.asarray(pd.to_numeric(fp_input, errors='coerce').fillna(0.0), dtype=float))
                continue
            base_model = model['base_models'][base_name]
            base_preds.append(np.asarray(base_model.predict(X), dtype=float))
        blend_X = np.column_stack(base_preds)
        return np.asarray(model['blender'].predict(blend_X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _make_blend_base_models(target: str, reg_params: Dict) -> Dict[str, object]:
    depth = int(reg_params.get('max_depth', 6))
    n_estimators = int(reg_params.get('n_estimators', 300))
    min_child_weight = int(reg_params.get('min_child_weight', 3))
    reg_alpha = float(reg_params.get('reg_alpha', 0.1))
    xgb_params = dict(reg_params)
    xgb_params.update({'random_state': 42, 'n_jobs': -1})
    xgb_model = xgb.XGBRegressor(**xgb_params)
    etr_model = ExtraTreesRegressor(
        n_estimators=max(320, int(n_estimators * 1.1)),
        max_depth=max(depth + 1, 6),
        min_samples_leaf=max(2, min_child_weight - 1),
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    bayes_model = Pipeline([
        ('scaler', StandardScaler()),
        ('bayes', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)),
    ])
    return {
        'xgb': xgb_model,
        'etr': etr_model,
        'bayes': bayes_model,
    }


def _fit_regression_model(model_name: str, model, X: pd.DataFrame, y: pd.Series,
                          sample_weight: np.ndarray) -> object:
    if model_name == 'xgb':
        model.fit(X, y, sample_weight=sample_weight, verbose=False)
    elif model_name == 'bayes':
        model.fit(X, y, bayes__sample_weight=sample_weight)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def _make_quantile_sidecar_model(target: str, alpha: float, reg_params: Dict) -> HistGradientBoostingRegressor:
    depth = int(reg_params.get('max_depth', 5))
    n_estimators = int(reg_params.get('n_estimators', 300))
    min_child_weight = int(reg_params.get('min_child_weight', 3))
    return HistGradientBoostingRegressor(
        loss='quantile',
        quantile=alpha,
        max_iter=max(140, int(n_estimators * 0.50)),
        learning_rate=min(0.05, float(reg_params.get('learning_rate', 0.05))),
        max_depth=max(3, min(6, depth)),
        max_features=float(reg_params.get('colsample_bytree', 0.8)),
        min_samples_leaf=max(12, min_child_weight * 3),
        l2_regularization=max(0.0, float(reg_params.get('reg_lambda', 1.0)) * 0.05),
        early_stopping=False,
        random_state=42,
    )


def _make_ordinal_probability_model(target: str, threshold: int, clf_params: Dict) -> xgb.XGBClassifier:
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
        'max_depth': min(4, int(clf_params.get('max_depth', 4))),
        'learning_rate': float(clf_params.get('learning_rate', 0.05)),
        'n_estimators': max(160, int(clf_params.get('n_estimators', 200) * 0.8)),
        'subsample': float(clf_params.get('subsample', 0.8)),
        'colsample_bytree': float(clf_params.get('colsample_bytree', 0.8)),
        'min_child_weight': max(3, int(clf_params.get('min_child_weight', 3))),
        'reg_alpha': max(0.15, float(clf_params.get('reg_alpha', 0.1))),
        'reg_lambda': max(1.0, float(clf_params.get('reg_lambda', 1.0))),
    }
    if target in {'stl', 'blk'} and threshold >= 2:
        params['max_depth'] = max(2, params['max_depth'] - 1)
        params['n_estimators'] = max(120, int(params['n_estimators'] * 0.85))
    return xgb.XGBClassifier(**params)


def _make_poisson_probability_model(target: str, reg_params: Dict) -> xgb.XGBRegressor:
    params = {
        'objective': 'count:poisson',
        'random_state': 42,
        'n_jobs': -1,
        'max_depth': min(5, int(reg_params.get('max_depth', 5))),
        'learning_rate': float(reg_params.get('learning_rate', 0.05)),
        'n_estimators': max(200, int(reg_params.get('n_estimators', 250) * 0.85)),
        'subsample': float(reg_params.get('subsample', 0.8)),
        'colsample_bytree': float(reg_params.get('colsample_bytree', 0.8)),
        'min_child_weight': max(3, int(reg_params.get('min_child_weight', 3))),
        'reg_alpha': max(0.10, float(reg_params.get('reg_alpha', 0.1))),
        'reg_lambda': max(1.0, float(reg_params.get('reg_lambda', 1.0))),
    }
    return xgb.XGBRegressor(**params)


def _predict_probability_sidecar(sidecar: Dict[str, object],
                                 X: pd.DataFrame,
                                 lines: Optional[pd.Series] = None) -> Dict[str, np.ndarray]:
    lines_series = pd.Series(np.nan, index=X.index, dtype=float)
    if lines is not None:
        lines_series = pd.to_numeric(pd.Series(lines, index=X.index), errors='coerce')
    result: Dict[str, np.ndarray] = {}
    kind = sidecar.get('kind')

    if kind == 'quantile_probability_sidecar':
        tags = sidecar.get('quantile_tags', quantile_tags(sidecar.get('quantiles', QUANTILE_LEVELS)))
        q_preds = []
        for tag in tags:
            q_preds.append(np.asarray(sidecar['models'][tag].predict(X), dtype=float))
        q_matrix = ensure_monotonic_quantiles(np.column_stack(q_preds))
        for idx, tag in enumerate(tags):
            result[tag] = q_matrix[:, idx]
        raw = np.full(len(X), np.nan, dtype=float)
        valid_line_mask = lines_series.notna().values
        if valid_line_mask.any():
            raw[valid_line_mask] = over_probability_from_quantiles(
                lines_series.loc[valid_line_mask].values,
                q_matrix[valid_line_mask],
                sidecar.get('quantiles', QUANTILE_LEVELS),
            )
        result['p_over_raw'] = raw
        return result

    if kind == 'ordinal_tail_probability_sidecar':
        thresholds = sidecar.get('thresholds', [])
        prob_tags = [tail_threshold_tag(threshold) for threshold in thresholds]
        prob_matrix = []
        for threshold in thresholds:
            tag = tail_threshold_tag(threshold)
            model = sidecar['models'].get(tag)
            if model is None:
                prob_matrix.append(np.full(len(X), 0.5, dtype=float))
            else:
                prob_matrix.append(np.asarray(model.predict_proba(X)[:, 1], dtype=float))
        if prob_matrix:
            prob_matrix = ensure_monotonic_tail_probs(np.column_stack(prob_matrix))
            for idx, tag in enumerate(prob_tags):
                result[tag] = prob_matrix[:, idx]
            raw = np.full(len(X), np.nan, dtype=float)
            valid_line_mask = lines_series.notna().values
            if valid_line_mask.any():
                raw[valid_line_mask] = ordinal_over_probability(
                    lines_series.loc[valid_line_mask].values,
                    prob_matrix[valid_line_mask],
                    thresholds,
                )
            result['p_over_raw'] = raw
        else:
            result['p_over_raw'] = np.full(len(X), np.nan, dtype=float)
        return result

    if kind == 'poisson_probability_sidecar':
        rate = np.clip(np.asarray(sidecar['model'].predict(X), dtype=float), 1e-6, None)
        result['rate'] = rate
        raw = np.full(len(X), np.nan, dtype=float)
        valid_line_mask = lines_series.notna().values
        if valid_line_mask.any():
            raw[valid_line_mask] = poisson_over_probability(
                lines_series.loc[valid_line_mask].values,
                rate[valid_line_mask],
            )
        result['p_over_raw'] = raw
        return result

    raise ValueError(f"Unsupported probability sidecar kind: {kind}")


def _score_predictions(pred_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict], float, float, float, float, float, float]:
    """Compute fold-level regression and line-direction metrics from OOF predictions."""
    cv_results = []
    cal_records = []

    for fold in sorted(pred_df['fold'].dropna().unique()):
        fold_df = pred_df[pred_df['fold'] == fold].copy()
        if len(fold_df) == 0:
            continue

        mae = mean_absolute_error(fold_df['actual'], fold_df['pred'])
        rmse = np.sqrt(mean_squared_error(fold_df['actual'], fold_df['pred']))

        def _score_mask(mask: pd.Series, collect_calibration: bool = False) -> Tuple[float, int]:
            sub = fold_df[mask].copy()
            n_scored = int(len(sub))
            if n_scored < 10:
                return np.nan, n_scored

            edge_arr = sub['pred'].values - sub['line'].values
            beat_line = (sub['actual'].values > sub['line'].values).astype(int)
            if collect_calibration:
                for e, b, a, l, p in zip(edge_arr, beat_line, sub['actual'].values, sub['line'].values, sub['pred'].values):
                    cal_records.append({
                        'edge': float(e),
                        'actual_beat_line': int(b),
                        'actual': float(a),
                        'line': float(l),
                        'pred': float(p),
                    })
            hit_arr = ((edge_arr > 0) == beat_line.astype(bool)).astype(float)
            return float(hit_arr.mean()), n_scored

        all_mask = fold_df['line'].notna()
        real_mask = fold_df['line_source'].eq('real') & all_mask
        proxy_mask = fold_df['line_source'].eq('proxy') & all_mask

        real_bet_acc, n_real_scored = _score_mask(real_mask, collect_calibration=True)
        proxy_bet_acc, n_proxy_scored = _score_mask(proxy_mask)
        all_bet_acc, n_scored = _score_mask(all_mask)

        cv_results.append({
            'fold': int(fold),
            'mae': float(mae),
            'rmse': float(rmse),
            'bet_accuracy': real_bet_acc,
            'proxy_bet_accuracy': proxy_bet_acc,
            'all_line_bet_accuracy': all_bet_acc,
            'n_scored': int(n_scored),
            'n_real_scored': int(n_real_scored),
            'n_proxy_scored': int(n_proxy_scored),
        })

    avg_mae = float(np.mean([r['mae'] for r in cv_results])) if cv_results else np.nan
    avg_rmse = float(np.mean([r['rmse'] for r in cv_results])) if cv_results else np.nan
    valid_real_acc = [r['bet_accuracy'] for r in cv_results if not np.isnan(r['bet_accuracy'])]
    valid_proxy_acc = [r['proxy_bet_accuracy'] for r in cv_results if not np.isnan(r['proxy_bet_accuracy'])]
    valid_all_acc = [r['all_line_bet_accuracy'] for r in cv_results if not np.isnan(r['all_line_bet_accuracy'])]
    avg_bet_acc = float(np.mean(valid_real_acc)) if valid_real_acc else np.nan
    avg_proxy_bet_acc = float(np.mean(valid_proxy_acc)) if valid_proxy_acc else np.nan
    avg_all_bet_acc = float(np.mean(valid_all_acc)) if valid_all_acc else np.nan
    mean_bias = float((pred_df['pred'] - pred_df['actual']).mean()) if len(pred_df) else 0.0
    return cv_results, cal_records, avg_mae, avg_rmse, avg_bet_acc, avg_proxy_bet_acc, avg_all_bet_acc, mean_bias


def _load_projection_archive_predictions(projection_dir: Path) -> pd.DataFrame:
    """Load latest archived first-principles projections per game date."""
    if not projection_dir.exists():
        return pd.DataFrame()

    date_to_file: Dict[str, Path] = {}
    for path in projection_dir.glob('player_projections_*.csv'):
        match = re.match(r'^player_projections_(\d{4}-\d{2}-\d{2})(?:_\d{6})?$', path.stem)
        if not match:
            continue
        game_date = match.group(1)
        keep = date_to_file.get(game_date)
        if keep is None or path.stat().st_mtime_ns > keep.stat().st_mtime_ns:
            date_to_file[game_date] = path

    frames = []
    for game_date, path in sorted(date_to_file.items()):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if 'player' not in frame.columns:
            continue
        frame = frame.copy()
        frame['game_date'] = pd.to_datetime(game_date)
        frame['player_norm'] = frame['player'].apply(_normalize_player_name)
        keep_cols = ['game_date', 'player_norm']
        for col in ['proj_pts_final', 'proj_reb', 'proj_ast', 'proj_pra']:
            if col in frame.columns:
                keep_cols.append(col)
        frame = frame[keep_cols]
        if 'proj_pra' not in frame.columns and {'proj_pts_final', 'proj_reb', 'proj_ast'}.issubset(frame.columns):
            frame['proj_pra'] = frame['proj_pts_final'] + frame['proj_reb'] + frame['proj_ast']
        if {'proj_pts_final', 'proj_reb'}.issubset(frame.columns):
            frame['proj_pr'] = frame['proj_pts_final'] + frame['proj_reb']
        if {'proj_pts_final', 'proj_ast'}.issubset(frame.columns):
            frame['proj_pa'] = frame['proj_pts_final'] + frame['proj_ast']
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class AdvancedPropModel:
    """
    XGBoost model for NBA player props with advanced features.
    """
    
    # Target stat configurations
    TARGETS = {
        'pts': {'primary': 'pts', 'combo': None},
        'trb': {'primary': 'trb', 'combo': None},
        'ast': {'primary': 'ast', 'combo': None},
        'stl': {'primary': 'stl', 'combo': None},
        'blk': {'primary': 'blk', 'combo': None},
        'tov': {'primary': 'tov', 'combo': None},
        'pra': {'primary': None, 'combo': ['pts', 'trb', 'ast']},
        'pr': {'primary': None, 'combo': ['pts', 'trb']},
        'pa': {'primary': None, 'combo': ['pts', 'ast']},
    }
    
    # Feature groups
    FEATURE_GROUPS = {
        'core_rolling': [
            'pts_l5', 'pts_l10', 'pts_l20',
            'trb_l5', 'trb_l10', 'trb_l20',
            'ast_l5', 'ast_l10', 'ast_l20',
            'mp_l5', 'mp_l10', 'mp_l20',
            'fga_l5', 'fga_l10', 'fga_l20',
            'fta_l5', 'fta_l10', 'fta_l20',
            '3pa_l5', '3pa_l10', '3pa_l20',
            'stl_l5', 'stl_l10', 'stl_l20',
            'blk_l5', 'blk_l10', 'blk_l20',
            'tov_l5', 'tov_l10', 'tov_l20',
        ],
        'layer_blowout': [
            'clean_pts', 'clean_reb', 'clean_ast', 'clean_fg3m', 'clean_min',
            'pts_delta', 'n_exclude', 'n_heavy', 'n_partial',
        ],
        'layer_possession': [
            'possessions', 'raw_ppp', 'garbage_time_flag', 'competitive_poss',
            'adj_ppp', 'raw_ppp_season', 'opp_def_rtg_avg', 'total_possessions',
            'efg_pct', 'ts_pct',
        ],
        'layer_context_dynamic': [
            'possessions_l5_ctx', 'possessions_l10_ctx',
            'competitive_poss_l5_ctx', 'competitive_poss_l10_ctx',
            'raw_ppp_l5_ctx', 'raw_ppp_l10_ctx',
            'garbage_rate_l10_ctx', 'competitive_share_l10_ctx',
        ],
        'layer_usage_context': [
            'usg_pct', 'pts_per_poss', 'ast_per_poss', 'reb_per_poss',
            'min_pg', 'poss_pg', 'off_rtg', 'net_rtg', 'role_weight',
        ],
        'layer_luck': [
            'total_luck_score', 'efg_luck_score', 'fg3_luck_score', 'ft_luck_score',
            'pts_luck_adj', 'fg3_regressed', 'ft_regressed',
        ],
        'ewma': [
            'pts_ewma5', 'trb_ewma5', 'ast_ewma5', 'mp_ewma5',
            'fga_ewma5', 'fta_ewma5', '3pa_ewma5',
            'stl_ewma5', 'blk_ewma5', 'tov_ewma5',
        ],
        'recency_strategy': [
            'season_games_played', 'season_sample_reliability',
            'pts_season_avg', 'trb_season_avg', 'ast_season_avg', 'mp_season_avg',
            'fga_season_avg', 'fta_season_avg', '3pa_season_avg',
            'stl_season_avg', 'blk_season_avg', 'tov_season_avg',
            'pts_recency_anchor', 'trb_recency_anchor', 'ast_recency_anchor', 'mp_recency_anchor',
            'fga_recency_anchor', 'fta_recency_anchor', '3pa_recency_anchor',
            'stl_recency_anchor', 'blk_recency_anchor', 'tov_recency_anchor',
            'pts_form_vs_season', 'trb_form_vs_season', 'ast_form_vs_season', 'mp_form_vs_season',
        ],
        'consistency': [
            'pts_consistency', 'trb_consistency', 'ast_consistency', 'mp_consistency',
            'pts_ewma_std', 'trb_ewma_std', 'ast_ewma_std', 'mp_ewma_std',
            'pts_std_l10', 'trb_std_l10', 'ast_std_l10', 'mp_std_l10',
        ],
        'trend': [
            'pts_trend', 'trb_trend', 'ast_trend',
            'mp_trend_ratio', 'usage_trend', 'fga_trend',
        ],
        'situational': [
            'pts_home_l15', 'pts_away_l15', 'pts_home_away_diff',
            'trb_home_l15', 'trb_away_l15', 'trb_home_away_diff',
            'ast_home_l15', 'ast_away_l15', 'ast_home_away_diff',
            'pts_situational_avg', 'trb_situational_avg', 'ast_situational_avg',
            'pts_vs_good_def', 'pts_vs_bad_def', 'pts_defense_adj',
            'trb_vs_good_def', 'trb_vs_bad_def', 'trb_defense_adj',
            'ast_vs_good_def', 'ast_vs_bad_def', 'ast_defense_adj',
            'pts_on_b2b', 'pts_rested', 'pts_b2b_penalty',
            'trb_on_b2b', 'trb_rested', 'trb_b2b_penalty',
            'ast_on_b2b', 'ast_rested', 'ast_b2b_penalty',
        ],
        'opponent': [
            'opp_pts_allowed_l10', 'opp_trb_allowed_l10', 'opp_ast_allowed_l10',
            'opp_fga_allowed_l10', 'opp_pace_factor',
            'opp_pts_allowed_rank', 'opp_trb_allowed_rank', 'opp_ast_allowed_rank',
            'opp_def_tier',
        ],
        'role_usage': [
            'games_played', 'mp_l3', 'likely_starter',
            'role_expanding', 'role_shrinking',
            'usage_l5', 'usage_l10',
            'fga_l3',
        ],
        'schedule': [
            'is_home', 'is_b2b_second', 'days_rest',
            'high_min_prev', 'games_last_7d',
        ],
        'vegas': [
            'pts_edge_vs_avg', 'trb_edge_vs_avg', 'ast_edge_vs_avg',
            'pts_vegas_slow', 'trb_vegas_slow', 'ast_vegas_slow',
            'pts_pct_diff', 'trb_pct_diff', 'ast_pct_diff',
            'team_scoring_trend', 'high_scoring_game',
            'game_environment_score',
        ],
        'composite': [
            'pts_best_estimate', 'trb_best_estimate', 'ast_best_estimate',
            'pts_confidence', 'trb_confidence', 'ast_confidence',
            'risk_factor',
        ],
    }
    
    def __init__(self, model_dir: str = r'C:\Users\jakep\Downloads\Fresh_Start_NBA\models',
                 include_season_context_training: bool = False):
        """Initialize the model trainer."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}          # XGBRegressor per target
        self.classifiers = {}     # XGBClassifier per target
        self.edge_models = {}     # XGBRegressor per target on actual-minus-line
        self.meta_models = {}     # XGBClassifier per target for abstention
        self.feature_importance = {}
        self.selected_features = {}
        self.market_feature_sets = {'edge': {}, 'meta': {}}
        self.biases = {}
        self.regression_oof = {}
        self.probability_sidecars = {}
        self.probability_oof_rows = []
        self.probability_calibrators = {}
        self.probability_diagnostics = {}
        self.fp_blend_weights = {}
        self.include_season_context_training = include_season_context_training
        
    def get_feature_columns(self, df: pd.DataFrame, 
                            include_groups: Optional[List[str]] = None) -> List[str]:
        """
        Get available feature columns from the DataFrame.
        
        Args:
            df: DataFrame with features
            include_groups: List of feature groups to include (None = all)
        """
        if include_groups is None:
            include_groups = list(self.FEATURE_GROUPS.keys())
        
        all_features = []
        for group in include_groups:
            if group in self.FEATURE_GROUPS:
                all_features.extend(self.FEATURE_GROUPS[group])
        
        # Filter to columns that exist
        available = [c for c in all_features if c in df.columns]
        available = [c for c in available if c not in CURRENT_ROW_LEAK_FEATURES]

        return available
    
    def prepare_data(self, df: pd.DataFrame, target: str,
                     lines_lookup: dict = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare feature matrix, target, and per-row prop line for training.

        Returns
        -------
        X        : feature DataFrame
        y        : actual stat Series
        line_col : Series with the betting line for each row
                   (real line if known, rolling-L10 proxy if not, NaN if unavailable)
        line_source : Series with values {'real', 'proxy', ''} for each row
        """
        import unicodedata, re

        def _norm(name):
            n = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
            return re.sub(r'[^a-z0-9 ]', '', n.lower()).strip()

        if self.include_season_context_training:
            df = _seasonalize_feature_block(df, SEASON_CONTEXT_FEATURES)

        # Get feature columns
        feature_cols = self.get_feature_columns(df)
        if not self.include_season_context_training:
            feature_cols = [c for c in feature_cols if c not in SEASON_CONTEXT_FEATURES]
        feature_cols = [c for c in feature_cols if c not in CURRENT_ROW_LEAK_FEATURES]

        # Create target
        target_config = self.TARGETS.get(target, {'primary': target, 'combo': None})
        if target_config['combo']:
            y = df[target_config['combo']].sum(axis=1)
        else:
            y = df[target_config['primary']]

        X = df[feature_cols].copy()
        X = X.fillna(X.median())

        # ── Build per-row prop line ───────────────────────────────────────────
        line_col = pd.Series(np.nan, index=df.index)
        line_source = pd.Series('', index=df.index, dtype='object')

        # 1. Real lines from historical_lines.csv (only available for recent games)
        if lines_lookup and 'player' in df.columns and 'game_date' in df.columns:
            date_str = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')
            player_norm = df['player'].apply(_norm)
            for i in df.index:
                key = (player_norm.at[i], date_str.at[i], target)
                if key in lines_lookup:
                    line_col.at[i] = lines_lookup[key]
                    line_source.at[i] = 'real'

        # 2. Recency-weighted proxies for rows without a real line
        proxy_candidates = [
            _sum_target_feature(df, target, 'best_estimate'),
            _sum_target_feature(df, target, 'recency_anchor'),
            _sum_target_feature(df, target, 'season_avg'),
        ]
        proxy_col = PROXY_LINE_COL.get(target)
        if proxy_col and proxy_col in df.columns:
            proxy_candidates.append(pd.to_numeric(df[proxy_col], errors='coerce'))

        for proxy_series in proxy_candidates:
            if proxy_series is None:
                continue
            missing_mask = line_col.isna()
            if not missing_mask.any():
                break
            proxy_vals = pd.to_numeric(proxy_series.loc[missing_mask], errors='coerce')
            line_col.loc[missing_mask] = proxy_vals
            proxy_filled = missing_mask.copy()
            proxy_filled.loc[missing_mask] = proxy_vals.notna().values
            line_source.loc[proxy_filled] = 'proxy'

        return X, y, line_col, line_source

    def _train_blended_regression(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                                  target: str, line_col: pd.Series, line_source: pd.Series,
                                  sample_weights: np.ndarray, reg_params: Dict,
                                  n_splits: int, fp_series: pd.Series) -> Dict[str, object]:
        """Train complementary base regressors, then blend them with a positive Ridge meta-model."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_rows = []
        base_cols = [f'pred_{name}' for name in BLEND_BASE_MODEL_ORDER]

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_tr = sample_weights[train_idx]

            row = {
                'fold': fold,
                'actual': y_val.values,
                'line': pd.to_numeric(line_col.iloc[val_idx], errors='coerce').values,
                'line_source': line_source.iloc[val_idx].values,
                'game_date': pd.to_datetime(df.iloc[val_idx]['game_date']).values,
                'sample_weight': sample_weights[val_idx],
                'pred_fp': pd.to_numeric(fp_series.iloc[val_idx], errors='coerce').fillna(0.0).values,
            }
            if 'player' in df.columns:
                row['player_norm'] = df.iloc[val_idx]['player'].apply(_normalize_player_name).values

            for model_name in [m for m in BLEND_BASE_MODEL_ORDER if m != 'fp']:
                model = _make_blend_base_models(target, reg_params)[model_name]
                model = _fit_regression_model(model_name, model, X_tr, y_tr, w_tr)
                row[f'pred_{model_name}'] = _predict_regression_model(model, X_val)

            oof_rows.append(pd.DataFrame(row))

        oof_df = pd.concat(oof_rows, ignore_index=True).sort_values('game_date').reset_index(drop=True)
        meta_cv_splits = max(3, min(5, len(oof_df) // 300))
        blend_oof = np.full(len(oof_df), np.nan)

        if meta_cv_splits >= 3:
            meta_cv = TimeSeriesSplit(n_splits=meta_cv_splits)
            for meta_train_idx, meta_val_idx in meta_cv.split(oof_df):
                blender = Ridge(alpha=1.0, positive=True)
                blender.fit(
                    oof_df.iloc[meta_train_idx][base_cols],
                    oof_df.iloc[meta_train_idx]['actual'],
                    sample_weight=oof_df.iloc[meta_train_idx]['sample_weight'].values,
                )
                blend_oof[meta_val_idx] = blender.predict(oof_df.iloc[meta_val_idx][base_cols])

        valid_mask = ~np.isnan(blend_oof)
        if valid_mask.sum() < 200:
            oof_df['pred'] = oof_df['pred_xgb']
        else:
            oof_df = oof_df.loc[valid_mask].copy()
            oof_df['pred'] = blend_oof[valid_mask]

        cv_results, cal_records, avg_mae, avg_rmse, avg_bet_acc, avg_proxy_bet_acc, avg_all_bet_acc, mean_bias = _score_predictions(oof_df)

        final_base_models = _make_blend_base_models(target, reg_params)
        for model_name, model in final_base_models.items():
            final_base_models[model_name] = _fit_regression_model(
                model_name, model, X, y, sample_weights
            )

        final_blender = Ridge(alpha=1.0, positive=True)
        final_blender.fit(
            oof_df[base_cols],
            oof_df['actual'],
            sample_weight=oof_df['sample_weight'].values,
        )
        final_model = {
            'kind': 'positive_ridge_blend_regressor',
            'base_models': final_base_models,
            'base_order': BLEND_BASE_MODEL_ORDER,
            'blender': final_blender,
            'feature_names': list(X.columns),
            'target': target,
        }

        return {
            'model': final_model,
            'oof_frame': oof_df,
            'cv_results': cv_results,
            'cal_records': cal_records,
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'avg_bet_accuracy': avg_bet_acc,
            'avg_proxy_bet_accuracy': avg_proxy_bet_acc,
            'avg_all_line_bet_accuracy': avg_all_bet_acc,
            'oof_bias': mean_bias,
            'blend_summary': {
                'base_models': BLEND_BASE_MODEL_ORDER,
                'coefficients': {
                    name: float(weight)
                    for name, weight in zip(BLEND_BASE_MODEL_ORDER, final_blender.coef_)
                },
                'intercept': float(final_blender.intercept_),
                'n_oof_rows': int(len(oof_df)),
            },
        }

    def _store_probability_oof(self, oof_df: Optional[pd.DataFrame]) -> None:
        if oof_df is None or len(oof_df) == 0:
            return
        self.probability_oof_rows.append(oof_df.copy())

    def _train_quantile_probability_sidecar(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                                            target: str, line_col: pd.Series, line_source: pd.Series,
                                            sample_weights: np.ndarray, reg_params: Dict,
                                            n_splits: int) -> Optional[Dict[str, object]]:
        oof_preds = {tag: np.full(len(X), np.nan, dtype=float) for tag in quantile_tags()}
        tscv = TimeSeriesSplit(n_splits=n_splits)
        print(f"  Training probability sidecar for {target.upper()} ({len(QUANTILE_LEVELS)} quantiles x {n_splits} folds)...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            fold_start = time.perf_counter()
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            w_tr = sample_weights[train_idx]
            for alpha, tag in zip(QUANTILE_LEVELS, quantile_tags()):
                model = _make_quantile_sidecar_model(target, alpha, reg_params)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                oof_preds[tag][val_idx] = np.asarray(model.predict(X_val), dtype=float)
            print(f"    Quantile fold {fold}/{n_splits} complete in {time.perf_counter() - fold_start:.1f}s")

        valid_mask = np.ones(len(X), dtype=bool)
        for tag in quantile_tags():
            valid_mask &= ~np.isnan(oof_preds[tag])
        if valid_mask.sum() < 200:
            print(f"  Probability sidecar skipped for {target.upper()} - only {int(valid_mask.sum())} OOF rows")
            return None

        q_matrix = ensure_monotonic_quantiles(
            np.column_stack([oof_preds[tag][valid_mask] for tag in quantile_tags()])
        )
        oof_df = pd.DataFrame({
            'target': target,
            'game_date': pd.to_datetime(df.loc[valid_mask, 'game_date']).values,
            'actual': pd.to_numeric(y.loc[valid_mask], errors='coerce').values,
            'line': pd.to_numeric(line_col.loc[valid_mask], errors='coerce').values,
            'line_source': line_source.loc[valid_mask].values,
            'sample_weight': sample_weights[valid_mask],
        })
        if 'player' in df.columns:
            oof_df['player_norm'] = df.loc[valid_mask, 'player'].apply(_normalize_player_name).values
        for idx, tag in enumerate(quantile_tags()):
            oof_df[tag] = q_matrix[:, idx]
        with_line_mask = oof_df['line'].notna()
        oof_df['p_over_raw'] = np.nan
        oof_df['actual_over'] = np.nan
        if with_line_mask.any():
            oof_df.loc[with_line_mask, 'p_over_raw'] = over_probability_from_quantiles(
                oof_df.loc[with_line_mask, 'line'].values,
                q_matrix[with_line_mask.values],
                QUANTILE_LEVELS,
            )
            oof_df.loc[with_line_mask, 'actual_over'] = (
                oof_df.loc[with_line_mask, 'actual'].values > oof_df.loc[with_line_mask, 'line'].values
            ).astype(int)

        real_mask = oof_df['line_source'].eq('real') & oof_df['line'].notna() & oof_df['actual_over'].notna()
        raw_metrics = compute_binary_probability_metrics(
            oof_df.loc[real_mask, 'actual_over'].astype(int).values,
            oof_df.loc[real_mask, 'p_over_raw'].astype(float).values,
        )
        pinball = {}
        for alpha, tag in zip(QUANTILE_LEVELS, quantile_tags()):
            pinball[tag] = float(mean_pinball_loss(
                oof_df['actual'].values,
                oof_df[tag].values,
                alpha=alpha,
                sample_weight=oof_df['sample_weight'].values,
            ))

        final_models = {}
        print(f"  Training final quantile models for {target.upper()}...")
        for alpha, tag in zip(QUANTILE_LEVELS, quantile_tags()):
            tag_start = time.perf_counter()
            model = _make_quantile_sidecar_model(target, alpha, reg_params)
            model.fit(X, y, sample_weight=sample_weights)
            final_models[tag] = model
            print(f"    Final quantile {tag} complete in {time.perf_counter() - tag_start:.1f}s")

        sidecar = {
            'kind': 'quantile_probability_sidecar',
            'target': target,
            'feature_names': list(X.columns),
            'quantiles': list(QUANTILE_LEVELS),
            'quantile_tags': quantile_tags(),
            'models': final_models,
        }
        summary = {
            'kind': 'quantile_probability_sidecar',
            'n_oof_rows': int(len(oof_df)),
            'n_real_line_rows': int(real_mask.sum()),
            'pinball_loss': pinball,
            'raw_probability_metrics': raw_metrics,
        }
        brier = raw_metrics.get('brier')
        log_loss_val = raw_metrics.get('log_loss')
        ece = raw_metrics.get('ece')
        brier_txt = f"{brier:.4f}" if brier is not None else "n/a"
        log_loss_txt = f"{log_loss_val:.4f}" if log_loss_val is not None else "n/a"
        ece_txt = f"{ece:.4f}" if ece is not None else "n/a"
        print(f"  Probability sidecar ({target.upper()}): Brier={brier_txt} LogLoss={log_loss_txt} ECE={ece_txt}")
        return {'sidecar': sidecar, 'oof_frame': oof_df, 'summary': summary}

    def _train_ordinal_probability_sidecar(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                                           target: str, line_col: pd.Series, line_source: pd.Series,
                                           sample_weights: np.ndarray, clf_params: Dict,
                                           n_splits: int) -> Optional[Dict[str, object]]:
        max_threshold = int(min(4, max(2, np.nanpercentile(y.values, 97))))
        thresholds = []
        for threshold in range(1, max_threshold + 1):
            positives = int((y >= threshold).sum())
            negatives = int((y < threshold).sum())
            if positives >= 75 and negatives >= 75:
                thresholds.append(threshold)
        if not thresholds:
            print(f"  Probability sidecar skipped for {target.upper()} - no stable ordinal thresholds")
            return None

        oof_preds = {tail_threshold_tag(threshold): np.full(len(X), np.nan, dtype=float) for threshold in thresholds}
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            w_tr = sample_weights[train_idx]
            for threshold in thresholds:
                tag = tail_threshold_tag(threshold)
                y_tr = (y.iloc[train_idx] >= threshold).astype(int)
                if y_tr.nunique() < 2:
                    continue
                model = _make_ordinal_probability_model(target, threshold, clf_params)
                model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
                oof_preds[tag][val_idx] = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)

        valid_mask = np.ones(len(X), dtype=bool)
        for threshold in thresholds:
            valid_mask &= ~np.isnan(oof_preds[tail_threshold_tag(threshold)])
        if valid_mask.sum() < 150:
            print(f"  Probability sidecar skipped for {target.upper()} - only {int(valid_mask.sum())} OOF rows")
            return None

        prob_matrix = ensure_monotonic_tail_probs(
            np.column_stack([oof_preds[tail_threshold_tag(threshold)][valid_mask] for threshold in thresholds])
        )
        oof_df = pd.DataFrame({
            'target': target,
            'game_date': pd.to_datetime(df.loc[valid_mask, 'game_date']).values,
            'actual': pd.to_numeric(y.loc[valid_mask], errors='coerce').values,
            'line': pd.to_numeric(line_col.loc[valid_mask], errors='coerce').values,
            'line_source': line_source.loc[valid_mask].values,
            'sample_weight': sample_weights[valid_mask],
        })
        if 'player' in df.columns:
            oof_df['player_norm'] = df.loc[valid_mask, 'player'].apply(_normalize_player_name).values
        for idx, threshold in enumerate(thresholds):
            oof_df[tail_threshold_tag(threshold)] = prob_matrix[:, idx]
        with_line_mask = oof_df['line'].notna()
        oof_df['p_over_raw'] = np.nan
        oof_df['actual_over'] = np.nan
        if with_line_mask.any():
            oof_df.loc[with_line_mask, 'p_over_raw'] = ordinal_over_probability(
                oof_df.loc[with_line_mask, 'line'].values,
                prob_matrix[with_line_mask.values],
                thresholds,
            )
            oof_df.loc[with_line_mask, 'actual_over'] = (
                oof_df.loc[with_line_mask, 'actual'].values > oof_df.loc[with_line_mask, 'line'].values
            ).astype(int)

        real_mask = oof_df['line_source'].eq('real') & oof_df['line'].notna() & oof_df['actual_over'].notna()
        raw_metrics = compute_binary_probability_metrics(
            oof_df.loc[real_mask, 'actual_over'].astype(int).values,
            oof_df.loc[real_mask, 'p_over_raw'].astype(float).values,
        )

        final_models = {}
        for threshold in thresholds:
            tag = tail_threshold_tag(threshold)
            y_full = (y >= threshold).astype(int)
            if y_full.nunique() < 2:
                continue
            model = _make_ordinal_probability_model(target, threshold, clf_params)
            model.fit(X, y_full, sample_weight=sample_weights, verbose=False)
            final_models[tag] = model
        if not final_models:
            return None

        sidecar = {
            'kind': 'ordinal_tail_probability_sidecar',
            'target': target,
            'feature_names': list(X.columns),
            'thresholds': thresholds,
            'models': final_models,
        }
        summary = {
            'kind': 'ordinal_tail_probability_sidecar',
            'n_oof_rows': int(len(oof_df)),
            'n_real_line_rows': int(real_mask.sum()),
            'thresholds': thresholds,
            'raw_probability_metrics': raw_metrics,
        }
        brier = raw_metrics.get('brier')
        log_loss_val = raw_metrics.get('log_loss')
        ece = raw_metrics.get('ece')
        brier_txt = f"{brier:.4f}" if brier is not None else "n/a"
        log_loss_txt = f"{log_loss_val:.4f}" if log_loss_val is not None else "n/a"
        ece_txt = f"{ece:.4f}" if ece is not None else "n/a"
        print(f"  Probability sidecar ({target.upper()}): Brier={brier_txt} LogLoss={log_loss_txt} ECE={ece_txt}")
        return {'sidecar': sidecar, 'oof_frame': oof_df, 'summary': summary}

    def _train_poisson_probability_sidecar(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                                           target: str, line_col: pd.Series, line_source: pd.Series,
                                           sample_weights: np.ndarray, reg_params: Dict,
                                           n_splits: int) -> Optional[Dict[str, object]]:
        rate_oof = np.full(len(X), np.nan, dtype=float)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            w_tr = sample_weights[train_idx]
            model = _make_poisson_probability_model(target, reg_params)
            model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            rate_oof[val_idx] = np.clip(np.asarray(model.predict(X_val), dtype=float), 1e-6, None)

        valid_mask = ~np.isnan(rate_oof)
        if valid_mask.sum() < 150:
            print(f"  Probability sidecar skipped for {target.upper()} - only {int(valid_mask.sum())} OOF rows")
            return None

        oof_df = pd.DataFrame({
            'target': target,
            'game_date': pd.to_datetime(df.loc[valid_mask, 'game_date']).values,
            'actual': pd.to_numeric(y.loc[valid_mask], errors='coerce').values,
            'line': pd.to_numeric(line_col.loc[valid_mask], errors='coerce').values,
            'line_source': line_source.loc[valid_mask].values,
            'sample_weight': sample_weights[valid_mask],
            'rate': rate_oof[valid_mask],
        })
        if 'player' in df.columns:
            oof_df['player_norm'] = df.loc[valid_mask, 'player'].apply(_normalize_player_name).values
        with_line_mask = oof_df['line'].notna()
        oof_df['p_over_raw'] = np.nan
        oof_df['actual_over'] = np.nan
        if with_line_mask.any():
            oof_df.loc[with_line_mask, 'p_over_raw'] = poisson_over_probability(
                oof_df.loc[with_line_mask, 'line'].values,
                oof_df.loc[with_line_mask, 'rate'].values,
            )
            oof_df.loc[with_line_mask, 'actual_over'] = (
                oof_df.loc[with_line_mask, 'actual'].values > oof_df.loc[with_line_mask, 'line'].values
            ).astype(int)

        real_mask = oof_df['line_source'].eq('real') & oof_df['line'].notna() & oof_df['actual_over'].notna()
        raw_metrics = compute_binary_probability_metrics(
            oof_df.loc[real_mask, 'actual_over'].astype(int).values,
            oof_df.loc[real_mask, 'p_over_raw'].astype(float).values,
        )

        final_model = _make_poisson_probability_model(target, reg_params)
        final_model.fit(X, y, sample_weight=sample_weights, verbose=False)
        sidecar = {
            'kind': 'poisson_probability_sidecar',
            'target': target,
            'feature_names': list(X.columns),
            'model': final_model,
        }
        summary = {
            'kind': 'poisson_probability_sidecar',
            'n_oof_rows': int(len(oof_df)),
            'n_real_line_rows': int(real_mask.sum()),
            'raw_probability_metrics': raw_metrics,
        }
        brier = raw_metrics.get('brier')
        log_loss_val = raw_metrics.get('log_loss')
        ece = raw_metrics.get('ece')
        brier_txt = f"{brier:.4f}" if brier is not None else "n/a"
        log_loss_txt = f"{log_loss_val:.4f}" if log_loss_val is not None else "n/a"
        ece_txt = f"{ece:.4f}" if ece is not None else "n/a"
        print(f"  Probability sidecar ({target.upper()}): Brier={brier_txt} LogLoss={log_loss_txt} ECE={ece_txt}")
        return {'sidecar': sidecar, 'oof_frame': oof_df, 'summary': summary}

    def get_probability_oof_frame(self) -> pd.DataFrame:
        if not self.probability_oof_rows:
            return pd.DataFrame()
        return pd.concat(self.probability_oof_rows, ignore_index=True).sort_values(
            ['game_date', 'target']
        ).reset_index(drop=True)

    def fit_probability_calibrators(self) -> Dict[str, object]:
        prob_oof = self.get_probability_oof_frame()
        if len(prob_oof) == 0:
            self.probability_calibrators = {}
            self.probability_diagnostics = {}
            return {'calibrators': {}, 'diagnostics': {}}
        real_only = prob_oof[prob_oof['line_source'].eq('real')].copy()
        calibrators, diagnostics = fit_isotonic_calibrators(real_only)
        self.probability_calibrators = calibrators
        self.probability_diagnostics = diagnostics
        return {'calibrators': calibrators, 'diagnostics': diagnostics}

    def train_first_principles_blend(self, projection_dir: Optional[Path] = None) -> Dict[str, dict]:
        """Fit learned live blend weights between model OOF predictions and first-principles projections."""
        projection_dir = projection_dir or (DATA_DIR / 'projection_archive')
        archive = _load_projection_archive_predictions(Path(projection_dir))
        if archive.empty:
            print("  No projection archive found; skipping learned first-principles blend.")
            self.fp_blend_weights = {}
            return {}

        fp_col_map = {
            'pts': 'proj_pts_final',
            'trb': 'proj_reb',
            'ast': 'proj_ast',
            'pra': 'proj_pra',
            'pr': 'proj_pr',
            'pa': 'proj_pa',
        }

        results = {}
        for target, fp_col in fp_col_map.items():
            oof_df = self.regression_oof.get(target)
            if oof_df is None or len(oof_df) == 0 or fp_col not in archive.columns or 'player_norm' not in oof_df.columns:
                continue

            merged = oof_df.merge(
                archive[['game_date', 'player_norm', fp_col]],
                on=['game_date', 'player_norm'],
                how='inner',
            ).dropna(subset=['pred', 'actual', fp_col])
            if len(merged) < 100:
                continue

            merged = merged.sort_values('game_date').reset_index(drop=True)
            cv_splits = max(3, min(5, len(merged) // 60))
            blend_preds = np.full(len(merged), np.nan)
            if cv_splits >= 3:
                tscv = TimeSeriesSplit(n_splits=cv_splits)
                for train_idx, val_idx in tscv.split(merged):
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(
                        merged.iloc[train_idx][['pred', fp_col]],
                        merged.iloc[train_idx]['actual'],
                        sample_weight=merged.iloc[train_idx]['sample_weight'].values,
                    )
                    blend_preds[val_idx] = ridge.predict(merged.iloc[val_idx][['pred', fp_col]])

            valid_mask = ~np.isnan(blend_preds)
            ridge = Ridge(alpha=1.0)
            ridge.fit(
                merged[['pred', fp_col]],
                merged['actual'],
                sample_weight=merged['sample_weight'].values,
            )

            payload = {
                'model_weight': float(ridge.coef_[0]),
                'fp_weight': float(ridge.coef_[1]),
                'intercept': float(ridge.intercept_),
                'n_rows': int(len(merged)),
                'fp_column': fp_col,
            }
            if valid_mask.sum() >= 50:
                payload['cv_mae_model_only'] = float(mean_absolute_error(merged.loc[valid_mask, 'actual'], merged.loc[valid_mask, 'pred']))
                payload['cv_mae_fp_only'] = float(mean_absolute_error(merged.loc[valid_mask, 'actual'], merged.loc[valid_mask, fp_col]))
                payload['cv_mae_blend'] = float(mean_absolute_error(merged.loc[valid_mask, 'actual'], blend_preds[valid_mask]))

            results[target] = payload

        self.fp_blend_weights = results
        return results
    
    def train_model(self, df: pd.DataFrame, target: str,
                    n_splits: int = 5,
                    lines_lookup: dict = None,
                    importance_threshold: float = 0.001) -> Dict:
        """
        Train one regression model (raw stat) and one classifier (OVER/UNDER vs line).

        Evaluation:
          - Regression accuracy = (pred > line) == (actual > line), aggregated by edge bucket
          - CLV-style correlation: how well pred-line tracks actual-line deviation
        Training:
          - Recency-weighted: 2025-26 rows get 3x, 2024-25 2x, prior 1x
          - Per-target hyperparameters (REGRESSION_PARAMS / CLASSIFIER_PARAMS)
          - Feature selection: drop features with avg importance < threshold
        """
        print(f"\n{'='*60}")
        print(f"Training model for: {target.upper()}")
        print("="*60)

        X, y, line_col, line_source = self.prepare_data(df, target, lines_lookup=lines_lookup)
        sample_weights = _get_sample_weights(df)
        n_real_lines = int(line_source.eq('real').sum())
        n_proxy_lines = int(line_source.eq('proxy').sum())
        n_with_lines = n_real_lines + n_proxy_lines
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}  "
              f"| Real lines: {n_real_lines:,} ({n_real_lines/len(X):.0%})  "
              f"| Proxy lines: {n_proxy_lines:,} ({n_proxy_lines/len(X):.0%})")

        # ── Per-target params ──────────────────────────────────────────────────
        reg_params = {
            'objective': 'reg:squarederror',
            'random_state': 42, 'n_jobs': -1,
            **REGRESSION_PARAMS.get(target, dict(
                max_depth=6, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0)),
        }
        clf_params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'random_state': 42, 'n_jobs': -1,
            **CLASSIFIER_PARAMS.get(target, dict(
                max_depth=5, learning_rate=0.05, n_estimators=250,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0)),
        }
        edge_params = {
            'objective': 'reg:squarederror',
            'random_state': 42, 'n_jobs': -1,
            **EDGE_REGRESSION_PARAMS.get(target, dict(
                max_depth=4, learning_rate=0.05, n_estimators=220,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0)),
        }
        meta_params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'random_state': 42, 'n_jobs': -1,
            **META_CLASSIFIER_PARAMS.get(target, dict(
                max_depth=4, learning_rate=0.05, n_estimators=180,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.15, reg_lambda=1.0)),
        }

        # ── Cross-validation ──────────────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []
        feature_imp = []
        cal_records = []   # real-line records for calibration table
        market_records = []
        oof_errors = []
        meta_oof_frames = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_val   = X.iloc[train_idx],          X.iloc[val_idx]
            y_tr, y_val   = y.iloc[train_idx],           y.iloc[val_idx]
            w_tr           = sample_weights[train_idx]
            line_tr        = line_col.iloc[train_idx]
            line_val       = line_col.iloc[val_idx]
            source_tr      = line_source.iloc[train_idx]
            source_val     = line_source.iloc[val_idx]
            df_val         = df.iloc[val_idx]

            model = xgb.XGBRegressor(**reg_params)
            model.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)
            oof_errors.extend((y_pred - y_val.values).tolist())

            mae  = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            all_mask   = line_val.notna()
            real_mask  = source_val.eq('real') & all_mask
            proxy_mask = source_val.eq('proxy') & all_mask

            def _score_direction(mask: pd.Series, collect_calibration: bool = False) -> Tuple[float, int]:
                n_scored = int(mask.sum())
                if n_scored < 10:
                    return np.nan, n_scored

                lines_arr  = line_val[mask].values
                actual_arr = y_val[mask].values
                pred_arr   = y_pred[mask]
                edge_arr   = pred_arr - lines_arr
                beat_line  = (actual_arr > lines_arr).astype(int)
                if collect_calibration:
                    for e, b, a, l, p in zip(edge_arr, beat_line, actual_arr, lines_arr, pred_arr):
                        cal_records.append({
                            'edge': e,
                            'actual_beat_line': b,
                            'actual': a,
                            'line': l,
                            'pred': p,
                        })
                hit_arr = ((edge_arr > 0) == beat_line.astype(bool)).astype(float)
                return float(hit_arr.mean()), n_scored

            real_bet_acc, n_real_scored = _score_direction(real_mask, collect_calibration=True)
            proxy_bet_acc, n_proxy_scored = _score_direction(proxy_mask)
            all_bet_acc, n_scored = _score_direction(all_mask)

            real_train_mask = source_tr.eq('real') & line_tr.notna()
            if int(real_train_mask.sum()) >= 400 and int(real_mask.sum()) >= 25:
                market_train = _build_market_feature_block(
                    df.iloc[train_idx].loc[real_train_mask], target, line_tr.loc[real_train_mask]
                )
                market_val = _build_market_feature_block(
                    df_val.loc[real_mask],
                    target,
                    line_val.loc[real_mask],
                )
                y_edge_train = y_tr.loc[real_train_mask] - line_tr.loc[real_train_mask]
                y_edge_val = y_val.loc[real_mask] - line_val.loc[real_mask]

                edge_model = xgb.XGBRegressor(**edge_params)
                edge_model.fit(
                    market_train,
                    y_edge_train,
                    sample_weight=w_tr[real_train_mask.values],
                    verbose=False,
                )
                edge_pred_val = pd.Series(edge_model.predict(market_val), index=market_val.index)
                for row_idx in market_val.index:
                    market_records.append({
                        'edge': float(edge_pred_val.at[row_idx]),
                        'actual_beat_line': int(y_edge_val.at[row_idx] > 0),
                        'actual': float(y_val.at[row_idx]),
                        'line': float(line_val.at[row_idx]),
                        'pred': float(line_val.at[row_idx] + edge_pred_val.at[row_idx]),
                    })

                meta_val = _build_meta_feature_block(
                    market_val,
                    raw_edge=pd.Series(y_pred[real_mask], index=market_val.index) - line_val.loc[real_mask],
                    market_edge_pred=edge_pred_val,
                )
                meta_val['meta_target'] = (
                    ((edge_pred_val > 0) == (y_edge_val > 0)) &
                    (y_edge_val.abs() >= 0.75)
                ).astype(int).values
                meta_val['game_date'] = pd.to_datetime(df_val.loc[real_mask, 'game_date']).values
                meta_val['sample_weight'] = sample_weights[val_idx][real_mask.values]
                meta_oof_frames.append(meta_val.reset_index(drop=True))

            cv_results.append({'fold': fold, 'mae': mae, 'rmse': rmse,
                                'bet_accuracy': real_bet_acc,
                                'proxy_bet_accuracy': proxy_bet_acc,
                                'all_line_bet_accuracy': all_bet_acc,
                                'n_scored': n_scored,
                                'n_real_scored': n_real_scored,
                                'n_proxy_scored': n_proxy_scored})
            feature_imp.append(pd.DataFrame({'feature': X.columns,
                                             'importance': model.feature_importances_}))

            real_str = f"{real_bet_acc:.1%} (n={n_real_scored})" if not np.isnan(real_bet_acc) else "n/a"
            proxy_str = f"{proxy_bet_acc:.1%} (n={n_proxy_scored})" if not np.isnan(proxy_bet_acc) else "n/a"
            print(f"  Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
                  f"RealBetAcc={real_str}, ProxyBetAcc={proxy_str}")

        # ── CV summary ────────────────────────────────────────────────────────
        avg_mae   = float(np.mean([r['mae']  for r in cv_results]))
        avg_rmse  = float(np.mean([r['rmse'] for r in cv_results]))
        valid_real_acc = [r['bet_accuracy'] for r in cv_results if not np.isnan(r['bet_accuracy'])]
        valid_proxy_acc = [r['proxy_bet_accuracy'] for r in cv_results if not np.isnan(r['proxy_bet_accuracy'])]
        valid_all_acc = [r['all_line_bet_accuracy'] for r in cv_results if not np.isnan(r['all_line_bet_accuracy'])]
        avg_bet_acc = float(np.mean(valid_real_acc)) if valid_real_acc else np.nan
        avg_proxy_bet_acc = float(np.mean(valid_proxy_acc)) if valid_proxy_acc else np.nan
        avg_all_bet_acc = float(np.mean(valid_all_acc)) if valid_all_acc else np.nan
        real_str = f"{avg_bet_acc:.1%}" if not np.isnan(avg_bet_acc) else "n/a"
        proxy_str = f"{avg_proxy_bet_acc:.1%}" if not np.isnan(avg_proxy_bet_acc) else "n/a"
        all_str = f"{avg_all_bet_acc:.1%}" if not np.isnan(avg_all_bet_acc) else "n/a"
        print(f"\n  Average: MAE={avg_mae:.2f}, RMSE={avg_rmse:.2f}, "
              f"RealBetAcc={real_str}, ProxyBetAcc={proxy_str}, AllLines={all_str}")

        mean_bias = float(np.mean(oof_errors)) if oof_errors else 0.0
        self.biases[target] = round(mean_bias, 4)
        print(f"  OOF bias: {mean_bias:+.3f}")

        # ── Edge calibration table ────────────────────────────────────────────
        real_line_summary = _summarize_calibration(cal_records)
        _print_calibration(cal_records, target, label='real lines')
        market_edge_summary = _summarize_calibration(market_records)
        _print_calibration(market_records, target, label='market-edge model')

        # ── Feature selection ─────────────────────────────────────────────────
        all_imp  = pd.concat(feature_imp).groupby('feature')['importance'].mean()
        selected = all_imp[all_imp >= importance_threshold].index.tolist()
        dropped  = len(all_imp) - len(selected)
        if dropped > 0:
            print(f"\n  Feature selection: {len(selected)}/{len(all_imp)} kept "
                  f"({dropped} dropped, importance < {importance_threshold})")
        X_sel = X[selected]

        # ── Final regression model ────────────────────────────────────────────
        blend_summary = None
        if target in BLEND_REGRESSION_TARGETS:
            fp_series = _build_fp_projection_proxy(df, target).fillna(_sum_target_feature(df, target, 'best_estimate')).fillna(0.0)
            print("  Training blended regression bundle...")
            blend_result = self._train_blended_regression(
                df=df,
                X=X_sel,
                y=y,
                target=target,
                line_col=line_col,
                line_source=line_source,
                sample_weights=sample_weights,
                reg_params=reg_params,
                n_splits=n_splits,
                fp_series=fp_series,
            )
            final_reg = blend_result['model']
            oof_cols = [c for c in ['game_date', 'player_norm', 'pred', 'actual', 'sample_weight'] if c in blend_result['oof_frame'].columns]
            self.regression_oof[target] = blend_result['oof_frame'][oof_cols].copy()
            cv_results = blend_result['cv_results']
            cal_records = blend_result['cal_records']
            avg_mae = blend_result['avg_mae']
            avg_rmse = blend_result['avg_rmse']
            avg_bet_acc = blend_result['avg_bet_accuracy']
            avg_proxy_bet_acc = blend_result['avg_proxy_bet_accuracy']
            avg_all_bet_acc = blend_result['avg_all_line_bet_accuracy']
            mean_bias = blend_result['oof_bias']
            self.biases[target] = round(mean_bias, 4)
            blend_summary = blend_result.get('blend_summary')
            real_str = f"{avg_bet_acc:.1%}" if not np.isnan(avg_bet_acc) else "n/a"
            proxy_str = f"{avg_proxy_bet_acc:.1%}" if not np.isnan(avg_proxy_bet_acc) else "n/a"
            all_str = f"{avg_all_bet_acc:.1%}" if not np.isnan(avg_all_bet_acc) else "n/a"
            print(f"  Blended OOF: MAE={avg_mae:.2f}, RMSE={avg_rmse:.2f}, "
                  f"RealBetAcc={real_str}, ProxyBetAcc={proxy_str}, AllLines={all_str}")
            print(f"  Blended OOF bias: {mean_bias:+.3f}")
            real_line_summary = _summarize_calibration(cal_records)
            _print_calibration(cal_records, target, label='real lines (ridge blend)')
        else:
            print("  Training final regression model...")
            final_reg = xgb.XGBRegressor(**reg_params)
            final_reg.fit(X_sel, y, sample_weight=sample_weights, verbose=False)

        self.models[target] = final_reg
        self.selected_features[target] = selected
        self.feature_importance[target] = all_imp.sort_values(ascending=False)
        probability_summary = None
        probability_sidecar_result = None
        if target in PROBABILITY_TARGETS:
            probability_sidecar_result = self._train_quantile_probability_sidecar(
                df=df,
                X=X_sel,
                y=y,
                target=target,
                line_col=line_col,
                line_source=line_source,
                sample_weights=sample_weights,
                reg_params=reg_params,
                n_splits=n_splits,
            )
        elif target in {'stl', 'blk'}:
            probability_sidecar_result = self._train_ordinal_probability_sidecar(
                df=df,
                X=X_sel,
                y=y,
                target=target,
                line_col=line_col,
                line_source=line_source,
                sample_weights=sample_weights,
                clf_params=clf_params,
                n_splits=n_splits,
            )
        elif target == 'tov':
            probability_sidecar_result = self._train_poisson_probability_sidecar(
                df=df,
                X=X_sel,
                y=y,
                target=target,
                line_col=line_col,
                line_source=line_source,
                sample_weights=sample_weights,
                reg_params=reg_params,
                n_splits=n_splits,
            )
        if probability_sidecar_result:
            self.probability_sidecars[target] = probability_sidecar_result['sidecar']
            self._store_probability_oof(probability_sidecar_result['oof_frame'])
            probability_summary = probability_sidecar_result.get('summary')
        edge_feature_names = []
        meta_feature_names = []
        meta_summary = None
        final_edge = None

        # ── Final classifier model ────────────────────────────────────────────
        # Train only on rows that have a real historical line.
        # Target: 1 = OVER (actual > line), 0 = UNDER
        clf_mask = line_source.eq('real')
        n_clf    = int(clf_mask.sum())
        if n_clf >= 500:
            print(f"  Training final classifier on {n_clf:,} rows with real lines...")
            clf_target = (y[clf_mask] > line_col[clf_mask]).astype(int)
            w_clf      = sample_weights[clf_mask.values]
            X_clf      = X_sel[clf_mask]
            final_clf  = xgb.XGBClassifier(**clf_params)
            final_clf.fit(X_clf, clf_target, sample_weight=w_clf, verbose=False)
            self.classifiers[target] = final_clf
        else:
            print(f"  Classifier skipped — only {n_clf} rows with lines (need ≥500)")

        if n_clf >= 500:
            print(f"  Training market-edge model on {n_clf:,} real-line rows...")
            market_X_full = _build_market_feature_block(df.loc[clf_mask], target, line_col.loc[clf_mask])
            edge_feature_names = market_X_full.columns.tolist()
            y_edge_full = y.loc[clf_mask] - line_col.loc[clf_mask]
            final_edge = xgb.XGBRegressor(**edge_params)
            final_edge.fit(
                market_X_full,
                y_edge_full,
                sample_weight=sample_weights[clf_mask.values],
                verbose=False,
            )
            self.edge_models[target] = final_edge
            self.market_feature_sets['edge'][target] = edge_feature_names

            if market_edge_summary is None:
                edge_pred_full = pd.Series(final_edge.predict(market_X_full), index=market_X_full.index)
                fallback_records = []
                for row_idx in market_X_full.index:
                    fallback_records.append({
                        'edge': float(edge_pred_full.at[row_idx]),
                        'actual_beat_line': int(y_edge_full.at[row_idx] > 0),
                        'actual': float(y.loc[row_idx]),
                        'line': float(line_col.at[row_idx]),
                        'pred': float(line_col.at[row_idx] + edge_pred_full.at[row_idx]),
                    })
                market_edge_summary = _summarize_calibration(fallback_records)

        meta_df = pd.concat(meta_oof_frames, ignore_index=True) if meta_oof_frames else pd.DataFrame()
        if n_clf >= 500 and final_edge is not None and (len(meta_df) < 500 or market_edge_summary is None):
            real_df = df.loc[clf_mask].copy()
            real_df['game_date'] = pd.to_datetime(real_df['game_date'], errors='coerce')
            real_df = real_df.sort_values('game_date').reset_index(drop=True)
            real_line = pd.to_numeric(line_col.loc[clf_mask], errors='coerce').reset_index(drop=True)
            real_y = y.loc[clf_mask].reset_index(drop=True)
            real_market = _build_market_feature_block(real_df, target, real_line)
            real_y_edge = (real_y - real_line).reset_index(drop=True)
            raw_proxy = (_sum_target_feature(real_df, target, 'best_estimate') - real_line).fillna(
                _sum_target_feature(real_df, target, 'l10') - real_line
            )

            dedicated_market_records = []
            dedicated_meta_frames = []
            real_splits = max(3, min(5, len(real_df) // 250))
            if real_splits >= 3:
                real_cv = TimeSeriesSplit(n_splits=real_splits)
                for edge_train_idx, edge_val_idx in real_cv.split(real_market):
                    if len(edge_train_idx) < 300 or len(edge_val_idx) < 50:
                        continue
                    edge_fold = xgb.XGBRegressor(**edge_params)
                    edge_fold.fit(
                        real_market.iloc[edge_train_idx],
                        real_y_edge.iloc[edge_train_idx],
                        sample_weight=sample_weights[clf_mask.values][edge_train_idx],
                        verbose=False,
                    )
                    edge_val_pred = pd.Series(edge_fold.predict(real_market.iloc[edge_val_idx]), index=edge_val_idx)
                    for local_idx in edge_val_idx:
                        dedicated_market_records.append({
                            'edge': float(edge_val_pred.at[local_idx]),
                            'actual_beat_line': int(real_y_edge.iloc[local_idx] > 0),
                            'actual': float(real_y.iloc[local_idx]),
                            'line': float(real_line.iloc[local_idx]),
                            'pred': float(real_line.iloc[local_idx] + edge_val_pred.at[local_idx]),
                        })
                    meta_block = _build_meta_feature_block(
                        real_market.iloc[edge_val_idx],
                        raw_edge=raw_proxy.iloc[edge_val_idx],
                        market_edge_pred=edge_val_pred,
                    )
                    meta_block['meta_target'] = (
                        ((edge_val_pred > 0) == (real_y_edge.iloc[edge_val_idx] > 0)) &
                        (real_y_edge.iloc[edge_val_idx].abs() >= 0.75)
                    ).astype(int).values
                    meta_block['game_date'] = real_df.iloc[edge_val_idx]['game_date'].values
                    meta_block['sample_weight'] = sample_weights[clf_mask.values][edge_val_idx]
                    dedicated_meta_frames.append(meta_block.reset_index(drop=True))

            if dedicated_market_records:
                market_edge_summary = _summarize_calibration(dedicated_market_records)
            if dedicated_meta_frames:
                meta_df = pd.concat(dedicated_meta_frames, ignore_index=True)
                print(f"  Dedicated real-line OOF rows: {len(meta_df):,}")

        if len(meta_df) == 0 and n_clf >= 500 and final_edge is not None:
            fallback_fp = _build_fp_projection_proxy(df.loc[clf_mask], target).fillna(_sum_target_feature(df.loc[clf_mask], target, 'best_estimate')).fillna(0.0)
            fallback_raw_pred = pd.Series(
                _predict_regression_model(final_reg, X_sel.loc[clf_mask], fp_input=fallback_fp),
                index=df.loc[clf_mask].index,
            )
            fallback_market = _build_market_feature_block(
                df.loc[clf_mask],
                target,
                line_col.loc[clf_mask],
            )
            fallback_edge_pred = pd.Series(final_edge.predict(fallback_market), index=fallback_market.index)
            fallback_meta = _build_meta_feature_block(
                fallback_market,
                raw_edge=fallback_raw_pred - line_col.loc[clf_mask],
                market_edge_pred=fallback_edge_pred,
            )
            fallback_meta['meta_target'] = (
                ((fallback_edge_pred > 0) == (y_edge_full > 0)) &
                (y_edge_full.abs() >= 0.75)
            ).astype(int).values
            fallback_meta['game_date'] = pd.to_datetime(df.loc[clf_mask, 'game_date']).values
            fallback_meta['sample_weight'] = sample_weights[clf_mask.values]
            meta_df = fallback_meta.reset_index(drop=True)
            print("  Meta model fallback: using full real-line rows because OOF coverage was too thin.")

        if len(meta_df) > 0:
            if meta_df['meta_target'].nunique() >= 2 and len(meta_df) >= 500:
                meta_df = meta_df.sort_values('game_date').reset_index(drop=True)
                meta_feature_names = [
                    c for c in meta_df.columns
                    if c not in {'meta_target', 'game_date', 'sample_weight'}
                ]
                meta_probs = np.full(len(meta_df), np.nan)
                meta_cv_splits = max(3, min(5, len(meta_df) // 300))
                if meta_cv_splits >= 3:
                    meta_cv = TimeSeriesSplit(n_splits=meta_cv_splits)
                    X_meta = meta_df[meta_feature_names]
                    y_meta = meta_df['meta_target'].astype(int)
                    w_meta = meta_df['sample_weight'].values
                    for meta_train_idx, meta_val_idx in meta_cv.split(X_meta):
                        meta_model_fold = xgb.XGBClassifier(**meta_params)
                        meta_model_fold.fit(
                            X_meta.iloc[meta_train_idx],
                            y_meta.iloc[meta_train_idx],
                            sample_weight=w_meta[meta_train_idx],
                            verbose=False,
                        )
                        meta_probs[meta_val_idx] = meta_model_fold.predict_proba(X_meta.iloc[meta_val_idx])[:, 1]
                    valid_meta = ~np.isnan(meta_probs)
                    if valid_meta.sum() >= 100:
                        auc = roc_auc_score(y_meta[valid_meta], meta_probs[valid_meta])
                        top_mask = meta_probs[valid_meta] >= 0.60
                        top_hit = float(y_meta[valid_meta][top_mask].mean()) if top_mask.sum() else None
                        meta_summary = {
                            'n_records': int(valid_meta.sum()),
                            'positive_rate': float(y_meta[valid_meta].mean()),
                            'auc': float(auc),
                            'top_prob_threshold': 0.60,
                            'top_prob_n': int(top_mask.sum()),
                            'top_prob_hit_rate': float(top_hit) if top_hit is not None else None,
                        }
                        if top_hit is not None:
                            print(f"  Abstention meta-model AUC={auc:.3f}; top-prob hit={top_hit:.1%}")
                        else:
                            print(f"  Abstention meta-model AUC={auc:.3f}")

                final_meta = xgb.XGBClassifier(**meta_params)
                final_meta.fit(
                    meta_df[meta_feature_names],
                    meta_df['meta_target'].astype(int),
                    sample_weight=meta_df['sample_weight'].values,
                    verbose=False,
                )
                self.meta_models[target] = final_meta
                self.market_feature_sets['meta'][target] = meta_feature_names

        return {
            'target':              target,
            'cv_results':          cv_results,
            'avg_mae':             avg_mae,
            'avg_rmse':            avg_rmse,
            'avg_bet_accuracy':    avg_bet_acc,
            'avg_proxy_bet_accuracy': avg_proxy_bet_acc,
            'avg_all_line_bet_accuracy': avg_all_bet_acc,
            'n_features_original': X.shape[1],
            'n_features_selected': len(selected),
            'n_samples':           X.shape[0],
            'n_real_lines':        n_real_lines,
            'n_proxy_lines':       n_proxy_lines,
            'n_cal_records':       len(cal_records),
            'oof_bias':            mean_bias,
            'real_line_summary':   real_line_summary,
            'market_edge_summary': market_edge_summary,
            'market_edge_sign_accuracy': market_edge_summary.get('hit_rate') if market_edge_summary else None,
            'meta_summary':        meta_summary,
            'blend_summary':       blend_summary,
            'probability_summary': probability_summary,
        }
    
    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """Train models for all targets."""
        results = {}
        
        for target in self.TARGETS.keys():
            try:
                results[target] = self.train_model(df, target)
            except Exception as e:
                print(f"Error training {target}: {e}")
                results[target] = {'error': str(e)}
        
        return results
    
    def _get_X(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Prepare X aligned to the selected feature set for this target."""
        X, _, _, _ = self.prepare_data(df, target)
        selected = self.selected_features.get(target)
        if selected:
            X = X[[c for c in selected if c in X.columns]]
        return X

    def predict(self, df: pd.DataFrame, target: str) -> np.ndarray:
        """Predict raw stat value (regression model)."""
        if target not in self.models:
            raise ValueError(f"No regression model for {target}")
        fp_input = _build_fp_projection_proxy(df, target) if target in BLEND_REGRESSION_TARGETS else None
        return _predict_regression_model(self.models[target], self._get_X(df, target), fp_input=fp_input)

    def predict_proba(self, df: pd.DataFrame, target: str) -> np.ndarray:
        """Predict P(OVER) from the classifier model. Returns array of float in [0,1]."""
        if target not in self.classifiers:
            raise ValueError(f"No classifier for {target}")
        return self.classifiers[target].predict_proba(self._get_X(df, target))[:, 1]

    def predict_probability_details(self, df: pd.DataFrame, target: str,
                                    lines: Optional[pd.Series] = None,
                                    calibrators: Optional[Dict[str, object]] = None) -> Dict[str, np.ndarray]:
        if target not in self.probability_sidecars:
            raise ValueError(f"No probability sidecar for {target}")
        sidecar = self.probability_sidecars[target]
        X_prob = self._get_X(df, target)
        details = _predict_probability_sidecar(sidecar, X_prob, lines=lines)
        cal_map = calibrators if calibrators is not None else self.probability_calibrators
        raw = details.get('p_over_raw')
        if raw is not None:
            details['p_over_cal'] = apply_isotonic_calibrators(raw, target, cal_map)
        return details

    def predict_quantiles(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        if target not in self.probability_sidecars:
            raise ValueError(f"No probability sidecar for {target}")
        details = self.predict_probability_details(df, target, lines=None)
        qtags = [tag for tag in quantile_tags() if tag in details]
        return pd.DataFrame({tag: details[tag] for tag in qtags}, index=df.index)
    
    def save_models(self, suffix: str = ''):
        """Save regression models, classifiers, feature lists, importances, and bias calibration."""
        def _load_json_dict(path: Path, default):
            if not path.exists():
                return default
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return default

        # Regression models
        for target, model in self.models.items():
            path = self.model_dir / f"xgb_{target}{suffix}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(model, f)

        # Classifiers
        for target, clf in self.classifiers.items():
            path = self.model_dir / f"clf_{target}{suffix}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(clf, f)
        if self.classifiers:
            print(f"  Classifiers saved: {sorted(self.classifiers.keys())}")

        for target, model in self.edge_models.items():
            path = self.model_dir / f"edge_xgb_{target}{suffix}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(model, f)

        for target, clf in self.meta_models.items():
            path = self.model_dir / f"meta_clf_{target}{suffix}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(clf, f)

        for target, sidecar in self.probability_sidecars.items():
            path = self.model_dir / f"prob_sidecar_{target}{suffix}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(sidecar, f)

        # Feature importance
        imp_path = self.model_dir / f"feature_importance{suffix}.json"
        merged_importance = _load_json_dict(imp_path, {})
        merged_importance.update({k: v.to_dict() for k, v in self.feature_importance.items()})
        with open(imp_path, 'w') as f:
            json.dump(merged_importance, f, indent=2)

        # Selected feature lists (required so inference uses same columns as training)
        if self.selected_features:
            sel_path = self.model_dir / f"selected_features{suffix}.json"
            merged_selected = _load_json_dict(sel_path, {})
            merged_selected.update(self.selected_features)
            with open(sel_path, 'w') as f:
                json.dump(merged_selected, f, indent=2)

        if any(self.market_feature_sets.values()):
            market_path = self.model_dir / f"market_feature_sets{suffix}.json"
            merged_market = _load_json_dict(market_path, {'edge': {}, 'meta': {}})
            for layer_name, target_map in self.market_feature_sets.items():
                merged_market.setdefault(layer_name, {})
                merged_market[layer_name].update(target_map)
            with open(market_path, 'w') as f:
                json.dump(merged_market, f, indent=2)

        if self.biases:
            bias_path = self.model_dir / f"bias{suffix}.json"
            merged_bias = _load_json_dict(bias_path, {})
            merged_bias.update(self.biases)
            with open(bias_path, 'w') as f:
                json.dump(merged_bias, f, indent=2)

        if self.fp_blend_weights:
            fp_blend_path = self.model_dir / f"fp_blend_weights{suffix}.json"
            merged_fp_weights = _load_json_dict(fp_blend_path, {})
            merged_fp_weights.update(self.fp_blend_weights)
            with open(fp_blend_path, 'w') as f:
                json.dump(merged_fp_weights, f, indent=2)

        probability_oof = self.get_probability_oof_frame()
        if len(probability_oof) > 0:
            probability_oof_path = self.model_dir / f"probability_oof{suffix}.csv"
            if probability_oof_path.exists():
                try:
                    existing_oof = pd.read_csv(probability_oof_path)
                    if 'target' in existing_oof.columns:
                        existing_oof = existing_oof[~existing_oof['target'].isin(probability_oof['target'].unique())]
                        probability_oof = pd.concat([existing_oof, probability_oof], ignore_index=True)
                except Exception:
                    pass
            probability_oof.sort_values(['target', 'game_date']).to_csv(probability_oof_path, index=False)

        print(f"\nModels saved to: {self.model_dir}")
    
    def load_models(self, suffix: str = ''):
        """Load regression models, classifiers, and selected feature lists."""
        for target in self.TARGETS.keys():
            reg_path = self.model_dir / f"xgb_{target}{suffix}.pkl"
            if reg_path.exists():
                with open(reg_path, 'rb') as f:
                    self.models[target] = pickle.load(f)
            clf_path = self.model_dir / f"clf_{target}{suffix}.pkl"
            if clf_path.exists():
                with open(clf_path, 'rb') as f:
                    self.classifiers[target] = pickle.load(f)
            edge_path = self.model_dir / f"edge_xgb_{target}{suffix}.pkl"
            if edge_path.exists():
                with open(edge_path, 'rb') as f:
                    self.edge_models[target] = pickle.load(f)
            meta_path = self.model_dir / f"meta_clf_{target}{suffix}.pkl"
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    self.meta_models[target] = pickle.load(f)
            prob_path = self.model_dir / f"prob_sidecar_{target}{suffix}.pkl"
            if prob_path.exists():
                with open(prob_path, 'rb') as f:
                    self.probability_sidecars[target] = pickle.load(f)

        sel_path = self.model_dir / f"selected_features{suffix}.json"
        if sel_path.exists():
            with open(sel_path) as f:
                self.selected_features = json.load(f)

        bias_path = self.model_dir / f"bias{suffix}.json"
        if bias_path.exists():
            with open(bias_path) as f:
                self.biases = json.load(f)

        market_path = self.model_dir / f"market_feature_sets{suffix}.json"
        if market_path.exists():
            with open(market_path) as f:
                self.market_feature_sets = json.load(f)

        fp_blend_path = self.model_dir / f"fp_blend_weights{suffix}.json"
        if fp_blend_path.exists():
            with open(fp_blend_path) as f:
                self.fp_blend_weights = json.load(f)

        probability_oof_path = self.model_dir / f"probability_oof{suffix}.csv"
        if probability_oof_path.exists():
            self.probability_oof_rows = [pd.read_csv(probability_oof_path)]

        print(
            f"Loaded {len(self.models)} regression + {len(self.classifiers)} classifier + "
            f"{len(self.edge_models)} edge + {len(self.meta_models)} meta + "
            f"{len(self.probability_sidecars)} probability sidecars"
        )
    
    def print_feature_importance(self, target: str, top_n: int = 20):
        """Print top feature importances for a target."""
        if target not in self.feature_importance:
            print(f"No feature importance for {target}")
            return
        
        print(f"\nTop {top_n} features for {target.upper()}:")
        imp = self.feature_importance[target].head(top_n)
        for i, (feat, score) in enumerate(imp.items(), 1):
            print(f"  {i:2d}. {feat}: {score:.4f}")


def _expected_feature_columns(trainer: AdvancedPropModel) -> List[str]:
    cols = []
    for group_cols in trainer.FEATURE_GROUPS.values():
        cols.extend(group_cols)
    return list(dict.fromkeys(cols))


def _print_feature_audit(df: pd.DataFrame, trainer: AdvancedPropModel, label: str):
    expected = _expected_feature_columns(trainer)
    available = trainer.get_feature_columns(df)
    missing = [c for c in expected if c not in df.columns]
    fill = df[available].notna().mean().sort_values(ascending=False) if available else pd.Series(dtype=float)

    print(f"\nFeature audit: {label}")
    print(f"  Expected advanced features: {len(expected)}")
    print(f"  Available advanced features: {len(available)} ({len(available) / len(expected):.1%})")
    print(f"  Missing advanced features: {len(missing)}")
    if not fill.empty:
        print(f"  Average fill on available features: {fill.mean():.1%}")
        print(f"  Median fill on available features: {fill.median():.1%}")
        sparse = fill[fill < 0.5]
        print(f"  Available features below 50% fill: {len(sparse)}")
        if len(sparse) > 0:
            for feat, pct in sparse.head(10).items():
                print(f"    {feat}: {pct:.1%}")


def _json_safe(value):
    """Convert NaN/inf values into JSON-safe None recursively."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _build_edge_analysis(results: Dict) -> Dict:
    """Build compact saved diagnostics for real-line performance by prop and edge bucket."""
    per_target = {}
    bucket_totals = {}
    direction_totals = {}

    for target, res in results.items():
        if not isinstance(res, dict) or 'error' in res:
            continue
        summary = res.get('real_line_summary') or {}
        per_target[target] = summary

        for row in summary.get('bucket_summary', []):
            key = row['label']
            agg = bucket_totals.setdefault(key, {'n': 0, 'hit_sum': 0.0, 'edge_sum': 0.0, 'actual_minus_line_sum': 0.0})
            n = int(row.get('n') or 0)
            agg['n'] += n
            if n and row.get('hit_rate') is not None:
                agg['hit_sum'] += float(row['hit_rate']) * n
            if n and row.get('avg_edge') is not None:
                agg['edge_sum'] += float(row['avg_edge']) * n
            if n and row.get('avg_actual_minus_line') is not None:
                agg['actual_minus_line_sum'] += float(row['avg_actual_minus_line']) * n

        for row in summary.get('direction_summary', []):
            key = row['label']
            agg = direction_totals.setdefault(key, {'n': 0, 'hit_sum': 0.0, 'edge_sum': 0.0, 'actual_minus_line_sum': 0.0})
            n = int(row.get('n') or 0)
            agg['n'] += n
            if n and row.get('hit_rate') is not None:
                agg['hit_sum'] += float(row['hit_rate']) * n
            if n and row.get('avg_edge') is not None:
                agg['edge_sum'] += float(row['avg_edge']) * n
            if n and row.get('avg_actual_minus_line') is not None:
                agg['actual_minus_line_sum'] += float(row['avg_actual_minus_line']) * n

    def _finalize(table: Dict[str, Dict[str, float]]) -> List[Dict]:
        rows = []
        for label, agg in table.items():
            n = agg['n']
            rows.append({
                'label': label,
                'n': n,
                'hit_rate': (agg['hit_sum'] / n) if n else None,
                'avg_edge': (agg['edge_sum'] / n) if n else None,
                'avg_actual_minus_line': (agg['actual_minus_line_sum'] / n) if n else None,
            })
        return rows

    return {
        'per_target': per_target,
        'cross_target_bucket_summary': _finalize(bucket_totals),
        'cross_target_direction_summary': _finalize(direction_totals),
    }


def _prepare_training_frame(df: pd.DataFrame, args, trainer: AdvancedPropModel) -> pd.DataFrame:
    expected = _expected_feature_columns(trainer)
    df = _build_dynamic_context_features(df)
    available = trainer.get_feature_columns(df)
    coverage = len(available) / len(expected) if expected else 1.0
    base_required = {'player', 'game_date', 'pts', 'trb', 'ast', 'mp', 'fga', 'opp'}

    _print_feature_audit(df, trainer, "loaded input")

    if not args.auto_build_features:
        return df

    if coverage >= args.min_feature_coverage:
        print("\nAdvanced feature coverage is already sufficient; skipping feature rebuild.")
        return df

    if not base_required.issubset(df.columns):
        print("\nInput is missing base columns required to build advanced features.")
        return df

    print("\nAdvanced feature coverage is low; running FeaturePipeline on the input data...")
    from feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline(ewma_span=5, split_window=15, min_games=args.min_games)
    df = pipeline.fit_transform(df)
    df = _build_dynamic_context_features(df)

    if 'games_played' in df.columns:
        before = len(df)
        df = df[df['games_played'] >= args.min_games].copy()
        print(f"Filtered to {len(df):,} rows with {args.min_games}+ prior games (from {before:,})")

    if args.prepared_output:
        out_path = Path(args.prepared_output)
        df.to_csv(out_path, index=False)
        print(f"Saved prepared training data to: {out_path}")

    _print_feature_audit(df, trainer, "post-feature-pipeline")
    return df


def main():
    """Train all models with advanced features."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train advanced prop models')
    parser.add_argument('--input', type=str, default=str(DATA_DIR / 'model_data_advanced.csv'),
                        help='Input data file with advanced features')
    parser.add_argument('--model-dir', type=str, default=str(BASE_DIR / 'models'),
                        help='Directory to save models')
    parser.add_argument('--targets', type=str, nargs='+', 
                        default=['pts', 'trb', 'ast', 'pra', 'pr', 'pa', 'stl', 'blk', 'tov'],
                        help='Targets to train')
    parser.add_argument('--auto-build-features', action=argparse.BooleanOptionalAction, default=True,
                        help='If advanced features are missing, run feature_pipeline on the input first')
    parser.add_argument('--prepared-output', type=str, default='',
                        help='Optional path to save the prepared feature table before training')
    parser.add_argument('--min-feature-coverage', type=float, default=0.75,
                        help='Minimum fraction of advanced features required to skip feature rebuilding')
    parser.add_argument('--min-games', type=int, default=10,
                        help='Minimum prior games required after feature generation')
    parser.add_argument('--include-season-context-training', action=argparse.BooleanOptionalAction, default=False,
                        help='Use season-level enriched layer fields in training. Off by default because current historical values are hindsight aggregates.')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading data from: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    trainer = AdvancedPropModel(
        model_dir=args.model_dir,
        include_season_context_training=args.include_season_context_training,
    )
    if not args.include_season_context_training:
        print("Season-context layer features are excluded from training to avoid hindsight leakage.")
    df = _prepare_training_frame(df, args, trainer)

    if 'game_date' not in df.columns:
        raise ValueError("Input data must contain game_date after preparation")

    # Sort by date for proper time series split
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)
    
    print("\nLoading historical prop lines for bet-accuracy metric...")
    lines_lookup = _load_lines_lookup()
    print(f"  {len(lines_lookup):,} (player, date, prop) line entries loaded")

    results = {}
    for target in args.targets:
        if target in trainer.TARGETS:
            try:
                results[target] = trainer.train_model(df, target, lines_lookup=lines_lookup)
            except Exception as e:
                print(f"Error training {target}: {e}")

    # Save models
    trainer.save_models(suffix='_advanced')
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for target, res in results.items():
        if 'error' not in res:
            bet_acc = res.get('avg_bet_accuracy')
            proxy_bet_acc = res.get('avg_proxy_bet_accuracy')
            acc_str = f"{bet_acc:.1%}" if bet_acc is not None and not np.isnan(bet_acc) else "n/a"
            proxy_str = f"{proxy_bet_acc:.1%}" if proxy_bet_acc is not None and not np.isnan(proxy_bet_acc) else "n/a"
            print(f"\n{target.upper()}:")
            print(f"  MAE:          {res['avg_mae']:.2f}")
            print(f"  RMSE:         {res['avg_rmse']:.2f}")
            print(f"  Real BetAcc:  {acc_str}  (scored only on real historical prop lines)")
            print(f"  Proxy BetAcc: {proxy_str}  (scored on L10 fallback lines)")
            print(f"  OOF Bias:     {res.get('oof_bias', 0.0):+.3f}")
            print(f"  Features:     {res['n_features_selected']}/{res['n_features_original']} kept")
    
    # Print feature importance for main targets
    for target in ['pts', 'trb', 'ast']:
        trainer.print_feature_importance(target, top_n=15)
    
    # Save results
    results_path = Path(args.model_dir) / 'training_results.json'
    payload = {
        k: _json_safe({kk: vv for kk, vv in v.items() if kk != 'cv_results'})
        for k, v in results.items()
    }
    if results_path.exists():
        try:
            with open(results_path) as f:
                existing_payload = json.load(f)
            existing_payload.update(payload)
            payload = existing_payload
        except Exception:
            pass
    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    edge_analysis_path = Path(args.model_dir) / 'training_edge_analysis.json'
    with open(edge_analysis_path, 'w') as f:
        json.dump(_json_safe(_build_edge_analysis(payload)), f, indent=2)
    print(f"Edge analysis saved to: {edge_analysis_path}")

    probability_analysis_path = Path(args.model_dir) / 'training_probability_analysis.json'
    probability_payload = {
        target: result.get('probability_summary')
        for target, result in payload.items()
        if isinstance(result, dict) and result.get('probability_summary')
    }
    with open(probability_analysis_path, 'w') as f:
        json.dump(_json_safe(probability_payload), f, indent=2)
    print(f"Probability analysis saved to: {probability_analysis_path}")


if __name__ == "__main__":
    main()
