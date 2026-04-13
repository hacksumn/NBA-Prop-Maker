"""
NBA Player Props Betting System
================================
Single-file pipeline: Data → Features → Train → Predict → Output

Usage:
    python nba_props.py update      # Fetch latest game data
    python nba_props.py train       # Train/retrain models
    python nba_props.py predict     # Generate today's picks
    python nba_props.py backtest    # Backtest model performance
    python nba_props.py rebuild-lines # Rebuild historical_lines.csv from local sources
    python nba_props.py all         # Run full pipeline

Author: Fresh Start Build
"""

import pandas as pd
import numpy as np
import requests
import json
import pickle
import io
import os
import sys
import math
import gc
import re
from itertools import combinations
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from player_pool_guard import sanitize_player_names
from probability_utils import apply_isotonic_calibrators, over_probability_from_quantiles, quantile_tags
from injury_feed import fetch_injury_data

# =============================================================================
# CONFIGURATION
# =============================================================================

_HERE = Path(__file__).parent.resolve()

CONFIG = {
    # Paths — always relative to this script's location, regardless of CWD
    'data_dir': _HERE / 'data',
    'line_archive_dir': (_HERE / 'data' / 'line_archive'),
    'models_dir': _HERE / 'models',
    'output_dir': _HERE / 'output',
    'output_archive_dir': (_HERE / 'output' / 'archive'),
    
    # API Keys
    'odds_api_key': '8269e5e566cd8d8f8c294b85244ffab7',
    
    # Model settings
    'min_games': 10,           # Minimum games for prediction
    'ewma_span': 5,            # EWMA span for recent performance
    'confidence_threshold': 0.6,  # Min confidence for picks
    'edge_threshold': 1.5,     # Min edge (points) for picks — legacy fallback
    'edge_z_threshold': 0.5,   # Min edge in units of player std dev (primary filter)
    'require_meta_for_live_picks': True,   # Only allow picks supported by the abstention layer
    'allow_structural_micro_props': False, # Disable hardcoded STL/BLK fallback picks by default
    'live_pick_min': 5,        # Minimum number of live picks to publish each day
    'live_pick_target': 5,     # Target size of the final live card
    'live_pick_candidate_pool': 25,  # Pull a deeper pool, then sanitize and diversify

    # Targets
    'targets': ['pts', 'trb', 'ast', 'pra', 'pr', 'pa'],
}

# Ensure directories exist
for dir_key in ['data_dir', 'line_archive_dir', 'models_dir', 'output_dir', 'output_archive_dir']:
    CONFIG[dir_key].mkdir(parents=True, exist_ok=True)


_ODDS_PROP_TO_STAT = {
    'player_points': 'pts',
    'player_rebounds': 'trb',
    'player_assists': 'ast',
    'player_points_rebounds_assists': 'pra',
    'player_points_rebounds': 'pr',
    'player_points_assists': 'pa',
    'player_steals': 'stl',
    'player_blocks': 'blk',
    'player_turnovers': 'tov',
}

_STAT_TO_ODDS_PROP = {v: k for k, v in _ODDS_PROP_TO_STAT.items()}


def _season_from_game_date(date_value) -> str:
    """Infer NBA season string from a game date."""
    ts = pd.to_datetime(date_value, errors='coerce')
    if pd.isna(ts):
        return ""
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _ensure_nba_season_columns(df: pd.DataFrame, date_col: str = 'game_date') -> pd.DataFrame:
    """Attach season labels when the frame only has game dates."""
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    if 'season' not in out.columns:
        out['season'] = out[date_col].apply(_season_from_game_date) if date_col in out.columns else ""
    else:
        if date_col in out.columns:
            inferred = out[date_col].apply(_season_from_game_date)
            out['season'] = out['season'].replace('', np.nan).fillna(inferred)
        out['season'] = out['season'].astype(str)
    return out


def _prop_component_cols(target: str, suffix: str) -> List[str]:
    combo_map = {
        'pra': ['pts', 'trb', 'ast'],
        'pr': ['pts', 'trb'],
        'pa': ['pts', 'ast'],
    }
    if target in combo_map:
        return [f'{part}_{suffix}' for part in combo_map[target]]
    return [f'{target}_{suffix}']


def _series_prop_feature(df: pd.DataFrame, target: str, suffix: str, default=np.nan) -> pd.Series:
    cols = [c for c in _prop_component_cols(target, suffix) if c in df.columns]
    if not cols:
        return pd.Series(default, index=df.index, dtype=float)
    stacked = pd.concat([pd.to_numeric(df[c], errors='coerce') for c in cols], axis=1)
    if len(cols) == 1:
        return stacked.iloc[:, 0]
    return stacked.sum(axis=1, min_count=len(cols))


def _row_prop_feature(row: pd.Series, target: str, suffix: str):
    cols = _prop_component_cols(target, suffix)
    values = []
    for col in cols:
        if col not in row.index:
            return None
        value = row.get(col)
        if value is None or pd.isna(value):
            return None
        values.append(float(value))
    return values[0] if len(values) == 1 else sum(values)


def _series_fp_projection(df: pd.DataFrame, target: str) -> pd.Series:
    """Return the best available first-principles projection proxy for a stat."""
    def _series(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce')
        return pd.Series(np.nan, index=df.index, dtype=float)

    if target == 'pts':
        if 'proj_pts_final' in df.columns:
            return pd.to_numeric(df['proj_pts_final'], errors='coerce')
        base = _series('pts_luck_adj')
        if base.isna().all():
            base = _series('clean_pts')
        if base.isna().all():
            base = _series_prop_feature(df, 'pts', 'best_estimate')
        if 'adj_ppp' in df.columns:
            scale = (pd.to_numeric(df['adj_ppp'], errors='coerce') / 1.15).clip(0.7, 1.3)
            return base.fillna(_series_prop_feature(df, 'pts', 'best_estimate')).fillna(_series_prop_feature(df, 'pts', 'recency_anchor')) * scale.fillna(1.0)
        return base.fillna(_series_prop_feature(df, 'pts', 'best_estimate')).fillna(_series_prop_feature(df, 'pts', 'recency_anchor'))
    if target == 'trb':
        series = _series('proj_reb')
        if series.isna().all():
            series = _series('clean_reb')
        return series.fillna(_series_prop_feature(df, 'trb', 'best_estimate')).fillna(_series_prop_feature(df, 'trb', 'recency_anchor'))
    if target == 'ast':
        series = _series('proj_ast')
        if series.isna().all():
            series = _series('clean_ast')
        return series.fillna(_series_prop_feature(df, 'ast', 'best_estimate')).fillna(_series_prop_feature(df, 'ast', 'recency_anchor'))
    return pd.Series(np.nan, index=df.index, dtype=float)


def _load_live_fp_projection_lookup():
    proj_path = _HERE / 'data' / 'player_projections_today.csv'
    if not proj_path.exists():
        return None
    fp = pd.read_csv(proj_path)
    if 'player' not in fp.columns:
        return None
    fp['_norm'] = fp['player'].apply(_normalize_name)
    return fp.drop_duplicates(subset='_norm', keep='first').set_index('_norm')


def _row_fp_projection(row: pd.Series, target: str, fp_lookup=None):
    if fp_lookup is not None:
        key = _normalize_name(row.get('player', ''))
        if key in fp_lookup.index:
            fp_row = fp_lookup.loc[key]
            mapping = {
                'pts': 'proj_pts_final',
                'trb': 'proj_reb',
                'ast': 'proj_ast',
            }
            fp_col = mapping.get(target)
            if fp_col and fp_col in fp_row and pd.notna(fp_row[fp_col]):
                return float(fp_row[fp_col])

    if target == 'pts':
        if pd.notna(row.get('proj_pts_final')):
            return float(row.get('proj_pts_final'))
        base = row.get('pts_luck_adj')
        if pd.isna(base):
            base = row.get('clean_pts')
        if pd.isna(base):
            base = _row_prop_feature(row, 'pts', 'best_estimate')
        scale = row.get('adj_ppp')
        if pd.notna(scale):
            return float(base or 0.0) * float(np.clip(float(scale) / 1.15, 0.7, 1.3))
        return float(base or 0.0)
    if target == 'trb':
        value = row.get('proj_reb', row.get('clean_reb'))
        if pd.isna(value):
            value = _row_prop_feature(row, 'trb', 'best_estimate')
        return float(value or 0.0)
    if target == 'ast':
        value = row.get('proj_ast', row.get('clean_ast'))
        if pd.isna(value):
            value = _row_prop_feature(row, 'ast', 'best_estimate')
        return float(value or 0.0)
    return None


def _safe_float(value, default=np.nan) -> float:
    """Best-effort float coercion that preserves NaN defaults."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _adaptive_fp_blend_weights(
    stat: str,
    pred_row: pd.Series,
    fp_row: pd.Series,
    feature_row: Optional[pd.Series],
    base_weights: Dict[str, float],
) -> Tuple[float, float, float]:
    """Return per-player model/FP weights using only current-row live features."""
    base_model = float(base_weights.get('model_weight', 0.65))
    base_fp = float(base_weights.get('fp_weight', 0.35))
    intercept = float(base_weights.get('intercept', 0.0))

    feature_row = feature_row if feature_row is not None else pred_row
    games_played = max(0.0, _safe_float(feature_row.get('games_played', pred_row.get('games_played', 0.0)), 0.0))
    season_games = max(0.0, _safe_float(feature_row.get('season_games_played', games_played), games_played))
    mp_cv = _safe_float(feature_row.get('mp_cv', pred_row.get('mp_cv')), np.nan)
    mp_trend_pct = _safe_float(feature_row.get('mp_trend_pct', pred_row.get('mp_trend_pct')), np.nan)
    best_estimate = _safe_float(feature_row.get(f'{stat}_best_estimate', pred_row.get(f'{stat}_anchor')), np.nan)
    form_vs_season = _safe_float(feature_row.get(f'{stat}_form_vs_season'), np.nan)
    absence_recent_gp = _safe_float(fp_row.get('absence_recent_gp'), np.nan)

    clean_col = {'pts': 'clean_pts', 'trb': 'clean_reb', 'ast': 'clean_ast'}.get(stat)
    clean_available = 1.0 if clean_col and clean_col in fp_row.index and pd.notna(fp_row.get(clean_col)) else 0.0

    career_depth = float(np.clip(np.log1p(games_played) / np.log1p(250.0), 0.0, 1.0))
    current_season_share = float(np.clip(season_games / max(games_played, 1.0), 0.0, 1.0))

    stability_parts = []
    if pd.notna(mp_cv):
        stability_parts.append(1.0 - float(np.clip(mp_cv / 0.45, 0.0, 1.0)))
    if pd.notna(mp_trend_pct):
        stability_parts.append(1.0 - float(np.clip(abs(mp_trend_pct) / 0.35, 0.0, 1.0)))
    role_stability = float(np.mean(stability_parts)) if stability_parts else 0.5

    form_scale = max(abs(best_estimate), 4.0) if pd.notna(best_estimate) else 8.0
    form_shift = (
        float(np.clip(abs(form_vs_season) / form_scale, 0.0, 1.0))
        if pd.notna(form_vs_season)
        else 0.0
    )

    delta = 0.0
    delta += 0.22 * (0.55 - career_depth)
    delta += 0.18 * (current_season_share - 0.35)
    delta += 0.08 * (role_stability - 0.70)
    delta += 0.08 * (form_shift - 0.15)
    delta += 0.03 if clean_available else -0.05
    if pd.notna(absence_recent_gp) and absence_recent_gp <= 2:
        delta -= 0.10

    fp_weight = float(np.clip(base_fp + delta, 0.15, 0.60))
    model_weight = float(np.clip(base_model - delta, 0.40, 0.85))
    total = model_weight + fp_weight
    if total <= 0:
        return base_model, base_fp, intercept
    return model_weight / total, fp_weight / total, intercept


def _build_pick_probability_payload(
    direction: str,
    dir_prob: Optional[float],
    raw_p_over: Optional[float] = None,
    cal_p_over: Optional[float] = None,
    prob_source: str = 'unknown',
) -> Dict[str, Optional[float]]:
    """Return a pick-safe probability payload with explicit side semantics."""
    dir_prob_val = (
        float(dir_prob)
        if dir_prob is not None and not pd.isna(dir_prob)
        else None
    )
    raw_over_val = (
        float(raw_p_over)
        if raw_p_over is not None and not pd.isna(raw_p_over)
        else None
    )
    cal_over_val = (
        float(cal_p_over)
        if cal_p_over is not None and not pd.isna(cal_p_over)
        else None
    )

    raw_under_val = (1.0 - raw_over_val) if raw_over_val is not None else None
    cal_under_val = (1.0 - cal_over_val) if cal_over_val is not None else None

    return {
        # Legacy alias: keep the column but make it match the selected pick side.
        'ou_prob': round(dir_prob_val * 100, 1) if dir_prob_val is not None else None,
        'dir_prob': round(dir_prob_val * 100, 1) if dir_prob_val is not None else None,
        'p_over_raw': round(raw_over_val * 100, 1) if raw_over_val is not None else None,
        'p_over_cal': round(cal_over_val * 100, 1) if cal_over_val is not None else None,
        'p_under_raw': round(raw_under_val * 100, 1) if raw_under_val is not None else None,
        'p_under_cal': round(cal_under_val * 100, 1) if cal_under_val is not None else None,
        'prob_source': prob_source,
    }


def _load_probability_sidecars():
    sidecars = {}
    for target in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov']:
        path = CONFIG['models_dir'] / f'prob_sidecar_{target}_advanced.pkl'
        if not path.exists():
            continue
        try:
            with open(path, 'rb') as f:
                sidecars[target] = pickle.load(f)
        except Exception:
            continue
    return sidecars


def _load_probability_calibrators():
    path = CONFIG['models_dir'] / 'probability_calibrator_advanced.pkl'
    if not path.exists():
        return {}
    try:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        return payload.get('calibrators', {})
    except Exception:
        return {}


def _attach_probability_outputs(pred_df: pd.DataFrame,
                                feature_df: pd.DataFrame,
                                probability_sidecars: Dict[str, object],
                                probability_calibrators: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    if len(pred_df) == 0 or not probability_sidecars:
        return pred_df

    from train_advanced_models import _predict_probability_sidecar

    working = pred_df.copy().reset_index(drop=True)
    feature_working = feature_df.copy().reset_index(drop=True)
    probability_calibrators = probability_calibrators or {}

    for target, sidecar in probability_sidecars.items():
        feature_names = sidecar.get('feature_names', [])
        X_prob = pd.DataFrame(index=feature_working.index)
        for col in feature_names:
            if col in feature_working.columns:
                X_prob[col] = pd.to_numeric(feature_working[col], errors='coerce')
            else:
                X_prob[col] = 0.0
        line_col = f'{target}_line'
        lines = working[line_col] if line_col in working.columns else pd.Series(np.nan, index=working.index)
        details = _predict_probability_sidecar(sidecar, X_prob.fillna(0.0), lines=lines)

        qtags = [tag for tag in quantile_tags() if tag in details]
        if qtags:
            scale = pd.Series(1.0, index=working.index, dtype=float)
            if 'game_total_scale' in working.columns and target in {'pts', 'ast'}:
                scale = scale * pd.to_numeric(working['game_total_scale'], errors='coerce').fillna(1.0)
            if 'injury_adj_strength' in working.columns:
                boost = pd.to_numeric(working['injury_adj_strength'], errors='coerce').fillna(0.0)
                if target == 'pts':
                    scale = scale * (1.0 + boost)
                elif target == 'ast':
                    scale = scale * (1.0 + boost * 0.90)
            q_matrix = np.column_stack([np.asarray(details[tag], dtype=float) for tag in qtags])
            q_matrix = q_matrix * scale.values.reshape(-1, 1)
            for idx, tag in enumerate(qtags):
                details[tag] = q_matrix[:, idx]
                working[f'{target}_{tag}'] = np.round(q_matrix[:, idx], 3)
            valid_line_mask = pd.to_numeric(lines, errors='coerce').notna().values
            raw = np.full(len(working), np.nan, dtype=float)
            if valid_line_mask.any():
                raw[valid_line_mask] = over_probability_from_quantiles(
                    pd.to_numeric(lines.loc[valid_line_mask], errors='coerce').values,
                    q_matrix[valid_line_mask],
                )
            details['p_over_raw'] = raw

            # σ = (q75 - q25) / 1.35  — unbiased normal-equivalent std from quantile spread.
            # Context-conditioned because the quantile models are trained on minutes, usage,
            # pace, and matchup features.  Exposed as {target}_sigma for downstream use.
            q25_tag = 'q25'
            q75_tag = 'q75'
            if q25_tag in qtags and q75_tag in qtags:
                q25_idx = qtags.index(q25_tag)
                q75_idx = qtags.index(q75_tag)
                sigma_vals = np.clip(
                    (q_matrix[:, q75_idx] - q_matrix[:, q25_idx]) / 1.35,
                    0.1, None,
                )
                working[f'{target}_sigma'] = np.round(sigma_vals, 3)

        raw_probs = np.asarray(details.get('p_over_raw', np.full(len(working), np.nan)), dtype=float)
        cal_probs = apply_isotonic_calibrators(raw_probs, target, probability_calibrators) if probability_calibrators else raw_probs
        working[f'{target}_p_over_raw'] = np.round(raw_probs, 4)
        working[f'{target}_p_over_cal'] = np.round(cal_probs, 4)
        if 'rate' in details:
            working[f'{target}_rate'] = np.round(np.asarray(details['rate'], dtype=float), 4)

    return working

def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def _get_output_archive_dir(day_str: str) -> Path:
    archive_dir = CONFIG['output_archive_dir'] / day_str
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def _get_line_archive_dir(day_str: str) -> Path:
    archive_dir = CONFIG['line_archive_dir'] / day_str
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def _normalize_projection_type(value: object) -> str:
    projection_type = str(value or 'standard').strip().lower()
    return projection_type if projection_type in {'standard', 'goblin', 'demon'} else 'standard'


def _normalize_historical_lines_frame(df: pd.DataFrame,
                                      game_date: Optional[str] = None,
                                      source: str = 'unknown',
                                      snapshot_ts: Optional[str] = None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=[
                'game_date', 'player', 'player_norm', 'prop', 'line', 'num_books',
                'source', 'snapshot_ts', 'projection_type', 'is_promo',
            ]
        )

    out = df.copy()
    if game_date is not None:
        out['game_date'] = game_date
    out['game_date'] = pd.to_datetime(out.get('game_date'), errors='coerce').dt.strftime('%Y-%m-%d')
    if 'player_norm' not in out.columns and 'player' in out.columns:
        out['player_norm'] = out['player'].apply(_normalize_name)
    out['line'] = pd.to_numeric(out.get('line'), errors='coerce')
    out['num_books'] = pd.to_numeric(out.get('num_books', 1), errors='coerce').fillna(1).clip(lower=1)
    if 'source' in out.columns:
        out['source'] = out['source'].fillna(source)
    else:
        out['source'] = source
    snap_value = snapshot_ts or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'snapshot_ts' in out.columns:
        out['snapshot_ts'] = out['snapshot_ts'].replace('', pd.NA).fillna(snap_value)
    else:
        out['snapshot_ts'] = snap_value
    raw_projection_type = out['projection_type'] if 'projection_type' in out.columns else out.get('odds_type')
    if raw_projection_type is not None:
        out['projection_type'] = raw_projection_type.apply(_normalize_projection_type)
    else:
        out['projection_type'] = 'standard'
    if 'is_promo' in out.columns:
        out['is_promo'] = out['is_promo'].fillna(False).astype(bool)
    else:
        out['is_promo'] = False

    keep_cols = [
        'game_date', 'player', 'player_norm', 'prop', 'line', 'num_books',
        'source', 'snapshot_ts', 'projection_type', 'is_promo',
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[keep_cols]
    out = out.dropna(subset=['game_date', 'player', 'player_norm', 'prop', 'line'])
    out['player'] = out['player'].astype(str)
    out['player_norm'] = out['player_norm'].astype(str)
    out['prop'] = out['prop'].astype(str)
    out['source'] = out['source'].astype(str)
    out['snapshot_ts'] = out['snapshot_ts'].astype(str)
    out['projection_type'] = out['projection_type'].astype(str)
    out['is_promo'] = out['is_promo'].astype(bool)
    return out.drop_duplicates(subset=['game_date', 'player_norm', 'prop', 'line', 'source', 'snapshot_ts']).reset_index(drop=True)


def _load_historical_lines_frame(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or (CONFIG['data_dir'] / 'historical_lines.csv')
    if not Path(path).exists():
        return _normalize_historical_lines_frame(pd.DataFrame())
    try:
        raw = pd.read_csv(path, low_memory=False)
    except Exception:
        return _normalize_historical_lines_frame(pd.DataFrame())
    return _normalize_historical_lines_frame(raw)


def _merge_historical_lines(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    existing = _normalize_historical_lines_frame(existing)
    incoming = _normalize_historical_lines_frame(incoming)
    if len(existing) == 0:
        merged = incoming.copy()
    elif len(incoming) == 0:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, incoming], ignore_index=True)

    if len(merged) == 0:
        return merged

    merged['snapshot_sort'] = pd.to_datetime(merged['snapshot_ts'], errors='coerce')
    merged = merged.sort_values(
        ['game_date', 'player_norm', 'prop', 'num_books', 'snapshot_sort', 'line'],
        ascending=[True, True, True, True, True, True],
    )
    merged = merged.drop_duplicates(subset=['game_date', 'player_norm', 'prop'], keep='last')
    merged = merged.sort_values(['game_date', 'prop', 'player']).reset_index(drop=True)
    return merged.drop(columns=['snapshot_sort'], errors='ignore')


def _write_historical_lines(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = path or (CONFIG['data_dir'] / 'historical_lines.csv')
    clean = _normalize_historical_lines_frame(df)
    clean, summary = sanitize_player_names(
        clean,
        player_col='player',
        date_col='game_date',
        player_norm_col='player_norm',
        data_dir=CONFIG['data_dir'],
        drop_unknown=True,
        require_roster=False,
    )
    _atomic_csv(clean, Path(path))
    if summary['canonicalized_rows'] or summary['dropped_rows']:
        print(
            f"  Historical line cleanup: {summary['canonicalized_rows']} canonicalized, "
            f"{summary['dropped_rows']} dropped"
        )
    return Path(path)


def _archive_line_snapshot(df: pd.DataFrame, game_date: str, stem: str = 'lines') -> Tuple[Path, Path]:
    archive_dir = _get_line_archive_dir(game_date)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    daily_path = archive_dir / f'{stem}_{game_date}.csv'
    run_path = archive_dir / f'{stem}_{stamp}.csv'
    _atomic_csv(df, daily_path)
    _atomic_csv(df, run_path)
    return daily_path, run_path


def _extract_prediction_archive_lines(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()

    line_cols = [c for c in df.columns if c.endswith('_line')]
    if 'player' not in df.columns or not line_cols:
        return pd.DataFrame()

    game_date = None
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(path))
    if match:
        game_date = match.group(1)
    rows = []
    for stat, prop in _STAT_TO_ODDS_PROP.items():
        col = f'{stat}_line'
        if col not in df.columns:
            continue
        subset = df[['player', col]].copy()
        subset[col] = pd.to_numeric(subset[col], errors='coerce')
        subset = subset.dropna(subset=[col])
        if len(subset) == 0:
            continue
        subset = subset.rename(columns={col: 'line'})
        subset['prop'] = prop
        subset['game_date'] = game_date
        subset['player_norm'] = subset['player'].apply(_normalize_name)
        subset['num_books'] = 1
        subset['source'] = 'prediction_archive'
        subset['snapshot_ts'] = path.stem
        rows.append(subset[['game_date', 'player', 'player_norm', 'prop', 'line', 'num_books', 'source', 'snapshot_ts']])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def rebuild_historical_lines_from_local_sources(write: bool = True) -> pd.DataFrame:
    """Rebuild/repair historical_lines.csv from local archives and snapshots."""
    merged = _load_historical_lines_frame()

    morning_paths = sorted(CONFIG['data_dir'].glob('lines_morning_*.csv'))
    for path in morning_paths:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', path.name)
        if not match:
            continue
        game_date = match.group(1)
        try:
            morning = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        morning = _normalize_historical_lines_frame(
            morning.assign(num_books=1, source='morning_snapshot', snapshot_ts=path.stem),
            game_date=game_date,
        )
        merged = _merge_historical_lines(merged, morning)

    archive_paths = sorted(CONFIG['line_archive_dir'].rglob('lines_*.csv'))
    for path in archive_paths:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', str(path))
        if not match:
            continue
        game_date = match.group(1)
        try:
            snap = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        snap = _normalize_historical_lines_frame(
            snap.assign(source='line_archive', snapshot_ts=path.stem),
            game_date=game_date,
        )
        merged = _merge_historical_lines(merged, snap)

    pred_paths = sorted((CONFIG['output_archive_dir']).rglob('predictions_*.csv'))
    for path in pred_paths:
        extracted = _extract_prediction_archive_lines(path)
        if len(extracted) == 0:
            continue
        merged = _merge_historical_lines(merged, extracted)

    latest_pred = CONFIG['output_dir'] / 'predictions_latest.csv'
    if latest_pred.exists():
        merged = _merge_historical_lines(merged, _extract_prediction_archive_lines(latest_pred))

    if write:
        _write_historical_lines(merged)
    return merged


def _archive_csv_snapshot(df: pd.DataFrame, stem: str, day_str: str, run_stamp: str) -> Tuple[Path, Path]:
    archive_dir = _get_output_archive_dir(day_str)
    daily_path = archive_dir / f'{stem}_{day_str}.csv'
    run_path = archive_dir / f'{stem}_{run_stamp}.csv'
    _atomic_csv(df, daily_path)
    _atomic_csv(df, run_path)
    return daily_path, run_path


def _archive_text_snapshot(text: str, stem: str, day_str: str, run_stamp: str, suffix: str) -> Tuple[Path, Path]:
    archive_dir = _get_output_archive_dir(day_str)
    daily_path = archive_dir / f'{stem}_{day_str}{suffix}'
    run_path = archive_dir / f'{stem}_{run_stamp}{suffix}'
    _atomic_text(text, daily_path)
    _atomic_text(text, run_path)
    return daily_path, run_path


def _migrate_legacy_output_artifacts() -> None:
    """
    Move old flat timestamped output files into the dated archive tree.
    Keeps latest/history files in place.
    """
    pattern = re.compile(r'^(picks|predictions|dashboard)_(\d{8})_(\d{6})\.(csv|html)$')
    for path in CONFIG['output_dir'].iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        stem, ymd, hms, ext = match.groups()
        day_str = f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}'
        archive_dir = _get_output_archive_dir(day_str)
        dest = archive_dir / f'{stem}_{day_str}_{hms}.{ext}'
        if dest.exists():
            path.unlink()
            continue
        path.replace(dest)


# =============================================================================
# DATA FETCHING (Free NBA API)
# =============================================================================

def fetch_nba_data(seasons: List[str] = None, local_file: str = None) -> pd.DataFrame:
    """
    Fetch NBA player game logs.
    
    First tries to load from local file, then falls back to nba_api.
    
    Args:
        seasons: List of seasons like ['2024-25', '2023-24']
        local_file: Path to local CSV file (optional)
    """
    
    # Try local file first
    if local_file and Path(local_file).exists():
        print(f"  Loading from local file: {local_file}")
        df = pd.read_csv(local_file)
        
        # Handle different date formats
        if 'GAME_DATE' in df.columns:
            # Format like "Apr 01, 2025"
            try:
                df['game_date'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')
            except:
                df['game_date'] = pd.to_datetime(df['GAME_DATE'])
        elif 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Standardize column names
        col_map = {
            'PLAYER_NAME': 'player',
            'TEAM_ABBREVIATION': 'team',
            'MATCHUP': 'matchup',
            'WL': 'result',
            'MIN': 'mp',
            'FGM': 'fg',
            'FG_PCT': 'fg_pct',
            'FG3M': '3p',
            'FG3A': '3pa',
            'FG3_PCT': '3p_pct',
            'FTM': 'ft',
            'FT_PCT': 'ft_pct',
            'OREB': 'orb',
            'DREB': 'drb',
            'REB': 'trb',
            'TOV': 'tov',
            'PF': 'pf',
            'PTS': 'pts',
            'AST': 'ast',
            'STL': 'stl',
            'BLK': 'blk',
            'FTA': 'fta',
            'FGA': 'fga',
            'PLUS_MINUS': 'plus_minus',
            # Rolling averages from uploaded data
            'PTS_ROLL_20': 'pts_l20_imported',
            'REB_ROLL_20': 'trb_l20_imported',
            'AST_ROLL_20': 'ast_l20_imported',
            'STL_ROLL_20': 'stl_l20_imported',
            'BLK_ROLL_20': 'blk_l20_imported',
            'TOV_ROLL_20': 'tov_l20_imported',
            'MIN_ROLL_20': 'mp_l20_imported',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        
        # Ensure required columns exist
        if 'is_home' not in df.columns:
            if 'matchup' in df.columns:
                df['is_home'] = df['matchup'].apply(lambda x: 0 if '@' in str(x) else 1)
            elif 'MATCHUP' in df.columns:
                df['is_home'] = df['MATCHUP'].apply(lambda x: 0 if '@' in str(x) else 1)
            else:
                df['is_home'] = 1
        
        if 'opp' not in df.columns:
            if 'matchup' in df.columns:
                df['opp'] = df['matchup'].apply(lambda x: str(x).split()[-1] if pd.notna(x) else None)
            elif 'MATCHUP' in df.columns:
                df['opp'] = df['MATCHUP'].apply(lambda x: str(x).split()[-1] if pd.notna(x) else None)
        
        if 'team' not in df.columns:
            if 'matchup' in df.columns:
                df['team'] = df['matchup'].apply(lambda x: str(x).split()[0] if pd.notna(x) else None)
            elif 'MATCHUP' in df.columns:
                df['team'] = df['MATCHUP'].apply(lambda x: str(x).split()[0] if pd.notna(x) else None)
        
        df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
        print(f"  Loaded {len(df):,} player-games, {df['player'].nunique()} players")
        print(f"  Date range: {df['game_date'].min().strftime('%Y-%m-%d')} to {df['game_date'].max().strftime('%Y-%m-%d')}")
        return df
    
    # Try nba_api
    try:
        from nba_api.stats.endpoints import playergamelogs
    except ImportError:
        print("Installing nba_api...")
        os.system('pip install nba_api --break-system-packages -q')
        from nba_api.stats.endpoints import playergamelogs
    
    if seasons is None:
        seasons = ['2025-26', '2024-25', '2023-24']
    
    all_data = []
    
    for season in seasons:
        print(f"  Fetching {season} from NBA API...")
        try:
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            df = logs.get_data_frames()[0]
            df['season'] = season
            all_data.append(df)
            
            import time
            time.sleep(1)
            
        except Exception as e:
            print(f"    Error: {e}")
    
    if not all_data:
        raise ValueError("No data fetched! Provide a local_file path.")
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Standardize column names
    df = df.rename(columns={
        'PLAYER_NAME': 'player',
        'TEAM_ABBREVIATION': 'team',
        'MATCHUP': 'matchup',
        'GAME_DATE': 'game_date',
        'WL': 'result',
        'MIN': 'mp',
        'FGM': 'fg',
        'FGA': 'fga',
        'FG_PCT': 'fg_pct',
        'FG3M': '3p',
        'FG3A': '3pa',
        'FG3_PCT': '3p_pct',
        'FTM': 'ft',
        'FTA': 'fta',
        'FT_PCT': 'ft_pct',
        'OREB': 'orb',
        'DREB': 'drb',
        'REB': 'trb',
        'AST': 'ast',
        'STL': 'stl',
        'BLK': 'blk',
        'TOV': 'tov',
        'PF': 'pf',
        'PTS': 'pts',
        'PLUS_MINUS': 'plus_minus',
    })
    
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['is_home'] = df['matchup'].apply(lambda x: 0 if '@' in str(x) else 1)
    df['opp'] = df['matchup'].apply(lambda x: x.split(' ')[-1] if pd.notna(x) else None)
    
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)
    
    print(f"  Total: {len(df):,} player-games")
    
    return df


def fetch_todays_games() -> pd.DataFrame:
    """Fetch today's NBA schedule."""
    try:
        from nba_api.stats.endpoints import scoreboardv2
    except ImportError:
        os.system('pip install nba_api -q')
        from nba_api.stats.endpoints import scoreboardv2
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
        games = scoreboard.get_data_frames()[0]
        
        if len(games) == 0:
            print(f"  No games scheduled for {today}")
            return pd.DataFrame()
        
        # Extract matchups
        matchups = []
        for _, game in games.iterrows():
            matchups.append({
                'game_id': game.get('GAME_ID'),
                'home_team': game.get('HOME_TEAM_ABBREVIATION', game.get('HOME_TEAM_ID')),
                'away_team': game.get('VISITOR_TEAM_ABBREVIATION', game.get('VISITOR_TEAM_ID')),
                'game_date': today
            })
        
        return pd.DataFrame(matchups)
        
    except Exception as e:
        print(f"  Error fetching schedule: {e}")
        return pd.DataFrame()


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation for fuzzy name matching."""
    import unicodedata, re
    name = unicodedata.normalize('NFD', str(name))
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    return re.sub(r"[^a-z ]", '', name.lower()).strip()


def _coerce_slate_date(value: object) -> Optional[str]:
    ts = pd.to_datetime(value, errors='coerce')
    if pd.isna(ts):
        return None
    return ts.strftime('%Y-%m-%d')


def _resolve_active_slate_date(lines_df: pd.DataFrame,
                               as_of: Optional[datetime] = None) -> Optional[str]:
    if lines_df is None or len(lines_df) == 0 or 'game_date' not in lines_df.columns:
        return None

    dates = pd.to_datetime(lines_df['game_date'], errors='coerce').dropna()
    if dates.empty:
        return None

    unique_dates = sorted(dates.dt.normalize().unique())
    as_of_ts = pd.Timestamp(as_of or datetime.now()).normalize()
    upcoming = [d for d in unique_dates if d >= as_of_ts]
    chosen = upcoming[0] if upcoming else unique_dates[-1]
    return pd.Timestamp(chosen).strftime('%Y-%m-%d')


def _select_active_slate_lines(lines_df: pd.DataFrame,
                               as_of: Optional[datetime] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    if lines_df is None or len(lines_df) == 0:
        return pd.DataFrame(), None

    working = lines_df.copy()
    if 'game_date' not in working.columns:
        fallback_date = pd.Timestamp(as_of or datetime.now()).strftime('%Y-%m-%d')
        return working.reset_index(drop=True), fallback_date

    working['game_date'] = pd.to_datetime(working['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    working = working.dropna(subset=['game_date']).copy()
    slate_date = _resolve_active_slate_date(working, as_of=as_of)
    if slate_date is None:
        return working.iloc[0:0].copy(), None
    active = working[working['game_date'] == slate_date].copy()
    return active.reset_index(drop=True), slate_date


def _infer_line_source(lines_df: pd.DataFrame, default: str = 'cached_unknown') -> str:
    if lines_df is None or len(lines_df) == 0 or 'source' not in lines_df.columns:
        return default
    src_vals = lines_df['source'].dropna().astype(str).tolist()
    if any('prizepicks' in s.lower() for s in src_vals):
        return 'prizepicks'
    if any('odds_api' in s.lower() for s in src_vals):
        return 'odds_api'
    return default


def fetch_prizepicks_lines() -> pd.DataFrame:
    """Fetch today's NBA player props from PrizePicks (free, no API key required)."""
    # PrizePicks stat type → our internal prop key
    stat_map = {
        'Points':           'player_points',
        'Rebounds':         'player_rebounds',
        'Assists':          'player_assists',
        'Pts+Rebs+Asts':    'player_points_rebounds_assists',
        'Pts+Rebs':         'player_points_rebounds',
        'Pts+Asts':         'player_points_assists',
        'Steals':           'player_steals',
        'Blocked Shots':    'player_blocks',
        'Turnovers':        'player_turnovers',
    }
    url = 'https://api.prizepicks.com/projections'
    params = {'league_id': 7, 'per_page': 500, 'single_stat': 'true', 'game_mode': 'pickem'}
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"  PrizePicks error: {r.status_code}")
            return pd.DataFrame()
        data = r.json()

        # Build player id → name lookup from included
        player_lookup = {}
        game_lookup = {}
        for item in data.get('included', []):
            if item.get('type') in ('new_player', 'player'):
                pid = item['id']
                attrs = item.get('attributes', {})
                player_lookup[pid] = attrs.get('display_name', attrs.get('name', ''))
            elif item.get('type') == 'game':
                game_lookup[item['id']] = item.get('attributes', {})

        rows = []
        for proj in data.get('data', []):
            attrs = proj.get('attributes', {})
            stat_type = attrs.get('stat_type', '')
            prop_key = stat_map.get(stat_type)
            if not prop_key:
                continue
            line = attrs.get('line_score')
            if line is None:
                continue
            # Resolve player name
            rels = proj.get('relationships', {})
            player_rel = rels.get('new_player') or rels.get('player') or {}
            pid = (player_rel.get('data') or {}).get('id', '')
            player_name = player_lookup.get(pid, attrs.get('description', ''))
            if not player_name:
                continue
            game_rel = rels.get('game') or {}
            game_id = (game_rel.get('data') or {}).get('id', '')
            game_attrs = game_lookup.get(game_id, {})
            slate_date = (
                _coerce_slate_date(attrs.get('start_time')) or
                _coerce_slate_date(attrs.get('board_time')) or
                _coerce_slate_date(attrs.get('end_time')) or
                _coerce_slate_date(game_attrs.get('start_time')) or
                _coerce_slate_date(game_attrs.get('end_time'))
            )
            if slate_date is None:
                continue
            projection_type = _normalize_projection_type(
                attrs.get('odds_type') or attrs.get('projection_type') or 'standard'
            )
            rows.append({
                'game_date': slate_date,
                'player': player_name,
                'prop': prop_key,
                'line': float(line),
                'odds': -110,       # PrizePicks is pick'em; use standard juice as placeholder
                'num_books': 1,
                'projection_type': projection_type,
                'is_promo': bool(attrs.get('is_promo', False)),
            })

        if not rows:
            print("  PrizePicks: no props found")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # De-dup per slate; PrizePicks can expose tomorrow's board before the
        # calendar day rolls over, so the real slate date must survive grouping.
        df = df.groupby(['game_date', 'player', 'prop'], as_index=False).agg(
            line=('line', 'median'),
            odds=('odds', 'first'),
            num_books=('num_books', 'sum'),
            projection_type=('projection_type', 'first'),
            is_promo=('is_promo', 'max'),
        )
        df, summary = sanitize_player_names(
            df,
            player_col='player',
            date_col='game_date',
            player_norm_col=None,
            data_dir=CONFIG['data_dir'],
            drop_unknown=True,
            require_roster=True,
        )
        print(
            f"  Fetched {len(df)} validated player props from PrizePicks "
            f"({summary['canonicalized_rows']} canonicalized, {summary['dropped_rows']} dropped)"
        )
        return df

    except Exception as e:
        print(f"  PrizePicks error: {e}")
        return pd.DataFrame()


def fetch_vegas_lines() -> tuple:
    """Fetch current NBA player prop lines. Tries Odds API first, falls back to PrizePicks.

    Returns:
        (DataFrame, source_label) where source_label is 'odds_api' or 'prizepicks'.
    """
    api_key = CONFIG['odds_api_key']

    # Check if Odds API has remaining credits
    try:
        check = requests.get(
            'https://api.the-odds-api.com/v4/sports/basketball_nba/events',
            params={'apiKey': api_key}, timeout=10
        )
        remaining = int(check.headers.get('x-requests-remaining', 0))
        if check.status_code == 200 and remaining > 50:
            events = check.json()
            if events:
                all_lines = []
                markets = [
                    'player_points', 'player_rebounds', 'player_assists',
                    'player_points_rebounds_assists', 'player_points_rebounds',
                    'player_points_assists', 'player_steals', 'player_blocks',
                    'player_turnovers'
                ]
                for event in events[:10]:
                    event_game_date = _coerce_slate_date(event.get('commence_time')) or datetime.now().strftime('%Y-%m-%d')
                    props_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event['id']}/odds"
                    pr = requests.get(props_url, params={
                        'apiKey': api_key, 'regions': 'us',
                        'markets': ','.join(markets), 'oddsFormat': 'american'
                    })
                    if pr.status_code != 200:
                        continue
                    for bookmaker in pr.json().get('bookmakers', []):
                        for market in bookmaker.get('markets', []):
                            for outcome in market.get('outcomes', []):
                                all_lines.append({
                                    'game_date': event_game_date,
                                    'player': outcome.get('description', ''),
                                    'prop': market['key'],
                                    'line': outcome.get('point', 0),
                                    'direction': outcome.get('name', ''),
                                    'odds': outcome.get('price', 0),
                                    'book': bookmaker['key'],
                                })
                if all_lines:
                    df = pd.DataFrame(all_lines)
                    consolidated = df.groupby(['game_date', 'player', 'prop'], as_index=False).agg(
                        line=('line', 'median'), odds=('odds', 'first'), num_books=('book', 'count')
                    )
                    consolidated, summary = sanitize_player_names(
                        consolidated,
                        player_col='player',
                        date_col='game_date',
                        player_norm_col=None,
                        data_dir=CONFIG['data_dir'],
                        drop_unknown=True,
                        require_roster=True,
                    )
                    print(
                        f"  Fetched {len(consolidated)} validated player props "
                        f"(Odds API, {remaining} credits left; "
                        f"{summary['canonicalized_rows']} canonicalized, {summary['dropped_rows']} dropped)"
                    )
                    return consolidated, 'odds_api'
        else:
            print(f"  Odds API: {remaining} credits remaining — switching to PrizePicks")
    except Exception as e:
        print(f"  Odds API unavailable ({e}) — switching to PrizePicks")

    pp_df = fetch_prizepicks_lines()
    return pp_df, 'prizepicks'


def save_lines_snapshot(vegas_df: pd.DataFrame, game_date: Optional[str] = None, source: str = 'live_fetch'):
    """Append fetched lines to data/historical_lines.csv keyed by their real slate date."""
    if vegas_df is None or len(vegas_df) == 0:
        return
    path = CONFIG['data_dir'] / 'historical_lines.csv'
    if 'game_date' in vegas_df.columns:
        snap_input = vegas_df[
            [c for c in ['game_date', 'player', 'prop', 'line', 'num_books', 'projection_type', 'is_promo'] if c in vegas_df.columns]
        ].copy()
        snap_input['game_date'] = pd.to_datetime(snap_input['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        snap_input = snap_input.dropna(subset=['game_date']).copy()
    else:
        resolved_game_date = game_date or datetime.now().strftime('%Y-%m-%d')
        snap_input = vegas_df[
            [c for c in ['player', 'prop', 'line', 'num_books', 'projection_type', 'is_promo'] if c in vegas_df.columns]
        ].copy()
        snap_input['game_date'] = resolved_game_date
    snap = _normalize_historical_lines_frame(snap_input, source=source)
    if len(snap) == 0:
        return
    combined = _merge_historical_lines(_load_historical_lines_frame(path), snap)
    _write_historical_lines(combined, path)
    for slate_date in sorted(snap['game_date'].dropna().unique()):
        slate_snap = snap[snap['game_date'] == slate_date].copy()
        _archive_line_snapshot(slate_snap, slate_date, stem='lines')
        day_total = int((combined['game_date'] == slate_date).sum())
        print(f"  Saved {len(slate_snap)} lines -> historical_lines.csv ({slate_date}, {day_total} rows kept for day)")
        morning_path = CONFIG['data_dir'] / f'lines_morning_{slate_date}.csv'
        if not morning_path.exists():
            slate_snap[['player', 'prop', 'line']].assign(
                player_norm=slate_snap['player'].apply(_normalize_name)
            ).to_csv(morning_path, index=False)
            print(f"  Morning snapshot saved -> {morning_path.name}")


def backfill_historical_lines(days: int = 30):
    """Pull historical prop lines from Odds API and save to historical_lines.csv."""
    api_key = CONFIG['odds_api_key']
    base = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba"
    markets = [
        'player_points', 'player_rebounds', 'player_assists',
        'player_points_rebounds_assists', 'player_points_rebounds',
        'player_points_assists', 'player_steals', 'player_blocks', 'player_turnovers'
    ]

    # Probe: check historical access before burning API quota
    probe_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%dT12:00:00Z')
    probe = requests.get(f"{base}/events", params={'apiKey': api_key, 'date': probe_date})
    if probe.status_code in (401, 402, 403):
        print(f"  Historical endpoint not accessible with current key (HTTP {probe.status_code}).")
        print("  Lines will still be collected going forward each time you run predict.")
        repaired = rebuild_historical_lines_from_local_sources(write=True)
        print(f"  Local repair complete — {len(repaired):,} rows available from local sources.")
        print("  Upgrade at https://the-odds-api.com to enable backfill.")
        return False

    print(f"  Historical API accessible — backfilling {days} days...")
    all_rows = []

    for d in range(1, days + 1):
        dt = datetime.now() - timedelta(days=d)
        date_str = dt.strftime('%Y-%m-%dT12:00:00Z')
        game_date = dt.strftime('%Y-%m-%d')

        r = requests.get(f"{base}/events", params={'apiKey': api_key, 'date': date_str})
        if r.status_code != 200:
            print(f"  {game_date}: HTTP {r.status_code} — skipping")
            continue

        for event in r.json()['data'][:10]:
            pr = requests.get(
                f"{base}/events/{event['id']}/odds",
                params={'apiKey': api_key, 'date': date_str, 'regions': 'us',
                        'markets': ','.join(markets), 'oddsFormat': 'american'}
            )
            if pr.status_code != 200:
                continue
            for bkm in pr.json()['data'].get('bookmakers', []):
                for mkt in bkm.get('markets', []):
                    for outcome in mkt.get('outcomes', []):
                        all_rows.append({
                            'game_date': game_date,
                            'player': outcome.get('description', ''),
                            'player_norm': _normalize_name(outcome.get('description', '')),
                            'prop': mkt['key'],
                            'line': outcome.get('point', 0),
                            'num_books': 1,
                        })
        print(f"  {game_date}: done")

    if not all_rows:
        print("  No historical data returned.")
        return False

    # Median line per (date, player, prop) across bookmakers
    df = (pd.DataFrame(all_rows)
            .groupby(['game_date', 'player', 'player_norm', 'prop'])
            .agg(line=('line', 'median'), num_books=('num_books', 'sum'))
            .reset_index())

    path = CONFIG['data_dir'] / 'historical_lines.csv'
    df = _normalize_historical_lines_frame(df.assign(source='odds_api_backfill'))
    merged = _merge_historical_lines(_load_historical_lines_frame(path), df)
    _write_historical_lines(merged, path)
    print(f"  Backfill complete — {len(merged)} total rows in historical_lines.csv")
    return True


# =============================================================================
# INJURY / LINEUP DATA
# =============================================================================

def fetch_game_totals(game_date: str = None) -> pd.DataFrame:
    """Fetch game totals, spreads and moneylines from ESPN Core API.
    Returns DataFrame with columns: home_team, away_team, total, spread, home_ml, away_ml
    Team abbreviations match NBA team codes (LAL, BOS, etc.).
    """
    if game_date is None:
        game_date = datetime.now().strftime('%Y%m%d')
    else:
        game_date = game_date.replace('-', '')
    try:
        url = f'http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events?dates={game_date}&limit=20'
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        items = r.json().get('items', [])
        rows = []
        for item in items:
            ref = item.get('$ref', '')
            if not ref:
                continue
            try:
                ev = requests.get(ref, timeout=8).json()
                comps = ev.get('competitions', [])
                if not comps:
                    continue
                comp = comps[0]
                competitors = comp.get('competitors', [])
                home_abbr = away_abbr = ''
                for c in competitors:
                    team = c.get('team', {})
                    abbr = team.get('abbreviation', '')
                    if c.get('homeAway') == 'home':
                        home_abbr = abbr
                    else:
                        away_abbr = abbr
                # Fetch odds
                event_id = ev.get('id', '')
                odds_url = f'http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event_id}/competitions/{event_id}/odds'
                odds_r = requests.get(odds_url, timeout=8)
                if odds_r.status_code != 200:
                    continue
                odds_items = odds_r.json().get('items', [])
                if not odds_items:
                    continue
                o = odds_items[0]  # use first provider (DraftKings)
                rows.append({
                    'home_team': home_abbr,
                    'away_team': away_abbr,
                    'total': o.get('overUnder'),
                    'spread': o.get('spread'),
                    'home_ml': o.get('homeTeamOdds', {}).get('moneyLine'),
                    'away_ml': o.get('awayTeamOdds', {}).get('moneyLine'),
                })
                time.sleep(0.3)
            except Exception:
                continue
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  Game totals unavailable: {e}")
        return pd.DataFrame()


def fetch_player_positions() -> pd.DataFrame:
    """Fetch player positions from NBA API. Caches to data/player_positions.csv.
    Returns DataFrame with columns: player_norm, position_group (G/F/C).
    """
    cache_path = CONFIG['data_dir'] / 'player_positions.csv'
    # Refresh cache if >7 days old or missing
    if cache_path.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit='s')).days
        if age_days < 7:
            return pd.read_csv(cache_path)
    try:
        from nba_api.stats.endpoints import playerindex
        import time as _time
        df = playerindex.PlayerIndex(season='2025-26').get_data_frames()[0]
        _time.sleep(0.6)
        df['full_name'] = df['PLAYER_FIRST_NAME'] + ' ' + df['PLAYER_LAST_NAME']
        df['player_norm'] = df['full_name'].apply(_normalize_name)
        # Collapse multi-position tags to primary group
        def _pos_group(pos):
            if not pos or pd.isna(pos):
                return 'F'
            p = str(pos).upper()
            if 'G' in p and 'F' not in p and 'C' not in p:
                return 'G'
            if 'C' in p and 'G' not in p:
                return 'C'
            return 'F'
        df['position_group'] = df['POSITION'].apply(_pos_group)
        out = df[['player_norm', 'position_group']].drop_duplicates('player_norm')
        out.to_csv(cache_path, index=False)
        print(f"  Player positions cached: {len(out)} players")
        return out
    except Exception as e:
        print(f"  Position fetch failed: {e}")
        return pd.DataFrame(columns=['player_norm', 'position_group'])


def apply_injury_adjustments(pred_df, injury_data, df):
    """Boost offensive predictions only for likely beneficiaries of high-impact absences."""
    pred_df = pred_df.copy()
    pred_df['injury_adj'] = False
    pred_df['teammates_out'] = ''
    pred_df['injury_adj_strength'] = 0.0
    if not injury_data or pred_df is None or len(pred_df) == 0:
        return pred_df

    latest = df.sort_values('game_date').groupby('player').last().reset_index().copy()
    latest['player_norm'] = latest['player'].apply(_normalize_name)
    for col in ['mp_l10', 'pts_l10', 'ast_l10', 'trb_l10', 'usg_pct', 'likely_starter', 'is_starter_proxy']:
        if col in latest.columns:
            latest[col] = pd.to_numeric(latest[col], errors='coerce').fillna(0)
        else:
            latest[col] = 0.0
    latest['starter_proxy'] = latest[['likely_starter', 'is_starter_proxy']].max(axis=1)
    latest['starter_proxy'] = latest['starter_proxy'].where(latest['starter_proxy'] > 0, (latest['mp_l10'] >= 28).astype(float))

    def _clip01(val: float, denom: float) -> float:
        if denom <= 0:
            return 0.0
        return float(max(0.0, min(1.0, val / denom)))

    def _absence_impact_score(player_row: pd.Series) -> float:
        mp = float(player_row.get('mp_l10', 0.0) or 0.0)
        pts = float(player_row.get('pts_l10', 0.0) or 0.0)
        ast = float(player_row.get('ast_l10', 0.0) or 0.0)
        usg = float(player_row.get('usg_pct', 0.0) or 0.0)
        starter = float(player_row.get('starter_proxy', 0.0) or 0.0)
        return (
            0.38 * _clip01(mp, 36.0) +
            0.32 * _clip01(pts, 25.0) +
            0.18 * _clip01(ast, 8.0) +
            0.08 * _clip01(usg, 0.30) +
            0.04 * min(1.0, starter)
        )

    def _beneficiary_score(player_row: pd.Series) -> float:
        mp = float(player_row.get('mp_l10', 0.0) or 0.0)
        pts = float(player_row.get('pts_l10', 0.0) or 0.0)
        ast = float(player_row.get('ast_l10', 0.0) or 0.0)
        starter = float(player_row.get('starter_proxy', 0.0) or 0.0)
        return (
            0.42 * _clip01(mp, 36.0) +
            0.30 * _clip01(pts, 20.0) +
            0.20 * _clip01(ast, 6.0) +
            0.08 * min(1.0, starter)
        )

    team_adjustments: Dict[str, Dict[str, Dict[str, object]]] = {}
    for team, team_data in injury_data.items():
        team_players = latest[latest['team'] == team].copy()
        if len(team_players) == 0:
            continue

        absent_norms = set(team_data.get('out', [])) | set(team_data.get('doubtful', []))
        if not absent_norms:
            continue

        high_impact_absences = []
        for out_norm in absent_norms:
            match = team_players[team_players['player_norm'] == out_norm]
            if len(match) == 0:
                continue
            player_row = match.iloc[0]
            impact_score = _absence_impact_score(player_row)
            high_impact = (
                impact_score >= 0.72 or
                (
                    float(player_row.get('mp_l10', 0.0) or 0.0) >= 28.0 and
                    (
                        float(player_row.get('pts_l10', 0.0) or 0.0) >= 18.0 or
                        float(player_row.get('ast_l10', 0.0) or 0.0) >= 5.0 or
                        float(player_row.get('usg_pct', 0.0) or 0.0) >= 0.24
                    )
                )
            )
            if not high_impact:
                continue
            high_impact_absences.append({
                'player': player_row['player'],
                'player_norm': out_norm,
                'impact_score': impact_score,
            })

        if not high_impact_absences:
            continue

        severity = sum(item['impact_score'] for item in high_impact_absences)
        base_boost = min(0.12, 0.025 + 0.035 * severity)
        blocked_norms = absent_norms | set(team_data.get('questionable', [])) | set(team_data.get('day_to_day', []))
        active = team_players[~team_players['player_norm'].isin(blocked_norms)].copy()
        active = active[active['mp_l10'] >= 18.0].copy()
        if len(active) == 0:
            continue

        active['benefit_score'] = active.apply(_beneficiary_score, axis=1)
        active = active[active['benefit_score'] >= 0.48].copy()
        if len(active) == 0:
            continue
        active = active.sort_values(['benefit_score', 'mp_l10', 'pts_l10', 'ast_l10'], ascending=False).head(5)

        multipliers = [1.00, 0.78, 0.58, 0.42, 0.28]
        out_names = [item['player'] for item in high_impact_absences[:3]]
        players_map: Dict[str, Dict[str, object]] = {}
        for rank, (_, player_row) in enumerate(active.iterrows()):
            player_boost = round(base_boost * multipliers[rank], 4)
            if player_boost < 0.015:
                continue
            players_map[player_row['player_norm']] = {
                'boost': player_boost,
                'out_names': out_names,
            }
        if players_map:
            team_adjustments[team] = players_map

    for idx, row in pred_df.iterrows():
        team = row.get('team', '')
        player_norm = _normalize_name(str(row.get('player', '')))
        player_adjustment = team_adjustments.get(team, {}).get(player_norm)
        if not player_adjustment:
            continue

        boost = float(player_adjustment['boost'])
        if 'pts_pred' in pred_df.columns and pd.notna(pred_df.at[idx, 'pts_pred']):
            pred_df.at[idx, 'pts_pred'] = pred_df.at[idx, 'pts_pred'] * (1.0 + boost)
        if 'ast_pred' in pred_df.columns and pd.notna(pred_df.at[idx, 'ast_pred']):
            pred_df.at[idx, 'ast_pred'] = pred_df.at[idx, 'ast_pred'] * (1.0 + boost * 0.90)

        pts_val = pd.to_numeric(pd.Series([pred_df.at[idx, 'pts_pred'] if 'pts_pred' in pred_df.columns else np.nan]), errors='coerce').iloc[0]
        trb_val = pd.to_numeric(pd.Series([pred_df.at[idx, 'trb_pred'] if 'trb_pred' in pred_df.columns else np.nan]), errors='coerce').iloc[0]
        ast_val = pd.to_numeric(pd.Series([pred_df.at[idx, 'ast_pred'] if 'ast_pred' in pred_df.columns else np.nan]), errors='coerce').iloc[0]
        if pd.notna(pts_val) and pd.notna(ast_val):
            pred_df.at[idx, 'pa_pred'] = round(float(pts_val + ast_val), 1)
        if pd.notna(pts_val) and pd.notna(trb_val):
            pred_df.at[idx, 'pr_pred'] = round(float(pts_val + trb_val), 1)
        if pd.notna(pts_val) and pd.notna(trb_val) and pd.notna(ast_val):
            pred_df.at[idx, 'pra_pred'] = round(float(pts_val + trb_val + ast_val), 1)

        pred_df.at[idx, 'injury_adj'] = True
        pred_df.at[idx, 'injury_adj_strength'] = round(boost, 4)
        pred_df.at[idx, 'teammates_out'] = ', '.join(player_adjustment['out_names'])

    n_adj = int(pred_df['injury_adj'].sum())
    if n_adj > 0:
        print(f"  Injury adjustments applied to {n_adj} players")
    return pred_df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(df: pd.DataFrame, stats_cutoff=None) -> pd.DataFrame:
    """
    Create all features for the model.
    
    Features:
    - Rolling averages (L5, L10, L20)
    - EWMA weighted averages
    - Consistency/variance metrics
    - Home/away splits
    - Opponent defense context
    - Situational adjustments
    - Composite predictions
    """
    print("Creating features...")

    df = df.copy()
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    df = _ensure_nba_season_columns(df)
    df = df.sort_values(['player', 'game_date']).reset_index(drop=True)

    # stats_cutoff: when set, global league-average constants are computed from
    # training-period rows only (game_date < cutoff) to prevent leakage.
    
    # Basic stats
    stats = ['pts', 'trb', 'ast', 'mp', 'fga', 'fta', '3pa', 'stl', 'blk', 'tov']
    
    # Group by player
    grouped = df.groupby('player')
    season_grouped = df.groupby(['player', 'season'], dropna=False)
    
    # Games played counter
    df['games_played'] = grouped.cumcount()
    df['season_games_played'] = season_grouped.cumcount()
    df['season_sample_reliability'] = (
        df['season_games_played'] / (df['season_games_played'] + 10.0)
    ).fillna(0.0).clip(0.0, 1.0)
    
    # -------------------------------------------------------------------------
    # 1. Rolling Averages (L5, L10, L20)
    # -------------------------------------------------------------------------
    print("  Rolling averages...")
    for stat in stats:
        if stat not in df.columns:
            continue
        for window in [5, 10, 20]:
            df[f'{stat}_l{window}'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        df[f'{stat}_season_avg'] = season_grouped[stat].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
    
    # Standard deviation
    for stat in ['pts', 'trb', 'ast', 'mp']:
        if stat in df.columns:
            df[f'{stat}_std'] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).std()
            )

    # -------------------------------------------------------------------------
    # 1b. Blowout-Discounted Weighted Rolling Averages
    # -------------------------------------------------------------------------
    # Games with large score margins distort rolling averages — stars rest early,
    # backups inflate stats. Down-weight blowouts so the model focuses on
    # competitive games where roles reflect normal usage.
    #   weight = 1.0 for |margin| <= 15pts, linearly decays to 0.6 at |margin| = 40
    if 'plus_minus' in df.columns:
        _pm = pd.to_numeric(df['plus_minus'], errors='coerce').fillna(0).abs()
        _excess = (_pm - 15).clip(lower=0)
        df['_game_weight'] = (1.0 - 0.4 * (_excess / 25).clip(0, 1))

        def _weighted_roll(group, stat, weight_col, window):
            """Weighted rolling mean using blowout discounts."""
            out = pd.Series(np.nan, index=group.index)
            for i in range(len(group)):
                start = max(0, i - window)
                v = group[stat].iloc[start:i].values
                w = group[weight_col].iloc[start:i].values
                if len(v) >= 2:
                    out.iloc[i] = np.average(v, weights=w)
            return out

        for stat in ['pts', 'trb', 'ast', 'mp']:
            if stat not in df.columns:
                continue
            df[f'{stat}_wt_l10'] = (
                df.sort_values(['player', 'game_date'])
                  .groupby('player', group_keys=False)
                  .apply(lambda g: _weighted_roll(g, stat, '_game_weight', 10))
                  .reset_index(level=0, drop=True)
            )

        # _game_weight dropped in cleanup block at end of create_features()
    else:
        for stat in ['pts', 'trb', 'ast', 'mp']:
            if f'{stat}_l10' in df.columns:
                df[f'{stat}_wt_l10'] = df[f'{stat}_l10']

    # -------------------------------------------------------------------------
    # 2. EWMA Features
    # -------------------------------------------------------------------------
    print("  EWMA features...")
    span = CONFIG['ewma_span']
    for stat in stats:
        if stat not in df.columns:
            continue
        df[f'{stat}_ewma'] = grouped[stat].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
        )
    
    # Consistency (coefficient of variation)
    for stat in ['pts', 'trb', 'ast']:
        if f'{stat}_ewma' in df.columns and f'{stat}_std' in df.columns:
            df[f'{stat}_consistency'] = df[f'{stat}_std'] / df[f'{stat}_ewma'].replace(0, np.nan)
            df[f'{stat}_consistency'] = df[f'{stat}_consistency'].fillna(1.0).clip(0, 3)

    # Recency-weighted anchor: combine multi-window form with a season-to-date baseline.
    season_reliability = df.get('season_sample_reliability', pd.Series(0.0, index=df.index)).fillna(0.0)
    for stat in stats:
        ewma_col = f'{stat}_ewma'
        season_col = f'{stat}_season_avg'
        l5_col = f'{stat}_l5'
        l10_col = f'{stat}_l10'
        l20_col = f'{stat}_l20'
        if not all(col in df.columns for col in [ewma_col, l5_col, l10_col, l20_col]):
            continue

        recent_stack = (
            df[ewma_col] * 0.40 +
            df[l5_col] * 0.25 +
            df[l10_col] * 0.20 +
            df[l20_col] * 0.15
        )
        season_anchor = df[season_col] if season_col in df.columns else df[l20_col]
        season_anchor = season_anchor.fillna(df[l20_col]).fillna(recent_stack)
        season_share = 0.12 + 0.23 * season_reliability
        df[f'{stat}_recency_anchor'] = (
            recent_stack * (1.0 - season_share) +
            season_anchor * season_share
        )
        if season_col in df.columns:
            df[f'{stat}_form_vs_season'] = df[f'{stat}_recency_anchor'] - df[season_col]
    
    # -------------------------------------------------------------------------
    # 3. Trends
    # -------------------------------------------------------------------------
    print("  Trend features...")
    for stat in ['pts', 'trb', 'ast', 'mp']:
        if f'{stat}_l5' in df.columns and f'{stat}_l20' in df.columns:
            df[f'{stat}_trend'] = df[f'{stat}_l5'] / df[f'{stat}_l20'].replace(0, np.nan)
            df[f'{stat}_trend'] = df[f'{stat}_trend'].fillna(1.0)
    
    # -------------------------------------------------------------------------
    # 4. Home/Away Splits
    # -------------------------------------------------------------------------
    print("  Home/away splits...")
    for stat in ['pts', 'trb', 'ast']:
        if stat not in df.columns:
            continue
        # Home average
        df[f'{stat}_home_temp'] = np.where(df['is_home'] == 1, df[stat], np.nan)
        df[f'{stat}_home_avg'] = grouped[f'{stat}_home_temp'].transform(
            lambda x: x.shift(1).rolling(15, min_periods=3).mean()
        )
        
        # Away average
        df[f'{stat}_away_temp'] = np.where(df['is_home'] == 0, df[stat], np.nan)
        df[f'{stat}_away_avg'] = grouped[f'{stat}_away_temp'].transform(
            lambda x: x.shift(1).rolling(15, min_periods=3).mean()
        )
        
        # Differential
        df[f'{stat}_home_diff'] = df[f'{stat}_home_avg'] - df[f'{stat}_away_avg']
        
        # Situational average (use home if home game, away if away)
        df[f'{stat}_sit_avg'] = np.where(
            df['is_home'] == 1,
            df[f'{stat}_home_avg'],
            df[f'{stat}_away_avg']
        )
        df[f'{stat}_sit_avg'] = df[f'{stat}_sit_avg'].fillna(df[f'{stat}_l10'])
    
    # -------------------------------------------------------------------------
    # 5. Opponent Defense
    # -------------------------------------------------------------------------
    print("  Opponent defense...")
    
    # Aggregate team defense (what they allow)
    team_def = df.groupby(['opp', 'game_date']).agg({
        'pts': 'mean',
        'trb': 'mean', 
        'ast': 'mean',
        'fga': 'mean'
    }).reset_index()
    
    team_def.columns = ['opp', 'game_date', 'def_pts', 'def_trb', 'def_ast', 'def_fga']
    team_def = team_def.sort_values(['opp', 'game_date'])
    
    # Rolling defense averages
    team_grouped = team_def.groupby('opp')
    for stat in ['def_pts', 'def_trb', 'def_ast', 'def_fga']:
        team_def[f'{stat}_l10'] = team_grouped[stat].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
    
    # Merge to player data
    df = df.merge(
        team_def[['opp', 'game_date', 'def_pts_l10', 'def_trb_l10', 'def_ast_l10', 'def_fga_l10']],
        on=['opp', 'game_date'],
        how='left'
    )
    
    # ── Positional defense: how many pts/trb/ast does each team allow per position group ──
    pos_df = fetch_player_positions()
    if len(pos_df) > 0:
        df['player_norm_tmp'] = df['player'].apply(_normalize_name)
        df = df.merge(pos_df, left_on='player_norm_tmp', right_on='player_norm', how='left')
        df['position_group'] = df['position_group'].fillna('F')
        df.drop(columns=['player_norm_tmp'], inplace=True, errors='ignore')
        pos_def = df.groupby(['opp', 'game_date', 'position_group']).agg(
            pos_def_pts=('pts', 'mean'),
            pos_def_trb=('trb', 'mean'),
            pos_def_ast=('ast', 'mean'),
        ).reset_index().sort_values(['opp', 'position_group', 'game_date'])
        for stat in ['pos_def_pts', 'pos_def_trb', 'pos_def_ast']:
            pos_def[f'{stat}_l10'] = pos_def.groupby(['opp', 'position_group'])[stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
        df = df.merge(
            pos_def[['opp', 'game_date', 'position_group', 'pos_def_pts_l10', 'pos_def_trb_l10', 'pos_def_ast_l10']],
            on=['opp', 'game_date', 'position_group'], how='left'
        )
    else:
        for col in ['pos_def_pts_l10', 'pos_def_trb_l10', 'pos_def_ast_l10', 'position_group']:
            df[col] = np.nan

    # Defense rank (1 = allows most, 30 = allows least)
    for stat in ['def_pts', 'def_trb', 'def_ast']:
        col = f'{stat}_l10'
        if col in df.columns:
            df[f'{stat}_rank'] = df.groupby('game_date')[col].rank(ascending=False, pct=True)
            df[f'{stat}_rank'] = df[f'{stat}_rank'].fillna(0.5)
    
    # Defense tier (1=easy, 5=hard)
    df['def_tier'] = pd.cut(
        df['def_pts_rank'].fillna(0.5),
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[1, 2, 3, 4, 5]
    ).astype(float).fillna(3)
    
    # -------------------------------------------------------------------------
    # 6. Schedule/Fatigue
    # -------------------------------------------------------------------------
    print("  Schedule features...")
    
    # Days rest
    df['prev_date'] = grouped['game_date'].shift(1)
    df['days_rest'] = (df['game_date'] - df['prev_date']).dt.days
    df['days_rest'] = df['days_rest'].fillna(3).clip(0, 7)
    
    # Back-to-back
    df['is_b2b'] = (df['days_rest'] <= 1).astype(int)

    # Opponent rest — how many days rest does the opposing team have?
    # Build one row per (team, game_date), compute that team's rest, then join as opponent.
    team_dates = (
        df[['team', 'game_date']].drop_duplicates()
        .sort_values(['team', 'game_date'])
    )
    team_dates['team_days_rest'] = (
        team_dates.groupby('team')['game_date']
        .diff().dt.days.fillna(3).clip(0, 7)
    )
    df = df.merge(
        team_dates.rename(columns={'team': 'opp', 'team_days_rest': 'opp_days_rest'}),
        on=['opp', 'game_date'], how='left'
    )
    df['opp_days_rest'] = df['opp_days_rest'].fillna(3).clip(0, 7)
    df['opp_is_b2b'] = (df['opp_days_rest'] <= 1).astype(int)
    # Positive = we are more rested; negative = opponent is more rested
    df['rest_advantage'] = df['days_rest'] - df['opp_days_rest']

    # High minutes previous game
    df['prev_mp'] = grouped['mp'].shift(1)
    df['high_mp_prev'] = (df['prev_mp'] > 36).astype(int)
    
    # -------------------------------------------------------------------------
    # 6b. Minutes Model Features
    # -------------------------------------------------------------------------
    print("  Minutes model features...")
    
    # Additional rolling windows for minutes
    df['mp_l3'] = grouped['mp'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).mean()
    )
    
    # EWMA for minutes
    df['mp_ewma3'] = grouped['mp'].transform(
        lambda x: x.shift(1).ewm(span=3, min_periods=2).mean()
    )
    df['mp_ewma5'] = grouped['mp'].transform(
        lambda x: x.shift(1).ewm(span=5, min_periods=3).mean()
    )
    
    # Minutes std and CV
    df['mp_std_l10'] = grouped['mp'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).std()
    )
    df['mp_cv'] = df['mp_std_l10'] / df['mp_l10'].clip(lower=1)
    
    # Role indicators based on L10 minutes
    df['is_star'] = (df['mp_l10'] >= 32).astype(int)
    df['is_starter'] = (df['mp_l10'] >= 24).astype(int)
    df['is_rotation'] = ((df['mp_l10'] >= 15) & (df['mp_l10'] < 24)).astype(int)
    df['is_bench'] = (df['mp_l10'] < 15).astype(int)
    
    # Schedule intensity
    df['is_heavy_schedule'] = df['is_b2b']  # Simplified proxy
    
    # Previous game context
    df['prev_mp_high'] = (df['prev_mp'] > 38).astype(int)
    df['prev_mp_low'] = (df['prev_mp'] < 20).astype(int)
    
    # Previous game result/margin
    if 'plus_minus' in df.columns:
        df['prev_margin'] = grouped['plus_minus'].shift(1).abs()
        df['prev_blowout'] = (df['prev_margin'] > 15).astype(int)
    else:
        df['prev_blowout'] = 0
    
    if 'result' in df.columns:
        df['prev_win'] = (grouped['result'].shift(1) == 'W').astype(int)
    else:
        df['prev_win'] = 0
    
    # Minutes trend features
    df['mp_trend_l3_l10'] = df['mp_l3'] - df['mp_l10']
    df['mp_trend_pct'] = (df['mp_l3'] - df['mp_l10']) / df['mp_l10'].clip(lower=1)
    
    # Role expanding/shrinking
    df['role_expanding'] = (df['mp_trend_l3_l10'] > 2).astype(int)
    df['role_shrinking'] = (df['mp_trend_l3_l10'] < -2).astype(int)
    
    # Home/away minutes split (simplified - use existing trend)
    # Use L10 as baseline since proper expanding calculation is complex
    df['mp_location_diff'] = df['mp_home_diff'].fillna(0) if 'mp_home_diff' in df.columns else 0
    
    # Opponent pace factor (simplified)
    if 'fga' in df.columns:
        _fga_ref = df[df['game_date'] < stats_cutoff] if stats_cutoff is not None else df
        league_avg_fga = _fga_ref['fga'].mean() if len(_fga_ref) > 0 else df['fga'].mean()
        df['opp_pace_factor'] = df['def_fga_l10'].fillna(league_avg_fga) / league_avg_fga
    else:
        df['opp_pace_factor'] = 1.0
    
    # Fatigue score
    df['fatigue_score'] = (
        df['is_b2b'] * 2 +
        df['prev_mp_high'] * 1 +
        df['is_heavy_schedule'] * 1
    )
    
    # Expected minutes adjustment
    df['mp_expected_adj'] = (
        df['mp_trend_l3_l10'].fillna(0) +
        df.get('mp_location_diff', 0) -
        df['fatigue_score'] * 1.5
    )
    
    # -------------------------------------------------------------------------
    # 7. Composite Features
    # -------------------------------------------------------------------------
    print("  Composite features...")
    
    for stat in ['pts', 'trb', 'ast']:
        # Best estimate (weighted combo)
        # Incorporates: EWMA (recency), situational avg, blowout-weighted L10,
        # luck-adjusted projection (pts only), quality-adjusted p100,
        # and L20 season baseline (regression anchor — prevents chasing hot/cold streaks).
        ewma = df.get(f'{stat}_ewma', df.get(f'{stat}_l5', 0))
        sit = df.get(f'{stat}_sit_avg', ewma)
        wt10 = df.get(f'{stat}_wt_l10', df.get(f'{stat}_l10', ewma))
        qa = df.get(f'{stat}_qa_p100_l10', df.get(f'{stat}_adj_l10', wt10))
        # Season baseline — L20 prevents over-extrapolating recent streaks
        season = df.get(f'{stat}_season_avg', df.get(f'{stat}_l20', df.get(f'{stat}_l10', ewma)))
        anchor = df.get(f'{stat}_recency_anchor', ewma)
        df[f'{stat}_best_estimate'] = (
            anchor * 0.45 +
            sit * 0.15 +
            wt10 * 0.15 +
            qa * 0.10 +
            season * 0.15
        )
        df[f'{stat}_synthetic_line'] = season

        if stat == 'pts' and 'pts_regression_proj' in df.columns:
            luck_proj = df['pts_regression_proj']
            df[f'{stat}_pred'] = (anchor * 0.25 + sit * 0.15 + wt10 * 0.10
                                  + qa * 0.10 + luck_proj * 0.15 + season * 0.25)
        else:
            df[f'{stat}_pred'] = (anchor * 0.30 + sit * 0.15 + wt10 * 0.15
                                  + qa * 0.15 + season * 0.25)

        df[f'{stat}_edge_vs_avg'] = df[f'{stat}_best_estimate'] - df[f'{stat}_synthetic_line']
        df[f'{stat}_pct_diff'] = (
            df[f'{stat}_edge_vs_avg'] / df[f'{stat}_synthetic_line'].replace(0, np.nan)
        ).fillna(0.0)
        df[f'{stat}_vegas_slow'] = (
            df[f'{stat}_edge_vs_avg'].abs() > df[f'{stat}_synthetic_line'].abs().fillna(0) * 0.1
        ).astype(int)

        # Confidence (inverse of consistency, scaled by games)
        cons = df.get(f'{stat}_consistency', 0.5)
        games_factor = (df['games_played'] / 20).clip(0, 1)
        sample_factor = 0.6 + 0.4 * df.get('season_sample_reliability', pd.Series(0.0, index=df.index)).fillna(0.0)
        df[f'{stat}_conf'] = ((1 / cons.clip(0.3, 2)) * games_factor * sample_factor).clip(0, 1)
    
    # Combo stats predictions
    df['pra_pred'] = df['pts_pred'] + df['trb_pred'] + df['ast_pred']
    df['pr_pred'] = df['pts_pred'] + df['trb_pred']
    df['pa_pred'] = df['pts_pred'] + df['ast_pred']
    
    # -------------------------------------------------------------------------
    # 8. Quality-Adjusted Rolling Averages (Strength of Schedule)
    # -------------------------------------------------------------------------
    # Normalise each game's stats to a league-average defensive context so that
    # scoring 30 vs a weak defence doesn't inflate the rolling average as much
    # as scoring 30 vs a great defence.
    # pts_adj = pts * (league_avg_allowed / opp_allowed_l10)
    print("  Quality-adjusted rolling averages...")
    for stat in ['pts', 'trb', 'ast']:
        def_col = f'def_{stat}_l10'
        if def_col in df.columns:
            # Use training-period rows only to avoid contamination from future games
            _def_ref = df[df['game_date'] < stats_cutoff] if stats_cutoff is not None else df
            league_avg = _def_ref[def_col].median() if len(_def_ref) > 0 else df[def_col].median()
            adj_col = f'_{stat}_qa'   # temp column — added to df before groupby
            df[adj_col] = df[stat] * (league_avg / df[def_col].replace(0, np.nan)).fillna(1).clip(0.5, 2.0)
            qa_grp = df.groupby('player')   # fresh groupby so it sees adj_col
            df[f'{stat}_adj_l5']  = qa_grp[adj_col].transform(lambda x: x.shift(1).rolling(5,  min_periods=2).mean())
            df[f'{stat}_adj_l10'] = qa_grp[adj_col].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
            # Edge: raw average vs quality-adj (positive = player benefited from easy schedule)
            df[f'{stat}_sched_edge'] = df[f'{stat}_l10'] - df[f'{stat}_adj_l10']
            df = df.drop(columns=[adj_col])

    # -------------------------------------------------------------------------
    # 9. Shooting Efficiency Trends
    # -------------------------------------------------------------------------
    print("  Shooting efficiency features...")
    for pct_stat in ['fg_pct', '3p_pct', 'ft_pct']:
        if pct_stat in df.columns:
            df[f'{pct_stat}_l5'] = grouped[pct_stat].transform(
                lambda x: x.shift(1).rolling(5, min_periods=2).mean()
            )
            df[f'{pct_stat}_l10'] = grouped[pct_stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            df[f'{pct_stat}_l20'] = grouped[pct_stat].transform(
                lambda x: x.shift(1).rolling(20, min_periods=5).mean()
            )
            # Luck: how much is recent % deviating from the longer baseline?
            # Positive = running hot (expect regression); negative = running cold (expect bounce)
            df[f'{pct_stat}_luck'] = df[f'{pct_stat}_l5'] - df[f'{pct_stat}_l20']

    # -------------------------------------------------------------------------
    # 9b. Luck Model — Free Throw & 3P Variance / Regression to Mean
    # -------------------------------------------------------------------------
    # Identifies how many points a player is generating from unsustainable
    # shooting variance vs. true talent. Models:
    #   3P luck pts  = (recent 3P% - career 3P%) * recent 3PA * 3pts
    #   FT luck pts  = (recent FT% - career FT%) * recent FTA * 1pt
    # Positive = running hot above career rate (expect regression down).
    # Negative = running cold below career rate (expect bounce back up).
    _luck_cols = []
    pg_luck = df.sort_values(['player', 'game_date']).groupby('player')

    if all(c in df.columns for c in ['3p_pct', '3pa', '3p_pct_l5', '3pa_l5']):
        # Career-to-date 3P% as true-talent anchor (expanding mean, min 20 games)
        df['3p_pct_career'] = pg_luck['3p_pct'].transform(
            lambda x: x.shift(1).expanding(min_periods=20).mean()
        ).fillna(df.get('3p_pct_l20', df['3p_pct_l5']))  # fall back to l20 if <20 games

        df['3p_pct_luck_pts'] = (
            (df['3p_pct_l5'].fillna(0) - df['3p_pct_career'].fillna(0))
            * df['3pa_l5'].fillna(0) * 3.0
        )
        _luck_cols += ['3p_pct_career', '3p_pct_luck_pts']

    if all(c in df.columns for c in ['ft_pct', 'fta', 'ft_pct_l5', 'fta_l5']):
        df['ft_pct_career'] = pg_luck['ft_pct'].transform(
            lambda x: x.shift(1).expanding(min_periods=20).mean()
        ).fillna(df.get('ft_pct_l20', df['ft_pct_l5']))

        df['ft_luck_pts'] = (
            (df['ft_pct_l5'].fillna(0) - df['ft_pct_career'].fillna(0))
            * df['fta_l5'].fillna(0) * 1.0
        )
        _luck_cols += ['ft_pct_career', 'ft_luck_pts']

    if '3p_pct_luck_pts' in df.columns and 'ft_luck_pts' in df.columns:
        df['pts_luck_total'] = df['3p_pct_luck_pts'] + df['ft_luck_pts']
        # Luck-adjusted points: what the player "should" have scored at career rates
        if 'pts_l5' in df.columns:
            df['pts_luck_adj_l5'] = df['pts_l5'] - df['pts_luck_total']
        # Regression projection: blend EWMA with luck-adjusted estimate
        if 'pts_ewma' in df.columns and 'pts_luck_adj_l5' in df.columns:
            df['pts_regression_proj'] = (
                df['pts_ewma'] * 0.6 + df['pts_luck_adj_l5'] * 0.4
            )
        _luck_cols += ['pts_luck_total', 'pts_luck_adj_l5', 'pts_regression_proj']

    if 'fta' in df.columns and 'fta_l5' not in df.columns:
        # Ensure fta rolling exists for luck model
        df['fta_l5'] = pg_luck['fta'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )

    if 'fga' in df.columns and 'fta' in df.columns and 'pts' in df.columns:
        df['ts_raw'] = df['pts'] / (2 * (df['fga'] + 0.44 * df['fta'] + 1e-6)).clip(lower=0.01)
        ts_grouped = df.groupby('player')
        df['ts_pct_l10'] = ts_grouped['ts_raw'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        df['ts_pct_trend'] = (
            ts_grouped['ts_raw'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
            / df['ts_pct_l10'].replace(0, np.nan)
        )
        df = df.drop(columns=['ts_raw'])

    # -------------------------------------------------------------------------
    # 9. Player vs Specific Opponent History (Bayesian shrinkage)
    # -------------------------------------------------------------------------
    # With only 1-2 games vs a specific opponent the sample mean is pure noise.
    # Shrink toward the player's L10 prior; as games accumulate the estimate
    # shifts toward the actual opponent history.
    # vs_opp_shrunk = (n*sample_sum + k*prior) / (n + k),  k = prior weight
    print("  Opponent matchup history...")
    _PRIOR_K = 5   # equivalent to 5 games of prior data
    for stat in ['pts', 'trb', 'ast']:
        df_s = df.sort_values(['player', 'opp', 'game_date'])
        grp = df_s.groupby(['player', 'opp'])[stat]
        n_opp   = grp.transform(lambda x: x.shift(1).expanding(min_periods=1).count()).fillna(0)
        sum_opp = grp.transform(lambda x: x.shift(1).expanding(min_periods=1).sum()).fillna(0)
        prior = df[f'{stat}_l10'].fillna(df[stat].median())
        df[f'{stat}_vs_opp'] = (sum_opp + _PRIOR_K * prior) / (n_opp + _PRIOR_K)
        df[f'{stat}_vs_opp_edge'] = df[f'{stat}_vs_opp'] - prior

    # -------------------------------------------------------------------------
    # 10. Last Game Spike/Crash (z-score)
    # -------------------------------------------------------------------------
    print("  Spike/crash features...")
    for stat in ['pts', 'trb', 'ast']:
        if f'{stat}_std' in df.columns and f'{stat}_l10' in df.columns:
            prev = grouped[stat].shift(1)
            std_safe = df[f'{stat}_std'].clip(lower=0.5)
            df[f'{stat}_last_zscore'] = ((prev - df[f'{stat}_l10']) / std_safe).fillna(0).clip(-4, 4)
            df[f'{stat}_spike'] = (df[f'{stat}_last_zscore'] > 1.5).astype(int)
            df[f'{stat}_crash'] = (df[f'{stat}_last_zscore'] < -1.5).astype(int)

    # -------------------------------------------------------------------------
    # 11. Season Context
    # -------------------------------------------------------------------------
    df['season_early'] = (df['games_played'] < 15).astype(int)
    df['season_late'] = (df['games_played'] > 60).astype(int)

    # -------------------------------------------------------------------------
    # 12. Game Pace — estimated possessions per game
    # -------------------------------------------------------------------------
    # Possessions ≈ FGA + 0.44*FTA + TOV - 0.5*ORB (Oliver formula)
    # We sum player-level stats to team totals, compute rolling team pace,
    # and combine team + opponent pace for an expected game-pace feature.
    if all(c in df.columns for c in ['fga', 'fta', 'tov']):
        print("  Pace & usage features...")
        orb_vals = df['orb'] if 'orb' in df.columns else 0
        df['_player_poss'] = df['fga'] + 0.44 * df['fta'] + df['tov'] - 0.5 * orb_vals

        # Team total possessions per game (sum across all players on team that game)
        df['_team_poss_game'] = df.groupby(['team', 'game_date'])['_player_poss'].transform('sum')

        # One row per (team, game_date) for rolling pace
        team_poss_df = (
            df.groupby(['team', 'game_date'])['_player_poss'].sum()
            .reset_index().rename(columns={'_player_poss': '_team_poss'})
            .sort_values(['team', 'game_date'])
        )
        team_poss_df['team_pace_l10'] = team_poss_df.groupby('team')['_team_poss'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )

        # Merge own team pace and opponent team pace onto player rows
        df = df.merge(team_poss_df[['team', 'game_date', 'team_pace_l10']],
                      on=['team', 'game_date'], how='left')
        df = df.merge(
            team_poss_df.rename(columns={'team': 'opp', 'team_pace_l10': 'opp_pace_l10'}
                                )[['opp', 'game_date', 'opp_pace_l10']],
            on=['opp', 'game_date'], how='left'
        )

        # League-average pace — computed from training period only to avoid leakage
        if stats_cutoff is not None:
            _pace_ref = df[df['game_date'] < stats_cutoff]
            if len(_pace_ref) > 0 and all(c in _pace_ref.columns for c in ['fga', 'fta', 'tov']):
                _ref_orb = _pace_ref['orb'] if 'orb' in _pace_ref.columns else 0
                _ref_poss = _pace_ref['fga'] + 0.44 * _pace_ref['fta'] + _pace_ref['tov'] - 0.5 * _ref_orb
                _ref_team_poss = _ref_poss.groupby([_pace_ref['team'], _pace_ref['game_date']]).sum()
                league_avg_pace = _ref_team_poss.median()
            else:
                league_avg_pace = team_poss_df['_team_poss'].median()
        else:
            league_avg_pace = team_poss_df['_team_poss'].median()
        df['game_pace_est'] = (
            df['team_pace_l10'].fillna(league_avg_pace) +
            df['opp_pace_l10'].fillna(league_avg_pace)
        ) / 2
        df['pace_factor'] = (df['game_pace_est'] / league_avg_pace).clip(0.8, 1.2)

        # Pace-adjusted stat projections (explicit interaction: volume × pace)
        for stat in ['pts', 'trb', 'ast']:
            l10 = f'{stat}_l10'
            if l10 in df.columns:
                df[f'{stat}_pace_adj'] = df[l10] * df['pace_factor']

        # -------------------------------------------------------------------------
        # 12b. Per-100 Possession Normalization
        # -------------------------------------------------------------------------
        # Normalise each game's raw stat to what a player would produce in 100
        # team possessions. Removes pace noise — a slow-tempo game and a fast-tempo
        # game with the same raw box score are not equivalent.
        _team_poss_safe = df['_team_poss_game'].clip(lower=1)
        for stat in ['pts', 'trb', 'ast']:
            if stat in df.columns:
                df[f'_{stat}_p100_raw'] = df[stat] / _team_poss_safe * 100

        pg_p100 = df.sort_values(['player', 'game_date']).groupby('player')
        for stat in ['pts', 'trb', 'ast']:
            raw_col = f'_{stat}_p100_raw'
            if raw_col in df.columns:
                df[f'{stat}_p100_l5']  = pg_p100[raw_col].transform(
                    lambda x: x.shift(1).rolling(5,  min_periods=2).mean()
                )
                df[f'{stat}_p100_l10'] = pg_p100[raw_col].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=3).mean()
                )
                df = df.drop(columns=[raw_col], errors='ignore')

        # -------------------------------------------------------------------------
        # 12c. Opponent Defensive Quality (Pace-Adjusted)
        # -------------------------------------------------------------------------
        # Compute how many pts/reb/ast each team allows per 100 possessions, rolling
        # L10. A player's stats against a poor defence should be discounted; against
        # elite defence they should be boosted. Replaces the raw def_col multiplier.
        if 'opp' in df.columns:
            for stat in ['pts', 'trb', 'ast']:
                raw_col = f'_{stat}_p100_raw_opp'  # scored AGAINST each team
                # Build team-level allowed-per-100 by tagging from opponent side
                _opp_stats = df.groupby(['opp', 'game_date'])[stat].sum().reset_index()
                _opp_poss  = df.groupby(['opp', 'game_date'])['_team_poss_game'].first().reset_index()
                _opp_merge = _opp_stats.merge(_opp_poss, on=['opp', 'game_date'])
                _opp_merge[raw_col] = _opp_merge[stat] / _opp_merge['_team_poss_game'].clip(lower=1) * 100
                _opp_merge = _opp_merge.sort_values(['opp', 'game_date'])
                _opp_merge[f'opp_allowed_{stat}_p100_l10'] = (
                    _opp_merge.groupby('opp')[raw_col]
                    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
                )
                df = df.merge(
                    _opp_merge[['opp', 'game_date', f'opp_allowed_{stat}_p100_l10']],
                    on=['opp', 'game_date'], how='left'
                )

            # Quality multiplier: league_avg_defense / opponent_defense
            # > 1.0 means opponent is weak (boost projection); < 1.0 means strong
            for stat in ['pts', 'trb', 'ast']:
                qa_col = f'opp_allowed_{stat}_p100_l10'
                if qa_col in df.columns:
                    _def_ref = df[df['game_date'] < stats_cutoff] if stats_cutoff is not None else df
                    _league_def = _def_ref[qa_col].median() if len(_def_ref) > 0 else df[qa_col].median()
                    df[f'{stat}_def_quality_mult'] = (
                        (_league_def / df[qa_col].replace(0, np.nan)).fillna(1.0).clip(0.6, 1.6)
                    )
                    # Quality-adjusted p100 rolling average
                    if f'{stat}_p100_l10' in df.columns:
                        df[f'{stat}_qa_p100_l10'] = df[f'{stat}_p100_l10'] * df[f'{stat}_def_quality_mult']

        # -------------------------------------------------------------------------
        # 13. Usage Rate Approximation
        # -------------------------------------------------------------------------
        # usage_rate = player's share of team possessions used
        df['usage_rate'] = df['_player_poss'] / df['_team_poss_game'].clip(lower=1)

        # Rolling usage (re-group because df gained new columns after merges)
        pg = df.sort_values(['player', 'game_date']).groupby('player')
        df['usage_l5'] = pg['usage_rate'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )
        df['usage_l10'] = pg['usage_rate'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        df['usage_trend'] = df['usage_l5'] / df['usage_l10'].replace(0, np.nan)
        # Spike: recent usage ≥15% above baseline — signals teammate absence
        df['usage_spike'] = (df['usage_l5'] >= df['usage_l10'] * 1.15).astype(int)

        # Drop intermediates — they are current-game data, not valid features
        df = df.drop(columns=['_player_poss', '_team_poss_game', 'usage_rate'], errors='ignore')

    # -------------------------------------------------------------------------
    # 14. Interaction Features
    # -------------------------------------------------------------------------
    # XGBoost can build interactions via tree depth, but explicit terms give the
    # model a direct, pre-multiplied signal for the most informative pairs.
    print("  Interaction features...")

    # Back-to-back × role (stars and rotation players are affected differently)
    if 'is_b2b' in df.columns:
        for role in ['is_star', 'is_starter', 'is_rotation']:
            if role in df.columns:
                df[f'b2b_{role}'] = df['is_b2b'] * df[role]

    # Opponent on B2B × scoring volume (easy defence = more stats for scorers)
    if 'opp_is_b2b' in df.columns and 'pts_l10' in df.columns:
        df['opp_b2b_pts_vol'] = df['opp_is_b2b'] * df['pts_l10']

    # Rest advantage × star player (stars benefit disproportionately from rest)
    if 'rest_advantage' in df.columns and 'is_star' in df.columns:
        df['rest_adv_star'] = df['rest_advantage'] * df['is_star']

    # Pace × usage: high-usage players benefit more from fast-paced games
    if 'pace_factor' in df.columns and 'usage_l10' in df.columns:
        df['pace_usage'] = df['pace_factor'] * df['usage_l10']

    # Hot player (trending up) vs easy defence
    if 'pts_trend' in df.columns and 'def_pts_rank' in df.columns:
        # def_pts_rank near 1 = bad defence (allows most pts); easier for hot players
        df['hot_vs_weak_def'] = df['pts_trend'] * df['def_pts_rank']

    # Schedule edge × usage spike (big mismatch player against a worn-out opponent)
    if 'usage_spike' in df.columns and 'opp_is_b2b' in df.columns:
        df['usage_spike_opp_b2b'] = df['usage_spike'] * df['opp_is_b2b']

    # -------------------------------------------------------------------------
    # 15. Starter proxy, foul trouble & blowout features (minutes model inputs)
    # -------------------------------------------------------------------------
    print("  Starter/foul/blowout features...")
    df = df.sort_values(['player', 'game_date'])
    pg2 = df.groupby('player')

    # --- Starter proxy (from minutes threshold) ---
    # Players who average 28+ min are almost certainly starters (no API needed)
    if 'mp_l10' in df.columns:
        df['is_starter_proxy']    = (df['mp_l10'] >= 28).astype(int)
        df['is_rotation_proxy']   = ((df['mp_l10'] >= 18) & (df['mp_l10'] < 28)).astype(int)
        df['is_bench_proxy']      = (df['mp_l10'] < 18).astype(int)
        # Consistency of starter minutes — low std = reliable role
        df['mp_consistency']      = 1.0 / (df['mp_std'].clip(lower=1) if 'mp_std' in df.columns
                                           else pd.Series(5.0, index=df.index))
        # How many of last 5 games did player play 28+ min
        mp_vals = pd.to_numeric(df['mp'], errors='coerce')
        df['starter_rate_l5'] = pg2['mp'].transform(
            lambda x: (pd.to_numeric(x, errors='coerce').shift(1) >= 28)
                       .rolling(5, min_periods=2).mean()
        )
        df['starter_rate_l10'] = pg2['mp'].transform(
            lambda x: (pd.to_numeric(x, errors='coerce').shift(1) >= 28)
                       .rolling(10, min_periods=3).mean()
        )

    # --- Foul trouble features (pf column required) ---
    if 'pf' in df.columns:
        df['pf'] = pd.to_numeric(df['pf'], errors='coerce').fillna(0)
        df['pf_l5'] = pg2['pf'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )
        df['pf_l10'] = pg2['pf'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        # Foul-out risk: % of last 10 games with 4+ fouls (high risk of reduced minutes)
        df['foul_risk_l10'] = pg2['pf'].transform(
            lambda x: (x.shift(1) >= 4).rolling(10, min_periods=3).mean()
        )
        # Foul rate per minute (foul-prone per unit of play)
        if 'mp_l10' in df.columns:
            df['pf_per_min_l10'] = df['pf_l10'] / df['mp_l10'].clip(lower=1)
        # Trend: getting more/fewer fouls recently
        df['pf_trend'] = df['pf_l5'] / df['pf_l10'].replace(0, np.nan).fillna(1)

    # --- Blowout & game margin features (plus_minus required) ---
    if 'plus_minus' in df.columns:
        df['plus_minus'] = pd.to_numeric(df['plus_minus'], errors='coerce').fillna(0)

        # Team-level margin per game (sum of plus_minus / 5 ≈ point differential)
        team_margin = (
            df.groupby(['team', 'game_date'])['plus_minus'].mean()
            .reset_index().rename(columns={'plus_minus': '_team_margin'})
            .sort_values(['team', 'game_date'])
        )
        team_margin['team_margin_l10'] = team_margin.groupby('team')['_team_margin'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        # Blowout rate: % of last 10 games with |margin| > 15 pts
        team_margin['team_blowout_rate_l10'] = team_margin.groupby('team')['_team_margin'].transform(
            lambda x: (x.shift(1).abs() > 15).rolling(10, min_periods=3).mean()
        )
        df = df.merge(
            team_margin[['team', 'game_date', 'team_margin_l10', 'team_blowout_rate_l10']],
            on=['team', 'game_date'], how='left'
        )

        # Per-player: minutes drop in blowouts (rolling corr of mp vs plus_minus)
        df['_mp_num']      = pd.to_numeric(df['mp'], errors='coerce')
        df['_pm_shifted']  = df.groupby('player')['plus_minus'].transform(lambda x: x.shift(1))
        df = df.sort_values(['player', 'game_date'])
        # Use expanding window corr computed per player
        df['mp_margin_corr'] = (
            df.groupby('player')[['_mp_num', '_pm_shifted']]
              .apply(lambda g: g['_mp_num'].rolling(15, min_periods=5).corr(g['_pm_shifted']))
              .reset_index(level=0, drop=True)
        )
        df = df.drop(columns=['_mp_num', '_pm_shifted', '_team_margin'], errors='ignore')

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    temp_cols = [c for c in df.columns if '_temp' in c or c.startswith('_')]
    df = df.drop(columns=temp_cols + ['prev_date', 'prev_margin', '_game_weight'], errors='ignore')
    
    print(f"  Created {len([c for c in df.columns if c not in ['player', 'game_date', 'team', 'opp']])} features")

    # Downcast float64 -> float32 to halve memory footprint (~600MB -> ~300MB)
    float_cols = df.select_dtypes(include='float64').columns
    df[float_cols] = df[float_cols].astype(np.float32)
    int_cols = df.select_dtypes(include='int64').columns
    df[int_cols] = df[int_cols].astype(np.int32)
    gc.collect()

    return df


# =============================================================================
# ADVANCED FEATURE ENGINEERING (for advanced models)
# =============================================================================

def _build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the extra features required by train_advanced_models.py models.
    Must be called on the FULL sorted game log (not just latest) so rolling
    windows have history.  Returns df with new columns added.
    """
    df = _build_dynamic_context_features(df)
    df = df.sort_values(['player', 'game_date']).copy()
    stats = ['pts', 'trb', 'ast']

    # --- L20 rolling windows (nba_props already has L10; advanced models want L20) ---
    for stat in stats + ['mp', 'fga', 'fta', '3pa']:
        col = stat if stat in df.columns else None
        if col is None:
            continue
        if f'{stat}_l20' not in df.columns:
            df[f'{stat}_l20'] = df.groupby('player')[col].transform(
                lambda x: x.shift(1).rolling(20, min_periods=5).mean()
            )

    # --- Home / Away L15 splits ---
    window = 15
    for stat in stats:
        if f'{stat}_home_l{window}' in df.columns:
            continue
        df[f'{stat}_home_temp'] = np.where(df.get('is_home', pd.Series(dtype=float)) == 1, df[stat], np.nan)
        df[f'{stat}_away_temp'] = np.where(df.get('is_home', pd.Series(dtype=float)) == 0, df[stat], np.nan)
        df[f'{stat}_home_l{window}'] = df.groupby('player')[f'{stat}_home_temp'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )
        df[f'{stat}_away_l{window}'] = df.groupby('player')[f'{stat}_away_temp'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )
        df[f'{stat}_situational_avg'] = np.where(
            df.get('is_home', pd.Series(dtype=float)) == 1,
            df[f'{stat}_home_l{window}'],
            df[f'{stat}_away_l{window}']
        )
        df[f'{stat}_situational_avg'] = df[f'{stat}_situational_avg'].fillna(df[f'{stat}_l10'])

    if 'days_rest' in df.columns and 'is_b2b_second' not in df.columns:
        df['is_b2b_second'] = (pd.to_numeric(df['days_rest'], errors='coerce') == 1).astype(int)

    if 'is_b2b_second' in df.columns and 'days_rest' in df.columns:
        for stat in stats:
            if stat not in df.columns:
                continue
            on_b2b_col = f'{stat}_on_b2b'
            rested_col = f'{stat}_rested'
            penalty_col = f'{stat}_b2b_penalty'
            if on_b2b_col not in df.columns:
                df[f'{stat}_b2b_temp'] = np.where(df['is_b2b_second'] == 1, df[stat], np.nan)
                df[on_b2b_col] = df.groupby('player')[f'{stat}_b2b_temp'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=2).mean()
                )
            if rested_col not in df.columns:
                rested_mask = pd.to_numeric(df['days_rest'], errors='coerce').fillna(2) >= 2
                df[f'{stat}_rested_temp'] = np.where(rested_mask, df[stat], np.nan)
                df[rested_col] = df.groupby('player')[f'{stat}_rested_temp'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=2).mean()
                )
            if penalty_col not in df.columns:
                df[penalty_col] = df[on_b2b_col] - df[rested_col]

    # Drop temp columns
    temp_cols = [c for c in df.columns if c.endswith('_temp')]
    df = df.drop(columns=temp_cols, errors='ignore')

    # --- Opponent allowed stats (MUST come before opp_def_tier so tier has real ranks) ---
    opp_col = 'opp' if 'opp' in df.columns else None
    if opp_col and 'opp_pts_allowed_l10' not in df.columns:
        team_allowed = (
            df.groupby(['opp', 'game_date'])[['pts', 'trb', 'ast', 'fga']]
            .sum().reset_index().sort_values(['opp', 'game_date'])
        )
        tg = team_allowed.groupby('opp')
        for stat, out in [('pts','opp_pts_allowed_l10'), ('trb','opp_trb_allowed_l10'),
                          ('ast','opp_ast_allowed_l10'), ('fga','opp_fga_allowed_l10')]:
            team_allowed[out] = tg[stat].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
        # Per-game rank: rank each team's defensive avg on each date
        team_allowed['opp_pts_allowed_rank'] = team_allowed.groupby('game_date')['opp_pts_allowed_l10'].rank(method='min')
        team_allowed['opp_trb_allowed_rank'] = team_allowed.groupby('game_date')['opp_trb_allowed_l10'].rank(method='min')
        team_allowed['opp_ast_allowed_rank'] = team_allowed.groupby('game_date')['opp_ast_allowed_l10'].rank(method='min')
        df = df.merge(
            team_allowed[['opp', 'game_date', 'opp_pts_allowed_l10', 'opp_trb_allowed_l10',
                          'opp_ast_allowed_l10', 'opp_fga_allowed_l10',
                          'opp_pts_allowed_rank', 'opp_trb_allowed_rank', 'opp_ast_allowed_rank']],
            on=['opp', 'game_date'], how='left'
        )

    # --- Defense-tier adjusted averages ---
    # opp_def_tier: bucket opponent defensive rank into 1-5 (1=best defense)
    if 'opp_def_tier' not in df.columns:
        if 'opp_pts_allowed_rank' in df.columns:
            df['opp_def_tier'] = pd.cut(
                df['opp_pts_allowed_rank'].fillna(15),
                bins=[0, 6, 12, 18, 24, 30],
                labels=[1, 2, 3, 4, 5]
            ).astype(float)
        else:
            df['opp_def_tier'] = 3.0  # neutral fallback

    for stat in stats:
        if f'{stat}_defense_adj' in df.columns:
            continue
        df[f'{stat}_vs_good_temp'] = np.where(df['opp_def_tier'] <= 2, df[stat], np.nan)
        df[f'{stat}_vs_bad_temp']  = np.where(df['opp_def_tier'] >= 4, df[stat], np.nan)
        df[f'{stat}_vs_good_def'] = df.groupby('player')[f'{stat}_vs_good_temp'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        df[f'{stat}_vs_bad_def'] = df.groupby('player')[f'{stat}_vs_bad_temp'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        df[f'{stat}_defense_adj'] = np.where(
            df['opp_def_tier'] <= 2, df[f'{stat}_vs_good_def'],
            np.where(df['opp_def_tier'] >= 4, df[f'{stat}_vs_bad_def'], df[f'{stat}_l10'])
        )
        df[f'{stat}_defense_adj'] = df[f'{stat}_defense_adj'].fillna(df[f'{stat}_l10'])

    temp_cols = [c for c in df.columns if c.endswith('_vs_good_temp') or c.endswith('_vs_bad_temp')]
    df = df.drop(columns=temp_cols, errors='ignore')

    # --- Best estimate (weighted composite) ---
    for stat in stats:
        if f'{stat}_best_estimate' in df.columns:
            continue
        components, weights = [], []
        anchor_col = f'{stat}_recency_anchor'
        ewma_col = f'{stat}_ewma5'
        if anchor_col in df.columns:
            components.append(df[anchor_col]); weights.append(0.45)
        elif ewma_col in df.columns:
            components.append(df[ewma_col]); weights.append(0.35)
        if f'{stat}_defense_adj' in df.columns:
            components.append(df[f'{stat}_defense_adj']); weights.append(0.20)
        if f'{stat}_situational_avg' in df.columns:
            components.append(df[f'{stat}_situational_avg']); weights.append(0.15)
        if f'{stat}_season_avg' in df.columns:
            components.append(df[f'{stat}_season_avg'].fillna(df.get(f'{stat}_l20')))
            weights.append(0.15)
        if f'{stat}_l20' in df.columns:
            components.append(df[f'{stat}_l20']); weights.append(0.05)
        if components:
            total_w = sum(weights)
            df[f'{stat}_best_estimate'] = sum(c * w for c, w in zip(components, weights)) / total_w
        else:
            df[f'{stat}_best_estimate'] = df[f'{stat}_l10']

    # --- Vegas-slow flag ---
    for stat in stats:
        if f'{stat}_vegas_slow' in df.columns:
            continue
        anchor_col = f'{stat}_recency_anchor' if f'{stat}_recency_anchor' in df.columns else (
            f'{stat}_ewma5' if f'{stat}_ewma5' in df.columns else None
        )
        baseline_col = f'{stat}_synthetic_line' if f'{stat}_synthetic_line' in df.columns else (
            f'{stat}_season_avg' if f'{stat}_season_avg' in df.columns else f'{stat}_l10'
        )
        if anchor_col is not None and baseline_col in df.columns:
            edge = df[anchor_col] - df[baseline_col]
            df[f'{stat}_vegas_slow'] = (edge.abs() > df[baseline_col].abs().fillna(0) * 0.1).astype(int)
        else:
            df[f'{stat}_vegas_slow'] = 0

    # --- Usage features (mirror feature_pipeline) ---
    if 'usage_proxy' not in df.columns and 'fga' in df.columns:
        df['usage_proxy'] = (df['fga'] + df.get('fta', 0) * 0.44 + df.get('tov', 0)) / df['mp'].replace(0, np.nan)
        df['usage_l5']  = df.groupby('player')['usage_proxy'].transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
        df['usage_l10'] = df.groupby('player')['usage_proxy'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        df['usage_trend'] = (df['usage_l5'] / df['usage_l10'].replace(0, np.nan)).fillna(1.0)

    if 'mp_l3' not in df.columns and 'mp' in df.columns:
        df['mp_l3'] = df.groupby('player')['mp'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    if 'mp_trend_ratio' not in df.columns and 'mp_l3' in df.columns and 'mp_l10' in df.columns:
        df['mp_trend_ratio'] = (df['mp_l3'] / df['mp_l10'].replace(0, np.nan)).fillna(1.0)
        df['role_expanding'] = (df['mp_trend_ratio'] > 1.1).astype(int)
        df['role_shrinking'] = (df['mp_trend_ratio'] < 0.9).astype(int)

    if 'fga_l3' not in df.columns and 'fga' in df.columns:
        df['fga_l3'] = df.groupby('player')['fga'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df['fga_trend'] = (df['fga_l3'] / df['fga_l10'].replace(0, np.nan)).fillna(1.0) if 'fga_l10' in df.columns else 1.0

    # --- Game environment score ---
    if 'game_environment_score' not in df.columns:
        pace    = df.get('opp_pace_factor', pd.Series(1.0, index=df.index))
        scoring = df.get('team_scoring_trend', pd.Series(1.0, index=df.index))
        def_rank = df.get('opp_pts_allowed_rank', pd.Series(15, index=df.index)).fillna(15)
        raw = pace * 0.4 + scoring * 0.3 + (30 - def_rank) / 30 * 0.3
        df['game_environment_score'] = ((raw - 0.4) / 0.6).clip(0, 1)

    if 'high_scoring_game' not in df.columns:
        trend = df['team_scoring_trend'] if 'team_scoring_trend' in df.columns else pd.Series(1.0, index=df.index)
        df['high_scoring_game'] = (trend > 1.05).astype(int)

    # --- L20 for shooting pcts ---
    for stat in ['3p_pct', 'ft_pct']:
        if stat in df.columns and f'{stat}_l20' not in df.columns:
            df[f'{stat}_l20'] = df.groupby('player')[stat].transform(
                lambda x: x.shift(1).rolling(20, min_periods=5).mean()
            )

    # --- Alias ewma names: nba_props uses pts_ewma, models expect pts_ewma5 ---
    ewma_stats = ['pts', 'trb', 'ast', 'fga', 'fta', '3pa', 'stl', 'blk', 'tov', 'mp']
    for stat in ewma_stats:
        src = f'{stat}_ewma'
        dst = f'{stat}_ewma5'
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # --- Alias std names: pts_std → pts_std_l10 ---
    for stat in ['pts', 'trb', 'ast']:
        src = f'{stat}_std'
        dst = f'{stat}_std_l10'
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # --- EWMA std (rolling std of the ewma series) ---
    for stat in ['pts', 'trb', 'ast', 'mp']:
        dst = f'{stat}_ewma_std'
        if dst not in df.columns:
            ewma_col = f'{stat}_ewma5' if f'{stat}_ewma5' in df.columns else f'{stat}_ewma'
            if ewma_col in df.columns:
                df[dst] = df.groupby('player')[ewma_col].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=3).std()
                ).fillna(0)

    # --- Home/away diff ---
    window = 15
    for stat in stats:
        dst = f'{stat}_home_away_diff'
        if dst not in df.columns:
            home_col = f'{stat}_home_l{window}'
            away_col = f'{stat}_away_l{window}'
            if home_col in df.columns and away_col in df.columns:
                df[dst] = df[home_col] - df[away_col]

    # (Opponent allowed stats merged earlier, before opp_def_tier computation)

    # --- Alias likely_starter ---
    if 'likely_starter' not in df.columns:
        if 'is_starter_proxy' in df.columns:
            df['likely_starter'] = df['is_starter_proxy'].astype(int)
        elif 'mp_l10' in df.columns:
            df['likely_starter'] = (df['mp_l10'] >= 28).astype(int)

    # --- Edge vs avg and pct_diff ---
    for stat in stats:
        edge_col = f'{stat}_edge_vs_avg'
        pct_col  = f'{stat}_pct_diff'
        if edge_col not in df.columns:
            anchor_col = f'{stat}_recency_anchor' if f'{stat}_recency_anchor' in df.columns else (
                f'{stat}_ewma5' if f'{stat}_ewma5' in df.columns else f'{stat}_ewma'
            )
            baseline_col = f'{stat}_synthetic_line' if f'{stat}_synthetic_line' in df.columns else (
                f'{stat}_season_avg' if f'{stat}_season_avg' in df.columns else f'{stat}_l10'
            )
            if anchor_col in df.columns and baseline_col in df.columns:
                df[edge_col] = df[anchor_col] - df[baseline_col]
                df[pct_col]  = (df[edge_col] / df[baseline_col].replace(0, np.nan)).fillna(0)

    # --- Team scoring trend ---
    if 'team_scoring_trend' not in df.columns and 'team' in df.columns:
        team_pts = (
            df.groupby(['team', 'game_date'])['pts'].sum()
            .reset_index()
            .sort_values(['team', 'game_date'])
        )
        tg = team_pts.groupby('team')
        team_pts['team_pts_l5']  = tg['pts'].transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
        team_pts['team_pts_l10'] = tg['pts'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        team_pts['team_scoring_trend'] = (
            team_pts['team_pts_l5'] / team_pts['team_pts_l10'].replace(0, np.nan)
        ).fillna(1.0)
        df = df.merge(team_pts[['team', 'game_date', 'team_scoring_trend']], on=['team', 'game_date'], how='left')
        df['team_scoring_trend'] = df['team_scoring_trend'].fillna(1.0)

    # --- Confidence and risk scores ---
    for stat in stats:
        conf_col = f'{stat}_confidence'
        if conf_col not in df.columns:
            std_col = f'{stat}_std_l10' if f'{stat}_std_l10' in df.columns else f'{stat}_std'
            l10_col = f'{stat}_l10'
            if std_col in df.columns and l10_col in df.columns:
                cv = (df[std_col] / df[l10_col].replace(0, np.nan)).fillna(1.0).clip(0.1, 3.0)
                games_factor = (df.get('games_played', pd.Series(20, index=df.index)) / 20).clip(0, 1)
                df[conf_col] = (games_factor / cv).clip(0, 1)
            else:
                df[conf_col] = 0.5

    if 'risk_factor' not in df.columns:
        cv_sum = sum(
            (df.get(f'{s}_std_l10', df.get(f'{s}_std', pd.Series(1.0, index=df.index))) /
             df.get(f'{s}_l10', pd.Series(1.0, index=df.index)).replace(0, np.nan)).fillna(1.0)
            for s in stats
        )
        df['risk_factor'] = (cv_sum / len(stats)).clip(0, 3.0)

    return df


def _load_advanced_models():
    """
    Load regression models, classifiers, and selected feature lists from train_advanced_models.py.
    Returns advanced model artifacts or empty fallbacks.
    """
    models_dir = CONFIG['models_dir']
    feature_imp_path = models_dir / 'feature_importance_advanced.json'
    if not feature_imp_path.exists():
        return None, None, {}, {}, {}, {}, {'edge': {}, 'meta': {}}

    try:
        models = {}
        classifiers = {}
        edge_models = {}
        meta_models = {}
        feature_cols = None
        ALL_STATS = ['pts', 'trb', 'ast', 'pra', 'pr', 'pa', 'stl', 'blk', 'tov']

        selected_features = {}
        sel_path = models_dir / 'selected_features_advanced.json'
        if sel_path.exists():
            with open(sel_path) as f:
                selected_features = json.load(f)

        for stat in ALL_STATS:
            reg_path = models_dir / f'xgb_{stat}_advanced.pkl'
            if not reg_path.exists():
                return None, None, {}, {}, {}, {}, {'edge': {}, 'meta': {}}
            with open(reg_path, 'rb') as f:
                models[stat] = pickle.load(f)
            if feature_cols is None:
                model_obj = models[stat]
                if isinstance(model_obj, dict):
                    feature_cols = model_obj.get('feature_names')
                elif hasattr(model_obj, 'get_booster'):
                    feature_cols = model_obj.get_booster().feature_names

            clf_path = models_dir / f'clf_{stat}_advanced.pkl'
            if clf_path.exists():
                with open(clf_path, 'rb') as f:
                    classifiers[stat] = pickle.load(f)
            edge_path = models_dir / f'edge_xgb_{stat}_advanced.pkl'
            if edge_path.exists():
                with open(edge_path, 'rb') as f:
                    edge_models[stat] = pickle.load(f)
            meta_path = models_dir / f'meta_clf_{stat}_advanced.pkl'
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    meta_models[stat] = pickle.load(f)
        if feature_cols is None and selected_features:
            merged_features = []
            for cols in selected_features.values():
                merged_features.extend(cols)
            feature_cols = list(dict.fromkeys(merged_features))

        market_feature_sets = {'edge': {}, 'meta': {}}
        market_path = models_dir / 'market_feature_sets_advanced.json'
        if market_path.exists():
            with open(market_path) as f:
                market_feature_sets = json.load(f)

        print(f"  Loaded advanced models: {len(models)} regression + {len(classifiers)} classifiers + {len(edge_models)} edge + {len(meta_models)} meta")
        return models, feature_cols, classifiers, selected_features, edge_models, meta_models, market_feature_sets
    except Exception as e:
        print(f"  WARNING: Could not load advanced models ({e}), falling back to standard models")
        return None, None, {}, {}, {}, {}, {'edge': {}, 'meta': {}}


def _attach_market_model_scores(pred_df: pd.DataFrame,
                                feature_df: Optional[pd.DataFrame],
                                edge_models: Dict[str, object],
                                meta_models: Dict[str, object],
                                market_feature_sets: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
    """Attach market-edge and abstention scores using the advanced line-aware models."""
    def _model_feature_names(model) -> Optional[List[str]]:
        try:
            if hasattr(model, 'get_booster'):
                names = model.get_booster().feature_names
                if names:
                    return list(names)
        except Exception:
            pass
        names = getattr(model, 'feature_names_in_', None)
        return list(names) if names is not None else None

    if len(pred_df) == 0 or not edge_models:
        return pred_df

    from train_advanced_models import _build_market_feature_block, _build_meta_feature_block

    working = pred_df.copy()
    merged = working.copy()
    if feature_df is not None and len(feature_df) > 0:
        base = feature_df.copy()
        if 'player_norm' not in base.columns and 'player' in base.columns:
            base['player_norm'] = base['player'].apply(_normalize_name)
        if 'player_norm' not in merged.columns and 'player' in merged.columns:
            merged['player_norm'] = merged['player'].apply(_normalize_name)
        merge_keys = []
        if 'game_date' in base.columns and 'game_date' in merged.columns and 'player_norm' in base.columns and 'player_norm' in merged.columns:
            base['game_date'] = pd.to_datetime(base['game_date'], errors='coerce').dt.normalize()
            merged['game_date'] = pd.to_datetime(merged['game_date'], errors='coerce').dt.normalize()
            merge_keys = ['game_date', 'player_norm']
        elif 'player_norm' in base.columns and 'player_norm' in merged.columns:
            merge_keys = ['player_norm']
        elif 'player' in base.columns and 'player' in merged.columns:
            merge_keys = ['player']
        if merge_keys:
            extra_cols = [c for c in base.columns if c not in merged.columns or c in merge_keys]
            merged = merged.merge(base[extra_cols].drop_duplicates(subset=merge_keys, keep='last'), on=merge_keys, how='left')

    for target, edge_model in edge_models.items():
        line_col = f'{target}_line'
        pred_col = f'{target}_pred'
        if line_col not in merged.columns or pred_col not in merged.columns:
            continue

        market_block = _build_market_feature_block(merged, target, merged[line_col], raw_pred=merged[pred_col])
        edge_features = _model_feature_names(edge_model) or market_feature_sets.get('edge', {}).get(target, list(market_block.columns))
        for col in edge_features:
            if col not in market_block.columns:
                market_block[col] = 0.0
        market_X = market_block[edge_features].fillna(0.0)
        edge_pred = edge_model.predict(market_X)
        raw_edge = pd.to_numeric(merged[pred_col], errors='coerce') - pd.to_numeric(merged[line_col], errors='coerce')

        working[f'{target}_market_edge_pred'] = np.round(edge_pred, 3)
        working[f'{target}_model_signal_edge'] = np.round(raw_edge, 3)
        working[f'{target}_edge_disagreement'] = np.round(np.abs(edge_pred - raw_edge.fillna(0.0)), 3)
        working[f'{target}_regime_score'] = np.round(pd.to_numeric(market_block.get('market_role_shock', 0.0), errors='coerce').fillna(0.0), 3)

        if target in meta_models:
            meta_block = _build_meta_feature_block(
                market_block,
                raw_edge=raw_edge,
                market_edge_pred=pd.Series(edge_pred, index=market_block.index),
            )
            meta_features = _model_feature_names(meta_models[target]) or market_feature_sets.get('meta', {}).get(target, list(meta_block.columns))
            for col in meta_features:
                if col not in meta_block.columns:
                    meta_block[col] = 0.0
            meta_X = meta_block[meta_features].fillna(0.0)
            working[f'{target}_meta_prob'] = np.round(meta_models[target].predict_proba(meta_X)[:, 1], 4)

    return working


# =============================================================================
# MODEL TRAINING
# =============================================================================

LAYER_FEATURE_GROUPS = {
    'pbp': ['possessions', 'raw_ppp', 'garbage_time_flag', 'competitive_poss'],
    'season_ppp': ['adj_ppp', 'raw_ppp_season', 'opp_def_rtg_avg', 'total_possessions', 'efg_pct', 'ts_pct'],
    'blowout': ['clean_pts', 'clean_reb', 'clean_ast', 'clean_fg3m', 'clean_min', 'pts_delta', 'n_exclude', 'n_heavy', 'n_partial'],
    'usage': ['usg_pct', 'pts_per_poss', 'ast_per_poss', 'reb_per_poss', 'min_pg', 'poss_pg', 'off_rtg', 'net_rtg', 'role_weight'],
    'luck': ['total_luck_score', 'efg_luck_score', 'fg3_luck_score', 'ft_luck_score', 'pts_luck_adj', 'fg3_regressed', 'ft_regressed'],
}


def _summarize_layer_feature_coverage(
    frame: pd.DataFrame,
    all_layer_cols: List[str],
    added_layer_cols: Optional[List[str]] = None,
) -> Dict[str, object]:
    present_cols = [c for c in all_layer_cols if c in frame.columns]
    total_rows = len(frame)
    if total_rows == 0 or not present_cols:
        return {
            'rows_with_any_layer': 0,
            'rows_without_any_layer': total_rows,
            'rows_with_any_added': 0,
            'group_counts': {},
            'group_missing_counts': {},
        }

    layer_non_null = frame[present_cols].notna()
    any_layer_mask = layer_non_null.any(axis=1)

    present_added_cols = [c for c in (added_layer_cols or []) if c in frame.columns]
    if present_added_cols:
        any_added_mask = frame[present_added_cols].notna().any(axis=1)
    else:
        any_added_mask = pd.Series(False, index=frame.index)

    group_counts = {}
    group_missing_counts = {}
    for group_name, group_cols in LAYER_FEATURE_GROUPS.items():
        present_group_cols = [c for c in group_cols if c in frame.columns]
        if not present_group_cols:
            continue
        count = int(frame[present_group_cols].notna().any(axis=1).sum())
        group_counts[group_name] = count
        group_missing_counts[group_name] = total_rows - count

    return {
        'rows_with_any_layer': int(any_layer_mask.sum()),
        'rows_without_any_layer': int((~any_layer_mask).sum()),
        'rows_with_any_added': int(any_added_mask.sum()),
        'group_counts': group_counts,
        'group_missing_counts': group_missing_counts,
    }


def _merge_layer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge enriched layer features (PBP/PPP, blowout, usage, luck) onto a game-log
    DataFrame. Joins on (player_id, season) when available, else (player, season).

    Sources (tried in order):
      1. data/training_dataset_enriched.csv  — built by build_historical_dataset.py
      2. Individual cache files in data/hist_cache/
    """
    LAYER_COLS = [
        # Layer 1 — possession-level PBP context
        'possessions', 'raw_ppp', 'garbage_time_flag', 'competitive_poss',
        # Layer 2 — season-level PPP adjustments
        'adj_ppp', 'raw_ppp_season', 'opp_def_rtg_avg', 'total_possessions',
        'efg_pct', 'ts_pct',
        # Layer 5 — blowout-discounted baselines
        'clean_pts', 'clean_reb', 'clean_ast', 'clean_fg3m', 'clean_min',
        'pts_delta', 'n_exclude', 'n_heavy', 'n_partial',
        # Layer 3 — usage & role
        'usg_pct', 'pts_per_poss', 'ast_per_poss', 'reb_per_poss',
        'min_pg', 'poss_pg', 'off_rtg', 'net_rtg', 'role_weight',
        # Layer 4 — luck & shooting regression
        'total_luck_score', 'efg_luck_score', 'fg3_luck_score', 'ft_luck_score',
        'pts_luck_adj', 'fg3_regressed', 'ft_regressed',
    ]

    # Already merged — skip
    already = [c for c in LAYER_COLS if c in df.columns]
    if len(already) == len(LAYER_COLS):
        return df

    enriched_path = _HERE / 'data' / 'training_dataset_enriched.csv'
    cache_dir     = _HERE / 'data' / 'hist_cache'

    layer_df = None

    def _season_aggregate(part: pd.DataFrame) -> pd.DataFrame:
        avail = [c for c in LAYER_COLS if c in part.columns]
        if not avail:
            return pd.DataFrame()
        if 'PLAYER_ID' in part.columns and 'player_id' not in part.columns:
            part = part.rename(columns={'PLAYER_ID': 'player_id'})
        if 'player_name' in part.columns and 'player' not in part.columns:
            part = part.rename(columns={'player_name': 'player'})
        if 'season' not in part.columns:
            return pd.DataFrame()

        group_keys = None
        if 'player_id' in part.columns and 'player' in part.columns:
            part['player_id'] = pd.to_numeric(part['player_id'], errors='coerce')
            group_keys = ['player_id', 'player', 'season']
        elif 'player_id' in part.columns:
            part['player_id'] = pd.to_numeric(part['player_id'], errors='coerce')
            group_keys = ['player_id', 'season']
        elif 'player' in part.columns:
            group_keys = ['player', 'season']
        if group_keys is None:
            return pd.DataFrame()

        agg = part[group_keys + avail].copy()
        for col in avail:
            agg[col] = pd.to_numeric(agg[col], errors='coerce')
        return agg.groupby(group_keys, dropna=False)[avail].mean().reset_index()

    if enriched_path.exists():
        try:
            enr = pd.read_csv(enriched_path, low_memory=False)
            layer_df = _season_aggregate(enr)
            if not layer_df.empty:
                loaded_cols = len([c for c in LAYER_COLS if c in layer_df.columns])
                print(f"  [Layer features] Loaded {loaded_cols} cols from training_dataset_enriched.csv")
        except Exception as e:
            print(f"  [Layer features] enriched CSV failed ({e}), trying cache files...")

    if layer_df is None or layer_df.empty:
        # Fallback: load individual cache files and join
        frames = []
        for fname, name_col in [
            ('hist_layer1_pbp_master.csv', 'player_name'),
            ('hist_layer2_adj_ppp.csv',    'player_name'),
            ('hist_layer5_blowout.csv', 'player_name'),
            ('hist_layer3_usage.csv',   'player_name'),
            ('hist_layer4_luck.csv',    'player_name'),
        ]:
            p = cache_dir / fname
            if not p.exists():
                continue
            try:
                part = pd.read_csv(p, low_memory=False)
                if name_col in part.columns and 'player' not in part.columns:
                    part = part.rename(columns={name_col: 'player'})
                part_agg = _season_aggregate(part)
                if not part_agg.empty:
                    frames.append(part_agg)
            except Exception:
                pass

        if frames:
            from functools import reduce
            join_keys = ['player', 'season']
            if all('player_id' in frame.columns for frame in frames):
                if all('player' in frame.columns for frame in frames):
                    join_keys = ['player_id', 'player', 'season']
                else:
                    join_keys = ['player_id', 'season']
            layer_df = reduce(lambda a, b: a.merge(b, on=join_keys, how='outer'), frames)
            print(f"  [Layer features] Loaded from cache files ({len(layer_df.columns)-len(join_keys)} cols)")

    if layer_df is None or layer_df.empty:
        print("  [Layer features] No enriched features found — skipping merge")
        return df

    # Normalize join keys.
    layer_df['season'] = layer_df['season'].astype(str)
    df['season'] = df['season'].astype(str)
    join_keys = ['player', 'season']
    if 'player_id' in layer_df.columns and 'player_id' in df.columns:
        layer_df['player_id'] = pd.to_numeric(layer_df['player_id'], errors='coerce')
        df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce')
        join_keys = ['player_id', 'season']
    else:
        if 'player' not in layer_df.columns:
            print("  [Layer features] Missing player names for name-based merge — skipping merge")
            return df
        layer_df['player'] = layer_df['player'].astype(str)
        df['player'] = df['player'].astype(str)

    cols_to_add = [c for c in layer_df.columns if c not in ('player_id', 'player', 'season') and c not in df.columns]
    if not cols_to_add:
        return df

    merged = df.merge(layer_df[join_keys + cols_to_add],
                      on=join_keys, how='left')

    unmatched_mask = merged[cols_to_add[0]].isna() if cols_to_add else pd.Series(False, index=merged.index)
    n_recovered = 0

    # Name-based fallback: when the primary join was on player_id and some rows still
    # didn't match (e.g. new callups, ID format drift), attempt a second pass using
    # diacritic-normalized player name + season.
    if join_keys == ['player_id', 'season'] and unmatched_mask.any() \
            and 'player' in layer_df.columns and 'player' in merged.columns:
        import unicodedata

        def _norm(s: str) -> str:
            s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')
            return s.lower().strip()

        nl = layer_df[['player', 'season'] + cols_to_add].copy()
        nl['_pnorm'] = nl['player'].apply(_norm)
        nl = nl.drop(columns=['player']).drop_duplicates(subset=['_pnorm', 'season'])

        merged['_pnorm'] = merged['player'].apply(_norm)

        um_rows = merged.loc[unmatched_mask, ['_pnorm', 'season']].merge(
            nl, on=['_pnorm', 'season'], how='left'
        )
        for col in cols_to_add:
            if col in um_rows.columns:
                merged.loc[unmatched_mask, col] = um_rows[col].values

        merged.drop(columns=['_pnorm'], inplace=True)

        still_unmatched = merged[cols_to_add[0]].isna()
        n_recovered = int(unmatched_mask.sum()) - int(still_unmatched.sum())
        unmatched_mask = still_unmatched

    coverage = _summarize_layer_feature_coverage(merged, LAYER_COLS, cols_to_add)
    n_matched_final = coverage['rows_with_any_layer']
    recover_note = f" (+{n_recovered} via name fallback)" if n_recovered > 0 else ""
    added_note = ""
    if cols_to_add:
        added_note = f"; {coverage['rows_with_any_added']:,}/{len(df):,} rows received >=1 newly merged value"
    print(
        f"  [Layer features] Merged {len(cols_to_add)} features — "
        f"{n_matched_final:,}/{len(df):,} rows have layer coverage{recover_note}{added_note}"
    )

    group_counts = coverage['group_counts']
    if group_counts:
        group_summary = ", ".join(
            f"{group_name}: {count:,}/{len(df):,}"
            for group_name, count in group_counts.items()
        )
        print(f"  [Layer features] Coverage by layer — {group_summary}")

    partial_gaps = [
        f"{group_name}: {missing_count}"
        for group_name, missing_count in coverage['group_missing_counts'].items()
        if missing_count > 0
    ]
    if partial_gaps:
        print(f"  [Layer features] Partial gaps — {', '.join(partial_gaps)}")

    no_layer_mask = ~merged[[c for c in LAYER_COLS if c in merged.columns]].notna().any(axis=1)
    if no_layer_mask.any() and 'player' in merged.columns:
        missing = sorted(merged.loc[no_layer_mask, 'player'].dropna().unique().tolist())
        print(f"  [Layer features] WARNING — {len(missing)} players without any layer features:")
        for name in missing[:40]:
            print(f"    - {name}")
        if len(missing) > 40:
            print(f"    ... and {len(missing) - 40} more")

    return merged


def _merge_current_pbp_context(df: pd.DataFrame) -> pd.DataFrame:
    """Merge current-season per-game PBP context from pbp_possessions_master.csv."""
    pbp_path = _HERE / 'data' / 'pbp_possessions_master.csv'
    if not pbp_path.exists() or 'game_date' not in df.columns:
        return df

    try:
        pbp = pd.read_csv(pbp_path, low_memory=False)
    except Exception:
        return df

    if pbp.empty:
        return df

    if 'possessions_used' in pbp.columns and 'possessions' not in pbp.columns:
        pbp = pbp.rename(columns={'possessions_used': 'possessions'})
    if 'ppp_raw' in pbp.columns and 'raw_ppp' not in pbp.columns:
        pbp = pbp.rename(columns={'ppp_raw': 'raw_ppp'})
    if 'garbage_time' in pbp.columns and 'garbage_time_flag' not in pbp.columns:
        pbp = pbp.rename(columns={'garbage_time': 'garbage_time_flag'})
    if 'player_name' in pbp.columns and 'player' not in pbp.columns:
        pbp = pbp.rename(columns={'player_name': 'player'})

    if 'game_date' not in pbp.columns or 'possessions' not in pbp.columns:
        return df

    pbp['game_date'] = pd.to_datetime(pbp['game_date'], errors='coerce')
    df = df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

    df = df.copy()
    df_player_id_col = 'player_id' if 'player_id' in df.columns else ('PLAYER_ID' if 'PLAYER_ID' in df.columns else None)
    df_game_id_col = 'game_id' if 'game_id' in df.columns else ('GAME_ID' if 'GAME_ID' in df.columns else None)
    pbp_player_id_col = 'player_id' if 'player_id' in pbp.columns else None
    pbp_game_id_col = 'game_id' if 'game_id' in pbp.columns else None

    if pbp_player_id_col:
        pbp[pbp_player_id_col] = pd.to_numeric(pbp[pbp_player_id_col], errors='coerce')
    if df_player_id_col:
        df[df_player_id_col] = pd.to_numeric(df[df_player_id_col], errors='coerce')
    if pbp_game_id_col:
        pbp[pbp_game_id_col] = pd.to_numeric(pbp[pbp_game_id_col], errors='coerce')
    if df_game_id_col:
        df[df_game_id_col] = pd.to_numeric(df[df_game_id_col], errors='coerce')

    for col in ['possessions', 'pts', 'raw_ppp', 'garbage_time_flag']:
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors='coerce')

    group_keys = ['player', 'game_date']
    if pbp_player_id_col and df_player_id_col and pbp_game_id_col and df_game_id_col:
        group_keys = [pbp_player_id_col, pbp_game_id_col]
    elif pbp_player_id_col and df_player_id_col:
        group_keys = [pbp_player_id_col, 'game_date']
    elif 'player' in pbp.columns and 'player' in df.columns and pbp_game_id_col and df_game_id_col:
        group_keys = ['player', pbp_game_id_col]

    agg_rows = []
    for keys, grp in pbp.groupby(group_keys, sort=False):
        total_poss = grp['possessions'].fillna(0).sum()
        total_pts = grp['pts'].fillna(0).sum() if 'pts' in grp.columns else np.nan
        if 'garbage_time_flag' in grp.columns:
            comp_mask = grp['garbage_time_flag'].fillna(0) == 0
            comp_poss = grp.loc[comp_mask, 'possessions'].fillna(0).sum()
            garbage_flag = int(grp['garbage_time_flag'].fillna(0).max())
        else:
            comp_poss = total_poss
            garbage_flag = 0

        row = {
            'possessions': round(float(total_poss), 2),
            'raw_ppp': round(float(total_pts / total_poss), 4) if pd.notna(total_pts) and total_poss > 0 else np.nan,
            'garbage_time_flag': garbage_flag,
            'competitive_poss': round(float(comp_poss), 2),
        }
        if 'game_date' in group_keys:
            row['game_date'] = grp['game_date'].iloc[0]
        if pbp_game_id_col in group_keys:
            row[pbp_game_id_col] = grp[pbp_game_id_col].iloc[0]
        if group_keys[0] == pbp_player_id_col:
            row[pbp_player_id_col] = grp[pbp_player_id_col].iloc[0]
        else:
            row['player'] = str(grp['player'].iloc[0])
        agg_rows.append(row)

    if not agg_rows:
        return df

    pbp_agg = pd.DataFrame(agg_rows)

    left_keys = []
    right_keys = []
    for key in group_keys:
        if key == pbp_player_id_col and df_player_id_col:
            left_keys.append(df_player_id_col)
            right_keys.append(pbp_player_id_col)
        elif key == pbp_game_id_col and df_game_id_col:
            left_keys.append(df_game_id_col)
            right_keys.append(pbp_game_id_col)
        else:
            left_keys.append(key)
            right_keys.append(key)

    ctx_cols = ['possessions', 'raw_ppp', 'garbage_time_flag', 'competitive_poss']
    merged = df.merge(
        pbp_agg[right_keys + ctx_cols].rename(columns={rk: lk for lk, rk in zip(left_keys, right_keys)}),
        on=left_keys,
        how='left',
        suffixes=('', '_pbp_ctx'),
    )
    for col in ctx_cols:
        ctx_col = f'{col}_pbp_ctx'
        if ctx_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[ctx_col].combine_first(merged[col])
            else:
                merged[col] = merged[ctx_col]
            merged = merged.drop(columns=[ctx_col])
    return merged


def _build_dynamic_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leak-safe rolling context features from per-game PBP layer fields."""
    required = [c for c in ['possessions', 'raw_ppp', 'competitive_poss', 'garbage_time_flag'] if c in df.columns]
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


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns for modeling."""
    exclude = [
        # Identifiers - CRITICAL: these are NOT predictive
        'player', 'game_date', 'team', 'opp', 'matchup', 'result', 'season',
        # Current-game derived features (leakage: computed from same game being predicted)
        'usage_proxy', 'usage_rate',
        'GAME_ID', 'Game_ID', 'game_id',
        'PLAYER_ID', 'Player_ID', 'player_id',
        'TEAM_ID', 'team_id',
        'SEASON_ID', 'Season_ID', 'season_id',
        'VIDEO_AVAILABLE', 'matchup_raw',
        'SEASON_YEAR', 'NICKNAME', 'TEAM_NAME', 'MIN_SEC', 'TEAM_COUNT',
        # Target columns (actual stats from the game)
        'pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'fg', 'fga', '3p', '3pa',
        'ft', 'fta', 'orb', 'drb', 'pf', 'plus_minus', 'mp',
        'fg_pct', '3p_pct', 'ft_pct',
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG3M', 'FG3A',
        'FTM', 'FTA', 'OREB', 'DREB', 'PF', 'PLUS_MINUS', 'MIN',
        'FG_PCT', 'FG3_PCT', 'FT_PCT',
        # Derived prediction columns
        'pts_pred', 'trb_pred', 'ast_pred', 'stl_pred', 'blk_pred', 'tov_pred',
        'pra_pred', 'pr_pred', 'pa_pred',
        'pts_conf', 'trb_conf', 'ast_conf',
        # NBA API specific columns
        'NBA_FANTASY_PTS', 'DD2', 'TD3', 'WNBA_FANTASY_PTS', 'AVAILABLE_FLAG',
        # Imported rolling averages from uploaded data - use our computed versions
        'pts_l20_imported', 'trb_l20_imported', 'ast_l20_imported',
        'stl_l20_imported', 'blk_l20_imported', 'tov_l20_imported', 'mp_l20_imported',
        'FG3M_ROLL_20', 'FGA_ROLL_20', 'FGM_ROLL_20', 'FTA_ROLL_20', 'FTM_ROLL_20',
        'FG_PCT_ROLL_20', 'FT_PCT_ROLL_20',
    ]
    
    # Also exclude any column ending with _RANK (NBA API rank columns)
    # Also exclude columns that look like identifiers
    feature_cols = []
    for c in df.columns:
        c_lower = c.lower()
        if c in exclude:
            continue
        if c.endswith('_RANK'):
            continue
        # Exclude anything with 'id' in the name (identifiers)
        if '_id' in c_lower or c_lower.endswith('id'):
            continue
        if df[c].dtype not in ['float64', 'int64', 'float32', 'int32']:
            continue
        feature_cols.append(c)
    
    return feature_cols


def _train_lgbm(X_train, y_train, X_val, y_val):
    """Train a LightGBM model with early stopping. Returns None if not installed."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    m = lgb.LGBMRegressor(
        objective='regression_l1', num_leaves=63, learning_rate=0.05,
        n_estimators=500, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=30, random_state=42, n_jobs=1, verbose=-1,
    )
    m.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return m


def train_models(df: pd.DataFrame) -> Dict:
    """Train XGBoost models for all targets."""
    try:
        import xgboost as xgb
    except ImportError:
        print("Installing xgboost...")
        os.system('pip install xgboost --break-system-packages -q')
        import xgboost as xgb
    
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
    from sklearn.linear_model import Ridge
    
    print("\nTraining models...")

    # Filter to players with enough games
    df = df[df['games_played'] >= CONFIG['min_games']].copy()

    # Sort by date
    df = df.sort_values('game_date').reset_index(drop=True)

    # ── Holdout split ─────────────────────────────────────────────────────────
    # Hold back the last 30 days as a HARD test set — the model never sees this
    # data during training or hyperparameter search. Reported MAE/Acc come ONLY
    # from this holdout so the numbers are honest out-of-sample estimates.
    # After validation, we retrain on ALL data for the production model.
    max_date = df['game_date'].max()
    holdout_cutoff = max_date - pd.Timedelta(days=30)
    train_mask = df['game_date'] < holdout_cutoff
    test_mask  = df['game_date'] >= holdout_cutoff

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()

    n_train = train_mask.sum()
    n_test  = test_mask.sum()
    print(f"  Train: {n_train:,} rows (up to {holdout_cutoff.date()})")
    print(f"  Holdout test: {n_test:,} rows ({holdout_cutoff.date()} → {max_date.date()})")

    # Merge enriched layer features (blowout, usage, luck) before feature selection
    df       = _merge_layer_features(df)
    df_train = _merge_layer_features(df_train)
    df_test  = _merge_layer_features(df_test)

    # Get features
    feature_cols = get_feature_columns(df)
    print(f"  Using {len(feature_cols)} features")

    # Target configurations: name → stat_col
    targets = {
        'pts': 'pts',
        'trb': 'trb',
        'ast': 'ast',
        'stl': 'stl',
        'blk': 'blk',
        'tov': 'tov',
    }

    models = {}
    results = {}
    biases = {}  # per-stat holdout bias: mean(pred - actual)

    # XGBoost parameters
    params = {
        'objective': 'reg:pseudohubererror',  # robust to outlier games vs squarederror
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 500,       # upper bound — actual count set by CV early stopping
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': 1,
        'verbosity': 0,
    }

    for target_name, target_col in targets.items():
        print(f"\n  Training {target_name.upper()}...")

        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[target_col]
        X_test  = df_test[feature_cols].fillna(0)
        y_test  = df_test[target_col]

        # Time series CV within training set only — purely for finding opt n_estimators
        tscv = TimeSeriesSplit(n_splits=5)
        best_iterations = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            Xf_tr, Xf_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            yf_tr, yf_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            xgb_cv = xgb.XGBRegressor(**params, early_stopping_rounds=25)
            xgb_cv.fit(Xf_tr, yf_tr, eval_set=[(Xf_val, yf_val)], verbose=False)
            best_iterations.append(xgb_cv.best_iteration)

        opt_n = max(50, int(np.mean(best_iterations) * 1.1))

        # Validation model: trained on df_train, evaluated on holdout
        val_model = xgb.XGBRegressor(**{**params, 'n_estimators': opt_n})
        val_model.fit(X_train, y_train, verbose=False)
        holdout_preds = val_model.predict(X_test)

        holdout_mae = mean_absolute_error(y_test, holdout_preds)
        # Per-player threshold: compare each holdout game against that player's
        # own training median, not the global median.  This matters most for
        # skewed rare-event stats (BLK, STL) where the global median is near 0
        # and produces misleading accuracy numbers.
        player_train_medians = df_train.groupby('player')[target_col].median()
        test_thresholds = df_test['player'].map(player_train_medians).fillna(float(y_train.median()))
        holdout_acc = float(
            ((holdout_preds > test_thresholds.values) == (y_test.values > test_thresholds.values)).mean()
        )
        holdout_bias = float(np.mean(holdout_preds - y_test.values))
        biases[target_name] = round(holdout_bias, 4)

        print(f"    Holdout MAE: {holdout_mae:.2f} | Acc: {holdout_acc:.1%} | "
              f"n_est: {opt_n} | bias: {holdout_bias:+.3f} | "
              f"n_test={len(y_test):,}")

        # Production model: retrain on ALL data with the same opt_n
        # (more data = better predictions; holdout was only for honest evaluation)
        final_model = xgb.XGBRegressor(**{**params, 'n_estimators': opt_n})
        final_model.fit(df[feature_cols].fillna(0), df[target_col], verbose=False)

        lgb_model = None

        models[target_name] = {
            'xgb': final_model, 'lgb': lgb_model,
            'xgb_w': 1.0, 'lgb_w': 0.0,
        }
        results[target_name] = {'mae': holdout_mae, 'accuracy': holdout_acc}

    # ── Minutes deviation model ───────────────────────────────────────────────
    # Predict how many minutes a player will play relative to their L10 average.
    # Target: mp - mp_l10  (positive = more minutes than usual, negative = fewer)
    # This is saved separately and loaded at inference to scale stat predictions.
    print("\n  Training MP (minutes deviation)...")
    try:
        df['mp'] = pd.to_numeric(df['mp'], errors='coerce')
        mp_mask = df['mp_l10'].notna() & df['mp'].notna() & (df['mp_l10'] > 3)
        df_mp = df[mp_mask].copy()
        df_mp['mp_deviation'] = df_mp['mp'] - df_mp['mp_l10']

        # Use minutes-relevant features — starter role, fouls, blowout, schedule
        mp_feature_candidates = [
            c for c in feature_cols if any(k in c for k in [
                'mp_', 'rest', 'is_home', 'days_', 'games_played',
                'back_to_back', 'schedule', 'opp_pace', 'opp_pts',
                'home_away', 'trend', 'ewma', 'streak',
                # New: starter proxy
                'starter', 'rotation', 'bench', 'consistency',
                # New: foul trouble
                'pf_', 'foul_risk', 'pf_per_min',
                # New: blowout/margin
                'blowout', 'margin', 'mp_margin',
                # Star/role identifiers
                'is_star', 'is_starter', 'is_rotation',
            ])
        ]
        # Fall back to all features if too few mp-specific ones found
        mp_feat_cols = mp_feature_candidates if len(mp_feature_candidates) >= 10 else feature_cols

        X_mp = df_mp[mp_feat_cols].fillna(0)
        y_mp = df_mp['mp_deviation'].clip(-15, 15)  # cap extreme outliers

        tscv_mp = TimeSeriesSplit(n_splits=5)
        mp_iters = []
        mp_oof = np.zeros(len(X_mp))
        for fold, (tr, va) in enumerate(tscv_mp.split(X_mp)):
            m = xgb.XGBRegressor(**{**params, 'n_estimators': 300}, early_stopping_rounds=20)
            m.fit(X_mp.iloc[tr], y_mp.iloc[tr],
                  eval_set=[(X_mp.iloc[va], y_mp.iloc[va])], verbose=False)
            mp_oof[va] = m.predict(X_mp.iloc[va])
            mp_iters.append(m.best_iteration)

        mp_opt_n = max(30, int(np.mean(mp_iters) * 1.1))
        mp_mae   = mean_absolute_error(y_mp, mp_oof)
        mp_final = xgb.XGBRegressor(**{**params, 'n_estimators': mp_opt_n})
        mp_final.fit(X_mp, y_mp, verbose=False)

        mp_model_path = CONFIG['models_dir'] / 'minutes_deviation_model.pkl'
        with open(mp_model_path, 'wb') as f:
            pickle.dump({'model': mp_final, 'feature_cols': mp_feat_cols}, f)
        print(f"    MAE: {mp_mae:.2f} min  |  n_est: {mp_opt_n}  |  saved -> {mp_model_path}")
    except Exception as e:
        print(f"    Minutes model training failed (non-fatal): {e}")

    # Save models
    models_path = CONFIG['models_dir'] / 'models.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'models': models,
            'feature_cols': feature_cols,
            'residual': False,
        }, f)

    # Save per-stat prediction biases (used at inference to de-bias predictions)
    bias_path = CONFIG['models_dir'] / 'bias.json'
    with open(bias_path, 'w') as f:
        json.dump(biases, f, indent=2)
    print(f"\n  Bias calibration saved: { {k: f'{v:+.3f}' for k, v in biases.items()} }")

    print(f"  Models saved to {models_path}")
    gc.collect()

    # Save results
    results_path = CONFIG['models_dir'] / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return models


def train_ou_classifiers(df: pd.DataFrame) -> bool:
    """Train binary OVER/UNDER classifiers using real Vegas lines as training signal."""
    try:
        import xgboost as xgb
    except ImportError:
        print("Installing xgboost...")
        os.system('pip install xgboost -q')
        import xgboost as xgb

    lines_path = CONFIG['data_dir'] / 'historical_lines.csv'
    if not lines_path.exists():
        print("  No historical_lines.csv — skipping O/U classifier")
        return False

    hist = pd.read_csv(lines_path)
    hist['game_date'] = pd.to_datetime(hist['game_date'])
    if 'player_norm' not in hist.columns:
        hist['player_norm'] = hist['player'].apply(_normalize_name)

    _prop_map = {
        'player_points': 'pts', 'player_rebounds': 'trb', 'player_assists': 'ast',
        'player_steals': 'stl', 'player_blocks': 'blk', 'player_turnovers': 'tov',
    }
    hist['stat'] = hist['prop'].map(_prop_map)
    hist = hist.dropna(subset=['stat'])

    feat_df = df.copy()
    feat_df['player_norm'] = feat_df['player'].apply(_normalize_name)

    models_path = CONFIG['models_dir'] / 'models.pkl'
    with open(models_path, 'rb') as f:
        saved = pickle.load(f)
    base_features = saved['feature_cols']

    from sklearn.model_selection import TimeSeriesSplit
    ou_models = {}
    for prop_key, stat_name in _prop_map.items():
        if stat_name not in feat_df.columns:
            continue
        stat_hist = hist[hist['stat'] == stat_name][['game_date', 'player_norm', 'line']]
        merged = feat_df.merge(stat_hist, on=['game_date', 'player_norm'], how='inner')
        if len(merged) < 200:
            print(f"  {stat_name.upper()}: only {len(merged)} samples — skipping O/U classifier")
            continue

        merged = merged.sort_values('game_date')
        y = (merged[stat_name] > merged['line']).astype(int)
        feature_set = base_features + ['line']
        X = merged[[c for c in feature_set if c in merged.columns]].fillna(0)
        actual_features = X.columns.tolist()

        tscv = TimeSeriesSplit(n_splits=5)
        cv_iters = []
        for tr_idx, val_idx in tscv.split(X):
            clf = xgb.XGBClassifier(
                objective='binary:logistic', max_depth=4, learning_rate=0.05,
                n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_lambda=1.5, random_state=42,
                eval_metric='logloss', early_stopping_rounds=20, n_jobs=1
            )
            clf.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                    eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=False)
            cv_iters.append(clf.best_iteration)

        opt_n = max(30, int(np.mean(cv_iters) * 1.1))
        final_clf = xgb.XGBClassifier(
            objective='binary:logistic', max_depth=4, learning_rate=0.05,
            n_estimators=opt_n, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_lambda=1.5, random_state=42, n_jobs=1
        )
        # Train on first 70%, calibrate on last 30% (time-ordered — no leakage)
        cal_split = int(len(X) * 0.70)
        final_clf.fit(X.iloc[:cal_split], y.iloc[:cal_split])
        # Isotonic calibration: maps raw probabilities to better-calibrated outputs
        from sklearn.isotonic import IsotonicRegression
        raw_cal_probs = final_clf.predict_proba(X.iloc[cal_split:])[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_cal_probs, y.iloc[cal_split:].values)
        cal_probs_adj = iso.transform(raw_cal_probs)
        cal_acc = ((cal_probs_adj > 0.5) == y.iloc[cal_split:].values).mean()
        ou_models[stat_name] = {'clf': final_clf, 'iso': iso, 'features': actual_features}
        print(f"  {stat_name.upper()} O/U: {len(y)} samples, cal acc {cal_acc:.1%}, n_est={opt_n}")

    if not ou_models:
        return False

    path = CONFIG['models_dir'] / 'ou_classifiers.pkl'
    with open(path, 'wb') as f:
        pickle.dump(ou_models, f)
    print(f"  O/U classifiers saved ({len(ou_models)} stats)")
    gc.collect()
    return True


def predict_ou_proba(pred_df, feat_df):
    """Add P(OVER) probabilities from binary classifiers to pred_df."""
    path = CONFIG['models_dir'] / 'ou_classifiers.pkl'
    if not path.exists():
        return pred_df
    with open(path, 'rb') as f:
        ou_models = pickle.load(f)

    feat_df = feat_df.copy()
    feat_df['player_norm'] = feat_df['player'].apply(_normalize_name)
    latest = feat_df.sort_values('game_date').groupby('player').last().reset_index()

    pred_df = pred_df.copy()
    for stat_name, m in ou_models.items():
        clf     = m['clf']
        iso     = m.get('iso')          # isotonic calibrator (may be absent in old models)
        features = m['features']
        line_col = f'{stat_name}_line'
        prob_col = f'{stat_name}_ou_prob'
        if line_col not in pred_df.columns:
            continue
        non_line_features = [f for f in features if f != 'line' and f in latest.columns]
        # Preserve pred_df row order by using index-safe merge
        pred_subset = pred_df[['player', line_col]].reset_index(drop=True)
        rows = pred_subset.merge(latest[['player'] + non_line_features], on='player', how='left')
        rows = rows.rename(columns={line_col: 'line'})
        X = rows[[c for c in features if c in rows.columns]].fillna(0)
        if len(X) == 0 or len(X) != len(pred_df):
            continue
        raw_probs = clf.predict_proba(X)[:, 1]
        # Apply isotonic calibration if available
        pred_df[prob_col] = iso.transform(raw_probs) if iso is not None else raw_probs

    return pred_df


# =============================================================================
# PREDICTION
# =============================================================================

def _predict(model_obj, X, fp_input=None):
    """Get prediction from either an ensemble dict or a plain model."""
    if isinstance(model_obj, dict):
        if model_obj.get('kind') in {'ridge_blend_regressor', 'positive_ridge_blend_regressor'}:
            base_preds = []
            for base_name in model_obj.get('base_order', []):
                if base_name == 'fp':
                    if fp_input is None:
                        base_preds.append(np.zeros(len(X), dtype=float))
                    else:
                        if isinstance(fp_input, pd.Series):
                            fp_values = pd.to_numeric(fp_input, errors='coerce').to_numpy(dtype=float, copy=False)
                        else:
                            fp_values = pd.to_numeric(
                                pd.Series(np.asarray(fp_input).reshape(-1)),
                                errors='coerce',
                            ).to_numpy(dtype=float, copy=False)
                        fp_values = np.nan_to_num(fp_values, nan=0.0, posinf=0.0, neginf=0.0)
                        if fp_values.size == 1 and len(X) != 1:
                            fp_values = np.full(len(X), float(fp_values[0]), dtype=float)
                        elif fp_values.size != len(X):
                            raise ValueError(
                                f"FP input length {fp_values.size} does not match feature rows {len(X)}"
                            )
                        base_preds.append(fp_values.astype(float, copy=False))
                    continue
                base_model = model_obj['base_models'][base_name]
                base_preds.append(np.asarray(base_model.predict(X), dtype=float))
            blend_X = np.column_stack(base_preds)
            return np.asarray(model_obj['blender'].predict(blend_X), dtype=float)
        p = model_obj['xgb'].predict(X) * model_obj['xgb_w']
        if model_obj['lgb'] is not None:
            p = p + model_obj['lgb'].predict(X) * model_obj['lgb_w']
        return p
    return model_obj.predict(X)


def _file_signature(path: Path) -> Dict[str, object]:
    """Return lightweight cache invalidation metadata for a file path."""
    if not path.exists():
        return {'path': str(path), 'exists': False}
    stat = path.stat()
    return {
        'path': str(path),
        'exists': True,
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def _prediction_feature_cache_paths(using_advanced: bool) -> Tuple[Path, Path]:
    suffix = 'advanced' if using_advanced else 'standard'
    cache_dir = CONFIG['data_dir'] / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f'prediction_feature_frame_{suffix}.pkl',
        cache_dir / f'prediction_feature_frame_{suffix}.json',
    )


def _prediction_feature_cache_signature(using_advanced: bool) -> Dict[str, object]:
    paths = [
        CONFIG['data_dir'] / 'nba_data.csv',
        CONFIG['data_dir'] / 'pbp_possessions_master.csv',
        CONFIG['data_dir'] / 'training_dataset_enriched.csv',
        CONFIG['data_dir'] / 'player_positions.csv',
        Path(__file__),
    ]
    if using_advanced:
        paths.extend([
            CONFIG['models_dir'] / 'selected_features_advanced.json',
            CONFIG['models_dir'] / 'bias_advanced.json',
            CONFIG['models_dir'] / 'minutes_deviation_model.pkl',
        ])
    return {
        'using_advanced': using_advanced,
        'sources': [_file_signature(path) for path in paths],
    }


def _prepare_latest_prediction_frame(
    df: pd.DataFrame,
    using_advanced: bool,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build or load the latest per-player feature frame used at inference."""
    cache_pkl, cache_meta = _prediction_feature_cache_paths(using_advanced)
    signature = _prediction_feature_cache_signature(using_advanced)

    if use_cache and cache_pkl.exists() and cache_meta.exists():
        try:
            with open(cache_meta) as f:
                cached_meta = json.load(f)
            if cached_meta == signature:
                latest_cached = pd.read_pickle(cache_pkl)
                print(f"  Using cached prediction feature frame: {cache_pkl.name}")
                return latest_cached
        except Exception:
            pass

    base_feature_cols = ['games_played', 'pts_l10', 'trb_l10', 'ast_l10', 'mp_l10']
    working = df.copy()
    if any(col not in working.columns for col in base_feature_cols):
        print("  Building base rolling features...")
        working = create_features(working)

    working = _merge_current_pbp_context(working)
    working = _build_dynamic_context_features(working)

    if using_advanced:
        print("  Building advanced features (rolling splits, defense-adj, best_estimate)...")
        working = _build_advanced_features(working)

    working = working.sort_values('game_date')
    latest_group_col = 'player_id' if 'player_id' in working.columns else ('PLAYER_ID' if 'PLAYER_ID' in working.columns else 'player')
    latest = working.groupby(latest_group_col).last().reset_index()
    latest = latest[latest['games_played'] >= CONFIG['min_games']]
    latest = _merge_layer_features(latest)

    try:
        latest.to_pickle(cache_pkl)
        with open(cache_meta, 'w') as f:
            json.dump(signature, f, indent=2)
        print(f"  Cached prediction feature frame: {cache_pkl.name}")
    except Exception as e:
        print(f"  WARNING: Could not save prediction feature cache ({e})")

    return latest


def generate_predictions(
    df: pd.DataFrame,
    vegas_lines: pd.DataFrame = None,
    use_feature_cache: bool = False,
    apply_bias_correction: bool = True,
    apply_injury_boosts: bool = True,
    apply_game_totals: bool = True,
    apply_fp_blend: bool = True,
) -> pd.DataFrame:
    """Generate predictions for upcoming games using minutes-adjusted approach."""

    # Try advanced models first (trained by train_advanced_models.py)
    adv_models, adv_feature_cols, adv_classifiers, adv_selected, adv_edge_models, adv_meta_models, adv_market_sets = _load_advanced_models()
    using_advanced = adv_models is not None

    # Load standard models (always needed as fallback)
    models_path = CONFIG['models_dir'] / 'models.pkl'
    if not models_path.exists() and not using_advanced:
        raise FileNotFoundError("No trained models found. Run 'train' first.")

    models, feature_cols, is_residual = {}, [], False
    if models_path.exists():
        with open(models_path, 'rb') as f:
            saved = pickle.load(f)
        models = saved['models']
        feature_cols = saved['feature_cols']
        is_residual = saved.get('residual', False)

    if using_advanced:
        models = adv_models
        feature_cols = adv_feature_cols
        is_residual = False
        clf_models   = adv_classifiers   # stat -> XGBClassifier P(OVER)
        clf_selected = adv_selected       # stat -> list of selected feature names
        probability_sidecars = _load_probability_sidecars()
        probability_calibrators = _load_probability_calibrators()
    else:
        clf_models   = {}
        clf_selected = {}
        adv_edge_models = {}
        adv_meta_models = {}
        adv_market_sets = {'edge': {}, 'meta': {}}
        probability_sidecars = {}
        probability_calibrators = {}

    # Load prediction bias calibration (computed from OOF residuals during training)
    pred_biases = {}
    bias_path = CONFIG['models_dir'] / ('bias_advanced.json' if using_advanced else 'bias.json')
    if bias_path.exists():
        with open(bias_path) as f:
            pred_biases = json.load(f)
    elif using_advanced:
        print("  WARNING: No bias_advanced.json found; skipping advanced bias correction")

    # Try to load minutes deviation model
    minutes_model = None
    minutes_feature_cols = None
    mp_model_path = CONFIG['models_dir'] / 'minutes_deviation_model.pkl'
    if mp_model_path.exists():
        with open(mp_model_path, 'rb') as f:
            mp_saved = pickle.load(f)
        minutes_model = mp_saved['model']
        minutes_feature_cols = mp_saved['feature_cols']
        print("  Using minutes deviation model for adjustments")

    print("\nGenerating predictions...")
    latest = _prepare_latest_prediction_frame(df, using_advanced, use_cache=use_feature_cache)
    primitive_fp_integrated = using_advanced and any(
        isinstance(models.get(stat), dict) and models.get(stat, {}).get('kind') == 'positive_ridge_blend_regressor'
        for stat in ['pts', 'trb', 'ast']
    )
    live_fp_lookup = _load_live_fp_projection_lookup() if primitive_fp_integrated else None

    print(f"  {len(latest)} players with sufficient history")

    # Warn about data staleness
    max_date = pd.to_datetime(latest['game_date'], errors='coerce').max()
    days_old = (pd.Timestamp.now() - max_date).days
    if days_old > 7:
        print(f"  WARNING: Data is {days_old} days old (last game: {max_date.strftime('%Y-%m-%d')})")
    
    required_feature_cols = list(feature_cols) if feature_cols else []
    if using_advanced:
        for target_name, model in models.items():
            target_features = clf_selected.get(target_name)
            if not target_features and isinstance(model, dict):
                target_features = model.get('feature_names')
            if not target_features and hasattr(model, 'get_booster'):
                target_features = model.get_booster().feature_names
            if target_features:
                required_feature_cols.extend(target_features)
    if minutes_feature_cols:
        required_feature_cols.extend(minutes_feature_cols)
    required_feature_cols = list(dict.fromkeys(required_feature_cols))

    # Check if feature columns exist in latest
    missing_cols = [c for c in required_feature_cols if c not in latest.columns]
    if missing_cols:
        print(f"  WARNING: {len(missing_cols)} feature columns not found, filling with 0")
        for c in missing_cols:
            latest[c] = 0

    # Prepare features
    X = latest[required_feature_cols].fillna(0) if required_feature_cols else latest.copy()
    
    # Calculate per-minute rates for each player (for minutes-based adjustment)
    for stat in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov']:
        l10_col = f'{stat}_l10'
        if l10_col in latest.columns and 'mp_l10' in latest.columns:
            latest[f'{stat}_per_min'] = latest[l10_col] / latest['mp_l10'].clip(lower=1)
    
    # Generate predictions
    predictions = []
    
    for idx, row in latest.iterrows():
        player = row['player']
        team = row.get('team', row.get('TEAM_ABBREVIATION', ''))
        player_fp_applied = False
        
        player_preds = {
            'player': player,
            'team': team,
            'games_played': row['games_played'],
        }
        
        # Get minutes deviation prediction if model available
        mp_deviation_pred = 0
        mp_l10 = row.get('mp_l10', 0)
        
        if minutes_model is not None and minutes_feature_cols is not None:
            # Check if we have the minutes features
            mp_features_available = all(c in row.index for c in minutes_feature_cols)
            if mp_features_available:
                mp_X = pd.DataFrame([row[minutes_feature_cols].fillna(0)])
                mp_deviation_pred = minutes_model.predict(mp_X)[0]
                
                # Clip to reasonable bounds
                mp_deviation_pred = max(-8, min(8, mp_deviation_pred))
        
        player_preds['mp_l10'] = round(mp_l10, 1) if pd.notna(mp_l10) else None
        player_preds['mp_deviation_pred'] = round(mp_deviation_pred, 1)
        player_preds['mp_expected'] = round(mp_l10 + mp_deviation_pred, 1) if pd.notna(mp_l10) else None
        
        # Get features for this player
        X_player = X.loc[[idx]]

        for target_name, model in models.items():
            # Get L10 average
            l10_col = f'{target_name}_l10'
            l10_val = row.get(l10_col) if l10_col in row.index else None
            if pd.isna(l10_val):
                l10_val = None
            anchor_val = _row_prop_feature(row, target_name, 'best_estimate')
            if anchor_val is None:
                anchor_val = _row_prop_feature(row, target_name, 'recency_anchor')
            if anchor_val is None:
                anchor_val = l10_val

            player_preds[f'{target_name}_l10'] = round(l10_val, 1) if l10_val else None
            player_preds[f'{target_name}_anchor'] = round(anchor_val, 1) if anchor_val is not None else None

            # Store L5 as well (needed for L5 confirmation filter in filter_best_picks)
            l5_col = f'{target_name}_l5'
            l5_val = row.get(l5_col) if l5_col in row.index else None
            if pd.isna(l5_val): l5_val = None
            player_preds[f'{target_name}_l5'] = round(l5_val, 1) if l5_val else None

            reg_features = None
            if using_advanced:
                reg_features = clf_selected.get(target_name)
                if not reg_features and isinstance(model, dict):
                    reg_features = model.get('feature_names')
                if not reg_features and hasattr(model, 'get_booster'):
                    reg_features = model.get_booster().feature_names
            else:
                reg_features = feature_cols

            reg_features = [c for c in (reg_features or []) if c in X_player.columns]
            X_reg = X_player[reg_features].fillna(0) if reg_features else X_player
            fp_input = None
            if primitive_fp_integrated and target_name in {'pts', 'trb', 'ast'} and isinstance(model, dict) and model.get('kind') == 'positive_ridge_blend_regressor':
                fp_input = np.array([_row_fp_projection(row, target_name, live_fp_lookup)])
            raw_pred = _predict(model, X_reg, fp_input=fp_input)[0]

            if is_residual:
                base_pred = (anchor_val or 0) + raw_pred
            else:
                if primitive_fp_integrated and target_name in {'pts', 'trb', 'ast'} and isinstance(model, dict) and model.get('kind') == 'positive_ridge_blend_regressor':
                    base_pred = raw_pred
                    player_fp_applied = True
                elif anchor_val is not None:
                    base_pred = raw_pred * 0.7 + anchor_val * 0.3
                else:
                    base_pred = raw_pred

            # Minutes-based scaling adjustment
            if mp_l10 > 0 and mp_deviation_pred != 0:
                per_min_rate = row.get(f'{target_name}_per_min', 0)
                if pd.notna(per_min_rate) and per_min_rate > 0:
                    minutes_adj = per_min_rate * mp_deviation_pred
                    # Cap the adjustment at 15% of baseline
                    minutes_adj = max(-0.15 * base_pred, min(0.15 * base_pred, minutes_adj))
                    base_pred += minutes_adj

            # Subtract OOF bias to de-bias predictions (positive bias = model predicts too high)
            bias_correction = pred_biases.get(target_name, 0.0) if apply_bias_correction else 0.0
            final_pred = max(0, base_pred - bias_correction)

            player_preds[f'{target_name}_pred'] = round(final_pred, 1)

            # ── Classifier: P(OVER) from binary clf trained vs actual line ────
            clf = clf_models.get(target_name)
            if clf is not None:
                try:
                    sel = clf_selected.get(target_name, reg_features or feature_cols)
                    avail = [c for c in sel if c in X_player.columns]
                    X_clf = X_player[avail].fillna(0)
                    p_over = float(clf.predict_proba(X_clf)[0, 1])
                    player_preds[f'{target_name}_clf_prob'] = round(p_over, 4)
                except Exception:
                    pass

        # Combo predictions
        pts = player_preds.get('pts_pred', 0)
        trb = player_preds.get('trb_pred', 0)
        ast = player_preds.get('ast_pred', 0)
        
        player_preds['pra_pred'] = round(pts + trb + ast, 1)
        player_preds['pr_pred'] = round(pts + trb, 1)
        player_preds['pa_pred'] = round(pts + ast, 1)
        
        # Combo l10 averages
        pts_l10 = player_preds.get('pts_l10') or 0
        trb_l10 = player_preds.get('trb_l10') or 0
        ast_l10 = player_preds.get('ast_l10') or 0
        
        if pts_l10 and trb_l10 and ast_l10:
            player_preds['pra_l10'] = round(pts_l10 + trb_l10 + ast_l10, 1)
            player_preds['pr_l10'] = round(pts_l10 + trb_l10, 1)
            player_preds['pa_l10'] = round(pts_l10 + ast_l10, 1)

        # Composite L5s
        pts_l5 = player_preds.get('pts_l5') or 0
        trb_l5 = player_preds.get('trb_l5') or 0
        ast_l5 = player_preds.get('ast_l5') or 0
        if pts_l5 and trb_l5 and ast_l5:
            player_preds['pra_l5'] = round(pts_l5 + trb_l5 + ast_l5, 1)
            player_preds['pr_l5'] = round(pts_l5 + trb_l5, 1)
            player_preds['pa_l5'] = round(pts_l5 + ast_l5, 1)

        player_preds['fp_applied'] = player_fp_applied
        
        predictions.append(player_preds)
    
    pred_df = pd.DataFrame(predictions)

    # Apply injury adjustments (before Vegas merge so combo preds are boosted too)
    if apply_injury_boosts:
        inj = fetch_injury_data()
        pred_df = apply_injury_adjustments(pred_df, inj, latest)
    else:
        pred_df['injury_adj'] = False
        pred_df['teammates_out'] = ''
        pred_df['injury_adj_strength'] = 0.0

    # Apply game total scaling: if game total is above league avg → boost predictions slightly
    if apply_game_totals:
        print("  Fetching game totals...")
        totals_df = fetch_game_totals()
    else:
        totals_df = pd.DataFrame()
    if len(totals_df) > 0:
        league_avg_total = 225.0  # approximate NBA season average
        # Build team → total + spread lookup
        team_totals = {}
        for _, row in totals_df.iterrows():
            scale = 1 + 0.003 * (row['total'] - league_avg_total) if pd.notna(row.get('total')) else 1.0
            team_totals[row['home_team']] = {'scale': scale, 'total': row.get('total'), 'spread': row.get('spread', 0)}
            team_totals[row['away_team']] = {'scale': scale, 'total': row.get('total'), 'spread': -row.get('spread', 0) if pd.notna(row.get('spread')) else 0}
        pred_df['game_total'] = pred_df['team'].map(lambda t: team_totals.get(t, {}).get('total'))
        pred_df['team_spread'] = pred_df['team'].map(lambda t: team_totals.get(t, {}).get('spread', 0))
        pred_df['game_total_scale'] = pred_df['team'].map(lambda t: team_totals.get(t, {}).get('scale', 1.0)).fillna(1.0)
        # Scale pts/ast predictions by game environment (trb/stl/blk less affected by pace)
        for stat in ['pts', 'ast', 'pra', 'pa']:
            pred_col = f'{stat}_pred'
            if pred_col in pred_df.columns:
                scale_vec = pred_df['game_total_scale']
                pred_df[pred_col] = (pred_df[pred_col] * scale_vec).round(1)
        n_games = len(totals_df)
        avg_total = totals_df['total'].mean()
        print(f"  Game totals applied: {n_games} games, avg total {avg_total:.1f}")
    else:
        pred_df['game_total'] = None
        pred_df['team_spread'] = 0
        pred_df['game_total_scale'] = 1.0
        print("  Game totals unavailable — skipping")

    # Merge with Vegas lines if available
    if vegas_lines is not None and len(vegas_lines) > 0:
        pred_df = merge_vegas_lines(pred_df, vegas_lines)
        pred_df = _attach_probability_outputs(
            pred_df,
            latest,
            probability_sidecars,
            probability_calibrators,
        )

    # Line movement: compare current lines vs morning snapshot
    today_str = datetime.now().strftime('%Y-%m-%d')
    morning_path = CONFIG['data_dir'] / f'lines_morning_{today_str}.csv'
    if morning_path.exists() and vegas_lines is not None and len(vegas_lines) > 0:
        morning = pd.read_csv(morning_path)
        morning['player_norm'] = morning.get('player_norm', morning['player'].apply(_normalize_name))
        current = vegas_lines.copy()
        current['player_norm'] = current['player'].apply(_normalize_name)
        _prop_map_rev = {
            'player_points': 'pts', 'player_rebounds': 'trb', 'player_assists': 'ast',
            'player_steals': 'stl', 'player_blocks': 'blk', 'player_turnovers': 'tov',
            'player_points_rebounds_assists': 'pra', 'player_points_rebounds': 'pr',
            'player_points_assists': 'pa',
        }
        current['stat'] = current['prop'].map(_prop_map_rev)
        morning['stat'] = morning['prop'].map(_prop_map_rev)
        merged_lines = current.merge(
            morning[['player_norm', 'stat', 'line']].rename(columns={'line': 'morning_line'}),
            on=['player_norm', 'stat'], how='left'
        )
        merged_lines['line_move'] = merged_lines['line'] - merged_lines['morning_line']
        # Pivot line_move per player/stat and attach to pred_df
        pred_df['player_norm_tmp'] = pred_df['player'].apply(_normalize_name)
        for stat in ['pts', 'trb', 'ast', 'pra', 'pr', 'pa', 'stl', 'blk', 'tov']:
            stat_moves = merged_lines[merged_lines['stat'] == stat][['player_norm', 'line_move']]
            if len(stat_moves) > 0:
                pred_df = pred_df.merge(
                    stat_moves.rename(columns={'line_move': f'{stat}_line_move', 'player_norm': 'player_norm_tmp'}),
                    on='player_norm_tmp', how='left'
                )
        pred_df.drop(columns=['player_norm_tmp'], inplace=True, errors='ignore')
        n_moved = sum(1 for c in pred_df.columns if c.endswith('_line_move') and pred_df[c].abs().gt(0).any())
        if n_moved > 0:
            print(f"  Line movement tracked: {n_moved} props with movement")

    # The advanced path already adds per-target classifier probabilities directly.
    # Keep the legacy O/U classifier file only for the standard-model path.
    if not using_advanced:
        pred_df = predict_ou_proba(pred_df, latest)

    # Blend in First-Principles projections using learned weights when available
    if apply_fp_blend and not primitive_fp_integrated:
        pred_df = apply_first_principles_adjustments(pred_df, latest)
    elif 'fp_applied' not in pred_df.columns:
        pred_df['fp_applied'] = False

    if using_advanced:
        pred_df = _attach_market_model_scores(
            pred_df,
            latest,
            adv_edge_models,
            adv_meta_models,
            adv_market_sets,
        )

    return pred_df


def apply_first_principles_adjustments(pred_df: pd.DataFrame, feature_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Blend player_projections_today.csv into model predictions using adaptive per-player weights."""
    proj_path = _HERE / 'data' / 'player_projections_today.csv'
    if not proj_path.exists():
        print("  [First-Principles] player_projections_today.csv not found -- skipping blend")
        pred_df['fp_applied'] = False
        return pred_df

    fp = pd.read_csv(proj_path)
    fp['_norm'] = fp['player'].apply(_normalize_name)
    pred_df['_norm'] = pred_df['player'].apply(_normalize_name)
    fp_lookup = fp.drop_duplicates(subset='_norm', keep='first').set_index('_norm')
    feature_lookup = None
    if feature_df is not None and 'player' in feature_df.columns:
        feature_working = feature_df.copy()
        feature_working['_norm'] = feature_working['player'].apply(_normalize_name)
        feature_lookup = feature_working.drop_duplicates(subset='_norm', keep='first').set_index('_norm')

    learned_weights = {}
    weights_path = CONFIG['models_dir'] / 'fp_blend_weights_advanced.json'
    if weights_path.exists():
        try:
            with open(weights_path) as f:
                learned_weights = json.load(f)
        except Exception:
            learned_weights = {}

    default_weights = {
        'model_weight': 0.65,
        'fp_weight': 0.35,
        'intercept': 0.0,
    }
    blend_map = {
        'pts': ('proj_pts_final', 'pts_pred'),
        'trb': ('proj_reb', 'trb_pred'),
        'ast': ('proj_ast', 'ast_pred'),
    }
    applied = 0
    learned_prior_stats = 0
    weight_stats = {stat: [] for stat in blend_map}

    for idx, row in pred_df.iterrows():
        key = row['_norm']
        if key not in fp_lookup.index:
            pred_df.at[idx, 'fp_applied'] = False
            continue
        fp_row = fp_lookup.loc[key]
        feature_row = feature_lookup.loc[key] if feature_lookup is not None and key in feature_lookup.index else row

        used_any = False
        for stat, (fp_col, pred_col) in blend_map.items():
            if fp_col not in fp_row or pd.isna(fp_row[fp_col]) or pred_col not in pred_df.columns:
                continue
            orig = pred_df.at[idx, pred_col]
            if pd.isna(orig):
                continue
            weights = learned_weights.get(stat, default_weights)
            model_weight, fp_weight, intercept = _adaptive_fp_blend_weights(
                stat,
                row,
                fp_row,
                feature_row,
                weights,
            )
            blended = (
                model_weight * float(orig) +
                fp_weight * float(fp_row[fp_col]) +
                intercept
            )
            pred_df.at[idx, pred_col] = round(max(0.0, blended), 1)
            used_any = True
            if stat in learned_weights:
                learned_prior_stats += 1
            weight_stats[stat].append(fp_weight)

        pts = pred_df.at[idx, 'pts_pred'] or 0
        trb = pred_df.at[idx, 'trb_pred'] or 0
        ast = pred_df.at[idx, 'ast_pred'] or 0
        pred_df.at[idx, 'pra_pred'] = round(pts + trb + ast, 1)
        pred_df.at[idx, 'pr_pred']  = round(pts + trb, 1)
        pred_df.at[idx, 'pa_pred']  = round(pts + ast, 1)
        pred_df.at[idx, 'fp_applied'] = used_any
        if used_any:
            applied += 1

    pred_df.drop(columns=['_norm'], inplace=True)
    summary_parts = []
    for stat in ['pts', 'trb', 'ast']:
        if weight_stats[stat]:
            summary_parts.append(
                f"{stat.upper()} fp_w {np.mean(weight_stats[stat]):.2f} [{np.min(weight_stats[stat]):.2f},{np.max(weight_stats[stat]):.2f}]"
            )
    if learned_weights:
        print(
            f"  [First-Principles] Blended {applied}/{len(pred_df)} players using adaptive FP weights "
            f"with learned stat priors ({learned_prior_stats} stat-level learned applications)"
        )
    else:
        print(f"  [First-Principles] Blended {applied}/{len(pred_df)} players using adaptive fallback weights")
    if summary_parts:
        print(f"    {' | '.join(summary_parts)}")
    return pred_df


def merge_vegas_lines(pred_df: pd.DataFrame, vegas_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Vegas lines with predictions and calculate edges."""
    
    # Map prop names
    prop_map = {
        'player_points': 'pts',
        'player_rebounds': 'trb',
        'player_assists': 'ast',
        'player_points_rebounds_assists': 'pra',
        'player_points_rebounds': 'pr',
        'player_points_assists': 'pa',
        'player_steals': 'stl',
        'player_blocks': 'blk',
        'player_turnovers': 'tov',
    }
    
    has_proj_type = 'projection_type' in vegas_df.columns
    has_is_promo = 'is_promo' in vegas_df.columns

    # Process each prop type
    for vegas_prop, our_prop in prop_map.items():
        subset_cols = ['player', 'line']
        if has_proj_type:
            subset_cols.append('projection_type')
        if has_is_promo:
            subset_cols.append('is_promo')
        prop_lines = vegas_df[vegas_df['prop'] == vegas_prop][subset_cols].copy()

        if has_proj_type:
            # Preserve projection_type alongside the median line dedup
            pt_map = prop_lines.groupby('player')['projection_type'].first().to_dict()
        if has_is_promo:
            promo_map = prop_lines.groupby('player')['is_promo'].max().to_dict()
        if has_proj_type or has_is_promo:
            prop_lines = prop_lines.groupby('player', as_index=False)['line'].median()
            if has_proj_type:
                prop_lines[f'{our_prop}_projection_type'] = (
                    prop_lines['player'].map(pt_map).fillna('standard').apply(_normalize_projection_type)
                )
            else:
                prop_lines[f'{our_prop}_projection_type'] = 'standard'
            if has_is_promo:
                prop_lines[f'{our_prop}_is_promo'] = prop_lines['player'].map(promo_map).fillna(False).astype(bool)
            else:
                prop_lines[f'{our_prop}_is_promo'] = False
        else:
            prop_lines = prop_lines.groupby('player', as_index=False)['line'].median()
            prop_lines[f'{our_prop}_projection_type'] = 'standard'
            prop_lines[f'{our_prop}_is_promo'] = False

        prop_lines = prop_lines.rename(columns={'line': f'{our_prop}_line'})

        if len(prop_lines) > 0:
            pred_df = pred_df.merge(prop_lines, on='player', how='left')

            # Calculate edge
            pred_col = f'{our_prop}_pred'
            line_col = f'{our_prop}_line'

            if pred_col in pred_df.columns and line_col in pred_df.columns:
                pred_df[f'{our_prop}_edge'] = pred_df[pred_col] - pred_df[line_col]

    return pred_df


def _build_target_quality_policy(results: Optional[Dict[str, Dict]] = None,
                                 edge_analysis: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
    """Build per-target betting policy from a results/edge-analysis snapshot."""
    fallback = {
        'pts': {'allowed': True,  'allow_over': True,  'allow_under': True,  'shrunk_acc': 0.60, 'min_prob': 0.58, 'min_meta_prob': 0.57, 'min_edge': 1.8, 'min_edge_over': 2.0, 'min_edge_under': 2.0, 'rmse': 5.0},
        'trb': {'allowed': False, 'allow_over': False, 'allow_under': False, 'shrunk_acc': 0.54, 'min_prob': 0.62, 'min_meta_prob': 0.59, 'min_edge': 1.0, 'min_edge_over': 1.4, 'min_edge_under': 1.2, 'rmse': 2.5},
        'ast': {'allowed': True,  'allow_over': False, 'allow_under': True,  'shrunk_acc': 0.58, 'min_prob': 0.60, 'min_meta_prob': 0.57, 'min_edge': 0.8, 'min_edge_over': 1.6, 'min_edge_under': 0.9, 'rmse': 1.8},
        'pra': {'allowed': True,  'allow_over': True,  'allow_under': True,  'shrunk_acc': 0.62, 'min_prob': 0.59, 'min_meta_prob': 0.56, 'min_edge': 2.5, 'min_edge_over': 2.5, 'min_edge_under': 2.8, 'rmse': 6.9},
        'pr':  {'allowed': True,  'allow_over': True,  'allow_under': True,  'shrunk_acc': 0.63, 'min_prob': 0.59, 'min_meta_prob': 0.56, 'min_edge': 2.2, 'min_edge_over': 2.3, 'min_edge_under': 2.6, 'rmse': 6.3},
        'pa':  {'allowed': True,  'allow_over': True,  'allow_under': True,  'shrunk_acc': 0.62, 'min_prob': 0.59, 'min_meta_prob': 0.56, 'min_edge': 2.0, 'min_edge_over': 2.3, 'min_edge_under': 2.4, 'rmse': 5.8},
        'stl': {'allowed': False, 'allow_over': False, 'allow_under': False, 'shrunk_acc': 0.57, 'min_prob': 0.62, 'min_meta_prob': 0.60, 'min_edge': 0.5, 'min_edge_over': 0.8, 'min_edge_under': 0.8, 'rmse': 0.9},
        'blk': {'allowed': False, 'allow_over': False, 'allow_under': False, 'shrunk_acc': 0.56, 'min_prob': 0.64, 'min_meta_prob': 0.61, 'min_edge': 0.5, 'min_edge_over': 0.8, 'min_edge_under': 0.8, 'rmse': 0.75},
        'tov': {'allowed': False, 'allow_over': False, 'allow_under': False, 'shrunk_acc': 0.58, 'min_prob': 0.61, 'min_meta_prob': 0.58, 'min_edge': 0.8, 'min_edge_over': 1.2, 'min_edge_under': 1.0, 'rmse': 1.1},
    }

    results = results or {}
    edge_analysis = edge_analysis or {}
    if not isinstance(results, dict):
        return fallback

    policy = {}
    for stat, default in fallback.items():
        res = results.get(stat, {})
        # Support both flat {stat: {...}} and nested {'per_target': {stat: {...}}} formats
        if isinstance(edge_analysis, dict):
            if 'per_target' in edge_analysis:
                edge_res = edge_analysis['per_target'].get(stat, {})
            else:
                edge_res = edge_analysis.get(stat, {})
        else:
            edge_res = {}
        real_acc = res.get('avg_bet_accuracy')
        proxy_acc = res.get('avg_proxy_bet_accuracy')
        market_hit = res.get('market_edge_sign_accuracy')
        meta_summary = res.get('meta_summary') or {}
        rmse = float(res.get('avg_rmse', default['rmse']))
        n_real = int(res.get('n_real_lines') or 0)

        if real_acc is None:
            policy[stat] = default
            continue

        reliability = min(1.0, n_real / 3000.0)
        shrunk_acc = 0.5 + (float(real_acc) - 0.5) * reliability
        divergence = abs(float(real_acc) - float(proxy_acc)) if proxy_acc is not None else 0.0
        market_hit = float(market_hit) if market_hit is not None else None
        meta_auc = float(meta_summary.get('auc')) if meta_summary.get('auc') is not None else None
        meta_top_hit = float(meta_summary.get('top_prob_hit_rate')) if meta_summary.get('top_prob_hit_rate') is not None else None
        clv_corr = edge_res.get('clv_corr')
        clv_corr = float(clv_corr) if clv_corr is not None else 0.0

        bucket_map = {row.get('label'): row for row in edge_res.get('bucket_summary', [])}
        direction_map = {row.get('label'): row for row in edge_res.get('direction_summary', [])}
        under_summary = direction_map.get('under_edges', {})
        over_summary = direction_map.get('over_edges', {})
        strong_summary = direction_map.get('strong_edges_abs_ge_2', {})

        under_hit = float(under_summary.get('hit_rate')) if under_summary.get('hit_rate') is not None else None
        over_hit = float(over_summary.get('hit_rate')) if over_summary.get('hit_rate') is not None else None
        strong_hit = float(strong_summary.get('hit_rate')) if strong_summary.get('hit_rate') is not None else None
        under_n = int(under_summary.get('n') or 0)
        over_n = int(over_summary.get('n') or 0)
        strong_n = int(strong_summary.get('n') or 0)

        under_0_1 = bucket_map.get('UNDER 0-1', {})
        under_1_2 = bucket_map.get('UNDER 1-2', {})
        over_0_1 = bucket_map.get('OVER  0-1', {})
        over_1_2 = bucket_map.get('OVER  1-2', {})
        under_0_1_hit = float(under_0_1.get('hit_rate')) if under_0_1.get('hit_rate') is not None else None
        under_1_2_hit = float(under_1_2.get('hit_rate')) if under_1_2.get('hit_rate') is not None else None
        over_0_1_hit = float(over_0_1.get('hit_rate')) if over_0_1.get('hit_rate') is not None else None
        over_1_2_hit = float(over_1_2.get('hit_rate')) if over_1_2.get('hit_rate') is not None else None

        min_edge = max(0.75, rmse * 0.35)
        if n_real < 2500:
            min_edge += 0.15
        if shrunk_acc < 0.60:
            min_edge += 0.15
        if divergence > 0.08:
            min_edge += 0.20

        min_prob = 0.57
        if shrunk_acc < 0.62:
            min_prob += 0.02
        if shrunk_acc < 0.58:
            min_prob += 0.02
        if n_real < 2500:
            min_prob += 0.01
        if divergence > 0.08:
            min_prob += 0.02
        min_prob = min(0.70, min_prob)

        min_meta_prob = float(default.get('min_meta_prob', 0.57))
        if market_hit is not None and market_hit < 0.56:
            min_meta_prob += 0.02
        if meta_auc is not None and meta_auc < 0.56:
            min_meta_prob += 0.02
        if meta_top_hit is not None and meta_top_hit >= 0.62:
            min_meta_prob -= 0.01
        min_meta_prob = min(0.70, max(0.52, min_meta_prob))

        if meta_auc is not None and meta_auc >= 0.62 and meta_top_hit is not None and meta_top_hit >= 0.54:
            meta_gate_mode = 'hard'
            meta_blend_weight = 0.30
            meta_conf_weight = 0.10
        elif (meta_auc is not None and meta_auc >= 0.56) or (meta_top_hit is not None and meta_top_hit >= 0.53):
            meta_gate_mode = 'medium'
            meta_blend_weight = 0.18
            meta_conf_weight = 0.07
        else:
            meta_gate_mode = 'soft'
            meta_blend_weight = 0.10
            meta_conf_weight = 0.04

        baseline_allowed = (
            n_real >= 1500 and
            shrunk_acc >= 0.56 and
            divergence <= 0.12
        )

        strong_allowed = (
            strong_n >= 250 and
            strong_hit is not None and strong_hit >= 0.63 and
            clv_corr >= 0.18
        )
        under_only_allowed = (
            n_real >= 2500 and
            under_n >= 1000 and
            under_hit is not None and under_hit >= 0.64 and
            shrunk_acc >= 0.58 and
            clv_corr >= 0.14
        )
        # Tracking lane: if the market is clearly only useful on UNDERS,
        # keep it available even when the broader model/proxy relationship is noisy.
        # This avoids the all-or-nothing behavior where every prop is disabled.
        tracking_under_allowed = (
            n_real >= 1500 and
            under_n >= 500 and
            under_hit is not None and under_hit >= 0.61 and
            (
                (market_hit is not None and market_hit >= 0.50) or
                clv_corr >= 0.10
            )
        )
        allowed = (baseline_allowed and (strong_allowed or under_only_allowed)) or tracking_under_allowed

        allow_under = (
            allowed and
            under_n >= 300 and
            under_hit is not None and under_hit >= 0.60
        )
        allow_over = (
            allowed and
            over_n >= 300 and
            over_hit is not None and over_hit >= 0.60 and
            strong_n >= 250
        )

        min_edge_under = min_edge
        if under_0_1_hit is not None and under_0_1_hit < 0.56:
            min_edge_under += 0.25
        if under_1_2_hit is not None and under_1_2_hit < 0.62:
            min_edge_under += 0.15

        min_edge_over = min_edge
        if over_0_1_hit is not None and over_0_1_hit < 0.53:
            min_edge_over += 0.35
        if over_1_2_hit is not None and over_1_2_hit < 0.58:
            min_edge_over += 0.20
        if over_hit is not None and over_hit < 0.62:
            min_edge_over += 0.10

        if not allow_over:
            min_edge_over = max(min_edge_over, min_edge + 0.50)
        if not allow_under:
            min_edge_under = max(min_edge_under, min_edge + 0.30)
        elif tracking_under_allowed and not allow_over:
            min_prob = min(min_prob, 0.585)
            min_meta_prob = min(min_meta_prob, 0.56)
            min_edge_under = min(min_edge_under, max(0.60, rmse * 0.30))

        priority_under_lane = False
        priority_under_min_edge = min_edge_under
        priority_under_support_gap = 0.0
        priority_under_conf_bonus = 0.0
        priority_under_rank_bonus = 0.0
        priority_under_selection_priority = 0.0
        priority_under_require_l10 = True
        priority_under_require_l5 = False
        if stat in {'ast', 'trb'} and allow_under:
            under_hit_floor = 0.64 if stat == 'ast' else 0.61
            under_01_floor = 0.63 if stat == 'ast' else 0.58
            under_12_floor = 0.66 if stat == 'ast' else 0.64
            priority_under_lane = (
                under_n >= 800 and
                under_hit is not None and under_hit >= under_hit_floor and
                under_0_1_hit is not None and under_0_1_hit >= under_01_floor and
                under_1_2_hit is not None and under_1_2_hit >= under_12_floor
            )
            if priority_under_lane:
                priority_under_min_edge = max(min_edge_under, 0.85 if stat == 'ast' else 0.90)
                priority_under_support_gap = 0.60 if stat == 'ast' else 0.75
                priority_under_conf_bonus = 0.045 if stat == 'ast' else 0.030
                priority_under_rank_bonus = 0.035 if stat == 'ast' else 0.025
                priority_under_selection_priority = 1.00 if stat == 'ast' else 0.70
                priority_under_require_l5 = True

        policy[stat] = {
            'allowed': allowed,
            'allow_over': allow_over,
            'allow_under': allow_under,
            'meta_gate_mode': meta_gate_mode,
            'meta_blend_weight': round(meta_blend_weight, 3),
            'meta_conf_weight': round(meta_conf_weight, 3),
            'shrunk_acc': round(shrunk_acc, 4),
            'min_prob': round(min_prob, 4),
            'min_meta_prob': round(min_meta_prob, 4),
            'min_edge': round(min_edge, 3),
            'min_edge_over': round(min_edge_over, 3),
            'min_edge_under': round(min_edge_under, 3),
            'rmse': rmse,
            'n_real': n_real,
            'divergence': round(divergence, 4),
            'clv_corr': round(clv_corr, 4),
            'strong_n': strong_n,
            'strong_hit': round(strong_hit, 4) if strong_hit is not None else None,
            'under_hit': round(under_hit, 4) if under_hit is not None else None,
            'over_hit': round(over_hit, 4) if over_hit is not None else None,
            'tracking_under_allowed': tracking_under_allowed,
            'market_hit': round(market_hit, 4) if market_hit is not None else None,
            'meta_auc': round(meta_auc, 4) if meta_auc is not None else None,
            'meta_top_hit': round(meta_top_hit, 4) if meta_top_hit is not None else None,
            'priority_under_lane': priority_under_lane,
            'priority_under_min_edge': round(priority_under_min_edge, 3),
            'priority_under_support_gap': round(priority_under_support_gap, 3),
            'priority_under_conf_bonus': round(priority_under_conf_bonus, 3),
            'priority_under_rank_bonus': round(priority_under_rank_bonus, 3),
            'priority_under_selection_priority': round(priority_under_selection_priority, 3),
            'priority_under_require_l10': priority_under_require_l10,
            'priority_under_require_l5': priority_under_require_l5,
        }

    return policy


def _load_target_quality_policy() -> Dict[str, Dict]:
    """
    Build per-target live-betting policy from the latest training results.

    Targets are allowed only when real-line validation is sufficiently strong and
    not wildly inconsistent with proxy-line behavior.
    """
    path = CONFIG['models_dir'] / 'training_results.json'
    edge_path = CONFIG['models_dir'] / 'training_edge_analysis.json'
    if not path.exists():
        return _build_target_quality_policy()

    try:
        with open(path) as f:
            results = json.load(f)
    except Exception:
        return _build_target_quality_policy()

    edge_analysis = {}
    if edge_path.exists():
        try:
            with open(edge_path) as f:
                edge_analysis = json.load(f).get('per_target', {})
        except Exception:
            edge_analysis = {}

    return _build_target_quality_policy(results=results, edge_analysis=edge_analysis)


def _filter_best_picks_legacy(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Deprecated compatibility shim for older scripts/notebooks."""
    return filter_best_picks(pred_df)

    picks = []

    # Per-prop UNDER thresholds: min gap (line - L10) required to take UNDER
    # Raised from 0.75-1.5 → 1.5-3.0 based on live data showing low-edge picks lose.
    # STL/BLK removed from props loop entirely — models too weak (61%/66%).
    UNDER_GAP_THRESHOLD = {
        'ast': 1.5,   # raised from 0.75 — small gaps were noise
        'trb': 2.0,   # raised from 1.0
        'pts': 3.0,   # raised from 1.5 — PTS UNDER needs strong signal
        'pra': 3.0,   # raised from 1.5
        'pr':  2.5,   # raised from 1.5
        'pa':  2.5,   # raised from 1.5
        'tov': 1.5,
    }
    # Max gap: if line > L10 by this much, Vegas is probably pricing a role change
    # we don't know about → skip (too risky either direction).
    UNDER_GAP_MAX = {
        'ast': 3.5, 'trb': 4.0, 'pts': 8.0,
        'pra': 12.0, 'pr': 9.0, 'pa': 9.0,
        'stl': 2.0, 'blk': 2.0, 'tov': 2.0,
    }
    # OVER threshold: L10 must exceed line by this much.
    # OVER picks win 44-53% historically — require very large gap to even consider.
    # PA OVER is the best at 53%, others are close to coin-flip or worse.
    OVER_GAP_THRESHOLD = {
        'pts': 99.0,  # PTS OVER: 44.5% historically — never bet
        'trb': 99.0,  # TRB OVER: 44.4% — never bet
        'ast': 99.0,  # AST OVER: insufficient sample
        'pra': 99.0,  # PRA OVER: 48.7% — never bet
        'pr':  99.0,  # PR OVER: 50.0% — coin flip
        'pa':  99.0,  # PA OVER: 52.5% — not enough edge vs -110 juice
        'tov': 99.0,
    }

    # STL and BLK excluded: weak models (61% / 66% accuracy) — not worth betting
    props = ['pts', 'trb', 'ast', 'pra', 'pr', 'pa', 'tov']

    for _, row in pred_df.iterrows():
        for prop in props:
            line_col = f'{prop}_line'
            l10_col  = f'{prop}_l10'
            pred_col = f'{prop}_pred'

            line = row.get(line_col)
            l10_val = row.get(l10_col)
            if line is None or l10_val is None or pd.isna(line) or pd.isna(l10_val):
                continue

            gap = line - l10_val          # positive  → line above L10 → UNDER edge
            l10_to_line = l10_val - line  # positive  → L10 above line → OVER edge

            # --- Determine direction and whether this qualifies ---
            under_thresh = UNDER_GAP_THRESHOLD.get(prop, 1.5)
            over_thresh  = OVER_GAP_THRESHOLD.get(prop, 3.0)

            # Also check L5 (last 5 games) — require both L5 and L10 below line.
            # This filters out players who are "trending up" (L5 near/above line)
            # even if their L10 average is still below the line.
            # Composite props: sum component L5s (same logic as composite L10).
            _composite_l5 = {
                'pra': ['pts_l5', 'trb_l5', 'ast_l5'],
                'pr':  ['pts_l5', 'trb_l5'],
                'pa':  ['pts_l5', 'ast_l5'],
            }
            if prop in _composite_l5:
                comps = [row.get(c) for c in _composite_l5[prop]]
                valid = [v for v in comps if v is not None and not pd.isna(v)]
                l5_val = sum(valid) if len(valid) == len(comps) else None
            else:
                l5_col  = f'{prop}_l5'
                l5_val  = row.get(l5_col)
            gap_l5  = (line - float(l5_val)) if l5_val is not None and not pd.isna(l5_val) else gap

            under_max = UNDER_GAP_MAX.get(prop, 99.0)
            if gap >= under_thresh and gap <= under_max:
                # Require L5 also below line (gap_l5 > 0) to filter hot-streak fade picks
                if gap_l5 <= 0:
                    continue  # player trending up — L5 is at/above line, skip
                direction = 'UNDER'
                # Weighted gap: 60% L10 gap + 40% L5 gap (more weight on recent confirmation)
                primary_gap = gap * 0.60 + gap_l5 * 0.40
            elif l10_to_line >= over_thresh:
                direction = 'OVER'
                primary_gap = l10_to_line
            else:
                continue  # No meaningful edge

            # --- Get player σ for this prop ---
            # Primary: model-predicted σ = (q75 - q25) / 1.35.  Context-conditioned on
            # minutes stability, usage, pace, and matchup via the quantile sidecar.
            # Fallback: rolling historical std from the feature pipeline.
            _composite_std_components = {
                'pra': ['pts_std', 'trb_std', 'ast_std'],
                'pr':  ['pts_std', 'trb_std'],
                'pa':  ['pts_std', 'ast_std'],
            }
            std_val = row.get(f'{prop}_sigma')
            if not std_val or pd.isna(std_val) or float(std_val) <= 0:
                # No quantile sidecar σ available — fall back to rolling std
                if prop in _composite_std_components:
                    comp_stds = [row.get(c) for c in _composite_std_components[prop]]
                    valid_stds = [s for s in comp_stds if s and not pd.isna(s) and s > 0]
                    std_val = sum(valid_stds) if valid_stds else None
                else:
                    std_val = row.get(f'{prop}_std')
                    if std_val is None or pd.isna(std_val):
                        std_val = row.get(f'{prop.split("_")[0]}_std')
            if std_val is not None:
                std_val = float(std_val)

            if std_val and std_val > 0:
                zscore = primary_gap / std_val
            else:
                # Final fallback: use prop-specific typical std
                default_stds = {'pts': 6.0, 'trb': 2.5, 'ast': 2.0,
                                'pra': 9.0, 'pr': 7.5, 'pa': 7.5,
                                'stl': 0.8, 'blk': 0.8, 'tov': 1.0}
                zscore = primary_gap / default_stds.get(prop, 4.0)

            # Confidence: primary signal is zscore of the gap
            conf_pct = 50 + 49 * (1 - math.exp(-0.55 * zscore))

            # Secondary: prediction agreement bonus (small)
            pred = row.get(pred_col)
            if pred is not None and not pd.isna(pred):
                pred_direction = 'OVER' if pred > line else 'UNDER'
                if pred_direction == direction:
                    conf_pct = min(99, conf_pct + 3)

            # Line movement adjustment
            line_move = row.get(f'{prop}_line_move')
            if line_move is not None and not pd.isna(line_move) and line_move != 0:
                move_with = (direction == 'OVER' and line_move > 0) or \
                            (direction == 'UNDER' and line_move < 0)
                if move_with:
                    conf_pct = min(99, conf_pct + min(4, abs(line_move) * 1.5))
                else:
                    conf_pct = max(50, conf_pct - min(6, abs(line_move) * 2.0))

            # ── Blend classifier P(direction) into confidence ─────────────────
            # clf_prob = P(OVER); invert for UNDER picks.
            # Weight: 40% classifier, 60% existing heuristic signals.
            clf_prob_raw = row.get(f'{prop}_clf_prob')
            if clf_prob_raw is not None and not pd.isna(clf_prob_raw):
                clf_dir_prob = (1.0 - clf_prob_raw) if direction == 'UNDER' else clf_prob_raw
                clf_conf_pct = clf_dir_prob * 100.0
                conf_pct     = round(conf_pct * 0.60 + clf_conf_pct * 0.40, 1)
                prob_payload = _build_pick_probability_payload(
                    direction=direction,
                    dir_prob=clf_dir_prob,
                    raw_p_over=clf_prob_raw,
                    cal_p_over=None,
                    prob_source='legacy_classifier_raw',
                )
            else:
                prob_payload = _build_pick_probability_payload(
                    direction=direction,
                    dir_prob=conf_pct / 100.0,
                    raw_p_over=None,
                    cal_p_over=None,
                    prob_source='legacy_confidence_only',
                )

            conf_pct = round(max(50.0, min(99.0, conf_pct)), 1)

            mp_exp = row.get('mp_expected')
            line_move_val = row.get(f'{prop}_line_move')
            picks.append({
                'player':         row['player'],
                'team':           row.get('team', ''),
                'prop':           prop.upper(),
                'direction':      direction,
                'line':           line,
                'prediction':     round(float(pred), 1) if pred is not None and not pd.isna(pred) else None,
                'edge':           round(primary_gap, 1),
                'confidence':     conf_pct,
                'mp_predicted':   round(float(mp_exp), 1) if mp_exp is not None and not pd.isna(mp_exp) else None,
                'l10_avg':        round(float(l10_val), 1),
                'l10_edge':       round(float(l10_val - line), 1),
                'l5_avg':         round(float(l5_val), 1) if l5_val is not None and not pd.isna(l5_val) else None,
                'line_move':      round(float(line_move_val), 2) if line_move_val is not None and not pd.isna(line_move_val) else None,
                'injury_adj':     row.get('injury_adj', False),
                'teammates_out':  row.get('teammates_out', ''),
                **prob_payload,
            })

    picks_df = pd.DataFrame(picks)

    if len(picks_df) == 0:
        return picks_df

    # De-duplicate: one pick per player+prop (keep highest confidence)
    picks_df = picks_df.sort_values('confidence', ascending=False)
    picks_df = picks_df.drop_duplicates(subset=['player', 'prop'], keep='first')
    picks_df = picks_df.reset_index(drop=True)

    # ── Load calibrator for empirical win-rate based ranking ─────────────────
    _cal_path = CONFIG['models_dir'] / 'confidence_calibrator.pkl'
    _stat_dir_rates = {}
    _iso = None
    _overall_wr = 0.54
    if _cal_path.exists():
        try:
            with open(_cal_path, 'rb') as _f:
                _cal = pickle.load(_f)
            _iso = _cal.get('iso')
            _stat_dir_rates = _cal.get('stat_dir_rates', {})
            _overall_wr = _cal.get('overall_wr', 0.54)
        except Exception:
            pass

    # Calibrated confidence: replace broken formula with isotonic-calibrated value
    if _iso is not None:
        picks_df['calibrated_conf'] = (_iso.predict(picks_df['confidence'].values) * 100).round(1)
    else:
        picks_df['calibrated_conf'] = picks_df['confidence']

    # Empirical stat-direction win rate as primary ranking signal
    picks_df['stat_wr'] = picks_df.apply(
        lambda r: _stat_dir_rates.get(f"{r['prop'].upper()}_{r['direction'].upper()}", _overall_wr),
        axis=1,
    )

    # Rank score: 70% empirical stat win-rate + 30% calibrated confidence
    picks_df['rank_score'] = (picks_df['stat_wr'] * 0.70 +
                              picks_df['calibrated_conf'] / 100 * 0.30).round(4)

    # Sort by rank score, cap at top 10
    picks_df = picks_df.sort_values('rank_score', ascending=False)
    picks_df = picks_df.head(10).reset_index(drop=True)

    return picks_df


def compute_regression_candidates(df: pd.DataFrame, lookback: int = 20, z_thresh: float = 2.5) -> dict:
    """
    Identify players who had an outlier performance in their last game.

    When a player scores 2.5+ standard deviations above their L20 mean, they
    statistically regress toward their mean in the next game. This is a model-
    independent signal — pure statistical mechanics, no XGBoost required.

    Returns {player_normalized: {stat1, stat2, ...}} for stats where UNDER
    confidence should be boosted in today's picks.
    """
    candidates: dict = {}
    stats = ['pts', 'trb', 'ast']
    df_sorted = df.sort_values('game_date')
    for player, grp in df_sorted.groupby('player'):
        if len(grp) < lookback + 1:
            continue
        last_game = grp.iloc[-1]
        lookback_games = grp.iloc[-(lookback + 1):-1]
        norm = _normalize_name(str(player))
        for stat in stats:
            if stat not in grp.columns:
                continue
            vals = lookback_games[stat].dropna()
            if len(vals) < 5:
                continue
            mean_val = vals.mean()
            std_val  = vals.std()
            if std_val < 0.5:
                continue
            last_val = last_game.get(stat)
            if last_val is None or pd.isna(last_val):
                continue
            z = (float(last_val) - mean_val) / std_val
            if z >= z_thresh:
                candidates.setdefault(norm, set()).add(stat)
    n = sum(len(v) for v in candidates.values())
    if n:
        print(f"  [Regression] {len(candidates)} players flagged for mean-reversion UNDER ({n} stat flags)")
    return candidates


def _load_sigma_thresholds() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Load per-target σ p25 and p75 thresholds from models/sigma_context_analysis.json.

    Returns (p25_dict, p75_dict), each mapping target → float threshold.
    - p25: low-σ boundary — below this the book is systematically mispriced
      (AST low-σ UNDER: 68% hit, TRB low-σ UNDER: 63% hit)
    - p75: high-σ boundary — above this real variance dominates; require more edge

    Both dicts are empty when the file does not exist (e.g. before first calibration run).
    """
    path = CONFIG['models_dir'] / 'sigma_context_analysis.json'
    if not path.exists():
        return {}, {}
    try:
        with open(path) as f:
            data = json.load(f)
        p25: Dict[str, float] = {}
        p75: Dict[str, float] = {}
        for target, info in data.items():
            if info.get('sigma_p25') is not None:
                p25[target] = float(info['sigma_p25'])
            if info.get('sigma_p75') is not None:
                p75[target] = float(info['sigma_p75'])
        return p25, p75
    except Exception:
        return {}, {}


def filter_best_picks(pred_df: pd.DataFrame, quality_policy: Optional[Dict[str, Dict]] = None,
                      regression_candidates: Optional[dict] = None, max_picks: int = 10) -> pd.DataFrame:
    """Filter to best betting opportunities using validated model edge."""

    picks = []
    quality = quality_policy or _load_target_quality_policy()
    props = [prop for prop, q in quality.items() if q.get('allowed')]
    sigma_p25, sigma_p75 = _load_sigma_thresholds()  # p25: {'ast': 1.68, 'trb': 2.10, ...}  p75: {'ast': 2.40, 'trb': 2.87, ...}

    for _, row in pred_df.iterrows():
        for prop in props:
            line = row.get(f'{prop}_line')
            pred = row.get(f'{prop}_pred')
            l10_val = row.get(f'{prop}_l10')
            anchor_val = row.get(f'{prop}_anchor')
            if anchor_val is None or pd.isna(anchor_val):
                anchor_val = _row_prop_feature(row, prop, 'best_estimate')
            if anchor_val is None or pd.isna(anchor_val):
                anchor_val = _row_prop_feature(row, prop, 'recency_anchor')
            if line is None or pred is None or pd.isna(line) or pd.isna(pred):
                continue

            model_edge = float(pred) - float(line)
            market_edge_pred = row.get(f'{prop}_market_edge_pred')
            market_edge_pred = float(market_edge_pred) if market_edge_pred is not None and not pd.isna(market_edge_pred) else None
            if market_edge_pred is not None:
                combined_edge = 0.45 * model_edge + 0.55 * market_edge_pred
            else:
                combined_edge = model_edge
            if abs(combined_edge) < 1e-9:
                continue
            direction = 'OVER' if combined_edge > 0 else 'UNDER'
            if direction == 'OVER' and not quality[prop].get('allow_over', quality[prop].get('allowed', False)):
                continue
            if direction == 'UNDER' and not quality[prop].get('allow_under', quality[prop].get('allowed', False)):
                continue

            # Promo-line gate: PrizePicks does not offer UNDER on goblin lines (artificially
            # low, More-only) or OVER on demon lines (artificially high, Less-only).
            # Picks on these lines are structurally unplaceable on PrizePicks.
            proj_type = _normalize_projection_type(row.get(f'{prop}_projection_type') or 'standard')
            is_promo = bool(row.get(f'{prop}_is_promo', False))
            if direction == 'UNDER' and proj_type == 'goblin':
                continue
            if direction == 'OVER' and proj_type == 'demon':
                continue
            if direction == 'UNDER' and is_promo:
                continue

            # σ-tier: flag low-σ UNDER picks where the book is systematically mispriced.
            # σ-context analysis shows that when the model predicts a tight distribution
            # (σ < p25 threshold), the actual OVER hit rate is 31–37% for AST/TRB —
            # a 13–19 point UNDER edge before even considering model prediction direction.
            sigma_val = row.get(f'{prop}_sigma')
            sigma_val = float(sigma_val) if sigma_val is not None and not pd.isna(sigma_val) and float(sigma_val) > 0 else None
            p25_threshold = sigma_p25.get(prop)
            low_sigma_under = (
                direction == 'UNDER'
                and sigma_val is not None
                and p25_threshold is not None
                and sigma_val < p25_threshold
            )

            min_edge = float(
                quality[prop].get('min_edge_over' if direction == 'OVER' else 'min_edge_under',
                                  quality[prop]['min_edge'])
            )
            # σ-tiered edge gate: only applied to UNDER picks where σ signal is available.
            # Low-σ (σ < p25) → book is systematically mispriced → lower bar by 25%.
            # High-σ (σ > p75) → real variance dominates → raise bar by 25%.
            # PTS has near-zero σ signal (buckets 45–49%); adjustment still applies but
            # is conservative since the σ spread itself is tight for PTS.
            if direction == 'UNDER' and sigma_val is not None:
                p75_threshold = sigma_p75.get(prop)
                if p25_threshold is not None and sigma_val < p25_threshold:
                    min_edge = max(0.50, min_edge * 0.75)
                elif p75_threshold is not None and sigma_val > p75_threshold:
                    min_edge = min_edge * 1.25
            if abs(combined_edge) < min_edge:
                continue

            l10_edge = None
            l10_agrees = None
            if l10_val is not None and not pd.isna(l10_val):
                l10_edge = float(l10_val) - float(line)
                l10_agrees = np.sign(l10_edge) == np.sign(combined_edge)

            anchor_edge = None
            anchor_agrees = None
            if anchor_val is not None and not pd.isna(anchor_val):
                anchor_edge = float(anchor_val) - float(line)
                anchor_agrees = np.sign(anchor_edge) == np.sign(combined_edge)

            composite_l5 = {
                'pra': ['pts_l5', 'trb_l5', 'ast_l5'],
                'pr':  ['pts_l5', 'trb_l5'],
                'pa':  ['pts_l5', 'ast_l5'],
            }
            if prop in composite_l5:
                comps = [row.get(c) for c in composite_l5[prop]]
                valid = [v for v in comps if v is not None and not pd.isna(v)]
                l5_val = sum(valid) if len(valid) == len(comps) else None
            else:
                l5_val = row.get(f'{prop}_l5')

            l5_edge = None
            l5_agrees = None
            if l5_val is not None and not pd.isna(l5_val):
                l5_edge = float(l5_val) - float(line)
                l5_agrees = np.sign(l5_edge) == np.sign(combined_edge)

            rmse = max(0.5, float(quality[prop].get('rmse', 1.0)))
            raw_p_over = row.get(f'{prop}_p_over_raw')
            raw_p_over = float(raw_p_over) if raw_p_over is not None and not pd.isna(raw_p_over) else None
            cal_p_over = row.get(f'{prop}_p_over_cal')
            cal_p_over = float(cal_p_over) if cal_p_over is not None and not pd.isna(cal_p_over) else None
            use_calibrated_prob = cal_p_over is not None
            clf_prob_raw = row.get(f'{prop}_clf_prob')
            prob_source = 'edge_logistic'
            if use_calibrated_prob:
                dir_prob = (1.0 - cal_p_over) if direction == 'UNDER' else cal_p_over
                prob_source = 'calibrated_p_over'
            elif clf_prob_raw is not None and not pd.isna(clf_prob_raw):
                dir_prob = (1.0 - float(clf_prob_raw)) if direction == 'UNDER' else float(clf_prob_raw)
                raw_p_over = float(clf_prob_raw)
                prob_source = 'classifier_raw'
            else:
                dir_prob = 1.0 / (1.0 + math.exp(-abs(combined_edge) / rmse))

            meta_prob_raw = row.get(f'{prop}_meta_prob')
            meta_prob = float(meta_prob_raw) if meta_prob_raw is not None and not pd.isna(meta_prob_raw) else None
            meta_gate_mode = str(quality[prop].get('meta_gate_mode', 'hard')).lower()
            meta_blend_weight = float(quality[prop].get('meta_blend_weight', 0.35))
            meta_conf_weight = float(quality[prop].get('meta_conf_weight', 0.10))
            if (not use_calibrated_prob and
                    CONFIG.get('require_meta_for_live_picks', False) and
                    meta_prob is None and meta_gate_mode == 'hard'):
                continue
            regime_score = row.get(f'{prop}_regime_score')
            regime_score = float(regime_score) if regime_score is not None and not pd.isna(regime_score) else 0.0
            edge_disagreement = row.get(f'{prop}_edge_disagreement')
            edge_disagreement = float(edge_disagreement) if edge_disagreement is not None and not pd.isna(edge_disagreement) else (abs(model_edge - market_edge_pred) if market_edge_pred is not None else 0.0)
            market_prob = 1.0 / (1.0 + math.exp(-abs(market_edge_pred if market_edge_pred is not None else combined_edge) / rmse))
            if use_calibrated_prob:
                dir_prob = dir_prob
            elif meta_prob is not None:
                market_blend_weight = 0.15 if meta_gate_mode == 'hard' else 0.20
                model_blend_weight = max(0.0, 1.0 - meta_blend_weight - market_blend_weight)
                dir_prob = (
                    dir_prob * model_blend_weight +
                    meta_prob * meta_blend_weight +
                    market_prob * market_blend_weight
                )
                prob_source = f'{prob_source}+meta_market'
            else:
                dir_prob = dir_prob * 0.75 + market_prob * 0.25
                if prob_source != 'calibrated_p_over':
                    prob_source = f'{prob_source}+market'

            min_prob = float(quality[prop]['min_prob'])
            if direction == 'OVER':
                min_prob += 0.02
            if anchor_agrees is False:
                min_prob += 0.01
            if l10_agrees is False:
                min_prob += 0.02
            if l5_agrees is False:
                min_prob += 0.02
            min_meta_prob = float(quality[prop].get('min_meta_prob', 0.55))
            if not use_calibrated_prob:
                if meta_prob is None:
                    if meta_gate_mode == 'medium':
                        min_prob += 0.01
                elif meta_prob < min_meta_prob:
                    if meta_gate_mode == 'hard':
                        continue
                    if meta_gate_mode == 'medium':
                        min_prob += 0.015
                    else:
                        min_prob += 0.01
            if regime_score >= 0.55 and edge_disagreement >= min_edge * 0.75:
                continue
            if dir_prob < min_prob:
                continue

            if direction == 'OVER' and l10_agrees is not True and abs(combined_edge) < min_edge * 1.50:
                continue
            if anchor_agrees is False and abs(combined_edge) < min_edge * 1.20:
                continue
            if l10_agrees is False and abs(combined_edge) < min_edge * 1.35:
                continue
            if l5_agrees is False and abs(combined_edge) < min_edge * 1.25:
                continue
            if market_edge_pred is not None and np.sign(model_edge) != np.sign(market_edge_pred) and abs(combined_edge) < min_edge * 1.50:
                continue
            # Extreme dampening gate: edge model sees >60% less downside than raw model AND
            # combined_edge is barely above threshold — market context provides no independent signal.
            if (market_edge_pred is not None
                    and np.sign(model_edge) == np.sign(market_edge_pred)
                    and abs(market_edge_pred) < abs(model_edge) * 0.40
                    and abs(combined_edge) < min_edge * 1.15):
                continue

            line_move = row.get(f'{prop}_line_move')
            if line_move is not None and not pd.isna(line_move) and line_move != 0:
                move_with = (direction == 'OVER' and line_move > 0) or \
                            (direction == 'UNDER' and line_move < 0)
                if not move_with and abs(float(line_move)) >= 0.9:
                    continue
            else:
                move_with = None

            priority_under_lane = False
            priority_lane_bonus = 0.0
            priority_rank_bonus = 0.0
            selection_priority = 0.0
            priority_support_gap = None
            if direction == 'UNDER' and quality[prop].get('priority_under_lane', False):
                support_vals = [float(v) for v in [pred, anchor_val, l10_val, l5_val] if v is not None and not pd.isna(v)]
                support_anchor = max(support_vals) if support_vals else float(pred)
                priority_support_gap = float(line) - support_anchor
                lane_edge_req = float(quality[prop].get('priority_under_min_edge', min_edge))
                lane_gap_req = float(quality[prop].get('priority_under_support_gap', 0.0))
                require_l10 = bool(quality[prop].get('priority_under_require_l10', True))
                require_l5 = bool(quality[prop].get('priority_under_require_l5', False))
                market_disagrees = (
                    market_edge_pred is not None and
                    np.sign(model_edge) != np.sign(market_edge_pred)
                )
                if (
                    abs(combined_edge) >= lane_edge_req and
                    priority_support_gap >= lane_gap_req and
                    (not require_l10 or l10_agrees is not False) and
                    (not require_l5 or l5_agrees is not False) and
                    not market_disagrees
                ):
                    priority_under_lane = True
                    priority_lane_bonus = float(quality[prop].get('priority_under_conf_bonus', 0.0))
                    priority_rank_bonus = float(quality[prop].get('priority_under_rank_bonus', 0.0))
                    selection_priority = float(quality[prop].get('priority_under_selection_priority', 0.0))

            edge_strength = min(1.0, abs(combined_edge) / (min_edge * 1.5))
            base_conf = (
                dir_prob * 0.45 +
                float(quality[prop]['shrunk_acc']) * 0.25 +
                edge_strength * 0.15
            )
            if meta_prob is not None and not use_calibrated_prob:
                base_conf += meta_prob * meta_conf_weight
            if market_edge_pred is not None:
                if np.sign(market_edge_pred) != np.sign(combined_edge):
                    # Edge model disagrees with bet direction — reduce confidence
                    base_conf -= min(0.03, abs(market_edge_pred) / max(1.0, rmse) * 0.04)
                elif abs(market_edge_pred) >= abs(model_edge) * 0.90:
                    # Amplifying or roughly matching — add confidence bonus (original behavior)
                    base_conf += min(0.08, abs(market_edge_pred) / max(1.0, rmse) * 0.08)
                # else: dampening (edge model less convinced than raw model) — no contribution
            if anchor_agrees:
                base_conf += 0.02
            if l10_agrees:
                base_conf += 0.03
            if l5_agrees:
                base_conf += 0.03
            if move_with is True:
                base_conf += 0.02
            elif move_with is False:
                base_conf -= 0.03
            if regime_score >= 0.40:
                base_conf -= min(0.08, regime_score * 0.10)
            if edge_disagreement >= min_edge:
                base_conf -= 0.04
            if priority_under_lane:
                base_conf += priority_lane_bonus
            # σ-tier bonus: low-σ UNDER picks are in the highest-predictability context.
            # +0.04 base_conf (~+4 conf_pct points) and +0.50 selection_priority boost.
            # Stacks with priority_under_lane when both conditions hold.
            if low_sigma_under:
                base_conf += 0.04
                selection_priority = max(selection_priority, 0.0) + 0.50

            conf_pct = round(max(50.0, min(99.0, base_conf * 100)), 1)
            mp_exp = row.get('mp_expected')
            line_move_val = row.get(f'{prop}_line_move')

            # Regression-to-mean boost: if player had a 2.5+ SD outlier last game
            # → they statistically regress → UNDER confidence gets +5 bonus.
            # Disabled when injury_adj=True (injury usage boost conflicts with regression).
            regression_flag = False
            if (regression_candidates and direction == 'UNDER'
                    and not row.get('injury_adj', False)):
                player_norm = _normalize_name(str(row['player']))
                if prop in regression_candidates.get(player_norm, set()):
                    regression_flag = True
                    conf_pct = min(99.0, conf_pct + 5.0)

            prob_payload = _build_pick_probability_payload(
                direction=direction,
                dir_prob=dir_prob,
                raw_p_over=raw_p_over,
                cal_p_over=cal_p_over,
                prob_source=prob_source,
            )
            break_even_prob = round((1.0 / _PP_MULTIPLIERS[2]) ** 0.5, 4)

            picks.append({
                'player':         row['player'],
                'team':           row.get('team', ''),
                'opponent':       row.get('opp', row.get('opponent', '')),
                'prop':           prop.upper(),
                'direction':      direction,
                'line':           line,
                'projection_type': proj_type,
                'is_promo':       is_promo,
                'prediction':     round(float(pred), 1),
                'edge':           round(float(combined_edge), 1),
                'confidence':     conf_pct,
                'meta_prob':      round(meta_prob * 100, 1) if meta_prob is not None else None,
                'market_edge':    round(float(market_edge_pred), 2) if market_edge_pred is not None else None,
                'regime_score':   round(regime_score, 3),
                'edge_disagreement': round(edge_disagreement, 3),
                'pick_source':    'market_model',
                'mp_predicted':   round(float(mp_exp), 1) if mp_exp is not None and not pd.isna(mp_exp) else None,
                'anchor_avg':     round(float(anchor_val), 1) if anchor_val is not None and not pd.isna(anchor_val) else None,
                'anchor_edge':    round(float(anchor_edge), 1) if anchor_edge is not None else None,
                'l10_avg':        round(float(l10_val), 1) if l10_val is not None and not pd.isna(l10_val) else None,
                'l10_edge':       round(float(l10_edge), 1) if l10_edge is not None else None,
                'l5_avg':         round(float(l5_val), 1) if l5_val is not None and not pd.isna(l5_val) else None,
                'line_move':      round(float(line_move_val), 2) if line_move_val is not None and not pd.isna(line_move_val) else None,
                'model_wr':       round(float(quality[prop]['shrunk_acc']) * 100, 1),
                'min_edge_req':   round(min_edge, 2),
                'injury_adj':     row.get('injury_adj', False),
                'teammates_out':  row.get('teammates_out', ''),
                'regression_flag': regression_flag,
                'priority_under_lane': priority_under_lane,
                'priority_support_gap': round(float(priority_support_gap), 2) if priority_support_gap is not None else None,
                'selection_priority': selection_priority,
                'lane_rank_bonus': priority_rank_bonus,
                'sigma': round(sigma_val, 3) if sigma_val is not None else None,
                'low_sigma_under': low_sigma_under,
                'break_even_prob': break_even_prob,
                'exceeds_ev_threshold': bool(dir_prob >= break_even_prob),
                **prob_payload,
            })

    # ── Structural STL/BLK UNDER picks ──────────────────────────────────────
    # STL UNDER: 73.1% historical win rate (1,345 picks). BLK UNDER: 79.5% (1,160 picks).
    # These are excluded from the model path because the regressor is too noisy.
    # The structural edge is pure: PrizePicks sets STL/BLK lines at 1.5+ for players
    # who average < 1.0 in those categories. No model needed — the line itself is wrong.
    if CONFIG.get('allow_structural_micro_props', False):
        _structural_meta = [
            ('stl', 73.0, 'STL'),
            ('blk', 78.0, 'BLK'),
        ]
        for prop_name, hist_wr, prop_upper in _structural_meta:
            line_col = f'{prop_name}_line'
            l10_col  = f'{prop_name}_l10'
            l5_col   = f'{prop_name}_l5'
            for _, row in pred_df.iterrows():
                line   = row.get(line_col)
                l10_val = row.get(l10_col)
                if line is None or l10_val is None or pd.isna(line) or pd.isna(l10_val):
                    continue
                if float(line) < 1.5:
                    continue  # line too low — might be a different PrizePicks type
                # L10 must be below the line to confirm the structural UNDER thesis
                if float(l10_val) >= float(line):
                    continue
                l5_val   = row.get(l5_col)
                edge_val = round(float(l10_val) - float(line), 1)  # negative = UNDER edge
                picks.append({
                    'player':         row['player'],
                    'team':           row.get('team', ''),
                    'prop':           prop_upper,
                    'direction':      'UNDER',
                    'line':           float(line),
                    'prediction':     round(float(l10_val), 1),  # L10 avg as proxy prediction
                    'edge':           edge_val,
                    'confidence':     hist_wr,
                    'ou_prob':        None,
                    'p_over_raw':     None,
                    'p_over_cal':     None,
                    'dir_prob':       hist_wr,
                    'meta_prob':      None,
                    'market_edge':    None,
                    'regime_score':   None,
                    'edge_disagreement': None,
                    'pick_source':    'structural_micro',
                    'mp_predicted':   None,
                    'l10_avg':        round(float(l10_val), 1),
                    'l10_edge':       edge_val,
                    'l5_avg':         round(float(l5_val), 1) if l5_val is not None and not pd.isna(l5_val) else None,
                    'line_move':      None,
                    'model_wr':       hist_wr,
                    'min_edge_req':   0.5,
                    'injury_adj':     row.get('injury_adj', False),
                    'teammates_out':  row.get('teammates_out', ''),
                    'regression_flag': False,
                    'sigma': None,
                    'low_sigma_under': False,
                })

    # ── Volume-fill pass ─────────────────────────────────────────────────────
    # When the primary filter + structural path produce fewer picks than needed
    # for a full card, do a relaxed second pass. Rules: must be an allowed UNDER
    # prop, edge must be ≥ 65% of the policy minimum, dir_prob must be ≥ 53%,
    # and the meta model must not actively disagree (meta_prob ≥ 0.48 if present).
    # These picks are capped at 63% confidence so they rank below primary picks.
    _FILL_POOL_TARGET = max(10, max_picks)
    if len(picks) < _FILL_POOL_TARGET:
        _existing = {(p['player'], p['prop']) for p in picks}
        _fill_limit = _FILL_POOL_TARGET - len(picks)
        _fill_count = 0

        for _, row in pred_df.iterrows():
            if _fill_count >= _fill_limit:
                break
            for prop in props:
                if _fill_count >= _fill_limit:
                    break
                # Volume fill is UNDER-only — OVERs have weak historical win rates
                if not quality[prop].get('allow_under', False):
                    continue

                line = row.get(f'{prop}_line')
                pred = row.get(f'{prop}_pred')
                if line is None or pred is None or pd.isna(line) or pd.isna(pred):
                    continue

                model_edge = float(pred) - float(line)
                market_edge_pred_raw = row.get(f'{prop}_market_edge_pred')
                market_edge_pred_fill = (
                    float(market_edge_pred_raw)
                    if market_edge_pred_raw is not None and not pd.isna(market_edge_pred_raw)
                    else None
                )
                combined_edge = (
                    0.45 * model_edge + 0.55 * market_edge_pred_fill
                    if market_edge_pred_fill is not None
                    else model_edge
                )

                # Must be a genuine UNDER signal
                if combined_edge >= 0:
                    continue

                # Skip if already captured by primary pass
                if (row['player'], prop.upper()) in _existing:
                    continue

                # Relaxed minimum edge: 65% of the policy floor
                min_edge_fill = float(
                    quality[prop].get('min_edge_under', quality[prop]['min_edge'])
                ) * 0.65
                if abs(combined_edge) < min_edge_fill:
                    continue

                # Compute directional probability
                rmse_fill = max(0.5, float(quality[prop].get('rmse', 1.0)))
                cal_p_over_raw = row.get(f'{prop}_p_over_cal')
                cal_p_over_fill = (
                    float(cal_p_over_raw)
                    if cal_p_over_raw is not None and not pd.isna(cal_p_over_raw)
                    else None
                )
                clf_prob_fill = row.get(f'{prop}_clf_prob')
                if cal_p_over_fill is not None:
                    dir_prob_fill = 1.0 - cal_p_over_fill
                    prob_source_fill = 'calibrated_p_over'
                elif clf_prob_fill is not None and not pd.isna(clf_prob_fill):
                    dir_prob_fill = 1.0 - float(clf_prob_fill)
                    prob_source_fill = 'classifier_raw'
                else:
                    dir_prob_fill = 1.0 / (1.0 + math.exp(-abs(combined_edge) / rmse_fill))
                    prob_source_fill = 'edge_logistic'

                if dir_prob_fill < 0.53:
                    continue

                # Reject if meta model actively disagrees
                meta_prob_fill_raw = row.get(f'{prop}_meta_prob')
                meta_prob_fill = (
                    float(meta_prob_fill_raw)
                    if meta_prob_fill_raw is not None and not pd.isna(meta_prob_fill_raw)
                    else None
                )
                if meta_prob_fill is not None and meta_prob_fill < 0.48:
                    continue

                # Cap confidence below primary picks so they sort last
                conf_pct_fill = round(max(50.0, min(63.0, dir_prob_fill * 100)), 1)

                l10_fill = row.get(f'{prop}_l10')
                l5_fill = row.get(f'{prop}_l5')
                anchor_fill = row.get(f'{prop}_anchor')
                if anchor_fill is None or pd.isna(anchor_fill):
                    anchor_fill = _row_prop_feature(row, prop, 'best_estimate')
                mp_exp_fill = row.get('mp_expected')
                lm_fill = row.get(f'{prop}_line_move')
                raw_p_over_fill = row.get(f'{prop}_p_over_raw')
                raw_p_over_fill = (
                    float(raw_p_over_fill)
                    if raw_p_over_fill is not None and not pd.isna(raw_p_over_fill)
                    else (float(clf_prob_fill) if clf_prob_fill is not None and not pd.isna(clf_prob_fill) else None)
                )
                prob_payload_fill = _build_pick_probability_payload(
                    direction='UNDER',
                    dir_prob=dir_prob_fill,
                    raw_p_over=raw_p_over_fill,
                    cal_p_over=cal_p_over_fill,
                    prob_source=prob_source_fill,
                )
                proj_type_fill = _normalize_projection_type(row.get(f'{prop}_projection_type') or 'standard')
                is_promo_fill = bool(row.get(f'{prop}_is_promo', False))
                if proj_type_fill == 'goblin' or is_promo_fill:
                    continue
                break_even_prob_fill = round((1.0 / _PP_MULTIPLIERS[2]) ** 0.5, 4)

                picks.append({
                    'player':               row['player'],
                    'team':                 row.get('team', ''),
                    'opponent':             row.get('opp', row.get('opponent', '')),
                    'prop':                 prop.upper(),
                    'direction':            'UNDER',
                    'line':                 line,
                    'projection_type':      proj_type_fill,
                    'is_promo':             is_promo_fill,
                    'prediction':           round(float(pred), 1),
                    'edge':                 round(float(combined_edge), 1),
                    'confidence':           conf_pct_fill,
                    'meta_prob':            round(meta_prob_fill * 100, 1) if meta_prob_fill is not None else None,
                    'market_edge':          round(market_edge_pred_fill, 2) if market_edge_pred_fill is not None else None,
                    'regime_score':         0.0,
                    'edge_disagreement':    0.0,
                    'pick_source':          'volume_fill',
                    'mp_predicted':         round(float(mp_exp_fill), 1) if mp_exp_fill is not None and not pd.isna(mp_exp_fill) else None,
                    'anchor_avg':           round(float(anchor_fill), 1) if anchor_fill is not None and not pd.isna(anchor_fill) else None,
                    'anchor_edge':          round(float(anchor_fill) - float(line), 1) if anchor_fill is not None and not pd.isna(anchor_fill) else None,
                    'l10_avg':              round(float(l10_fill), 1) if l10_fill is not None and not pd.isna(l10_fill) else None,
                    'l10_edge':             round(float(l10_fill) - float(line), 1) if l10_fill is not None and not pd.isna(l10_fill) else None,
                    'l5_avg':               round(float(l5_fill), 1) if l5_fill is not None and not pd.isna(l5_fill) else None,
                    'line_move':            round(float(lm_fill), 2) if lm_fill is not None and not pd.isna(lm_fill) else None,
                    'model_wr':             round(float(quality[prop]['shrunk_acc']) * 100, 1),
                    'min_edge_req':         round(min_edge_fill, 2),
                    'injury_adj':           row.get('injury_adj', False),
                    'teammates_out':        row.get('teammates_out', ''),
                    'regression_flag':      False,
                    'priority_under_lane':  False,
                    'priority_support_gap': None,
                    'selection_priority':   0.0,
                    'lane_rank_bonus':      0.0,
                    'sigma':                None,
                    'low_sigma_under':      False,
                    'break_even_prob':      break_even_prob_fill,
                    'exceeds_ev_threshold': bool(dir_prob_fill >= break_even_prob_fill),
                    **prob_payload_fill,
                })
                _existing.add((row['player'], prop.upper()))
                _fill_count += 1

        if _fill_count > 0:
            print(
                f"  [volume_fill] Primary pass: {len(picks) - _fill_count} picks. "
                f"Added {_fill_count} fill picks (relaxed thresholds) to reach pool of {len(picks)}."
            )

    # ── Emergency-fill pass ──────────────────────────────────────────────────
    # Hard floor for publication volume: if the candidate pool is still too thin,
    # take the best remaining model-aligned props with lighter thresholds, cap
    # confidence, and keep them out of betslips.
    if len(picks) < max_picks:
        _existing = {(p['player'], p['prop']) for p in picks}
        _emergency_limit = max_picks - len(picks)
        _emergency_count = 0

        for _, row in pred_df.iterrows():
            if _emergency_count >= _emergency_limit:
                break
            for prop in props:
                if _emergency_count >= _emergency_limit:
                    break
                if (row['player'], prop.upper()) in _existing:
                    continue

                line = row.get(f'{prop}_line')
                pred = row.get(f'{prop}_pred')
                if line is None or pred is None or pd.isna(line) or pd.isna(pred):
                    continue

                model_edge = float(pred) - float(line)
                market_edge_pred_raw = row.get(f'{prop}_market_edge_pred')
                market_edge_pred = (
                    float(market_edge_pred_raw)
                    if market_edge_pred_raw is not None and not pd.isna(market_edge_pred_raw)
                    else None
                )
                combined_edge = (
                    0.45 * model_edge + 0.55 * market_edge_pred
                    if market_edge_pred is not None
                    else model_edge
                )
                if abs(combined_edge) < 1e-9:
                    continue

                direction = 'OVER' if combined_edge > 0 else 'UNDER'
                if direction == 'OVER' and not quality[prop].get('allow_over', False):
                    continue
                if direction == 'UNDER' and not quality[prop].get('allow_under', False):
                    continue

                min_edge_base = float(
                    quality[prop].get(
                        'min_edge_over' if direction == 'OVER' else 'min_edge_under',
                        quality[prop]['min_edge'],
                    )
                )
                min_edge_emergency = max(0.35, min_edge_base * 0.35)
                if abs(combined_edge) < min_edge_emergency:
                    continue

                rmse_emergency = max(0.5, float(quality[prop].get('rmse', 1.0)))
                cal_p_over_raw = row.get(f'{prop}_p_over_cal')
                cal_p_over = (
                    float(cal_p_over_raw)
                    if cal_p_over_raw is not None and not pd.isna(cal_p_over_raw)
                    else None
                )
                clf_prob_emergency = row.get(f'{prop}_clf_prob')
                if cal_p_over is not None:
                    dir_prob_emergency = (1.0 - cal_p_over) if direction == 'UNDER' else cal_p_over
                    prob_source_emergency = 'calibrated_p_over'
                elif clf_prob_emergency is not None and not pd.isna(clf_prob_emergency):
                    dir_prob_emergency = (1.0 - float(clf_prob_emergency)) if direction == 'UNDER' else float(clf_prob_emergency)
                    prob_source_emergency = 'classifier_raw'
                else:
                    dir_prob_emergency = 1.0 / (1.0 + math.exp(-abs(combined_edge) / rmse_emergency))
                    prob_source_emergency = 'edge_logistic'

                if dir_prob_emergency < 0.50:
                    continue

                meta_prob_raw = row.get(f'{prop}_meta_prob')
                meta_prob = (
                    float(meta_prob_raw)
                    if meta_prob_raw is not None and not pd.isna(meta_prob_raw)
                    else None
                )
                if meta_prob is not None and meta_prob < 0.45:
                    continue

                conf_pct_emergency = round(max(50.0, min(60.0, dir_prob_emergency * 100)), 1)
                l10_emergency = row.get(f'{prop}_l10')
                l5_emergency = row.get(f'{prop}_l5')
                anchor_emergency = row.get(f'{prop}_anchor')
                if anchor_emergency is None or pd.isna(anchor_emergency):
                    anchor_emergency = _row_prop_feature(row, prop, 'best_estimate')
                mp_exp_emergency = row.get('mp_expected')
                lm_emergency = row.get(f'{prop}_line_move')
                raw_p_over_emergency = row.get(f'{prop}_p_over_raw')
                raw_p_over_emergency = (
                    float(raw_p_over_emergency)
                    if raw_p_over_emergency is not None and not pd.isna(raw_p_over_emergency)
                    else (float(clf_prob_emergency) if clf_prob_emergency is not None and not pd.isna(clf_prob_emergency) else None)
                )
                prob_payload_emergency = _build_pick_probability_payload(
                    direction=direction,
                    dir_prob=dir_prob_emergency,
                    raw_p_over=raw_p_over_emergency,
                    cal_p_over=cal_p_over,
                    prob_source=prob_source_emergency,
                )
                proj_type_emergency = _normalize_projection_type(row.get(f'{prop}_projection_type') or 'standard')
                is_promo_emergency = bool(row.get(f'{prop}_is_promo', False))
                if direction == 'UNDER' and (proj_type_emergency == 'goblin' or is_promo_emergency):
                    continue
                if direction == 'OVER' and proj_type_emergency == 'demon':
                    continue
                break_even_prob_emergency = round((1.0 / _PP_MULTIPLIERS[2]) ** 0.5, 4)

                picks.append({
                    'player':               row['player'],
                    'team':                 row.get('team', ''),
                    'opponent':             row.get('opp', row.get('opponent', '')),
                    'prop':                 prop.upper(),
                    'direction':            direction,
                    'line':                 line,
                    'projection_type':      proj_type_emergency,
                    'is_promo':             is_promo_emergency,
                    'prediction':           round(float(pred), 1),
                    'edge':                 round(float(combined_edge), 1),
                    'confidence':           conf_pct_emergency,
                    'meta_prob':            round(meta_prob * 100, 1) if meta_prob is not None else None,
                    'market_edge':          round(market_edge_pred, 2) if market_edge_pred is not None else None,
                    'regime_score':         0.0,
                    'edge_disagreement':    0.0,
                    'pick_source':          'emergency_fill',
                    'mp_predicted':         round(float(mp_exp_emergency), 1) if mp_exp_emergency is not None and not pd.isna(mp_exp_emergency) else None,
                    'anchor_avg':           round(float(anchor_emergency), 1) if anchor_emergency is not None and not pd.isna(anchor_emergency) else None,
                    'anchor_edge':          round(float(anchor_emergency) - float(line), 1) if anchor_emergency is not None and not pd.isna(anchor_emergency) else None,
                    'l10_avg':              round(float(l10_emergency), 1) if l10_emergency is not None and not pd.isna(l10_emergency) else None,
                    'l10_edge':             round(float(l10_emergency) - float(line), 1) if l10_emergency is not None and not pd.isna(l10_emergency) else None,
                    'l5_avg':               round(float(l5_emergency), 1) if l5_emergency is not None and not pd.isna(l5_emergency) else None,
                    'line_move':            round(float(lm_emergency), 2) if lm_emergency is not None and not pd.isna(lm_emergency) else None,
                    'model_wr':             round(float(quality[prop]['shrunk_acc']) * 100, 1),
                    'min_edge_req':         round(min_edge_emergency, 2),
                    'injury_adj':           row.get('injury_adj', False),
                    'teammates_out':        row.get('teammates_out', ''),
                    'regression_flag':      False,
                    'priority_under_lane':  False,
                    'priority_support_gap': None,
                    'selection_priority':   -0.5,
                    'lane_rank_bonus':      0.0,
                    'sigma':                None,
                    'low_sigma_under':      False,
                    'break_even_prob':      break_even_prob_emergency,
                    'exceeds_ev_threshold': bool(dir_prob_emergency >= break_even_prob_emergency),
                    **prob_payload_emergency,
                })
                _existing.add((row['player'], prop.upper()))
                _emergency_count += 1

        if _emergency_count > 0:
            print(
                f"  [emergency_fill] Added {_emergency_count} fallback picks "
                f"to guarantee publishable card depth ({len(picks)} total candidates)."
            )

    picks_df = pd.DataFrame(picks)
    if len(picks_df) == 0:
        return picks_df
    if 'projection_type' not in picks_df.columns:
        picks_df['projection_type'] = 'standard'
    picks_df['projection_type'] = picks_df['projection_type'].apply(_normalize_projection_type)
    if 'is_promo' not in picks_df.columns:
        picks_df['is_promo'] = False
    picks_df['is_promo'] = picks_df['is_promo'].fillna(False).astype(bool)
    market_block_mask = (
        (
            picks_df['direction'].astype(str).str.upper().eq('UNDER') &
            (picks_df['projection_type'].eq('goblin') | picks_df['is_promo'])
        ) |
        (
            picks_df['direction'].astype(str).str.upper().eq('OVER') &
            picks_df['projection_type'].eq('demon')
        )
    )
    if market_block_mask.any():
        picks_df = picks_df.loc[~market_block_mask].copy()
    if len(picks_df) == 0:
        return picks_df

    picks_df = picks_df.sort_values('confidence', ascending=False)
    picks_df = picks_df.drop_duplicates(subset=['player', 'prop'], keep='first').reset_index(drop=True)
    picks_df['calibrated_conf'] = picks_df['confidence']
    picks_df['stat_wr'] = (picks_df['model_wr'] / 100.0).round(4)
    if 'meta_prob' not in picks_df.columns:
        picks_df['meta_prob'] = np.nan
    if 'p_over_raw' not in picks_df.columns:
        picks_df['p_over_raw'] = np.nan
    if 'p_over_cal' not in picks_df.columns:
        picks_df['p_over_cal'] = np.nan
    if 'p_under_raw' not in picks_df.columns:
        picks_df['p_under_raw'] = np.nan
    if 'p_under_cal' not in picks_df.columns:
        picks_df['p_under_cal'] = np.nan
    if 'prob_source' not in picks_df.columns:
        picks_df['prob_source'] = ''
    if 'selection_priority' not in picks_df.columns:
        picks_df['selection_priority'] = 0.0
    if 'lane_rank_bonus' not in picks_df.columns:
        picks_df['lane_rank_bonus'] = 0.0
    if 'sigma' not in picks_df.columns:
        picks_df['sigma'] = np.nan
    if 'low_sigma_under' not in picks_df.columns:
        picks_df['low_sigma_under'] = False
    edge_norm = picks_df.apply(
        lambda r: min(1.0, abs(r['edge']) / max(0.5, float(r['min_edge_req']) * 1.5)),
        axis=1,
    )
    meta_norm = picks_df['meta_prob'].fillna(picks_df['dir_prob']).clip(lower=50, upper=99) / 100.0
    picks_df['rank_score'] = (
        picks_df['stat_wr'] * 0.30 +
        (picks_df['dir_prob'] / 100.0) * 0.25 +
        meta_norm * 0.30 +
        edge_norm * 0.15 +
        picks_df['lane_rank_bonus'].fillna(0.0)
    ).round(4)

    max_picks = max(1, int(max_picks))
    picks_df = picks_df.sort_values(['selection_priority', 'rank_score', 'confidence'], ascending=False)
    picks_df = picks_df.head(max_picks).reset_index(drop=True)
    return picks_df


def build_live_pick_card(picks_df: pd.DataFrame,
                         target_picks: int = 5,
                         min_picks: int = 5) -> pd.DataFrame:
    """
    Build the published live card from a sanitized candidate pool.

    Prefer prop diversity first, then controlled duplicates. STL/BLK are capped
    so the board does not collapse into one stat family unless the pool is too
    thin to hit the requested card size.
    """
    if picks_df is None or len(picks_df) == 0:
        return pd.DataFrame() if picks_df is None else picks_df

    target_picks = max(1, int(target_picks))
    min_picks = max(1, min(int(min_picks), target_picks))

    working = picks_df.copy()
    sort_cols = [c for c in ['selection_priority', 'rank_score', 'confidence', 'dir_prob', 'edge'] if c in working.columns]
    if sort_cols:
        working = working.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    working = working.drop_duplicates(subset=['player', 'prop'], keep='first').reset_index(drop=True)

    micro_props = {'STL', 'BLK'}
    phases = [
        {'prefer_new_prop': True, 'max_same_prop': 1, 'max_micro_props': 2, 'max_blk_props': 1},
        {'prefer_new_prop': False, 'max_same_prop': 2, 'max_micro_props': 2, 'max_blk_props': 1},
        {'prefer_new_prop': False, 'max_same_prop': 2, 'max_micro_props': 3, 'max_blk_props': 1},
        {'prefer_new_prop': False, 'max_same_prop': 3, 'max_micro_props': target_picks, 'max_blk_props': 2},
    ]

    selected_idx = []
    selected_players = set()
    player_counts: Dict[str, int] = {}
    prop_counts: Dict[str, int] = {}
    micro_count = 0

    def _prop_count(prop_name: str) -> int:
        return int(prop_counts.get(prop_name, 0))

    for phase in phases:
        changed = True
        while changed and len(selected_idx) < target_picks:
            changed = False
            for idx, row in working.iterrows():
                if idx in selected_idx:
                    continue
                player = str(row.get('player', ''))
                prop = str(row.get('prop', '')).upper()
                if not player or player in selected_players:
                    continue
                if phase['prefer_new_prop'] and _prop_count(prop) > 0:
                    continue
                if _prop_count(prop) >= phase['max_same_prop']:
                    continue
                if prop == 'BLK' and _prop_count(prop) >= phase.get('max_blk_props', phase['max_same_prop']):
                    continue
                if prop in micro_props and micro_count >= phase['max_micro_props']:
                    continue

                selected_idx.append(idx)
                selected_players.add(player)
                player_counts[player] = int(player_counts.get(player, 0)) + 1
                prop_counts[prop] = _prop_count(prop) + 1
                if prop in micro_props:
                    micro_count += 1
                changed = True
                if len(selected_idx) >= target_picks:
                    break

    if len(selected_idx) < min_picks:
        for idx, row in working.iterrows():
            if idx in selected_idx:
                continue
            player = str(row.get('player', ''))
            if not player or player in selected_players:
                continue
            selected_idx.append(idx)
            selected_players.add(player)
            player_counts[player] = int(player_counts.get(player, 0)) + 1
            if len(selected_idx) >= min_picks:
                break

    if len(selected_idx) < min_picks:
        for idx, row in working.iterrows():
            if idx in selected_idx:
                continue
            player = str(row.get('player', ''))
            if not player:
                continue
            if int(player_counts.get(player, 0)) >= 2:
                continue
            selected_idx.append(idx)
            player_counts[player] = int(player_counts.get(player, 0)) + 1
            if len(selected_idx) >= min_picks:
                break

    final_df = working.loc[selected_idx].copy() if selected_idx else working.iloc[0:0].copy()
    if sort_cols and len(final_df) > 0:
        final_df = final_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return final_df.head(target_picks).reset_index(drop=True)


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest(df: pd.DataFrame, test_days: int = 30) -> Dict:
    """Backtest model performance on recent games."""
    
    print("\nBacktesting...")
    
    # Sort by date
    df = df.sort_values('game_date')
    
    # Get test cutoff
    max_date = df['game_date'].max()
    test_start = max_date - timedelta(days=test_days)
    
    train_df = df[df['game_date'] < test_start].copy()
    test_df = df[df['game_date'] >= test_start].copy()
    
    print(f"  Train: {len(train_df):,} games (up to {test_start.strftime('%Y-%m-%d')})")
    print(f"  Test: {len(test_df):,} games ({test_days} days)")
    
    # Create features
    train_df = create_features(train_df)
    
    # Train on training set
    models = train_models(train_df)
    
    # Create features for test set (using only past data for global stats)
    full_df = create_features(df, stats_cutoff=test_start)
    test_with_features = full_df[full_df['game_date'] >= test_start].copy()
    
    # Filter to players with enough history
    test_with_features = test_with_features[test_with_features['games_played'] >= CONFIG['min_games']]

    # Load historical prop lines (real Vegas lines saved by predict runs / backfill)
    lines_path = CONFIG['data_dir'] / 'historical_lines.csv'
    hist_lines = None
    if lines_path.exists():
        hist_lines = pd.read_csv(lines_path)
        hist_lines['game_date'] = pd.to_datetime(hist_lines['game_date'])
        if 'player_norm' not in hist_lines.columns:
            hist_lines['player_norm'] = hist_lines['player'].apply(_normalize_name)
        # Map Odds API prop names → our internal stat names
        _prop_map = {
            'player_points': 'pts', 'player_rebounds': 'trb', 'player_assists': 'ast',
            'player_points_rebounds_assists': 'pra', 'player_points_rebounds': 'pr',
            'player_points_assists': 'pa', 'player_steals': 'stl',
            'player_blocks': 'blk', 'player_turnovers': 'tov',
        }
        hist_lines['stat'] = hist_lines['prop'].map(_prop_map)
        hist_lines = hist_lines.dropna(subset=['stat'])
        n_real = hist_lines[hist_lines['game_date'] >= test_with_features['game_date'].min()].shape[0]
        print(f"  Real lines available for test period: {n_real} player-prop rows")
    else:
        print("  No historical_lines.csv found — using L10 as synthetic line for all bets")

    # Add normalised name column for joining
    test_with_features = test_with_features.copy()
    test_with_features['player_norm'] = test_with_features['player'].apply(_normalize_name)

    # Get predictions
    models_path = CONFIG['models_dir'] / 'models.pkl'
    with open(models_path, 'rb') as f:
        saved = pickle.load(f)
    
    models = saved['models']
    feature_cols = saved['feature_cols']
    is_residual = saved.get('residual', False)

    X_test = test_with_features[feature_cols].fillna(0)

    results = {}

    for target_name, model in models.items():
        if target_name not in test_with_features.columns:
            continue

        y_true = test_with_features[target_name]
        y_pred = _predict(model, X_test)

        # Reconstruct raw stat predictions from residual model
        l10_col = f'{target_name}_l10'
        anchor_vals = _series_prop_feature(test_with_features, target_name, 'best_estimate')
        anchor_vals = anchor_vals.fillna(_series_prop_feature(test_with_features, target_name, 'recency_anchor'))
        anchor_vals = anchor_vals.fillna(
            pd.to_numeric(test_with_features[l10_col], errors='coerce') if l10_col in test_with_features.columns
            else pd.Series(np.nan, index=test_with_features.index)
        )
        if is_residual and l10_col in test_with_features.columns:
            y_pred_actual = y_pred + anchor_vals.fillna(0).values
        else:
            y_pred_actual = y_pred

        # MAE on actual scale
        mae = np.mean(np.abs(y_true - y_pred_actual))

        # Determine line: use real Vegas line where available, then fall back to the
        # recency-weighted anchor instead of a simple L10 average.
        if l10_col in test_with_features.columns or hist_lines is not None:
            real_line_vals = np.full(len(test_with_features), np.nan)
            if hist_lines is not None:
                stat_hist = hist_lines[hist_lines['stat'] == target_name][
                    ['game_date', 'player_norm', 'line']
                ].rename(columns={'line': 'real_line'})
                merged = (test_with_features[['game_date', 'player_norm']]
                          .reset_index(drop=True)
                          .merge(stat_hist, on=['game_date', 'player_norm'], how='left'))
                real_line_vals = merged['real_line'].values.astype(float)

            anchor_arr = anchor_vals.values if len(anchor_vals) == len(test_with_features) else np.full(len(test_with_features), np.nan)
            line = np.where(~np.isnan(real_line_vals), real_line_vals, anchor_arr)

            real_mask = ~np.isnan(real_line_vals)
            n_real = real_mask.sum()
            real_pct = real_mask.mean() * 100

            if n_real < 20:
                print(f"\n  {target_name.upper()}: only {n_real} real lines — skipping (need 20+)")
                continue

            # Only evaluate against real lines — no L10 fallback (which inflates accuracy)
            line_real       = real_line_vals[real_mask]
            y_pred_real     = y_pred_actual[real_mask]
            y_true_real     = y_true.values[real_mask]

            pred_over   = y_pred_real > line_real
            actual_over = y_true_real > line_real
            accuracy    = (pred_over == actual_over).mean()

            # Over/under breakdown
            over_mask  = pred_over
            under_mask = ~pred_over
            over_acc   = (actual_over[over_mask]).mean()  if over_mask.sum()  > 0 else None
            under_acc  = (~actual_over[under_mask]).mean() if under_mask.sum() > 0 else None

            # Volatility-normalized high-confidence picks
            edge   = y_pred_real - line_real
            std_col = f'{target_name}_std'
            if std_col in test_with_features.columns:
                std_vals = test_with_features[std_col].values[real_mask]
                std_vals = np.clip(std_vals, 0.5, None)
                edge_z   = edge / std_vals
                high_edge = (
                    (np.abs(edge_z) >= CONFIG['edge_z_threshold']) &
                    (np.abs(edge)   >= CONFIG['edge_threshold'])
                )
            else:
                high_edge = np.abs(edge) >= CONFIG['edge_threshold']

            # Apply grade filter: only Grade A/B+ (consensus: L10 agrees with edge direction)
            l10_col = f'{target_name}_l10'
            if l10_col in test_with_features.columns:
                l10_real    = test_with_features[l10_col].values[real_mask]
                l10_edge    = l10_real - line_real
                l10_agrees  = np.sign(l10_edge) == np.sign(edge)
                grade_filter = l10_agrees & high_edge
            else:
                grade_filter = high_edge

            if grade_filter.sum() > 0:
                high_edge_acc = (pred_over[grade_filter] == actual_over[grade_filter]).mean()
                high_edge_n   = int(grade_filter.sum())
                he_over_n  = int((pred_over & grade_filter).sum())
                he_under_n = int((~pred_over & grade_filter).sum())
                he_over_acc  = actual_over[pred_over & grade_filter].mean()  if he_over_n  > 0 else None
                he_under_acc = (~actual_over[~pred_over & grade_filter]).mean() if he_under_n > 0 else None
            else:
                high_edge_acc = None
                high_edge_n = he_over_n = he_under_n = 0
                he_over_acc = he_under_acc = None

            results[target_name] = {
                'mae': round(float(mae), 2),
                'real_lines': int(n_real),
                'accuracy': round(float(accuracy) * 100, 1),
                'over_accuracy':  round(float(over_acc)  * 100, 1) if over_acc  is not None else None,
                'under_accuracy': round(float(under_acc) * 100, 1) if under_acc is not None else None,
                'grade_ab_accuracy': round(float(high_edge_acc) * 100, 1) if high_edge_acc is not None else None,
                'grade_ab_bets':   high_edge_n,
                'grade_ab_over':   {'n': he_over_n,  'acc': round(float(he_over_acc)  * 100, 1) if he_over_acc  is not None else None},
                'grade_ab_under':  {'n': he_under_n, 'acc': round(float(he_under_acc) * 100, 1) if he_under_acc is not None else None},
            }

            print(f"\n  {target_name.upper()}:  ({n_real} real lines, {real_pct:.0f}% coverage)")
            print(f"    MAE:            {mae:.2f}")
            print(f"    Overall:        {accuracy:.1%}  (OVER {over_acc:.1%} | UNDER {under_acc:.1%})" if over_acc and under_acc else f"    Overall: {accuracy:.1%}")
            if high_edge_acc is not None:
                print(f"    Grade A/B+:     {high_edge_acc:.1%}  ({high_edge_n} bets)")
                if he_over_acc  is not None: print(f"      -> OVER:    {he_over_acc:.1%}  ({he_over_n} bets)")
                if he_under_acc is not None: print(f"      -> UNDER:   {he_under_acc:.1%}  ({he_under_n} bets)")
    
    return results


# --- Modern daily-card backtest override ------------------------------------

def _load_historical_lines_wide_v2(start_date: pd.Timestamp = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical lines and optional line-move history in wide player-date form."""
    lines_path = CONFIG['data_dir'] / 'historical_lines.csv'
    if not lines_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    hist_lines = pd.read_csv(lines_path)
    hist_lines['game_date'] = pd.to_datetime(hist_lines['game_date'], errors='coerce').dt.normalize()
    hist_lines = hist_lines.dropna(subset=['game_date'])
    if start_date is not None:
        hist_lines = hist_lines[hist_lines['game_date'] >= pd.Timestamp(start_date).normalize()].copy()
    if len(hist_lines) == 0:
        return pd.DataFrame(), pd.DataFrame()

    if 'player_norm' not in hist_lines.columns:
        hist_lines['player_norm'] = hist_lines['player'].apply(_normalize_name)
    hist_lines['stat'] = hist_lines['prop'].map(_ODDS_PROP_TO_STAT)
    hist_lines = hist_lines.dropna(subset=['stat'])
    hist_lines = hist_lines.sort_values(['game_date', 'player_norm', 'stat'])
    hist_lines = hist_lines.drop_duplicates(subset=['game_date', 'player_norm', 'stat'], keep='last')

    line_wide = hist_lines.pivot_table(
        index=['game_date', 'player_norm'], columns='stat', values='line', aggfunc='last'
    ).reset_index()
    line_wide.columns = [
        f'{col}_line' if col not in {'game_date', 'player_norm'} else col
        for col in line_wide.columns
    ]

    move_rows = []
    for game_day in hist_lines['game_date'].dt.strftime('%Y-%m-%d').unique():
        morning_path = CONFIG['data_dir'] / f'lines_morning_{game_day}.csv'
        if not morning_path.exists():
            continue
        try:
            morning = pd.read_csv(morning_path)
        except Exception:
            continue
        if len(morning) == 0:
            continue
        morning['player_norm'] = morning.get('player_norm', morning['player'].apply(_normalize_name))
        morning['stat'] = morning['prop'].map(_ODDS_PROP_TO_STAT)
        morning = morning.dropna(subset=['stat'])
        morning = morning.sort_values(['player_norm', 'stat']).drop_duplicates(subset=['player_norm', 'stat'], keep='last')
        current = hist_lines[hist_lines['game_date'] == pd.Timestamp(game_day)][['player_norm', 'stat', 'line']].copy()
        merged = current.merge(
            morning[['player_norm', 'stat', 'line']].rename(columns={'line': 'morning_line'}),
            on=['player_norm', 'stat'],
            how='left',
        )
        merged['game_date'] = pd.Timestamp(game_day)
        merged['line_move'] = pd.to_numeric(merged['line'], errors='coerce') - pd.to_numeric(merged['morning_line'], errors='coerce')
        move_rows.append(merged[['game_date', 'player_norm', 'stat', 'line_move']])

    if not move_rows:
        return line_wide, pd.DataFrame()

    move_df = pd.concat(move_rows, ignore_index=True)
    move_wide = move_df.pivot_table(
        index=['game_date', 'player_norm'], columns='stat', values='line_move', aggfunc='last'
    ).reset_index()
    move_wide.columns = [
        f'{col}_line_move' if col not in {'game_date', 'player_norm'} else col
        for col in move_wide.columns
    ]
    return line_wide, move_wide


def _build_backtest_feature_frame_v2(df: pd.DataFrame, using_advanced: bool) -> pd.DataFrame:
    """Build historical per-game features for backtesting the current model stack."""
    working = df.copy()
    working['game_date'] = pd.to_datetime(working['game_date'], errors='coerce').dt.normalize()
    if 'player_norm' not in working.columns:
        working['player_norm'] = working['player'].apply(_normalize_name)

    base_feature_cols = ['games_played', 'pts_l10', 'trb_l10', 'ast_l10', 'mp_l10']
    if any(col not in working.columns for col in base_feature_cols):
        print("  Building base features for backtest...")
        working = create_features(working)

    working = _build_dynamic_context_features(working)
    if using_advanced:
        print("  Building advanced historical features for backtest...")
        working = _build_advanced_features(working)
        working = _merge_layer_features(working)

    if 'player_norm' not in working.columns:
        working['player_norm'] = working['player'].apply(_normalize_name)

    for stat in ['pra', 'pr', 'pa']:
        if stat not in working.columns:
            if stat == 'pra':
                working[stat] = working.get('pts', 0).fillna(0) + working.get('trb', 0).fillna(0) + working.get('ast', 0).fillna(0)
            elif stat == 'pr':
                working[stat] = working.get('pts', 0).fillna(0) + working.get('trb', 0).fillna(0)
            elif stat == 'pa':
                working[stat] = working.get('pts', 0).fillna(0) + working.get('ast', 0).fillna(0)

    return working


def _predict_historical_frame_v2(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Generate historical row-by-row predictions using the current saved model artifacts."""
    adv_models, adv_feature_cols, adv_classifiers, adv_selected, adv_edge_models, adv_meta_models, adv_market_sets = _load_advanced_models()
    using_advanced = adv_models is not None

    models_path = CONFIG['models_dir'] / 'models.pkl'
    if not models_path.exists() and not using_advanced:
        raise FileNotFoundError("No trained models found. Run 'train' first.")

    models, feature_cols, is_residual = {}, [], False
    if models_path.exists():
        with open(models_path, 'rb') as f:
            saved = pickle.load(f)
        models = saved['models']
        feature_cols = saved['feature_cols']
        is_residual = saved.get('residual', False)

    if using_advanced:
        models = adv_models
        feature_cols = adv_feature_cols
        is_residual = False
        clf_models = adv_classifiers
        clf_selected = adv_selected
    else:
        clf_models = {}
        clf_selected = {}
        adv_edge_models = {}
        adv_meta_models = {}
        adv_market_sets = {'edge': {}, 'meta': {}}

    bias_path = CONFIG['models_dir'] / ('bias_advanced.json' if using_advanced else 'bias.json')
    pred_biases = {}
    if bias_path.exists():
        with open(bias_path) as f:
            pred_biases = json.load(f)

    minutes_model = None
    minutes_feature_cols = None
    mp_model_path = CONFIG['models_dir'] / 'minutes_deviation_model.pkl'
    if mp_model_path.exists():
        with open(mp_model_path, 'rb') as f:
            mp_saved = pickle.load(f)
        minutes_model = mp_saved.get('model')
        minutes_feature_cols = mp_saved.get('feature_cols')

    working = feature_df.copy()
    if 'player_norm' not in working.columns:
        working['player_norm'] = working['player'].apply(_normalize_name)
    primitive_fp_integrated = using_advanced and any(
        isinstance(models.get(stat), dict) and models.get(stat, {}).get('kind') == 'positive_ridge_blend_regressor'
        for stat in ['pts', 'trb', 'ast']
    )
    required_cols = list(feature_cols or [])
    if using_advanced:
        for target_name, model in models.items():
            target_features = clf_selected.get(target_name)
            if not target_features and isinstance(model, dict):
                target_features = model.get('feature_names')
            if not target_features and hasattr(model, 'get_booster'):
                target_features = model.get_booster().feature_names
            if target_features:
                required_cols.extend(target_features)
    if minutes_feature_cols:
        required_cols.extend(minutes_feature_cols)
    required_cols = list(dict.fromkeys(required_cols))

    missing_cols = [c for c in required_cols if c not in working.columns]
    for col in missing_cols:
        working[col] = 0

    mp_deviation_pred = np.zeros(len(working))
    if minutes_model is not None and minutes_feature_cols:
        minute_cols = [c for c in minutes_feature_cols if c in working.columns]
        if minute_cols:
            mp_deviation_pred = minutes_model.predict(working[minute_cols].fillna(0))

    pred_df = working[['game_date', 'player', 'player_norm']].copy()
    if 'team' in working.columns:
        pred_df['team'] = working['team']
    pred_df['mp_expected'] = pd.to_numeric(working.get('mp_expected', working.get('mp_l10', 0)), errors='coerce')

    for target_name, model in models.items():
        if target_name not in ['pts', 'trb', 'ast', 'pra', 'pr', 'pa', 'stl', 'blk', 'tov']:
            continue

        l10_col = f'{target_name}_l10'
        l5_col = f'{target_name}_l5'
        per_min_col = f'{target_name}_per_min'
        l10_src = working[l10_col] if l10_col in working.columns else pd.Series(np.nan, index=working.index)
        l10_vals = pd.to_numeric(l10_src, errors='coerce')
        anchor_vals = _series_prop_feature(working, target_name, 'best_estimate')
        anchor_vals = anchor_vals.fillna(_series_prop_feature(working, target_name, 'recency_anchor'))
        anchor_vals = anchor_vals.fillna(l10_vals)
        pred_df[l10_col] = l10_vals.round(1)
        pred_df[f'{target_name}_anchor'] = anchor_vals.round(1)
        if l5_col in working.columns:
            pred_df[l5_col] = pd.to_numeric(working[l5_col], errors='coerce').round(1)

        reg_features = None
        if using_advanced:
            reg_features = clf_selected.get(target_name)
            if not reg_features and isinstance(model, dict):
                reg_features = model.get('feature_names')
            if not reg_features and hasattr(model, 'get_booster'):
                reg_features = model.get_booster().feature_names
        else:
            reg_features = feature_cols
        reg_features = [c for c in (reg_features or []) if c in working.columns]
        X_reg = working[reg_features].fillna(0) if reg_features else working.fillna(0)

        fp_series = None
        if primitive_fp_integrated and target_name in {'pts', 'trb', 'ast'} and isinstance(model, dict) and model.get('kind') == 'positive_ridge_blend_regressor':
            fp_series = _series_fp_projection(working, target_name)
        raw_pred = _predict(model, X_reg, fp_input=fp_series)
        if is_residual:
            base_pred = anchor_vals.fillna(0).values + raw_pred
        else:
            anchor_arr = anchor_vals.fillna(0).values
            anchor_mask = ~anchor_vals.isna().values
            if primitive_fp_integrated and target_name in {'pts', 'trb', 'ast'} and isinstance(model, dict) and model.get('kind') == 'positive_ridge_blend_regressor':
                base_pred = raw_pred
            else:
                base_pred = np.where(anchor_mask, raw_pred * 0.7 + anchor_arr * 0.3, raw_pred)

        mp_l10_src = working['mp_l10'] if 'mp_l10' in working.columns else pd.Series(0, index=working.index)
        per_min_src = working[per_min_col] if per_min_col in working.columns else pd.Series(0, index=working.index)
        mp_l10 = pd.to_numeric(mp_l10_src, errors='coerce').fillna(0).values
        per_min_rate = pd.to_numeric(per_min_src, errors='coerce').fillna(0).values
        if np.any(mp_l10 > 0) and np.any(mp_deviation_pred != 0):
            minutes_adj = per_min_rate * mp_deviation_pred
            cap = np.abs(base_pred) * 0.15
            minutes_adj = np.clip(minutes_adj, -cap, cap)
            base_pred = base_pred + minutes_adj

        bias_correction = float(pred_biases.get(target_name, 0.0))
        final_pred = np.maximum(0, base_pred - bias_correction)
        pred_df[f'{target_name}_pred'] = np.round(final_pred, 1)

        clf = clf_models.get(target_name)
        if clf is not None:
            try:
                sel = clf_selected.get(target_name, reg_features or feature_cols)
                avail = [c for c in sel if c in working.columns]
                if avail:
                    X_clf = working[avail].fillna(0)
                    pred_df[f'{target_name}_clf_prob'] = np.round(clf.predict_proba(X_clf)[:, 1], 4)
            except Exception:
                pass

    pred_df['pra_pred'] = (pred_df.get('pts_pred', 0).fillna(0) + pred_df.get('trb_pred', 0).fillna(0) + pred_df.get('ast_pred', 0).fillna(0)).round(1)
    pred_df['pr_pred'] = (pred_df.get('pts_pred', 0).fillna(0) + pred_df.get('trb_pred', 0).fillna(0)).round(1)
    pred_df['pa_pred'] = (pred_df.get('pts_pred', 0).fillna(0) + pred_df.get('ast_pred', 0).fillna(0)).round(1)

    if {'pts_l10', 'trb_l10', 'ast_l10'}.issubset(pred_df.columns):
        pred_df['pra_l10'] = (pred_df['pts_l10'].fillna(0) + pred_df['trb_l10'].fillna(0) + pred_df['ast_l10'].fillna(0)).round(1)
        pred_df['pr_l10'] = (pred_df['pts_l10'].fillna(0) + pred_df['trb_l10'].fillna(0)).round(1)
        pred_df['pa_l10'] = (pred_df['pts_l10'].fillna(0) + pred_df['ast_l10'].fillna(0)).round(1)
    if {'pts_l5', 'trb_l5', 'ast_l5'}.issubset(pred_df.columns):
        pred_df['pra_l5'] = (pred_df['pts_l5'].fillna(0) + pred_df['trb_l5'].fillna(0) + pred_df['ast_l5'].fillna(0)).round(1)
        pred_df['pr_l5'] = (pred_df['pts_l5'].fillna(0) + pred_df['trb_l5'].fillna(0)).round(1)
        pred_df['pa_l5'] = (pred_df['pts_l5'].fillna(0) + pred_df['ast_l5'].fillna(0)).round(1)

    pred_df['game_total'] = None
    pred_df['team_spread'] = 0
    pred_df['game_total_scale'] = 1.0
    pred_df['injury_adj'] = False
    pred_df['injury_adj_strength'] = 0.0
    pred_df['teammates_out'] = ''
    return pred_df


def _score_backtest_pick_v2(row: pd.Series) -> Tuple[str, Optional[float]]:
    prop = str(row.get('prop', '')).strip().lower()
    actual = row.get(f'{prop}_actual')
    if actual is None or pd.isna(actual):
        return 'DNP', None
    line = pd.to_numeric(row.get('line'), errors='coerce')
    actual = float(actual)
    if pd.isna(line):
        return 'DNP', actual
    if actual == float(line):
        return 'PUSH', actual
    direction = str(row.get('direction', '')).upper()
    win = (direction == 'OVER' and actual > float(line)) or (direction == 'UNDER' and actual < float(line))
    return ('WIN' if win else 'LOSS'), actual


def _summarize_pick_records_v2(picks_df: pd.DataFrame) -> Dict[str, object]:
    if len(picks_df) == 0:
        return {'picks': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'dnps': 0, 'hit_rate': None}
    wins = int((picks_df['result'] == 'WIN').sum())
    losses = int((picks_df['result'] == 'LOSS').sum())
    pushes = int((picks_df['result'] == 'PUSH').sum())
    dnps = int((picks_df['result'] == 'DNP').sum())
    graded = wins + losses
    return {
        'picks': int(len(picks_df)),
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'dnps': dnps,
        'hit_rate': round(wins / graded, 4) if graded else None,
    }


def _daily_card_backtest_v2(pred_df: pd.DataFrame,
                            actual_df: pd.DataFrame,
                            quality_policy: Optional[Dict[str, Dict]] = None) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Simulate daily top-card picks using the current pick filter/ranking."""
    if len(pred_df) == 0:
        empty = pd.DataFrame()
        return {'days': 0, 'days_with_picks': 0, 'top_k': {}, 'by_prop': {}, 'by_direction': {}}, empty, empty

    actual = actual_df[['game_date', 'player_norm', 'pts', 'trb', 'ast', 'stl', 'blk', 'tov']].copy()
    actual['pra_actual'] = actual['pts'].fillna(0) + actual['trb'].fillna(0) + actual['ast'].fillna(0)
    actual['pr_actual'] = actual['pts'].fillna(0) + actual['trb'].fillna(0)
    actual['pa_actual'] = actual['pts'].fillna(0) + actual['ast'].fillna(0)
    actual = actual.rename(columns={
        'pts': 'pts_actual',
        'trb': 'trb_actual',
        'ast': 'ast_actual',
        'stl': 'stl_actual',
        'blk': 'blk_actual',
        'tov': 'tov_actual',
    })

    detail_frames = []
    daily_summary = pd.DataFrame({'game_date': sorted(pred_df['game_date'].dropna().dt.strftime('%Y-%m-%d').unique())})
    for game_date, day_pred in pred_df.sort_values(['game_date', 'player'], ascending=[True, True]).groupby('game_date'):
        day_picks = filter_best_picks(day_pred.copy(), quality_policy=quality_policy)
        if len(day_picks) == 0:
            continue
        day_picks['game_date'] = pd.Timestamp(game_date)
        day_picks['player_norm'] = day_picks['player'].apply(_normalize_name)
        merged = day_picks.merge(actual, on=['game_date', 'player_norm'], how='left')
        scored = merged.copy()
        results = scored.apply(_score_backtest_pick_v2, axis=1, result_type='expand')
        scored['result'] = results[0]
        scored['actual'] = results[1]
        scored['game_date'] = pd.to_datetime(scored['game_date']).dt.strftime('%Y-%m-%d')
        detail_frames.append(scored)

    if detail_frames:
        detail_df = pd.concat(detail_frames, ignore_index=True)
    else:
        detail_df = pd.DataFrame(columns=['game_date', 'player', 'prop', 'direction', 'line', 'prediction', 'edge', 'confidence', 'rank_score', 'result', 'actual'])

    top_k_summary = {}
    if len(detail_df) > 0:
        counts_by_day = detail_df.groupby('game_date').size()
        for k in [1, 3, 5, 10]:
            qualified_days = counts_by_day[counts_by_day >= k].index.tolist()
            card = (
                detail_df[detail_df['game_date'].isin(qualified_days)]
                .sort_values(['game_date', 'rank_score'], ascending=[True, False])
                .groupby('game_date')
                .head(k)
                .copy()
            )
            summary = _summarize_pick_records_v2(card)
            summary['days'] = int(len(qualified_days))
            summary['avg_picks_per_day'] = round(float(len(card) / len(qualified_days)), 2) if qualified_days else None
            top_k_summary[f'top_{k}'] = summary
            graded_card = card[card['result'].isin(['WIN', 'LOSS'])]
            hit_map = graded_card.groupby('game_date').apply(lambda g: round(float((g['result'] == 'WIN').mean()), 4)).to_dict() if len(graded_card) > 0 else {}
            daily_summary[f'top_{k}_picks'] = daily_summary['game_date'].map(card.groupby('game_date').size().to_dict()).fillna(0).astype(int)
            daily_summary[f'top_{k}_wins'] = daily_summary['game_date'].map(card[card['result'] == 'WIN'].groupby('game_date').size().to_dict()).fillna(0).astype(int)
            daily_summary[f'top_{k}_losses'] = daily_summary['game_date'].map(card[card['result'] == 'LOSS'].groupby('game_date').size().to_dict()).fillna(0).astype(int)
            daily_summary[f'top_{k}_pushes'] = daily_summary['game_date'].map(card[card['result'] == 'PUSH'].groupby('game_date').size().to_dict()).fillna(0).astype(int)
            daily_summary[f'top_{k}_hit_rate'] = daily_summary['game_date'].map(hit_map)
    else:
        top_k_summary = {f'top_{k}': {'picks': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'dnps': 0, 'hit_rate': None, 'days': 0, 'avg_picks_per_day': None} for k in [1, 3, 5, 10]}

    by_prop = {str(prop): _summarize_pick_records_v2(group) for prop, group in detail_df.groupby('prop')} if len(detail_df) > 0 else {}
    by_direction = {str(direction): _summarize_pick_records_v2(group) for direction, group in detail_df.groupby('direction')} if len(detail_df) > 0 else {}

    summary = {
        'days': int(daily_summary['game_date'].nunique()) if len(daily_summary) > 0 else 0,
        'days_with_picks': int(detail_df['game_date'].nunique()) if len(detail_df) > 0 else 0,
        'all_generated': _summarize_pick_records_v2(detail_df),
        'top_k': top_k_summary,
        'by_prop': by_prop,
        'by_direction': by_direction,
    }
    return summary, detail_df, daily_summary


def backtest(df: pd.DataFrame, test_days: int = 30) -> Dict:
    """Backtest current model artifacts on recent real-line slates, including daily cards."""

    print("\nBacktesting current model stack...")
    working = df.copy()
    working['game_date'] = pd.to_datetime(working['game_date'], errors='coerce').dt.normalize()
    working = working.sort_values('game_date')

    max_date = working['game_date'].max()
    test_start = (max_date - timedelta(days=test_days)).normalize()
    print(f"  Test window: {test_start.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({test_days} days)")

    prepared_path = CONFIG['data_dir'] / 'training_dataset_advanced_from_enriched.csv'
    backtest_mode = 'artifact_replay'
    backtest_quality_policy = None
    live_edge_models = {}
    live_meta_models = {}
    live_market_sets = {'edge': {}, 'meta': {}}
    live_prob_sidecars = {}
    live_prob_calibrators = {}

    if prepared_path.exists():
        from train_advanced_models import AdvancedPropModel as CutoffAdvancedPropModel, _load_lines_lookup as _load_train_lines_lookup

        prepared_df = pd.read_csv(prepared_path, low_memory=False)
        prepared_df['game_date'] = pd.to_datetime(prepared_df['game_date'], errors='coerce').dt.normalize()
        prepared_df = prepared_df.dropna(subset=['game_date']).copy()
        if 'player_norm' not in prepared_df.columns:
            prepared_df['player_norm'] = prepared_df['player'].apply(_normalize_name)

        train_feature_df = prepared_df[prepared_df['game_date'] < test_start].copy()
        feature_df = prepared_df[prepared_df['game_date'] >= test_start].copy()
        if 'games_played' in feature_df.columns:
            feature_df = feature_df[feature_df['games_played'] >= CONFIG['min_games']].copy()
        print(f"  Prepared train rows: {len(train_feature_df):,}")
        print(f"  Prepared test rows after min-games filter: {len(feature_df):,}")

        trainer = CutoffAdvancedPropModel(model_dir=str(CONFIG['output_dir'] / 'backtest_temp_models'))
        train_lines_lookup = _load_train_lines_lookup()
        train_results = {}
        for target_name in trainer.TARGETS.keys():
            try:
                train_results[target_name] = trainer.train_model(
                    train_feature_df, target_name, lines_lookup=train_lines_lookup
                )
            except Exception as exc:
                print(f"Error training {target_name}: {exc}")
                train_results[target_name] = {'error': str(exc)}
        train_errors = [k for k, v in train_results.items() if isinstance(v, dict) and v.get('error')]
        if train_errors:
            print(f"  WARNING: cutoff training errors for {', '.join(train_errors)}")
        trainer.fit_probability_calibrators()

        local_edge_analysis = {
            target_name: result.get('real_line_summary', {})
            for target_name, result in train_results.items()
            if isinstance(result, dict) and result.get('real_line_summary')
        }
        backtest_quality_policy = _build_target_quality_policy(
            results={k: v for k, v in train_results.items() if isinstance(v, dict)},
            edge_analysis=local_edge_analysis,
        )

        pred_df = feature_df[['game_date', 'player', 'team', 'player_norm']].copy()
        pred_df['mp_expected'] = pd.to_numeric(feature_df.get('mp_expected', feature_df.get('mp_l10', 0)), errors='coerce')
        for target_name in ['pts', 'trb', 'ast', 'pra', 'pr', 'pa', 'stl', 'blk', 'tov']:
            if target_name not in trainer.models:
                continue
            pred_df[f'{target_name}_pred'] = np.round(trainer.predict(feature_df, target_name), 1)
            if target_name in trainer.classifiers:
                pred_df[f'{target_name}_clf_prob'] = np.round(trainer.predict_proba(feature_df, target_name), 4)
            l10_col = f'{target_name}_l10'
            l5_col = f'{target_name}_l5'
            anchor_col = f'{target_name}_anchor'
            if l10_col in feature_df.columns:
                pred_df[l10_col] = pd.to_numeric(feature_df[l10_col], errors='coerce').round(1)
            if l5_col in feature_df.columns:
                pred_df[l5_col] = pd.to_numeric(feature_df[l5_col], errors='coerce').round(1)
            anchor_vals = _series_prop_feature(feature_df, target_name, 'best_estimate')
            anchor_vals = anchor_vals.fillna(_series_prop_feature(feature_df, target_name, 'recency_anchor'))
            anchor_vals = anchor_vals.fillna(pd.to_numeric(feature_df[l10_col], errors='coerce') if l10_col in feature_df.columns else np.nan)
            pred_df[anchor_col] = pd.to_numeric(anchor_vals, errors='coerce').round(1)

        if {'pts_l10', 'trb_l10', 'ast_l10'}.issubset(pred_df.columns):
            pred_df['pra_l10'] = (pred_df['pts_l10'].fillna(0) + pred_df['trb_l10'].fillna(0) + pred_df['ast_l10'].fillna(0)).round(1)
            pred_df['pr_l10'] = (pred_df['pts_l10'].fillna(0) + pred_df['trb_l10'].fillna(0)).round(1)
            pred_df['pa_l10'] = (pred_df['pts_l10'].fillna(0) + pred_df['ast_l10'].fillna(0)).round(1)
        if {'pts_l5', 'trb_l5', 'ast_l5'}.issubset(pred_df.columns):
            pred_df['pra_l5'] = (pred_df['pts_l5'].fillna(0) + pred_df['trb_l5'].fillna(0) + pred_df['ast_l5'].fillna(0)).round(1)
            pred_df['pr_l5'] = (pred_df['pts_l5'].fillna(0) + pred_df['trb_l5'].fillna(0)).round(1)
            pred_df['pa_l5'] = (pred_df['pts_l5'].fillna(0) + pred_df['ast_l5'].fillna(0)).round(1)

        pred_df['game_total'] = None
        pred_df['team_spread'] = 0
        pred_df['game_total_scale'] = 1.0
        pred_df['injury_adj'] = False
        pred_df['injury_adj_strength'] = 0.0
        pred_df['teammates_out'] = ''
        live_edge_models = trainer.edge_models
        live_meta_models = trainer.meta_models
        live_market_sets = trainer.market_feature_sets
        live_prob_sidecars = trainer.probability_sidecars
        live_prob_calibrators = trainer.probability_calibrators
        using_advanced = True
        backtest_mode = 'cutoff_trained'
    else:
        adv_models, _, _, _, live_edge_models, live_meta_models, live_market_sets = _load_advanced_models()
        using_advanced = adv_models is not None
        live_prob_sidecars = _load_probability_sidecars() if using_advanced else {}
        live_prob_calibrators = _load_probability_calibrators() if using_advanced else {}
        feature_df = _build_backtest_feature_frame_v2(working, using_advanced=using_advanced)
        feature_df = feature_df[feature_df['game_date'] >= test_start].copy()
        feature_df = feature_df[feature_df['games_played'] >= CONFIG['min_games']].copy()
        print(f"  Feature rows after min-games filter: {len(feature_df):,}")
        pred_df = _predict_historical_frame_v2(feature_df)

    line_wide, move_wide = _load_historical_lines_wide_v2(start_date=test_start)
    if len(line_wide) == 0:
        raise FileNotFoundError("No historical_lines.csv found for the requested backtest window.")
    n_real_lines = int((line_wide.drop(columns=['game_date', 'player_norm']).notna()).sum().sum())
    print(f"  Real lines available in test window: {n_real_lines:,}")

    pred_df = pred_df.merge(line_wide, on=['game_date', 'player_norm'], how='left')
    if len(move_wide) > 0:
        pred_df = pred_df.merge(move_wide, on=['game_date', 'player_norm'], how='left')
    pred_df = _attach_probability_outputs(
        pred_df,
        feature_df,
        live_prob_sidecars,
        live_prob_calibrators,
    )
    pred_df = _attach_market_model_scores(
        pred_df,
        feature_df,
        live_edge_models,
        live_meta_models,
        live_market_sets,
    )

    row_results = {}
    for target_name in ['pts', 'trb', 'ast', 'pra', 'pr', 'pa', 'stl', 'blk', 'tov']:
        pred_col = f'{target_name}_pred'
        line_col = f'{target_name}_line'
        if pred_col not in pred_df.columns or line_col not in pred_df.columns or target_name not in feature_df.columns:
            continue

        eval_df = pred_df[['game_date', 'player_norm', pred_col, line_col]].merge(
            feature_df[['game_date', 'player_norm', target_name]],
            on=['game_date', 'player_norm'],
            how='left',
        )
        eval_df = eval_df.dropna(subset=[pred_col, line_col, target_name])
        if len(eval_df) == 0:
            continue

        y_true = pd.to_numeric(eval_df[target_name], errors='coerce').values
        y_pred = pd.to_numeric(eval_df[pred_col], errors='coerce').values
        line_vals = pd.to_numeric(eval_df[line_col], errors='coerce').values

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        pred_over = y_pred > line_vals
        actual_over = y_true > line_vals
        accuracy = float((pred_over == actual_over).mean())
        over_mask = pred_over
        under_mask = ~pred_over
        over_acc = float(actual_over[over_mask].mean()) if over_mask.sum() > 0 else None
        under_acc = float((~actual_over[under_mask]).mean()) if under_mask.sum() > 0 else None

        row_results[target_name] = {
            'mae': round(float(mae), 3),
            'rmse': round(float(rmse), 3),
            'real_lines': int(len(eval_df)),
            'accuracy': round(accuracy * 100, 1),
            'over_accuracy': round(over_acc * 100, 1) if over_acc is not None else None,
            'under_accuracy': round(under_acc * 100, 1) if under_acc is not None else None,
        }

        print(f"\n  {target_name.upper()}: {len(eval_df)} real lines")
        print(f"    MAE: {mae:.2f}")
        print(f"    Overall: {accuracy:.1%}")
        if over_acc is not None and under_acc is not None:
            print(f"    OVER {over_acc:.1%} | UNDER {under_acc:.1%}")
        elif over_acc is not None:
            print(f"    OVER {over_acc:.1%}")
        elif under_acc is not None:
            print(f"    UNDER {under_acc:.1%}")

    daily_card_summary, card_detail_df, daily_summary_df = _daily_card_backtest_v2(
        pred_df,
        feature_df,
        quality_policy=backtest_quality_policy,
    )

    print("\nDaily-card backtest:")
    for label, summary in daily_card_summary.get('top_k', {}).items():
        hit_rate = summary.get('hit_rate')
        hit_txt = f"{hit_rate:.1%}" if hit_rate is not None else "n/a"
        print(f"  {label}: {summary.get('wins', 0)}W-{summary.get('losses', 0)}L ({hit_txt}) across {summary.get('days', 0)} qualifying days")

    return {
        'metadata': {
            'test_days': int(test_days),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': max_date.strftime('%Y-%m-%d'),
            'using_advanced_models': bool(using_advanced),
            'mode': backtest_mode,
            'feature_rows': int(len(feature_df)),
            'real_line_rows': int(n_real_lines),
            'notes': [
                'Daily-card backtest uses the current pick filter on real historical lines.',
                'When training_dataset_advanced_from_enriched.csv is available, models are retrained on rows before the test window for an out-of-sample cutoff test.',
                'When cutoff training is available, the pick policy is rebuilt from training-window metrics instead of the live full-sample artifacts.',
                'Live-only pregame extras such as injury boosts, historical game-total scaling, and first-principles blends are not reconstructed for older slates unless archived separately.',
            ],
        },
        'row_level': row_results,
        'daily_card_backtest': daily_card_summary,
        'detail_records': card_detail_df.to_dict(orient='records'),
        'daily_summary': daily_summary_df.to_dict(orient='records'),
    }


# =============================================================================
# OUTPUT
# =============================================================================

_PP_MULTIPLIERS = {1: 2.0, 2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0}

def _calc_payout(wager: float, n_picks: int) -> float:
    mult = _PP_MULTIPLIERS.get(n_picks, n_picks * 2.0)
    return round(wager * mult, 2)


def _power_play_break_even_prob(n_picks: int) -> float:
    mult = _PP_MULTIPLIERS.get(n_picks, n_picks * 2.0)
    return float((1.0 / max(mult, 1e-9)) ** (1.0 / max(int(n_picks), 1)))


def _coerce_pick_side_probability(row: pd.Series) -> float:
    dir_prob = row.get('dir_prob', row.get('ou_prob'))
    if dir_prob is not None and not pd.isna(dir_prob):
        prob = float(dir_prob)
    else:
        conf = pd.to_numeric(pd.Series([row.get('confidence')]), errors='coerce').iloc[0]
        prob = float(conf) / 100.0 if pd.notna(conf) else 0.50
    return float(np.clip(prob, 0.01, 0.99))


def _estimate_leg_correlation(left: pd.Series, right: pd.Series) -> float:
    left_player = _normalize_name(left.get('player', ''))
    right_player = _normalize_name(right.get('player', ''))
    if left_player and left_player == right_player:
        return 0.70

    left_team = str(left.get('team', '') or '').strip().upper()
    right_team = str(right.get('team', '') or '').strip().upper()
    if left_team and left_team == right_team:
        return 0.35

    left_opp = str(left.get('opponent', '') or '').strip().upper()
    right_opp = str(right.get('opponent', '') or '').strip().upper()
    if (left_team and left_team == right_opp) or (right_team and right_team == left_opp):
        return 0.15

    return 0.0


def _estimate_joint_probabilities(subset: pd.DataFrame) -> Dict[str, float]:
    if subset is None or len(subset) == 0:
        return {
            'joint_prob_raw': 0.0,
            'estimated_correlation': 0.0,
            'joint_prob_adj': 0.0,
        }

    probs = [_coerce_pick_side_probability(row) for _, row in subset.iterrows()]
    joint_raw = float(np.prod(probs))
    if len(probs) == 1:
        return {
            'joint_prob_raw': joint_raw,
            'estimated_correlation': 0.0,
            'joint_prob_adj': joint_raw,
        }

    running_joint = probs[0]
    pair_rhos: List[float] = []
    rows = [row for _, row in subset.iterrows()]
    for idx in range(1, len(rows)):
        current_prob = probs[idx]
        prior_rhos = [_estimate_leg_correlation(rows[idx], rows[j]) for j in range(idx)]
        pair_rhos.extend(prior_rhos)
        rho = float(np.mean(prior_rhos)) if prior_rhos else 0.0
        variance_term = max(running_joint * (1.0 - running_joint) * current_prob * (1.0 - current_prob), 0.0)
        running_joint = running_joint * current_prob + rho * math.sqrt(variance_term)
        running_joint = float(np.clip(running_joint, 0.0, 0.999))

    return {
        'joint_prob_raw': joint_raw,
        'estimated_correlation': float(np.mean(pair_rhos)) if pair_rhos else 0.0,
        'joint_prob_adj': running_joint,
    }


def _sort_betslip_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates is None or len(candidates) == 0:
        return pd.DataFrame()

    working = candidates.copy()
    working['_selection_priority'] = pd.to_numeric(working.get('selection_priority'), errors='coerce').fillna(0.0)
    working['_dir_prob'] = [
        _coerce_pick_side_probability(row)
        for _, row in working.iterrows()
    ]
    working['_confidence'] = pd.to_numeric(working.get('confidence'), errors='coerce').fillna(0.0)
    working['_edge_abs'] = pd.to_numeric(working.get('edge'), errors='coerce').fillna(0.0).abs()
    working = working.sort_values(
        ['_selection_priority', '_dir_prob', '_confidence', '_edge_abs'],
        ascending=False,
    )

    # Keep one prop per player in slips. This preserves the existing bankroll rule
    # while still allowing correlation-aware selection across teams/games.
    deduped = []
    seen_players = set()
    for _, row in working.iterrows():
        player_key = _normalize_name(row.get('player', ''))
        if player_key in seen_players:
            continue
        seen_players.add(player_key)
        deduped.append(row)
    if not deduped:
        return working.iloc[0:0].copy()

    out = pd.DataFrame(deduped).reset_index(drop=True)
    return out.drop(columns=['_selection_priority', '_dir_prob', '_confidence', '_edge_abs'], errors='ignore')


def _prepare_betslip_candidate_pool(picks_df: pd.DataFrame,
                                    *,
                                    direction: Optional[str] = None,
                                    allowed_props: Optional[set] = None) -> pd.DataFrame:
    eligible_df = picks_df.copy()
    if 'pick_source' in eligible_df.columns:
        eligible_df = eligible_df[~eligible_df['pick_source'].isin(['volume_fill', 'emergency_fill'])]
    if 'confidence' in eligible_df.columns:
        conf = pd.to_numeric(eligible_df['confidence'], errors='coerce').fillna(0.0)
        eligible_df = eligible_df[conf >= 70.0]
    if direction is not None and 'direction' in eligible_df.columns:
        eligible_df = eligible_df[eligible_df['direction'].astype(str).str.upper() == str(direction).upper()]
    if allowed_props is not None and 'prop' in eligible_df.columns:
        prop_set = {str(prop).upper() for prop in allowed_props}
        eligible_df = eligible_df[eligible_df['prop'].astype(str).str.upper().isin(prop_set)]
    return _sort_betslip_candidates(eligible_df)


def _select_best_power_play_subset(candidates: pd.DataFrame,
                                   size: int,
                                   wager: float,
                                   candidate_limit: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if candidates is None or len(candidates) < size:
        return pd.DataFrame(), {}

    pool = candidates.head(candidate_limit).reset_index(drop=True)
    best_subset = pd.DataFrame()
    best_metrics: Dict[str, float] = {}
    best_score = -float('inf')
    best_prob = -float('inf')

    for combo in combinations(range(len(pool)), size):
        subset = pool.iloc[list(combo)].reset_index(drop=True)
        metrics = _estimate_joint_probabilities(subset)
        gross_mult = float(_PP_MULTIPLIERS.get(size, size * 2.0))
        break_even_joint_prob = 1.0 / gross_mult
        ev_per_unit = metrics['joint_prob_adj'] * gross_mult - 1.0
        metrics.update({
            'break_even_joint_prob': break_even_joint_prob,
            'ev_per_unit': ev_per_unit,
            'ev_dollar': wager * ev_per_unit,
        })
        if (
            ev_per_unit > best_score or
            (math.isclose(ev_per_unit, best_score, rel_tol=1e-9, abs_tol=1e-9) and metrics['joint_prob_adj'] > best_prob)
        ):
            best_subset = subset
            best_metrics = metrics
            best_score = ev_per_unit
            best_prob = metrics['joint_prob_adj']

    return best_subset, best_metrics


def _build_betslip_row(subset: pd.DataFrame,
                       *,
                       game_date: str,
                       slip_type: str,
                       wager: float,
                       max_legs: int = 4) -> Dict[str, object]:
    metrics = _estimate_joint_probabilities(subset)
    n_legs = len(subset)
    gross_mult = float(_PP_MULTIPLIERS.get(n_legs, n_legs * 2.0))
    metrics.update({
        'break_even_joint_prob': 1.0 / gross_mult,
        'ev_per_unit': metrics['joint_prob_adj'] * gross_mult - 1.0,
        'ev_dollar': wager * (metrics['joint_prob_adj'] * gross_mult - 1.0),
    })

    row_data = {
        'game_date': game_date,
        'slip_type': slip_type,
        'wager': wager,
        'potential_payout': _calc_payout(wager, n_legs),
        'joint_prob_raw': round(metrics['joint_prob_raw'], 4),
        'estimated_correlation': round(metrics['estimated_correlation'], 3),
        'joint_prob_adj': round(metrics['joint_prob_adj'], 4),
        'break_even_joint_prob': round(metrics['break_even_joint_prob'], 4),
        'ev_per_unit': round(metrics['ev_per_unit'], 4),
        'ev_dollar': round(metrics['ev_dollar'], 2),
        'result': '',
        'actual_payout': '',
    }
    for i, (_, leg) in enumerate(subset.iterrows(), start=1):
        row_data[f'Pick_{i}_Player'] = leg['player']
        row_data[f'Pick_{i}_Prop'] = str(leg['prop']).upper()
        row_data[f'Pick_{i}_Dir'] = str(leg['direction']).upper()
        row_data[f'Pick_{i}_Line'] = leg['line']
    for i in range(len(subset) + 1, max_legs + 1):
        row_data[f'Pick_{i}_Player'] = ''
        row_data[f'Pick_{i}_Prop'] = ''
        row_data[f'Pick_{i}_Dir'] = ''
        row_data[f'Pick_{i}_Line'] = ''
    return row_data


def _grade_power_play_results(results: List[Optional[str]], wager: float) -> Tuple[Optional[str], Optional[float], int]:
    graded = [r for r in results if r in {'WIN', 'LOSS', 'PUSH'}]
    effective_picks = len(graded)
    if effective_picks == 0:
        return None, None, 0
    if any(r == 'LOSS' for r in graded):
        return 'LOSS', 0.0, effective_picks
    if all(r == 'WIN' for r in graded):
        return 'WIN', _calc_payout(wager, effective_picks), effective_picks
    return 'PUSH', float(wager), effective_picks


def log_betslips(picks_df: pd.DataFrame, game_date: str, wager: float = 20.0):
    """
    Auto-log suggested PrizePicks Power Play bet slips to betslips_history.csv.
    Uses EV-ranked candidate selection under the current Power Play payout model.
    """
    slip_path = CONFIG['output_dir'] / 'betslips_history.csv'

    # Build column schema: Pick_1..Pick_4, each with Player/Prop/Dir/Line
    MAX_LEGS = 4
    pick_cols = []
    for i in range(1, MAX_LEGS + 1):
        pick_cols += [f'Pick_{i}_Player', f'Pick_{i}_Prop', f'Pick_{i}_Dir', f'Pick_{i}_Line']
    metric_cols = [
        'joint_prob_raw',
        'estimated_correlation',
        'joint_prob_adj',
        'break_even_joint_prob',
        'ev_per_unit',
        'ev_dollar',
    ]

    rows = []
    power_pool = _prepare_betslip_candidate_pool(picks_df)
    for size in [2, 3, 4]:
        candidate_limit = {2: 6, 3: 8, 4: 10}.get(size, max(6, size * 2))
        subset, _ = _select_best_power_play_subset(power_pool, size, wager, candidate_limit)
        if len(subset) < size:
            continue
        rows.append(
            _build_betslip_row(
                subset,
                game_date=game_date,
                slip_type=f'{size}-pick Power Play',
                wager=wager,
                max_legs=MAX_LEGS,
            )
        )

    # ── UNDER-Only slip track ────────────────────────────────────────────────
    # Build dedicated slips from UNDER picks on the post-retrain high-signal props.
    # A hard confidence floor is already applied above, so STL/BLK only enter this
    # pool when they survive both the live quality policy and the >=70 betslip gate.
    _under_props = {'AST', 'TRB', 'STL', 'BLK'}
    under_pool = _prepare_betslip_candidate_pool(
        picks_df,
        direction='UNDER',
        allowed_props=_under_props,
    )

    for size in [2, 3]:
        candidate_limit = {2: 6, 3: 8}.get(size, max(6, size * 2))
        subset, _ = _select_best_power_play_subset(under_pool, size, wager, candidate_limit)
        if len(subset) < size:
            continue
        rows.append(
            _build_betslip_row(
                subset,
                game_date=game_date,
                slip_type=f'{size}-pick UNDER-Only',
                wager=wager,
                max_legs=MAX_LEGS,
            )
        )
    # ── End UNDER-Only track ─────────────────────────────────────────────────

    if not rows:
        return pd.DataFrame()

    # Define final column order
    col_order = [
        'game_date', 'slip_type', 'wager', 'potential_payout',
    ] + metric_cols + pick_cols + ['result', 'actual_payout']

    new_df = pd.DataFrame(rows)[col_order]

    if slip_path.exists():
        existing = pd.read_csv(slip_path, dtype=str)
        # Ensure existing file has all columns (backward compat with old pipe format)
        for col in col_order:
            if col not in existing.columns:
                existing[col] = ''
        # If old file still has a 'picks' column, migrate it
        if 'picks' in existing.columns:
            existing = _migrate_picks_column(existing)
        existing = existing[existing['game_date'] != game_date]  # replace today
        existing = existing.reindex(columns=col_order, fill_value='')
        combined = pd.concat([existing, new_df.astype(str)], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(slip_path, index=False)
    print(f"  Betslips logged -> {slip_path}  ({len(rows)} slips for {game_date})")
    return new_df.reset_index(drop=True)


def backtest_under_only_slips(history_df: Optional[pd.DataFrame] = None, wager: float = 20.0) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Replay current under-only Power Play slip rules against graded picks_history rows."""
    if history_df is None:
        history_path = CONFIG['output_dir'] / 'picks_history.csv'
        if not history_path.exists():
            return pd.DataFrame(), {'error': 'picks_history.csv not found'}
        history_df = pd.read_csv(history_path, low_memory=False)

    if history_df is None or len(history_df) == 0:
        return pd.DataFrame(), {'error': 'no history rows'}

    working = history_df.copy()
    if 'game_date' not in working.columns or 'result' not in working.columns:
        return pd.DataFrame(), {'error': 'missing required picks_history columns'}

    working['game_date'] = pd.to_datetime(working['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    working = working.dropna(subset=['game_date']).copy()
    working['result'] = working['result'].astype(str).str.upper()
    working = working[working['result'].isin(['WIN', 'LOSS', 'PUSH', 'DNP'])].copy()
    if len(working) == 0:
        return pd.DataFrame(), {'error': 'no graded under-only candidates'}

    detail_rows = []
    for game_date, day_df in working.groupby('game_date', sort=True):
        candidate_pool = _prepare_betslip_candidate_pool(
            day_df,
            direction='UNDER',
            allowed_props={'AST', 'TRB', 'STL', 'BLK'},
        )
        for size in [2, 3]:
            candidate_limit = {2: 6, 3: 8}[size]
            subset, metrics = _select_best_power_play_subset(candidate_pool, size, wager, candidate_limit)
            if len(subset) < size:
                continue
            results = [str(result).upper() for result in subset.get('result', pd.Series(dtype=str)).tolist()]
            slip_result, actual_payout, effective_legs = _grade_power_play_results(results, wager)
            if slip_result is None:
                continue
            row_data = _build_betslip_row(
                subset,
                game_date=game_date,
                slip_type=f'{size}-pick UNDER-Only Backtest',
                wager=wager,
            )
            row_data['result'] = slip_result
            row_data['actual_payout'] = round(actual_payout, 2)
            row_data['effective_legs'] = effective_legs
            row_data['eligible_picks'] = len(candidate_pool)
            detail_rows.append(row_data)

    detail_df = pd.DataFrame(detail_rows)
    if len(detail_df) == 0:
        return detail_df, {'error': 'no backtest slips generated'}

    summary: Dict[str, object] = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'wager_per_slip': wager,
        'slip_types': {},
    }
    for slip_type, slip_df in detail_df.groupby('slip_type'):
        total_wagered = float(slip_df['wager'].sum())
        total_payout = float(pd.to_numeric(slip_df['actual_payout'], errors='coerce').fillna(0.0).sum())
        wins = int((slip_df['result'] == 'WIN').sum())
        losses = int((slip_df['result'] == 'LOSS').sum())
        pushes = int((slip_df['result'] == 'PUSH').sum())
        summary['slip_types'][slip_type] = {
            'slips': int(len(slip_df)),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': round(wins / max(wins + losses, 1), 4),
            'total_wagered': round(total_wagered, 2),
            'total_payout': round(total_payout, 2),
            'roi': round((total_payout - total_wagered) / max(total_wagered, 1.0), 4),
            'avg_ev_per_unit': round(pd.to_numeric(slip_df['ev_per_unit'], errors='coerce').fillna(0.0).mean(), 4),
            'avg_joint_prob_adj': round(pd.to_numeric(slip_df['joint_prob_adj'], errors='coerce').fillna(0.0).mean(), 4),
        }

    return detail_df.reset_index(drop=True), summary


def _migrate_picks_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-time migration: convert old pipe-delimited 'picks' column into
    individual Pick_N_Player / Pick_N_Prop / Pick_N_Dir / Pick_N_Line columns.
    """
    MAX_LEGS = 4
    for i in range(1, MAX_LEGS + 1):
        for suffix in ['Player', 'Prop', 'Dir', 'Line']:
            col = f'Pick_{i}_{suffix}'
            if col not in df.columns:
                df[col] = ''

    def _parse_row(picks_str):
        parts = str(picks_str).split('|') if pd.notna(picks_str) else []
        result = {}
        for i, part in enumerate(parts[:MAX_LEGS], start=1):
            tokens = part.strip().rsplit(' ', 3)
            if len(tokens) == 4:
                result[f'Pick_{i}_Player'] = tokens[0]
                result[f'Pick_{i}_Prop']   = tokens[1].upper()
                result[f'Pick_{i}_Dir']    = tokens[2].upper()
                result[f'Pick_{i}_Line']   = tokens[3]
        return result

    for idx, row in df.iterrows():
        if 'picks' in df.columns and pd.notna(row.get('picks')) and str(row.get('picks', '')).strip():
            parsed = _parse_row(row['picks'])
            for col, val in parsed.items():
                df.at[idx, col] = val

    df = df.drop(columns=['picks'], errors='ignore')
    return df


def sanity_check_picks(picks: pd.DataFrame, df: pd.DataFrame, injury_data: dict = None) -> pd.DataFrame:
    """
    Remove picks that fail a role-change or injury sanity check:

    0. INJURED              — player is listed OUT or DOUBTFUL on today's injury report
    1. MINUTES SPIKE        — last game MP > 1.6x their L10 average AND the pick is UNDER
    2. MINUTES CRASH        — last game MP < 0.4x their L10 average AND the pick is OVER
    3. LINE/PRED RATIO      — Vegas line is > 1.6x model prediction (UNDER) or
                               model prediction is > 1.6x Vegas line (OVER)
    4. L3 TREND             — player's L3 average for the relevant stat already at/above
                               line on an UNDER play (recent hot streak, do not fade)
    5. RECENT MINUTES TREND — L3 avg minutes > 1.4x L10 avg minutes on an UNDER play
    6. HIGH VOLATILITY      — L10 coefficient of variation > 0.55 (stat is unpredictable)
    7. VOLATILITY SPIKE     — L10 max value > 2x L10 average (blowup games in recent history)
    """
    if picks.empty:
        return picks

    # Build flat set of OUT/DOUBTFUL players from injury report
    injured_out = set()
    doubtful_players = set()
    questionable_players = set()
    probable_players = set()
    day_to_day_players = set()
    status_map = {}
    pending_teams = set()
    if injury_data:
        for team_abbr, team_data in injury_data.items():
            if team_data.get('not_yet_submitted'):
                pending_teams.add(team_abbr)
            for name in team_data.get('out', []):
                injured_out.add(name)
            for name in team_data.get('doubtful', []):
                doubtful_players.add(name)
            for name in team_data.get('questionable', []):
                questionable_players.add(name)
            for name in team_data.get('probable', []):
                probable_players.add(name)
            for name in team_data.get('day_to_day', []):
                day_to_day_players.add(name)
            status_map.update(team_data.get('status_map', {}))

    df = df.copy()
    df['mp'] = pd.to_numeric(df['mp'], errors='coerce').fillna(0)
    df['pts'] = pd.to_numeric(df['pts'], errors='coerce').fillna(0)
    df['trb'] = pd.to_numeric(df['trb'], errors='coerce').fillna(0)
    df['ast'] = pd.to_numeric(df['ast'], errors='coerce').fillna(0)
    df = df.sort_values('game_date')

    # Pre-compute composite stat series for a player/prop from raw df tail
    def _get_stat_series(player_df, prop):
        p = player_df.copy()
        if prop == 'pts':  return p['pts']
        if prop == 'trb':  return p['trb']
        if prop == 'ast':  return p['ast']
        if prop == 'stl':  return p.get('stl', pd.Series(dtype=float))
        if prop == 'blk':  return p.get('blk', pd.Series(dtype=float))
        if prop == 'tov':  return p.get('tov', pd.Series(dtype=float))
        if prop == 'pa':   return p['pts'] + p['ast']
        if prop == 'pr':   return p['pts'] + p['trb']
        if prop == 'pra':  return p['pts'] + p['trb'] + p['ast']
        return pd.Series(dtype=float)

    # Build per-player L10 stat series for volatility analysis
    df['pa_val']  = df['pts'] + df['ast']
    df['pr_val']  = df['pts'] + df['trb']
    df['pra_val'] = df['pts'] + df['trb'] + df['ast']

    # Last game, L3, and L10 averages per player
    last_game = df.groupby('player').tail(1).set_index('player')[['mp', 'pts', 'trb', 'ast']]
    l3        = df.groupby('player').tail(3).groupby('player')[['mp', 'pts', 'trb', 'ast']].mean()
    l10       = df.groupby('player').tail(10).groupby('player')[['mp', 'pts', 'trb', 'ast']].mean()

    # L10 std dev for volatility checks (per player, per composite stat)
    stat_cols_for_vol = ['pts', 'trb', 'ast', 'pa_val', 'pr_val', 'pra_val']
    l10_std = df.groupby('player').tail(10).groupby('player')[stat_cols_for_vol].std()
    l10_max = df.groupby('player').tail(10).groupby('player')[stat_cols_for_vol].max()

    _vol_col = {'pts': 'pts', 'trb': 'trb', 'ast': 'ast',
                'pa': 'pa_val', 'pr': 'pr_val', 'pra': 'pra_val',
                'stl': 'pts', 'blk': 'pts', 'tov': 'pts'}  # fallback single stats

    # Pre-compute composite stats for L3 and L10
    def _get_composite(table, player, prop):
        """Return L3 or L10 average for a composite prop (pa, pr, pra, pts, trb, ast, stl, blk, tov)."""
        if player not in table.index:
            return None
        r = table.loc[player]
        if prop == 'pts':   return float(r['pts'])
        if prop == 'trb':   return float(r['trb'])
        if prop == 'ast':   return float(r['ast'])
        if prop == 'pa':    return float(r['pts']) + float(r['ast'])
        if prop == 'pr':    return float(r['pts']) + float(r['trb'])
        if prop == 'pra':   return float(r['pts']) + float(r['trb']) + float(r['ast'])
        return None

    flagged = []
    clean_idx = []

    for idx, row in picks.iterrows():
        player    = row['player']
        direction = row.get('direction', '')
        line      = float(row.get('line', 0))
        pred      = float(row.get('prediction', 0))
        prop      = str(row.get('prop', '')).lower()

        last_mp     = float(last_game.loc[player, 'mp']) if player in last_game.index else None
        avg_mp      = float(l10.loc[player, 'mp'])       if player in l10.index     else None
        l3_mp       = float(l3.loc[player, 'mp'])        if player in l3.index      else None

        # Use model-predicted minutes if available (from pred_df mp_expected column)
        mp_expected = row.get('mp_expected', None)
        if pd.isna(mp_expected) if mp_expected is not None else True:
            mp_expected = None

        reason = None

        # --- Injury report: skip any player listed OUT or DOUBTFUL ---
        player_norm = _normalize_name(player)
        if player_norm in injured_out:
            reason = f'PLAYER OUT/DOUBTFUL on today\'s injury report -- never bet on players who may not play'

        # --- Minutes model: predicted minutes vs line-implied minutes ---
        if reason is None and mp_expected is not None and avg_mp and avg_mp > 5 and pred > 0:
            # Estimate implied minutes from the line using player's per-minute rate
            per_min = avg_mp and (pred / avg_mp) if avg_mp > 0 else None
            if per_min and per_min > 0:
                implied_mp = line / per_min
                if direction == 'UNDER' and implied_mp < mp_expected * 0.65:
                    reason = (f'MINUTES MODEL: predicted {mp_expected:.0f}min but line implies '
                              f'{implied_mp:.0f}min — Vegas pricing more playing time than model expects')
                elif direction == 'OVER' and implied_mp > mp_expected * 1.4:
                    reason = (f'MINUTES MODEL: predicted {mp_expected:.0f}min but line implies only '
                              f'{implied_mp:.0f}min — model may be overestimating playing time')

        # --- Last-game minutes spike (fallback when no minutes model) ---
        if reason is None and last_mp is not None and avg_mp is not None and avg_mp > 5:
            ratio = last_mp / avg_mp
            if ratio >= 1.6 and direction == 'UNDER':
                reason = f'MINUTES SPIKE: {last_mp:.0f}min last game vs {avg_mp:.0f}min L10 avg — Vegas line may reflect new role'
            elif ratio <= 0.4 and direction == 'OVER':
                reason = f'MINUTES CRASH: {last_mp:.0f}min last game vs {avg_mp:.0f}min L10 avg — player may have reduced role'

        # L3 TREND and RECENT MINUTES TREND filters removed:
        # Data shows that "hot recently" (L3 near/above line) does NOT predict
        # the player will go OVER — lines are still set above L10 systematically,
        # so UNDER is still the correct bet regardless of recent 3-game form.

        # --- High volatility: stat is too unpredictable to bet ---
        # Raised CV threshold to 0.70 — volatile players still go under on mispriced lines.
        if reason is None and line > 0:
            vcol = _vol_col.get(prop)
            if vcol and player in l10_std.index and player in l10.index:
                std_val = float(l10_std.loc[player, vcol]) if vcol in l10_std.columns else 0
                avg_val = _get_composite(l10, player, prop) or 0
                max_val = float(l10_max.loc[player, vcol]) if vcol in l10_max.columns else 0
                # CV > 0.70: truly chaotic stat (e.g. a player who scores 0 or 30 randomly)
                if avg_val > 0 and std_val / avg_val > 0.70:
                    reason = (f'HIGH VOLATILITY: L10 std dev {std_val:.1f} vs avg {avg_val:.1f} '
                              f'(CV={std_val/avg_val:.2f}) — stat is too unpredictable to bet')
                # L10 max spike > 3x average (extreme outlier game distorts L10 avg itself)
                elif avg_val > 0 and max_val > avg_val * 3.0:
                    reason = (f'VOLATILITY SPIKE: L10 max {max_val:.0f} vs avg {avg_val:.1f} '
                              f'— single huge outlier game inflates L10 avg, line may not be mispriced')

        # --- Line vs prediction mismatch (tightened from 2.0x to 1.6x) ---
        if reason is None and pred > 0:
            if direction == 'UNDER' and line >= 1.6 * pred:
                reason = f'LINE/PRED MISMATCH: line {line} is {line/pred:.1f}x model prediction {pred} — Vegas likely pricing a role change'
            elif direction == 'OVER' and pred >= 1.6 * line:
                reason = f'LINE/PRED MISMATCH: prediction {pred} is {pred/line:.1f}x line {line} — model may be using stale data'

        if reason:
            flagged.append({'player': player, 'prop': row.get('prop',''), 'direction': direction,
                            'line': line, 'prediction': pred, 'reason': reason})
        else:
            clean_idx.append(idx)

    if flagged:
        print(f"\n  Sanity check removed {len(flagged)} pick(s):")
        for f in flagged:
            print(f"    [REMOVED] {f['player']} {f['prop']} {f['direction']} {f['line']} -- {f['reason']}")

    return picks.loc[clean_idx].reset_index(drop=True)


def sanity_check_picks(picks: pd.DataFrame, df: pd.DataFrame, injury_data: dict = None) -> pd.DataFrame:
    """
    Remove picks that fail a role-change or injury sanity check.

    For STL/BLK unders, ratio-only mismatch checks are too harsh because the
    denominator is tiny. Use model/L10/L3 support together instead.
    """
    if picks.empty:
        return picks

    injured_out = set()
    doubtful_players = set()
    questionable_players = set()
    probable_players = set()
    day_to_day_players = set()
    status_map = {}
    pending_teams = set()
    if injury_data:
        for team_abbr, team_data in injury_data.items():
            if team_data.get('not_yet_submitted'):
                pending_teams.add(team_abbr)
            for name in team_data.get('out', []):
                injured_out.add(name)
            for name in team_data.get('doubtful', []):
                doubtful_players.add(name)
            for name in team_data.get('questionable', []):
                questionable_players.add(name)
            for name in team_data.get('probable', []):
                probable_players.add(name)
            for name in team_data.get('day_to_day', []):
                day_to_day_players.add(name)
            status_map.update(team_data.get('status_map', {}))

    df = df.copy()
    for col in ['mp', 'pts', 'trb', 'ast', 'stl', 'blk', 'tov']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0
    df = df.sort_values('game_date')

    df['pa_val'] = df['pts'] + df['ast']
    df['pr_val'] = df['pts'] + df['trb']
    df['pra_val'] = df['pts'] + df['trb'] + df['ast']

    base_stat_cols = ['mp', 'pts', 'trb', 'ast', 'stl', 'blk', 'tov']
    last_game = df.groupby('player').tail(1).set_index('player')[base_stat_cols]
    last3_rows = df.groupby('player').tail(3)
    last10_rows = df.groupby('player').tail(10)
    l3 = last3_rows.groupby('player')[base_stat_cols].mean()
    l10 = last10_rows.groupby('player')[base_stat_cols].mean()
    last3_hist = {name: grp for name, grp in last3_rows.groupby('player')}
    last10_hist = {name: grp for name, grp in last10_rows.groupby('player')}

    stat_cols_for_vol = ['pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'pa_val', 'pr_val', 'pra_val']
    l10_std = df.groupby('player').tail(10).groupby('player')[stat_cols_for_vol].std()
    l10_max = df.groupby('player').tail(10).groupby('player')[stat_cols_for_vol].max()

    vol_col = {
        'pts': 'pts',
        'trb': 'trb',
        'ast': 'ast',
        'stl': 'stl',
        'blk': 'blk',
        'tov': 'tov',
        'pa': 'pa_val',
        'pr': 'pr_val',
        'pra': 'pra_val',
    }

    def get_prop_avg(table, player_name, prop_name):
        if player_name not in table.index:
            return None
        row = table.loc[player_name]
        if prop_name == 'pts':
            return float(row['pts'])
        if prop_name == 'trb':
            return float(row['trb'])
        if prop_name == 'ast':
            return float(row['ast'])
        if prop_name == 'stl':
            return float(row['stl'])
        if prop_name == 'blk':
            return float(row['blk'])
        if prop_name == 'tov':
            return float(row['tov'])
        if prop_name == 'pa':
            return float(row['pts']) + float(row['ast'])
        if prop_name == 'pr':
            return float(row['pts']) + float(row['trb'])
        if prop_name == 'pra':
            return float(row['pts']) + float(row['trb']) + float(row['ast'])
        return None

    flagged = []
    clean_idx = []

    for idx, row in picks.iterrows():
        player = row['player']
        team = str(row.get('team', '') or '')
        direction = row.get('direction', '')
        line = float(row.get('line', 0))
        pred = float(row.get('prediction', 0))
        prop = str(row.get('prop', '')).lower()

        last_mp = float(last_game.loc[player, 'mp']) if player in last_game.index else None
        avg_mp = float(l10.loc[player, 'mp']) if player in l10.index else None

        mp_expected = row.get('mp_predicted', row.get('mp_expected', None))
        if mp_expected is not None:
            mp_expected = pd.to_numeric(pd.Series([mp_expected]), errors='coerce').iloc[0]
        if pd.isna(mp_expected) if mp_expected is not None else True:
            mp_expected = None

        reason = None

        player_norm = _normalize_name(player)
        status_detail = status_map.get(player_norm, {})
        status_label = str(status_detail.get('status', '') or status_detail.get('bucket', '')).replace('_', ' ').strip().upper()
        if team in pending_teams:
            reason = "TEAM STATUS PENDING on official NBA injury report -- skip until that team submits"
        elif player_norm in injured_out or player_norm in doubtful_players:
            reason = f"PLAYER {status_label or 'OUT'} on today's injury report -- never bet on players who may not play"
        elif player_norm in questionable_players:
            reason = f"PLAYER {status_label or 'QUESTIONABLE'} on today's injury report -- skip uncertain pregame availability"
        elif player_norm in day_to_day_players and player_norm not in probable_players:
            reason = "PLAYER DAY-TO-DAY on today's injury report -- skip until status upgrades"

        if reason is None and mp_expected is not None and avg_mp and avg_mp > 5 and pred > 0:
            per_min = (pred / avg_mp) if avg_mp > 0 else None
            if per_min and per_min > 0:
                implied_mp = line / per_min
                if direction == 'UNDER' and implied_mp < mp_expected * 0.65:
                    reason = (
                        f"MINUTES MODEL: predicted {mp_expected:.0f}min but line implies "
                        f"{implied_mp:.0f}min -- Vegas pricing more playing time than model expects"
                    )
                elif direction == 'OVER' and implied_mp > mp_expected * 1.4:
                    reason = (
                        f"MINUTES MODEL: predicted {mp_expected:.0f}min but line implies only "
                        f"{implied_mp:.0f}min -- model may be overestimating playing time"
                    )

        if reason is None and last_mp is not None and avg_mp is not None and avg_mp > 5:
            ratio = last_mp / avg_mp
            if ratio >= 1.6 and direction == 'UNDER':
                reason = f"MINUTES SPIKE: {last_mp:.0f}min last game vs {avg_mp:.0f}min L10 avg -- Vegas line may reflect new role"
            elif ratio <= 0.4 and direction == 'OVER':
                reason = f"MINUTES CRASH: {last_mp:.0f}min last game vs {avg_mp:.0f}min L10 avg -- player may have reduced role"
            elif last_mp <= min(12.0, avg_mp * 0.45):
                reason = (
                    f"ROLE INSTABILITY: only {last_mp:.0f}min last game vs {avg_mp:.0f}min L10 avg "
                    f"-- elevated DNP/rotation risk"
                )

        if reason is None and line > 0:
            vcol = vol_col.get(prop)
            avg_val = get_prop_avg(l10, player, prop) or 0
            if vcol and player in l10_std.index and player in l10.index:
                std_val = float(l10_std.loc[player, vcol]) if vcol in l10_std.columns else 0
                max_val = float(l10_max.loc[player, vcol]) if vcol in l10_max.columns else 0
                cv_val = (std_val / avg_val) if avg_val > 0 else None
                micro_prop_under = prop in {'stl', 'blk'} and direction == 'UNDER'
                strict_under_l10 = None
                strict_under_l3 = None
                relaxed_micro_profile = False
                if micro_prop_under and player in last10_hist and prop in last10_hist[player].columns:
                    recent10 = pd.to_numeric(last10_hist[player][prop], errors='coerce').fillna(0)
                    strict_under_l10 = float((recent10 < line).mean()) if len(recent10) else None
                    if player in last3_hist and prop in last3_hist[player].columns:
                        recent3 = pd.to_numeric(last3_hist[player][prop], errors='coerce').fillna(0)
                        strict_under_l3 = float((recent3 < line).mean()) if len(recent3) else None
                    elite_under_profile = (
                        strict_under_l10 is not None and strict_under_l10 >= 0.85 and
                        (strict_under_l3 is None or strict_under_l3 >= 0.67)
                    )
                    low_line_binary_profile = (
                        line <= 1.5 and
                        strict_under_l10 is not None and strict_under_l10 >= 0.70 and
                        (strict_under_l3 is None or strict_under_l3 >= 0.67) and
                        pred <= line - 0.4
                    )
                    relaxed_micro_profile = elite_under_profile or low_line_binary_profile

                cv_limit = 0.70
                spike_limit = avg_val * 3.0 if avg_val > 0 else None
                if relaxed_micro_profile:
                    cv_limit = 1.75 if line <= 1.5 else 0.95
                    spike_limit = max(line + 1.5, avg_val * (4.5 if line <= 1.5 else 4.0))

                if cv_val is not None and cv_val > cv_limit:
                    reason = (
                        f"HIGH VOLATILITY: L10 std dev {std_val:.1f} vs avg {avg_val:.1f} "
                        f"(CV={cv_val:.2f}) -- stat is too unpredictable to bet"
                    )
                elif spike_limit is not None and max_val > spike_limit:
                    reason = (
                        f"VOLATILITY SPIKE: L10 max {max_val:.0f} vs avg {avg_val:.1f} "
                        f"-- single huge outlier game inflates L10 avg, line may not be mispriced"
                    )

        if reason is None and pred > 0:
            prop_l10 = get_prop_avg(l10, player, prop)
            prop_l3 = get_prop_avg(l3, player, prop)
            if prop in {'stl', 'blk'} and direction == 'UNDER':
                support_vals = [float(v) for v in [pred, prop_l10, prop_l3] if v is not None and not pd.isna(v)]
                support_floor = max(support_vals) if support_vals else pred
                support_gap = line - support_floor
                support_ratio = (line / support_floor) if support_floor and support_floor > 0 else np.inf
                if support_gap >= 1.0 and support_ratio >= 2.0:
                    reason = (
                        f"MICRO-PROP ROLE MISMATCH: line {line} is still {support_ratio:.1f}x above "
                        f"best support {support_floor:.1f} (model/L10/L3) -- Vegas likely pricing a role shift"
                    )
            else:
                if direction == 'UNDER' and line >= 1.6 * pred:
                    reason = f"LINE/PRED MISMATCH: line {line} is {line/pred:.1f}x model prediction {pred} -- Vegas likely pricing a role change"
                elif direction == 'OVER' and pred >= 1.6 * line:
                    reason = f"LINE/PRED MISMATCH: prediction {pred} is {pred/line:.1f}x line {line} -- model may be using stale data"

        if reason:
            flagged.append({
                'player': player,
                'prop': row.get('prop', ''),
                'direction': direction,
                'line': line,
                'prediction': pred,
                'reason': reason,
            })
        else:
            clean_idx.append(idx)

    if flagged:
        print(f"\n  Sanity check removed {len(flagged)} pick(s):")
        for item in flagged:
            print(f"    [REMOVED] {item['player']} {item['prop']} {item['direction']} {item['line']} -- {item['reason']}")

    return picks.loc[clean_idx].reset_index(drop=True)


def save_picks(picks_df: pd.DataFrame, pred_df: pd.DataFrame, slate_date: Optional[str] = None):
    """Save picks to CSV and generate simple HTML report."""
    _migrate_legacy_output_artifacts()

    for col, default in [
        ('dir_prob', np.nan),
        ('ou_prob', np.nan),
        ('p_over_raw', np.nan),
        ('p_over_cal', np.nan),
        ('p_under_raw', np.nan),
        ('p_under_cal', np.nan),
        ('prob_source', ''),
        ('sigma', np.nan),
        ('low_sigma_under', False),
        ('projection_type', 'standard'),
        ('is_promo', False),
        ('break_even_prob', np.nan),
        ('exceeds_ev_threshold', False),
    ]:
        if col not in picks_df.columns:
            picks_df[col] = default

    run_dt = datetime.now()
    run_stamp = run_dt.strftime('%Y-%m-%d_%H%M%S')
    slate_date = slate_date or run_dt.strftime('%Y-%m-%d')

    # Save full history-independent snapshots in a dated archive tree.
    picks_daily_path, picks_run_path = _archive_csv_snapshot(
        picks_df, 'picks', slate_date, run_stamp
    )
    predictions_daily_path, predictions_run_path = _archive_csv_snapshot(
        pred_df, 'predictions', slate_date, run_stamp
    )
    print(f"\n  Picks archived to {picks_run_path}")
    print(f"  Predictions archived to {predictions_run_path}")

    # Append to cumulative history file for tracking
    history_path = CONFIG['output_dir'] / 'picks_history.csv'
    daily = picks_df.copy()
    daily.insert(0, 'game_date', slate_date)
    if history_path.exists():
        existing = pd.read_csv(history_path, dtype=str)
        existing = existing[existing['game_date'] != slate_date]  # replace current slate rows
        combined = pd.concat([existing, daily.astype(str)], ignore_index=True)
    else:
        combined = daily
    combined.to_csv(history_path, index=False)
    print(f"  History updated -> {history_path}  ({len(combined):,} total rows)")

    # Generate HTML dashboard
    html = generate_html_report(picks_df, run_stamp)
    dashboard_daily_path, dashboard_run_path = _archive_text_snapshot(
        html, 'dashboard', slate_date, run_stamp, '.html'
    )
    print(f"  Dashboard archived to {dashboard_run_path}")

    # Log suggested betslips for the active slate (pass injury data so risky legs are skipped)
    slips_df = log_betslips(picks_df, slate_date)

    # Save stable "latest" working files for the current day.
    _atomic_csv(picks_df, CONFIG['output_dir'] / 'picks_latest.csv')
    _atomic_csv(pred_df, CONFIG['output_dir'] / 'predictions_latest.csv')
    _atomic_text(html, CONFIG['output_dir'] / 'dashboard_latest.html')
    if isinstance(slips_df, pd.DataFrame) and len(slips_df) > 0:
        _atomic_csv(slips_df, CONFIG['output_dir'] / 'betslips_latest.csv')
        slips_daily_path, slips_run_path = _archive_csv_snapshot(
            slips_df, 'betslips', slate_date, run_stamp
        )
        print(f"  Betslips archived to {slips_run_path}")

    print(f"  Daily archive folder: {picks_daily_path.parent}")


def generate_html_report(picks_df: pd.DataFrame, timestamp: str) -> str:
    """Generate HTML dashboard showing all picks with sort/filter controls."""

    n_over   = len(picks_df[picks_df['direction'] == 'OVER'])
    n_under  = len(picks_df[picks_df['direction'] == 'UNDER'])
    avg_edge = picks_df['edge'].abs().mean()
    avg_conf = picks_df['confidence'].mean() if 'confidence' in picks_df.columns else 0
    date_str = datetime.now().strftime('%B %d, %Y')
    time_str = datetime.now().strftime('%I:%M %p')

    # Sort by confidence descending (highest confidence first)
    if 'confidence' in picks_df.columns:
        picks_df = picks_df.sort_values('confidence', ascending=False)

    # Build table rows
    rows_html = ''
    for i, (_, pick) in enumerate(picks_df.iterrows()):
        conf      = f"{pick['confidence']:.0f}" if pd.notna(pick.get('confidence')) else '-'
        pick_prob = pick.get('dir_prob', pick.get('ou_prob'))
        ou        = f"{pick_prob:.0f}" if pd.notna(pick_prob) else '-'
        l10       = f"{pick['l10_avg']:.1f}" if pd.notna(pick.get('l10_avg')) else '-'
        edge_val  = pick['edge']
        edge_fmt  = f"{edge_val:+.1f}"
        direction = pick['direction']
        inj       = '&#9889;' if pick.get('injury_adj') else ''
        row_class = 'row-even' if i % 2 == 0 else 'row-odd'
        lm = pick.get('line_move')
        if lm is not None and not pd.isna(lm) and lm != 0:
            lm_fmt  = f"{lm:+.1f}"
            lm_class = 'edge-pos' if (lm > 0 and direction == 'OVER') or (lm < 0 and direction == 'UNDER') else 'edge-neg'
            line_move_td = f'<td class="td-num {lm_class}">{lm_fmt}</td>'
        else:
            line_move_td = '<td class="td-num td-muted">-</td>'
        game_total = pick.get('game_total', '')
        gt_td = f'<td class="td-num td-muted">{game_total if game_total else "-"}</td>'

        rows_html += (
            f'<tr class="{row_class}" data-dir="{direction}">\n'
            f'  <td class="td-player">{pick["player"]}{" " + inj if inj else ""}</td>\n'
            f'  <td class="td-team">{pick.get("team", "")}</td>\n'
            f'  <td class="td-prop"><span class="prop-badge">{pick["prop"]}</span></td>\n'
            f'  <td><span class="dir-badge dir-{direction.lower()}">{direction}</span></td>\n'
            f'  <td class="td-num">{pick["line"]}</td>\n'
            f'  <td class="td-num">{pick["prediction"]}</td>\n'
            f'  <td class="td-num {"edge-pos" if edge_val > 0 else "edge-neg"}">{edge_fmt}</td>\n'
            f'  <td class="td-num td-muted">{l10}</td>\n'
            f'  <td class="td-num">{conf}%</td>\n'
            f'  <td class="td-num td-muted">{ou}{"%" if ou != "-" else ""}</td>\n'
            f'  {line_move_td}\n'
            f'  {gt_td}\n'
            f'</tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NBA Props &mdash; {date_str}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
@font-face {{
  font-family: 'Street Bit';
  src: url('StreetBit.ttf') format('truetype');
  font-weight: normal;
  font-style: normal;
}}
/* ── Reset & base ─────────────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg:        #09090f;
  --bg2:       #0f0f1a;
  --bg3:       #14142a;
  --bg4:       #1a1a35;
  --border:    #252545;
  --accent:    #7c3aed;
  --accent2:   #a855f7;
  --cyan:      #06b6d4;
  --green:     #22c55e;
  --red:       #ef4444;
  --gold:      #f59e0b;
  --text:      #e2e8f0;
  --muted:     #64748b;
  --pixel:     'Street Bit', 'Press Start 2P', monospace;
  --mono:      'Share Tech Mono', monospace;
  --sans:      'Inter', sans-serif;
}}

body {{
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}}

/* ── Scanline overlay ─────────────────────────────────────────────────────── */
body::before {{
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.08) 2px,
    rgba(0,0,0,0.08) 4px
  );
}}

.wrapper {{ position: relative; z-index: 1; max-width: 1440px; margin: 0 auto; padding: 28px 24px; }}

/* ── Header ──────────────────────────────────────────────────────────────── */
.header {{
  display: flex; align-items: flex-end; justify-content: space-between;
  flex-wrap: wrap; gap: 16px;
  border-bottom: 2px solid var(--accent);
  padding-bottom: 20px; margin-bottom: 28px;
}}
.header-left {{ display: flex; flex-direction: column; gap: 6px; }}
.logo {{
  font-family: var(--pixel);
  font-size: clamp(14px, 2.2vw, 22px);
  color: var(--accent2);
  letter-spacing: 2px;
  text-shadow: 0 0 20px rgba(168,85,247,0.6), 0 0 40px rgba(168,85,247,0.3);
  line-height: 1.4;
}}
.logo span {{ color: var(--cyan); }}
.dateline {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 1px;
  text-transform: uppercase;
}}
.header-right {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  text-align: right;
  line-height: 1.8;
}}

/* ── Stat cards ──────────────────────────────────────────────────────────── */
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 28px;
}}
.card {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  position: relative;
  overflow: hidden;
  transition: border-color .2s, transform .15s;
}}
.card:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
.card::after {{
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--accent-line, var(--accent));
}}
.card-label {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 10px;
}}
.card-value {{
  font-family: var(--pixel);
  font-size: clamp(16px, 2.5vw, 24px);
  line-height: 1;
  color: var(--card-color, var(--text));
}}
.card-sub {{
  font-size: 11px;
  color: var(--muted);
  margin-top: 6px;
  font-family: var(--mono);
}}

/* ── Filter bar ──────────────────────────────────────────────────────────── */
.filter-bar {{
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
}}
.filter-group {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.filter-label {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  white-space: nowrap;
}}
.filter-bar select,
.filter-bar input {{
  background: var(--bg3);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 7px 12px;
  font-family: var(--mono);
  font-size: 12px;
  outline: none;
  transition: border-color .2s;
  min-width: 120px;
}}
.filter-bar select:focus,
.filter-bar input:focus {{ border-color: var(--accent); }}
.filter-bar input {{ width: 200px; }}
.result-count {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  margin-left: auto;
  white-space: nowrap;
}}
.result-count span {{ color: var(--cyan); font-weight: bold; }}

/* ── Tab pills ───────────────────────────────────────────────────────────── */
.tab-pills {{
  display: flex;
  gap: 6px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}}
.tab-pill {{
  font-family: var(--mono);
  font-size: 11px;
  padding: 6px 14px;
  border-radius: 20px;
  border: 1px solid var(--border);
  background: var(--bg2);
  color: var(--muted);
  cursor: pointer;
  transition: all .2s;
  user-select: none;
}}
.tab-pill:hover, .tab-pill.active {{
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}}

/* ── Table ───────────────────────────────────────────────────────────────── */
.table-wrap {{
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  overflow-x: auto;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
thead tr {{
  background: var(--bg3);
  border-bottom: 1px solid var(--accent);
}}
th {{
  padding: 13px 14px;
  font-family: var(--mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--muted);
  text-align: left;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  transition: color .15s;
}}
th:hover {{ color: var(--cyan); }}
th.sort-asc::after  {{ content: ' ↑'; color: var(--cyan); }}
th.sort-desc::after {{ content: ' ↓'; color: var(--cyan); }}

.row-even {{ background: var(--bg); }}
.row-odd  {{ background: var(--bg2); }}
td {{ padding: 11px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: var(--bg4) !important; }}

.td-player {{
  font-weight: 600;
  font-size: 13px;
  color: var(--text);
  white-space: nowrap;
}}
.td-team {{
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
}}
.td-num {{
  font-family: var(--mono);
  font-size: 12px;
  text-align: right;
}}
.td-muted {{ color: var(--muted); }}

/* ── Badges ──────────────────────────────────────────────────────────────── */
.prop-badge {{
  display: inline-block;
  background: var(--bg4);
  border: 1px solid var(--border);
  color: var(--cyan);
  font-family: var(--mono);
  font-size: 10px;
  padding: 3px 8px;
  border-radius: 4px;
  letter-spacing: 0.5px;
}}
.dir-badge {{
  display: inline-block;
  font-family: var(--mono);
  font-size: 10px;
  font-weight: bold;
  padding: 4px 10px;
  border-radius: 4px;
  letter-spacing: 1px;
}}
.dir-over  {{ background: rgba(34,197,94,0.15);  color: var(--green); border: 1px solid rgba(34,197,94,0.3); }}
.dir-under {{ background: rgba(239,68,68,0.15);  color: var(--red);   border: 1px solid rgba(239,68,68,0.3); }}

.grade-badge {{
  display: inline-flex;
  align-items: center; justify-content: center;
  font-family: var(--pixel);
  font-size: 9px;
  width: 32px; height: 22px;
  border-radius: 4px;
}}
.grade-A    {{ background: rgba(34,197,94,0.2);  color: var(--green); border: 1px solid rgba(34,197,94,0.4); }}
.grade-Bplus{{ background: rgba(6,182,212,0.15); color: var(--cyan);  border: 1px solid rgba(6,182,212,0.3); }}
.grade-B    {{ background: rgba(245,158,11,0.15);color: var(--gold);  border: 1px solid rgba(245,158,11,0.3); }}
.grade-C    {{ background: rgba(100,116,139,0.15);color: var(--muted);border: 1px solid rgba(100,116,139,0.3); }}

.edge-pos {{ color: var(--green) !important; }}
.edge-neg {{ color: var(--red)   !important; }}

/* ── Footer ──────────────────────────────────────────────────────────────── */
.footer {{
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}}
.footer-text {{
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  letter-spacing: 0.5px;
}}
.footer-text strong {{ color: var(--accent2); }}

/* ── Glow pulse on logo ──────────────────────────────────────────────────── */
@keyframes pulse {{
  0%, 100% {{ text-shadow: 0 0 20px rgba(168,85,247,0.6), 0 0 40px rgba(168,85,247,0.3); }}
  50%       {{ text-shadow: 0 0 30px rgba(168,85,247,0.9), 0 0 60px rgba(168,85,247,0.5); }}
}}
.logo {{ animation: pulse 3s ease-in-out infinite; }}
</style>
</head>
<body>
<div class="wrapper">

  <!-- Header -->
  <header class="header">
    <div class="header-left">
      <div class="logo">NBA<span>&nbsp;PROP</span>&nbsp;MODEL</div>
      <div class="dateline">{date_str} &nbsp;&#9670;&nbsp; Generated {time_str}</div>
    </div>
    <div class="header-right">
      PICKS: {len(picks_df)} &nbsp;|&nbsp; AVG CONF: {avg_conf:.0f}%<br>
      AVG EDGE: {avg_edge:.2f} &nbsp;|&nbsp; OVER / UNDER: {n_over} / {n_under}
    </div>
  </header>

  <!-- Stat cards -->
  <div class="cards">
    <div class="card" style="--accent-line:var(--accent2); --card-color:var(--accent2);">
      <div class="card-label">Total Picks</div>
      <div class="card-value">{len(picks_df)}</div>
      <div class="card-sub">Today&rsquo;s slate</div>
    </div>
    <div class="card" style="--accent-line:var(--cyan); --card-color:var(--cyan);">
      <div class="card-label">Avg Confidence</div>
      <div class="card-value">{avg_conf:.0f}%</div>
      <div class="card-sub">Model conviction</div>
    </div>
    <div class="card" style="--accent-line:var(--green); --card-color:var(--green);">
      <div class="card-label">OVERs</div>
      <div class="card-value">{n_over}</div>
      <div class="card-sub">{n_over / max(len(picks_df),1)*100:.0f}% of picks</div>
    </div>
    <div class="card" style="--accent-line:var(--red); --card-color:var(--red);">
      <div class="card-label">UNDERs</div>
      <div class="card-value">{n_under}</div>
      <div class="card-sub">{n_under / max(len(picks_df),1)*100:.0f}% of picks</div>
    </div>
    <div class="card" style="--accent-line:var(--gold); --card-color:var(--gold);">
      <div class="card-label">Avg Edge</div>
      <div class="card-value">{avg_edge:.1f}</div>
      <div class="card-sub">Points vs line</div>
    </div>
  </div>

  <!-- Filter bar -->
  <div class="filter-bar">
    <div class="filter-group">
      <span class="filter-label">Direction</span>
      <select id="dirFilter" onchange="applyFilters()">
        <option value="">All</option>
        <option value="OVER">OVER</option>
        <option value="UNDER">UNDER</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Prop</span>
      <select id="propFilter" onchange="applyFilters()">
        <option value="">All props</option>
        <option value="PTS">PTS</option>
        <option value="TRB">TRB</option>
        <option value="AST">AST</option>
        <option value="PRA">PRA</option>
        <option value="PR">PR</option>
        <option value="PA">PA</option>
        <option value="STL">STL</option>
        <option value="BLK">BLK</option>
        <option value="TOV">TOV</option>
      </select>
    </div>
    <div class="filter-group">
      <span class="filter-label">Search</span>
      <input id="searchBox" type="text" placeholder="Player name..." oninput="applyFilters()">
    </div>
    <div class="result-count">Showing <span id="visCount">{len(picks_df)}</span> of {len(picks_df)} picks</div>
  </div>

  <!-- Table -->
  <div class="table-wrap">
    <table id="picksTable">
      <thead>
        <tr>
          <th onclick="sortTable(0,this)">Player</th>
          <th onclick="sortTable(1,this)">Team</th>
          <th onclick="sortTable(2,this)">Prop</th>
          <th onclick="sortTable(3,this)">Dir</th>
          <th onclick="sortTable(4,this)">Line</th>
          <th onclick="sortTable(5,this)">Pred</th>
          <th onclick="sortTable(6,this)">Edge</th>
          <th onclick="sortTable(7,this)">L10</th>
          <th onclick="sortTable(8,this)">Conf%</th>
          <th onclick="sortTable(9,this)">Pick%</th>
          <th onclick="sortTable(10,this)">Line Mv</th>
          <th onclick="sortTable(11,this)">Tot</th>
        </tr>
      </thead>
      <tbody id="tableBody">
{rows_html}      </tbody>
    </table>
  </div>

  <!-- Footer -->
  <footer class="footer">
    <div class="footer-text">
      <strong>NBA PROP MODEL</strong> &nbsp;&mdash;&nbsp;
      {date_str} &nbsp;&mdash;&nbsp;
      XGBoost + LightGBM &nbsp;|&nbsp; 239 features &nbsp;|&nbsp; Time-series CV
    </div>
    <div class="footer-text">
      For informational use only. Not financial advice.
    </div>
  </footer>

</div><!-- /wrapper -->

<script>
(function() {{
  let sortCol = -1, sortAsc = true;

  window.sortTable = function(col, th) {{
    const tbody = document.getElementById('tableBody');
    const rows  = Array.from(tbody.querySelectorAll('tr'));
    if (sortCol === col) {{ sortAsc = !sortAsc; }}
    else {{ sortAsc = true; sortCol = col; }}

    document.querySelectorAll('thead th').forEach(h => h.classList.remove('sort-asc','sort-desc'));
    th.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');

    rows.sort((a, b) => {{
      let av = a.cells[col].innerText.replace(/[%+]/g,'').trim();
      let bv = b.cells[col].innerText.replace(/[%+]/g,'').trim();
      const an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return sortAsc ? an - bn : bn - an;
      return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    rows.forEach(r => tbody.appendChild(r));
    applyFilters();
  }};

  window.applyFilters = function() {{
    const dir    = document.getElementById('dirFilter').value;
    const prop   = document.getElementById('propFilter').value;
    const search = document.getElementById('searchBox').value.toLowerCase();
    let vis = 0;
    document.querySelectorAll('#tableBody tr').forEach(row => {{
      const d      = row.dataset.dir;
      const pCell  = row.cells[2].innerText.trim();
      const player = row.cells[0].innerText.toLowerCase();
      const show   = (dir   === '' || d === dir)
                  && (prop  === '' || pCell === prop)
                  && (search === '' || player.includes(search));
      row.style.display = show ? '' : 'none';
      if (show) vis++;
    }});
    document.getElementById('visCount').innerText = vis;
  }};
}})();
</script>
</body>
</html>"""

    return html


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    """Main entry point."""
    # Ensure UTF-8 output on Windows terminals
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands: update, train, predict, backtest, backfill, rebuild-lines, all")
        return
    
    command = sys.argv[1].lower()
    
    print("\n" + "="*60)
    print(f"NBA PROPS - {command.upper()}")
    print("="*60)
    
    data_path = CONFIG['data_dir'] / 'nba_data.csv'
    
    if command == 'update':
        # Fetch fresh data
        print("\nFetching NBA data...")
        
        # Check for local file argument
        local_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        if local_file:
            df = fetch_nba_data(local_file=local_file)
        else:
            df = fetch_nba_data(seasons=['2025-26', '2024-25', '2023-24'])

        df.to_csv(data_path, index=False)
        print(f"\nData saved to {data_path}")
        print(f"Total rows: {len(df):,}")

        # Warm the prediction feature cache so the next `predict` run is fast.
        try:
            adv_models, _, _, _, _, _, _ = _load_advanced_models()
            using_advanced = adv_models is not None
            cache_mode = 'advanced' if using_advanced else 'standard'
            print(f"\nWarming {cache_mode} prediction feature cache...")
            _prepare_latest_prediction_frame(df, using_advanced=using_advanced, use_cache=False)
        except Exception as e:
            print(f"  WARNING: Prediction cache warm-up failed ({e})")

    elif command == 'train':
        # Load data and train
        if not data_path.exists():
            print("No data found. Running 'update' first...")
            df = fetch_nba_data(seasons=['2025-26', '2024-25', '2023-24'])
            df.to_csv(data_path, index=False)
        else:
            df = pd.read_csv(data_path, parse_dates=['game_date'])
        
        df = create_features(df)
        train_models(df)
        train_ou_classifiers(df)

    elif command == 'predict':
        # Generate predictions
        if not data_path.exists():
            raise FileNotFoundError("No data found. Run 'update' first.")

        df = pd.read_csv(data_path, parse_dates=['game_date'])

        # Use the active slate from historical_lines.csv if already fetched (avoids rate limits)
        today_str = datetime.now().strftime('%Y-%m-%d')
        lines_path = CONFIG['data_dir'] / 'historical_lines.csv'
        vegas = pd.DataFrame()
        _line_source = 'unknown'
        active_slate_date = today_str
        if lines_path.exists():
            _hist = pd.read_csv(lines_path)
            active_hist, hist_slate_date = _select_active_slate_lines(_hist)
            if len(active_hist) > 0:
                active_slate_date = hist_slate_date or today_str
                _line_source = _infer_line_source(active_hist)
                print(
                    f"\nUsing {len(active_hist)} cached lines from historical_lines.csv "
                    f"(active slate {active_slate_date}) [source: {_line_source}]"
                )
                _market_cols = [c for c in ['projection_type', 'is_promo'] if c in active_hist.columns]
                vegas = active_hist[['player', 'prop', 'line'] + _market_cols].copy()
                vegas['num_books'] = active_hist['num_books'].values if 'num_books' in active_hist.columns else 1
        if len(vegas) == 0:
            print("\nFetching Vegas lines...")
            vegas_raw, _line_source = fetch_vegas_lines()
            if len(vegas_raw) > 0:
                save_lines_snapshot(vegas_raw, source=_line_source)
                active_live, live_slate_date = _select_active_slate_lines(vegas_raw)
                active_slate_date = live_slate_date or today_str
                _market_cols = [c for c in ['projection_type', 'is_promo'] if c in active_live.columns]
                vegas = active_live[['player', 'prop', 'line'] + _market_cols].copy()
                vegas['num_books'] = active_live['num_books'].values if 'num_books' in active_live.columns else 1

        print(f"\nActive slate date: {active_slate_date}")

        # Warn loudly when lines did not come from PrizePicks — picks are bet on PrizePicks,
        # so a line source mismatch corrupts the performance-tracking record.
        if _line_source != 'prizepicks':
            print(
                f"\n{'!'*60}\n"
                f"  LINE SOURCE WARNING: lines sourced from '{_line_source}'\n"
                f"  PrizePicks lines could not be fetched (Step 2 failed or blocked).\n"
                f"  Picks below are generated against {_line_source} lines, which may\n"
                f"  differ from the lines PrizePicks actually offers today.\n"
                f"  ACTION REQUIRED: verify each pick's line on PrizePicks before betting.\n"
                f"  picks_history.csv will record '{_line_source}' as the line source.\n"
                f"{'!'*60}"
            )

        # Generate predictions
        pred_df = generate_predictions(df, vegas, use_feature_cache=True)

        # Compute regression-to-mean candidates from historical data
        reg_candidates = compute_regression_candidates(df)

        # Build a deeper candidate pool, then sanitize and diversify the published card.
        candidate_pool_size = max(
            int(CONFIG.get('live_pick_candidate_pool', 25)),
            int(CONFIG.get('live_pick_min', 5)) * 10,
        )
        picks = filter_best_picks(
            pred_df,
            regression_candidates=reg_candidates,
            max_picks=candidate_pool_size,
        )

        # Fetch fresh injury report and remove risky availability picks
        inj_report = fetch_injury_data()
        picks = sanity_check_picks(picks, df, inj_report)
        picks = build_live_pick_card(
            picks,
            target_picks=int(CONFIG.get('live_pick_target', 5)),
            min_picks=int(CONFIG.get('live_pick_min', 5)),
        )

        print(f"\n{len(picks)} picks found after live-card filtering")
        if len(picks) < int(CONFIG.get('live_pick_min', 5)):
            print(
                f"  WARNING: only {len(picks)} live picks survived today "
                f"(minimum target {int(CONFIG.get('live_pick_min', 5))})"
            )
        
        if len(picks) > 0:
            # Top 10 display: one pick per player (best confidence), different players only
            seen = []
            top_display = []
            for _, row in picks.iterrows():
                if row['player'] not in seen:
                    seen.append(row['player'])
                    top_display.append(row)
                if len(top_display) >= 10:
                    break
            top_display_df = pd.DataFrame(top_display)
            print("\nTop 10 picks (1 per player):")
            display_cols = ['player', 'prop', 'direction', 'line', 'prediction', 'edge', 'confidence', 'dir_prob', 'p_over_cal', 'p_over_raw', 'meta_prob', 'market_edge', 'model_wr', 'pick_source', 'mp_predicted', 'prob_source']
            display_cols = [c for c in display_cols if c in top_display_df.columns]
            print(top_display_df[display_cols].to_string(index=False))

            # Injury adjustment summary
            inj_picks = picks[picks.get('injury_adj', pd.Series(False, index=picks.index))]
            if 'injury_adj' in picks.columns:
                inj_picks = picks[picks['injury_adj'] == True]
                if len(inj_picks) > 0:
                    print(f"\n  [{len(inj_picks)} picks boosted by injury adjustments]")
                    for _, p in inj_picks.iterrows():
                        print(f"    {p['player']} ({p['prop']}): {p.get('teammates_out', '')}")

            # Stamp the line source onto every pick so picks_history.csv tracks it.
            # This is essential for identifying runs where PrizePicks was unavailable.
            picks['line_source'] = _line_source

            save_picks(picks, pred_df, slate_date=active_slate_date)

    elif command == 'backfill':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        backfill_historical_lines(days)

    elif command == 'rebuild-lines':
        rebuilt = rebuild_historical_lines_from_local_sources(write=True)
        print(f"\nRebuilt historical_lines.csv from local sources: {len(rebuilt):,} rows")
        if len(rebuilt) > 0:
            print(f"Coverage: {rebuilt['game_date'].nunique()} distinct game dates")

    elif command == 'backtest':
        # Backtest
        if not data_path.exists():
            raise FileNotFoundError("No data found. Run 'update' first.")
        
        df = pd.read_csv(data_path, parse_dates=['game_date'])
        
        test_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        results = backtest(df, test_days=test_days)
        
        # Save results
        results_path = CONFIG['output_dir'] / 'backtest_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        detail_records = results.get('detail_records', [])
        if detail_records:
            detail_path = CONFIG['output_dir'] / 'backtest_daily_cards.csv'
            _atomic_csv(pd.DataFrame(detail_records), detail_path)
            print(f"Detailed daily-card picks saved to {detail_path}")
        daily_summary = results.get('daily_summary', [])
        if daily_summary:
            summary_path = CONFIG['output_dir'] / 'backtest_daily_summary.csv'
            _atomic_csv(pd.DataFrame(daily_summary), summary_path)
            print(f"Daily-card summary saved to {summary_path}")
        
    elif command == 'all':
        # Full pipeline
        print("\n[1/4] Fetching data...")
        
        # Check for local file argument
        local_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        if local_file:
            df = fetch_nba_data(local_file=local_file)
        else:
            df = fetch_nba_data(seasons=['2025-26', '2024-25', '2023-24'])
        
        df.to_csv(data_path, index=False)
        
        print("\n[2/4] Creating features...")
        df = create_features(df)
        
        print("\n[3/4] Training models...")
        train_models(df)
        train_ou_classifiers(df)

        print("\n[4/4] Generating predictions...")
        vegas_raw, _line_source_all = fetch_vegas_lines()
        active_slate_date_all = datetime.now().strftime('%Y-%m-%d')
        vegas = pd.DataFrame()
        if len(vegas_raw) > 0:
            save_lines_snapshot(vegas_raw, source=_line_source_all)
            active_live_all, live_slate_date_all = _select_active_slate_lines(vegas_raw)
            active_slate_date_all = live_slate_date_all or active_slate_date_all
            _market_cols_all = [c for c in ['projection_type', 'is_promo'] if c in active_live_all.columns]
            vegas = active_live_all[['player', 'prop', 'line'] + _market_cols_all].copy()
            vegas['num_books'] = active_live_all['num_books'].values if 'num_books' in active_live_all.columns else 1
        pred_df = generate_predictions(df, vegas, use_feature_cache=False)
        reg_candidates = compute_regression_candidates(df)
        candidate_pool_size = max(
            int(CONFIG.get('live_pick_candidate_pool', 25)),
            int(CONFIG.get('live_pick_min', 5)) * 10,
        )
        picks = filter_best_picks(
            pred_df,
            regression_candidates=reg_candidates,
            max_picks=candidate_pool_size,
        )
        inj_report_all = fetch_injury_data()
        picks = sanity_check_picks(picks, df, inj_report_all)
        picks = build_live_pick_card(
            picks,
            target_picks=int(CONFIG.get('live_pick_target', 5)),
            min_picks=int(CONFIG.get('live_pick_min', 5)),
        )

        if len(picks) > 0:
            picks['line_source'] = _line_source_all
            print(f"\n{len(picks)} picks found:")

            print("\n=== TOP 20 PICKS (by rank score) ===")
            display_cols = ['player', 'prop', 'direction', 'line', 'prediction', 'edge', 'confidence', 'dir_prob', 'p_over_cal', 'p_over_raw', 'meta_prob', 'market_edge', 'model_wr', 'pick_source', 'l10_avg', 'prob_source']
            display_cols = [c for c in display_cols if c in picks.columns]
            print(picks.head(20)[display_cols].to_string(index=False))

            save_picks(picks, pred_df, slate_date=active_slate_date_all)
        
    else:
        print(f"Unknown command: {command}")
        print("Available commands: update, train, predict, backtest, backfill, rebuild-lines, all")


if __name__ == "__main__":
    main()
