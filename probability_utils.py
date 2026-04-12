import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

QUANTILE_LEVELS: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90]
# Targets that receive a quantile-regression probability sidecar during training.
# pra/pr/pa are continuous combined stats — quantile regression is the right family
# (nonparametric, better than Poisson for high-range correlated sums).
PROBABILITY_TARGETS = {'pts', 'trb', 'ast', 'pra', 'pr', 'pa'}
# Sparse integer count stats that use Poisson or ordinal sidecars instead.
COUNT_TARGETS = {'stl', 'blk', 'tov'}


def quantile_tag(alpha: float) -> str:
    return f"q{int(round(alpha * 100)):02d}"


def quantile_tags(levels: Iterable[float] = QUANTILE_LEVELS) -> List[str]:
    return [quantile_tag(alpha) for alpha in levels]


def tail_threshold_tag(threshold: int) -> str:
    return f"ge_{int(threshold)}"


def ensure_monotonic_quantiles(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.maximum.accumulate(arr)
    return np.maximum.accumulate(arr, axis=1)


def ensure_monotonic_tail_probs(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.minimum.accumulate(arr)
    return np.minimum.accumulate(arr, axis=1)


def over_probability_from_quantiles(lines: Iterable[float],
                                    quantile_values: np.ndarray,
                                    levels: Iterable[float] = QUANTILE_LEVELS) -> np.ndarray:
    lines_arr = np.asarray(lines, dtype=float)
    q_vals = ensure_monotonic_quantiles(np.asarray(quantile_values, dtype=float))
    q_levels = np.asarray(list(levels), dtype=float)
    if q_vals.ndim == 1:
        q_vals = q_vals.reshape(1, -1)

    probs = np.zeros(len(lines_arr), dtype=float)
    for i, line in enumerate(lines_arr):
        row = q_vals[i]
        if len(row) < 2:
            probs[i] = 0.5
            continue
        lower_span = max(row[1] - row[0], 1e-6)
        upper_span = max(row[-1] - row[-2], 1e-6)
        lower_anchor = row[0] - lower_span * (q_levels[0] / max(q_levels[1] - q_levels[0], 1e-6))
        upper_anchor = row[-1] + upper_span * ((1.0 - q_levels[-1]) / max(q_levels[-1] - q_levels[-2], 1e-6))
        x_points = np.concatenate(([lower_anchor], row, [upper_anchor]))
        x_points = np.maximum.accumulate(x_points + np.arange(len(x_points)) * 1e-6)
        p_points = np.concatenate(([0.0], q_levels, [1.0]))
        cdf = float(np.interp(line, x_points, p_points))
        probs[i] = 1.0 - cdf
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def ordinal_over_probability(lines: Iterable[float],
                             tail_probabilities: np.ndarray,
                             thresholds: Iterable[int]) -> np.ndarray:
    lines_arr = np.asarray(lines, dtype=float)
    tail_probs = np.clip(np.asarray(tail_probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    threshold_arr = np.asarray(list(thresholds), dtype=int)
    if tail_probs.ndim == 1:
        tail_probs = tail_probs.reshape(1, -1)
    tail_probs = ensure_monotonic_tail_probs(tail_probs)

    out = np.full(len(lines_arr), 0.5, dtype=float)
    threshold_to_idx = {int(threshold): idx for idx, threshold in enumerate(threshold_arr)}
    for i, line in enumerate(lines_arr):
        threshold = int(math.floor(line) + 1)
        idx = threshold_to_idx.get(threshold)
        if idx is None:
            if threshold <= int(threshold_arr.min()):
                out[i] = float(tail_probs[i, 0])
            elif threshold > int(threshold_arr.max()):
                out[i] = float(tail_probs[i, -1] * 0.5)
            else:
                lower = threshold_arr[threshold_arr < threshold].max()
                out[i] = float(tail_probs[i, threshold_to_idx[int(lower)]])
        else:
            out[i] = float(tail_probs[i, idx])
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def poisson_over_probability(lines: Iterable[float], rates: Iterable[float]) -> np.ndarray:
    lines_arr = np.asarray(lines, dtype=float)
    rates_arr = np.clip(np.asarray(rates, dtype=float), 1e-6, None)
    out = np.zeros(len(lines_arr), dtype=float)
    for i, (line, rate) in enumerate(zip(lines_arr, rates_arr)):
        threshold = int(math.floor(line) + 1)
        pmf = math.exp(-rate)
        cdf = pmf
        for k in range(1, max(threshold, 1)):
            pmf *= rate / k
            cdf += pmf
        out[i] = 1.0 - cdf if threshold > 0 else 1.0
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def expected_calibration_error(y_true: Iterable[int],
                               p_pred: Iterable[float],
                               n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(y) == 0:
        return float('nan')

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(p, bins[1:-1], right=False)
    ece = 0.0
    for idx in range(n_bins):
        mask = indices == idx
        if not mask.any():
            continue
        pred_mean = p[mask].mean()
        actual_mean = y[mask].mean()
        ece += (mask.sum() / len(y)) * abs(pred_mean - actual_mean)
    return float(ece)


def build_reliability_curve(y_true: Iterable[int],
                            p_pred: Iterable[float],
                            n_bins: int = 10) -> List[Dict]:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1.0 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(p, bins[1:-1], right=False)
    rows = []
    for idx in range(n_bins):
        mask = indices == idx
        rows.append({
            'bin_lower': float(bins[idx]),
            'bin_upper': float(bins[idx + 1]),
            'n': int(mask.sum()),
            'pred_mean': float(p[mask].mean()) if mask.any() else None,
            'actual_mean': float(y[mask].mean()) if mask.any() else None,
        })
    return rows


def compute_binary_probability_metrics(y_true: Iterable[int],
                                       p_pred: Iterable[float],
                                       n_bins: int = 10) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(y) == 0:
        return {
            'n_records': 0,
            'positive_rate': None,
            'brier': None,
            'log_loss': None,
            'ece': None,
        }
    return {
        'n_records': int(len(y)),
        'positive_rate': float(y.mean()),
        'brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'ece': expected_calibration_error(y, p, n_bins=n_bins),
    }


def fit_isotonic_calibrators(prob_oof_df: pd.DataFrame,
                             n_bins: int = 10) -> Tuple[Dict[str, IsotonicRegression], Dict[str, Dict]]:
    calibrators: Dict[str, IsotonicRegression] = {}
    diagnostics: Dict[str, Dict] = {}
    if prob_oof_df is None or len(prob_oof_df) == 0:
        return calibrators, diagnostics

    working = prob_oof_df.copy()
    working = working.dropna(subset=['target', 'p_over_raw', 'actual_over'])
    if len(working) == 0:
        return calibrators, diagnostics

    for target, group in working.groupby('target'):
        y_true = group['actual_over'].astype(int).values
        p_raw = np.clip(group['p_over_raw'].astype(float).values, 1e-6, 1.0 - 1e-6)
        if len(group) < 100 or len(np.unique(y_true)) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds='clip')
        sample_weight = group['sample_weight'].values if 'sample_weight' in group.columns else None
        iso.fit(p_raw, y_true, sample_weight=sample_weight)
        p_cal = np.clip(iso.transform(p_raw), 1e-6, 1.0 - 1e-6)
        calibrators[target] = iso
        diagnostics[target] = {
            'raw_metrics': compute_binary_probability_metrics(y_true, p_raw, n_bins=n_bins),
            'calibrated_metrics': compute_binary_probability_metrics(y_true, p_cal, n_bins=n_bins),
            'reliability_raw': build_reliability_curve(y_true, p_raw, n_bins=n_bins),
            'reliability_calibrated': build_reliability_curve(y_true, p_cal, n_bins=n_bins),
        }
    return calibrators, diagnostics


def compute_sigma_context_analysis(
    prob_oof_df: pd.DataFrame,
    n_sigma_bins: int = 4,
    n_bins: int = 10,
) -> Dict[str, Dict]:
    """
    Analyze how P(over) calibration varies with predicted distributional width (σ).

    For each target in the OOF frame, computes σ = (q75 - q25) / 1.35 per row, then
    buckets rows by σ quartile and reports Brier, ECE, hit rate, and mean predicted
    probability per bucket.  Rows without q25/q75 columns (e.g. ordinal/Poisson sidecars)
    are skipped gracefully.

    Args:
        prob_oof_df: OOF probability frame, must contain 'target', 'p_over_raw',
                     'actual_over', 'q25', 'q75'.
        n_sigma_bins: Number of sigma quantile buckets (default 4 = quartiles).
        n_bins: Bins for ECE calculation.

    Returns:
        Dict keyed by target string, each value is a diagnostics dict.
    """
    if prob_oof_df is None or len(prob_oof_df) == 0:
        return {}

    working = prob_oof_df.copy()
    if 'q25' not in working.columns or 'q75' not in working.columns:
        return {}

    working = working.dropna(subset=['target', 'p_over_raw', 'actual_over', 'q25', 'q75'])
    if len(working) == 0:
        return {}

    working['_sigma'] = ((working['q75'] - working['q25']) / 1.35).clip(lower=0.01)

    result: Dict[str, Dict] = {}
    for target, group in working.groupby('target'):
        group = group.copy()
        y_true = group['actual_over'].astype(int).values
        p_raw = np.clip(group['p_over_raw'].astype(float).values, 1e-6, 1.0 - 1e-6)
        sigma = group['_sigma'].values

        percentile_edges = np.linspace(0, 100, n_sigma_bins + 1)
        sigma_edges = np.unique(np.nanpercentile(sigma, percentile_edges))

        bucket_rows = []
        for i in range(len(sigma_edges) - 1):
            lo = sigma_edges[i]
            hi = sigma_edges[i + 1]
            mask = (sigma >= lo) & (sigma <= hi if i == len(sigma_edges) - 2 else sigma < hi)
            if mask.sum() < 20:
                continue
            metrics = compute_binary_probability_metrics(y_true[mask], p_raw[mask], n_bins=n_bins)
            bucket_rows.append({
                'sigma_lo': round(float(lo), 3),
                'sigma_hi': round(float(hi), 3),
                'n': int(mask.sum()),
                'mean_sigma': round(float(sigma[mask].mean()), 3),
                'mean_p_over': round(float(p_raw[mask].mean()), 3),
                'hit_rate': round(float(y_true[mask].mean()), 3),
                'brier': round(float(metrics['brier']), 4) if metrics.get('brier') is not None else None,
                'ece': round(float(metrics['ece']), 4) if metrics.get('ece') is not None else None,
                'log_loss': round(float(metrics['log_loss']), 4) if metrics.get('log_loss') is not None else None,
            })

        result[target] = {
            'n_rows': int(len(group)),
            'mean_sigma': round(float(sigma.mean()), 3),
            'sigma_p25': round(float(np.percentile(sigma, 25)), 3),
            'sigma_p50': round(float(np.percentile(sigma, 50)), 3),
            'sigma_p75': round(float(np.percentile(sigma, 75)), 3),
            'sigma_p90': round(float(np.percentile(sigma, 90)), 3),
            'sigma_buckets': bucket_rows,
        }
    return result


def apply_isotonic_calibrators(raw_probs: Iterable[float],
                               target: str,
                               calibrators: Dict[str, IsotonicRegression]) -> np.ndarray:
    raw = np.asarray(raw_probs, dtype=float)
    out = np.full(len(raw), np.nan, dtype=float)
    valid = np.isfinite(raw)
    if not valid.any():
        return out
    raw_valid = np.clip(raw[valid], 1e-6, 1.0 - 1e-6)
    calibrator = calibrators.get(target)
    if calibrator is None:
        out[valid] = raw_valid
        return out
    out[valid] = np.clip(np.asarray(calibrator.transform(raw_valid), dtype=float), 1e-6, 1.0 - 1e-6)
    return out
