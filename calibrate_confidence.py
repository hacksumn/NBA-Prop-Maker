"""
calibrate_confidence.py
=======================
Calibrates model probabilities from OOF sidecar outputs and, for backward
compatibility, also rebuilds the legacy confidence calibrator used for ranking.

Run after advanced-model training:
    python calibrate_confidence.py

Outputs:
    models/probability_calibrator_advanced.pkl
    models/probability_diagnostics_advanced.json
    models/confidence_calibrator.pkl
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from probability_utils import fit_isotonic_calibrators, compute_sigma_context_analysis

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
HIST_PATH = ROOT / "output" / "picks_history.csv"
PROB_OOF_PATH = MODEL_DIR / "probability_oof_advanced.csv"
LEGACY_OUT_PATH = MODEL_DIR / "confidence_calibrator.pkl"
PROB_OUT_PATH = MODEL_DIR / "probability_calibrator_advanced.pkl"
PROB_DIAG_PATH = MODEL_DIR / "probability_diagnostics_advanced.json"
SIGMA_CONTEXT_PATH = MODEL_DIR / "sigma_context_analysis.json"

MODEL_DIR.mkdir(exist_ok=True)


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def calibrate_probability_outputs() -> Tuple[Dict, Dict]:
    if not PROB_OOF_PATH.exists():
        print("probability_oof_advanced.csv not found - probability calibration skipped")
        return {}, {}

    prob_oof = pd.read_csv(PROB_OOF_PATH)
    if 'line_source' in prob_oof.columns:
        prob_oof = prob_oof[prob_oof['line_source'] == 'real'].copy()
    prob_oof = prob_oof.dropna(subset=['target', 'p_over_raw', 'actual_over'])
    if len(prob_oof) == 0:
        print("No real-line OOF probability rows available - probability calibration skipped")
        return {}, {}

    calibrators, diagnostics = fit_isotonic_calibrators(prob_oof)
    payload = {
        'calibrators': calibrators,
        'diagnostics': diagnostics,
        'n_rows': int(len(prob_oof)),
        'targets': sorted(calibrators.keys()),
    }
    with open(PROB_OUT_PATH, 'wb') as f:
        pickle.dump(payload, f)
    with open(PROB_DIAG_PATH, 'w') as f:
        json.dump(_json_safe({
            'n_rows': int(len(prob_oof)),
            'targets': sorted(diagnostics.keys()),
            'diagnostics': diagnostics,
        }), f, indent=2)

    print(f"Probability OOF rows: {len(prob_oof):,}")
    for target in sorted(diagnostics.keys()):
        raw = diagnostics[target].get('raw_metrics', {})
        cal = diagnostics[target].get('calibrated_metrics', {})
        raw_brier = raw.get('brier')
        cal_brier = cal.get('brier')
        raw_ece = raw.get('ece')
        cal_ece = cal.get('ece')
        raw_ll = raw.get('log_loss')
        cal_ll = cal.get('log_loss')
        print(
            f"  {target.upper():3s} | "
            f"Brier {raw_brier:.4f} -> {cal_brier:.4f} | "
            f"LogLoss {raw_ll:.4f} -> {cal_ll:.4f} | "
            f"ECE {raw_ece:.4f} -> {cal_ece:.4f}"
        )

    print(f"Probability calibrators saved -> {PROB_OUT_PATH}")
    print(f"Probability diagnostics saved -> {PROB_DIAG_PATH}")

    # sigma-context analysis: how does calibration quality vary with predicted spread?
    sigma_context = compute_sigma_context_analysis(prob_oof)
    if sigma_context:
        with open(SIGMA_CONTEXT_PATH, 'w') as f:
            json.dump(_json_safe(sigma_context), f, indent=2)
        print("\nsigma-context analysis (sigma = IQR/1.35 from quantile spread):")
        for target in sorted(sigma_context.keys()):
            info = sigma_context[target]
            print(f"  {target.upper():3s} | n={info['n_rows']:,} | "
                  f"sigma p25={info['sigma_p25']:.2f} p50={info['sigma_p50']:.2f} "
                  f"p75={info['sigma_p75']:.2f} p90={info['sigma_p90']:.2f}")
            for bucket in info.get('sigma_buckets', []):
                brier_str = f"{bucket['brier']:.4f}" if bucket.get('brier') is not None else "n/a"
                ece_str = f"{bucket['ece']:.4f}" if bucket.get('ece') is not None else "n/a"
                print(f"    sigma [{bucket['sigma_lo']:.2f}, {bucket['sigma_hi']:.2f}) "
                      f"n={bucket['n']:>5}  hit={bucket['hit_rate']:.1%}  "
                      f"Brier={brier_str}  ECE={ece_str}")
        print(f"sigma-context analysis saved -> {SIGMA_CONTEXT_PATH}")

    return calibrators, diagnostics


def calibrate_legacy_confidence() -> Dict:
    if not HIST_PATH.exists():
        print("picks_history.csv not found - legacy confidence calibration skipped")
        return {}

    hist = pd.read_csv(HIST_PATH)
    graded = hist[hist["result"].isin(["WIN", "LOSS"])].copy()
    graded["win"] = (graded["result"] == "WIN").astype(int)
    graded["confidence"] = pd.to_numeric(graded["confidence"], errors="coerce")
    graded = graded.dropna(subset=["confidence", "win"])
    if len(graded) == 0:
        print("No graded picks found - legacy confidence calibration skipped")
        return {}

    print(f"\nLegacy graded picks: {len(graded):,}")
    print(f"Legacy overall win rate: {graded['win'].mean()*100:.1f}%")

    X = graded["confidence"].values
    y = graded["win"].values
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X, y)

    stat_dir_rates = {}
    for (prop, direction), group in graded.groupby(["prop", "direction"]):
        if len(group) < 10:
            continue
        stat_dir_rates[f"{prop.upper()}_{direction.upper()}"] = round(float(group["win"].mean()), 4)

    calibrator = {
        "iso": iso,
        "stat_dir_rates": stat_dir_rates,
        "n_graded": int(len(graded)),
        "overall_wr": float(graded["win"].mean()),
    }
    with open(LEGACY_OUT_PATH, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"Legacy confidence calibrator saved -> {LEGACY_OUT_PATH}")
    return calibrator


def compute_sigma_forward_validation() -> Dict:
    """
    Report forward hit rates for real picks stratified by sigma tier.

    Reads picks_history.csv, finds graded rows (WIN/LOSS) that have a sigma
    value populated, then buckets by prop + direction + sigma tier (low/mid/high
    relative to p25/p75 thresholds from sigma_context_analysis.json).  Reports
    n, win_rate, mean confidence, mean edge, and mean sigma per bucket.

    This is the live forward-validation counterpart to the OOF-based
    compute_sigma_context_analysis() which runs on training data.  After enough
    graded picks accumulate with sigma populated, this reveals whether the
    low-sigma mispricing edge (68% UNDER hit for AST, 63% for TRB) persists
    in production data.

    Output: models/sigma_forward_validation.json
    """
    if not HIST_PATH.exists():
        print("picks_history.csv not found - sigma forward validation skipped")
        return {}

    hist = pd.read_csv(HIST_PATH)
    if 'sigma' not in hist.columns:
        print("picks_history.csv has no sigma column yet - sigma forward validation skipped")
        print("  (sigma will appear after the next daily run; re-run calibrate_confidence.py then)")
        return {}

    graded = hist[hist['result'].isin(['WIN', 'LOSS'])].copy()
    graded['win'] = (graded['result'] == 'WIN').astype(int)
    graded['sigma'] = pd.to_numeric(graded['sigma'], errors='coerce')
    # confidence and edge are optional; older history rows may not have them
    if 'confidence' in graded.columns:
        graded['confidence'] = pd.to_numeric(graded['confidence'], errors='coerce')
    else:
        graded['confidence'] = np.nan
    if 'edge' in graded.columns:
        graded['edge'] = pd.to_numeric(graded['edge'], errors='coerce')
    else:
        graded['edge'] = np.nan

    has_sigma = graded['sigma'].notna()
    if not has_sigma.any():
        print("No graded picks with sigma values yet - sigma forward validation skipped")
        print("  (sigma will appear in history after the next daily run)")
        return {}

    sigma_graded = graded[has_sigma].copy()
    print(f"\nsigma forward validation: {len(sigma_graded):,} graded picks with sigma "
          f"({has_sigma.sum()}/{len(graded)} graded rows)")

    # Load p25/p75 thresholds from sigma_context_analysis.json
    thresholds: Dict[str, Dict[str, float]] = {}
    if SIGMA_CONTEXT_PATH.exists():
        try:
            with open(SIGMA_CONTEXT_PATH) as f:
                ctx = json.load(f)
            for target, info in ctx.items():
                entry: Dict[str, float] = {}
                if info.get('sigma_p25') is not None:
                    entry['p25'] = float(info['sigma_p25'])
                if info.get('sigma_p75') is not None:
                    entry['p75'] = float(info['sigma_p75'])
                if entry:
                    thresholds[target] = entry
        except Exception:
            pass

    result: Dict = {}
    for (prop, direction), group in sigma_graded.groupby(['prop', 'direction']):
        tgt = str(prop).lower()
        p25 = thresholds.get(tgt, {}).get('p25')
        p75 = thresholds.get(tgt, {}).get('p75')

        def _bucket(label: str, rows: pd.DataFrame) -> Dict:
            if len(rows) == 0:
                return {}
            return {
                'tier': label,
                'n': int(len(rows)),
                'win_rate': round(float(rows['win'].mean()), 4),
                'mean_confidence': round(float(rows['confidence'].dropna().mean()), 2) if rows['confidence'].notna().any() else None,
                'mean_edge': round(float(rows['edge'].dropna().mean()), 3) if rows['edge'].notna().any() else None,
                'mean_sigma': round(float(rows['sigma'].mean()), 3),
            }

        tiers = [_bucket('all', group)]
        if p25 is not None:
            low = group[group['sigma'] < p25]
            mid = group[(group['sigma'] >= p25) & (group['sigma'] <= (p75 if p75 else 1e9))]
            tiers.append(_bucket('low_sigma', low))
            tiers.append(_bucket('mid_sigma', mid))
        if p75 is not None:
            high = group[group['sigma'] > p75]
            tiers.append(_bucket('high_sigma', high))

        buckets = [b for b in tiers if b and b.get('n', 0) >= 5]
        if buckets:
            key = f"{str(prop).upper()}_{str(direction).upper()}"
            result[key] = {
                'prop': str(prop).upper(),
                'direction': str(direction).upper(),
                'p25_threshold': p25,
                'p75_threshold': p75,
                'buckets': buckets,
            }
            print(f"  {key}")
            for b in buckets:
                wr = b['win_rate']
                edge_str = f"{b['mean_edge']:+.3f}" if b['mean_edge'] is not None else " n/a "
                print(f"    {b['tier']:12s}  n={b['n']:>4}  win={wr:.1%}  "
                      f"edge={edge_str}  sigma={b['mean_sigma']:.2f}")

    out_path = MODEL_DIR / 'sigma_forward_validation.json'
    with open(out_path, 'w') as f:
        json.dump(_json_safe(result), f, indent=2)
    print(f"sigma forward validation saved -> {out_path}")
    return result


def main():
    calibrate_probability_outputs()
    calibrate_legacy_confidence()
    compute_sigma_forward_validation()


if __name__ == "__main__":
    main()
