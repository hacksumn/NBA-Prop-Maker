"""Audit production-selected feature parity between training and live builders.

The advanced models are trained from ``FeaturePipeline`` output while live and
historical scoring rebuild features through ``nba_props``.  This audit aligns
the two products on the same player-game keys and reports definition drift only
for features selected by the active production models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_RAW = ROOT / "data" / "nba_data.csv"
DEFAULT_TRAINING = ROOT / "data" / "model_data_defenserating_advanced.csv"
DEFAULT_SELECTED = ROOT / "models" / "selected_features_advanced.json"
DEFAULT_DETAIL = ROOT / "output" / "training_live_selected_feature_parity.csv"
DEFAULT_SUMMARY = ROOT / "output" / "training_live_selected_feature_parity_summary.json"
KEY = ["GAME_ID", "PLAYER_ID", "team"]
EXTERNAL_LIVE_FEATURES = {"opp_def_rtg_pt", "opp_pace_pt", "opp_def_tier_pt"}


def _canonicalize_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = sorted(set(KEY) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing parity keys: {missing}")
    out = frame.copy()
    out["GAME_ID"] = pd.to_numeric(out["GAME_ID"], errors="raise").astype("int64")
    out["PLAYER_ID"] = pd.to_numeric(out["PLAYER_ID"], errors="raise").astype("int64")
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    if out.duplicated(KEY).any():
        raise ValueError(f"{label} contains duplicate player-game keys")
    return out


def _selected_feature_map(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Selected-feature artifact is empty or invalid")
    return {str(prop): list(features) for prop, features in payload.items()}


def compare_selected_features(
    training: pd.DataFrame,
    live: pd.DataFrame,
    selected: dict[str, list[str]],
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> tuple[pd.DataFrame, dict]:
    """Compare selected features on identical player-game rows."""
    training = _canonicalize_keys(training, "training frame")
    live = _canonicalize_keys(live, "live frame")
    feature_props: dict[str, list[str]] = {}
    for prop, features in selected.items():
        for feature in features:
            feature_props.setdefault(feature, []).append(prop)

    selected_union = sorted(feature_props)
    comparable = [
        feature for feature in selected_union
        if feature not in EXTERNAL_LIVE_FEATURES
        and feature in training.columns
        and feature in live.columns
    ]
    joined = training[KEY + comparable].merge(
        live[KEY + comparable], on=KEY, how="inner", validate="one_to_one",
        suffixes=("_training", "_live"),
    )
    if joined.empty:
        raise ValueError("Training/live parity join produced zero rows")

    records = []
    for feature in selected_union:
        record = {
            "feature": feature,
            "props": ",".join(sorted(feature_props[feature])),
            "selected_prop_count": len(feature_props[feature]),
            "training_present": feature in training.columns,
            "live_present": feature in live.columns,
            "external_live_attachment": feature in EXTERNAL_LIVE_FEATURES,
            "aligned_rows": int(len(joined)),
        }
        if feature in EXTERNAL_LIVE_FEATURES:
            record["status"] = "external_live_attachment"
            records.append(record)
            continue
        if feature not in training.columns or feature not in live.columns:
            record["status"] = "missing"
            records.append(record)
            continue

        left = pd.to_numeric(joined[f"{feature}_training"], errors="coerce")
        right = pd.to_numeric(joined[f"{feature}_live"], errors="coerce")
        both = left.notna() & right.notna()
        either = left.notna() | right.notna()
        diff = (left[both] - right[both]).abs()
        close = np.isclose(
            left[both].to_numpy(dtype=float), right[both].to_numpy(dtype=float),
            atol=atol, rtol=rtol, equal_nan=True,
        )
        record.update(
            {
                "training_nonnull_rate": float(left.notna().mean()),
                "live_nonnull_rate": float(right.notna().mean()),
                "both_nonnull_rows": int(both.sum()),
                "nonnull_disagreement_rows": int((left.notna() != right.notna()).sum()),
                "exact_match_rate_on_both": float(close.mean()) if len(close) else None,
                "mean_abs_diff": float(diff.mean()) if len(diff) else None,
                "median_abs_diff": float(diff.median()) if len(diff) else None,
                "p95_abs_diff": float(diff.quantile(0.95)) if len(diff) else None,
                "max_abs_diff": float(diff.max()) if len(diff) else None,
                "material_diff_rate": float((diff > 1e-6).mean()) if len(diff) else None,
                "populated_rows": int(either.sum()),
            }
        )
        record["status"] = (
            "match" if record["nonnull_disagreement_rows"] == 0 and bool(close.all())
            else "drift"
        )
        records.append(record)

    detail = pd.DataFrame(records).sort_values(
        ["status", "selected_prop_count", "feature"], ascending=[True, False, True]
    ).reset_index(drop=True)
    status_counts = {str(k): int(v) for k, v in detail["status"].value_counts().items()}
    drifted = detail.loc[detail["status"].eq("drift"), "feature"].tolist()
    missing = detail.loc[detail["status"].eq("missing"), "feature"].tolist()
    affected_props = sorted(
        prop for prop, features in selected.items()
        if set(features) & set(drifted + missing)
    )
    summary = {
        "status": "contract_pass" if not drifted and not missing else "contract_failed",
        "grain": KEY,
        "training_rows": int(len(training)),
        "live_rows": int(len(live)),
        "aligned_rows": int(len(joined)),
        "selected_features": int(len(selected_union)),
        "status_counts": status_counts,
        "drifted_features": drifted,
        "missing_features": missing,
        "affected_props": affected_props,
        "external_live_features": sorted(set(selected_union) & EXTERNAL_LIVE_FEATURES),
        "tolerance": {"atol": atol, "rtol": rtol},
    }
    return detail, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    from nba_props import _build_backtest_feature_frame_v2

    raw = pd.read_csv(args.raw, low_memory=False)
    training = pd.read_csv(args.training, low_memory=False)
    selected = _selected_feature_map(args.selected)
    live = _build_backtest_feature_frame_v2(raw, using_advanced=True)
    detail, summary = compare_selected_features(training, live, selected)
    summary.update(
        {
            "raw_input": str(args.raw),
            "training_input": str(args.training),
            "selected_input": str(args.selected),
        }
    )
    args.detail.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail, index=False)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(detail.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "contract_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
