"""Chronologically evaluate probability-ranked UNDER lanes on frozen OOF rows.

The earliest 64 dates fit isotonic calibration. The latest 20 dates are never
used to fit calibration or choose the declared probability floors. Within each
prop/date, the evaluator selects at most the highest-probability eligible UNDER
and compares it with the expected hit rate of a random UNDER from that exact
prop/date pool.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probability_utils import apply_isotonic_calibrators, fit_isotonic_calibrators  # noqa: E402
from train_advanced_models import _sha256_file  # noqa: E402


DECLARED_FLOORS = {
    "ast": 0.60,
    "blk": 0.70,
    "stl": 0.70,
    "trb": 0.55,
    "tov": 0.55,
}
PRODUCTION_LANES = {"ast", "blk", "stl"}


def _paired_summary(rows: pd.DataFrame) -> dict:
    if len(rows) == 0:
        return {
            "n": 0,
            "dates": 0,
            "hit_rate": None,
            "matched_random_under_rate": None,
            "lift": None,
            "lift_ci95_low": None,
            "lift_ci95_high": None,
        }
    delta = rows["hit_under"].astype(float) - rows["matched_random_under_rate"].astype(float)
    lift = float(delta.mean())
    se = float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
    return {
        "n": int(len(rows)),
        "dates": int(rows["game_date"].nunique()),
        "hit_rate": float(rows["hit_under"].mean()),
        "matched_random_under_rate": float(rows["matched_random_under_rate"].mean()),
        "lift": lift,
        "lift_ci95_low": lift - 1.96 * se,
        "lift_ci95_high": lift + 1.96 * se,
    }


def evaluate(oof_path: Path, holdout_dates: int = 20) -> tuple[pd.DataFrame, dict]:
    """Return selected rows and a chronological development/holdout summary."""
    frame = pd.read_csv(oof_path)
    required = {"target", "game_date", "actual", "line", "line_source", "p_over_raw", "actual_over"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF artifact is missing columns: {missing}")
    frame = frame[frame["line_source"].eq("real")].copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise").dt.normalize()
    for col in ["actual", "line", "p_over_raw", "actual_over"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["actual", "line", "p_over_raw", "actual_over"])
    frame = frame[frame["actual"].ne(frame["line"])].copy()
    frame["hit_under"] = frame["actual"].lt(frame["line"]).astype(int)

    dates = sorted(frame["game_date"].unique())
    if holdout_dates < 1 or len(dates) <= holdout_dates:
        raise ValueError(f"Need more than {holdout_dates} OOF dates; found {len(dates)}")
    holdout_start = pd.Timestamp(dates[-holdout_dates])
    development = frame[frame["game_date"] < holdout_start].copy()
    calibrators, _ = fit_isotonic_calibrators(development)

    selected_parts = []
    for split, split_rows in (
        ("development", development),
        ("holdout", frame[frame["game_date"] >= holdout_start].copy()),
    ):
        split_rows = split_rows.copy()
        pool_rates = split_rows.groupby(["game_date", "target"])["hit_under"].mean()
        for target, floor in DECLARED_FLOORS.items():
            target_rows = split_rows[split_rows["target"].eq(target)].copy()
            if len(target_rows) == 0:
                continue
            target_rows["p_under_cal"] = 1.0 - apply_isotonic_calibrators(
                target_rows["p_over_raw"].values,
                target,
                calibrators,
            )
            eligible = target_rows[target_rows["p_under_cal"] >= floor].copy()
            eligible = eligible.sort_values(
                ["game_date", "p_under_cal"], ascending=[True, False]
            ).drop_duplicates(["game_date", "target"], keep="first")
            eligible["split"] = split
            eligible["declared_floor"] = floor
            eligible["production_lane"] = target in PRODUCTION_LANES
            eligible["matched_random_under_rate"] = [
                float(pool_rates.loc[(day, target)]) for day in eligible["game_date"]
            ]
            selected_parts.append(eligible)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    summary = {
        "method": "chronological_calibration_fit_then_latest_date_holdout",
        "oof_path": str(Path(oof_path).resolve()),
        "oof_sha256": _sha256_file(oof_path),
        "development": {
            "date_min": development["game_date"].min().date().isoformat(),
            "date_max": development["game_date"].max().date().isoformat(),
            "dates": int(development["game_date"].nunique()),
        },
        "holdout": {
            "date_min": holdout_start.date().isoformat(),
            "date_max": frame["game_date"].max().date().isoformat(),
            "dates": int(frame.loc[frame["game_date"] >= holdout_start, "game_date"].nunique()),
        },
        "declared_floors": DECLARED_FLOORS,
        "production_lanes": sorted(PRODUCTION_LANES),
        "scope_caveat": (
            "These are frozen real-line OOF rows, not a new unseen NBA season and not a "
            "PrizePicks-standard-only closing-line panel. The test supports ranking and "
            "abstention policy; it is not a guaranteed future win-rate claim."
        ),
        "results": {},
    }
    for split in ["development", "holdout"]:
        split_selected = selected[selected["split"].eq(split)].copy()
        summary["results"][split] = {
            "production_lanes_combined": _paired_summary(
                split_selected[split_selected["production_lane"]]
            ),
            "all_declared_lanes_combined": _paired_summary(split_selected),
            "per_target": {
                target: _paired_summary(split_selected[split_selected["target"].eq(target)])
                for target in DECLARED_FLOORS
            },
        }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=ROOT / "models" / "probability_oof_advanced.csv")
    parser.add_argument("--holdout-dates", type=int, default=20)
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=ROOT / "output" / "probability_under_selection_oof.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "output" / "probability_under_selection_summary.json",
    )
    args = parser.parse_args()
    selected, summary = evaluate(args.oof, holdout_dates=args.holdout_dates)
    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.detail_output, index=False)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["results"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
