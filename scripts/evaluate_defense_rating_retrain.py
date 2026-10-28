"""Compare production and full-layer candidate on identical real-line OOF rows.

The trainer persists quantile OOF predictions but not its primary-regression OOF
vector.  Therefore this audit compares the persisted pre-line q50 projection for
both systems against the *same frozen production line*.  It never compares the
two runs' potentially different line lookups.  Pushes are excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTION = ROOT / "models" / "probability_oof_advanced.csv"
DEFAULT_CANDIDATE = ROOT / "models" / "defense_rating_retrain" / "probability_oof_advanced.csv"
DEFAULT_PRODUCTION_FEATURES = ROOT / "models" / "selected_features_advanced.json"
DEFAULT_CANDIDATE_FEATURES = ROOT / "models" / "defense_rating_retrain" / "selected_features_advanced.json"
DEFAULT_DETAIL = ROOT / "output" / "defense_rating_same_row_oof.csv"
DEFAULT_SUMMARY = ROOT / "output" / "defense_rating_same_row_oof_summary.json"
NEW_FEATURES = {
    "opp_def_rtg_pt",
    "opp_pace_pt",
    "opp_def_tier_pt",
    "availability_probability",
    "conditional_minutes_prediction",
    "expected_minutes_prediction",
}
KEY = ["target", "game_date", "player_norm", "actual"]


def _load_oof(path: Path, production: bool) -> pd.DataFrame:
    required = [*KEY, "line", "line_source", "q50"]
    frame = pd.read_csv(path, usecols=required)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise").dt.normalize()
    if frame.duplicated(KEY).any():
        count = int(frame.duplicated(KEY, keep=False).sum())
        raise ValueError(f"{path} has {count} duplicate comparison keys")
    if production:
        frame = frame.loc[frame["line_source"].eq("real")].copy()
        if frame.empty:
            raise ValueError("Production OOF has no real-line rows")
    return frame


def _cluster_interval(daily: pd.DataFrame, value_column: str, draws: int, seed: int) -> tuple[float, float]:
    values = daily[value_column].to_numpy(dtype=float)
    counts = daily["n"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for start in range(0, draws, 500):
        size = min(500, draws - start)
        sampled = rng.integers(0, len(daily), size=(size, len(daily)))
        estimates[start:start + size] = (
            values[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def evaluate(
    production: pd.DataFrame,
    candidate: pd.DataFrame,
    production_features: dict,
    candidate_features: dict,
    draws: int = 5000,
    seed: int = 5051,
    minimum_overlap: float = 0.99,
) -> tuple[pd.DataFrame, dict]:
    right = candidate[KEY + ["q50"]].rename(columns={"q50": "candidate_q50"})
    aligned = production.merge(right, on=KEY, how="inner", validate="one_to_one")
    overlap_rate = len(aligned) / len(production) if len(production) else 0.0
    if overlap_rate < minimum_overlap:
        raise ValueError(
            f"Candidate covers {len(aligned)}/{len(production)} frozen production real-line rows "
            f"({overlap_rate:.2%}); required {minimum_overlap:.2%}"
        )
    missing = production.merge(
        right[KEY], on=KEY, how="left", validate="one_to_one", indicator=True
    )
    missing = missing.loc[missing["_merge"].eq("left_only")].drop(columns="_merge")
    aligned_rows_before_push_exclusion = int(len(aligned))
    aligned = aligned.rename(columns={"q50": "production_q50", "line": "frozen_line"})
    aligned = aligned.loc[aligned["actual"].ne(aligned["frozen_line"])].copy()
    aligned["actual_over"] = aligned["actual"].gt(aligned["frozen_line"])
    aligned["production_direction"] = np.where(
        aligned["production_q50"].gt(aligned["frozen_line"]), "OVER", "UNDER"
    )
    aligned["candidate_direction"] = np.where(
        aligned["candidate_q50"].gt(aligned["frozen_line"]), "OVER", "UNDER"
    )
    aligned["production_hit"] = aligned["production_direction"].eq(
        np.where(aligned["actual_over"], "OVER", "UNDER")
    ).astype(int)
    aligned["candidate_hit"] = aligned["candidate_direction"].eq(
        np.where(aligned["actual_over"], "OVER", "UNDER")
    ).astype(int)
    aligned["always_under_hit"] = (~aligned["actual_over"]).astype(int)
    aligned["candidate_minus_production"] = aligned["candidate_hit"] - aligned["production_hit"]
    aligned["candidate_minus_under"] = aligned["candidate_hit"] - aligned["always_under_hit"]

    rows = []
    for target, group in aligned.groupby("target", sort=True):
        daily = group.groupby("game_date").agg(
            n=("target", "size"),
            candidate_minus_production=("candidate_minus_production", "sum"),
            candidate_minus_under=("candidate_minus_under", "sum"),
        ).reset_index()
        prod_ci = _cluster_interval(daily, "candidate_minus_production", draws, seed)
        under_ci = _cluster_interval(daily, "candidate_minus_under", draws, seed + 1)
        selected = sorted(NEW_FEATURES & set(candidate_features.get(target, [])))
        production_rate = float(group["production_hit"].mean())
        candidate_rate = float(group["candidate_hit"].mean())
        under_rate = float(group["always_under_hit"].mean())
        rows.append(
            {
                "target": target,
                "rows": int(len(group)),
                "dates": int(group["game_date"].nunique()),
                "production_hit_rate": production_rate,
                "candidate_hit_rate": candidate_rate,
                "always_under_hit_rate": under_rate,
                "candidate_minus_production": candidate_rate - production_rate,
                "candidate_minus_production_ci_low": prod_ci[0],
                "candidate_minus_production_ci_high": prod_ci[1],
                "candidate_minus_under": candidate_rate - under_rate,
                "candidate_minus_under_ci_low": under_ci[0],
                "candidate_minus_under_ci_high": under_ci[1],
                "new_features_selected": selected,
                "passes_oof_evidence_gate": bool(selected and prod_ci[0] > 0 and under_ci[0] > 0),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "status": "same_row_oof_complete_not_a_reserved_future_holdout",
        "comparison_projection": "persisted probability-sidecar q50",
        "line_contract": "production real line frozen on identical player/date/prop/actual rows",
        "pushes_excluded": True,
        "bootstrap": {
            "method": "game-date clustered percentile bootstrap",
            "draws": draws,
            "seed": seed,
        },
        "production_real_rows": int(len(production)),
        "aligned_rows_before_push_exclusion": aligned_rows_before_push_exclusion,
        "overlap_rate": float(overlap_rate),
        "minimum_overlap_rate": float(minimum_overlap),
        "excluded_nonoverlap_rows": int(len(missing)),
        "excluded_nonoverlap_by_target": {
            str(target): int(count)
            for target, count in missing.groupby("target").size().items()
        },
        "excluded_nonoverlap_date_min": (
            missing["game_date"].min().date().isoformat() if not missing.empty else None
        ),
        "excluded_nonoverlap_date_max": (
            missing["game_date"].max().date().isoformat() if not missing.empty else None
        ),
        "aligned_nonpush_rows": int(len(aligned)),
        "date_min": aligned["game_date"].min().date().isoformat(),
        "date_max": aligned["game_date"].max().date().isoformat(),
        "new_feature_survival": {
            target: sorted(NEW_FEATURES & set(features))
            for target, features in candidate_features.items()
        },
        "production_defense_features": {
            target: sorted(feature for feature in features if "defense" in feature or feature.startswith("opp_"))
            for target, features in production_features.items()
        },
        "results": table.to_dict("records"),
        "props_passing_oof_evidence_gate": table.loc[
            table["passes_oof_evidence_gate"], "target"
        ].tolist(),
        "promotion_authorized": False,
        "promotion_blocker": (
            "This is chronological OOF evidence, not a newly reserved future period. "
            "No production model may be replaced from this result."
        ),
    }
    return aligned, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--production-features", type=Path, default=DEFAULT_PRODUCTION_FEATURES)
    parser.add_argument("--candidate-features", type=Path, default=DEFAULT_CANDIDATE_FEATURES)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=5051)
    parser.add_argument("--minimum-overlap", type=float, default=0.99)
    args = parser.parse_args()
    production = _load_oof(args.production, production=True)
    candidate = _load_oof(args.candidate, production=False)
    production_features = json.loads(args.production_features.read_text(encoding="utf-8"))
    candidate_features = json.loads(args.candidate_features.read_text(encoding="utf-8"))
    detail, summary = evaluate(
        production, candidate, production_features, candidate_features,
        draws=args.bootstrap_draws, seed=args.seed,
        minimum_overlap=args.minimum_overlap,
    )
    args.detail.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail, index=False, date_format="%Y-%m-%d")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary["results"]).to_string(index=False))
    print(json.dumps({
        "status": summary["status"],
        "aligned_nonpush_rows": summary["aligned_nonpush_rows"],
        "props_passing_oof_evidence_gate": summary["props_passing_oof_evidence_gate"],
        "promotion_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
