"""Evaluate sportsbook market features on the expanded 80-date CLV panel.

This is the high-power complement to the PrizePicks-specific test. It evaluates
consensus/no-vig/dispersion/movement/availability against consensus UNDER
outcomes, uses expanding chronological folds for feature selection, and keeps
the latest 20 dates with archived model predictions untouched for a 100-pick
selection holdout and apples-to-apples current-model comparison.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scripts.evaluate_clv_walk_forward as wf
    from scripts.execute_clv_snapshot_manifest import ROOT, atomic_write_csv, atomic_write_json
except ModuleNotFoundError:
    import evaluate_clv_walk_forward as wf
    from execute_clv_snapshot_manifest import ROOT, atomic_write_csv, atomic_write_json


ACTIVE_PROPS = {"player_points", "player_rebounds", "player_assists"}
HOLDOUT_DATES = 20
MIN_BOOKS = 2
MIN_PAIRED_PRICE_SHARE = 0.90

SPECS = {
    "t_minus_60": wf.SnapshotSpec(
        name="t_minus_60",
        line_column="consensus_line_t_minus_60",
        label_column="under_win_consensus_t_minus_60",
        push_column="push_consensus_t_minus_60",
        candidate_groups={
            "sportsbook_consensus_line": ["consensus_line_t_minus_60"],
            "no_vig_probability": ["consensus_p_over_t_minus_60"],
            "book_dispersion": [
                "line_range_t_minus_60", "line_std_t_minus_60", "p_over_std_t_minus_60",
            ],
            "market_availability": [
                "book_count_t_minus_60", "paired_price_share_t_minus_60",
            ],
        },
        forbidden_groups={
            "t60_to_t5_movement": "not known until T-5",
            "market_withdrawal": "not known until T-5",
        },
    ),
    "t_minus_5": wf.SnapshotSpec(
        name="t_minus_5",
        line_column="consensus_line_t_minus_5",
        label_column="under_win_consensus_t_minus_5",
        push_column="push_consensus_t_minus_5",
        candidate_groups={
            "sportsbook_consensus_line": ["consensus_line_t_minus_5"],
            "no_vig_probability": ["consensus_p_over_t_minus_5"],
            "book_dispersion": [
                "line_range_t_minus_5", "line_std_t_minus_5", "p_over_std_t_minus_5",
            ],
            "t60_to_t5_movement": [
                "line_move_60_to_5", "p_over_move_60_to_5", "dispersion_move_60_to_5",
            ],
            "market_withdrawal_availability": [
                "book_count_t_minus_5", "paired_price_share_t_minus_5",
                "market_withdrawn_by_t5",
            ],
        },
        forbidden_groups={},
    ),
}


def load_combined_features() -> pd.DataFrame:
    paths = [
        ROOT / "data" / "clv_features_2026_expansion.csv",
        ROOT / "data" / "clv_features_2026_live.csv",
    ]
    frames = [pd.read_csv(path, low_memory=False, dtype={"nba_game_id": str}) for path in paths]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined = combined.sort_values(["game_date", "nba_game_id", "player_norm", "prop"])
    return combined.drop_duplicates(["nba_game_id", "player_norm", "prop"], keep="last").reset_index(drop=True)


def build_pool(features: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    spec = SPECS[snapshot]
    kind = snapshot
    line = pd.to_numeric(features.get(spec.line_column), errors="coerce")
    label = pd.to_numeric(features.get(spec.label_column), errors="coerce")
    push = pd.to_numeric(features.get(spec.push_column), errors="coerce").fillna(0).astype(bool)
    books = pd.to_numeric(features.get(f"book_count_{kind}"), errors="coerce")
    paired = pd.to_numeric(features.get(f"paired_price_share_{kind}"), errors="coerce")
    eligible = (
        features["prop"].isin(ACTIVE_PROPS)
        & line.notna()
        & label.notna()
        & ~push
        & books.ge(MIN_BOOKS)
        & paired.ge(MIN_PAIRED_PRICE_SHARE)
    )
    pool = features.loc[eligible].copy()
    pool["under_win"] = label.loc[eligible].astype(int)
    pool["pp_projection_type"] = "sportsbook_consensus"
    pool["snapshot_kind"] = snapshot
    predictions = wf.load_archived_predictions(pool["game_date"].unique())
    pool = pool.merge(
        predictions[["game_date", "player_norm", "prop", "model_prediction"]],
        on=["game_date", "player_norm", "prop"],
        how="left",
        validate="many_to_one",
    )
    pool["model_under_edge"] = line.loc[eligible].to_numpy() - pool["model_prediction"]
    return pool


def make_development_folds(dates: list[str]) -> list[tuple[list[str], list[str]]]:
    ordered = sorted(dates)
    if len(ordered) < 24:
        raise ValueError(f"Need at least 24 development dates, found {len(ordered)}")
    validation_dates = 24
    initial_train = ordered[:-validation_dates]
    validation = ordered[-validation_dates:]
    chunks = [list(chunk) for chunk in np.array_split(validation, 4) if len(chunk)]
    folds = []
    seen = list(initial_train)
    for chunk in chunks:
        folds.append((list(seen), chunk))
        seen.extend(chunk)
    return folds


def _holdout_dates(pool: pd.DataFrame) -> list[str]:
    dates = (
        pool.loc[pool["model_prediction"].notna(), "game_date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if len(dates) < HOLDOUT_DATES:
        raise ValueError(f"Need {HOLDOUT_DATES} archived-prediction dates, found {len(dates)}")
    return dates[-HOLDOUT_DATES:]


def _comparison_rows(scored: pd.DataFrame, snapshot: str) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    overlap = scored[scored["model_prediction"].notna()].copy()
    rows = []
    daily = []
    selected = []
    for method, score in [("market_feature_overlay", "overlay_score"), ("current_model_edge", "model_under_edge")]:
        card = wf._top_card(overlap, score)
        metrics = wf._score_predictions(overlap, score)
        random_rates = wf.matched_random_card_distribution(overlap, card, seed=wf.SEED + len(rows) + 71)
        low, high = wf._wilson_interval(int(metrics["wins"]), int(metrics["n"]))
        rows.append({
            "snapshot_kind": snapshot,
            "method": method,
            **metrics,
            "hit_rate_ci_low": low,
            "hit_rate_ci_high": high,
            "matched_random_hit_rate": float(np.nanmean(random_rates)),
            "lift_vs_random_mean_pp": float(metrics["hit_rate"] - np.nanmean(random_rates)) * 100.0,
            "one_sided_random_p_value": float((1 + np.sum(random_rates >= metrics["hit_rate"])) / (len(random_rates) + 1)),
        })
        keep = [
            "game_date", "nba_game_id", "player", "player_norm", "prop", "actual", "under_win",
            f"consensus_line_{snapshot}", f"consensus_p_over_{snapshot}",
            "model_prediction", "model_under_edge", "overlay_score",
        ]
        picked = card[[column for column in keep if column in card.columns]].copy()
        picked["method"] = method
        picked["snapshot_kind"] = snapshot
        selected.append(picked)
        d = card.groupby("game_date").agg(n=("under_win", "size"), wins=("under_win", "sum")).reset_index()
        d["hit_rate"] = d["wins"] / d["n"]
        d["method"] = method
        d["snapshot_kind"] = snapshot
        daily.append(d)
    wins = int(overlap["under_win"].sum())
    n = int(len(overlap))
    low, high = wf._wilson_interval(wins, n)
    rows.append({
        "snapshot_kind": snapshot,
        "method": "always_under_pool",
        "n": n,
        "wins": wins,
        "hit_rate": wins / n,
        "dates": int(overlap["game_date"].nunique()),
        "hit_rate_ci_low": low,
        "hit_rate_ci_high": high,
        "matched_random_hit_rate": np.nan,
        "lift_vs_random_mean_pp": np.nan,
        "one_sided_random_p_value": np.nan,
    })
    return rows, pd.concat(daily, ignore_index=True), pd.concat(selected, ignore_index=True)


def run() -> tuple[
    dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    features = load_combined_features()
    atomic_write_csv(ROOT / "data" / "clv_features_2026_combined.csv", features)
    old_base = list(wf.BASE_NUMERIC_FEATURES)
    wf.BASE_NUMERIC_FEATURES = []
    all_pools, all_selection, all_folds, all_comparisons, all_daily, all_selected = [], [], [], [], [], []
    summaries = {}
    try:
        for snapshot, spec in SPECS.items():
            pool = build_pool(features, snapshot)
            holdout_dates = _holdout_dates(pool)
            cutoff = min(holdout_dates)
            development = pool[pool["game_date"] < cutoff].copy()
            holdout = pool[pool["game_date"].isin(holdout_dates)].copy()
            folds = make_development_folds(sorted(development["game_date"].unique()))
            selected_groups, selection, fold_results = wf.select_feature_groups(development, spec, folds)
            selected_features = [feature for group in selected_groups for feature in spec.candidate_groups[group]]
            model, result = wf._fit_and_score(development, holdout, selected_features)
            scored = result["scored"]
            comparisons, daily, selected = _comparison_rows(scored, snapshot)
            model_path = ROOT / "models" / f"sportsbook_selection_overlay_{snapshot}.pkl"
            overlay_row = next(row for row in comparisons if row["method"] == "market_feature_overlay")
            current_row = next(row for row in comparisons if row["method"] == "current_model_edge")
            sportsbook_holdout_pass = bool(
                selected_groups
                and overlay_row["n"] >= 100
                and overlay_row["dates"] >= 20
                and overlay_row["hit_rate"] > overlay_row["matched_random_hit_rate"]
                and overlay_row["hit_rate"] > current_row["hit_rate"]
            )
            with model_path.open("wb") as handle:
                pickle.dump({
                    "model": model,
                    "selected_feature_groups": selected_groups,
                    "selected_features": selected_features,
                    "sportsbook_holdout_gate_pass": sportsbook_holdout_pass,
                    "enabled_for_production": False,
                    "production_blocker": "label and line are sportsbook consensus; live card source is PrizePicks",
                    "holdout_dates": holdout_dates,
                    "policy_grain": "sportsbook_consensus_under",
                }, handle)
            summaries[snapshot] = {
                "eligible_rows": int(len(pool)),
                "eligible_dates": int(pool["game_date"].nunique()),
                "development_rows": int(len(development)),
                "development_dates": int(development["game_date"].nunique()),
                "validation_dates": int(sum(len(v) for _, v in folds)),
                "holdout_rows": int(len(holdout)),
                "holdout_dates": len(holdout_dates),
                "holdout_dates_list": holdout_dates,
                "holdout_selected_n": int(overlay_row["n"]),
                "selected_feature_groups": selected_groups,
                "selected_features": selected_features,
                "sportsbook_holdout_gate_pass": sportsbook_holdout_pass,
                "production_gate_pass": False,
                "production_blocker": "label and line are sportsbook consensus; live card source is PrizePicks",
                "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            }
            all_pools.append(pool)
            all_selection.append(selection)
            all_folds.append(fold_results)
            all_comparisons.extend(comparisons)
            all_daily.append(daily)
            all_selected.append(selected)
    finally:
        wf.BASE_NUMERIC_FEATURES = old_base
    summary = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z"),
        "combined_feature_rows": int(len(features)),
        "combined_dates": int(features["game_date"].nunique()),
        "policy_grain": "sportsbook consensus UNDER; PTS/TRB/AST; >=2 books; >=90% paired prices",
        "holdout_contract": "latest 20 dates with archived model predictions; exactly 100 top-five selections when each date is populated",
        "snapshots": summaries,
        "odds_api_expansion_credits_used": 42129,
        "odds_api_balance_after_expansion": 13436,
        "prizepicks_gap_included": False,
        "prizepicks_gap_reason": "Odds API has no historical PrizePicks market; evaluated separately on the frozen PP overlap.",
    }
    return (
        summary,
        pd.concat(all_pools, ignore_index=True),
        pd.concat(all_selection, ignore_index=True),
        pd.concat(all_folds, ignore_index=True),
        pd.DataFrame(all_comparisons),
        pd.concat(all_daily, ignore_index=True),
        pd.concat(all_selected, ignore_index=True),
    )


def main() -> int:
    summary, pool, selection, folds, comparisons, daily, selected = run()
    atomic_write_json(ROOT / "output" / "sportsbook_expansion_summary.json", summary)
    policy = {
        snapshot: {
            "selected_feature_groups": details["selected_feature_groups"],
            "selected_features": details["selected_features"],
            "sportsbook_holdout_gate_pass": details["sportsbook_holdout_gate_pass"],
            "enabled_for_production": False,
            "production_blocker": details["production_blocker"],
            "model_path": details["model_path"],
        }
        for snapshot, details in summary["snapshots"].items()
    }
    atomic_write_json(ROOT / "models" / "sportsbook_selection_policy.json", policy)
    atomic_write_csv(ROOT / "data" / "sportsbook_expansion_eligible_pool.csv", pool)
    atomic_write_csv(ROOT / "output" / "sportsbook_expansion_feature_selection.csv", selection)
    atomic_write_csv(ROOT / "output" / "sportsbook_expansion_fold_results.csv", folds)
    atomic_write_csv(ROOT / "output" / "sportsbook_expansion_holdout_comparison.csv", comparisons)
    atomic_write_csv(ROOT / "output" / "sportsbook_expansion_daily.csv", daily)
    atomic_write_csv(ROOT / "output" / "sportsbook_expansion_holdout_selections.csv", selected)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
