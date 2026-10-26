"""Analyze the paid CLV features and the PrizePicks-minus-consensus hypothesis."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

try:
    from scripts.execute_clv_snapshot_manifest import (
        ROOT, atomic_write_csv, atomic_write_json, atomic_write_text, normalize_name,
    )
except ModuleNotFoundError:
    from execute_clv_snapshot_manifest import (
        ROOT, atomic_write_csv, atomic_write_json, atomic_write_text, normalize_name,
    )


PROP_MAP = {
    "PTS": "player_points", "TRB": "player_rebounds", "AST": "player_assists",
    "STL": "player_steals", "BLK": "player_blocks", "TOV": "player_turnovers",
    "PA": "player_points_assists", "PR": "player_points_rebounds",
    "PRA": "player_points_rebounds_assists",
}


def analyze() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    features = pd.read_csv(ROOT / "data" / "clv_features_2026_live.csv", dtype={"nba_game_id": str})
    gap = features.dropna(
        subset=["pp_minus_consensus_t_minus_60", "under_win_pp_t_minus_60"]
    ).copy()
    gap = gap[gap["push_pp_t_minus_60"].eq(0)]
    gap["gap_bucket"] = pd.cut(
        gap["pp_minus_consensus_t_minus_60"],
        [-np.inf, -0.001, 0.001, 0.5, 1.0, np.inf],
        labels=["PP below", "equal", "PP +0..0.5", "PP +0.5..1", "PP +1+"],
    )
    dose = gap.groupby("gap_bucket", observed=True).agg(
        n=("under_win_pp_t_minus_60", "size"),
        under_win_rate=("under_win_pp_t_minus_60", "mean"),
        mean_gap=("pp_minus_consensus_t_minus_60", "mean"),
    ).reset_index()

    movement = features.dropna(
        subset=["line_move_60_to_5", "under_win_consensus_t_minus_5"]
    ).copy()
    movement = movement[movement["push_consensus_t_minus_5"].eq(0)]
    movement["move_bucket"] = pd.cut(
        movement["line_move_60_to_5"],
        [-np.inf, -0.01, 0.01, np.inf],
        labels=["line_down", "unchanged", "line_up"],
    )
    move_table = movement.groupby("move_bucket", observed=True).agg(
        n=("under_win_consensus_t_minus_5", "size"),
        under_win_rate=("under_win_consensus_t_minus_5", "mean"),
        mean_move=("line_move_60_to_5", "mean"),
    ).reset_index()

    full_minutes = features[
        features["mp"].ge(30)
        & features["under_win_consensus_t_minus_5"].notna()
        & features["push_consensus_t_minus_5"].eq(0)
    ].copy()
    regime = full_minutes.groupby("prop").agg(
        n=("under_win_consensus_t_minus_5", "size"),
        consensus_close_under_rate=("under_win_consensus_t_minus_5", "mean"),
    ).reset_index()
    pp_by_prop = gap.groupby("prop").agg(
        pp_n=("under_win_pp_t_minus_60", "size"),
        pp_under_rate=("under_win_pp_t_minus_60", "mean"),
        mean_pp_gap=("pp_minus_consensus_t_minus_60", "mean"),
    ).reset_index()
    regime = regime.merge(pp_by_prop, on="prop", how="outer")

    picks = pd.read_csv(ROOT / "output" / "picks_history.csv", low_memory=False)
    picks["game_date"] = pd.to_datetime(picks["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    picks["player_norm"] = picks["player"].map(normalize_name)
    picks["prop_key"] = picks["prop"].astype(str).str.upper().map(PROP_MAP)
    picks = picks[picks["game_date"].between("2026-03-30", "2026-05-18")].copy()
    joined = picks.merge(
        features,
        left_on=["game_date", "player_norm", "prop_key"],
        right_on=["game_date", "player_norm", "prop"],
        how="left",
        suffixes=("_pick", "_feature"),
    )
    close = joined.dropna(subset=["consensus_line_t_minus_5", "line"]).copy()
    direction_factor = np.where(close["direction"].str.upper().eq("OVER"), 1.0, -1.0)
    close["cross_market_line_clv"] = direction_factor * (
        close["consensus_line_t_minus_5"] - close["line"]
    )
    close["closing_side_probability"] = np.where(
        close["direction"].str.upper().eq("OVER"),
        close["consensus_p_over_t_minus_5"],
        1.0 - close["consensus_p_over_t_minus_5"],
    )

    pp_close = joined.dropna(subset=["pp_line_t_minus_5", "line"]).copy()
    pp_factor = np.where(pp_close["direction"].str.upper().eq("OVER"), 1.0, -1.0)
    pp_close["same_book_line_clv"] = pp_factor * (pp_close["pp_line_t_minus_5"] - pp_close["line"])
    pp_age = pp_close["pp_snapshot_age_minutes_t_minus_5"]

    summary = {
        "gap": {
            "graded_rows": int(len(gap)),
            "pp_below_under_rate": float(dose.loc[dose.gap_bucket.eq("PP below"), "under_win_rate"].iloc[0]),
            "pp_plus_one_under_rate": float(dose.loc[dose.gap_bucket.eq("PP +1+"), "under_win_rate"].iloc[0]),
        },
        "movement": {
            "graded_rows": int(len(movement)),
            "line_down_under_rate": float(move_table.loc[move_table.move_bucket.eq("line_down"), "under_win_rate"].iloc[0]),
            "unchanged_under_rate": float(move_table.loc[move_table.move_bucket.eq("unchanged"), "under_win_rate"].iloc[0]),
            "line_up_under_rate": float(move_table.loc[move_table.move_bucket.eq("line_up"), "under_win_rate"].iloc[0]),
        },
        "regime_validation": {
            "blk_consensus_close_n_30plus": int(regime.loc[regime.prop.eq("player_blocks"), "n"].iloc[0]),
            "blk_consensus_close_under_rate_30plus": float(regime.loc[regime.prop.eq("player_blocks"), "consensus_close_under_rate"].iloc[0]),
            "stl_consensus_close_n_30plus": int(regime.loc[regime.prop.eq("player_steals"), "n"].iloc[0]),
            "stl_consensus_close_under_rate_30plus": float(regime.loc[regime.prop.eq("player_steals"), "consensus_close_under_rate"].iloc[0]),
        },
        "card": {
            "logged_rows": int(len(joined)),
            "graded_rows": int(joined["result"].isin(["WIN", "LOSS"]).sum()),
            "close_consensus_coverage": int(len(close)),
            "mean_cross_market_line_clv": float(close["cross_market_line_clv"].mean()),
            "positive_cross_market_line_clv_rate": float(close["cross_market_line_clv"].gt(0).mean()),
            "mean_closing_side_probability": float(close["closing_side_probability"].mean()),
            "closing_side_probability_above_half_rate": float(close["closing_side_probability"].gt(0.5).mean()),
            "pp_reference_coverage": int(len(pp_close)),
            "median_pp_reference_age_minutes": float(pp_age.median()),
            "pp_reference_within_60_minutes": int(pp_age.le(60).sum()),
        },
        "interpretation": {
            "gap_signal": "confirmed_in_sample_at_larger_scale",
            "late_line_movement": "weak_not_monotonic",
            "same_book_prizepicks_clv": "not_measurable_from_stale_archived_pp_snapshots",
            "sportsbook_price_confirmation": "not_established",
            "stl_market_wide_softness": "not_confirmed_at_scale",
            "blk_market_wide_softness": "weaker_than_prior_small_sample",
        },
    }

    report = f"""# CLV Feature Findings

## What the paid panel resolves

The PrizePicks-minus-consensus gap survives the larger panel. At T-60, UNDER win
rate rises from **{summary['gap']['pp_below_under_rate']:.1%}** when PrizePicks is
below consensus to **{summary['gap']['pp_plus_one_under_rate']:.1%}** when it is at
least one unit higher (`n={summary['gap']['graded_rows']:,}`). This expands the
original 973-key diagnostic to a much larger, price-bearing panel.

The selected card has closing consensus coverage on **{summary['card']['close_consensus_coverage']}**
rows. Direction-adjusted line advantage versus sportsbook consensus averages
**{summary['card']['mean_cross_market_line_clv']:+.3f}** units and is positive on
**{summary['card']['positive_cross_market_line_clv_rate']:.1%}**. This is primarily
a cross-market PrizePicks shading advantage—not proof the model predicts market
movement.

## What remains unresolved

The average sportsbook closing no-vig probability on the selected side is only
**{summary['card']['mean_closing_side_probability']:.1%}**, and exceeds 50% on
**{summary['card']['closing_side_probability_above_half_rate']:.1%}** of covered
picks. Sportsbook prices therefore do not independently confirm the selected
direction.

Archived PrizePicks references are not true closes: their median age at T-5 is
**{summary['card']['median_pp_reference_age_minutes']:.0f} minutes**, with only
**{summary['card']['pp_reference_within_60_minutes']}** observations captured in
the final hour. Odds API credits cannot repair this because PrizePicks is not an
Odds API bookmaker. Next-season collection must snapshot the PrizePicks board
near T-60 and T-5 if same-book CLV is required.

Late sportsbook line movement is weak rather than monotonic: UNDER rates are
{summary['movement']['line_down_under_rate']:.1%} after a downward move,
{summary['movement']['unchanged_under_rate']:.1%} unchanged, and
{summary['movement']['line_up_under_rate']:.1%} after an upward move.

The larger sportsbook sample corrects DEC-040's market-wide claim. At 30+
minutes, T-5 consensus UNDER is only
**{summary['regime_validation']['stl_consensus_close_under_rate_30plus']:.1%}**
for STL (`n={summary['regime_validation']['stl_consensus_close_n_30plus']:,}`)
and **{summary['regime_validation']['blk_consensus_close_under_rate_30plus']:.1%}**
for BLK (`n={summary['regime_validation']['blk_consensus_close_n_30plus']:,}`).
STL is efficiently priced at the book; BLK retains only a modest skew. The much
higher PrizePicks UNDER rates are therefore primarily book-specific shading.

## Productive next model work

1. Add the pre-tip PP-minus-consensus gap, book dispersion, and no-vig consensus
   probability as source-specific candidate features.
2. Evaluate them chronologically against same-date random UNDER and always-UNDER
   controls; do not tune on this complete live window and then quote it as holdout.
3. Keep STL/BLK, AST/TRB, and composite regimes separate.
4. Begin scheduled PrizePicks T-60/T-5 capture next season; do not spend more
   Odds API credits trying to manufacture a missing PrizePicks close.
"""
    return summary, dose, move_table, regime, report


def main() -> int:
    summary, dose, movement, regime, report = analyze()
    atomic_write_json(ROOT / "output" / "clv_feature_findings.json", summary)
    atomic_write_csv(ROOT / "output" / "clv_gap_dose_response.csv", dose)
    atomic_write_csv(ROOT / "output" / "clv_movement_outcomes.csv", movement)
    atomic_write_csv(ROOT / "output" / "clv_regime_validation.csv", regime)
    atomic_write_text(ROOT / "output" / "CLV_FEATURE_FINDINGS.md", report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
