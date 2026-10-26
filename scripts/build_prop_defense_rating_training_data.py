"""Join point-in-time team-defense ratings onto advanced prop training rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROP_INPUT = ROOT / "data" / "model_data_advanced.csv"
DEFAULT_RATING_INPUT = ROOT / "data" / "pointintime_team_defense_ratings.csv"
DEFAULT_OUTPUT = ROOT / "data" / "model_data_defenserating_advanced.csv"
DEFAULT_SUMMARY = ROOT / "output" / "defense_rating_prop_training_join_summary.json"

PROP_KEY = ["GAME_ID", "team"]
RATING_KEY = ["GAME_ID", "team"]
DEFENSE_RATING_PROP_FEATURES = [
    "opp_def_rtg_pt",
    "opp_pace_pt",
    "opp_def_tier_pt",
]
RATING_REQUIRED = [
    *RATING_KEY,
    "game_date",
    "opponent",
    "is_home",
    "rating_asof_date",
    "team_rating_source",
    "opp_rating_source",
    "n_prior_games",
    "opp_n_prior_games",
    "off_rtg_pt",
    "def_rtg_pt",
    "net_rtg_pt",
    "pace_pt",
    *DEFENSE_RATING_PROP_FEATURES,
    "opp_def_rank_pt",
    "matchup_expected_ortg_pt",
    "rating_league_avg",
    "rating_hca",
]
PROP_REQUIRED = [*PROP_KEY, "game_date"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _normalise(prop: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prop = prop.copy()
    ratings = ratings.copy()
    prop["GAME_ID"] = pd.to_numeric(prop["GAME_ID"], errors="raise").astype("int64")
    ratings["GAME_ID"] = pd.to_numeric(ratings["GAME_ID"], errors="raise").astype("int64")
    for frame in [prop, ratings]:
        frame["team"] = frame["team"].astype(str).str.strip().str.upper()
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise").dt.normalize()
    ratings["opponent"] = ratings["opponent"].astype(str).str.strip().str.upper()
    ratings["rating_asof_date"] = pd.to_datetime(
        ratings["rating_asof_date"], errors="raise"
    ).dt.normalize()
    if "opp" in prop.columns:
        prop["opp"] = prop["opp"].astype("string").str.strip().str.upper()
    return prop, ratings


def _validate_ratings(ratings: pd.DataFrame) -> None:
    if ratings.duplicated(RATING_KEY).any():
        count = int(ratings.duplicated(RATING_KEY, keep=False).sum())
        raise ValueError(f"Point-in-time ratings have {count} rows on duplicate keys")
    if ratings[RATING_REQUIRED].isna().any().any():
        nulls = ratings[RATING_REQUIRED].isna().sum()
        raise ValueError(f"Point-in-time ratings contain required nulls: {nulls[nulls.gt(0)].to_dict()}")
    chronology = ratings["rating_asof_date"] >= ratings["game_date"]
    if chronology.any():
        raise ValueError(f"Point-in-time ratings have {int(chronology.sum())} chronology violations")
    for column in ["off_rtg_pt", "def_rtg_pt", "opp_def_rtg_pt", "matchup_expected_ortg_pt"]:
        if not pd.to_numeric(ratings[column], errors="raise").between(80, 150).all():
            raise ValueError(f"{column} is outside the accepted 80..150 range")
    for column in ["pace_pt", "opp_pace_pt"]:
        if not pd.to_numeric(ratings[column], errors="raise").between(70, 130).all():
            raise ValueError(f"{column} is outside the accepted 70..130 range")
    if not pd.to_numeric(ratings["opp_def_rank_pt"], errors="raise").between(1, 30).all():
        raise ValueError("opp_def_rank_pt must be between 1 and 30")
    if not pd.to_numeric(ratings["opp_def_tier_pt"], errors="raise").between(1, 5).all():
        raise ValueError("opp_def_tier_pt must be between 1 and 5")
    if (pd.to_numeric(ratings[["n_prior_games", "opp_n_prior_games"]].stack(), errors="raise") < 0).any():
        raise ValueError("Prior-game counts cannot be negative")


def build_training_table(prop: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Perform a fail-closed one-to-one team-game join and return QA metadata."""
    _require_columns(prop, PROP_REQUIRED, "Prop training input")
    _require_columns(ratings, RATING_REQUIRED, "Point-in-time rating input")
    prop, ratings = _normalise(prop, ratings)
    _validate_ratings(ratings)

    right = ratings[RATING_REQUIRED].rename(
        columns={
            "game_date": "defense_rating_game_date",
            "is_home": "defense_rating_is_home",
        }
    )
    merged = prop.merge(
        right,
        on=RATING_KEY,
        how="left",
        validate="many_to_one",
        indicator="_defense_rating_merge",
    )
    if len(merged) != len(prop):
        raise RuntimeError("Defense-rating merge changed the prop table row count")
    unmatched = merged["_defense_rating_merge"].ne("both")
    if unmatched.any():
        examples = merged.loc[unmatched, PROP_KEY + ["game_date"]].head(5).to_dict("records")
        raise ValueError(f"Defense-rating join left {int(unmatched.sum())} prop rows unmatched: {examples}")
    if not merged["defense_rating_game_date"].eq(merged["game_date"]).all():
        raise ValueError("Matched defense-rating rows disagree on game_date")
    chronology = merged["rating_asof_date"] >= merged["game_date"]
    if chronology.any():
        raise ValueError(f"Joined table has {int(chronology.sum())} rating chronology violations")

    if "opp" in merged.columns:
        known = merged["opp"].notna()
        disagreement = known & merged["opp"].ne(merged["opponent"])
        if disagreement.any():
            raise ValueError(f"Opponent identity disagrees on {int(disagreement.sum())} prop rows")
        # Repair the known 2025-26 source defect from the audited GAME_ID pairing.
        merged["opp"] = merged["opp"].fillna(merged["opponent"])
    else:
        merged["opp"] = merged["opponent"]
    if "is_home" in merged.columns:
        supplied_home = pd.to_numeric(merged["is_home"], errors="coerce")
        known_home = supplied_home.notna()
        disagreement = known_home & supplied_home.ne(merged["defense_rating_is_home"])
        if disagreement.any():
            raise ValueError(f"Home/away identity disagrees on {int(disagreement.sum())} prop rows")
        merged["is_home"] = supplied_home.fillna(merged["defense_rating_is_home"]).astype(int)
    else:
        merged["is_home"] = merged["defense_rating_is_home"].astype(int)

    if merged[DEFENSE_RATING_PROP_FEATURES].isna().any().any():
        raise RuntimeError("Joined table contains null defense-rating candidate features")
    output = merged.drop(columns=["_defense_rating_merge", "defense_rating_game_date", "defense_rating_is_home"])
    summary = {
        "status": "ready_for_isolated_defense_rating_retraining",
        "join_key": PROP_KEY,
        "source_rows": int(len(prop)),
        "rating_rows": int(len(ratings)),
        "matched_rows": int(len(output)),
        "match_rate": 1.0,
        "date_min": output["game_date"].min().date().isoformat(),
        "date_max": output["game_date"].max().date().isoformat(),
        "rating_chronology_violations": 0,
        "rating_duplicate_keys": 0,
        "feature_columns": DEFENSE_RATING_PROP_FEATURES,
        "feature_null_cells": int(output[DEFENSE_RATING_PROP_FEATURES].isna().sum().sum()),
        "repaired_opponent_rows": int(prop.get("opp", pd.Series(pd.NA, index=prop.index)).isna().sum()),
        "repaired_is_home_rows": int(pd.to_numeric(prop.get("is_home", pd.Series(pd.NA, index=prop.index)), errors="coerce").isna().sum()),
        "rating_source_counts": output["opp_rating_source"].value_counts().to_dict(),
    }
    return output.sort_values(["game_date", "GAME_ID", "PLAYER_ID"] if "PLAYER_ID" in output else ["game_date", "GAME_ID"]).reset_index(drop=True), summary


def build_from_files(
    prop_input: Path = DEFAULT_PROP_INPUT,
    rating_input: Path = DEFAULT_RATING_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict:
    prop = pd.read_csv(prop_input, low_memory=False)
    ratings = pd.read_csv(rating_input, low_memory=False)
    output, summary = build_training_table(prop, ratings)
    summary.update(
        {
            "prop_input": str(prop_input),
            "prop_input_sha256": _sha256(prop_input),
            "rating_input": str(rating_input),
            "rating_input_sha256": _sha256(rating_input),
            "output": str(output_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(output_path)
    summary["output_sha256"] = _sha256(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prop-input", type=Path, default=DEFAULT_PROP_INPUT)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATING_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = build_from_files(args.prop_input, args.ratings, args.output, args.summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
