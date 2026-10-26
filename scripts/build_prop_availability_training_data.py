"""Build the leakage-safe prop-training table with availability/minutes OOF features.

The prop target table contains appearance rows.  The availability/minutes table
contains reconstructed pregame candidates, including non-appearances.  This
adapter performs a one-to-one key merge, verifies that every retained prediction
was trained strictly before its target game, and emits only matched appearance
rows suitable for conditional player-prop model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROP_INPUT = ROOT / "data" / "model_data_advanced.csv"
DEFAULT_OOF_INPUT = ROOT / "data" / "availability_minutes_oof_predictions.csv"
DEFAULT_OUTPUT = ROOT / "data" / "model_data_availability_advanced.csv"
DEFAULT_SUMMARY = ROOT / "output" / "availability_prop_training_join_summary.json"

PROP_KEY = ["GAME_ID", "PLAYER_ID", "team"]
OOF_KEY = ["game_id", "player_id", "team"]
AVAILABILITY_MINUTES_PROP_FEATURES = [
    "availability_probability",
    "conditional_minutes_prediction",
    "expected_minutes_prediction",
]
OOF_REQUIRED = [
    *OOF_KEY,
    "game_date",
    "available",
    "minutes",
    *AVAILABILITY_MINUTES_PROP_FEATURES,
    "prediction_split",
    "model_train_end",
]
PROP_REQUIRED = [*PROP_KEY, "game_date", "mp"]


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


def _normalise_keys(prop: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prop = prop.copy()
    oof = oof.copy()
    for column in ["GAME_ID", "PLAYER_ID"]:
        prop[column] = pd.to_numeric(prop[column], errors="raise").astype("int64")
    for column in ["game_id", "player_id"]:
        oof[column] = pd.to_numeric(oof[column], errors="raise").astype("int64")
    prop["team"] = prop["team"].astype(str).str.strip().str.upper()
    oof["team"] = oof["team"].astype(str).str.strip().str.upper()
    prop["game_date"] = pd.to_datetime(prop["game_date"], errors="raise").dt.normalize()
    oof["game_date"] = pd.to_datetime(oof["game_date"], errors="raise").dt.normalize()
    oof["model_train_end"] = pd.to_datetime(oof["model_train_end"], errors="raise").dt.normalize()
    return prop, oof


def _validate_oof(oof: pd.DataFrame) -> None:
    if oof.duplicated(OOF_KEY).any():
        count = int(oof.duplicated(OOF_KEY, keep=False).sum())
        raise ValueError(f"Availability OOF has {count} rows on duplicate candidate keys")
    if oof[OOF_REQUIRED].isna().any().any():
        nulls = oof[OOF_REQUIRED].isna().sum()
        raise ValueError(f"Availability OOF contains required nulls: {nulls[nulls.gt(0)].to_dict()}")
    if (oof["model_train_end"] >= oof["game_date"]).any():
        count = int((oof["model_train_end"] >= oof["game_date"]).sum())
        raise ValueError(f"Availability OOF has {count} chronology violations")

    probability = pd.to_numeric(oof["availability_probability"], errors="raise")
    conditional = pd.to_numeric(oof["conditional_minutes_prediction"], errors="raise")
    expected = pd.to_numeric(oof["expected_minutes_prediction"], errors="raise")
    if not probability.between(0, 1).all():
        raise ValueError("availability_probability must be between zero and one")
    if not conditional.between(0, 48).all() or not expected.between(0, 48).all():
        raise ValueError("Availability/minutes predictions must be between zero and 48")
    if not np.allclose(expected, probability * conditional, rtol=1e-9, atol=1e-9):
        raise ValueError("expected_minutes_prediction is not probability times conditional minutes")


def build_training_table(
    prop: pd.DataFrame,
    oof: pd.DataFrame,
    min_window_coverage: float = 0.90,
) -> tuple[pd.DataFrame, dict]:
    """Join the two inputs and return the conditional prop-training table and QA summary."""
    _require_columns(prop, PROP_REQUIRED, "Prop training input")
    _require_columns(oof, OOF_REQUIRED, "Availability OOF input")
    prop, oof = _normalise_keys(prop, oof)
    _validate_oof(oof)

    if prop.duplicated(PROP_KEY).any():
        count = int(prop.duplicated(PROP_KEY, keep=False).sum())
        raise ValueError(f"Prop training input has {count} rows on duplicate player-game keys")

    oof_min = oof["game_date"].min()
    oof_max = oof["game_date"].max()
    prop_max = prop["game_date"].max()
    if prop_max > oof_max:
        raise ValueError(
            f"Prop data extends through {prop_max.date()} but OOF ends {oof_max.date()}; "
            "regenerate availability OOF before retraining"
        )

    right = oof[OOF_REQUIRED].rename(
        columns={
            "game_date": "availability_game_date",
            "available": "availability_target",
            "minutes": "availability_actual_minutes",
            "prediction_split": "availability_prediction_split",
            "model_train_end": "availability_model_train_end",
        }
    )
    merged = prop.merge(
        right,
        left_on=PROP_KEY,
        right_on=OOF_KEY,
        how="left",
        validate="one_to_one",
        indicator="_availability_merge",
    )
    if len(merged) != len(prop):
        raise RuntimeError("Availability merge changed the prop table row count")

    in_window = prop["game_date"].between(oof_min, oof_max)
    matched = merged["_availability_merge"].eq("both")
    window_rows = int(in_window.sum())
    matched_window_rows = int((in_window & matched).sum())
    coverage = matched_window_rows / window_rows if window_rows else 0.0
    if coverage < min_window_coverage:
        raise ValueError(
            f"Availability OOF coverage {coverage:.2%} is below required "
            f"{min_window_coverage:.2%} ({matched_window_rows}/{window_rows})"
        )

    matched_rows = merged.loc[matched].copy()
    if not matched_rows["availability_game_date"].eq(matched_rows["game_date"]).all():
        raise ValueError("Matched availability rows disagree on game_date")
    if not (matched_rows["availability_model_train_end"] < matched_rows["game_date"]).all():
        raise ValueError("Matched availability rows contain a chronology violation")

    prop_minutes = pd.to_numeric(matched_rows["mp"], errors="raise")
    oof_minutes = pd.to_numeric(matched_rows["availability_actual_minutes"], errors="raise")
    if not np.allclose(prop_minutes, oof_minutes, rtol=0, atol=1e-6):
        raise ValueError("Matched OOF minutes do not reconcile to prop-training minutes")

    appearance = matched_rows["availability_target"].eq(1) & prop_minutes.gt(0)
    disagreement = matched_rows["availability_target"].eq(1).ne(prop_minutes.gt(0))
    if disagreement.any():
        raise ValueError(f"Availability target disagrees with minutes on {int(disagreement.sum())} rows")

    output = matched_rows.loc[appearance].copy()
    output = output.drop(
        columns=[
            "game_id",
            "player_id",
            "availability_game_date",
            "availability_target",
            "availability_actual_minutes",
            "_availability_merge",
        ]
    )
    if output.duplicated(PROP_KEY).any():
        raise RuntimeError("Output contains duplicate player-game keys")
    if output[AVAILABILITY_MINUTES_PROP_FEATURES].isna().any().any():
        raise RuntimeError("Output contains null availability/minutes features")

    if "mp_l10" in prop.columns:
        rotation_mask = in_window & pd.to_numeric(prop["mp_l10"], errors="coerce").ge(15)
    else:
        rotation_mask = pd.Series(False, index=prop.index)
    rotation_rows = int(rotation_mask.sum())
    rotation_matches = int((rotation_mask & matched).sum())
    summary = {
        "status": "ready_for_prop_retraining",
        "join_key": PROP_KEY,
        "source_rows": int(len(prop)),
        "source_key_duplicates": 0,
        "oof_rows": int(len(oof)),
        "oof_key_duplicates": 0,
        "oof_date_min": oof_min.date().isoformat(),
        "oof_date_max": oof_max.date().isoformat(),
        "oof_chronology_violations": 0,
        "source_rows_in_oof_window": window_rows,
        "matched_rows_in_oof_window": matched_window_rows,
        "window_match_rate": coverage,
        "rotation_proxy_rows": rotation_rows,
        "rotation_proxy_matched_rows": rotation_matches,
        "rotation_proxy_match_rate": rotation_matches / rotation_rows if rotation_rows else None,
        "excluded_before_oof_window": int((prop["game_date"] < oof_min).sum()),
        "excluded_unmatched_in_window": int((in_window & ~matched).sum()),
        "excluded_nonappearances": int((~appearance).sum()),
        "output_rows": int(len(output)),
        "output_date_min": output["game_date"].min().date().isoformat(),
        "output_date_max": output["game_date"].max().date().isoformat(),
        "feature_columns": AVAILABILITY_MINUTES_PROP_FEATURES,
        "feature_null_cells": int(output[AVAILABILITY_MINUTES_PROP_FEATURES].isna().sum().sum()),
        "output_key_duplicates": int(output.duplicated(PROP_KEY).sum()),
    }
    return output.sort_values(["game_date", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True), summary


def build_from_files(
    prop_input: Path = DEFAULT_PROP_INPUT,
    oof_input: Path = DEFAULT_OOF_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    min_window_coverage: float = 0.90,
) -> dict:
    prop = pd.read_csv(prop_input, low_memory=False)
    oof = pd.read_csv(oof_input, low_memory=False)
    output, summary = build_training_table(prop, oof, min_window_coverage=min_window_coverage)
    summary.update(
        {
            "prop_input": str(prop_input),
            "prop_input_sha256": _sha256(prop_input),
            "availability_oof_input": str(oof_input),
            "availability_oof_sha256": _sha256(oof_input),
            "output": str(output_path),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(output_path)
    summary["output_sha256"] = _sha256(output_path)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prop-input", type=Path, default=DEFAULT_PROP_INPUT)
    parser.add_argument("--availability-oof", type=Path, default=DEFAULT_OOF_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-window-coverage", type=float, default=0.90)
    args = parser.parse_args()

    summary = build_from_files(
        prop_input=args.prop_input,
        oof_input=args.availability_oof,
        output_path=args.output,
        summary_path=args.summary,
        min_window_coverage=args.min_window_coverage,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
