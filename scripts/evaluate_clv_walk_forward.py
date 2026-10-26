"""Leakage-safe walk-forward evaluation for the frozen CLV feature panel.

This script evaluates a source-specific PrizePicks UNDER selection overlay.  It
does not retrain the underlying stat projection models.  Archived prediction
files supply the model information that was genuinely available on each slate;
the frozen CLV table supplies timestamped market features and outcomes.

The final dates are held out before feature selection. Candidate feature
groups are promoted only when they improve the expanding-window validation
card in at least three folds, improve mean top-five hit rate by at least one
percentage point, and do not reduce mean row-level AUC. The gate additionally
requires at least 12 validation dates and 60 validation-card observations so a
thin fold sequence cannot promote a feature. Passing that research
gate creates a shadow model; production enablement also requires at least 100
holdout selections, 15 holdout dates, and 20 standard-line selections.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from scripts.build_clv_features import parse_local_snapshot
    from scripts.execute_clv_snapshot_manifest import (
        ROOT,
        atomic_write_csv,
        atomic_write_json,
        normalize_name,
    )
except ModuleNotFoundError:
    from build_clv_features import parse_local_snapshot
    from execute_clv_snapshot_manifest import (
        ROOT,
        atomic_write_csv,
        atomic_write_json,
        normalize_name,
    )


PROP_TO_STAT = {
    "player_points": "pts",
    "player_rebounds": "trb",
    "player_assists": "ast",
    "player_points_rebounds_assists": "pra",
    "player_points_rebounds": "pr",
    "player_points_assists": "pa",
    "player_steals": "stl",
    "player_blocks": "blk",
    "player_turnovers": "tov",
}

# Frozen representation of the production policy at evaluation time.  STL and
# BLK are excluded by CONFIG; composites and TOV are disabled by model policy.
ELIGIBLE_PROPS = {
    "player_points",
    "player_rebounds",
    "player_assists",
}
ELIGIBLE_PROJECTION_TYPES = {"standard"}
MIN_MODEL_UNDER_EDGE = 0.05
CARD_SIZE = 5
HOLDOUT_DATE_COUNT = 7
MIN_TRAIN_DATES = 10
N_VALIDATION_FOLDS = 4
RANDOM_REPETITIONS = 2_000
SEED = 20260812
MIN_PRODUCTION_HOLDOUT_PICKS = 100
MIN_PRODUCTION_HOLDOUT_DATES = 15
MIN_PRODUCTION_STANDARD_PICKS = 20

BASE_NUMERIC_FEATURES = [
    "model_under_edge",
    "model_prediction",
    "archived_l10",
    "archived_clf_prob",
    "archived_market_edge_pred",
    "archived_meta_prob",
]
BASE_CATEGORICAL_FEATURES = ["prop"]


@dataclass(frozen=True)
class SnapshotSpec:
    name: str
    line_column: str
    label_column: str
    push_column: str
    candidate_groups: dict[str, list[str]]
    forbidden_groups: dict[str, str]


SNAPSHOT_SPECS = {
    "t_minus_60": SnapshotSpec(
        name="t_minus_60",
        line_column="pp_line_t_minus_60",
        label_column="under_win_pp_t_minus_60",
        push_column="push_pp_t_minus_60",
        candidate_groups={
            "pp_minus_consensus_gap": ["pp_minus_consensus_t_minus_60"],
            "sportsbook_consensus_line": ["consensus_line_t_minus_60"],
            "no_vig_probability": ["consensus_p_over_t_minus_60"],
            "book_dispersion": [
                "book_count_t_minus_60",
                "line_range_t_minus_60",
                "line_std_t_minus_60",
                "p_over_std_t_minus_60",
            ],
            "market_availability": ["available_decision", "available_t_minus_60"],
        },
        forbidden_groups={
            "t60_to_t5_movement": "not known until T-5",
            "market_withdrawal": "not known until T-5",
        },
    ),
    "t_minus_5": SnapshotSpec(
        name="t_minus_5",
        line_column="pp_line_t_minus_5",
        label_column="under_win_pp_t_minus_5",
        push_column="push_pp_t_minus_5",
        candidate_groups={
            "pp_minus_consensus_gap": ["pp_minus_consensus_t_minus_5"],
            "sportsbook_consensus_line": ["consensus_line_t_minus_5"],
            "no_vig_probability": ["consensus_p_over_t_minus_5"],
            "book_dispersion": [
                "book_count_t_minus_5",
                "line_range_t_minus_5",
                "line_std_t_minus_5",
                "p_over_std_t_minus_5",
            ],
            "t60_to_t5_movement": [
                "line_move_60_to_5",
                "p_over_move_60_to_5",
                "dispersion_move_60_to_5",
            ],
            "market_withdrawal_availability": [
                "available_t_minus_60",
                "available_t_minus_5",
                "market_withdrawn_by_t5",
            ],
        },
        forbidden_groups={},
    ),
}


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def _canonical_prediction_path(game_date: str) -> Path:
    return ROOT / "output" / "archive" / game_date / f"predictions_{game_date}.csv"


def load_archived_predictions(game_dates: Iterable[str]) -> pd.DataFrame:
    """Load one canonical archived prediction per player/date/prop."""
    frames: list[pd.DataFrame] = []
    for game_date in sorted(set(game_dates)):
        path = _canonical_prediction_path(game_date)
        if not path.exists():
            continue
        prediction = pd.read_csv(path, low_memory=False).copy()
        if prediction.empty or "player" not in prediction.columns:
            continue
        prediction["game_date"] = game_date
        prediction["player_norm"] = prediction["player"].map(normalize_name)
        for prop, stat in PROP_TO_STAT.items():
            pred_col = f"{stat}_pred"
            if pred_col not in prediction.columns:
                continue
            mapped = pd.DataFrame(
                {
                    "game_date": prediction["game_date"],
                    "player_norm": prediction["player_norm"],
                    "prop": prop,
                    "model_prediction": pd.to_numeric(prediction[pred_col], errors="coerce"),
                    "archived_l10": pd.to_numeric(prediction.get(f"{stat}_l10"), errors="coerce"),
                    "archived_anchor": pd.to_numeric(prediction.get(f"{stat}_anchor"), errors="coerce"),
                    "archived_clf_prob": pd.to_numeric(prediction.get(f"{stat}_clf_prob"), errors="coerce"),
                    "archived_market_edge_pred": pd.to_numeric(
                        prediction.get(f"{stat}_market_edge_pred"), errors="coerce"
                    ),
                    "archived_meta_prob": pd.to_numeric(prediction.get(f"{stat}_meta_prob"), errors="coerce"),
                    "prediction_source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                }
            )
            frames.append(mapped)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["game_date", "player_norm", "prop"], keep="last")


def _prizepicks_metadata_for_features(features: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """Recover the market type for the exact archived PP reference in the frozen table."""
    manifest = _read_csv(
        ROOT / "data" / "clv_snapshot_manifest_2026_live.csv",
        dtype={"nba_game_id": str},
    )
    manifest = manifest[manifest["snapshot_kind"].eq(snapshot)][
        ["nba_game_id", "requested_snapshot_ts"]
    ].drop_duplicates("nba_game_id")
    left = features[[
        "nba_game_id", "game_date", "player_norm", "prop",
        f"pp_line_{snapshot}", f"pp_snapshot_age_minutes_{snapshot}",
    ]].copy()
    left = left.merge(manifest, on="nba_game_id", how="left", validate="many_to_one")
    left["requested_snapshot_ts"] = pd.to_datetime(left["requested_snapshot_ts"], utc=True, errors="coerce")
    age = pd.to_numeric(left[f"pp_snapshot_age_minutes_{snapshot}"], errors="coerce")
    left["matched_pp_snapshot_ts"] = (
        left["requested_snapshot_ts"] - pd.to_timedelta(age, unit="m")
    ).dt.round("s")
    left["matched_pp_line"] = pd.to_numeric(left[f"pp_line_{snapshot}"], errors="coerce")

    master = _read_csv(ROOT / "data" / "historical_lines.csv")
    source = master["source"].fillna("").astype(str).str.lower()
    pp = master[source.str.contains("prizepicks")].copy()
    pp["game_date"] = pd.to_datetime(pp["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pp["player_norm"] = pp["player"].map(normalize_name)
    pp["matched_pp_snapshot_ts"] = pp["snapshot_ts"].map(parse_local_snapshot).dt.round("s")
    pp["matched_pp_line"] = pd.to_numeric(pp["line"], errors="coerce")
    pp["projection_type"] = pp["projection_type"].fillna("standard").astype(str).str.lower()
    pp["is_promo"] = pp["is_promo"].fillna(False).astype(bool)
    pp = pp.sort_values("matched_pp_snapshot_ts").drop_duplicates(
        ["game_date", "player_norm", "prop", "matched_pp_line", "matched_pp_snapshot_ts"],
        keep="last",
    )
    keys = ["game_date", "player_norm", "prop", "matched_pp_line", "matched_pp_snapshot_ts"]
    metadata = left.merge(
        pp[keys + ["projection_type", "is_promo"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    return metadata[["nba_game_id", "player_norm", "prop", "projection_type", "is_promo"]].rename(
        columns={
            "projection_type": f"pp_projection_type_{snapshot}",
            "is_promo": f"pp_is_promo_{snapshot}",
        }
    )


def build_eligible_pool(features: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """Create the frozen placeable-line, current-policy UNDER candidate pool."""
    spec = SNAPSHOT_SPECS[snapshot]
    working = features.copy()
    working["game_date"] = pd.to_datetime(working["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working = working.merge(
        load_archived_predictions(working["game_date"].dropna().unique()),
        on=["game_date", "player_norm", "prop"],
        how="left",
        validate="many_to_one",
    )
    working = working.merge(
        _prizepicks_metadata_for_features(working, snapshot),
        on=["nba_game_id", "player_norm", "prop"],
        how="left",
        validate="one_to_one",
    )
    working["pp_line"] = pd.to_numeric(working[spec.line_column], errors="coerce")
    working["under_win"] = pd.to_numeric(working[spec.label_column], errors="coerce")
    working["push"] = pd.to_numeric(working[spec.push_column], errors="coerce")
    working["model_under_edge"] = working["pp_line"] - working["model_prediction"]
    market_type = working[f"pp_projection_type_{snapshot}"].fillna("").astype(str).str.lower()
    promo = working[f"pp_is_promo_{snapshot}"].fillna(False).astype(bool)
    working["pp_projection_type"] = market_type
    eligible = (
        working["prop"].isin(ELIGIBLE_PROPS)
        & market_type.isin(ELIGIBLE_PROJECTION_TYPES)
        & ~promo
        & working["model_prediction"].notna()
        & working["pp_line"].notna()
        & working["under_win"].notna()
        & working["push"].eq(0)
        & working["model_under_edge"].ge(MIN_MODEL_UNDER_EDGE)
    )
    pool = working[eligible].copy()
    pool["snapshot_kind"] = snapshot
    pool["eligible_policy"] = "placeable_nonpromo_current_active_props_model_under"
    return pool.sort_values(["game_date", "nba_game_id", "player_norm", "prop"]).reset_index(drop=True)


def chronological_split_dates(dates: Iterable[str]) -> tuple[list[str], list[str], list[tuple[list[str], list[str]]]]:
    """Return development dates, untouched holdout dates, and expanding folds."""
    ordered = sorted(set(str(date) for date in dates))
    if len(ordered) < 7:
        raise ValueError(f"Only {len(ordered)} eligible dates; at least seven are required")
    holdout_n = HOLDOUT_DATE_COUNT if len(ordered) >= 24 else max(2, math.ceil(len(ordered) * 0.25))
    development = ordered[:-holdout_n]
    holdout = ordered[-holdout_n:]
    min_train_dates = MIN_TRAIN_DATES if len(development) >= MIN_TRAIN_DATES + 4 else max(3, len(development) // 2)
    validation_dates = development[min_train_dates:]
    split_count = min(N_VALIDATION_FOLDS, len(validation_dates))
    chunks = [list(chunk) for chunk in np.array_split(validation_dates, split_count) if len(chunk)]
    folds: list[tuple[list[str], list[str]]] = []
    for chunk in chunks:
        first = development.index(chunk[0])
        train_dates = development[:first]
        if len(train_dates) < min_train_dates:
            continue
        folds.append((train_dates, chunk))
    if len(folds) < 3:
        raise ValueError("Fewer than three valid chronological folds")
    return development, holdout, folds


def _make_model(numeric_features: list[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transform = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, BASE_CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("features", transform),
            ("model", LogisticRegression(C=0.5, max_iter=2_000, random_state=SEED)),
        ]
    )


def _top_card(frame: pd.DataFrame, score_column: str, card_size: int = CARD_SIZE) -> pd.DataFrame:
    return (
        frame.sort_values(["game_date", score_column, "player_norm", "prop"], ascending=[True, False, True, True])
        .groupby("game_date", sort=True)
        .head(card_size)
        .copy()
    )


def _score_predictions(frame: pd.DataFrame, score_column: str) -> dict[str, float | int | None]:
    card = _top_card(frame, score_column)
    n = int(len(card))
    wins = int(card["under_win"].sum())
    return {
        "n": n,
        "wins": wins,
        "hit_rate": float(wins / n) if n else None,
        "dates": int(card["game_date"].nunique()),
    }


def _fit_and_score(train: pd.DataFrame, test: pd.DataFrame, numeric_features: list[str]) -> tuple[Pipeline, dict]:
    model = _make_model(numeric_features)
    X_train = train[numeric_features + BASE_CATEGORICAL_FEATURES]
    y_train = train["under_win"].astype(int)
    model.fit(X_train, y_train)
    scored = test.copy()
    scored["overlay_score"] = model.predict_proba(
        scored[numeric_features + BASE_CATEGORICAL_FEATURES]
    )[:, 1]
    card_metrics = _score_predictions(scored, "overlay_score")
    y_test = scored["under_win"].astype(int)
    auc = float(roc_auc_score(y_test, scored["overlay_score"])) if y_test.nunique() > 1 else np.nan
    card_metrics["auc"] = auc
    return model, {"metrics": card_metrics, "scored": scored}


def _validation_score(
    pool: pd.DataFrame,
    folds: list[tuple[list[str], list[str]]],
    numeric_features: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scored_frames = []
    for fold_index, (train_dates, validation_dates) in enumerate(folds, start=1):
        train = pool[pool["game_date"].isin(train_dates)].copy()
        validation = pool[pool["game_date"].isin(validation_dates)].copy()
        _, result = _fit_and_score(train, validation, numeric_features)
        scored = result["scored"]
        scored["fold"] = fold_index
        scored["model_name"] = model_name
        scored_frames.append(scored)
        metrics = result["metrics"]
        rows.append(
            {
                "fold": fold_index,
                "model_name": model_name,
                "train_start": min(train_dates),
                "train_end": max(train_dates),
                "validation_start": min(validation_dates),
                "validation_end": max(validation_dates),
                "train_dates": len(train_dates),
                "validation_dates": len(validation_dates),
                "train_rows": len(train),
                "validation_rows": len(validation),
                **metrics,
            }
        )
    return pd.DataFrame(rows), pd.concat(scored_frames, ignore_index=True)


def select_feature_groups(
    pool: pd.DataFrame,
    spec: SnapshotSpec,
    folds: list[tuple[list[str], list[str]]],
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Greedy fixed-gate selection using validation folds only."""
    selected_groups: list[str] = []
    selected_features = list(BASE_NUMERIC_FEATURES)
    fold_outputs: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    base_fold, _ = _validation_score(pool, folds, selected_features, "base")
    fold_outputs.append(base_fold.assign(snapshot_kind=spec.name, feature_group="base"))
    current_fold = base_fold
    remaining = list(spec.candidate_groups)
    step = 0
    while remaining:
        step += 1
        candidates = []
        for group in remaining:
            trial_features = selected_features + spec.candidate_groups[group]
            trial_fold, _ = _validation_score(pool, folds, trial_features, f"trial_{group}")
            lift = trial_fold["hit_rate"].to_numpy() - current_fold["hit_rate"].to_numpy()
            auc_lift = trial_fold["auc"].to_numpy() - current_fold["auc"].to_numpy()
            record = {
                "snapshot_kind": spec.name,
                "step": step,
                "feature_group": group,
                "features": "|".join(spec.candidate_groups[group]),
                "mean_top5_lift_pp": float(np.nanmean(lift) * 100.0),
                "positive_folds": int(np.sum(lift > 0)),
                "nonnegative_folds": int(np.sum(lift >= 0)),
                "mean_auc_lift": float(np.nanmean(auc_lift)),
                "validation_top5_n": int(trial_fold["n"].sum()),
            }
            total_validation_dates = sum(len(validation_dates) for _, validation_dates in folds)
            record["validation_dates"] = total_validation_dates
            record["survival_gate"] = bool(
                record["mean_top5_lift_pp"] >= 1.0
                and record["positive_folds"] >= min(3, len(folds))
                and record["mean_auc_lift"] >= 0.0
                and record["validation_top5_n"] >= 60
                and total_validation_dates >= 12
            )
            candidates.append((record, trial_fold))
        candidates.sort(
            key=lambda item: (
                item[0]["survival_gate"],
                item[0]["mean_top5_lift_pp"],
                item[0]["mean_auc_lift"],
            ),
            reverse=True,
        )
        selection_rows.extend(record for record, _ in candidates)
        winner, winner_fold = candidates[0]
        if not winner["survival_gate"]:
            break
        selected_groups.append(str(winner["feature_group"]))
        selected_features.extend(spec.candidate_groups[str(winner["feature_group"])])
        current_fold = winner_fold
        fold_outputs.append(
            winner_fold.assign(snapshot_kind=spec.name, feature_group=winner["feature_group"])
        )
        remaining.remove(str(winner["feature_group"]))
    for group, reason in spec.forbidden_groups.items():
        selection_rows.append(
            {
                "snapshot_kind": spec.name,
                "step": 0,
                "feature_group": group,
                "features": "",
                "mean_top5_lift_pp": np.nan,
                "positive_folds": 0,
                "nonnegative_folds": 0,
                "mean_auc_lift": np.nan,
                "validation_top5_n": 0,
                "validation_dates": 0,
                "survival_gate": False,
                "exclusion_reason": reason,
            }
        )
    return selected_groups, pd.DataFrame(selection_rows), pd.concat(fold_outputs, ignore_index=True)


def random_card_distribution(
    frame: pd.DataFrame,
    repetitions: int = RANDOM_REPETITIONS,
    card_size: int = CARD_SIZE,
    seed: int = SEED,
) -> np.ndarray:
    """Return deterministic same-date random UNDER card hit rates."""
    groups = [group["under_win"].to_numpy(dtype=float) for _, group in frame.groupby("game_date", sort=True)]
    rng = np.random.default_rng(seed)
    rates = np.empty(repetitions, dtype=float)
    for iteration in range(repetitions):
        wins = 0.0
        total = 0
        for values in groups:
            take = min(card_size, len(values))
            if take == 0:
                continue
            chosen = rng.choice(len(values), size=take, replace=False)
            wins += float(values[chosen].sum())
            total += take
        rates[iteration] = wins / total if total else np.nan
    return rates


def matched_random_card_distribution(
    pool: pd.DataFrame,
    selected_card: pd.DataFrame,
    repetitions: int = RANDOM_REPETITIONS,
    seed: int = SEED,
) -> np.ndarray:
    """Randomize within the selected card's date/prop/line-type composition."""
    strata = ["game_date", "prop", "pp_projection_type"]
    selected_counts = selected_card.groupby(strata, dropna=False).size().to_dict()
    candidates = {
        key: group["under_win"].to_numpy(dtype=float)
        for key, group in pool.groupby(strata, dropna=False, sort=True)
    }
    rng = np.random.default_rng(seed)
    rates = np.empty(repetitions, dtype=float)
    for iteration in range(repetitions):
        wins = 0.0
        total = 0
        for key, take in selected_counts.items():
            values = candidates[key]
            replace = take > len(values)
            chosen = rng.choice(len(values), size=take, replace=replace)
            wins += float(values[chosen].sum())
            total += int(take)
        rates[iteration] = wins / total if total else np.nan
    return rates


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return centre - half, centre + half


def evaluate_holdout(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    selected_features: list[str],
    snapshot: str,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, dict]:
    model, result = _fit_and_score(development, holdout, selected_features)
    scored = result["scored"].copy()
    overlay = _score_predictions(scored, "overlay_score")
    current = _score_predictions(scored, "model_under_edge")
    overlay_card = _top_card(scored, "overlay_score")
    current_card = _top_card(scored, "model_under_edge")
    overlay_random_rates = matched_random_card_distribution(
        scored, overlay_card, seed=SEED + 11
    )
    current_random_rates = matched_random_card_distribution(
        scored, current_card, seed=SEED + 29
    )
    always_n = int(len(scored))
    always_wins = int(scored["under_win"].sum())
    comparisons = []
    for name, metrics, random_rates in [
        ("shadow_overlay", overlay, overlay_random_rates),
        ("current_model_edge", current, current_random_rates),
    ]:
        low, high = _wilson_interval(int(metrics["wins"]), int(metrics["n"]))
        comparisons.append(
            {
                "snapshot_kind": snapshot,
                "method": name,
                **metrics,
                "hit_rate_ci_low": low,
                "hit_rate_ci_high": high,
                "lift_vs_random_mean_pp": (
                    float(metrics["hit_rate"] - np.nanmean(random_rates)) * 100.0
                    if metrics["hit_rate"] is not None else None
                ),
            }
        )
    low, high = _wilson_interval(always_wins, always_n)
    comparisons.append(
        {
            "snapshot_kind": snapshot,
            "method": "always_under_pool",
            "n": always_n,
            "wins": always_wins,
            "hit_rate": always_wins / always_n if always_n else None,
            "dates": int(scored["game_date"].nunique()),
            "auc": None,
            "hit_rate_ci_low": low,
            "hit_rate_ci_high": high,
            "lift_vs_random_mean_pp": None,
        }
    )
    comparisons.append(
        {
            "snapshot_kind": snapshot,
            "method": "matched_random_for_overlay",
            "n": int(overlay["n"]),
            "wins": None,
            "hit_rate": float(np.nanmean(overlay_random_rates)),
            "dates": int(scored["game_date"].nunique()),
            "auc": None,
            "hit_rate_ci_low": float(np.nanquantile(overlay_random_rates, 0.025)),
            "hit_rate_ci_high": float(np.nanquantile(overlay_random_rates, 0.975)),
            "lift_vs_random_mean_pp": 0.0,
        }
    )
    comparisons.append(
        {
            "snapshot_kind": snapshot,
            "method": "matched_random_for_current",
            "n": int(current["n"]),
            "wins": None,
            "hit_rate": float(np.nanmean(current_random_rates)),
            "dates": int(scored["game_date"].nunique()),
            "auc": None,
            "hit_rate_ci_low": float(np.nanquantile(current_random_rates, 0.025)),
            "hit_rate_ci_high": float(np.nanquantile(current_random_rates, 0.975)),
            "lift_vs_random_mean_pp": 0.0,
        }
    )
    for projection_type in sorted(ELIGIBLE_PROJECTION_TYPES):
        sensitivity = scored[scored["pp_projection_type"].eq(projection_type)].copy()
        if sensitivity.empty:
            continue
        for name, score_column in [
            (f"shadow_overlay_{projection_type}", "overlay_score"),
            (f"current_model_edge_{projection_type}", "model_under_edge"),
        ]:
            metrics = _score_predictions(sensitivity, score_column)
            low, high = _wilson_interval(int(metrics["wins"]), int(metrics["n"]))
            comparisons.append(
                {
                    "snapshot_kind": snapshot,
                    "method": name,
                    **metrics,
                    "hit_rate_ci_low": low,
                    "hit_rate_ci_high": high,
                    "lift_vs_random_mean_pp": None,
                }
            )
    daily_frames = []
    for method, score_col in [("shadow_overlay", "overlay_score"), ("current_model_edge", "model_under_edge")]:
        card = _top_card(scored, score_col)
        daily = card.groupby("game_date").agg(n=("under_win", "size"), wins=("under_win", "sum")).reset_index()
        daily["hit_rate"] = daily["wins"] / daily["n"]
        daily["method"] = method
        daily["snapshot_kind"] = snapshot
        daily_frames.append(daily)
    summary = {
        "holdout_start": str(holdout["game_date"].min()),
        "holdout_end": str(holdout["game_date"].max()),
        "holdout_dates": int(holdout["game_date"].nunique()),
        "holdout_rows": int(len(holdout)),
        "random_repetitions": RANDOM_REPETITIONS,
        "research_holdout_lift_pass": bool(
            overlay["hit_rate"] is not None
            and current["hit_rate"] is not None
            and overlay["hit_rate"] > current["hit_rate"]
            and overlay["hit_rate"] > float(np.nanmean(overlay_random_rates))
        ),
        "holdout_selected_standard_n": int(overlay_card["pp_projection_type"].eq("standard").sum()),
        "minimum_production_holdout_picks": MIN_PRODUCTION_HOLDOUT_PICKS,
        "minimum_production_holdout_dates": MIN_PRODUCTION_HOLDOUT_DATES,
        "minimum_production_standard_picks": MIN_PRODUCTION_STANDARD_PICKS,
    }
    summary["production_gate_pass"] = bool(
        summary["research_holdout_lift_pass"]
        and int(overlay["n"]) >= MIN_PRODUCTION_HOLDOUT_PICKS
        and int(overlay["dates"]) >= MIN_PRODUCTION_HOLDOUT_DATES
        and summary["holdout_selected_standard_n"] >= MIN_PRODUCTION_STANDARD_PICKS
    )
    return model, pd.DataFrame(comparisons), pd.concat(daily_frames, ignore_index=True), summary


def run_evaluation() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _read_csv(ROOT / "data" / "clv_features_2026_live.csv", dtype={"nba_game_id": str})
    pool_frames = []
    selection_frames = []
    fold_frames = []
    comparison_frames = []
    daily_frames = []
    snapshot_summaries = {}
    for snapshot, spec in SNAPSHOT_SPECS.items():
        pool = build_eligible_pool(features, snapshot)
        development_dates, holdout_dates, folds = chronological_split_dates(pool["game_date"].unique())
        development = pool[pool["game_date"].isin(development_dates)].copy()
        holdout = pool[pool["game_date"].isin(holdout_dates)].copy()
        selected_groups, selection, fold_results = select_feature_groups(development, spec, folds)
        selected_features = list(BASE_NUMERIC_FEATURES)
        for group in selected_groups:
            selected_features.extend(spec.candidate_groups[group])
        final_model, comparison, daily, holdout_summary = evaluate_holdout(
            development, holdout, selected_features, snapshot
        )
        model_path = ROOT / "models" / f"clv_selection_overlay_{snapshot}.pkl"
        model_payload = {
            "model": final_model,
            "snapshot_kind": snapshot,
            "selected_feature_groups": selected_groups,
            "numeric_features": selected_features,
            "categorical_features": BASE_CATEGORICAL_FEATURES,
            "eligible_props": sorted(ELIGIBLE_PROPS),
            "eligible_projection_types": sorted(ELIGIBLE_PROJECTION_TYPES),
            "minimum_model_under_edge": MIN_MODEL_UNDER_EDGE,
            "development_end": max(development_dates),
            "holdout_start": min(holdout_dates),
            "enabled_for_production": bool(holdout_summary["production_gate_pass"] and selected_groups),
        }
        tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(model_payload, handle)
        tmp_path.replace(model_path)

        pool_frames.append(pool)
        selection_frames.append(selection)
        fold_frames.append(fold_results)
        comparison_frames.append(comparison)
        daily_frames.append(daily)
        snapshot_summaries[snapshot] = {
            "eligible_rows": int(len(pool)),
            "eligible_dates": int(pool["game_date"].nunique()),
            "development_start": min(development_dates),
            "development_end": max(development_dates),
            "development_dates": len(development_dates),
            "holdout_dates_list": holdout_dates,
            "selected_feature_groups": selected_groups,
            "selected_features": selected_features,
            "forbidden_feature_groups": spec.forbidden_groups,
            "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            **holdout_summary,
        }
    combined_pool = pd.concat(pool_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    policy = {
        snapshot: {
            "enabled": bool(details["production_gate_pass"] and details["selected_feature_groups"]),
            "selected_feature_groups": details["selected_feature_groups"],
            "model_path": details["model_path"],
            "reason": (
                "walk-forward features and untouched holdout both passed"
                if details["production_gate_pass"] and details["selected_feature_groups"]
                else "research-only: no validated production lift"
            ),
        }
        for snapshot, details in snapshot_summaries.items()
    }
    summary = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z"),
        "source_feature_table": "data/clv_features_2026_live.csv",
        "source_feature_rows": int(len(features)),
        "policy_definition": {
            "direction": "UNDER",
            "projection_type": sorted(ELIGIBLE_PROJECTION_TYPES),
            "is_promo": False,
            "eligible_props": sorted(ELIGIBLE_PROPS),
            "minimum_archived_model_under_edge": MIN_MODEL_UNDER_EDGE,
            "card_size": CARD_SIZE,
        },
        "feature_survival_gate": {
            "minimum_mean_top5_lift_pp": 1.0,
            "minimum_positive_folds": 3,
            "minimum_mean_auc_lift": 0.0,
            "minimum_validation_dates": 12,
            "minimum_validation_card_rows": 60,
            "holdout_not_used_for_selection": True,
        },
        "snapshots": snapshot_summaries,
        "production_policy": policy,
        "odds_api_credits_spent": 0,
        "odds_api_credits_reserved": 64120,
        "limitations": [
            "Archived PrizePicks references are not true T-60/T-5 captures; direct forward capture is required.",
            "Standard non-promo metadata spans only eight T-60 and nine T-5 eligible dates; the resulting two/three-date holdouts are too small for promotion.",
            "Archived prediction files reproduce the then-live model outputs, not one immutable model version across all dates.",
            "The overlay is a selection/ranking model; it does not replace the underlying stat regression models.",
        ],
    }
    return summary, combined_pool, selections, folds, pd.concat([comparisons, daily], ignore_index=True, sort=False)


def main() -> int:
    summary, pool, selections, folds, comparison_and_daily = run_evaluation()
    comparisons = comparison_and_daily[comparison_and_daily["method"].isin(
        [
            "shadow_overlay", "current_model_edge", "always_under_pool",
            "matched_random_for_overlay", "matched_random_for_current",
            "shadow_overlay_standard", "current_model_edge_standard",
            "shadow_overlay_demon", "current_model_edge_demon",
            "shadow_overlay_goblin", "current_model_edge_goblin",
        ]
    ) & comparison_and_daily["n"].notna() & comparison_and_daily["game_date"].isna()].copy()
    daily = comparison_and_daily[comparison_and_daily["game_date"].notna()].copy()
    atomic_write_csv(ROOT / "data" / "clv_eligible_pool_frozen.csv", pool)
    atomic_write_csv(ROOT / "output" / "clv_walk_forward_feature_selection.csv", selections)
    atomic_write_csv(ROOT / "output" / "clv_walk_forward_fold_results.csv", folds)
    atomic_write_csv(ROOT / "output" / "clv_holdout_comparison.csv", comparisons)
    atomic_write_csv(ROOT / "output" / "clv_walk_forward_daily.csv", daily)
    atomic_write_json(ROOT / "output" / "clv_walk_forward_summary.json", summary)
    atomic_write_json(ROOT / "models" / "clv_selection_policy.json", summary["production_policy"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
