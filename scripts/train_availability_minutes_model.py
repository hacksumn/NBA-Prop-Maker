"""Train and validate a leakage-safe pregame availability/minutes hurdle model.

The historical NBA game-log table contains players who appeared, not a complete
pregame roster. This module reconstructs a conservative candidate roster using
only each player's prior appearances for that team. It then predicts:

1. P(player appears), and
2. minutes conditional on appearing.

Expected minutes is the product of those two estimates. Model selection uses
expanding date folds inside the development seasons. The latest complete season
is untouched until one final holdout evaluation. No artifact from this script is
enabled in the live prop pipeline; prop integration is a separate decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "nba_data.csv"
DEFAULT_CANDIDATES = ROOT / "data" / "availability_minutes_candidates.csv"
DEFAULT_MODEL = ROOT / "models" / "availability_minutes_model.pkl"
DEFAULT_POLICY = ROOT / "models" / "availability_minutes_policy.json"
DEFAULT_SUMMARY = ROOT / "output" / "availability_minutes_validation_summary.json"
DEFAULT_FOLDS = ROOT / "output" / "availability_minutes_walk_forward.csv"
DEFAULT_HOLDOUT = ROOT / "output" / "availability_minutes_holdout_predictions.csv"
DEFAULT_OOF = ROOT / "data" / "availability_minutes_oof_predictions.csv"

MODEL_VERSION = 1
MAX_CANDIDATE_DAYS_SINCE_PLAYED = 45
MIN_PRIOR_TEAM_APPEARANCES = 2
MIN_ROTATION_MINUTES = 8.0
RANDOM_STATE = 42
MIN_COMPLETE_SEASON_DATES = 100


FEATURE_COLUMNS = [
    "avail_l1",
    "avail_rate_l3",
    "avail_rate_l5",
    "avail_rate_l10",
    "avail_rate_l20",
    "missed_streak",
    "days_since_played",
    "mp_last",
    "mp_played_mean_l3",
    "mp_played_mean_l5",
    "mp_played_mean_l10",
    "mp_played_mean_l20",
    "mp_played_std_l10",
    "mp_teamgame_mean_l5",
    "mp_teamgame_mean_l10",
    "mp_trend_3v10",
    "team_days_rest",
    "is_b2b",
    "team_games_last_4d",
    "team_games_prior",
    "player_team_appearances",
    "is_home",
    "season_progress",
    "month_sin",
    "month_cos",
    "team_candidate_count",
    "teammate_available_count_l1",
    "teammate_missing_rotation_minutes_l1",
    "teammate_available_rotation_minutes_l1",
]

IDENTITY_COLUMNS = [
    "game_date", "season", "game_id", "team", "player_id", "player",
]
LABEL_COLUMNS = ["available", "minutes"]


@dataclass
class PlayerState:
    player_id: str
    player: str
    last_play_date: pd.Timestamp
    statuses: deque = field(default_factory=lambda: deque(maxlen=20))
    teamgame_minutes: deque = field(default_factory=lambda: deque(maxlen=20))
    played_minutes: deque = field(default_factory=lambda: deque(maxlen=20))
    appearances: int = 0


@dataclass
class TeamState:
    players: dict[str, PlayerState] = field(default_factory=dict)
    game_dates: deque = field(default_factory=lambda: deque(maxlen=20))


def _safe_mean(values: Iterable[float], fallback: float = 0.0) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)) if len(array) else float(fallback)


def _tail(values: deque, size: int) -> list:
    return list(values)[-size:]


def _missed_streak(statuses: deque) -> int:
    count = 0
    for value in reversed(statuses):
        if int(value) == 1:
            break
        count += 1
    return count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_game_logs(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a game-log frame to the availability model's canonical grain."""
    frame = frame.copy()
    rename = {
        "GAME_ID": "game_id",
        "PLAYER_ID": "player_id",
        "SEASON_YEAR": "season_year",
    }
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame})
    required = {"game_id", "player_id", "player", "team", "game_date", "mp", "season"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"NBA game log is missing columns: {sorted(missing)}")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["mp"] = pd.to_numeric(frame["mp"], errors="coerce")
    frame["player_id"] = frame["player_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    frame["game_id"] = frame["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    frame["team"] = frame["team"].astype(str).str.strip().str.upper()
    frame["season"] = frame["season"].astype(str)
    frame = frame.dropna(subset=["game_date", "mp", "player", "team"])
    frame = frame[frame["mp"].between(0, 65, inclusive="both")].copy()
    duplicate_count = int(frame.duplicated(["game_id", "player_id"]).sum())
    if duplicate_count:
        raise RuntimeError(f"NBA game log contains {duplicate_count} duplicate player-game rows")
    return frame.sort_values(["game_date", "game_id", "team", "player_id"]).reset_index(drop=True)


def load_game_logs(path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    return prepare_game_logs(pd.read_csv(path, low_memory=False))


def _season_progress(game_date: pd.Timestamp, season: str) -> float:
    """Return a pregame-known calendar progress value without dataset-end leakage."""
    try:
        start_year = int(str(season).split("-")[0])
    except (TypeError, ValueError):
        start_year = int(pd.Timestamp(game_date).year)
    season_start = pd.Timestamp(year=start_year, month=10, day=1)
    season_end = pd.Timestamp(year=start_year + 1, month=4, day=30)
    span = max(1, (season_end - season_start).days)
    return float(np.clip((pd.Timestamp(game_date) - season_start).days / span, 0.0, 1.0))


def _player_features(
    state: PlayerState,
    team_state: TeamState,
    game_date: pd.Timestamp,
    is_home: float,
    season_progress: float,
) -> dict[str, float]:
    statuses = list(state.statuses)
    team_minutes = list(state.teamgame_minutes)
    played_minutes = list(state.played_minutes)
    l3_played = _safe_mean(played_minutes[-3:], _safe_mean(played_minutes))
    l5_played = _safe_mean(played_minutes[-5:], l3_played)
    l10_played = _safe_mean(played_minutes[-10:], l5_played)
    l20_played = _safe_mean(played_minutes[-20:], l10_played)
    month_angle = 2.0 * math.pi * (int(game_date.month) - 1) / 12.0
    prior_dates = list(team_state.game_dates)
    team_days_rest = (game_date - prior_dates[-1]).days if prior_dates else 3
    recent_four_day_games = sum((game_date - value).days <= 3 for value in prior_dates)
    return {
        "avail_l1": float(statuses[-1]) if statuses else 1.0,
        "avail_rate_l3": _safe_mean(statuses[-3:], 1.0),
        "avail_rate_l5": _safe_mean(statuses[-5:], 1.0),
        "avail_rate_l10": _safe_mean(statuses[-10:], 1.0),
        "avail_rate_l20": _safe_mean(statuses[-20:], 1.0),
        "missed_streak": float(_missed_streak(state.statuses)),
        "days_since_played": float((game_date - state.last_play_date).days),
        "mp_last": float(played_minutes[-1]),
        "mp_played_mean_l3": l3_played,
        "mp_played_mean_l5": l5_played,
        "mp_played_mean_l10": l10_played,
        "mp_played_mean_l20": l20_played,
        "mp_played_std_l10": float(np.std(played_minutes[-10:])) if len(played_minutes) >= 2 else 0.0,
        "mp_teamgame_mean_l5": _safe_mean(team_minutes[-5:], l5_played),
        "mp_teamgame_mean_l10": _safe_mean(team_minutes[-10:], l10_played),
        "mp_trend_3v10": l3_played - l10_played,
        "team_days_rest": float(np.clip(team_days_rest, 0, 10)),
        "is_b2b": float(team_days_rest <= 1),
        "team_games_last_4d": float(recent_four_day_games),
        "team_games_prior": float(len(prior_dates)),
        "player_team_appearances": float(state.appearances),
        "is_home": float(is_home),
        "season_progress": float(season_progress),
        "month_sin": float(math.sin(month_angle)),
        "month_cos": float(math.cos(month_angle)),
    }


def build_candidate_table(logs: pd.DataFrame) -> pd.DataFrame:
    """Build player/team/game candidates using only state known before each game."""
    rows: list[dict] = []
    schedule = (
        logs.groupby(["season", "game_date", "game_id", "team"], as_index=False)
        .agg(is_home=("is_home", "first") if "is_home" in logs else ("mp", "size"))
        .sort_values(["season", "game_date", "game_id", "team"])
    )
    actual_groups = {
        key: group.set_index("player_id")
        for key, group in logs.groupby(["season", "game_date", "game_id", "team"], sort=False)
    }

    for season, season_schedule in schedule.groupby("season", sort=False):
        team_states: dict[str, TeamState] = {}
        global_last_team: dict[str, str] = {}
        for game_date, date_schedule in season_schedule.groupby("game_date", sort=True):
            pending_global_updates: dict[str, str] = {}
            for game in date_schedule.itertuples(index=False):
                key = (str(season), pd.Timestamp(game_date), str(game.game_id), str(game.team))
                actual = actual_groups[key]
                actual_minutes = {
                    str(player_id): float(row["mp"])
                    for player_id, row in actual.iterrows()
                    if float(row["mp"]) > 0
                }
                team_state = team_states.setdefault(str(game.team), TeamState())
                season_progress = _season_progress(pd.Timestamp(game_date), str(season))
                game_rows: list[dict] = []

                for player_id, state in list(team_state.players.items()):
                    days_since = (pd.Timestamp(game_date) - state.last_play_date).days
                    if days_since > MAX_CANDIDATE_DAYS_SINCE_PLAYED:
                        continue
                    if global_last_team.get(player_id, str(game.team)) != str(game.team):
                        continue
                    recent_rotation_minutes = _safe_mean(_tail(state.played_minutes, 5), 0.0)
                    if state.appearances < MIN_PRIOR_TEAM_APPEARANCES:
                        continue
                    if max(recent_rotation_minutes, float(state.played_minutes[-1])) < MIN_ROTATION_MINUTES:
                        continue
                    feature_row = _player_features(
                        state,
                        team_state,
                        pd.Timestamp(game_date),
                        float(game.is_home) if pd.notna(game.is_home) else 0.5,
                        season_progress,
                    )
                    minutes = float(actual_minutes.get(player_id, 0.0))
                    feature_row.update(
                        {
                            "game_date": pd.Timestamp(game_date),
                            "season": str(season),
                            "game_id": str(game.game_id),
                            "team": str(game.team),
                            "player_id": player_id,
                            "player": state.player,
                            "available": int(minutes > 0),
                            "minutes": minutes,
                        }
                    )
                    game_rows.append(feature_row)

                if game_rows:
                    candidate_count = len(game_rows)
                    total_available_count = sum(row["avail_l1"] for row in game_rows)
                    total_missing_minutes = sum(
                        row["mp_played_mean_l10"] * (1.0 - row["avail_l1"])
                        for row in game_rows
                    )
                    total_available_minutes = sum(
                        row["mp_played_mean_l10"] * row["avail_l1"]
                        for row in game_rows
                    )
                    for row in game_rows:
                        own_missing = row["mp_played_mean_l10"] * (1.0 - row["avail_l1"])
                        own_available = row["mp_played_mean_l10"] * row["avail_l1"]
                        row["team_candidate_count"] = float(candidate_count)
                        row["teammate_available_count_l1"] = float(
                            total_available_count - row["avail_l1"]
                        )
                        row["teammate_missing_rotation_minutes_l1"] = float(
                            total_missing_minutes - own_missing
                        )
                        row["teammate_available_rotation_minutes_l1"] = float(
                            total_available_minutes - own_available
                        )
                    rows.extend(game_rows)

                # Update states only after every feature for this team-game exists.
                for player_id, state in list(team_state.players.items()):
                    minutes = float(actual_minutes.get(player_id, 0.0))
                    appeared = int(minutes > 0)
                    state.statuses.append(appeared)
                    state.teamgame_minutes.append(minutes)
                    if appeared:
                        state.played_minutes.append(minutes)
                        state.last_play_date = pd.Timestamp(game_date)
                        state.appearances += 1
                for player_id, minutes in actual_minutes.items():
                    source_row = actual.loc[player_id]
                    if player_id not in team_state.players:
                        state = PlayerState(
                            player_id=player_id,
                            player=str(source_row["player"]),
                            last_play_date=pd.Timestamp(game_date),
                        )
                        state.statuses.append(1)
                        state.teamgame_minutes.append(minutes)
                        state.played_minutes.append(minutes)
                        state.appearances = 1
                        team_state.players[player_id] = state
                    pending_global_updates[player_id] = str(game.team)
                team_state.game_dates.append(pd.Timestamp(game_date))
            global_last_team.update(pending_global_updates)

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates = candidates[IDENTITY_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS].copy()
    candidates = candidates.sort_values(["game_date", "game_id", "team", "player_id"])
    candidates = candidates.reset_index(drop=True)
    return candidates


def build_live_feature_frame(
    logs: pd.DataFrame,
    schedule: pd.DataFrame,
    target_date: object,
) -> pd.DataFrame:
    """Reconstruct pregame availability features for every current rotation player.

    Synthetic zero-minute marker rows add the scheduled team-games without
    adding a fake player to the candidate pool.  ``build_candidate_table`` then
    advances state chronologically and emits the target-date feature rows using
    only games strictly before the scheduled game.
    """
    history = prepare_game_logs(logs)
    target = pd.Timestamp(target_date).normalize()
    if schedule is None or schedule.empty:
        raise ValueError("A non-empty schedule is required for live availability features")
    required = {"home_team", "away_team"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ValueError(f"Schedule is missing required columns: {missing}")
    game_id_col = "game_id" if "game_id" in schedule.columns else (
        "GAME_ID" if "GAME_ID" in schedule.columns else None
    )
    if game_id_col is None:
        raise ValueError("Schedule is missing game_id")
    season = str(history.loc[history["game_date"].le(target), "season"].iloc[-1])
    markers = []
    for game in schedule.to_dict("records"):
        for team, is_home in [
            (str(game["home_team"]).strip().upper(), 1.0),
            (str(game["away_team"]).strip().upper(), 0.0),
        ]:
            markers.append({
                "game_id": str(game[game_id_col]),
                "player_id": f"__schedule_marker__{team}",
                "player": f"__schedule_marker__{team}",
                "team": team,
                "game_date": target,
                "mp": 0.0,
                "season": season,
                "is_home": is_home,
            })
    extended = pd.concat([history, pd.DataFrame(markers)], ignore_index=True, sort=False)
    candidates = build_candidate_table(extended)
    if candidates.empty:
        return candidates
    live = candidates.loc[candidates["game_date"].eq(target)].copy()
    return live.drop(columns=LABEL_COLUMNS, errors="ignore").reset_index(drop=True)


def make_expanding_date_folds(
    frame: pd.DataFrame,
    n_splits: int = 4,
    min_train_dates: int = 120,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = np.asarray(
        sorted(pd.to_datetime(frame["game_date"]).dt.normalize().unique()),
        dtype="datetime64[ns]",
    )
    if len(dates) < min_train_dates + n_splits:
        min_train_dates = max(20, len(dates) // 2)
    validation_dates = dates[min_train_dates:]
    blocks = [block for block in np.array_split(validation_dates, n_splits) if len(block)]
    folds = []
    normalized = pd.to_datetime(frame["game_date"]).dt.normalize().to_numpy(dtype="datetime64[ns]")
    for block in blocks:
        start = block[0]
        train_index = np.flatnonzero(normalized < start)
        validation_index = np.flatnonzero(np.isin(normalized, block))
        if len(train_index) and len(validation_index):
            folds.append((train_index, validation_index))
    return folds


def availability_models() -> dict[str, Callable[[], object]]:
    return {
        "logistic": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        max_iter=1000,
                        class_weight=None,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def minutes_models() -> dict[str, Callable[[], object]]:
    return {
        "ridge": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def _availability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    prediction = probability >= 0.5
    unavailable = np.asarray(y_true) == 0
    predicted_unavailable = ~prediction
    unavailable_precision = (
        float(np.mean(unavailable[predicted_unavailable])) if predicted_unavailable.any() else 0.0
    )
    unavailable_recall = (
        float(np.mean(predicted_unavailable[unavailable])) if unavailable.any() else 0.0
    )
    return {
        "availability_brier": float(brier_score_loss(y_true, probability)),
        "availability_log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "availability_auc": float(roc_auc_score(y_true, probability)),
        "availability_accuracy": float(np.mean(prediction == np.asarray(y_true))),
        "unavailable_average_precision": float(average_precision_score(unavailable, 1 - probability)),
        "unavailable_precision_at_050": unavailable_precision,
        "unavailable_recall_at_050": unavailable_recall,
    }


def _minutes_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    conditional_minutes: np.ndarray,
) -> dict[str, float]:
    available = frame["available"].to_numpy(dtype=bool)
    actual = frame["minutes"].to_numpy(dtype=float)
    conditional_minutes = np.clip(np.asarray(conditional_minutes, dtype=float), 0, 48)
    expected_minutes = np.clip(probability, 0, 1) * conditional_minutes
    baseline_conditional = frame["mp_played_mean_l5"].to_numpy(dtype=float)
    baseline_probability = frame["avail_rate_l10"].clip(0.02, 0.98).to_numpy(dtype=float)
    baseline_expected = baseline_probability * baseline_conditional
    last_status_baseline = frame["avail_l1"].to_numpy(dtype=float) * baseline_conditional
    hard_gated_minutes = np.where(np.asarray(probability) >= 0.50, conditional_minutes, 0.0)
    strongest_baseline_mae = min(
        mean_absolute_error(actual, baseline_expected),
        mean_absolute_error(actual, last_status_baseline),
    )
    strongest_baseline_rmse = min(
        mean_squared_error(actual, baseline_expected) ** 0.5,
        mean_squared_error(actual, last_status_baseline) ** 0.5,
    )
    return {
        "conditional_minutes_mae": float(mean_absolute_error(actual[available], conditional_minutes[available])),
        "conditional_minutes_rmse": float(mean_squared_error(actual[available], conditional_minutes[available]) ** 0.5),
        "conditional_baseline_mae": float(mean_absolute_error(actual[available], baseline_conditional[available])),
        "expected_minutes_mae": float(mean_absolute_error(actual, expected_minutes)),
        "expected_minutes_rmse": float(mean_squared_error(actual, expected_minutes) ** 0.5),
        "expected_baseline_mae": float(mean_absolute_error(actual, baseline_expected)),
        "expected_baseline_rmse": float(mean_squared_error(actual, baseline_expected) ** 0.5),
        "expected_last_status_baseline_mae": float(mean_absolute_error(actual, last_status_baseline)),
        "expected_last_status_baseline_rmse": float(mean_squared_error(actual, last_status_baseline) ** 0.5),
        "strongest_expected_baseline_mae": float(strongest_baseline_mae),
        "strongest_expected_baseline_rmse": float(strongest_baseline_rmse),
        "hard_gated_minutes_mae": float(mean_absolute_error(actual, hard_gated_minutes)),
        "hard_gated_minutes_rmse": float(mean_squared_error(actual, hard_gated_minutes) ** 0.5),
        "hard_gated_mae_improvement": float(
            strongest_baseline_mae - mean_absolute_error(actual, hard_gated_minutes)
        ),
        "soft_expected_rmse_improvement": float(
            strongest_baseline_rmse - mean_squared_error(actual, expected_minutes) ** 0.5
        ),
        "conditional_within_3": float(np.mean(np.abs(actual[available] - conditional_minutes[available]) <= 3)),
        "conditional_within_5": float(np.mean(np.abs(actual[available] - conditional_minutes[available]) <= 5)),
    }


def evaluate_model_pair(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    availability_factory: Callable[[], object],
    minutes_factory: Callable[[], object],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, object, object]:
    x_train = train[FEATURE_COLUMNS]
    x_validation = validation[FEATURE_COLUMNS]
    availability_model = availability_factory()
    availability_model.fit(x_train, train["available"])
    probability = availability_model.predict_proba(x_validation)[:, 1]

    played_train = train[train["available"].eq(1)]
    minutes_model = minutes_factory()
    minutes_model.fit(played_train[FEATURE_COLUMNS], played_train["minutes"])
    conditional_minutes = minutes_model.predict(x_validation)
    metrics = _availability_metrics(validation["available"].to_numpy(), probability)
    metrics.update(_minutes_metrics(validation, probability, conditional_minutes))
    return metrics, probability, conditional_minutes, availability_model, minutes_model


def walk_forward_select(development: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    folds = make_expanding_date_folds(development)
    if len(folds) < 3:
        raise RuntimeError("Fewer than three expanding date folds are available")
    records = []
    for availability_name, availability_factory in availability_models().items():
        for minutes_name, minutes_factory in minutes_models().items():
            for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
                train = development.iloc[train_index]
                validation = development.iloc[validation_index]
                metrics, _, _, _, _ = evaluate_model_pair(
                    train, validation, availability_factory, minutes_factory
                )
                records.append(
                    {
                        "availability_model": availability_name,
                        "minutes_model": minutes_name,
                        "fold": fold_number,
                        "train_start": train["game_date"].min().date().isoformat(),
                        "train_end": train["game_date"].max().date().isoformat(),
                        "validation_start": validation["game_date"].min().date().isoformat(),
                        "validation_end": validation["game_date"].max().date().isoformat(),
                        "train_rows": int(len(train)),
                        "validation_rows": int(len(validation)),
                        **metrics,
                    }
                )
    results = pd.DataFrame(records)
    pair_summary = (
        results.groupby(["availability_model", "minutes_model"], as_index=False)
        .agg(
            availability_brier=("availability_brier", "mean"),
            availability_auc=("availability_auc", "mean"),
            conditional_minutes_mae=("conditional_minutes_mae", "mean"),
            conditional_baseline_mae=("conditional_baseline_mae", "mean"),
            expected_minutes_mae=("expected_minutes_mae", "mean"),
            expected_baseline_mae=("expected_baseline_mae", "mean"),
        )
    )
    pair_summary = pair_summary.sort_values(
        ["expected_minutes_mae", "availability_brier", "conditional_minutes_mae"]
    )
    selected = pair_summary.iloc[0]
    return str(selected["availability_model"]), str(selected["minutes_model"]), results


def _prediction_frame(
    frame: pd.DataFrame,
    probability: np.ndarray,
    conditional_minutes: np.ndarray,
) -> pd.DataFrame:
    output = frame[IDENTITY_COLUMNS + LABEL_COLUMNS].copy()
    output["availability_probability"] = np.asarray(probability, dtype=float)
    output["conditional_minutes_prediction"] = np.clip(conditional_minutes, 0, 48)
    output["expected_minutes_prediction"] = (
        output["availability_probability"] * output["conditional_minutes_prediction"]
    )
    output["hard_gated_minutes_prediction"] = np.where(
        output["availability_probability"] >= 0.50,
        output["conditional_minutes_prediction"],
        0.0,
    )
    output["rolling_availability_baseline"] = frame["avail_rate_l10"].clip(0.02, 0.98).to_numpy()
    output["last_game_available"] = frame["avail_l1"].to_numpy()
    output["conditional_minutes_baseline"] = frame["mp_played_mean_l5"].to_numpy()
    output["expected_minutes_baseline"] = (
        output["rolling_availability_baseline"] * output["conditional_minutes_baseline"]
    )
    output["last_status_minutes_baseline"] = (
        output["last_game_available"] * output["conditional_minutes_baseline"]
    )
    return output


def generate_oof_predictions(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    availability_factory: Callable[[], object],
    minutes_factory: Callable[[], object],
    holdout_probability: np.ndarray,
    holdout_conditional_minutes: np.ndarray,
) -> pd.DataFrame:
    """Materialize only predictions made by models that precede each target date."""
    frames = []
    for fold_number, (train_index, validation_index) in enumerate(
        make_expanding_date_folds(development), start=1
    ):
        train = development.iloc[train_index]
        validation = development.iloc[validation_index]
        _, probability, conditional, _, _ = evaluate_model_pair(
            train, validation, availability_factory, minutes_factory
        )
        predicted = _prediction_frame(validation, probability, conditional)
        predicted["prediction_split"] = f"walk_forward_{fold_number}"
        predicted["model_train_end"] = train["game_date"].max().date().isoformat()
        frames.append(predicted)
    holdout_predictions = _prediction_frame(
        holdout, holdout_probability, holdout_conditional_minutes
    )
    holdout_predictions["prediction_split"] = "untouched_holdout"
    holdout_predictions["model_train_end"] = development["game_date"].max().date().isoformat()
    frames.append(holdout_predictions)
    output = pd.concat(frames, ignore_index=True)
    target_dates = pd.to_datetime(output["game_date"])
    train_dates = pd.to_datetime(output["model_train_end"])
    if not (train_dates < target_dates).all():
        raise RuntimeError("OOF predictions include a model trained through the target date")
    if output.duplicated(["game_id", "team", "player_id"]).any():
        raise RuntimeError("OOF prediction keys are not unique")
    return output.sort_values(["game_date", "game_id", "team", "player_id"]).reset_index(drop=True)


def select_complete_holdout_season(
    candidates: pd.DataFrame,
    minimum_unique_dates: int = MIN_COMPLETE_SEASON_DATES,
) -> tuple[str, pd.DataFrame]:
    """Choose the latest season with enough calendar coverage to be a holdout.

    The active season must never become the acceptance holdout merely because
    it is the latest label in the file.  A 100-date floor is deliberately well
    below a normal NBA regular season while excluding an opening-week fragment.
    """
    profile = (
        candidates.groupby("season", as_index=False)
        .agg(
            rows=("game_date", "size"),
            unique_dates=("game_date", "nunique"),
            start_date=("game_date", "min"),
            end_date=("game_date", "max"),
        )
        .sort_values(["end_date", "season"])
        .reset_index(drop=True)
    )
    complete = profile[profile["unique_dates"].ge(int(minimum_unique_dates))]
    if complete.empty:
        raise RuntimeError(
            f"No season has the required {minimum_unique_dates} unique dates for an untouched holdout"
        )
    holdout_season = str(complete.iloc[-1]["season"])
    return holdout_season, profile


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def predict_availability_minutes_features(
    frame: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Score a precomputed pregame feature frame with the deployment package."""
    with model_path.open("rb") as handle:
        package = pickle.load(handle)
    feature_columns = list(package["feature_columns"])
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Availability/minutes feature frame is missing: {sorted(missing)}")
    probability = package["availability_model"].predict_proba(frame[feature_columns])[:, 1]
    conditional = np.clip(package["minutes_model"].predict(frame[feature_columns]), 0, 48)
    output_columns = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    output = frame[output_columns].copy()
    output["availability_probability"] = probability
    output["conditional_minutes_prediction"] = conditional
    output["expected_minutes_prediction"] = probability * conditional
    output["hard_gated_minutes_prediction"] = np.where(probability >= 0.50, conditional, 0.0)
    return output


def train_and_validate(
    input_path: Path = DEFAULT_INPUT,
    candidates_path: Path = DEFAULT_CANDIDATES,
    model_path: Path = DEFAULT_MODEL,
    policy_path: Path = DEFAULT_POLICY,
    summary_path: Path = DEFAULT_SUMMARY,
    folds_path: Path = DEFAULT_FOLDS,
    holdout_path: Path = DEFAULT_HOLDOUT,
    oof_path: Path = DEFAULT_OOF,
) -> dict:
    logs = load_game_logs(input_path)
    candidates = build_candidate_table(logs)
    if candidates.empty:
        raise RuntimeError("Candidate construction returned no rows")
    if candidates.duplicated(["game_id", "team", "player_id"]).any():
        raise RuntimeError("Candidate table is not unique at player/team/game grain")
    if candidates[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Candidate feature table contains unexpected nulls")

    holdout_season, season_profile = select_complete_holdout_season(candidates)
    holdout_start = candidates.loc[
        candidates["season"].eq(holdout_season), "game_date"
    ].min()
    development = candidates[candidates["game_date"].lt(holdout_start)].copy()
    holdout = candidates[candidates["season"].eq(holdout_season)].copy()
    forward = candidates[candidates["game_date"].gt(holdout["game_date"].max())].copy()
    if development.empty or holdout.empty:
        raise RuntimeError("Need development seasons before the latest complete-season holdout")

    availability_name, minutes_name, fold_results = walk_forward_select(development)
    availability_factory = availability_models()[availability_name]
    minutes_factory = minutes_models()[minutes_name]
    holdout_metrics, probability, conditional_minutes, _, _ = evaluate_model_pair(
        development, holdout, availability_factory, minutes_factory
    )
    holdout_predictions = _prediction_frame(holdout, probability, conditional_minutes)
    oof_predictions = generate_oof_predictions(
        development,
        holdout,
        availability_factory,
        minutes_factory,
        probability,
        conditional_minutes,
    )
    forward_metrics = None
    if not forward.empty:
        forward_train = pd.concat([development, holdout], ignore_index=True)
        forward_metrics, forward_probability, forward_conditional, _, _ = evaluate_model_pair(
            forward_train,
            forward,
            availability_factory,
            minutes_factory,
        )
        forward_predictions = _prediction_frame(
            forward, forward_probability, forward_conditional
        )
        forward_predictions["prediction_split"] = "forward_current_season"
        forward_predictions["model_train_end"] = (
            forward_train["game_date"].max().date().isoformat()
        )
        oof_predictions = pd.concat(
            [oof_predictions, forward_predictions], ignore_index=True
        ).sort_values(["game_date", "game_id", "team", "player_id"]).reset_index(drop=True)
        if not (
            pd.to_datetime(oof_predictions["model_train_end"])
            < pd.to_datetime(oof_predictions["game_date"])
        ).all():
            raise RuntimeError("Forward availability predictions violate chronology")
        if oof_predictions.duplicated(["game_id", "team", "player_id"]).any():
            raise RuntimeError("Availability prediction keys are not unique after forward scoring")

    selected_folds = fold_results[
        fold_results["availability_model"].eq(availability_name)
        & fold_results["minutes_model"].eq(minutes_name)
    ].copy()
    brier_baseline_by_fold = []
    for _, row in selected_folds.iterrows():
        validation = development[
            development["game_date"].between(row["validation_start"], row["validation_end"])
        ]
        brier_baseline_by_fold.append(
            brier_score_loss(
                validation["available"],
                validation["avail_rate_l10"].clip(0.02, 0.98),
            )
        )
    selected_folds["availability_baseline_brier"] = brier_baseline_by_fold
    selected_folds["availability_brier_improvement"] = (
        selected_folds["availability_baseline_brier"] - selected_folds["availability_brier"]
    )
    selected_folds["conditional_mae_improvement"] = (
        selected_folds["conditional_baseline_mae"] - selected_folds["conditional_minutes_mae"]
    )
    selected_folds["expected_mae_improvement"] = (
        selected_folds["expected_baseline_mae"] - selected_folds["expected_minutes_mae"]
    )

    holdout_availability_baseline_brier = brier_score_loss(
        holdout["available"], holdout["avail_rate_l10"].clip(0.02, 0.98)
    )
    holdout_metrics["availability_baseline_brier"] = float(holdout_availability_baseline_brier)
    holdout_metrics["availability_brier_improvement"] = float(
        holdout_availability_baseline_brier - holdout_metrics["availability_brier"]
    )
    holdout_metrics["conditional_mae_improvement"] = float(
        holdout_metrics["conditional_baseline_mae"] - holdout_metrics["conditional_minutes_mae"]
    )
    holdout_metrics["expected_mae_improvement"] = float(
        holdout_metrics["expected_baseline_mae"] - holdout_metrics["expected_minutes_mae"]
    )

    validation_passes = {
        "availability_positive_folds": int(selected_folds["availability_brier_improvement"].gt(0).sum()),
        "conditional_minutes_positive_folds": int(selected_folds["conditional_mae_improvement"].gt(0).sum()),
        "expected_minutes_positive_folds": int(selected_folds["expected_mae_improvement"].gt(0).sum()),
        "hard_gated_minutes_positive_folds": int(selected_folds["hard_gated_mae_improvement"].gt(0).sum()),
        "soft_expected_rmse_positive_folds": int(selected_folds["soft_expected_rmse_improvement"].gt(0).sum()),
        "fold_count": int(len(selected_folds)),
    }
    accepted = bool(
        validation_passes["availability_positive_folds"] >= 3
        and validation_passes["conditional_minutes_positive_folds"] >= 3
        and validation_passes["expected_minutes_positive_folds"] >= 3
        and validation_passes["hard_gated_minutes_positive_folds"] >= 3
        and validation_passes["soft_expected_rmse_positive_folds"] >= 3
        and holdout_metrics["availability_brier_improvement"] > 0
        and holdout_metrics["conditional_mae_improvement"] > 0
        and holdout_metrics["hard_gated_mae_improvement"] > 0
        and holdout_metrics["soft_expected_rmse_improvement"] > 0
    )

    # Deployment artifacts are trained only after the model families and gates
    # have been frozen and the untouched holdout has been evaluated once.
    deployment_availability = availability_factory()
    deployment_availability.fit(candidates[FEATURE_COLUMNS], candidates["available"])
    deployment_minutes = minutes_factory()
    played_candidates = candidates[candidates["available"].eq(1)]
    deployment_minutes.fit(played_candidates[FEATURE_COLUMNS], played_candidates["minutes"])

    data_quality = {
        "source_rows": int(len(logs)),
        "source_player_game_duplicates": int(logs.duplicated(["game_id", "player_id"]).sum()),
        "source_date_min": logs["game_date"].min().date().isoformat(),
        "source_date_max": logs["game_date"].max().date().isoformat(),
        "source_seasons": sorted(logs["season"].unique().tolist()),
        "candidate_rows": int(len(candidates)),
        "candidate_unique_dates": int(candidates["game_date"].nunique()),
        "candidate_unique_players": int(candidates["player_id"].nunique()),
        "candidate_duplicate_keys": int(candidates.duplicated(["game_id", "team", "player_id"]).sum()),
        "candidate_availability_rate": float(candidates["available"].mean()),
        "candidate_feature_null_cells": int(candidates[FEATURE_COLUMNS].isna().sum().sum()),
        "development_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "holdout_season": holdout_season,
        "holdout_start": holdout["game_date"].min().date().isoformat(),
        "holdout_end": holdout["game_date"].max().date().isoformat(),
        "holdout_minimum_unique_dates": MIN_COMPLETE_SEASON_DATES,
        "season_profile": season_profile.assign(
            start_date=season_profile["start_date"].dt.date.astype(str),
            end_date=season_profile["end_date"].dt.date.astype(str),
        ).to_dict("records"),
        "forward_rows": int(len(forward)),
        "forward_start": (
            forward["game_date"].min().date().isoformat() if not forward.empty else None
        ),
        "forward_end": (
            forward["game_date"].max().date().isoformat() if not forward.empty else None
        ),
        "oof_rows": int(len(oof_predictions)),
        "oof_start": pd.to_datetime(oof_predictions["game_date"]).min().date().isoformat(),
        "oof_end": pd.to_datetime(oof_predictions["game_date"]).max().date().isoformat(),
        "oof_same_or_future_train_dates": int(
            (
                pd.to_datetime(oof_predictions["model_train_end"])
                >= pd.to_datetime(oof_predictions["game_date"])
            ).sum()
        ),
    }
    summary = {
        "model_version": MODEL_VERSION,
        "accepted_for_prop_retraining": accepted,
        "enabled_for_live": False,
        "selected_availability_model": availability_name,
        "selected_minutes_model": minutes_name,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_contract": {
            "max_days_since_played": MAX_CANDIDATE_DAYS_SINCE_PLAYED,
            "minimum_prior_team_appearances": MIN_PRIOR_TEAM_APPEARANCES,
            "minimum_rotation_minutes": MIN_ROTATION_MINUTES,
            "candidate_source": "prior same-team appearances only",
            "availability_label": "mp > 0 in the scheduled team game",
            "minutes_label": "box-score mp if available, otherwise 0",
        },
        "data_quality": data_quality,
        "validation_gate": validation_passes,
        "walk_forward_selected_mean": {
            column: float(selected_folds[column].mean())
            for column in [
                "availability_brier",
                "availability_baseline_brier",
                "availability_brier_improvement",
                "availability_auc",
                "conditional_minutes_mae",
                "conditional_baseline_mae",
                "conditional_mae_improvement",
                "expected_minutes_mae",
                "expected_baseline_mae",
                "expected_mae_improvement",
                "hard_gated_minutes_mae",
                "hard_gated_mae_improvement",
                "expected_minutes_rmse",
                "strongest_expected_baseline_rmse",
                "soft_expected_rmse_improvement",
            ]
        },
        "untouched_holdout": holdout_metrics,
        "forward_current_season": forward_metrics,
        "source": {
            "path": str(input_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(input_path),
        },
    }
    policy = {
        "version": MODEL_VERSION,
        "accepted_for_prop_retraining": accepted,
        "enabled_for_live": False,
        "availability_model": availability_name,
        "minutes_model": minutes_name,
        "holdout_season": holdout_season,
        "feature_columns": FEATURE_COLUMNS,
        "integration_rule": (
            "May be used as an input to the next prop-model retraining only when "
            "accepted_for_prop_retraining=true; live use remains disabled until "
            "the retrained prop stack passes its own untouched chronological test."
        ),
    }
    model_package = {
        "version": MODEL_VERSION,
        "availability_model": deployment_availability,
        "minutes_model": deployment_minutes,
        "availability_model_name": availability_name,
        "minutes_model_name": minutes_name,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_contract": summary["candidate_contract"],
        "training_max_date": candidates["game_date"].max().date().isoformat(),
        "accepted_for_prop_retraining": accepted,
        "enabled_for_live": False,
    }

    for path in [
        candidates_path, model_path, policy_path, summary_path, folds_path,
        holdout_path, oof_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_path, index=False)
    fold_results.to_csv(folds_path, index=False)
    holdout_predictions.to_csv(holdout_path, index=False)
    oof_predictions.to_csv(oof_path, index=False)
    with model_path.open("wb") as handle:
        pickle.dump(model_package, handle)
    policy_path.write_text(json.dumps(json_ready(policy), indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    args = parser.parse_args()
    summary = train_and_validate(
        input_path=args.input,
        candidates_path=args.candidates,
        model_path=args.model,
        policy_path=args.policy,
        summary_path=args.summary,
        folds_path=args.folds,
        holdout_path=args.holdout,
        oof_path=args.oof,
    )
    print(json.dumps(json_ready(summary), indent=2))
    return 0 if summary["accepted_for_prop_retraining"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
