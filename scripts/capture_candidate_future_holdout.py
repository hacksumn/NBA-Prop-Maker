"""Capture and grade production-vs-candidate predictions on future FanDuel rows.

The active shadow candidate is frozen in ``models/shadow_candidate_registry.json``.
Predictions are captured before games and graded only after box scores arrive.
No result from this script can promote or overwrite production artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"
REGISTRY = MODELS / "shadow_candidate_registry.json"
STORE = DATA / "candidate_future_holdout_predictions.csv"
SUMMARY = OUTPUT / "candidate_future_holdout_summary.json"
TARGETS = ["pts", "trb", "ast", "pra", "pr", "pa", "stl", "blk", "tov"]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp") as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_active_candidate() -> tuple[dict, dict]:
    if not REGISTRY.exists():
        return {}, {}
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return registry, registry.get("active") or {}


def _active_lines(slate_date: str, as_of=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    import nba_props

    history = pd.read_csv(DATA / "historical_lines.csv", low_memory=False)
    active, active_date, source, mode = nba_props._select_cached_lines_for_predict(
        history,
        slate_date,
        allowed_sources=["fanduel"],
        standard_only=True,
        allow_future_slate=False,
        as_of=as_of,
    )
    if active.empty or active_date != slate_date or source != "fanduel":
        raise RuntimeError(
            f"No exact cached FanDuel slate for {slate_date} (date={active_date}, source={source}, mode={mode})"
        )
    market_cols = [c for c in ["projection_type", "is_promo"] if c in active.columns]
    vegas = active[["player", "prop", "line", *market_cols]].copy()
    vegas["num_books"] = active["num_books"].values if "num_books" in active.columns else 1
    return active, vegas


def _score_model(model_dir: Path, history: pd.DataFrame, vegas: pd.DataFrame, slate_date: str) -> pd.DataFrame:
    import nba_props

    original = nba_props.CONFIG["models_dir"]
    try:
        nba_props.CONFIG["models_dir"] = model_dir
        return nba_props.generate_predictions(
            history,
            vegas,
            use_feature_cache=False,
            slate_date=slate_date,
        )
    finally:
        nba_props.CONFIG["models_dir"] = original


def _long_predictions(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        line_col = f"{target}_line"
        pred_col = f"{target}_pred"
        if line_col not in frame.columns or pred_col not in frame.columns:
            continue
        columns = ["player", line_col, pred_col]
        probability_col = f"{target}_p_over_cal"
        if probability_col in frame.columns:
            columns.append(probability_col)
        part = frame[columns].copy()
        part = part.rename(columns={line_col: "line", pred_col: f"{prefix}_prediction"})
        if probability_col in part.columns:
            part = part.rename(columns={probability_col: f"{prefix}_p_over"})
        part["prop"] = target
        part = part.dropna(subset=["line", f"{prefix}_prediction"])
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pregame_capture_mask(frame: pd.DataFrame, captured_at) -> pd.Series:
    """Require known pre-tip times and a line observed by capture time."""
    captured = pd.to_datetime(captured_at, errors="coerce", utc=True, format="mixed")
    start = pd.to_datetime(frame["commence_time"], errors="coerce", utc=True, format="mixed")
    snapshot = pd.to_datetime(frame["snapshot_ts"], errors="coerce", utc=True, format="mixed")
    return snapshot.notna() & start.notna() & snapshot.le(captured) & start.gt(captured)


def _evaluation_rows(stored: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evaluate only first, provably pregame captures at exact contract grain.

    Raw snapshots remain in the append-only store. Reruns must not give one
    contract extra weight or replace the prediction originally made for it.
    """
    eligible = _pregame_capture_mask(stored, stored["captured_at"])
    valid = stored.loc[eligible].copy()
    valid["_capture_sort"] = pd.to_datetime(valid["captured_at"], utc=True, format="mixed")
    key = ["candidate_id", "game_date", "player_norm", "prop", "line"]
    first = valid.sort_values("_capture_sort", kind="stable").drop_duplicates(key, keep="first")
    return first.drop(columns="_capture_sort"), {
        "stored_rows": int(len(stored)),
        "excluded_unverified_or_posttip_rows": int((~eligible).sum()),
        "excluded_repeat_contract_snapshots": int(len(valid) - len(first)),
        "evaluation_contracts": int(len(first)),
    }


def capture(slate_date: str | None = None) -> dict:
    registry, active_candidate = load_active_candidate()
    if not active_candidate:
        return {"status": "no_active_shadow_candidate", "rows": 0}
    slate_date = slate_date or pd.Timestamp.now(tz="America/Chicago").date().isoformat()
    if slate_date < str(active_candidate["future_holdout_start"]):
        return {"status": "before_future_holdout_start", "rows": 0, "slate_date": slate_date}
    import nba_props

    candidate_dir = ROOT / str(active_candidate["model_dir"])
    if not candidate_dir.is_dir():
        raise FileNotFoundError(candidate_dir)

    started_at = pd.Timestamp.now(tz="UTC")
    active_lines, _ = _active_lines(slate_date, as_of=started_at)
    active_lines = active_lines.loc[_pregame_capture_mask(active_lines, started_at)].copy()
    if active_lines.empty:
        return {"status": "no_verified_pregame_lines", "rows": 0, "slate_date": slate_date}
    market_cols = [c for c in ["projection_type", "is_promo"] if c in active_lines.columns]
    vegas = active_lines[["player", "prop", "line", *market_cols]].copy()
    vegas["num_books"] = active_lines["num_books"].values if "num_books" in active_lines else 1
    history = pd.read_csv(DATA / "nba_data.csv", parse_dates=["game_date"], low_memory=False)
    production = _score_model(MODELS, history, vegas, slate_date)
    candidate = _score_model(candidate_dir, history, vegas, slate_date)
    prod_long = _long_predictions(production, "production")
    cand_long = _long_predictions(candidate, "candidate")
    aligned = prod_long.merge(cand_long, on=["player", "prop", "line"], how="inner", validate="one_to_one")
    aligned["player_norm"] = aligned["player"].map(nba_props._normalize_name)
    timestamps = active_lines.copy()
    timestamps["player_norm"] = timestamps["player"].map(nba_props._normalize_name)
    timestamps["prop"] = timestamps["prop"].map(nba_props._ODDS_PROP_TO_STAT).fillna(timestamps["prop"])
    timestamps["line"] = pd.to_numeric(timestamps["line"], errors="coerce")
    timestamps = timestamps.sort_values("snapshot_ts").drop_duplicates(
        ["player_norm", "prop", "line"], keep="last"
    )
    aligned = aligned.merge(
        timestamps[["player_norm", "prop", "line", "snapshot_ts", "event_id", "commence_time"]],
        on=["player_norm", "prop", "line"], how="left", validate="one_to_one",
    )
    aligned["candidate_id"] = str(active_candidate["candidate_id"])
    aligned["game_date"] = slate_date
    captured_at = pd.Timestamp.now(tz="UTC")
    # Scoring can cross tip-off. Check again immediately before persistence.
    aligned = aligned.loc[_pregame_capture_mask(aligned, captured_at)].copy()
    aligned["captured_at"] = captured_at.isoformat()
    aligned["production_direction"] = np.where(
        aligned["production_prediction"] > aligned["line"], "OVER", "UNDER"
    )
    aligned["candidate_direction"] = np.where(
        aligned["candidate_prediction"] > aligned["line"], "OVER", "UNDER"
    )
    aligned["actual"] = np.nan

    key = ["candidate_id", "game_date", "player_norm", "prop", "line", "snapshot_ts"]
    _, added, _ = nba_props._append_only_history(STORE, aligned, key)
    return {
        "status": "captured",
        "candidate_id": active_candidate["candidate_id"],
        "slate_date": slate_date,
        "rows": int(len(aligned)),
        "new_rows": int(added),
    }


def _actuals() -> pd.DataFrame:
    import nba_props

    games = pd.read_csv(DATA / "nba_data.csv", low_memory=False)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    games["player_norm"] = games["player"].map(nba_props._normalize_name)
    for stat in ["pts", "trb", "ast", "stl", "blk", "tov"]:
        games[stat] = pd.to_numeric(games[stat], errors="coerce")
    games["pra"] = games["pts"] + games["trb"] + games["ast"]
    games["pr"] = games["pts"] + games["trb"]
    games["pa"] = games["pts"] + games["ast"]
    return games[["game_date", "player_norm", *TARGETS]].melt(
        id_vars=["game_date", "player_norm"], var_name="prop", value_name="actual"
    )


def _date_bootstrap(frame: pd.DataFrame, candidate_col: str, comparator_col: str, draws: int = 2000) -> list[float]:
    dates = sorted(frame["game_date"].unique())
    if len(dates) < 2:
        return [float("nan"), float("nan")]
    # Cluster by slate, but retain the row-weighted estimand reported above.
    by_date = frame.groupby("game_date")[[candidate_col, comparator_col]].sum()
    by_date["n"] = frame.groupby("game_date").size()
    rng = np.random.default_rng(42)
    delta = np.empty(draws)
    for index in range(draws):
        sample = rng.choice(dates, size=len(dates), replace=True)
        sampled = by_date.loc[sample]
        delta[index] = float((sampled[candidate_col] - sampled[comparator_col]).sum() / sampled["n"].sum())
    return [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))]


def grade() -> dict:
    registry, active_candidate = load_active_candidate()
    if not STORE.exists() or not active_candidate:
        return {"status": "nothing_to_grade"}
    stored = pd.read_csv(STORE, low_memory=False)
    actuals = _actuals()
    working = stored.drop(columns=["actual"], errors="ignore").merge(
        actuals, on=["game_date", "player_norm", "prop"], how="left", validate="many_to_one"
    )
    atomic_csv(STORE, working)
    eligible, eligibility = _evaluation_rows(working[working["candidate_id"].eq(active_candidate["candidate_id"])])
    graded = eligible[
        eligible["actual"].notna()
        & eligible["actual"].ne(eligible["line"])
    ].copy()
    if graded.empty:
        summary = {"status": "pending_results", "candidate_id": active_candidate["candidate_id"], "rows": 0,
                   "eligibility": eligibility}
        atomic_json(SUMMARY, summary)
        return summary
    actual_over = graded["actual"] > graded["line"]
    graded["candidate_correct"] = graded["candidate_direction"].eq(np.where(actual_over, "OVER", "UNDER")).astype(float)
    graded["production_correct"] = graded["production_direction"].eq(np.where(actual_over, "OVER", "UNDER")).astype(float)
    graded["always_under_correct"] = (~actual_over).astype(float)
    rows = []
    for prop, group in [("ALL", graded), *list(graded.groupby("prop"))]:
        candidate_rate = float(group["candidate_correct"].mean())
        production_rate = float(group["production_correct"].mean())
        under_rate = float(group["always_under_correct"].mean())
        rows.append({
            "prop": prop,
            "rows": int(len(group)),
            "dates": int(group["game_date"].nunique()),
            "candidate_accuracy": candidate_rate,
            "production_accuracy": production_rate,
            "always_under_accuracy": under_rate,
            "candidate_minus_production": candidate_rate - production_rate,
            "candidate_minus_production_ci95": _date_bootstrap(group, "candidate_correct", "production_correct"),
            "candidate_minus_always_under": candidate_rate - under_rate,
            "candidate_minus_always_under_ci95": _date_bootstrap(group, "candidate_correct", "always_under_correct"),
        })
    overall = rows[0]
    complete = bool(
        overall["dates"] >= int(active_candidate.get("minimum_dates", 15))
        and overall["rows"] >= int(active_candidate.get("minimum_rows", 100))
    )
    summary = {
        "status": "complete_pending_manual_review" if complete else "accumulating_future_holdout",
        "candidate_id": active_candidate["candidate_id"],
        "future_holdout_start": active_candidate["future_holdout_start"],
        "minimum_dates": active_candidate.get("minimum_dates", 15),
        "minimum_rows": active_candidate.get("minimum_rows", 100),
        "results": rows,
        "eligibility": eligibility,
        "production_promotion_performed": False,
    }
    atomic_json(SUMMARY, summary)
    if complete:
        registry["active"]["status"] = "future_holdout_complete_pending_manual_review"
        registry["active"]["holdout_summary"] = str(SUMMARY.relative_to(ROOT)).replace("\\", "/")
        registry["updated_at"] = pd.Timestamp.now(tz="America/Chicago").isoformat()
        atomic_json(REGISTRY, registry)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    do_both = not args.capture and not args.grade
    result = {}
    if args.grade or do_both:
        result["grade"] = grade()
    if args.capture or do_both:
        result["capture"] = capture(args.date)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
