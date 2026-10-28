"""Rebuild training data daily and train isolated prop candidates periodically.

This orchestrator never promotes or overwrites production prop models.  It
rebuilds the leakage-safe availability/minutes and point-in-time defense joins,
then optionally trains a timestamped candidate and records the date from which
its genuinely future shadow holdout begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"
FULL_TABLE = DATA / "model_data_full_advanced.csv"
REGISTRY = MODELS / "shadow_candidate_registry.json"

PRODUCTION_PATTERNS = (
    "xgb_*_advanced.pkl",
    "clf_*_advanced.pkl",
    "edge_xgb_*_advanced.pkl",
    "meta_clf_*_advanced.pkl",
    "prob_sidecar_*_advanced.pkl",
    "probability_calibrator_advanced.pkl",
    "bias_advanced.json",
    "defense_rating_feature_contract.json",
    "feature_importance_advanced.json",
    "market_feature_sets_advanced.json",
    "probability_oof_advanced.csv",
    "training_edge_analysis.json",
    "training_probability_analysis.json",
    "training_results.json",
    "selected_features_advanced.json",
    "probability_diagnostics_advanced.json",
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def production_hashes() -> dict[str, str]:
    files: set[Path] = set()
    for pattern in PRODUCTION_PATTERNS:
        files.update(path for path in MODELS.glob(pattern) if path.is_file())
    return {str(path.relative_to(ROOT)): _hash_file(path) for path in sorted(files)}


def run_step(name: str, arguments: list[str]) -> dict[str, object]:
    started = datetime.now().astimezone()
    command = [sys.executable, *arguments]
    print(f"\n[{name}]\n{' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    ended = datetime.now().astimezone()
    result = {
        "name": name,
        "command": arguments,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "returncode": int(completed.returncode),
    }
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    return result


def _date_profile(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    if "game_date" not in frame.columns:
        raise ValueError(f"{path.name} has no game_date")
    dates = pd.to_datetime(frame["game_date"], errors="raise")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(frame)),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "columns": int(len(frame.columns)),
    }


def rebuild_training_tables() -> tuple[list[dict[str, object]], dict[str, object]]:
    steps = [
        run_step("Advanced feature table", ["feature_pipeline.py"]),
        run_step(
            "Availability/minutes OOF model",
            ["scripts/train_availability_minutes_model.py"],
        ),
        run_step(
            "Availability/minutes holdout audit",
            ["scripts/audit_availability_minutes_model.py"],
        ),
        run_step(
            "Availability feature join",
            [
                "scripts/build_prop_availability_training_data.py",
                "--prop-input", "data/model_data_advanced.csv",
                "--output", "data/model_data_availability_advanced.csv",
                "--summary", "output/availability_prop_training_join_summary.json",
            ],
        ),
        run_step(
            "Point-in-time team defense",
            ["scripts/build_pointintime_team_defense.py"],
        ),
        run_step(
            "Unified availability + defense join",
            [
                "scripts/build_prop_defense_rating_training_data.py",
                "--prop-input", "data/model_data_availability_advanced.csv",
                "--output", str(FULL_TABLE.relative_to(ROOT)),
                "--summary", "output/full_prop_training_join_summary.json",
            ],
        ),
    ]
    raw_profile = _date_profile(DATA / "nba_data.csv")
    full_profile = _date_profile(FULL_TABLE)
    if full_profile["date_max"] > raw_profile["date_max"]:
        raise RuntimeError("Unified training table extends beyond the raw game log")
    freshness_gap = (
        pd.Timestamp(raw_profile["date_max"]) - pd.Timestamp(full_profile["date_max"])
    ).days
    if freshness_gap > 1:
        raise RuntimeError(
            f"Unified training table is {freshness_gap} completed dates behind raw game data"
        )
    return steps, {
        "raw": raw_profile,
        "unified_training": full_profile,
        "freshness_gap_days": int(freshness_gap),
    }


def _load_registry() -> dict[str, object]:
    if not REGISTRY.exists():
        return {"version": 1, "active": None, "history": []}
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    payload.setdefault("version", 1)
    payload.setdefault("active", None)
    payload.setdefault("history", [])
    return payload


def train_isolated_candidate() -> tuple[list[dict[str, object]], dict[str, object]]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate_id = f"full_layers_{stamp}"
    candidate_dir = MODELS / "weekly_candidates" / candidate_id
    detail = OUTPUT / f"{candidate_id}_same_row_oof.csv"
    comparison = OUTPUT / f"{candidate_id}_same_row_oof_summary.json"
    probability_selection = OUTPUT / f"{candidate_id}_probability_holdout.csv"
    probability_summary = OUTPUT / f"{candidate_id}_probability_holdout_summary.json"

    steps = [
        run_step(
            "Isolated full-layer candidate",
            [
                "train_advanced_models.py",
                "--input", str(FULL_TABLE.relative_to(ROOT)),
                "--model-dir", str(candidate_dir.relative_to(ROOT)),
                "--require-availability-features",
                "--require-defense-rating-features",
                "--no-auto-build-features",
            ],
        ),
    ]
    for artifact in ["availability_minutes_model.pkl", "availability_minutes_policy.json"]:
        source = MODELS / artifact
        if not source.exists():
            raise FileNotFoundError(f"Required live availability artifact is missing: {source}")
        shutil.copy2(source, candidate_dir / artifact)
    steps.extend([
        run_step(
            "Same-row chronological OOF audit",
            [
                "scripts/evaluate_defense_rating_retrain.py",
                "--production", "models/probability_oof_advanced.csv",
                "--candidate", str((candidate_dir / "probability_oof_advanced.csv").relative_to(ROOT)),
                "--production-features", "models/selected_features_advanced.json",
                "--candidate-features", str((candidate_dir / "selected_features_advanced.json").relative_to(ROOT)),
                "--detail", str(detail.relative_to(ROOT)),
                "--summary", str(comparison.relative_to(ROOT)),
            ],
        ),
        run_step(
            "Reserved latest-date probability audit",
            [
                "scripts/evaluate_probability_under_selection.py",
                "--oof", str((candidate_dir / "probability_oof_advanced.csv").relative_to(ROOT)),
                "--detail-output", str(probability_selection.relative_to(ROOT)),
                "--summary-output", str(probability_summary.relative_to(ROOT)),
                "--holdout-dates", "20",
            ],
        ),
    ])

    registry = _load_registry()
    record = {
        "candidate_id": candidate_id,
        "model_dir": str(candidate_dir.relative_to(ROOT)).replace("\\", "/"),
        "created_at": datetime.now().astimezone().isoformat(),
        "future_holdout_start": (datetime.now().date() + timedelta(days=1)).isoformat(),
        "minimum_dates": 15,
        "minimum_rows": 100,
        "status": "shadow_pending_future_results",
        "same_row_oof_summary": str(comparison.relative_to(ROOT)).replace("\\", "/"),
        "probability_holdout_summary": str(probability_summary.relative_to(ROOT)).replace("\\", "/"),
    }
    if not registry.get("active"):
        registry["active"] = record
    else:
        record["status"] = "retained_not_active_existing_shadow_in_progress"
    registry["history"].append(record)
    registry["updated_at"] = datetime.now().astimezone().isoformat()
    _atomic_json(REGISTRY, registry)
    return steps, record


def audit_existing_candidate(candidate_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Finish audits/registration for an already-trained isolated candidate."""
    candidate_dir = candidate_dir if candidate_dir.is_absolute() else ROOT / candidate_dir
    candidate_dir = candidate_dir.resolve()
    weekly_root = (MODELS / "weekly_candidates").resolve()
    if weekly_root not in candidate_dir.parents or not candidate_dir.is_dir():
        raise ValueError(f"Candidate must be an existing child of {weekly_root}: {candidate_dir}")
    candidate_id = candidate_dir.name
    required = ["probability_oof_advanced.csv", "selected_features_advanced.json"]
    missing = [name for name in required if not (candidate_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Candidate audit artifacts are missing: {missing}")
    detail = OUTPUT / f"{candidate_id}_same_row_oof.csv"
    comparison = OUTPUT / f"{candidate_id}_same_row_oof_summary.json"
    probability_selection = OUTPUT / f"{candidate_id}_probability_holdout.csv"
    probability_summary = OUTPUT / f"{candidate_id}_probability_holdout_summary.json"
    steps = [
        run_step(
            "Same-row chronological OOF audit",
            [
                "scripts/evaluate_defense_rating_retrain.py",
                "--production", "models/probability_oof_advanced.csv",
                "--candidate", str((candidate_dir / "probability_oof_advanced.csv").relative_to(ROOT)),
                "--production-features", "models/selected_features_advanced.json",
                "--candidate-features", str((candidate_dir / "selected_features_advanced.json").relative_to(ROOT)),
                "--detail", str(detail.relative_to(ROOT)),
                "--summary", str(comparison.relative_to(ROOT)),
            ],
        ),
        run_step(
            "Reserved latest-date probability audit",
            [
                "scripts/evaluate_probability_under_selection.py",
                "--oof", str((candidate_dir / "probability_oof_advanced.csv").relative_to(ROOT)),
                "--detail-output", str(probability_selection.relative_to(ROOT)),
                "--summary-output", str(probability_summary.relative_to(ROOT)),
                "--holdout-dates", "20",
            ],
        ),
    ]
    registry = _load_registry()
    record = {
        "candidate_id": candidate_id,
        "model_dir": str(candidate_dir.relative_to(ROOT)).replace("\\", "/"),
        "created_at": datetime.fromtimestamp(candidate_dir.stat().st_ctime).astimezone().isoformat(),
        "future_holdout_start": (datetime.now().date() + timedelta(days=1)).isoformat(),
        "minimum_dates": 15,
        "minimum_rows": 100,
        "status": "shadow_pending_future_results",
        "same_row_oof_summary": str(comparison.relative_to(ROOT)).replace("\\", "/"),
        "probability_holdout_summary": str(probability_summary.relative_to(ROOT)).replace("\\", "/"),
    }
    existing = next(
        (item for item in registry["history"] if item.get("candidate_id") == candidate_id),
        None,
    )
    if existing is None:
        registry["history"].append(record)
    else:
        existing.update(record)
    if not registry.get("active") or registry["active"].get("candidate_id") == candidate_id:
        registry["active"] = record
    registry["updated_at"] = datetime.now().astimezone().isoformat()
    _atomic_json(REGISTRY, registry)
    return steps, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["refresh", "candidate", "audit", "all"], default="refresh")
    parser.add_argument("--candidate-dir", type=Path)
    args = parser.parse_args()

    before = production_hashes()
    result: dict[str, object] = {
        "started_at": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "steps": [],
        "production_hashes_before": before,
    }
    try:
        if args.mode in {"refresh", "all"}:
            steps, profile = rebuild_training_tables()
            result["steps"].extend(steps)
            result["training_profile"] = profile
        elif args.mode == "candidate" and not FULL_TABLE.exists():
            raise FileNotFoundError(f"{FULL_TABLE} does not exist; run --mode refresh first")

        if args.mode in {"candidate", "all"}:
            steps, candidate = train_isolated_candidate()
            result["steps"].extend(steps)
            result["candidate"] = candidate
        elif args.mode == "audit":
            if args.candidate_dir is None:
                raise ValueError("--candidate-dir is required with --mode audit")
            steps, candidate = audit_existing_candidate(args.candidate_dir)
            result["steps"].extend(steps)
            result["candidate"] = candidate

        after = production_hashes()
        result["production_hashes_after"] = after
        result["production_unchanged"] = before == after
        if before != after:
            raise RuntimeError("Production prop-model artifacts changed during isolated maintenance")
        result["status"] = "complete"
        return_code = 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["production_hashes_after"] = production_hashes()
        result["production_unchanged"] = before == result["production_hashes_after"]
        return_code = 1
        print(f"\nTraining maintenance failed: {exc}", file=sys.stderr)
    finally:
        result["ended_at"] = datetime.now().astimezone().isoformat()
        _atomic_json(OUTPUT / "training_maintenance_latest.json", result)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
