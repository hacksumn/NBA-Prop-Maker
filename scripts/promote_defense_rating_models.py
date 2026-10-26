"""Activate the isolated DEC-051 defense-rating candidate artifacts safely.

This is an operator-directed activation. Every overwritten production artifact
is copied to a timestamped backup first, all replacements are atomic, and a
SHA-256 manifest records before/source/after content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "models" / "defense_rating_retrain"
DEFAULT_DESTINATION = ROOT / "models"
DEFAULT_BACKUP = ROOT / "models" / "backups" / "dec052_pre_defense_activation"
DEFAULT_MANIFEST = ROOT / "output" / "defense_rating_activation_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".dec052.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _validate_candidate(source_dir: Path) -> tuple[dict, dict]:
    """Validate chronology, defense selection, and probability-bundle lineage before writes."""
    required = [
        "selected_features_advanced.json",
        "training_results.json",
        "defense_rating_feature_contract.json",
        "probability_oof_advanced.csv",
        "probability_calibrator_advanced.pkl",
        "probability_diagnostics_advanced.json",
    ]
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Candidate directory is incomplete: {missing}")

    contract = json.loads(
        (source_dir / "defense_rating_feature_contract.json").read_text(encoding="utf-8")
    )
    if contract.get("chronology_violations") != 0 or int(contract.get("rows", 0)) <= 0:
        raise ValueError("Candidate defense-rating contract failed chronology or row-count checks")
    required_features = {"opp_def_rtg_pt", "opp_pace_pt", "opp_def_tier_pt"}
    if set(contract.get("feature_columns", [])) != required_features:
        raise ValueError("Candidate defense-rating contract has the wrong feature columns")

    selected = json.loads(
        (source_dir / "selected_features_advanced.json").read_text(encoding="utf-8")
    )
    for target in ("stl", "blk"):
        retained = required_features & set(selected.get(target, []))
        if not retained:
            raise ValueError(f"Candidate {target} model does not retain rigorous defense")

    sidecar_paths = sorted(source_dir.glob("prob_sidecar_*_advanced.pkl"))
    sidecar_hashes = {
        path.name.removeprefix("prob_sidecar_").removesuffix("_advanced.pkl"): _sha256(path)
        for path in sidecar_paths
    }
    with (source_dir / "probability_calibrator_advanced.pkl").open("rb") as handle:
        calibrator_payload = pickle.load(handle)
    bundle_contract = calibrator_payload.get("sidecar_contract") or {}
    if bundle_contract.get("sidecar_sha256") != sidecar_hashes:
        raise ValueError("Candidate calibrator does not match its probability sidecars")
    if bundle_contract.get("oof_sha256") != _sha256(source_dir / "probability_oof_advanced.csv"):
        raise ValueError("Candidate calibrator does not match its OOF artifact")

    diagnostics = json.loads(
        (source_dir / "probability_diagnostics_advanced.json").read_text(encoding="utf-8")
    )
    split = diagnostics.get("split") or calibrator_payload.get("diagnostics", {}).get("split", {})
    if (
        split.get("method") != "chronological_latest_dates_holdout"
        or int(split.get("holdout_dates", 0)) < 1
        or int(split.get("holdout_rows", 0)) < 1
    ):
        raise ValueError("Candidate probability calibration lacks a chronological holdout")
    return contract, selected


def promote(
    source_dir: Path = DEFAULT_SOURCE,
    destination_dir: Path = DEFAULT_DESTINATION,
    backup_dir: Path = DEFAULT_BACKUP,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict:
    """Back up and atomically activate every candidate artifact."""
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    backup_dir = backup_dir.resolve()
    if source_dir == destination_dir:
        raise ValueError("Source and destination model directories must differ")
    contract, selected = _validate_candidate(source_dir)

    source_files = sorted(path for path in source_dir.iterdir() if path.is_file())
    if backup_dir.exists() and any(backup_dir.iterdir()):
        raise FileExistsError(
            f"Backup directory already contains files; refusing to overwrite: {backup_dir}"
        )
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # Complete and verify every backup before replacing the first production file.
    for source in source_files:
        destination = destination_dir / source.name
        before_hash = None
        backup = None
        if destination.exists():
            before_hash = _sha256(destination)
            backup = backup_dir / destination.name
            shutil.copy2(destination, backup)
            if _sha256(backup) != before_hash:
                raise RuntimeError(f"Backup verification failed for {destination.name}")
        source_hash = _sha256(source)
        records.append(
            {
                "name": source.name,
                "before_sha256": before_hash,
                "candidate_sha256": source_hash,
                "after_sha256": None,
                "backup": str(backup) if backup else None,
            }
        )

    try:
        for source, record in zip(source_files, records):
            destination = destination_dir / source.name
            _atomic_copy(source, destination)
            after_hash = _sha256(destination)
            if after_hash != record["candidate_sha256"]:
                raise RuntimeError(f"Activation verification failed for {destination.name}")
            record["after_sha256"] = after_hash
    except Exception:
        for record in records:
            destination = destination_dir / record["name"]
            backup_value = record.get("backup")
            if backup_value:
                _atomic_copy(Path(backup_value), destination)
            elif destination.exists():
                destination.unlink()
        raise

    manifest = {
        "status": "activated_by_operator_direction",
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_dir),
        "destination": str(destination_dir),
        "backup": str(backup_dir),
        "files_activated": len(records),
        "training_rows": int(contract["rows"]),
        "production_selected_defense_features": {
            target: [
                feature for feature in features
                if feature in {"opp_def_rtg_pt", "opp_pace_pt", "opp_def_tier_pt"}
            ]
            for target, features in selected.items()
        },
        "reason": (
            "Jake explicitly directed that the corrected team-defense logic be active "
            "and not reverted because the OOF betting-direction comparison was null."
        ),
        "evidence_caveat": (
            "DEC-051 found no positive same-row OOF betting-direction interval and no "
            "reserved future holdout exists; activation is a data-contract correctness decision, "
            "not a validated performance-improvement claim."
        ),
        "files": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = promote(args.source, args.destination, args.backup, args.manifest)
    print(json.dumps({
        "status": manifest["status"],
        "files_activated": manifest["files_activated"],
        "backup": manifest["backup"],
        "production_selected_defense_features": manifest["production_selected_defense_features"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
