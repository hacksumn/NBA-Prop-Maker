"""Rebuild the production probability calibrator without retraining prop models.

The active sidecars and OOF predictions are treated as one immutable input
bundle. Calibration is fitted on the earlier OOF dates only; the latest 20
dates remain outside the calibration fit and are recorded as the audit period.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_advanced_models import (  # noqa: E402
    AdvancedPropModel,
    PROBABILITY_BUNDLE_SCHEMA_VERSION,
    _json_safe,
    _sha256_file,
)


def _atomic_pickle(payload: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    os.replace(temporary, path)


def _atomic_json(payload: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    os.replace(temporary, path)


def rebuild(
    model_dir: Path = ROOT / "models",
    holdout_dates: int = 20,
    backup_dir: Path = ROOT / "models" / "backups" / "dec056_pre_probability_bundle",
    manifest_path: Path = ROOT / "output" / "probability_bundle_rebuild_manifest.json",
) -> dict:
    """Fit and atomically activate a sidecar-bound chronological calibrator."""
    model_dir = Path(model_dir).resolve()
    backup_dir = Path(backup_dir).resolve()
    oof_path = model_dir / "probability_oof_advanced.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(oof_path)

    sidecar_paths = sorted(model_dir.glob("prob_sidecar_*_advanced.pkl"))
    if not sidecar_paths:
        raise FileNotFoundError(f"No probability sidecars found in {model_dir}")
    sidecar_hashes = {
        path.name.removeprefix("prob_sidecar_").removesuffix("_advanced.pkl"): _sha256_file(path)
        for path in sidecar_paths
    }

    trainer = AdvancedPropModel(model_dir=str(model_dir))
    trainer.probability_oof_rows = [pd.read_csv(oof_path)]
    fitted = trainer.fit_probability_calibrators(holdout_dates=holdout_dates)
    missing = sorted(set(sidecar_hashes) - set(fitted["calibrators"]))
    if missing:
        raise RuntimeError(f"Calibration could not be fitted for active sidecars: {missing}")

    contract = {
        "schema_version": PROBABILITY_BUNDLE_SCHEMA_VERSION,
        "targets": sorted(sidecar_hashes),
        "sidecar_sha256": sidecar_hashes,
        "oof_sha256": _sha256_file(oof_path),
    }
    calibrator_payload = {
        "calibrators": fitted["calibrators"],
        "diagnostics": fitted["diagnostics"],
        "sidecar_contract": contract,
    }
    diagnostics_payload = {
        "sidecar_contract": contract,
        **fitted["diagnostics"],
    }

    calibrator_path = model_dir / "probability_calibrator_advanced.pkl"
    diagnostics_path = model_dir / "probability_diagnostics_advanced.json"
    if backup_dir.exists() and any(backup_dir.iterdir()):
        raise FileExistsError(f"Backup directory is not empty: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    before = {}
    for path in (calibrator_path, diagnostics_path):
        if path.exists():
            destination = backup_dir / path.name
            shutil.copy2(path, destination)
            before[path.name] = {
                "sha256": _sha256_file(path),
                "backup": str(destination),
            }

    _atomic_pickle(calibrator_payload, calibrator_path)
    _atomic_json(diagnostics_payload, diagnostics_path)

    manifest = {
        "status": "activated",
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "holdout_dates": int(holdout_dates),
        "split": fitted["diagnostics"]["split"],
        "contract": contract,
        "before": before,
        "after": {
            calibrator_path.name: _sha256_file(calibrator_path),
            diagnostics_path.name: _sha256_file(diagnostics_path),
        },
        "regression_or_sidecar_models_changed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--holdout-dates", type=int, default=20)
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=ROOT / "models" / "backups" / "dec056_pre_probability_bundle",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "output" / "probability_bundle_rebuild_manifest.json",
    )
    args = parser.parse_args()
    manifest = rebuild(
        model_dir=args.model_dir,
        holdout_dates=args.holdout_dates,
        backup_dir=args.backup_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps({
        "status": manifest["status"],
        "split": manifest["split"],
        "targets": manifest["contract"]["targets"],
        "regression_or_sidecar_models_changed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
