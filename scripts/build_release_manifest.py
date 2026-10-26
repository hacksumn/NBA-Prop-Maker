"""Build a hash-audited manifest for the active source, data, and model release.

This script is read-only apart from its JSON output. It deliberately records
large canonical data files in place instead of duplicating them into another
directory. The manifest therefore creates a precise rollback/audit boundary
without risking a second drifting copy of the project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output" / "release_manifest_20261025.json"

SOURCE_ROOT_FILES = (
    ".gitignore",
    "README.md",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "CLAUDE.md",
    "CONTEXT.md",
    "PROJECT_STATE.md",
    "feature_cols.json",
    "feature_cols_advanced.json",
)

MEMORY_FILES = (
    "output/AGENTS.md",
    "output/ARCHITECTURE.md",
    "output/STATUS.md",
    "output/TASKS.md",
    "output/DECISIONS.md",
)

CANONICAL_DATA_FILES = (
    "data/nba_data.csv",
    "data/historical_lines.csv",
    "data/nba_team_game_logs.csv",
    "data/model_data_advanced.csv",
    "data/model_data_defenserating_advanced.csv",
    "data/pointintime_team_defense_ratings.csv",
    "data/player_profiles.csv",
    "data/player_clean_baselines.csv",
    "data/player_positions.csv",
    "data/game_blowout_index.csv",
    "data/player_projections_today.csv",
)

POLICY_FILES = (
    "models/availability_minutes_policy.json",
    "models/clv_selection_policy.json",
    "models/sportsbook_selection_policy.json",
    "models/selected_features_advanced.json",
    "models/defense_rating_feature_contract.json",
    "models/probability_diagnostics_advanced.json",
)

EVIDENCE_FILES = (
    "output/dec057_production_recovery_manifest.json",
    "output/zip_recovery_20261025_manifest.json",
    "output/probability_under_selection_summary.json",
    "output/training_live_selected_feature_parity_summary.json",
    "output/weekly_candidate_20261025_recovery_validation_same_row_oof_summary.json",
)

PRODUCTION_MODEL_PATTERNS = (
    "xgb_*_advanced.pkl",
    "clf_*_advanced.pkl",
    "edge_xgb_*_advanced.pkl",
    "meta_clf_*_advanced.pkl",
    "prob_sidecar_*_advanced.pkl",
    "probability_calibrator_advanced.pkl",
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    """Return a stable path/size/hash record for a required file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required release file is missing: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def records(paths: Iterable[Path]) -> list[dict[str, object]]:
    """Build sorted records and reject duplicate paths."""
    unique = sorted(set(paths), key=lambda item: item.as_posix().lower())
    return [record(path) for path in unique]


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def source_paths() -> list[Path]:
    paths = [ROOT / name for name in SOURCE_ROOT_FILES]
    paths.extend(path for path in ROOT.glob("*.py") if ".bak_" not in path.name)
    paths.extend(path for path in (ROOT / "scripts").glob("*.py"))
    paths.extend(path for path in (ROOT / "scripts").glob("*.ps1"))
    paths.extend(path for path in (ROOT / "tests").glob("test_*.py"))
    paths.extend(ROOT / name for name in MEMORY_FILES)
    return paths


def production_model_paths() -> list[Path]:
    paths: list[Path] = []
    for pattern in PRODUCTION_MODEL_PATTERNS:
        paths.extend((ROOT / "models").glob(pattern))
    if len(set(paths)) < 46:
        raise RuntimeError(
            f"Production bundle is incomplete: found only {len(set(paths))} binary artifacts"
        )
    return paths


def build_manifest() -> dict[str, object]:
    """Assemble the complete release manifest and validate core contracts."""
    diagnostics_path = ROOT / "models" / "probability_diagnostics_advanced.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    sidecar_contract = diagnostics.get("sidecar_contract") or {}
    if not sidecar_contract.get("sidecar_sha256") or not sidecar_contract.get("oof_sha256"):
        raise RuntimeError("Probability diagnostics are missing the bound sidecar/OOF hash contract")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_name": "DEC-059 operator-safe release boundary",
        "git": {
            "head": git_text("rev-parse", "HEAD"),
            "branch": git_text("branch", "--show-current"),
            "working_tree_clean": not bool(git_text("status", "--porcelain")),
        },
        "contracts": {
            "production_models_are_immutable_during_daily_operations": True,
            "automatic_model_promotion": False,
            "odds_api_spend_default": False,
            "prizepicks_capture_windows": ["T-60", "T-5"],
            "calibration_fit_rows": diagnostics.get("calibration_fit_rows"),
            "calibration_holdout_rows": diagnostics.get("calibration_holdout_rows"),
            "calibration_holdout_dates": diagnostics.get("calibration_holdout_dates"),
            "probability_sidecar_contract": sidecar_contract,
        },
        "source": records(source_paths()),
        "canonical_data": records(ROOT / name for name in CANONICAL_DATA_FILES),
        "production_models": records(production_model_paths()),
        "policies": records(ROOT / name for name in POLICY_FILES),
        "evidence": records(ROOT / name for name in EVIDENCE_FILES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = build_manifest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "source_files": len(manifest["source"]),
                "canonical_data_files": len(manifest["canonical_data"]),
                "production_model_files": len(manifest["production_models"]),
                "working_tree_clean": manifest["git"]["working_tree_clean"],
                "manifest_sha256": file_sha256(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

