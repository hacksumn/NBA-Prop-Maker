#!/usr/bin/env python3
"""Recover only audited, useful data assets from Fresh_Start_NBA.zip.

This intentionally does not restore source code, output history, OOF predictions,
or model artifacts.  All validation completes before the first production data
file is changed, and every replaced active file is copied to a dedicated backup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EXPECTED_ZIP_SHA256 = "0c83fcee1a867445ce91cdb2e39e3f40deae2ec9917ffce1cc19d2719d8cfc72"
ZIP_PREFIX = "Fresh_Start_NBA/"
MINIMUM_ROWS = {
    "data/player_profiles.csv": 300,
    "data/player_clean_baselines.csv": 300,
    "data/player_positions.csv": 300,
    "data/game_blowout_index.csv": 1000,
}
REQUIRED_COLUMNS = {
    "data/player_profiles.csv": {"player_id", "player_name", "team_abbr", "gp", "usg_pct"},
    "data/player_clean_baselines.csv": {
        "player_id",
        "player_name",
        "team_abbr",
        "games_analyzed",
        "clean_pts",
        "clean_reb",
        "clean_ast",
    },
    "data/player_positions.csv": {"player_norm", "position_group"},
    "data/game_blowout_index.csv": {
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "blowout_tier",
        "blowout_weight",
    },
}
KEY_COLUMNS = {
    "data/player_profiles.csv": "player_id",
    "data/player_clean_baselines.csv": "player_id",
    "data/player_positions.csv": "player_norm",
    "data/game_blowout_index.csv": "game_id",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv(payload: bytes) -> tuple[list[dict[str, str]], list[str]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = list(reader)
    return rows, list(reader.fieldnames or [])


def _validate_rows(rel_path: str, rows: list[dict[str, str]], columns: list[str]) -> None:
    required = REQUIRED_COLUMNS[rel_path]
    missing_columns = sorted(required - set(columns))
    if missing_columns:
        raise ValueError(f"{rel_path}: missing required columns {missing_columns}")
    minimum = MINIMUM_ROWS[rel_path]
    if len(rows) < minimum:
        raise ValueError(f"{rel_path}: {len(rows)} rows is below the recovery floor {minimum}")

    key_column = KEY_COLUMNS[rel_path]
    keys = [str(row.get(key_column, "")).strip() for row in rows]
    blank_keys = sum(not key for key in keys)
    duplicate_rows = len(keys) - len(set(keys))
    if blank_keys or duplicate_rows:
        raise ValueError(
            f"{rel_path}: invalid {key_column} grain "
            f"(blank={blank_keys}, duplicate_rows={duplicate_rows})"
        )

    if rel_path.endswith("player_positions.csv"):
        invalid = sorted({row.get("position_group", "") for row in rows} - {"G", "F", "C"})
        if invalid:
            raise ValueError(f"{rel_path}: invalid position groups {invalid}")
    if rel_path.endswith("game_blowout_index.csv"):
        dates = [row["game_date"] for row in rows]
        if min(dates) < "2025-10-01" or max(dates) >= "2026-10-01":
            raise ValueError(
                f"{rel_path}: expected only 2025-26 games, observed {min(dates)}..{max(dates)}"
            )


def _csv_bytes(rows: Iterable[dict[str, str]], columns: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _read_existing_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    return _parse_csv(path.read_bytes())


def recover_assets(
    zip_path: Path,
    project_root: Path,
    *,
    expected_sha256: str | None = EXPECTED_ZIP_SHA256,
    backup_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> dict:
    zip_path = zip_path.resolve()
    project_root = project_root.resolve()
    backup_dir = (backup_dir or project_root / "data" / "backups" / "zip_recovery_20261025_pre").resolve()
    manifest_path = (
        manifest_path or project_root / "output" / "zip_recovery_20261025_manifest.json"
    ).resolve()

    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    observed_zip_hash = _sha256_file(zip_path)
    if expected_sha256 and observed_zip_hash.lower() != expected_sha256.lower():
        raise ValueError(
            f"ZIP hash mismatch: expected {expected_sha256.lower()}, observed {observed_zip_hash.lower()}"
        )
    if backup_dir.exists() and any(backup_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace existing recovery backup: {backup_dir}")

    requested = list(MINIMUM_ROWS)
    archive_bytes: dict[str, bytes] = {}
    archive_rows: dict[str, list[dict[str, str]]] = {}
    archive_columns: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for rel_path in requested:
            entry_name = ZIP_PREFIX + rel_path
            try:
                payload = archive.read(entry_name)
            except KeyError as exc:
                raise ValueError(f"ZIP is missing required recovery entry {entry_name}") from exc
            rows, columns = _parse_csv(payload)
            _validate_rows(rel_path, rows, columns)
            archive_bytes[rel_path] = payload
            archive_rows[rel_path] = rows
            archive_columns[rel_path] = columns

    data_dir = project_root / "data"
    snapshot_dir = data_dir / "season_snapshots"
    active_profiles = data_dir / "player_profiles.csv"
    active_baselines = data_dir / "player_clean_baselines.csv"
    active_positions = data_dir / "player_positions.csv"
    active_blowout = data_dir / "game_blowout_index.csv"
    snapshot_profiles = snapshot_dir / "player_profiles_2025-26.csv"
    snapshot_baselines = snapshot_dir / "player_clean_baselines_2025-26.csv"
    snapshot_blowout = snapshot_dir / "game_blowout_index_2025-26.csv"

    current_position_rows, current_position_columns = _read_existing_csv(active_positions)
    if current_position_rows and not {"player_norm", "position_group"}.issubset(current_position_columns):
        raise ValueError(f"Current {active_positions.name} has an invalid schema")
    current_positions: dict[str, dict[str, str]] = {}
    for row in current_position_rows:
        key = str(row.get("player_norm", "")).strip()
        if not key:
            raise ValueError(f"Current {active_positions.name} contains a blank player_norm")
        if key in current_positions:
            raise ValueError(f"Current {active_positions.name} contains duplicate player_norm={key!r}")
        current_positions[key] = row

    archived_positions = {
        row["player_norm"].strip(): row for row in archive_rows["data/player_positions.csv"]
    }
    conflicts = {
        key: (archived_positions[key]["position_group"], current_positions[key]["position_group"])
        for key in archived_positions.keys() & current_positions.keys()
        if archived_positions[key]["position_group"] != current_positions[key]["position_group"]
    }
    if conflicts:
        sample = dict(list(sorted(conflicts.items()))[:10])
        raise ValueError(f"Position recovery has conflicting assignments: {sample}")
    merged_positions = dict(archived_positions)
    merged_positions.update(current_positions)  # Current values win on matching keys.
    merged_position_rows = [merged_positions[key] for key in sorted(merged_positions)]
    merged_position_bytes = _csv_bytes(merged_position_rows, ["player_norm", "position_group"])

    active_paths = [active_profiles, active_baselines, active_positions, active_blowout]
    backup_dir.mkdir(parents=True, exist_ok=False)
    backups: dict[str, dict[str, object]] = {}
    for path in active_paths:
        if not path.exists():
            continue
        backup_path = backup_dir / path.name
        shutil.copy2(path, backup_path)
        backups[path.name] = {
            "path": str(backup_path),
            "bytes": backup_path.stat().st_size,
            "sha256": _sha256_file(backup_path),
        }

    writes = {
        active_profiles: archive_bytes["data/player_profiles.csv"],
        active_baselines: archive_bytes["data/player_clean_baselines.csv"],
        active_positions: merged_position_bytes,
        snapshot_profiles: archive_bytes["data/player_profiles.csv"],
        snapshot_baselines: archive_bytes["data/player_clean_baselines.csv"],
        snapshot_blowout: archive_bytes["data/game_blowout_index.csv"],
    }
    preexisting_destinations = {path: path.exists() for path in writes}
    completed: list[Path] = []
    try:
        for path, payload in writes.items():
            _atomic_write_bytes(path, payload)
            completed.append(path)
    except Exception:
        for path in (active_profiles, active_baselines, active_positions):
            backup_path = backup_dir / path.name
            if backup_path.exists():
                shutil.copy2(backup_path, path)
        for path in completed:
            if not preexisting_destinations[path] and path.exists():
                path.unlink()
        raise

    restored = {
        "player_profiles": {
            "active_path": str(active_profiles),
            "snapshot_path": str(snapshot_profiles),
            "rows": len(archive_rows["data/player_profiles.csv"]),
            "sha256": _sha256_file(active_profiles),
        },
        "player_clean_baselines": {
            "active_path": str(active_baselines),
            "snapshot_path": str(snapshot_baselines),
            "rows": len(archive_rows["data/player_clean_baselines.csv"]),
            "sha256": _sha256_file(active_baselines),
        },
        "player_positions": {
            "active_path": str(active_positions),
            "archive_rows": len(archived_positions),
            "current_rows_before": len(current_positions),
            "rows_after_union": len(merged_positions),
            "archive_only_added": len(archived_positions.keys() - current_positions.keys()),
            "current_only_preserved": len(current_positions.keys() - archived_positions.keys()),
            "conflicts": 0,
            "sha256": _sha256_file(active_positions),
        },
        "game_blowout_index_2025-26": {
            "active_current_season_path_untouched": str(active_blowout),
            "snapshot_path": str(snapshot_blowout),
            "rows": len(archive_rows["data/game_blowout_index.csv"]),
            "date_min": min(row["game_date"] for row in archive_rows["data/game_blowout_index.csv"]),
            "date_max": max(row["game_date"] for row in archive_rows["data/game_blowout_index.csv"]),
            "sha256": _sha256_file(snapshot_blowout),
        },
    }
    manifest = {
        "status": "recovered_and_validated",
        "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_zip": str(zip_path),
        "source_zip_sha256": observed_zip_hash,
        "project_root": str(project_root),
        "backup_dir": str(backup_dir),
        "backups": backups,
        "restored": restored,
        "explicitly_not_restored": [
            "source code from the ZIP",
            "model artifacts from the ZIP",
            "OOF predictions from the ZIP or worktree",
            "picks/slips from the worktree",
            "live injury snapshots",
            "player_projections_today.csv",
        ],
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path.home() / "Downloads" / "Fresh_Start_NBA.zip",
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--expected-sha256", default=EXPECTED_ZIP_SHA256)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    manifest = recover_assets(
        args.zip_path,
        args.project_root,
        expected_sha256=args.expected_sha256,
        backup_dir=args.backup_dir,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest["restored"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
