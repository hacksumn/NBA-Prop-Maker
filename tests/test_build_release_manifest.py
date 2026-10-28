import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import build_release_manifest as release


def test_record_hashes_and_sizes_file(tmp_path: Path):
    payload = tmp_path / "payload.txt"
    payload.write_bytes(b"release-boundary")
    with patch.object(release, "ROOT", tmp_path):
        item = release.record(payload)
    assert item["path"] == "payload.txt"
    assert item["bytes"] == len(b"release-boundary")
    assert len(item["sha256"]) == 64


def test_record_fails_closed_for_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Required release file is missing"):
        release.record(tmp_path / "missing.csv")


def test_git_blob_record_hashes_exact_committed_bytes(tmp_path: Path):
    payload = b"git-blob\n"
    completed = type("Completed", (), {"stdout": payload})()
    path = tmp_path / "source.py"
    with (
        patch.object(release, "ROOT", tmp_path),
        patch.object(release.subprocess, "run", return_value=completed) as run,
    ):
        item = release.git_blob_record(path, "abc123")
    assert item["path"] == "source.py"
    assert item["bytes"] == len(payload)
    assert item["sha256"] == release.hashlib.sha256(payload).hexdigest()
    run.assert_called_once()


def test_build_manifest_rejects_unbound_calibration(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "probability_diagnostics_advanced.json").write_text(
        json.dumps({"sidecar_contract": {}}), encoding="utf-8"
    )
    with patch.object(release, "ROOT", tmp_path):
        with pytest.raises(RuntimeError, match="bound sidecar/OOF hash contract"):
            release.build_manifest()
