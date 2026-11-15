"""Atomic artifact writes and training-refresh receipts.

Readers of a file written here see either the complete old version or the
complete new version, never a partial file: bytes go to a unique temp file in
the same directory, are flushed to disk, and then swapped in with os.replace.

On Windows os.replace fails with PermissionError while another process has the
destination open (for example `predict` loading the model), so the swap is
retried for a bounded time and then fails loudly. The temp file is removed on
every failure path; the destination is never left truncated.

Training receipts let the production-bundle guard in run_daily.py tell an
expected TrainingRefresh update apart from an unexplained mutation. A receipt
is written *before* the swap and names the exact SHA-256 of each new artifact,
so a guard check can never see the new file without its receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

RECEIPT_DIR_NAME = "training_receipts"
PIPELINE_CONTEXT_ENV = "NBA_PIPELINE_CONTEXT"
REPLACE_TIMEOUT_SECONDS = 60.0


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def _replace_with_retry(source: Path, destination: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    delay = 0.05
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise PermissionError(
                    f"Could not atomically replace {destination} within {timeout:.0f}s; "
                    "another process kept it open. The previous version is intact."
                )
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def atomic_write_bytes(path, data: bytes, timeout: float = REPLACE_TIMEOUT_SECONDS) -> str:
    """Atomically replace ``path`` with ``data``; return the SHA-256 written."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"atomic_write_bytes needs bytes, got {type(data).__name__}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temp, path, timeout)
    finally:
        if temp.exists():
            temp.unlink()
    return sha256_bytes(bytes(data))


def atomic_write_text(path, text: str, timeout: float = REPLACE_TIMEOUT_SECONDS) -> str:
    """Atomically replace ``path`` with UTF-8 ``text``; return the SHA-256 written."""
    return atomic_write_bytes(path, text.encode("utf-8"), timeout=timeout)


def atomic_write_csv(frame, path, timeout: float = REPLACE_TIMEOUT_SECONDS) -> str:
    """Atomically replace ``path`` with ``frame`` as CSV (no index)."""
    return atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"), timeout=timeout)


def write_training_receipt(model_dir, artifacts: dict[str, str], writer: str) -> Path:
    """Record the SHA-256 of top-level model artifacts a trainer is about to install.

    Each receipt is a new file (append-only); existing receipts are never changed.
    """
    if not artifacts:
        raise ValueError("A training receipt must name at least one artifact")
    for name in artifacts:
        if Path(name).name != name:
            raise ValueError(f"Receipt artifacts must be top-level file names: {name!r}")
    directory = Path(model_dir) / RECEIPT_DIR_NAME
    now = datetime.now(timezone.utc)
    payload = {
        "writer": writer,
        "written_at_utc": now.isoformat(),
        "pid": os.getpid(),
        "pipeline_context": os.environ.get(PIPELINE_CONTEXT_ENV, ""),
        "artifacts": dict(artifacts),
    }
    receipt = directory / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex}.json"
    atomic_write_text(receipt, json.dumps(payload, indent=2) + "\n")
    return receipt


def load_training_receipts(model_dir, since_utc: datetime, exclude_context: str = "") -> list[dict]:
    """Return receipts written at or after ``since_utc``.

    Receipts whose pipeline_context equals ``exclude_context`` are dropped, so a
    caller can refuse to excuse writes made by its own child processes.
    Unreadable receipts are skipped (they can only fail to excuse a change).
    """
    if since_utc.tzinfo is None:
        raise ValueError("since_utc must be timezone-aware")
    directory = Path(model_dir) / RECEIPT_DIR_NAME
    if not directory.is_dir():
        return []
    receipts = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            written = datetime.fromisoformat(payload["written_at_utc"])
            artifacts = payload["artifacts"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if written.tzinfo is None or written < since_utc or not isinstance(artifacts, dict):
            continue
        if exclude_context and payload.get("pipeline_context", "") == exclude_context:
            continue
        payload["_receipt"] = path.name
        receipts.append(payload)
    return receipts
