"""Preserve every replaced pipeline artifact as a compressed, hashed version.

The Python audit hook observes writes, replacements and removals in the project's
data, output, models and logs directories. Retention failures block the mutation.
Archives have no automatic expiry. Use --snapshot to retain existing artifacts,
--run SCRIPT [ARGS] to protect a script, or --restore HASH --output PATH.
"""
from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import json
import os
import runpy
import shutil
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_INSTALLED = {}


class RetentionStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.archive = self.root / ".retained_versions"
        self.local = threading.local()
        self.pending = set()
        self.cache = {}

    def protected(self, value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        path = Path(os.fsdecode(value)).resolve()
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return None
        if not relative.parts or relative.parts[0] not in {"data", "output", "models", "logs"}:
            if path.parent != self.root or path.name not in {"feature_cols.json", "feature_cols_advanced.json"}:
                return None
        if path.suffix in {".tmp", ".lock", ".pyc"} or "__pycache__" in relative.parts:
            return None
        return path

    def preserve(self, path, logical_path=None):
        """Copy bytes before mutation and record their original logical path."""
        path = Path(path)
        if not path.is_file():
            return None
        logical = Path(logical_path or path).resolve()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        cached = self.cache.get((str(path), str(logical)))
        if cached and cached[0] == signature:
            return cached[1]
        self.local.busy = True
        temporary = None
        try:
            objects = self.archive / "objects"
            objects.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(dir=objects, suffix=".tmp")
            temporary = Path(temporary_name)
            digest = hashlib.sha256()
            with os.fdopen(fd, "wb") as target, path.open("rb") as source:
                with gzip.GzipFile(fileobj=target, mode="wb", compresslevel=1, mtime=0) as compressed:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        compressed.write(block)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ino) != signature:
                raise RuntimeError(f"Artifact changed during retention: {logical}")
            sha = digest.hexdigest()
            destination = objects / f"{sha}.gz"
            if destination.exists():
                temporary.unlink()
            else:
                os.replace(temporary, destination)
            relative = logical.relative_to(self.root).as_posix()
            record_dir = self.archive / "records" / hashlib.sha256(relative.encode()).hexdigest()
            record_dir.mkdir(parents=True, exist_ok=True)
            record = record_dir / f"{sha}.json"
            if not record.exists():
                payload = {"path": relative, "sha256": sha, "bytes": stat.st_size,
                           "preserved_at": datetime.now(timezone.utc).isoformat()}
                staged = record.with_name(uuid.uuid4().hex + ".tmp")
                staged.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(staged, record)
            self.cache[(str(path), str(logical))] = (signature, sha)
            return sha
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
            self.local.busy = False

    def audit(self, event, args):
        if getattr(self.local, "busy", False):
            return
        if event == "open":
            value, mode, flags = args
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            path = self.protected(value) if writing else None
            if path is not None:
                self.cache.pop((str(path), str(path)), None)
                self.preserve(path)
                self.cache.pop((str(path), str(path)), None)
                self.pending.add(path)
        elif event == "os.rename":
            source, destination = args[:2]
            target = self.protected(destination)
            original = self.protected(source)
            if original is not None:
                self.preserve(original)
            if target is not None:
                self.preserve(target)
                # os.fsdecode: byte paths are valid for os.replace but not for Path().
                self.preserve(Path(os.fsdecode(source)), logical_path=target)
                self.pending.add(target)
        elif event == "os.truncate":
            # Path-based os.truncate empties a file without an "open" event. A
            # descriptor-based truncate needs an earlier writable open, which the
            # "open" branch already preserved.
            value = args[0]
            path = self.protected(value) if isinstance(value, (str, bytes, os.PathLike)) else None
            if path is not None:
                self.preserve(path)
                self.pending.add(path)
        elif event == "os.remove":
            path = self.protected(args[0])
            if path is not None:
                self.preserve(path)

    def checkpoint(self, all_files=False):
        paths = set(self.pending)
        if not paths and not all_files:
            return None, 0
        if all_files:
            for folder in ["data", "output", "models", "logs"]:
                paths.update(p for p in (self.root / folder).rglob("*") if self.protected(p) and p.is_file())
            paths.update(p for p in self.root.glob("feature_cols*.json") if self.protected(p))
        versions = {}
        for path in sorted(paths):
            sha = self.preserve(path)
            if sha is not None:
                versions[path.relative_to(self.root).as_posix()] = sha
        self.local.busy = True
        try:
            directory = self.archive / "checkpoints"
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            manifest = directory / f"{stamp}_{uuid.uuid4().hex}.json"
            manifest.write_text(json.dumps({"created_at": stamp, "versions": versions}, indent=2), encoding="utf-8")
        finally:
            self.local.busy = False
        return manifest, len(versions)


def install_retention(root=None):
    """Install once per process/project, including ordinary direct CLI runs."""
    root = Path(root or Path(__file__).parent).resolve()
    if root not in _INSTALLED:
        store = RetentionStore(root)
        _INSTALLED[root] = store
        sys.addaudithook(store.audit)
        atexit.register(store.checkpoint)
    return _INSTALLED[root]


@contextmanager
def artifact_lock(path):
    """Serialize cooperating read/merge/write operations across processes."""
    path = Path(path).resolve()
    directory = path.parent / ".write_locks"
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / (hashlib.sha256(path.name.encode()).hexdigest() + ".lock")
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--retain", nargs="+", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--list-versions", type=Path)
    parser.add_argument("--restore")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    store = install_retention()
    if args.snapshot:
        path, count = store.checkpoint(all_files=True)
        print(f"Retained {count} files. Manifest {path}")
    if args.retain:
        for path in args.retain:
            protected = store.protected(path)
            if protected is None:
                parser.error(f"Outside protected artifact paths: {path}")
            store.preserve(protected)
    if args.verify:
        count = 0
        for path in (store.archive / "objects").glob("*.gz"):
            digest = hashlib.sha256()
            with gzip.open(path, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != path.stem:
                raise RuntimeError(f"Retained object checksum mismatch: {path.name}")
            count += 1
        print(f"Verified {count} retained objects by decompression and SHA-256")
    if args.list_versions:
        path = store.protected(args.list_versions)
        if path is None:
            parser.error("--list-versions requires a protected artifact path")
        relative = path.relative_to(store.root).as_posix()
        directory = store.archive / "records" / hashlib.sha256(relative.encode()).hexdigest()
        versions = [json.loads(record.read_text(encoding="utf-8")) for record in directory.glob("*.json")]
        print(json.dumps(sorted(versions, key=lambda item: item["preserved_at"]), indent=2))
    if args.restore:
        if len(args.restore) != 64 or any(c not in "0123456789abcdef" for c in args.restore):
            parser.error("--restore requires a SHA-256 hash")
        if args.output is None or args.output.exists():
            parser.error("--output must name a new file")
        source = store.archive / "objects" / f"{args.restore}.gz"
        with gzip.open(source, "rb") as archived, args.output.open("xb") as target:
            shutil.copyfileobj(archived, target)
    if args.run:
        sys.argv = args.run
        script = Path(args.run[0]).resolve()
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    sys.modules.setdefault("artifact_retention", sys.modules[__name__])
    main()
