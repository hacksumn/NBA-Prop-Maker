"""DEC-092: atomic model writes and a training-aware production-bundle guard."""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import atomic_io
import run_daily

ROOT = Path(__file__).resolve().parents[1]
MODEL = "availability_minutes_model.pkl"
POLICY = "availability_minutes_policy.json"


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _leftovers(self):
        return [p.name for p in self.dir.iterdir() if p.suffix == ".tmp"]

    def test_replaces_content_and_returns_hash_without_temp_files(self):
        path = self.dir / "m.pkl"
        path.write_bytes(b"old")
        sha = atomic_io.atomic_write_bytes(path, b"new")
        self.assertEqual(path.read_bytes(), b"new")
        self.assertEqual(sha, atomic_io.sha256_bytes(b"new"))
        self.assertEqual(self._leftovers(), [])

    def test_failed_swap_keeps_original_and_removes_temp(self):
        path = self.dir / "m.pkl"
        path.write_bytes(b"old")
        with patch.object(atomic_io.os, "replace", side_effect=OSError("disk gone")):
            with self.assertRaises(OSError):
                atomic_io.atomic_write_bytes(path, b"new")
        self.assertEqual(path.read_bytes(), b"old")
        self.assertEqual(self._leftovers(), [])

    def test_transient_permission_error_is_retried(self):
        path = self.dir / "m.pkl"
        path.write_bytes(b"old")
        real = os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise PermissionError("held open by reader")
            return real(src, dst)

        with patch.object(atomic_io.os, "replace", side_effect=flaky), \
                patch.object(atomic_io.time, "sleep"):
            atomic_io.atomic_write_bytes(path, b"new")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(path.read_bytes(), b"new")

    def test_persistent_lock_times_out_loudly_and_keeps_original(self):
        path = self.dir / "m.pkl"
        path.write_bytes(b"old")
        with patch.object(atomic_io.os, "replace", side_effect=PermissionError("locked")), \
                patch.object(atomic_io.time, "sleep"):
            with self.assertRaisesRegex(PermissionError, "previous version is intact"):
                atomic_io.atomic_write_bytes(path, b"new", timeout=0.0)
        self.assertEqual(path.read_bytes(), b"old")
        self.assertEqual(self._leftovers(), [])

    def test_real_open_reader_delays_but_does_not_break_the_swap(self):
        # On Windows an open handle blocks os.replace; the retry must wait it out.
        path = self.dir / "m.pkl"
        path.write_bytes(b"old")
        handle = path.open("rb")
        closer = threading.Timer(0.4, handle.close)
        closer.start()
        try:
            atomic_io.atomic_write_bytes(path, b"new", timeout=10)
        finally:
            closer.join()
            handle.close()
        self.assertEqual(path.read_bytes(), b"new")

    def test_concurrent_reader_never_sees_a_partial_file(self):
        path = self.dir / "model.bin"
        a, b = b"A" * 2_000_000, b"B" * 2_000_000
        path.write_bytes(a)
        writer = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
            import sys; sys.path.insert(0, {str(ROOT)!r})
            from atomic_io import atomic_write_bytes
            for i in range(40):
                atomic_write_bytes({str(path)!r}, (b"A" if i % 2 else b"B") * 2_000_000)
        """)], stderr=subprocess.PIPE, text=True)
        bad, reads = [], 0
        while writer.poll() is None:
            try:
                data = path.read_bytes()
            except (PermissionError, FileNotFoundError):
                continue
            reads += 1
            if data not in (a, b):
                bad.append(len(data))
        _, err = writer.communicate(timeout=120)
        self.assertEqual(writer.returncode, 0, err)
        self.assertGreater(reads, 0)
        self.assertEqual(bad, [], "reader saw a partial or mixed file")


class TrainerInstallTests(unittest.TestCase):
    def test_install_writes_receipt_before_files_with_exact_hashes(self):
        from scripts.train_availability_minutes_model import install_model_artifacts

        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            model_bytes = pickle.dumps({"v": 1})
            policy_bytes = b'{"p": 1}\n'
            receipt = install_model_artifacts(
                models / MODEL, model_bytes, models / POLICY, policy_bytes,
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt.parent.name, atomic_io.RECEIPT_DIR_NAME)
            self.assertEqual(payload["artifacts"][MODEL], atomic_io.sha256_bytes(model_bytes))
            self.assertEqual(payload["artifacts"][POLICY], atomic_io.sha256_bytes(policy_bytes))
            self.assertEqual((models / MODEL).read_bytes(), model_bytes)
            self.assertEqual((models / POLICY).read_bytes(), policy_bytes)
            self.assertEqual([p for p in models.iterdir() if p.suffix == ".tmp"], [])


class TrainingAwareGuardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.models = Path(self._tmp.name)
        (self.models / MODEL).write_bytes(b"model-v1")
        (self.models / POLICY).write_bytes(b"policy-v1")
        (self.models / "clf_ast_advanced.pkl").write_bytes(b"prod")
        self._patches = [
            patch.object(run_daily, "MODEL_DIR", self.models),
            patch.object(run_daily, "_fail"),
            patch.object(run_daily, "_warn"),
        ]
        for p in self._patches:
            p.start()
        self.started = datetime.now(timezone.utc)
        self.before = run_daily._production_model_hashes(self.models)
        self.context = "run_daily_weekly_test_1"

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def _refresh(self, name, data, receipt_name=None, receipt_hash=None, context=""):
        env = {atomic_io.PIPELINE_CONTEXT_ENV: context}
        with patch.dict(os.environ, env):
            atomic_io.write_training_receipt(
                self.models, {receipt_name or name: receipt_hash or atomic_io.sha256_bytes(data)},
                writer="test",
            )
        atomic_io.atomic_write_bytes(self.models / name, data)

    def _check(self):
        errors = []
        ok = run_daily._record_production_immutability_failure(
            self.before, errors, self.started, self.context,
        )
        return ok, errors

    def test_unchanged_bundle_passes(self):
        self.assertEqual(self._check(), (True, []))

    def test_training_refresh_update_with_matching_receipt_is_expected(self):
        self._refresh(MODEL, b"model-v2")
        self._refresh(POLICY, b"policy-v2")
        self.assertEqual(self._check(), (True, []))
        run_daily._fail.assert_not_called()
        self.assertEqual(run_daily._warn.call_count, 2)

    def test_owned_file_change_without_receipt_still_fails(self):
        atomic_io.atomic_write_bytes(self.models / MODEL, b"model-v2")
        ok, errors = self._check()
        self.assertFalse(ok)
        self.assertIn(MODEL, errors[0])

    def test_receipt_from_this_runs_own_steps_does_not_excuse(self):
        self._refresh(MODEL, b"model-v2", context=self.context)
        ok, errors = self._check()
        self.assertFalse(ok)
        self.assertIn(MODEL, errors[0])

    def test_receipt_hash_must_match_current_file(self):
        self._refresh(MODEL, b"model-v2")
        atomic_io.atomic_write_bytes(self.models / MODEL, b"model-v3-unreceipted")
        self.assertFalse(self._check()[0])

    def test_receipt_cannot_excuse_a_non_owned_production_file(self):
        self._refresh("clf_ast_advanced.pkl", b"tampered")
        ok, errors = self._check()
        self.assertFalse(ok)
        self.assertIn("clf_ast_advanced.pkl", errors[0])

    def test_deleted_owned_file_fails(self):
        (self.models / MODEL).unlink()
        self.assertFalse(self._check()[0])

    def test_stale_receipt_outside_window_does_not_excuse(self):
        self._refresh(MODEL, b"model-v2")
        self.started = datetime.now(timezone.utc) + timedelta(hours=1)
        self.assertFalse(self._check()[0])

    def test_mixed_change_reports_only_the_unexplained_file(self):
        self._refresh(MODEL, b"model-v2")
        atomic_io.atomic_write_bytes(self.models / "clf_ast_advanced.pkl", b"changed")
        ok, errors = self._check()
        self.assertFalse(ok)
        self.assertIn("clf_ast_advanced.pkl", errors[0])
        self.assertNotIn(MODEL, errors[0])

    def test_legacy_call_without_window_keeps_strict_behavior(self):
        self._refresh(MODEL, b"model-v2")
        errors = []
        self.assertFalse(run_daily._record_production_immutability_failure(self.before, errors))

    def test_in_flight_temp_files_are_not_bundle_artifacts(self):
        (self.models / f".{MODEL}.abc123.tmp").write_bytes(b"partial")
        self.assertEqual(run_daily._production_model_hashes(self.models), self.before)

    def test_corrupt_receipt_is_ignored_not_trusted(self):
        receipts = self.models / atomic_io.RECEIPT_DIR_NAME
        receipts.mkdir()
        (receipts / "bad.json").write_text("{not json", encoding="utf-8")
        atomic_io.atomic_write_bytes(self.models / MODEL, b"model-v2")
        self.assertFalse(self._check()[0])


class WeeklyContextTests(unittest.TestCase):
    def test_child_steps_run_under_this_runs_context_and_env_is_restored(self):
        class Sunday:
            @classmethod
            def now(cls):
                return datetime(2026, 11, 15, 9, 20, 0)

        seen = {}

        def capture(args, errors, run_stamp, context):
            seen["env"] = os.environ.get(atomic_io.PIPELINE_CONTEXT_ENV)
            seen["context"] = context

        os.environ.pop(atomic_io.PIPELINE_CONTEXT_ENV, None)
        with patch.object(run_daily, "datetime", Sunday), \
                patch.object(run_daily, "_weekly_maintenance_steps", side_effect=capture):
            run_daily.step8_5_weekly_maintenance(SimpleNamespace(skip_weekly_retrain=False), [])
        self.assertTrue(seen["context"].startswith("run_daily_weekly_20261115_092000_"))
        self.assertEqual(seen["env"], seen["context"])
        self.assertNotIn(atomic_io.PIPELINE_CONTEXT_ENV, os.environ)


if __name__ == "__main__":
    unittest.main()
