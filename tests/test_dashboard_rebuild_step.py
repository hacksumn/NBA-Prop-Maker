"""The daily dashboard rebuild hook in scripts/project_operator.ps1 (DEC-088).

The hook runs inside the live scheduled job, so the guarantees that matter are negative ones: it must never
block picks, never throw, never mask a real failure and never hang the run. The real DailySafe action cannot be
run from a test (paid APIs), so these tests (1) pin the structure of the script and (2) run the REAL
Invoke-DashboardRebuild function, extracted from the script, against stand-in builders that succeed, fail, hang
or are missing.
"""
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "project_operator.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _function_text() -> str:
    """Invoke-DashboardRebuild exactly as it appears in the script."""
    return _block("Invoke-DashboardRebuild")      # just this function: Invoke-DashboardPublish follows it (DEC-100)


def _block(name: str) -> str:
    """The text of one function or switch clause in the script."""
    text = _script_text()
    if name.startswith("'"):
        start = text.index(f"    {name} {{")
        end = re.search(r"\n    '[A-Za-z]+' \{", text[start + 10:])
        return text[start:start + 10 + end.start()] if end else text[start:]
    start = text.index(f"function {name}")
    return text[start:start + text[start:].index("\n}\n") + 3]      # up to the function's closing brace


class StructureTests(unittest.TestCase):
    def test_daily_safe_rebuilds_in_a_finally_after_both_existing_steps(self):
        daily = _block("'DailySafe'")
        self.assertIn("Show-Preflight", daily)
        try_part, finally_part = daily.split("finally", 1)
        self.assertIn("try {", try_part)
        self.assertLess(try_part.index("Safe daily pipeline"), try_part.index("capture_candidate_future_holdout.py"))
        self.assertIn("Invoke-DashboardRebuild", finally_part)
        self.assertNotIn("Invoke-DashboardRebuild", try_part)   # in the finally only: runs even when a step failed

    def test_dashboard_can_never_block_preflight_or_required_files(self):
        for name in ("Assert-RequiredFiles", "Show-Preflight"):
            self.assertNotIn("dashboard", _block(name).lower(), name)

    def test_rebuild_function_cannot_throw_and_bypasses_the_retention_wrapper(self):
        fn = _function_text()
        self.assertNotRegex(fn, r"\bthrow\b")                        # failures are logged, never raised
        self.assertNotIn("Invoke-PythonStep", fn)                     # would keep every replaced page forever
        self.assertNotIn("Invoke-ProtectedPythonStep", fn)
        self.assertIn("catch {", fn)
        self.assertIn("WaitForExit(", fn)                             # time-limited
        self.assertIn("NOT refreshed", fn)                            # says so plainly when it did not run

    def test_script_still_parses(self):
        if not POWERSHELL:
            self.skipTest("PowerShell not available")
        cmd = ("$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
               f"'{SCRIPT}',[ref]$t,[ref]$e);if($e.Count){{$e|ForEach-Object{{$_.Message}};exit 1}}")
        done = subprocess.run([POWERSHELL, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


@unittest.skipUnless(POWERSHELL, "PowerShell not available")
class RebuildBehaviourTests(unittest.TestCase):
    """Run the real function against a stand-in scripts/build_dashboard.py."""

    def _run(self, builder_source, call="Invoke-DashboardRebuild -TimeoutSeconds 60", wrapper=None, timeout=90):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            if builder_source is not None:
                (root / "scripts").mkdir()
                (root / "scripts" / "build_dashboard.py").write_text(builder_source, encoding="utf-8")
            body = wrapper.format(call=call) if wrapper else call
            harness = root / "harness.ps1"
            harness.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                f"$ProjectRoot = '{root}'\n{_function_text()}\n{body}\nWrite-Host 'HARNESS-FINISHED'\n",
                encoding="utf-8")
            started = time.time()
            done = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
                                  capture_output=True, text=True, errors="replace", timeout=timeout)
            return done.returncode, done.stdout + done.stderr, time.time() - started

    def test_success_relays_the_builder_output_and_says_ok(self):
        code, out, _ = self._run('print("[dashboard] wrote stub page (1 KB); warnings=0")\n')
        self.assertEqual(code, 0, out)
        self.assertIn("[Dashboard rebuild] [dashboard] wrote stub page", out)
        self.assertIn("[Dashboard rebuild] OK", out)

    def test_builder_failure_is_logged_but_never_raised(self):
        code, out, _ = self._run('import sys\nprint("partial output")\nsys.stderr.write("boom reason\\n")\nsys.exit(3)\n')
        self.assertEqual(code, 0, out)                    # the harness (and so the scheduled job) is unaffected
        self.assertIn("FAILED with exit code 3", out)
        self.assertIn("boom reason", out)                 # the builder's own explanation reaches the log
        self.assertIn("HARNESS-FINISHED", out)

    def test_a_hung_builder_is_stopped_and_the_run_continues(self):
        code, out, elapsed = self._run("import time\ntime.sleep(120)\n", call="Invoke-DashboardRebuild -TimeoutSeconds 3")
        self.assertEqual(code, 0, out)
        self.assertIn("no result after 3 s", out)
        self.assertIn("NOT refreshed", out)
        self.assertIn("HARNESS-FINISHED", out)
        self.assertLess(elapsed, 60)                      # it did not wait for the 120 s sleep

    def test_missing_builder_is_skipped_loudly(self):
        code, out, _ = self._run(None)
        self.assertEqual(code, 0, out)
        self.assertIn("SKIPPED", out)
        self.assertIn("NOT refreshed", out)

    def test_pipeline_error_survives_the_finally_and_the_page_is_still_rebuilt(self):
        wrapper = ("try {{\n  try {{ throw 'PIPELINE-BOOM' }}\n  finally {{ {call} }}\n}}\n"
                   "catch {{ Write-Host \"CAUGHT: $($_.Exception.Message)\" }}")
        code, out, _ = self._run('print("[dashboard] wrote stub page")\n', wrapper=wrapper)
        self.assertEqual(code, 0, out)
        self.assertIn("[Dashboard rebuild] OK", out)      # the rebuild ran although the pipeline step failed
        self.assertIn("CAUGHT: PIPELINE-BOOM", out)       # and the original error was not masked

    def test_failed_rebuild_cannot_turn_a_passing_run_into_a_failure(self):
        wrapper = "try {{\n  try {{ Write-Host 'PIPELINE-OK' }}\n  finally {{ {call} }}\n}}\ncatch {{ Write-Host 'WRONGLY-FAILED' }}"
        code, out, _ = self._run("import sys\nsys.exit(9)\n", wrapper=wrapper)
        self.assertEqual(code, 0, out)
        self.assertIn("PIPELINE-OK", out)
        self.assertIn("FAILED with exit code 9", out)
        self.assertNotIn("WRONGLY-FAILED", out)


if __name__ == "__main__":
    unittest.main()
