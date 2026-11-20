"""The daily dashboard upload hook in scripts/project_operator.ps1 (DEC-100).

It runs inside the live scheduled job and publishes a PUBLIC page, so the guarantees that matter are negative
ones: it must never block picks, never throw, never hang the run, never upload anything but the freshly built
page, never upload a page that was not rebuilt in this run, and never run unless explicitly enabled. Real uploads
cannot be made from a test, so these tests (1) pin the structure of the script and (2) run the REAL
Invoke-DashboardPublish function, extracted from the script, against a stand-in `npx.cmd` that records what it
was asked to do and succeeds, fails, hangs or is missing.
"""
import json
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "project_operator.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")

GOOD_CONFIG = {"enabled": True, "project": "my-test-project", "branch": "main"}
PAGE = b"<!doctype html><title>stand-in page</title><body>PAGE-BYTES-\xe2\x9c\x93</body>"

# A stand-in for npx.cmd: records the arguments, working directory and staged files, then behaves as told.
STAND_IN = r"""@echo off
> "%STANDIN_OUT%\args.txt" echo %*
> "%STANDIN_OUT%\cwd.txt" cd
dir /b > "%STANDIN_OUT%\files.txt"
copy /y index.html "%STANDIN_OUT%\index_copy.html" >nul
copy /y _headers "%STANDIN_OUT%\headers_copy.txt" >nul
> "%STANDIN_OUT%\env.txt" echo CI=%CI% METRICS=%WRANGLER_SEND_METRICS% XDG=%XDG_CONFIG_HOME%
if "%STANDIN_MODE%"=="fail" (
  echo deploy exploded 1>&2
  exit /b 3
)
if "%STANDIN_MODE%"=="fancy" (
  echo ESC[31mX [ERROR]ESC[0m shown ACCENTS
  exit /b 0
)
if "%STANDIN_MODE%"=="hang" (
  ping -n 60 127.0.0.1 >nul
  exit /b 0
)
echo Deployment complete! https://abc123.example.pages.dev
exit /b 0
"""


def _user_token_set() -> bool:
    """True when a Cloudflare API token is stored in the user's environment (the function reads it at user scope)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return bool(winreg.QueryValueEx(key, "CLOUDFLARE_API_TOKEN")[0])
    except (ImportError, OSError):
        return False


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _function_text() -> str:
    """Invoke-DashboardPublish exactly as it appears in the script."""
    body = _script_text().split("function Invoke-DashboardPublish", 1)[1].split("\nswitch ($Action)", 1)[0]
    return "function Invoke-DashboardPublish" + body


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
    def test_publish_runs_after_the_rebuild_inside_the_finally_only(self):
        daily = _block("'DailySafe'")
        try_part, finally_part = daily.split("finally", 1)
        self.assertNotIn("Invoke-DashboardPublish", try_part)         # a failed step must not skip it, nor it any step
        self.assertIn("Invoke-DashboardPublish", finally_part)
        self.assertLess(finally_part.index("Invoke-DashboardRebuild"), finally_part.index("Invoke-DashboardPublish"))

    def test_publish_can_never_block_preflight_or_required_files(self):
        for name in ("Assert-RequiredFiles", "Show-Preflight"):
            self.assertNotRegex(_block(name).lower(), r"publish|cloudflare|wrangler", name)

    def test_publish_function_cannot_throw_and_bypasses_the_retention_wrapper(self):
        fn = _function_text()
        self.assertNotRegex(fn, r"\bthrow\b")
        self.assertNotIn("Invoke-PythonStep", fn)
        self.assertNotIn("Invoke-ProtectedPythonStep", fn)
        self.assertIn("catch {", fn)
        self.assertIn("WaitForExit(", fn)                             # time-limited
        self.assertIn("taskkill.exe", fn)                             # kills the whole npx -> node process tree
        self.assertIn("NOT updated", fn)                              # says so plainly when it did not run

    def test_it_uploads_only_a_throwaway_copy_with_a_pinned_tool_version(self):
        fn = _function_text()
        self.assertRegex(fn, r"wrangler@\d+\.\d+\.\d+")               # exact version: no surprise upgrades overnight
        self.assertNotRegex(fn, r"wrangler@\d+['\"]")                 # never a floating major
        self.assertIn("'deploy', '.'", fn)                            # deploys the staging folder, nothing else
        self.assertIn("-WorkingDirectory $stageDir", fn)              # never from the project folder / its git history
        self.assertIn("dash_site_", fn)
        self.assertIn("Remove-Item -LiteralPath $stageDir -Recurse", fn)
        self.assertIn("noindex", fn)
        self.assertIn("Disallow: /", fn)

    def test_settings_file_with_the_public_project_name_is_never_tracked(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("scripts/dashboard_publish.local.json", ignore)
        example = json.loads((ROOT / "scripts" / "dashboard_publish.example.json").read_text(encoding="utf-8"))
        self.assertIs(example["enabled"], False)                      # the tracked template does nothing by itself
        self.assertEqual(set(example), {"enabled", "project", "branch", "wranglerConfigHome"})
        self.assertIn("dashboard_publish.local.json", _function_text())   # the function reads the ignored file

    def test_login_script_matches_the_pinned_tool_version_and_stays_out_of_appdata(self):
        bat = (ROOT / "scripts" / "cloudflare_login.bat").read_text(encoding="utf-8")
        in_script = re.search(r"wrangler@(\d+\.\d+\.\d+)", _function_text()).group(1)
        self.assertIn(f"wrangler@{in_script}", bat)                   # one version everywhere
        self.assertIn(r"XDG_CONFIG_HOME=%USERPROFILE%\.fresh_start_nba\cloudflare", bat)   # outside AppData
        self.assertNotIn("APPDATA", bat.replace("AppData", "").replace("REM", ""))  # never stores the login under AppData
        for scope in ("account:read", "user:read", "pages:write"):     # only what a Pages upload needs
            self.assertIn(scope, bat)
        self.assertNotIn("workers:write", bat)

    def test_pages_upload_never_runs_without_the_login_check(self):
        fn = _function_text()
        self.assertIn("wranglerConfigHome", fn)
        self.assertLess(fn.index("no Cloudflare login found"), fn.index("Start-Process"))   # checked before launching

    def test_script_still_parses(self):
        if not POWERSHELL:
            self.skipTest("PowerShell not available")
        cmd = ("$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
               f"'{SCRIPT}',[ref]$t,[ref]$e);if($e.Count){{$e|ForEach-Object{{$_.Message}};exit 1}}")
        done = subprocess.run([POWERSHELL, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


@unittest.skipUnless(POWERSHELL, "PowerShell not available")
class PublishBehaviourTests(unittest.TestCase):
    """Run the real function against a stand-in npx.cmd in a synthetic project folder."""

    def _run(self, config=GOOD_CONFIG, page=True, mode="ok", age_minutes=0, timeout_s=60, no_npx=False, wall=120,
             appdata_login=True, home_login=False, make_home_dir=True):
        """appdata_login: a wrangler login exists under the (test) %APPDATA%. home_login: one exists in the folder named
        by wranglerConfigHome. A config value of "{HOME}" is replaced by that test folder's real path."""
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root, out, tmp, binp = base / "proj", base / "out", base / "tmp", base / "bin"
            appdata, cfhome = base / "appdata", base / "cfhome"
            for d in (root / "scripts", root / "output", out, tmp, binp, appdata):
                d.mkdir(parents=True)
            if make_home_dir:
                cfhome.mkdir()
            if appdata_login:
                (appdata / "xdg.config" / ".wrangler" / "config").mkdir(parents=True)
                (appdata / "xdg.config" / ".wrangler" / "config" / "default.toml").write_text("oauth_token = \"x\"\n", encoding="ascii")
            if home_login:
                (cfhome / ".wrangler" / "config").mkdir(parents=True)
                (cfhome / ".wrangler" / "config" / "default.toml").write_text("oauth_token = \"x\"\n", encoding="ascii")
            (binp / "npx.cmd").write_bytes(STAND_IN.replace("ESC", "\x1b").replace("ACCENTS", "â\u0095\u0090").encode("latin-1"))
            if config is not None:
                if isinstance(config, str):
                    text = config
                else:
                    text = json.dumps({k: (str(cfhome) if v == "{HOME}" else v) for k, v in config.items()})
                (root / "scripts" / "dashboard_publish.local.json").write_text(text, encoding="utf-8")
            if page:
                target = root / "output" / "dashboard_pro.html"
                target.write_bytes(PAGE)
                stamp = time.time() - age_minutes * 60
                import os
                os.utime(target, (stamp, stamp))
            path_line = ("$env:PATH = \"$env:SystemRoot\\System32\\WindowsPowerShell\\v1.0;$env:SystemRoot\\System32\""
                         if no_npx else f"$env:PATH = '{binp};' + $env:PATH")
            harness = base / "harness.ps1"
            harness.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                f"$ProjectRoot = '{root}'\n$env:TMP = '{tmp}'\n$env:TEMP = '{tmp}'\n$env:APPDATA = '{appdata}'\n"
                f"$env:STANDIN_OUT = '{out}'\n$env:STANDIN_MODE = '{mode}'\n{path_line}\n"
                "$env:CI = $null; $env:WRANGLER_SEND_METRICS = $null; $env:XDG_CONFIG_HOME = $null; $env:CLOUDFLARE_API_TOKEN = $null\n"
                f"{_function_text()}\nInvoke-DashboardPublish -TimeoutSeconds {timeout_s}\n"
                "Write-Host \"AFTER CI=[$env:CI] METRICS=[$env:WRANGLER_SEND_METRICS] XDG=[$env:XDG_CONFIG_HOME]\"\nWrite-Host 'HARNESS-FINISHED'\n",
                encoding="utf-8")
            started = time.time()
            done = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
                                  capture_output=True, text=True, errors="replace", timeout=wall)
            read = lambda name: (out / name).read_text(encoding="utf-8", errors="replace") if (out / name).exists() else None
            return {
                "rc": done.returncode, "log": done.stdout + done.stderr, "seconds": time.time() - started,
                "called": (out / "args.txt").exists(), "args": read("args.txt"), "cwd": (read("cwd.txt") or "").strip(),
                "files": sorted((read("files.txt") or "").split()), "headers": read("headers_copy.txt"),
                "index": (out / "index_copy.html").read_bytes() if (out / "index_copy.html").exists() else None,
                "env": (read("env.txt") or "").strip(), "root": str(root), "left_in_tmp": sorted(p.name for p in tmp.iterdir()),
                "home": str(cfhome), "appdata": str(appdata),
            }

    def _assert_finished_quietly(self, r):
        self.assertEqual(r["rc"], 0, r["log"])
        self.assertIn("HARNESS-FINISHED", r["log"])                   # nothing escaped the function
        self.assertEqual(r["left_in_tmp"], [], "staging folder or capture files were left behind")

    def test_success_uploads_only_the_page_and_noindex_files_from_a_throwaway_folder(self):
        r = self._run()
        self._assert_finished_quietly(r)
        self.assertIn("[Dashboard publish] OK", r["log"])
        self.assertIn("Deployment complete", r["log"])                # the tool's output is relayed
        for part in ("--yes", "pages deploy .", "--project-name my-test-project", "--branch main"):
            self.assertIn(part, r["args"])
        self.assertRegex(r["args"], r"wrangler@\d+\.\d+\.\d+")
        self.assertEqual(r["files"], ["_headers", "index.html", "robots.txt"])
        self.assertEqual(r["index"], PAGE)                            # the page, byte for byte
        self.assertIn("X-Robots-Tag: noindex", r["headers"])
        self.assertNotEqual(Path(r["cwd"]).resolve(), Path(r["root"]).resolve())   # not run from the project folder
        self.assertNotIn(r["root"].lower(), r["cwd"].lower())
        self.assertEqual(r["env"], "CI=true METRICS=false XDG=")      # non-interactive, no telemetry, no login-folder override...
        self.assertIn("AFTER CI=[] METRICS=[] XDG=[]", r["log"])      # ...and restored afterwards

    def test_login_in_a_folder_outside_appdata_is_used_and_handed_to_wrangler(self):
        # The fix for DEC-102: the scheduled task cannot see a login made inside the Claude desktop app (AppData is
        # redirected there), so the settings file can name a folder outside AppData and wrangler is pointed at it.
        r = self._run(config={**GOOD_CONFIG, "wranglerConfigHome": "{HOME}"}, appdata_login=False, home_login=True)
        self._assert_finished_quietly(r)
        self.assertIn("[Dashboard publish] OK", r["log"])
        self.assertEqual(r["env"], f"CI=true METRICS=false XDG={r['home']}")
        self.assertIn("AFTER CI=[] METRICS=[] XDG=[]", r["log"])      # restored for the rest of the run

    def test_a_login_that_exists_only_under_appdata_is_not_trusted_when_a_login_folder_is_configured(self):
        # This is the exact failure of 2026-11-19: a login existed, but only where the scheduled task cannot see it.
        r = self._run(config={**GOOD_CONFIG, "wranglerConfigHome": "{HOME}"}, appdata_login=True, home_login=False)
        self._assert_finished_quietly(r)
        self.assertFalse(r["called"])
        self.assertIn("no Cloudflare login found at", r["log"])
        self.assertIn("cfhome", r["log"])
        self.assertIn("NOT updated", r["log"])

    def test_missing_login_is_reported_plainly_and_nothing_is_uploaded(self):
        import os
        if os.environ.get("CLOUDFLARE_API_TOKEN") or _user_token_set():
            self.skipTest("a Cloudflare API token is set on this machine, which legitimately replaces the login file")
        r = self._run(appdata_login=False)
        self._assert_finished_quietly(r)
        self.assertFalse(r["called"])
        self.assertIn("no Cloudflare login found at", r["log"])
        self.assertIn("cloudflare_login.bat", r["log"])
        self.assertIn("NOT updated", r["log"])

    def test_bad_login_folder_setting_fails_safe(self):
        for bad, kwargs in (("relative\\folder", {}), ("{HOME}", {"make_home_dir": False})):
            with self.subTest(bad=bad):
                r = self._run(config={**GOOD_CONFIG, "wranglerConfigHome": bad}, **kwargs)
                self._assert_finished_quietly(r)
                self.assertFalse(r["called"])
                self.assertIn("not an existing absolute folder", r["log"])

    def test_tool_output_is_cleaned_of_colour_codes_and_box_characters(self):
        r = self._run(mode="fancy")
        self._assert_finished_quietly(r)
        self.assertIn("[Dashboard publish] X [ERROR] shown", r["log"])
        self.assertNotIn("\x1b", r["log"])
        relayed = [l for l in r["log"].splitlines() if l.startswith("[Dashboard publish]")]
        self.assertTrue(relayed)
        for line in relayed:
            self.assertTrue(all(32 <= ord(c) < 127 for c in line), repr(line))

    def test_disabled_does_nothing_and_says_why(self):
        r = self._run(config={**GOOD_CONFIG, "enabled": False})
        self._assert_finished_quietly(r)
        self.assertFalse(r["called"])
        self.assertIn("SKIPPED: not enabled", r["log"])
        self.assertIn("NOT updated", r["log"])

    def test_only_a_real_json_true_enables_it(self):
        for value in ("true", "yes", 1, None):
            with self.subTest(enabled=value):
                r = self._run(config={**GOOD_CONFIG, "enabled": value})
                self._assert_finished_quietly(r)
                self.assertFalse(r["called"])
                self.assertIn("SKIPPED", r["log"])

    def test_missing_settings_file_is_skipped(self):
        r = self._run(config=None)
        self._assert_finished_quietly(r)
        self.assertFalse(r["called"])
        self.assertIn("SKIPPED", r["log"])

    def test_unreadable_settings_fail_safe(self):
        r = self._run(config="{ this is not json")
        self._assert_finished_quietly(r)
        self.assertFalse(r["called"])
        self.assertIn("FAILED", r["log"])

    def test_unsafe_project_or_branch_names_are_refused_before_anything_runs(self):
        for bad in ({"project": "Bad Name"}, {"project": "x; calc"}, {"project": ""}, {"project": "a" * 80},
                    {"branch": "main --force"}, {"branch": "a&b"}):
            with self.subTest(bad=bad):
                r = self._run(config={**GOOD_CONFIG, **bad})
                self._assert_finished_quietly(r)
                self.assertFalse(r["called"])
                self.assertIn("invalid project or branch", r["log"])

    def test_missing_page_is_skipped(self):
        r = self._run(page=False)
        self._assert_finished_quietly(r)
        self.assertFalse(r["called"])
        self.assertIn("SKIPPED", r["log"])

    def test_a_page_not_rebuilt_in_this_run_is_never_uploaded(self):
        r = self._run(age_minutes=45)
        self._assert_finished_quietly(r)
        self.assertFalse(r["called"])
        self.assertIn("was not rebuilt in this run", r["log"])
        self.assertIn("NOT updated", r["log"])

    def test_upload_failure_is_logged_and_never_raised(self):
        r = self._run(mode="fail")
        self._assert_finished_quietly(r)
        self.assertIn("FAILED with exit code 3", r["log"])
        self.assertIn("deploy exploded", r["log"])                    # the real reason is shown
        self.assertIn("NOT updated", r["log"])
        self.assertNotIn("[Dashboard publish] OK", r["log"])

    def test_a_hung_upload_is_stopped_and_the_run_continues(self):
        r = self._run(mode="hang", timeout_s=3)
        self._assert_finished_quietly(r)
        self.assertIn("no result after 3 s", r["log"])
        self.assertLess(r["seconds"], 60)

    def test_missing_node_tooling_is_a_logged_failure_not_a_crash(self):
        r = self._run(no_npx=True)
        self._assert_finished_quietly(r)
        self.assertFalse(r["called"])
        self.assertIn("FAILED", r["log"])
        self.assertIn("NOT updated", r["log"])


if __name__ == "__main__":
    unittest.main()
