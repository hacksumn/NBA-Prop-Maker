import tempfile
import unittest
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_daily


class DailyRuntimeContractTests(unittest.TestCase):
    def test_daily_exact_team_log_refresh_uses_active_season(self):
        errors = []
        args = SimpleNamespace(seasons=["2026-27"])
        with patch.object(run_daily, "_run_python_step", return_value=True) as runner:
            self.assertTrue(run_daily.step1_1_team_game_logs(args, errors))
        self.assertEqual(
            runner.call_args.args[2],
            ["scripts/fetch_nba_team_game_logs.py", "--seasons", "2026-27"],
        )

    def test_failed_live_line_snapshot_blocks_pick_generation(self):
        args = SimpleNamespace(skip_picks=False)
        errors = []
        try:
            with patch.dict(
                "nba_props.CONFIG",
                {"live_line_source_mode": "fanduel", "live_sportsbook_key": "fanduel"},
            ), patch(
                "nba_props.fetch_vegas_lines",
                return_value=([], "odds_api"),
            ):
                run_daily.step2_prizepicks(args, errors)
            self.assertTrue(run_daily._RUN_STATE["live_lines_step_failed"])
            self.assertEqual(len(errors), 1)
            with patch.object(run_daily, "_run_python_step") as runner:
                run_daily.step9_generate_picks(args, errors)
            runner.assert_not_called()
            self.assertEqual(len(errors), 1)
        finally:
            run_daily._RUN_STATE["live_lines_step_failed"] = False

    def test_project_operator_verifies_model_hashes_in_finally(self):
        script = (Path(run_daily.__file__).parent / "scripts" / "project_operator.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("function Invoke-ProtectedPythonStep", script)
        protected = script.split("function Invoke-ProtectedPythonStep", 1)[1].split(
            "function Assert-RequiredFiles", 1
        )[0]
        self.assertIn("finally", protected)
        self.assertIn("Assert-ProductionUnchanged", protected)

    def test_project_operator_refreshes_persisted_odds_key_for_old_terminals(self):
        script = (Path(run_daily.__file__).parent / "scripts" / "project_operator.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "GetEnvironmentVariable('ODDS_API_KEY', 'User')",
            script,
        )
        self.assertIn("$env:ODDS_API_KEY = $PersistedOddsApiKey", script)

    def test_active_season_rolls_at_october_boundary(self):
        self.assertEqual(run_daily._active_season(as_of="2026-09-30"), "2025-26")
        self.assertEqual(run_daily._active_season(as_of="2026-10-01"), "2026-27")

    def test_active_season_uses_latest_explicit_override(self):
        args = SimpleNamespace(seasons=["2024-25", "2026-27", "2025-26"])
        self.assertEqual(run_daily._active_season(args), "2026-27")

    def test_active_season_rejects_malformed_override(self):
        with self.assertRaisesRegex(ValueError, "Invalid NBA season"):
            run_daily._active_season(SimpleNamespace(seasons=["2026-99"]))

    def test_usage_profile_empty_build_cannot_reuse_header_only_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "player_profiles.csv"
            cache.write_text("player_id,player_name,team_abbr\n", encoding="utf-8")
            errors = []
            with patch.object(run_daily, "PLAYER_PROFILES", cache), patch.object(
                run_daily, "_retry_step_call", return_value=[]
            ):
                run_daily.step6_usage_injury(errors, season="2026-27")
            self.assertTrue(errors)
            self.assertEqual(cache.read_text(encoding="utf-8"), "player_id,player_name,team_abbr\n")

    def test_empty_blowout_build_preserves_existing_cache(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "game_blowout_index.csv"
            cache.write_text("game_id,game_date\nold,2026-10-20\n", encoding="utf-8")
            snapshot_dir = Path(tmp) / "season_snapshots"
            errors = []
            with patch.object(run_daily, "BLOWOUT_INDEX", cache), patch.object(
                run_daily, "LAYER_SNAPSHOT_DIR", snapshot_dir
            ), patch(
                "blowout_discount_model.build_game_blowout_index",
                return_value=pd.DataFrame(),
            ):
                run_daily.step3_blowout_index(SimpleNamespace(seasons=["2026-27"]), errors)
            self.assertTrue(errors)
            self.assertEqual(cache.read_text(encoding="utf-8"), "game_id,game_date\nold,2026-10-20\n")
            self.assertFalse(snapshot_dir.exists())

    def test_closing_watcher_is_prizepicks_only_by_default(self):
        args = SimpleNamespace(skip_pp_closing_capture=False, pair_sportsbook_t60=False)
        errors = []
        with self.subTest("default-spend-contract"), patch.object(
            run_daily, "LOG_DIR", Path(self._testMethodName)
        ):
            run_daily.LOG_DIR.mkdir(exist_ok=True)
            try:
                with patch.dict(
                    "nba_props.CONFIG", {"live_line_source_mode": "prizepicks"}
                ), patch("run_daily.subprocess.Popen") as popen:
                    run_daily.step2_5_start_prizepicks_closing_capture(args, errors)
                command = popen.call_args[0][0]
                self.assertNotIn("--pair-sportsbook-t60", command)
                self.assertNotIn("--odds-credit-budget", command)
                self.assertEqual(errors, [])
            finally:
                log_path = run_daily.LOG_DIR / "prizepicks_snapshot_watcher.log"
                if log_path.exists():
                    log_path.unlink()
                run_daily.LOG_DIR.rmdir()

    def test_fanduel_mode_never_starts_prizepicks_watcher(self):
        args = SimpleNamespace(skip_pp_closing_capture=False, pair_sportsbook_t60=False)
        errors = []
        with patch.dict(
            "nba_props.CONFIG", {"live_line_source_mode": "fanduel"}
        ), patch("run_daily.subprocess.Popen") as popen:
            run_daily.step2_5_start_prizepicks_closing_capture(args, errors)
        popen.assert_not_called()
        self.assertEqual(errors, [])

    def test_paid_pairing_requires_explicit_flag_and_config_gate(self):
        args = SimpleNamespace(skip_pp_closing_capture=False, pair_sportsbook_t60=True)
        errors = []
        temp_dir = Path(self._testMethodName)
        temp_dir.mkdir(exist_ok=True)
        try:
            with patch.dict(
                "nba_props.CONFIG", {"live_line_source_mode": "prizepicks"}
            ), patch.object(run_daily, "LOG_DIR", temp_dir), patch(
                "run_daily.subprocess.Popen"
            ) as popen, patch.dict(
                "nba_props.CONFIG",
                {"allow_live_odds_api_spend": True, "odds_api_credit_reserve": 321},
            ):
                run_daily.step2_5_start_prizepicks_closing_capture(args, errors)
            command = popen.call_args[0][0]
            self.assertIn("--pair-sportsbook-t60", command)
            self.assertEqual(command[command.index("--odds-credit-budget") + 1], "321")
        finally:
            log_path = temp_dir / "prizepicks_snapshot_watcher.log"
            if log_path.exists():
                log_path.unlink()
            temp_dir.rmdir()

    def test_weekly_maintenance_stops_before_training_when_feature_build_fails(self):
        class Sunday:
            @classmethod
            def now(cls):
                return real_datetime(2026, 10, 25, 12, 0, 0)

        args = SimpleNamespace(skip_weekly_retrain=False, seasons=["2026-27"])
        errors = []
        with patch.object(run_daily, "datetime", Sunday), patch.object(
            run_daily, "_production_model_hashes", return_value={"bundle": "same"}
        ), patch.object(
            run_daily, "_record_production_immutability_failure", return_value=True
        ), patch.object(
            run_daily, "_run_python_step", side_effect=[True, False]
        ) as runner:
            run_daily.step8_5_weekly_maintenance(args, errors)

        self.assertEqual(runner.call_count, 2)
        commands = [call.args[2] for call in runner.call_args_list]
        self.assertEqual(commands[0][0], "scripts/fetch_nba_team_game_logs.py")
        self.assertEqual(commands[1], ["feature_pipeline.py"])

    def test_weekly_maintenance_trains_only_an_isolated_defense_candidate(self):
        class Sunday:
            @classmethod
            def now(cls):
                return real_datetime(2026, 10, 25, 12, 0, 0)

        args = SimpleNamespace(skip_weekly_retrain=False, seasons=["2026-27"])
        errors = []
        with patch.object(run_daily, "datetime", Sunday), patch.object(
            run_daily, "_production_model_hashes", return_value={"bundle": "same"}
        ), patch.object(
            run_daily, "_record_production_immutability_failure", return_value=True
        ), patch.object(
            run_daily, "_run_python_step", return_value=True
        ) as runner:
            run_daily.step8_5_weekly_maintenance(args, errors)

        self.assertEqual(runner.call_count, 6)
        commands = [call.args[2] for call in runner.call_args_list]
        self.assertIn(["scripts/build_pointintime_team_defense.py"], commands)
        self.assertIn(["scripts/build_prop_defense_rating_training_data.py"], commands)
        train_command = next(command for command in commands if command[0] == "train_advanced_models.py")
        self.assertIn("--require-defense-rating-features", train_command)
        self.assertIn("--model-dir", train_command)
        self.assertNotEqual(
            Path(train_command[train_command.index("--model-dir") + 1]).resolve(),
            run_daily.MODEL_DIR.resolve(),
        )
        self.assertFalse(any(command[0] == "calibrate_confidence.py" for command in commands))


if __name__ == "__main__":
    unittest.main()
