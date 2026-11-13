import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import nba_props as nba
import gate_decisions as decisions
from scripts import evaluate_gate_experiment as experiment
from scripts import reconcile_grade_evidence as reconciliation


class GateDecisionTests(unittest.TestCase):
    def test_disabled_auxiliary_refresh_does_not_change_production_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            (models / "xgb_ast_advanced.pkl").write_bytes(b"production")
            auxiliary = models / "availability_minutes_model.pkl"
            auxiliary.write_bytes(b"first")
            policy = models / "availability_minutes_policy.json"
            policy.write_text(json.dumps({"enabled_for_live": False}))
            first = decisions.start_run({"models_dir": models}, {})
            auxiliary.write_bytes(b"refreshed")
            second = decisions.start_run({"models_dir": models}, {})
            self.assertEqual(first["model_sha256"], second["model_sha256"])
            self.assertIn("xgb_ast_advanced.pkl", first["model_sha256"])
            policy.write_text(json.dumps({"enabled_for_live": True}))
            enabled = decisions.start_run({"models_dir": models}, {})
            self.assertIn(auxiliary.name, enabled["model_sha256"])

    def test_reconciliation_preserves_original_grade_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output").mkdir()
            (root / "data").mkdir()
            row = {"game_date": "2026-11-10", "player": "Test", "prop": "AST",
                   "direction": "OVER", "line": 4, "actual": 4, "result": "LOSS"}
            paths = [root / "output" / name for name in ["picks_history.csv", "candidates_history.csv"]]
            for path in paths:
                pd.DataFrame([row]).to_csv(path, index=False)
            original = [p.read_bytes() for p in paths]
            pd.DataFrame([{"game_date": "2026-11-10", "player": "Test", "pts": 10,
                           "trb": 2, "ast": 4}]).to_csv(root / "data/nba_data.csv", index=False)
            with patch.object(reconciliation, "ROOT", root):
                first = reconciliation.reconcile()
                second = reconciliation.reconcile()
            self.assertEqual(first, second)
            self.assertEqual(first["outcome_conflicts"], 2)
            self.assertEqual(first["conflicts"][0]["observed_result"], "PUSH")
            self.assertEqual([p.read_bytes() for p in paths], original)
            self.assertEqual(len(list((root / "output/grade_reconciliation").glob("*.json"))), 2)

    def test_generation_trace_preserves_selection_and_records_disabled_lanes(self):
        policy = {"ast": {"allowed": True, "allow_under": True, "allow_over": False,
                           "min_prob": .5, "min_edge": .05, "min_edge_under": .05,
                           "rmse": 1.8, "shrunk_acc": .6, "meta_gate_mode": "off"},
                  "pts": {"allowed": False}}
        board = pd.DataFrame([{"player": "Test", "team": "DET", "ast_line": 4.5,
                               "ast_pred": 5.5, "pts_line": 20.5, "pts_pred": 18}])
        events = []
        original = nba.filter_best_picks(board, quality_policy=policy)
        traced = nba.filter_best_picks(board, quality_policy=policy, rejection_events=events)
        pd.testing.assert_frame_equal(original, traced)
        self.assertIn("OVER direction disabled", [e["reason"] for e in events])
        self.assertIn("prop disabled by policy or configuration", [e["reason"] for e in events])

    def test_sanity_trace_retains_removed_candidate_and_reason(self):
        picks = pd.DataFrame([{"player": "Test", "team": "DET", "prop": "AST",
                               "direction": "UNDER", "line": 4.5, "prediction": 3.5}])
        history = pd.DataFrame([{"player": "Test", "game_date": "2026-11-01", "mp": 30}])
        events = []
        injury = {"DET": {"out": ["test"]}}
        result = nba.sanity_check_picks(picks, history, injury, rejection_events=events)
        self.assertTrue(result.empty)
        self.assertEqual(events[0]["line"], 4.5)
        self.assertIn("PLAYER OUT", events[0]["reason"])

    def test_run_keeps_publication_price_and_never_overwrites_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            (models / "model.pkl").write_bytes(b"model")
            config = {"models_dir": models, "output_dir": root, "odds_api_key": "secret",
                      "live_sportsbook_key": "fanduel"}
            run = decisions.start_run(config, {})
            candidates = pd.DataFrame([{"side_price": -158, "meta_prob": float("nan")}])
            published = pd.DataFrame([{"side_price": -138}])
            path = decisions.save_run(run, root, "2026-11-12", [], [], candidates, published, pd.DataFrame())
            saved = json.loads(path.read_text())
            self.assertEqual(saved["final_candidates"][0]["side_price"], -158)
            self.assertEqual(saved["published"][0]["side_price"], -138)
            self.assertNotIn("secret", path.read_text())
            self.assertEqual(saved["final_candidates"][0]["meta_prob"], None)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                decisions.save_run(run, root, "2026-11-12", [], [], candidates, published, pd.DataFrame())
            self.assertEqual(path.read_bytes(), before)

    def test_fallback_diagnostic_names_source(self):
        row = pd.Series({"price_gate_pass": True, "meta_gate_pass": True,
                         "volatility_gate_pass": True, "estimated_hit_rate_pass": True,
                         "pick_source_gate_pass": True, "pick_source": "volume_fill"})
        self.assertEqual(nba._format_live_gate_failures(row), "excluded fallback source=volume_fill")

    def test_future_evaluation_deduplicates_and_excludes_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output/gate_runs").mkdir(parents=True)
            (root / "data").mkdir()
            spec = {"start_date": "2026-11-01", "policy_sha256": "policy", "model_sha256": {},
                    "source_sha256": "source", "minimum_binary_contracts": 300,
                    "minimum_completed_dates": 30, "scope": "synthetic final candidates"}
            (root / "output/gate_experiment.json").write_text(json.dumps(spec))
            row = {"player": "Test", "prop": "AST", "direction": "UNDER", "line": 4.5,
                   "side_decimal_odds": 2.0, "line_source": "fanduel", "line_snapshot_ts": "2026-11-10T11:00:00Z",
                   "commence_time": "2026-11-11T01:00:00Z", "price_gate_pass": True,
                   "estimated_hit_rate_pass": True, "pick_source_gate_pass": True,
                   "pick_source": "market_model", "meta_gate_pass": True,
                   "volatility_gate_pass": True, "meta_gate_mode": "soft", "meta_prob": 44,
                   "meta_floor": 42, "volatility_metric": 3.2, "volatility_ceiling": 3.506}
            run = {"slate_date": "2026-11-10", "started_at": "2026-11-10T12:00:00Z",
                   "publication_saved_at": "2026-11-10T12:10:00Z", "policy_sha256": "policy",
                   "model_sha256": {}, "source_sha256": "source", "final_candidates": [row], "published": [row]}
            for name in ["a", "b"]:
                (root / f"output/gate_runs/{name}.json").write_text(json.dumps(run))
            (root / "output/gate_runs/drift.json").write_text(json.dumps({**run, "policy_sha256": "changed"}))
            pd.DataFrame([{"player": "Test", "game_date": "2026-11-10", "ast": 3}]).to_csv(root / "data/nba_data.csv", index=False)
            with patch.object(experiment, "ROOT", root):
                result = experiment.evaluate()
            self.assertEqual(result["binary_contracts"], 1)
            self.assertEqual(result["runs_excluded_for_drift"], 1)
            self.assertEqual(result["comparisons"]["incumbent_eligible"]["profit_units"], 1)
            self.assertEqual(result["comparisons"]["meta_floor_46"]["n"], 0)
            self.assertEqual(result["comparisons"]["volatility_mult_1_25"]["n"], 0)
            self.assertFalse(result["review_eligible"])


if __name__ == "__main__":
    unittest.main()
