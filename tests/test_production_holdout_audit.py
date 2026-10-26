import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.audit_production_holdout_simple import run_audit


class ProductionHoldoutAuditTests(unittest.TestCase):
    def test_audit_uses_result_and_direction_and_excludes_nonbinary_results(self):
        rows = [
            {"game_date": "2026-01-01", "prop": "pts", "result": "WIN", "direction": "OVER"},
            {"game_date": "2026-01-02", "prop": "pts", "result": "LOSS", "direction": "OVER"},
            {"game_date": "2026-01-03", "prop": "pts", "result": "WIN", "direction": "UNDER"},
            {"game_date": "2026-01-04", "prop": "pts", "result": "LOSS", "direction": "UNDER"},
            {"game_date": "2026-01-05", "prop": "pts", "result": "PUSH", "direction": "OVER"},
            {"game_date": "2026-01-06", "prop": "pts", "result": "DNP", "direction": "UNDER"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            picks_path = tmp_path / "picks.csv"
            pd.DataFrame(rows).to_csv(picks_path, index=False)

            results, summary = run_audit(picks_path, tmp_path, min_picks=1)

            self.assertEqual(int(results.loc[0, "graded_picks"]), 4)
            self.assertEqual(float(results.loc[0, "model_accuracy"]), 50.0)
            self.assertEqual(float(results.loc[0, "always_under_accuracy"]), 50.0)
            self.assertEqual(summary["total_picks_audited"], 4)
            self.assertTrue((tmp_path / "production_holdout_audit.json").exists())
            self.assertNotIn(
                "NaN", (tmp_path / "production_holdout_audit.json").read_text()
            )

    def test_audit_fails_closed_when_direction_column_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            picks_path = tmp_path / "picks.csv"
            pd.DataFrame([{
                "game_date": "2026-01-01", "prop": "pts", "result": "WIN",
            }]).to_csv(picks_path, index=False)

            with self.assertRaisesRegex(ValueError, "direction"):
                run_audit(picks_path, tmp_path, min_picks=1)


if __name__ == "__main__":
    unittest.main()
