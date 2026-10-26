import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluate_probability_under_selection import evaluate


class ProbabilityUnderSelectionTests(unittest.TestCase):
    def test_latest_dates_are_reserved_from_calibration_fit(self):
        rows = []
        dates = pd.date_range("2026-01-01", periods=24, freq="D")
        for day_idx, day in enumerate(dates):
            for row_idx in range(10):
                actual_under = row_idx < 7
                rows.append(
                    {
                        "target": "ast",
                        "game_date": day.date().isoformat(),
                        "actual": 4.0 if actual_under else 8.0,
                        "line": 6.5,
                        "line_source": "real",
                        "p_over_raw": 0.20 + row_idx * 0.06,
                        "actual_over": int(not actual_under),
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oof.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            selected, summary = evaluate(path, holdout_dates=4)

        self.assertEqual(summary["development"]["dates"], 20)
        self.assertEqual(summary["holdout"]["dates"], 4)
        self.assertEqual(summary["development"]["date_max"], "2026-01-20")
        self.assertEqual(summary["holdout"]["date_min"], "2026-01-21")
        development_dates = set(selected.loc[selected["split"].eq("development"), "game_date"])
        holdout_dates = set(selected.loc[selected["split"].eq("holdout"), "game_date"])
        self.assertFalse(development_dates & holdout_dates)

    def test_missing_contract_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oof.csv"
            pd.DataFrame([{"target": "ast"}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                evaluate(path, holdout_dates=1)


if __name__ == "__main__":
    unittest.main()
