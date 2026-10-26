import unittest

import pandas as pd

from scripts.evaluate_sportsbook_expansion import HOLDOUT_DATES, _holdout_dates, make_development_folds


class SportsbookExpansionEvaluationTests(unittest.TestCase):
    def test_holdout_uses_latest_twenty_prediction_dates(self):
        dates = pd.date_range("2026-01-01", periods=30, freq="D").strftime("%Y-%m-%d")
        pool = pd.DataFrame({"game_date": dates, "model_prediction": 1.0})
        selected = _holdout_dates(pool)
        self.assertEqual(len(selected), HOLDOUT_DATES)
        self.assertEqual(selected[0], dates[-HOLDOUT_DATES])
        self.assertEqual(selected[-1], dates[-1])

    def test_development_folds_are_strictly_chronological(self):
        dates = pd.date_range("2026-01-01", periods=54, freq="D").strftime("%Y-%m-%d").tolist()
        folds = make_development_folds(dates)
        self.assertEqual(sum(len(validation) for _, validation in folds), 24)
        for train, validation in folds:
            self.assertLess(max(train), min(validation))


if __name__ == "__main__":
    unittest.main()
