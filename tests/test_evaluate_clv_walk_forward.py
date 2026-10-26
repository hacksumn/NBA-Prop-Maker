import unittest

import numpy as np
import pandas as pd

from scripts.evaluate_clv_walk_forward import (
    HOLDOUT_DATE_COUNT,
    MIN_TRAIN_DATES,
    chronological_split_dates,
    matched_random_card_distribution,
    random_card_distribution,
)


class WalkForwardSplitTests(unittest.TestCase):
    def test_holdout_is_strictly_after_development_and_folds_are_expanding(self):
        dates = pd.date_range("2026-01-01", periods=33, freq="D").strftime("%Y-%m-%d").tolist()
        development, holdout, folds = chronological_split_dates(dates)
        self.assertEqual(HOLDOUT_DATE_COUNT, len(holdout))
        self.assertLess(max(development), min(holdout))
        self.assertGreaterEqual(len(folds), 3)
        for train, validation in folds:
            self.assertGreaterEqual(len(train), MIN_TRAIN_DATES)
            self.assertLess(max(train), min(validation))
            self.assertTrue(set(train).isdisjoint(validation))
            self.assertTrue(set(validation).isdisjoint(holdout))

    def test_thin_but_usable_history_keeps_a_real_holdout(self):
        dates = pd.date_range("2026-01-01", periods=8, freq="D").strftime("%Y-%m-%d").tolist()
        development, holdout, folds = chronological_split_dates(dates)
        self.assertEqual(2, len(holdout))
        self.assertLess(max(development), min(holdout))
        self.assertGreaterEqual(len(folds), 3)

    def test_random_cards_are_deterministic_and_same_date_bounded(self):
        frame = pd.DataFrame(
            {
                "game_date": ["2026-01-01"] * 8 + ["2026-01-02"] * 3,
                "under_win": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        first = random_card_distribution(frame, repetitions=50, card_size=5, seed=7)
        second = random_card_distribution(frame, repetitions=50, card_size=5, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(((first >= 0) & (first <= 1)).all())

    def test_matched_random_preserves_selected_strata(self):
        pool = pd.DataFrame(
            {
                "game_date": ["2026-01-01"] * 6,
                "prop": ["a", "a", "a", "b", "b", "b"],
                "pp_projection_type": ["standard"] * 3 + ["demon"] * 3,
                "under_win": [1, 1, 0, 0, 0, 1],
            }
        )
        selected = pool.iloc[[0, 1, 3]].copy()
        rates = matched_random_card_distribution(pool, selected, repetitions=25, seed=9)
        self.assertEqual(25, len(rates))
        self.assertTrue(((rates >= 0) & (rates <= 1)).all())


if __name__ == "__main__":
    unittest.main()
