"""Boundary and accounting checks for the read-only DEC-104 A replay."""
import unittest

import numpy as np
import pandas as pd

from scripts.replay_model_market_cutoff import classify, comparison, summarize


class CutoffReplayTests(unittest.TestCase):
    def test_exact_boundary_signed_gap_and_unknowns(self):
        rows = pd.DataFrame({
            "model_side_prob": [.6, .6001, .3, np.nan, 60, .8, .8],
            "no_vig_side_prob": [.5] * 7,
            "direction": ["UNDER"] * 5 + ["OVER", "UNDER"],
            "line_source": ["fanduel"] * 6 + ["prizepicks"],
        })
        self.assertEqual(classify(rows).replay_action.tolist(),
                         ["keep", "remove", "keep", "unknown", "unknown", "outside_scope", "outside_scope"])

    def test_payouts_and_nonbinary_denominators(self):
        frame = pd.DataFrame({
            "result": ["WIN", "LOSS", "PUSH", "DNP", None, "WIN"],
            "side_decimal_odds": [2.5, 1.8, 2, 2, 2, np.nan],
            "game_date": ["2026-11-19"] * 6,
        })
        result = summarize(frame)
        self.assertEqual(result["settled_binary"], 3)
        self.assertEqual(result["priced_binary"], 2)
        self.assertEqual(result["unpriced_binary"], 1)
        self.assertAlmostEqual(result["profit_units"], .5)
        self.assertAlmostEqual(result["roi"], .25)
        self.assertEqual(result["pushes"], 1)
        self.assertEqual(result["dnps"], 1)
        self.assertEqual(result["pending_or_other"], 1)

    def test_unassessable_history_not_counted_as_retained(self):
        frame = pd.DataFrame({
            "result": ["WIN", "LOSS", "WIN"], "side_decimal_odds": [2., 2., 2.],
            "game_date": ["2026-11-19"] * 3,
            "replay_action": ["keep", "remove", "unknown"],
        })
        result = comparison(frame)
        self.assertEqual(result["all_logged"]["settled_binary"], 3)
        self.assertEqual(result["comparable_before"]["settled_binary"], 2)
        self.assertEqual(result["after_cutoff"]["profit_units"], 1)
        self.assertEqual(result["removed"]["profit_units"], -1)
        self.assertEqual(result["unknown"]["settled_binary"], 1)


if __name__ == "__main__":
    unittest.main()
