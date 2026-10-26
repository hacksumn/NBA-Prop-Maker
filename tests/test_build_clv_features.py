import unittest

import pandas as pd

from scripts.build_clv_features import _column_or_nan, choose_main_book_lines, parse_local_snapshot


class BuildClvFeaturesTests(unittest.TestCase):
    def test_missing_decision_snapshot_materializes_as_nan(self):
        frame = pd.DataFrame({"consensus_line_t_minus_5": [10.5, 11.5]})
        missing = _column_or_nan(frame, "consensus_line_decision")
        self.assertEqual(len(missing), 2)
        self.assertTrue(missing.isna().all())

    def test_balanced_price_line_is_selected_as_book_main(self):
        base = {
            "manifest_key": "game:t_minus_5",
            "player_norm": "jayson tatum",
            "prop": "player_points",
            "bookmaker": "bovada",
        }
        rows = pd.DataFrame([
            {**base, "line": 28.5, "over_price": -110, "under_price": -110},
            {**base, "line": 31.5, "over_price": 220, "under_price": -300},
        ])
        selected = choose_main_book_lines(rows)
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected.iloc[0]["line"]), 28.5)

    def test_naive_archive_timestamp_is_interpreted_as_chicago_time(self):
        stamp = parse_local_snapshot("2026-04-03 09:20:41")
        self.assertEqual(stamp.isoformat(), "2026-04-03T14:20:41+00:00")


if __name__ == "__main__":
    unittest.main()
