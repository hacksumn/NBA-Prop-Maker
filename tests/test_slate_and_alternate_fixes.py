"""Regressions for CodeRabbit findings fixed under DEC-082."""
import unittest
from datetime import datetime, timezone

import pandas as pd

import nba_props as nb


class SlateDateTimezoneTests(unittest.TestCase):
    def test_utc_aware_as_of_does_not_crash_missing_today_path(self):
        lines = pd.DataFrame([{
            "game_date": "2026-11-12", "player": "A", "prop": "player_assists", "line": 4.5,
            "source": "odds_api_fanduel", "projection_type": "standard", "is_promo": False,
        }])
        out = nb._select_cached_lines_for_predict(
            lines, "2026-11-13", as_of=datetime(2026, 11, 13, 15, 0, tzinfo=timezone.utc),
            allowed_sources=["fanduel"], standard_only=True, allow_future_slate=False,
        )
        self.assertEqual(out[3], "missing_exact_today")
        self.assertTrue(out[0].empty)

    def test_aware_and_naive_as_of_pick_the_same_slate(self):
        lines = pd.DataFrame({"game_date": ["2026-11-12", "2026-11-14"]})
        naive = nb._resolve_active_slate_date(lines, as_of=datetime(2026, 11, 13, 9, 0))
        aware = nb._resolve_active_slate_date(
            lines, as_of=datetime(2026, 11, 13, 15, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(naive, "2026-11-14")
        self.assertEqual(aware, naive)


if __name__ == "__main__":
    unittest.main()
