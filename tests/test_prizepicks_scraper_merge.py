import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from prizepicks_scraper import _merge_historical_lines, save_lines


class PrizePicksScraperMergeTests(unittest.TestCase):
    def test_empty_live_fetch_fails_instead_of_reporting_stale_file_as_updated(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "historical_lines.csv"
            output.write_text("sentinel\nkeep\n", encoding="utf-8")
            before = output.read_bytes()
            with patch("prizepicks_scraper.fetch_todays_lines", return_value=pd.DataFrame()):
                with self.assertRaisesRegex(RuntimeError, "No current PrizePicks lines"):
                    save_lines(str(output))
            self.assertEqual(output.read_bytes(), before)

    def test_prizepicks_snapshot_and_fanduel_line_are_both_retained(self):
        existing = pd.DataFrame(
            [
                {
                    "game_date": "2026-05-12",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 7.5,
                    "num_books": 1,
                    "source": "odds_api_fanduel",
                    "snapshot_ts": "2026-05-11 13:20:00",
                    "projection_type": "standard",
                    "is_promo": False,
                }
            ]
        )
        fresh = pd.DataFrame(
            [
                {
                    "game_date": "2026-05-12",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 8.5,
                    "num_books": 1,
                    "source": "prizepicks_scraper",
                    "snapshot_ts": "2026-05-11 13:24:00",
                    "projection_type": "demon",
                    "is_promo": False,
                }
            ]
        )

        merged = _merge_historical_lines(existing, fresh)

        self.assertEqual(len(merged), 2)
        self.assertEqual(set(merged["source"]), {"odds_api_fanduel", "prizepicks_scraper"})

    def test_exact_source_snapshot_is_idempotent(self):
        row = pd.DataFrame([{
            "game_date": "2026-05-12", "player": "Al Horford", "player_norm": "al horford",
            "prop": "player_points", "line": 7.5, "num_books": 1,
            "source": "odds_api_fanduel", "snapshot_ts": "2026-05-11 13:20:00",
            "projection_type": "standard", "is_promo": False,
        }])

        merged = _merge_historical_lines(row, row)

        self.assertEqual(len(merged), 1)

    def test_same_book_snapshot_retains_alternate_lines(self):
        first = pd.DataFrame([{
            "game_date": "2026-05-12", "player": "Al Horford", "player_norm": "al horford",
            "prop": "player_points", "line": 7.5, "num_books": 1,
            "source": "odds_api_bovada", "snapshot_ts": "2026-05-11 13:20:00",
            "projection_type": "standard", "is_promo": False,
        }])
        alternate = first.copy()
        alternate["line"] = 8.5

        merged = _merge_historical_lines(first, alternate)

        self.assertEqual(len(merged), 2)
        self.assertEqual(set(merged["line"]), {7.5, 8.5})


if __name__ == "__main__":
    unittest.main()
