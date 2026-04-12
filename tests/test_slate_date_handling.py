import unittest
from datetime import datetime

import pandas as pd

from nba_props import _select_active_slate_lines
from prizepicks_scraper import _parse_projections


class SlateDateHandlingTests(unittest.TestCase):
    def test_parse_projections_uses_payload_slate_date(self):
        payload = {
            "data": [
                {
                    "type": "projection",
                    "id": "proj-1",
                    "attributes": {
                        "stat_type": "Points",
                        "line_score": 20.5,
                        "start_time": "2026-04-12T18:10:00.000-04:00",
                    },
                    "relationships": {
                        "new_player": {"data": {"id": "player-1"}},
                        "game": {"data": {"id": "game-1"}},
                    },
                },
                {
                    "type": "projection",
                    "id": "proj-2",
                    "attributes": {
                        "stat_type": "Points",
                        "line_score": 21.5,
                        "start_time": "2026-04-13T18:10:00.000-04:00",
                    },
                    "relationships": {
                        "new_player": {"data": {"id": "player-1"}},
                        "game": {"data": {"id": "game-2"}},
                    },
                },
            ],
            "included": [
                {
                    "type": "new_player",
                    "id": "player-1",
                    "attributes": {"display_name": "Scottie Barnes"},
                },
                {
                    "type": "game",
                    "id": "game-1",
                    "attributes": {"start_time": "2026-04-12T18:00:00.000-04:00"},
                },
                {
                    "type": "game",
                    "id": "game-2",
                    "attributes": {"start_time": "2026-04-13T18:00:00.000-04:00"},
                },
            ],
        }

        parsed = _parse_projections(payload)

        self.assertEqual(parsed["game_date"].tolist(), ["2026-04-12", "2026-04-13"])
        self.assertEqual(parsed["player"].tolist(), ["Scottie Barnes", "Scottie Barnes"])
        self.assertEqual(parsed["prop"].tolist(), ["player_points", "player_points"])

    def test_select_active_slate_lines_prefers_nearest_upcoming_date(self):
        lines = pd.DataFrame(
            [
                {"game_date": "2026-04-12", "player": "Scottie Barnes", "prop": "player_points", "line": 20.5},
                {"game_date": "2026-04-13", "player": "Devin Carter", "prop": "player_assists", "line": 6.0},
                {"game_date": "2026-04-13", "player": "Brandon Ingram", "prop": "player_rebounds", "line": 7.5},
            ]
        )

        active, slate_date = _select_active_slate_lines(lines, as_of=datetime(2026, 4, 11, 19, 0, 0))

        self.assertEqual(slate_date, "2026-04-12")
        self.assertEqual(active["game_date"].tolist(), ["2026-04-12"])

    def test_select_active_slate_lines_falls_back_to_latest_past_date(self):
        lines = pd.DataFrame(
            [
                {"game_date": "2026-04-10", "player": "Scottie Barnes", "prop": "player_points", "line": 20.5},
                {"game_date": "2026-04-11", "player": "Devin Carter", "prop": "player_assists", "line": 6.0},
            ]
        )

        active, slate_date = _select_active_slate_lines(lines, as_of=datetime(2026, 4, 12, 9, 0, 0))

        self.assertEqual(slate_date, "2026-04-11")
        self.assertEqual(active["game_date"].tolist(), ["2026-04-11"])


if __name__ == "__main__":
    unittest.main()
