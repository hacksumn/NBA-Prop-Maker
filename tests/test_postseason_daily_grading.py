import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

import nba_scraper
from run_daily import _fetch_live_box_scores_for_date


def _raw_game_log_row(game_id: str, game_date: str, player_name: str, pts: int = 12):
    return {
        "PLAYER_ID": "1",
        "TEAM_ID": "10",
        "GAME_ID": game_id,
        "PLAYER_NAME": player_name,
        "TEAM_ABBREVIATION": "MIA",
        "GAME_DATE": game_date,
        "MATCHUP": "MIA vs. BOS",
        "WL": "W",
        "MIN": "30",
        "FGM": "5",
        "FGA": "10",
        "FG_PCT": "0.500",
        "FG3M": "1",
        "FG3A": "3",
        "FG3_PCT": "0.333",
        "FTM": "1",
        "FTA": "2",
        "REB": "4",
        "AST": "3",
        "STL": "1",
        "BLK": "0",
        "TOV": "2",
        "PTS": str(pts),
        "PLUS_MINUS": "5",
        "PF": "2",
    }


class FakeSeasonEndpoint:
    calls = []

    def __init__(self, season, player_or_team_abbreviation, season_type_all_star):
        self.season = season
        self.season_type_all_star = season_type_all_star
        FakeSeasonEndpoint.calls.append((season, season_type_all_star))

    def get_data_frames(self):
        rows_by_phase = {
            "Regular Season": [_raw_game_log_row("100", "2026-04-12", "Regular Player", pts=11)],
            "PlayIn": [_raw_game_log_row("200", "2026-04-15", "PlayIn Player", pts=14)],
            "Playoffs": [_raw_game_log_row("300", "2026-04-18", "Playoff Player", pts=17)],
        }
        return [pd.DataFrame(rows_by_phase[self.season_type_all_star])]


class FakeDailyEndpoint:
    calls = []

    def __init__(self, season, season_type_all_star, date_from_nullable, date_to_nullable, player_or_team_abbreviation):
        self.season = season
        self.season_type_all_star = season_type_all_star
        FakeDailyEndpoint.calls.append((season, season_type_all_star, date_from_nullable, date_to_nullable))

    def get_data_frames(self):
        if self.season_type_all_star == "Regular Season":
            return [pd.DataFrame()]
        if self.season_type_all_star == "PlayIn":
            return [pd.DataFrame([_raw_game_log_row("400", "2026-04-15", "Jimmy Butler", pts=21)])]
        return [pd.DataFrame()]


class PostseasonDailyGradingTests(unittest.TestCase):
    def test_default_seasons_roll_forward_without_manual_year_bump(self):
        self.assertEqual(
            nba_scraper._default_seasons(as_of=datetime(2026, 10, 20)),
            ["2024-25", "2025-26", "2026-27"],
        )

    def test_nba_scraper_fetch_season_includes_postseason_phases(self):
        FakeSeasonEndpoint.calls = []
        with patch.object(nba_scraper.leaguegamelog, "LeagueGameLog", FakeSeasonEndpoint):
            with patch.object(nba_scraper.time, "sleep", lambda *_: None):
                season_df = nba_scraper.fetch_season("2025-26", retries=1, delay=0)

        self.assertEqual(
            FakeSeasonEndpoint.calls,
            [
                ("2025-26", "Regular Season"),
                ("2025-26", "PlayIn"),
                ("2025-26", "Playoffs"),
            ],
        )
        self.assertEqual(set(season_df["game_date"]), {"2026-04-12", "2026-04-15", "2026-04-18"})
        self.assertIn("PlayIn Player", season_df["player"].tolist())

    def test_run_daily_live_fetch_falls_through_to_playin(self):
        FakeDailyEndpoint.calls = []
        live_box, season_type = _fetch_live_box_scores_for_date(
            "2026-04-15",
            lambda name: str(name).lower(),
            fetcher_cls=FakeDailyEndpoint,
            sleep_fn=lambda *_: None,
        )

        self.assertEqual(season_type, "PlayIn")
        self.assertEqual(
            FakeDailyEndpoint.calls,
            [
                ("2025-26", "Regular Season", "04/15/2026", "04/15/2026"),
                ("2025-26", "PlayIn", "04/15/2026", "04/15/2026"),
            ],
        )
        self.assertEqual(live_box["game_date"].tolist(), ["2026-04-15"])
        self.assertEqual(live_box["player"].tolist(), ["Jimmy Butler"])
        self.assertEqual(live_box["pts"].tolist(), [21])


if __name__ == "__main__":
    unittest.main()
