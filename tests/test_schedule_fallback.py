from unittest.mock import patch

import nba_props
import pandas as pd


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "events": [
                {
                    "id": "401810001",
                    "competitions": [
                        {
                            "competitors": [
                                {
                                    "homeAway": "home",
                                    "team": {"abbreviation": "NO"},
                                },
                                {
                                    "homeAway": "away",
                                    "team": {"abbreviation": "NY"},
                                },
                            ]
                        }
                    ],
                }
            ]
        }


def test_fetch_todays_games_falls_back_to_espn_and_normalizes_teams():
    with (
        patch(
            "nba_api.stats.endpoints.scoreboardv2.ScoreboardV2",
            side_effect=ValueError("malformed NBA Stats response"),
        ),
        patch.object(nba_props.requests, "get", return_value=_Response()) as request,
    ):
        games = nba_props.fetch_todays_games("2026-10-25")

    assert games.to_dict("records") == [
        {
            "game_id": "401810001",
            "home_team": "NOP",
            "away_team": "NYK",
            "game_date": "2026-10-25",
        }
    ]
    assert request.call_args.kwargs["params"]["dates"] == "20261025"


def test_fetch_todays_games_removes_identical_nba_stats_duplicates():
    duplicated = pd.DataFrame([
        {
            "GAME_ID": "0022600127",
            "HOME_TEAM_ABBREVIATION": "DEN",
            "VISITOR_TEAM_ABBREVIATION": "GSW",
        },
        {
            "GAME_ID": "0022600127",
            "HOME_TEAM_ABBREVIATION": "DEN",
            "VISITOR_TEAM_ABBREVIATION": "GSW",
        },
    ])

    class _Scoreboard:
        def get_data_frames(self):
            return [duplicated]

    with patch(
        "nba_api.stats.endpoints.scoreboardv2.ScoreboardV2",
        return_value=_Scoreboard(),
    ), patch("nba_api.stats.static.teams.get_teams", return_value=[]):
        games = nba_props.fetch_todays_games("2026-10-26")

    assert games.to_dict("records") == [{
        "game_id": "0022600127",
        "home_team": "DEN",
        "away_team": "GSW",
        "game_date": "2026-10-26",
    }]
