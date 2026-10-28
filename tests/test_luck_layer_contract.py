from pathlib import Path
from unittest.mock import patch

import pandas as pd

import luck_regression_model as luck
import run_daily


class _EndpointResult:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_data_frames(self):
        return [self.frame.copy()]


def test_early_season_player_luck_returns_schema_valid_empty_frame():
    base = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "PLAYER_NAME": ["Early Player"],
            "TEAM_ABBREVIATION": ["AAA"],
            "GP": [3],
            "FG3M": [1.0],
            "FG3A": [3.0],
            "FG3_PCT": [0.333],
            "FTM": [2.0],
            "FTA": [2.5],
            "FT_PCT": [0.8],
            "FGM": [5.0],
            "FGA": [10.0],
            "FG_PCT": [0.5],
            "PTS": [13.0],
            "REB": [4.0],
            "AST": [2.0],
        }
    )
    advanced = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "TS_PCT": [0.55],
            "EFG_PCT": [0.55],
            "USG_PCT": [0.2],
            "OFF_RATING": [110.0],
        }
    )
    shot_locations = pd.DataFrame({"PLAYER_ID": [1]})

    with (
        patch.object(
            luck.leaguedashplayerstats,
            "LeagueDashPlayerStats",
            side_effect=[_EndpointResult(base), _EndpointResult(advanced)],
        ),
        patch.object(
            luck.leaguedashplayershotlocations,
            "LeagueDashPlayerShotLocations",
            return_value=_EndpointResult(shot_locations),
        ),
        patch.object(luck.time, "sleep"),
    ):
        result = luck.build_player_luck_scores(season="2026-27", min_gp=15)

    assert result.empty
    assert list(result.columns) == luck.PLAYER_LUCK_COLUMNS


def test_luck_snapshot_contract_is_season_scoped():
    expected = Path(run_daily.LAYER_SNAPSHOT_DIR) / "player_luck_scores_2026-27.csv"
    assert expected.name == "player_luck_scores_2026-27.csv"
    assert run_daily.MAX_LUCK_SNAPSHOT_AGE_HOURS == 168.0
