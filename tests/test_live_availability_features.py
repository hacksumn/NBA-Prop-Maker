import pandas as pd

from scripts.train_availability_minutes_model import build_live_feature_frame


def test_live_availability_features_use_only_prior_team_games():
    logs = pd.DataFrame(
        [
            {
                "GAME_ID": f"002260000{index}",
                "PLAYER_ID": 1,
                "player": "Rotation Player",
                "team": "AAA",
                "game_date": day,
                "mp": minutes,
                "season": "2026-27",
                "is_home": index % 2,
            }
            for index, (day, minutes) in enumerate(
                [
                    ("2026-10-20", 30.0),
                    ("2026-10-22", 28.0),
                    ("2026-10-24", 32.0),
                ],
                start=1,
            )
        ]
    )
    schedule = pd.DataFrame(
        [{
            "game_id": "0022600100",
            "home_team": "AAA",
            "away_team": "BBB",
        }]
    )

    live = build_live_feature_frame(logs, schedule, "2026-10-27")

    assert len(live) == 1
    assert live.loc[0, "player"] == "Rotation Player"
    assert live.loc[0, "mp_last"] == 32.0
    assert live.loc[0, "mp_played_mean_l3"] == 30.0
    assert live.loc[0, "days_since_played"] == 3.0
    assert 0.0 < live.loc[0, "season_progress"] < 1.0
