import copy

import numpy as np
import pandas as pd
import pytest

from scripts.build_pointintime_team_defense import (
    RATING_COLUMNS,
    build_point_in_time_ratings,
    build_live_matchup_ratings,
    build_team_game_aggregates,
    build_team_game_aggregates_from_team_logs,
    validate_rating_table,
)
from scripts.build_prop_defense_rating_training_data import (
    RATING_REQUIRED,
    build_training_table,
)
from train_advanced_models import (
    AdvancedPropModel,
    _validate_defense_rating_training_features,
)
from feature_pipeline import FeaturePipeline, _repair_matchup_columns


TEAMS = [f"T{i:02d}" for i in range(30)]


def _team_games(n_dates: int = 8) -> pd.DataFrame:
    rows = []
    for day in range(n_dates):
        date = pd.Timestamp("2025-10-01") + pd.Timedelta(days=day)
        for pair in range(15):
            home = TEAMS[2 * pair]
            away = TEAMS[2 * pair + 1]
            game_id = 1000 + day * 20 + pair
            for team, opponent, is_home, offset in [
                (home, away, 1, pair % 5),
                (away, home, 0, (pair + 2) % 5),
            ]:
                possessions = 98.0 + (pair % 3)
                points = 108.0 + offset + day
                rows.append(
                    {
                        "season": "2025-26",
                        "GAME_ID": game_id,
                        "game_date": date,
                        "team_tricode": team,
                        "opponent_tricode": opponent,
                        "is_home": is_home,
                        "team_pts": points,
                        "team_fga": 85.0,
                        "team_fta": 20.0,
                        "team_orb": 10.0,
                        "team_tov": 14.0,
                        "team_possessions": possessions,
                        "team_ortg": points / possessions * 100.0,
                    }
                )
    return pd.DataFrame(rows)


def _rating_rows_for_adapter() -> pd.DataFrame:
    values = {
        "GAME_ID": [1, 1],
        "team": ["AAA", "BBB"],
        "game_date": pd.to_datetime(["2025-01-10", "2025-01-10"]),
        "opponent": ["BBB", "AAA"],
        "is_home": [1, 0],
        "rating_asof_date": pd.to_datetime(["2025-01-09", "2025-01-09"]),
        "team_rating_source": ["current_season_expanding"] * 2,
        "opp_rating_source": ["current_season_expanding"] * 2,
        "n_prior_games": [20, 20],
        "opp_n_prior_games": [20, 20],
        "off_rtg_pt": [116.0, 114.0],
        "def_rtg_pt": [111.0, 113.0],
        "net_rtg_pt": [5.0, 1.0],
        "pace_pt": [99.0, 100.0],
        "opp_def_rtg_pt": [113.0, 111.0],
        "opp_pace_pt": [100.0, 99.0],
        "opp_def_rank_pt": [10, 4],
        "opp_def_tier_pt": [2, 1],
        "matchup_expected_ortg_pt": [116.5, 110.0],
        "rating_league_avg": [115.0, 115.0],
        "rating_hca": [2.5, 2.5],
    }
    return pd.DataFrame(values)[RATING_REQUIRED]


def test_target_date_outcomes_cannot_change_same_date_ratings():
    games = _team_games()
    original = build_point_in_time_ratings(games, min_prior_games=2)
    changed = games.copy()
    target = pd.Timestamp("2025-10-06")
    changed.loc[changed["game_date"].eq(target), "team_ortg"] += 25.0
    rebuilt = build_point_in_time_ratings(changed, min_prior_games=2)
    left = original.loc[original["game_date"].eq(target), RATING_COLUMNS].reset_index(drop=True)
    right = rebuilt.loc[rebuilt["game_date"].eq(target), RATING_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_rating_validator_rejects_duplicate_and_chronology_violation():
    ratings = build_point_in_time_ratings(_team_games(), min_prior_games=2)
    duplicate = pd.concat([ratings, ratings.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_rating_table(duplicate)
    bad = ratings.copy()
    bad.loc[0, "rating_asof_date"] = bad.loc[0, "game_date"]
    with pytest.raises(ValueError, match="chronology"):
        validate_rating_table(bad)


def test_box_score_aggregate_repairs_missing_latest_season_matchups():
    rows = []
    for team, matchup in [("AAA", "AAA vs. BBB"), ("BBB", "BBB @ AAA")]:
        rows.append(
            {
                "season": "2025-26", "GAME_ID": 1, "game_date": "2025-10-20",
                "team": team, "opp": np.nan, "is_home": np.nan, "matchup": matchup,
                "pts": 110, "fga": 90, "fta": 20, "orb": 10, "tov": 12,
            }
        )
    aggregate = build_team_game_aggregates(pd.DataFrame(rows))
    assert aggregate.set_index("team_tricode")["opponent_tricode"].to_dict() == {
        "AAA": "BBB", "BBB": "AAA"
    }
    assert aggregate.set_index("team_tricode")["is_home"].to_dict() == {"AAA": 1, "BBB": 0}


def test_exact_team_logs_supply_possessions_and_reject_missing_orb():
    logs = pd.DataFrame(
        {
            "GAME_ID": [1, 1],
            "GAME_DATE": ["2025-10-20", "2025-10-20"],
            "TEAM_ABBREVIATION": ["AAA", "BBB"],
            "MATCHUP": ["AAA vs. BBB", "BBB @ AAA"],
            "PTS": [110, 105], "FGA": [90, 88], "FTA": [20, 18],
            "OREB": [10, 8], "TOV": [12, 14], "season": ["2025-26"] * 2,
        }
    )
    aggregate = build_team_game_aggregates_from_team_logs(logs)
    expected = {"AAA": 90 + 0.44 * 20 - 10 + 12, "BBB": 88 + 0.44 * 18 - 8 + 14}
    assert aggregate.set_index("team_tricode")["team_possessions"].to_dict() == pytest.approx(expected)

    broken = logs.copy()
    broken.loc[0, "OREB"] = np.nan
    with pytest.raises(ValueError, match="null"):
        build_team_game_aggregates_from_team_logs(broken)


def test_prop_adapter_repairs_missing_matchup_and_is_one_to_one():
    prop = pd.DataFrame(
        {
            "GAME_ID": [1, 1], "PLAYER_ID": [11, 12], "team": ["AAA", "BBB"],
            "game_date": ["2025-01-10", "2025-01-10"],
            "opp": [pd.NA, pd.NA], "is_home": [np.nan, np.nan], "pts": [10, 12],
        }
    )
    output, summary = build_training_table(prop, _rating_rows_for_adapter())
    assert len(output) == len(prop)
    assert summary["match_rate"] == 1.0
    assert output["opp"].tolist() == ["BBB", "AAA"]
    assert output["is_home"].tolist() == [1, 0]


def test_prop_adapter_rejects_rating_chronology_violation():
    prop = pd.DataFrame({"GAME_ID": [1], "team": ["AAA"], "game_date": ["2025-01-10"]})
    ratings = _rating_rows_for_adapter().iloc[[0]].copy()
    ratings["rating_asof_date"] = ratings["game_date"]
    with pytest.raises(ValueError, match="chronology"):
        build_training_table(prop, ratings)


def test_trainer_defense_contract_is_registered_and_fail_closed():
    prop = pd.DataFrame(
        {
            "GAME_ID": [1, 1], "PLAYER_ID": [11, 12], "team": ["AAA", "BBB"],
            "game_date": ["2025-01-10", "2025-01-10"],
        }
    )
    joined, _ = build_training_table(prop, _rating_rows_for_adapter())
    contract = _validate_defense_rating_training_features(joined, AdvancedPropModel(model_dir="models/test_defense_contract"))
    assert contract["chronology_violations"] == 0
    broken = joined.drop(columns=["opp_def_rtg_pt"])
    with pytest.raises(ValueError, match="missing columns"):
        _validate_defense_rating_training_features(broken)


def test_crude_context_is_ranked_once_per_team_and_pace_normalized():
    rows = []
    for day in range(6):
        date = pd.Timestamp("2025-11-01") + pd.Timedelta(days=day)
        for index, team in enumerate(TEAMS):
            opponent = TEAMS[index ^ 1]
            rows.append(
                {
                    "season": "2025-26", "game_date": date, "player": f"P{index}",
                    "team": team, "opp": opponent, "pts": 80 + index,
                    "trb": 30 + index / 10, "ast": 15 + index / 10,
                    "fga": 80 + index / 5, "fta": 20, "orb": 10, "tov": 12,
                }
            )
    context = FeaturePipeline()._create_opponent_context(pd.DataFrame(rows))
    target = context[context["game_date"].eq(pd.Timestamp("2025-11-06"))]
    by_opponent = target.drop_duplicates("opp")
    assert len(by_opponent) == 30
    assert sorted(by_opponent["opp_pts_allowed_rank"].astype(int).tolist()) == list(range(1, 31))
    assert by_opponent["opp_pace_factor"].mean() == pytest.approx(1.0)
    best = by_opponent.loc[by_opponent["opp_pts_allowed_per100_l10"].idxmin()]
    worst = by_opponent.loc[by_opponent["opp_pts_allowed_per100_l10"].idxmax()]
    assert best["opp_def_tier"] == 1
    assert worst["opp_def_tier"] == 5


def test_feature_pipeline_repairs_blank_2025_26_matchup_columns():
    raw = pd.DataFrame(
        {
            "GAME_ID": [1, 1], "team": ["AAA", "BBB"],
            "opp": [pd.NA, pd.NA], "is_home": [np.nan, np.nan],
            "matchup": ["AAA vs. BBB", "BBB @ AAA"],
        }
    )
    repaired = _repair_matchup_columns(raw)
    assert repaired["opp"].tolist() == ["BBB", "AAA"]
    assert repaired["is_home"].tolist() == [1, 0]


def test_live_adapter_scores_only_scheduled_opponents_from_prior_games():
    aggregates = _team_games()
    box = aggregates.rename(
        columns={
            "team_tricode": "team", "opponent_tricode": "opp",
            "team_pts": "pts", "team_fga": "fga", "team_fta": "fta",
            "team_orb": "orb", "team_tov": "tov",
        }
    )
    box["matchup"] = np.where(
        box["is_home"].eq(1),
        box["team"] + " vs. " + box["opp"],
        box["team"] + " @ " + box["opp"],
    )
    schedule = pd.DataFrame(
        {"home_team": [TEAMS[0], TEAMS[2]], "away_team": [TEAMS[1], TEAMS[3]]}
    )
    slate = pd.Timestamp("2025-10-10")
    live = build_live_matchup_ratings(box, schedule, slate, min_prior_games=2)
    assert set(live["team"]) == set(TEAMS[:4])
    assert live["opp_def_rtg_pt"].notna().all()
    assert (pd.to_datetime(live["rating_asof_date"]) < slate).all()
