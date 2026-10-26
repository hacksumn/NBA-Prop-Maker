import unittest

import pandas as pd

from scripts.train_availability_minutes_model import (
    FEATURE_COLUMNS,
    build_candidate_table,
    make_expanding_date_folds,
)


def synthetic_logs(player_two_plays_game_three: bool = False) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2025-10-20", periods=5, freq="2D")
    for game_number, game_date in enumerate(dates, start=1):
        rows.append(
            {
                "season": "2025-26",
                "game_date": game_date,
                "game_id": f"g{game_number}",
                "team": "AAA",
                "player_id": "p1",
                "player": "Always Active",
                "mp": 32.0,
                "is_home": float(game_number % 2),
            }
        )
        if game_number <= 2 or (game_number == 3 and player_two_plays_game_three):
            rows.append(
                {
                    "season": "2025-26",
                    "game_date": game_date,
                    "game_id": f"g{game_number}",
                    "team": "AAA",
                    "player_id": "p2",
                    "player": "Becomes Absent",
                    "mp": 24.0,
                    "is_home": float(game_number % 2),
                }
            )
    return pd.DataFrame(rows)


class AvailabilityMinutesModelTests(unittest.TestCase):
    def test_candidate_table_includes_pregame_known_nonappearance(self):
        candidates = build_candidate_table(synthetic_logs(False))
        row = candidates[
            candidates["game_id"].eq("g3") & candidates["player_id"].eq("p2")
        ].iloc[0]
        self.assertEqual(0, row["available"])
        self.assertEqual(0.0, row["minutes"])
        self.assertEqual(1.0, row["avail_l1"])
        self.assertEqual(0.0, row["missed_streak"])

    def test_current_game_outcome_does_not_change_pregame_features(self):
        absent = build_candidate_table(synthetic_logs(False))
        active = build_candidate_table(synthetic_logs(True))
        absent_row = absent[
            absent["game_id"].eq("g3") & absent["player_id"].eq("p2")
        ].iloc[0]
        active_row = active[
            active["game_id"].eq("g3") & active["player_id"].eq("p2")
        ].iloc[0]
        pd.testing.assert_series_equal(
            absent_row[FEATURE_COLUMNS],
            active_row[FEATURE_COLUMNS],
            check_names=False,
        )
        self.assertNotEqual(absent_row["available"], active_row["available"])

    def test_missed_streak_uses_only_prior_team_games(self):
        candidates = build_candidate_table(synthetic_logs(False))
        row = candidates[
            candidates["game_id"].eq("g4") & candidates["player_id"].eq("p2")
        ].iloc[0]
        self.assertEqual(0.0, row["avail_l1"])
        self.assertEqual(1.0, row["missed_streak"])

    def test_expanding_folds_keep_dates_disjoint_and_ordered(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="D")
        frame = pd.DataFrame(
            {
                "game_date": dates.repeat(2),
                "available": 1,
                "minutes": 20.0,
            }
        )
        folds = make_expanding_date_folds(frame, n_splits=4, min_train_dates=40)
        self.assertEqual(4, len(folds))
        for train_index, validation_index in folds:
            train_dates = set(frame.iloc[train_index]["game_date"])
            validation_dates = set(frame.iloc[validation_index]["game_date"])
            self.assertFalse(train_dates & validation_dates)
            self.assertLess(max(train_dates), min(validation_dates))

    def test_model_features_exclude_identity_and_outcomes(self):
        forbidden = {"game_date", "game_id", "player_id", "player", "available", "minutes"}
        self.assertFalse(forbidden & set(FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
