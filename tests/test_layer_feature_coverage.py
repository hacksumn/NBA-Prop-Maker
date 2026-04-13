import unittest

import pandas as pd

from nba_props import _summarize_layer_feature_coverage


class LayerFeatureCoverageTests(unittest.TestCase):
    def test_partial_layer_coverage_is_not_treated_as_no_layer_data(self):
        layer_cols = [
            "possessions",
            "raw_ppp",
            "garbage_time_flag",
            "competitive_poss",
            "adj_ppp",
            "raw_ppp_season",
            "opp_def_rtg_avg",
            "total_possessions",
            "efg_pct",
            "ts_pct",
            "clean_pts",
            "clean_reb",
            "clean_ast",
            "clean_fg3m",
            "clean_min",
            "pts_delta",
            "n_exclude",
            "n_heavy",
            "n_partial",
            "usg_pct",
            "pts_per_poss",
            "ast_per_poss",
            "reb_per_poss",
            "min_pg",
            "poss_pg",
            "off_rtg",
            "net_rtg",
            "role_weight",
            "total_luck_score",
            "efg_luck_score",
            "fg3_luck_score",
            "ft_luck_score",
            "pts_luck_adj",
            "fg3_regressed",
            "ft_regressed",
        ]
        frame = pd.DataFrame(
            [
                {
                    "player": "Partial Player",
                    "possessions": 0.0,
                    "raw_ppp": 1.0,
                    "garbage_time_flag": 0.0,
                    "competitive_poss": 12.0,
                    "clean_pts": 2.0,
                },
                {
                    "player": "Full Player",
                    "possessions": 1.0,
                    "raw_ppp": 1.1,
                    "garbage_time_flag": 0.0,
                    "competitive_poss": 18.0,
                    "adj_ppp": 1.04,
                    "clean_pts": 6.5,
                    "usg_pct": 0.19,
                    "total_luck_score": 5.5,
                },
                {
                    "player": "Missing Player",
                },
            ]
        )

        summary = _summarize_layer_feature_coverage(
            frame,
            layer_cols,
            added_layer_cols=["adj_ppp", "clean_pts", "usg_pct", "total_luck_score"],
        )

        self.assertEqual(summary["rows_with_any_layer"], 2)
        self.assertEqual(summary["rows_without_any_layer"], 1)
        self.assertEqual(summary["rows_with_any_added"], 2)
        self.assertEqual(summary["group_counts"]["pbp"], 2)
        self.assertEqual(summary["group_counts"]["season_ppp"], 1)
        self.assertEqual(summary["group_counts"]["blowout"], 2)
        self.assertEqual(summary["group_counts"]["usage"], 1)
        self.assertEqual(summary["group_counts"]["luck"], 1)

    def test_zero_true_misses_when_players_only_lack_specific_layers(self):
        frame = pd.DataFrame(
            [
                {"player": "A", "possessions": 0.0, "clean_pts": 1.2},
                {"player": "B", "possessions": 1.0, "clean_pts": 3.4, "adj_ppp": 1.05},
                {"player": "C", "possessions": 2.0, "clean_pts": 0.8, "usg_pct": 0.11},
            ]
        )
        layer_cols = ["possessions", "adj_ppp", "clean_pts", "usg_pct", "total_luck_score"]

        summary = _summarize_layer_feature_coverage(
            frame,
            layer_cols,
            added_layer_cols=["adj_ppp", "clean_pts", "usg_pct", "total_luck_score"],
        )

        self.assertEqual(summary["rows_with_any_layer"], 3)
        self.assertEqual(summary["rows_without_any_layer"], 0)
        self.assertEqual(summary["group_missing_counts"]["season_ppp"], 2)
        self.assertEqual(summary["group_missing_counts"]["blowout"], 0)


if __name__ == "__main__":
    unittest.main()
