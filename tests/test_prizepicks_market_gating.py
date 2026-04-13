import unittest

import pandas as pd

from nba_props import filter_best_picks


class PrizePicksMarketGatingTests(unittest.TestCase):
    def test_unplaceable_goblin_demon_and_promo_lines_are_filtered(self):
        quality_policy = {
            "ast": {
                "allowed": True,
                "allow_under": True,
                "allow_over": True,
                "min_edge": 0.5,
                "min_edge_under": 0.5,
                "min_edge_over": 0.5,
                "rmse": 1.0,
                "shrunk_acc": 0.60,
                "min_prob": 0.0,
                "meta_gate_mode": "soft",
                "priority_under_lane": False,
            }
        }
        pred_df = pd.DataFrame(
            [
                {
                    "player": "Standard Under",
                    "team": "DAL",
                    "ast_line": 6.0,
                    "ast_pred": 4.0,
                    "ast_p_over_cal": 0.20,
                    "ast_projection_type": "standard",
                    "ast_is_promo": False,
                },
                {
                    "player": "Goblin Under",
                    "team": "DAL",
                    "ast_line": 6.0,
                    "ast_pred": 4.0,
                    "ast_p_over_cal": 0.20,
                    "ast_projection_type": "goblin",
                    "ast_is_promo": False,
                },
                {
                    "player": "Promo Under",
                    "team": "DAL",
                    "ast_line": 6.0,
                    "ast_pred": 4.0,
                    "ast_p_over_cal": 0.20,
                    "ast_projection_type": "standard",
                    "ast_is_promo": True,
                },
                {
                    "player": "Standard Over",
                    "team": "DAL",
                    "ast_line": 6.0,
                    "ast_pred": 8.0,
                    "ast_p_over_cal": 0.80,
                    "ast_projection_type": "standard",
                    "ast_is_promo": False,
                },
                {
                    "player": "Demon Over",
                    "team": "DAL",
                    "ast_line": 6.0,
                    "ast_pred": 8.0,
                    "ast_p_over_cal": 0.80,
                    "ast_projection_type": "demon",
                    "ast_is_promo": False,
                },
            ]
        )

        picks = filter_best_picks(pred_df, quality_policy=quality_policy, max_picks=10)

        self.assertEqual(set(picks["player"]), {"Standard Under", "Standard Over"})
        self.assertEqual(
            picks.set_index("player")["projection_type"].to_dict(),
            {"Standard Under": "standard", "Standard Over": "standard"},
        )
        self.assertEqual(
            picks.set_index("player")["is_promo"].to_dict(),
            {"Standard Under": False, "Standard Over": False},
        )


if __name__ == "__main__":
    unittest.main()
