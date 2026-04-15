import unittest

import pandas as pd

from nba_props import _apply_consistency_scores, _apply_estimated_hit_rates, build_live_pick_card


class LiveCardSelectionTests(unittest.TestCase):
    def test_consistency_score_prefers_lower_volatility_and_higher_support(self):
        picks = pd.DataFrame(
            [
                {
                    "player": "Stable Guard",
                    "prop": "AST",
                    "direction": "UNDER",
                    "selection_priority": 0.5,
                    "dir_prob": 72.0,
                    "meta_prob": 66.0,
                    "confidence": 71.0,
                    "rank_score": 0.68,
                    "edge": -1.1,
                    "min_edge_req": 0.9,
                    "sigma": 1.40,
                    "volatility_metric": 1.40,
                    "support_agreement_count": 3,
                    "edge_disagreement": 0.20,
                    "regime_score": 0.10,
                },
                {
                    "player": "Wild Guard",
                    "prop": "AST",
                    "direction": "UNDER",
                    "selection_priority": 0.5,
                    "dir_prob": 72.0,
                    "meta_prob": 66.0,
                    "confidence": 74.0,
                    "rank_score": 0.69,
                    "edge": -1.3,
                    "min_edge_req": 0.9,
                    "sigma": 2.90,
                    "volatility_metric": 2.90,
                    "support_agreement_count": 1,
                    "edge_disagreement": 0.90,
                    "regime_score": 0.25,
                },
            ]
        )

        scored = _apply_consistency_scores(picks)
        stable = scored.loc[scored["player"] == "Stable Guard", "consistency_score"].iloc[0]
        wild = scored.loc[scored["player"] == "Wild Guard", "consistency_score"].iloc[0]

        self.assertGreater(stable, wild)

    def test_estimated_hit_rate_prefers_history_backed_profiles_over_confidence(self):
        history_rows = []
        for idx in range(12):
            history_rows.append(
                {
                    "player": f"AST Winner {idx}",
                    "prop": "AST",
                    "direction": "UNDER",
                    "result": "WIN",
                    "meta_prob": 58.0 + idx,
                    "dir_prob": 72.0 + idx,
                    "selection_priority": 1.0,
                }
            )
        for idx in range(12):
            history_rows.append(
                {
                    "player": f"PTS Loser {idx}",
                    "prop": "PTS",
                    "direction": "OVER",
                    "result": "LOSS",
                    "meta_prob": 35.0 + idx,
                    "dir_prob": 60.0 + idx,
                    "selection_priority": 0.0,
                }
            )
        history = pd.DataFrame(history_rows)

        picks = pd.DataFrame(
            [
                {
                    "player": "Stable Play",
                    "prop": "AST",
                    "direction": "UNDER",
                    "meta_prob": 64.0,
                    "dir_prob": 80.0,
                    "selection_priority": 1.0,
                    "confidence": 68.0,
                },
                {
                    "player": "Fragile Play",
                    "prop": "PTS",
                    "direction": "OVER",
                    "meta_prob": 40.0,
                    "dir_prob": 65.0,
                    "selection_priority": 0.0,
                    "confidence": 86.0,
                },
            ]
        )

        scored = _apply_estimated_hit_rates(picks, history_df=history)
        stable = scored.loc[scored["player"] == "Stable Play", "estimated_hit_rate"].iloc[0]
        fragile = scored.loc[scored["player"] == "Fragile Play", "estimated_hit_rate"].iloc[0]

        self.assertGreater(stable, fragile)

    def test_build_live_pick_card_enforces_accuracy_gates_and_unique_players(self):
        picks = pd.DataFrame(
            [
                {
                    "player": "Alpha",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 72.0,
                    "selection_priority": 1.0,
                    "consistency_score": 0.92,
                    "rank_score": 0.71,
                    "meta_prob": 52.0,
                    "volatility_metric": 1.40,
                    "dir_prob": 71.0,
                    "confidence": 74.0,
                    "edge": -1.2,
                    "pick_source": "market_model",
                },
                {
                    "player": "Alpha",
                    "prop": "TRB",
                    "direction": "UNDER",
                    "estimated_hit_rate": 71.5,
                    "selection_priority": 0.7,
                    "consistency_score": 0.89,
                    "rank_score": 0.73,
                    "meta_prob": 54.0,
                    "volatility_metric": 2.30,
                    "dir_prob": 69.0,
                    "confidence": 76.0,
                    "edge": -1.1,
                    "pick_source": "market_model",
                },
                {
                    "player": "Beta",
                    "prop": "STL",
                    "direction": "UNDER",
                    "estimated_hit_rate": 70.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.88,
                    "rank_score": 0.72,
                    "meta_prob": 54.0,
                    "volatility_metric": 0.80,
                    "dir_prob": 78.0,
                    "confidence": 79.0,
                    "edge": -0.8,
                    "pick_source": "market_model",
                },
                {
                    "player": "Gamma",
                    "prop": "BLK",
                    "direction": "UNDER",
                    "estimated_hit_rate": 69.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.83,
                    "rank_score": 0.70,
                    "meta_prob": 53.0,
                    "volatility_metric": 0.80,
                    "dir_prob": 77.0,
                    "confidence": 78.0,
                    "edge": -0.7,
                    "pick_source": "market_model",
                },
                {
                    "player": "Delta",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 68.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.81,
                    "rank_score": 0.69,
                    "meta_prob": 40.0,
                    "volatility_metric": 1.30,
                    "dir_prob": 68.0,
                    "confidence": 73.0,
                    "edge": -0.9,
                    "pick_source": "market_model",
                },
                {
                    "player": "Epsilon",
                    "prop": "TRB",
                    "direction": "UNDER",
                    "estimated_hit_rate": 90.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.62,
                    "rank_score": 0.75,
                    "meta_prob": 66.0,
                    "volatility_metric": 4.90,
                    "dir_prob": 80.0,
                    "confidence": 82.0,
                    "edge": -1.6,
                    "pick_source": "market_model",
                },
                {
                    "player": "Zeta",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 67.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.77,
                    "rank_score": 0.68,
                    "meta_prob": 48.0,
                    "volatility_metric": 1.80,
                    "dir_prob": 69.0,
                    "confidence": 72.0,
                    "edge": -0.8,
                    "pick_source": "market_model",
                },
                {
                    "player": "Theta",
                    "prop": "TRB",
                    "direction": "UNDER",
                    "estimated_hit_rate": 66.5,
                    "selection_priority": 0.0,
                    "consistency_score": 0.76,
                    "rank_score": 0.67,
                    "meta_prob": 49.0,
                    "volatility_metric": 2.60,
                    "dir_prob": 68.0,
                    "confidence": 71.0,
                    "edge": -0.9,
                    "pick_source": "market_model",
                },
                {
                    "player": "Volume Fill",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 80.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.95,
                    "rank_score": 0.80,
                    "meta_prob": 60.0,
                    "volatility_metric": 1.20,
                    "dir_prob": 82.0,
                    "confidence": 82.0,
                    "edge": -1.0,
                    "pick_source": "volume_fill",
                },
            ]
        )

        card = build_live_pick_card(picks, target_picks=5, min_picks=5)

        self.assertEqual(card["player"].tolist(), ["Alpha", "Beta", "Zeta", "Theta", "Gamma"])
        self.assertNotIn("Delta", card["player"].tolist())
        self.assertNotIn("Epsilon", card["player"].tolist())
        self.assertNotIn("Volume Fill", card["player"].tolist())
        self.assertTrue(card["accuracy_gate_pass"].all())

    def test_build_live_pick_card_backfills_to_minimum_five_with_relaxed_market_model_tier(self):
        picks = pd.DataFrame(
            [
                {
                    "player": "Weak A",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 61.0,
                    "selection_priority": 1.0,
                    "consistency_score": 0.88,
                    "rank_score": 0.74,
                    "meta_prob": 55.0,
                    "volatility_metric": 1.20,
                    "dir_prob": 76.0,
                    "confidence": 78.0,
                    "edge": -1.1,
                    "pick_source": "market_model",
                },
                {
                    "player": "Weak B",
                    "prop": "TRB",
                    "direction": "UNDER",
                    "estimated_hit_rate": 60.5,
                    "selection_priority": 0.7,
                    "consistency_score": 0.86,
                    "rank_score": 0.72,
                    "meta_prob": 57.0,
                    "volatility_metric": 2.20,
                    "dir_prob": 72.0,
                    "confidence": 75.0,
                    "edge": -1.3,
                    "pick_source": "market_model",
                },
                {
                    "player": "Weak C",
                    "prop": "TRB",
                    "direction": "UNDER",
                    "estimated_hit_rate": 60.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.84,
                    "rank_score": 0.71,
                    "meta_prob": 58.0,
                    "volatility_metric": 2.40,
                    "dir_prob": 70.0,
                    "confidence": 74.0,
                    "edge": -1.2,
                    "pick_source": "market_model",
                },
                {
                    "player": "Weak D",
                    "prop": "STL",
                    "direction": "UNDER",
                    "estimated_hit_rate": 59.5,
                    "selection_priority": 0.0,
                    "consistency_score": 0.83,
                    "rank_score": 0.70,
                    "meta_prob": 54.0,
                    "volatility_metric": 0.80,
                    "dir_prob": 79.0,
                    "confidence": 73.0,
                    "edge": -0.8,
                    "pick_source": "market_model",
                },
                {
                    "player": "Weak E",
                    "prop": "BLK",
                    "direction": "UNDER",
                    "estimated_hit_rate": 59.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.82,
                    "rank_score": 0.69,
                    "meta_prob": 53.0,
                    "volatility_metric": 0.80,
                    "dir_prob": 77.0,
                    "confidence": 72.0,
                    "edge": -0.7,
                    "pick_source": "market_model",
                },
                {
                    "player": "Volume Reserve",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 75.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.95,
                    "rank_score": 0.81,
                    "meta_prob": 62.0,
                    "volatility_metric": 1.10,
                    "dir_prob": 84.0,
                    "confidence": 83.0,
                    "edge": -1.0,
                    "pick_source": "volume_fill",
                },
            ]
        )

        card = build_live_pick_card(picks, target_picks=5, min_picks=5)

        self.assertEqual(card["player"].tolist(), ["Weak A", "Weak B", "Weak C", "Weak D", "Weak E"])
        self.assertEqual(card["publish_tier"].tolist(), [
            "market_lane_relaxed",
            "market_lane_relaxed",
            "market_lane_relaxed",
            "market_lane_relaxed",
            "market_relaxed",
        ])
        self.assertNotIn("Volume Reserve", card["player"].tolist())

    def test_build_live_pick_card_prefers_top_tier_lane_over_non_lane(self):
        picks = pd.DataFrame(
            [
                {
                    "player": "Lane Pick",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 66.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.70,
                    "rank_score": 0.70,
                    "meta_prob": 55.0,
                    "volatility_metric": 1.60,
                    "dir_prob": 70.0,
                    "confidence": 71.0,
                    "edge": -0.9,
                    "pick_source": "market_model",
                    "projection_type": "demon",
                },
                {
                    "player": "Non Lane Pick",
                    "prop": "BLK",
                    "direction": "UNDER",
                    "estimated_hit_rate": 69.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.78,
                    "rank_score": 0.74,
                    "meta_prob": 58.0,
                    "volatility_metric": 0.80,
                    "dir_prob": 79.0,
                    "confidence": 78.0,
                    "edge": -0.8,
                    "pick_source": "market_model",
                    "projection_type": "demon",
                },
            ]
        )

        card = build_live_pick_card(picks, target_picks=1, min_picks=1)

        self.assertEqual(card["player"].tolist(), ["Lane Pick"])
        self.assertEqual(card["publish_tier"].tolist(), ["core_lane"])

    def test_build_live_pick_card_prefers_standard_lines_when_available(self):
        picks = pd.DataFrame(
            [
                {
                    "player": "Standard Pick",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 67.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.72,
                    "rank_score": 0.72,
                    "meta_prob": 56.0,
                    "volatility_metric": 1.70,
                    "dir_prob": 71.0,
                    "confidence": 72.0,
                    "edge": -0.9,
                    "pick_source": "market_model",
                    "projection_type": "standard",
                },
                {
                    "player": "Demon Pick",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 67.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.72,
                    "rank_score": 0.72,
                    "meta_prob": 56.0,
                    "volatility_metric": 1.70,
                    "dir_prob": 71.0,
                    "confidence": 72.0,
                    "edge": -0.9,
                    "pick_source": "market_model",
                    "projection_type": "demon",
                },
            ]
        )

        card = build_live_pick_card(picks, target_picks=1, min_picks=1)

        self.assertEqual(card["player"].tolist(), ["Standard Pick"])

    def test_build_live_pick_card_prefers_stable_roles_in_postseason_context(self):
        history_rows = []
        for idx in range(12):
            history_rows.append(
                {
                    "player": f"AST Winner {idx}",
                    "prop": "AST",
                    "direction": "UNDER",
                    "result": "WIN",
                    "meta_prob": 58.0 + idx,
                    "dir_prob": 72.0 + idx,
                    "selection_priority": 1.0,
                    "game_date": "2026-04-13",
                }
            )
        for idx in range(12):
            history_rows.append(
                {
                    "player": f"PTS Loser {idx}",
                    "prop": "PTS",
                    "direction": "OVER",
                    "result": "LOSS",
                    "meta_prob": 40.0 + idx,
                    "dir_prob": 58.0 + idx,
                    "selection_priority": 0.0,
                    "game_date": "2026-04-13",
                }
            )
        history = pd.DataFrame(history_rows)

        picks = pd.DataFrame(
            [
                {
                    "player": "Veteran Starter",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 65.5,
                    "selection_priority": 0.0,
                    "consistency_score": 0.82,
                    "rank_score": 0.72,
                    "meta_prob": 56.0,
                    "volatility_metric": 1.50,
                    "dir_prob": 72.0,
                    "confidence": 74.0,
                    "edge": -0.9,
                    "pick_source": "market_model",
                    "projection_type": "standard",
                    "games_played": 210,
                    "season_games_played": 72,
                    "mp_predicted": 36.0,
                    "mp_l10": 35.0,
                },
                {
                    "player": "Fragile Flier",
                    "prop": "AST",
                    "direction": "UNDER",
                    "estimated_hit_rate": 67.0,
                    "selection_priority": 0.0,
                    "consistency_score": 0.61,
                    "rank_score": 0.74,
                    "meta_prob": 56.0,
                    "volatility_metric": 2.10,
                    "dir_prob": 73.0,
                    "confidence": 75.0,
                    "edge": -1.0,
                    "pick_source": "market_model",
                    "projection_type": "demon",
                    "games_played": 28,
                    "season_games_played": 18,
                    "mp_predicted": 23.0,
                    "mp_l10": 29.0,
                },
            ]
        )

        card = build_live_pick_card(
            picks,
            target_picks=2,
            min_picks=2,
            slate_date="2026-04-17",
            history_df=history,
        )

        self.assertEqual(card["season_stage"].tolist(), ["play_in", "play_in"])
        self.assertEqual(card["player"].tolist(), ["Veteran Starter", "Fragile Flier"])
        self.assertGreater(
            card.loc[card["player"] == "Veteran Starter", "postseason_pick_score"].iloc[0],
            card.loc[card["player"] == "Fragile Flier", "postseason_pick_score"].iloc[0],
        )


if __name__ == "__main__":
    unittest.main()
