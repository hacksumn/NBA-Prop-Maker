import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import nba_props


class BetslipEVSelectionTests(unittest.TestCase):
    def test_two_pick_power_play_prefers_higher_ev_same_team_pair(self):
        picks = pd.DataFrame(
            [
                {
                    "player": "Alpha Guard",
                    "team": "DAL",
                    "opponent": "LAL",
                    "prop": "AST",
                    "direction": "UNDER",
                    "line": 7.5,
                    "confidence": 82.0,
                    "dir_prob": 0.60,
                    "edge": -1.8,
                    "pick_source": "market_model",
                    "selection_priority": 0.0,
                },
                {
                    "player": "Beta Wing",
                    "team": "DAL",
                    "opponent": "LAL",
                    "prop": "TRB",
                    "direction": "UNDER",
                    "line": 6.5,
                    "confidence": 81.0,
                    "dir_prob": 0.60,
                    "edge": -1.7,
                    "pick_source": "market_model",
                    "selection_priority": 0.0,
                },
                {
                    "player": "Gamma Center",
                    "team": "DEN",
                    "opponent": "PHX",
                    "prop": "AST",
                    "direction": "UNDER",
                    "line": 5.5,
                    "confidence": 90.0,
                    "dir_prob": 0.62,
                    "edge": -1.5,
                    "pick_source": "market_model",
                    "selection_priority": 0.0,
                },
                {
                    "player": "Delta Forward",
                    "team": "BOS",
                    "opponent": "NYK",
                    "prop": "TRB",
                    "direction": "UNDER",
                    "line": 8.5,
                    "confidence": 78.0,
                    "dir_prob": 0.58,
                    "edge": -1.4,
                    "pick_source": "market_model",
                    "selection_priority": 0.0,
                },
            ]
        )

        original_output_dir = nba_props.CONFIG["output_dir"]
        with TemporaryDirectory() as tmpdir:
            nba_props.CONFIG["output_dir"] = Path(tmpdir)
            slips = nba_props.log_betslips(picks, "2026-04-12", wager=20.0)
        nba_props.CONFIG["output_dir"] = original_output_dir

        pair = slips[slips["slip_type"] == "2-pick Power Play"].iloc[0]
        selected_players = {pair["Pick_1_Player"], pair["Pick_2_Player"]}

        self.assertEqual(selected_players, {"Alpha Guard", "Beta Wing"})
        self.assertIn("joint_prob_adj", slips.columns)
        self.assertIn("ev_per_unit", slips.columns)
        self.assertGreater(float(pair["estimated_correlation"]), 0.0)
        self.assertGreater(float(pair["ev_per_unit"]), 0.0)


if __name__ == "__main__":
    unittest.main()
