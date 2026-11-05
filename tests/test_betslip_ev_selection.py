import io
import unittest
from pathlib import Path
from unittest.mock import patch
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
        original_mode = nba_props.CONFIG["live_line_source_mode"]
        try:
            with TemporaryDirectory() as tmpdir:
                nba_props.CONFIG["output_dir"] = Path(tmpdir)
                nba_props.CONFIG["live_line_source_mode"] = "prizepicks"
                slips = nba_props.log_betslips(picks, "2026-04-12", wager=20.0)
        finally:
            nba_props.CONFIG["output_dir"] = original_output_dir
            nba_props.CONFIG["live_line_source_mode"] = original_mode

        pair = slips[slips["slip_type"] == "2-pick Power Play"].iloc[0]
        selected_players = {pair["Pick_1_Player"], pair["Pick_2_Player"]}

        self.assertEqual(selected_players, {"Alpha Guard", "Beta Wing"})
        self.assertIn("joint_prob_adj", slips.columns)
        self.assertIn("ev_per_unit", slips.columns)
        self.assertGreater(float(pair["estimated_correlation"]), 0.0)
        self.assertGreater(float(pair["ev_per_unit"]), 0.0)


    def _fanduel_leg(self, player, event_id, price, dir_prob, team):
        decimal = nba_props._american_to_decimal(price)
        return {
            "player": player, "team": team, "opponent": "", "prop": "AST",
            "direction": "UNDER", "line": 4.5, "confidence": 75.0, "dir_prob": dir_prob,
            "edge": -1.0, "pick_source": "market_model", "selection_priority": 0.0,
            "side_price": price, "side_decimal_odds": decimal, "event_id": event_id,
        }

    def _log_fanduel(self, picks):
        original_output_dir = nba_props.CONFIG["output_dir"]
        original_mode = nba_props.CONFIG["live_line_source_mode"]
        try:
            with TemporaryDirectory() as tmpdir:
                nba_props.CONFIG["output_dir"] = Path(tmpdir)
                nba_props.CONFIG["live_line_source_mode"] = "fanduel"
                return nba_props.log_betslips(picks, "2026-11-01", wager=20.0)
        finally:
            nba_props.CONFIG["output_dir"] = original_output_dir
            nba_props.CONFIG["live_line_source_mode"] = original_mode

    def test_fanduel_parlay_priced_as_product_of_leg_odds(self):
        picks = pd.DataFrame([
            self._fanduel_leg("A Player", "ev1", -110, 62.0, "DAL"),
            self._fanduel_leg("B Player", "ev2", +120, 55.0, "BOS"),
        ])
        slips = self._log_fanduel(picks)
        pair = slips[slips["slip_type"] == "2-pick FanDuel Parlay"].iloc[0]
        expected_mult = (1 + 100 / 110) * 2.2
        self.assertAlmostEqual(float(pair["potential_payout"]), round(20 * expected_mult, 2), places=2)
        self.assertAlmostEqual(float(pair["break_even_joint_prob"]), round(1 / expected_mult, 4), places=4)
        # dir_prob is a percentage on pick rows; joint prob must use 0.62*0.55, not 0.99*0.99.
        self.assertAlmostEqual(float(pair["joint_prob_raw"]), 0.62 * 0.55, places=3)
        self.assertEqual(float(pair["Pick_1_Price"]) in (-110.0, 120.0), True)

    def test_fanduel_parlay_rejects_same_game_and_unpriced_legs(self):
        same_game = pd.DataFrame([
            self._fanduel_leg("A Player", "ev1", -110, 62.0, "DAL"),
            self._fanduel_leg("B Player", "ev1", +120, 55.0, "LAL"),
        ])
        self.assertEqual(len(self._log_fanduel(same_game)), 0)
        unpriced = pd.DataFrame([
            self._fanduel_leg("A Player", "ev1", -110, 62.0, "DAL"),
            self._fanduel_leg("B Player", "ev2", float("nan"), 55.0, "BOS"),
        ])
        self.assertEqual(len(self._log_fanduel(unpriced)), 0)

    def test_zero_slips_logs_confidence_gate_reason(self):
        low_a = self._fanduel_leg("A Player", "ev1", -110, 62.0, "DAL")
        low_b = self._fanduel_leg("B Player", "ev2", +120, 55.0, "BOS")
        low_a["confidence"] = 68.8
        low_b["confidence"] = 61.1
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            slips = self._log_fanduel(pd.DataFrame([low_a, low_b]))
        self.assertEqual(len(slips), 0)
        self.assertIn("0 at confidence >= 70 (max 68.8)", out.getvalue())

    def test_zero_slips_logs_unpriceable_pool_reason(self):
        same_game = pd.DataFrame([
            self._fanduel_leg("A Player", "ev1", -110, 62.0, "DAL"),
            self._fanduel_leg("B Player", "ev1", +120, 55.0, "LAL"),
        ])
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self._log_fanduel(same_game)
        self.assertIn("2 eligible legs, but no combination could be priced", out.getvalue())


if __name__ == "__main__":
    unittest.main()
