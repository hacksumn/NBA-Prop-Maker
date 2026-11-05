import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import nba_props


NOW = pd.Timestamp("2026-11-01T15:00:00Z")


def _lines(**overrides):
    row = {
        "player": "Darius Garland",
        "player_norm": "darius garland",
        "prop": "player_steals",
        "line": 1.5,
        "over_price": 174.0,
        "under_price": -240.0,
        "commence_time": "2026-11-02T01:40:00Z",
        "event_id": "ev1",
        "snapshot_ts": "2026-11-01T06:38:15-06:00",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _pick(dir_prob, direction="UNDER", prop="STL", line=1.5, player="Darius Garland"):
    return pd.DataFrame([{
        "player": player, "prop": prop, "direction": direction, "line": line,
        "dir_prob": dir_prob, "confidence": dir_prob,
        "break_even_prob": 0.5774, "exceeds_ev_threshold": True,
    }])


class FanDuelPriceGateTests(unittest.TestCase):
    def test_heavy_juice_under_is_blocked_despite_high_confidence(self):
        # The 2026-11-01 card: 72% model vs a -240 price (70.6% break-even).
        out = nba_props._attach_live_market_prices(_pick(72.0), _lines(), now_utc=NOW)
        row = out.iloc[0]
        self.assertAlmostEqual(row["break_even_prob"], 240 / 340, places=4)
        self.assertEqual(row["side_price"], -240.0)
        self.assertFalse(bool(row["price_gate_pass"]))
        self.assertFalse(bool(row["exceeds_ev_threshold"]))
        self.assertIn("cap", row["price_gate_reason"])

    def test_positive_ev_standard_price_passes(self):
        lines = _lines(prop="player_assists", line=3.5, over_price=-114.0, under_price=-114.0)
        out = nba_props._attach_live_market_prices(_pick(62.6, prop="AST", line=3.5), lines, now_utc=NOW)
        row = out.iloc[0]
        self.assertAlmostEqual(row["ev_per_unit"], round(0.626 * (1 + 100 / 114) - 1, 4), places=4)
        self.assertAlmostEqual(row["no_vig_side_prob"], 0.5, places=4)
        self.assertTrue(bool(row["price_gate_pass"]))
        self.assertEqual(row["event_id"], "ev1")

    def test_thin_ev_is_blocked(self):
        lines = _lines(prop="player_assists", line=3.5, over_price=-110.0, under_price=-110.0)
        out = nba_props._attach_live_market_prices(_pick(53.0, prop="AST", line=3.5), lines, now_utc=NOW)
        self.assertFalse(bool(out.iloc[0]["price_gate_pass"]))
        self.assertIn("EV", out.iloc[0]["price_gate_reason"])

    def test_unmatched_line_and_started_game_are_blocked(self):
        lines = _lines(prop="player_assists", line=3.5, over_price=-110.0, under_price=-110.0)
        wrong_line = nba_props._attach_live_market_prices(_pick(70.0, prop="AST", line=4.5), lines, now_utc=NOW)
        self.assertIn("no paired", wrong_line.iloc[0]["price_gate_reason"])
        started = nba_props._attach_live_market_prices(
            _pick(70.0, prop="AST", line=3.5),
            lines.assign(commence_time="2026-11-01T14:00:00Z"),
            now_utc=NOW,
        )
        self.assertIn("started", started.iloc[0]["price_gate_reason"])

    def test_punctuated_names_match_store_normalization(self):
        # Store player_norm uses a different normalizer than picks; match on raw names.
        lines = _lines(player="Collin Murray-Boyles", player_norm="collin murray boyles",
                       prop="player_assists", line=1.5, over_price=-125.0, under_price=-106.0)
        out = nba_props._attach_live_market_prices(
            _pick(60.0, prop="AST", line=1.5, player="Collin Murray-Boyles"), lines, now_utc=NOW
        )
        self.assertEqual(out.iloc[0]["side_price"], -106.0)
        self.assertNotIn("no paired", out.iloc[0]["price_gate_reason"])

    def test_one_sided_price_is_not_bettable(self):
        lines = _lines(prop="player_assists", line=3.5, over_price=float("nan"), under_price=-110.0)
        out = nba_props._attach_live_market_prices(_pick(70.0, prop="AST", line=3.5), lines, now_utc=NOW)
        self.assertFalse(bool(out.iloc[0]["price_gate_pass"]))

    def test_american_to_decimal(self):
        self.assertAlmostEqual(nba_props._american_to_decimal(-110), 1 + 100 / 110)
        self.assertAlmostEqual(nba_props._american_to_decimal(150), 2.5)
        self.assertTrue(pd.isna(nba_props._american_to_decimal(0)))
        self.assertTrue(pd.isna(nba_props._american_to_decimal(None)))


class SeasonStageTests(unittest.TestCase):
    def _history(self, last_date):
        return pd.DataFrame({"game_date": pd.to_datetime(["2026-10-21", last_date])})

    def test_regular_season_day_after_last_game_is_not_play_in(self):
        info = nba_props._infer_season_stage("2026-11-01", self._history("2026-10-31"))
        self.assertEqual(info["season_stage"], "regular_season")
        self.assertFalse(info["postseason_context"])

    def test_stale_logs_in_november_are_not_playoffs(self):
        info = nba_props._infer_season_stage("2026-11-10", self._history("2026-10-31"))
        self.assertEqual(info["season_stage"], "regular_season")

    def test_april_gap_after_regular_season_is_play_in(self):
        history = pd.DataFrame({"game_date": pd.to_datetime(["2026-01-10", "2026-04-12"])})
        info = nba_props._infer_season_stage("2026-04-14", history)
        self.assertEqual(info["season_stage"], "play_in")
        self.assertEqual(info["days_after_regular_end"], 2)


class TrackingOnlyLogTests(unittest.TestCase):
    def test_points_leans_logged_but_not_published(self):
        pred = pd.DataFrame([{
            "player": "Brandon Williams", "team": "GSW", "opp": "LAC",
            "pts_line": 14.5, "pts_pred": 13.0, "pts_p_over_cal": 0.40,
        }])
        lines = _lines(player="Brandon Williams", player_norm="brandon williams",
                       prop="player_points", line=14.5, over_price=-110.0, under_price=-120.0)
        original = nba_props.CONFIG["output_dir"]
        try:
            with TemporaryDirectory() as tmpdir:
                nba_props.CONFIG["output_dir"] = Path(tmpdir)
                # Pin the clock before the fixture's 2026-11-02T01:40Z tip-off: leans for
                # games already started are not recorded (DEC-070).
                with patch("nba_props._utc_now", return_value=pd.Timestamp("2026-11-01T18:00:00Z")):
                    out = nba_props._log_tracking_only_props(pred, lines, "2026-11-01")
                saved = pd.read_csv(Path(tmpdir) / "tracking_only_history.csv")
                self.assertFalse((Path(tmpdir) / "picks_history.csv").exists())
        finally:
            nba_props.CONFIG["output_dir"] = original
        self.assertEqual(out.iloc[0]["direction"], "UNDER")
        self.assertEqual(out.iloc[0]["side_price"], -120.0)
        self.assertEqual(saved.iloc[0]["opponent"], "LAC")
        self.assertTrue(bool(saved.iloc[0]["tracking_only"]))


if __name__ == "__main__":
    unittest.main()
