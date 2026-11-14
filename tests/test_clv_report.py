import unittest

import numpy as np
import pandas as pd

import tempfile
from pathlib import Path
from unittest.mock import patch

import scripts.clv_report as clv
from scripts.clv_report import attach_clv, attach_outcomes, side_prices, summarize


def _snap(ts, line, over, under, commence="2026-10-31T23:10:00Z", source="odds_api_fanduel"):
    return {
        "game_date": "2026-10-31", "player": "Test Guard", "_pn": "test guard",
        "prop": "player_assists", "line": line, "source": source, "snapshot_ts": ts,
        "over_price": over, "under_price": under,
        "_snap": pd.Timestamp(ts), "_commence": pd.Timestamp(commence),
    }


def _row(line=4.5, direction="UNDER"):
    return pd.DataFrame([{
        "game_date": "2026-10-31", "player": "Test Guard", "_pn": "test guard",
        "stat": "ast", "odds_prop": "player_assists", "line": line, "direction": direction,
    }])


RUN = {"2026-10-31": (pd.Timestamp("2026-10-31T14:00:00Z"), None)}


class ClvReportTests(unittest.TestCase):
    def test_side_prices_no_vig(self):
        sp = side_prices(-110, -110, "UNDER")
        self.assertAlmostEqual(sp["no_vig"], 0.5)
        self.assertAlmostEqual(sp["decimal"], 1 + 100 / 110)

    def test_close_moving_toward_under_beats_close(self):
        lines = pd.DataFrame([
            _snap("2026-10-31T13:00:00Z", 4.5, -110, -110),   # bet snapshot
            _snap("2026-10-31T23:05:00Z", 4.5, +130, -160),   # T-5 close, under steamed
            _snap("2026-10-31T23:30:00Z", 4.5, +500, -900),   # after tip: must be ignored
        ])
        out = attach_clv(_row(), lines, RUN).iloc[0]
        close_nv = side_prices(130, -160, "UNDER")["no_vig"]
        self.assertEqual(out["close_quality"], "exact_line")
        self.assertEqual(out["close_price"], -160)
        self.assertAlmostEqual(out["ev_vs_close"], close_nv * (1 + 100 / 110) - 1)
        self.assertTrue(out["beat_close"])
        self.assertAlmostEqual(out["no_vig_move"], close_nv - 0.5)
        self.assertAlmostEqual(out["close_minutes_before_tip"], 5.0)

    def test_unchanged_price_costs_the_vig(self):
        lines = pd.DataFrame([
            _snap("2026-10-31T13:00:00Z", 4.5, -110, -110),
            _snap("2026-10-31T23:05:00Z", 4.5, -110, -110),
        ])
        out = attach_clv(_row(), lines, RUN).iloc[0]
        self.assertFalse(out["beat_close"])
        self.assertAlmostEqual(out["no_vig_move"], 0.0)
        self.assertLess(out["ev_vs_close"], 0)

    def test_line_move_scored_in_stat_units(self):
        lines = pd.DataFrame([
            _snap("2026-10-31T13:00:00Z", 4.5, -110, -110),
            _snap("2026-10-31T23:05:00Z", 3.5, -110, -110),  # line dropped: under 4.5 was the better number
        ])
        out = attach_clv(_row(), lines, RUN).iloc[0]
        self.assertEqual(out["close_quality"], "line_moved")
        self.assertAlmostEqual(out["line_clv"], 1.0)
        self.assertTrue(np.isnan(out["ev_vs_close"]))

    def test_snapshots_after_the_run_are_not_the_bet_price(self):
        lines = pd.DataFrame([
            _snap("2026-10-31T13:00:00Z", 4.5, -110, -110),
            _snap("2026-10-31T15:00:00Z", 4.5, +120, -150),  # after the 14:00 run
        ])
        out = attach_clv(_row(), lines, RUN).iloc[0]
        self.assertEqual(out["bet_price"], -110)
        self.assertEqual(out["close_price"], -150)

    def test_no_later_snapshot_is_excluded_from_summary(self):
        lines = pd.DataFrame([_snap("2026-10-31T13:00:00Z", 4.5, -110, -110)])
        out = attach_clv(_row(), lines, RUN)
        out["result"] = None
        out["profit_units"] = np.nan
        self.assertEqual(out.iloc[0]["close_quality"], "no_later_snapshot")
        summary = summarize(out, ["stat"])
        self.assertIsNone(summary["overall"])
        self.assertEqual(summary["close_quality_counts"], {"no_later_snapshot": 1})


    def test_publication_time_beats_later_same_day_rerun(self):
        lines = pd.DataFrame([
            _snap("2026-10-31T13:00:00Z", 4.5, -110, -110),   # morning price the pick saw
            _snap("2026-10-31T21:00:00Z", 4.5, +120, -150),   # evening price
            _snap("2026-10-31T23:05:00Z", 4.5, +130, -160),   # close
        ])
        evening_run = {"2026-10-31": (pd.Timestamp("2026-10-31T21:30:00Z"), None)}
        row = _row().assign(_published_utc=pd.Timestamp("2026-10-31T14:00:00Z"))
        out = attach_clv(row, lines, evening_run).iloc[0]
        self.assertEqual(out["bet_price"], -110)
        legacy = attach_clv(_row(), lines, evening_run).iloc[0]   # no published time: legacy fallback
        self.assertEqual(legacy["bet_price"], -150)

    def test_missing_exact_close_uses_primary_line_not_row_order(self):
        for order in (1, -1):
            close_rows = [
                _snap("2026-10-31T23:05:00Z", 10.5, -900, +550),  # alternate
                _snap("2026-10-31T23:05:00Z", 30.5, +600, -1000), # alternate
                _snap("2026-10-31T23:05:00Z", 21.5, -112, -108),  # primary (balanced)
            ][::order]
            lines = pd.DataFrame([_snap("2026-10-31T13:00:00Z", 20.5, -110, -110), *close_rows])
            out = attach_clv(_row(line=20.5), lines, RUN).iloc[0]
            self.assertEqual(out["close_quality"], "line_moved")
            self.assertEqual(out["close_line"], 21.5)
            self.assertAlmostEqual(out["line_clv"], -1.0)

    def test_incomplete_box_score_is_left_ungraded(self):
        box = pd.DataFrame([{"player": "Test Guard", "game_date": "2026-10-31",
                             "pts": 20, "trb": None, "ast": 5, "stl": 1, "blk": 0, "tov": 2}])
        rows = pd.DataFrame([
            {"game_date": "2026-10-31", "_pn": "test guard", "stat": "pr", "line": 25.5,
             "direction": d, "bet_decimal": 1.9} for d in ("OVER", "UNDER")
        ])
        with tempfile.TemporaryDirectory() as tmp:
            box.to_csv(Path(tmp) / "nba_data.csv", index=False)
            with patch.object(clv, "DATA_DIR", Path(tmp)):
                out = attach_outcomes(rows)
        self.assertTrue(out["result"].isna().all())
        self.assertTrue(out["profit_units"].isna().all())


if __name__ == "__main__":
    unittest.main()
