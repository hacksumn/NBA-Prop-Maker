import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import nba_props
from probability_utils import over_probability_from_quantiles, poisson_over_probability


QUANTS = {"ast_q10": 1.4, "ast_q25": 2.6, "ast_q50": 3.8, "ast_q75": 5.0, "ast_q90": 6.2}  # raw P(over 4.5) ~0.35


def _pred():
    return pd.DataFrame([{"player": "Test Guard", **QUANTS}])


def _pick(line=3.5, direction="UNDER"):
    return pd.DataFrame([{"player": "Test Guard", "prop": "AST", "direction": direction,
                          "line": line, "dir_prob": 60.0}])


def _line(line, over, under, prop="player_assists_alternate"):
    return {"player": "Test Guard", "prop": prop, "line": line, "over_price": over,
            "under_price": under, "snapshot_ts": "s1", "game_date": "2026-11-01"}


def _raw(line):
    q = np.array([list(QUANTS.values())])
    return float(over_probability_from_quantiles([line], q)[0])


class LinePricingTests(unittest.TestCase):
    def test_raw_probability_from_quantiles_tails_and_rate(self):
        row = _pred().iloc[0]
        self.assertAlmostEqual(nba_props._line_raw_over_probability(row, "ast", 3.5), _raw(3.5))
        tails = pd.Series({"stl_ge_1": 0.6, "stl_ge_2": 0.25, "stl_ge_3": 0.08})
        self.assertAlmostEqual(nba_props._line_raw_over_probability(tails, "stl", 1.5), 0.25)
        rate = pd.Series({"blk_rate": 1.2})
        self.assertAlmostEqual(nba_props._line_raw_over_probability(rate, "blk", 0.5),
                               float(poisson_over_probability([0.5], [1.2])[0]))
        self.assertTrue(np.isnan(nba_props._line_raw_over_probability(pd.Series({"x": 1}), "ast", 3.5)))

    def test_identity_calibration_returns_raw(self):
        row = _pred().iloc[0]
        self.assertAlmostEqual(nba_props._line_over_probability(row, "ast", 3.5, calibrators={}), _raw(3.5))


class BestContractTests(unittest.TestCase):
    def _choose(self, lines, band=(0.30, 0.70)):
        with patch.dict(nba_props.CONFIG, {"live_alt_line_raw_prob_band": band,
                                           "live_alt_line_max_distance": 2.0,
                                           "live_max_favorite_american": -200}):
            return nba_props._select_best_contract_lines(_pick(), _pred(), pd.DataFrame(lines), calibrators={})

    def test_moves_to_higher_ev_alternate_in_band(self):
        lines = [_line(3.5, -110, -110, prop="player_assists"), _line(4.5, 150, -190)]
        out = self._choose(lines).iloc[0]
        ev_main = (1 - _raw(3.5)) * (1 + 100 / 110) - 1
        ev_alt = (1 - _raw(4.5)) * (1 + 100 / 190) - 1
        self.assertGreater(ev_alt, ev_main)  # fixture sanity
        self.assertEqual(out["line"], 4.5)
        self.assertEqual(out["main_line"], 3.5)
        self.assertEqual(out["line_choice"], "alternate")
        self.assertAlmostEqual(out["dir_prob"], round((1 - _raw(4.5)) * 100, 1))

    def test_out_of_band_alternate_is_ignored(self):
        lines = [_line(3.5, -110, -110, prop="player_assists"), _line(4.5, 150, -190)]
        out = self._choose(lines, band=(0.40, 0.60)).iloc[0]  # raw P(over 4.5) is below 0.40
        self.assertEqual(out["line"], 3.5)
        self.assertEqual(out["line_choice"], "main")

    def test_heavy_favorite_and_one_sided_alternates_are_ignored(self):
        lines = [_line(3.5, -110, -110, prop="player_assists"),
                 _line(4.5, 190, -250),             # beyond -200 cap
                 _line(2.5, float("nan"), 120)]     # no paired over price
        out = self._choose(lines).iloc[0]
        self.assertEqual(out["line"], 3.5)

    def test_no_lines_keeps_main(self):
        out = nba_props._select_best_contract_lines(_pick(), _pred(), pd.DataFrame(), calibrators={})
        self.assertEqual(out.iloc[0]["line_choice"], "main")


class AlternateStoreTests(unittest.TestCase):
    def test_save_and_load_round_trip_by_snapshot(self):
        original = nba_props.CONFIG["data_dir"]
        try:
            with TemporaryDirectory() as tmp:
                nba_props.CONFIG["data_dir"] = Path(tmp)
                nba_props._save_alternate_lines(pd.DataFrame([_line(4.5, 150, -190)]))
                nba_props._save_alternate_lines(pd.DataFrame([{**_line(4.5, 140, -180), "snapshot_ts": "s2"}]))
                self.assertEqual(len(nba_props._load_alternate_lines("2026-11-01")), 2)
                s2 = nba_props._load_alternate_lines("2026-11-01", snapshot_ts="s2")
                self.assertEqual(s2.iloc[0]["over_price"], 140)
                self.assertFalse((Path(tmp) / "historical_lines.csv").exists())
        finally:
            nba_props.CONFIG["data_dir"] = original

    def test_alternate_fetch_is_off_by_default(self):
        self.assertFalse(nba_props.CONFIG["live_fetch_alternate_markets"])


if __name__ == "__main__":
    unittest.main()
