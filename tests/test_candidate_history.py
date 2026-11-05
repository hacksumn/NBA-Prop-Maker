"""Every final candidate (published or gated out) is logged and graded (tracking only)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import nba_props
import run_daily


def _candidate(player, prop="AST", direction="UNDER", line=4.5, price_pass=True,
               reason="", commence="2026-11-05T00:00:00Z", decimal=1.91, price=-110):
    return {
        "player": player, "team": "DET", "opponent": "PHI", "prop": prop,
        "direction": direction, "line": line, "confidence": 70.0,
        "side_price": price, "side_decimal_odds": decimal, "ev_per_unit": 0.05,
        "price_gate_pass": price_pass, "price_gate_reason": reason,
        "accuracy_gate_pass": True, "gate_failures": "", "line_source": "fanduel",
        "commence_time": commence,
    }


class LogCandidateHistoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self._cfg = patch.dict(nba_props.CONFIG, {"output_dir": self.out})
        self._cfg.start()
        self._print = patch("builtins.print")
        self._print.start()
        self.pre_tip = pd.Timestamp("2026-11-04T18:00:00Z")

    def tearDown(self):
        self._print.stop()
        self._cfg.stop()
        self._tmp.cleanup()

    def _read(self):
        return pd.read_csv(self.out / "candidates_history.csv", dtype=str, keep_default_na=False)

    def test_buckets_published_price_and_other(self):
        cands = pd.DataFrame([
            _candidate("A Pub"),
            _candidate("B Price", price_pass=False, reason="EV +0.010<+0.030"),
            _candidate("C Other"),
        ])
        published = pd.DataFrame([_candidate("A Pub")])
        nba_props._log_candidate_history(cands, published, "2026-11-04", now_utc=self.pre_tip)
        hist = self._read().set_index("player")
        self.assertEqual(hist.loc["A Pub", "gate_bucket"], "published")
        self.assertEqual(hist.loc["B Price", "gate_bucket"], "blocked_price")
        self.assertEqual(hist.loc["C Other", "gate_bucket"], "blocked_other")
        self.assertEqual(hist.loc["A Pub", "game_date"], "2026-11-04")
        self.assertEqual(hist.loc["A Pub", "line"], "4.5")

    def test_started_games_are_not_logged(self):
        cands = pd.DataFrame([
            _candidate("Late", commence="2026-11-04T17:00:00Z"),
            _candidate("Early ok"),
        ])
        nba_props._log_candidate_history(cands, pd.DataFrame(), "2026-11-04", now_utc=self.pre_tip)
        self.assertEqual(self._read()["player"].tolist(), ["Early ok"])

    def test_late_rerun_cannot_overwrite_pregame_record(self):
        nba_props._log_candidate_history(
            pd.DataFrame([_candidate("A")]), pd.DataFrame([_candidate("A")]),
            "2026-11-04", now_utc=self.pre_tip,
        )
        after_tip = pd.Timestamp("2026-11-05T05:07:00Z")
        nba_props._log_candidate_history(
            pd.DataFrame([_candidate("A", price_pass=False, reason="game started")]),
            pd.DataFrame(), "2026-11-04", now_utc=after_tip,
        )
        hist = self._read()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist.loc[0, "gate_bucket"], "published")

    def test_upsert_keeps_absent_rows_and_published_is_sticky(self):
        nba_props._log_candidate_history(
            pd.DataFrame([_candidate("A"), _candidate("B")]), pd.DataFrame([_candidate("A")]),
            "2026-11-04", now_utc=self.pre_tip,
        )
        nba_props._log_candidate_history(
            pd.DataFrame([_candidate("A", price_pass=False, reason="EV moved")]), pd.DataFrame(),
            "2026-11-04", now_utc=self.pre_tip + pd.Timedelta(hours=1),
        )
        hist = self._read().set_index("player")
        self.assertEqual(sorted(hist.index), ["A", "B"])
        self.assertEqual(hist.loc["A", "published"], "True")
        self.assertEqual(hist.loc["A", "gate_bucket"], "published")
        self.assertEqual(hist.loc["A", "price_gate_reason"], "EV moved")

    def test_graded_rows_are_never_modified(self):
        nba_props._log_candidate_history(
            pd.DataFrame([_candidate("A")]), pd.DataFrame(), "2026-11-04", now_utc=self.pre_tip
        )
        hist = self._read()
        hist.loc[0, "result"] = "WIN"
        hist.to_csv(self.out / "candidates_history.csv", index=False)
        nba_props._log_candidate_history(
            pd.DataFrame([_candidate("A", price_pass=False)]), pd.DataFrame(),
            "2026-11-04", now_utc=self.pre_tip,
        )
        hist = self._read()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist.loc[0, "result"], "WIN")
        self.assertEqual(hist.loc[0, "gate_bucket"], "blocked_other")

    def test_failure_never_raises(self):
        with patch("nba_props._atomic_csv", side_effect=OSError("disk full")):
            out = nba_props._log_candidate_history(
                pd.DataFrame([_candidate("A")]), pd.DataFrame(), "2026-11-04", now_utc=self.pre_tip
            )
        self.assertTrue(out.empty)


class GradeCandidateTests(unittest.TestCase):
    @staticmethod
    def _norm(name):
        return str(name).lower().strip()

    def _hist(self, rows):
        base = {"result": "", "actual": "", "profit_units": "", "graded_at": "",
                "side_decimal_odds": "1.91", "side_price": "-110"}
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_win_loss_push_dnp_and_profit(self):
        hist = self._hist([
            {"game_date": "2026-11-03", "player": "A", "prop": "AST", "direction": "UNDER", "line": "4.5", "gate_bucket": "blocked_price"},
            {"game_date": "2026-11-03", "player": "B", "prop": "PRA", "direction": "OVER", "line": "30.5", "gate_bucket": "published"},
            {"game_date": "2026-11-03", "player": "C", "prop": "TRB", "direction": "UNDER", "line": "8", "gate_bucket": "blocked_other"},
            {"game_date": "2026-11-03", "player": "Gone", "prop": "AST", "direction": "UNDER", "line": "2.5", "gate_bucket": "blocked_other"},
            {"game_date": "2026-11-04", "player": "A", "prop": "AST", "direction": "UNDER", "line": "4.5", "gate_bucket": "blocked_price"},
        ])
        box = pd.DataFrame([
            {"game_date": "2026-11-03", "player_norm": "a", "ast": 3, "pra": 0, "trb": 0},
            {"game_date": "2026-11-03", "player_norm": "b", "ast": 0, "pra": 28, "trb": 0},
            {"game_date": "2026-11-03", "player_norm": "c", "ast": 0, "pra": 0, "trb": 8},
        ])
        graded = run_daily._grade_candidate_rows(
            hist, box, {"2026-11-03"}, "2026-11-04", "2026-10-31", "now", self._norm
        )
        self.assertEqual(len(graded), 4)
        self.assertEqual(hist["result"].tolist(), ["WIN", "LOSS", "PUSH", "DNP", ""])
        self.assertEqual(hist.loc[0, "profit_units"], "0.91")
        self.assertEqual(hist.loc[1, "profit_units"], "-1")
        self.assertEqual(hist.loc[2, "profit_units"], "0")
        self.assertEqual(hist.loc[3, "profit_units"], "")

        summary = run_daily._summarize_candidate_record(hist)
        self.assertEqual(summary["blocked_price"]["w"], 1)
        self.assertEqual(summary["blocked_price"]["roi_pct"], 91.0)
        self.assertEqual(summary["published"]["l"], 1)
        self.assertEqual(summary["blocked_other"]["p"], 1)
        self.assertEqual(summary["blocked_other"]["dnp"], 1)

    def test_stale_date_without_box_scores_is_dnp_recent_waits(self):
        hist = self._hist([
            {"game_date": "2026-10-20", "player": "A", "prop": "AST", "direction": "UNDER", "line": "4.5", "gate_bucket": "published"},
            {"game_date": "2026-11-03", "player": "A", "prop": "AST", "direction": "UNDER", "line": "4.5", "gate_bucket": "published"},
        ])
        empty_box = pd.DataFrame(columns=["game_date", "player_norm", "ast"])
        run_daily._grade_candidate_rows(hist, empty_box, set(), "2026-11-04", "2026-11-01", "now", self._norm)
        self.assertEqual(hist["result"].tolist(), ["DNP", ""])

    def test_profit_falls_back_to_american_price_and_unpriced_is_none(self):
        self.assertEqual(run_daily._candidate_profit_units("WIN", "", "+150"), 1.5)
        self.assertEqual(run_daily._candidate_profit_units("WIN", "", "-200"), 0.5)
        self.assertIsNone(run_daily._candidate_profit_units("WIN", "", ""))
        self.assertIsNone(run_daily._candidate_profit_units("DNP", "1.91", "-110"))


if __name__ == "__main__":
    unittest.main()
