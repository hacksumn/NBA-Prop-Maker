"""DEC-070: history files are append-only; the pipeline never deletes saved data.

Before this, every predict run replaced a slate's rows in picks_history.csv and
betslips_history.csv, so a late rerun erased the morning card (70 picks and 116
slips were recovered from the archive on 2026-11-05).
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import nba_props
from nba_scraper import merge_game_logs_preserving


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


class AppendOnlyHelperTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "history.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def test_existing_rows_are_never_deleted_or_modified(self):
        pd.DataFrame([
            {"game_date": "2026-11-04", "player": "A", "prop": "AST", "direction": "UNDER", "line": "4.5", "result": "WIN", "old_col": "x"},
        ]).to_csv(self.path, index=False)
        new = pd.DataFrame([
            {"game_date": "2026-11-04", "player": "A", "prop": "ast", "direction": "under", "line": "4.50", "result": "", "new_col": "y"},
            {"game_date": "2026-11-04", "player": "B", "prop": "TRB", "direction": "UNDER", "line": "8", "result": "", "new_col": "z"},
        ])
        combined, added, already = nba_props._append_only_history(self.path, new, nba_props.PICKS_HISTORY_KEY)
        self.assertEqual((added, already), (1, 1))
        stored = _read(self.path)
        self.assertEqual(stored["player"].tolist(), ["A", "B"])
        self.assertEqual(stored.loc[0, "result"], "WIN")      # untouched
        self.assertEqual(stored.loc[0, "old_col"], "x")       # no column dropped
        self.assertEqual(stored.loc[0, "new_col"], "")        # new column blank for old row
        self.assertEqual(stored.loc[1, "new_col"], "z")

    def test_empty_batch_writes_nothing(self):
        pd.DataFrame([{"game_date": "d", "player": "A", "prop": "AST", "direction": "U", "line": "1"}]).to_csv(self.path, index=False)
        before = self.path.read_bytes()
        nba_props._append_only_history(self.path, pd.DataFrame(), nba_props.PICKS_HISTORY_KEY)
        self.assertEqual(self.path.read_bytes(), before)

    def test_betslip_key_ignores_leg_order(self):
        a = pd.DataFrame([{"game_date": "d", "slip_type": "2-pick", "Pick_1_Player": "A", "Pick_1_Prop": "AST",
                           "Pick_1_Dir": "UNDER", "Pick_1_Line": "4.5", "Pick_2_Player": "B", "Pick_2_Prop": "TRB",
                           "Pick_2_Dir": "UNDER", "Pick_2_Line": "8.0"}])
        b = pd.DataFrame([{"game_date": "d", "slip_type": "2-pick", "Pick_1_Player": "B", "Pick_1_Prop": "TRB",
                           "Pick_1_Dir": "UNDER", "Pick_1_Line": "8", "Pick_2_Player": "A", "Pick_2_Prop": "AST",
                           "Pick_2_Dir": "UNDER", "Pick_2_Line": "4.5"}])
        self.assertEqual(nba_props._betslip_history_key(a)[0], nba_props._betslip_history_key(b)[0])


class PipelineWritersTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self._patches = [
            patch.dict(nba_props.CONFIG, {"output_dir": self.out, "output_archive_dir": self.out / "archive"}),
            patch("builtins.print"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def _pick(self, player):
        return {"player": player, "team": "DET", "opponent": "PHI", "prop": "AST", "direction": "UNDER",
                "line": 4.5, "prediction": 3.9, "edge": -0.6, "confidence": 70.0}

    def test_late_rerun_with_zero_picks_keeps_morning_picks(self):
        pred = pd.DataFrame([{"player": "A"}])
        nba_props.save_picks(pd.DataFrame([self._pick("Morning A"), self._pick("Morning B")]), pred, "2026-11-04")
        nba_props.save_picks(pd.DataFrame([self._pick("Noon C")]), pred, "2026-11-04")
        nba_props.save_picks(pd.DataFrame(columns=list(self._pick("x").keys())), pred, "2026-11-04")
        stored = _read(self.out / "picks_history.csv")
        self.assertEqual(sorted(stored["player"]), ["Morning A", "Morning B", "Noon C"])
        self.assertTrue(stored["published_at"].str.len().gt(0).all())

    def test_zero_slip_rerun_does_not_touch_betslip_history(self):
        path = self.out / "betslips_history.csv"
        pd.DataFrame([{"game_date": "2026-11-04", "slip_type": "2-pick", "Pick_1_Player": "A",
                       "result": "WIN", "custom_note": "keep me"}]).to_csv(path, index=False)
        before = path.read_bytes()
        nba_props.log_betslips(pd.DataFrame(), "2026-11-04")
        self.assertEqual(path.read_bytes(), before)

    def test_historical_lines_shrink_is_backed_up_first(self):
        path = self.out / "lines.csv"
        pd.DataFrame({"a": range(5)}).to_csv(path, index=False)
        backup = nba_props._backup_before_shrink(path, 3)
        self.assertIsNotNone(backup)
        self.assertEqual(len(pd.read_csv(backup)), 5)
        self.assertIsNone(nba_props._backup_before_shrink(path, 5))


class GameLogMergeTests(unittest.TestCase):
    def _rows(self, pairs, pts=10):
        return pd.DataFrame([{"PLAYER_ID": p, "GAME_ID": g, "pts": pts, "season": "2026-27"} for p, g in pairs])

    def test_partial_or_empty_api_response_never_shrinks(self):
        stored = self._rows([("1", "0022600001"), ("2", "0022600001"), ("1", "0022600002")]).astype(str)
        self.assertEqual(len(merge_game_logs_preserving(stored, self._rows([(1, 22600001)]))), 3)
        self.assertEqual(len(merge_game_logs_preserving(stored, pd.DataFrame())), 3)

    def test_same_player_game_is_refreshed_new_games_added(self):
        stored = self._rows([("1", "0022600001")], pts=5).astype(str)
        fresh = self._rows([(1, 22600001), (1, 22600009)], pts=31)
        merged = merge_game_logs_preserving(stored, fresh)
        self.assertEqual(len(merged), 2)
        self.assertEqual(sorted(merged["pts"].astype(int)), [31, 31])


if __name__ == "__main__":
    unittest.main()
