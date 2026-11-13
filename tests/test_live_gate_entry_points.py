import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import nba_props


class LiveGateEntryPointTests(unittest.TestCase):
    def test_all_delegates_to_predict_after_training_with_fresh_features(self):
        original_main = nba_props.main
        data = pd.DataFrame({"player": ["Test Player"]})
        with patch.object(sys, "argv", ["nba_props.py", "all", "input.csv"]):
            with patch.object(nba_props, "fetch_nba_data", return_value=data) as fetch, \
                    patch.object(nba_props, "_save_nba_data_preserving", return_value=data), \
                    patch.object(nba_props, "create_features", return_value=data), \
                    patch.object(nba_props, "train_models") as train, \
                    patch.object(nba_props, "train_ou_classifiers") as classifiers, \
                    patch.object(nba_props, "main") as predict:
                original_main()
        fetch.assert_called_once_with(local_file="input.csv")
        train.assert_called_once_with(data)
        classifiers.assert_called_once_with(data)
        predict.assert_called_once_with(command="predict", use_feature_cache=False)

    def test_predict_cli_still_loads_the_configured_data_path(self):
        with TemporaryDirectory() as directory:
            with patch.dict(nba_props.CONFIG, {"data_dir": Path(directory)}), \
                    patch.object(sys, "argv", ["nba_props.py", "predict"]):
                with self.assertRaisesRegex(FileNotFoundError, "No data found"):
                    nba_props.main()

    def test_predict_filters_cached_lines_at_the_run_time(self):
        now = pd.Timestamp("2026-11-01T15:00:00Z")
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            pd.DataFrame({"game_date": ["2026-10-31"]}).to_csv(data_dir / "nba_data.csv", index=False)
            pd.DataFrame({"game_date": ["2026-11-01"]}).to_csv(data_dir / "historical_lines.csv", index=False)
            with patch.dict(nba_props.CONFIG, {"data_dir": data_dir}), \
                    patch.object(nba_props, "_utc_now", return_value=now), \
                    patch.object(nba_props, "_select_cached_lines_for_predict", side_effect=RuntimeError("selection reached")) as select:
                with self.assertRaisesRegex(RuntimeError, "selection reached"):
                    nba_props.main(command="predict")
        self.assertEqual(select.call_args.kwargs["as_of"], now)

    def test_future_and_unknown_snapshots_fail_the_price_gate(self):
        now = pd.Timestamp("2026-11-01T15:00:00Z")
        picks = pd.DataFrame([{
            "player": "Test Player", "prop": "AST", "direction": "UNDER",
            "line": 3.5, "dir_prob": 65.0,
        }])
        for snapshot in [None, "invalid", "2026-11-01T16:00:00Z"]:
            with self.subTest(snapshot=snapshot):
                lines = pd.DataFrame([{
                    "player": "Test Player", "prop": "player_assists", "line": 3.5,
                    "over_price": -110.0, "under_price": -110.0,
                    "commence_time": "2026-11-02T01:00:00Z",
                    "snapshot_ts": snapshot, "event_id": "test-event",
                }])
                result = nba_props._attach_live_market_prices(picks, lines, now_utc=now).iloc[0]
                self.assertFalse(result["price_gate_pass"])
                self.assertIn("snapshot", result["price_gate_reason"])

    def test_invalid_probabilities_and_prices_fail_the_exact_price_gate(self):
        now = pd.Timestamp("2026-11-01T15:00:00Z")
        picks = pd.DataFrame([{
            "player": "Test Player", "prop": "AST", "direction": "UNDER",
            "line": 3.5, "dir_prob": 65.0, "confidence": 65.0,
        }])
        lines = pd.DataFrame([{
            "player": "Test Player", "prop": "player_assists", "line": 3.5,
            "over_price": -110.0, "under_price": -110.0,
            "commence_time": "2026-11-02T01:00:00Z",
            "snapshot_ts": "2026-11-01T14:00:00Z", "event_id": "test-event",
        }])
        for invalid in [np.inf, -np.inf, 101.0, -1.0]:
            with self.subTest(probability=invalid):
                result = nba_props._attach_live_market_prices(
                    picks.assign(dir_prob=invalid), lines, now_utc=now
                ).iloc[0]
                self.assertFalse(result["price_gate_pass"])
                self.assertIn("probability", result["price_gate_reason"])
        for invalid in [np.inf, -np.inf]:
            with self.subTest(price=invalid):
                result = nba_props._attach_live_market_prices(
                    picks, lines.assign(under_price=invalid), now_utc=now
                ).iloc[0]
                self.assertFalse(result["price_gate_pass"])
                self.assertIn("no paired", result["price_gate_reason"])


if __name__ == "__main__":
    unittest.main()
