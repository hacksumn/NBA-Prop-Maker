"""Future holdout predictions must be frozen before tip-off."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import capture_candidate_future_holdout as holdout


def _row(**updates):
    row = dict(candidate_id="frozen", game_date="2026-11-10", player="Alpha",
               player_norm="alpha", prop="ast", line=3.5,
               snapshot_ts="2026-11-10T16:00:00Z", commence_time="2026-11-11T00:00:00Z",
               captured_at="2026-11-10T17:00:00Z", event_id="event",
               production_prediction=3.0, candidate_prediction=2.8, actual=np.nan)
    row.update(updates)
    return row


def test_evaluation_rejects_posttip_unknown_and_future_snapshot():
    rows = pd.DataFrame([
        _row(),
        _row(player_norm="posttip", captured_at="2026-11-11T00:00:00Z"),
        _row(player_norm="unknown", commence_time=None),
        _row(player_norm="future", snapshot_ts="2026-11-10T18:00:00Z"),
        _row(player_norm="unknown_capture", captured_at=None),
    ])
    result, counts = holdout._evaluation_rows(rows)
    assert result.player_norm.tolist() == ["alpha"]
    assert counts["excluded_unverified_or_posttip_rows"] == 4


def test_evaluation_freezes_first_contract_without_deleting_raw_snapshots():
    rows = pd.DataFrame([
        _row(captured_at="2026-11-10T19:00:00Z", candidate_prediction=9.0),
        _row(),
        _row(snapshot_ts="2026-11-10T18:00:00Z", captured_at="2026-11-10T20:00:00Z"),
        _row(line=4.5),
    ])
    result, counts = holdout._evaluation_rows(rows)
    assert len(result) == 2
    assert result.loc[result.line.eq(3.5), "candidate_prediction"].item() == 2.8
    assert counts["excluded_repeat_contract_snapshots"] == 2
    assert len(rows) == 4


def test_cluster_bootstrap_matches_row_weighted_accuracy():
    frame = pd.DataFrame({"game_date": ["a"] * 100 + ["b"],
                          "candidate": [1.] * 100 + [0.], "baseline": [0.] * 101})
    low, high = holdout._date_bootstrap(frame, "candidate", "baseline", draws=2000)
    assert low == 0.0 and high == 1.0
    # A sampled a,b pair must have delta 100/101, not the old 0.5.
    class FixedRng:
        def choice(self, *args, **kwargs):
            return np.array(["a", "b"])
    with patch.object(holdout.np.random, "default_rng", return_value=FixedRng()):
        assert holdout._date_bootstrap(frame, "candidate", "baseline", draws=2) == [100/101, 100/101]


def test_capture_rerun_preserves_first_prediction_and_actual():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "candidate").mkdir()
        history = root / "nba_data.csv"
        pd.DataFrame({"game_date": ["2026-11-09"]}).to_csv(history, index=False)
        store = root / "holdout.csv"
        original = _row(actual=2.0, snapshot_ts="2000-01-01T00:00:00Z",
                        captured_at="2026-11-10T16:30:00Z")
        pd.DataFrame([original]).to_csv(store, index=False)
        lines = pd.DataFrame([dict(player="Alpha", prop="Assists", line=3.5,
                                  snapshot_ts=original["snapshot_ts"],
                                  commence_time="2099-01-01T00:00:00Z", event_id="event")])
        pred = pd.DataFrame({"player": ["Alpha"], "ast_line": [3.5], "ast_pred": [9.0]})
        with patch.multiple(holdout, ROOT=root, DATA=root, STORE=store), \
             patch.object(holdout, "load_active_candidate", return_value=({}, dict(
                 candidate_id="frozen", model_dir="candidate", future_holdout_start="2026-10-28"))), \
             patch.object(holdout, "_active_lines", return_value=(lines, pd.DataFrame())), \
             patch.object(holdout, "_score_model", return_value=pred):
            result = holdout.capture("2026-11-10")
        saved = pd.read_csv(store)
        assert result["new_rows"] == 0
        assert len(saved) == 1
        assert saved.candidate_prediction.item() == 2.8
        assert saved.actual.item() == 2.0


def test_capture_skips_started_events_without_scoring():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "candidate").mkdir()
        lines = pd.DataFrame([dict(player="Alpha", prop="Assists", line=3.5,
                                  snapshot_ts="2000-01-01T00:00:00Z",
                                  commence_time="2000-01-02T00:00:00Z")])
        with patch.object(holdout, "ROOT", root), \
             patch.object(holdout, "load_active_candidate", return_value=({}, dict(
                 candidate_id="frozen", model_dir="candidate", future_holdout_start="2026-10-28"))), \
             patch.object(holdout, "_active_lines", return_value=(lines, pd.DataFrame())), \
             patch.object(holdout, "_score_model") as score:
            assert holdout.capture("2026-11-10")["status"] == "no_verified_pregame_lines"
            score.assert_not_called()
