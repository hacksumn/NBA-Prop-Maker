"""Legacy and ISO timestamps must coexist without silently changing selection."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import nba_props
import train_advanced_models as training


def _lines():
    return pd.DataFrame([
        dict(game_date="2026-11-10", player="Alpha", player_norm="alpha", prop="AST",
             line=2.5, source="fanduel", snapshot_ts="2026-11-10 16:00:00",
             commence_time="2026-11-10 23:00:00", over_price=-110, under_price=-110),
        dict(game_date="2026-11-10", player="Alpha", player_norm="alpha", prop="AST",
             line=3.5, source="fanduel", snapshot_ts="2026-11-10T17:00:00Z",
             commence_time="2026-11-10T23:00:00Z", over_price=-110, under_price=-110),
        dict(game_date="2026-11-10", player="Alpha", player_norm="alpha", prop="AST",
             line=9.5, source="fanduel", snapshot_ts="2026-11-11T00:00:00Z",
             commence_time="2026-11-10T23:00:00Z", over_price=-110, under_price=-110),
    ])


def test_live_selector_keeps_latest_pregame_iso_among_legacy_rows():
    for rows in [_lines(), _lines().iloc[::-1]]:
        selected = nba_props._select_preferred_line_rows(rows, require_known_snapshot=True)
        assert selected.line.tolist() == [3.5]


def test_live_selector_honors_asof_with_mixed_formats():
    selected = nba_props._select_preferred_line_rows(_lines(), as_of="2026-11-10T16:30:00Z")
    assert selected.line.tolist() == [2.5]


def test_training_lookup_selects_same_latest_pregame_line():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _lines().assign(prop="player_assists").to_csv(root / "historical_lines.csv", index=False)
        with patch.object(training, "DATA_DIR", root):
            lookup = training._load_lines_lookup()
    assert lookup[("alpha", "2026-11-10", "ast")] == 3.5
