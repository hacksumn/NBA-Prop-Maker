import pandas as pd

from scripts.audit_project_integrity import _pick_summary


def test_roi_excludes_unpriced_losses_and_wins_and_other_sources():
    frame = pd.DataFrame([
        dict(game_date="2026-11-01", line_source="fanduel", prop="AST", result="WIN", side_price=-200),
        dict(game_date="2026-11-01", line_source="fanduel", prop="AST", result="LOSS", side_price=None),
        dict(game_date="2026-11-01", line_source="fanduel", prop="AST", result="WIN", side_price=None),
        dict(game_date="2026-11-01", line_source="prizepicks", prop="AST", result="WIN", side_price=200),
        dict(game_date="2026-11-01", line_source="fanduel", prop="AST", result="PUSH", side_price=-110),
    ])
    row = _pick_summary(frame)[0]
    assert row["graded"] == 3
    assert row["priced_graded"] == 1
    assert row["profit_units"] == 0.5
    assert row["roi"] == 0.5


def test_missing_prices_do_not_imply_zero_roi():
    row = _pick_summary(pd.DataFrame([dict(game_date="2026-11-01", line_source="fanduel", prop="AST",
                                           result="LOSS", side_price=None)]))[0]
    assert row["roi"] is None
