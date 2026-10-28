import json

import numpy as np
import pandas as pd

import nba_props
from feature_pipeline import FeaturePipeline


def _history() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-01", periods=12, freq="2D")
    for index, game_date in enumerate(dates):
        game_id = 990_000 + index
        for player_id, player, team, opponent, is_home, base in [
            (1, "Alpha Player", "AAA", "BBB", 1, 10.0),
            (2, "Beta Player", "BBB", "AAA", 0, 20.0),
        ]:
            pts = base + index
            rows.append(
                {
                    "GAME_ID": game_id,
                    "PLAYER_ID": player_id,
                    "player": player,
                    "team": team,
                    "game_date": game_date,
                    "matchup": f"{team} {'vs.' if is_home else '@'} {opponent}",
                    "is_home": is_home,
                    "opp": opponent,
                    "pts": pts,
                    "trb": 5.0 + index,
                    "ast": 3.0 + index,
                    "mp": 30.0,
                    "fga": 12.0 + index,
                    "fta": 4.0,
                    "3pa": 5.0,
                    "stl": 1.0,
                    "blk": 1.0,
                    "tov": 2.0,
                    "orb": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_upcoming_target_row_uses_latest_completed_game_in_shifted_features():
    history = _history()
    slate_date = pd.Timestamp("2026-01-27")
    schedule = pd.DataFrame(
        [{"game_id": "0022600999", "home_team": "AAA", "away_team": "BBB", "game_date": slate_date}]
    )

    with_target = nba_props._append_upcoming_prediction_rows(history, slate_date, schedule)
    target = with_target.loc[with_target["_prediction_target_row"]]
    assert len(target) == 2
    assert target[["pts", "trb", "ast", "mp"]].isna().all().all()

    featured = FeaturePipeline(ewma_span=5, split_window=15, min_games=10).fit_transform(with_target)
    alpha_target = featured.loc[
        featured["_prediction_target_row"] & featured["PLAYER_ID"].eq(1)
    ].iloc[0]
    completed_alpha = history.loc[history["PLAYER_ID"].eq(1), "pts"]

    assert alpha_target["games_played"] == len(completed_alpha)
    assert np.isclose(alpha_target["pts_l5"], completed_alpha.tail(5).mean())
    assert np.isclose(alpha_target["pts_l10"], completed_alpha.tail(10).mean())
    assert alpha_target["game_date"] == slate_date
    assert alpha_target["opp"] == "BBB"
    assert alpha_target["GAME_ID"] == "0022600999"


def test_prediction_cache_is_scoped_to_slate_date():
    first = nba_props._prediction_feature_cache_signature(True, "2026-01-27")
    second = nba_props._prediction_feature_cache_signature(True, "2026-01-28")
    assert first["slate_date"] == "2026-01-27"
    assert second["slate_date"] == "2026-01-28"
    assert first != second


def test_active_prop_features_are_overwritten_by_training_pipeline(tmp_path, monkeypatch):
    selected = {"pts": ["pts_ewma5", "pts_confidence"]}
    (tmp_path / "selected_features_advanced.json").write_text(json.dumps(selected))
    monkeypatch.setitem(nba_props.CONFIG, "models_dir", tmp_path)

    source = _history()
    live_copy = source.copy()
    live_copy["GAME_ID"] = live_copy["GAME_ID"].astype(str).str.zfill(10)
    live_copy["pts_ewma5"] = -999.0
    live_copy["pts_confidence"] = -999.0

    aligned = nba_props._align_selected_features_to_training_pipeline(live_copy, source)
    canonical = FeaturePipeline(ewma_span=5, split_window=15, min_games=10).fit_transform(source)
    keys = ["GAME_ID", "PLAYER_ID", "team"]
    canonical["GAME_ID"] = pd.to_numeric(canonical["GAME_ID"]).astype("Int64").astype("string")
    canonical["PLAYER_ID"] = pd.to_numeric(canonical["PLAYER_ID"]).astype("Int64").astype("string")
    check = aligned[keys + selected["pts"]].merge(
        canonical[keys + selected["pts"]],
        on=keys,
        suffixes=("_live", "_training"),
        validate="one_to_one",
    )
    for feature in selected["pts"]:
        assert np.allclose(
            check[f"{feature}_live"],
            check[f"{feature}_training"],
            equal_nan=True,
        )
