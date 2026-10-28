from pathlib import Path

import pandas as pd

import nba_props


def test_current_projection_layers_override_stale_enriched_gap(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "player": "Current Player",
                "season": "2026-27",
                "clean_pts": 21.5,
                "adj_ppp": 1.18,
                "raw_ppp": 1.11,
                "avg_opp_def_rtg": 112.4,
                "usg_pct": 0.24,
            }
        ]
    ).to_csv(data_dir / "player_projections_today.csv", index=False)
    pd.DataFrame(
        [
            {
                "player_id": 1,
                "player": "Current Player",
                "season": "2025-26",
                "clean_pts": 19.0,
                "adj_ppp": 1.05,
            }
        ]
    ).to_csv(data_dir / "training_dataset_enriched.csv", index=False)
    monkeypatch.setattr(nba_props, "_HERE", tmp_path)

    target = pd.DataFrame(
        [{"player_id": 1, "player": "Current Player", "season": "2026-27"}]
    )
    merged = nba_props._merge_layer_features(target)

    assert merged.loc[0, "clean_pts"] == 21.5
    assert merged.loc[0, "adj_ppp"] == 1.18
    assert merged.loc[0, "raw_ppp_season"] == 1.11
    assert merged.loc[0, "opp_def_rtg_avg"] == 112.4
    assert "raw_ppp" not in merged.columns


def test_prediction_cache_signature_tracks_current_layer_snapshot(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    layer = data_dir / "player_projections_today.csv"
    layer.write_text("player,season\nA,2026-27\n", encoding="utf-8")
    monkeypatch.setitem(nba_props.CONFIG, "data_dir", data_dir)
    monkeypatch.setitem(nba_props.CONFIG, "models_dir", models_dir)

    signature = nba_props._prediction_feature_cache_signature(True, "2026-10-27")

    tracked = {Path(item["path"]).name for item in signature["sources"]}
    assert "player_projections_today.csv" in tracked
