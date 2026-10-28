from pathlib import Path

import pandas as pd

import pbp_fetcher


def test_pbp_index_deduplicates_padded_and_unpadded_game_ids(tmp_path: Path):
    path = tmp_path / "pbp_index.csv"
    pd.DataFrame(
        [
            {
                "game_id": "22600118",
                "game_date": "2026-10-25",
                "home_team": "SAC",
                "away_team": "MEM",
                "season": "2026-27",
                "fetched": True,
            },
            {
                "game_id": "0022600118",
                "game_date": "2026-10-25",
                "home_team": "SAC",
                "away_team": "MEM",
                "season": "2026-27",
                "fetched": True,
            },
        ]
    ).to_csv(path, index=False)

    repaired = pbp_fetcher.normalize_index_file(path)

    assert repaired["game_id"].tolist() == ["0022600118"]
    reread = pd.read_csv(path, dtype={"game_id": "string"})
    assert reread["game_id"].tolist() == ["0022600118"]


def test_pbp_game_id_normalization_rejects_empty_values():
    assert pbp_fetcher._normalize_game_id("0022600001") == "0022600001"
    assert pbp_fetcher._normalize_game_id(22600001) == "0022600001"
    assert pbp_fetcher._normalize_game_id(None) == ""
