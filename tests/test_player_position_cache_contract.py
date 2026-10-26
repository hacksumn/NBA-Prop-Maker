import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

import nba_props


class PlayerPositionCacheContractTests(unittest.TestCase):
    def test_nba_season_rolls_in_october(self):
        self.assertEqual(nba_props._nba_season_for_date("2026-09-30"), "2025-26")
        self.assertEqual(nba_props._nba_season_for_date("2026-10-01"), "2026-27")

    def test_partial_refresh_preserves_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            cache = data_dir / "player_positions.csv"
            cached = pd.DataFrame(
                {
                    "player_norm": [f"cached {i}" for i in range(350)],
                    "position_group": ["F"] * 350,
                }
            )
            cached.to_csv(cache, index=False)
            old = time.time() - 8 * 86400
            os.utime(cache, (old, old))

            partial = pd.DataFrame(
                {
                    "PLAYER_FIRST_NAME": ["Partial"] * 10,
                    "PLAYER_LAST_NAME": [str(i) for i in range(10)],
                    "POSITION": ["G"] * 10,
                }
            )
            endpoint = MagicMock()
            endpoint.return_value.get_data_frames.return_value = [partial]
            with patch.dict(nba_props.CONFIG, {"data_dir": data_dir}), patch(
                "nba_api.stats.endpoints.playerindex.PlayerIndex", endpoint
            ), patch("time.sleep", return_value=None):
                result = nba_props.fetch_player_positions(season="2026-27")

            self.assertEqual(len(result), 350)
            self.assertEqual(len(pd.read_csv(cache)), 350)


if __name__ == "__main__":
    unittest.main()
