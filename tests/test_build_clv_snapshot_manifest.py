import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_clv_snapshot_manifest import build_manifest, resolve_decision_timestamp


class ClvSnapshotManifestTests(unittest.TestCase):
    def test_resolves_first_archive_matching_final_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026-04-21"
            day.mkdir()
            final_bytes = b"player,prop\nA,PTS\n"
            (day / "picks_2026-04-21.csv").write_bytes(final_bytes)
            (day / "picks_2026-04-21_080000.csv").write_bytes(b"player,prop\nB,AST\n")
            source = day / "picks_2026-04-21_090539.csv"
            source.write_bytes(final_bytes)

            resolved = resolve_decision_timestamp(root, "2026-04-21")

            self.assertEqual(resolved["decision_status"], "resolved")
            self.assertEqual(resolved["decision_ts_utc"], "2026-04-21T14:05:39Z")
            self.assertEqual(resolved["decision_source_sha256"], hashlib.sha256(final_bytes).hexdigest())

    def test_builds_exact_offsets_and_blocks_missing_decision(self):
        games = pd.DataFrame(
            [{"game_date": "2026-04-21", "nba_game_id": "0042500112", "away_team": "PHI", "home_team": "BOS"}]
        )
        schedule = pd.DataFrame(
            [{
                "game_date": "2026-04-21",
                "away_team": "PHI",
                "home_team": "BOS",
                "tip_ts_utc": pd.Timestamp("2026-04-21T23:00:00Z"),
                "schedule_event_id": "401869396",
                "schedule_source": "test",
                "schedule_source_url": "https://example.invalid",
            }]
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_manifest(games, schedule, Path(tmp))

        indexed = manifest.set_index("snapshot_kind")
        self.assertEqual(indexed.loc["t_minus_60", "requested_snapshot_ts"], "2026-04-21T22:00:00Z")
        self.assertEqual(indexed.loc["t_minus_5", "requested_snapshot_ts"], "2026-04-21T22:55:00Z")
        self.assertEqual(indexed.loc["decision", "execution_status"], "blocked")
        self.assertEqual(indexed.loc["decision", "block_reason"], "final_daily_card_missing")
        self.assertEqual(int(indexed.loc["t_minus_60", "total_credit_ceiling"]), 91)


if __name__ == "__main__":
    unittest.main()
