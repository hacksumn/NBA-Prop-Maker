import csv
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.recover_zip_assets import recover_assets


def _csv_text(columns, rows):
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class RecoverZipAssetsTests(unittest.TestCase):
    def _fixture(self, base: Path):
        root = base / "project"
        data = root / "data"
        data.mkdir(parents=True)

        (data / "player_profiles.csv").write_text("player_id,player_name,team_abbr,gp,usg_pct\n", encoding="utf-8")
        (data / "player_clean_baselines.csv").write_text("\n", encoding="utf-8")
        (data / "player_positions.csv").write_text(
            "player_norm,position_group\narchive overlap,G\ncurrent only,C\n", encoding="utf-8"
        )
        current_blowout = "game_id,game_date,home_team,away_team,blowout_tier,blowout_weight\nnew,2026-10-24,A,B,FULL,1.0\n"
        (data / "game_blowout_index.csv").write_text(current_blowout, encoding="utf-8")

        profiles = [
            {"player_id": str(i), "player_name": f"Player {i}", "team_abbr": "AAA", "gp": "10", "usg_pct": ".2"}
            for i in range(3)
        ]
        baselines = [
            {
                "player_id": str(i),
                "player_name": f"Player {i}",
                "team_abbr": "AAA",
                "games_analyzed": "10",
                "clean_pts": "10",
                "clean_reb": "5",
                "clean_ast": "3",
            }
            for i in range(3)
        ]
        positions = [
            {"player_norm": "archive overlap", "position_group": "G"},
            {"player_norm": "archive only", "position_group": "F"},
            {"player_norm": "archive center", "position_group": "C"},
        ]
        blowouts = [
            {
                "game_id": str(i),
                "game_date": "2026-04-12",
                "home_team": "AAA",
                "away_team": "BBB",
                "blowout_tier": "FULL",
                "blowout_weight": "1.0",
            }
            for i in range(3)
        ]

        zip_path = base / "Fresh_Start_NBA.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "Fresh_Start_NBA/data/player_profiles.csv",
                _csv_text(list(profiles[0]), profiles),
            )
            archive.writestr(
                "Fresh_Start_NBA/data/player_clean_baselines.csv",
                _csv_text(list(baselines[0]), baselines),
            )
            archive.writestr(
                "Fresh_Start_NBA/data/player_positions.csv",
                _csv_text(list(positions[0]), positions),
            )
            archive.writestr(
                "Fresh_Start_NBA/data/game_blowout_index.csv",
                _csv_text(list(blowouts[0]), blowouts),
            )
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        return root, zip_path, digest, current_blowout

    def test_selective_recovery_preserves_current_blowout_and_unions_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, zip_path, digest, current_blowout = self._fixture(Path(tmp))
            with patch.dict(
                "scripts.recover_zip_assets.MINIMUM_ROWS",
                {
                    "data/player_profiles.csv": 3,
                    "data/player_clean_baselines.csv": 3,
                    "data/player_positions.csv": 3,
                    "data/game_blowout_index.csv": 3,
                },
                clear=True,
            ):
                manifest = recover_assets(zip_path, root, expected_sha256=digest)

            self.assertEqual(manifest["restored"]["player_profiles"]["rows"], 3)
            self.assertEqual(manifest["restored"]["player_positions"]["rows_after_union"], 4)
            self.assertEqual(manifest["restored"]["player_positions"]["conflicts"], 0)
            self.assertEqual(
                (root / "data" / "game_blowout_index.csv").read_text(encoding="utf-8"),
                current_blowout,
            )
            self.assertTrue(root.joinpath("data/season_snapshots/game_blowout_index_2025-26.csv").exists())
            self.assertTrue(root.joinpath("data/backups/zip_recovery_20261025_pre/player_profiles.csv").exists())
            saved_manifest = json.loads(
                root.joinpath("output/zip_recovery_20261025_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["status"], "recovered_and_validated")

    def test_hash_mismatch_fails_before_creating_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, zip_path, _, _ = self._fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "ZIP hash mismatch"):
                recover_assets(zip_path, root, expected_sha256="0" * 64)
            self.assertFalse(root.joinpath("data/backups/zip_recovery_20261025_pre").exists())


if __name__ == "__main__":
    unittest.main()
