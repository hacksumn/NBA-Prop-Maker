import unittest
from unittest.mock import patch

import pandas as pd

from player_pool_guard import SeasonPlayerLookup, canonicalize_player_name, sanitize_player_names


def _lookup(has_roster: bool = True) -> SeasonPlayerLookup:
    return SeasonPlayerLookup(
        season="2025-26",
        alias_to_canonical={
            "aj green": "AJ Green",
            "kelly oubre jr": "Kelly Oubre Jr.",
            "kelly oubre": "Kelly Oubre Jr.",
            "jaime jaquez jr": "Jaime Jaquez Jr.",
            "jaime jaquez": "Jaime Jaquez Jr.",
            "ronald holland ii": "Ronald Holland II",
            "ronald holland": "Ronald Holland II",
            "moritz wagner": "Moritz Wagner",
            "isaiah stewart": "Isaiah Stewart",
            "isaiah stewart ii": "Isaiah Stewart",
        },
        fuzzy_records=(
            ("Bub Carrington", "bub", "carrington"),
            ("Moritz Wagner", "moritz", "wagner"),
            ("Ronald Holland II", "ronald", "holland"),
            ("Tristan da Silva", "tristan", "silva"),
        ),
        has_roster=has_roster,
    )


class PlayerPoolGuardTests(unittest.TestCase):
    def test_canonicalize_alias_variants(self):
        lookup = _lookup()
        self.assertEqual(canonicalize_player_name("A.J. Green", lookup), "AJ Green")
        self.assertEqual(canonicalize_player_name("Kelly Oubre", lookup), "Kelly Oubre Jr.")
        self.assertEqual(canonicalize_player_name("Ron Holland", lookup), "Ronald Holland II")
        self.assertEqual(canonicalize_player_name("Carlton Carrington", lookup), "Bub Carrington")
        self.assertEqual(canonicalize_player_name("Moe Wagner", lookup), "Moritz Wagner")
        self.assertEqual(canonicalize_player_name("Tristan Silva", lookup), "Tristan da Silva")

    def test_sanitize_player_names_canonicalizes_and_drops_unknown(self):
        raw = pd.DataFrame(
            [
                {"game_date": "2026-04-09", "player": "Kelly Oubre", "player_norm": "kelly oubre", "prop": "player_points", "line": 12.5},
                {"game_date": "2026-04-09", "player": "Carlton Carrington", "player_norm": "carlton carrington", "prop": "player_assists", "line": 4.5},
                {"game_date": "2026-04-09", "player": "Fake Prospect", "player_norm": "fake prospect", "prop": "player_points", "line": 10.5},
            ]
        )

        with patch("player_pool_guard.load_season_player_lookup", return_value=_lookup()):
            cleaned, summary = sanitize_player_names(raw)

        self.assertEqual(cleaned["player"].tolist(), ["Kelly Oubre Jr.", "Bub Carrington"])
        self.assertEqual(cleaned["player_norm"].tolist(), ["kelly oubre jr", "bub carrington"])
        self.assertEqual(summary["canonicalized_rows"], 2)
        self.assertEqual(summary["dropped_rows"], 1)
        self.assertEqual(summary["unknown_players"], ["Fake Prospect"])

    def test_sanitize_player_names_requires_roster(self):
        raw = pd.DataFrame(
            [{"game_date": "2026-04-09", "player": "Kelly Oubre", "player_norm": "kelly oubre"}]
        )

        with patch("player_pool_guard.load_season_player_lookup", return_value=_lookup(has_roster=False)):
            with self.assertRaisesRegex(RuntimeError, "Official NBA roster validation unavailable"):
                sanitize_player_names(raw, require_roster=True)


if __name__ == "__main__":
    unittest.main()
