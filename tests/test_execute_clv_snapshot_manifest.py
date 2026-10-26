import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.execute_clv_snapshot_manifest import (
    PHYSICAL_KEY,
    canonical_team,
    execute,
    execution_lock,
    normalize_name,
    parse_event_odds,
)


class ExecuteClvManifestTests(unittest.TestCase):
    def test_execution_lock_rejects_overlapping_paid_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            lock = checkpoint.with_suffix(".json.lock")
            lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with execution_lock(checkpoint):
                    pass

    def test_parse_pairs_prices_and_retains_book(self):
        payload = {
            "timestamp": "2026-04-21T22:55:00Z",
            "data": {
                "id": "event-1",
                "commence_time": "2026-04-21T23:00:00Z",
                "bookmakers": [{
                    "key": "fanduel",
                    "markets": [{
                        "key": "player_points",
                        "outcomes": [
                            {"name": "Over", "description": "Jayson Tatum", "point": 28.5, "price": -115},
                            {"name": "Under", "description": "Jayson Tatum", "point": 28.5, "price": -105},
                        ],
                    }],
                }],
            },
        }
        rows = parse_event_odds(payload, "2026-04-21T22:55:00Z")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.iloc[0]["source"], "odds_api_fanduel")
        self.assertEqual(rows.iloc[0]["over_price"], -115)
        self.assertEqual(rows.iloc[0]["under_price"], -105)

    def test_parse_converts_utc_tip_to_chicago_game_date(self):
        payload = {
            "timestamp": "2026-03-31T00:55:00Z",
            "data": {
                "id": "event-2",
                "commence_time": "2026-03-31T01:00:00Z",
                "bookmakers": [{
                    "key": "fanduel",
                    "markets": [{
                        "key": "player_points",
                        "outcomes": [
                            {"name": "Over", "description": "Jayson Tatum", "point": 28.5, "price": -110},
                            {"name": "Under", "description": "Jayson Tatum", "point": 28.5, "price": -110},
                        ],
                    }],
                }],
            },
        }
        rows = parse_event_odds(payload, "2026-03-31T00:55:00Z")
        self.assertEqual(rows.iloc[0]["game_date"], "2026-03-30")

    def test_team_names_match_manifest_abbreviations(self):
        self.assertEqual(canonical_team("Philadelphia 76ers"), "PHI")
        self.assertEqual(canonical_team("Los Angeles Clippers"), "LAC")
        self.assertEqual(canonical_team("BOS"), "BOS")

    def test_name_normalization_matches_production_punctuation_contract(self):
        self.assertEqual(normalize_name("Kelly Oubre Jr."), "kelly oubre jr")
        self.assertEqual(normalize_name("Nikola Jokić"), "nikola jokic")

    def test_line_is_part_of_lossless_physical_key(self):
        self.assertIn("line", PHYSICAL_KEY)

    def test_zero_cap_is_a_no_request_dry_run(self):
        class BombSession:
            def get(self, *args, **kwargs):
                raise AssertionError("dry run sent a network request")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            pd.DataFrame([{
                "manifest_key": "game:t_minus_5",
                "game_date": "2026-04-21",
                "nba_game_id": "0042500112",
                "away_team": "PHI",
                "home_team": "BOS",
                "snapshot_kind": "t_minus_5",
                "requested_snapshot_ts": "2026-04-21T22:55:00Z",
                "event_lookup_group": "2026-04-21:t_minus_5",
                "markets": "player_points",
                "regions": "us",
                "odds_format": "american",
                "event_odds_credit_ceiling": 10,
                "execution_status": "ready_for_event_lookup",
            }]).to_csv(manifest, index=False)
            result = execute(
                manifest, root / "checkpoint.json", root / "ledger.csv",
                root / "summary.json", root / "raw", root / "parsed",
                root / "master.csv", "unused", 0, None, 1, BombSession()
            )
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["requests_sent"], 0)


if __name__ == "__main__":
    unittest.main()
