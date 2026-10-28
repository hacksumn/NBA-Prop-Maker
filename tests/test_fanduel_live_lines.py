import unittest
from datetime import datetime
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import nba_props


class _Response:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload


class FanDuelLiveLineTests(unittest.TestCase):
    def test_paid_live_response_is_atomically_checkpointed(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)
            with patch.dict(nba_props.CONFIG, {"line_archive_dir": archive}):
                path = nba_props._write_live_odds_checkpoint(
                    {"bookmakers": [{"key": "fanduel"}]},
                    event_id="event/1",
                    slate_date="2026-10-26",
                    bookmaker="fanduel",
                    snapshot_ts="2026-10-26T10:00:00-05:00",
                )
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["event_id"], "event/1")
            self.assertEqual(record["payload"]["bookmakers"][0]["key"], "fanduel")

    def test_deactivated_api_key_is_reported_as_authentication_failure(self):
        response = _Response(
            {
                "error_code": "DEACTIVATED_KEY",
                "message": "API key is deactivated",
            },
            status_code=401,
        )
        with patch.dict(
            nba_props.CONFIG,
            {"odds_api_key": "test", "allow_live_odds_api_spend": True},
        ), patch("nba_props.requests.get", return_value=response), patch(
            "builtins.print"
        ) as printer:
            lines, source = nba_props.fetch_vegas_lines(
                preferred_bookmaker="fanduel",
                allow_prizepicks_fallback=False,
            )
        self.assertTrue(lines.empty)
        self.assertEqual(source, "odds_api")
        messages = " ".join(str(call.args[0]) for call in printer.call_args_list)
        self.assertIn("DEACTIVATED_KEY", messages)
        self.assertNotIn("0 credits remaining", messages)

    def test_fanduel_alternates_remain_exact_and_prices_are_paired(self):
        event = {
            "id": "event-1",
            "commence_time": f"{datetime.now().strftime('%Y-%m-%d')}T23:00:00Z",
        }
        odds_payload = {
            "bookmakers": [{
                "key": "fanduel",
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {"description": "Test Player", "name": "Over", "point": 20.5, "price": -110},
                        {"description": "Test Player", "name": "Under", "point": 20.5, "price": -110},
                        {"description": "Test Player", "name": "Over", "point": 17.5, "price": -200},
                        {"description": "Test Player", "name": "Under", "point": 17.5, "price": 154},
                    ],
                }],
            }],
        }
        responses = [
            _Response([event], headers={"x-requests-remaining": "99999"}),
            _Response(odds_payload),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                nba_props.CONFIG,
                {
                    "odds_api_key": "test",
                    "allow_live_odds_api_spend": True,
                    "line_archive_dir": Path(tmp),
                },
            ), patch("nba_props.requests.get", side_effect=responses), patch(
                "nba_props.sanitize_player_names",
                side_effect=lambda frame, **kwargs: (
                    frame,
                    {"canonicalized_rows": 0, "dropped_rows": 0},
                ),
            ):
                lines, source = nba_props.fetch_vegas_lines(
                    prefer_prizepicks=False,
                    preferred_bookmaker="fanduel",
                    allow_prizepicks_fallback=False,
                )

        self.assertEqual(source, "fanduel")
        self.assertEqual(sorted(lines["line"].tolist()), [17.5, 20.5])
        self.assertNotIn(19.0, lines["line"].tolist())
        main = lines.loc[lines["line"].eq(20.5)].iloc[0]
        self.assertEqual(main["over_price"], -110)
        self.assertEqual(main["under_price"], -110)
        self.assertEqual(main["source"], "odds_api_fanduel")
        self.assertEqual(main["bookmaker"], "fanduel")

    def test_main_line_selection_prefers_balanced_fanduel_contract(self):
        lines = pd.DataFrame([
            {
                "game_date": "2026-10-26", "player": "Test Player",
                "prop": "player_points", "line": 17.5,
                "source": "odds_api_fanduel", "snapshot_ts": "2026-10-26T12:00:00Z",
                "over_price": -200, "under_price": 154,
            },
            {
                "game_date": "2026-10-26", "player": "Test Player",
                "prop": "player_points", "line": 20.5,
                "source": "odds_api_fanduel", "snapshot_ts": "2026-10-26T12:00:00Z",
                "over_price": -110, "under_price": -110,
            },
        ])
        selected = nba_props._select_preferred_line_rows(
            lines, allowed_sources=["fanduel"]
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["line"], 20.5)


if __name__ == "__main__":
    unittest.main()
