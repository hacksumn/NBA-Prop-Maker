import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from prizepicks_scraper import _parse_projections
from scripts import capture_prizepicks_closing_snapshots as capture_module
from scripts.capture_prizepicks_closing_snapshots import (
    capture_sportsbook_t60,
    classify_due_rows,
    match_odds_event,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class PrizePicksClosingCaptureTests(unittest.TestCase):
    def test_parser_preserves_event_and_utc_tip_but_uses_local_slate_date(self):
        payload = {
            "included": [
                {"type": "new_player", "id": "p1", "attributes": {"display_name": "Test Player"}},
                {
                    "type": "game",
                    "id": "g1",
                    "attributes": {
                        "start_time": "2026-11-02T01:00:00Z",
                        "description": "LAL @ BOS",
                    },
                },
            ],
            "data": [
                {
                    "attributes": {
                        "stat_type": "Points",
                        "line_score": 20.5,
                        "odds_type": "standard",
                        "is_promo": False,
                    },
                    "relationships": {
                        "new_player": {"data": {"id": "p1"}},
                        "game": {"data": {"id": "g1"}},
                    },
                }
            ],
        }
        parsed = _parse_projections(payload)
        self.assertEqual("g1", parsed.loc[0, "event_id"])
        self.assertEqual("2026-11-02T01:00:00Z", parsed.loc[0, "commence_time"])
        self.assertEqual("2026-11-01", parsed.loc[0, "game_date"])
        self.assertEqual("LAL @ BOS", parsed.loc[0, "game_description"])

    def test_rows_are_classified_only_inside_t60_or_t5_window(self):
        lines = pd.DataFrame(
            [
                {"event_id": "g60", "commence_time": "2026-11-02T02:00:00Z"},
                {"event_id": "g5", "commence_time": "2026-11-02T01:05:00Z"},
                {"event_id": "later", "commence_time": "2026-11-02T04:00:00Z"},
            ]
        )
        due = classify_due_rows(lines, "2026-11-02T01:00:00Z")
        self.assertEqual({"g60"}, set(due["t_minus_60"]["event_id"]))
        self.assertEqual({"g5"}, set(due["t_minus_5"]["event_id"]))

    def test_match_odds_event_uses_both_teams_when_tips_overlap(self):
        rows = pd.DataFrame(
            [{
                "event_id": "pp1",
                "commence_time": "2026-11-02T01:00:00Z",
                "game_description": "LAL @ BOS",
            }]
        )
        events = [
            {
                "id": "wrong",
                "commence_time": "2026-11-02T01:00:00Z",
                "home_team": "New York Knicks",
                "away_team": "Chicago Bulls",
            },
            {
                "id": "right",
                "commence_time": "2026-11-02T01:00:00Z",
                "home_team": "Boston Celtics",
                "away_team": "Los Angeles Lakers",
            },
        ]
        self.assertEqual("right", match_odds_event(events, rows)["id"])

    def test_paired_t60_capture_persists_prices_and_actual_credit_cost(self):
        rows = pd.DataFrame(
            [{
                "event_id": "pp1",
                "commence_time": "2026-11-02T01:00:00Z",
                "game_description": "LAL @ BOS",
            }]
        )
        events = [{
            "id": "odds1",
            "commence_time": "2026-11-02T01:00:00Z",
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
        }]
        odds = {
            **events[0],
            "bookmakers": [{
                "key": "fanduel",
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {"name": "Over", "description": "Test Player", "point": 20.5, "price": -115},
                        {"name": "Under", "description": "Test Player", "point": 20.5, "price": -105},
                    ],
                }],
            }],
        }
        session = FakeSession([
            FakeResponse(events, headers={"x-requests-last": "0", "x-requests-remaining": "13436"}),
            FakeResponse(odds, headers={"x-requests-last": "1", "x-requests-remaining": "13435"}),
        ])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(capture_module, "ROOT", root),
                patch.object(capture_module, "PAIR_CHECKPOINT_PATH", root / "pair.json"),
                patch.object(capture_module, "PAIR_RAW_DIR", root / "raw"),
                patch.object(capture_module, "PAIR_PARSED_DIR", root / "parsed"),
                patch.object(capture_module, "MASTER_PATH", root / "historical_lines.csv"),
            ):
                result = capture_sportsbook_t60(
                    rows,
                    "2026-11-02T00:00:00Z",
                    api_key="test-key",
                    max_credits=9,
                    session=session,
                )
                checkpoint = capture_module.load_pair_checkpoint(root / "pair.json")
                master = pd.read_csv(root / "historical_lines.csv")
        self.assertEqual("completed_with_data", result["status"])
        self.assertEqual(1, result["credits_used"])
        self.assertEqual(1, checkpoint["credits_used"])
        self.assertEqual(-115, master.loc[0, "over_price"])
        self.assertEqual(-105, master.loc[0, "under_price"])
        self.assertEqual(2, len(session.calls))

    def test_credit_cap_stops_before_paid_event_request(self):
        rows = pd.DataFrame(
            [{
                "event_id": "pp1",
                "commence_time": "2026-11-02T01:00:00Z",
                "game_description": "LAL @ BOS",
            }]
        )
        events = [{
            "id": "odds1",
            "commence_time": "2026-11-02T01:00:00Z",
            "home_team": "Boston Celtics",
            "away_team": "Los Angeles Lakers",
        }]
        session = FakeSession([FakeResponse(events, headers={"x-requests-last": "0"})])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(capture_module, "ROOT", root),
                patch.object(capture_module, "PAIR_CHECKPOINT_PATH", root / "pair.json"),
                patch.object(capture_module, "PAIR_RAW_DIR", root / "raw"),
                patch.object(capture_module, "PAIR_PARSED_DIR", root / "parsed"),
                patch.object(capture_module, "MASTER_PATH", root / "historical_lines.csv"),
            ):
                result = capture_sportsbook_t60(
                    rows,
                    "2026-11-02T00:00:00Z",
                    api_key="test-key",
                    max_credits=8,
                    session=session,
                )
        self.assertEqual("credit_cap_reached", result["status"])
        self.assertEqual(1, len(session.calls))

    def test_t5_capture_never_calls_sportsbook_pairing(self):
        lines = pd.DataFrame(
            [{
                "game_date": "2026-11-01",
                "player": "Test Player",
                "player_norm": "test player",
                "prop": "player_points",
                "line": 20.5,
                "num_books": 1,
                "projection_type": "standard",
                "is_promo": False,
                "event_id": "pp5",
                "commence_time": "2026-11-02T01:05:00Z",
                "game_description": "LAL @ BOS",
            }]
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(capture_module, "ROOT", root),
                patch.object(capture_module, "RAW_DIR", root / "pp_raw"),
                patch.object(capture_module, "PARSED_DIR", root / "pp_parsed"),
                patch.object(capture_module, "CHECKPOINT_PATH", root / "pp_checkpoint.json"),
                patch.object(capture_module, "MASTER_PATH", root / "historical_lines.csv"),
                patch.object(capture_module, "parse_prizepicks_payload", return_value=lines),
                patch.object(capture_module, "capture_sportsbook_t60") as pair_mock,
            ):
                result = capture_module.capture_once(
                    captured_at_utc="2026-11-02T01:00:00Z",
                    payload={},
                    write=True,
                    pair_sportsbook_t60=True,
                )
        pair_mock.assert_not_called()
        self.assertEqual(0, result["odds_api_credits_used"])


if __name__ == "__main__":
    unittest.main()
