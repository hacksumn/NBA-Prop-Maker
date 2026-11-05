"""Live FanDuel fetch must never pay for or store in-play lines; credit alarms and
fetch messages must reach the run log.

Context: on 2026-11-04 a 23:07 CT run stored 135 in-play rows (3 events that
tipped at 20:48 CT) in historical_lines.csv because /events also lists live
games. Step 2's fetch messages also reached the console only, not the run log.
"""
import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import nba_props
import run_daily


class _Response:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _odds_payload(player):
    return {
        "bookmakers": [{
            "key": "fanduel",
            "markets": [{
                "key": "player_assists",
                "outcomes": [
                    {"description": player, "name": "Over", "point": 4.5, "price": -110},
                    {"description": player, "name": "Under", "point": 4.5, "price": -110},
                ],
            }],
        }],
    }


def _no_sanitize(frame, **kwargs):
    return frame, {"canonicalized_rows": 0, "dropped_rows": 0}


class SplitPregameEventsTests(unittest.TestCase):
    def test_started_unknown_and_future_events_are_split(self):
        now = pd.Timestamp("2026-11-05T05:07:00Z")
        events = [
            {"id": "tipped", "commence_time": "2026-11-05T02:48:00Z"},
            {"id": "exact", "commence_time": "2026-11-05T05:07:00Z"},
            {"id": "later", "commence_time": "2026-11-05T05:30:00Z"},
            {"id": "unknown", "commence_time": ""},
        ]
        pregame, started = nba_props._split_pregame_events(events, now)
        self.assertEqual([e["id"] for e in started], ["tipped", "exact"])
        self.assertEqual([e["id"] for e in pregame], ["later", "unknown"])

    def test_naive_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            nba_props._split_pregame_events([], pd.Timestamp("2026-11-05T05:07:00"))


class FetchSkipsStartedEventsTests(unittest.TestCase):
    def _config(self, tmp, **extra):
        cfg = {
            "odds_api_key": "test",
            "allow_live_odds_api_spend": True,
            "line_archive_dir": Path(tmp),
            "live_odds_min_remaining_credits": 100,
            "odds_api_low_credit_warning": 750,
        }
        cfg.update(extra)
        return cfg

    def test_only_pregame_events_are_requested_and_stored(self):
        today = _today()
        events = [
            {"id": "started", "commence_time": f"{today}T12:00:00Z"},
            {"id": "pregame", "commence_time": f"{today}T23:00:00Z"},
        ]
        responses = [
            _Response(events, headers={"x-requests-remaining": "19982"}),
            _Response(_odds_payload("Pregame Player")),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            nba_props.CONFIG, self._config(tmp)
        ), patch("nba_props.requests.get", side_effect=responses) as getter, patch(
            "nba_props._utc_now", return_value=pd.Timestamp(f"{today}T18:00:00Z")
        ), patch("nba_props.sanitize_player_names", side_effect=_no_sanitize), patch(
            "builtins.print"
        ) as printer:
            lines, source = nba_props.fetch_vegas_lines(
                preferred_bookmaker="fanduel", allow_prizepicks_fallback=False
            )
        self.assertEqual(source, "fanduel")
        self.assertEqual(getter.call_count, 2)
        self.assertIn("/events/pregame/odds", getter.call_args_list[1].args[0])
        self.assertEqual(set(lines["event_id"]), {"pregame"})
        messages = " ".join(str(c.args[0]) for c in printer.call_args_list)
        self.assertIn("Skipping 1 of 2 slate event(s)", messages)

    def test_budget_counts_only_pregame_events(self):
        # 12 listed but 11 tipped: 1 x 9 + 100 reserve = 109 needed, 120 available.
        today = _today()
        events = [
            {"id": f"started-{i}", "commence_time": f"{today}T12:00:00Z"} for i in range(11)
        ] + [{"id": "pregame", "commence_time": f"{today}T23:00:00Z"}]
        responses = [
            _Response(events, headers={"x-requests-remaining": "120"}),
            _Response(_odds_payload("Pregame Player")),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            nba_props.CONFIG, self._config(tmp)
        ), patch("nba_props.requests.get", side_effect=responses) as getter, patch(
            "nba_props._utc_now", return_value=pd.Timestamp(f"{today}T18:00:00Z")
        ), patch("nba_props.sanitize_player_names", side_effect=_no_sanitize), patch(
            "builtins.print"
        ):
            lines, _ = nba_props.fetch_vegas_lines(
                preferred_bookmaker="fanduel", allow_prizepicks_fallback=False
            )
        self.assertEqual(getter.call_count, 2)
        self.assertEqual(len(lines), 1)

    def test_all_started_spends_nothing_and_says_why(self):
        today = _today()
        events = [{"id": "a", "commence_time": f"{today}T12:00:00Z"}]
        with patch.dict(
            nba_props.CONFIG, {"odds_api_key": "test", "allow_live_odds_api_spend": True}
        ), patch(
            "nba_props.requests.get",
            return_value=_Response(events, headers={"x-requests-remaining": "19982"}),
        ) as getter, patch(
            "nba_props._utc_now", return_value=pd.Timestamp(f"{today}T18:00:00Z")
        ), patch("builtins.print") as printer:
            lines, source = nba_props.fetch_vegas_lines(
                preferred_bookmaker="fanduel", allow_prizepicks_fallback=False
            )
        self.assertTrue(lines.empty)
        self.assertEqual(source, "fanduel")
        self.assertEqual(getter.call_count, 1)
        messages = " ".join(str(c.args[0]) for c in printer.call_args_list)
        self.assertIn("have already started", messages)
        self.assertIn("none spent", messages)

    def test_low_credit_warning_is_printed_below_floor_only(self):
        for remaining, expect in (("749", True), ("750", False)):
            with patch.dict(
                nba_props.CONFIG,
                {
                    "odds_api_key": "test",
                    "allow_live_odds_api_spend": True,
                    "odds_api_low_credit_warning": 750,
                },
            ), patch(
                "nba_props.requests.get",
                return_value=_Response([], headers={"x-requests-remaining": remaining}),
            ), patch("builtins.print") as printer:
                nba_props.fetch_vegas_lines(
                    preferred_bookmaker="fanduel", allow_prizepicks_fallback=False
                )
            messages = " ".join(str(c.args[0]) for c in printer.call_args_list)
            self.assertEqual("ODDS API CREDITS LOW" in messages, expect, remaining)


class StdoutToLogTests(unittest.TestCase):
    def test_printed_lines_reach_logger_and_alarms_are_warnings(self):
        def noisy(value):
            print("  Fetched 77 validated player props")
            print("  !!! ODDS API CREDITS LOW: 109 remaining !!!")
            print("partial line without newline", end="")
            return value

        with self.assertLogs(run_daily.logger, level=logging.INFO) as captured:
            result = run_daily._call_with_logged_stdout(noisy, 42)
        self.assertEqual(result, 42)
        joined = "\n".join(captured.output)
        self.assertIn("INFO", captured.output[0])
        self.assertIn("Fetched 77 validated player props", joined)
        self.assertTrue(any(r.startswith("WARNING") and "CREDITS LOW" in r for r in captured.output))
        self.assertIn("partial line without newline", joined)

    def test_output_is_flushed_when_callee_raises(self):
        def failing():
            print("  Odds API event x returned HTTP 500")
            raise RuntimeError("boom")

        with self.assertLogs(run_daily.logger, level=logging.INFO) as captured:
            with self.assertRaises(RuntimeError):
                run_daily._call_with_logged_stdout(failing)
        self.assertIn("HTTP 500", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
