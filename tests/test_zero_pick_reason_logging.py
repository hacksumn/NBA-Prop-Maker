"""Zero-pick days must say why (logging-only contract; no behavior change).

Covers nba_props._describe_empty_slate and the fetch_vegas_lines path that
prints it when the Odds API lists events but none is dated on the target slate.
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import nba_props


class _Response:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload


class DescribeEmptySlateTests(unittest.TestCase):
    def test_reports_listed_dates_clock_and_credits(self):
        events = [
            {"id": "a", "commence_time": "2026-11-05T01:00:00Z"},
            {"id": "b", "commence_time": "2026-11-06T01:00:00Z"},
        ]
        message = nba_props._describe_empty_slate(
            events, "2026-11-03", remaining_credits=452, now=datetime(2026, 11, 3, 23, 6)
        )
        self.assertIn("2026-11-03", message)
        self.assertIn("2 event(s) listed", message)
        self.assertIn("2026-11-04..2026-11-05", message)  # UTC 01:00Z is the prior Chicago evening
        self.assertIn("local time 23:06", message)
        self.assertIn("452 provider credits remaining", message)
        self.assertIn("already started/finished", message)

    def test_handles_empty_and_unparseable_events(self):
        message = nba_props._describe_empty_slate(
            [{"id": "x", "commence_time": "not-a-date"}], "2026-11-03", None, now=datetime(2026, 11, 3, 9, 0)
        )
        self.assertIn("no events listed", message)
        self.assertIn("provider credits unknown", message)
        self.assertIn("local time 09:00", message)
        self.assertIn("no events listed", nba_props._describe_empty_slate(None, "2026-11-03"))


class FetchVegasLinesEmptySlateTests(unittest.TestCase):
    def test_no_event_today_prints_reason_and_returns_empty_frame(self):
        later = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        response = _Response(
            [{"id": "event-1", "commence_time": f"{later}T23:00:00Z"}],
            headers={"x-requests-remaining": "452"},
        )
        with patch.dict(
            nba_props.CONFIG,
            {"odds_api_key": "test", "allow_live_odds_api_spend": True},
        ), patch("nba_props.requests.get", return_value=response) as getter, patch(
            "builtins.print"
        ) as printer:
            lines, source = nba_props.fetch_vegas_lines(
                prefer_prizepicks=False,
                preferred_bookmaker="fanduel",
                allow_prizepicks_fallback=False,
            )
        self.assertTrue(lines.empty)
        self.assertEqual(source, "fanduel")
        self.assertEqual(getter.call_count, 1)  # events lookup only; no paid event request
        messages = " ".join(str(call.args[0]) for call in printer.call_args_list if call.args)
        self.assertIn("returned no NBA events for today's slate", messages)
        self.assertIn("1 event(s) listed", messages)
        self.assertIn("452 provider credits remaining", messages)


if __name__ == "__main__":
    unittest.main()
