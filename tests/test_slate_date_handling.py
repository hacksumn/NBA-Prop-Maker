import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from nba_props import (
    _line_source_policy_allows_publish,
    _parse_historical_event_odds,
    backfill_historical_lines,
    _coerce_slate_date,
    _merge_historical_lines,
    _select_placeable_projection_offers,
    _select_preferred_line_rows,
    _select_active_slate_lines,
    _select_cached_lines_for_predict,
    _select_lines_for_exact_slate_date,
    fetch_vegas_lines,
)
from prizepicks_scraper import _parse_projections


class SlateDateHandlingTests(unittest.TestCase):
    def test_live_card_publish_policy_requires_exact_configured_source(self):
        self.assertTrue(_line_source_policy_allows_publish("PrizePicks", "prizepicks"))
        self.assertFalse(_line_source_policy_allows_publish("fanduel", "prizepicks"))
        self.assertFalse(_line_source_policy_allows_publish("", "prizepicks"))

    def test_prizepicks_failure_does_not_spend_reserved_odds_credits(self):
        with patch("nba_props.fetch_prizepicks_lines", return_value=pd.DataFrame()), patch(
            "nba_props.requests.get"
        ) as odds_get:
            lines, source = fetch_vegas_lines(prefer_prizepicks=True)

        self.assertTrue(lines.empty)
        self.assertEqual("prizepicks", source)
        odds_get.assert_not_called()

    @patch("nba_props.requests.get")
    def test_backfill_without_credit_cap_sends_no_requests(self, mock_get):
        self.assertFalse(backfill_historical_lines(days=2, max_credits=0))
        mock_get.assert_not_called()

    def test_coerce_slate_date_converts_utc_event_time_to_local_date(self):
        self.assertEqual(_coerce_slate_date("2026-05-12T00:00:00Z"), "2026-05-11")

    def test_parse_projections_uses_payload_slate_date(self):
        payload = {
            "data": [
                {
                    "type": "projection",
                    "id": "proj-1",
                    "attributes": {
                        "stat_type": "Points",
                        "line_score": 20.5,
                        "start_time": "2026-04-12T18:10:00.000-04:00",
                        "projection_type": "Single Stat",
                        "odds_type": "goblin",
                        "is_promo": True,
                    },
                    "relationships": {
                        "new_player": {"data": {"id": "player-1"}},
                        "game": {"data": {"id": "game-1"}},
                    },
                },
                {
                    "type": "projection",
                    "id": "proj-2",
                    "attributes": {
                        "stat_type": "Points",
                        "line_score": 21.5,
                        "start_time": "2026-04-13T18:10:00.000-04:00",
                        "projection_type": "Single Stat",
                        "odds_type": "standard",
                        "is_promo": False,
                    },
                    "relationships": {
                        "new_player": {"data": {"id": "player-1"}},
                        "game": {"data": {"id": "game-2"}},
                    },
                },
            ],
            "included": [
                {
                    "type": "new_player",
                    "id": "player-1",
                    "attributes": {"display_name": "Scottie Barnes"},
                },
                {
                    "type": "game",
                    "id": "game-1",
                    "attributes": {"start_time": "2026-04-12T18:00:00.000-04:00"},
                },
                {
                    "type": "game",
                    "id": "game-2",
                    "attributes": {"start_time": "2026-04-13T18:00:00.000-04:00"},
                },
            ],
        }

        parsed = _parse_projections(payload)

        self.assertEqual(parsed["game_date"].tolist(), ["2026-04-12", "2026-04-13"])
        self.assertEqual(parsed["player"].tolist(), ["Scottie Barnes", "Scottie Barnes"])
        self.assertEqual(parsed["prop"].tolist(), ["player_points", "player_points"])
        self.assertEqual(parsed["projection_type"].tolist(), ["goblin", "standard"])
        self.assertEqual(parsed["is_promo"].tolist(), [True, False])

    def test_prizepicks_contracts_never_create_a_median_line(self):
        payload = {
            "data": [
                {
                    "type": "projection", "id": "standard",
                    "attributes": {
                        "stat_type": "Points", "line_score": 20.5,
                        "start_time": "2026-10-21T19:00:00-05:00",
                        "odds_type": "standard", "is_promo": False,
                    },
                    "relationships": {"new_player": {"data": {"id": "p1"}}},
                },
                {
                    "type": "projection", "id": "goblin",
                    "attributes": {
                        "stat_type": "Points", "line_score": 17.5,
                        "start_time": "2026-10-21T19:00:00-05:00",
                        "odds_type": "goblin", "is_promo": False,
                    },
                    "relationships": {"new_player": {"data": {"id": "p1"}}},
                },
            ],
            "included": [{
                "type": "new_player", "id": "p1",
                "attributes": {"display_name": "Test Player"},
            }],
        }

        parsed = _parse_projections(payload)

        self.assertEqual(set(parsed["line"]), {17.5, 20.5})
        self.assertNotIn(19.0, parsed["line"].tolist())
        selected = _select_placeable_projection_offers(parsed)
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected.loc[0, "line"]), 20.5)
        self.assertEqual(selected.loc[0, "projection_type"], "standard")

    def test_select_active_slate_lines_prefers_nearest_upcoming_date(self):
        lines = pd.DataFrame(
            [
                {"game_date": "2026-04-12", "player": "Scottie Barnes", "prop": "player_points", "line": 20.5},
                {"game_date": "2026-04-13", "player": "Devin Carter", "prop": "player_assists", "line": 6.0},
                {"game_date": "2026-04-13", "player": "Brandon Ingram", "prop": "player_rebounds", "line": 7.5},
            ]
        )

        active, slate_date = _select_active_slate_lines(lines, as_of=datetime(2026, 4, 11, 19, 0, 0))

        self.assertEqual(slate_date, "2026-04-12")
        self.assertEqual(active["game_date"].tolist(), ["2026-04-12"])

    def test_select_active_slate_lines_falls_back_to_latest_past_date(self):
        lines = pd.DataFrame(
            [
                {"game_date": "2026-04-10", "player": "Scottie Barnes", "prop": "player_points", "line": 20.5},
                {"game_date": "2026-04-11", "player": "Devin Carter", "prop": "player_assists", "line": 6.0},
            ]
        )

        active, slate_date = _select_active_slate_lines(lines, as_of=datetime(2026, 4, 12, 9, 0, 0))

        self.assertEqual(slate_date, "2026-04-11")
        self.assertEqual(active["game_date"].tolist(), ["2026-04-11"])

    def test_select_lines_for_exact_slate_date_requires_exact_match(self):
        lines = pd.DataFrame(
            [
                {"game_date": "2026-04-12", "player": "Scottie Barnes", "prop": "player_points", "line": 20.5},
                {"game_date": "2026-04-14", "player": "Devin Carter", "prop": "player_assists", "line": 6.0},
            ]
        )

        active, slate_date = _select_lines_for_exact_slate_date(lines, "2026-04-13")

        self.assertTrue(active.empty)
        self.assertIsNone(slate_date)

    def test_select_lines_for_exact_slate_date_returns_only_requested_day(self):
        lines = pd.DataFrame(
            [
                {"game_date": "2026-04-12", "player": "Scottie Barnes", "prop": "player_points", "line": 20.5},
                {"game_date": "2026-04-13", "player": "Devin Carter", "prop": "player_assists", "line": 6.0},
                {"game_date": "2026-04-13", "player": "Brandon Ingram", "prop": "player_rebounds", "line": 7.5},
            ]
        )

        active, slate_date = _select_lines_for_exact_slate_date(lines, "2026-04-13")

        self.assertEqual(slate_date, "2026-04-13")
        self.assertEqual(active["game_date"].tolist(), ["2026-04-13", "2026-04-13"])

    def test_predict_prefers_cached_future_slate_over_refresh(self):
        lines = pd.DataFrame(
            [
                {
                    "game_date": "2026-04-17",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 8.5,
                    "num_books": 1,
                    "source": "prizepicks_scraper",
                    "snapshot_ts": "2026-04-16 09:43:23",
                },
                {
                    "game_date": "2026-04-18",
                    "player": "Scottie Barnes",
                    "player_norm": "scottie barnes",
                    "prop": "player_points",
                    "line": 20.5,
                    "num_books": 1,
                    "source": "prizepicks_scraper",
                    "snapshot_ts": "2026-04-16 09:43:23",
                },
            ]
        )

        active, slate_date, line_source, mode = _select_cached_lines_for_predict(
            lines,
            "2026-04-16",
            as_of=datetime(2026, 4, 16, 17, 3, 18),
        )

        self.assertEqual(slate_date, "2026-04-17")
        self.assertEqual(line_source, "prizepicks")
        self.assertEqual(mode, "active_future")
        self.assertEqual(active["game_date"].tolist(), ["2026-04-17"])

    def test_predict_can_ignore_prizepicks_cache_for_fanduel_mode(self):
        lines = pd.DataFrame(
            [
                {
                    "game_date": "2026-04-17",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 8.5,
                    "num_books": 1,
                    "source": "prizepicks_scraper",
                    "snapshot_ts": "2026-04-16 09:43:23",
                    "projection_type": "demon",
                    "is_promo": False,
                },
                {
                    "game_date": "2026-04-17",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 7.5,
                    "num_books": 1,
                    "source": "odds_api_fanduel",
                    "snapshot_ts": "2026-04-16 10:43:23",
                    "projection_type": "standard",
                    "is_promo": False,
                },
            ]
        )

        active, slate_date, line_source, mode = _select_cached_lines_for_predict(
            lines,
            "2026-04-17",
            allowed_sources=["fanduel"],
            standard_only=True,
        )

        self.assertEqual(slate_date, "2026-04-17")
        self.assertEqual(line_source, "fanduel")
        self.assertEqual(mode, "exact_today")
        self.assertEqual(float(active.loc[0, "line"]), 7.5)

    def test_fanduel_mode_does_not_fall_forward_to_future_cached_slate(self):
        lines = pd.DataFrame(
            [
                {
                    "game_date": "2026-05-12",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 7.5,
                    "num_books": 1,
                    "source": "fanduel",
                    "snapshot_ts": "2026-05-11 10:43:23",
                    "projection_type": "standard",
                    "is_promo": False,
                }
            ]
        )

        active, slate_date, line_source, mode = _select_cached_lines_for_predict(
            lines,
            "2026-05-11",
            allowed_sources=["fanduel"],
            standard_only=True,
            allow_future_slate=False,
        )

        self.assertEqual(len(active), 0)
        self.assertEqual(slate_date, "2026-05-12")
        self.assertEqual(line_source, "fanduel")
        self.assertEqual(mode, "missing_exact_today")

    def test_historical_line_merge_preserves_prizepicks_and_odds_api(self):
        existing = pd.DataFrame(
            [
                {
                    "game_date": "2026-04-17",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 8.5,
                    "num_books": 1,
                    "source": "prizepicks_scraper",
                    "snapshot_ts": "2026-04-16 09:43:23",
                    "projection_type": "standard",
                    "is_promo": False,
                }
            ]
        )
        incoming = pd.DataFrame(
            [
                {
                    "game_date": "2026-04-17",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 9.5,
                    "num_books": 24,
                    "source": "odds_api",
                    "snapshot_ts": "2026-04-16 17:03:38",
                    "projection_type": "standard",
                    "is_promo": False,
                }
            ]
        )

        merged = _merge_historical_lines(existing, incoming)

        self.assertEqual(len(merged), 2)
        preferred = _select_preferred_line_rows(merged)
        self.assertEqual(len(preferred), 1)
        self.assertEqual(preferred.loc[0, "source"], "prizepicks_scraper")
        self.assertEqual(float(preferred.loc[0, "line"]), 8.5)

    def test_historical_line_merge_preserves_fanduel_and_prizepicks(self):
        existing = pd.DataFrame(
            [
                {
                    "game_date": "2026-04-17",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 8.5,
                    "num_books": 1,
                    "source": "prizepicks_scraper",
                    "snapshot_ts": "2026-04-16 09:43:23",
                    "projection_type": "demon",
                    "is_promo": False,
                }
            ]
        )
        incoming = pd.DataFrame(
            [
                {
                    "game_date": "2026-04-17",
                    "player": "Al Horford",
                    "player_norm": "al horford",
                    "prop": "player_points",
                    "line": 7.5,
                    "num_books": 1,
                    "source": "odds_api_fanduel",
                    "snapshot_ts": "2026-04-16 17:03:38",
                    "projection_type": "standard",
                    "is_promo": False,
                }
            ]
        )

        merged = _merge_historical_lines(existing, incoming)

        self.assertEqual(len(merged), 2)
        preferred = _select_preferred_line_rows(merged)
        self.assertEqual(len(preferred), 1)
        self.assertEqual(preferred.loc[0, "source"], "prizepicks_scraper")
        self.assertEqual(float(preferred.loc[0, "line"]), 8.5)

    def test_historical_line_merge_preserves_intraday_snapshots(self):
        first = pd.DataFrame([{
            "game_date": "2026-04-17", "player": "Al Horford", "player_norm": "al horford",
            "prop": "player_points", "line": 7.5, "num_books": 1,
            "source": "odds_api_fanduel", "snapshot_ts": "2026-04-16 09:00:00",
        }])
        second = first.copy()
        second["line"] = 8.5
        second["snapshot_ts"] = "2026-04-16 16:00:00"

        merged = _merge_historical_lines(first, second)

        self.assertEqual(len(merged), 2)
        preferred = _select_preferred_line_rows(merged)
        self.assertEqual(float(preferred.loc[0, "line"]), 8.5)

    def test_preferred_view_excludes_post_tip_snapshot(self):
        rows = pd.DataFrame([
            {
                "game_date": "2026-04-17", "player": "Al Horford", "player_norm": "al horford",
                "prop": "player_points", "line": 7.5, "num_books": 1,
                "source": "odds_api_fanduel", "snapshot_ts": "2026-04-16T23:55:00Z",
                "commence_time": "2026-04-17T00:00:00Z",
            },
            {
                "game_date": "2026-04-17", "player": "Al Horford", "player_norm": "al horford",
                "prop": "player_points", "line": 8.5, "num_books": 1,
                "source": "odds_api_fanduel", "snapshot_ts": "2026-04-17T00:05:00Z",
                "commence_time": "2026-04-17T00:00:00Z",
            },
        ])

        preferred = _select_preferred_line_rows(rows)

        self.assertEqual(len(preferred), 1)
        self.assertEqual(float(preferred.loc[0, "line"]), 7.5)

    def test_cutoff_excludes_unknown_snapshot_timestamp(self):
        rows = pd.DataFrame([{
            "game_date": "2026-04-17", "player": "Al Horford",
            "player_norm": "al horford", "prop": "player_points",
            "line": 7.5, "num_books": 1, "source": "odds_api_fanduel",
            "snapshot_ts": "unknown",
        }])

        preferred = _select_preferred_line_rows(rows, as_of="2026-04-16T23:00:00Z")

        self.assertTrue(preferred.empty)

    def test_invalid_historical_cutoff_fails_closed(self):
        rows = pd.DataFrame([{
            "game_date": "2026-04-17", "player": "Al Horford",
            "player_norm": "al horford", "prop": "player_points",
            "line": 7.5, "num_books": 1, "source": "odds_api_fanduel",
            "snapshot_ts": "2026-04-16T22:00:00Z",
        }])

        with self.assertRaisesRegex(ValueError, "Invalid line-selection cutoff"):
            _select_preferred_line_rows(rows, as_of="not-a-timestamp")

    def test_historical_mode_excludes_unknown_snapshot_timestamp(self):
        rows = pd.DataFrame([{
            "game_date": "2026-04-17", "player": "Al Horford",
            "player_norm": "al horford", "prop": "player_points",
            "line": 7.5, "num_books": 1, "source": "odds_api_fanduel",
            "snapshot_ts": "unknown",
        }])

        preferred = _select_preferred_line_rows(rows, require_known_snapshot=True)

        self.assertTrue(preferred.empty)

    def test_historical_odds_parser_keeps_book_and_both_prices(self):
        payload = {
            "timestamp": "2026-04-16T16:00:00Z",
            "data": {
                "id": "event-1",
                "commence_time": "2026-04-17T00:00:00Z",
                "bookmakers": [{
                    "key": "fanduel",
                    "markets": [{
                        "key": "player_points",
                        "outcomes": [
                            {"name": "Over", "description": "Al Horford", "point": 7.5, "price": -105},
                            {"name": "Under", "description": "Al Horford", "point": 7.5, "price": -115},
                        ],
                    }],
                }],
            },
        }

        parsed = _parse_historical_event_odds(payload, "2026-04-16T16:02:00Z")

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed.loc[0, "source"], "odds_api_fanduel")
        self.assertEqual(parsed.loc[0, "bookmaker"], "fanduel")
        self.assertEqual(float(parsed.loc[0, "over_price"]), -105.0)
        self.assertEqual(float(parsed.loc[0, "under_price"]), -115.0)

    def test_preferred_view_selects_balanced_main_line_from_alternates(self):
        rows = pd.DataFrame([
            {
                "game_date": "2026-04-17", "player": "Al Horford", "player_norm": "al horford",
                "prop": "player_points", "line": 7.5, "num_books": 1,
                "source": "odds_api_bovada", "snapshot_ts": "2026-04-16T16:00:00Z",
                "over_price": -110, "under_price": -110,
            },
            {
                "game_date": "2026-04-17", "player": "Al Horford", "player_norm": "al horford",
                "prop": "player_points", "line": 8.5, "num_books": 1,
                "source": "odds_api_bovada", "snapshot_ts": "2026-04-16T16:00:00Z",
                "over_price": 180, "under_price": -240,
            },
        ])

        preferred = _select_preferred_line_rows(rows)

        self.assertEqual(len(preferred), 1)
        self.assertEqual(float(preferred.loc[0, "line"]), 7.5)


if __name__ == "__main__":
    unittest.main()
