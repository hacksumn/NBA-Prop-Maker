from pathlib import Path

import pandas as pd

from scripts import capture_fanduel_closing_snapshots as capture


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return self.response


def test_plan_creates_t60_and_t5_for_each_local_slate_event():
    events = [
        {
            "id": "event-1",
            "commence_time": "2026-10-28T00:00:00Z",
            "home_team": "Home",
            "away_team": "Away",
        }
    ]

    plan = capture.build_plan(events, "2026-10-27")

    assert plan["snapshot_kind"].tolist() == ["t_minus_60", "t_minus_5"]
    assert plan["target_ts_utc"].tolist() == [
        "2026-10-27T23:00:00Z",
        "2026-10-27T23:55:00Z",
    ]


def test_event_capture_saves_exact_fanduel_prices_and_charges_actual_cost(
    tmp_path: Path, monkeypatch
):
    checkpoint_path = tmp_path / "checkpoint.json"
    raw_root = tmp_path / "raw"
    parsed_root = tmp_path / "parsed"
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    monkeypatch.setattr(capture, "CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(capture, "RAW_ROOT", raw_root)
    monkeypatch.setattr(capture, "PARSED_ROOT", parsed_root)
    monkeypatch.setattr(capture, "save_checkpoint", lambda payload, path=checkpoint_path: capture.atomic_write_json(checkpoint_path, payload))
    now = pd.Timestamp("2026-10-27T23:00:00Z")
    monkeypatch.setattr(capture, "utc_now", lambda: now)
    payload = {
        "id": "event-1",
        "commence_time": "2026-10-28T00:00:00Z",
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {"name": "Over", "description": "Test Player", "point": 20.5, "price": -110},
                            {"name": "Under", "description": "Test Player", "point": 20.5, "price": -120},
                        ],
                    }
                ],
            }
        ],
    }
    session = FakeSession(
        FakeResponse(payload, headers={
            "x-requests-last": "9",
            "x-requests-used": "109",
            "x-requests-remaining": "891",
        })
    )
    row = pd.Series({
        "event_id": "event-1",
        "snapshot_kind": "t_minus_60",
        "game_date": "2026-10-27",
        "target_ts_utc": "2026-10-27T23:00:00Z",
        "commence_time": "2026-10-28T00:00:00Z",
    })
    checkpoint = capture.load_checkpoint(checkpoint_path)
    checkpoint["last_api_balance"] = {"requests_remaining": 900}

    result = capture.capture_event(
        row, "secret", 200, 100, checkpoint, session=session
    )

    assert result["credits_used"] == 9
    assert result["rows_parsed"] == 1
    parsed = pd.read_csv(tmp_path / result["parsed_path"])
    assert parsed.loc[0, "line"] == 20.5
    assert parsed.loc[0, "over_price"] == -110
    assert parsed.loc[0, "under_price"] == -120
    assert parsed.loc[0, "source"] == "odds_api_fanduel_t_minus_60"
    assert session.calls[0][1]["bookmakers"] == "fanduel"


def test_provider_reserve_blocks_before_paid_request():
    row = pd.Series({
        "event_id": "event-1",
        "snapshot_kind": "t_minus_5",
        "game_date": "2026-10-27",
        "target_ts_utc": "2026-10-27T23:55:00Z",
        "commence_time": "2026-10-28T00:00:00Z",
    })
    checkpoint = capture.load_checkpoint(Path("does-not-exist.json"))
    checkpoint["last_api_balance"] = {"requests_remaining": 105}
    session = FakeSession(FakeResponse({}))

    result = capture.capture_event(row, "secret", 200, 100, checkpoint, session=session)

    assert result["status"] == "provider_reserve_reached"
    assert session.calls == []
