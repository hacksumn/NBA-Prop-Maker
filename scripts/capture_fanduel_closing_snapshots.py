"""Capture exact FanDuel player-prop contracts near T-60 and T-5.

The watcher uses the free Odds API event list to plan each NBA game's windows,
then requests only that event and only FanDuel.  Every paid response is saved
before parsing, checkpointed by event/window, and merged losslessly into
``data/historical_lines.csv``.  Daily and provider-balance guards stop spend
before the configured limits are crossed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_props import _merge_historical_lines, _write_historical_lines  # noqa: E402
from scripts.execute_clv_snapshot_manifest import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    execution_lock,
    parse_event_odds,
    response_balance,
    response_cost,
)


LOGGER = logging.getLogger(__name__)
EVENTS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
EVENT_ODDS_URL = EVENTS_URL + "/{event_id}/odds"
MARKETS = (
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
    "player_steals",
    "player_blocks",
    "player_turnovers",
)
CHECKPOINT = ROOT / "data" / "fanduel_closing_capture_checkpoint.json"
RAW_ROOT = ROOT / "data" / "odds_api_fanduel_closing_raw"
PARSED_ROOT = ROOT / "data" / "odds_api_fanduel_closing_parsed"
MASTER = ROOT / "data" / "historical_lines.csv"
LOCK = ROOT / "data" / "fanduel_closing_watcher.lock"
LOCAL_TIMEZONE = "America/Chicago"
REQUEST_CEILING = len(MARKETS)


@dataclass(frozen=True)
class CaptureWindow:
    name: str
    target_minutes: int
    early_tolerance_minutes: int
    late_tolerance_minutes: int


WINDOWS = (
    CaptureWindow("t_minus_60", 60, 6, 6),
    CaptureWindow("t_minus_5", 5, 4, 4),
)


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def iso_utc(value: object) -> str:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        raise ValueError(f"Invalid timestamp: {value!r}")
    return stamp.isoformat().replace("+00:00", "Z")


def load_checkpoint(path: Path = CHECKPOINT) -> dict:
    if not path.exists():
        return {
            "version": 1,
            "completed": {},
            "blocked": {},
            "failures": [],
            "daily_credits": {},
            "last_api_balance": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, default in {
        "completed": {}, "blocked": {}, "failures": [],
        "daily_credits": {}, "last_api_balance": {},
    }.items():
        payload.setdefault(key, default)
    return payload


def save_checkpoint(payload: dict, path: Path = CHECKPOINT) -> None:
    payload["updated_at_utc"] = iso_utc(utc_now())
    atomic_write_json(path, payload)


def event_local_date(event: dict) -> str:
    tip = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
    return "" if pd.isna(tip) else tip.tz_convert(LOCAL_TIMEZONE).strftime("%Y-%m-%d")


def build_plan(events: object, local_date: str) -> pd.DataFrame:
    rows = []
    if not isinstance(events, list):
        return pd.DataFrame()
    for event in events:
        if not isinstance(event, dict) or not event.get("id"):
            continue
        if event_local_date(event) != local_date:
            continue
        tip = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(tip):
            continue
        for window in WINDOWS:
            target = tip - pd.Timedelta(minutes=window.target_minutes)
            rows.append({
                "event_id": str(event["id"]),
                "home_team": str(event.get("home_team", "")),
                "away_team": str(event.get("away_team", "")),
                "commence_time": iso_utc(tip),
                "game_date": local_date,
                "snapshot_kind": window.name,
                "target_ts_utc": iso_utc(target),
                "early_tolerance_minutes": window.early_tolerance_minutes,
                "late_tolerance_minutes": window.late_tolerance_minutes,
            })
    return pd.DataFrame(rows).sort_values(
        ["target_ts_utc", "event_id", "snapshot_kind"]
    ).reset_index(drop=True) if rows else pd.DataFrame()


def fetch_events(api_key: str, session: requests.Session | None = None) -> tuple[list, dict]:
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is not configured")
    client = session or requests.Session()
    response = client.get(EVENTS_URL, params={"apiKey": api_key, "dateFormat": "iso"}, timeout=30)
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": response.text[:1000]}
    if response.status_code != 200 or not isinstance(payload, list):
        raise RuntimeError(f"Odds API event lookup failed with HTTP {response.status_code}")
    return payload, response_balance(response)


def _merge_parsed(paths: list[Path]) -> None:
    frames = [pd.read_csv(path, low_memory=False) for path in paths if path.is_file()]
    if not frames:
        return
    existing = pd.read_csv(MASTER, low_memory=False) if MASTER.exists() else pd.DataFrame()
    merged = _merge_historical_lines(existing, pd.concat(frames, ignore_index=True))
    _write_historical_lines(merged, MASTER)


def recover_completed_to_master(checkpoint: dict) -> int:
    paths = [ROOT / str(item.get("parsed_path", "")) for item in checkpoint["completed"].values()]
    paths = [path for path in paths if path.is_file()]
    if paths:
        _merge_parsed(paths)
    return len(paths)


def capture_event(
    row: pd.Series,
    api_key: str,
    max_daily_credits: int,
    min_remaining_credits: int,
    checkpoint: dict,
    session: requests.Session | None = None,
) -> dict:
    client = session or requests.Session()
    event_id = str(row["event_id"])
    kind = str(row["snapshot_kind"])
    key = f"{event_id}:{kind}"
    day = str(row["game_date"])
    if key in checkpoint["completed"]:
        return {"status": "already_captured", "checkpoint_key": key, "credits_used": 0}
    if key in checkpoint["blocked"]:
        return {"status": "blocked", "checkpoint_key": key, "credits_used": 0}

    daily_spent = int(checkpoint["daily_credits"].get(day, 0))
    if daily_spent + REQUEST_CEILING > int(max_daily_credits):
        return {"status": "daily_credit_cap_reached", "checkpoint_key": key, "credits_used": 0}
    remaining = checkpoint.get("last_api_balance", {}).get("requests_remaining")
    if remaining is not None and int(remaining) - REQUEST_CEILING < int(min_remaining_credits):
        return {"status": "provider_reserve_reached", "checkpoint_key": key, "credits_used": 0}

    now = utc_now()
    requested = str(row["target_ts_utc"])
    response = client.get(
        EVENT_ODDS_URL.format(event_id=event_id),
        params={
            "apiKey": api_key,
            "bookmakers": "fanduel",
            "markets": ",".join(MARKETS),
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
        timeout=45,
    )
    try:
        event_payload = response.json()
    except ValueError:
        event_payload = {"error": response.text[:2000]}
    cost = response_cost(response, REQUEST_CEILING)
    balance = response_balance(response)
    checkpoint["daily_credits"][day] = daily_spent + cost
    checkpoint["last_api_balance"] = balance

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    raw_path = RAW_ROOT / day / f"{event_id}_{kind}_{stamp}.json"
    wrapped = {"timestamp": iso_utc(now), "data": event_payload}
    atomic_write_json(raw_path, wrapped)

    if response.status_code != 200:
        failure = {
            "checkpoint_key": key,
            "http_status": int(response.status_code),
            "at_utc": iso_utc(now),
            "credits_used": int(cost),
            "raw_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        }
        checkpoint["failures"].append(failure)
        checkpoint["failures"] = checkpoint["failures"][-500:]
        if response.status_code in {400, 401, 402, 403, 404, 422, 429}:
            checkpoint["blocked"][key] = failure
        save_checkpoint(checkpoint)
        return {"status": f"http_{response.status_code}", **failure}

    parsed = parse_event_odds(wrapped, requested, game_date_override=day)
    parsed = parsed.loc[parsed["bookmaker"].astype(str).str.lower().eq("fanduel")].copy()
    parsed["source"] = f"odds_api_fanduel_{kind}"
    parsed_path = PARSED_ROOT / day / f"{event_id}_{kind}_{stamp}.csv"
    atomic_write_csv(parsed_path, parsed)
    paired = int(parsed[["over_price", "under_price"]].notna().all(axis=1).sum()) if len(parsed) else 0
    record = {
        "status": "completed_with_data" if len(parsed) else "completed_empty",
        "checkpoint_key": key,
        "event_id": event_id,
        "snapshot_kind": kind,
        "captured_at_utc": iso_utc(now),
        "requested_snapshot_ts": requested,
        "commence_time": str(row["commence_time"]),
        "rows_parsed": int(len(parsed)),
        "paired_price_rows": paired,
        "credits_used": int(cost),
        "raw_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "parsed_path": str(parsed_path.relative_to(ROOT)).replace("\\", "/"),
        "provider_balance": balance,
    }
    checkpoint["completed"][key] = record
    save_checkpoint(checkpoint)
    return record


def capture_due(
    plan: pd.DataFrame,
    now: pd.Timestamp,
    api_key: str,
    max_daily_credits: int,
    min_remaining_credits: int,
    session: requests.Session | None = None,
) -> list[dict]:
    checkpoint = load_checkpoint()
    target = pd.to_datetime(plan["target_ts_utc"], utc=True)
    early = pd.to_timedelta(plan["early_tolerance_minutes"], unit="m")
    late = pd.to_timedelta(plan["late_tolerance_minutes"], unit="m")
    due = plan[(now >= target - early) & (now <= target + late)].copy()
    results = []
    new_paths = []
    for _, row in due.iterrows():
        checkpoint_key = f"{row['event_id']}:{row['snapshot_kind']}"
        if checkpoint_key in checkpoint["completed"] or checkpoint_key in checkpoint["blocked"]:
            continue
        result = capture_event(
            row, api_key, max_daily_credits, min_remaining_credits, checkpoint, session=session
        )
        results.append(result)
        parsed_path = result.get("parsed_path")
        if parsed_path:
            new_paths.append(ROOT / str(parsed_path))
    if new_paths:
        _merge_parsed(new_paths)
    return results


def acquire_lock() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started_at_utc": iso_utc(utc_now())}, handle)
    except FileExistsError as exc:
        age_hours = (time.time() - LOCK.stat().st_mtime) / 3600.0
        if age_hours <= 24:
            raise RuntimeError(f"FanDuel closing watcher lock already exists: {LOCK}") from exc
        LOCK.unlink()
        with LOCK.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started_at_utc": iso_utc(utc_now())}, handle)


def watch(
    local_date: str,
    poll_seconds: int,
    max_daily_credits: int,
    min_remaining_credits: int,
) -> dict:
    api_key = os.environ.get("ODDS_API_KEY", "")
    events, balance = fetch_events(api_key)
    plan = build_plan(events, local_date)
    checkpoint = load_checkpoint()
    checkpoint["last_api_balance"] = balance
    save_checkpoint(checkpoint)
    event_path = RAW_ROOT / local_date / "events.json"
    atomic_write_json(event_path, events)
    if plan.empty:
        return {"status": "no_nba_games", "local_date": local_date}

    acquire_lock()
    attempts = []
    try:
        recover_completed_to_master(checkpoint)
        targets = pd.to_datetime(plan["target_ts_utc"], utc=True)
        final_deadline = max(
            target + pd.Timedelta(minutes=int(late))
            for target, late in zip(targets, plan["late_tolerance_minutes"])
        )
        while utc_now() <= final_deadline:
            now = utc_now()
            results = capture_due(
                plan, now, api_key, max_daily_credits, min_remaining_credits
            )
            attempts.extend(results)
            if any(
                item.get("status") in {"daily_credit_cap_reached", "provider_reserve_reached"}
                for item in results
            ):
                break
            if now >= final_deadline:
                break
            time.sleep(max(10, min(int(poll_seconds), 60)))
    finally:
        if LOCK.exists():
            LOCK.unlink()
    return {
        "status": "complete",
        "local_date": local_date,
        "planned_windows": int(len(plan)),
        "attempts": attempts,
        "credits_used": int(sum(int(item.get("credits_used", 0)) for item in attempts)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-daily-credits", type=int, default=200)
    parser.add_argument("--min-remaining-credits", type=int, default=100)
    args = parser.parse_args()
    if args.max_daily_credits < REQUEST_CEILING:
        raise ValueError(f"--max-daily-credits must be at least {REQUEST_CEILING}")
    if args.plan:
        api_key = os.environ.get("ODDS_API_KEY", "")
        events, balance = fetch_events(api_key)
        plan = build_plan(events, args.date)
        print(plan.to_csv(index=False) if not plan.empty else "No NBA games scheduled.")
        print(json.dumps({"provider_balance": balance}, indent=2))
        return 0
    if args.watch:
        result = watch(
            args.date, args.poll_seconds, args.max_daily_credits, args.min_remaining_credits
        )
    else:
        api_key = os.environ.get("ODDS_API_KEY", "")
        events, balance = fetch_events(api_key)
        plan = build_plan(events, args.date)
        checkpoint = load_checkpoint()
        checkpoint["last_api_balance"] = balance
        save_checkpoint(checkpoint)
        result = {
            "status": "complete",
            "results": capture_due(
                plan, utc_now(), api_key, args.max_daily_credits, args.min_remaining_credits
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())
