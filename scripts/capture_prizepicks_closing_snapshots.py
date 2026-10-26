"""Capture PrizePicks near T-60/T-5 and pair T-60 with live sportsbook odds.

Run once from a scheduler near a target window, or use ``--watch`` from the
morning pipeline. The watcher consults ESPN's NBA scoreboard only to determine
when a capture is due. PrizePicks is fetched directly at both windows. When
explicitly enabled, the live Odds API is called only for T-60 games, under a
durable lifetime credit cap; no Odds API credits are spent at T-5.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from prizepicks_scraper import (
        _merge_historical_lines,
        fetch_prizepicks_payload,
        parse_prizepicks_payload,
    )
    from scripts.execute_clv_snapshot_manifest import (
        ROOT,
        atomic_write_csv,
        atomic_write_json,
        execution_lock,
        parse_event_odds,
        response_balance,
        response_cost,
    )
except ModuleNotFoundError:
    from prizepicks_scraper import (
        _merge_historical_lines,
        fetch_prizepicks_payload,
        parse_prizepicks_payload,
    )
    from execute_clv_snapshot_manifest import (
        ROOT,
        atomic_write_csv,
        atomic_write_json,
        execution_lock,
        parse_event_odds,
        response_balance,
        response_cost,
    )


LOGGER = logging.getLogger(__name__)
RAW_DIR = ROOT / "data" / "prizepicks_clv_raw"
PARSED_DIR = ROOT / "data" / "prizepicks_clv_snapshots"
CHECKPOINT_PATH = ROOT / "data" / "prizepicks_clv_capture_checkpoint.json"
MASTER_PATH = ROOT / "data" / "historical_lines.csv"
LOCK_PATH = ROOT / "data" / "prizepicks_clv_watcher.lock"
PAIR_CHECKPOINT_PATH = ROOT / "data" / "prizepicks_t60_pair_checkpoint.json"
PAIR_RAW_DIR = ROOT / "data" / "odds_api_prizepicks_t60_raw"
PAIR_PARSED_DIR = ROOT / "data" / "odds_api_prizepicks_t60_parsed"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ODDS_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
ODDS_EVENT_URL = ODDS_EVENTS_URL + "/{event_id}/odds"
ODDS_REGIONS = "us"
ODDS_MARKETS = (
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
ODDS_EVENT_CREDIT_CEILING = len(ODDS_MARKETS)
DEFAULT_ODDS_CREDIT_BUDGET = 13_436

TEAM_ABBR = {
    "atlanta hawks": "ATL", "boston celtics": "BOS", "brooklyn nets": "BKN",
    "charlotte hornets": "CHA", "chicago bulls": "CHI", "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL", "denver nuggets": "DEN", "detroit pistons": "DET",
    "golden state warriors": "GSW", "houston rockets": "HOU", "indiana pacers": "IND",
    "los angeles clippers": "LAC", "la clippers": "LAC", "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM", "miami heat": "MIA", "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN", "new orleans pelicans": "NOP", "new york knicks": "NYK",
    "oklahoma city thunder": "OKC", "orlando magic": "ORL", "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX", "portland trail blazers": "POR", "sacramento kings": "SAC",
    "san antonio spurs": "SAS", "toronto raptors": "TOR", "utah jazz": "UTA",
    "washington wizards": "WAS",
}


@dataclass(frozen=True)
class CaptureWindow:
    name: str
    target_minutes: float
    early_tolerance_minutes: float
    late_tolerance_minutes: float


CAPTURE_WINDOWS = (
    CaptureWindow("t_minus_60", 60.0, 6.0, 6.0),
    CaptureWindow("t_minus_5", 5.0, 4.0, 4.0),
)


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _iso_utc(value: object) -> str:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    return stamp.isoformat().replace("+00:00", "Z")


def load_checkpoint(path: Path | None = None) -> dict:
    path = path or CHECKPOINT_PATH
    if not path.exists():
        return {"version": 1, "completed": {}, "attempts": []}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("version", 1)
    payload.setdefault("completed", {})
    payload.setdefault("attempts", [])
    return payload


def load_pair_checkpoint(path: Path | None = None) -> dict:
    path = path or PAIR_CHECKPOINT_PATH
    if not path.exists():
        return {
            "version": 1,
            "credit_budget": DEFAULT_ODDS_CREDIT_BUDGET,
            "credits_used": 0,
            "completed": {},
            "blocked": {},
            "failures": [],
            "last_api_balance": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("version", 1)
    payload.setdefault("credit_budget", DEFAULT_ODDS_CREDIT_BUDGET)
    payload.setdefault("credits_used", 0)
    payload.setdefault("completed", {})
    payload.setdefault("blocked", {})
    payload.setdefault("failures", [])
    payload.setdefault("last_api_balance", {})
    return payload


def save_pair_checkpoint(checkpoint: dict, path: Path | None = None) -> None:
    checkpoint["updated_at_utc"] = _iso_utc(_utc_now())
    atomic_write_json(path or PAIR_CHECKPOINT_PATH, checkpoint)


def _normalize_team_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _team_is_mentioned(team: object, description: object) -> bool:
    team_text = _normalize_team_text(team)
    description_text = _normalize_team_text(description)
    if not team_text or not description_text:
        return False
    description_tokens = set(description_text.split())
    abbreviation = TEAM_ABBR.get(team_text, "").casefold()
    nickname = team_text.split()[-1]
    return (
        team_text in description_text
        or (abbreviation and abbreviation in description_tokens)
        or nickname in description_tokens
    )


def match_odds_event(events: object, event_rows: pd.DataFrame) -> dict | None:
    """Match one PrizePicks game to one current Odds API event without guessing."""
    if not isinstance(events, list) or event_rows.empty:
        return None
    tip = pd.to_datetime(event_rows.get("commence_time").iloc[0], utc=True, errors="coerce")
    if pd.isna(tip):
        return None
    candidates = []
    for event in events:
        if not isinstance(event, dict) or not event.get("id"):
            continue
        event_tip = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(event_tip) or abs((event_tip - tip).total_seconds()) > 15 * 60:
            continue
        candidates.append(event)
    if len(candidates) == 1:
        return candidates[0]
    descriptions = [
        value for value in event_rows.get("game_description", pd.Series(dtype=object)).dropna().unique()
        if str(value).strip()
    ]
    matches = []
    for event in candidates:
        if any(
            _team_is_mentioned(event.get("home_team"), description)
            and _team_is_mentioned(event.get("away_team"), description)
            for description in descriptions
        ):
            matches.append(event)
    return matches[0] if len(matches) == 1 else None


def _response_json(response: requests.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return {"error": "non_json_response", "body": response.text[:2000]}


def capture_sportsbook_t60(
    event_rows: pd.DataFrame,
    captured_at_utc: object,
    api_key: str,
    max_credits: int = DEFAULT_ODDS_CREDIT_BUDGET,
    session: requests.Session | None = None,
) -> dict:
    """Persist the sportsbook panel paired to one direct PrizePicks T-60 game."""
    event_id = str(event_rows["event_id"].iloc[0])
    checkpoint_key = f"{event_id}:t_minus_60"
    captured_at = pd.to_datetime(captured_at_utc, utc=True, errors="coerce")
    if pd.isna(captured_at):
        raise ValueError("captured_at_utc must be valid")
    if max_credits <= 0:
        return {"status": "disabled", "checkpoint_key": checkpoint_key, "credits_used": 0, "requests_sent": 0}
    if not api_key:
        return {"status": "missing_api_key", "checkpoint_key": checkpoint_key, "credits_used": 0, "requests_sent": 0}

    client = session or requests.Session()
    with execution_lock(PAIR_CHECKPOINT_PATH):
        checkpoint = load_pair_checkpoint()
        stored_budget = int(checkpoint.get("credit_budget", max_credits))
        if stored_budget != int(max_credits) and checkpoint.get("completed"):
            max_credits = min(int(max_credits), stored_budget)
        checkpoint["credit_budget"] = int(max_credits)
        if checkpoint_key in checkpoint["completed"]:
            record = checkpoint["completed"][checkpoint_key]
            return {
                "status": "already_captured",
                "checkpoint_key": checkpoint_key,
                "credits_used": 0,
                "requests_sent": 0,
                "parsed_rows": int(record.get("rows_parsed", 0)),
            }
        if checkpoint_key in checkpoint["blocked"]:
            return {
                "status": "blocked_after_terminal_http_error",
                "checkpoint_key": checkpoint_key,
                "credits_used": 0,
                "requests_sent": 0,
            }

        event_response = client.get(
            ODDS_EVENTS_URL,
            params={"apiKey": api_key, "dateFormat": "iso"},
            timeout=30,
        )
        event_payload = _response_json(event_response)
        checkpoint["last_api_balance"] = response_balance(event_response)
        if event_response.status_code != 200:
            failure = {
                "checkpoint_key": checkpoint_key,
                "reason": f"event_lookup_http_{event_response.status_code}",
                "at_utc": _iso_utc(captured_at),
            }
            checkpoint["failures"].append(failure)
            checkpoint["failures"] = checkpoint["failures"][-500:]
            save_pair_checkpoint(checkpoint)
            return {"status": failure["reason"], "checkpoint_key": checkpoint_key, "credits_used": 0, "requests_sent": 1}

        matched = match_odds_event(event_payload, event_rows)
        lookup_path = (
            PAIR_RAW_DIR / captured_at.strftime("%Y-%m-%d") / f"{event_id}_event_lookup.json"
        )
        atomic_write_json(lookup_path, event_payload)
        if matched is None:
            failure = {
                "checkpoint_key": checkpoint_key,
                "reason": "event_match_ambiguous_or_missing",
                "at_utc": _iso_utc(captured_at),
                "lookup_path": str(lookup_path.relative_to(ROOT)).replace("\\", "/"),
            }
            checkpoint["failures"].append(failure)
            checkpoint["failures"] = checkpoint["failures"][-500:]
            save_pair_checkpoint(checkpoint)
            return {"status": failure["reason"], "checkpoint_key": checkpoint_key, "credits_used": 0, "requests_sent": 1}

        raw_path = PAIR_RAW_DIR / captured_at.strftime("%Y-%m-%d") / f"{event_id}_t_minus_60.json"
        requested = pd.to_datetime(event_rows["commence_time"].iloc[0], utc=True) - pd.Timedelta(minutes=60)
        cost = 0
        if raw_path.exists():
            wrapped_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            status_code = 200
        else:
            spent = int(checkpoint.get("credits_used", 0))
            if spent + ODDS_EVENT_CREDIT_CEILING > int(max_credits):
                return {
                    "status": "credit_cap_reached",
                    "checkpoint_key": checkpoint_key,
                    "credits_used": 0,
                    "requests_sent": 1,
                    "credit_budget_remaining": max(0, int(max_credits) - spent),
                }
            odds_response = client.get(
                ODDS_EVENT_URL.format(event_id=matched["id"]),
                params={
                    "apiKey": api_key,
                    "regions": ODDS_REGIONS,
                    "markets": ",".join(ODDS_MARKETS),
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
                timeout=45,
            )
            event_odds = _response_json(odds_response)
            status_code = odds_response.status_code
            cost = response_cost(odds_response, ODDS_EVENT_CREDIT_CEILING)
            checkpoint["credits_used"] = spent + cost
            checkpoint["last_api_balance"] = response_balance(odds_response)
            save_pair_checkpoint(checkpoint)
            wrapped_payload = {"timestamp": _iso_utc(captured_at), "data": event_odds}
            saved_raw_path = raw_path if status_code == 200 else raw_path.with_name(
                f"{raw_path.stem}.http_{status_code}{raw_path.suffix}"
            )
            atomic_write_json(saved_raw_path, wrapped_payload)
            if status_code != 200:
                failure = {
                    "checkpoint_key": checkpoint_key,
                    "reason": f"event_odds_http_{status_code}",
                    "at_utc": _iso_utc(captured_at),
                    "credits_used": cost,
                }
                checkpoint["failures"].append(failure)
                checkpoint["failures"] = checkpoint["failures"][-500:]
                if 400 <= status_code < 500:
                    checkpoint["blocked"][checkpoint_key] = failure
                save_pair_checkpoint(checkpoint)
                return {
                    "status": failure["reason"],
                    "checkpoint_key": checkpoint_key,
                    "credits_used": cost,
                    "requests_sent": 2,
                }

        parsed = parse_event_odds(wrapped_payload, _iso_utc(requested))
        parsed_path = (
            PAIR_PARSED_DIR
            / captured_at.strftime("%Y-%m-%d")
            / f"{event_id}_t_minus_60.csv"
        )
        atomic_write_csv(parsed_path, parsed)
        if not parsed.empty:
            _write_master_snapshot(parsed)
        paired_prices = int(parsed[["over_price", "under_price"]].notna().all(axis=1).sum()) if len(parsed) else 0
        checkpoint["completed"][checkpoint_key] = {
            "status": "completed_with_data" if len(parsed) else "completed_empty",
            "completed_at_utc": _iso_utc(captured_at),
            "prizepicks_event_id": event_id,
            "odds_api_event_id": str(matched["id"]),
            "rows_parsed": int(len(parsed)),
            "paired_price_rows": paired_prices,
            "credits_used": int(cost),
            "raw_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
            "parsed_path": str(parsed_path.relative_to(ROOT)).replace("\\", "/"),
        }
        save_pair_checkpoint(checkpoint)
        return {
            "status": checkpoint["completed"][checkpoint_key]["status"],
            "checkpoint_key": checkpoint_key,
            "credits_used": int(cost),
            "requests_sent": 1 if raw_path.exists() and cost == 0 else 2,
            "parsed_rows": int(len(parsed)),
            "paired_price_rows": paired_prices,
            "credit_budget_remaining": max(0, int(max_credits) - int(checkpoint["credits_used"])),
        }


def classify_due_rows(lines: pd.DataFrame, captured_at_utc: object) -> dict[str, pd.DataFrame]:
    """Classify parsed projection rows into the configured pre-tip windows."""
    if lines is None or lines.empty:
        return {}
    working = lines.copy()
    working["commence_utc"] = pd.to_datetime(working.get("commence_time"), utc=True, errors="coerce")
    captured = pd.to_datetime(captured_at_utc, utc=True, errors="coerce")
    if pd.isna(captured):
        raise ValueError("captured_at_utc must be a valid timestamp")
    working["minutes_until_tip"] = (working["commence_utc"] - captured).dt.total_seconds() / 60.0
    output: dict[str, pd.DataFrame] = {}
    for window in CAPTURE_WINDOWS:
        lower = window.target_minutes - window.late_tolerance_minutes
        upper = window.target_minutes + window.early_tolerance_minutes
        due = working[working["minutes_until_tip"].between(lower, upper, inclusive="both")].copy()
        due = due[due["event_id"].fillna("").astype(str).str.len().gt(0)]
        if not due.empty:
            output[window.name] = due
    return output


def _write_master_snapshot(captured: pd.DataFrame) -> None:
    existing = pd.read_csv(MASTER_PATH, low_memory=False) if MASTER_PATH.exists() else pd.DataFrame()
    combined = _merge_historical_lines(existing, captured)
    atomic_write_csv(MASTER_PATH, combined)


def capture_once(
    captured_at_utc: object | None = None,
    payload: dict | None = None,
    write: bool = True,
    pair_sportsbook_t60: bool = False,
    odds_credit_budget: int = DEFAULT_ODDS_CREDIT_BUDGET,
    odds_session: requests.Session | None = None,
) -> dict:
    """Fetch once and persist every game whose board is in a target window."""
    captured_value = _utc_now() if captured_at_utc is None else captured_at_utc
    captured_at = pd.to_datetime(captured_value, utc=True, errors="coerce")
    if pd.isna(captured_at):
        raise ValueError("captured_at_utc must be valid")
    fetched_payload = payload if payload is not None else fetch_prizepicks_payload()
    if fetched_payload is None:
        return {
            "captured_at_utc": _iso_utc(captured_at),
            "status": "fetch_failed",
            "captured_events": 0,
            "captured_rows": 0,
            "odds_api_requests": 0,
        }
    parsed = parse_prizepicks_payload(fetched_payload, require_roster=True)
    due = classify_due_rows(parsed, captured_at)
    stamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
    raw_path = RAW_DIR / captured_at.strftime("%Y-%m-%d") / f"prizepicks_{stamp}.json"
    checkpoint = load_checkpoint()
    captured_frames = []
    completed_now = []
    sportsbook_results = []
    if write:
        atomic_write_json(raw_path, fetched_payload)
    for snapshot_kind, snapshot_rows in due.items():
        target_minutes = next(
            window.target_minutes for window in CAPTURE_WINDOWS if window.name == snapshot_kind
        )
        for event_id, event_rows in snapshot_rows.groupby("event_id", sort=True):
            checkpoint_key = f"{event_id}:{snapshot_kind}"
            commence = pd.to_datetime(event_rows["commence_utc"].iloc[0], utc=True)
            requested = commence - pd.Timedelta(minutes=target_minutes)
            if checkpoint_key not in checkpoint["completed"]:
                saved = event_rows.drop(columns=["commence_utc", "minutes_until_tip"], errors="ignore").copy()
                saved["source"] = f"prizepicks_{snapshot_kind}"
                saved["bookmaker"] = "prizepicks"
                saved["snapshot_ts"] = _iso_utc(captured_at)
                saved["requested_snapshot_ts"] = _iso_utc(requested)
                saved["num_books"] = 1
                parsed_path = (
                    PARSED_DIR
                    / str(saved["game_date"].iloc[0])
                    / f"{event_id}_{snapshot_kind}_{stamp}.csv"
                )
                if write:
                    atomic_write_csv(parsed_path, saved)
                captured_frames.append(saved)
                completed_now.append(
                    {
                        "checkpoint_key": checkpoint_key,
                        "event_id": str(event_id),
                        "snapshot_kind": snapshot_kind,
                        "captured_at_utc": _iso_utc(captured_at),
                        "requested_snapshot_ts": _iso_utc(requested),
                        "commence_time": _iso_utc(commence),
                        "rows": int(len(saved)),
                        "raw_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
                        "parsed_path": str(parsed_path.relative_to(ROOT)).replace("\\", "/"),
                    }
                )
            if write and pair_sportsbook_t60 and snapshot_kind == "t_minus_60":
                try:
                    sportsbook_results.append(
                        capture_sportsbook_t60(
                            event_rows,
                            captured_at,
                            api_key=os.environ.get("ODDS_API_KEY", ""),
                            max_credits=odds_credit_budget,
                            session=odds_session,
                        )
                    )
                except Exception as exc:
                    LOGGER.exception("Sportsbook T-60 pairing failed for PrizePicks event %s", event_id)
                    sportsbook_results.append(
                        {
                            "status": "pairing_exception",
                            "checkpoint_key": checkpoint_key,
                            "credits_used": 0,
                            "error": str(exc),
                        }
                    )
    captured_rows = int(sum(len(frame) for frame in captured_frames))
    if write and captured_frames:
        _write_master_snapshot(pd.concat(captured_frames, ignore_index=True))
        for item in completed_now:
            checkpoint["completed"][item["checkpoint_key"]] = item
    attempt = {
        "captured_at_utc": _iso_utc(captured_at),
        "status": "captured" if captured_frames else "no_due_games",
        "parsed_rows": int(len(parsed)),
        "captured_events": int(len(completed_now)),
        "captured_rows": captured_rows,
        "raw_path": str(raw_path.relative_to(ROOT)).replace("\\", "/") if write else None,
        "sportsbook_t60_results": sportsbook_results,
        "odds_api_requests": int(sum(int(result.get("requests_sent", 0)) for result in sportsbook_results)),
        "odds_api_credits_used": int(sum(int(result.get("credits_used", 0)) for result in sportsbook_results)),
    }
    if write:
        checkpoint["attempts"].append(attempt)
        checkpoint["attempts"] = checkpoint["attempts"][-500:]
        checkpoint["updated_at_utc"] = _iso_utc(captured_at)
        atomic_write_json(CHECKPOINT_PATH, checkpoint)
    return attempt


def fetch_tip_times(local_date: str) -> list[pd.Timestamp]:
    """Fetch scheduled NBA tip times for one local calendar date from ESPN."""
    day = pd.Timestamp(local_date)
    response = requests.get(
        ESPN_SCOREBOARD,
        params={"dates": day.strftime("%Y%m%d"), "limit": 100},
        timeout=20,
    )
    response.raise_for_status()
    tips = []
    for event in response.json().get("events", []):
        tip = pd.to_datetime(event.get("date"), utc=True, errors="coerce")
        if pd.notna(tip):
            tips.append(tip)
    return sorted(set(tips))


def build_capture_plan(local_date: str) -> pd.DataFrame:
    rows = []
    for tip in fetch_tip_times(local_date):
        for window in CAPTURE_WINDOWS:
            target = tip - pd.Timedelta(minutes=window.target_minutes)
            rows.append(
                {
                    "local_date": local_date,
                    "tip_ts_utc": _iso_utc(tip),
                    "snapshot_kind": window.name,
                    "target_ts_utc": _iso_utc(target),
                    "early_tolerance_minutes": window.early_tolerance_minutes,
                    "late_tolerance_minutes": window.late_tolerance_minutes,
                }
            )
    return pd.DataFrame(rows)


def _acquire_lock() -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK_PATH.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at_utc": _iso_utc(_utc_now())}))
    except FileExistsError as exc:
        age_hours = (time.time() - LOCK_PATH.stat().st_mtime) / 3600.0
        if age_hours <= 24:
            raise RuntimeError(f"PrizePicks closing watcher already has an active lock: {LOCK_PATH}") from exc
        LOCK_PATH.unlink()
        with LOCK_PATH.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at_utc": _iso_utc(_utc_now())}))


def watch(
    local_date: str,
    poll_seconds: int = 60,
    pair_sportsbook_t60: bool = False,
    odds_credit_budget: int = DEFAULT_ODDS_CREDIT_BUDGET,
) -> dict:
    """Wait for each planned window and attempt direct PrizePicks capture."""
    plan = build_capture_plan(local_date)
    if plan.empty:
        return {"status": "no_nba_games", "local_date": local_date, "odds_api_requests": 0}
    _acquire_lock()
    attempts = []
    try:
        plan["target"] = pd.to_datetime(plan["target_ts_utc"], utc=True)
        final_deadline = plan["target"].max() + pd.Timedelta(minutes=7)
        last_attempt: dict[str, pd.Timestamp] = {}
        while _utc_now() <= final_deadline:
            now = _utc_now()
            due = plan[
                (now >= plan["target"] - pd.Timedelta(minutes=4))
                & (now <= plan["target"] + pd.Timedelta(minutes=7))
            ]
            due_keys = [f"{row.tip_ts_utc}:{row.snapshot_kind}" for row in due.itertuples()]
            should_attempt = any(
                key not in last_attempt or (now - last_attempt[key]).total_seconds() >= poll_seconds
                for key in due_keys
            )
            if should_attempt:
                result = capture_once(
                    captured_at_utc=now,
                    write=True,
                    pair_sportsbook_t60=pair_sportsbook_t60,
                    odds_credit_budget=odds_credit_budget,
                )
                attempts.append(result)
                for key in due_keys:
                    last_attempt[key] = now
            if now > final_deadline:
                break
            time.sleep(max(10, min(int(poll_seconds), 60)))
    finally:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    return {
        "status": "complete",
        "local_date": local_date,
        "planned_windows": int(len(plan)),
        "attempts": attempts,
        "odds_api_requests": int(sum(item.get("odds_api_requests", 0) for item in attempts)),
        "odds_api_credits_used": int(sum(item.get("odds_api_credits_used", 0) for item in attempts)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Wait for today's T-60 and T-5 windows")
    parser.add_argument("--plan", action="store_true", help="Print the schedule without fetching PrizePicks")
    parser.add_argument("--date", default=date.today().isoformat(), help="Local slate date (YYYY-MM-DD)")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--pair-sportsbook-t60",
        action="store_true",
        help="Spend live Odds API credits only when a direct PrizePicks T-60 game is captured",
    )
    parser.add_argument(
        "--odds-credit-budget",
        type=int,
        default=DEFAULT_ODDS_CREDIT_BUDGET,
        help="Lifetime credit cap for paired live sportsbook T-60 captures",
    )
    args = parser.parse_args()
    if args.plan:
        plan = build_capture_plan(args.date)
        print(plan.to_csv(index=False) if not plan.empty else "No NBA games scheduled.")
        return 0
    result = (
        watch(
            args.date,
            poll_seconds=args.poll_seconds,
            pair_sportsbook_t60=args.pair_sportsbook_t60,
            odds_credit_budget=args.odds_credit_budget,
        )
        if args.watch
        else capture_once(
            pair_sportsbook_t60=args.pair_sportsbook_t60,
            odds_credit_budget=args.odds_credit_budget,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"fetch_failed"} else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())
