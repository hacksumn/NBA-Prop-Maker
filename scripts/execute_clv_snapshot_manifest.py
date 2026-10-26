"""Execute the paid Odds API CLV manifest safely and resumably.

Every paid response is saved as raw JSON and parsed CSV before its manifest key
is marked complete. The raw store is the recovery boundary; the master line CSV
is materialized atomically in batches. A positive lifetime credit cap is
mandatory, and completed keys are never requested twice.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba"
LINE_COLUMNS = [
    "game_date", "player", "player_norm", "prop", "line", "num_books",
    "source", "snapshot_ts", "projection_type", "is_promo", "bookmaker",
    "event_id", "commence_time", "requested_snapshot_ts", "over_price",
    "under_price",
]
PHYSICAL_KEY = ["game_date", "player_norm", "prop", "line", "source", "snapshot_ts"]
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z ]", "", text).strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_with_retry(temp_path: Path, path: Path, attempts: int = 10) -> None:
    """Retry transient Windows file-sharing failures during atomic replace."""
    for attempt in range(attempts):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.5 * (attempt + 1))


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


@contextmanager
def execution_lock(checkpoint_path: Path):
    """Prevent overlapping paid executors from issuing duplicate requests."""
    lock_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            existing_pid = int(existing.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_pid = 0
        if _pid_is_running(existing_pid):
            raise RuntimeError(
                f"Paid Odds API executor already running with PID {existing_pid}: {lock_path}"
            )
        lock_path.unlink(missing_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "started_at_utc": utc_now()}, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        yield lock_path
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            current = {}
        if int(current.get("pid", 0) or 0) == os.getpid():
            lock_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    _replace_with_retry(temp_path, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        frame.to_csv(handle, index=False)
        temp_path = Path(handle.name)
    _replace_with_retry(temp_path, path)


def empty_lines() -> pd.DataFrame:
    return pd.DataFrame(columns=LINE_COLUMNS)


def normalize_lines(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in LINE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out = out[LINE_COLUMNS]
    out["player"] = out["player"].fillna("").astype(str).str.strip()
    out["player_norm"] = out["player_norm"].fillna("").astype(str).str.strip()
    missing_norm = out["player_norm"].eq("")
    out.loc[missing_norm, "player_norm"] = out.loc[missing_norm, "player"].map(normalize_name)
    for column in ["line", "num_books", "over_price", "under_price"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["source"] = out["source"].fillna("unknown").astype(str)
    out["snapshot_ts"] = out["snapshot_ts"].fillna("").astype(str)
    out["is_promo"] = out["is_promo"].fillna(False).astype(bool)
    return out


def parse_event_odds(
    payload: object,
    requested_snapshot_ts: str,
    game_date_override: Optional[str] = None,
) -> pd.DataFrame:
    """Parse one event response into paired book/player/prop/line observations."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return empty_lines()
    event = payload["data"]
    event_id = str(event.get("id", "") or "")
    commence_time = str(event.get("commence_time", "") or "")
    game_date = str(game_date_override or "")
    if not game_date and commence_time:
        commence = pd.to_datetime(commence_time, errors="coerce", utc=True)
        if pd.notna(commence):
            game_date = commence.tz_convert("America/Chicago").strftime("%Y-%m-%d")
    snapshot_ts = str(payload.get("timestamp") or requested_snapshot_ts)
    rows: Dict[Tuple[str, str, str, float], Dict[str, object]] = {}
    for bookmaker in event.get("bookmakers", []) or []:
        if not isinstance(bookmaker, dict):
            continue
        book = str(bookmaker.get("key", "") or "").strip().lower()
        if not book:
            continue
        for market in bookmaker.get("markets", []) or []:
            prop = str(market.get("key", "") or "").strip()
            if not prop:
                continue
            for outcome in market.get("outcomes", []) or []:
                if not isinstance(outcome, dict):
                    continue
                player = str(outcome.get("description", "") or "").strip()
                point = pd.to_numeric(outcome.get("point"), errors="coerce")
                price = pd.to_numeric(outcome.get("price"), errors="coerce")
                if not player or pd.isna(point) or not game_date:
                    continue
                key = (book, player, prop, float(point))
                row = rows.setdefault(
                    key,
                    {
                        "game_date": game_date,
                        "player": player,
                        "player_norm": normalize_name(player),
                        "prop": prop,
                        "line": float(point),
                        "num_books": 1,
                        "source": f"odds_api_{book}",
                        "snapshot_ts": snapshot_ts,
                        "projection_type": "standard",
                        "is_promo": False,
                        "bookmaker": book,
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "requested_snapshot_ts": requested_snapshot_ts,
                        "over_price": pd.NA,
                        "under_price": pd.NA,
                    },
                )
                side = str(outcome.get("name", "") or "").strip().lower()
                if side == "over" and not pd.isna(price):
                    row["over_price"] = float(price)
                elif side == "under" and not pd.isna(price):
                    row["under_price"] = float(price)
    return normalize_lines(pd.DataFrame(rows.values())) if rows else empty_lines()


def response_cost(response: requests.Response, fallback: int) -> int:
    raw = response.headers.get("x-requests-last")
    try:
        return max(0, int(raw)) if raw is not None else fallback
    except (TypeError, ValueError):
        return fallback


def response_balance(response: requests.Response) -> Dict[str, Optional[int]]:
    result: Dict[str, Optional[int]] = {"requests_used": None, "requests_remaining": None}
    for header, key in [("x-requests-used", "requests_used"), ("x-requests-remaining", "requests_remaining")]:
        try:
            result[key] = int(response.headers[header]) if header in response.headers else None
        except (TypeError, ValueError):
            pass
    return result


def load_checkpoint(path: Path, manifest_hash: str) -> Dict[str, object]:
    if not path.exists():
        return {
            "version": 1,
            "manifest_sha256": manifest_hash,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "credits_used": 0,
            "completed": {},
            "event_groups": {},
            "failures": [],
            "last_api_balance": {},
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Checkpoint manifest hash does not match the current manifest")
    return checkpoint


def save_checkpoint(path: Path, checkpoint: Dict[str, object]) -> None:
    checkpoint["updated_at_utc"] = utc_now()
    atomic_write_json(path, checkpoint)


def safe_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def event_list_path(raw_root: Path, group: str) -> Path:
    return raw_root / "event_lists" / f"{safe_token(group)}.json"


def row_raw_path(raw_root: Path, row: pd.Series) -> Path:
    return raw_root / "event_odds" / str(row["game_date"]) / f"{safe_token(row['manifest_key'])}.json"


def row_parsed_path(parsed_root: Path, row: pd.Series) -> Path:
    return parsed_root / str(row["game_date"]) / f"{safe_token(row['manifest_key'])}.csv"


def payload_json(response: requests.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return {"non_json_body": response.text[:2000]}


def canonical_team(value: object) -> str:
    text = str(value or "").strip()
    return TEAM_ABBR.get(text.casefold(), text.upper())


def match_event(events_payload: object, home_team: str, away_team: str) -> Optional[Dict[str, object]]:
    if not isinstance(events_payload, dict) or not isinstance(events_payload.get("data"), list):
        return None
    matches = [
        event for event in events_payload["data"]
        if isinstance(event, dict)
        and canonical_team(event.get("home_team")) == home_team
        and canonical_team(event.get("away_team")) == away_team
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Odds API events match {away_team} at {home_team}")
    return matches[0] if matches else None


def merge_master(master_path: Path, parsed_paths: Iterable[Path]) -> Dict[str, int]:
    paths = [path for path in parsed_paths if path.exists() and path.stat().st_size > 0]
    if not paths:
        return {"rows_before": 0, "rows_after": 0, "rows_added": 0}
    before = pd.read_csv(master_path, low_memory=False) if master_path.exists() else empty_lines()
    before = normalize_lines(before)
    additions = [normalize_lines(pd.read_csv(path, low_memory=False)) for path in paths]
    additions = [frame for frame in additions if len(frame)]
    if not additions:
        return {"rows_before": len(before), "rows_after": len(before), "rows_added": 0}
    merged = pd.concat([before, *additions], ignore_index=True)
    merged = merged.drop_duplicates(PHYSICAL_KEY, keep="last").reset_index(drop=True)
    atomic_write_csv(master_path, merged)
    return {
        "rows_before": int(len(before)),
        "rows_after": int(len(merged)),
        "rows_added": int(len(merged) - len(before)),
    }


def write_ledger(path: Path, manifest: pd.DataFrame, checkpoint: Dict[str, object]) -> None:
    completed = checkpoint.get("completed", {})
    rows = []
    for row in manifest.itertuples(index=False):
        record = dict(completed.get(row.manifest_key, {}))
        rows.append(
            {
                "manifest_key": row.manifest_key,
                "game_date": row.game_date,
                "nba_game_id": row.nba_game_id,
                "snapshot_kind": row.snapshot_kind,
                "requested_snapshot_ts": row.requested_snapshot_ts,
                "manifest_status": row.execution_status,
                "execution_status": record.get("status", "pending" if row.execution_status == "ready_for_event_lookup" else "blocked"),
                "odds_api_event_id": record.get("odds_api_event_id", ""),
                "actual_snapshot_ts": record.get("actual_snapshot_ts", ""),
                "rows_parsed": record.get("rows_parsed", 0),
                "bookmakers": record.get("bookmakers", 0),
                "markets": record.get("markets", 0),
                "paired_price_rows": record.get("paired_price_rows", 0),
                "credits_used": record.get("credits_used", 0),
                "http_status": record.get("http_status", ""),
                "raw_path": record.get("raw_path", ""),
                "parsed_path": record.get("parsed_path", ""),
            }
        )
    atomic_write_csv(path, pd.DataFrame(rows))


def build_summary(manifest: pd.DataFrame, checkpoint: Dict[str, object]) -> Dict[str, object]:
    completed = checkpoint.get("completed", {})
    records = list(completed.values())
    return {
        "updated_at_utc": utc_now(),
        "manifest_rows": int(len(manifest)),
        "ready_manifest_rows": int(manifest["execution_status"].eq("ready_for_event_lookup").sum()),
        "blocked_manifest_rows": int(manifest["execution_status"].eq("blocked").sum()),
        "completed_rows": int(len(completed)),
        "completed_with_data": int(sum(int(r.get("rows_parsed", 0)) > 0 for r in records)),
        "completed_empty": int(sum(int(r.get("rows_parsed", 0)) == 0 for r in records)),
        "parsed_line_rows": int(sum(int(r.get("rows_parsed", 0)) for r in records)),
        "paired_price_rows": int(sum(int(r.get("paired_price_rows", 0)) for r in records)),
        "credits_used": int(checkpoint.get("credits_used", 0)),
        "last_api_balance": checkpoint.get("last_api_balance", {}),
        "failures": checkpoint.get("failures", []),
    }


def execute(
    manifest_path: Path,
    checkpoint_path: Path,
    ledger_path: Path,
    summary_path: Path,
    raw_root: Path,
    parsed_root: Path,
    master_path: Path,
    api_key: str,
    max_credits: int,
    max_events: Optional[int],
    flush_every: int,
    session: Optional[requests.Session] = None,
) -> Dict[str, object]:
    manifest = pd.read_csv(manifest_path, dtype={"nba_game_id": str})
    required = {
        "manifest_key", "game_date", "nba_game_id", "away_team", "home_team",
        "snapshot_kind", "requested_snapshot_ts", "event_lookup_group", "markets",
        "regions", "odds_format", "event_odds_credit_ceiling", "execution_status",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")
    if manifest["manifest_key"].duplicated().any():
        raise RuntimeError("Manifest keys are not unique")
    ready = manifest[manifest["execution_status"].eq("ready_for_event_lookup")].copy()
    if ready["requested_snapshot_ts"].isna().any():
        raise RuntimeError("Ready manifest rows contain blank request timestamps")

    manifest_hash = file_sha256(manifest_path)
    checkpoint = load_checkpoint(checkpoint_path, manifest_hash)
    completed: Dict[str, object] = checkpoint.setdefault("completed", {})
    if max_credits <= 0:
        summary = build_summary(manifest, checkpoint)
        summary["dry_run"] = True
        summary["requests_sent"] = 0
        return summary
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is not configured")
    if int(checkpoint.get("credits_used", 0)) >= max_credits:
        raise RuntimeError("Stored lifetime credit usage already meets or exceeds the requested cap")

    client = session or requests.Session()
    event_groups: Dict[str, object] = checkpoint.setdefault("event_groups", {})
    pending_parsed: List[Path] = []
    events_attempted = 0
    requests_sent = 0

    def charge(response: requests.Response, fallback: int) -> int:
        nonlocal requests_sent
        cost = response_cost(response, fallback)
        checkpoint["credits_used"] = int(checkpoint.get("credits_used", 0)) + cost
        checkpoint["last_api_balance"] = response_balance(response)
        requests_sent += 1
        save_checkpoint(checkpoint_path, checkpoint)
        return cost

    def fetch_event_list(group: str, timestamp: str, path: Path) -> object:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if int(checkpoint["credits_used"]) + 1 > max_credits:
            raise RuntimeError("Credit cap reached before event-list request")
        response = client.get(
            f"{BASE_URL}/events", params={"apiKey": api_key, "date": timestamp}, timeout=30
        )
        payload = payload_json(response)
        cost = charge(response, 1)
        saved_path = path if response.status_code == 200 else path.with_name(
            f"{path.stem}.http_{response.status_code}{path.suffix}"
        )
        atomic_write_json(saved_path, payload)
        event_groups[group] = {
            "lookup_ts": timestamp,
            "http_status": response.status_code,
            "credits_used": cost,
            "raw_path": str(saved_path.relative_to(ROOT)).replace("\\", "/"),
        }
        save_checkpoint(checkpoint_path, checkpoint)
        if response.status_code != 200:
            raise RuntimeError(f"Event-list request failed HTTP {response.status_code}")
        return payload

    # A crash can occur after a parsed event checkpoint but before the next
    # batched master rewrite. Re-materialize all completed parsed checkpoints
    # before sending another paid request; the physical-key merge is idempotent.
    recovery_paths = [ROOT / str(record.get("parsed_path", "")) for record in completed.values()]
    recovery_paths = [path for path in recovery_paths if str(path) and path.is_file()]
    if recovery_paths:
        recovered = merge_master(master_path, recovery_paths)
        print(f"Recovered parsed checkpoints into master: {recovered['rows_after']:,} rows")

    for group, group_rows in ready.groupby("event_lookup_group", sort=False):
        group_rows = group_rows.sort_values(["requested_snapshot_ts", "nba_game_id"])
        incomplete = group_rows[~group_rows["manifest_key"].isin(completed)]
        if incomplete.empty:
            continue
        lookup_ts = str(group_rows["requested_snapshot_ts"].min())
        group_path = event_list_path(raw_root, group)
        group_payload = fetch_event_list(group, lookup_ts, group_path)

        for _, row in incomplete.iterrows():
            if max_events is not None and events_attempted >= max_events:
                break
            manifest_key = str(row["manifest_key"])
            raw_path = row_raw_path(raw_root, row)
            parsed_path = row_parsed_path(parsed_root, row)
            event = match_event(group_payload, str(row["home_team"]), str(row["away_team"]))

            # If the earliest group lookup does not yet list a later game, use
            # that row's exact request timestamp once, preserving the raw lookup.
            if event is None:
                fallback_group = f"{group}:{manifest_key}"
                fallback_path = event_list_path(raw_root, fallback_group)
                fallback_payload = fetch_event_list(
                    fallback_group, str(row["requested_snapshot_ts"]), fallback_path
                )
                event = match_event(
                    fallback_payload, str(row["home_team"]), str(row["away_team"])
                )
            if event is None or not event.get("id"):
                checkpoint.setdefault("failures", []).append(
                    {"manifest_key": manifest_key, "reason": "event_not_found", "at_utc": utc_now()}
                )
                save_checkpoint(checkpoint_path, checkpoint)
                continue

            event_id = str(event["id"])
            request_ceiling = int(row["event_odds_credit_ceiling"])
            if raw_path.exists():
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                status_code = 200
                cost = 0
            else:
                if int(checkpoint["credits_used"]) + request_ceiling > max_credits:
                    print(f"Credit cap would be exceeded before {manifest_key}; stopping safely.")
                    break
                response = client.get(
                    f"{BASE_URL}/events/{event_id}/odds",
                    params={
                        "apiKey": api_key,
                        "date": str(row["requested_snapshot_ts"]),
                        "regions": str(row["regions"]),
                        "markets": str(row["markets"]),
                        "oddsFormat": str(row["odds_format"]),
                    },
                    timeout=45,
                )
                payload = payload_json(response)
                status_code = response.status_code
                cost = charge(response, request_ceiling)
                saved_raw_path = raw_path if status_code == 200 else raw_path.with_name(
                    f"{raw_path.stem}.http_{status_code}{raw_path.suffix}"
                )
                atomic_write_json(saved_raw_path, payload)
                if status_code in {401, 402, 403, 429}:
                    raise RuntimeError(f"Paid request stopped on HTTP {status_code}")
                if status_code != 200:
                    checkpoint.setdefault("failures", []).append(
                        {"manifest_key": manifest_key, "reason": f"http_{status_code}", "at_utc": utc_now()}
                    )
                    save_checkpoint(checkpoint_path, checkpoint)
                    continue

            parsed = parse_event_odds(
                payload,
                str(row["requested_snapshot_ts"]),
                game_date_override=str(row["game_date"]),
            )
            atomic_write_csv(parsed_path, parsed)
            pending_parsed.append(parsed_path)
            paired = int(parsed[["over_price", "under_price"]].notna().all(axis=1).sum()) if len(parsed) else 0
            completed[manifest_key] = {
                "status": "completed_with_data" if len(parsed) else "completed_empty",
                "completed_at_utc": utc_now(),
                "odds_api_event_id": event_id,
                "actual_snapshot_ts": str(payload.get("timestamp", "")) if isinstance(payload, dict) else "",
                "rows_parsed": int(len(parsed)),
                "bookmakers": int(parsed["bookmaker"].nunique()) if len(parsed) else 0,
                "markets": int(parsed["prop"].nunique()) if len(parsed) else 0,
                "paired_price_rows": paired,
                "credits_used": int(cost),
                "http_status": int(status_code),
                "raw_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
                "parsed_path": str(parsed_path.relative_to(ROOT)).replace("\\", "/"),
            }
            events_attempted += 1
            save_checkpoint(checkpoint_path, checkpoint)
            write_ledger(ledger_path, manifest, checkpoint)
            print(
                f"{manifest_key}: {len(parsed):,} rows, {paired:,} paired prices, "
                f"{cost} credits ({checkpoint['credits_used']:,}/{max_credits:,})"
            )
            if len(pending_parsed) >= flush_every:
                result = merge_master(master_path, pending_parsed)
                pending_parsed.clear()
                print(f"  master checkpoint: +{result['rows_added']:,} rows -> {result['rows_after']:,}")

        if max_events is not None and events_attempted >= max_events:
            break
        if int(checkpoint["credits_used"]) >= max_credits:
            break

    if pending_parsed:
        result = merge_master(master_path, pending_parsed)
        print(f"  final master checkpoint: +{result['rows_added']:,} rows -> {result['rows_after']:,}")
    write_ledger(ledger_path, manifest, checkpoint)
    summary = build_summary(manifest, checkpoint)
    summary.update({"dry_run": False, "requests_sent_this_run": requests_sent})
    atomic_write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "clv_snapshot_manifest_2026_live.csv")
    parser.add_argument("--max-credits", type=int, default=0, help="Mandatory lifetime cap; zero is dry-run")
    parser.add_argument("--max-events", type=int, default=None, help="Canary limit for event-odds calls")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "clv_snapshot_execution_checkpoint.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "data" / "clv_snapshot_execution_ledger.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "output" / "clv_snapshot_execution_summary.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "odds_api_clv_raw")
    parser.add_argument("--parsed-root", type=Path, default=ROOT / "data" / "odds_api_clv_parsed")
    parser.add_argument("--master", type=Path, default=ROOT / "data" / "historical_lines.csv")
    args = parser.parse_args()
    with execution_lock(args.checkpoint):
        summary = execute(
            manifest_path=args.manifest,
            checkpoint_path=args.checkpoint,
            ledger_path=args.ledger,
            summary_path=args.summary,
            raw_root=args.raw_root,
            parsed_root=args.parsed_root,
            master_path=args.master,
            api_key=os.environ.get("ODDS_API_KEY", ""),
            max_credits=args.max_credits,
            max_events=args.max_events,
            flush_every=max(1, args.flush_every),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
