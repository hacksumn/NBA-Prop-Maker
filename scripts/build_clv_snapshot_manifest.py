"""Build the audited Odds API decision/T-60/T-5 CLV request manifest.

This script does not call The Odds API. It reconciles local NBA game IDs to
scheduled tip times, recovers card-decision timestamps from immutable archived
pick files, and writes one instruction row per game and snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START = "2026-03-30"
DEFAULT_END = "2026-05-18"
LOCAL_TIMEZONE = "America/Chicago"
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
ESPN_TEAM_ALIASES = {
    "GS": "GSW",
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}
STAMPED_PICK_RE = re.compile(r"picks_(\d{4}-\d{2}-\d{2})_(\d{6})\.csv$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    return str(shown).replace("\\", "/")


def _parse_matchup(value: object) -> Tuple[str, str]:
    text = str(value).replace("vs.", "vs")
    if " @ " in text:
        away, home = text.split(" @ ", 1)
        return away.strip(), home.strip()
    if " vs " in text:
        home, away = text.split(" vs ", 1)
        return away.strip(), home.strip()
    raise ValueError(f"Unrecognized matchup: {value!r}")


def load_local_games(path: Path, start: str, end: str) -> pd.DataFrame:
    games = pd.read_csv(
        path,
        usecols=["GAME_ID", "game_date", "matchup"],
        dtype={"GAME_ID": str},
        low_memory=False,
    )
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    games = games[games["game_date"].between(start, end)].drop_duplicates(
        ["game_date", "GAME_ID"]
    )
    parsed = games["matchup"].apply(_parse_matchup)
    games["away_team"] = parsed.str[0]
    games["home_team"] = parsed.str[1]
    return games.rename(columns={"GAME_ID": "nba_game_id"})[
        ["game_date", "nba_game_id", "away_team", "home_team"]
    ].sort_values(["game_date", "nba_game_id"]).reset_index(drop=True)


def _schedule_cache_path(start: str, end: str) -> Path:
    return ROOT / "data" / f"clv_schedule_{start}_{end}.json"


def fetch_schedule(game_dates: Iterable[str], timeout: int = 30) -> List[Dict[str, str]]:
    session = requests.Session()
    rows: List[Dict[str, str]] = []
    for game_date in sorted(set(game_dates)):
        source_url = (
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            f"?dates={game_date.replace('-', '')}&limit=100"
        )
        response = session.get(source_url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        for event in payload.get("events", []):
            competition = event["competitions"][0]
            teams = {
                item["homeAway"]: ESPN_TEAM_ALIASES.get(
                    item["team"]["abbreviation"], item["team"]["abbreviation"]
                )
                for item in competition["competitors"]
            }
            rows.append(
                {
                    "game_date": game_date,
                    "away_team": teams["away"],
                    "home_team": teams["home"],
                    "tip_ts_utc": event["date"],
                    "schedule_event_id": str(event["id"]),
                    "schedule_source": "espn_scoreboard_api",
                    "schedule_source_url": source_url,
                }
            )
    return rows


def load_or_fetch_schedule(
    game_dates: Iterable[str], start: str, end: str, refresh: bool = False
) -> pd.DataFrame:
    cache_path = _schedule_cache_path(start, end)
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        rows = cached["games"]
    else:
        rows = fetch_schedule(game_dates)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "window_start": start,
                    "window_end": end,
                    "games": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    schedule = pd.DataFrame(rows)
    if len(schedule):
        schedule["tip_ts_utc"] = pd.to_datetime(schedule["tip_ts_utc"], utc=True)
    return schedule


def resolve_decision_timestamp(
    archive_root: Path, slate_date: str, local_timezone: str = LOCAL_TIMEZONE
) -> Dict[str, str]:
    """Resolve the first timestamped archive identical to the final daily card."""
    day_dir = archive_root / slate_date
    alias = day_dir / f"picks_{slate_date}.csv"
    base = {
        "decision_ts_utc": "",
        "decision_ts_local": "",
        "decision_status": "blocked",
        "decision_block_reason": "",
        "decision_source_path": "",
        "decision_source_sha256": "",
        "decision_rule": "first_timestamped_archive_identical_to_final_daily_card",
    }
    if not alias.exists():
        base["decision_block_reason"] = "final_daily_card_missing"
        return base

    alias_hash = _sha256(alias)
    matches: List[Tuple[datetime, Path]] = []
    for path in day_dir.glob("picks_*.csv"):
        match = STAMPED_PICK_RE.fullmatch(path.name)
        if not match or _sha256(path) != alias_hash:
            continue
        stamp = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H%M%S")
        matches.append((stamp.replace(tzinfo=ZoneInfo(local_timezone)), path))

    if not matches:
        base["decision_block_reason"] = "no_timestamped_archive_matches_final_daily_card"
        base["decision_source_path"] = _display_path(alias)
        base["decision_source_sha256"] = alias_hash
        return base

    local_dt, source_path = min(matches, key=lambda item: item[0])
    utc_dt = local_dt.astimezone(timezone.utc)
    base.update(
        {
            "decision_ts_utc": utc_dt.isoformat().replace("+00:00", "Z"),
            "decision_ts_local": local_dt.isoformat(),
            "decision_status": "resolved",
            "decision_source_path": _display_path(source_path),
            "decision_source_sha256": alias_hash,
        }
    )
    return base


def _iso_z(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def build_manifest(
    games: pd.DataFrame,
    schedule: pd.DataFrame,
    archive_root: Path,
    local_timezone: str = LOCAL_TIMEZONE,
) -> pd.DataFrame:
    join_keys = ["game_date", "away_team", "home_team"]
    if schedule.duplicated(join_keys).any():
        raise RuntimeError("Schedule contains duplicate date/matchup keys")
    merged = games.merge(schedule, on=join_keys, how="left", validate="one_to_one")
    if merged["tip_ts_utc"].isna().any():
        missing = merged.loc[merged["tip_ts_utc"].isna(), join_keys].to_dict("records")
        raise RuntimeError(f"Schedule did not match every local game: {missing}")

    decision_by_date = {
        game_date: resolve_decision_timestamp(archive_root, game_date, local_timezone)
        for game_date in merged["game_date"].unique()
    }
    market_string = ",".join(MARKETS)
    rows: List[Dict[str, object]] = []
    for game in merged.itertuples(index=False):
        tip_ts = pd.Timestamp(game.tip_ts_utc)
        decision = decision_by_date[game.game_date]
        snapshots: List[Tuple[str, int, Optional[pd.Timestamp]]] = [
            (
                "decision",
                0,
                pd.Timestamp(decision["decision_ts_utc"])
                if decision["decision_ts_utc"]
                else None,
            ),
            ("t_minus_60", 60, tip_ts - timedelta(minutes=60)),
            ("t_minus_5", 5, tip_ts - timedelta(minutes=5)),
        ]
        for order, (kind, minutes_before_tip, requested_ts) in enumerate(snapshots, start=1):
            status = "ready_for_event_lookup"
            block_reason = ""
            if requested_ts is None:
                status = "blocked"
                block_reason = decision["decision_block_reason"]
            elif requested_ts >= tip_ts:
                status = "blocked"
                block_reason = "decision_timestamp_not_before_tip"
            request_text = _iso_z(requested_ts) if requested_ts is not None else ""
            snapshot_key = f"{game.nba_game_id}:{kind}"
            rows.append(
                {
                    "manifest_key": snapshot_key,
                    "game_date": game.game_date,
                    "nba_game_id": game.nba_game_id,
                    "away_team": game.away_team,
                    "home_team": game.home_team,
                    "tip_ts_utc": _iso_z(tip_ts),
                    "schedule_event_id": game.schedule_event_id,
                    "schedule_source": game.schedule_source,
                    "schedule_source_url": game.schedule_source_url,
                    "snapshot_order": order,
                    "snapshot_kind": kind,
                    "minutes_before_tip": minutes_before_tip,
                    "requested_snapshot_ts": request_text,
                    "event_lookup_group": f"{game.game_date}:{kind}",
                    "odds_api_event_id": "",
                    "endpoint_template": (
                        "/v4/historical/sports/basketball_nba/events/"
                        "{odds_api_event_id}/odds"
                    ),
                    "regions": "us",
                    "bookmakers": "",
                    "markets": market_string,
                    "market_count": len(MARKETS),
                    "odds_format": "american",
                    "event_odds_credit_ceiling": len(MARKETS) * 10,
                    "event_list_credit_ceiling": 0,
                    "total_credit_ceiling": len(MARKETS) * 10,
                    "execution_status": status,
                    "block_reason": block_reason,
                    "checkpoint_key": snapshot_key,
                    "decision_ts_local": decision["decision_ts_local"] if kind == "decision" else "",
                    "decision_rule": decision["decision_rule"] if kind == "decision" else "",
                    "decision_source_path": decision["decision_source_path"] if kind == "decision" else "",
                    "decision_source_sha256": decision["decision_source_sha256"] if kind == "decision" else "",
                }
            )

    manifest = pd.DataFrame(rows).sort_values(
        ["game_date", "snapshot_order", "tip_ts_utc", "nba_game_id"]
    ).reset_index(drop=True)
    ready = manifest["execution_status"].eq("ready_for_event_lookup")
    # One historical event-list lookup per date/snapshot group. Its event IDs
    # are reused by all games in that exact group.
    leaders = manifest.loc[ready].groupby("event_lookup_group", sort=False).head(1).index
    manifest.loc[leaders, "event_list_credit_ceiling"] = 1
    manifest["total_credit_ceiling"] = (
        manifest["event_odds_credit_ceiling"] + manifest["event_list_credit_ceiling"]
    )
    return manifest


def validate_manifest(manifest: pd.DataFrame, expected_games: int) -> Dict[str, object]:
    if len(manifest) != expected_games * 3:
        raise RuntimeError(f"Expected {expected_games * 3} rows, found {len(manifest)}")
    if manifest["manifest_key"].duplicated().any():
        raise RuntimeError("Manifest keys are not unique")
    per_game = manifest.groupby("nba_game_id")["snapshot_kind"].nunique()
    if not per_game.eq(3).all():
        raise RuntimeError("Every game must have decision, T-60, and T-5 rows")
    ready = manifest[manifest["execution_status"].eq("ready_for_event_lookup")]
    blocked = manifest[manifest["execution_status"].eq("blocked")]
    return {
        "manifest_rows": int(len(manifest)),
        "games": int(expected_games),
        "game_dates": int(manifest["game_date"].nunique()),
        "ready_rows": int(len(ready)),
        "blocked_rows": int(len(blocked)),
        "rows_by_snapshot": manifest.groupby("snapshot_kind").size().astype(int).to_dict(),
        "ready_rows_by_snapshot": ready.groupby("snapshot_kind").size().astype(int).to_dict(),
        "blocked_reasons": blocked["block_reason"].value_counts().astype(int).to_dict(),
        "ready_event_odds_credit_ceiling": int(ready["event_odds_credit_ceiling"].sum()),
        "ready_event_list_credit_ceiling": int(ready["event_list_credit_ceiling"].sum()),
        "ready_total_credit_ceiling": int(ready["total_credit_ceiling"].sum()),
        "full_panel_event_odds_credit_ceiling": int(len(manifest) * len(MARKETS) * 10),
        "full_panel_event_list_credit_ceiling": int(
            manifest[["game_date", "snapshot_kind"]].drop_duplicates().shape[0]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--refresh-schedule", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "clv_snapshot_manifest_2026_live.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "output" / "clv_snapshot_manifest_summary.json",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.summary = args.summary.resolve()

    games = load_local_games(ROOT / "data" / "nba_data.csv", args.start, args.end)
    schedule = load_or_fetch_schedule(
        games["game_date"], args.start, args.end, refresh=args.refresh_schedule
    )
    manifest = build_manifest(games, schedule, ROOT / "output" / "archive")
    summary = validate_manifest(manifest, len(games))
    summary.update(
        {
            "window_start": args.start,
            "window_end": args.end,
            "manifest_path": str(args.output.relative_to(ROOT)).replace("\\", "/"),
            "schedule_cache_path": str(_schedule_cache_path(args.start, args.end).relative_to(ROOT)).replace("\\", "/"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "odds_api_requests_sent": 0,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
