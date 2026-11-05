"""Fetch exact NBA team-game box scores used by team-defense features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "nba_team_game_logs.csv"
DEFAULT_SUMMARY = ROOT / "output" / "nba_team_game_logs_summary.json"
def current_season() -> str:
    now = datetime.now()
    start_year = now.year if now.month >= 10 else now.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


DEFAULT_SEASONS = [current_season()]
SEASON_TYPES = ["Regular Season", "PlayIn", "Playoffs"]


def fetch_team_game_logs(seasons: list[str], request_pause: float = 0.6) -> pd.DataFrame:
    """Fetch regular-season and playoff team logs with exact OREB/possessions inputs."""
    from nba_api.stats.endpoints import leaguegamelog

    frames = []
    for season in seasons:
        for season_type in SEASON_TYPES:
            endpoint = leaguegamelog.LeagueGameLog(
                counter=0,
                direction="DESC",
                league_id="00",
                player_or_team_abbreviation="T",
                season=season,
                season_type_all_star=season_type,
                timeout=60,
            )
            frame = endpoint.get_data_frames()[0]
            if frame.empty:
                continue
            frame["season"] = season
            frame["season_type"] = season_type
            frames.append(frame)
            time.sleep(max(0.0, request_pause))
    if not frames:
        raise RuntimeError("NBA API returned no team-game logs")
    output = pd.concat(frames, ignore_index=True)
    required = [
        "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP",
        "PTS", "FGA", "FTA", "OREB", "TOV", "season", "season_type",
    ]
    missing = sorted(set(required) - set(output.columns))
    if missing:
        raise ValueError(f"NBA team-game response is missing columns: {missing}")
    output = output.drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"], keep="last")
    if not output.groupby("GAME_ID").size().eq(2).all():
        raise ValueError("NBA team-game response does not contain exactly two teams per game")
    if output[required].isna().any().any():
        nulls = output[required].isna().sum()
        raise ValueError(f"NBA team-game response has required nulls: {nulls[nulls.gt(0)].to_dict()}")
    return output.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--request-pause", type=float, default=0.6)
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="Replace the complete output instead of merging refreshed seasons into it.",
    )
    args = parser.parse_args()
    output = fetch_team_game_logs(args.seasons, request_pause=args.request_pause)
    if args.output.exists() and not args.replace_all:
        existing = pd.read_csv(args.output, low_memory=False)
        if "season" in existing.columns:
            existing = existing[~existing["season"].astype(str).isin(args.seasons)]
        output = pd.concat([existing, output], ignore_index=True)
        output = (
            output.drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"], keep="last")
            .sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"])
            .reset_index(drop=True)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(args.output)
    summary = {
        "status": "complete",
        "output": str(args.output),
        "rows": int(len(output)),
        "games": int(output["GAME_ID"].nunique()),
        "date_min": str(output["GAME_DATE"].min()),
        "date_max": str(output["GAME_DATE"].max()),
        "seasons": output.groupby(["season", "season_type"]).size().to_dict(),
        "duplicate_team_game_keys": int(output.duplicated(["GAME_ID", "TEAM_ABBREVIATION"]).sum()),
        "required_null_cells": int(output[["PTS", "FGA", "FTA", "OREB", "TOV"]].isna().sum().sum()),
    }
    # JSON cannot encode tuple keys.
    summary["seasons"] = {
        f"{season}|{season_type}": int(count)
        for (season, season_type), count in output.groupby(["season", "season_type"]).size().items()
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
