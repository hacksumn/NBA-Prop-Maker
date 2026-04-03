"""
nba_scraper.py
--------------
Pulls player game logs from NBA.com via the nba_api package.
Covers 3 seasons (2023-24, 2024-25, 2025-26) and saves to data/nba_data.csv.

Columns saved:
    player, team, game_date, matchup, home_away,
    pts, trb, ast, stl, blk, tov, mp,
    fga, fg, fg_pct, fta, ft, 3pa, 3p, 3p_pct,
    result, plus_minus, season
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime

from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import players as nba_players

logger = logging.getLogger(__name__)

# Seasons to pull — add more here if you want deeper history
SEASONS = ["2023-24", "2024-25", "2025-26"]

# nba_api column name -> our target column name
COLUMN_MAP = {
    "PLAYER_ID":      "PLAYER_ID",
    "TEAM_ID":        "TEAM_ID",
    "GAME_ID":        "GAME_ID",
    "PLAYER_NAME":    "player",
    "TEAM_ABBREVIATION": "team",
    "GAME_DATE":      "game_date",
    "MATCHUP":        "matchup",
    "WL":             "result",
    "MIN":            "mp",
    "FGM":            "fg",
    "FGA":            "fga",
    "FG_PCT":         "fg_pct",
    "FG3M":           "3p",
    "FG3A":           "3pa",
    "FG3_PCT":        "3p_pct",
    "FTM":            "ft",
    "FTA":            "fta",
    "REB":            "trb",
    "AST":            "ast",
    "STL":            "stl",
    "BLK":            "blk",
    "TOV":            "tov",
    "PTS":            "pts",
    "PLUS_MINUS":     "plus_minus",
    "PF":             "pf",
}

FINAL_COLUMNS = [
    "PLAYER_ID", "TEAM_ID", "GAME_ID",
    "player", "team", "game_date", "matchup", "home_away",
    "pts", "trb", "ast", "stl", "blk", "tov", "mp",
    "fga", "fg", "fg_pct", "fta", "ft",
    "3pa", "3p", "3p_pct",
    "result", "plus_minus", "pf", "season",
]


def _derive_home_away(matchup: str) -> str:
    """Return 'home' if the team is hosting (matchup uses 'vs.'), else 'away'."""
    return "home" if "vs." in matchup else "away"


def fetch_season(season: str, retries: int = 3, delay: float = 2.0) -> pd.DataFrame:
    """
    Fetch all player game logs for a given season string (e.g. '2024-25').
    Returns a cleaned DataFrame with FINAL_COLUMNS.
    """
    logger.info(f"Fetching player game logs for season {season} ...")
    for attempt in range(1, retries + 1):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",   # player-level logs
                season_type_all_star="Regular Season",
            )
            df = log.get_data_frames()[0]
            break
        except Exception as exc:
            logger.warning(f"Attempt {attempt}/{retries} failed for {season}: {exc}")
            if attempt == retries:
                logger.error(f"Could not fetch {season} after {retries} attempts.")
                return pd.DataFrame(columns=FINAL_COLUMNS)
            time.sleep(delay * attempt)

    # Rename columns
    df = df.rename(columns=COLUMN_MAP)

    # Derive home/away from matchup string
    df["home_away"] = df["matchup"].apply(_derive_home_away)

    # Tag season
    df["season"] = season

    # Normalise game_date to YYYY-MM-DD
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

    # Keep only the columns we care about (ignore extras from nba_api)
    available = [c for c in FINAL_COLUMNS if c in df.columns]
    df = df[available]

    logger.info(f"  -> {len(df):,} rows fetched for {season}")
    return df


def fetch_all_seasons(seasons: list[str] | None = None) -> pd.DataFrame:
    """Fetch multiple seasons and concatenate into one DataFrame."""
    if seasons is None:
        seasons = SEASONS
    frames = []
    for season in seasons:
        df = fetch_season(season)
        if not df.empty:
            frames.append(df)
        # Be polite to NBA.com rate limits
        time.sleep(1.5)
    if not frames:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["game_date", "player"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def incremental_update(existing_path: str, seasons: list[str] | None = None) -> pd.DataFrame:
    """
    If a CSV already exists, only re-fetch the current season to pick up
    last night's games, then merge with the historical data.
    This avoids re-pulling 3 seasons of data every morning.
    """
    if seasons is None:
        seasons = SEASONS

    current_season = seasons[-1]  # most recent season

    if os.path.exists(existing_path):
        logger.info(f"Existing data found at {existing_path}. Running incremental update ...")
        existing = pd.read_csv(existing_path, dtype=str)

        # Drop current season rows (will be replaced with fresh pull)
        historical = existing[existing["season"] != current_season].copy()
        logger.info(f"  -> Kept {len(historical):,} historical rows from prior seasons")

        fresh = fetch_season(current_season)
        combined = pd.concat([historical, fresh], ignore_index=True)
    else:
        logger.info("No existing data found. Performing full 3-season pull (this may take ~60s) ...")
        combined = fetch_all_seasons(seasons)

    combined.sort_values(["game_date", "player"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def save_game_logs(output_path: str = "data/nba_data.csv",
                   incremental: bool = True,
                   seasons: list[str] | None = None) -> str:
    """
    Main entry point.  Fetches game logs and saves to output_path.
    Returns the absolute path of the saved file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if incremental:
        df = incremental_update(output_path, seasons)
    else:
        df = fetch_all_seasons(seasons)

    df.to_csv(output_path, index=False)
    logger.info(f"Game logs saved -> {output_path}  ({len(df):,} total rows)")
    return os.path.abspath(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    save_game_logs("data/nba_data.csv", incremental=True)
