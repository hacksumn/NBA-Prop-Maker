"""
nba_scraper.py
--------------
Pulls player game logs from NBA.com via the nba_api package.
Covers a rolling 3-season window and includes regular season plus postseason
logs so the daily runner can keep grading active slates after the regular
season ends.

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

logger = logging.getLogger(__name__)

SEASON_TYPES = ("Regular Season", "PlayIn", "Playoffs")

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


def _season_from_date(date_value=None) -> str:
    """Infer NBA season string from a date-like value."""
    ts = pd.to_datetime(date_value or datetime.now(), errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp(datetime.now())
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _default_seasons(as_of=None, depth: int = 3) -> list[str]:
    """Return the rolling season window the daily runner should maintain."""
    current_season = _season_from_date(as_of)
    current_start = int(current_season.split("-")[0])
    return [
        f"{year}-{(year + 1) % 100:02d}"
        for year in range(current_start - depth + 1, current_start + 1)
    ]


# Seasons to pull by default — keeps the repo current without manual year bumps.
SEASONS = _default_seasons()


def _derive_home_away(matchup: str) -> str:
    """Return 'home' if the team is hosting (matchup uses 'vs.'), else 'away'."""
    return "home" if "vs." in matchup else "away"


def _clean_game_log_frame(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Normalize raw nba_api player game logs to the shared CSV schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    df = df.rename(columns=COLUMN_MAP).copy()

    if "matchup" in df.columns:
        df["home_away"] = df["matchup"].fillna("").astype(str).apply(_derive_home_away)
    else:
        df["home_away"] = ""

    df["season"] = season
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["game_date"] = ""

    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[FINAL_COLUMNS]


def _fetch_season_phase(season: str,
                        season_type: str,
                        retries: int = 3,
                        delay: float = 2.0) -> pd.DataFrame:
    """Fetch one season phase (regular season / play-in / playoffs)."""
    for attempt in range(1, retries + 1):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",
                season_type_all_star=season_type,
            )
            return _clean_game_log_frame(log.get_data_frames()[0], season)
        except Exception as exc:
            logger.warning(
                f"Attempt {attempt}/{retries} failed for {season} [{season_type}]: {exc}"
            )
            if attempt == retries:
                logger.error(f"Could not fetch {season} [{season_type}] after {retries} attempts.")
                return pd.DataFrame(columns=FINAL_COLUMNS)
            time.sleep(delay * attempt)

    return pd.DataFrame(columns=FINAL_COLUMNS)


def fetch_season(season: str, retries: int = 3, delay: float = 2.0) -> pd.DataFrame:
    """
    Fetch all player game logs for a given season string (e.g. '2024-25').
    Returns a cleaned DataFrame with FINAL_COLUMNS.
    """
    logger.info(f"Fetching player game logs for season {season} ...")
    frames = []
    phase_counts = []

    for idx, season_type in enumerate(SEASON_TYPES):
        phase_df = _fetch_season_phase(season, season_type, retries=retries, delay=delay)
        if not phase_df.empty:
            frames.append(phase_df)
            phase_counts.append(f"{season_type}={len(phase_df):,}")
        if idx < len(SEASON_TYPES) - 1:
            time.sleep(0.75)

    if not frames:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    if {"PLAYER_ID", "GAME_ID"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="last")
    else:
        combined = combined.drop_duplicates(subset=["player", "team", "game_date"], keep="last")
    combined = combined.sort_values(["game_date", "player"]).reset_index(drop=True)

    phase_label = ", ".join(phase_counts) if phase_counts else "no phase rows"
    logger.info(f"  -> {len(combined):,} rows fetched for {season} ({phase_label})")
    return combined


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
