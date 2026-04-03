"""
pbp_fetcher.py — Layer 1: Play-by-Play Data Fetcher
=====================================================
Pulls possession-level play-by-play data from nba_api (PlayByPlayV3),
tags every play with a garbage_time flag, calculates exact possession
counts per player per game, and saves clean output to:

    data/pbp_raw/          — raw PBP CSVs, one per game
    data/pbp_possessions/  — per-player possession summaries per game
    data/pbp_index.csv     — index of all fetched games

This is the data foundation for Layer 2 (Opponent-Adjusted PPP).

Garbage Time Thresholds (industry standard):
    - Period 3: score differential >= 25 at any point
    - Period 4: score differential >= 15 at any point
    - Period 4 final 3 minutes: score differential >= 10

Usage:
    # Fetch PBP for all games in a season
    python pbp_fetcher.py --season 2024-25

    # Fetch PBP for a specific date range
    python pbp_fetcher.py --season 2024-25 --start 2025-01-01 --end 2025-03-01

    # Fetch PBP for a single game
    python pbp_fetcher.py --game-id 0022401188

    # Rebuild possession summaries from already-fetched raw PBP files
    python pbp_fetcher.py --rebuild-summaries
"""

import os
import re
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playbyplayv3, leaguegamefinder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PBP_RAW_DIR = DATA_DIR / "pbp_raw"
PBP_POSS_DIR = DATA_DIR / "pbp_possessions"
PBP_INDEX_FILE = DATA_DIR / "pbp_index.csv"

# Garbage time thresholds
GARBAGE_Q3_DIFF = 25      # Period 3: differential >= this → garbage
GARBAGE_Q4_DIFF = 15      # Period 4: differential >= this → garbage
GARBAGE_Q4_LATE_DIFF = 10 # Period 4, final 3 min: differential >= this → garbage
GARBAGE_Q4_LATE_SECS = 180  # 3 minutes = 180 seconds

# API rate limiting — nba_api is rate-limited; be respectful
REQUEST_DELAY = 1.1  # seconds between requests
PBP_TIMEOUT = 90
MAX_FETCH_RETRIES = 5
RETRY_BACKOFF_SECONDS = [20, 45, 90, 180]
LONG_COOLDOWN_EVERY = 240
LONG_COOLDOWN_SECONDS = 180
FAILURE_STREAK_COOLDOWN = 3
FAILURE_STREAK_SLEEP = 600

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def _sleep_with_log(seconds: float, reason: str):
    if seconds <= 0:
        return
    log.info(f"  Cooling down {seconds:.0f}s â€” {reason}")
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# Utility: Parse ISO 8601 clock string → seconds remaining in period
# ---------------------------------------------------------------------------

def clock_to_seconds(clock_str: str) -> float:
    """
    Convert NBA API clock format 'PT11M54.00S' to seconds remaining.
    Returns float seconds remaining in the current period.
    """
    if not clock_str or pd.isna(clock_str):
        return 0.0
    match = re.match(r"PT(\d+)M([\d.]+)S", str(clock_str))
    if not match:
        return 0.0
    minutes = int(match.group(1))
    seconds = float(match.group(2))
    return minutes * 60 + seconds


# ---------------------------------------------------------------------------
# Utility: Forward-fill score columns
# ---------------------------------------------------------------------------

def fill_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    scoreHome and scoreAway are only populated on scoring plays.
    Forward-fill them so every row has the current score.
    """
    df = df.copy()
    df["scoreHome"] = pd.to_numeric(df["scoreHome"], errors="coerce")
    df["scoreAway"] = pd.to_numeric(df["scoreAway"], errors="coerce")
    df["scoreHome"] = df["scoreHome"].ffill().fillna(0).astype(int)
    df["scoreAway"] = df["scoreAway"].ffill().fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Core: Tag garbage time
# ---------------------------------------------------------------------------

def tag_garbage_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean 'garbage_time' column to a PBP DataFrame.

    Rules:
      - Period 3: |home - away| >= GARBAGE_Q3_DIFF at any point in Q3 → all
        remaining Q3 possessions are garbage.
      - Period 4: |home - away| >= GARBAGE_Q4_DIFF at any point in Q4 → all
        remaining Q4 possessions are garbage.
      - Period 4, final 3 min: |home - away| >= GARBAGE_Q4_LATE_DIFF → garbage.
      - Overtime: treated as non-garbage (tight games only go to OT).
    """
    df = df.copy()
    df["seconds_remaining"] = df["clock"].apply(clock_to_seconds)
    df["score_diff"] = (df["scoreHome"] - df["scoreAway"]).abs()
    df["garbage_time"] = False

    for period in df["period"].unique():
        period_mask = df["period"] == period

        if period == 3:
            # Once differential hits threshold in Q3, all subsequent Q3 plays are garbage
            q3_rows = df[period_mask].copy()
            garbage_triggered = False
            garbage_indices = []
            for idx, row in q3_rows.iterrows():
                if garbage_triggered:
                    garbage_indices.append(idx)
                elif row["score_diff"] >= GARBAGE_Q3_DIFF:
                    garbage_triggered = True
                    garbage_indices.append(idx)
            df.loc[garbage_indices, "garbage_time"] = True

        elif period == 4:
            q4_rows = df[period_mask].copy()
            garbage_triggered = False
            garbage_indices = []
            for idx, row in q4_rows.iterrows():
                if garbage_triggered:
                    garbage_indices.append(idx)
                elif row["score_diff"] >= GARBAGE_Q4_DIFF:
                    garbage_triggered = True
                    garbage_indices.append(idx)
                elif (row["seconds_remaining"] <= GARBAGE_Q4_LATE_SECS and
                      row["score_diff"] >= GARBAGE_Q4_LATE_DIFF):
                    garbage_triggered = True
                    garbage_indices.append(idx)
            df.loc[garbage_indices, "garbage_time"] = True

        # Periods 1, 2, and OT (5+) are never garbage

    return df


# ---------------------------------------------------------------------------
# Core: Identify possession-ending events
# ---------------------------------------------------------------------------

def tag_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each row with a 'possession_end' boolean.

    A possession ends on:
      - Made field goal (actionType == 'Made Shot')
      - Defensive rebound (actionType == 'Rebound', subType contains 'Def')
      - Turnover (actionType == 'Turnover')
      - Final free throw of a sequence that is made (subType contains 'of 2' or
        'of 3' and the shot is made, or it's a 1-of-1)
      - End of period

    This gives us exact possession counts per team per game segment.
    """
    df = df.copy()
    df["possession_end"] = False

    action = df["actionType"].str.lower().fillna("")
    sub = df["subType"].str.lower().fillna("")
    desc = df["description"].str.lower().fillna("")

    # Made field goal
    df.loc[action == "made shot", "possession_end"] = True

    # Turnover
    df.loc[action == "turnover", "possession_end"] = True

    # Defensive rebound (ends the opponent's possession)
    def_reb = (action == "rebound") & (
        sub.str.contains("def", na=False) |
        desc.str.contains("def:", na=False)
    )
    df.loc[def_reb, "possession_end"] = True

    # Final free throw (made or missed — either way possession ends)
    # Patterns: "1 of 1", "2 of 2", "3 of 3"
    final_ft = (action == "free throw") & (
        desc.str.contains(r"\b1 of 1\b", regex=True, na=False) |
        desc.str.contains(r"\b2 of 2\b", regex=True, na=False) |
        desc.str.contains(r"\b3 of 3\b", regex=True, na=False)
    )
    df.loc[final_ft, "possession_end"] = True

    # End of period
    df.loc[action == "period", "possession_end"] = True

    return df


# ---------------------------------------------------------------------------
# Core: Build per-player possession summary for one game
# ---------------------------------------------------------------------------

def build_possession_summary(df: pd.DataFrame, game_id: str,
                              game_date: str, home_team: str,
                              away_team: str) -> pd.DataFrame:
    """
    From a tagged PBP DataFrame, compute per-player stats split by
    garbage_time flag. Returns a summary DataFrame with columns:

        game_id, game_date, player_id, player_name, team_tricode,
        is_home, period, garbage_time,
        possessions_used,   # plays where this player was the primary actor
        pts, fgm, fga, fg3m, fg3a, ftm, fta, orb, drb, ast, tov, stl, blk,
        ppp_raw             # pts / possessions_used (NaN if 0 possessions)
    """
    rows = []

    # Only look at plays where a specific player was involved
    player_plays = df[df["personId"].notna() & (df["personId"] != 0)].copy()

    for (player_id, player_name, team, garbage), grp in player_plays.groupby(
        ["personId", "playerNameI", "teamTricode", "garbage_time"]
    ):
        action = grp["actionType"].str.lower().fillna("")
        sub = grp["subType"].str.lower().fillna("")
        desc = grp["description"].str.lower().fillna("")

        # Scoring plays
        made_fg = action == "made shot"
        missed_fg = action == "missed shot"
        fg3 = grp["shotValue"].fillna(0) == 3
        fgm = made_fg.sum()
        fga = (made_fg | missed_fg).sum()
        fg3m = (made_fg & fg3).sum()
        fg3a = ((made_fg | missed_fg) & fg3).sum()

        # Free throws
        ft_rows = action == "free throw"
        ftm = (ft_rows & desc.str.contains(r"\(\d+ pts\)", regex=True, na=False)).sum()
        fta = ft_rows.sum()

        # Points
        pts = grp.loc[made_fg, "pointsTotal"].fillna(0).diff().clip(lower=0).sum()
        # Simpler: use shotValue for FG + count made FTs
        pts_fg = grp.loc[made_fg, "shotValue"].fillna(0).sum()
        pts_ft = grp.loc[ft_rows & desc.str.contains(r"\(\d+ pts\)", regex=True, na=False)].shape[0]
        pts = int(pts_fg + pts_ft)

        # Rebounds — parse from description pattern "(Off:N Def:N)"
        # NBA PBP V3 uses subType='Unknown' for most rebounds, so we must
        # parse the description directly. Pattern: "REBOUND (Off:1 Def:0)"
        # Off:N > 0 means offensive rebound; Def:N > 0 means defensive rebound.
        reb_rows = action == "rebound"
        # Offensive rebound: Off: followed by a non-zero digit
        orb = (reb_rows & desc.str.contains(r"\(off:[1-9]", regex=True, na=False)).sum()
        # Defensive rebound: Def: followed by a non-zero digit
        drb = (reb_rows & desc.str.contains(r"def:[1-9]", regex=True, na=False)).sum()

        # Assists
        ast = desc.str.contains(r"\d+ ast\b", regex=True, na=False).sum()

        # Turnovers
        tov = (action == "turnover").sum()

        # Steals
        stl = desc.str.contains(r"steal", na=False).sum()

        # Blocks
        blk = desc.str.contains(r"\d+ blk\b", regex=True, na=False).sum()

        # Possessions used: FGA + 0.44*FTA + TOV - ORB
        possessions_used = fga + 0.44 * fta + tov - orb
        possessions_used = max(0, round(possessions_used, 2))

        ppp_raw = round(pts / possessions_used, 4) if possessions_used > 0 else np.nan

        rows.append({
            "game_id": game_id,
            "game_date": game_date,
            "player_id": int(player_id),
            "player_name": player_name,
            "team_tricode": team,
            "is_home": 1 if team == home_team else 0,
            "opponent_tricode": away_team if team == home_team else home_team,
            "garbage_time": int(garbage),
            "possessions_used": possessions_used,
            "pts": pts,
            "fgm": int(fgm),
            "fga": int(fga),
            "fg3m": int(fg3m),
            "fg3a": int(fg3a),
            "ftm": int(ftm),
            "fta": int(fta),
            "orb": int(orb),
            "drb": int(drb),
            "ast": int(ast),
            "tov": int(tov),
            "stl": int(stl),
            "blk": int(blk),
            "ppp_raw": ppp_raw,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core: Fetch and process a single game
# ---------------------------------------------------------------------------

def fetch_game(game_id: str, game_date: str, home_team: str,
               away_team: str, force: bool = False) -> bool:
    """
    Fetch PBP for one game, tag garbage time and possessions,
    save raw PBP and possession summary CSVs.
    Returns True on success, False on failure.
    """
    PBP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PBP_POSS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = PBP_RAW_DIR / f"{game_id}.csv"
    poss_path = PBP_POSS_DIR / f"{game_id}_possessions.csv"

    if raw_path.exists() and poss_path.exists() and not force:
        log.debug(f"  Already fetched: {game_id} — skipping")
        return True

    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=PBP_TIMEOUT)
            df = pbp.get_data_frames()[0]

            if df.empty:
                log.warning(f"  Empty PBP for game {game_id}")
                return False

            # Fill scores forward
            df = fill_scores(df)

            # Tag garbage time
            df = tag_garbage_time(df)

            # Tag possession ends
            df = tag_possessions(df)

            # Add metadata
            df["game_id"] = game_id
            df["game_date"] = game_date
            df["home_team"] = home_team
            df["away_team"] = away_team

            # Save raw PBP
            df.to_csv(raw_path, index=False)

            # Build and save possession summary
            summary = build_possession_summary(df, game_id, game_date,
                                               home_team, away_team)
            summary.to_csv(poss_path, index=False)

            log.info(f"  ✓ {game_id}  {away_team} @ {home_team}  ({game_date})  "
                     f"{len(df)} plays  "
                     f"{df['garbage_time'].sum()} garbage  "
                     f"{df['possession_end'].sum()} possessions")
            return True

        except Exception as e:
            if attempt >= MAX_FETCH_RETRIES:
                log.error(f"  ✗ Failed {game_id}: {e}")
                return False

            backoff = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
            log.warning(f"  Retry {attempt}/{MAX_FETCH_RETRIES - 1} for {game_id} after error: {e}")
            _sleep_with_log(backoff, f"retrying game {game_id}")

    return False


# ---------------------------------------------------------------------------
# Core: Fetch all games for a season
# ---------------------------------------------------------------------------

def fetch_season(season: str, start_date: str = None, end_date: str = None,
                 force: bool = False):
    """
    Fetch PBP for all completed games in a season.
    Optionally filter by date range (YYYY-MM-DD format).
    """
    log.info(f"Fetching game list for {season}...")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00"
    )
    games = finder.get_data_frames()[0]
    time.sleep(REQUEST_DELAY)

    # Keep only one row per game (LeagueGameFinder returns 2 rows per game)
    games = games.drop_duplicates(subset="GAME_ID")
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    # Filter by date range if provided
    if start_date:
        games = games[games["GAME_DATE"] >= pd.to_datetime(start_date)]
    if end_date:
        games = games[games["GAME_DATE"] <= pd.to_datetime(end_date)]

    # Only completed games (past today)
    today = pd.Timestamp.now().normalize()
    games = games[games["GAME_DATE"] < today]

    log.info(f"Found {len(games)} games to process")

    # Build index of game metadata
    index_rows = []
    success = 0
    failed = 0

    consecutive_failures = 0
    processed_since_cooldown = 0

    for _, row in games.iterrows():
        game_id = row["GAME_ID"]
        game_date = row["GAME_DATE"].strftime("%Y-%m-%d")
        matchup = row["MATCHUP"]  # e.g. "BKN vs. NYK" or "BKN @ NYK"

        # Parse home/away from matchup
        if " vs. " in matchup:
            home_team = matchup.split(" vs. ")[0].strip()
            away_team = matchup.split(" vs. ")[1].strip()
        elif " @ " in matchup:
            away_team = matchup.split(" @ ")[0].strip()
            home_team = matchup.split(" @ ")[1].strip()
        else:
            home_team = matchup[:3]
            away_team = matchup[-3:]

        result = fetch_game(game_id, game_date, home_team, away_team, force)
        if result:
            success += 1
            consecutive_failures = 0
            index_rows.append({
                "game_id": game_id,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "season": season,
                "fetched": True
            })
        else:
            failed += 1
            consecutive_failures += 1
            index_rows.append({
                "game_id": game_id,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "season": season,
                "fetched": False
            })

        processed_since_cooldown += 1
        if consecutive_failures >= FAILURE_STREAK_COOLDOWN:
            _sleep_with_log(FAILURE_STREAK_SLEEP, f"{consecutive_failures} consecutive failures")
            consecutive_failures = 0
            processed_since_cooldown = 0
        elif processed_since_cooldown >= LONG_COOLDOWN_EVERY:
            _sleep_with_log(LONG_COOLDOWN_SECONDS, f"processed {processed_since_cooldown} games")
            processed_since_cooldown = 0

        time.sleep(REQUEST_DELAY)

    # Save/update index
    new_index = pd.DataFrame(index_rows)
    if PBP_INDEX_FILE.exists():
        try:
            existing = pd.read_csv(PBP_INDEX_FILE)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, new_index]).drop_duplicates(
            subset="game_id", keep="last"
        )
        combined.to_csv(PBP_INDEX_FILE, index=False)
    else:
        new_index.to_csv(PBP_INDEX_FILE, index=False)

    log.info(f"\nDone. {success} succeeded, {failed} failed.")
    log.info(f"Raw PBP saved to:        {PBP_RAW_DIR}")
    log.info(f"Possession summaries to: {PBP_POSS_DIR}")
    log.info(f"Index updated at:        {PBP_INDEX_FILE}")


# ---------------------------------------------------------------------------
# Core: Rebuild possession summaries from existing raw PBP files
# ---------------------------------------------------------------------------

def rebuild_summaries():
    """
    Re-process all existing raw PBP CSVs and regenerate possession summaries.
    Useful after updating the garbage time or possession tagging logic.
    """
    raw_files = list(PBP_RAW_DIR.glob("*.csv"))
    log.info(f"Rebuilding summaries for {len(raw_files)} games...")

    try:
        index = pd.read_csv(PBP_INDEX_FILE) if PBP_INDEX_FILE.exists() else pd.DataFrame()
    except Exception:
        index = pd.DataFrame()

    for raw_path in raw_files:
        game_id = raw_path.stem
        df = pd.read_csv(raw_path)

        # Get metadata from index
        if not index.empty and game_id in index["game_id"].values:
            meta = index[index["game_id"] == game_id].iloc[0]
            game_date = meta["game_date"]
            home_team = meta["home_team"]
            away_team = meta["away_team"]
        else:
            game_date = df["game_date"].iloc[0] if "game_date" in df.columns else "unknown"
            home_team = df["home_team"].iloc[0] if "home_team" in df.columns else "UNK"
            away_team = df["away_team"].iloc[0] if "away_team" in df.columns else "UNK"

        # Re-tag
        df = fill_scores(df)
        df = tag_garbage_time(df)
        df = tag_possessions(df)

        # Rebuild summary
        summary = build_possession_summary(df, game_id, game_date,
                                           home_team, away_team)
        poss_path = PBP_POSS_DIR / f"{game_id}_possessions.csv"
        summary.to_csv(poss_path, index=False)
        log.info(f"  Rebuilt: {game_id}")

    log.info("Rebuild complete.")


# ---------------------------------------------------------------------------
# Utility: Merge all possession summaries into one master file
# ---------------------------------------------------------------------------

def merge_possession_summaries(output_path: str = None) -> pd.DataFrame:
    """
    Concatenate all per-game possession summary CSVs into one master DataFrame.
    Saves to data/pbp_possessions_master.csv by default.
    Returns the merged DataFrame.
    """
    poss_files = list(PBP_POSS_DIR.glob("*_possessions.csv"))
    if not poss_files:
        log.warning("No possession summary files found.")
        return pd.DataFrame()

    log.info(f"Merging {len(poss_files)} possession summary files...")
    dfs = []
    for f in poss_files:
        try:
            df = pd.read_csv(f)
            if df.empty or len(df.columns) == 0:
                log.warning(f"  Skipping empty file: {f.name}")
                continue
            dfs.append(df)
        except Exception as e:
            log.warning(f"  Skipping corrupt file {f.name}: {e}")
    if not dfs:
        log.warning("No valid possession summary files to merge.")
        return pd.DataFrame()
    master = pd.concat(dfs, ignore_index=True)

    out = output_path or str(DATA_DIR / "pbp_possessions_master.csv")
    master.to_csv(out, index=False)
    log.info(f"Master possession file saved: {out}  ({len(master):,} rows)")
    return master


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1: NBA Play-by-Play Fetcher with Garbage Time Tagging"
    )
    parser.add_argument("--season", default="2025-26",
                        help="NBA season string, e.g. 2025-26 (default: 2025-26)")
    parser.add_argument("--start", default=None,
                        help="Start date filter YYYY-MM-DD")
    parser.add_argument("--end", default=None,
                        help="End date filter YYYY-MM-DD")
    parser.add_argument("--game-id", default=None,
                        help="Fetch a single game by ID")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if already downloaded")
    parser.add_argument("--rebuild-summaries", action="store_true",
                        help="Rebuild possession summaries from existing raw files")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all possession summaries into master CSV")
    args = parser.parse_args()

    # Create output directories
    PBP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PBP_POSS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_summaries:
        rebuild_summaries()
    elif args.merge:
        merge_possession_summaries()
    elif args.game_id:
        # Single game — need to look up metadata
        finder = leaguegamefinder.LeagueGameFinder(
            game_id_nullable=args.game_id,
            league_id_nullable="00"
        )
        games = finder.get_data_frames()[0]
        if games.empty:
            log.error(f"Game {args.game_id} not found.")
            return
        row = games.iloc[0]
        matchup = row["MATCHUP"]
        game_date = str(row["GAME_DATE"])[:10]
        if " vs. " in matchup:
            home_team = matchup.split(" vs. ")[0].strip()
            away_team = matchup.split(" vs. ")[1].strip()
        elif " @ " in matchup:
            away_team = matchup.split(" @ ")[0].strip()
            home_team = matchup.split(" @ ")[1].strip()
        else:
            home_team = matchup[:3]
            away_team = matchup[-3:]
        fetch_game(args.game_id, game_date, home_team, away_team, args.force)
    else:
        fetch_season(args.season, args.start, args.end, args.force)

    # Always merge after fetching
    if not args.rebuild_summaries:
        merge_possession_summaries()


if __name__ == "__main__":
    main()
