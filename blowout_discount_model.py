"""
blowout_discount_model.py — Layer 5: Blowout Discount Model
============================================================
Identifies blowout games, strips garbage time stats from player game logs,
and produces clean "competitive minutes only" baselines for all downstream
projections (Layers 2, 3, and 4).

The Problem:
    Standard rolling averages treat all games equally. A player who scores
    35 points in a 40-point blowout win (garbage time included) looks the
    same as a player who scores 35 points in a tight 2-point game. They are
    NOT the same. The blowout game is statistical noise — the player was
    likely padding stats against scrubs in the final 8 minutes.

The Solution — Three-Tier Blowout Classification:
    TIER 1 — FULL GAME (no discount):
        Final margin ≤ 15 points. Competitive throughout.
        Weight: 1.00 (full credit)

    TIER 2 — PARTIAL DISCOUNT (moderate blowout):
        Final margin 16-24 points. Likely garbage time in Q4.
        Weight: 0.75 (25% discount applied to stat line)
        Competitive minutes estimated as first 3 quarters + early Q4.

    TIER 3 — HEAVY DISCOUNT (blowout):
        Final margin ≥ 25 points. Heavy garbage time in Q3 and Q4.
        Weight: 0.50 (50% discount applied to stat line)
        Only Q1 and Q2 stats considered fully reliable.

    TIER 4 — EXCLUDE (extreme blowout):
        Final margin ≥ 35 points. Game was decided by halftime.
        Weight: 0.00 (game excluded entirely from rolling averages)

The Plus/Minus Adjustment:
    Beyond final margin, we also check each player's individual on-court
    plus/minus (plusMinusPoints from BoxScoreTraditionalV3). If a player
    had a +25 on-court differential in a 20-point game, they were clearly
    on the floor during the blowout portion. Their stats get an additional
    discount on top of the game-level tier.

Outputs:
    data/game_blowout_index.csv      — all games tagged with blowout tier
    data/player_blowout_log.csv      — per-player per-game weighted stats
    data/player_clean_baselines.csv  — rolling competitive-minutes averages

Usage:
    # Build the full blowout index and clean baselines
    python blowout_discount_model.py --build [--last-n 30]

    # Show blowout game index for a specific team
    python blowout_discount_model.py --team DEN

    # Show clean vs raw stat comparison for a player
    python blowout_discount_model.py --player "Nikola Jokic"

    # Show all games flagged as blowouts this season
    python blowout_discount_model.py --blowouts [--threshold 25]
"""

import os
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import (
    leaguegamelog,
    boxscoretraditionalv3,
    playergamelog,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

GAME_INDEX_OUT    = DATA_DIR / "game_blowout_index.csv"
PLAYER_LOG_OUT    = DATA_DIR / "player_blowout_log.csv"
CLEAN_BASELINE_OUT = DATA_DIR / "player_clean_baselines.csv"

SEASON    = "2024-25"
API_DELAY = 0.65

# Blowout tier thresholds (absolute final margin)
TIER_FULL_MAX     = 15   # ≤15 = full credit
TIER_PARTIAL_MAX  = 24   # 16-24 = 75% weight
TIER_HEAVY_MAX    = 34   # 25-34 = 50% weight
# ≥35 = 0% weight (excluded)

# Weights per tier
TIER_WEIGHTS = {
    "FULL":    1.00,
    "PARTIAL": 0.75,
    "HEAVY":   0.50,
    "EXCLUDE": 0.00,
}

# Individual on-court plus/minus threshold for additional player-level discount
# If player's on-court +/- exceeds this in a blowout game, apply extra discount
PLAYER_PM_THRESHOLD = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Build the Game Blowout Index
# ---------------------------------------------------------------------------

def classify_blowout_tier(margin: float) -> str:
    """Classify a game by its absolute final score margin."""
    margin = abs(margin)
    if margin <= TIER_FULL_MAX:
        return "FULL"
    elif margin <= TIER_PARTIAL_MAX:
        return "PARTIAL"
    elif margin <= TIER_HEAVY_MAX:
        return "HEAVY"
    else:
        return "EXCLUDE"


def build_game_blowout_index(season: str = SEASON) -> pd.DataFrame:
    """
    Fetch all games for the season and classify each by blowout tier.

    Returns one row per game (not per team) with:
        game_id, game_date, home_team, away_team, home_pts, away_pts,
        final_margin, blowout_tier, blowout_weight
    """
    log.info("Fetching league game log...")
    lgl = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T"
    ).get_data_frames()[0]
    time.sleep(API_DELAY)

    log.info(f"Processing {len(lgl)} team-game records...")

    # Pivot to one row per game (two teams → one row)
    # We use PLUS_MINUS which is the same absolute value for both teams
    lgl["abs_margin"] = lgl["PLUS_MINUS"].abs()

    # Get unique games — take the home team row (MATCHUP contains 'vs.')
    home_games = lgl[lgl["MATCHUP"].str.contains(r"vs\.", na=False)].copy()
    away_games = lgl[lgl["MATCHUP"].str.contains("@", na=False)].copy()

    # Build game index
    rows = []
    for _, hrow in home_games.iterrows():
        game_id = hrow["GAME_ID"]
        arow = away_games[away_games["GAME_ID"] == game_id]
        if arow.empty:
            continue
        arow = arow.iloc[0]

        margin = abs(hrow["PLUS_MINUS"])
        tier = classify_blowout_tier(margin)
        weight = TIER_WEIGHTS[tier]

        # Determine winner
        winner = hrow["TEAM_ABBREVIATION"] if hrow["WL"] == "W" else arow["TEAM_ABBREVIATION"]
        loser  = arow["TEAM_ABBREVIATION"] if hrow["WL"] == "W" else hrow["TEAM_ABBREVIATION"]

        rows.append({
            "game_id":       game_id,
            "game_date":     hrow["GAME_DATE"],
            "home_team":     hrow["TEAM_ABBREVIATION"],
            "away_team":     arow["TEAM_ABBREVIATION"],
            "home_pts":      int(hrow["PTS"]),
            "away_pts":      int(arow["PTS"]),
            "final_margin":  margin,
            "winner":        winner,
            "loser":         loser,
            "blowout_tier":  tier,
            "blowout_weight": weight,
        })

    df = pd.DataFrame(rows).sort_values("game_date", ascending=False)
    log.info(f"Built blowout index for {len(df)} games")

    # Summary
    tier_counts = df["blowout_tier"].value_counts()
    log.info(f"  FULL (≤{TIER_FULL_MAX}):    {tier_counts.get('FULL', 0)} games")
    log.info(f"  PARTIAL (≤{TIER_PARTIAL_MAX}): {tier_counts.get('PARTIAL', 0)} games")
    log.info(f"  HEAVY (≤{TIER_HEAVY_MAX}):   {tier_counts.get('HEAVY', 0)} games")
    log.info(f"  EXCLUDE (≥35): {tier_counts.get('EXCLUDE', 0)} games")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Build Per-Player Weighted Game Log
# ---------------------------------------------------------------------------

def parse_minutes(min_str) -> float:
    """Convert 'MM:SS' string to float minutes."""
    if pd.isna(min_str) or min_str == "" or min_str is None:
        return 0.0
    try:
        parts = str(min_str).split(":")
        return float(parts[0]) + float(parts[1]) / 60.0
    except Exception:
        return 0.0


def apply_player_pm_discount(base_weight: float, player_pm: float, game_tier: str) -> float:
    """
    Apply an additional discount if a player's individual on-court +/-
    suggests they were heavily involved in garbage time.

    A player with +25 on-court in a PARTIAL blowout was likely on the floor
    for most of the blowout portion — discount more aggressively.
    """
    if game_tier == "FULL":
        return base_weight  # No additional discount for competitive games

    pm = abs(player_pm)
    if pm >= PLAYER_PM_THRESHOLD * 1.5:  # e.g., +30 in a blowout
        return max(base_weight * 0.60, 0.0)
    elif pm >= PLAYER_PM_THRESHOLD:       # e.g., +20 in a blowout
        return max(base_weight * 0.80, 0.0)
    else:
        return base_weight


def build_player_blowout_log(
    player_id: str,
    player_name: str,
    team_abbr: str,
    blowout_index: pd.DataFrame,
    season: str = SEASON,
    last_n: int = 30
) -> pd.DataFrame:
    """
    Fetch a player's game log and apply blowout discounts to each game.

    Returns DataFrame with one row per game:
        game_id, game_date, opponent, raw_pts, raw_reb, raw_ast, raw_min,
        blowout_tier, blowout_weight, player_pm_discount,
        weighted_pts, weighted_reb, weighted_ast, weighted_min
    """
    try:
        time.sleep(API_DELAY)
        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season
        ).get_data_frames()[0]
    except Exception as e:
        log.warning(f"Failed to fetch game log for {player_name}: {e}")
        return pd.DataFrame()

    if gl.empty:
        return pd.DataFrame()

    # Limit to last N games
    gl = gl.head(last_n).copy()

    rows = []
    for _, game in gl.iterrows():
        game_id = game["Game_ID"]
        raw_min = parse_minutes(game.get("MIN", 0))

        # Look up blowout tier for this game
        game_info = blowout_index[blowout_index["game_id"] == game_id]
        if game_info.empty:
            tier   = "FULL"
            weight = 1.0
        else:
            tier   = game_info.iloc[0]["blowout_tier"]
            weight = game_info.iloc[0]["blowout_weight"]

        # Get player's individual on-court plus/minus from box score
        # (Use PLUS_MINUS from game log as proxy — same concept)
        player_pm = game.get("PLUS_MINUS", 0) or 0
        final_weight = apply_player_pm_discount(weight, player_pm, tier)

        raw_pts = game.get("PTS", 0) or 0
        raw_reb = game.get("REB", 0) or 0
        raw_ast = game.get("AST", 0) or 0
        raw_fg3m = game.get("FG3M", 0) or 0
        raw_stl = game.get("STL", 0) or 0
        raw_blk = game.get("BLK", 0) or 0
        raw_tov = game.get("TOV", 0) or 0
        raw_fgm = game.get("FGM", 0) or 0
        raw_fga = game.get("FGA", 0) or 0
        raw_ftm = game.get("FTM", 0) or 0
        raw_fta = game.get("FTA", 0) or 0

        rows.append({
            "player_id":       player_id,
            "player_name":     player_name,
            "team_abbr":       team_abbr,
            "game_id":         game_id,
            "game_date":       game["GAME_DATE"],
            "matchup":         game.get("MATCHUP", ""),
            "wl":              game.get("WL", ""),
            "raw_min":         round(raw_min, 1),
            "raw_pts":         raw_pts,
            "raw_reb":         raw_reb,
            "raw_ast":         raw_ast,
            "raw_fg3m":        raw_fg3m,
            "raw_stl":         raw_stl,
            "raw_blk":         raw_blk,
            "raw_tov":         raw_tov,
            "raw_fgm":         raw_fgm,
            "raw_fga":         raw_fga,
            "raw_ftm":         raw_ftm,
            "raw_fta":         raw_fta,
            "player_pm":       player_pm,
            "blowout_tier":    tier,
            "blowout_weight":  weight,
            "final_weight":    round(final_weight, 3),
            # Weighted stats — what we use for projections
            "w_pts":           round(raw_pts * final_weight, 2),
            "w_reb":           round(raw_reb * final_weight, 2),
            "w_ast":           round(raw_ast * final_weight, 2),
            "w_fg3m":          round(raw_fg3m * final_weight, 2),
            "w_stl":           round(raw_stl * final_weight, 2),
            "w_blk":           round(raw_blk * final_weight, 2),
            "w_tov":           round(raw_tov * final_weight, 2),
            "w_fgm":           round(raw_fgm * final_weight, 2),
            "w_fga":           round(raw_fga * final_weight, 2),
            "w_ftm":           round(raw_ftm * final_weight, 2),
            "w_fta":           round(raw_fta * final_weight, 2),
            "w_min":           round(raw_min * final_weight, 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: Build Clean Baselines for All Players
# ---------------------------------------------------------------------------

def build_clean_baselines(
    player_ids: list,
    blowout_index: pd.DataFrame,
    season: str = SEASON,
    last_n: int = 20
) -> pd.DataFrame:
    """
    Build clean (blowout-discounted) rolling averages for a list of players.

    For each player, computes:
        - Raw rolling average (standard, no discount)
        - Clean rolling average (blowout-discounted weighted average)
        - Delta (clean - raw) — shows how much blowouts inflated/deflated stats
    """
    all_logs = []
    for pid, pname, team in player_ids:
        log.info(f"  Processing {pname} ({team})...")
        player_log = build_player_blowout_log(
            player_id=pid,
            player_name=pname,
            team_abbr=team,
            blowout_index=blowout_index,
            season=season,
            last_n=last_n
        )
        if not player_log.empty:
            all_logs.append(player_log)

    if not all_logs:
        return pd.DataFrame()

    combined = pd.concat(all_logs, ignore_index=True)

    # Compute per-player summaries
    rows = []
    for (pid, pname, team), grp in combined.groupby(["player_id", "player_name", "team_abbr"]):
        n_games = len(grp)
        if n_games == 0:
            continue

        # Raw averages (simple mean — what standard models use)
        raw_pts = grp["raw_pts"].mean()
        raw_reb = grp["raw_reb"].mean()
        raw_ast = grp["raw_ast"].mean()
        raw_fg3m = grp["raw_fg3m"].mean()
        raw_min = grp["raw_min"].mean()

        # Blowout-discounted weighted averages
        # Weighted mean: sum(stat * weight) / sum(weight)
        total_weight = grp["final_weight"].sum()
        if total_weight <= 0:
            continue

        clean_pts  = (grp["raw_pts"]  * grp["final_weight"]).sum() / total_weight
        clean_reb  = (grp["raw_reb"]  * grp["final_weight"]).sum() / total_weight
        clean_ast  = (grp["raw_ast"]  * grp["final_weight"]).sum() / total_weight
        clean_fg3m = (grp["raw_fg3m"] * grp["final_weight"]).sum() / total_weight
        clean_min  = (grp["raw_min"]  * grp["final_weight"]).sum() / total_weight

        # Count blowout games
        n_exclude = (grp["blowout_tier"] == "EXCLUDE").sum()
        n_heavy   = (grp["blowout_tier"] == "HEAVY").sum()
        n_partial = (grp["blowout_tier"] == "PARTIAL").sum()
        n_full    = (grp["blowout_tier"] == "FULL").sum()

        # Blowout inflation: how much did blowouts inflate raw stats?
        pts_delta = round(clean_pts - raw_pts, 2)

        rows.append({
            "player_id":      pid,
            "player_name":    pname,
            "team_abbr":      team,
            "games_analyzed": n_games,
            "n_full":         n_full,
            "n_partial":      n_partial,
            "n_heavy":        n_heavy,
            "n_exclude":      n_exclude,
            # Raw averages
            "raw_pts":        round(raw_pts, 2),
            "raw_reb":        round(raw_reb, 2),
            "raw_ast":        round(raw_ast, 2),
            "raw_fg3m":       round(raw_fg3m, 2),
            "raw_min":        round(raw_min, 1),
            # Clean (blowout-discounted) averages
            "clean_pts":      round(clean_pts, 2),
            "clean_reb":      round(clean_reb, 2),
            "clean_ast":      round(clean_ast, 2),
            "clean_fg3m":     round(clean_fg3m, 2),
            "clean_min":      round(clean_min, 1),
            # Delta
            "pts_delta":      pts_delta,
            "reb_delta":      round(clean_reb - raw_reb, 2),
            "ast_delta":      round(clean_ast - raw_ast, 2),
        })

    df = pd.DataFrame(rows).sort_values("pts_delta", ascending=True)
    log.info(f"Built clean baselines for {len(df)} players")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Print Reports
# ---------------------------------------------------------------------------

def print_blowout_index(df: pd.DataFrame, team: str = None, threshold: int = 25):
    """Print blowout games, optionally filtered by team."""
    if team:
        filtered = df[(df["home_team"] == team.upper()) | (df["away_team"] == team.upper())]
        title = f"BLOWOUT GAMES — {team.upper()} (2024-25)"
    else:
        filtered = df[df["final_margin"] >= threshold]
        title = f"ALL BLOWOUT GAMES ≥{threshold} pts — 2024-25"

    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")
    print(f"  {'Date':<14} {'Matchup':<22} {'Score':<12} {'Margin':>7} {'Tier':<10} {'Weight':>7}")
    print("  " + "-" * 78)
    for _, row in filtered.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        score   = f"{row['away_pts']}-{row['home_pts']}"
        print(
            f"  {row['game_date']:<14} {matchup:<22} {score:<12} "
            f"{row['final_margin']:>+7.0f} {row['blowout_tier']:<10} {row['blowout_weight']:>7.2f}"
        )
    print(f"\n  Total games shown: {len(filtered)}")
    print()


def print_player_comparison(df: pd.DataFrame, name: str):
    """Print raw vs clean stat comparison for a specific player."""
    matches = df[df["player_name"].str.lower().str.contains(name.lower(), na=False)]
    if matches.empty:
        print(f"  No player found matching '{name}'")
        return

    row = matches.iloc[0]
    print(f"\n{'='*65}")
    print(f"BLOWOUT DISCOUNT REPORT: {row['player_name']} ({row['team_abbr']})")
    print(f"{'='*65}")
    print(f"  Games Analyzed:  {row['games_analyzed']} games")
    print(f"  Game Breakdown:  {row['n_full']} FULL | {row['n_partial']} PARTIAL | "
          f"{row['n_heavy']} HEAVY | {row['n_exclude']} EXCLUDED")
    print()
    print(f"  {'Stat':<12} {'Raw Avg':>10} {'Clean Avg':>10} {'Delta':>10} {'Impact'}")
    print(f"  {'-'*55}")

    stats = [
        ("Points",   row["raw_pts"],  row["clean_pts"],  row["pts_delta"]),
        ("Rebounds", row["raw_reb"],  row["clean_reb"],  row["reb_delta"]),
        ("Assists",  row["raw_ast"],  row["clean_ast"],  row["ast_delta"]),
        ("Minutes",  row["raw_min"],  row["clean_min"],  round(row["clean_min"] - row["raw_min"], 1)),
    ]
    for stat_name, raw, clean, delta in stats:
        direction = "↑ inflated by blowouts" if delta > 0.3 else ("↓ deflated by blowouts" if delta < -0.3 else "≈ minimal impact")
        print(f"  {stat_name:<12} {raw:>10.1f} {clean:>10.1f} {delta:>+10.2f}   {direction}")

    print()
    if abs(row["pts_delta"]) > 1.0:
        if row["pts_delta"] > 0:
            print(f"  ⚠ Raw stats INFLATED by {row['pts_delta']:+.1f} pts/g due to blowout garbage time.")
            print(f"  → Use clean_pts ({row['clean_pts']:.1f}) for projections, not raw ({row['raw_pts']:.1f})")
        else:
            print(f"  ⚠ Raw stats DEFLATED by {row['pts_delta']:.1f} pts/g (played fewer garbage time minutes).")
            print(f"  → Use clean_pts ({row['clean_pts']:.1f}) for projections, not raw ({row['raw_pts']:.1f})")
    else:
        print(f"  ✓ Blowout impact is minimal — raw and clean stats are essentially equivalent.")
    print()


def print_most_inflated(df: pd.DataFrame, n: int = 10):
    """Print players whose stats are most inflated by blowout garbage time."""
    print(f"\n{'='*80}")
    print("PLAYERS MOST INFLATED BY BLOWOUT GARBAGE TIME (Raw > Clean)")
    print(f"{'='*80}")
    print(f"  {'Player':<26} {'Tm':>4} {'GP':>4} {'Raw Pts':>8} {'Clean Pts':>10} {'Delta':>8} {'Blowouts':>10}")
    print("  " + "-" * 78)
    # Most inflated = raw_pts >> clean_pts (positive delta means raw > clean)
    inflated = df.sort_values("pts_delta", ascending=False).head(n)
    for _, row in inflated.iterrows():
        blowout_games = row["n_partial"] + row["n_heavy"] + row["n_exclude"]
        print(
            f"  {row['player_name']:<26} {row['team_abbr']:>4} {row['games_analyzed']:>4} "
            f"{row['raw_pts']:>8.1f} {row['clean_pts']:>10.1f} {row['pts_delta']:>+8.2f} "
            f"{blowout_games:>10}"
        )

    print(f"\n{'='*80}")
    print("PLAYERS MOST DEFLATED (Played less garbage time than average)")
    print(f"{'='*80}")
    print(f"  {'Player':<26} {'Tm':>4} {'GP':>4} {'Raw Pts':>8} {'Clean Pts':>10} {'Delta':>8} {'Blowouts':>10}")
    print("  " + "-" * 78)
    deflated = df.sort_values("pts_delta", ascending=True).head(n)
    for _, row in deflated.iterrows():
        blowout_games = row["n_partial"] + row["n_heavy"] + row["n_exclude"]
        print(
            f"  {row['player_name']:<26} {row['team_abbr']:>4} {row['games_analyzed']:>4} "
            f"{row['raw_pts']:>8.1f} {row['clean_pts']:>10.1f} {row['pts_delta']:>+8.2f} "
            f"{blowout_games:>10}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Layer 5: Blowout Discount Model")
    parser.add_argument("--build",     action="store_true", help="Build full blowout index and clean baselines")
    parser.add_argument("--blowouts",  action="store_true", help="Show all blowout games")
    parser.add_argument("--team",      type=str,            help="Show blowout games for a specific team (e.g. DEN)")
    parser.add_argument("--player",    type=str,            help="Show raw vs clean comparison for a player")
    parser.add_argument("--inflated",  action="store_true", help="Show most stat-inflated players")
    parser.add_argument("--threshold", type=int, default=25, help="Blowout margin threshold for --blowouts")
    parser.add_argument("--last-n",    type=int, default=20, help="Number of recent games to analyze")
    parser.add_argument("--season",    type=str, default=SEASON)
    args = parser.parse_args()

    if args.build:
        log.info("Building game blowout index...")
        blowout_index = build_game_blowout_index(args.season)
        blowout_index.to_csv(GAME_INDEX_OUT, index=False)
        log.info(f"Saved → {GAME_INDEX_OUT}")

        # Build clean baselines for top players
        # Fetch a list of active players from the game log
        log.info("Fetching player list for clean baselines...")
        from nba_api.stats.endpoints import leaguedashplayerstats
        time.sleep(API_DELAY)
        player_list_df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=args.season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base"
        ).get_data_frames()[0]
        # Filter to players with meaningful minutes
        active = player_list_df[player_list_df["GP"] >= 15].copy()
        player_ids = list(zip(
            active["PLAYER_ID"].astype(str),
            active["PLAYER_NAME"],
            active["TEAM_ABBREVIATION"]
        ))
        log.info(f"Building clean baselines for {len(player_ids)} players...")
        baselines = build_clean_baselines(player_ids, blowout_index, args.season, args.last_n)
        baselines.to_csv(CLEAN_BASELINE_OUT, index=False)
        log.info(f"Saved → {CLEAN_BASELINE_OUT}")
        return

    # Load cached data
    if GAME_INDEX_OUT.exists():
        blowout_index = pd.read_csv(GAME_INDEX_OUT)
    else:
        log.info("Building game blowout index...")
        blowout_index = build_game_blowout_index(args.season)
        blowout_index.to_csv(GAME_INDEX_OUT, index=False)

    if args.blowouts:
        print_blowout_index(blowout_index, threshold=args.threshold)
        return

    if args.team:
        print_blowout_index(blowout_index, team=args.team)
        return

    if CLEAN_BASELINE_OUT.exists():
        baselines = pd.read_csv(CLEAN_BASELINE_OUT)
    else:
        log.info("Clean baselines not found — run --build first")
        baselines = pd.DataFrame()

    if args.player and not baselines.empty:
        print_player_comparison(baselines, args.player)
        return

    if args.inflated and not baselines.empty:
        print_most_inflated(baselines)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
