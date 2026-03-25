"""
usage_injury_model.py — Layer 3: Player Usage & Injury Model
=============================================================
Models how usage redistributes across a roster when players are out,
applies a Dean Oliver efficiency penalty to absorbing players, and outputs
adjusted stat projections for any injury scenario.

Architecture:
    1. Baseline Profiler     → Fetches each player's USG%, efficiency, minutes,
                               and per-possession stat rates for the season.
    2. On/Off Impact Engine  → Uses TeamPlayerOnOffDetails to measure how the
                               team's offensive rating changes with/without each
                               player (true impact score).
    3. Usage Redistributor   → When a player is OUT, distributes their freed
                               usage to teammates proportional to their role
                               weight, applying a Skill Curve efficiency penalty.
    4. Injury Simulator      → Accepts any lineup scenario (list of OUT players)
                               and outputs adjusted projections for every active
                               player: new USG%, new pts/reb/ast projections.

Key Concepts:
    - Usage Rate (USG%): % of team possessions used by a player while on court
    - Skill Curve: efficiency degrades as usage increases beyond a player's
      natural rate (based on Dean Oliver's research — roughly -0.5% TS% per
      +1% USG above baseline)
    - Role Weight: determines how much freed usage each teammate absorbs
      (primary ball-handlers absorb more than role players)

Outputs:
    data/player_profiles.csv      — baseline USG%, efficiency, per-poss rates
    data/team_onoff.csv           — on/off impact for every player on every team
    data/injury_projections.csv   — adjusted projections for a given scenario

Usage:
    # Build all player profiles and on/off data (run once per week)
    python usage_injury_model.py --build

    # Simulate an injury scenario
    python usage_injury_model.py --out "Giannis Antetokounmpo" --team MIL

    # Simulate multiple players out
    python usage_injury_model.py --out "LeBron James,Anthony Davis" --team LAL

    # Show full roster profile for a team
    python usage_injury_model.py --roster LAL
"""

import os
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    teamplayeronoffdetails,
    commonteamroster,
    leaguegamelog,
)
from nba_api.stats.static import teams as nba_teams_static

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

PLAYER_PROFILES_OUT  = DATA_DIR / "player_profiles.csv"
TEAM_ONOFF_OUT       = DATA_DIR / "team_onoff.csv"
INJURY_PROJ_OUT      = DATA_DIR / "injury_projections.csv"

SEASON = "2024-25"
API_DELAY = 0.65  # seconds between API calls

# Dean Oliver Skill Curve: efficiency penalty per 1% USG above baseline
# Research shows ~0.5% TS% drop per +1% USG — we express this as PPP penalty
# 0.005 PPP per 1% extra usage (conservative estimate)
SKILL_CURVE_PENALTY = 0.005

# Minimum games played to include a player in the model
MIN_GP = 10

# Role weight tiers — determines how much freed usage each player absorbs
# Higher = absorbs more usage when a teammate is out
ROLE_WEIGHTS = {
    "primary_handler":  1.00,   # USG% >= 28% (star/primary ball-handler)
    "secondary_handler": 0.70,  # USG% 22-28% (secondary scorer)
    "tertiary":          0.40,  # USG% 18-22% (third option)
    "role_player":       0.20,  # USG% 12-18% (role player)
    "bench_filler":      0.05,  # USG% < 12%  (specialist/end of bench)
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Build baseline player profiles
# ---------------------------------------------------------------------------

def build_player_profiles(season: str = SEASON) -> pd.DataFrame:
    """
    Fetch season-level advanced + base stats for all players.
    Computes per-possession rates and role classification.

    Returns DataFrame with one row per player with columns:
        player_id, player_name, team_id, team_abbr, gp, min_pg,
        usg_pct, ts_pct, efg_pct, off_rtg, net_rtg, pace, pie,
        pts_pg, reb_pg, ast_pg, stl_pg, blk_pg, tov_pg,
        pts_per_poss, reb_per_poss, ast_per_poss,
        role_tier, role_weight
    """
    log.info("Fetching advanced stats...")
    adv = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    time.sleep(API_DELAY)

    log.info("Fetching base stats...")
    base = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    time.sleep(API_DELAY)

    # Merge on player_id
    adv_cols = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
        "GP", "MIN", "USG_PCT", "TS_PCT", "EFG_PCT",
        "OFF_RATING", "NET_RATING", "PACE", "PIE", "POSS"
    ]
    base_cols = [
        "PLAYER_ID", "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FGA", "FTA", "OREB"
    ]

    merged = adv[adv_cols].merge(base[base_cols], on="PLAYER_ID")

    # Filter minimum games
    merged = merged[merged["GP"] >= MIN_GP].copy()

    # Rename for clarity
    merged.columns = [c.lower() for c in merged.columns]
    merged = merged.rename(columns={
        "player_id": "player_id",
        "player_name": "player_name",
        "team_id": "team_id",
        "team_abbreviation": "team_abbr",
        "gp": "gp",
        "min": "min_pg",
        "usg_pct": "usg_pct",
        "ts_pct": "ts_pct",
        "efg_pct": "efg_pct",
        "off_rating": "off_rtg",
        "net_rating": "net_rtg",
        "pace": "pace",
        "pie": "pie",
        "poss": "season_poss",
        "pts": "pts_pg",
        "reb": "reb_pg",
        "ast": "ast_pg",
        "stl": "stl_pg",
        "blk": "blk_pg",
        "tov": "tov_pg",
        "fga": "fga_pg",
        "fta": "fta_pg",
        "oreb": "oreb_pg",
    })

    # Per-possession rates (normalized to per-100 possessions)
    # season_poss is total possessions used this season
    # We compute per-game possessions from season_poss / gp
    merged["poss_pg"] = merged["season_poss"] / merged["gp"]
    merged["poss_pg"] = merged["poss_pg"].clip(lower=1)

    merged["pts_per_poss"]  = (merged["pts_pg"]  / merged["poss_pg"]).round(4)
    merged["reb_per_poss"]  = (merged["reb_pg"]  / merged["poss_pg"]).round(4)
    merged["ast_per_poss"]  = (merged["ast_pg"]  / merged["poss_pg"]).round(4)
    merged["stl_per_poss"]  = (merged["stl_pg"]  / merged["poss_pg"]).round(4)
    merged["blk_per_poss"]  = (merged["blk_pg"]  / merged["poss_pg"]).round(4)
    merged["tov_per_poss"]  = (merged["tov_pg"]  / merged["poss_pg"]).round(4)

    # Role classification based on USG%
    def classify_role(usg):
        if usg >= 0.28:   return "primary_handler"
        elif usg >= 0.22: return "secondary_handler"
        elif usg >= 0.18: return "tertiary"
        elif usg >= 0.12: return "role_player"
        else:             return "bench_filler"

    merged["role_tier"]   = merged["usg_pct"].apply(classify_role)
    merged["role_weight"] = merged["role_tier"].map(ROLE_WEIGHTS)

    log.info(f"Built profiles for {len(merged)} players across {merged['team_abbr'].nunique()} teams")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Build team on/off impact data
# ---------------------------------------------------------------------------

def build_team_onoff(season: str = SEASON) -> pd.DataFrame:
    """
    For every team, fetch on/off data showing how the team performs
    with and without each player on the court.

    Returns DataFrame with columns:
        team_id, team_abbr, player_id, player_name,
        on_ortg, off_ortg, ortg_impact,   (team offensive rating with/without)
        on_net, off_net, net_impact,       (team net rating with/without)
        on_pace, off_pace,                 (pace with/without)
        on_min, off_min, on_poss, off_poss
    """
    all_teams = nba_teams_static.get_teams()
    rows = []
    errors = 0

    log.info(f"Fetching on/off data for {len(all_teams)} teams...")

    for i, team in enumerate(all_teams):
        team_id  = team["id"]
        team_abbr = team["abbreviation"]

        try:
            onoff = teamplayeronoffdetails.TeamPlayerOnOffDetails(
                team_id=str(team_id),
                season=season,
                measure_type_detailed_defense="Advanced"
            )
            on_df  = onoff.get_data_frames()[1]   # ON court
            off_df = onoff.get_data_frames()[2]   # OFF court

            for _, on_row in on_df.iterrows():
                pid  = on_row["VS_PLAYER_ID"]
                pname = on_row["VS_PLAYER_NAME"]

                # Find matching OFF row
                off_match = off_df[off_df["VS_PLAYER_ID"] == pid]
                if off_match.empty:
                    continue
                off_row = off_match.iloc[0]

                rows.append({
                    "team_id":      team_id,
                    "team_abbr":    team_abbr,
                    "player_id":    pid,
                    "player_name":  pname,
                    "on_ortg":      on_row["OFF_RATING"],
                    "off_ortg":     off_row["OFF_RATING"],
                    "ortg_impact":  round(on_row["OFF_RATING"] - off_row["OFF_RATING"], 2),
                    "on_net":       on_row["NET_RATING"],
                    "off_net":      off_row["NET_RATING"],
                    "net_impact":   round(on_row["NET_RATING"] - off_row["NET_RATING"], 2),
                    "on_pace":      on_row["PACE"],
                    "off_pace":     off_row["PACE"],
                    "on_min":       on_row["MIN"],
                    "off_min":      off_row["MIN"],
                    "on_poss":      on_row["POSS"],
                    "off_poss":     off_row["POSS"],
                    "on_gp":        on_row["GP"],
                })

            log.info(f"  [{i+1:2d}/30] {team_abbr} — {len(on_df)} players")

        except Exception as e:
            log.warning(f"  [{i+1:2d}/30] {team_abbr} FAILED: {e}")
            errors += 1

        time.sleep(API_DELAY)

    df = pd.DataFrame(rows)
    log.info(f"Built on/off data: {len(df)} player-team rows, {errors} team errors")
    return df


# ---------------------------------------------------------------------------
# Step 3: Usage redistribution with Skill Curve penalty
# ---------------------------------------------------------------------------

def get_role_weight(usg_pct: float) -> float:
    """Return the role weight for a given usage rate."""
    if usg_pct >= 0.28:   return ROLE_WEIGHTS["primary_handler"]
    elif usg_pct >= 0.22: return ROLE_WEIGHTS["secondary_handler"]
    elif usg_pct >= 0.18: return ROLE_WEIGHTS["tertiary"]
    elif usg_pct >= 0.12: return ROLE_WEIGHTS["role_player"]
    else:                 return ROLE_WEIGHTS["bench_filler"]


def redistribute_usage(
    roster_profiles: pd.DataFrame,
    out_player_ids: list,
    rotation_size: int = 9
) -> pd.DataFrame:
    """
    Redistribute freed usage from injured players to active teammates.

    Algorithm:
        1. Identify active players (not in out_player_ids)
        2. Sum freed usage from all OUT players
        3. Distribute freed usage proportional to each active player's role_weight
        4. Apply Skill Curve penalty: for each % of extra usage absorbed,
           reduce efficiency by SKILL_CURVE_PENALTY PPP
        5. Return updated roster with new_usg_pct and efficiency_adj columns

    Args:
        roster_profiles: DataFrame from build_player_profiles() for ONE team
        out_player_ids:  list of player_ids who are OUT
        rotation_size:   how many players are in the rotation (default 9)

    Returns:
        DataFrame with added columns:
            is_out, freed_usg, new_usg_pct, extra_usg_pct,
            skill_curve_penalty, pts_adj, reb_adj, ast_adj
    """
    df = roster_profiles.copy()

    # Mark OUT players
    df["is_out"] = df["player_id"].isin(out_player_ids)

    # Total freed usage from OUT players
    freed_usg = df.loc[df["is_out"], "usg_pct"].sum()

    if freed_usg <= 0:
        df["freed_usg"] = 0.0
        df["new_usg_pct"] = df["usg_pct"]
        df["extra_usg_pct"] = 0.0
        df["skill_curve_penalty"] = 0.0
        df["pts_adj"] = df["pts_pg"]
        df["reb_adj"] = df["reb_pg"]
        df["ast_adj"] = df["ast_pg"]
        return df

    # Active players in rotation
    active = df[~df["is_out"]].copy()

    # Limit to rotation size (top N by minutes)
    active = active.nlargest(rotation_size, "min_pg")

    # Total role weight of active players
    total_weight = active["role_weight"].sum()
    if total_weight <= 0:
        total_weight = 1.0

    # Each active player absorbs freed_usg * (their_weight / total_weight)
    active["absorbed_usg"] = freed_usg * (active["role_weight"] / total_weight)
    active["new_usg_pct"]  = (active["usg_pct"] + active["absorbed_usg"]).clip(upper=0.50)
    active["extra_usg_pct"] = active["new_usg_pct"] - active["usg_pct"]

    # Skill Curve efficiency penalty (PPP)
    # For every 1% of extra usage, efficiency drops by SKILL_CURVE_PENALTY
    active["skill_curve_penalty"] = (
        active["extra_usg_pct"] * 100 * SKILL_CURVE_PENALTY
    ).round(4)

    # Adjusted per-game stats
    # New possessions per game = new_usg_pct / usg_pct * original poss_pg
    usg_ratio = active["new_usg_pct"] / active["usg_pct"].clip(lower=0.01)

    # Points: scale by usage ratio, then apply efficiency penalty
    # pts_adj = (pts_per_poss - skill_curve_penalty) * new_poss_pg
    new_poss_pg = active["poss_pg"] * usg_ratio
    active["pts_adj"] = (
        (active["pts_per_poss"] - active["skill_curve_penalty"]) * new_poss_pg
    ).clip(lower=0).round(1)

    # Rebounds and assists scale with usage (ball-handling creates opportunities)
    active["reb_adj"] = (active["reb_per_poss"] * new_poss_pg).round(1)
    active["ast_adj"] = (active["ast_per_poss"] * new_poss_pg).round(1)

    # Merge back into full roster
    df = df.merge(
        active[["player_id", "absorbed_usg", "new_usg_pct",
                "extra_usg_pct", "skill_curve_penalty",
                "pts_adj", "reb_adj", "ast_adj"]],
        on="player_id", how="left"
    )

    # Fill OUT players and non-rotation players
    df["new_usg_pct"]        = df["new_usg_pct"].fillna(df["usg_pct"])
    df["extra_usg_pct"]      = df["extra_usg_pct"].fillna(0.0)
    df["skill_curve_penalty"] = df["skill_curve_penalty"].fillna(0.0)
    df["pts_adj"]            = df["pts_adj"].fillna(df["pts_pg"])
    df["reb_adj"]            = df["reb_adj"].fillna(df["reb_pg"])
    df["ast_adj"]            = df["ast_adj"].fillna(df["ast_pg"])

    # OUT players get zeroed out
    df.loc[df["is_out"], ["pts_adj", "reb_adj", "ast_adj",
                          "new_usg_pct", "extra_usg_pct"]] = 0.0

    df["freed_usg"] = freed_usg
    return df


# ---------------------------------------------------------------------------
# Step 4: Injury scenario simulator
# ---------------------------------------------------------------------------

def simulate_injury_scenario(
    player_profiles: pd.DataFrame,
    onoff_data: pd.DataFrame,
    team_abbr: str,
    out_player_names: list,
    rotation_size: int = 9,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Full injury simulation for a team.

    Given a list of player names who are OUT, returns adjusted projections
    for every active player on the team.

    Args:
        player_profiles: full league profiles from build_player_profiles()
        onoff_data:      full league on/off from build_team_onoff()
        team_abbr:       3-letter team code (e.g. 'LAL', 'MIL')
        out_player_names: list of player name strings (partial match OK)
        rotation_size:   rotation depth to distribute usage into
        verbose:         print summary table

    Returns:
        DataFrame with adjusted projections for the active roster
    """
    # Get team roster
    team_roster = player_profiles[
        player_profiles["team_abbr"] == team_abbr
    ].copy()

    if team_roster.empty:
        log.error(f"No players found for team {team_abbr}")
        return pd.DataFrame()

    # Resolve player names to IDs (fuzzy match)
    out_ids = []
    for name in out_player_names:
        name_lower = name.lower()
        matches = team_roster[
            team_roster["player_name"].str.lower().str.contains(name_lower, na=False)
        ]
        if matches.empty:
            # Try league-wide
            matches = player_profiles[
                player_profiles["player_name"].str.lower().str.contains(name_lower, na=False)
            ]
        if not matches.empty:
            pid = matches.iloc[0]["player_id"]
            pname = matches.iloc[0]["player_name"]
            out_ids.append(pid)
            log.info(f"  Matched '{name}' → {pname} (ID: {pid})")
        else:
            log.warning(f"  Could not find player matching '{name}'")

    if not out_ids:
        log.error("No valid OUT players found. Check player names.")
        return pd.DataFrame()

    # Run redistribution
    result = redistribute_usage(team_roster, out_ids, rotation_size)

    # Merge on/off impact data
    team_onoff = onoff_data[onoff_data["team_abbr"] == team_abbr]
    if not team_onoff.empty:
        result = result.merge(
            team_onoff[["player_id", "ortg_impact", "net_impact"]],
            on="player_id", how="left"
        )
    else:
        result["ortg_impact"] = np.nan
        result["net_impact"]  = np.nan

    # Sort: OUT players first, then active by pts_adj descending
    result["sort_key"] = result.apply(
        lambda r: -99 if r["is_out"] else -r["pts_adj"], axis=1
    )
    result = result.sort_values("sort_key").drop(columns="sort_key")

    if verbose:
        out_names = [
            result.loc[result["player_id"] == pid, "player_name"].values[0]
            for pid in out_ids
            if pid in result["player_id"].values
        ]
        print(f"\n{'='*72}")
        print(f"INJURY SIMULATION: {team_abbr} — OUT: {', '.join(out_names)}")
        print(f"{'='*72}")
        print(f"  Freed usage: {result['freed_usg'].iloc[0]:.1%}")
        print()
        print(f"  {'Player':<24} {'Status':<7} {'USG%':>6} {'→':>2} {'NewUSG':>7} "
              f"{'Penalty':>8} {'Pts':>5} {'→':>2} {'AdjPts':>7} "
              f"{'Reb':>5} {'→':>2} {'AdjReb':>7} {'Ast':>5} {'→':>2} {'AdjAst':>7}")
        print("  " + "-" * 100)
        for _, row in result.iterrows():
            status = "OUT" if row["is_out"] else "ACTIVE"
            penalty_str = f"-{row['skill_curve_penalty']:.3f}" if row["skill_curve_penalty"] > 0 else "  —  "
            print(
                f"  {row['player_name']:<24} {status:<7} "
                f"{row['usg_pct']:>6.1%} {'→':>2} {row['new_usg_pct']:>7.1%} "
                f"{penalty_str:>8} "
                f"{row['pts_pg']:>5.1f} {'→':>2} {row['pts_adj']:>7.1f} "
                f"{row['reb_pg']:>5.1f} {'→':>2} {row['reb_adj']:>7.1f} "
                f"{row['ast_pg']:>5.1f} {'→':>2} {row['ast_adj']:>7.1f}"
            )
        print()

    return result


# ---------------------------------------------------------------------------
# Step 5: Detect likely DNP / injured players from recent game logs
# ---------------------------------------------------------------------------

def detect_recent_absences(
    player_profiles: pd.DataFrame,
    team_abbr: str,
    last_n_games: int = 5,
    season: str = SEASON
) -> list:
    """
    Scan recent game logs to detect players who have missed games.
    Returns list of player_ids who missed 2+ of the last N games.

    This is a proxy for injury detection — not a real injury report,
    but a reliable signal for who is likely to miss upcoming games.
    """
    team_players = player_profiles[
        player_profiles["team_abbr"] == team_abbr
    ]["player_id"].tolist()

    log.info(f"Scanning last {last_n_games} games for {team_abbr} absences...")

    # Fetch recent team game log (last_n_games not supported — filter manually)
    try:
        lg = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation="P",
            season_type_all_star="Regular Season",
            team_id_nullable=str(
                player_profiles.loc[
                    player_profiles["team_abbr"] == team_abbr, "team_id"
                ].iloc[0]
            )
        )
        gl = lg.get_data_frames()[0]
        # Keep only the most recent last_n_games unique game dates
        if "GAME_DATE" in gl.columns:
            recent_dates = sorted(gl["GAME_DATE"].unique())[-last_n_games:]
            gl = gl[gl["GAME_DATE"].isin(recent_dates)]
        time.sleep(API_DELAY)
    except Exception as e:
        log.warning(f"Could not fetch game log: {e}")
        return []

    # Players who appeared in at least one game
    appeared = set(gl["PLAYER_ID"].unique())
    games_played = gl.groupby("PLAYER_ID")["GAME_ID"].nunique()

    # Players on roster who appeared in fewer than last_n_games - 2 games
    likely_out = []
    for pid in team_players:
        gp = games_played.get(pid, 0)
        if gp <= max(1, last_n_games - 3):
            name = player_profiles.loc[
                player_profiles["player_id"] == pid, "player_name"
            ].values
            if len(name) > 0:
                likely_out.append({"player_id": pid, "player_name": name[0], "recent_gp": gp})

    if likely_out:
        log.info(f"  Likely absent players for {team_abbr}:")
        for p in likely_out:
            log.info(f"    {p['player_name']} — {p['recent_gp']}/{last_n_games} recent games")

    return likely_out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Layer 3: Usage & Injury Model")
    parser.add_argument("--build",   action="store_true", help="Build all profiles and on/off data")
    parser.add_argument("--out",     type=str,  help="Comma-separated player names who are OUT")
    parser.add_argument("--team",    type=str,  help="Team abbreviation (e.g. LAL, MIL)")
    parser.add_argument("--roster",  type=str,  help="Show full roster profile for a team")
    parser.add_argument("--absences",type=str,  help="Detect recent absences for a team")
    parser.add_argument("--season",  type=str,  default=SEASON)
    args = parser.parse_args()

    if args.build:
        log.info("Building player profiles...")
        profiles = build_player_profiles(args.season)
        profiles.to_csv(PLAYER_PROFILES_OUT, index=False)
        log.info(f"Saved {len(profiles)} player profiles → {PLAYER_PROFILES_OUT}")

        log.info("Building team on/off data (this takes ~3 minutes)...")
        onoff = build_team_onoff(args.season)
        onoff.to_csv(TEAM_ONOFF_OUT, index=False)
        log.info(f"Saved {len(onoff)} on/off rows → {TEAM_ONOFF_OUT}")
        return

    # Load cached data if available
    if PLAYER_PROFILES_OUT.exists():
        profiles = pd.read_csv(PLAYER_PROFILES_OUT)
        log.info(f"Loaded {len(profiles)} player profiles from cache")
    else:
        log.info("No cached profiles found — building now...")
        profiles = build_player_profiles(args.season)
        profiles.to_csv(PLAYER_PROFILES_OUT, index=False)

    if TEAM_ONOFF_OUT.exists():
        onoff = pd.read_csv(TEAM_ONOFF_OUT)
        log.info(f"Loaded {len(onoff)} on/off rows from cache")
    else:
        log.info("No cached on/off data found — building now...")
        onoff = build_team_onoff(args.season)
        onoff.to_csv(TEAM_ONOFF_OUT, index=False)

    if args.roster:
        team = args.roster.upper()
        roster = profiles[profiles["team_abbr"] == team].sort_values("usg_pct", ascending=False)
        print(f"\nROSTER PROFILE: {team}")
        print(roster[["player_name", "gp", "min_pg", "usg_pct", "ts_pct",
                       "pts_pg", "reb_pg", "ast_pg", "role_tier",
                       "off_rtg", "net_rtg"]].to_string(index=False))
        return

    if args.absences:
        team = args.absences.upper()
        detect_recent_absences(profiles, team, last_n_games=5, season=args.season)
        return

    if args.out and args.team:
        out_names = [n.strip() for n in args.out.split(",")]
        result = simulate_injury_scenario(
            profiles, onoff, args.team.upper(), out_names
        )
        if not result.empty:
            result.to_csv(INJURY_PROJ_OUT, index=False)
            log.info(f"Saved projections → {INJURY_PROJ_OUT}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
