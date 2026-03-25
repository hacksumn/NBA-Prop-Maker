"""
luck_regression_model.py — Layer 4: Luck/Regression Model
==========================================================
Identifies which teams and players are running above or below their true
talent level due to luck, quantifies each luck component mathematically,
and regresses stats back toward true talent for forward-looking projections.

The Three Pillars of Luck:
    1. 3-Point Variance (3PT%)
       - 3PT% is the highest-variance stat in basketball (sample size ~100-300 attempts)
       - A player's "true" 3PT% is estimated from shot quality (zone + distance)
       - Deviation from expected = luck component
       - Regression weight: ~60% toward league average at 100 attempts

    2. Free Throw Rate Luck (FT%)
       - FT% is highly stable year-over-year (r ≈ 0.90)
       - BUT opponent FT% allowed is nearly random (r ≈ 0.15)
       - Teams that allow opponents to shoot 82% FT are not "bad defenders"
       - Regression: neutralize opponent FT% to league average (77%)

    3. Expected eFG% vs Actual eFG% (Shot Quality)
       - xeFG% = what a player SHOULD shoot given shot location mix
       - Uses league-average FG% by zone as the baseline
       - Actual eFG% - xeFG% = "shooting luck" component
       - Positive = running hot (expect regression down)
       - Negative = running cold (expect regression up)

Outputs:
    data/team_luck_scores.csv    — per-team luck breakdown and regression
    data/player_luck_scores.csv  — per-player luck breakdown and regression
    data/luck_adjusted_proj.csv  — forward-looking luck-neutralized projections

Usage:
    # Build all luck scores (run once per week)
    python luck_regression_model.py --build

    # Show luckiest/unluckiest teams right now
    python luck_regression_model.py --teams

    # Show luckiest/unluckiest players right now
    python luck_regression_model.py --players [--min-gp 20]

    # Show luck-adjusted projection for a specific player
    python luck_regression_model.py --player "Stephen Curry"

    # Show full luck report for a team
    python luck_regression_model.py --team GSW
"""

import os
import time
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import (
    leaguedashteamstats,
    leaguedashplayerstats,
    leaguedashteamshotlocations,
    leaguedashplayershotlocations,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TEAM_LUCK_OUT   = DATA_DIR / "team_luck_scores.csv"
PLAYER_LUCK_OUT = DATA_DIR / "player_luck_scores.csv"
LUCK_ADJ_OUT    = DATA_DIR / "luck_adjusted_proj.csv"

SEASON    = "2024-25"
API_DELAY = 0.65

# League-average FG% by shot zone (2024-25 season averages)
# These are the "expected" make rates for each zone regardless of who shoots
LEAGUE_AVG_FG_BY_ZONE = {
    "Restricted Area":          0.640,
    "In The Paint (Non-RA)":    0.410,
    "Mid-Range":                0.440,
    "Left Corner 3":            0.380,
    "Right Corner 3":           0.385,
    "Above the Break 3":        0.360,
    "Backcourt":                0.050,
}

# League-average 3PT% for regression baseline
LEAGUE_AVG_3PT_PCT = 0.362

# League-average FT% for regression baseline (used for opponent FT neutralization)
LEAGUE_AVG_FT_PCT = 0.778

# Regression weights by sample size
# At N attempts, how much do we regress toward the mean?
# Formula: regress_weight = N / (N + K) where K is the "stabilization point"
# 3PT% stabilizes at ~300 attempts; FT% at ~50 attempts
K_3PT = 300   # stabilization point for 3PT%
K_FT  = 50    # stabilization point for FT%
K_EFG = 250   # stabilization point for eFG%

# Minimum games played to include in luck analysis
MIN_GP = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: Bayesian regression toward mean
# ---------------------------------------------------------------------------

def regress_to_mean(actual: float, mean: float, n: float, k: float) -> float:
    """
    Bayesian regression toward the mean.

    actual: observed rate (e.g., 0.38 for 3PT%)
    mean:   league average (e.g., 0.362)
    n:      sample size (e.g., 200 attempts)
    k:      stabilization point (e.g., 300 for 3PT%)

    Returns the regressed estimate of the player's true rate.
    The more attempts, the less regression applied.
    """
    weight = n / (n + k)
    return weight * actual + (1 - weight) * mean


def luck_score(actual: float, expected: float, scale: float = 100.0) -> float:
    """
    Convert a raw luck delta into a normalized luck score.
    Positive = lucky (running above expected), Negative = unlucky.
    Scale of 100 means ±1 is ±1% above/below expected.
    """
    return round((actual - expected) * scale, 2)


# ---------------------------------------------------------------------------
# Step 1: Fetch and process shot location data
# ---------------------------------------------------------------------------

def flatten_shot_location_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the multi-level column structure from shot location endpoints.
    Converts (zone, stat) tuples to zone_stat column names.
    """
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            zone, stat = col
            if zone == "":
                new_cols.append(str(stat))
            else:
                zone_clean = str(zone).replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
                new_cols.append(f"{zone_clean}_{stat}")
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df


def compute_xefg(shot_loc_row: pd.Series, zones: list) -> float:
    """
    Compute expected eFG% (xeFG%) for a player/team based on shot location mix.

    xeFG% = sum over zones of (FGA_zone / FGA_total * league_avg_FG%_zone * point_value)
    where point_value = 1.5 for 3PT zones, 1.0 for 2PT zones.

    This tells us: "Given WHERE this player shoots, what eFG% should we expect?"
    """
    total_fga = sum(
        shot_loc_row.get(f"{z.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')}_FGA", 0)
        for z in zones
    )
    if total_fga <= 0:
        return np.nan

    xefg = 0.0
    three_pt_zones = {"Left Corner 3", "Right Corner 3", "Above the Break 3"}

    for zone in zones:
        zone_key = zone.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        fga_col = f"{zone_key}_FGA"
        fga = shot_loc_row.get(fga_col, 0)
        if fga <= 0:
            continue

        lg_fg_pct = LEAGUE_AVG_FG_BY_ZONE.get(zone, 0.45)
        point_val = 1.5 if zone in three_pt_zones else 1.0
        xefg += (fga / total_fga) * lg_fg_pct * point_val

    return round(xefg, 4)


def compute_actual_efg(row: pd.Series) -> float:
    """
    Compute actual eFG% from shot location data.
    eFG% = (FGM + 0.5 * FG3M) / FGA
    """
    zones = list(LEAGUE_AVG_FG_BY_ZONE.keys())
    three_pt_zones = {"Left Corner 3", "Right Corner 3", "Above the Break 3"}

    total_fga = 0
    total_fgm = 0
    total_fg3m = 0

    for zone in zones:
        zone_key = zone.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        fga = row.get(f"{zone_key}_FGA", 0)
        fgm = row.get(f"{zone_key}_FGM", 0)
        total_fga += fga
        total_fgm += fgm
        if zone in three_pt_zones:
            total_fg3m += fgm

    if total_fga <= 0:
        return np.nan
    return round((total_fgm + 0.5 * total_fg3m) / total_fga, 4)


# ---------------------------------------------------------------------------
# Step 2: Build team luck scores
# ---------------------------------------------------------------------------

def build_team_luck_scores(season: str = SEASON) -> pd.DataFrame:
    """
    Compute luck scores for all 30 teams across three dimensions:
        1. Offensive 3PT% luck (actual vs regressed)
        2. Defensive FT% luck (opponent FT% vs league average)
        3. Offensive eFG% luck (actual eFG% vs xeFG%)

    Returns DataFrame with one row per team with columns:
        team_id, team_name, gp,
        fg3_pct, fg3a_pg, fg3_regressed, fg3_luck_score,
        opp_ft_pct, opp_ft_luck_score,
        actual_efg, xefg, efg_luck_score,
        total_luck_score, luck_label
    """
    log.info("Fetching team base stats...")
    base = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    time.sleep(API_DELAY)

    log.info("Fetching team opponent stats...")
    opp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Opponent"
    ).get_data_frames()[0]
    time.sleep(API_DELAY)

    log.info("Fetching team shot location data...")
    shot_loc = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
        season=season,
        distance_range="By Zone",
        per_mode_detailed="PerGame",
        measure_type_simple="Base"
    ).get_data_frames()[0]
    time.sleep(API_DELAY)

    shot_loc = flatten_shot_location_df(shot_loc)

    # Merge base + opponent + shot location
    merged = base[["TEAM_ID", "TEAM_NAME", "GP", "FG3M", "FG3A", "FG3_PCT",
                   "FTA", "FT_PCT", "FGM", "FGA", "PTS"]].copy()
    merged = merged.merge(
        opp[["TEAM_ID", "OPP_FG3A", "OPP_FG3_PCT", "OPP_FTA", "OPP_FT_PCT"]],
        on="TEAM_ID"
    )
    merged = merged.merge(shot_loc, on="TEAM_ID")
    # After merging with shot_loc (which also has TEAM_NAME), pandas renames to TEAM_NAME_x
    if "TEAM_NAME_x" in merged.columns:
        merged = merged.rename(columns={"TEAM_NAME_x": "TEAM_NAME"})

    zones = list(LEAGUE_AVG_FG_BY_ZONE.keys())

    rows = []
    for _, row in merged.iterrows():
        # --- Luck 1: Offensive 3PT% ---
        fg3a_season = row["FG3A"] * row["GP"]  # total 3PT attempts this season
        fg3_regressed = regress_to_mean(
            actual=row["FG3_PCT"],
            mean=LEAGUE_AVG_3PT_PCT,
            n=fg3a_season,
            k=K_3PT
        )
        fg3_luck = luck_score(row["FG3_PCT"], fg3_regressed)

        # --- Luck 2: Defensive FT% (opponent FT%) ---
        # Teams have almost no control over opponent FT% — it's nearly random
        # Luck = how much better/worse than league avg the opponent shot from FT line
        opp_ft_luck = luck_score(row["OPP_FT_PCT"], LEAGUE_AVG_FT_PCT)
        # Negative = opponents shot WORSE than average (lucky for defense)
        # Positive = opponents shot BETTER than average (unlucky for defense)
        # Flip sign: if opponents shot better, that's BAD luck for the team
        opp_ft_luck_team = -opp_ft_luck  # negative = unlucky (gave up more FT pts)

        # --- Luck 3: Offensive eFG% vs xeFG% ---
        xefg = compute_xefg(row, zones)
        actual_efg = compute_actual_efg(row)
        if xefg is not None and actual_efg is not None:
            efg_luck = luck_score(actual_efg, xefg)
        else:
            efg_luck = 0.0
            xefg = actual_efg

        # --- Total luck score ---
        # Weighted: eFG% luck is most impactful, 3PT% next, FT% smallest
        total_luck = round(
            0.50 * efg_luck +
            0.35 * fg3_luck +
            0.15 * opp_ft_luck_team,
            2
        )

        # Label
        if total_luck > 3.0:
            label = "VERY LUCKY"
        elif total_luck > 1.5:
            label = "Lucky"
        elif total_luck < -3.0:
            label = "VERY UNLUCKY"
        elif total_luck < -1.5:
            label = "Unlucky"
        else:
            label = "Neutral"

        rows.append({
            "team_id":           row["TEAM_ID"],
            "team_name":         row["TEAM_NAME"],
            "gp":                row["GP"],
            "fg3_pct":           round(row["FG3_PCT"], 4),
            "fg3a_pg":           round(row["FG3A"], 1),
            "fg3_regressed":     round(fg3_regressed, 4),
            "fg3_luck_score":    fg3_luck,
            "opp_ft_pct":        round(row["OPP_FT_PCT"], 4),
            "opp_ft_luck_score": round(opp_ft_luck_team, 2),
            "actual_efg":        actual_efg,
            "xefg":              xefg,
            "efg_luck_score":    efg_luck,
            "total_luck_score":  total_luck,
            "luck_label":        label,
        })

    df = pd.DataFrame(rows).sort_values("total_luck_score", ascending=False)
    log.info(f"Built luck scores for {len(df)} teams")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: Build player luck scores
# ---------------------------------------------------------------------------

def build_player_luck_scores(season: str = SEASON, min_gp: int = MIN_GP) -> pd.DataFrame:
    """
    Compute luck scores for all qualifying players across three dimensions:
        1. 3PT% luck (actual vs regressed toward league mean)
        2. FT% luck (actual vs career-expected; FT% is skill, not luck,
           but extreme single-season deviations are partially luck)
        3. eFG% luck (actual eFG% vs xeFG% from shot location mix)

    Returns DataFrame with one row per player.
    """
    log.info("Fetching player base stats...")
    base = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    time.sleep(API_DELAY)

    log.info("Fetching player advanced stats...")
    adv = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced"
    ).get_data_frames()[0]
    time.sleep(API_DELAY)

    log.info("Fetching player shot location data...")
    for _attempt in range(3):
        try:
            time.sleep(API_DELAY * (1 + _attempt))  # back off on retries
            shot_loc = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
                season=season,
                distance_range="By Zone",
                per_mode_detailed="PerGame",
                measure_type_simple="Base"
            ).get_data_frames()[0]
            break
        except Exception as _e:
            if _attempt == 2:
                raise
            log.warning(f"Shot location fetch attempt {_attempt+1} failed: {_e} — retrying...")

    shot_loc = flatten_shot_location_df(shot_loc)

    # Merge
    base_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP",
                 "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
                 "FGM", "FGA", "FG_PCT", "PTS", "REB", "AST"]
    adv_cols  = ["PLAYER_ID", "TS_PCT", "EFG_PCT", "USG_PCT", "OFF_RATING"]

    merged = base[[c for c in base_cols if c in base.columns]].merge(
        adv[[c for c in adv_cols if c in adv.columns]], on="PLAYER_ID"
    )
    merged = merged[merged["GP"] >= min_gp].copy()
    merged = merged.merge(shot_loc, left_on="PLAYER_ID", right_on="PLAYER_ID", how="left")
    # shot_loc also has PLAYER_NAME, TEAM_ABBREVIATION — pandas renames to _x/_y
    for col in ["PLAYER_NAME", "TEAM_ABBREVIATION"]:
        if f"{col}_x" in merged.columns:
            merged = merged.rename(columns={f"{col}_x": col})

    zones = list(LEAGUE_AVG_FG_BY_ZONE.keys())

    rows = []
    for _, row in merged.iterrows():
        # --- Luck 1: 3PT% ---
        fg3a_season = row.get("FG3A", 0) * row["GP"]
        if fg3a_season >= 10:
            fg3_regressed = regress_to_mean(
                actual=row.get("FG3_PCT", LEAGUE_AVG_3PT_PCT),
                mean=LEAGUE_AVG_3PT_PCT,
                n=fg3a_season,
                k=K_3PT
            )
            fg3_luck = luck_score(row.get("FG3_PCT", LEAGUE_AVG_3PT_PCT), fg3_regressed)
        else:
            fg3_regressed = LEAGUE_AVG_3PT_PCT
            fg3_luck = 0.0

        # --- Luck 2: FT% ---
        # FT% is a skill stat, but extreme deviations in a single season
        # partially regress. We regress toward the player's own expected rate
        # (approximated as league average for simplicity; ideally use career avg)
        fta_season = row.get("FTA", 0) * row["GP"]
        if fta_season >= 20:
            ft_regressed = regress_to_mean(
                actual=row.get("FT_PCT", LEAGUE_AVG_FT_PCT),
                mean=LEAGUE_AVG_FT_PCT,
                n=fta_season,
                k=K_FT
            )
            ft_luck = luck_score(row.get("FT_PCT", LEAGUE_AVG_FT_PCT), ft_regressed)
        else:
            ft_regressed = LEAGUE_AVG_FT_PCT
            ft_luck = 0.0

        # --- Luck 3: eFG% vs xeFG% ---
        xefg = compute_xefg(row, zones)
        actual_efg = compute_actual_efg(row)
        if xefg is not None and actual_efg is not None and not np.isnan(xefg):
            efg_luck = luck_score(actual_efg, xefg)
            efg_regressed = regress_to_mean(
                actual=actual_efg,
                mean=xefg,
                n=row.get("FGA", 0) * row["GP"],
                k=K_EFG
            )
        else:
            efg_luck = 0.0
            efg_regressed = row.get("EFG_PCT", np.nan)
            xefg = row.get("EFG_PCT", np.nan)

        # --- Total luck score ---
        total_luck = round(
            0.50 * efg_luck +
            0.35 * fg3_luck +
            0.15 * ft_luck,
            2
        )

        # Luck-adjusted points per game
        # If eFG% regresses from actual to regressed, pts change proportionally
        # pts_adj = pts * (efg_regressed / actual_efg) if actual_efg > 0
        pts_pg = row.get("PTS", 0)
        if actual_efg and actual_efg > 0 and not np.isnan(actual_efg):
            pts_luck_adj = round(pts_pg * (efg_regressed / actual_efg), 1)
        else:
            pts_luck_adj = pts_pg

        # 3PT luck-adjusted points
        fg3m_pg = row.get("FG3M", 0)
        fg3_pts_luck = round((fg3_regressed - row.get("FG3_PCT", fg3_regressed)) * fg3a_season / row["GP"] * 3, 2)

        # Label
        if total_luck > 3.0:
            label = "VERY LUCKY"
        elif total_luck > 1.5:
            label = "Lucky"
        elif total_luck < -3.0:
            label = "VERY UNLUCKY"
        elif total_luck < -1.5:
            label = "Unlucky"
        else:
            label = "Neutral"

        rows.append({
            "player_id":         row["PLAYER_ID"],
            "player_name":       row["PLAYER_NAME"],
            "team_abbr":         row.get("TEAM_ABBREVIATION", ""),
            "gp":                row["GP"],
            "pts_pg":            round(pts_pg, 1),
            "fg3_pct":           round(row.get("FG3_PCT", 0), 4),
            "fg3a_pg":           round(row.get("FG3A", 0), 1),
            "fg3_regressed":     round(fg3_regressed, 4),
            "fg3_luck_score":    fg3_luck,
            "ft_pct":            round(row.get("FT_PCT", 0), 4),
            "fta_pg":            round(row.get("FTA", 0), 1),
            "ft_regressed":      round(ft_regressed, 4),
            "ft_luck_score":     ft_luck,
            "actual_efg":        actual_efg if actual_efg else np.nan,
            "xefg":              xefg if xefg else np.nan,
            "efg_luck_score":    efg_luck,
            "efg_regressed":     round(efg_regressed, 4) if efg_regressed else np.nan,
            "pts_luck_adj":      pts_luck_adj,
            "fg3_pts_luck_adj":  round(fg3_pts_luck, 2),
            "total_luck_score":  total_luck,
            "luck_label":        label,
        })

    df = pd.DataFrame(rows).sort_values("total_luck_score", ascending=False)
    log.info(f"Built luck scores for {len(df)} players")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Print reports
# ---------------------------------------------------------------------------

def print_team_luck_report(df: pd.DataFrame, n: int = 10):
    """Print the luckiest and unluckiest teams."""
    print(f"\n{'='*80}")
    print("TEAM LUCK REPORT — 2024-25 Season")
    print(f"{'='*80}")
    print(f"\n{'LUCKIEST TEAMS (expect regression DOWN)':^80}")
    print("-" * 80)
    print(f"  {'Team':<26} {'3PT%':>6} {'xeFG%':>6} {'eFG%':>6} "
          f"{'OppFT':>6} {'eFG Luck':>9} {'3PT Luck':>9} {'TOTAL':>7} {'Label':<12}")
    print("  " + "-" * 78)
    for _, row in df.head(n).iterrows():
        print(
            f"  {row['team_name']:<26} {row['fg3_pct']:>6.3f} "
            f"{row['xefg']:>6.3f} {row['actual_efg']:>6.3f} "
            f"{row['opp_ft_pct']:>6.3f} {row['efg_luck_score']:>+9.2f} "
            f"{row['fg3_luck_score']:>+9.2f} {row['total_luck_score']:>+7.2f} "
            f"  {row['luck_label']}"
        )

    print(f"\n{'UNLUCKIEST TEAMS (expect regression UP)':^80}")
    print("-" * 80)
    print(f"  {'Team':<26} {'3PT%':>6} {'xeFG%':>6} {'eFG%':>6} "
          f"{'OppFT':>6} {'eFG Luck':>9} {'3PT Luck':>9} {'TOTAL':>7} {'Label':<12}")
    print("  " + "-" * 78)
    for _, row in df.tail(n).iloc[::-1].iterrows():
        print(
            f"  {row['team_name']:<26} {row['fg3_pct']:>6.3f} "
            f"{row['xefg']:>6.3f} {row['actual_efg']:>6.3f} "
            f"{row['opp_ft_pct']:>6.3f} {row['efg_luck_score']:>+9.2f} "
            f"{row['fg3_luck_score']:>+9.2f} {row['total_luck_score']:>+7.2f} "
            f"  {row['luck_label']}"
        )
    print()


def print_player_luck_report(df: pd.DataFrame, n: int = 15):
    """Print the luckiest and unluckiest players."""
    print(f"\n{'='*90}")
    print("PLAYER LUCK REPORT — 2024-25 Season")
    print(f"{'='*90}")

    print(f"\n{'LUCKIEST PLAYERS (expect regression DOWN in scoring)':^90}")
    print("-" * 90)
    print(f"  {'Player':<26} {'Tm':>4} {'GP':>4} {'Pts':>5} {'AdjPts':>7} "
          f"{'3PT%':>6} {'xeFG':>6} {'eFG':>6} {'eFGLck':>7} {'3PTLck':>7} {'TOTAL':>7} {'Label'}")
    print("  " + "-" * 88)
    for _, row in df.head(n).iterrows():
        print(
            f"  {row['player_name']:<26} {row['team_abbr']:>4} {row['gp']:>4} "
            f"{row['pts_pg']:>5.1f} {row['pts_luck_adj']:>7.1f} "
            f"{row['fg3_pct']:>6.3f} {row['xefg']:>6.3f} {row['actual_efg']:>6.3f} "
            f"{row['efg_luck_score']:>+7.2f} {row['fg3_luck_score']:>+7.2f} "
            f"{row['total_luck_score']:>+7.2f}  {row['luck_label']}"
        )

    print(f"\n{'UNLUCKIEST PLAYERS (expect regression UP in scoring)':^90}")
    print("-" * 90)
    print(f"  {'Player':<26} {'Tm':>4} {'GP':>4} {'Pts':>5} {'AdjPts':>7} "
          f"{'3PT%':>6} {'xeFG':>6} {'eFG':>6} {'eFGLck':>7} {'3PTLck':>7} {'TOTAL':>7} {'Label'}")
    print("  " + "-" * 88)
    for _, row in df.tail(n).iloc[::-1].iterrows():
        print(
            f"  {row['player_name']:<26} {row['team_abbr']:>4} {row['gp']:>4} "
            f"{row['pts_pg']:>5.1f} {row['pts_luck_adj']:>7.1f} "
            f"{row['fg3_pct']:>6.3f} {row['xefg']:>6.3f} {row['actual_efg']:>6.3f} "
            f"{row['efg_luck_score']:>+7.2f} {row['fg3_luck_score']:>+7.2f} "
            f"{row['total_luck_score']:>+7.2f}  {row['luck_label']}"
        )
    print()


def print_player_detail(df: pd.DataFrame, name: str):
    """Print a detailed luck breakdown for a specific player."""
    matches = df[df["player_name"].str.lower().str.contains(name.lower(), na=False)]
    if matches.empty:
        print(f"  No player found matching '{name}'")
        return

    row = matches.iloc[0]
    print(f"\n{'='*60}")
    print(f"LUCK BREAKDOWN: {row['player_name']} ({row['team_abbr']})")
    print(f"{'='*60}")
    print(f"  Games Played:      {row['gp']}")
    print(f"  Actual Pts/G:      {row['pts_pg']:.1f}")
    print(f"  Luck-Adj Pts/G:    {row['pts_luck_adj']:.1f}  "
          f"({'↓' if row['pts_luck_adj'] < row['pts_pg'] else '↑'} "
          f"{abs(row['pts_luck_adj'] - row['pts_pg']):.1f} pts)")
    print()
    print(f"  ── 3-Point Luck ──────────────────────────────")
    print(f"  Actual 3PT%:       {row['fg3_pct']:.3f}")
    print(f"  Regressed 3PT%:    {row['fg3_regressed']:.3f}")
    print(f"  3PT Luck Score:    {row['fg3_luck_score']:+.2f}  "
          f"({'running HOT' if row['fg3_luck_score'] > 0 else 'running COLD'})")
    print()
    print(f"  ── Shot Quality Luck (eFG% vs xeFG%) ────────")
    print(f"  Actual eFG%:       {row['actual_efg']:.3f}")
    print(f"  Expected eFG%:     {row['xefg']:.3f}  (based on shot locations)")
    print(f"  eFG Luck Score:    {row['efg_luck_score']:+.2f}  "
          f"({'shooting above location quality' if row['efg_luck_score'] > 0 else 'shooting below location quality'})")
    print()
    print(f"  ── Free Throw Luck ───────────────────────────")
    print(f"  Actual FT%:        {row['ft_pct']:.3f}")
    print(f"  Regressed FT%:     {row['ft_regressed']:.3f}")
    print(f"  FT Luck Score:     {row['ft_luck_score']:+.2f}")
    print()
    print(f"  ── Overall ───────────────────────────────────")
    print(f"  Total Luck Score:  {row['total_luck_score']:+.2f}")
    print(f"  Verdict:           {row['luck_label']}")
    if row['total_luck_score'] > 1.5:
        print(f"  → Expect scoring to REGRESS DOWN by ~{abs(row['pts_pg'] - row['pts_luck_adj']):.1f} pts/g")
    elif row['total_luck_score'] < -1.5:
        print(f"  → Expect scoring to BOUNCE BACK UP by ~{abs(row['pts_pg'] - row['pts_luck_adj']):.1f} pts/g")
    else:
        print(f"  → Scoring is roughly at true talent level")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Layer 4: Luck/Regression Model")
    parser.add_argument("--build",   action="store_true", help="Build all luck scores")
    parser.add_argument("--teams",   action="store_true", help="Show team luck report")
    parser.add_argument("--players", action="store_true", help="Show player luck report")
    parser.add_argument("--player",  type=str, help="Show luck detail for a specific player")
    parser.add_argument("--team",    type=str, help="Show luck detail for a specific team")
    parser.add_argument("--min-gp",  type=int, default=MIN_GP, help="Minimum games played")
    parser.add_argument("--season",  type=str, default=SEASON)
    args = parser.parse_args()

    if args.build:
        log.info("Building team luck scores...")
        team_luck = build_team_luck_scores(args.season)
        team_luck.to_csv(TEAM_LUCK_OUT, index=False)
        log.info(f"Saved team luck scores → {TEAM_LUCK_OUT}")

        log.info("Building player luck scores...")
        player_luck = build_player_luck_scores(args.season, args.min_gp)
        player_luck.to_csv(PLAYER_LUCK_OUT, index=False)
        log.info(f"Saved player luck scores → {PLAYER_LUCK_OUT}")
        return

    # Load cached data
    if TEAM_LUCK_OUT.exists():
        team_luck = pd.read_csv(TEAM_LUCK_OUT)
    else:
        log.info("Building team luck scores...")
        team_luck = build_team_luck_scores(args.season)
        team_luck.to_csv(TEAM_LUCK_OUT, index=False)

    if PLAYER_LUCK_OUT.exists():
        player_luck = pd.read_csv(PLAYER_LUCK_OUT)
    else:
        log.info("Building player luck scores...")
        player_luck = build_player_luck_scores(args.season, args.min_gp)
        player_luck.to_csv(PLAYER_LUCK_OUT, index=False)

    if args.teams:
        print_team_luck_report(team_luck)
        return

    if args.players:
        print_player_luck_report(player_luck)
        return

    if args.player:
        print_player_detail(player_luck, args.player)
        return

    if args.team:
        team_row = team_luck[team_luck["team_name"].str.lower().str.contains(args.team.lower(), na=False)]
        if not team_row.empty:
            r = team_row.iloc[0]
            print(f"\nTeam: {r['team_name']}")
            print(f"  3PT%: {r['fg3_pct']:.3f} → Regressed: {r['fg3_regressed']:.3f} | Luck: {r['fg3_luck_score']:+.2f}")
            print(f"  eFG%: {r['actual_efg']:.3f} → xeFG%: {r['xefg']:.3f} | Luck: {r['efg_luck_score']:+.2f}")
            print(f"  Opp FT%: {r['opp_ft_pct']:.3f} | Luck: {r['opp_ft_luck_score']:+.2f}")
            print(f"  TOTAL LUCK: {r['total_luck_score']:+.2f} — {r['luck_label']}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
