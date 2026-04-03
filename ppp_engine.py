"""
ppp_engine.py — Layer 2: Opponent-Adjusted PPP Engine
======================================================
Builds a Ridge Regression model that simultaneously solves for every
team's true offensive and defensive ratings (points per 100 possessions),
adjusted for opponent quality and home-court advantage.

Then uses those team ratings to compute each player's Opponent-Adjusted
Points Per Possession (adj_ppp) — their true scoring efficiency stripped
of the noise of who they played.

Architecture:
    Team Level  → Ridge Regression over all game-level observations
                  Solves: Actual_PPP = League_Avg + OffRtg(team) + DefRtg(opp) + HCA
    Player Level → adj_ppp = raw_ppp - opponent_def_adjustment

Outputs:
    data/team_ratings.csv          — all 30 teams: off_rtg, def_rtg, net_rtg
    data/player_adj_ppp.csv        — per-player adj_ppp and context
    data/matchup_context.csv       — pre-computed today's matchup adjustments

Usage:
    # Build ratings from the possession master CSV
    python ppp_engine.py --build

    # Show current team ratings table
    python ppp_engine.py --ratings

    # Compute adjusted PPP for all players in today's games
    python ppp_engine.py --today

    # Compute adjusted PPP for a specific player vs opponent
    python ppp_engine.py --player "L. Shamet" --opponent BKN
"""

import os
import time
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

POSSESSION_MASTER = DATA_DIR / "pbp_possessions_master.csv"
TEAM_RATINGS_OUT  = DATA_DIR / "team_ratings.csv"
PLAYER_ADJ_PPP    = DATA_DIR / "player_adj_ppp.csv"
MATCHUP_CONTEXT   = DATA_DIR / "matchup_context.csv"
RATINGS_MODEL_OUT = MODELS_DIR / "ppp_ridge_model.pkl"

# Ridge regularization — higher alpha = more regression toward league mean
# 5.0 is a good starting point; tune with cross-validation in Layer 4
RIDGE_ALPHA = 5.0

# League average offensive rating (points per 100 possessions, 2024-25 season)
# This is the "prior" the Ridge model regresses toward
LEAGUE_AVG_ORTG = 115.0

# Minimum possessions for a player to be included in adj_ppp output
MIN_POSSESSIONS = 20

# Home court advantage in points per 100 possessions (historical NBA average)
HOME_COURT_ADV = 2.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Build game-level team aggregates from possession master
# ---------------------------------------------------------------------------

def build_game_team_aggregates(master: pd.DataFrame,
                               pbp_raw_dir: str = None) -> pd.DataFrame:
    """
    Aggregate possession master into per-team-per-game rows.
    Each row = one team's offensive performance in one game (non-garbage only).

    Team possessions are computed using the Dean Oliver formula applied at the
    team level from the player-level data:
        Team_Poss = sum(FGA) + 0.44*sum(FTA) - sum(ORB) + sum(TOV)
    This avoids double-counting that occurs when summing individual player
    possessions_used (which includes shared plays).

    Returns DataFrame with columns:
        game_id, game_date, team_tricode, opponent_tricode, is_home,
        team_pts, team_possessions, team_ortg (pts per 100 poss)
    """
    # Exclude garbage time from all calculations
    clean = master[master["garbage_time"] == 0].copy()

    # Filter to valid 3-letter team codes only
    clean = clean[
        (clean["team_tricode"].str.len() == 3) &
        (clean["opponent_tricode"].str.len() == 3)
    ].copy()

    # Aggregate using team-level possession formula
    agg = (
        clean.groupby(["game_id", "game_date", "team_tricode", "opponent_tricode"])
        .agg(
            team_pts=("pts", "sum"),
            team_fga=("fga", "sum"),
            team_fta=("fta", "sum"),
            team_orb=("orb", "sum"),
            team_tov=("tov", "sum"),
            is_home=("is_home", "max"),
        )
        .reset_index()
    )

    # Team possessions via Dean Oliver formula (team-level, not sum of players)
    # This is the correct way: FGA + 0.44*FTA - ORB + TOV
    agg["team_possessions"] = (
        agg["team_fga"] +
        0.44 * agg["team_fta"] -
        agg["team_orb"] +
        agg["team_tov"]
    ).clip(lower=1)

    # Compute offensive rating: pts per 100 possessions
    agg["team_ortg"] = (
        agg["team_pts"] / agg["team_possessions"]
    ) * 100

    # Drop rows with implausible values
    agg = agg[
        agg["team_ortg"].between(60, 160) &
        (agg["team_possessions"] >= 30)
    ].copy()

    log.info(f"Built {len(agg)} team-game observations from {agg['game_id'].nunique()} games")
    return agg


# ---------------------------------------------------------------------------
# Step 2: Ridge Regression — solve for team offensive and defensive ratings
# ---------------------------------------------------------------------------

def fit_team_ratings(game_agg: pd.DataFrame, alpha: float = RIDGE_ALPHA) -> dict:
    """
    Fit a Ridge Regression model to solve simultaneously for:
        - Each team's Offensive Rating adjustment (above/below league average)
        - Each team's Defensive Rating adjustment (above/below league average)
        - Home Court Advantage

    Model:
        team_ortg = league_avg + off_adj[team] + def_adj[opponent] + hca * is_home

    This is equivalent to the Simple Ratings System used by Basketball-Reference,
    but with Ridge regularization to prevent overfitting on small samples.

    Returns a dict with:
        'off_rtg'    : {team: adjusted offensive rating (pts/100 poss)}
        'def_rtg'    : {team: adjusted defensive rating (pts/100 poss allowed)}
        'net_rtg'    : {team: off_rtg - def_rtg}
        'hca'        : float, estimated home court advantage
        'league_avg' : float
        'model'      : fitted Ridge model
        'teams'      : sorted list of all teams
    """
    teams = sorted(set(game_agg["team_tricode"].unique()) |
                   set(game_agg["opponent_tricode"].unique()))
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    log.info(f"Fitting Ridge Regression over {len(game_agg)} observations, "
             f"{n_teams} teams, alpha={alpha}")

    # Build design matrix
    # Columns: [off_team_0, off_team_1, ..., off_team_N,
    #           def_team_0, def_team_1, ..., def_team_N,
    #           home_court]
    n_features = 2 * n_teams + 1
    X = np.zeros((len(game_agg), n_features))
    y = np.zeros(len(game_agg))

    for i, (_, row) in enumerate(game_agg.iterrows()):
        off_col = team_idx[row["team_tricode"]]
        def_col = n_teams + team_idx[row["opponent_tricode"]]
        X[i, off_col] = 1.0          # offensive team indicator
        X[i, def_col] = 1.0          # defensive team indicator
        X[i, -1] = float(row["is_home"])  # home court
        y[i] = row["team_ortg"] - LEAGUE_AVG_ORTG  # center on league average

    # Fit Ridge — fit_intercept=False because we already centered y
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y)

    coefs = model.coef_
    off_adj = {teams[i]: round(float(coefs[i]), 3) for i in range(n_teams)}
    def_adj = {teams[i]: round(float(coefs[n_teams + i]), 3) for i in range(n_teams)}
    hca = round(float(coefs[-1]), 3)

    # Absolute ratings = league_avg + adjustment
    off_rtg = {t: round(LEAGUE_AVG_ORTG + off_adj[t], 2) for t in teams}
    # For defensive rating: lower is better (fewer points allowed per 100)
    # def_adj is positive when a defense ALLOWS more points → bad defense
    def_rtg = {t: round(LEAGUE_AVG_ORTG + def_adj[t], 2) for t in teams}
    net_rtg  = {t: round(off_rtg[t] - def_rtg[t], 2) for t in teams}

    log.info(f"  Home court advantage: {hca:+.2f} pts/100 poss")
    log.info(f"  Best offense:  {max(off_rtg, key=off_rtg.get)} "
             f"({max(off_rtg.values()):.1f})")
    log.info(f"  Best defense:  {min(def_rtg, key=def_rtg.get)} "
             f"({min(def_rtg.values()):.1f})")
    log.info(f"  Best net:      {max(net_rtg, key=net_rtg.get)} "
             f"({max(net_rtg.values()):.1f})")

    return {
        "off_rtg": off_rtg,
        "def_rtg": def_rtg,
        "net_rtg": net_rtg,
        "off_adj": off_adj,
        "def_adj": def_adj,
        "hca": hca,
        "league_avg": LEAGUE_AVG_ORTG,
        "alpha": alpha,
        "n_games": len(game_agg),
        "n_teams": n_teams,
        "teams": teams,
        "model": model,
        "team_idx": team_idx,
    }


# ---------------------------------------------------------------------------
# Step 3: Player-level opponent-adjusted PPP
# ---------------------------------------------------------------------------

def compute_player_adj_ppp(master: pd.DataFrame, ratings: dict) -> pd.DataFrame:
    """
    For each player, compute their Opponent-Adjusted PPP.

    The adjustment works as follows:
        raw_ppp          = player's actual pts / possessions (non-garbage)
        league_avg_ppp   = LEAGUE_AVG_ORTG / 100
        opp_def_adj      = (def_rtg[opponent] - league_avg) / 100
                           (positive = opponent is a bad defense = easier game)
        adj_ppp          = raw_ppp - opp_def_adj

    Intuitively: if a player scored 1.40 PPP against a defense that allows
    1.20 PPP (bad defense), their adj_ppp is pulled down to ~1.20 to reflect
    that the opponent made it easier than average.

    Returns DataFrame with:
        player_id, player_name, team_tricode, games_played,
        total_possessions, raw_ppp, avg_opp_def_rtg,
        adj_ppp, adj_ppp_rank (within team)
    """
    clean = master[master["garbage_time"] == 0].copy()

    def_rtg = ratings["def_rtg"]
    league_avg_ppp = ratings["league_avg"] / 100.0

    # Map opponent defensive rating onto each player-game row
    clean["opp_def_rtg"] = clean["opponent_tricode"].map(def_rtg)
    clean["opp_def_adj_ppp"] = (clean["opp_def_rtg"].fillna(ratings["league_avg"]) -
                                  ratings["league_avg"]) / 100.0

    # Aggregate to player level (weighted by possessions)
    player_agg = (
        clean.groupby(["player_id", "player_name", "team_tricode"])
        .apply(lambda g: pd.Series({
            "games_played": g["game_id"].nunique(),
            "total_possessions": g["possessions_used"].sum(),
            "total_pts": g["pts"].sum(),
            "total_fga": g["fga"].sum(),
            "total_fgm": g["fgm"].sum(),
            "total_fg3a": g["fg3a"].sum(),
            "total_fg3m": g["fg3m"].sum(),
            "total_fta": g["fta"].sum(),
            "total_ftm": g["ftm"].sum(),
            "total_ast": g["ast"].sum(),
            "total_tov": g["tov"].sum(),
            "avg_opp_def_rtg": np.average(
                g["opp_def_rtg"].fillna(ratings["league_avg"]),
                weights=g["possessions_used"].clip(lower=0.01)
            ),
            "weighted_opp_adj": np.average(
                g["opp_def_adj_ppp"],
                weights=g["possessions_used"].clip(lower=0.01)
            ),
        }), include_groups=False)
        .reset_index()
    )

    # Filter minimum possessions
    player_agg = player_agg[player_agg["total_possessions"] >= MIN_POSSESSIONS].copy()

    # Raw PPP
    player_agg["raw_ppp"] = np.where(
        player_agg["total_possessions"] > 0,
        player_agg["total_pts"] / player_agg["total_possessions"],
        np.nan
    ).round(4)

    # Adjusted PPP: subtract the opponent quality bonus/penalty
    player_agg["adj_ppp"] = (
        player_agg["raw_ppp"] - player_agg["weighted_opp_adj"]
    ).round(4)

    # Efficiency stats
    player_agg["efg_pct"] = np.where(
        player_agg["total_fga"] > 0,
        (player_agg["total_fgm"] + 0.5 * player_agg["total_fg3m"]) /
        player_agg["total_fga"],
        np.nan
    ).round(4)

    player_agg["ts_pct"] = np.where(
        (player_agg["total_fga"] + 0.44 * player_agg["total_fta"]) > 0,
        player_agg["total_pts"] /
        (2 * (player_agg["total_fga"] + 0.44 * player_agg["total_fta"])),
        np.nan
    ).round(4)

    # Rank within team by adj_ppp
    player_agg["adj_ppp_team_rank"] = (
        player_agg.groupby("team_tricode")["adj_ppp"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Sort by adj_ppp descending
    player_agg = player_agg.sort_values("adj_ppp", ascending=False).reset_index(drop=True)
    player_agg["adj_ppp_overall_rank"] = player_agg.index + 1

    cols = [
        "player_id", "player_name", "team_tricode",
        "games_played", "total_possessions",
        "raw_ppp", "avg_opp_def_rtg", "weighted_opp_adj", "adj_ppp",
        "efg_pct", "ts_pct",
        "total_pts", "total_fga", "total_fgm", "total_fg3a", "total_fg3m",
        "total_fta", "total_ftm", "total_ast", "total_tov",
        "adj_ppp_team_rank", "adj_ppp_overall_rank"
    ]
    return player_agg[cols]


# ---------------------------------------------------------------------------
# Step 4: Matchup context — pre-compute adjustment for today's games
# ---------------------------------------------------------------------------

def compute_matchup_context(player_adj: pd.DataFrame, ratings: dict,
                             matchups: list) -> pd.DataFrame:
    """
    Given a list of today's matchups [(home_team, away_team), ...],
    compute the expected PPP adjustment for every player in those games.

    matchups: list of (home_tricode, away_tricode) tuples

    Returns DataFrame with:
        player_name, team_tricode, opponent_tricode, is_home,
        adj_ppp, opp_def_rtg, matchup_ppp_projection
    """
    rows = []
    for home, away in matchups:
        for team, opp, is_home in [(home, away, 1), (away, home, 0)]:
            players = player_adj[player_adj["team_tricode"] == team].copy()
            opp_def = ratings["def_rtg"].get(opp, ratings["league_avg"])
            opp_adj = (opp_def - ratings["league_avg"]) / 100.0
            hca_adj = (ratings["hca"] / 100.0) * is_home

            players = players.copy()
            players["opponent_tricode"] = opp
            players["is_home"] = is_home
            players["opp_def_rtg_today"] = opp_def
            # Matchup projection: adj_ppp + opponent difficulty + home court
            players["matchup_ppp_proj"] = (
                players["adj_ppp"] + opp_adj + hca_adj
            ).round(4)
            rows.append(players)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values("matchup_ppp_proj", ascending=False)
    return result[[
        "player_name", "team_tricode", "opponent_tricode", "is_home",
        "adj_ppp", "opp_def_rtg_today", "matchup_ppp_proj",
        "total_possessions", "efg_pct", "ts_pct"
    ]]


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build(possession_master_path: str = None, alpha: float = RIDGE_ALPHA):
    """
    Full build pipeline:
    1. Load possession master
    2. Aggregate to team-game level
    3. Fit Ridge Regression for team ratings
    4. Compute player-level adj_ppp
    5. Save all outputs
    """
    path = possession_master_path or str(POSSESSION_MASTER)

    if not Path(path).exists():
        log.error(f"Possession master not found: {path}")
        log.error("Run pbp_fetcher.py first to generate the data.")
        return None, None

    log.info(f"Loading possession master: {path}")
    master = pd.read_csv(path)
    log.info(f"  {len(master):,} rows, {master['game_id'].nunique()} games, "
             f"{master['player_id'].nunique()} players")

    # Step 1: Team aggregates
    game_agg = build_game_team_aggregates(master)

    # Step 2: Fit team ratings
    ratings = fit_team_ratings(game_agg, alpha=alpha)

    # Step 3: Player adj_ppp
    player_adj = compute_player_adj_ppp(master, ratings)
    log.info(f"Computed adj_ppp for {len(player_adj)} players")

    # Step 4: Save outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Team ratings CSV
    team_df = pd.DataFrame({
        "team": ratings["teams"],
        "off_rtg": [ratings["off_rtg"][t] for t in ratings["teams"]],
        "def_rtg": [ratings["def_rtg"][t] for t in ratings["teams"]],
        "net_rtg": [ratings["net_rtg"][t] for t in ratings["teams"]],
        "off_adj": [ratings["off_adj"][t] for t in ratings["teams"]],
        "def_adj": [ratings["def_adj"][t] for t in ratings["teams"]],
    }).sort_values("net_rtg", ascending=False).reset_index(drop=True)
    team_df["rank"] = team_df.index + 1
    team_df.to_csv(TEAM_RATINGS_OUT, index=False)
    log.info(f"Team ratings saved: {TEAM_RATINGS_OUT}")

    # Player adj_ppp CSV
    player_adj.to_csv(PLAYER_ADJ_PPP, index=False)
    log.info(f"Player adj_ppp saved: {PLAYER_ADJ_PPP}")

    # Save ratings metadata as JSON (for use by other modules)
    meta = {
        "league_avg": ratings["league_avg"],
        "hca": ratings["hca"],
        "alpha": ratings["alpha"],
        "n_games": ratings["n_games"],
        "n_teams": ratings["n_teams"],
        "off_rtg": ratings["off_rtg"],
        "def_rtg": ratings["def_rtg"],
        "net_rtg": ratings["net_rtg"],
        "off_adj": ratings["off_adj"],
        "def_adj": ratings["def_adj"],
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    ratings_json = DATA_DIR / "team_ratings.json"
    with open(ratings_json, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Ratings JSON saved: {ratings_json}")

    # Save model
    joblib.dump(ratings["model"], RATINGS_MODEL_OUT)
    log.info(f"Ridge model saved: {RATINGS_MODEL_OUT}")

    return ratings, player_adj


# ---------------------------------------------------------------------------
# Convenience: Load pre-built ratings from JSON
# ---------------------------------------------------------------------------

def load_ratings(ratings_json_path: str = None) -> dict:
    """Load team ratings from the saved JSON file."""
    path = ratings_json_path or str(DATA_DIR / "team_ratings.json")
    if not Path(path).exists():
        raise FileNotFoundError(f"Ratings not found: {path}. Run --build first.")
    with open(path) as f:
        return json.load(f)


def load_player_adj_ppp(path: str = None) -> pd.DataFrame:
    """Load player adj_ppp from the saved CSV."""
    p = path or str(PLAYER_ADJ_PPP)
    if not Path(p).exists():
        raise FileNotFoundError(f"Player adj_ppp not found: {p}. Run --build first.")
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer 2: Opponent-Adjusted PPP Engine"
    )
    parser.add_argument("--build", action="store_true",
                        help="Build team ratings and player adj_ppp from possession master")
    parser.add_argument("--master", default=None,
                        help="Path to possession master CSV (default: data/pbp_possessions_master.csv)")
    parser.add_argument("--alpha", type=float, default=RIDGE_ALPHA,
                        help=f"Ridge regularization alpha (default: {RIDGE_ALPHA})")
    parser.add_argument("--ratings", action="store_true",
                        help="Print current team ratings table")
    parser.add_argument("--players", action="store_true",
                        help="Print top player adj_ppp rankings")
    parser.add_argument("--player", default=None,
                        help="Show adj_ppp detail for a specific player name")
    parser.add_argument("--opponent", default=None,
                        help="Compute matchup projection vs this opponent tricode")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top players to show (default: 30)")
    args = parser.parse_args()

    if args.build:
        ratings, player_adj = build(args.master, args.alpha)
        if ratings is None:
            return

        print("\n" + "=" * 60)
        print("TEAM RATINGS (Opponent-Adjusted, pts/100 possessions)")
        print("=" * 60)
        team_df = pd.read_csv(TEAM_RATINGS_OUT)
        print(team_df[["rank", "team", "off_rtg", "def_rtg", "net_rtg"]].to_string(index=False))

        print("\n" + "=" * 60)
        print(f"TOP {min(args.top, len(player_adj))} PLAYERS BY ADJ_PPP")
        print("=" * 60)
        cols = ["adj_ppp_overall_rank", "player_name", "team_tricode",
                "total_possessions", "raw_ppp", "avg_opp_def_rtg", "adj_ppp", "efg_pct"]
        print(player_adj[cols].head(args.top).to_string(index=False))
        return

    if args.ratings:
        try:
            team_df = pd.read_csv(TEAM_RATINGS_OUT)
            print("\nTEAM RATINGS (Opponent-Adjusted, pts/100 possessions)")
            print("=" * 60)
            print(team_df[["rank", "team", "off_rtg", "def_rtg", "net_rtg"]].to_string(index=False))
        except FileNotFoundError:
            print("No ratings found. Run --build first.")
        return

    if args.players:
        try:
            player_adj = load_player_adj_ppp()
            cols = ["adj_ppp_overall_rank", "player_name", "team_tricode",
                    "total_possessions", "raw_ppp", "avg_opp_def_rtg", "adj_ppp", "efg_pct"]
            print(f"\nTOP {args.top} PLAYERS BY ADJ_PPP")
            print(player_adj[cols].head(args.top).to_string(index=False))
        except FileNotFoundError as e:
            print(e)
        return

    if args.player:
        try:
            player_adj = load_player_adj_ppp()
            ratings = load_ratings()
            match = player_adj[player_adj["player_name"].str.contains(
                args.player, case=False, na=False
            )]
            if match.empty:
                print(f"Player '{args.player}' not found.")
                return
            row = match.iloc[0]
            print(f"\nPlayer: {row['player_name']} ({row['team_tricode']})")
            print(f"  Possessions:     {row['total_possessions']:.0f}")
            print(f"  Raw PPP:         {row['raw_ppp']:.4f}")
            print(f"  Avg Opp Def Rtg: {row['avg_opp_def_rtg']:.2f}")
            print(f"  Opp Adj:         {row['weighted_opp_adj']:+.4f}")
            print(f"  Adj PPP:         {row['adj_ppp']:.4f}")
            print(f"  eFG%:            {row['efg_pct']:.1%}")
            print(f"  TS%:             {row['ts_pct']:.1%}")
            print(f"  Overall Rank:    #{row['adj_ppp_overall_rank']}")

            if args.opponent:
                opp = args.opponent.upper()
                opp_def = ratings["def_rtg"].get(opp)
                if opp_def is None:
                    print(f"\n  Opponent '{opp}' not found in ratings.")
                    return
                opp_adj = (opp_def - ratings["league_avg"]) / 100.0
                proj = row["adj_ppp"] + opp_adj
                print(f"\n  vs {opp}:")
                print(f"    Opp Def Rtg:     {opp_def:.2f} pts/100")
                print(f"    Opp Difficulty:  {opp_adj:+.4f} PPP")
                print(f"    Matchup Proj:    {proj:.4f} PPP")
                print(f"    Proj pts/poss:   {proj:.4f}  (~{proj*100:.1f} pts per 100 poss)")
        except FileNotFoundError as e:
            print(e)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
