"""Build leakage-safe, point-in-time opponent-adjusted NBA team ratings.

The existing :mod:`ppp_engine` Ridge model is reused unchanged for the rating
fit.  This script changes only the time semantics: one fit is made per target
date using games strictly before that date.  Ratings are emitted at one row per
team-game so historical prop rows can join by ``(GAME_ID, team)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_INPUT = ROOT / "data" / "nba_data.csv"
DEFAULT_TEAM_LOG_INPUT = ROOT / "data" / "nba_team_game_logs.csv"
DEFAULT_OUTPUT = ROOT / "data" / "pointintime_team_defense_ratings.csv"
DEFAULT_SUMMARY = ROOT / "output" / "pointintime_team_defense_summary.json"

TEAM_GAME_KEY = ["GAME_ID", "team"]
RATING_COLUMNS = [
    "off_rtg_pt",
    "def_rtg_pt",
    "net_rtg_pt",
    "pace_pt",
    "opp_def_rtg_pt",
    "opp_pace_pt",
    "opp_def_rank_pt",
    "opp_def_tier_pt",
    "matchup_expected_ortg_pt",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _season_from_date(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    years = np.where(dates.dt.month >= 10, dates.dt.year, dates.dt.year - 1)
    return pd.Series(
        [f"{year}-{(year + 1) % 100:02d}" for year in years],
        index=values.index,
    )


def _repair_matchups(box_scores: pd.DataFrame) -> pd.DataFrame:
    """Repair missing opponent/home fields from reciprocal game rows and matchup."""
    out = box_scores.copy()
    out["team"] = out["team"].astype(str).str.strip().str.upper()

    game_teams = (
        out[["GAME_ID", "team"]]
        .drop_duplicates()
        .groupby("GAME_ID")["team"]
        .agg(list)
    )
    bad_games = game_teams[game_teams.map(len).ne(2)]
    if not bad_games.empty:
        raise ValueError(
            f"Expected exactly two teams per GAME_ID; found {len(bad_games)} invalid games"
        )
    opponent_lookup = {
        (game_id, teams[0]): teams[1]
        for game_id, teams in game_teams.items()
    }
    opponent_lookup.update(
        {(game_id, teams[1]): teams[0] for game_id, teams in game_teams.items()}
    )
    derived_opp = pd.Series(
        [opponent_lookup[(game_id, team)] for game_id, team in zip(out["GAME_ID"], out["team"])],
        index=out.index,
    )

    if "opp" not in out.columns:
        out["opp"] = derived_opp
    else:
        supplied = out["opp"].astype("string").str.strip().str.upper()
        disagreement = supplied.notna() & supplied.ne(derived_opp)
        if disagreement.any():
            raise ValueError(f"Opponent disagrees with GAME_ID pairing on {int(disagreement.sum())} rows")
        out["opp"] = supplied.fillna(derived_opp)

    matchup = out.get("matchup", pd.Series("", index=out.index)).astype(str)
    home_marker = matchup.str.contains(r"\bvs\.?", case=False, regex=True)
    away_marker = matchup.str.contains("@", regex=False)
    derived_home = home_marker.astype(int)
    unknown_home = ~(home_marker | away_marker)
    if unknown_home.any():
        # A reciprocal GAME_ID still identifies the home team when one row has a
        # usable matchup string.  Reject a game when neither side does.
        known = pd.DataFrame(
            {"GAME_ID": out["GAME_ID"], "team": out["team"], "home": derived_home, "unknown": unknown_home}
        )
        home_by_game = known.loc[~known["unknown"] & known["home"].eq(1)].groupby("GAME_ID")["team"].first()
        missing = out.loc[unknown_home, "GAME_ID"].map(home_by_game).isna()
        if missing.any():
            raise ValueError(f"Cannot reconstruct home team for {int(missing.sum())} rows")
        derived_home.loc[unknown_home] = (
            out.loc[unknown_home, "team"].eq(out.loc[unknown_home, "GAME_ID"].map(home_by_game))
        ).astype(int)

    if "is_home" in out.columns:
        supplied_home = pd.to_numeric(out["is_home"], errors="coerce")
        disagreement = supplied_home.notna() & supplied_home.ne(derived_home)
        if disagreement.any():
            raise ValueError(f"is_home disagrees with matchup on {int(disagreement.sum())} rows")
    out["is_home"] = derived_home.astype(int)
    return out


def build_team_game_aggregates(box_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player box scores to one offensive observation per team-game."""
    required = {"GAME_ID", "game_date", "team", "pts", "fga", "fta", "orb", "tov"}
    missing = sorted(required - set(box_scores.columns))
    if missing:
        raise ValueError(f"Box-score input is missing required columns: {missing}")

    source = box_scores.copy()
    source["game_date"] = pd.to_datetime(source["game_date"], errors="raise").dt.normalize()
    source["GAME_ID"] = pd.to_numeric(source["GAME_ID"], errors="raise").astype("int64")
    source = _repair_matchups(source)
    if "season" not in source.columns:
        source["season"] = _season_from_date(source["game_date"])
    else:
        inferred = _season_from_date(source["game_date"])
        source["season"] = source["season"].astype("string").replace("", pd.NA).fillna(inferred)

    for column in ["pts", "fga", "fta", "tov"]:
        source[column] = pd.to_numeric(source[column], errors="raise").fillna(0.0)
    source["orb"] = pd.to_numeric(source["orb"], errors="coerce")
    if source["orb"].isna().any():
        raise ValueError(
            "Player box-score ORB is incomplete; use data/nba_team_game_logs.csv "
            "instead of treating missing offensive rebounds as zero"
        )
    agg = (
        source.groupby(
            ["season", "GAME_ID", "game_date", "team", "opp", "is_home"],
            as_index=False,
            dropna=False,
        )
        .agg(
            team_pts=("pts", "sum"),
            team_fga=("fga", "sum"),
            team_fta=("fta", "sum"),
            team_orb=("orb", "sum"),
            team_tov=("tov", "sum"),
        )
        .rename(columns={"team": "team_tricode", "opp": "opponent_tricode"})
    )
    agg["team_possessions"] = (
        agg["team_fga"] + 0.44 * agg["team_fta"] - agg["team_orb"] + agg["team_tov"]
    ).clip(lower=1.0)
    agg["team_ortg"] = 100.0 * agg["team_pts"] / agg["team_possessions"]
    invalid = ~agg["team_ortg"].between(60, 180) | agg["team_possessions"].lt(30)
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} implausible team-game rating observations")
    if agg.duplicated(["GAME_ID", "team_tricode"]).any():
        raise ValueError("Team-game aggregates contain duplicate keys")
    if not agg.groupby("GAME_ID").size().eq(2).all():
        raise ValueError("Team-game aggregates do not contain exactly two rows per game")
    return agg.sort_values(["game_date", "GAME_ID", "team_tricode"]).reset_index(drop=True)


def build_team_game_aggregates_from_team_logs(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Build exact team-game observations from the NBA team game-log endpoint."""
    required = {
        "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP",
        "PTS", "FGA", "FTA", "OREB", "TOV",
    }
    missing = sorted(required - set(team_logs.columns))
    if missing:
        raise ValueError(f"Team-game logs are missing required columns: {missing}")
    source = team_logs.copy()
    source["GAME_ID"] = pd.to_numeric(source["GAME_ID"], errors="raise").astype("int64")
    source["game_date"] = pd.to_datetime(source["GAME_DATE"], errors="raise").dt.normalize()
    source["team_tricode"] = source["TEAM_ABBREVIATION"].astype(str).str.strip().str.upper()
    source["opponent_tricode"] = source["MATCHUP"].astype(str).str.split().str[-1].str.upper()
    source["is_home"] = (~source["MATCHUP"].astype(str).str.contains("@", regex=False)).astype(int)
    if "season" not in source.columns:
        source["season"] = _season_from_date(source["game_date"])
    else:
        source["season"] = source["season"].astype(str)
    for column in ["PTS", "FGA", "FTA", "OREB", "TOV"]:
        source[column] = pd.to_numeric(source[column], errors="raise")
    required_stats = ["PTS", "FGA", "FTA", "OREB", "TOV"]
    if source[required_stats].isna().any().any():
        nulls = source[required_stats].isna().sum()
        raise ValueError(
            "Team-game logs contain required stat nulls: "
            f"{nulls[nulls.gt(0)].to_dict()}"
        )
    aggregate = source.rename(
        columns={
            "PTS": "team_pts", "FGA": "team_fga", "FTA": "team_fta",
            "OREB": "team_orb", "TOV": "team_tov",
        }
    )[[
        "season", "GAME_ID", "game_date", "team_tricode", "opponent_tricode",
        "is_home", "team_pts", "team_fga", "team_fta", "team_orb", "team_tov",
    ]].copy()
    if aggregate.duplicated(["GAME_ID", "team_tricode"]).any():
        raise ValueError("Team-game logs contain duplicate team-game keys")
    if not aggregate.groupby("GAME_ID").size().eq(2).all():
        raise ValueError("Team-game logs do not contain exactly two teams per game")
    aggregate["team_possessions"] = (
        aggregate["team_fga"] + 0.44 * aggregate["team_fta"]
        - aggregate["team_orb"] + aggregate["team_tov"]
    ).clip(lower=1.0)
    aggregate["team_ortg"] = 100.0 * aggregate["team_pts"] / aggregate["team_possessions"]
    invalid = ~aggregate["team_ortg"].between(60, 180) | aggregate["team_possessions"].lt(30)
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} implausible exact team-game observations")
    return aggregate.sort_values(["game_date", "GAME_ID", "team_tricode"]).reset_index(drop=True)


def _final_season_state(game_agg: pd.DataFrame, teams: list[str]) -> dict:
    """Fit a final-season fallback state using only that completed season."""
    from ppp_engine import LEAGUE_AVG_ORTG, fit_team_ratings

    ratings = fit_team_ratings(game_agg)
    pace = game_agg.groupby("team_tricode")["team_possessions"].mean().to_dict()
    counts = game_agg.groupby("team_tricode").size().to_dict()
    league_pace = float(game_agg["team_possessions"].mean())
    return {
        "off": {team: ratings["off_rtg"].get(team, LEAGUE_AVG_ORTG) for team in teams},
        "def": {team: ratings["def_rtg"].get(team, LEAGUE_AVG_ORTG) for team in teams},
        "pace": {team: float(pace.get(team, league_pace)) for team in teams},
        "counts": {team: int(counts.get(team, 0)) for team in teams},
        "hca": float(ratings["hca"]),
        "league_avg": float(ratings["league_avg"]),
        "asof": pd.Timestamp(game_agg["game_date"].max()),
    }


def build_point_in_time_ratings(
    game_agg: pd.DataFrame,
    min_prior_games: int = 5,
) -> pd.DataFrame:
    """Return one pregame rating row per team-game using strict expanding fits."""
    from ppp_engine import LEAGUE_AVG_ORTG, fit_team_ratings, log as ppp_log

    if min_prior_games < 1:
        raise ValueError("min_prior_games must be positive")
    frame = game_agg.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise").dt.normalize()
    teams = sorted(set(frame["team_tricode"]) | set(frame["opponent_tricode"]))
    if len(teams) != 30:
        raise ValueError(f"Expected 30 NBA teams, found {len(teams)}")

    original_level = ppp_log.level
    ppp_log.setLevel(logging.WARNING)
    rows: list[dict] = []
    prior_final: dict | None = None
    try:
        for season in sorted(frame["season"].astype(str).unique()):
            season_games = frame.loc[frame["season"].astype(str).eq(season)].copy()
            for target_date in sorted(season_games["game_date"].unique()):
                target_date = pd.Timestamp(target_date)
                history = season_games.loc[season_games["game_date"].lt(target_date)].copy()
                counts = history.groupby("team_tricode").size().to_dict()
                league_pace = (
                    float(history["team_possessions"].mean())
                    if not history.empty
                    else (float(np.mean(list(prior_final["pace"].values()))) if prior_final else 100.0)
                )
                pace_l10 = {
                    team: float(
                        history.loc[history["team_tricode"].eq(team)]
                        .sort_values("game_date")["team_possessions"]
                        .tail(10)
                        .mean()
                    )
                    for team in teams
                }
                current = fit_team_ratings(history) if len(history) >= 2 else None
                current_asof = history["game_date"].max() if not history.empty else pd.NaT

                off: dict[str, float] = {}
                defense: dict[str, float] = {}
                pace: dict[str, float] = {}
                source: dict[str, str] = {}
                asof: dict[str, pd.Timestamp] = {}
                for team in teams:
                    if current is not None and counts.get(team, 0) >= min_prior_games and team in current["teams"]:
                        off[team] = float(current["off_rtg"][team])
                        defense[team] = float(current["def_rtg"][team])
                        pace[team] = pace_l10[team]
                        source[team] = "current_season_expanding"
                        asof[team] = pd.Timestamp(current_asof)
                    elif prior_final is not None:
                        off[team] = float(prior_final["off"][team])
                        defense[team] = float(prior_final["def"][team])
                        pace[team] = float(prior_final["pace"][team])
                        source[team] = "prior_season_final"
                        asof[team] = pd.Timestamp(prior_final["asof"])
                    else:
                        off[team] = float(LEAGUE_AVG_ORTG)
                        defense[team] = float(LEAGUE_AVG_ORTG)
                        pace[team] = league_pace
                        source[team] = "league_average"
                        # Constant league-average fallback consumed no historical
                        # games. Use an explicit sentinel, not a fake "yesterday".
                        asof[team] = pd.Timestamp("1900-01-01")

                # Lower defensive rating is better.  Team-code tie breaking keeps
                # the audit rank a deterministic integer permutation 1..30.
                ranked = sorted(teams, key=lambda team: (defense[team], team))
                def_rank = {team: rank for rank, team in enumerate(ranked, start=1)}

                schedule = season_games.loc[season_games["game_date"].eq(target_date)]
                hca = float(current["hca"] if current is not None else (prior_final["hca"] if prior_final else 0.0))
                league_avg = float(current["league_avg"] if current is not None else LEAGUE_AVG_ORTG)
                for game in schedule.itertuples(index=False):
                    team = game.team_tricode
                    opponent = game.opponent_tricode
                    expected = league_avg + (off[team] - league_avg) + (defense[opponent] - league_avg) + hca * int(game.is_home)
                    rating_asof = max(asof[team], asof[opponent])
                    rows.append(
                        {
                            "season": season,
                            "GAME_ID": int(game.GAME_ID),
                            "game_date": target_date,
                            "team": team,
                            "opponent": opponent,
                            "is_home": int(game.is_home),
                            "rating_asof_date": rating_asof,
                            "team_rating_source": source[team],
                            "opp_rating_source": source[opponent],
                            "n_prior_games": int(counts.get(team, 0)),
                            "opp_n_prior_games": int(counts.get(opponent, 0)),
                            "off_rtg_pt": off[team],
                            "def_rtg_pt": defense[team],
                            "net_rtg_pt": off[team] - defense[team],
                            "pace_pt": pace[team],
                            "opp_def_rtg_pt": defense[opponent],
                            "opp_pace_pt": pace[opponent],
                            "opp_def_rank_pt": def_rank[opponent],
                            "opp_def_tier_pt": int((def_rank[opponent] - 1) // 6 + 1),
                            "matchup_expected_ortg_pt": expected,
                            "rating_league_avg": league_avg,
                            "rating_hca": hca,
                        }
                    )
            prior_final = _final_season_state(season_games, teams)
    finally:
        ppp_log.setLevel(original_level)

    output = pd.DataFrame(rows).sort_values(["game_date", "GAME_ID", "team"]).reset_index(drop=True)
    validate_rating_table(output)
    return output


def validate_rating_table(frame: pd.DataFrame) -> None:
    """Fail closed on key, chronology, coverage, and numeric contract defects."""
    required = [
        *TEAM_GAME_KEY,
        "game_date",
        "team",
        "opponent",
        "rating_asof_date",
        "n_prior_games",
        "opp_n_prior_games",
        *RATING_COLUMNS,
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Rating table is missing required columns: {missing}")
    if frame.duplicated(TEAM_GAME_KEY).any():
        raise ValueError("Rating table contains duplicate team-game keys")
    if frame[required].isna().any().any():
        nulls = frame[required].isna().sum()
        raise ValueError(f"Rating table contains required nulls: {nulls[nulls.gt(0)].to_dict()}")
    game_date = pd.to_datetime(frame["game_date"], errors="raise")
    asof = pd.to_datetime(frame["rating_asof_date"], errors="raise")
    if (asof >= game_date).any():
        raise ValueError(f"Rating table contains {int((asof >= game_date).sum())} chronology violations")
    for column in ["off_rtg_pt", "def_rtg_pt", "opp_def_rtg_pt", "matchup_expected_ortg_pt"]:
        if not pd.to_numeric(frame[column], errors="raise").between(80, 150).all():
            raise ValueError(f"{column} is outside the accepted 80..150 range")
    for column in ["pace_pt", "opp_pace_pt"]:
        if not pd.to_numeric(frame[column], errors="raise").between(70, 130).all():
            raise ValueError(f"{column} is outside the accepted 70..130 range")
    if not pd.to_numeric(frame["opp_def_rank_pt"], errors="raise").between(1, 30).all():
        raise ValueError("opp_def_rank_pt must be between 1 and 30")
    if not pd.to_numeric(frame["opp_def_tier_pt"], errors="raise").between(1, 5).all():
        raise ValueError("opp_def_tier_pt must be between 1 and 5")
    consistency = frame.groupby(["game_date", "opponent"])["opp_def_rank_pt"].nunique()
    if consistency.gt(1).any():
        raise ValueError("An opponent has inconsistent defensive ranks on the same date")


def build_live_matchup_ratings(
    box_scores: pd.DataFrame,
    schedule: pd.DataFrame,
    slate_date: object,
    min_prior_games: int = 5,
) -> pd.DataFrame:
    """Score scheduled matchups from games completed strictly before ``slate_date``."""
    from ppp_engine import LEAGUE_AVG_ORTG, fit_team_ratings, log as ppp_log

    required_schedule = {"home_team", "away_team"}
    missing = sorted(required_schedule - set(schedule.columns))
    if missing:
        raise ValueError(f"Schedule is missing required columns: {missing}")
    target_date = pd.Timestamp(slate_date).normalize()
    target_season = _season_from_date(pd.Series([target_date])).iloc[0]
    use_team_logs = False
    team_logs = None
    if DEFAULT_TEAM_LOG_INPUT.exists():
        team_logs = pd.read_csv(DEFAULT_TEAM_LOG_INPUT, low_memory=False)
        known_teams = set(team_logs["TEAM_ABBREVIATION"].astype(str).str.upper())
        scheduled_teams = set(schedule["home_team"].astype(str).str.upper()) | set(
            schedule["away_team"].astype(str).str.upper()
        )
        use_team_logs = scheduled_teams.issubset(known_teams)
    if use_team_logs:
        game_agg = build_team_game_aggregates_from_team_logs(team_logs)
    else:
        game_agg = build_team_game_aggregates(box_scores)
    if (game_agg["game_date"] >= target_date).any():
        game_agg = game_agg.loc[game_agg["game_date"].lt(target_date)].copy()
    teams = sorted(set(game_agg["team_tricode"]) | set(schedule["home_team"]) | set(schedule["away_team"]))
    if len(teams) != 30:
        raise ValueError(f"Live rating adapter expected 30 NBA teams, found {len(teams)}")

    current_history = game_agg.loc[game_agg["season"].astype(str).eq(target_season)].copy()
    earlier_seasons = sorted(
        season for season in game_agg["season"].astype(str).unique() if season < target_season
    )
    prior_final = None
    if earlier_seasons:
        prior_games = game_agg.loc[game_agg["season"].astype(str).eq(earlier_seasons[-1])]
        prior_final = _final_season_state(prior_games, teams)

    original_level = ppp_log.level
    ppp_log.setLevel(logging.WARNING)
    try:
        current = fit_team_ratings(current_history) if len(current_history) >= 2 else None
    finally:
        ppp_log.setLevel(original_level)
    counts = current_history.groupby("team_tricode").size().to_dict()
    league_pace = (
        float(current_history["team_possessions"].mean())
        if not current_history.empty
        else (float(np.mean(list(prior_final["pace"].values()))) if prior_final else 100.0)
    )
    off: dict[str, float] = {}
    defense: dict[str, float] = {}
    pace: dict[str, float] = {}
    source: dict[str, str] = {}
    asof: dict[str, pd.Timestamp] = {}
    current_asof = current_history["game_date"].max() if not current_history.empty else pd.NaT
    for team in teams:
        if current is not None and counts.get(team, 0) >= min_prior_games and team in current["teams"]:
            off[team] = float(current["off_rtg"][team])
            defense[team] = float(current["def_rtg"][team])
            pace[team] = float(
                current_history.loc[current_history["team_tricode"].eq(team)]
                .sort_values("game_date")["team_possessions"].tail(10).mean()
            )
            source[team] = "current_season_expanding"
            asof[team] = pd.Timestamp(current_asof)
        elif prior_final is not None:
            off[team] = float(prior_final["off"][team])
            defense[team] = float(prior_final["def"][team])
            pace[team] = float(prior_final["pace"][team])
            source[team] = "prior_season_final"
            asof[team] = pd.Timestamp(prior_final["asof"])
        else:
            off[team] = float(LEAGUE_AVG_ORTG)
            defense[team] = float(LEAGUE_AVG_ORTG)
            pace[team] = league_pace
            source[team] = "league_average"
            asof[team] = pd.Timestamp("1900-01-01")
    ranked = sorted(teams, key=lambda team: (defense[team], team))
    ranks = {team: rank for rank, team in enumerate(ranked, start=1)}
    hca = float(current["hca"] if current is not None else (prior_final["hca"] if prior_final else 0.0))
    league_avg = float(current["league_avg"] if current is not None else LEAGUE_AVG_ORTG)
    rows = []
    for game in schedule.itertuples(index=False):
        home = str(game.home_team).upper()
        away = str(game.away_team).upper()
        for team, opponent, is_home in [(home, away, 1), (away, home, 0)]:
            rows.append({
                "game_date": target_date,
                "team": team,
                "opponent": opponent,
                "is_home": is_home,
                "rating_asof_date": max(asof[team], asof[opponent]),
                "team_rating_source": source[team],
                "opp_rating_source": source[opponent],
                "n_prior_games": int(counts.get(team, 0)),
                "opp_n_prior_games": int(counts.get(opponent, 0)),
                "off_rtg_pt": off[team],
                "def_rtg_pt": defense[team],
                "net_rtg_pt": off[team] - defense[team],
                "pace_pt": pace[team],
                "opp_def_rtg_pt": defense[opponent],
                "opp_pace_pt": pace[opponent],
                "opp_def_rank_pt": ranks[opponent],
                "opp_def_tier_pt": int((ranks[opponent] - 1) // 6 + 1),
                "matchup_expected_ortg_pt": (
                    league_avg + (off[team] - league_avg)
                    + (defense[opponent] - league_avg) + hca * is_home
                ),
                "rating_league_avg": league_avg,
                "rating_hca": hca,
            })
    output = pd.DataFrame(rows)
    if output.duplicated(["game_date", "team"]).any():
        raise ValueError("Live schedule contains a team more than once on the slate")
    if (pd.to_datetime(output["rating_asof_date"]) >= target_date).any():
        raise ValueError("Live matchup ratings contain a chronology violation")
    return output


def build_from_files(
    input_path: Path = DEFAULT_INPUT,
    team_log_input: Path = DEFAULT_TEAM_LOG_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    min_prior_games: int = 5,
) -> dict:
    box_scores = pd.read_csv(input_path, low_memory=False)
    if not team_log_input.exists():
        raise FileNotFoundError(
            f"Exact team-game source is missing: {team_log_input}. Run "
            "scripts/fetch_nba_team_game_logs.py first."
        )
    team_logs = pd.read_csv(team_log_input, low_memory=False)
    game_agg = build_team_game_aggregates_from_team_logs(team_logs)
    output = build_point_in_time_ratings(game_agg, min_prior_games=min_prior_games)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(output_path)
    chronology = (
        pd.to_datetime(output["rating_asof_date"]) >= pd.to_datetime(output["game_date"])
    )
    summary = {
        "status": "ready_for_prop_join",
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "team_log_input": str(team_log_input),
        "team_log_input_sha256": _sha256(team_log_input),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "min_prior_games": min_prior_games,
        "source_player_rows": int(len(box_scores)),
        "team_game_rows": int(len(game_agg)),
        "games": int(game_agg["GAME_ID"].nunique()),
        "output_rows": int(len(output)),
        "date_min": output["game_date"].min().date().isoformat(),
        "date_max": output["game_date"].max().date().isoformat(),
        "duplicate_keys": int(output.duplicated(TEAM_GAME_KEY).sum()),
        "required_null_cells": int(output[[*RATING_COLUMNS, "rating_asof_date"]].isna().sum().sum()),
        "chronology_violations": int(chronology.sum()),
        "rating_source_counts": output["opp_rating_source"].value_counts().to_dict(),
        "feature_columns": RATING_COLUMNS,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--team-logs", type=Path, default=DEFAULT_TEAM_LOG_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-prior-games", type=int, default=5)
    args = parser.parse_args()
    summary = build_from_files(
        args.input, args.team_logs, args.output, args.summary, args.min_prior_games
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
