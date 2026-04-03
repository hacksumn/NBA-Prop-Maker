"""
build_historical_dataset.py
============================
Historical Layer Builder -- retroactively applies all 5 model layers
across 3 seasons of NBA data to produce a fully enriched training dataset.

WHAT THIS SCRIPT DOES:
  1. Loads the raw game log (files/data/nba_data.csv)
  2. For each season, computes:
       Layer 5 -- Blowout discount weights per game
       Layer 1 -- PBP possession counts per game (incremental, cached)
       Layer 2 -- Opponent-adjusted PPP per player per season
       Layer 3 -- Usage rate & role tier per player per season
       Layer 4 -- Luck scores per player per season
  3. Merges all layer features onto the raw game log
  4. Saves the enriched dataset to data/training_dataset_enriched.csv

CHECKPOINTING:
  - Each layer's output is cached to data/hist_cache/
  - If the script is interrupted, re-running it will resume from the last
    completed checkpoint rather than starting over.
  - To force a full rebuild: python build_historical_dataset.py --rebuild

ESTIMATED RUNTIME:
  - Layer 5 (Blowout index):  ~5 min  (3 seasons x ~1,230 games)
  - Layer 1 (PBP):            ~4-6 hrs (469 game dates x ~10 games x API)
  - Layer 2 (PPP engine):     ~2 min  (pure math, no API)
  - Layer 3 (Usage):          ~10 min (3 seasons x API calls)
  - Layer 4 (Luck):           ~10 min (3 seasons x API calls)
  - Final merge:              ~30 sec
  Total: ~5-7 hours (mostly Layer 1 PBP -- can be run overnight)

USAGE:
  python build_historical_dataset.py              # full build with checkpoints
  python build_historical_dataset.py --rebuild    # force full rebuild
  python build_historical_dataset.py --skip-pbp  # skip slow PBP layer (faster)
  python build_historical_dataset.py --merge-only # just re-merge existing cache
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# -- Paths --------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
CACHE_DIR   = BASE_DIR / "data" / "hist_cache"
OUTPUT_DIR  = BASE_DIR / "data"
LOG_DIR     = BASE_DIR / "logs"

# Auto-detect nba_data.csv -- it may live in data/ or files/data/
def _find_raw_game_log() -> Path:
    candidates = [
        BASE_DIR / "data" / "nba_data.csv",
        BASE_DIR / "files" / "data" / "nba_data.csv",
        BASE_DIR / "nba_data.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Return the most likely path so the error message is helpful
    return candidates[0]

RAW_GAME_LOG = _find_raw_game_log()
OUTPUT_PATH  = OUTPUT_DIR / "training_dataset_enriched.csv"

SEASONS = ["2023-24", "2024-25", "2025-26"]

# -- Logging ------------------------------------------------------------------
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Force stdout to UTF-8 on Windows so Unicode chars don't crash cp1252
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"hist_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# -- Layer imports -------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR))

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"

def is_cached(name: str) -> bool:
    p = cache_path(name)
    return p.exists() and p.stat().st_size > 1000

def save_cache(df: pd.DataFrame, name: str):
    p = cache_path(name)
    df.to_csv(p, index=False)
    log.info(f"  [OK] Cached {name} -> {p.name}  ({len(df):,} rows)")

def load_cache(name: str) -> pd.DataFrame:
    return pd.read_csv(cache_path(name), low_memory=False)

def season_date_range(season: str):
    """Return (start_date, end_date) strings for a given season string."""
    year = int(season.split("-")[0])
    return f"{year}-10-01", f"{year + 1}-06-30"

def season_from_game_date(date_value) -> str:
    """Infer NBA season string from a game date."""
    dt = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(dt):
        return ""
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"

def ensure_season_column(df: pd.DataFrame, date_col: str = "game_date") -> pd.DataFrame:
    """Attach/infer season when the source data only has game_date."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "season" not in out.columns or out["season"].isna().all():
        if date_col in out.columns:
            out["season"] = out[date_col].apply(season_from_game_date)
        else:
            out["season"] = ""
    else:
        out["season"] = out["season"].fillna(out[date_col].apply(season_from_game_date) if date_col in out.columns else "")
    out["season"] = out["season"].astype(str)
    return out

def progress(current: int, total: int, label: str = "", width: int = 40):
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "#" * filled + "." * (width - filled)
    print(f"\r  [{bar}] {current}/{total}  {label:<30}", end="", flush=True)
    if current >= total:
        print()

# -----------------------------------------------------------------------------
# LAYER 5 -- BLOWOUT DISCOUNT INDEX (per season)
# -----------------------------------------------------------------------------

def _build_layer5_from_csv(raw_season: pd.DataFrame, blowout_idx: pd.DataFrame, season: str) -> pd.DataFrame:
    """Build blowout-discounted player baselines from local CSV. Zero per-player API calls."""
    TIER_WEIGHTS = {"FULL": 1.00, "PARTIAL": 0.75, "HEAVY": 0.50, "EXCLUDE": 0.00}
    PLAYER_PM_THRESHOLD = 20

    home = blowout_idx[["game_id", "game_date", "home_team", "blowout_tier", "blowout_weight"]].rename(columns={"home_team": "team"})
    away = blowout_idx[["game_id", "game_date", "away_team", "blowout_tier", "blowout_weight"]].rename(columns={"away_team": "team"})
    team_idx = pd.concat([home, away], ignore_index=True)
    team_idx["game_date"] = pd.to_datetime(team_idx["game_date"]).dt.strftime("%Y-%m-%d")

    df = raw_season.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

    merged = df.merge(team_idx[["game_date", "team", "blowout_tier", "blowout_weight"]], on=["game_date", "team"], how="left")
    df["blowout_tier"]   = merged["blowout_tier"].fillna("FULL").values
    df["blowout_weight"] = merged["blowout_weight"].fillna(1.0).values

    pm = pd.to_numeric(df.get("plus_minus", 0), errors="coerce").fillna(0)
    weight = df["blowout_weight"].copy()
    tier   = df["blowout_tier"]
    extra_discount  = (tier != "FULL") & (pm.abs() >= PLAYER_PM_THRESHOLD)
    high_pm_blowout = extra_discount & (pm.abs() >= 25)
    weight = weight.where(~high_pm_blowout, weight * 0.60)
    weight = weight.where(~(extra_discount & ~high_pm_blowout), weight * 0.80)
    df["final_weight"] = weight

    for col in ["pts","trb","ast","3p","stl","blk","tov","fg","fga","ft","fta","mp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    rows = []
    for player, grp in df.groupby("player"):
        n = len(grp)
        total_w = grp["final_weight"].sum()
        if total_w <= 0:
            continue
        def wavg(col):
            return (grp[col] * grp["final_weight"]).sum() / total_w if col in grp.columns else 0
        raw_pts = grp["pts"].mean() if "pts" in grp.columns else 0
        rows.append({
            "player_name": player, "team_abbr": grp["team"].iloc[-1], "season": season,
            "games_analyzed": n,
            "n_full":    int((grp["blowout_tier"] == "FULL").sum()),
            "n_partial": int((grp["blowout_tier"] == "PARTIAL").sum()),
            "n_heavy":   int((grp["blowout_tier"] == "HEAVY").sum()),
            "n_exclude": int((grp["blowout_tier"] == "EXCLUDE").sum()),
            "raw_pts":   round(raw_pts, 2),
            "raw_reb":   round(grp["trb"].mean() if "trb" in grp.columns else 0, 2),
            "raw_ast":   round(grp["ast"].mean() if "ast" in grp.columns else 0, 2),
            "clean_pts": round(wavg("pts"), 2),
            "clean_reb": round(wavg("trb"), 2),
            "clean_ast": round(wavg("ast"), 2),
            "clean_fg3m": round(wavg("3p"), 2),
            "clean_min": round(wavg("mp"), 1),
            "pts_delta": round(wavg("pts") - raw_pts, 2),
            "blowout_pct": round((n - int((grp["blowout_tier"] == "FULL").sum())) / n * 100, 1),
            "game_weight": round(total_w / n, 3),
        })
    return pd.DataFrame(rows)


def build_layer5_historical(raw: pd.DataFrame, rebuild: bool = False):
    """Build blowout discount weights for all 3 seasons using local CSV (no per-player API calls)."""
    cache_name = "hist_layer5_blowout"
    if not rebuild and is_cached(cache_name):
        log.info("  [OK] Layer 5 (Blowout) loaded from cache")
        return load_cache(cache_name)

    log.info("  Building Layer 5 (Blowout Discount) from local CSV...")
    from blowout_discount_model import build_game_blowout_index

    all_blowout = []
    for season in SEASONS:
        log.info(f"    Season {season}...")
        try:
            season_data = raw[raw["season"] == season].copy()
            if season_data.empty:
                log.warning(f"    No rows for {season} -- skipping")
                continue
            blowout_idx = build_game_blowout_index(season=season)
            baselines   = _build_layer5_from_csv(season_data, blowout_idx, season)
            all_blowout.append(baselines)
            log.info(f"    -> {len(baselines):,} player baselines")
            time.sleep(1)
        except Exception as e:
            log.warning(f"    Season {season} failed: {e}")

    if not all_blowout:
        log.error("  [X] Layer 5 produced no data")
        return pd.DataFrame()

    df = pd.concat(all_blowout, ignore_index=True)
    save_cache(df, cache_name)
    return df


# -----------------------------------------------------------------------------
# LAYER 1 -- PBP POSSESSION DATA (per game date, heavily cached)
# -----------------------------------------------------------------------------

def build_layer1_historical(game_dates: list, rebuild: bool = False):
    """
    Fetch PBP possession summaries for all historical game dates.

    FIX: The NBA API ignores date_from_nullable/date_to_nullable params and
    always returns the full season. We fetch each season's full game log ONCE,
    build a date->game_id index, then iterate dates from that index.
    Each date's PBP result is cached individually so Ctrl+C is safe.
    """
    cache_name = "hist_layer1_pbp_master"
    if not rebuild and is_cached(cache_name):
        log.info("  \u2713 Layer 1 (PBP) loaded from cache")
        return load_cache(cache_name)

    log.info(f"  Building Layer 1 (PBP Possessions) for {len(game_dates)} game dates...")
    log.info("  NOTE: This is the slow step (~4-6 hrs). Safe to Ctrl+C and resume later.")

    from pbp_fetcher import (
        fetch_game,
        PBP_POSS_DIR,
        REQUEST_DELAY,
        LONG_COOLDOWN_EVERY,
        LONG_COOLDOWN_SECONDS,
        FAILURE_STREAK_COOLDOWN,
        FAILURE_STREAK_SLEEP,
    )
    from nba_api.stats.endpoints import LeagueGameLog

    # -- Step 1: Build a full date->game_id index by fetching each season once --
    date_to_games = {}  # {date_str: [{game_id, home_team, away_team}, ...]}
    for season in SEASONS:
        season_cache = CACHE_DIR / f"season_game_index_{season}.json"
        if season_cache.exists():
            import json as _json
            with open(season_cache) as f:
                season_index = _json.load(f)
            date_to_games.update(season_index)
            log.info(f"  Season {season} game index loaded from cache ({len(season_index)} dates)")
            continue

        log.info(f"  Fetching season game index for {season}...")
        time.sleep(1.5)
        try:
            gl = LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                timeout=60,
            ).get_data_frames()[0]

            season_index = {}
            for date_str, grp in gl.groupby("GAME_DATE"):
                games_on_date = []
                for gid, ggrp in grp.groupby("GAME_ID"):
                    rows = ggrp.reset_index(drop=True)
                    # Home team is the one with "vs." in matchup
                    home_rows = rows[rows["MATCHUP"].str.contains(r"vs\.", na=False)]
                    if home_rows.empty:
                        home_rows = rows
                    home_team = home_rows.iloc[0]["TEAM_ABBREVIATION"]
                    matchup   = home_rows.iloc[0]["MATCHUP"]
                    away_team = matchup.split(" vs. ")[-1].strip() if " vs. " in matchup else ""
                    games_on_date.append({
                        "game_id":   str(gid),
                        "home_team": home_team,
                        "away_team": away_team,
                    })
                season_index[date_str] = games_on_date

            import json as _json
            with open(season_cache, "w") as f:
                _json.dump(season_index, f)
            date_to_games.update(season_index)
            log.info(f"  Season {season}: {len(season_index)} dates, {len(gl)} team-game rows")
        except Exception as e:
            log.warning(f"  Failed to fetch season index for {season}: {e}")

    # -- Step 2: Iterate dates and fetch PBP for each game ----------------------
    date_cache_dir = CACHE_DIR / "pbp_dates"
    date_cache_dir.mkdir(exist_ok=True)

    all_summaries = []
    failed_games = []
    sorted_dates = sorted(game_dates)

    consecutive_failures = 0
    processed_since_cooldown = 0
    processed_dates_since_cooldown = 0
    DATE_COOLDOWN_EVERY = 45
    DATE_COOLDOWN_SECONDS = 900

    for i, date_str in enumerate(sorted_dates):
        date_cache = date_cache_dir / f"pbp_{date_str}.csv"

        # Load from per-date cache if available
        if date_cache.exists() and date_cache.stat().st_size > 100:
            df_date = pd.read_csv(date_cache, low_memory=False)
            all_summaries.append(df_date)
            progress(i + 1, len(sorted_dates), f"{date_str} (cached)")
            continue

        games_on_date = date_to_games.get(date_str, [])
        if not games_on_date:
            progress(i + 1, len(sorted_dates), f"{date_str} (no games)")
            continue

        # Fetch PBP for each game on this date
        date_rows = []
        for game_info in games_on_date:
            game_id = str(game_info["game_id"])
            try:
                poss_path = PBP_POSS_DIR / f"{game_id}_possessions.csv"
                if rebuild or not poss_path.exists() or poss_path.stat().st_size <= 50:
                    ok = fetch_game(
                        game_id=game_id,
                        game_date=date_str,
                        home_team=game_info["home_team"],
                        away_team=game_info["away_team"],
                    )
                    if not ok:
                        failed_games.append(game_id)
                        consecutive_failures += 1
                        continue

                if not poss_path.exists() or poss_path.stat().st_size <= 50:
                    failed_games.append(game_id)
                    consecutive_failures += 1
                    continue

                summary = pd.read_csv(poss_path, low_memory=False)
                if not summary.empty:
                    date_rows.append(summary)
                    consecutive_failures = 0
                else:
                    failed_games.append(game_id)
                    consecutive_failures += 1

                processed_since_cooldown += 1
                if consecutive_failures >= FAILURE_STREAK_COOLDOWN:
                    log.warning(f"  Failure streak hit {consecutive_failures} games; cooling down for {FAILURE_STREAK_SLEEP // 60} minutes")
                    time.sleep(FAILURE_STREAK_SLEEP)
                    consecutive_failures = 0
                    processed_since_cooldown = 0
                elif processed_since_cooldown >= LONG_COOLDOWN_EVERY:
                    log.info(f"  Processed {processed_since_cooldown} games since last cooldown; pausing {LONG_COOLDOWN_SECONDS}s")
                    time.sleep(LONG_COOLDOWN_SECONDS)
                    processed_since_cooldown = 0
                else:
                    time.sleep(REQUEST_DELAY)
            except Exception:
                failed_games.append(game_id)
                consecutive_failures += 1
                processed_since_cooldown += 1
                if consecutive_failures >= FAILURE_STREAK_COOLDOWN:
                    log.warning(f"  Failure streak hit {consecutive_failures} games; cooling down for {FAILURE_STREAK_SLEEP // 60} minutes")
                    time.sleep(FAILURE_STREAK_SLEEP)
                    consecutive_failures = 0
                    processed_since_cooldown = 0
                elif processed_since_cooldown >= LONG_COOLDOWN_EVERY:
                    log.info(f"  Processed {processed_since_cooldown} games since last cooldown; pausing {LONG_COOLDOWN_SECONDS}s")
                    time.sleep(LONG_COOLDOWN_SECONDS)
                    processed_since_cooldown = 0
                else:
                    time.sleep(REQUEST_DELAY)

        if date_rows:
            df_date = pd.concat(date_rows, ignore_index=True)
            df_date.to_csv(date_cache, index=False)
            all_summaries.append(df_date)
            processed_dates_since_cooldown += 1

        if processed_dates_since_cooldown >= DATE_COOLDOWN_EVERY:
            log.info(f"  Processed {processed_dates_since_cooldown} game dates since last long cooldown; pausing {DATE_COOLDOWN_SECONDS // 60} minutes")
            time.sleep(DATE_COOLDOWN_SECONDS)
            processed_dates_since_cooldown = 0

        progress(i + 1, len(sorted_dates), f"{date_str} ({len(games_on_date)} games)")

    print()  # newline after progress bar

    if failed_games:
        log.warning(f"  \u26a0 {len(failed_games)} games failed: {failed_games[:5]}...")

    if not all_summaries:
        log.error("  \u2717 Layer 1 produced no data")
        return pd.DataFrame()

    df = pd.concat(all_summaries, ignore_index=True)
    df = ensure_season_column(df)
    save_cache(df, cache_name)
    return df


# -----------------------------------------------------------------------------
# LAYER 2 -- OPPONENT-ADJUSTED PPP (per season)
# -----------------------------------------------------------------------------

def build_layer2_historical(pbp_master: pd.DataFrame, rebuild: bool = False):
    """Run the Ridge Regression PPP engine on historical PBP data."""
    cache_name = "hist_layer2_adj_ppp"
    if not rebuild and is_cached(cache_name):
        log.info("  [OK] Layer 2 (Adj PPP) loaded from cache")
        return load_cache(cache_name)

    if pbp_master is None or pbp_master.empty:
        log.warning("  [!] Layer 2 skipped -- no PBP data available")
        return pd.DataFrame()

    pbp_master = ensure_season_column(pbp_master)

    log.info("  Building Layer 2 (Opponent-Adjusted PPP)...")
    from ppp_engine import build_game_team_aggregates, fit_team_ratings, compute_player_adj_ppp

    all_adj = []
    for season in SEASONS:
        season_pbp = pbp_master[pbp_master["season"] == season]
        if season_pbp.empty:
            continue
        try:
            game_agg = build_game_team_aggregates(season_pbp)
            if game_agg.empty:
                continue
            ratings  = fit_team_ratings(game_agg)
            adj_ppp  = compute_player_adj_ppp(season_pbp, ratings)
            adj_ppp["season"] = season
            all_adj.append(adj_ppp)
            log.info(f"    Season {season}: {len(adj_ppp):,} player rows")
        except Exception as e:
            log.warning(f"    [!] Season {season} PPP failed: {e}")

    if not all_adj:
        log.warning("  [!] Layer 2 produced no data")
        return pd.DataFrame()

    df = pd.concat(all_adj, ignore_index=True)
    save_cache(df, cache_name)
    return df


# -----------------------------------------------------------------------------
# LAYER 3 -- USAGE & ROLE (per season)
# -----------------------------------------------------------------------------

def build_layer3_historical(rebuild: bool = False):
    """Fetch usage rates and role tiers for all 3 seasons."""
    cache_name = "hist_layer3_usage"
    if not rebuild and is_cached(cache_name):
        log.info("  [OK] Layer 3 (Usage) loaded from cache")
        return load_cache(cache_name)

    log.info("  Building Layer 3 (Usage & Role Tiers)...")
    from usage_injury_model import build_player_profiles

    all_profiles = []
    for season in SEASONS:
        log.info(f"    Season {season}...")
        try:
            profiles = build_player_profiles(season=season)
            profiles["season"] = season
            all_profiles.append(profiles)
            log.info(f"    -> {len(profiles):,} player profiles")
            time.sleep(3)
        except Exception as e:
            log.warning(f"    [!] Season {season} failed: {e}")

    if not all_profiles:
        log.warning("  [!] Layer 3 produced no data")
        return pd.DataFrame()

    df = pd.concat(all_profiles, ignore_index=True)
    save_cache(df, cache_name)
    return df


# -----------------------------------------------------------------------------
# LAYER 4 -- LUCK SCORES (per season)
# -----------------------------------------------------------------------------

def build_layer4_historical(rebuild: bool = False):
    """Compute luck scores for all 3 seasons."""
    cache_name = "hist_layer4_luck"
    if not rebuild and is_cached(cache_name):
        log.info("  [OK] Layer 4 (Luck) loaded from cache")
        return load_cache(cache_name)

    log.info("  Building Layer 4 (Luck & Variance Scores)...")
    from luck_regression_model import build_player_luck_scores

    all_luck = []
    for season in SEASONS:
        log.info(f"    Season {season}...")
        try:
            luck = build_player_luck_scores(season=season)
            luck["season"] = season
            all_luck.append(luck)
            log.info(f"    -> {len(luck):,} player luck scores")
            time.sleep(3)
        except Exception as e:
            log.warning(f"    [!] Season {season} failed: {e}")

    if not all_luck:
        log.warning("  [!] Layer 4 produced no data")
        return pd.DataFrame()

    df = pd.concat(all_luck, ignore_index=True)
    save_cache(df, cache_name)
    return df


# -----------------------------------------------------------------------------
# STEP 6 -- MERGE ALL LAYERS INTO ENRICHED TRAINING DATASET
# -----------------------------------------------------------------------------

def merge_all_layers(
    raw: pd.DataFrame,
    layer5: pd.DataFrame,
    layer1: pd.DataFrame,
    layer2: pd.DataFrame,
    layer3: pd.DataFrame,
    layer4: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all 5 layer outputs onto the raw game log.
    Each layer is a LEFT JOIN -- missing layer data results in NaN columns,
    not dropped rows. The model can handle NaN via imputation.
    """
    log.info("  Merging all layers onto raw game log...")
    df = raw.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["player"] = df["player"].astype(str)
    df["season"] = df["season"].astype(str)
    if "PLAYER_ID" in df.columns and "player_id" not in df.columns:
        df["player_id"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce")

    # -- Layer 5: Blowout discount weights ---------------------------------
    # NOTE: build_clean_baselines() returns ONE row per player per season
    # (a summary, not per-game). We merge on player+season only.
    if layer5 is not None and not layer5.empty:
        l5 = layer5.copy()
        # Normalize player name column
        name_col = next((c for c in l5.columns if c.lower() in ("player_name", "player")), None)
        if name_col and name_col != "player":
            l5 = l5.rename(columns={name_col: "player"})
        l5["player"] = l5["player"].astype(str)
        l5["season"]  = l5["season"].astype(str)

        # Select relevant columns -- no game_date here, it's a season-level summary
        l5_cols = ["player", "season"]
        for col in ["clean_pts", "clean_reb", "clean_ast", "clean_fg3m", "clean_min",
                    "pts_delta", "reb_delta", "ast_delta",
                    "n_full", "n_partial", "n_heavy", "n_exclude",
                    "raw_pts", "raw_reb", "raw_ast"]:
            if col in l5.columns:
                l5_cols.append(col)

        if len(l5_cols) > 2:
            l5 = l5[l5_cols].drop_duplicates(subset=["player", "season"])
            df = df.merge(l5, on=["player", "season"], how="left")
            matched = (~df["clean_pts"].isna()).sum() if "clean_pts" in df.columns else 0
            log.info(f"    Layer 5 merged: {l5_cols} ({matched:,} matched)")
        else:
            log.warning("    [!] Layer 5 has no useful columns -- skipping")
    else:
        log.warning("    [!] Layer 5 not available -- skipping")

    # -- Layer 1: PBP possession features ----------------------------------
    if layer1 is not None and not layer1.empty:
        l1 = layer1.copy()
        name_col = next((c for c in ["player_name", "PLAYER_NAME", "player"] if c in l1.columns), None)
        if name_col and name_col != "player":
            l1 = l1.rename(columns={name_col: "player"})
        if "player" in l1.columns:
            l1["player"] = l1["player"].astype(str)
        if "player_id" in l1.columns:
            l1["player_id"] = pd.to_numeric(l1["player_id"], errors="coerce")
        date_col = next((c for c in l1.columns if "date" in c.lower()), None)
        if date_col:
            l1[date_col] = pd.to_datetime(l1[date_col])
            l1 = l1.rename(columns={date_col: "game_date"})
        if "possessions_used" in l1.columns and "possessions" not in l1.columns:
            l1 = l1.rename(columns={"possessions_used": "possessions"})
        if "ppp_raw" in l1.columns and "raw_ppp" not in l1.columns:
            l1 = l1.rename(columns={"ppp_raw": "raw_ppp"})
        if "garbage_time" in l1.columns and "garbage_time_flag" not in l1.columns:
            l1 = l1.rename(columns={"garbage_time": "garbage_time_flag"})

        if {"player", "game_date", "possessions"}.issubset(l1.columns):
            grp_cols = ["player", "game_date"]
            l1["possessions"] = pd.to_numeric(l1["possessions"], errors="coerce").fillna(0)
            if "pts" in l1.columns:
                l1["pts"] = pd.to_numeric(l1["pts"], errors="coerce").fillna(0)
            if "garbage_time_flag" in l1.columns:
                l1["garbage_time_flag"] = pd.to_numeric(l1["garbage_time_flag"], errors="coerce").fillna(0).astype(int)

            agg_rows = []
            for (player, game_date), grp in l1.groupby(grp_cols, sort=False):
                total_poss = grp["possessions"].sum()
                total_pts = grp["pts"].sum() if "pts" in grp.columns else np.nan
                if "garbage_time_flag" in grp.columns:
                    comp_mask = grp["garbage_time_flag"] == 0
                    comp_pts = grp.loc[comp_mask, "pts"].sum() if "pts" in grp.columns else np.nan
                    comp_poss = grp.loc[comp_mask, "possessions"].sum()
                    garbage_flag = int(grp["garbage_time_flag"].max())
                else:
                    comp_pts = total_pts
                    comp_poss = total_poss
                    garbage_flag = 0
                agg_rows.append({
                    "player": player,
                    "player_id": grp["player_id"].iloc[0] if "player_id" in grp.columns else np.nan,
                    "game_date": game_date,
                    "possessions": round(float(total_poss), 2),
                    "raw_ppp": round(float(total_pts / total_poss), 4) if pd.notna(total_pts) and total_poss > 0 else np.nan,
                    "garbage_time_flag": garbage_flag,
                    "competitive_pts": round(float(comp_pts), 2) if pd.notna(comp_pts) else np.nan,
                    "competitive_poss": round(float(comp_poss), 2) if pd.notna(comp_poss) else np.nan,
                })
            l1 = pd.DataFrame(agg_rows)

        l1_keys = ["player_id", "game_date"] if "player_id" in l1.columns and "player_id" in df.columns else ["player", "game_date"]
        l1_cols = l1_keys.copy()
        for col in ["possessions", "raw_ppp", "garbage_time_flag",
                    "competitive_pts", "competitive_poss"]:
            if col in l1.columns:
                l1_cols.append(col)

        if len(l1_cols) > len(l1_keys):
            l1 = l1[l1_cols].drop_duplicates(subset=l1_keys)
            df = df.merge(l1, on=l1_keys, how="left")
            log.info(f"    Layer 1 merged: {l1_cols}")
        else:
            log.warning("    [!] Layer 1 has no useful columns -- skipping")
    else:
        log.warning("    [!] Layer 1 not available -- skipping")

    # -- Layer 2: Opponent-adjusted PPP ------------------------------------
    if layer2 is not None and not layer2.empty:
        l2 = layer2.copy()
        name_col = next((c for c in ["player_name", "PLAYER_NAME", "player"] if c in l2.columns), None)
        if name_col and name_col != "player":
            l2 = l2.rename(columns={name_col: "player"})
        if "player" in l2.columns:
            l2["player"] = l2["player"].astype(str)
        if "player_id" in l2.columns:
            l2["player_id"] = pd.to_numeric(l2["player_id"], errors="coerce")
        if "raw_ppp" in l2.columns and "raw_ppp_season" not in l2.columns:
            l2 = l2.rename(columns={"raw_ppp": "raw_ppp_season"})
        if "avg_opp_def_rtg" in l2.columns and "opp_def_rtg_avg" not in l2.columns:
            l2 = l2.rename(columns={"avg_opp_def_rtg": "opp_def_rtg_avg"})
        l2 = ensure_season_column(l2)

        l2_keys = ["player_id", "season"] if "player_id" in l2.columns and "player_id" in df.columns else ["player", "season"]
        l2_cols = l2_keys.copy()
        for col in ["adj_ppp", "raw_ppp_season", "ppp_vs_avg", "opp_def_rtg_avg",
                    "total_possessions", "efg_pct", "ts_pct"]:
            if col in l2.columns:
                l2_cols.append(col)

        if len(l2_cols) > len(l2_keys):
            l2 = l2[l2_cols].drop_duplicates(subset=l2_keys)
            df = df.merge(l2, on=l2_keys, how="left")
            log.info(f"    Layer 2 merged: {l2_cols}")
        else:
            log.warning("    [!] Layer 2 has no useful columns -- skipping")
    else:
        log.warning("    [!] Layer 2 not available -- skipping")

    # -- Layer 3: Usage & role ---------------------------------------------
    if layer3 is not None and not layer3.empty:
        l3 = layer3.copy()
        # Normalize player name column to 'player'
        name_col = next((c for c in l3.columns if c in ("player_name", "PLAYER_NAME", "player")), None)
        if name_col and name_col != "player":
            l3 = l3.rename(columns={name_col: "player"})
        l3["player"] = l3["player"].astype(str)
        l3["season"] = l3["season"].astype(str)

        l3_cols = ["player", "season"]
        for col in ["usg_pct", "ts_pct", "role_tier", "pts_per_poss",
                    "ast_per_poss", "reb_per_poss", "min_pg", "poss_pg",
                    "off_rtg", "net_rtg", "role_weight"]:
            if col in l3.columns:
                l3_cols.append(col)

        if len(l3_cols) > 2:
            l3 = l3[l3_cols].drop_duplicates(subset=["player", "season"])
            df = df.merge(l3, on=["player", "season"], how="left")
            log.info(f"    Layer 3 merged: {l3_cols} ({(~df['usg_pct'].isna()).sum():,} matched)")
        else:
            log.warning("    Layer 3 has no useful columns -- skipping")
    else:
        log.warning("    Layer 3 not available -- skipping")

    # -- Layer 4: Luck scores ----------------------------------------------
    if layer4 is not None and not layer4.empty:
        l4 = layer4.copy()
        # Normalize player name column to 'player'
        name_col = next((c for c in l4.columns if c in ("player_name", "PLAYER_NAME", "player")), None)
        if name_col and name_col != "player":
            l4 = l4.rename(columns={name_col: "player"})
        l4["player"] = l4["player"].astype(str)
        l4["season"] = l4["season"].astype(str)

        l4_cols = ["player", "season"]
        for col in ["total_luck_score", "luck_label", "efg_luck_score",
                    "fg3_luck_score", "ft_luck_score", "xefg", "actual_efg",
                    "pts_luck_adj", "fg3_regressed", "ft_regressed"]:
            if col in l4.columns:
                l4_cols.append(col)

        if len(l4_cols) > 2:
            l4 = l4[l4_cols].drop_duplicates(subset=["player", "season"])
            df = df.merge(l4, on=["player", "season"], how="left")
            log.info(f"    Layer 4 merged: {l4_cols} ({(~df['total_luck_score'].isna()).sum():,} matched)")
        else:
            log.warning("    Layer 4 has no useful columns -- skipping")
    else:
        log.warning("    Layer 4 not available -- skipping")

    # -- Summary -----------------------------------------------------------
    new_cols = [c for c in df.columns if c not in raw.columns]
    log.info(f"  Merge complete: {len(df):,} rows x {len(df.columns)} cols")
    log.info(f"  New features added: {len(new_cols)} -- {new_cols}")

    # Fill rate report
    log.info("  Feature fill rates:")
    for col in new_cols:
        fill = df[col].notna().mean() * 100
        log.info(f"    {col:<35} {fill:5.1f}% filled")

    return df


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build enriched historical training dataset")
    parser.add_argument("--rebuild",    action="store_true", help="Force full rebuild, ignore all caches")
    parser.add_argument("--skip-pbp",  action="store_true", help="Skip Layer 1 PBP (fast mode, ~30 min total)")
    parser.add_argument("--merge-only", action="store_true", help="Only re-run the merge step using existing caches")
    args = parser.parse_args()

    start_time = time.time()

    print()
    print("=" * 64)
    print("  NBA PROP MODEL -- HISTORICAL LAYER BUILDER")
    print("  L5:Blowout  L1:PBP  L2:PPP  L3:Usage  L4:Luck  -> Dataset")
    print("=" * 64)
    print()

    # -- Load raw game log -------------------------------------------------
    log.info("Loading raw game log...")
    if not RAW_GAME_LOG.exists():
        log.error(f"Raw game log not found: {RAW_GAME_LOG}")
        sys.exit(1)

    raw = pd.read_csv(RAW_GAME_LOG, low_memory=False)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    game_dates = sorted(raw["game_date"].dt.strftime("%Y-%m-%d").unique().tolist())
    log.info(f"  {len(raw):,} rows | {len(game_dates)} unique game dates | {raw['season'].nunique()} seasons")

    # -- Layer 5: Blowout --------------------------------------------------
    print()
    log.info("-" * 60)
    log.info("  LAYER 5 -- Blowout Discount Index")
    log.info("-" * 60)
    if args.merge_only:
        layer5 = load_cache("hist_layer5_blowout") if is_cached("hist_layer5_blowout") else pd.DataFrame()
    else:
        layer5 = build_layer5_historical(raw, rebuild=args.rebuild)

    # -- Layer 1: PBP ------------------------------------------------------
    print()
    log.info("-" * 60)
    log.info("  LAYER 1 -- Play-by-Play Possession Data")
    log.info("-" * 60)
    if args.skip_pbp or args.merge_only:
        if is_cached("hist_layer1_pbp_master"):
            layer1 = load_cache("hist_layer1_pbp_master")
            log.info("  [OK] Layer 1 (PBP) loaded from cache")
        else:
            log.warning("  [!] Layer 1 skipped (--skip-pbp or --merge-only, no cache found)")
            layer1 = pd.DataFrame()
    else:
        layer1 = build_layer1_historical(game_dates, rebuild=args.rebuild)

    # -- Layer 2: PPP Engine -----------------------------------------------
    print()
    log.info("-" * 60)
    log.info("  LAYER 2 -- Opponent-Adjusted PPP Engine")
    log.info("-" * 60)
    if args.merge_only:
        layer2 = load_cache("hist_layer2_adj_ppp") if is_cached("hist_layer2_adj_ppp") else pd.DataFrame()
    else:
        layer2 = build_layer2_historical(layer1, rebuild=args.rebuild)

    # -- Layer 3: Usage ----------------------------------------------------
    print()
    log.info("-" * 60)
    log.info("  LAYER 3 -- Player Usage & Role Tiers")
    log.info("-" * 60)
    if args.merge_only:
        layer3 = load_cache("hist_layer3_usage") if is_cached("hist_layer3_usage") else pd.DataFrame()
    else:
        layer3 = build_layer3_historical(rebuild=args.rebuild)

    # -- Layer 4: Luck -----------------------------------------------------
    print()
    log.info("-" * 60)
    log.info("  LAYER 4 -- Luck & Variance Scores")
    log.info("-" * 60)
    if args.merge_only:
        layer4 = load_cache("hist_layer4_luck") if is_cached("hist_layer4_luck") else pd.DataFrame()
    else:
        layer4 = build_layer4_historical(rebuild=args.rebuild)

    # -- Final Merge -------------------------------------------------------
    print()
    log.info("-" * 60)
    log.info("  FINAL MERGE -- Building Enriched Training Dataset")
    log.info("-" * 60)
    enriched = merge_all_layers(raw, layer5, layer1, layer2, layer3, layer4)

    # Save
    enriched.to_csv(OUTPUT_PATH, index=False)
    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    log.info(f"  [OK] Enriched dataset saved -> {OUTPUT_PATH}")
    log.info(f"  Rows:     {len(enriched):,}")
    log.info(f"  Columns:  {len(enriched.columns)}")
    log.info(f"  Elapsed:  {elapsed/60:.1f} minutes")
    log.info(f"  Seasons:  {sorted(enriched['season'].unique().tolist())}")
    log.info(f"  Players:  {enriched['player'].nunique():,}")
    print()
    print("  NEXT STEP: Retrain with the refreshed historical dataset if you plan to use it.")
    print("  Command:   python train_advanced_models.py --input data/training_dataset_enriched.csv")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
