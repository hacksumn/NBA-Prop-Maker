#!/usr/bin/env python3
"""
run_daily.py — NBA Prop Model Daily Orchestrator
=================================================
Runs the full 5-layer model stack every morning in sequence, then merges
all layer outputs into a single enriched projection file ready for betting.

Pipeline Execution Order:
    STEP 1   — NBA Game Logs          (nba_scraper → data/nba_data.csv)
    STEP 1.5 — Auto-grade picks       (picks_history.csv)
    STEP 1.6 — Auto-grade betslips    (betslips_history.csv)
    STEP 2   — PrizePicks Lines       (prizepicks_scraper → data/historical_lines.csv)
    STEP 3   — Layer 5: Blowout Index (blowout_discount_model → game_blowout_index.csv)
    STEP 4   — Layer 1: PBP Fetch     (pbp_fetcher → pbp_possessions_master.csv)
    STEP 5   — Layer 2: PPP Engine    (ppp_engine → team_ratings.csv, player_adj_ppp.csv)
    STEP 6   — Layer 3: Usage/Injury  (usage_injury_model → player_profiles.csv)
    STEP 7   — Layer 4: Luck Model    (luck_regression_model → player_luck_scores.csv)
    STEP 8   — Merge & Output         (→ data/player_projections_today.csv)

Usage:
    python3 run_daily.py                    # full daily run (recommended)
    python3 run_daily.py --full-refresh     # re-pull all seasons from scratch
    python3 run_daily.py --lines-only       # only update PrizePicks lines
    python3 run_daily.py --logs-only        # only update NBA game logs
    python3 run_daily.py --skip-layers      # skip layers 1-5, only fetch data
    python3 run_daily.py --layers-only      # skip data fetch, only run layers
    python3 run_daily.py --seasons 2024-25  # custom season list
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent.resolve()
DATA_DIR      = SCRIPT_DIR / "data"
FILES_DIR     = SCRIPT_DIR / "files"
FILES_DATA    = FILES_DIR / "data"
LOG_DIR       = SCRIPT_DIR / "logs"

GAME_LOG_CSV  = FILES_DATA / "nba_data.csv"
LINES_CSV     = FILES_DATA / "historical_lines.csv"
LOG_FILE      = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d')}.log"

# Layer output files (all written to data/)
BLOWOUT_INDEX   = DATA_DIR / "game_blowout_index.csv"
CLEAN_BASELINES = DATA_DIR / "player_clean_baselines.csv"
PBP_MASTER      = DATA_DIR / "pbp_possessions_master.csv"
TEAM_RATINGS    = DATA_DIR / "team_ratings.csv"
PLAYER_ADJ_PPP  = DATA_DIR / "player_adj_ppp.csv"
PLAYER_PROFILES = DATA_DIR / "player_profiles.csv"
PLAYER_LUCK     = DATA_DIR / "player_luck_scores.csv"

# Final merged output — this is what your prediction model reads every day
PROJECTIONS_TODAY = DATA_DIR / "player_projections_today.csv"

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FILES_DATA.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(SCRIPT_DIR))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          NBA PROP MODEL — 5-LAYER DAILY PIPELINE            ║
║  L1:PBP  L2:PPP  L3:Usage  L4:Luck  L5:Blowout  → Picks    ║
╚══════════════════════════════════════════════════════════════╝""")


def _section(title: str):
    logger.info("=" * 65)
    logger.info(f"  {title}")
    logger.info("=" * 65)


def _summarise_csv(path: Path, label: str):
    if not path.exists():
        logger.warning(f"{label}: file not found at {path}")
        return
    import pandas as pd
    try:
        df = pd.read_csv(path)
        logger.info(f"  {label}: {len(df):,} rows × {len(df.columns)} cols — {path.name}")
        if "game_date" in df.columns:
            logger.info(f"  Latest game_date: {df['game_date'].max()}")
        if "player" in df.columns:
            logger.info(f"  Unique players: {df['player'].nunique():,}")
    except Exception as exc:
        logger.warning(f"Could not summarise {path}: {exc}")


def _step_header(n: int, title: str):
    logger.info(f"\n{'─'*65}")
    logger.info(f"  STEP {n} — {title}")
    logger.info(f"{'─'*65}")


def _ok(msg: str):
    logger.info(f"  ✓ {msg}")


def _warn(msg: str):
    logger.warning(f"  ⚠ {msg}")


def _fail(msg: str):
    logger.error(f"  ✗ {msg}")


# ── Step 1: NBA Game Logs ──────────────────────────────────────────────────────

def step1_game_logs(args, errors):
    _step_header(1, "NBA Game Logs  (nba_api → data/nba_data.csv)")
    try:
        from nba_scraper import save_game_logs
        save_game_logs(
            output_path=str(GAME_LOG_CSV),
            incremental=not args.full_refresh,
            seasons=args.seasons,
        )
        _summarise_csv(GAME_LOG_CSV, "nba_data.csv")
        _ok("Game logs updated")
    except Exception as exc:
        _fail(f"Game log fetch failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"Game logs: {exc}")


# ── Step 1.5: Auto-grade picks ─────────────────────────────────────────────────

def step1_5_grade_picks(errors):
    _step_header("1.5", "Auto-grade picks  (picks_history.csv)")
    try:
        import pandas as pd, unicodedata, re

        picks_history = FILES_DIR / "output" / "picks_history.csv"
        if not picks_history.exists() or not GAME_LOG_CSV.exists():
            _warn("picks_history.csv or nba_data.csv not found — skipping")
            return

        hist = pd.read_csv(picks_history, dtype=str)
        if "result" not in hist.columns:
            hist["result"] = ""
        if "actual" not in hist.columns:
            hist["actual"] = ""

        ungraded = hist[hist["result"].fillna("") == ""]
        if len(ungraded) == 0:
            _ok("All picks already graded")
            return

        def _norm(name):
            n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
            return re.sub(r"[^a-z0-9 ]", "", n.lower()).strip()

        box = pd.read_csv(GAME_LOG_CSV, low_memory=False)
        box["game_date"] = pd.to_datetime(box["game_date"]).dt.strftime("%Y-%m-%d")
        for col in ["pts", "trb", "ast", "stl", "blk", "tov"]:
            box[col] = pd.to_numeric(box[col], errors="coerce").fillna(0)
        box["pra"] = box["pts"] + box["trb"] + box["ast"]
        box["pr"]  = box["pts"] + box["trb"]
        box["pa"]  = box["pts"] + box["ast"]
        box["player_norm"] = box["player"].apply(_norm)
        available_dates = set(box["game_date"].unique())

        prop_map = {
            "PTS": "pts", "TRB": "trb", "AST": "ast", "STL": "stl",
            "BLK": "blk", "TOV": "tov", "PRA": "pra", "PR": "pr", "PA": "pa"
        }

        wins = losses = pushes = graded_count = 0
        for date, group in ungraded.groupby("game_date"):
            if date not in available_dates:
                continue
            day_box = box[box["game_date"] == date]
            for idx, pick in group.iterrows():
                stat = prop_map.get(str(pick.get("prop", "")).upper())
                if not stat:
                    continue
                pnorm = _norm(str(pick.get("player", "")))
                match = day_box[day_box["player_norm"] == pnorm]
                if match.empty:
                    continue
                actual = float(match.iloc[0][stat])
                line = float(pick.get("line", 0))
                direction = str(pick.get("direction", "")).upper()
                if actual == line:
                    result = "PUSH"; pushes += 1
                elif (actual > line and direction == "OVER") or (actual < line and direction == "UNDER"):
                    result = "WIN"; wins += 1
                else:
                    result = "LOSS"; losses += 1
                hist.at[idx, "result"] = result
                hist.at[idx, "actual"] = str(actual)
                graded_count += 1

        if graded_count > 0:
            hist.to_csv(picks_history, index=False)
            total = wins + losses
            pct = wins / total * 100 if total > 0 else 0
            _ok(f"Graded {graded_count} picks: {wins}W / {losses}L / {pushes}P  ({pct:.1f}%)")
        else:
            _ok("No ungraded picks with available box scores")
    except Exception as exc:
        _warn(f"Auto-grading picks failed (non-fatal): {exc}")


# ── Step 1.6: Auto-grade betslips ─────────────────────────────────────────────

def step1_6_grade_betslips(errors):
    _step_header("1.6", "Auto-grade betslips  (betslips_history.csv)")
    try:
        import pandas as pd, unicodedata, re

        betslips_path = FILES_DIR / "output" / "betslips_history.csv"
        if not betslips_path.exists() or not GAME_LOG_CSV.exists():
            _warn("betslips_history.csv or nba_data.csv not found — skipping")
            return

        slips = pd.read_csv(betslips_path, dtype=str)
        ungraded = slips[slips["result"].fillna("") == ""]
        if len(ungraded) == 0:
            _ok("All betslips already graded")
            return

        def _norm(name):
            n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
            return re.sub(r"[^a-z0-9 ]", "", n.lower()).strip()

        box = pd.read_csv(GAME_LOG_CSV, low_memory=False)
        box["game_date"] = pd.to_datetime(box["game_date"]).dt.strftime("%Y-%m-%d")
        for col in ["pts", "trb", "ast", "stl", "blk", "tov"]:
            box[col] = pd.to_numeric(box[col], errors="coerce").fillna(0)
        box["pra"] = box["pts"] + box["trb"] + box["ast"]
        box["pr"]  = box["pts"] + box["trb"]
        box["pa"]  = box["pts"] + box["ast"]
        box["ra"]  = box["trb"] + box["ast"]
        box["player_norm"] = box["player"].apply(_norm)
        available_dates = set(box["game_date"].unique())

        prop_map = {
            "PTS": "pts", "TRB": "trb", "AST": "ast", "STL": "stl",
            "BLK": "blk", "TOV": "tov", "PRA": "pra", "PR": "pr",
            "PA": "pa", "RA": "ra"
        }
        PP_MULT = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0}

        newly_graded = 0
        for idx, slip in ungraded.iterrows():
            slip_date = str(slip.get("game_date", ""))
            if slip_date not in available_dates:
                continue
            day_box = box[box["game_date"] == slip_date]
            picks_str = str(slip.get("picks", ""))
            if not picks_str:
                continue
            pick_parts = picks_str.split("|")
            n_picks = len(pick_parts)
            all_results = []
            for part in pick_parts:
                tokens = part.strip().rsplit(" ", 3)
                if len(tokens) < 4:
                    all_results.append(None)
                    continue
                player_name, prop_abbr, direction = tokens[0], tokens[1].upper(), tokens[2].upper()
                try:
                    line = float(tokens[3])
                except ValueError:
                    all_results.append(None)
                    continue
                stat = prop_map.get(prop_abbr)
                if not stat:
                    all_results.append(None)
                    continue
                pnorm = _norm(player_name)
                match = day_box[day_box["player_norm"] == pnorm]
                if match.empty:
                    all_results.append(None)
                    continue
                actual = float(match.iloc[0][stat])
                if actual == line:
                    all_results.append("PUSH")
                elif (actual > line and direction == "OVER") or (actual < line and direction == "UNDER"):
                    all_results.append("WIN")
                else:
                    all_results.append("LOSS")

            graded = [r for r in all_results if r is not None]
            if len(graded) < n_picks:
                continue
            if all(r == "WIN" for r in graded):
                slip_result = "WIN"
                wager = float(slip.get("wager", 20))
                mult = PP_MULT.get(n_picks, n_picks * 2.0)
                actual_payout = round(wager * mult, 2)
            elif any(r == "LOSS" for r in graded):
                slip_result = "LOSS"
                actual_payout = 0.0
            else:
                slip_result = "PUSH"
                actual_payout = float(slip.get("wager", 20))

            slips.at[idx, "result"] = slip_result
            slips.at[idx, "actual_payout"] = str(actual_payout)
            newly_graded += 1

        if newly_graded > 0:
            slips.to_csv(betslips_path, index=False)
            graded_now = slips[slips["result"].fillna("") != ""]
            wins = (graded_now["result"] == "WIN").sum()
            total = len(graded_now)
            _ok(f"Betslips: {wins}/{total} all-time | {newly_graded} newly graded")
        else:
            _ok("No betslips could be graded (box scores unavailable)")
    except Exception as exc:
        _warn(f"Betslip grading failed (non-fatal): {exc}")


# ── Step 2: PrizePicks Lines ───────────────────────────────────────────────────

def step2_prizepicks(args, errors):
    _step_header(2, "PrizePicks Lines  (prizepicks_scraper → historical_lines.csv)")
    try:
        from prizepicks_scraper import save_lines
        save_lines(output_path=str(LINES_CSV))
        _summarise_csv(LINES_CSV, "historical_lines.csv")
        _ok("PrizePicks lines updated")
    except Exception as exc:
        _fail(f"PrizePicks fetch failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"PrizePicks lines: {exc}")


# ── Step 3: Layer 5 — Blowout Discount Index ──────────────────────────────────

def step3_blowout_index(args, errors):
    _step_header(3, "Layer 5 — Blowout Discount Index")
    try:
        from blowout_discount_model import build_game_blowout_index

        # Rebuild index daily — it's fast (one API call for the full game log)
        logger.info("  Building game blowout index...")
        blowout_index = build_game_blowout_index(season="2024-25")
        blowout_index.to_csv(BLOWOUT_INDEX, index=False)

        tier_counts = blowout_index["blowout_tier"].value_counts()
        _ok(
            f"Blowout index: {len(blowout_index)} games — "
            f"FULL={tier_counts.get('FULL', 0)} | "
            f"PARTIAL={tier_counts.get('PARTIAL', 0)} | "
            f"HEAVY={tier_counts.get('HEAVY', 0)} | "
            f"EXCLUDE={tier_counts.get('EXCLUDE', 0)}"
        )

        # Build clean player baselines (top active players, last 20 games)
        # Only rebuild if the file is older than 12 hours to save API calls
        rebuild_baselines = True
        if CLEAN_BASELINES.exists():
            age_hours = (time.time() - CLEAN_BASELINES.stat().st_mtime) / 3600
            if age_hours < 12:
                rebuild_baselines = False
                _ok(f"Clean baselines are fresh ({age_hours:.1f}h old) — skipping rebuild")

        if rebuild_baselines:
            from blowout_discount_model import build_clean_baselines
            import pandas as pd
            from nba_api.stats.endpoints import leaguedashplayerstats

            logger.info("  Fetching active player list for clean baselines...")
            time.sleep(0.65)
            player_df = leaguedashplayerstats.LeagueDashPlayerStats(
                season="2024-25",
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Base"
            ).get_data_frames()[0]
            active = player_df[player_df["GP"] >= 15].copy()
            player_ids = list(zip(
                active["PLAYER_ID"].astype(str),
                active["PLAYER_NAME"],
                active["TEAM_ABBREVIATION"]
            ))
            logger.info(f"  Building clean baselines for {len(player_ids)} players...")
            baselines = build_clean_baselines(player_ids, blowout_index, "2024-25", last_n=20)
            baselines.to_csv(CLEAN_BASELINES, index=False)
            _ok(f"Clean baselines: {len(baselines)} players → {CLEAN_BASELINES.name}")

    except Exception as exc:
        _fail(f"Layer 5 (Blowout) failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"Layer 5 Blowout: {exc}")


# ── Step 4: Layer 1 — PBP Fetch (incremental, last 7 days) ───────────────────

def step4_pbp_fetch(args, errors):
    _step_header(4, "Layer 1 — Play-by-Play Fetch  (last 7 days, incremental)")
    try:
        from pbp_fetcher import fetch_season

        # Only fetch the last 7 days of PBP — full season fetch is too slow for daily use
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        if args.full_refresh:
            logger.info("  Full refresh: fetching entire 2024-25 PBP season (slow)...")
            fetch_season("2024-25")
        else:
            logger.info(f"  Incremental: fetching PBP for {start_date} → {end_date}")
            fetch_season("2024-25", start_date=start_date, end_date=end_date)

        # Check if master CSV exists
        if PBP_MASTER.exists():
            import pandas as pd
            master = pd.read_csv(PBP_MASTER)
            _ok(f"PBP master: {len(master):,} player-game rows → {PBP_MASTER.name}")
        else:
            _warn("PBP master CSV not yet built — run --full-refresh to build from scratch")

    except Exception as exc:
        _fail(f"Layer 1 (PBP Fetch) failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"Layer 1 PBP: {exc}")


# ── Step 5: Layer 2 — PPP Engine (team ratings + player adj_ppp) ──────────────

def step5_ppp_engine(errors):
    _step_header(5, "Layer 2 — Opponent-Adjusted PPP Engine")
    try:
        if not PBP_MASTER.exists():
            _warn("PBP master not found — skipping Layer 2. Run --full-refresh first.")
            return

        from ppp_engine import build
        logger.info("  Running Ridge Regression for team ratings...")
        build(possession_master_path=str(PBP_MASTER))

        if TEAM_RATINGS.exists():
            import pandas as pd
            ratings = pd.read_csv(TEAM_RATINGS)
            best_def = ratings.nsmallest(1, "def_rtg").iloc[0]
            worst_def = ratings.nlargest(1, "def_rtg").iloc[0]
            _ok(
                f"Team ratings: {len(ratings)} teams | "
                f"Best defense: {best_def['team']} ({best_def['def_rtg']:.1f}) | "
                f"Worst: {worst_def['team']} ({worst_def['def_rtg']:.1f})"
            )
        if PLAYER_ADJ_PPP.exists():
            import pandas as pd
            adj = pd.read_csv(PLAYER_ADJ_PPP)
            _ok(f"Player adj_ppp: {len(adj)} players → {PLAYER_ADJ_PPP.name}")

    except Exception as exc:
        _fail(f"Layer 2 (PPP Engine) failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"Layer 2 PPP: {exc}")


# ── Step 6: Layer 3 — Usage & Injury Model ────────────────────────────────────

def step6_usage_injury(errors):
    _step_header(6, "Layer 3 — Usage & Injury Model")
    try:
        from usage_injury_model import build_player_profiles, detect_recent_absences
        import pandas as pd

        # Rebuild profiles only if stale (>12 hours old)
        rebuild = True
        if PLAYER_PROFILES.exists():
            age_hours = (time.time() - PLAYER_PROFILES.stat().st_mtime) / 3600
            if age_hours < 12:
                rebuild = False
                _ok(f"Player profiles are fresh ({age_hours:.1f}h old) — skipping rebuild")

        if rebuild:
            logger.info("  Building player usage profiles...")
            profiles = build_player_profiles(season="2024-25")
            profiles.to_csv(PLAYER_PROFILES, index=False)
            _ok(f"Player profiles: {len(profiles)} players → {PLAYER_PROFILES.name}")

        # Always run absence detection — this catches today's injuries
        logger.info("  Detecting recent absences (potential injuries)...")
        profiles = pd.read_csv(PLAYER_PROFILES)
        teams = profiles["team_abbr"].unique().tolist()

        all_absences = []
        for team in teams:
            try:
                absences = detect_recent_absences(profiles, team, season="2024-25")
                if absences:
                    all_absences.extend(absences)
                time.sleep(0.3)
            except Exception:
                pass

        if all_absences:
            absence_df = pd.DataFrame(all_absences)
            absence_out = DATA_DIR / "detected_absences.csv"
            absence_df.to_csv(absence_out, index=False)
            _ok(f"Detected {len(absence_df)} potential absences/injuries → {absence_out.name}")
        else:
            _ok("No new absences detected")

    except Exception as exc:
        _fail(f"Layer 3 (Usage/Injury) failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"Layer 3 Usage: {exc}")


# ── Step 7: Layer 4 — Luck & Regression Model ─────────────────────────────────

def step7_luck_model(errors):
    _step_header(7, "Layer 4 — Luck & Regression Model")
    try:
        # Rebuild luck scores only if stale (>24 hours — luck scores are stable)
        rebuild = True
        if PLAYER_LUCK.exists():
            age_hours = (time.time() - PLAYER_LUCK.stat().st_mtime) / 3600
            if age_hours < 24:
                rebuild = False
                _ok(f"Luck scores are fresh ({age_hours:.1f}h old) — skipping rebuild")

        if rebuild:
            from luck_regression_model import build_player_luck_scores
            logger.info("  Computing player luck scores (3PT%, eFG% vs xeFG%, FT%)...")
            luck_df = build_player_luck_scores(season="2024-25")
            luck_df.to_csv(PLAYER_LUCK, index=False)

            very_lucky = (luck_df["luck_label"] == "VERY LUCKY").sum()
            very_unlucky = (luck_df["luck_label"] == "VERY UNLUCKY").sum()
            _ok(
                f"Luck scores: {len(luck_df)} players | "
                f"Very Lucky: {very_lucky} | Very Unlucky: {very_unlucky}"
            )

    except Exception as exc:
        _fail(f"Layer 4 (Luck Model) failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"Layer 4 Luck: {exc}")


# ── Step 8: Merge All Layers → player_projections_today.csv ──────────────────

def step8_merge_projections(errors):
    _step_header(8, "Merge All Layers → player_projections_today.csv")
    try:
        import pandas as pd

        # Load the base game log (raw rolling averages)
        if not GAME_LOG_CSV.exists():
            _warn("nba_data.csv not found — cannot build projections")
            return

        logger.info("  Loading base game log...")
        base = pd.read_csv(GAME_LOG_CSV, low_memory=False)
        base["game_date"] = pd.to_datetime(base["game_date"])

        # Build per-player rolling averages from last 20 games
        logger.info("  Computing L20 rolling averages per player...")
        stat_cols = ["pts", "trb", "ast", "stl", "blk", "tov", "mp"]
        for col in stat_cols:
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0)

        base = base.sort_values(["player", "game_date"])
        player_summaries = []
        for player, grp in base.groupby("player"):
            recent = grp.tail(20)
            row = {"player": player}
            for col in stat_cols:
                row[f"raw_l20_{col}"] = round(recent[col].mean(), 2)
            row["team_abbr"] = recent["team"].iloc[-1] if "team" in recent.columns else ""
            row["games_l20"] = len(recent)
            player_summaries.append(row)

        proj = pd.DataFrame(player_summaries)
        logger.info(f"  Base projections: {len(proj)} players")

        # ── Merge Layer 5: Blowout-clean baselines ────────────────────────────
        if CLEAN_BASELINES.exists():
            logger.info("  Merging Layer 5 (blowout-clean baselines)...")
            bl = pd.read_csv(CLEAN_BASELINES)[
                ["player_name", "clean_pts", "clean_reb", "clean_ast",
                 "pts_delta", "reb_delta", "ast_delta",
                 "n_full", "n_partial", "n_heavy", "n_exclude"]
            ].rename(columns={"player_name": "player"})
            proj = proj.merge(bl, on="player", how="left")
            merged_count = proj["clean_pts"].notna().sum()
            _ok(f"Layer 5 merged: {merged_count} players have blowout-clean baselines")
        else:
            _warn("Clean baselines not found — Layer 5 not merged")

        # ── Merge Layer 2: Opponent-adjusted PPP ─────────────────────────────
        if PLAYER_ADJ_PPP.exists():
            logger.info("  Merging Layer 2 (opponent-adjusted PPP)...")
            adj = pd.read_csv(PLAYER_ADJ_PPP)
            # Normalize player name for merge
            adj["player"] = adj["player_name"] if "player_name" in adj.columns else adj.get("player", "")
            adj_cols = [c for c in ["player", "adj_ppp", "raw_ppp", "opp_def_rtg",
                                     "ppp_vs_avg", "total_possessions"] if c in adj.columns]
            proj = proj.merge(adj[adj_cols], on="player", how="left")
            merged_count = proj["adj_ppp"].notna().sum() if "adj_ppp" in proj.columns else 0
            _ok(f"Layer 2 merged: {merged_count} players have adj_ppp")
        else:
            _warn("player_adj_ppp.csv not found — Layer 2 not merged")

        # ── Merge Layer 3: Usage profiles ─────────────────────────────────────
        if PLAYER_PROFILES.exists():
            logger.info("  Merging Layer 3 (usage profiles)...")
            prof = pd.read_csv(PLAYER_PROFILES)
            prof["player"] = prof["PLAYER_NAME"] if "PLAYER_NAME" in prof.columns else prof.get("player", "")
            usage_cols = [c for c in ["player", "USG_PCT", "TS_PCT", "role_tier",
                                       "pts_per_poss", "ast_per_poss"] if c in prof.columns]
            proj = proj.merge(prof[usage_cols], on="player", how="left")
            merged_count = proj["USG_PCT"].notna().sum() if "USG_PCT" in proj.columns else 0
            _ok(f"Layer 3 merged: {merged_count} players have usage profiles")
        else:
            _warn("player_profiles.csv not found — Layer 3 not merged")

        # ── Merge Layer 4: Luck scores ─────────────────────────────────────────
        if PLAYER_LUCK.exists():
            logger.info("  Merging Layer 4 (luck scores)...")
            luck = pd.read_csv(PLAYER_LUCK)
            luck["player"] = luck["PLAYER_NAME"] if "PLAYER_NAME" in luck.columns else luck.get("player", "")
            luck_cols = [c for c in ["player", "total_luck_score", "luck_label",
                                      "pts_luck_adj", "efg_luck", "three_pt_luck",
                                      "ft_luck"] if c in luck.columns]
            proj = proj.merge(luck[luck_cols], on="player", how="left")
            merged_count = proj["total_luck_score"].notna().sum() if "total_luck_score" in proj.columns else 0
            _ok(f"Layer 4 merged: {merged_count} players have luck scores")
        else:
            _warn("player_luck_scores.csv not found — Layer 4 not merged")

        # ── Compute final composite projection ────────────────────────────────
        logger.info("  Computing final composite projections...")

        # Best available points projection (prefer clean over raw)
        if "clean_pts" in proj.columns:
            proj["proj_pts"] = proj["clean_pts"].fillna(proj["raw_l20_pts"])
        else:
            proj["proj_pts"] = proj["raw_l20_pts"]

        # Apply luck adjustment if available
        if "pts_luck_adj" in proj.columns:
            proj["proj_pts_luck_adj"] = proj["pts_luck_adj"].fillna(proj["proj_pts"])
        else:
            proj["proj_pts_luck_adj"] = proj["proj_pts"]

        # Apply adj_ppp scaling if available
        # adj_ppp gives us a multiplier relative to league average (115 pts/100 poss)
        if "adj_ppp" in proj.columns:
            league_avg_ppp = 1.15  # 115 pts / 100 possessions
            proj["ppp_scale"] = (proj["adj_ppp"] / league_avg_ppp).fillna(1.0).clip(0.7, 1.3)
            proj["proj_pts_final"] = round(proj["proj_pts_luck_adj"] * proj["ppp_scale"], 2)
        else:
            proj["proj_pts_final"] = proj["proj_pts_luck_adj"]

        # Best available rebounds and assists projections
        proj["proj_reb"] = proj.get("clean_reb", proj["raw_l20_trb"]).fillna(proj["raw_l20_trb"])
        proj["proj_ast"] = proj.get("clean_ast", proj["raw_l20_ast"]).fillna(proj["raw_l20_ast"])

        # Composite PRA projection
        proj["proj_pra"] = round(
            proj["proj_pts_final"] + proj["proj_reb"] + proj["proj_ast"], 2
        )

        # Add metadata
        proj["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        proj["season"] = "2024-25"

        # Sort by projected points descending
        proj = proj.sort_values("proj_pts_final", ascending=False)

        # Save
        proj.to_csv(PROJECTIONS_TODAY, index=False)
        _ok(f"Final projections: {len(proj)} players → {PROJECTIONS_TODAY.name}")

        # Print top 10 projected scorers
        logger.info("\n  ── Top 10 Projected Scorers Today ──────────────────────────")
        top10 = proj.head(10)
        for _, row in top10.iterrows():
            luck_tag = ""
            if "luck_label" in row and pd.notna(row.get("luck_label")):
                luck_tag = f"  [{row['luck_label']}]"
            logger.info(
                f"  {row['player']:<28} "
                f"Pts: {row['proj_pts_final']:>5.1f}  "
                f"Reb: {row.get('proj_reb', 0):>4.1f}  "
                f"Ast: {row.get('proj_ast', 0):>4.1f}  "
                f"PRA: {row.get('proj_pra', 0):>5.1f}"
                f"{luck_tag}"
            )

    except Exception as exc:
        _fail(f"Merge step failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"Step 8 Merge: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    _banner()

    parser = argparse.ArgumentParser(
        description="NBA Prop Model — 5-Layer Daily Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--full-refresh",  action="store_true",
                        help="Re-pull all seasons from scratch (slow). Use once or after a long gap.")
    parser.add_argument("--lines-only",    action="store_true",
                        help="Only update PrizePicks lines; skip game logs and layers.")
    parser.add_argument("--logs-only",     action="store_true",
                        help="Only update NBA game logs; skip PrizePicks lines.")
    parser.add_argument("--skip-layers",   action="store_true",
                        help="Skip all 5 model layers; only fetch data.")
    parser.add_argument("--layers-only",   action="store_true",
                        help="Skip data fetch; only run the 5 model layers.")
    parser.add_argument("--seasons",       nargs="+", default=None, metavar="SEASON",
                        help="Override default seasons, e.g. --seasons 2023-24 2024-25")
    args = parser.parse_args()

    start_time = time.time()
    errors = []

    # ── Data Fetch Phase ──────────────────────────────────────────────────────
    if not args.layers_only:
        if not args.lines_only:
            step1_game_logs(args, errors)
            step1_5_grade_picks(errors)
            step1_6_grade_betslips(errors)
        else:
            logger.info("Skipping game logs (--lines-only mode)")

        if not args.logs_only:
            step2_prizepicks(args, errors)
        else:
            logger.info("Skipping PrizePicks lines (--logs-only mode)")

    # ── 5-Layer Model Phase ───────────────────────────────────────────────────
    if not args.skip_layers and not args.lines_only and not args.logs_only:
        step3_blowout_index(args, errors)
        step4_pbp_fetch(args, errors)
        step5_ppp_engine(errors)
        step6_usage_injury(errors)
        step7_luck_model(errors)
        step8_merge_projections(errors)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 65)
    logger.info(f"  Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Log saved to: {LOG_FILE}")

    if PROJECTIONS_TODAY.exists():
        logger.info(f"  Projections ready: {PROJECTIONS_TODAY}")

    if errors:
        logger.warning(f"\n  {len(errors)} non-fatal error(s):")
        for e in errors:
            logger.warning(f"    ✗ {e}")
        logger.info("  (Non-fatal errors do not stop the pipeline)")
    else:
        logger.info("  All steps completed successfully ✓")

    logger.info("=" * 65)


if __name__ == "__main__":
    main()
