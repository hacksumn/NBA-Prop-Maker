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
    python3 run_daily.py                    # full daily run incl. grading, lines, layers, picks
    python3 run_daily.py --full-refresh     # re-pull all seasons from scratch
    python3 run_daily.py --lines-only       # only update PrizePicks lines
    python3 run_daily.py --logs-only        # only update NBA game logs
    python3 run_daily.py --skip-layers      # skip layers 1-5, only fetch data
    python3 run_daily.py --layers-only      # skip data fetch, only run layers
    python3 run_daily.py --skip-picks       # run pipeline but do not generate today's picks
    python3 run_daily.py --skip-weekly-retrain  # skip Sunday model maintenance
    python3 run_daily.py --seasons 2024-25  # custom season list
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from injury_feed import fetch_injury_data

# Force UTF-8 output on Windows so Unicode chars in log messages don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent.resolve()
DATA_DIR      = SCRIPT_DIR / "data"
FILES_DIR     = SCRIPT_DIR / "files"
OUTPUT_DIR    = SCRIPT_DIR / "output"
LOG_DIR       = SCRIPT_DIR / "logs"

GAME_LOG_CSV  = DATA_DIR / "nba_data.csv"
LINES_CSV     = DATA_DIR / "historical_lines.csv"
PICKS_LATEST  = OUTPUT_DIR / "picks_latest.csv"
BETSLIPS_LATEST = OUTPUT_DIR / "betslips_latest.csv"
LOG_FILE      = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d')}.log"

# Layer output files (all written to data/)
BLOWOUT_INDEX   = DATA_DIR / "game_blowout_index.csv"
CLEAN_BASELINES = DATA_DIR / "player_clean_baselines.csv"
PBP_MASTER      = DATA_DIR / "pbp_possessions_master.csv"
TEAM_RATINGS    = DATA_DIR / "team_ratings.csv"
PLAYER_ADJ_PPP  = DATA_DIR / "player_adj_ppp.csv"
PLAYER_PROFILES = DATA_DIR / "player_profiles.csv"
PLAYER_LUCK     = DATA_DIR / "player_luck_scores.csv"
LIVE_INJURY_JSON = DATA_DIR / "live_injury_status.json"
LIVE_INJURY_CSV  = DATA_DIR / "live_injury_status.csv"

# Final merged output — this is what your prediction model reads every day
PROJECTIONS_TODAY = DATA_DIR / "player_projections_today.csv"
PROJECTIONS_ARCHIVE_DIR = DATA_DIR / "projection_archive"

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROJECTIONS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

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

def _atomic_csv(df, path):
    """Write DataFrame to CSV atomically so an open Excel lock on the target
    file doesn't cause a PermissionError.  Writes to a .tmp file first, then
    uses os.replace() which is atomic on Windows (renames over the target)."""
    import tempfile
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_json(payload, path):
    """Write JSON atomically to avoid partial files on interrupted runs."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _normalize_player_name(name: object) -> str:
    """ASCII-normalized player name key for cross-artifact joins."""
    normalized = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _flatten_live_injury_status(injury_data: dict, fetched_at: Optional[str] = None):
    """Flatten nested team injury payload into a CSV-friendly row contract."""
    import pandas as pd

    fetched_at = fetched_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for team_abbr, team_data in sorted((injury_data or {}).items()):
        base = {
            "team_abbr": str(team_abbr or "").upper(),
            "team_status_pending": bool(team_data.get("not_yet_submitted", False)),
            "report_label": team_data.get("report_label", ""),
            "report_url": team_data.get("report_url", ""),
            "fetched_at": fetched_at,
        }
        status_map = team_data.get("status_map", {}) or {}
        if not status_map:
            rows.append({
                **base,
                "player_norm": "",
                "injury_bucket": "",
                "injury_status": "",
                "injury_comment": "",
                "injury_source": "",
            })
            continue
        for player_norm, detail in sorted(status_map.items()):
            rows.append({
                **base,
                "player_norm": str(player_norm or ""),
                "injury_bucket": detail.get("bucket", ""),
                "injury_status": detail.get("status", ""),
                "injury_comment": detail.get("comment", ""),
                "injury_source": detail.get("source", ""),
            })
    return pd.DataFrame(
        rows,
        columns=[
            "team_abbr",
            "player_norm",
            "injury_bucket",
            "injury_status",
            "injury_comment",
            "injury_source",
            "team_status_pending",
            "report_label",
            "report_url",
            "fetched_at",
        ],
    )


def _apply_pending_team_status_fallback(proj_df):
    """Mark unresolved rows on pending-report teams with an explicit live status."""
    import pandas as pd

    proj = proj_df.copy()
    pending_mask = proj.get("live_team_status_pending", pd.Series(False, index=proj.index)).fillna(False).astype(bool)
    live_bucket = proj.get("live_injury_bucket", pd.Series(pd.NA, index=proj.index))
    live_bucket = live_bucket.fillna("").astype(str).str.strip()
    unresolved_mask = pending_mask & live_bucket.eq("")
    if not unresolved_mask.any():
        return proj, 0

    proj.loc[unresolved_mask, "live_injury_bucket"] = "team_pending"
    proj.loc[unresolved_mask, "live_injury_status"] = "Team Status Pending"
    proj.loc[unresolved_mask, "live_injury_comment"] = (
        "Official NBA injury report pending and no player-level ESPN status found"
    )
    proj.loc[unresolved_mask, "live_injury_source"] = "team_pending"
    return proj, int(unresolved_mask.sum())


def _banner():
    print("""
==============================================================
     NBA PROP MODEL -- 5-LAYER DAILY PIPELINE
     L1:PBP  L2:PPP  L3:Usage  L4:Luck  L5:Blowout -> Picks
==============================================================""")



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
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"  {label}: {len(df):,} rows × {len(df.columns)} cols — {path.name}")
        if "game_date" in df.columns:
            logger.info(f"  Latest game_date: {df['game_date'].max()}")
        if "player" in df.columns:
            logger.info(f"  Unique players: {df['player'].nunique():,}")
    except Exception as exc:
        logger.warning(f"Could not summarise {path}: {exc}")


def _summarise_live_pick_outputs():
    """Summarise the live-card artifacts produced by Step 9."""
    import pandas as pd

    if not PICKS_LATEST.exists():
        _warn(f"Live picks file not found at {PICKS_LATEST}")
        return

    try:
        picks = pd.read_csv(PICKS_LATEST, low_memory=False)
    except Exception as exc:
        _warn(f"Could not read {PICKS_LATEST.name}: {exc}")
        return

    if len(picks) == 0:
        _warn("Live card is empty after Step 9")
    else:
        _ok(f"Live card: {len(picks)} picks -> {PICKS_LATEST.name}")
        if "publish_tier" in picks.columns:
            tier_counts = picks["publish_tier"].fillna("missing").astype(str).value_counts().to_dict()
            logger.info(f"  Publish tiers: {tier_counts}")
        if "pick_source" in picks.columns:
            source_counts = picks["pick_source"].fillna("missing").astype(str).value_counts().to_dict()
            logger.info(f"  Pick sources: {source_counts}")

        # Print a clean fixed-width summary (avoids mangled wide-DataFrame log output)
        logger.info("  ── Live card ─────────────────────────────────────────────────────")
        logger.info(f"  {'PLAYER':<22} {'PROP':<5} {'DIR':<6} {'LINE':>5} {'CONF':>5} {'HIT%':>5}  TIER")
        logger.info("  " + "-" * 63)
        for _, r in picks.iterrows():
            player = str(r.get("player", ""))[:22]
            prop   = str(r.get("prop",   ""))
            dirn   = str(r.get("direction", ""))
            line   = r.get("line", "")
            conf   = r.get("confidence", "")
            hit    = r.get("estimated_hit_rate", "")
            tier   = str(r.get("publish_tier", r.get("pick_source", "")))[:16]
            try:
                line_s = f"{float(line):>5.1f}"
                conf_s = f"{float(conf):>4.0f}%"
                hit_s  = f"{float(hit)*100:>4.0f}%" if hit is not None and str(hit) not in ("", "nan") else "  -- "
            except Exception:
                line_s = conf_s = hit_s = "  ?  "
            logger.info(f"  {player:<22} {prop:<5} {dirn:<6} {line_s} {conf_s} {hit_s}  {tier}")
        logger.info("  " + "-" * 63)

    if not BETSLIPS_LATEST.exists():
        _warn(f"Live betslips file not found at {BETSLIPS_LATEST}")
        return

    try:
        slips = pd.read_csv(BETSLIPS_LATEST, low_memory=False)
    except Exception as exc:
        _warn(f"Could not read {BETSLIPS_LATEST.name}: {exc}")
        return

    if len(slips) == 0:
        _warn("No betslips were generated from the current live card")
    else:
        _ok(f"Live betslips: {len(slips)} slips -> {BETSLIPS_LATEST.name}")
        if "slip_type" in slips.columns:
            logger.info(f"  Slip types: {slips['slip_type'].fillna('missing').astype(str).value_counts().to_dict()}")


def _step_header(n: int, title: str):
    logger.info(f"\n{'─'*65}")
    logger.info(f"  STEP {n} — {title}")
    logger.info(f"{'─'*65}")


def _ok(msg: str):
    logger.info(f"  [OK] {msg}")


def _warn(msg: str):
    logger.warning(f"  ⚠ {msg}")


def _fail(msg: str):
    logger.error(f"  [FAIL] {msg}")


def _file_age_hours(path: Path) -> Optional[float]:
    """Return the file age in hours, or None when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 3600.0


def _retry_step_call(label: str, fn, attempts: int = 2, delay_s: float = 10.0):
    """Retry transient network-bound steps before falling back to cached outputs."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            wait_s = delay_s * attempt
            _warn(f"{label} attempt {attempt}/{attempts} failed: {exc}. Retrying in {wait_s:.0f}s")
            time.sleep(wait_s)
    raise last_exc


def _archive_projection_snapshot(df, archive_dir: Path, stamp: str):
    """Save a projection snapshot to the archive directory using a stable stamp."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"player_projections_{stamp}.csv"
    _atomic_csv(df, archive_path)
    return archive_path


def _run_python_step(step_no, title: str, script_args, errors, error_label: str,
                     success_msg: str, skip_msg: Optional[str] = None) -> bool:
    """Run a Python subprocess, stream output into the daily log, and track failures."""
    _step_header(step_no, title)
    cmd = [sys.executable, "-u", *script_args]
    logger.info(f"  Running: {' '.join(str(part) for part in cmd)}")
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logger.info(f"  {line.rstrip()}")
        rc = proc.wait()
    except Exception as exc:
        _fail(f"{error_label} failed to start: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"{error_label}: {exc}")
        return False

    if rc == 0:
        _ok(success_msg)
        return True

    msg = skip_msg or f"{error_label} exited with code {rc}"
    _fail(msg)
    errors.append(f"{error_label}: exit code {rc}")
    return False


# ── Step 1: NBA Game Logs ──────────────────────────────────────────────────────

def _blank_text_mask(series):
    return series.fillna("").astype(str).str.strip() == ""


def _normalize_game_date_series(series):
    import pandas as pd
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


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

        picks_history = OUTPUT_DIR / "picks_history.csv"
        if not picks_history.exists() or not GAME_LOG_CSV.exists():
            _warn("picks_history.csv or nba_data.csv not found — skipping")
            return

        from datetime import datetime as _dt
        hist = pd.read_csv(picks_history, dtype=str, low_memory=False)
        if "result"    not in hist.columns: hist["result"]    = ""
        if "actual"    not in hist.columns: hist["actual"]    = ""
        if "graded_at" not in hist.columns: hist["graded_at"] = ""
        if "game_date" not in hist.columns:
            _warn("picks_history.csv is missing game_date â€” skipping")
            return

        hist["game_date"] = _normalize_game_date_series(hist["game_date"])
        if "prop" in hist.columns:
            hist["prop"] = hist["prop"].fillna("").astype(str).str.strip().str.upper()
        if "direction" in hist.columns:
            hist["direction"] = hist["direction"].fillna("").astype(str).str.strip().str.upper()

        # ── Archive-gap backfill ──────────────────────────────────────────────
        # Scan output/archive/ for dated folders whose picks were never written
        # to picks_history.csv (e.g. predict step crashed before the append).
        # Backfill any missing dates so the normal grading loop below can grade them.
        archive_root = OUTPUT_DIR / "archive"
        today_str_pre = pd.Timestamp.now().strftime("%Y-%m-%d")
        if archive_root.is_dir():
            history_dates = set(hist["game_date"].dropna().unique())
            backfilled_count = 0
            for folder in sorted(archive_root.iterdir()):
                if not folder.is_dir():
                    continue
                folder_date = folder.name
                # Only process YYYY-MM-DD folders for past dates not in history
                try:
                    pd.Timestamp(folder_date)
                except Exception:
                    continue
                if folder_date >= today_str_pre:
                    continue
                if folder_date in history_dates:
                    continue
                # Find all picks CSVs for this slate date (picks_<date>*.csv)
                picks_files = sorted(folder.glob(f"picks_{folder_date}*.csv"))
                if not picks_files:
                    continue
                # Use the latest file (highest sort = latest timestamp)
                src = picks_files[-1]
                try:
                    bf = pd.read_csv(src, dtype=str, low_memory=False)
                    if bf.empty:
                        continue
                    bf.insert(0, "game_date", folder_date)
                    # Normalise columns to match history schema
                    bf["game_date"] = _normalize_game_date_series(bf["game_date"])
                    if "prop" in bf.columns:
                        bf["prop"] = bf["prop"].fillna("").astype(str).str.strip().str.upper()
                    if "direction" in bf.columns:
                        bf["direction"] = bf["direction"].fillna("").astype(str).str.strip().str.upper()
                    for col in ("result", "actual", "graded_at"):
                        if col not in bf.columns:
                            bf[col] = ""
                    # Align columns to existing history (add missing cols as empty)
                    for col in hist.columns:
                        if col not in bf.columns:
                            bf[col] = ""
                    bf = bf[[c for c in hist.columns if c in bf.columns] +
                             [c for c in bf.columns if c not in hist.columns]]
                    hist = pd.concat([hist, bf], ignore_index=True)
                    history_dates.add(folder_date)
                    backfilled_count += len(bf)
                    logger.warning(
                        f"  !!! ARCHIVE BACKFILL: {folder_date} had {len(bf)} picks in archive "
                        f"but was missing from picks_history.csv — restored from {src.name}"
                    )
                except Exception as e:
                    logger.warning(f"  Archive backfill failed for {folder_date}: {e}")
            if backfilled_count > 0:
                _atomic_csv(hist, picks_history)
                logger.warning(
                    f"  !!! Backfilled {backfilled_count} missing pick rows into picks_history.csv — "
                    f"they will now be graded below"
                )
        # ─────────────────────────────────────────────────────────────────────

        ungraded = hist[_blank_text_mask(hist["result"])]
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
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        gradeable = ungraded[ungraded["game_date"].fillna("").astype(str) < today_str].copy()

        # Fetch missing dates directly from nba_api (handles the 2-3 day data lag)
        missing_dates = sorted(set(gradeable["game_date"].dropna().unique()) - available_dates)
        if missing_dates:
            try:
                from nba_api.stats.endpoints import leaguegamelog
                import time as _time
                for md in missing_dates:
                    logger.info(f"  nba_data.csv missing {md} — fetching live from nba_api...")
                    try:
                        gl = leaguegamelog.LeagueGameLog(
                            season="2025-26",
                            date_from_nullable=pd.to_datetime(md).strftime("%m/%d/%Y"),
                            date_to_nullable=pd.to_datetime(md).strftime("%m/%d/%Y"),
                            player_or_team_abbreviation="P",
                        )
                        api_df = gl.get_data_frames()[0]
                        if not api_df.empty:
                            col_map = {
                                "PLAYER_NAME": "player", "GAME_DATE": "game_date",
                                "PTS": "pts", "REB": "trb", "AST": "ast",
                                "STL": "stl", "BLK": "blk", "TOV": "tov",
                            }
                            api_df = api_df.rename(columns=col_map)
                            api_df["game_date"] = pd.to_datetime(api_df["game_date"]).dt.strftime("%Y-%m-%d")
                            for col in ["pts", "trb", "ast", "stl", "blk", "tov"]:
                                api_df[col] = pd.to_numeric(api_df[col], errors="coerce").fillna(0)
                            api_df["pra"] = api_df["pts"] + api_df["trb"] + api_df["ast"]
                            api_df["pr"]  = api_df["pts"] + api_df["trb"]
                            api_df["pa"]  = api_df["pts"] + api_df["ast"]
                            api_df["player_norm"] = api_df["player"].apply(_norm)
                            box = pd.concat([box, api_df], ignore_index=True)
                            available_dates.add(md)
                            logger.info(f"    Fetched {len(api_df)} player rows for {md}")
                        _time.sleep(1)
                    except Exception as e:
                        logger.warning(f"    Could not fetch {md} from nba_api: {e}")
            except ImportError:
                logger.warning("  nba_api not available — cannot fetch missing dates")

        wins = losses = pushes = graded_count = 0
        newly_graded_rows = []  # track only picks graded in this run
        # Dates that are stale (>3 days old) and still have no box scores = no games that day → DNP
        _stale_cutoff = (pd.Timestamp.now() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        for date, group in gradeable.groupby("game_date"):
            if date not in available_dates:
                if date <= _stale_cutoff:
                    # No box scores found even from live API and date is old — mark all as DNP
                    logger.warning(
                        f"  No box scores found for {date} after live API check and date is >3 days old "
                        f"— marking {len(group)} picks as DNP (no games that day)"
                    )
                    for idx in group.index:
                        hist.at[idx, "result"]    = "DNP"
                        hist.at[idx, "actual"]    = ""
                        hist.at[idx, "graded_at"] = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                        graded_count += 1
                        newly_graded_rows.append(idx)
                else:
                    logger.info(f"  No box score data for {date} yet — skipping")
                continue
            day_box = box[box["game_date"] == date]
            for idx, pick in group.iterrows():
                stat = prop_map.get(str(pick.get("prop", "")).strip().upper())
                if not stat:
                    continue
                pnorm = _norm(str(pick.get("player", "")))
                match = day_box[day_box["player_norm"] == pnorm]
                if match.empty:
                    # Player not in box score = DNP — mark and exclude from W/L counting
                    hist.at[idx, "result"]    = "DNP"
                    hist.at[idx, "actual"]    = ""
                    hist.at[idx, "graded_at"] = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                    graded_count += 1
                    newly_graded_rows.append(idx)
                    continue
                actual = float(match.iloc[0][stat])
                line = float(pick.get("line", 0))
                direction = str(pick.get("direction", "")).strip().upper()
                if actual == line:
                    result = "PUSH"; pushes += 1
                elif (actual > line and direction == "OVER") or (actual < line and direction == "UNDER"):
                    result = "WIN"; wins += 1
                else:
                    result = "LOSS"; losses += 1
                hist.at[idx, "result"]    = result
                hist.at[idx, "actual"]    = str(actual)
                hist.at[idx, "graded_at"] = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                graded_count += 1
                newly_graded_rows.append(idx)

        if graded_count > 0:
            _atomic_csv(hist, picks_history)
            total = wins + losses
            pct = wins / total * 100 if total > 0 else 0

            # ── Print only picks graded in THIS run (exclude DNPs from display) ─
            newly_graded_df = hist.loc[newly_graded_rows]
            scored_df = newly_graded_df[newly_graded_df["result"].isin(["WIN", "LOSS", "PUSH"])]
            dnp_count = (newly_graded_df["result"] == "DNP").sum()
            if not scored_df.empty:
                print()
                print("=" * 72)
                print(f"  NEWLY GRADED PICKS ({len(scored_df)} scored, {dnp_count} DNP)")
                print("=" * 72)
                print(f"  {'PLAYER':<22} {'PROP':<5} {'DIR':<6} {'LINE':>6} {'ACTUAL':>7} {'RESULT'}")
                print("  " + "-" * 68)
                for date in sorted(scored_df["game_date"].unique()):
                    day_graded = scored_df[scored_df["game_date"] == date].sort_values("prop")
                    if day_graded.empty:
                        continue
                    print(f"  --- {date} ---")
                    for _, row in day_graded.iterrows():
                        player = str(row.get("player", ""))[:22]
                        prop   = str(row.get("prop", ""))
                        dirn   = str(row.get("direction", ""))
                        line   = row.get("line", "")
                        actual = row.get("actual", "")
                        result = str(row.get("result", ""))
                        marker = "+" if result == "WIN" else ("-" if result == "LOSS" else "=")
                        print(f"  [{marker}] {player:<22} {prop:<5} {dirn:<6} {line:>6} {actual:>7}   {result}")
                print("  " + "-" * 68)
                print(f"  TOTAL: {wins}W / {losses}L / {pushes}P  ({pct:.1f}%)")
                print("=" * 72)
                print()
            # ─────────────────────────────────────────────────────────────────

            # ── All-time record ───────────────────────────────────────────────
            all_graded = hist[hist["result"].isin(["WIN", "LOSS", "PUSH"])]
            at_wins  = (all_graded["result"] == "WIN").sum()
            at_loss  = (all_graded["result"] == "LOSS").sum()
            at_push  = (all_graded["result"] == "PUSH").sum()
            at_total = at_wins + at_loss
            at_pct   = at_wins / at_total * 100 if at_total > 0 else 0
            _ok(f"Graded {graded_count} new picks: {wins}W / {losses}L / {pushes}P  ({pct:.1f}%)  |  All-time: {at_wins}W / {at_loss}L / {at_push}P  ({at_pct:.1f}%)")
        else:
            if len(gradeable) == 0:
                _ok("No completed-slate picks pending grading")
            else:
                _ok("No ungraded picks with available box scores")

        # ── Diagnostic: pipeline health report ───────────────────────────────
        today_picks = hist[hist["game_date"] == today_str]
        still_ungraded = hist[_blank_text_mask(hist["result"])]
        overdue_ungraded = still_ungraded[still_ungraded["game_date"].fillna("").astype(str) < today_str]
        pending_ungraded = still_ungraded[still_ungraded["game_date"].fillna("").astype(str) >= today_str]
        overdue_by_date = overdue_ungraded.groupby("game_date").size()
        pending_by_date = pending_ungraded.groupby("game_date").size()

        logger.info("  ── Pick pipeline health ──────────────────────────────")
        logger.info(f"  Picks generated today ({today_str}): {len(today_picks)}")
        if len(overdue_ungraded) > 0:
            logger.info(f"  Overdue ungraded: {len(overdue_ungraded)} picks across {len(overdue_by_date)} completed dates")
            for date, cnt in sorted(overdue_by_date.items()):
                in_box = date in available_dates
                logger.info(f"    {date}: {cnt} ungraded  [box score {'available' if in_box else 'NOT FOUND'}]")
        else:
            logger.info("  All completed-slate picks graded — no gaps")
        if len(pending_ungraded) > 0:
            logger.info(f"  Pending current/future slates: {len(pending_ungraded)} picks across {len(pending_by_date)} dates")
            for date, cnt in sorted(pending_by_date.items()):
                logger.info(f"    {date}: {cnt} pending (not gradeable yet)")
        logger.info("  ─────────────────────────────────────────────────────")

    except Exception as exc:
        _warn(f"Auto-grading picks failed (non-fatal): {exc}")


# ── Step 1.6: Auto-grade betslips ─────────────────────────────────────────────

def step1_6_grade_betslips(errors):
    _step_header("1.6", "Auto-grade betslips  (betslips_history.csv)")
    try:
        import pandas as pd, unicodedata, re

        betslips_path = OUTPUT_DIR / "betslips_history.csv"
        if not betslips_path.exists() or not GAME_LOG_CSV.exists():
            _warn("betslips_history.csv or nba_data.csv not found — skipping")
            return

        slips = pd.read_csv(betslips_path, dtype=str)
        if "result" not in slips.columns:
            slips["result"] = ""
        if "actual_payout" not in slips.columns:
            slips["actual_payout"] = ""
        # Normalize game_date to ISO format (betslips may be stored as M/D/YYYY)
        slips["game_date"] = _normalize_game_date_series(slips["game_date"])
        ungraded = slips[_blank_text_mask(slips["result"])]
        if len(ungraded) == 0:
            _ok("All betslips already graded")
            return
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        gradeable = ungraded[ungraded["game_date"].fillna("").astype(str) < today_str].copy()
        if len(gradeable) == 0:
            _ok("No completed-slate betslips pending grading")
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

        # Live-fetch fallback: same as Step 1.5 — fetch any missing dates from nba_api
        missing_dates = sorted(
            set(gradeable["game_date"].dropna().astype(str).unique()) - available_dates
        )
        if missing_dates:
            try:
                from nba_api.stats.endpoints import leaguegamelog
                import time as _time2
                for md in missing_dates:
                    try:
                        gl = leaguegamelog.LeagueGameLog(
                            season="2025-26",
                            date_from_nullable=pd.to_datetime(md).strftime("%m/%d/%Y"),
                            date_to_nullable=pd.to_datetime(md).strftime("%m/%d/%Y"),
                            player_or_team_abbreviation="P",
                        )
                        api_df = gl.get_data_frames()[0]
                        if not api_df.empty:
                            col_map = {
                                "PLAYER_NAME": "player", "GAME_DATE": "game_date",
                                "PTS": "pts", "REB": "trb", "AST": "ast",
                                "STL": "stl", "BLK": "blk", "TOV": "tov",
                            }
                            api_df = api_df.rename(columns=col_map)
                            api_df["game_date"] = pd.to_datetime(api_df["game_date"]).dt.strftime("%Y-%m-%d")
                            for col in ["pts", "trb", "ast", "stl", "blk", "tov"]:
                                api_df[col] = pd.to_numeric(api_df[col], errors="coerce").fillna(0)
                            api_df["pra"] = api_df["pts"] + api_df["trb"] + api_df["ast"]
                            api_df["pr"]  = api_df["pts"] + api_df["trb"]
                            api_df["pa"]  = api_df["pts"] + api_df["ast"]
                            api_df["ra"]  = api_df["trb"] + api_df["ast"]
                            api_df["player_norm"] = api_df["player"].apply(_norm)
                            box = pd.concat([box, api_df], ignore_index=True)
                            available_dates.add(md)
                            logger.info(f"  [1.6] Live-fetched {len(api_df)} rows for betslip grading on {md}")
                        _time2.sleep(1)
                    except Exception as e:
                        logger.warning(f"  [1.6] Could not fetch {md} from nba_api: {e}")
            except ImportError:
                logger.warning("  [1.6] nba_api not available for betslip live-fetch")

        prop_map = {
            "PTS": "pts", "TRB": "trb", "AST": "ast", "STL": "stl",
            "BLK": "blk", "TOV": "tov", "PRA": "pra", "PR": "pr",
            "PA": "pa", "RA": "ra"
        }
        PP_MULT = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0}
        MAX_LEGS = 4

        def _grade_leg(player_name, prop_abbr, direction, line_val, day_box):
            stat = prop_map.get(str(prop_abbr).strip().upper())
            if not stat:
                return None
            pnorm = _norm(str(player_name))
            match = day_box[day_box["player_norm"] == pnorm]
            if match.empty:
                return None
            actual = float(match.iloc[0][stat])
            try:
                line = float(line_val)
            except (ValueError, TypeError):
                return None
            if actual == line:
                return "PUSH"
            elif (actual > line and str(direction).strip().upper() == "OVER") or (actual < line and str(direction).strip().upper() == "UNDER"):
                return "WIN"
            else:
                return "LOSS"

        newly_graded = 0
        for idx, slip in gradeable.iterrows():
            slip_date = str(slip.get("game_date", ""))
            if slip_date not in available_dates:
                continue
            day_box = box[box["game_date"] == slip_date]

            # Support both new column-per-pick format AND legacy pipe-delimited format
            all_results = []

            if "Pick_1_Player" in slips.columns:
                # New format: individual columns
                for i in range(1, MAX_LEGS + 1):
                    player = slip.get(f"Pick_{i}_Player", "")
                    prop   = slip.get(f"Pick_{i}_Prop", "")
                    dirn   = slip.get(f"Pick_{i}_Dir", "")
                    line   = slip.get(f"Pick_{i}_Line", "")
                    if not str(player).strip() or str(player).strip().lower() == "nan":
                        continue
                    result = _grade_leg(player, prop, dirn, line, day_box)
                    all_results.append(result)
            elif "picks" in slips.columns:
                # Legacy format: pipe-delimited string
                picks_str = str(slip.get("picks", ""))
                for part in picks_str.split("|"):
                    tokens = part.strip().rsplit(" ", 3)
                    if len(tokens) < 4:
                        all_results.append(None)
                        continue
                    result = _grade_leg(tokens[0], tokens[1], tokens[2], tokens[3], day_box)
                    all_results.append(result)

            n_picks = len(all_results)
            if n_picks == 0:
                continue
            # None = player not found in box scores (DNP) — treat as PUSH leg per PP policy
            graded = [r for r in all_results if r is not None]
            dnp_count = all_results.count(None)
            effective_picks = len(graded)
            if effective_picks == 0:
                continue  # All players DNP — can't grade yet

            if any(r == "LOSS" for r in graded):
                slip_result = "LOSS"
                actual_payout = 0.0
            elif all(r == "WIN" for r in graded):
                # All active legs won — pay at effective (DNP-reduced) tier
                slip_result = "WIN"
                wager = float(slip.get("wager", 20))
                mult = PP_MULT.get(effective_picks, effective_picks * 2.0)
                actual_payout = round(wager * mult, 2)
            else:
                slip_result = "PUSH"
                actual_payout = float(slip.get("wager", 20))

            slips.at[idx, "result"] = slip_result
            slips.at[idx, "actual_payout"] = str(actual_payout)
            newly_graded += 1

        if newly_graded > 0:
            _atomic_csv(slips, betslips_path)
            graded_now = slips[~_blank_text_mask(slips["result"])]
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
        blowout_index = build_game_blowout_index(season="2025-26")
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

            player_log_delay = float(os.getenv("NBA_PLAYER_LOG_DELAY", "1.5"))
            player_log_retries = int(os.getenv("NBA_PLAYER_LOG_RETRIES", "3"))
            player_log_retry_backoff = float(os.getenv("NBA_PLAYER_LOG_RETRY_BACKOFF", "4.0"))
            player_log_failure_cooldown = float(os.getenv("NBA_PLAYER_LOG_FAILURE_COOLDOWN", "6.0"))

            # Minimum games played to qualify for blowout-corrected baseline.
            # 5 GP is enough for meaningful sample while including injured returners
            # and callups who may have had blowout-distorted numbers in limited games.
            # 15 GP (old threshold) was excluding 40%+ of active players with props.
            MIN_GP_FOR_BASELINE = 5

            logger.info("  Fetching active player list for clean baselines...")
            time.sleep(0.65)
            player_df = leaguedashplayerstats.LeagueDashPlayerStats(
                season="2025-26",
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Base"
            ).get_data_frames()[0]
            n_total_active = len(player_df)
            active = player_df[player_df["GP"] >= MIN_GP_FOR_BASELINE].copy()
            n_excluded = n_total_active - len(active)
            player_ids = list(zip(
                active["PLAYER_ID"].astype(str),
                active["PLAYER_NAME"],
                active["TEAM_ABBREVIATION"]
            ))
            logger.info(
                f"  Active players this season: {n_total_active} total | "
                f"{len(active)} with GP≥{MIN_GP_FOR_BASELINE} | "
                f"{n_excluded} excluded (fewer than {MIN_GP_FOR_BASELINE} games)"
            )
            logger.info(f"  Building clean baselines for {len(player_ids)} players...")
            logger.info(
                "  Clean baseline throttle: %.2fs between player logs | %s retries | %.2fs retry backoff | %.2fs failure cooldown",
                player_log_delay,
                player_log_retries,
                player_log_retry_backoff,
                player_log_failure_cooldown,
            )
            baselines = build_clean_baselines(
                player_ids,
                blowout_index,
                "2025-26",
                last_n=20,
                request_delay=player_log_delay,
                request_retries=player_log_retries,
                retry_backoff=player_log_retry_backoff,
                failure_cooldown=player_log_failure_cooldown,
            )
            baselines.to_csv(CLEAN_BASELINES, index=False)
            coverage_pct = 100.0 * len(baselines) / len(active) if len(active) > 0 else 0.0
            _ok(
                f"Clean baselines: {len(baselines)}/{len(active)} qualifying players "
                f"({coverage_pct:.0f}% coverage) → {CLEAN_BASELINES.name}"
            )

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
            logger.info("  Full refresh: fetching entire 2025-26 PBP season (slow)...")
            fetch_season("2025-26")
        else:
            logger.info(f"  Incremental: fetching PBP for {start_date} -> {end_date}")
            fetch_season("2025-26", start_date=start_date, end_date=end_date)

        # Always rebuild master from all available per-game files
        from pbp_fetcher import merge_possession_summaries
        import pandas as pd
        master = merge_possession_summaries(output_path=str(PBP_MASTER))
        if not master.empty:
            _ok(f"PBP master: {len(master):,} player-game rows -> {PBP_MASTER.name}")
        else:
            _warn("No PBP possession files found yet — run with --full-refresh to backfill")

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
        from usage_injury_model import (
            build_player_profiles,
            detect_recent_absences,
        )
        import pandas as pd

        # Rebuild profiles only if stale (>12 hours old)
        rebuild = True
        if PLAYER_PROFILES.exists():
            age_hours = _file_age_hours(PLAYER_PROFILES) or 0.0
            if age_hours < 12:
                rebuild = False
                _ok(f"Player profiles are fresh ({age_hours:.1f}h old) — skipping rebuild")

        if rebuild:
            logger.info("  Building player usage profiles...")
            try:
                profiles = _retry_step_call(
                    "Player profile rebuild",
                    lambda: build_player_profiles(season="2025-26"),
                    attempts=2,
                    delay_s=12.0,
                )
                _atomic_csv(profiles, PLAYER_PROFILES)
                _ok(f"Player profiles: {len(profiles)} players → {PLAYER_PROFILES.name}")
            except Exception as exc:
                if not PLAYER_PROFILES.exists():
                    raise
                age_hours = _file_age_hours(PLAYER_PROFILES) or 0.0
                _warn(
                    f"Player profile rebuild failed ({exc}); "
                    f"reusing cached {PLAYER_PROFILES.name} ({age_hours:.1f}h old)"
                )

        # Always run absence detection — this catches today's injuries
        logger.info("  Detecting recent absences (potential injuries)...")
        profiles = pd.read_csv(PLAYER_PROFILES)
        teams = profiles["team_abbr"].unique().tolist()

        all_absences = []
        for team in teams:
            try:
                absences = detect_recent_absences(profiles, team, season="2025-26")
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

        logger.info("  Fetching live injury report (NBA official + ESPN fallback)...")
        live_injuries = {}
        try:
            live_injuries = fetch_injury_data() or {}
        except Exception as exc:
            logger.warning(f"  Live injury feed unavailable: {exc}")

        fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        live_injury_df = _flatten_live_injury_status(live_injuries, fetched_at=fetched_at)
        _atomic_json(live_injuries, LIVE_INJURY_JSON)
        _atomic_csv(live_injury_df, LIVE_INJURY_CSV)
        player_rows = int((live_injury_df["player_norm"] != "").sum()) if len(live_injury_df) > 0 else 0
        team_rows = int(live_injury_df["team_abbr"].nunique()) if len(live_injury_df) > 0 else 0
        pending_teams = int(
            live_injury_df.loc[live_injury_df["team_status_pending"] == True, "team_abbr"].nunique()
        ) if len(live_injury_df) > 0 else 0
        if player_rows > 0 or pending_teams > 0:
            _ok(
                f"Live injury feed: {player_rows} player statuses across {team_rows} teams "
                f"({pending_teams} pending official reports) → {LIVE_INJURY_CSV.name}"
            )
        else:
            _warn("Live injury feed returned no statuses; Step 8 will fall back to the absence proxy only")

    except Exception as exc:
        _fail(f"Layer 3 (Usage/Injury) failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"Layer 3 Usage: {exc}")


# ── Step 7: Layer 4 — Luck & Regression Model ─────────────────────────────────

def step7_luck_model(errors):
    _step_header(7, "Layer 4 — Luck & Regression Model")
    try:
        import pandas as pd
        from luck_regression_model import build_player_luck_scores
        # Layer 3 makes 30+ API calls — give the NBA API a cooldown before hitting it again
        logger.info("  Waiting 15s for API rate-limit cooldown after Layer 3...")
        time.sleep(15)
        logger.info("  Computing player luck scores (3PT%, eFG% vs xeFG%, FT%)...")
        try:
            luck_df = _retry_step_call(
                "Luck model rebuild",
                lambda: build_player_luck_scores(season="2025-26"),
                attempts=2,
                delay_s=15.0,
            )
            _atomic_csv(luck_df, PLAYER_LUCK)
        except Exception as exc:
            if not PLAYER_LUCK.exists():
                raise
            age_hours = _file_age_hours(PLAYER_LUCK) or 0.0
            _warn(
                f"Luck model refresh failed ({exc}); "
                f"reusing cached {PLAYER_LUCK.name} ({age_hours:.1f}h old)"
            )
            luck_df = pd.read_csv(PLAYER_LUCK)

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

        # Preserve the current working file before overwriting it, if it was never archived.
        if PROJECTIONS_TODAY.exists():
            try:
                existing = pd.read_csv(PROJECTIONS_TODAY, low_memory=False)
                if "generated_at" in existing.columns and len(existing) > 0:
                    existing_ts = pd.to_datetime(existing["generated_at"], errors="coerce").dropna()
                    if not existing_ts.empty:
                        existing_day = existing_ts.max().strftime("%Y-%m-%d")
                        existing_archive = PROJECTIONS_ARCHIVE_DIR / f"player_projections_{existing_day}.csv"
                        if not existing_archive.exists():
                            _archive_projection_snapshot(existing, PROJECTIONS_ARCHIVE_DIR, existing_day)
                            logger.info(f"  Archived prior working projections → {existing_archive.name}")
            except Exception as exc:
                logger.warning(f"  Could not preserve prior projection snapshot: {exc}")

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
        # Normalize player_id column name (nba_data.csv uses PLAYER_ID)
        if "PLAYER_ID" in base.columns and "player_id" not in base.columns:
            base["player_id"] = base["PLAYER_ID"]
        player_summaries = []
        for player, grp in base.groupby("player"):
            recent = grp.tail(20)
            row = {"player": player}
            for col in stat_cols:
                row[f"raw_l20_{col}"] = round(recent[col].mean(), 2)
            row["team_abbr"] = recent["team"].iloc[-1] if "team" in recent.columns else ""
            row["games_l20"] = len(recent)
            if "player_id" in recent.columns:
                row["player_id"] = recent["player_id"].iloc[-1]
            player_summaries.append(row)

        proj = pd.DataFrame(player_summaries)
        logger.info(f"  Base projections: {len(proj)} players")

        # Add normalized name key to proj once — used for all name-based merges below
        proj["player_norm"] = proj["player"].apply(_normalize_player_name)

        # ── Merge Layer 5: Blowout-clean baselines ────────────────────────────
        if CLEAN_BASELINES.exists():
            logger.info("  Merging Layer 5 (blowout-clean baselines)...")
            bl = pd.read_csv(CLEAN_BASELINES)[
                ["player_name", "clean_pts", "clean_reb", "clean_ast",
                 "pts_delta", "reb_delta", "ast_delta",
                 "n_full", "n_partial", "n_heavy", "n_exclude"]
            ].rename(columns={"player_name": "player"})
            bl["player_norm"] = bl["player"].apply(_normalize_player_name)
            bl = bl.drop_duplicates(subset=["player_norm"], keep="first")
            # Merge on normalized name to handle diacritics (e.g. Jokić vs Jokic)
            proj = proj.merge(bl.drop(columns=["player"]), on="player_norm", how="left")
            merged_count = proj["clean_pts"].notna().sum()
            _ok(f"Layer 5 merged: {merged_count} players have blowout-clean baselines")
        else:
            _warn("Clean baselines not found — Layer 5 not merged")

        # ── Merge Layer 2: Opponent-adjusted PPP ─────────────────────────────
        if PLAYER_ADJ_PPP.exists():
            logger.info("  Merging Layer 2 (opponent-adjusted PPP)...")
            adj = pd.read_csv(PLAYER_ADJ_PPP)
            adj_cols = [c for c in ["adj_ppp", "raw_ppp", "avg_opp_def_rtg",
                                     "total_possessions"] if c in adj.columns]
            # Merge on player_id (most reliable — not affected by diacritics)
            if "player_id" in adj.columns and "player_id" in proj.columns:
                adj["player_id"] = pd.to_numeric(adj["player_id"], errors="coerce")
                proj["player_id"] = pd.to_numeric(proj["player_id"], errors="coerce")
                adj = adj.drop_duplicates(subset=["player_id"], keep="first")
                proj = proj.merge(adj[["player_id"] + adj_cols], on="player_id", how="left")
            else:
                # fallback: normalized name match
                adj["player_norm"] = (adj["player_name"] if "player_name" in adj.columns else adj.get("player", pd.Series(dtype=str))).apply(_normalize_player_name)
                adj = adj.drop_duplicates(subset=["player_norm"], keep="first")
                proj = proj.merge(adj[["player_norm"] + adj_cols], on="player_norm", how="left")
            merged_count = proj["adj_ppp"].notna().sum() if "adj_ppp" in proj.columns else 0
            _ok(f"Layer 2 merged: {merged_count} players have adj_ppp")
        else:
            _warn("player_adj_ppp.csv not found — Layer 2 not merged")

        # ── Merge Layer 3: Usage profiles ─────────────────────────────────────
        if PLAYER_PROFILES.exists():
            logger.info("  Merging Layer 3 (usage profiles)...")
            prof = pd.read_csv(PLAYER_PROFILES)
            # profiles uses player_name (lowercase); normalize for diacritic-safe merge
            if "player_name" in prof.columns:
                prof = prof.rename(columns={"player_name": "player"})
            elif "PLAYER_NAME" in prof.columns:
                prof = prof.rename(columns={"PLAYER_NAME": "player"})
            prof["player_norm"] = prof["player"].apply(_normalize_player_name)
            prof = prof.drop_duplicates(subset=["player_norm"], keep="first")
            usage_cols = [c for c in ["player_norm", "usg_pct", "ts_pct", "role_tier",
                                       "pts_per_poss", "ast_per_poss"] if c in prof.columns]
            proj = proj.merge(prof[usage_cols], on="player_norm", how="left")
            merged_count = proj["usg_pct"].notna().sum() if "usg_pct" in proj.columns else 0
            _ok(f"Layer 3 merged: {merged_count} players have usage profiles")
        else:
            _warn("player_profiles.csv not found — Layer 3 not merged")

        # ── Merge Layer 4: Luck scores ─────────────────────────────────────────
        if PLAYER_LUCK.exists():
            logger.info("  Merging Layer 4 (luck scores)...")
            luck = pd.read_csv(PLAYER_LUCK)
            # player_luck_scores.csv uses "player_name" (lowercase)
            for col in ["player_name", "PLAYER_NAME", "player"]:
                if col in luck.columns:
                    luck["player"] = luck[col]
                    break
            luck["player_norm"] = luck["player"].apply(_normalize_player_name)
            luck = luck.drop_duplicates(subset=["player_norm"], keep="first")
            luck_cols = [c for c in ["player_norm", "total_luck_score", "luck_label",
                                      "pts_luck_adj", "fg3_luck_score", "ft_luck_score",
                                      "efg_luck_score"] if c in luck.columns]
            proj = proj.merge(luck[luck_cols], on="player_norm", how="left")
            merged_count = proj["total_luck_score"].notna().sum() if "total_luck_score" in proj.columns else 0
            _ok(f"Layer 4 merged: {merged_count} players have luck scores")
        else:
            _warn("player_luck_scores.csv not found — Layer 4 not merged")

        # Dedup: merges can produce multiple rows per player if source data has duplicates
        proj = proj.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True)
        logger.info(f"  After dedup: {len(proj)} unique players")

        # ── Merge detected absences (Step 6 output) ──────────────────────────
        detected_absences_path = DATA_DIR / "detected_absences.csv"
        if detected_absences_path.exists():
            logger.info("  Merging detected absences (Layer 3 absence scan)...")
            absences = pd.read_csv(detected_absences_path)
            if "player_id" in absences.columns and "player_id" in proj.columns:
                absences["player_id"] = pd.to_numeric(absences["player_id"], errors="coerce")
                proj["player_id"] = pd.to_numeric(proj["player_id"], errors="coerce")
                proj = proj.merge(
                    absences[["player_id", "recent_gp"]].rename(columns={"recent_gp": "absence_recent_gp"}),
                    on="player_id", how="left"
                )
                flagged = proj["absence_recent_gp"].notna().sum()
                _ok(f"Absence data merged: {flagged} players flagged with recent absence data")
            else:
                proj["absence_recent_gp"] = float("nan")
        else:
            proj["absence_recent_gp"] = float("nan")

        # Merge live injury report data captured in Step 6 so the projection
        # artifact has current availability context, not just a recent-games proxy.
        proj["live_team_status_pending"] = False
        if LIVE_INJURY_CSV.exists():
            logger.info("  Merging live injury feed (NBA official + ESPN fallback)...")
            live_inj = pd.read_csv(LIVE_INJURY_CSV)
            if len(live_inj) > 0:
                live_inj["team_abbr"] = live_inj["team_abbr"].astype(str).str.upper()
                live_inj["player_norm"] = live_inj["player_norm"].fillna("").astype(str)
                proj["team_abbr"] = proj["team_abbr"].astype(str).str.upper()
                proj["player_norm"] = proj["player"].apply(_normalize_player_name)

                pending_teams = set(
                    live_inj.loc[live_inj["team_status_pending"] == True, "team_abbr"].astype(str).tolist()
                )
                live_player_rows = live_inj[live_inj["player_norm"] != ""].copy()
                if len(live_player_rows) > 0:
                    live_player_rows = live_player_rows.drop_duplicates(
                        subset=["team_abbr", "player_norm"], keep="first"
                    )
                    proj = proj.merge(
                        live_player_rows[
                            [
                                "team_abbr",
                                "player_norm",
                                "injury_bucket",
                                "injury_status",
                                "injury_comment",
                                "injury_source",
                            ]
                        ].rename(columns={
                            "injury_bucket": "live_injury_bucket",
                            "injury_status": "live_injury_status",
                            "injury_comment": "live_injury_comment",
                            "injury_source": "live_injury_source",
                        }),
                        on=["team_abbr", "player_norm"],
                        how="left",
                    )
                proj["live_team_status_pending"] = proj["team_abbr"].isin(pending_teams)
                merged_count = int(proj["live_injury_bucket"].notna().sum())
                _ok(
                    f"Live injury feed merged: {merged_count} player statuses "
                    f"across {len(pending_teams)} pending-report teams"
                )
            else:
                _warn("Live injury status file is empty — Step 8 is using the absence proxy only")
        else:
            _warn("Live injury status file not found — Step 8 is using the absence proxy only")

        for col in ["live_injury_bucket", "live_injury_status", "live_injury_comment", "live_injury_source"]:
            if col not in proj.columns:
                proj[col] = pd.NA
        proj, pending_fallback_count = _apply_pending_team_status_fallback(proj)
        if pending_fallback_count > 0:
            _warn(
                f"Marked {pending_fallback_count} players as TEAM STATUS PENDING "
                f"(pending official report with no player-level live entry)"
            )

        # Suppress luck-driven signals for players flagged by the recent-absence scan.
        # These historical luck metrics remain valid in the standalone Layer 4 artifact,
        # but they are misleading in today's projection output when availability is uncertain.
        absence_recent_gp = pd.to_numeric(proj.get("absence_recent_gp"), errors="coerce")
        absence_risk_mask = absence_recent_gp.notna() & absence_recent_gp.le(2)
        live_injury_bucket = proj.get("live_injury_bucket", pd.Series(pd.NA, index=proj.index))
        live_injury_bucket = live_injury_bucket.fillna("").astype(str).str.lower()
        live_report_risk_mask = live_injury_bucket.isin(
            ["out", "doubtful", "questionable", "day_to_day", "team_pending"]
        )
        suppress_luck_mask = absence_risk_mask | live_report_risk_mask
        if suppress_luck_mask.any():
            numeric_luck_cols = [
                c for c in [
                    "total_luck_score", "pts_luck_adj", "fg3_luck_score", "ft_luck_score",
                    "efg_luck_score", "fg3_pts_luck_adj", "fg3_regressed", "ft_regressed",
                ]
                if c in proj.columns
            ]
            if numeric_luck_cols:
                proj.loc[suppress_luck_mask, numeric_luck_cols] = float("nan")
            if "luck_label" in proj.columns:
                proj.loc[suppress_luck_mask, "luck_label"] = pd.NA
            _ok(
                f"Suppressed luck adjustments for {int(suppress_luck_mask.sum())} players with "
                f"absence-risk or live injury flags"
            )

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
        run_dt = datetime.now()
        proj["generated_at"] = run_dt.strftime("%Y-%m-%d %H:%M:%S")
        proj["season"] = "2025-26"

        # Sort by projected points descending
        proj = proj.sort_values("proj_pts_final", ascending=False)
        proj = proj.drop(columns=["player_norm"], errors="ignore")

        # Save current working file plus archived snapshots.
        _atomic_csv(proj, PROJECTIONS_TODAY)
        archive_day = run_dt.strftime("%Y-%m-%d")
        archive_stamp = run_dt.strftime("%Y-%m-%d_%H%M%S")
        daily_archive = _archive_projection_snapshot(proj, PROJECTIONS_ARCHIVE_DIR, archive_day)
        run_archive = _archive_projection_snapshot(proj, PROJECTIONS_ARCHIVE_DIR, archive_stamp)
        _ok(f"Final projections: {len(proj)} players → {PROJECTIONS_TODAY.name}")
        logger.info(f"  Archived daily snapshot: {daily_archive.name}")
        logger.info(f"  Archived run snapshot:   {run_archive.name}")

        # Print top 10 projected scorers — skip players with 0 recent games (fully absent)
        logger.info("\n  ── Top 10 Projected Scorers Today (active players only) ──────────────")
        display_rows = []
        skipped_absent = []
        skipped_live_out = []
        skipped_team_pending = []
        for _, row in proj.iterrows():
            live_bucket = str(row.get("live_injury_bucket") or "").strip().lower()
            if live_bucket in {"out", "doubtful"}:
                skipped_live_out.append(f"{row['player']} ({live_bucket.upper()})")
                continue
            if live_bucket == "team_pending":
                skipped_team_pending.append(row["player"])
                continue
            recent_gp = row.get("absence_recent_gp")
            if pd.notna(recent_gp) and int(recent_gp) == 0:
                skipped_absent.append(row["player"])
                continue
            display_rows.append(row)
            if len(display_rows) >= 10:
                break
        top10 = pd.DataFrame(display_rows) if display_rows else proj.head(10)
        for _, row in top10.iterrows():
            luck_tag = ""
            if "luck_label" in row and pd.notna(row.get("luck_label")):
                luck_tag = f"  [{row['luck_label']}]"
            absence_tag = ""
            recent_gp = row.get("absence_recent_gp")
            if pd.notna(recent_gp) and int(recent_gp) <= 2:
                absence_tag = f"  [ABSENCE RISK: {int(recent_gp)}/5 games]"
            injury_tag = ""
            live_bucket = str(row.get("live_injury_bucket") or "").strip().lower()
            if live_bucket in {"questionable", "day_to_day", "probable"}:
                injury_tag = f"  [{live_bucket.replace('_', ' ').upper()}]"
            elif bool(row.get("live_team_status_pending", False)) and not live_bucket:
                injury_tag = "  [TEAM STATUS PENDING]"
            logger.info(
                f"  {row['player']:<28} "
                f"Pts: {row['proj_pts_final']:>5.1f}  "
                f"Reb: {row.get('proj_reb', 0):>4.1f}  "
                f"Ast: {row.get('proj_ast', 0):>4.1f}  "
                f"PRA: {row.get('proj_pra', 0):>5.1f}"
                f"{injury_tag}{luck_tag}{absence_tag}"
            )
        if skipped_absent:
            logger.info(
                f"\n  [Suppressed from top-10 (0/5 recent games): "
                f"{', '.join(skipped_absent[:8])}{'...' if len(skipped_absent) > 8 else ''}]"
            )
        if skipped_live_out:
            logger.info(
                f"  [Suppressed from top-10 (official OUT/DOUBTFUL): "
                f"{', '.join(skipped_live_out[:8])}{'...' if len(skipped_live_out) > 8 else ''}]"
            )
        if skipped_team_pending:
            logger.info(
                f"  [Suppressed from top-10 (team status pending): "
                f"{', '.join(skipped_team_pending[:8])}{'...' if len(skipped_team_pending) > 8 else ''}]"
            )

    except Exception as exc:
        _fail(f"Merge step failed: {exc}")
        logger.debug(exc, exc_info=True)
        errors.append(f"Step 8 Merge: {exc}")


def step8_5_weekly_maintenance(args, errors):
    if args.skip_weekly_retrain:
        logger.info("Skipping weekly maintenance (--skip-weekly-retrain)")
        return
    if datetime.now().weekday() != 6:
        logger.info("Skipping weekly maintenance (runs automatically on Sunday only)")
        return

    _section("WEEKLY MODEL MAINTENANCE")
    _run_python_step(
        "8.5a",
        "Weekly Feature Pipeline Rebuild",
        ["feature_pipeline.py"],
        errors,
        "Weekly feature rebuild",
        "Weekly feature rebuild complete",
    )
    _run_python_step(
        "8.5b",
        "Weekly Advanced Model Retrain",
        ["train_advanced_models.py"],
        errors,
        "Weekly advanced retrain",
        "Weekly advanced retrain complete",
    )
    _run_python_step(
        "8.5c",
        "Weekly Confidence Calibration",
        ["calibrate_confidence.py"],
        errors,
        "Weekly confidence calibration",
        "Weekly confidence calibration complete",
    )


def step9_generate_picks(args, errors):
    if args.skip_picks:
        logger.info("Skipping live pick generation (--skip-picks)")
        return

    ok = _run_python_step(
        9,
        "Generate Today's Picks  (nba_props.py predict)",
        ["nba_props.py", "predict"],
        errors,
        "Live pick generation",
        "Today's pick generation complete",
    )
    if ok:
        _summarise_live_pick_outputs()


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
    parser.add_argument("--skip-picks",    action="store_true",
                        help="Skip live pick generation at the end of the daily run.")
    parser.add_argument("--skip-weekly-retrain", action="store_true",
                        help="Skip automatic Sunday feature/model/calibration refresh.")
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
        step8_5_weekly_maintenance(args, errors)

    if not args.lines_only and not args.logs_only:
        step9_generate_picks(args, errors)

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
            logger.warning(f"    [FAIL] {e}")
        logger.info("  (Non-fatal errors do not stop the pipeline)")
    else:
        logger.info("  All steps completed successfully")

    logger.info("=" * 65)


if __name__ == "__main__":
    main()
