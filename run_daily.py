#!/usr/bin/env python3
"""
run_daily.py
------------
NBA Prop Model Data Fetcher — Daily Runner
==========================================
Run this each morning before your prediction model to pull:
  1. Last night's NBA box scores -> data/nba_data.csv
  2. Today's PrizePicks prop lines -> data/historical_lines.csv

Usage:
    python3 run_daily.py                  # incremental update (recommended daily)
    python3 run_daily.py --full-refresh   # re-pull all 3 seasons from scratch
    python3 run_daily.py --lines-only     # only update PrizePicks lines
    python3 run_daily.py --logs-only      # only update NBA game logs
    python3 run_daily.py --seasons 2024-25 2025-26  # custom season list
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
DATA_DIR     = SCRIPT_DIR / "files" / "data"   # model reads from files/data/
LOG_DIR      = SCRIPT_DIR / "logs"
GAME_LOG_CSV = DATA_DIR / "nba_data.csv"
LINES_CSV    = DATA_DIR / "historical_lines.csv"
LOG_FILE     = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d')}.log"

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Add script dir to path so we can import sibling modules
sys.path.insert(0, str(SCRIPT_DIR))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_banner():
    banner = """
==============================================================
         NBA PROP MODEL -- FREE DATA FETCHER
         Game Logs: nba_api  |  Lines: PrizePicks
=============================================================="""
    print(banner)


def _summarise_csv(path: Path, label: str):
    """Print a quick summary of a CSV file after saving."""
    if not path.exists():
        logger.warning(f"{label}: file not found at {path}")
        return
    import pandas as pd
    try:
        df = pd.read_csv(path)
        logger.info(f"{label}: {len(df):,} rows, {len(df.columns)} columns — {path}")
        if "game_date" in df.columns:
            latest = df["game_date"].max()
            logger.info(f"     Latest game_date in file: {latest}")
        if "player" in df.columns:
            logger.info(f"     Unique players: {df['player'].nunique():,}")
    except Exception as exc:
        logger.warning(f"Could not summarise {path}: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _print_banner()

    parser = argparse.ArgumentParser(
        description="NBA Prop Model — Daily Data Fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--full-refresh", action="store_true",
        help="Re-pull all 3 seasons from scratch (slow, ~60s). Use once or after a long gap.",
    )
    parser.add_argument(
        "--lines-only", action="store_true",
        help="Only update PrizePicks prop lines; skip game logs.",
    )
    parser.add_argument(
        "--logs-only", action="store_true",
        help="Only update NBA game logs; skip PrizePicks lines.",
    )
    parser.add_argument(
        "--seasons", nargs="+", default=None,
        metavar="SEASON",
        help="Override default seasons, e.g. --seasons 2023-24 2024-25 2025-26",
    )
    args = parser.parse_args()

    start_time = time.time()
    errors = []

    # ── Step 1: NBA Game Logs ─────────────────────────────────────────────────
    if not args.lines_only:
        logger.info("=" * 60)
        logger.info("STEP 1 — NBA Game Logs (nba_api -> data/nba_data.csv)")
        logger.info("=" * 60)
        try:
            from nba_scraper import save_game_logs
            save_game_logs(
                output_path=str(GAME_LOG_CSV),
                incremental=not args.full_refresh,
                seasons=args.seasons,
            )
            _summarise_csv(GAME_LOG_CSV, "nba_data.csv")
        except Exception as exc:
            logger.error(f"Game log fetch failed: {exc}", exc_info=True)
            errors.append(f"Game logs: {exc}")
    else:
        logger.info("Skipping game logs (--lines-only mode)")

    # ── Step 1.5: Auto-grade all ungraded picks against box scores ────────────
    if not args.lines_only:
        try:
            import pandas as pd, unicodedata, re
            from datetime import timedelta

            picks_history = SCRIPT_DIR / "files" / "output" / "picks_history.csv"
            if picks_history.exists() and GAME_LOG_CSV.exists():
                hist = pd.read_csv(picks_history, dtype=str)

                if 'result' not in hist.columns:
                    hist['result'] = ''
                if 'actual' not in hist.columns:
                    hist['actual'] = ''

                ungraded_picks = hist[hist['result'].fillna('') == '']

                if len(ungraded_picks) > 0:
                    def _norm(name):
                        n = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
                        return re.sub(r'[^a-z0-9 ]', '', n.lower()).strip()

                    box_all = pd.read_csv(GAME_LOG_CSV, low_memory=False)
                    box_all['game_date'] = pd.to_datetime(box_all['game_date']).dt.strftime('%Y-%m-%d')
                    for col in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov']:
                        box_all[col] = pd.to_numeric(box_all[col], errors='coerce').fillna(0)
                    box_all['pra'] = box_all['pts'] + box_all['trb'] + box_all['ast']
                    box_all['pr']  = box_all['pts'] + box_all['trb']
                    box_all['pa']  = box_all['pts'] + box_all['ast']
                    box_all['player_norm'] = box_all['player'].apply(_norm)
                    available_dates = set(box_all['game_date'].unique())

                    prop_map = {
                        'PTS':'pts','TRB':'trb','AST':'ast','STL':'stl',
                        'BLK':'blk','TOV':'tov','PRA':'pra','PR':'pr','PA':'pa'
                    }

                    wins = losses = pushes = graded_count = 0
                    for date, group in ungraded_picks.groupby('game_date'):
                        if date not in available_dates:
                            continue  # box scores not yet available for this date
                        box = box_all[box_all['game_date'] == date]
                        logger.info(f"Grading {len(group)} picks from {date}...")
                        for idx, pick in group.iterrows():
                            stat = prop_map.get(str(pick.get('prop', '')).upper())
                            if not stat:
                                continue
                            pnorm = _norm(str(pick.get('player', '')))
                            match = box[box['player_norm'] == pnorm]
                            if len(match) == 0:
                                continue
                            actual = float(match.iloc[0][stat])
                            line   = float(pick.get('line', 0))
                            direction = str(pick.get('direction', '')).upper()
                            if actual == line:
                                result = 'PUSH'; pushes += 1
                            elif (actual > line and direction == 'OVER') or (actual < line and direction == 'UNDER'):
                                result = 'WIN';  wins += 1
                            else:
                                result = 'LOSS'; losses += 1
                            hist.at[idx, 'result'] = result
                            hist.at[idx, 'actual'] = str(actual)
                            graded_count += 1

                    if graded_count > 0:
                        hist.to_csv(picks_history, index=False)
                        total = wins + losses
                        pct = wins / total * 100 if total > 0 else 0
                        logger.info(f"  Graded {graded_count} picks total: {wins}W / {losses}L / {pushes}P  ({pct:.1f}%)")
                    else:
                        logger.info(f"  No ungraded picks found with available box scores")
                else:
                    logger.info(f"  All picks already graded")
        except Exception as exc:
            logger.warning(f"Auto-grading failed (non-fatal): {exc}")

    # ── Step 1.6: Auto-grade yesterday's betslips ─────────────────────────────
    if not args.lines_only:
        try:
            import pandas as pd, unicodedata, re
            from datetime import timedelta

            betslips_path = SCRIPT_DIR / "files" / "output" / "betslips_history.csv"
            if betslips_path.exists() and GAME_LOG_CSV.exists():
                slips = pd.read_csv(betslips_path, dtype=str)

                ungraded = slips[slips['result'].fillna('') == '']

                if len(ungraded) == 0:
                    logger.info(f"  All betslips already graded")
                else:
                    def _norm(name):
                        n = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('ascii')
                        return re.sub(r'[^a-z0-9 ]', '', n.lower()).strip()

                    box_all = pd.read_csv(GAME_LOG_CSV, low_memory=False)
                    box_all['game_date'] = pd.to_datetime(box_all['game_date']).dt.strftime('%Y-%m-%d')
                    for col in ['pts', 'trb', 'ast', 'stl', 'blk', 'tov']:
                        box_all[col] = pd.to_numeric(box_all[col], errors='coerce').fillna(0)
                    box_all['pra'] = box_all['pts'] + box_all['trb'] + box_all['ast']
                    box_all['pr']  = box_all['pts'] + box_all['trb']
                    box_all['pa']  = box_all['pts'] + box_all['ast']
                    box_all['ra']  = box_all['trb'] + box_all['ast']
                    box_all['player_norm'] = box_all['player'].apply(_norm)
                    available_dates = set(box_all['game_date'].unique())

                    prop_map = {
                        'PTS':'pts','TRB':'trb','AST':'ast','STL':'stl',
                        'BLK':'blk','TOV':'tov','PRA':'pra','PR':'pr','PA':'pa','RA':'ra'
                    }

                    PP_MULT = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0}

                    newly_graded = 0
                    for idx, slip in ungraded.iterrows():
                        slip_date = str(slip.get('game_date', ''))
                        if slip_date not in available_dates:
                            continue  # box scores not yet available for this date
                        box = box_all[box_all['game_date'] == slip_date]

                        picks_str = str(slip.get('picks', ''))
                        if not picks_str:
                            continue
                        # Each pick: "Player Name PROP DIRECTION LINE"
                        pick_parts = picks_str.split('|')
                        n_picks = len(pick_parts)
                        all_results = []
                        for part in pick_parts:
                            tokens = part.strip().rsplit(' ', 3)  # split from right: prop dir line
                            if len(tokens) < 4:
                                all_results.append(None)
                                continue
                            player_name = tokens[0]
                            prop_abbr   = tokens[1].upper()
                            direction   = tokens[2].upper()
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
                            match = box[box['player_norm'] == pnorm]
                            if len(match) == 0:
                                all_results.append(None)
                                continue
                            actual = float(match.iloc[0][stat])
                            if actual == line:
                                all_results.append('PUSH')
                            elif (actual > line and direction == 'OVER') or (actual < line and direction == 'UNDER'):
                                all_results.append('WIN')
                            else:
                                all_results.append('LOSS')

                        graded = [r for r in all_results if r is not None]
                        if len(graded) < n_picks:
                            continue  # missing box scores, skip

                        if all(r == 'WIN' for r in graded):
                            slip_result = 'WIN'
                            wager = float(slip.get('wager', 20))
                            mult  = PP_MULT.get(n_picks, n_picks * 2.0)
                            actual_payout = round(wager * mult, 2)
                        elif any(r == 'LOSS' for r in graded):
                            slip_result   = 'LOSS'
                            actual_payout = 0.0
                        else:
                            slip_result   = 'PUSH'
                            actual_payout = float(slip.get('wager', 20))

                        slips.at[idx, 'result']        = slip_result
                        slips.at[idx, 'actual_payout'] = str(actual_payout)
                        newly_graded += 1

                    if newly_graded > 0:
                        slips.to_csv(betslips_path, index=False)
                        graded_now = slips[slips['result'].fillna('') != '']
                        wins  = (graded_now['result'] == 'WIN').sum()
                        total = len(graded_now)
                        logger.info(f"  Betslips graded: {wins}/{total} all-time | {newly_graded} newly graded")
                    else:
                        logger.info(f"  No betslips could be graded (box scores unavailable)")
        except Exception as exc:
            logger.warning(f"Betslip grading failed (non-fatal): {exc}")

    # ── Step 2: PrizePicks Prop Lines ─────────────────────────────────────────
    if not args.logs_only:
        logger.info("=" * 60)
        logger.info("STEP 2 — PrizePicks Prop Lines -> data/historical_lines.csv")
        logger.info("=" * 60)
        try:
            from prizepicks_scraper import save_lines
            save_lines(output_path=str(LINES_CSV))
            _summarise_csv(LINES_CSV, "historical_lines.csv")
        except Exception as exc:
            logger.error(f"PrizePicks fetch failed: {exc}", exc_info=True)
            errors.append(f"PrizePicks lines: {exc}")
    else:
        logger.info("Skipping PrizePicks lines (--logs-only mode)")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    if errors:
        logger.warning(f"Completed with {len(errors)} error(s) in {elapsed:.1f}s:")
        for e in errors:
            logger.warning(f"  x {e}")
        sys.exit(1)
    else:
        logger.info(f"All done in {elapsed:.1f}s — data is ready for your model.")
        logger.info(f"Log saved to: {LOG_FILE}")
        sys.exit(0)


if __name__ == "__main__":
    main()
