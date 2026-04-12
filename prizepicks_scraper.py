"""
prizepicks_scraper.py
---------------------
Pulls today's NBA prop lines from PrizePicks.

PrizePicks uses Cloudflare + PerimeterX bot detection, which blocks
headless browsers from a fresh profile. This scraper uses two strategies:

  Strategy A (recommended): Playwright with YOUR existing Chrome/Chromium
    profile — the one you already use to browse PrizePicks. Since your
    browser has already solved the CAPTCHA, it has valid cookies and
    passes the bot check automatically.

  Strategy B (fallback): Direct requests using a manually supplied
    __cf_clearance cookie (instructions below).

HOW TO SET UP (one-time):
  1. Open Chrome/Chromium on your machine and visit app.prizepicks.com
  2. Copy your Chrome profile path (see CHROME_PROFILE below)
  3. Set the PRIZEPICKS_CHROME_PROFILE env variable, OR edit the
     CHROME_PROFILE constant below.

  Alternatively, export your __cf_clearance cookie from your browser
  (DevTools -> Application -> Cookies -> app.prizepicks.com) and set the
  PRIZEPICKS_CF_COOKIE env variable.

Output CSV: data/historical_lines.csv
Columns:
    game_date, player, player_norm, prop, line, num_books
"""

import os
import re
import json
import time
import logging
import pandas as pd
from datetime import date
from unicodedata import normalize as uniNorm
from pathlib import Path

from player_pool_guard import sanitize_player_names

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Set this to your local Chrome profile directory, OR set the env variable.
# Common locations:
#   Windows: C:/Users/<YOU>/AppData/Local/Google/Chrome/User Data
#   macOS:   /Users/<YOU>/Library/Application Support/Google/Chrome
#   Linux:   /home/<YOU>/.config/google-chrome
CHROME_PROFILE = os.environ.get(
    "PRIZEPICKS_CHROME_PROFILE",
    r"C:\Users\jakep\AppData\Local\Google\Chrome\User Data",  # Windows default
)

# Alternatively, paste your __cf_clearance cookie value here or in env
CF_CLEARANCE_COOKIE = os.environ.get("PRIZEPICKS_CF_COOKIE", "")

# Props we care about for NBA player prop modelling
TARGET_PROPS = {
    "Points", "Rebounds", "Assists", "Steals", "Blocks",
    "Turnovers", "3-PT Made", "Pts+Rebs+Asts",
    "Pts+Rebs", "Pts+Asts", "Rebs+Asts",
    "Fantasy Score",
}

NBA_LEAGUE_ID = 7
PP_API_URL = "https://api.prizepicks.com/projections"

# PrizePicks stat name -> Odds API prop key (what nba_props.py expects)
PROP_NAME_MAP = {
    "Points":           "player_points",
    "Rebounds":         "player_rebounds",
    "Assists":          "player_assists",
    "Steals":           "player_steals",
    "Blocks":           "player_blocks",
    "Blocked Shots":    "player_blocks",
    "Turnovers":        "player_turnovers",
    "Pts+Rebs+Asts":    "player_points_rebounds_assists",
    "Pts+Rebs":         "player_points_rebounds",
    "Pts+Asts":         "player_points_assists",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    nfkd = uniNorm("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]", "", ascii_str.lower()).strip()


def _extract_player_map(included: list) -> dict:
    player_map = {}
    for item in included:
        if item.get("type") == "new_player":
            pid = item["id"]
            attrs = item.get("attributes", {})
            name = attrs.get("display_name") or attrs.get("name", "Unknown")
            player_map[pid] = name
    return player_map


def _extract_game_attr_map(included: list) -> dict:
    game_map = {}
    for item in included:
        if item.get("type") == "game":
            game_map[item.get("id", "")] = item.get("attributes", {})
    return game_map


def _coerce_slate_date(*values) -> str | None:
    for value in values:
        if not value:
            continue
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            continue
        return ts.strftime("%Y-%m-%d")
    return None


def _parse_projections(data: dict) -> pd.DataFrame:
    player_map = _extract_player_map(data.get("included", []))
    game_map = _extract_game_attr_map(data.get("included", []))
    projections = data.get("data", [])

    rows = []
    for proj in projections:
        attrs = proj.get("attributes", {})
        rels  = proj.get("relationships", {})
        stat_type = attrs.get("stat_type", "")
        prop_key = PROP_NAME_MAP.get(stat_type)
        if not prop_key:
            continue
        player_rel  = rels.get("new_player", {}).get("data", {})
        player_id   = player_rel.get("id", "")
        player_name = player_map.get(player_id, f"player_{player_id}")
        line = attrs.get("line_score")
        if line is None:
            continue
        game_rel = rels.get("game", {}).get("data", {})
        game_attrs = game_map.get(game_rel.get("id", ""), {})
        slate_date = _coerce_slate_date(
            attrs.get("start_time"),
            attrs.get("board_time"),
            attrs.get("end_time"),
            game_attrs.get("start_time"),
            game_attrs.get("end_time"),
        )
        if slate_date is None:
            continue
        rows.append({
            "game_date":   slate_date,
            "player":      player_name,
            "player_norm": _normalize_name(player_name),
            "prop":        prop_key,
            "line":        float(line),
            "num_books":   1,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Keep one line per slate/player/prop to avoid collapsing tomorrow's board
        # into today's run date when PrizePicks posts the next slate early.
        df = df.groupby(["game_date", "player", "prop"], as_index=False).agg(
            player_norm=("player_norm", "first"),
            line=("line", "median"),
            num_books=("num_books", "sum"),
        )
        df.sort_values(["game_date", "player", "prop"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


# ── Strategy A: Playwright with existing browser profile ──────────────────────

def _fetch_via_playwright_profile(profile_path: str, timeout_ms: int = 35_000) -> dict | None:
    """
    Use your existing Chrome/Chromium profile (which has valid PrizePicks
    cookies) to fetch the projections API response via network interception.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("playwright not installed. Run: pip install playwright && playwright install chromium")
        return None

    if not Path(profile_path).exists():
        logger.warning(f"Chrome profile not found at: {profile_path}")
        logger.warning("Set PRIZEPICKS_CHROME_PROFILE env variable to your Chrome profile path.")
        return None

    captured = {}

    def _handle_response(response):
        url = response.url
        if "prizepicks.com" in url:
            logger.info(f"  [network] {response.status} {url[:120]}")
        if "api.prizepicks.com/projections" in url or ("prizepicks.com" in url and "projection" in url.lower()):
            try:
                body = response.json()
                if "data" in body and not captured:
                    captured["data"] = body
                    logger.info(f"  Intercepted {len(body['data'])} projections from PrizePicks API")
            except Exception:
                pass

    # Use real Chrome if available — avoids bot detection that flags Playwright's Chromium
    chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not Path(chrome_exe).exists():
        chrome_exe = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

    launch_kwargs = dict(
        user_data_dir=profile_path,
        headless=False,   # visible window — harder to detect
        args=[
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--window-size=1280,800",
        ],
        ignore_default_args=["--enable-automation"],
    )
    if Path(chrome_exe).exists():
        launch_kwargs["executable_path"] = chrome_exe
        logger.info(f"  Using real Chrome: {chrome_exe}")

    logger.info(f"  Launching browser with profile: {profile_path}")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(**launch_kwargs)
        page = context.new_page()
        page.on("response", _handle_response)
        try:
            page.goto(
                f"https://app.prizepicks.com/board?leagueId={NBA_LEAGUE_ID}",
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            # Wait up to 15s for the API response to be intercepted
            for _ in range(30):
                if captured:
                    break
                page.wait_for_timeout(500)
        except Exception as exc:
            logger.warning(f"  Navigation warning: {exc}")
        context.close()

    return captured.get("data")


# ── Strategy B: requests with __cf_clearance cookie ───────────────────────────

def _fetch_via_cookie(cf_clearance: str) -> dict | None:
    """
    Use a manually supplied __cf_clearance cookie to bypass Cloudflare.
    Export this from your browser's DevTools after visiting app.prizepicks.com.
    """
    import requests as req

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://app.prizepicks.com/",
        "Origin": "https://app.prizepicks.com",
    }
    cookies = {"__cf_clearance": cf_clearance}
    params = {
        "league_id": NBA_LEAGUE_ID,
        "per_page": 250,
        "single_stat": "true",
    }
    try:
        resp = req.get(PP_API_URL, headers=headers, cookies=cookies, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data:
            logger.info(f"  Cookie strategy succeeded: {len(data['data'])} projections")
            return data
    except Exception as exc:
        logger.warning(f"  Cookie strategy failed: {exc}")
    return None


# ── Strategy 0: Direct API (no auth, works when Cloudflare isn't blocking) ────

def _fetch_direct() -> dict | None:
    """Try the PrizePicks API directly with multiple header variations."""
    import requests as req

    header_variants = [
        # Minimal headers — least detectable
        {"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        # Standard browser UA
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
         "Accept": "application/json"},
        # With referer
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
         "Accept": "application/json", "Referer": "https://app.prizepicks.com/"},
    ]
    params = {"league_id": NBA_LEAGUE_ID, "per_page": 500, "single_stat": "true", "game_mode": "pickem"}

    for i, headers in enumerate(header_variants):
        try:
            resp = req.get(PP_API_URL, headers=headers, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    logger.info(f"  Direct API succeeded (variant {i}): {len(data['data'])} projections")
                    return data
            else:
                logger.info(f"  Direct API variant {i}: HTTP {resp.status_code}")
        except Exception as exc:
            logger.warning(f"  Direct API variant {i} failed: {exc}")

    return None


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_todays_lines(chrome_profile: str | None = None,
                       cf_cookie: str | None = None) -> pd.DataFrame:
    """
    Fetch today's NBA prop lines from PrizePicks.
    Tries direct API first, then Playwright profile, then cookie fallback.
    """
    logger.info("Fetching today's PrizePicks NBA prop lines ...")

    cookie = cf_cookie or CF_CLEARANCE_COOKIE
    profile = chrome_profile or CHROME_PROFILE

    # Strategy 0: direct API (fastest, no browser needed) — works most days
    logger.info("  Trying direct API ...")
    data = _fetch_direct()

    # Strategy B: cookie fallback (needs a recently-exported __cf_clearance cookie)
    if data is None and cookie:
        logger.info("  Trying cookie fallback ...")
        data = _fetch_via_cookie(cookie)

    # Strategy A: Playwright with real Chrome profile (slowest, most reliable vs Cloudflare)
    # Requires: pip install playwright && playwright install chromium
    # Requires the machine to be in an interactive Windows session (not a locked screen).
    if data is None:
        logger.info("  Trying Playwright with Chrome profile (may open a browser window, ~30s) ...")
        data = _fetch_via_playwright_profile(profile)

    if data is None:
        logger.error(
            "Could not fetch PrizePicks data — all three strategies failed (direct, cookie, Playwright).\n"
            "  To fix this:\n"
            "  1. Make sure Chrome is open and you are logged into app.prizepicks.com\n"
            "  2. Set PRIZEPICKS_CHROME_PROFILE to your Chrome User Data directory\n"
            "  3. Or export __cf_clearance cookie and set PRIZEPICKS_CF_COOKIE=<value>"
        )
        return pd.DataFrame()

    df = _parse_projections(data)
    df, summary = sanitize_player_names(
        df,
        player_col="player",
        date_col="game_date",
        player_norm_col="player_norm",
        drop_unknown=True,
        require_roster=True,
    )
    logger.info(
        "Parsed %s validated NBA prop lines across %s slate date(s) (%s canonicalized, %s dropped)",
        len(df),
        df["game_date"].nunique(),
        summary["canonicalized_rows"],
        summary["dropped_rows"],
    )
    if summary["unknown_players"]:
        logger.warning("Dropped non-roster player names from PrizePicks feed: %s", ", ".join(summary["unknown_players"][:10]))
    return df


# ── Save / append ─────────────────────────────────────────────────────────────

def save_lines(output_path: str = "data/historical_lines.csv",
               chrome_profile: str | None = None,
               cf_cookie: str | None = None) -> str:
    """
    Fetch current PrizePicks lines and append them to the historical CSV.
    Idempotent per slate date: re-running replaces rows for the same game_date.
    Returns the absolute path of the saved file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fresh = fetch_todays_lines(chrome_profile=chrome_profile, cf_cookie=cf_cookie)

    if fresh.empty:
        logger.warning("No lines fetched; historical_lines.csv not updated.")
        return os.path.abspath(output_path)

    fresh = fresh.copy()
    fresh["num_books"] = pd.to_numeric(fresh.get("num_books", 1), errors="coerce").fillna(1).clip(lower=1)
    fresh["source"] = "prizepicks_scraper"
    fresh["snapshot_ts"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path, low_memory=False)
        if "player_norm" not in existing.columns and "player" in existing.columns:
            existing["player_norm"] = existing["player"].apply(_normalize_name)
        if "num_books" not in existing.columns:
            existing["num_books"] = 1
        if "source" not in existing.columns:
            existing["source"] = "legacy"
        if "snapshot_ts" not in existing.columns:
            existing["snapshot_ts"] = ""
        combined = pd.concat([existing, fresh], ignore_index=True)
    else:
        combined = fresh

    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined["line"] = pd.to_numeric(combined["line"], errors="coerce")
    combined["num_books"] = pd.to_numeric(combined["num_books"], errors="coerce").fillna(1).clip(lower=1)
    combined["source"] = combined["source"].fillna("unknown").astype(str)
    combined["snapshot_ts"] = combined["snapshot_ts"].replace("", pd.NA).fillna(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")).astype(str)
    combined = combined.dropna(subset=["game_date", "player", "player_norm", "prop", "line"])
    combined, summary = sanitize_player_names(
        combined,
        player_col="player",
        date_col="game_date",
        player_norm_col="player_norm",
        drop_unknown=True,
        require_roster=False,
    )
    combined["_snapshot_sort"] = pd.to_datetime(combined["snapshot_ts"], errors="coerce")
    combined = combined.sort_values(
        ["game_date", "player_norm", "prop", "num_books", "_snapshot_sort", "line"],
        ascending=[True, True, True, True, True, True],
    )
    combined = combined.drop_duplicates(subset=["game_date", "player_norm", "prop"], keep="last")
    combined = combined.drop(columns=["_snapshot_sort"], errors="ignore")
    combined.to_csv(output_path, index=False)
    logger.info(f"Prop lines saved -> {output_path}  ({len(combined):,} total rows)")
    if summary["canonicalized_rows"] or summary["dropped_rows"]:
        logger.info(
            "Historical line cleanup: %s canonicalized, %s dropped",
            summary["canonicalized_rows"],
            summary["dropped_rows"],
        )

    data_dir = Path(output_path).resolve().parent
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    for slate_date in sorted(fresh["game_date"].dropna().unique()):
        slate_fresh = fresh[fresh["game_date"] == slate_date].copy()
        morning_path = data_dir / f"lines_morning_{slate_date}.csv"
        if not morning_path.exists():
            slate_fresh[["player", "prop", "line", "player_norm"]].to_csv(morning_path, index=False)
            logger.info(f"Morning snapshot saved -> {morning_path.name}")

        archive_dir = data_dir / "line_archive" / slate_date
        archive_dir.mkdir(parents=True, exist_ok=True)
        slate_fresh.to_csv(archive_dir / f"lines_{slate_date}.csv", index=False)
        slate_fresh.to_csv(archive_dir / f"lines_{stamp}.csv", index=False)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    save_lines("data/historical_lines.csv")
