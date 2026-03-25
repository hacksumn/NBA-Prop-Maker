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


def _parse_projections(data: dict) -> pd.DataFrame:
    today_str = date.today().strftime("%Y-%m-%d")
    player_map = _extract_player_map(data.get("included", []))
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
        rows.append({
            "game_date":   today_str,
            "player":      player_name,
            "player_norm": _normalize_name(player_name),
            "prop":        prop_key,
            "line":        float(line),
            "num_books":   1,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Keep lowest line per player/prop (most conservative — avoids duplicates from multiple game modes)
        df = df.groupby(["player", "prop"], as_index=False).agg(
            game_date=("game_date", "first"),
            player_norm=("player_norm", "first"),
            line=("line", "median"),
            num_books=("num_books", "sum"),
        )
        df.sort_values(["player", "prop"], inplace=True)
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

    # Strategy 0: direct API (fastest, no browser needed) — works most days
    logger.info("  Trying direct API ...")
    data = _fetch_direct()

    # Strategy B: cookie fallback
    if data is None and cookie:
        logger.info("  Trying cookie fallback ...")
        data = _fetch_via_cookie(cookie)

    if data is None:
        logger.error(
            "Could not fetch PrizePicks data.\n"
            "  To fix this, do ONE of the following:\n"
            "  1. Set PRIZEPICKS_CHROME_PROFILE=/path/to/your/chrome/profile\n"
            "     (Visit app.prizepicks.com in that browser first to get valid cookies)\n"
            "  2. Export __cf_clearance cookie from your browser and set:\n"
            "     PRIZEPICKS_CF_COOKIE=<your_cookie_value>"
        )
        return pd.DataFrame()

    df = _parse_projections(data)
    logger.info(f"Parsed {len(df)} NBA prop lines for {date.today().strftime('%Y-%m-%d')}")
    return df


# ── Save / append ─────────────────────────────────────────────────────────────

def save_lines(output_path: str = "data/historical_lines.csv",
               chrome_profile: str | None = None,
               cf_cookie: str | None = None) -> str:
    """
    Fetch today's lines and append to the historical CSV.
    Idempotent: re-running on the same day replaces that day's rows.
    Returns the absolute path of the saved file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    today_str = date.today().strftime("%Y-%m-%d")
    fresh = fetch_todays_lines(chrome_profile=chrome_profile, cf_cookie=cf_cookie)

    if fresh.empty:
        logger.warning("No lines fetched; historical_lines.csv not updated.")
        return os.path.abspath(output_path)

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path, dtype=str)
        existing = existing[existing["game_date"] != today_str]
        combined = pd.concat([existing, fresh.astype(str)], ignore_index=True)
    else:
        combined = fresh

    combined.to_csv(output_path, index=False)
    logger.info(f"Prop lines saved -> {output_path}  ({len(combined):,} total rows)")
    return os.path.abspath(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    save_lines("data/historical_lines.csv")
