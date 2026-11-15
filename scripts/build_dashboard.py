#!/usr/bin/env python
"""Build the read-only Fresh Start NBA "Command Center" dashboard.

Reads the project's existing history / summary files and writes ONE self-contained HTML
file (default ``output/dashboard_pro.html``) with no network dependencies.

Safety contract (CLAUDE.md, DEC-070):
  * This script only READS ``data/``, ``models/``, ``output/`` and ``logs/``.
  * It writes exactly one file: the dashboard HTML (a derived view that can always be
    regenerated from the sources). It never touches a history, data or model file.
  * Every section is built inside an isolation wrapper. A missing or malformed input
    degrades that one section and is listed in the dashboard's warning banner; it never
    silently produces a number and never blocks the other sections.

Usage:
    python scripts/build_dashboard.py [--root PROJECT_ROOT] [--out OUTPUT_HTML] [--json]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_DIR = SCRIPT_DIR / "dashboard_assets"
LOGO_DIR = ASSET_DIR / "logos"       # committed <ABBR>.svg files, inlined into the page so it works offline
MAX_LOGO_BYTES = 64 * 1024
WALLPAPER_NAMES = ("wallpaper.jpg", "wallpaper.jpeg", "wallpaper.png", "wallpaper.webp")   # optional page background
MAX_WALLPAPER_BYTES = 1_500_000

# ---------------------------------------------------------------------------
# Era definitions. Boundaries come from DEC-036 (March rows were backfilled after the
# fact, not logged live) and DEC-038 (the live card moved from PrizePicks to FanDuel
# lines around 2026-05-11). 2026-27 is the current FanDuel season.
# ---------------------------------------------------------------------------
ERAS = [
    {"key": "backfill", "label": "Backfilled", "start": "2026-03-17", "end": "2026-03-29",
     "note": "Reconstructed after the fact (DEC-036). Not a forward record."},
    {"key": "live_pp", "label": "Live · PrizePicks", "start": "2026-03-30", "end": "2026-05-10",
     "note": "Real forward picks against PrizePicks lines."},
    {"key": "live_fd", "label": "Live · FanDuel (spring)", "start": "2026-05-11", "end": "2026-09-30",
     "note": "Card moved to FanDuel lines; UNDER-only engine met an efficient book (DEC-038)."},
    {"key": "season", "label": "2026-27 · FanDuel", "start": "2026-10-01", "end": "2099-12-31",
     "note": "Current season. FanDuel prices, UNDER-only AST/STL/BLK lanes."},
]
SCOPES = {
    "season": ["season"],
    "live": ["live_pp", "live_fd", "season"],
    "all": ["backfill", "live_pp", "live_fd", "season"],
}
SCOPE_LABELS = {
    "season": "2026-27 season",
    "live": "All live (excl. backfill)",
    "all": "Everything incl. backfill",
}

# NBA team metadata for logos. IDs are the official NBA team IDs (verified against TEAM_ID /
# team in data/nba_data.csv: all 30 abbreviations map one-to-one). Logos are read from
# scripts/dashboard_assets/logos/<ABBR>.svg and inlined into the page (see load_logos). A team with
# no usable file there falls back to the NBA's public CDN, and then to a text badge when offline;
# "color" is only that badge's tint.
TEAMS = {
    "ATL": (1610612737, "Atlanta Hawks", "#E03A3E"), "BOS": (1610612738, "Boston Celtics", "#1FA45B"),
    "CLE": (1610612739, "Cleveland Cavaliers", "#C23B6B"), "NOP": (1610612740, "New Orleans Pelicans", "#85714D"),
    "CHI": (1610612741, "Chicago Bulls", "#E0243C"), "DAL": (1610612742, "Dallas Mavericks", "#2F7FD1"),
    "DEN": (1610612743, "Denver Nuggets", "#FEC524"), "GSW": (1610612744, "Golden State Warriors", "#FFC72C"),
    "HOU": (1610612745, "Houston Rockets", "#E0243C"), "LAC": (1610612746, "LA Clippers", "#4A7BD0"),
    "LAL": (1610612747, "Los Angeles Lakers", "#FDB927"), "MIA": (1610612748, "Miami Heat", "#C8102E"),
    "MIL": (1610612749, "Milwaukee Bucks", "#2E8B57"), "MIN": (1610612750, "Minnesota Timberwolves", "#78BE20"),
    "BKN": (1610612751, "Brooklyn Nets", "#9AA0A6"), "NYK": (1610612752, "New York Knicks", "#F58426"),
    "ORL": (1610612753, "Orlando Magic", "#0077C0"), "IND": (1610612754, "Indiana Pacers", "#FDBB30"),
    "PHI": (1610612755, "Philadelphia 76ers", "#3A86D6"), "PHX": (1610612756, "Phoenix Suns", "#E56020"),
    "POR": (1610612757, "Portland Trail Blazers", "#E03A3E"), "SAC": (1610612758, "Sacramento Kings", "#7B4BB7"),
    "SAS": (1610612759, "San Antonio Spurs", "#C4CED4"), "OKC": (1610612760, "Oklahoma City Thunder", "#1B96E0"),
    "TOR": (1610612761, "Toronto Raptors", "#CE1141"), "UTA": (1610612762, "Utah Jazz", "#F9A01B"),
    "MEM": (1610612763, "Memphis Grizzlies", "#5D76A9"), "WAS": (1610612764, "Washington Wizards", "#E31837"),
    "DET": (1610612765, "Detroit Pistons", "#C8102E"), "CHA": (1610612766, "Charlotte Hornets", "#00A0C6"),
}

PRICE_BUCKETS = [  # (label, min_decimal inclusive, max_decimal exclusive)
    ("Plus money (+100 or better)", 2.0, 99.0),
    ("-100 to -120", 1.8333, 2.0),
    ("-121 to -140", 1.7143, 1.8333),
    ("-141 to -170", 1.5882, 1.7143),
    ("-171 to -200", 1.5, 1.5882),
]
PROB_BUCKETS = [("55-60%", 0.55, 0.60), ("60-65%", 0.60, 0.65), ("65-70%", 0.65, 0.70), ("70%+", 0.70, 1.01)]
EV_BUCKETS = [("0-5%", 0.0, 0.05), ("5-10%", 0.05, 0.10), ("10-20%", 0.10, 0.20), ("20%+", 0.20, 9.0)]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class Paths:
    """Resolved project locations. Everything is derived from one root so tests can
    point the builder at a synthetic project tree."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.output = self.root / "output"
        self.data = self.root / "data"
        self.models = self.root / "models"
        self.logs = self.root / "logs"


class Collector:
    """Runs sections in isolation and records anything that went wrong (fail loudly)."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def run(self, name: str, fn: Callable[[], Any], default: Any = None) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - isolation boundary, reported below
            msg = f"{name}: {type(exc).__name__}: {exc}"
            self.warnings.append(msg)
            print(f"[dashboard] WARNING {msg}", file=sys.stderr)
            return default


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None, None
    ph = k / n
    denom = 1 + z * z / n
    centre = ph + z * z / (2 * n)
    margin = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom


def american_to_decimal(a: Any) -> float | None:
    """Convert American odds to decimal odds (None if not a valid price)."""
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    if math.isnan(a) or a == 0:
        return None
    return 1 + a / 100.0 if a > 0 else 1 + 100.0 / abs(a)


def clean(obj: Any) -> Any:
    """Recursively convert numpy/pandas values to JSON-safe python (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_csv(path: Path, **kw: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing {path.name}")
    return pd.read_csv(path, low_memory=False, **kw)


def mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Picks: load, normalise, summarise
# ---------------------------------------------------------------------------
def assign_era(dates: pd.Series) -> pd.Series:
    """Map ISO date strings to era keys. Dates before the first era stay 'backfill'."""
    out = pd.Series("backfill", index=dates.index, dtype=object)
    for era in ERAS[1:]:
        out[dates >= era["start"]] = era["key"]
    return out


def load_picks(paths: Paths) -> pd.DataFrame:
    """Load picks_history.csv and add normalised helper columns (dec, units, era)."""
    p = read_csv(paths.output / "picks_history.csv")
    required = ["game_date", "player", "prop", "direction", "line", "result"]
    missing = [c for c in required if c not in p.columns]
    if missing:
        raise ValueError(f"picks_history.csv missing required columns: {missing}")
    # Rows with no date/player/prop cannot be picks. Drop from analytics, count, and report.
    unusable = p["game_date"].isna() | p["player"].isna() | p["prop"].isna() | p["direction"].isna()
    n_unusable = int(unusable.sum())
    p = p[~unusable].copy()
    p.attrs["unusable_rows"] = n_unusable
    for c in ["team", "opponent", "line_source", "pick_source", "publish_tier", "signal_grade", "published_at"]:
        if c not in p.columns:
            p[c] = np.nan
    for c in ["side_price", "side_decimal_odds", "ev_per_unit", "model_side_prob", "no_vig_side_prob",
              "confidence", "actual", "prediction", "l10_avg"]:
        p[c] = num(p[c]) if c in p.columns else np.nan

    p["game_date"] = p["game_date"].astype(str).str[:10]
    p["prop"] = p["prop"].astype(str).str.upper().str.strip()
    p["direction"] = p["direction"].astype(str).str.upper().str.strip()
    p["line"] = num(p["line"])
    res = p["result"]
    p["result"] = res.where(res.isna(), res.astype(str).str.upper().str.strip())
    p["era"] = assign_era(p["game_date"])

    # decimal odds: logged column, else derived from the American price
    dec = p["side_decimal_odds"].copy()
    from_price = p["side_price"].map(american_to_decimal)
    dec = dec.where(dec.notna(), from_price)
    p["price_source"] = np.where(dec.notna(), "logged", "none")

    # Fallback for early-season rows that were published before price columns existed:
    # the CLV report recorded the bet-time FanDuel price for the same pick.
    clv_path = paths.output / "clv_picks.csv"
    if clv_path.exists():
        clv = read_csv(clv_path)
        need = {"game_date", "player", "stat", "line", "direction", "bet_decimal"}
        if need.issubset(clv.columns):
            clv = clv.copy()
            clv["k"] = (clv["game_date"].astype(str).str[:10] + "|" + clv["player"].astype(str) + "|"
                        + clv["stat"].astype(str).str.upper() + "|" + num(clv["line"]).astype(str)
                        + "|" + clv["direction"].astype(str).str.upper())
            clv = clv.drop_duplicates("k").set_index("k")["bet_decimal"]
            keys = (p["game_date"] + "|" + p["player"].astype(str) + "|" + p["prop"] + "|"
                    + p["line"].astype(str) + "|" + p["direction"])
            fb = keys.map(clv).pipe(num)
            use = dec.isna() & fb.notna()
            dec = dec.where(~use, fb)
            p.loc[use, "price_source"] = "clv_fallback"
    p["dec"] = dec
    p["units"] = np.select(
        [p["result"].eq("WIN"), p["result"].eq("LOSS"), p["result"].eq("PUSH")],
        [dec - 1.0, -1.0, 0.0], default=np.nan)
    p.loc[dec.isna(), "units"] = np.nan  # no price -> no unit P&L (never guess)
    return p


def summarize(df: pd.DataFrame) -> dict:
    """Record, hit rate (Wilson CI) and flat 1-unit P&L at logged prices for a pick frame."""
    res = df["result"]
    w, l = int((res == "WIN").sum()), int((res == "LOSS").sum())
    pu, dnp, pend = int((res == "PUSH").sum()), int((res == "DNP").sum()), int(res.isna().sum())
    n = w + l
    lo, hi = wilson(w, n)
    priced = df[res.isin(["WIN", "LOSS"]) & df["dec"].notna()]
    units = float(priced["units"].sum()) if len(priced) else None
    roi = (units / len(priced)) if len(priced) else None
    out = {
        "n": n, "w": w, "l": l, "p": pu, "dnp": dnp, "pending": pend,
        "hit": (w / n) if n else None, "lo": lo, "hi": hi,
        "units": units, "priced_n": int(len(priced)), "roi": roi,
        "avg_model_prob": float(priced["model_side_prob"].mean()) if priced["model_side_prob"].notna().any() else None,
        "avg_no_vig": float(priced["no_vig_side_prob"].mean()) if priced["no_vig_side_prob"].notna().any() else None,
        "avg_ev": float(priced["ev_per_unit"].mean()) if priced["ev_per_unit"].notna().any() else None,
        "break_even": float((1.0 / priced["dec"]).mean()) if len(priced) else None,
        "priced_hit": (float((priced["result"] == "WIN").mean()) if len(priced) else None),
    }
    return out


def group_summaries(df: pd.DataFrame, cols: list[str], min_n: int = 1) -> list[dict]:
    out = []
    for key, g in df.groupby(cols, dropna=False, sort=False):
        s = summarize(g)
        if s["n"] < min_n:
            continue
        key = key if isinstance(key, tuple) else (key,)
        for c, k in zip(cols, key):
            s[c] = None if (isinstance(k, float) and math.isnan(k)) else k
        out.append(s)
    out.sort(key=lambda s: -s["n"])
    return out


def bucketed(df: pd.DataFrame, col: str, buckets: list[tuple[str, float, float]], priced_only: bool) -> list[dict]:
    g = df[df["result"].isin(["WIN", "LOSS"])]
    if priced_only:
        g = g[g["dec"].notna()]
    out = []
    for label, lo, hi in buckets:
        sub = g[(g[col] >= lo) & (g[col] < hi)]
        if len(sub):
            s = summarize(sub)
            s["label"] = label
            out.append(s)
    return out


def day_series(df: pd.DataFrame) -> dict:
    """Per-day results, cumulative curves, drawdown and streaks for a pick frame."""
    g = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]
    days = []
    cum_u = 0.0
    peak = 0.0
    max_dd = 0.0
    cw = cl = 0
    for date, d in g.groupby("game_date", sort=True):
        w = int((d["result"] == "WIN").sum())
        l = int((d["result"] == "LOSS").sum())
        pr = d[d["dec"].notna() & d["result"].isin(["WIN", "LOSS"])]
        u = float(pr["units"].sum()) if len(pr) else 0.0
        cum_u += u
        peak = max(peak, cum_u)
        max_dd = max(max_dd, peak - cum_u)
        cw += w
        cl += l
        days.append({
            "date": date, "w": w, "l": l, "p": int((d["result"] == "PUSH").sum()),
            "units": u, "priced": int(len(pr)), "cum_units": cum_u, "dd": peak - cum_u,
            "cum_w": cw, "cum_l": cl, "cum_hit": cw / (cw + cl) if (cw + cl) else None,
        })
    # day-level streak (a "green" day has more wins than losses)
    streak_type, streak_len = None, 0
    for d in reversed(days):
        kind = "W" if d["w"] > d["l"] else ("L" if d["w"] < d["l"] else "E")
        if kind == "E":
            break
        if streak_type is None:
            streak_type, streak_len = kind, 1
        elif kind == streak_type:
            streak_len += 1
        else:
            break
    green = sum(1 for d in days if d["w"] > d["l"])
    red = sum(1 for d in days if d["w"] < d["l"])
    best = max(days, key=lambda d: d["units"], default=None)
    worst = min(days, key=lambda d: d["units"], default=None)
    wl = g[g["result"].isin(["WIN", "LOSS"])].reset_index(drop=True)
    roll = []
    if len(wl) >= 10:
        win = (wl["result"] == "WIN").astype(float)
        r = win.rolling(20, min_periods=10).mean()
        for i in range(len(wl)):
            if not math.isnan(r.iloc[i]):
                roll.append({"i": i + 1, "date": wl.loc[i, "game_date"], "hit": float(r.iloc[i])})
    return {
        "days": days, "rolling20": roll,
        "max_drawdown": max_dd, "current_drawdown": (peak - cum_u) if days else 0.0,
        "streak": {"type": streak_type, "len": streak_len},
        "green_days": green, "red_days": red, "even_days": len(days) - green - red,
        "best_day": best, "worst_day": worst,
    }


def _with_team(rows: list[dict], df: pd.DataFrame) -> list[dict]:
    """Attach each player's most recent team by game date (for the logo next to the name).

    picks_history.csv is not guaranteed to be in date order (late backfills and re-grades are appended),
    so file order must not decide who a traded player belongs to."""
    last = df.dropna(subset=["team"]).sort_values("game_date", kind="stable").groupby("player")["team"].last()
    for r in rows:
        r["team"] = last.get(r["player"])
    return rows


def build_scope(p: pd.DataFrame, scope: str) -> dict:
    """The full analytics bundle for one scope (season / live / all)."""
    df = p[p["era"].isin(SCOPES[scope])]
    graded = df[df["result"].isin(["WIN", "LOSS"])]
    return {
        "label": SCOPE_LABELS[scope],
        "summary": summarize(df),
        "series": day_series(df),
        "by_prop": group_summaries(df, ["prop", "direction"]),
        "by_prop_only": group_summaries(df, ["prop"]),
        "by_price": bucketed(df, "dec", PRICE_BUCKETS, True),
        "by_prob": bucketed(df, "model_side_prob", PROB_BUCKETS, True),
        "by_ev": bucketed(df, "ev_per_unit", EV_BUCKETS, True),
        "by_line": [s for s in group_summaries(graded, ["prop", "line"], min_n=4)][:14],
        "players": _with_team(group_summaries(df, ["player"], min_n=3)[:18], df),
        "teams": group_summaries(df, ["team"], min_n=4)[:16],
        "teams_all": [r for r in group_summaries(df, ["team"], min_n=1) if r["team"]],      # logo board
        "opps": [r for r in group_summaries(df, ["opponent"], min_n=1) if r["opponent"]],   # vs each opponent
        "by_month": [dict(s, month=m) for m, g in df.assign(month=df["game_date"].str[:7]).groupby("month", sort=True)
                     for s in [summarize(g)]],
    }


def pick_rows(p: pd.DataFrame) -> list[dict]:
    """Compact per-pick rows for the explorer table."""
    rows = []
    for r in p.sort_values(["game_date"], kind="stable").itertuples(index=False):
        rows.append({
            "d": r.game_date, "pl": r.player, "tm": None if pd.isna(r.team) else r.team,
            "op": None if pd.isna(r.opponent) else r.opponent, "pr": r.prop, "di": r.direction,
            "ln": r.line, "px": None if pd.isna(r.side_price) else r.side_price,
            "dc": None if pd.isna(r.dec) else round(float(r.dec), 4),
            "ev": r.ev_per_unit, "mp": r.model_side_prob, "nv": r.no_vig_side_prob,
            "cf": r.confidence, "sg": None if pd.isna(r.signal_grade) else r.signal_grade,
            "rs": None if pd.isna(r.result) else r.result, "ac": r.actual,
            "u": r.units, "er": r.era, "ls": None if pd.isna(r.line_source) else r.line_source,
            "pt": None if pd.isna(r.publish_tier) else r.publish_tier,
        })
    return rows


# ---------------------------------------------------------------------------
# Today's card, slips, candidates
# ---------------------------------------------------------------------------
def build_today(paths: Paths, p: pd.DataFrame, build_date: str) -> dict:
    pend = p[p["result"].isna()]
    card_date = pend["game_date"].max() if len(pend) else p["game_date"].max()
    card = p[p["game_date"] == card_date].copy()
    latest = None
    latest_path = paths.output / "picks_latest.csv"
    if latest_path.exists():
        latest = read_csv(latest_path)
        latest["prop"] = latest["prop"].astype(str).str.upper()
        latest["direction"] = latest["direction"].astype(str).str.upper()
        latest_key = {(r.player, r.prop, r.direction): r for r in latest.itertuples(index=False)}
    else:
        latest_key = {}
    # Provisional outcomes: the CLV report grades from box scores as soon as they exist, while
    # picks_history.csv is graded the next morning (Step 1.5). Shown only as a labelled preview
    # on pending picks; they never enter any record or unit total.
    prov: dict[tuple, tuple] = {}
    clv_path = paths.output / "clv_picks.csv"
    if clv_path.exists():
        c = read_csv(clv_path)
        if {"game_date", "player", "stat", "line", "direction", "result", "actual"}.issubset(c.columns):
            for r in c.itertuples(index=False):
                res = str(r.result).upper()
                if res in ("WIN", "LOSS", "PUSH"):
                    prov[(str(r.game_date)[:10], r.player, str(r.stat).upper(), float(r.line), str(r.direction).upper())] = (res, r.actual)
    rows = []
    for r in card.itertuples(index=False):
        lt = latest_key.get((r.player, r.prop, r.direction))
        pv = prov.get((card_date, r.player, r.prop, float(r.line), r.direction)) if pd.isna(r.result) else None
        rows.append({
            "player": r.player, "team": None if pd.isna(r.team) else r.team,
            "opp": None if pd.isna(r.opponent) else r.opponent, "prop": r.prop, "dir": r.direction,
            "line": r.line, "price": None if pd.isna(r.side_price) else r.side_price,
            "ev": r.ev_per_unit, "mp": r.model_side_prob, "nv": r.no_vig_side_prob, "conf": r.confidence,
            "grade": None if pd.isna(r.signal_grade) else r.signal_grade,
            "tier": None if pd.isna(r.publish_tier) else r.publish_tier,
            "pred": r.prediction, "l10": r.l10_avg,
            "result": None if pd.isna(r.result) else r.result, "actual": r.actual,
            "prov": None if pv is None else {"result": pv[0], "actual": None if pd.isna(pv[1]) else float(pv[1])},
            "published_at": None if pd.isna(r.published_at) else str(r.published_at),
            "in_latest": lt is not None,
            "latest_price": (None if lt is None or pd.isna(getattr(lt, "side_price", np.nan)) else float(lt.side_price)),
            "tip": (None if lt is None or pd.isna(getattr(lt, "commence_time", np.nan)) else str(lt.commence_time)),
            "sigma": (None if lt is None or pd.isna(getattr(lt, "sigma", np.nan)) else float(lt.sigma)),
            "meta": (None if lt is None or pd.isna(getattr(lt, "meta_prob", np.nan)) else float(lt.meta_prob)),
        })
    dropped = [r["player"] for r in rows if not r["in_latest"]] if latest is not None else []
    # one entry per game on the card (home/away is not recorded in the pick files, so a/b is just
    # "first team seen" and "its opponent")
    games: dict[tuple, dict] = {}
    for r in rows:
        if not r["team"] or not r["opp"]:
            continue
        g = games.setdefault(tuple(sorted((r["team"], r["opp"]))),
                             {"a": r["team"], "b": r["opp"], "n": 0, "w": 0, "l": 0, "pw": 0, "pl": 0, "pending": 0, "tip": None})
        g["n"] += 1
        if r["result"] == "WIN":
            g["w"] += 1
        elif r["result"] == "LOSS":
            g["l"] += 1
        elif r["prov"] and r["prov"]["result"] == "WIN":
            g["pw"] += 1
        elif r["prov"] and r["prov"]["result"] == "LOSS":
            g["pl"] += 1
        elif r["result"] is None:
            g["pending"] += 1
        if r["tip"] and (g["tip"] is None or r["tip"] < g["tip"]):
            g["tip"] = r["tip"]
    pv_rows = [r["prov"]["result"] for r in rows if r["prov"]]
    provisional = {"n": len(pv_rows), "w": pv_rows.count("WIN"), "l": pv_rows.count("LOSS"), "p": pv_rows.count("PUSH")}
    return {
        "card_date": card_date, "build_date": build_date, "n": len(rows), "provisional": provisional,
        "games": sorted(games.values(), key=lambda g: (g["tip"] is None, g["tip"] or "", g["a"])),
        "pending": int(card["result"].isna().sum()), "picks": rows, "dropped_from_latest": dropped,
        "latest_rows": int(len(latest)) if latest is not None else None,
    }


def slip_legs(r: Any) -> list[dict]:
    legs = []
    for i in range(1, 5):
        pl = r.get(f"Pick_{i}_Player")
        if pl is None or (isinstance(pl, float) and math.isnan(pl)):
            continue
        px = r.get(f"Pick_{i}_Price")
        legs.append({"player": pl, "prop": r.get(f"Pick_{i}_Prop"), "dir": r.get(f"Pick_{i}_Dir"),
                     "line": r.get(f"Pick_{i}_Line"), "price": None if px is None or pd.isna(px) else px})
    return legs


def build_slips(paths: Paths) -> dict:
    """Betslip record and dollars. Net P&L = payout - wager on graded rows.

    The pipeline logs the same legs under two slip labels (e.g. "Power Play" and
    "UNDER-Only"). Raw rows and de-duplicated slips are both reported so a double
    count can never inflate the record unnoticed.
    """
    b = read_csv(paths.output / "betslips_history.csv")
    b["game_date"] = b["game_date"].astype(str).str[:10]
    b["wager"] = num(b["wager"])
    b["actual_payout"] = num(b["actual_payout"])
    b["potential_payout"] = num(b["potential_payout"])
    b["result"] = b["result"].where(b["result"].isna(), b["result"].astype(str).str.upper())
    b["legs"] = b.apply(lambda r: tuple((l["player"], l["prop"], l["dir"], l["line"]) for l in slip_legs(r)), axis=1)
    # A row with no date or no legs cannot be a real slip (seen in the wild: a lone
    # "LOSS / 0.0" row). Exclude it from every number and report it; never edit the file.
    orphan = b["game_date"].isin(["nan", "NaT", "None", ""]) | b["legs"].map(len).eq(0)
    orphans = {"n": int(orphan.sum()), "csv_lines": [int(i) + 2 for i in b.index[orphan]]}  # header is line 1
    b = b[~orphan].copy()
    b["key"] = b["game_date"] + "|" + b["legs"].astype(str)
    b["net"] = np.where(b["result"].isin(["WIN", "LOSS", "PUSH"]) & b["actual_payout"].notna(),
                        b["actual_payout"] - b["wager"], np.nan)
    b["era"] = assign_era(b["game_date"])

    def agg(df: pd.DataFrame) -> dict:
        r = df["result"]
        w, l, pu = int((r == "WIN").sum()), int((r == "LOSS").sum()), int((r == "PUSH").sum())
        settled = df[df["net"].notna()]
        staked = float(settled["wager"].sum())
        net = float(settled["net"].sum()) if len(settled) else 0.0
        lo, hi = wilson(w, w + l)
        return {"n": int(len(df)), "w": w, "l": l, "p": pu, "pending": int(r.isna().sum()),
                "hit": (w / (w + l)) if (w + l) else None, "lo": lo, "hi": hi,
                "staked": staked, "net": net, "roi": (net / staked) if staked else None}

    dedup = b.drop_duplicates("key", keep="first")
    by_type = []
    for t, g in b.groupby("slip_type", sort=False):
        s = agg(g)
        s["slip_type"] = t
        by_type.append(s)
    # cumulative dollars by day for de-duplicated slips
    daily = []
    cum = 0.0
    for d, g in dedup[dedup["net"].notna()].groupby("game_date", sort=True):
        day_net = float(g["net"].sum())
        cum += day_net
        daily.append({"date": d, "net": day_net, "cum": cum, "n": int(len(g))})
    recent = []
    for r in b.sort_values("game_date", kind="stable").tail(60).to_dict("records"):
        recent.append({"date": r["game_date"], "type": r["slip_type"], "wager": r["wager"],
                       "potential": r["potential_payout"], "payout": r["actual_payout"], "result": r["result"],
                       "legs": slip_legs(r), "joint": r.get("joint_prob_adj"), "ev": r.get("ev_per_unit")})
    eras = []
    for e in ERAS:
        g = b[b["era"] == e["key"]]
        if len(g):
            s = agg(g.drop_duplicates("key"))
            s["era"] = e["key"]
            eras.append(s)
    latest = []
    lp = paths.output / "betslips_latest.csv"
    if lp.exists():
        bl = read_csv(lp)
        latest = [{"type": r["slip_type"], "wager": r.get("wager"), "potential": r.get("potential_payout"),
                   "joint": r.get("joint_prob_adj"), "be": r.get("break_even_joint_prob"), "ev": r.get("ev_per_unit"),
                   "legs": slip_legs(r)} for r in bl.to_dict("records")]
    return {"raw": agg(b), "unique": agg(dedup), "by_type": by_type, "daily": daily, "recent": recent,
            "by_era": eras, "latest": latest, "duplicates": int(len(b) - len(dedup)), "orphans": orphans}


GATE_ORDER = ["Published", "Favorite cap", "Min EV", "Meta floor", "Volatility", "Tier / diversity",
              "Fallback source", "Other"]


def gate_category(bucket: str, failures: Any) -> str:
    """Primary human-readable reason a candidate did or did not reach the card."""
    if bucket == "published":
        return "Published"
    g = "" if failures is None or (isinstance(failures, float) and math.isnan(failures)) else str(failures)
    if "beyond" in g:
        return "Favorite cap"
    if re.search(r"EV\s*-?\d", g):
        return "Min EV"
    if "meta" in g:
        return "Meta floor"
    if "vol" in g:
        return "Volatility"
    if "tier selection" in g:
        return "Tier / diversity"
    if "fallback" in g:
        return "Fallback source"
    return "Other"


def build_gates(paths: Paths) -> dict:
    c = read_csv(paths.output / "candidates_history.csv")
    c["game_date"] = c["game_date"].astype(str).str[:10]
    c["prop"] = c["prop"].astype(str).str.upper()
    c["result"] = c["result"].where(c["result"].isna(), c["result"].astype(str).str.upper())
    c["dec"] = num(c["side_decimal_odds"])
    c["units"] = num(c["profit_units"])
    c.loc[c["dec"].isna(), "units"] = np.nan
    for col in ["model_side_prob", "no_vig_side_prob", "ev_per_unit"]:
        c[col] = num(c[col])
    c["category"] = [gate_category(b, f) for b, f in zip(c["gate_bucket"], c["gate_failures"])]

    buckets = []
    for b in ["published", "blocked_other", "blocked_price"]:
        g = c[c["gate_bucket"] == b]
        if len(g):
            s = summarize(g)
            s["bucket"] = b
            buckets.append(s)
    cats = []
    for cat in GATE_ORDER:
        g = c[c["category"] == cat]
        if len(g):
            s = summarize(g)
            s["category"] = cat
            cats.append(s)
    # model vs market probability quality on graded candidates (Brier, lower = better)
    gg = c[c["result"].isin(["WIN", "LOSS"]) & c["model_side_prob"].notna() & c["no_vig_side_prob"].notna()]
    y = (gg["result"] == "WIN").astype(float)
    brier = None
    rel = []
    if len(gg):
        brier = {"n": int(len(gg)),
                 "model": float(((gg["model_side_prob"] - y) ** 2).mean()),
                 "market": float(((gg["no_vig_side_prob"] - y) ** 2).mean())}
        edges = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (gg["model_side_prob"] >= lo) & (gg["model_side_prob"] < hi)
            if m.sum():
                rel.append({"label": f"{int(lo*100)}-{int(hi*100)}%", "n": int(m.sum()),
                            "pred": float(gg.loc[m, "model_side_prob"].mean()),
                            "mkt": float(gg.loc[m, "no_vig_side_prob"].mean()),
                            "act": float(y[m].mean())})
    by_day = []
    for d, g in c.groupby("game_date", sort=True):
        row = {"date": d}
        for b in ["published", "blocked_other", "blocked_price"]:
            sub = g[g["gate_bucket"] == b]
            s = summarize(sub)
            row[b] = {"w": s["w"], "l": s["l"], "units": s["units"] or 0.0, "n": int(len(sub))}
        by_day.append(row)

    board = []
    lp = paths.output / "candidates_latest.csv"
    if lp.exists():
        cl = read_csv(lp)
        for r in cl.to_dict("records"):
            gf = r.get("gate_failures")
            gf = None if gf is None or (isinstance(gf, float) and math.isnan(gf)) else str(gf)
            reason = None if gf is None else gate_category("blocked", gf)
            board.append({
                "player": r.get("player"), "team": r.get("team"), "opp": r.get("opponent"),
                "prop": str(r.get("prop")).upper(), "dir": r.get("direction"), "line": r.get("line"),
                "price": r.get("side_price"), "ev": r.get("ev_per_unit"), "mp": r.get("model_side_prob"),
                "failure": gf, "category": reason,
                "passed": (gf is None) or ("tier selection" in gf and "price" not in gf and "meta" not in gf),
            })
    return {"buckets": buckets, "categories": cats, "brier": brier, "reliability": rel,
            "by_day": by_day, "board": board, "dates": [c["game_date"].min(), c["game_date"].max()],
            "rows": int(len(c))}


# ---------------------------------------------------------------------------
# CLV
# ---------------------------------------------------------------------------
def build_clv(paths: Paths) -> dict:
    summ = load_json(paths.output / "clv_summary.json")
    picks = read_csv(paths.output / "clv_picks.csv")
    for col in ["ev_vs_close", "no_vig_move", "line_clv", "bet_no_vig", "close_no_vig", "profit_units"]:
        if col in picks.columns:
            picks[col] = num(picks[col])
    exact = picks[picks["close_quality"].eq("exact_line")] if "close_quality" in picks.columns else picks.iloc[0:0]
    hist = []
    if len(exact):
        edges = np.linspace(-0.12, 0.08, 11)
        counts, _ = np.histogram(exact["ev_vs_close"].dropna().clip(-0.1199, 0.0799), bins=edges)
        hist = [{"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(counts[i])} for i in range(len(counts))]
    scatter = [{"b": float(r.bet_no_vig), "c": float(r.close_no_vig), "stat": str(r.stat).upper(),
                "res": None if pd.isna(r.result) else str(r.result)}
               for r in exact.itertuples(index=False) if pd.notna(r.bet_no_vig) and pd.notna(r.close_no_vig)]
    return {"generated_at": summ.get("generated_at"), "start_date": summ.get("start_date"),
            "picks": summ.get("picks"), "model_leans": summ.get("model_leans"),
            "hist": hist, "scatter": scatter, "definition": summ.get("definition")}


# ---------------------------------------------------------------------------
# Model Lab
# ---------------------------------------------------------------------------
def build_models(paths: Paths, col: Collector) -> dict:
    out: dict[str, Any] = {}
    tr = col.run("models/training_results.json", lambda: load_json(paths.models / "training_results.json"), {})
    legacy = col.run("models/results.json", lambda: load_json(paths.models / "results.json"), {})
    audit = col.run("output/production_holdout_audit.json",
                    lambda: load_json(paths.output / "production_holdout_audit.json"), {})
    audit_by = {r["prop"]: r for r in (audit.get("results") or [])}
    props = []
    for k, v in (tr or {}).items():
        up = k.upper()
        meta = v.get("meta_summary") or {}
        a = audit_by.get(up) or {}
        props.append({
            "prop": up, "mae": v.get("avg_mae"), "rmse": v.get("avg_rmse"),
            "real_hit": (v.get("real_line_summary") or {}).get("hit_rate"),
            "n_real": v.get("n_real_lines"), "n_samples": v.get("n_samples"),
            "n_feat": v.get("n_features_selected"), "n_feat_orig": v.get("n_features_original"),
            "meta_auc": meta.get("auc"), "top_hit": meta.get("top_prob_hit_rate"),
            "top_n": meta.get("top_prob_n"), "mkt_sign": v.get("market_edge_sign_accuracy"),
            "legacy_acc": (legacy.get(k) or {}).get("accuracy"),
            "audit_n": a.get("graded_picks"), "audit_model": a.get("model_accuracy"),
            "audit_under": a.get("always_under_accuracy"), "audit_edge": a.get("edge_vs_always_under"),
            "audit_status": a.get("status"),
        })
    out["props"] = props
    out["audit_ts"] = audit.get("timestamp")

    def diag():
        d = load_json(paths.models / "probability_diagnostics_advanced.json")
        rows = []
        for k in sorted((d.get("holdout") or {}).keys()):
            h = d["holdout"][k]
            rows.append({"prop": k.upper(), "n": h.get("n_rows"),
                         "brier_raw": (h.get("raw") or {}).get("brier"),
                         "brier_cal": (h.get("calibrated") or {}).get("brier"),
                         "ece_raw": (h.get("raw") or {}).get("ece"),
                         "ece_cal": (h.get("calibrated") or {}).get("ece"),
                         "base_rate": (h.get("raw") or {}).get("positive_rate")})
        return {"split": d.get("split"), "rows": rows}
    out["calibration"] = col.run("models/probability_diagnostics_advanced.json", diag, {})

    def feats():
        fi = load_json(paths.models / "feature_importance_advanced.json")
        return {k.upper(): [{"f": f, "v": float(v)} for f, v in
                            sorted(m.items(), key=lambda kv: -float(kv[1]))[:10]] for k, m in fi.items()}
    out["features"] = col.run("models/feature_importance_advanced.json", feats, {})

    def lanes():
        u = load_json(paths.output / "probability_under_selection_summary.json")
        res = {}
        for part in ("development", "holdout"):
            blk = (u.get("results") or {}).get(part) or {}
            res[part] = {"combined": blk.get("production_lanes_combined"),
                         "per_target": blk.get("per_target")}
        return {"floors": u.get("declared_floors"), "lanes": u.get("production_lanes"),
                "dev": u.get("development"), "hold": u.get("holdout"), "results": res,
                "caveat": u.get("scope_caveat")}
    out["lanes"] = col.run("output/probability_under_selection_summary.json", lanes, {})

    def cand():
        c = load_json(paths.output / "candidate_future_holdout_summary.json")
        return {"id": c.get("candidate_id"), "start": c.get("future_holdout_start"),
                "min_dates": c.get("minimum_dates"), "min_rows": c.get("minimum_rows"),
                "promoted": c.get("production_promotion_performed"),
                "rows": (c.get("eligibility") or {}).get("evaluation_contracts"),
                "results": [{"prop": r["prop"].upper(), "rows": r.get("rows"), "dates": r.get("dates"),
                             "cand": r.get("candidate_accuracy"), "prod": r.get("production_accuracy"),
                             "under": r.get("always_under_accuracy"),
                             "vs_prod": r.get("candidate_minus_production"),
                             "vs_prod_ci": r.get("candidate_minus_production_ci95"),
                             "vs_under": r.get("candidate_minus_always_under"),
                             "vs_under_ci": r.get("candidate_minus_always_under_ci95")}
                            for r in c.get("results", [])]}
    out["candidate"] = col.run("output/candidate_future_holdout_summary.json", cand, {})

    def gexp():
        g = load_json(paths.output / "gate_experiment_summary.json")
        return {"status": g.get("status"), "start": g.get("start_date"), "dates": g.get("completed_dates"),
                "contracts": g.get("binary_contracts"), "review_eligible": g.get("review_eligible"),
                "comparisons": g.get("comparisons"), "excluded": g.get("runs_excluded_for_drift")}
    out["gate_experiment"] = col.run("output/gate_experiment_summary.json", gexp, {})

    def bt():
        d = read_csv(paths.output / "backtest_daily_summary.csv")
        res = []
        for k in (1, 3, 5, 10):
            w = float(num(d[f"top_{k}_wins"]).sum())
            l = float(num(d[f"top_{k}_losses"]).sum())
            res.append({"k": k, "w": w, "l": l, "hit": (w / (w + l)) if (w + l) else None})
        return {"days": int(len(d)), "start": str(d["game_date"].min()), "end": str(d["game_date"].max()), "tops": res}
    out["backtest"] = col.run("output/backtest_daily_summary.csv", bt, {})
    return out


# ---------------------------------------------------------------------------
# System health: run logs, data freshness, integrity
# ---------------------------------------------------------------------------
LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[(\w+)\]\s?(.*)$")
STEP_RE = re.compile(r"STEP\s+([\d.]+)\s+\S+\s+(.+?)(?:\s{2,}\(.*)?$")
CREDIT_RE = re.compile(r"(\d[\d,]*)\s+(?:provider\s+)?credits\s+(?:remaining|before)")


def _parse_log(path: Path) -> list[dict]:
    """Split one run log into run blocks, each a list of pipeline steps with warning/error counts."""
    runs: list[dict] = []
    cur_run: dict | None = None
    step: dict | None = None
    prev_dt: datetime | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LOG_TS.match(line.rstrip("\n"))
            if not m:
                continue
            ts, level, msg = m.groups()
            dt = datetime.fromisoformat(ts)
            # Other jobs can append to the same daily log hours later. A gap over 20 minutes
            # ends the run's measured wall time (the lines are still scanned for warnings).
            in_run = prev_dt is None or (dt - prev_dt) <= timedelta(minutes=20)
            prev_dt = dt
            if not in_run and cur_run is not None:
                cur_run["closed"] = True  # later lines (e.g. the next run's banner) are not this run
            sm = STEP_RE.search(msg)
            if sm and msg.lstrip().startswith("STEP"):
                sid, name = sm.group(1), sm.group(2).strip()
                if sid == "1":  # new run block
                    cur_run = {"file": path.name, "start": ts, "steps": [], "end": ts}
                    runs.append(cur_run)
                if cur_run is None:
                    cur_run = {"file": path.name, "start": ts, "steps": [], "end": ts}
                    runs.append(cur_run)
                step = {"id": sid, "name": name, "start": ts, "warn": 0, "err": 0, "msgs": []}
                cur_run["steps"].append(step)
                continue
            if cur_run is not None and not cur_run.get("closed"):
                cur_run["end"] = ts
            if step is not None and "already started" in msg:
                step["tipped"] = True  # every game had tipped when the line fetch ran
            if step is not None and level in ("WARNING", "ERROR", "CRITICAL"):
                key = "err" if level in ("ERROR", "CRITICAL") else "warn"
                step[key] += 1
                if len(step["msgs"]) < 3:
                    step["msgs"].append({"level": level, "text": msg.strip()[:260]})
    return runs


def build_health(paths: Paths, p: pd.DataFrame, build_dt: datetime, col: Collector) -> dict:
    h: dict[str, Any] = {}

    def runs():
        logs = sorted(paths.logs.glob("run_*.log"))
        if not logs:
            raise FileNotFoundError("no logs/run_*.log files")
        all_runs: list[dict] = []
        credits: list[dict] = []
        for lf in logs[-45:]:
            all_runs.extend(_parse_log(lf))
            with open(lf, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = LOG_TS.match(line.rstrip("\n"))
                    if not m:
                        continue
                    cm = CREDIT_RE.search(m.group(3))
                    if cm:
                        credits.append({"t": m.group(1), "credits": int(cm.group(1).replace(",", ""))})
        # per-run durations & statuses
        for r in all_runs:
            st = r["steps"]
            for i, s in enumerate(st):
                nxt = st[i + 1]["start"] if i + 1 < len(st) else r["end"]
                s["secs"] = max(0, int((datetime.fromisoformat(nxt) - datetime.fromisoformat(s["start"])).total_seconds()))
            r["warn"] = sum(s["warn"] for s in st)
            r["err"] = sum(s["err"] for s in st)
            r["secs"] = max(0, int((datetime.fromisoformat(r["end"]) - datetime.fromisoformat(r["start"])).total_seconds()))
            r["status"] = "fail" if r["err"] else ("warn" if r["warn"] else "ok")
            # An evening rerun after every game has tipped legitimately finds no pre-game lines
            # (DEC-068 skips started events). Label it instead of showing an unexplained FAIL.
            other_err = sum(s["err"] for s in st if s["id"] != "2")
            if r["err"] and not other_err and any(s.get("tipped") for s in st):
                r["status"] = "late"
                r["note"] = "Late rerun: every game had already tipped, so no pre-game lines were fetched (expected, DEC-068)."
        cutoff = (build_dt - timedelta(days=45)).strftime("%Y-%m-%d")
        credits = sorted({(c["t"], c["credits"]) for c in credits if c["t"][:10] >= cutoff})
        recent = all_runs[-14:]
        # step detail only for the newest runs keeps the payload small
        return {"last_run": all_runs[-1] if all_runs else None,
                "recent_runs": [r if i >= len(recent) - 6 else {k: v for k, v in r.items() if k != "steps"}
                                for i, r in enumerate(recent)],
                "n_runs": len(all_runs),
                "credits": [{"t": t, "credits": c} for t, c in credits[-120:]]}
    h["pipeline"] = col.run("logs/run_*.log", runs, {})

    def data_files():
        out = {}
        nba = pd.read_csv(paths.data / "nba_data.csv", usecols=["game_date", "player", "season"], low_memory=False)
        out["nba_data"] = {"rows": int(len(nba)), "players": int(nba["player"].nunique()),
                           "last_date": str(nba["game_date"].astype(str).str[:10].max()),
                           "seasons": {str(k): int(v) for k, v in nba["season"].value_counts().sort_index().items()}}
        hl_cols = pd.read_csv(paths.data / "historical_lines.csv", nrows=0).columns
        use = [c for c in ("game_date", "source", "player_norm") if c in hl_cols]
        hl = pd.read_csv(paths.data / "historical_lines.csv", usecols=use, low_memory=False)
        out["lines"] = {"rows": int(len(hl)), "last_date": str(hl["game_date"].astype(str).str[:10].max()),
                        "by_source": ({str(k): int(v) for k, v in hl["source"].fillna("unlabeled").value_counts().items()}
                                      if "source" in hl.columns else {})}
        return out
    h["data"] = col.run("data/nba_data.csv + historical_lines.csv", data_files, {})

    def freshness():
        files = ["picks_history.csv", "picks_latest.csv", "betslips_history.csv", "candidates_history.csv",
                 "clv_summary.json", "gate_experiment_summary.json", "grade_reconciliation_summary.json",
                 "training_maintenance_latest.json"]
        res = []
        for f in files:
            pth = paths.output / f
            mt = mtime_iso(pth)
            age = None if mt is None else round((build_dt - datetime.fromisoformat(mt)).total_seconds() / 3600, 1)
            res.append({"file": f, "mtime": mt, "age_h": age})
        return res
    h["files"] = col.run("file freshness", freshness, [])

    def maintenance():
        t = load_json(paths.output / "training_maintenance_latest.json")
        return {"status": t.get("status"), "mode": t.get("mode"), "ended_at": t.get("ended_at"),
                "production_unchanged": t.get("production_unchanged"),
                "artifacts_hashed": len(t.get("production_hashes_after") or {}),
                "profile": t.get("training_profile") if isinstance(t.get("training_profile"), str) else None}
    h["maintenance"] = col.run("output/training_maintenance_latest.json", maintenance, {})

    def recon():
        g = load_json(paths.output / "grade_reconciliation_summary.json")
        return {"checked": g.get("checked_history_rows"), "current_conflicts": g.get("current_conflicts"),
                "outcome_conflicts": g.get("outcome_conflicts"),
                "conflicts": [{k: c.get(k) for k in ("source_file", "game_date", "player", "prop", "stored_actual",
                                                      "observed_actual", "stored_result", "observed_result")}
                              for c in (g.get("conflicts") or [])[:8]]}
    h["reconciliation"] = col.run("output/grade_reconciliation_summary.json", recon, {})

    # integrity flags derived straight from pick history
    flags = []
    by_day = p.groupby("game_date")["result"].agg(lambda s: (int(s.eq("DNP").sum()), int(len(s))))
    for d, (dnp, n) in by_day.items():
        if n >= 3 and dnp == n:
            flags.append({"kind": "all_dnp", "date": d, "n": n,
                          "text": f"All {n} picks on {d} are stored as DNP, which usually means box scores for that date are missing.",
                          "hint": "Needs your decision under DEC-070 (see TASKS.md); the dashboard changes nothing."})
    cutoff = (build_dt - timedelta(days=3)).strftime("%Y-%m-%d")
    stale = p[p["result"].isna() & (p["game_date"] < cutoff)]
    if len(stale):
        flags.append({"kind": "ungraded", "date": str(stale["game_date"].min()), "n": int(len(stale)),
                      "text": f"{len(stale)} picks older than 3 days are still ungraded.",
                      "hint": "Check Step 1.5 grading and that box scores exist for those dates."})
    if p.attrs.get("unusable_rows"):
        flags.append({"kind": "unusable_picks", "n": int(p.attrs["unusable_rows"]),
                      "text": f"{p.attrs['unusable_rows']} rows in picks_history.csv have no date/player/prop/direction and were excluded.",
                      "hint": "Read-only: the file was not modified."})
    h["flags"] = flags

    def git():
        def run(*a: str) -> str:
            return subprocess.run(["git", *a], cwd=paths.root, capture_output=True, text=True, timeout=10).stdout.strip()
        return {"branch": run("rev-parse", "--abbrev-ref", "HEAD"), "commit": run("rev-parse", "--short", "HEAD"),
                "subject": run("log", "-1", "--format=%s")[:120],
                "dirty": len([l for l in run("status", "--porcelain").splitlines() if l.strip()])}
    h["git"] = col.run("git metadata", git, {})
    return h


# ---------------------------------------------------------------------------
# Decision log and open tasks (project memory)
# ---------------------------------------------------------------------------
def parse_decisions(path: Path) -> list[dict]:
    """Parse DECISIONS.md headings of every historical style into id/title/date/status."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    seen: set[str] = set()
    for i, ln in enumerate(lines):
        if not ln.startswith("## "):
            continue
        m = re.search(r"DEC-(\d{3})", ln)
        if not m:
            continue
        did = f"DEC-{m.group(1)}"
        title = re.sub(r"\[?DEC-\d{3}\]?", "", ln[3:])
        title = re.sub(r"\d{4}-\d{2}-\d{2}", "", title)
        title = re.sub(r"\(\s*\)", "", title)
        title = re.sub(r"^[\s\-—–?:]+|[\s\-—–?:]+$", "", title)
        title = re.sub(r"\s+[\-—–?]\s+", " — ", title).strip(" —")
        title = re.sub(r"\s{2,}", " ", title)
        window = "\n".join(lines[i:i + 10])
        dm = re.search(r"(\d{4}-\d{2}-\d{2})", ln) or re.search(r"Date:\**\s*(\d{4}-\d{2}-\d{2})", window)
        sm = re.search(r"Status:\**\s*([A-Za-z][A-Za-z \-]+)", window)
        if did in seen:
            continue
        seen.add(did)
        out.append({"id": did, "title": title[:170], "date": dm.group(1) if dm else None,
                    "status": sm.group(1).strip() if sm else None, "line": i + 1})
    out.sort(key=lambda d: d["id"], reverse=True)
    return out


def parse_tasks(path: Path) -> list[dict]:
    """Open/closed items from the NOW section of TASKS.md."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    in_now = False
    for ln in text:
        if ln.startswith("## "):
            in_now = ln.strip().lower().startswith("## now")
            continue
        if not in_now:
            continue
        m = re.match(r"^- \[( |x|X)\]\s+(.*)$", ln)
        if not m:
            continue
        body = m.group(2)
        tag = None
        tm = re.match(r"^\*\*\[([^\]]+)\]\s*(.*)$", body)
        if tm:
            tag, body = re.sub(r"[`*]", "", tm.group(1)).strip(), tm.group(2)
        body = re.sub(r"\*\*", "", body)
        body = re.sub(r"`", "", body)
        out.append({"done": m.group(1).lower() == "x", "tag": tag, "text": body[:330]})
    return out


# ---------------------------------------------------------------------------
# Auto-generated insights (every sentence is computed from the data above)
# ---------------------------------------------------------------------------
def pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def build_insights(d: dict) -> list[dict]:
    ins: list[dict] = []
    sc = (d.get("scopes") or {}).get("season")
    if sc:
        s = sc["summary"]
        if s["n"]:
            tone = "good" if (s["units"] or 0) > 0 else "bad"
            txt = (f"2026-27 record is {s['w']}-{s['l']} ({pct(s['hit'])}, 95% CI {pct(s['lo'], 0)}-{pct(s['hi'], 0)})"
                   f" with {s['units']:+.2f}u at logged FanDuel prices over {s['priced_n']} priced bets"
                   f" (ROI {pct(s['roi'])}).") if s["units"] is not None else \
                f"2026-27 record is {s['w']}-{s['l']} ({pct(s['hit'])}); no priced bets to compute units."
            ins.append({"tone": tone, "title": "Season scoreboard", "text": txt})
            if s["break_even"] and s["priced_hit"] is not None:
                gap = (s["priced_hit"] - s["break_even"]) * 100
                low_n = " Sample is small, so treat as a hint, not proof." if s["priced_n"] < 150 else ""
                ins.append({"tone": "good" if gap > 0 else "bad", "title": "Hit rate vs break-even",
                            "text": f"Average break-even at these prices is {pct(s['break_even'])}; the hit rate on priced bets is "
                                    f"{pct(s['priced_hit'])} ({gap:+.1f} pts).{low_n}"})
        props = [x for x in sc["by_prop"] if x["n"] >= 5]
        if props:
            best = max(props, key=lambda x: (x["units"] if x["units"] is not None else -99))
            worst = min(props, key=lambda x: (x["units"] if x["units"] is not None else 99))
            ins.append({"tone": "info", "title": "Lane leaders",
                        "text": f"Best lane: {best['prop']} {best['direction']} {best['w']}-{best['l']} "
                                f"({(best['units'] or 0):+.2f}u). Weakest: {worst['prop']} {worst['direction']} "
                                f"{worst['w']}-{worst['l']} ({(worst['units'] or 0):+.2f}u)."})
        ser = sc["series"]
        if ser["days"]:
            ins.append({"tone": "info", "title": "Drawdown",
                        "text": f"Max drawdown this season is {ser['max_drawdown']:.2f}u; currently {ser['current_drawdown']:.2f}u "
                                f"below the equity peak. {ser['green_days']} green days, {ser['red_days']} red, {ser['even_days']} even."})
        tot = sum(x["n"] for x in sc["by_prop"])
        ast = sum(x["n"] for x in sc["by_prop"] if x["prop"] == "AST")
        if tot:
            ins.append({"tone": "warn", "title": "Exposure concentration",
                        "text": f"{ast / tot * 100:.0f}% of graded picks are AST UNDERs and 100% of the card is UNDER-only. "
                                "Bad nights are correlated, so day-level swings are larger than the pick count implies (DEC-080)."})
    g = d.get("gates")
    if g and g.get("buckets"):
        bk = {b["bucket"]: b for b in g["buckets"]}
        if "published" in bk and ("blocked_price" in bk or "blocked_other" in bk):
            pub = bk["published"]
            parts = []
            for key, lab in (("blocked_price", "price-blocked"), ("blocked_other", "other-blocked")):
                if key in bk and bk[key]["n"]:
                    b = bk[key]
                    parts.append(f"{lab} {b['w']}-{b['l']} ({(b['units'] or 0):+.2f}u)")
            ins.append({"tone": "info", "title": "Do the gates earn their keep?",
                        "text": f"Published candidates went {pub['w']}-{pub['l']} ({(pub['units'] or 0):+.2f}u) vs " + ", ".join(parts) +
                                f" since {g['dates'][0]}. Small sample; the gates are being re-reviewed around 2026-11-30."})
        br = g.get("brier")
        if br:
            better = "model beats market" if br["model"] < br["market"] else "market beats model"
            ins.append({"tone": "good" if br["model"] < br["market"] else "bad", "title": "Probability quality",
                        "text": f"On {br['n']} graded candidates Brier is {br['model']:.4f} (model) vs {br['market']:.4f} "
                                f"(FanDuel no-vig): {better}. Lower is better."})
    clv = d.get("clv")
    if clv and (clv.get("picks") or {}).get("overall"):
        o = clv["picks"]["overall"]
        ins.append({"tone": "warn", "title": "Closing-line value is not confirming results yet",
                    "text": f"Beat the close on {pct(o.get('beat_close_rate'), 0)} of {o.get('rows_with_exact_close')} exact-close picks, "
                            f"while the no-vig price moved toward our side {pct(o.get('moved_toward_side_rate'), 0)} of the time "
                            f"(avg {((o.get('mean_no_vig_move') or 0) * 100):+.1f} pts). Realised ROI is {pct(o.get('roi_units_per_bet'))}/bet. "
                            "EV-vs-close always starts near -vig (about -4.5% at -110), so read it against that baseline."})
    h = d.get("health") or {}
    for f in h.get("flags", []):
        ins.append({"tone": "bad", "title": "Data integrity flag", "text": f["text"]})
    rc = h.get("reconciliation") or {}
    if rc.get("current_conflicts"):
        ins.append({"tone": "warn", "title": "Grade reconciliation",
                    "text": f"{rc['current_conflicts']} stored grades differ from the box score; "
                            f"{rc.get('outcome_conflicts') or 0} of them flip the W/L result."})
    lr = (h.get("pipeline") or {}).get("last_run")
    if lr and lr.get("status") == "late":
        ins.append({"tone": "info", "title": "Latest pipeline run was a late rerun",
                    "text": f"The run at {lr['start']} happened after every game had tipped, so it fetched no pre-game lines and "
                            "published no new card. That is expected (DEC-068), not a failure."})
    elif lr and lr.get("err"):
        ins.append({"tone": "warn", "title": "Last pipeline run logged errors",
                    "text": f"Run started {lr['start']} logged {lr['err']} error(s) and {lr['warn']} warning(s). See System Health."})
    return ins


# ---------------------------------------------------------------------------
# Team logos (embedded so the page works offline)
# ---------------------------------------------------------------------------
_SVG_DENY_TAGS = frozenset({"script", "foreignobject", "iframe", "object", "embed", "image", "animate", "set", "handler"})
_SVG_EXTERNAL_URL = re.compile(r"url\(\s*(?:['\"]\s*)?[^#\s'\")]", re.I)   # url(...) pointing anywhere but a #fragment


def validate_logo_svg(raw: bytes) -> None:
    """Raise ValueError unless ``raw`` is a small, well-formed, inert SVG.

    Logos are third-party files that get inlined as data: URIs and drawn as CSS backgrounds, where browsers
    already refuse scripts and subresource loads. This check is defence in depth: it keeps active content and
    external references out of the committed assets (and so out of the page) altogether.
    """
    if not raw or len(raw) > MAX_LOGO_BYTES:
        raise ValueError(f"size {len(raw)} bytes is outside 1..{MAX_LOGO_BYTES}")
    lowered = raw.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ValueError("contains a DOCTYPE/ENTITY declaration")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"not well-formed XML ({exc})") from exc
    if not isinstance(root.tag, str) or root.tag.split("}")[-1].lower() != "svg":
        raise ValueError(f"root element is {root.tag!r}, not <svg>")
    for el in root.iter():
        name = el.tag.split("}")[-1].lower() if isinstance(el.tag, str) else ""
        if name in _SVG_DENY_TAGS:
            raise ValueError(f"contains <{name}>")
        if name == "style" and el.text and ("@import" in el.text.lower() or _SVG_EXTERNAL_URL.search(el.text)):
            raise ValueError("<style> imports or references an external resource")
        for attr, val in el.attrib.items():
            a = attr.split("}")[-1].lower()
            v = val.strip().lower()
            if a.startswith("on"):
                raise ValueError(f"contains event attribute {a!r}")
            if a == "href" and not v.startswith("#"):
                raise ValueError("contains a non-fragment href")
            if "javascript:" in v or _SVG_EXTERNAL_URL.search(val):
                raise ValueError(f"attribute {a!r} references an external resource")


def load_logos(logo_dir: Path | None = None) -> tuple[dict[str, str], list[str]]:
    """Read ``<ABBR>.svg`` for every known team and return ({abbr: data: URI}, problems).

    A file that is rejected or does not belong to a known team is reported in ``problems`` and not embedded;
    that team keeps the CDN/badge fallback. Missing files are the caller's business (see build_payload).
    """
    logo_dir = LOGO_DIR if logo_dir is None else Path(logo_dir)
    logos: dict[str, str] = {}
    problems: list[str] = []
    if not logo_dir.is_dir():
        return logos, problems
    for f in sorted(logo_dir.glob("*.svg")):
        if f.stem not in TEAMS:
            problems.append(f"logo file {f.name} is not a known team abbreviation and was ignored")
            continue
        try:
            raw = f.read_bytes()
            validate_logo_svg(raw)
        except (OSError, ValueError) as exc:
            problems.append(f"logo file {f.name} rejected ({exc}); that team falls back to the CDN/badge")
            continue
        logos[f.stem] = "data:image/svg+xml;base64," + base64.b64encode(raw).decode("ascii")
    return logos, problems


def _image_mime(raw: bytes) -> str | None:
    """MIME type from the file's own magic bytes (never from its extension)."""
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def load_wallpaper(asset_dir: Path | None = None) -> tuple[str | None, str | None]:
    """Return (data: URI, problem) for the optional page wallpaper.

    The wallpaper is an optional ``wallpaper.jpg|jpeg|png|webp`` next to the other dashboard assets and is
    embedded byte for byte, so the page stays one offline file. No file at all is fine: (None, None).
    A file that is empty, too large or not really a JPEG/PNG/WebP is NOT embedded and is explained in ``problem``.
    """
    asset_dir = ASSET_DIR if asset_dir is None else Path(asset_dir)
    for name in WALLPAPER_NAMES:
        f = asset_dir / name
        if not f.is_file():
            continue
        try:
            raw = f.read_bytes()
        except OSError as exc:
            return None, f"wallpaper {name} could not be read ({exc}); the page has no wallpaper"
        if not raw or len(raw) > MAX_WALLPAPER_BYTES:
            return None, (f"wallpaper {name} is {len(raw):,} bytes, outside 1..{MAX_WALLPAPER_BYTES:,}; "
                          "it was not embedded and the page has no wallpaper")
        mime = _image_mime(raw)
        if mime is None:
            return None, f"wallpaper {name} is not a JPEG, PNG or WebP image; it was not embedded and the page has no wallpaper"
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii"), None
    return None, None


# ---------------------------------------------------------------------------
# Assemble + render
# ---------------------------------------------------------------------------
def build_payload(root: Path, now: datetime | None = None, logo_dir: Path | None = None,
                  wallpaper_dir: Path | None = None) -> dict:
    paths = Paths(root)
    col = Collector()
    now = now or datetime.now()
    build_date = now.strftime("%Y-%m-%d")
    payload: dict[str, Any] = {}

    picks = col.run("output/picks_history.csv", lambda: load_picks(paths))
    if picks is None:
        raise SystemExit("picks_history.csv is required to build the dashboard; see warnings above.")

    payload["teams"] = {k: {"id": v[0], "name": v[1], "color": v[2]} for k, v in TEAMS.items()}
    unknown = sorted((set(picks["team"].dropna()) | set(picks["opponent"].dropna())) - set(TEAMS))
    if unknown:  # loud, not silent: those teams get a text badge instead of a logo
        col.warnings.append(f"unknown team abbreviations (no logo available): {', '.join(unknown)}")
    payload["logos"], logo_problems = load_logos(logo_dir)
    col.warnings.extend(logo_problems)
    not_embedded = sorted(set(TEAMS) - set(payload["logos"]))
    if not_embedded:  # loud: the page still works, but those logos need internet (else a text badge)
        col.warnings.append(f"{len(not_embedded)} of {len(TEAMS)} team logos are not embedded and load from the NBA CDN "
                            f"when online (text badge offline): {', '.join(not_embedded)}")
    payload["wallpaper"], wallpaper_problem = load_wallpaper(wallpaper_dir)
    if wallpaper_problem:
        col.warnings.append(wallpaper_problem)
    payload["eras"] = []
    for e in ERAS:
        g = picks[picks["era"] == e["key"]]
        payload["eras"].append(dict(e, **summarize(g)))
    payload["scopes"] = {sc: build_scope(picks, sc) for sc in SCOPES}
    payload["scope_labels"] = SCOPE_LABELS
    payload["picks"] = pick_rows(picks)
    payload["today"] = col.run("today's card", lambda: build_today(paths, picks, build_date), {})
    payload["slips"] = col.run("output/betslips_history.csv", lambda: build_slips(paths), {})
    payload["gates"] = col.run("output/candidates_history.csv", lambda: build_gates(paths), {})
    payload["clv"] = col.run("output/clv_summary.json + clv_picks.csv", lambda: build_clv(paths), {})
    payload["models"] = build_models(paths, col)
    payload["health"] = build_health(paths, picks, now, col)
    orph = (payload["slips"] or {}).get("orphans") or {}
    if orph.get("n"):
        lines = ", ".join(str(x) for x in orph["csv_lines"])
        payload["health"]["flags"].append({
            "kind": "orphan_slip", "n": orph["n"],
            "text": f"betslips_history.csv has {orph['n']} row(s) with no date and no legs (CSV line {lines}), stored as a graded LOSS. "
                    "They are excluded from every slip number here.",
            "hint": "Read-only: the file was not modified. Removing the row is your call (DEC-070)."})
    payload["decisions"] = col.run("output/DECISIONS.md", lambda: parse_decisions(paths.output / "DECISIONS.md"), [])
    payload["tasks"] = col.run("output/TASKS.md", lambda: parse_tasks(paths.output / "TASKS.md"), [])
    payload["insights"] = build_insights(payload)
    payload["meta"] = {
        "generated_at": now.isoformat(timespec="seconds"), "build_date": build_date,
        "first_pick": picks["game_date"].min(), "last_pick": picks["game_date"].max(),
        "total_picks": int(len(picks)), "git": payload["health"].get("git"),
        "logos_embedded": len(payload["logos"]),
        "warnings": col.warnings,
    }
    return clean(payload)


def render_html(payload: dict) -> str:
    """Inline CSS/JS/data into the template. '</' is escaped so data can never close the script tag."""
    tpl = (ASSET_DIR / "template.html").read_text(encoding="utf-8")
    css = (ASSET_DIR / "style.css").read_text(encoding="utf-8")
    charts = (ASSET_DIR / "charts.js").read_text(encoding="utf-8")
    app = (ASSET_DIR / "app.js").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    data = data.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    for token, value in (("/*__CSS__*/", css), ("/*__DATA__*/", data), ("/*__CHARTS__*/", charts), ("/*__APP__*/", app)):
        if token not in tpl:
            raise RuntimeError(f"template token {token} missing")
        tpl = tpl.replace(token, value)
    return tpl


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _make_console_safe() -> None:
    """Never let a non-ASCII player name or team crash the run on a cp1252 / redirected console
    (as under Task Scheduler) after the page has already been written."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (e.g. replaced by a test harness): nothing to do


def main(argv: list[str] | None = None) -> int:
    _make_console_safe()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SCRIPT_DIR.parent), help="project root (default: repo root)")
    ap.add_argument("--out", default=None, help="output HTML (default: <root>/output/dashboard_pro.html)")
    ap.add_argument("--json", action="store_true", help="also print a short JSON summary to stdout")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.out) if args.out else root / "output" / "dashboard_pro.html"
    payload = build_payload(root)
    write_atomic(out, render_html(payload))
    w = payload["meta"]["warnings"]
    s = payload["scopes"]["season"]["summary"]
    print(f"[dashboard] wrote {out} ({out.stat().st_size / 1024:.0f} KB); picks={payload['meta']['total_picks']}; "
          f"season {s['w']}-{s['l']}; warnings={len(w)}")
    for line in w:
        print(f"[dashboard]   warning: {line}")
    if args.json:
        print(json.dumps({"season": s, "warnings": w}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
