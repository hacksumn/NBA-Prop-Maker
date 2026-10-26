"""Per-stat edge map: where the system has edge, and how PrizePicks compares to the book.

Built after the DEC-039 lossless-store recovery. Answers three separate questions
that were previously conflated:

  A. MARKET COMPARISON — for keys where both a PrizePicks line and a sportsbook
     line exist, how far apart are they per stat, and in which direction?
  B. BASELINE — what does naive always-UNDER return per stat, split by book and
     controlled for minutes played (the DEC-036 confound)?
  C. MODEL — what did the actual card return per stat/direction, versus the
     baseline for the same book?

Read-only. Writes nothing. Prices (`over_price`/`under_price`) are not yet
populated, so no-vig / CLV comparisons are deliberately not attempted here — see
the DEC-039 three-snapshot panel task.

Usage:
    python scripts/market_edge_map.py
    python scripts/market_edge_map.py --live-start 2026-03-30 --min-minutes 30
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

# Every prop the store carries, mapped to its box-score construction.
PROP_FORMULAS = {
    "PTS": lambda d: d["pts"],
    "TRB": lambda d: d["trb"],
    "AST": lambda d: d["ast"],
    "STL": lambda d: d["stl"],
    "BLK": lambda d: d["blk"],
    "TOV": lambda d: d["tov"],
    "PA": lambda d: d["pts"] + d["ast"],
    "PR": lambda d: d["pts"] + d["trb"],
    "PRA": lambda d: d["pts"] + d["trb"] + d["ast"],
}

PROP_ALIASES = {
    "player_points": "PTS",
    "player_rebounds": "TRB",
    "player_assists": "AST",
    "player_steals": "STL",
    "player_blocks": "BLK",
    "player_turnovers": "TOV",
    "player_points_assists": "PA",
    "player_points_rebounds": "PR",
    "player_points_rebounds_assists": "PRA",
    "points": "PTS",
    "rebounds": "TRB",
    "assists": "AST",
    "steals": "STL",
    "blocks": "BLK",
    "turnovers": "TOV",
    **{code: code for code in PROP_FORMULAS},
}

MINUTES_BINS = [-np.inf, 10, 20, 25, 30, 35, np.inf]
MINUTES_LABELS = ["<10", "10-20", "20-25", "25-30", "30-35", "35+"]


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def book_family(source: object) -> str:
    """Collapse source labels into the two markets that matter strategically."""
    s = str(source).lower()
    if "prizepicks" in s:
        return "prizepicks"
    if "fanduel" in s or "odds_api" in s:
        return "book"
    return "other"


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = wins / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (p, c - m, c + m)


def load_box() -> pd.DataFrame:
    box = pd.read_csv(
        DATA_DIR / "nba_data.csv",
        low_memory=False,
        usecols=["player", "game_date", "mp", "pts", "trb", "ast", "stl", "blk", "tov"],
    )
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce")
    box["player_norm"] = box["player"].map(normalize_name)
    for code, formula in PROP_FORMULAS.items():
        box[code] = formula(box)
    box = box.dropna(subset=["game_date", "player_norm"])
    return box.drop_duplicates(subset=["player_norm", "game_date"], keep="first")


def load_lines() -> pd.DataFrame:
    lines = pd.read_csv(DATA_DIR / "historical_lines.csv", low_memory=False)
    lines["game_date"] = pd.to_datetime(lines["game_date"], errors="coerce")
    lines["player_norm"] = lines["player"].map(normalize_name)
    lines["prop_code"] = lines["prop"].map(PROP_ALIASES)
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")
    lines["fam"] = lines["source"].map(book_family)
    lines["snap"] = pd.to_datetime(lines["snapshot_ts"], errors="coerce")
    return lines.dropna(subset=["game_date", "player_norm", "prop_code", "line"])


def grade(frame: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Attach realized outcome and the UNDER result. Pushes dropped (refunded)."""
    merged = frame.merge(
        box[["player_norm", "game_date", "mp", *PROP_FORMULAS]],
        on=["player_norm", "game_date"],
        how="inner",
    )
    merged["actual"] = merged.apply(lambda r: r[r["prop_code"]], axis=1)
    merged = merged[merged["actual"] != merged["line"]].copy()
    merged["under_win"] = (merged["actual"] < merged["line"]).astype(int)
    merged["minutes_bin"] = pd.cut(merged["mp"], MINUTES_BINS, labels=MINUTES_LABELS)
    return merged


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-start", default="2026-03-30")
    ap.add_argument("--min-minutes", type=float, default=30.0)
    args = ap.parse_args()

    box = load_box()
    lines = load_lines()

    print(f"Line store:  {len(lines):,} raw rows, "
          f"{lines['snapshot_ts'].nunique():,} distinct snapshots")
    has_price = lines[["over_price", "under_price"]].notna().all(axis=1).sum()
    print(f"Rows carrying both over/under prices: {has_price:,}")
    if has_price == 0:
        print("  -> No prices in the store yet. No-vig / CLV comparison NOT possible.")
        print("     (Scaffolding only until the DEC-039 paid three-snapshot panel runs.)")

    # ---- latest pre-tip line per (key, family) -----------------------------
    lines = lines.sort_values("snap")
    per_fam = lines.groupby(
        ["game_date", "player_norm", "prop_code", "fam"], as_index=False
    ).agg(line=("line", "last"), n_snap=("line", "size"))

    section("A. MARKET COMPARISON — PrizePicks line vs sportsbook line, same player/prop/day")
    pp = per_fam[per_fam.fam == "prizepicks"].rename(columns={"line": "pp_line"})
    bk = per_fam[per_fam.fam == "book"].rename(columns={"line": "bk_line"})
    both = pp.merge(
        bk[["game_date", "player_norm", "prop_code", "bk_line"]],
        on=["game_date", "player_norm", "prop_code"],
    )
    both["gap"] = both["pp_line"] - both["bk_line"]
    if len(both):
        tbl = both.groupby("prop_code").agg(
            n=("gap", "size"), pp_line=("pp_line", "mean"),
            book_line=("bk_line", "mean"), gap=("gap", "mean"),
            pp_higher=("gap", lambda s: (s > 0).mean()),
        ).sort_values("gap", ascending=False)
        print(tbl.round(3).to_string())
        print(f"\n  overlap window: {both.game_date.min().date()} -> {both.game_date.max().date()}")
        print(f"  ALL PROPS: n={len(both):,}  mean gap={both.gap.mean():+.3f}  "
              f"PrizePicks higher in {(both.gap > 0).mean():.1%} of keys")
        print("  Positive gap = PrizePicks posts a HIGHER line than the book, which")
        print("  mechanically favours UNDER on PrizePicks.")

        graded_both = grade(both.assign(line=both.pp_line), box)
        if len(graded_both):
            graded_both["gap_bucket"] = pd.cut(
                graded_both["gap"], [-np.inf, -0.001, 0.001, 0.5, 1.0, np.inf],
                labels=["PP lower", "equal", "PP +0..0.5", "PP +0.5..1", "PP +1 or more"],
            )
            section("A2. Does the PrizePicks-vs-book gap actually predict UNDER wins?")
            gb = graded_both.groupby("gap_bucket", observed=True).agg(
                n=("under_win", "size"), under_wr=("under_win", "mean")
            )
            print(gb.round(3).to_string())
            print("  (UNDER bet on the PrizePicks line, bucketed by how far it sits above the book)")
    else:
        print("  No overlapping keys found.")

    # ---- baselines ---------------------------------------------------------
    preferred = per_fam.sort_values("fam").groupby(
        ["game_date", "player_norm", "prop_code"], as_index=False
    ).agg(line=("line", "last"))
    universe = grade(preferred, box)

    section("B. NAIVE ALWAYS-UNDER BASELINE per stat (all sources, all dates)")
    base = universe.groupby("prop_code").agg(
        n=("under_win", "size"), under_wr=("under_win", "mean"),
        mean_line=("line", "mean"), mean_actual=("actual", "mean"),
    )
    base["line_minus_actual"] = base["mean_line"] - base["mean_actual"]
    print(base.round(3).sort_values("under_wr", ascending=False).to_string())
    print(f"  ALL: n={len(universe):,}  under_wr={universe.under_win.mean():.4f}")

    section(f"B2. SAME, minutes-controlled ({args.min_minutes:g}+ minutes only)")
    hi = universe[universe.mp >= args.min_minutes]
    ctrl = hi.groupby("prop_code").agg(
        n=("under_win", "size"), under_wr=("under_win", "mean"),
        mean_line=("line", "mean"), mean_actual=("actual", "mean"),
    )
    print(ctrl.round(3).sort_values("under_wr", ascending=False).to_string())
    print(f"  ALL: n={len(hi):,}  under_wr={hi.under_win.mean():.4f}")
    print("  Props holding well above 0.50 here carry a structural skew.")
    print("  Props collapsing toward/below 0.50 were minutes bets, not stat edges.")

    section("B3. BASELINE BY BOOK — the same stat priced by PrizePicks vs by the book")
    fam_graded = grade(per_fam[per_fam.fam.isin(["prizepicks", "book"])], box)
    rows = []
    for (prop, fam), grp in fam_graded.groupby(["prop_code", "fam"]):
        g30 = grp[grp.mp >= args.min_minutes]
        rows.append({
            "prop": prop, "book": fam, "n": len(grp),
            "under_wr": grp.under_win.mean(),
            "n_30plus": len(g30),
            "under_wr_30plus": g30.under_win.mean() if len(g30) else np.nan,
        })
    fam_tbl = pd.DataFrame(rows).pivot(index="prop", columns="book")
    print(fam_tbl.round(3).to_string())

    # ---- model card --------------------------------------------------------
    picks_path = OUTPUT_DIR / "picks_history.csv"
    if picks_path.exists():
        picks = pd.read_csv(picks_path, low_memory=False)
        picks["game_date"] = pd.to_datetime(picks["game_date"], errors="coerce")
        live = picks[
            (picks.game_date >= args.live_start) & picks.result.isin(["WIN", "LOSS"])
        ].copy()
        live["win"] = (live.result == "WIN").astype(int)
        live["fam"] = live["line_source"].map(book_family)

        section("C. THE ACTUAL CARD per stat x direction (live era only)")
        print(f"{'prop':6s} {'dir':6s} {'book':11s} {'n':>4s} {'WR':>7s} "
              f"{'95% CI':>18s} {'baseline':>9s}  verdict")
        base30 = hi.groupby("prop_code").under_win.mean().to_dict()
        base_all = universe.groupby("prop_code").under_win.mean().to_dict()
        for (prop, direction, fam), grp in live.groupby(["prop", "direction", "fam"]):
            n = len(grp)
            if n < 3:
                continue
            k = int(grp.win.sum())
            p, lo, up = wilson(k, n)
            ref = base_all.get(prop, np.nan)
            if direction == "OVER" and not np.isnan(ref):
                ref = 1 - ref
            if np.isnan(ref):
                verdict = "no baseline"
            elif lo > ref:
                verdict = "BEATS baseline"
            elif up < ref:
                verdict = "BELOW baseline"
            else:
                verdict = "not distinguishable"
            print(f"{prop:6s} {direction:6s} {fam:11s} {n:4d} {p:7.4f} "
                  f"[{lo:6.4f},{up:6.4f}] {ref:9.4f}  {verdict}")
        print("\n  baseline = naive always-UNDER for that stat (inverted for OVER).")
        print("  Rows with n<3 suppressed. Small n -> wide intervals; read the CI, not the point.")

    # ---- line movement -----------------------------------------------------
    section("D. LINE MOVEMENT — does the line move before tip, and does it inform?")
    multi = lines.groupby(["game_date", "player_norm", "prop_code", "fam"]).agg(
        first_line=("line", "first"), last_line=("line", "last"), n_snap=("line", "size")
    ).reset_index()
    multi = multi[multi.n_snap > 1].copy()
    multi["move"] = multi.last_line - multi.first_line
    print(f"keys with >1 snapshot: {len(multi):,}")
    if len(multi):
        mv = grade(multi.assign(line=multi.last_line), box)
        mv["moved"] = np.select(
            [mv.move < -0.001, mv.move > 0.001], ["line moved DOWN", "line moved UP"],
            default="unchanged",
        )
        print()
        print(mv.groupby("moved").agg(
            n=("under_win", "size"), under_wr=("under_win", "mean"),
            mean_move=("move", "mean"),
        ).round(3).to_string())
        print("\n  If a falling line predicted UNDER wins, movement carries information")
        print("  the card could use. Flat differences mean the archived snapshots are")
        print("  too sparse or too early to capture real pre-tip movement.")

    print("\nDone. Read-only; nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
