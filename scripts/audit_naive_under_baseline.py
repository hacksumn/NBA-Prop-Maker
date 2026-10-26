"""Audit whether the model adds edge over a naive always-UNDER baseline.

Motivation (DEC-036): headline pick win rates were never compared against the
trivial strategy of betting UNDER on every available line. For right-skewed
count stats (STL/BLK/AST/TRB) the PrizePicks line sits near the mean while the
median sits below it, so "always UNDER" is a strong baseline on its own. Any
claimed model edge must be measured *against that baseline*, not against 50%.

This script is read-only. It re-evaluates DEC-036 from
`data/nba_data.csv`, `data/historical_lines.csv`, and `output/picks_history.csv`.

Usage:
    python scripts/audit_naive_under_baseline.py
    python scripts/audit_naive_under_baseline.py --live-start 2026-03-30
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

# Props this audit can reconstruct directly from box score columns.
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

BOX_SCORE_COLUMNS = ["player", "game_date", "mp", "pts", "trb", "ast", "stl", "blk", "tov"]

# `historical_lines.csv` mixes Odds API market keys with internal prop codes.
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
    """NFKD-strip diacritics and casefold, matching the Step 8 join convention."""
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


def load_box_scores() -> pd.DataFrame:
    """Load box scores and derive every auditable prop column."""
    path = DATA_DIR / "nba_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing box score dataset: {path}")
    box = pd.read_csv(
        path,
        low_memory=False,
        usecols=BOX_SCORE_COLUMNS,
    )
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce")
    box["player_norm"] = box["player"].map(normalize_name)
    for code, formula in PROP_FORMULAS.items():
        box[code] = formula(box)
    box = box.dropna(subset=["game_date", "player_norm"])
    # One row per player-game; duplicates would fan out the line join.
    return box.drop_duplicates(subset=["player_norm", "game_date"], keep="first")


def load_lines(allowed_source_terms: list[str] | None = None) -> pd.DataFrame:
    """Load the preferred pregame line per player/prop/day from raw snapshots."""
    path = DATA_DIR / "historical_lines.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing lines dataset: {path}")
    lines = pd.read_csv(path, low_memory=False)
    lines["game_date"] = pd.to_datetime(lines["game_date"], errors="coerce")
    lines["player_norm"] = lines["player"].map(normalize_name)
    lines["prop_code"] = lines["prop"].map(PROP_ALIASES)
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")
    lines = lines.dropna(subset=["game_date", "prop_code", "line"])
    if "source" not in lines.columns:
        lines["source"] = "unknown"
    if "snapshot_ts" not in lines.columns:
        lines["snapshot_ts"] = "unknown"
    source = lines["source"].fillna("unknown").astype(str).str.lower()
    if allowed_source_terms:
        allowed = source.apply(
            lambda value: any(term.lower() in value for term in allowed_source_terms)
        )
        lines = lines[allowed].copy()
        source = lines["source"].fillna("unknown").astype(str).str.lower()
    lines["_source_priority"] = np.select(
        [source.str.contains("fanduel"), source.str.contains("prizepicks"), source.str.contains("odds_api")],
        [3, 2, 1],
        default=0,
    )
    lines["_snapshot_sort"] = pd.to_datetime(lines["snapshot_ts"], errors="coerce", utc=True)
    if "commence_time" in lines.columns:
        commence = pd.to_datetime(lines["commence_time"], errors="coerce", utc=True)
        lines = lines[
            lines["_snapshot_sort"].isna() | commence.isna() | lines["_snapshot_sort"].le(commence)
        ].copy()
    lines = lines.sort_values(
        ["game_date", "player_norm", "prop_code", "_source_priority", "_snapshot_sort"],
        kind="stable",
        na_position="first",
    ).drop_duplicates(["game_date", "player_norm", "prop_code"], keep="last")
    keep = ["player_norm", "game_date", "prop_code", "line"]
    if "source" in lines.columns:
        keep.append("source")
    return lines[keep].reset_index(drop=True)


def build_universe(box: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Join lines to realized outcomes and label the UNDER result.

    Pushes are dropped: PrizePicks refunds them, so they are neither a win nor a
    loss for hit-rate purposes.
    """
    merged = lines.merge(
        box[["player_norm", "game_date", "mp", *PROP_FORMULAS]],
        on=["player_norm", "game_date"],
        how="inner",
    )
    merged["actual"] = merged.apply(lambda r: r[r["prop_code"]], axis=1)
    merged = merged[merged["actual"] != merged["line"]].copy()
    merged["under_win"] = (merged["actual"] < merged["line"]).astype(int)
    merged["minutes_bin"] = pd.cut(
        merged["mp"], bins=MINUTES_BINS, labels=MINUTES_LABELS
    )
    return merged


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — stable at the small samples this repo works with."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - margin, centre + margin)


def report_live_picks(live_start: str) -> pd.DataFrame | None:
    """Recompute logged pick results from box scores and split by era."""
    path = OUTPUT_DIR / "picks_history.csv"
    if not path.exists():
        print(f"  (skipping pick audit — {path} not found)")
        return None
    picks = pd.read_csv(path, low_memory=False)
    picks["game_date"] = pd.to_datetime(picks["game_date"], errors="coerce")
    graded = picks[picks["result"].isin(["WIN", "LOSS"])].copy()
    graded["win"] = (graded["result"] == "WIN").astype(int)

    print("\n=== LOGGED PICK RECORD BY ERA ===")
    eras = {
        f"backfilled (< {live_start})": graded[graded["game_date"] < live_start],
        f"live (>= {live_start})": graded[graded["game_date"] >= live_start],
    }
    for name, era in eras.items():
        n = len(era)
        wins = int(era["win"].sum())
        rate = wins / n if n else float("nan")
        lo, hi = wilson_interval(wins, n)
        print(f"  {name:28s} n={n:5d}  wins={wins:5d}  WR={rate:.4f}  95%CI=[{lo:.4f},{hi:.4f}]")
    return eras[f"live (>= {live_start})"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-start",
        default="2026-03-30",
        help="First date of the genuinely live pick era (default: 2026-03-30)",
    )
    parser.add_argument(
        "--min-minutes",
        type=float,
        default=30.0,
        help="Minutes floor for the minutes-controlled baseline (default: 30)",
    )
    args = parser.parse_args()

    box = load_box_scores()
    lines = load_lines()
    universe = build_universe(box, lines)

    print(f"Box scores:      {len(box):,} player-games "
          f"({box['game_date'].min().date()} -> {box['game_date'].max().date()})")
    print(f"Line universe:   {len(universe):,} graded lines "
          f"({universe['game_date'].min().date()} -> {universe['game_date'].max().date()})")

    print("\n=== NAIVE ALWAYS-UNDER BASELINE (all lines, all dates) ===")
    overall = universe.groupby("prop_code").agg(
        n=("under_win", "size"),
        under_wr=("under_win", "mean"),
        mean_line=("line", "mean"),
        mean_actual=("actual", "mean"),
    )
    overall["line_minus_actual"] = overall["mean_line"] - overall["mean_actual"]
    print(overall.round(3).to_string())
    print(f"  ALL PROPS: n={len(universe):,}  under_wr={universe['under_win'].mean():.4f}")

    print("\n=== UNDER WIN RATE BY MINUTES PLAYED (the confound) ===")
    by_min = universe.pivot_table(
        index="prop_code", columns="minutes_bin", values="under_win",
        aggfunc="mean", observed=True,
    )
    print(by_min.round(3).to_string())
    totals = universe.groupby("minutes_bin", observed=True)["under_win"].agg(["size", "mean"])
    print("\n  all props combined:")
    print(totals.round(3).to_string())

    print(f"\n=== MINUTES-CONTROLLED BASELINE ({args.min_minutes:g}+ minutes only) ===")
    high = universe[universe["mp"] >= args.min_minutes]
    controlled = high.groupby("prop_code").agg(
        n=("under_win", "size"),
        under_wr=("under_win", "mean"),
        mean_line=("line", "mean"),
        mean_actual=("actual", "mean"),
    )
    print(controlled.round(3).to_string())
    print(f"  ALL PROPS: n={len(high):,}  under_wr={high['under_win'].mean():.4f}")
    print("  Interpretation: props that stay well above 0.50 here carry a structural")
    print("  line skew. Props that collapse toward 0.50 were minutes bets, not stat edges.")

    live_picks = report_live_picks(args.live_start)

    if live_picks is not None and len(live_picks):
        live_universe = universe[universe["game_date"] >= args.live_start]
        model_wr = live_picks["win"].mean()
        naive_wr = live_universe["under_win"].mean()
        print("\n=== MODEL vs PREFERRED-SOURCE NAIVE, SAME WINDOW (DESCRIPTIVE ONLY) ===")
        print(f"  model live picks:      n={len(live_picks):5d}  WR={model_wr:.4f}")
        print(f"  naive always-UNDER:    n={len(live_universe):5d}  WR={naive_wr:.4f}")
        print("  WARNING: this aggregate can mix pick and baseline sources; do not use it")
        print("  as an edge claim. Source/date-matched diagnostics follow.")

        print("\n=== SOURCE/DATE-MATCHED MODEL vs NAIVE ===")
        families = [
            ("prizepicks", ["prizepicks"], ["prizepicks"]),
            ("book", ["fanduel", "odds_api"], ["fanduel", "odds_api"]),
        ]
        for label, pick_terms, line_terms in families:
            pick_source = live_picks.get("line_source", pd.Series("", index=live_picks.index)).fillna("").astype(str).str.lower()
            selected = live_picks[pick_source.apply(lambda value: any(term in value for term in pick_terms))]
            if selected.empty:
                continue
            family_universe = build_universe(box, load_lines(line_terms))
            selected_dates = set(selected["game_date"].dt.normalize())
            family_universe = family_universe[family_universe["game_date"].dt.normalize().isin(selected_dates)]
            model_rate = selected["win"].mean()
            baseline_rate = family_universe["under_win"].mean()
            print(
                f"  {label:10s} model n={len(selected):4d} WR={model_rate:.4f} | "
                f"naive same-source/dates n={len(family_universe):5d} WR={baseline_rate:.4f} | "
                f"delta={model_rate - baseline_rate:+.4f}"
            )
        print("  Unknown-source picks are excluded. This still is not the required")
        print("  same-date/prop eligible-pool randomization test.")

    print("\nDone. This audit is read-only; no artifacts were written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
