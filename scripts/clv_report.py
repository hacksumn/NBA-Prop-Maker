"""Closing-line value (CLV) report for FanDuel picks and every model lean (DEC-066).

Answers: when the system takes a side at the morning FanDuel price, does the market
later move toward that side? Beating the close is the fastest honest test of edge —
it needs far fewer bets than win/loss records.

Two populations, reported separately:

  picks   Published card rows in output/picks_history.csv (line_source = fanduel).
  leans   The model's favored side on every line it scored, from the latest
          archived predictions file per slate. Not bet; measures model skill.

Bet snapshot = the latest FanDuel snapshot captured at or before the run that
produced the card/predictions. Close = the latest FanDuel snapshot strictly after
the bet snapshot and before tip-off (master store plus T-60/T-5 captures).

Per row:
  ev_vs_close     close_no_vig_side_prob * bet_decimal_odds - 1  (exact same line only)
  beat_close      ev_vs_close > 0   (the market moved by more than the vig)
  no_vig_move     close no-vig side prob - bet no-vig side prob  (vig-free direction signal)
  line_clv        favorable line movement in stat units when the close line differs
  close_minutes_before_tip, close_source   -- how good the "close" really is
Rows whose only snapshot is the bet snapshot are kept with close_quality =
'no_later_snapshot' and excluded from the summary.

Read-only apart from the three outputs. No network calls.

Usage:
    python scripts/clv_report.py
    python scripts/clv_report.py --start-date 2026-10-26
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
LOCAL_TZ = "America/Chicago"
DEFAULT_START = "2026-10-26"  # DEC-060: FanDuel became the authoritative live market

STAT_TO_ODDS = {
    "pts": "player_points", "trb": "player_rebounds", "ast": "player_assists",
    "pra": "player_points_rebounds_assists", "pr": "player_points_rebounds",
    "pa": "player_points_assists", "stl": "player_steals", "blk": "player_blocks",
    "tov": "player_turnovers",
}
STAT_FORMULAS = {
    "pts": ["pts"], "trb": ["trb"], "ast": ["ast"], "stl": ["stl"], "blk": ["blk"],
    "tov": ["tov"], "pra": ["pts", "trb", "ast"], "pr": ["pts", "trb"], "pa": ["pts", "ast"],
}


def normalize_name(name: object) -> str:
    """Lowercase, strip accents and punctuation (matches nba_props._normalize_name)."""
    text = unicodedata.normalize("NFD", str(name))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def american_to_decimal(price: object) -> float:
    """Decimal odds (stake included) from an American price; NaN when invalid."""
    value = pd.to_numeric(pd.Series([price]), errors="coerce").iloc[0]
    if pd.isna(value) or -100 < value < 100:
        return float("nan")
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / -value)


def side_prices(over: object, under: object, direction: str) -> Dict[str, float]:
    """Decimal odds and no-vig probability for the chosen side."""
    over_dec, under_dec = american_to_decimal(over), american_to_decimal(under)
    side_dec, other_dec = (under_dec, over_dec) if direction == "UNDER" else (over_dec, under_dec)
    if not (np.isfinite(side_dec) and np.isfinite(other_dec)):
        return {"decimal": side_dec, "no_vig": float("nan")}
    side_imp, other_imp = 1.0 / side_dec, 1.0 / other_dec
    return {"decimal": side_dec, "no_vig": side_imp / (side_imp + other_imp)}


def load_fanduel_snapshots(start_date: str) -> pd.DataFrame:
    """All FanDuel line rows on/after start_date: master store plus T-60/T-5 captures."""
    cols = ["game_date", "player", "prop", "line", "source", "snapshot_ts",
            "commence_time", "over_price", "under_price"]
    frames = []
    master = DATA_DIR / "historical_lines.csv"
    if master.exists():
        for chunk in pd.read_csv(master, usecols=cols, low_memory=False, chunksize=500_000):
            chunk = chunk[chunk["game_date"].astype(str) >= start_date]
            frames.append(chunk[chunk["source"].astype(str).str.contains("fanduel", na=False)])
    alt_path = DATA_DIR / "fanduel_alternate_lines.csv"  # DEC-066 alternate contracts
    if alt_path.exists():
        alt = pd.read_csv(alt_path, low_memory=False)
        alt["prop"] = alt["prop"].astype(str).str.replace("_alternate", "", regex=False)
        frames.append(alt[[c for c in cols if c in alt.columns]])
    closing_dir = DATA_DIR / "odds_api_fanduel_closing_parsed"
    if closing_dir.exists():
        for path in sorted(closing_dir.rglob("*.csv")):
            snap = pd.read_csv(path, low_memory=False)
            if "game_date" in snap.columns and len(snap):
                frames.append(snap[[c for c in cols if c in snap.columns]])
    if not frames:
        return pd.DataFrame(columns=cols + ["_pn", "_snap", "_commence"])
    lines = pd.concat(frames, ignore_index=True)
    lines = lines[lines["game_date"].astype(str) >= start_date].copy()
    lines["game_date"] = lines["game_date"].astype(str).str[:10]
    lines["_pn"] = lines["player"].apply(normalize_name)
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce").round(2)
    lines["_snap"] = pd.to_datetime(lines["snapshot_ts"], errors="coerce", utc=True, format="mixed")
    lines["_commence"] = pd.to_datetime(lines["commence_time"], errors="coerce", utc=True, format="mixed")
    lines = lines.dropna(subset=["_snap", "line"])
    return lines.drop_duplicates(subset=["game_date", "_pn", "prop", "line", "_snap"], keep="last")


def latest_run_times(kind: str, start_date: str) -> Dict[str, tuple]:
    """(UTC time, path) of the latest archived run per slate for 'picks' or 'predictions'."""
    out: Dict[str, tuple] = {}
    archive = OUTPUT_DIR / "archive"
    pattern = re.compile(rf"{kind}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{6}})\.csv$")
    for path in archive.rglob(f"{kind}_*.csv") if archive.exists() else []:
        match = pattern.search(path.name)
        if not match or match.group(1) < start_date:
            continue
        stamp = pd.Timestamp(datetime.strptime(match.group(1) + match.group(2), "%Y-%m-%d%H%M%S"))
        stamp = stamp.tz_localize(LOCAL_TZ).tz_convert("UTC")
        if match.group(1) not in out or stamp > out[match.group(1)][0]:
            out[match.group(1)] = (stamp, path)
    return out


def attach_clv(rows: pd.DataFrame, lines: pd.DataFrame, run_times: Dict[str, tuple]) -> pd.DataFrame:
    """Attach bet/close prices and CLV metrics to rows with game_date/_pn/odds_prop/line/direction."""
    grouped = {key: grp.sort_values("_snap") for key, grp in lines.groupby(["game_date", "_pn", "prop"])}
    records = []
    for _, row in rows.iterrows():
        rec = row.to_dict()
        rec.update({
            "bet_snapshot_ts": None, "bet_price": np.nan, "bet_decimal": np.nan, "bet_no_vig": np.nan,
            "close_snapshot_ts": None, "close_source": None, "close_line": np.nan, "close_price": np.nan,
            "close_no_vig": np.nan, "close_minutes_before_tip": np.nan, "ev_vs_close": np.nan,
            "beat_close": np.nan, "no_vig_move": np.nan, "line_clv": np.nan,
            "close_quality": "no_bet_snapshot",
        })
        grp = grouped.get((row["game_date"], row["_pn"], row["odds_prop"]))
        run_time = run_times.get(row["game_date"], (None,))[0]
        if grp is None or run_time is None:
            records.append(rec)
            continue
        before = grp[(grp["_snap"] <= run_time) & (grp["line"] == row["line"])]
        if before.empty:
            records.append(rec)
            continue
        bet = before.iloc[-1]
        price_col = "under_price" if row["direction"] == "UNDER" else "over_price"
        bet_side = side_prices(bet["over_price"], bet["under_price"], row["direction"])
        rec.update({
            "bet_snapshot_ts": str(bet["snapshot_ts"]), "bet_price": bet[price_col],
            "bet_decimal": bet_side["decimal"], "bet_no_vig": bet_side["no_vig"],
        })
        commence = bet["_commence"]
        later = grp[(grp["_snap"] > bet["_snap"]) & (grp["_snap"] < commence)] if pd.notna(commence) else grp.iloc[0:0]
        if later.empty:
            rec["close_quality"] = "no_later_snapshot"
            records.append(rec)
            continue
        last_snap = later["_snap"].max()
        at_close = later[later["_snap"] == last_snap]
        exact = at_close[at_close["line"] == row["line"]]
        close = exact.iloc[0] if not exact.empty else at_close.iloc[0]
        rec.update({
            "close_snapshot_ts": str(close["snapshot_ts"]), "close_source": close["source"],
            "close_line": close["line"],
            "close_minutes_before_tip": round((commence - last_snap).total_seconds() / 60.0, 1),
        })
        line_delta = float(row["line"]) - float(close["line"])
        rec["line_clv"] = line_delta if row["direction"] == "UNDER" else -line_delta
        if not exact.empty:
            close_side = side_prices(close["over_price"], close["under_price"], row["direction"])
            rec["close_price"] = close[price_col]
            rec["close_no_vig"] = close_side["no_vig"]
            if np.isfinite(close_side["no_vig"]) and np.isfinite(bet_side["decimal"]):
                rec["ev_vs_close"] = close_side["no_vig"] * bet_side["decimal"] - 1.0
                rec["beat_close"] = bool(rec["ev_vs_close"] > 0)
            if np.isfinite(close_side["no_vig"]) and np.isfinite(bet_side["no_vig"]):
                rec["no_vig_move"] = close_side["no_vig"] - bet_side["no_vig"]
            rec["close_quality"] = "exact_line"
        else:
            rec["close_quality"] = "line_moved"
        records.append(rec)
    return pd.DataFrame(records)


def attach_outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    """Grade rows against data/nba_data.csv and compute unit profit at the bet price."""
    rows = rows.copy()
    rows["actual"], rows["result"], rows["profit_units"] = np.nan, None, np.nan
    box_path = DATA_DIR / "nba_data.csv"
    if not box_path.exists() or rows.empty:
        return rows
    box = pd.read_csv(box_path, usecols=lambda c: c in {"player", "game_date", "pts", "trb", "ast", "stl", "blk", "tov"},
                      low_memory=False)
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    box = box[box["game_date"] >= rows["game_date"].min()]
    box["_pn"] = box["player"].apply(normalize_name)
    box = box.drop_duplicates(subset=["game_date", "_pn"], keep="last").set_index(["game_date", "_pn"])
    for idx, row in rows.iterrows():
        key = (row["game_date"], row["_pn"])
        if key not in box.index:
            continue
        actual = float(sum(pd.to_numeric(box.loc[key, c], errors="coerce") for c in STAT_FORMULAS[row["stat"]]))
        line = float(row["line"])
        if actual == line:
            result, profit = "PUSH", 0.0
        elif (actual < line) == (row["direction"] == "UNDER"):
            result = "WIN"
            profit = row["bet_decimal"] - 1.0 if np.isfinite(row["bet_decimal"]) else np.nan
        else:
            result, profit = "LOSS", -1.0
        rows.at[idx, "actual"], rows.at[idx, "result"], rows.at[idx, "profit_units"] = actual, result, profit
    return rows


def build_pick_rows(start_date: str) -> pd.DataFrame:
    """Published FanDuel picks from picks_history.csv."""
    path = OUTPUT_DIR / "picks_history.csv"
    if not path.exists():
        return pd.DataFrame()
    hist = pd.read_csv(path, dtype=str)
    hist = hist[(hist["game_date"] >= start_date) & (hist.get("line_source", "") == "fanduel")].copy()
    hist["stat"] = hist["prop"].str.lower()
    hist = hist[hist["stat"].isin(STAT_TO_ODDS)]
    return pd.DataFrame({
        "game_date": hist["game_date"], "player": hist["player"], "_pn": hist["player"].apply(normalize_name),
        "stat": hist["stat"], "odds_prop": hist["stat"].map(STAT_TO_ODDS),
        "line": pd.to_numeric(hist["line"], errors="coerce").round(2),
        "direction": hist["direction"].str.upper(),
        "model_side_prob": pd.to_numeric(hist.get("dir_prob"), errors="coerce") / 100.0,
    }).reset_index(drop=True)


def build_lean_rows(pred_runs: Dict[str, tuple]) -> pd.DataFrame:
    """Model's favored side on every scored line in the latest predictions archive per slate."""
    frames = []
    for game_date, (_, path) in sorted(pred_runs.items()):
        pred = pd.read_csv(path, low_memory=False)
        for stat, odds_prop in STAT_TO_ODDS.items():
            line_col, prob_col = f"{stat}_line", f"{stat}_p_over_cal"
            if line_col not in pred.columns or prob_col not in pred.columns:
                continue
            sub = pred[pred[line_col].notna() & pred[prob_col].notna()]
            if sub.empty:
                continue
            p_over = pd.to_numeric(sub[prob_col], errors="coerce")
            direction = np.where(p_over >= 0.5, "OVER", "UNDER")
            frames.append(pd.DataFrame({
                "game_date": game_date, "player": sub["player"].values,
                "_pn": sub["player"].apply(normalize_name).values, "stat": stat, "odds_prop": odds_prop,
                "line": pd.to_numeric(sub[line_col], errors="coerce").round(2).values,
                "direction": direction,
                "model_side_prob": np.where(direction == "OVER", p_over, 1.0 - p_over),
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(rows: pd.DataFrame, by: list) -> Dict[str, object]:
    """CLV and outcome summary overall and per group; excludes rows without a later close."""
    def _block(df: pd.DataFrame) -> Dict[str, object]:
        priced = df[df["ev_vs_close"].notna()]
        graded = df[df["result"].isin(["WIN", "LOSS"])]
        ev = priced["ev_vs_close"].astype(float)
        return {
            "rows": int(len(df)),
            "rows_with_exact_close": int(len(priced)),
            "mean_ev_vs_close": round(float(ev.mean()), 4) if len(ev) else None,
            "ev_vs_close_std_error": round(float(ev.std(ddof=1) / np.sqrt(len(ev))), 4) if len(ev) > 1 else None,
            "beat_close_rate": round(float(priced["beat_close"].astype(float).mean()), 4) if len(priced) else None,
            "mean_no_vig_move": round(float(priced["no_vig_move"].astype(float).mean()), 4) if len(priced) else None,
            "moved_toward_side_rate": round(float((priced["no_vig_move"].astype(float) > 0).mean()), 4) if len(priced) else None,
            "moved_away_rate": round(float((priced["no_vig_move"].astype(float) < 0).mean()), 4) if len(priced) else None,
            "mean_line_clv": round(float(df["line_clv"].dropna().mean()), 3) if df["line_clv"].notna().any() else None,
            "median_close_minutes_before_tip": (
                round(float(df["close_minutes_before_tip"].dropna().median()), 1)
                if df["close_minutes_before_tip"].notna().any() else None
            ),
            "graded": int(len(graded)),
            "hit_rate": round(float((graded["result"] == "WIN").mean()), 4) if len(graded) else None,
            "roi_units_per_bet": round(float(graded["profit_units"].mean()), 4) if graded["profit_units"].notna().any() else None,
        }
    usable = rows[rows["close_quality"].isin(["exact_line", "line_moved"])] if len(rows) else rows
    out: Dict[str, object] = {
        "all_rows": int(len(rows)),
        "close_quality_counts": rows["close_quality"].value_counts().to_dict() if len(rows) else {},
        "overall": _block(usable) if len(usable) else None,
        "groups": {},
    }
    if len(usable):
        for key, grp in usable.groupby(by):
            label = "|".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
            out["groups"][label] = _block(grp)
    return out


def main(argv: Optional[list] = None) -> int:
    """Build output/clv_picks.csv, output/clv_model_leans.csv and output/clv_summary.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-date", default=DEFAULT_START)
    args = parser.parse_args(argv)

    lines = load_fanduel_snapshots(args.start_date)
    print(f"FanDuel line rows loaded: {len(lines):,} across {lines['game_date'].nunique() if len(lines) else 0} slates")
    if lines.empty:
        print("No FanDuel snapshots on/after start date; CLV report not built.")
        return 0

    pick_runs = latest_run_times("picks", args.start_date)
    pred_runs = latest_run_times("predictions", args.start_date)
    picks = build_pick_rows(args.start_date)
    leans = build_lean_rows(pred_runs)
    print(f"Published FanDuel picks: {len(picks)} | model leans: {len(leans):,} "
          f"| slates with pick runs: {len(pick_runs)} | with prediction runs: {len(pred_runs)}")

    picks = attach_outcomes(attach_clv(picks, lines, pick_runs)) if len(picks) else picks
    leans = attach_outcomes(attach_clv(leans, lines, pred_runs)) if len(leans) else leans

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": args.start_date,
        "definition": "ev_vs_close = close no-vig side prob * bet decimal odds - 1; exact same line only",
        "picks": summarize(picks, ["stat"]) if len(picks) else None,
        "model_leans": summarize(leans, ["stat", "direction"]) if len(leans) else None,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    drop = [c for c in ["_pn"] if c in picks.columns]
    picks.drop(columns=drop).to_csv(OUTPUT_DIR / "clv_picks.csv", index=False)
    leans.drop(columns=[c for c in ["_pn"] if c in leans.columns]).to_csv(OUTPUT_DIR / "clv_model_leans.csv", index=False)
    (OUTPUT_DIR / "clv_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    for label, block in [("Picks", summary["picks"]), ("Model leans", summary["model_leans"])]:
        if not block or not block.get("overall"):
            reason = block.get("close_quality_counts") if block else "no rows"
            print(f"{label}: no rows with a later pre-tip snapshot yet ({reason})")
            continue
        o = block["overall"]
        print(f"{label}: {o['rows_with_exact_close']} with exact-line close | mean EV vs close "
              f"{o['mean_ev_vs_close']} (SE {o['ev_vs_close_std_error']}) | beat close {o['beat_close_rate']} | "
              f"no-vig move {o['mean_no_vig_move']} (toward {o['moved_toward_side_rate']}, away {o['moved_away_rate']}) | "
              f"graded {o['graded']} hit {o['hit_rate']} ROI {o['roi_units_per_bet']}")
    print("Wrote output/clv_picks.csv, output/clv_model_leans.csv, output/clv_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
