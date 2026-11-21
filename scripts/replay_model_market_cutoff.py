"""Replay DEC-104 A on immutable logged offers; never generate or publish picks.

Usage: python scripts/replay_model_market_cutoff.py
Outputs go to a new timestamped directory; existing output is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["game_date", "player", "prop", "direction", "line"]
CUTOFF = 0.10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(frame):
    """Keep exactly +10pp; unknown probabilities cannot support a replay decision."""
    out = frame.copy()
    model = pd.to_numeric(out.model_side_prob, errors="coerce")
    market = pd.to_numeric(out.no_vig_side_prob, errors="coerce")
    valid = model.between(0, 1) & market.between(0, 1)
    applicable = out.direction.eq("UNDER") & out.line_source.eq("fanduel")
    gap = model - market
    above = gap.gt(CUTOFF) & ~np.isclose(gap, CUTOFF, rtol=0, atol=1e-12)
    out["replay_gap"] = gap.where(valid)
    out["replay_action"] = np.select(
        [~applicable, ~valid, above],
        ["outside_scope", "unknown", "remove"], default="keep")
    return out


def summarize(frame):
    """Flat one-unit singles; ROI denominator is valid-priced WIN/LOSS stakes."""
    binary = frame[frame.result.isin(["WIN", "LOSS"])]
    decimal = pd.to_numeric(binary.side_decimal_odds, errors="coerce")
    valid = np.isfinite(decimal) & decimal.gt(1)
    priced = binary[valid]
    profit = np.where(priced.result.eq("WIN"), decimal[valid] - 1, -1)
    return {
        "rows": len(frame), "settled_binary": len(binary),
        "wins": int(binary.result.eq("WIN").sum()),
        "losses": int(binary.result.eq("LOSS").sum()),
        "pushes": int(frame.result.eq("PUSH").sum()),
        "dnps": int(frame.result.eq("DNP").sum()),
        "pending_or_other": int((~frame.result.isin(["WIN", "LOSS", "PUSH", "DNP"])).sum()),
        "dates": int(binary.game_date.nunique()),
        "hit_rate": float(binary.result.eq("WIN").mean()) if len(binary) else None,
        "priced_binary": len(priced),
        "unpriced_binary": int((~valid).sum()),
        "profit_units": float(profit.sum()) if len(priced) else None,
        "roi": float(profit.mean()) if len(priced) else None,
    }


def comparison(frame):
    known = frame[frame.replay_action.isin(["keep", "remove"])]
    return {
        "all_logged": summarize(frame),
        "comparable_before": summarize(known),
        "after_cutoff": summarize(known[known.replay_action.eq("keep")]),
        "removed": summarize(known[known.replay_action.eq("remove")]),
        "unknown": summarize(frame[frame.replay_action.eq("unknown")]),
        "outside_scope": summarize(frame[frame.replay_action.eq("outside_scope")]),
    }


def bootstrap(frame):
    """Exploratory paired date-cluster ROI intervals; includes nights with no keeps."""
    binary = frame[frame.result.isin(["WIN", "LOSS"]) &
                   frame.replay_action.isin(["keep", "remove"])].copy()
    odds = pd.to_numeric(binary.side_decimal_odds, errors="coerce")
    binary = binary[np.isfinite(odds) & odds.gt(1)].copy()
    if binary.empty:
        return {}
    binary["profit"] = np.where(binary.result.eq("WIN"), odds.loc[binary.index] - 1, -1)
    binary["stake"] = 1
    binary["kept_profit"] = binary.profit.where(binary.replay_action.eq("keep"), 0)
    binary["kept_stake"] = binary.replay_action.eq("keep").astype(int)
    daily = binary.groupby("game_date")[["profit", "stake", "kept_profit", "kept_stake"]].sum()
    draws = np.random.default_rng(104).integers(0, len(daily), (20000, len(daily)))
    sums = daily.to_numpy()[draws].sum(axis=1)
    usable = sums[:, 3] > 0
    sums = sums[usable]
    return {
        "dates": len(daily), "draws": 20000, "usable_draws": len(sums), "seed": 104,
        "before_roi_95": np.quantile(sums[:, 0] / sums[:, 1], [.025, .975]).tolist(),
        "after_roi_95": np.quantile(sums[:, 2] / sums[:, 3], [.025, .975]).tolist(),
        "paired_roi_difference_95": np.quantile(
            sums[:, 2] / sums[:, 3] - sums[:, 0] / sums[:, 1], [.025, .975]).tolist(),
    }


def run(root, destination):
    """Read histories once, validate keys, and write only a fresh replay directory."""
    sources = {name: root / "output" / f"{name}_history.csv" for name in ["picks", "candidates"]}
    hashes = {str(path.relative_to(root)): digest(path) for path in sources.values()}
    frames = {}
    for name, path in sources.items():
        frame = pd.read_csv(path, low_memory=False)
        if frame.duplicated(KEY).any():
            raise ValueError(f"Duplicate exact contracts in {path}; resolve before replay")
        frames[name] = classify(frame)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": CUTOFF, "rule": "FanDuel UNDER: model_side_prob - no_vig_side_prob > 0.10",
        "source_sha256": hashes, "script_sha256": digest(__file__),
        "method": "Removal only on recorded offers, no refill or retrospective reranking; no training.",
        "limitations": [
            "Threshold discovered on this history; this is exploratory, not an independent holdout.",
            "Candidate and published populations overlap; they are not independent replications.",
            "Older rows without probabilities are unassessable, never reconstructed with today's model/prices.",
            "Candidate promotion preserves first-observed prices/probabilities; use published history for published-card P&L.",
            "Flat one-unit singles only; no parlay or actual stake/bankroll simulation.",
            "Pushes, DNPs and pending rows excluded from binary hit rate and ROI denominators.",
        ],
        "populations": {name: comparison(frame) for name, frame in frames.items()},
        "exploratory_date_bootstrap": {name: bootstrap(frame) for name, frame in frames.items()},
    }
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    receipt["git_head"] = head.stdout.strip() if head.returncode == 0 else None
    # Detect concurrent scheduled writers; never label changed inputs as the same snapshot.
    if hashes != {str(path.relative_to(root)): digest(path) for path in sources.values()}:
        raise RuntimeError("Histories changed during replay; rerun on a stable snapshot")
    destination.mkdir(parents=True, exist_ok=False)
    grouped = []
    for name, frame in frames.items():
        frame.to_csv(destination / f"{name}_rows.csv", index=False)
        for dimension in ["game_date", "prop"]:
            for (value, action), group in frame.groupby([dimension, "replay_action"], dropna=False):
                grouped.append({"population": name, "dimension": dimension, "value": value,
                                "action": action, **summarize(group)})
    pd.DataFrame(grouped).to_csv(destination / "breakdowns.csv", index=False)
    (destination / "summary.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = args.output_dir or args.root / "output" / ("model_market_cutoff_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    receipt = run(args.root, destination)
    print(destination)
    for name, result in receipt["populations"].items():
        print(name, json.dumps({key: result[key] for key in ["comparable_before", "after_cutoff", "removed"]}))


if __name__ == "__main__":
    main()
