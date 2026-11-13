"""Freeze and evaluate future gate decisions from immutable live run receipts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nba_props as nba
from gate_decisions import start_run


def initialize():
    path = ROOT / "output/gate_experiment.json"
    if path.exists():
        return path
    run = start_run(nba.CONFIG, nba._load_target_quality_policy())
    specification = {"schema_version": 1, "created_at": run["started_at"],
                     "start_date": "2026-11-13", "minimum_completed_dates": 30,
                     "minimum_binary_contracts": 300,
                     "policy_sha256": run["policy_sha256"],
                     "model_sha256": run["model_sha256"],
                     "source_sha256": run["source_sha256"],
                     "policy": run["policy"], "config": run["config"],
                     "comparisons": ["published", "incumbent_eligible", "meta_floor_46", "volatility_mult_1_25"],
                     "primary_metric": "flat-unit return at recorded paired price",
                     "review_rule": "Review once both minimums are met. Compare same-offer policies and date-level uncertainty. No automatic promotion.",
                     "scope": "final candidate gates after sanity; alternatives measure candidate eligibility before one-player card selection"}
    with path.open("x", encoding="utf-8") as destination:
        json.dump(specification, destination, indent=2)
    return path


def evaluate():
    specification_path = ROOT / "output/gate_experiment.json"
    if not specification_path.exists():
        return {"status": "not_initialized"}
    spec = json.loads(specification_path.read_text(encoding="utf-8"))
    rows = []
    drift = 0
    for path in sorted((ROOT / "output/gate_runs").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["slate_date"] < spec["start_date"]:
            continue
        if (run["policy_sha256"] != spec["policy_sha256"] or
                run["model_sha256"] != spec["model_sha256"] or
                run["source_sha256"] != spec["source_sha256"]):
            drift += 1
            continue
        published = {(r["player"], r["prop"], r["direction"], float(r["line"])) for r in run["published"]}
        for candidate in run["final_candidates"]:
            row = {**candidate, "game_date": run["slate_date"], "run_started_at": run["started_at"],
                   "decision_saved_at": run["publication_saved_at"]}
            row["was_published"] = (row["player"], row["prop"], row["direction"], float(row["line"])) in published
            rows.append(row)
    result = {"status": "collecting", "start_date": spec["start_date"],
              "runs_excluded_for_drift": drift, "binary_contracts": 0,
              "completed_dates": 0, "comparisons": {},
              "scope": spec["scope"], "review_eligible": False}
    if rows:
        frame = pd.DataFrame(rows).sort_values("run_started_at")
        frame = frame.drop_duplicates(["game_date", "player", "prop", "direction", "line"], keep="first")
        snapshot = pd.to_datetime(frame.line_snapshot_ts, errors="coerce", utc=True, format="mixed")
        tip = pd.to_datetime(frame.commence_time, errors="coerce", utc=True, format="mixed")
        captured = pd.to_datetime(frame.decision_saved_at, utc=True, format="mixed")
        frame = frame[snapshot.le(captured) & captured.lt(tip)].copy()
        box = pd.read_csv(ROOT / "data/nba_data.csv", low_memory=False)
        box["game_date"] = pd.to_datetime(box.game_date).dt.strftime("%Y-%m-%d")
        box["player_norm"] = box.player.map(nba._normalize_name)
        lookup = box.set_index(["game_date", "player_norm"])
        graded = []
        for _, row in frame.iterrows():
            if row.game_date >= pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d"):
                continue
            key = (row.game_date, nba._normalize_name(row.player))
            prop = str(row.prop).lower()
            if key not in lookup.index or prop not in lookup.columns:
                continue
            actual = pd.to_numeric(lookup.loc[key, prop], errors="coerce")
            if pd.isna(actual) or actual == row.line:
                continue
            row = row.copy()
            row["win"] = bool((actual > row.line) == (row.direction == "OVER"))
            graded.append(row)
        if graded:
            data = pd.DataFrame(graded)
            source = data.get("line_source", pd.Series("", index=data.index))
            expected_source = spec.get("config", {}).get("live_sportsbook_key", "fanduel")
            base = (source.eq(expected_source) & data.price_gate_pass.eq(True) & data.estimated_hit_rate_pass.eq(True)
                    & data.pick_source_gate_pass.eq(True)
                    & ~data.pick_source.isin(["volume_fill", "emergency_fill"]))
            masks = {
                "published": data.was_published,
                "incumbent_eligible": base & data.meta_gate_pass & data.volatility_gate_pass,
                "meta_floor_46": base & data.volatility_gate_pass &
                    (data.meta_gate_mode.eq("off") | pd.to_numeric(data.meta_prob).ge(np.maximum(data.meta_floor, 46))),
                "volatility_mult_1_25": base & data.meta_gate_pass &
                    pd.to_numeric(data.volatility_metric).le(data.volatility_ceiling * (1.25 / 1.50))}
            result["binary_contracts"] = len(data)
            result["completed_dates"] = int(data.game_date.nunique())
            for name, mask in masks.items():
                part = data[mask].copy()
                profit = np.where(part.win, pd.to_numeric(part.side_decimal_odds) - 1, -1)
                result["comparisons"][name] = {"n": len(part), "wins": int(part.win.sum()),
                    "profit_units": float(profit.sum()), "roi": float(profit.mean()) if len(part) else None}
            unit_return = pd.Series(np.where(data.win, pd.to_numeric(data.side_decimal_odds) - 1, -1), index=data.index)
            baseline = unit_return.where(masks["incumbent_eligible"], 0)
            for name in ["meta_floor_46", "volatility_mult_1_25"]:
                delta = unit_return.where(masks[name], 0) - baseline
                daily = pd.DataFrame({"date": data.game_date, "delta": delta, "n": 1}).groupby("date").sum()
                draws = np.random.default_rng(123).integers(0, len(daily), (10000, len(daily)))
                totals = daily.to_numpy()[draws].sum(axis=1)
                result["comparisons"][name]["incremental_units_per_offered_contract"] = float(delta.mean())
                result["comparisons"][name]["exploratory_date_bootstrap_delta_95"] = np.quantile(totals[:, 0]/totals[:, 1], [.025, .975]).tolist()
            result["review_eligible"] = (len(data) >= spec["minimum_binary_contracts"] and
                                         result["completed_dates"] >= spec["minimum_completed_dates"])
    nba._atomic_text(json.dumps(result, indent=2), ROOT / "output/gate_experiment_summary.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    if args.initialize:
        initialize()
    print(json.dumps(evaluate(), indent=2))
