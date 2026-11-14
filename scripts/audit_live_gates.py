"""Read-only gate audit using recorded candidates, box scores and active policies.

Writes a separate audit receipt. Never refreshes providers or rewrites histories.
The candidate population starts AFTER generation and sanity filtering.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nba_props as nba


def summarize(frame):
    settled = frame[frame.result.isin(["WIN", "LOSS"])].copy()
    won = settled.result.eq("WIN")
    decimal = pd.to_numeric(settled.side_decimal_odds, errors="coerce")
    priced = np.isfinite(decimal) & decimal.gt(1)
    profit = pd.Series(np.where(won, decimal - 1, -1), index=settled.index)
    result = {
        "rows": len(frame), "wins": int(won.sum()),
        "losses": int((~won).sum()), "dnps": int(frame.result.eq("DNP").sum()),
        "pending": int(frame.result.eq("").sum()),
        "dates": int(settled.game_date.nunique()),
        "priced_n": int(priced.sum()),
        "profit_units": round(float(profit[priced].sum()), 5),
        "hit_rate": float(won.mean()) if len(settled) else None,
        "roi": float(profit[priced].mean()) if priced.any() else None,
    }
    if len(settled):
        daily = pd.DataFrame({"date": settled.game_date, "w": won.astype(int),
                              "n": 1, "units": profit.where(priced, 0),
                              "priced": priced.astype(int)}).groupby("date").sum()
        draws = np.random.default_rng(123).integers(0, len(daily), (20000, len(daily)))
        sums = daily.to_numpy()[draws].sum(axis=1)
        result["exploratory_date_bootstrap_hit_95"] = np.quantile(sums[:, 0] / sums[:, 1], [.025, .975]).tolist()
        if (sums[:, 3] > 0).all():
            result["exploratory_date_bootstrap_roi_95"] = np.quantile(sums[:, 2] / sums[:, 3], [.025, .975]).tolist()
    return result


def main():
    paths = ["output/candidates_history.csv", "output/picks_history.csv",
             "data/nba_data.csv", "models/training_results.json",
             "models/training_edge_analysis.json", "models/sigma_context_analysis.json"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    candidates = pd.read_csv(ROOT / paths[0], keep_default_na=False)
    picks = pd.read_csv(ROOT / paths[1], keep_default_na=False)
    box = pd.read_csv(ROOT / paths[2], low_memory=False)
    key = ["game_date", "player", "prop", "direction", "line"]
    binary = candidates[candidates.result.isin(["WIN", "LOSS"])].copy()
    box["game_date"] = pd.to_datetime(box.game_date).dt.strftime("%Y-%m-%d")
    box["player_norm"] = box.player.map(nba._normalize_name)
    checks = binary.copy()
    checks["player_norm"] = checks.player.map(nba._normalize_name)
    checks = checks.merge(box[["game_date", "player_norm", "ast", "stl", "blk"]],
                          on=["game_date", "player_norm"], how="left", validate="many_to_one")
    checks["box_actual"] = [row[str(row.prop).lower()] for _, row in checks.iterrows()]
    expected = np.where(checks.box_actual.eq(checks.line), "PUSH",
                        np.where(checks.box_actual.gt(checks.line).eq(checks.direction.eq("OVER")), "WIN", "LOSS"))
    checks["outcome_matches_box"] = checks.result.eq(expected)
    checks["actual_matches_box"] = checks.box_actual.eq(pd.to_numeric(checks.actual))
    snapshot = pd.to_datetime(candidates.line_snapshot_ts, utc=True, format="mixed", errors="coerce")
    tip = pd.to_datetime(candidates.commence_time, utc=True, format="mixed", errors="coerce")
    logged = pd.to_datetime(candidates.logged_at).dt.tz_localize("America/Chicago").dt.tz_convert("UTC")
    failure = candidates.gate_failures.astype(str)
    candidates["audit_gate_family"] = np.select(
        [candidates.gate_bucket.eq("published"), candidates.gate_bucket.eq("blocked_price"),
         failure.str.contains("meta ") & failure.str.contains("vol "),
         failure.str.contains("meta "), failure.eq("vol missing"),
         failure.str.startswith("vol ")],
        ["published", "price", "meta_and_volatility", "meta_only", "missing_volatility", "volatility_only"],
        default="selection_or_source")
    policy = nba._load_target_quality_policy()
    policies = {prop: {**rules, "final_meta_floor": nba._live_card_meta_floor(prop, policy),
                       "final_volatility_ceiling": nba._live_card_volatility_ceiling(prop)}
                for prop, rules in policy.items()}
    promoted = candidates[candidates.gate_bucket.eq("published") &
                          (candidates.price_gate_pass.astype(str).str.lower().eq("false") |
                           candidates.accuracy_gate_pass.astype(str).str.lower().eq("false"))]
    publication = promoted.merge(picks, on=key, suffixes=("_candidate", "_publication"), how="left")
    receipt = {
        "as_of_local": pd.Timestamp.now(tz="America/Chicago").isoformat(),
        "source_sha256": hashes, "population": "first pre-tip candidates after sanity checks; publication promotion is sticky",
        "dates": [candidates.game_date.min(), candidates.game_date.max()],
        "duplicate_candidate_keys": int(candidates.duplicated(key).sum()),
        "binary_grades_checked": len(checks),
        "box_outcome_mismatches": int((~checks.outcome_matches_box).sum()),
        "box_actual_mismatches": checks.loc[~checks.actual_matches_box,
            key + ["actual", "box_actual", "result"]].to_dict("records"),
        "unknown_snapshot": int(snapshot.isna().sum()), "unknown_tip": int(tip.isna().sum()),
        "snapshot_at_or_after_tip": int(snapshot.ge(tip).sum()),
        "candidate_logged_at_or_after_tip": int(logged.ge(tip).sum()),
        "snapshot_after_logged": int(snapshot.gt(logged).sum()),
        "max_snapshot_age_minutes": float(((logged-snapshot).dt.total_seconds()/60).max()),
        "by_bucket": {str(k): summarize(v) for k, v in candidates.groupby("gate_bucket")},
        "by_failure_family": {str(k): summarize(v) for k, v in candidates.groupby("audit_gate_family")},
        "by_price_failure": {str(k): summarize(v) for k, v in candidates[candidates.gate_bucket.eq("blocked_price")].assign(
            reason=lambda f: np.where(f.price_gate_reason.str.startswith("price "), "favorite_cap", "EV_or_other")).groupby("reason")},
        "active_policy": policies,
        "promoted_first_record_differs": publication.replace({np.nan: None}).to_dict("records"),
        "limitations": ["Early generation/sanity rejects are not in candidate history.",
                         "Stored first-candidate prices can differ from publication prices.",
                         "Bootstrap intervals are exploratory with few dates and prior outcome-driven tuning.",
                         "Current-policy evaluation requires a future frozen-policy sample."]}
    published = candidates[candidates.gate_bucket.eq("published")]
    for label, mask in {"meta_42_to_46": pd.to_numeric(published.meta_prob).between(42, 46, inclusive="left"),
                        "meta_at_least_46": pd.to_numeric(published.meta_prob).ge(46)}.items():
        receipt[label] = summarize(published[mask])
    receipt["probability_checks"] = {}
    for label, frame in [("all_settled_candidates", binary),
                         ("published_candidates", binary[binary.gate_bucket.eq("published")])]:
        y = frame.result.eq("WIN").astype(float)
        scores = {}
        for column in ["model_side_prob", "no_vig_side_prob"]:
            probability = pd.to_numeric(frame[column], errors="coerce")
            valid = np.isfinite(probability) & probability.between(0, 1)
            p = probability[valid].clip(1e-9, 1 - 1e-9)
            truth = y[valid]
            if len(p) == 0:
                # No settled rows with this probability: metrics are undefined, and NaN
                # would make json.dumps(allow_nan=False) abort the whole receipt.
                scores[column] = {"n": 0, "mean_probability": None, "observed_win_rate": None,
                                  "brier": None, "log_loss": None}
                continue
            scores[column] = {"n": len(p), "mean_probability": float(p.mean()),
                              "observed_win_rate": float(truth.mean()),
                              "brier": float(((p-truth)**2).mean()),
                              "log_loss": float(-(truth*np.log(p)+(1-truth)*np.log(1-p)).mean())}
        receipt["probability_checks"][label] = scores
    receipt["estimated_hit_floor"] = {
        "configured": nba.CONFIG["live_pick_estimated_hit_rate_floor"],
        "minimum_observed": float(pd.to_numeric(candidates.estimated_hit_rate).min()),
        "recorded_failures": int(failure.str.contains("hit ").sum())}
    destination = ROOT / "output/live_gate_audit_2026-11-12.json"
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    assert checks.outcome_matches_box.all(), "Inspect outcome mismatches in audit"
    print(json.dumps({k: receipt[k] for k in ["binary_grades_checked", "box_outcome_mismatches", "box_actual_mismatches", "duplicate_candidate_keys", "by_bucket", "by_price_failure"]}, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
