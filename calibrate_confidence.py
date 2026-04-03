"""
calibrate_confidence.py
=======================
Trains an isotonic regression calibrator that maps raw confidence scores
to real win probabilities, using picks_history.csv as ground truth.

Also builds per-stat-direction empirical win rates used for ranking picks.

Run after you have 500+ graded picks:
    python calibrate_confidence.py

Outputs:
    models/confidence_calibrator.pkl
"""
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT       = Path(__file__).parent
HIST_PATH  = ROOT / "output" / "picks_history.csv"
MODEL_DIR  = ROOT / "models"
OUT_PATH   = MODEL_DIR / "confidence_calibrator.pkl"

MODEL_DIR.mkdir(exist_ok=True)

def main():
    if not HIST_PATH.exists():
        print("picks_history.csv not found — run predict first")
        return

    hist = pd.read_csv(HIST_PATH)
    graded = hist[hist["result"].isin(["WIN", "LOSS"])].copy()
    graded["win"] = (graded["result"] == "WIN").astype(int)
    graded["confidence"] = pd.to_numeric(graded["confidence"], errors="coerce")
    graded = graded.dropna(subset=["confidence", "win"])

    print(f"Graded picks: {len(graded):,}")
    print(f"Overall win rate: {graded['win'].mean()*100:.1f}%\n")

    # ── 1. Global isotonic calibration: confidence → P(win) ──────────────────
    X = graded["confidence"].values
    y = graded["win"].values
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X, y)

    # ── 2. Per-stat-direction empirical win rates ─────────────────────────────
    stat_dir_rates = {}
    for (prop, direction), g in graded.groupby(["prop", "direction"]):
        if len(g) < 10:
            continue
        wr = g["win"].mean()
        stat_dir_rates[f"{prop.upper()}_{direction.upper()}"] = round(wr, 4)
        print(f"  {prop.upper()} {direction.upper():5s}: {g['win'].sum():3d}/{len(g):3d} = {wr*100:.1f}%")

    print()

    # ── 3. Confidence bucket breakdown (before vs after calibration) ──────────
    graded["bucket"] = pd.cut(
        graded["confidence"],
        bins=[0, 60, 65, 70, 75, 80, 85, 90, 100],
        labels=["<60","60-65","65-70","70-75","75-80","80-85","85-90","90+"],
    )
    graded["cal_conf"] = iso.predict(X) * 100

    print("Bucket  | Raw Conf | Actual Win% | Cal Conf")
    print("-" * 48)
    for bucket, g in graded.groupby("bucket", observed=True):
        actual = g["win"].mean() * 100
        cal    = g["cal_conf"].mean()
        raw    = g["confidence"].mean()
        print(f"  {str(bucket):6s}  |  {raw:5.1f}   |   {actual:5.1f}%     |  {cal:5.1f}%")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    calibrator = {
        "iso":            iso,
        "stat_dir_rates": stat_dir_rates,
        "n_graded":       len(graded),
        "overall_wr":     float(graded["win"].mean()),
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"\nCalibrator saved -> {OUT_PATH}")
    print(f"Stat-direction rates: {len(stat_dir_rates)} entries")


if __name__ == "__main__":
    main()
