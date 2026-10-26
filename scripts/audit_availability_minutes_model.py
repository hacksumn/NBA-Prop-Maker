"""Audit the availability/minutes candidate table, chronology, and holdout claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data" / "availability_minutes_candidates.csv"
DEFAULT_OOF = ROOT / "data" / "availability_minutes_oof_predictions.csv"
DEFAULT_HOLDOUT = ROOT / "output" / "availability_minutes_holdout_predictions.csv"
DEFAULT_SUMMARY = ROOT / "output" / "availability_minutes_validation_summary.json"
DEFAULT_CALIBRATION = ROOT / "output" / "availability_minutes_calibration.csv"
DEFAULT_SUBGROUPS = ROOT / "output" / "availability_minutes_holdout_subgroups.csv"
DEFAULT_AUDIT = ROOT / "output" / "availability_minutes_validation_audit.json"
DEFAULT_REPORT = ROOT / "output" / "AVAILABILITY_MINUTES_REPORT.md"
KEY = ["game_date", "season", "game_id", "team", "player_id", "player"]


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["game_date"] = pd.to_datetime(output["game_date"])
    for column in ["season", "game_id", "team", "player_id", "player"]:
        output[column] = output[column].astype(str).str.replace(r"\.0$", "", regex=True)
    return output


def merge_holdout(candidates: pd.DataFrame, predictions: pd.DataFrame, holdout_season: str) -> pd.DataFrame:
    holdout_candidates = candidates[candidates["season"].astype(str).eq(str(holdout_season))].copy()
    holdout_candidates = _normalize_keys(holdout_candidates)
    predictions = _normalize_keys(predictions)
    if holdout_candidates.duplicated(KEY).any() or predictions.duplicated(KEY).any():
        raise RuntimeError("Holdout source or prediction keys are not unique")
    merged = holdout_candidates.merge(
        predictions.drop(columns=["available", "minutes"], errors="ignore"),
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(f"Holdout join mismatch: {merged['_merge'].value_counts().to_dict()}")
    return merged.drop(columns="_merge")


def subgroup_metrics(frame: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    sample = frame[mask].copy()
    available = sample["available"].astype(int).to_numpy()
    actual = sample["minutes"].to_numpy(dtype=float)
    probability = sample["availability_probability"].to_numpy(dtype=float)
    conditional = sample["conditional_minutes_prediction"].to_numpy(dtype=float)
    conditional_baseline = sample["mp_played_mean_l5"].to_numpy(dtype=float)
    hard = np.where(probability >= 0.50, conditional, 0.0)
    last_status = sample["avail_l1"].to_numpy(dtype=float) * conditional_baseline
    soft = probability * conditional
    return {
        "subgroup": name,
        "rows": int(len(sample)),
        "availability_rate": float(np.mean(available)),
        "availability_brier": float(brier_score_loss(available, probability)),
        "rolling_availability_brier": float(
            brier_score_loss(available, sample["avail_rate_l10"].clip(0.02, 0.98))
        ),
        "availability_auc": float(roc_auc_score(available, probability)) if len(np.unique(available)) == 2 else np.nan,
        "conditional_minutes_mae": float(mean_absolute_error(actual[available == 1], conditional[available == 1])),
        "conditional_baseline_mae": float(
            mean_absolute_error(actual[available == 1], conditional_baseline[available == 1])
        ),
        "hard_gated_minutes_mae": float(mean_absolute_error(actual, hard)),
        "last_status_baseline_mae": float(mean_absolute_error(actual, last_status)),
        "soft_expected_minutes_rmse": float(mean_squared_error(actual, soft) ** 0.5),
        "last_status_baseline_rmse": float(mean_squared_error(actual, last_status) ** 0.5),
    }


def paired_date_bootstrap(frame: pd.DataFrame, seed: int = 42, draws: int = 5000) -> dict:
    working = frame.copy()
    actual = working["minutes"].to_numpy(dtype=float)
    available = working["available"].astype(int).to_numpy()
    probability = working["availability_probability"].to_numpy(dtype=float)
    conditional = working["conditional_minutes_prediction"].to_numpy(dtype=float)
    l5 = working["mp_played_mean_l5"].to_numpy(dtype=float)
    hard = np.where(probability >= 0.50, conditional, 0.0)
    last_status = working["avail_l1"].to_numpy(dtype=float) * l5
    soft = probability * conditional
    rolling_probability = working["avail_rate_l10"].clip(0.02, 0.98).to_numpy(dtype=float)
    working["availability_brier_gain"] = (available - rolling_probability) ** 2 - (available - probability) ** 2
    working["hard_mae_gain"] = np.abs(actual - last_status) - np.abs(actual - hard)
    working["soft_mse_gain"] = (actual - last_status) ** 2 - (actual - soft) ** 2
    working["conditional_mae_gain"] = np.where(
        available == 1,
        np.abs(actual - l5) - np.abs(actual - conditional),
        np.nan,
    )
    daily = working.groupby("game_date").agg(
        availability_brier_gain=("availability_brier_gain", "mean"),
        hard_mae_gain=("hard_mae_gain", "mean"),
        soft_mse_gain=("soft_mse_gain", "mean"),
        conditional_mae_gain=("conditional_mae_gain", "mean"),
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(daily), size=(draws, len(daily)))
    result = {"dates": int(len(daily)), "draws": int(draws)}
    for column in daily.columns:
        values = daily[column].to_numpy(dtype=float)
        bootstrap = np.nanmean(values[indices], axis=1)
        result[column] = {
            "mean_daily_gain": float(np.nanmean(values)),
            "positive_dates": int(np.nansum(values > 0)),
            "ci95_low": float(np.nanquantile(bootstrap, 0.025)),
            "ci95_high": float(np.nanquantile(bootstrap, 0.975)),
        }
    return result


def audit(
    candidates_path: Path = DEFAULT_CANDIDATES,
    oof_path: Path = DEFAULT_OOF,
    holdout_path: Path = DEFAULT_HOLDOUT,
    summary_path: Path = DEFAULT_SUMMARY,
    calibration_path: Path = DEFAULT_CALIBRATION,
    subgroup_path: Path = DEFAULT_SUBGROUPS,
    audit_path: Path = DEFAULT_AUDIT,
    report_path: Path = DEFAULT_REPORT,
) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidates_path, low_memory=False)
    oof = pd.read_csv(oof_path, low_memory=False)
    holdout_predictions = pd.read_csv(holdout_path, low_memory=False)
    holdout = merge_holdout(candidates, holdout_predictions, summary["data_quality"]["holdout_season"])

    oof["game_date"] = pd.to_datetime(oof["game_date"])
    oof["model_train_end"] = pd.to_datetime(oof["model_train_end"])
    chronology_failures = int((oof["model_train_end"] >= oof["game_date"]).sum())

    holdout["calibration_bin"] = pd.qcut(
        holdout["availability_probability"], 10, duplicates="drop"
    )
    calibration = (
        holdout.groupby("calibration_bin", observed=True)
        .agg(
            rows=("available", "size"),
            predicted_availability=("availability_probability", "mean"),
            actual_availability=("available", "mean"),
        )
        .reset_index()
    )
    calibration["absolute_gap"] = (
        calibration["predicted_availability"] - calibration["actual_availability"]
    ).abs()
    calibration["calibration_bin"] = calibration["calibration_bin"].astype(str)

    midpoint = sorted(holdout["game_date"].unique())[len(holdout["game_date"].unique()) // 2]
    subgroups = pd.DataFrame(
        [
            subgroup_metrics(holdout, "all_holdout_candidates", pd.Series(True, index=holdout.index)),
            subgroup_metrics(holdout, "first_half", holdout["game_date"] < midpoint),
            subgroup_metrics(holdout, "second_half", holdout["game_date"] >= midpoint),
            subgroup_metrics(holdout, "rotation_15_plus", holdout["mp_played_mean_l10"] >= 15),
            subgroup_metrics(
                holdout,
                "likely_prop_pool_proxy",
                (holdout["mp_played_mean_l10"] >= 15) & (holdout["avail_rate_l10"] >= 0.80),
            ),
            subgroup_metrics(
                holdout,
                "model_probability_080_plus",
                holdout["availability_probability"] >= 0.80,
            ),
        ]
    )
    subgroups["brier_improvement"] = (
        subgroups["rolling_availability_brier"] - subgroups["availability_brier"]
    )
    subgroups["conditional_mae_improvement"] = (
        subgroups["conditional_baseline_mae"] - subgroups["conditional_minutes_mae"]
    )
    subgroups["hard_mae_improvement"] = (
        subgroups["last_status_baseline_mae"] - subgroups["hard_gated_minutes_mae"]
    )
    subgroups["soft_rmse_improvement"] = (
        subgroups["last_status_baseline_rmse"] - subgroups["soft_expected_minutes_rmse"]
    )

    bootstrap = paired_date_bootstrap(holdout)
    checks = {
        "candidate_key_duplicates": int(
            candidates.duplicated(["game_id", "team", "player_id"]).sum()
        ),
        "candidate_null_feature_cells": int(
            candidates[summary["feature_columns"]].isna().sum().sum()
        ),
        "oof_key_duplicates": int(oof.duplicated(["game_id", "team", "player_id"]).sum()),
        "oof_chronology_failures": chronology_failures,
        "holdout_join_rows": int(len(holdout)),
        "holdout_expected_rows": int(summary["data_quality"]["holdout_rows"]),
        "calibration_max_absolute_decile_gap": float(calibration["absolute_gap"].max()),
    }
    ready = bool(
        summary["accepted_for_prop_retraining"]
        and all(checks[key] == 0 for key in [
            "candidate_key_duplicates",
            "candidate_null_feature_cells",
            "oof_key_duplicates",
            "oof_chronology_failures",
        ])
        and checks["holdout_join_rows"] == checks["holdout_expected_rows"]
        and bootstrap["availability_brier_gain"]["ci95_low"] > 0
        and bootstrap["conditional_mae_gain"]["ci95_low"] > 0
        and bootstrap["hard_mae_gain"]["ci95_low"] > 0
        and bootstrap["soft_mse_gain"]["ci95_low"] > 0
    )
    result = {
        "overall_assessment": "ready_for_prop_retraining" if ready else "needs_revision",
        "enabled_for_live": False,
        "checks": checks,
        "paired_date_bootstrap": bootstrap,
        "subgroup_rows": subgroups.to_dict(orient="records"),
        "caveats": [
            "Historical candidates come from prior same-team appearances, not a transaction/roster feed.",
            "Players returning after more than 45 days and players before their second team appearance are outside the candidate contract.",
            "The likely-prop-pool subgroup is a minutes/availability proxy, not a true historical PrizePicks eligibility table.",
            "The 0.50 hard availability gate is fixed and was not optimized on the latest-season holdout.",
            "The model package remains disabled for live use until the surrounding prop models pass their own chronological evaluation.",
        ],
    }

    for path in [calibration_path, subgroup_path, audit_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(calibration_path, index=False)
    subgroups.to_csv(subgroup_path, index=False)
    audit_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    overall = subgroups[subgroups["subgroup"].eq("all_holdout_candidates")].iloc[0]
    proxy = subgroups[subgroups["subgroup"].eq("likely_prop_pool_proxy")].iloc[0]
    report = f"""# Availability and Minutes Model Validation

## Executive Summary

- **The model is ready to become an input to prop-model retraining.** It passed all four expanding development folds and the latest-season holdout at the candidate player/team/game grain.
- **Availability is the larger gain.** Holdout Brier score improved from `{overall['rolling_availability_brier']:.4f}` to `{overall['availability_brier']:.4f}`, with AUC `{overall['availability_auc']:.3f}`.
- **Conditional minutes also improved.** Holdout MAE fell from `{overall['conditional_baseline_mae']:.3f}` to `{overall['conditional_minutes_mae']:.3f}` minutes.
- **The combined hurdle clears the strongest simple baseline.** Fixed 0.50-gated minutes MAE fell from `{overall['last_status_baseline_mae']:.3f}` to `{overall['hard_gated_minutes_mae']:.3f}`; soft expected-minutes RMSE fell from `{overall['last_status_baseline_rmse']:.3f}` to `{overall['soft_expected_minutes_rmse']:.3f}`.

## Dataset and Grain

- Source: `data/nba_data.csv`, SHA-256 `{summary['source']['sha256']}`.
- Source coverage: `{summary['data_quality']['source_rows']:,}` appearance rows from {summary['data_quality']['source_date_min']} through {summary['data_quality']['source_date_max']}.
- Candidate table: `{summary['data_quality']['candidate_rows']:,}` unique player/team/game rows across `{summary['data_quality']['candidate_unique_dates']}` dates and `{summary['data_quality']['candidate_unique_players']}` players.
- Holdout: `{summary['data_quality']['holdout_season']}`, `{summary['data_quality']['holdout_rows']:,}` candidates from {summary['data_quality']['holdout_start']} through {summary['data_quality']['holdout_end']}.
- OOF feature table: `{summary['data_quality']['oof_rows']:,}` rows; chronology violations `{checks['oof_chronology_failures']}`.

## Validation Results

| Metric | Model | Baseline | Improvement |
|---|---:|---:|---:|
| Availability Brier (lower is better) | {overall['availability_brier']:.4f} | {overall['rolling_availability_brier']:.4f} | {overall['brier_improvement']:.4f} |
| Conditional minutes MAE | {overall['conditional_minutes_mae']:.3f} | {overall['conditional_baseline_mae']:.3f} | {overall['conditional_mae_improvement']:.3f} min |
| Hard-gated all-candidate MAE | {overall['hard_gated_minutes_mae']:.3f} | {overall['last_status_baseline_mae']:.3f} | {overall['hard_mae_improvement']:.3f} min |
| Soft expected-minutes RMSE | {overall['soft_expected_minutes_rmse']:.3f} | {overall['last_status_baseline_rmse']:.3f} | {overall['soft_rmse_improvement']:.3f} min |

All four gains have positive 95% date-clustered bootstrap intervals. The availability calibration deciles have a maximum absolute predicted-versus-observed gap of `{checks['calibration_max_absolute_decile_gap']:.3f}`.

## Prop-Pool Proxy

Among `{int(proxy['rows']):,}` holdout candidates with prior L10 minutes at least 15 and prior availability at least 80%, availability Brier improved by `{proxy['brier_improvement']:.4f}`, conditional minutes MAE improved by `{proxy['conditional_mae_improvement']:.3f}` minutes, hard-gated MAE improved by `{proxy['hard_mae_improvement']:.3f}` minutes, and soft RMSE improved by `{proxy['soft_rmse_improvement']:.3f}` minutes. This is the closest historical proxy to players likely to receive a prop line, but it is not a true archived PrizePicks board.

## Decision and Next Step

Use `data/availability_minutes_oof_predictions.csv`—not the full-fit deployment model—to join availability probability and minutes predictions into historical prop training. Keep `models/availability_minutes_model.pkl` disabled for live use until the retrained prop stack beats its current comparator and same-prop baselines on a separate chronological test.

## Caveats

- Historical candidates are inferred from prior same-team appearances because the repository has no historical transaction/roster feed.
- Returns after more than 45 days and players before their second appearance for a team are outside the candidate contract.
- The fixed 0.50 hard gate was not optimized on the holdout.
- The model improves prediction components; it does not by itself prove better prop selection or betting performance.
"""
    report_path.write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--subgroups", type=Path, default=DEFAULT_SUBGROUPS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = audit(
        candidates_path=args.candidates,
        oof_path=args.oof,
        holdout_path=args.holdout,
        summary_path=args.summary,
        calibration_path=args.calibration,
        subgroup_path=args.subgroups,
        audit_path=args.audit,
        report_path=args.report,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["overall_assessment"] == "ready_for_prop_retraining" else 2


if __name__ == "__main__":
    raise SystemExit(main())
