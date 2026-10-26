import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_confidence


def test_standalone_calibration_preserves_latest_dates_as_holdout(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    rows = []
    for date_index, game_date in enumerate(dates):
        for row_index in range(10):
            probability = 0.15 + 0.7 * ((date_index * 10 + row_index) % 100) / 99
            rows.append(
                {
                    "target": "ast",
                    "game_date": game_date.strftime("%Y-%m-%d"),
                    "line_source": "real",
                    "p_over_raw": probability,
                    "actual_over": int((date_index + row_index) % 3 == 0),
                    "sample_weight": 1.0,
                }
            )
    oof_path = model_dir / "probability_oof_advanced.csv"
    pd.DataFrame(rows).to_csv(oof_path, index=False)
    with (model_dir / "prob_sidecar_ast_advanced.pkl").open("wb") as handle:
        pickle.dump({"target": "ast"}, handle)

    monkeypatch.setattr(calibrate_confidence, "MODEL_DIR", model_dir)
    monkeypatch.setattr(calibrate_confidence, "PROB_OOF_PATH", oof_path)
    monkeypatch.setattr(
        calibrate_confidence,
        "PROB_OUT_PATH",
        model_dir / "probability_calibrator_advanced.pkl",
    )
    monkeypatch.setattr(
        calibrate_confidence,
        "PROB_DIAG_PATH",
        model_dir / "probability_diagnostics_advanced.json",
    )
    monkeypatch.setattr(
        calibrate_confidence,
        "SIGMA_CONTEXT_PATH",
        model_dir / "sigma_context_analysis.json",
    )

    calibrate_confidence.calibrate_probability_outputs(holdout_dates=5)
    diagnostics = json.loads(
        (model_dir / "probability_diagnostics_advanced.json").read_text(encoding="utf-8")
    )
    assert diagnostics["split"]["fit_dates"] == 20
    assert diagnostics["split"]["holdout_dates"] == 5
    assert diagnostics["split"]["fit_date_max"] == "2026-01-20"
    assert diagnostics["split"]["holdout_date_min"] == "2026-01-21"
    assert "holdout" in diagnostics and "ast" in diagnostics["holdout"]
