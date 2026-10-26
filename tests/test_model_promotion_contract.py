import hashlib
import json
import pickle

from scripts.promote_defense_rating_models import _validate_candidate


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(tmp_path, *, split_method="chronological_latest_dates_holdout"):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    selected = {
        "stl": ["opp_def_rtg_pt"],
        "blk": ["opp_pace_pt"],
    }
    (candidate / "selected_features_advanced.json").write_text(
        json.dumps(selected), encoding="utf-8"
    )
    (candidate / "training_results.json").write_text("{}", encoding="utf-8")
    (candidate / "defense_rating_feature_contract.json").write_text(
        json.dumps(
            {
                "rows": 74254,
                "chronology_violations": 0,
                "feature_columns": [
                    "opp_def_rtg_pt",
                    "opp_pace_pt",
                    "opp_def_tier_pt",
                ],
            }
        ),
        encoding="utf-8",
    )
    oof = candidate / "probability_oof_advanced.csv"
    oof.write_text("target,game_date\nast,2026-01-01\n", encoding="utf-8")
    sidecars = {}
    for target in ("ast", "blk"):
        path = candidate / f"prob_sidecar_{target}_advanced.pkl"
        with path.open("wb") as handle:
            pickle.dump({"target": target}, handle)
        sidecars[target] = _sha256(path)
    bundle = {"sidecar_sha256": sidecars, "oof_sha256": _sha256(oof)}
    split = {
        "method": split_method,
        "holdout_dates": 20,
        "holdout_rows": 500,
    }
    with (candidate / "probability_calibrator_advanced.pkl").open("wb") as handle:
        pickle.dump({"sidecar_contract": bundle, "diagnostics": {"split": split}}, handle)
    (candidate / "probability_diagnostics_advanced.json").write_text(
        json.dumps({"sidecar_contract": bundle, "split": split}), encoding="utf-8"
    )
    return candidate


def test_candidate_validation_accepts_current_row_counts(tmp_path):
    contract, selected = _validate_candidate(_candidate(tmp_path))
    assert contract["rows"] == 74254
    assert selected["stl"] == ["opp_def_rtg_pt"]


def test_candidate_validation_rejects_in_sample_calibration(tmp_path):
    candidate = _candidate(tmp_path, split_method="all_rows_fit")
    try:
        _validate_candidate(candidate)
    except ValueError as exc:
        assert "chronological holdout" in str(exc)
    else:
        raise AssertionError("in-sample calibration should not be promotable")
