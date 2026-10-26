import pandas as pd

from scripts.audit_training_live_feature_parity import compare_selected_features


def _frame(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": [1, 2],
            "PLAYER_ID": [11, 11],
            "team": ["AAA", "AAA"],
            "stable": [1.0, 2.0],
            "drifted": [value, value + 1.0],
        }
    )


def test_parity_audit_identifies_selected_drift_and_affected_prop():
    training = _frame(3.0)
    live = _frame(4.0)
    selected = {"pts": ["stable", "drifted"], "trb": ["stable"]}

    detail, summary = compare_selected_features(training, live, selected)

    status = detail.set_index("feature")["status"].to_dict()
    assert status == {"drifted": "drift", "stable": "match"}
    assert summary["drifted_features"] == ["drifted"]
    assert summary["affected_props"] == ["pts"]


def test_parity_audit_separates_external_live_features_from_missing_features():
    training = _frame(3.0).assign(opp_def_rtg_pt=[110.0, 111.0])
    live = _frame(3.0)
    selected = {"stl": ["stable", "opp_def_rtg_pt", "not_built"]}

    detail, summary = compare_selected_features(training, live, selected)

    status = detail.set_index("feature")["status"].to_dict()
    assert status["opp_def_rtg_pt"] == "external_live_attachment"
    assert status["not_built"] == "missing"
    assert summary["external_live_features"] == ["opp_def_rtg_pt"]
    assert summary["missing_features"] == ["not_built"]
