"""A copied project must never default back to the original workspace."""
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import feature_pipeline
from scripts import run_training_maintenance as maintenance


def test_feature_cli_defaults_follow_script_location():
    frame = pd.DataFrame({"games_played": [10]})
    pipeline = Mock()
    pipeline.fit_transform.return_value = frame
    pipeline.get_feature_columns.return_value = []
    root = Path(feature_pipeline.__file__).resolve().parent
    with patch("sys.argv", ["feature_pipeline.py"]), \
         patch.object(feature_pipeline, "FeaturePipeline", return_value=pipeline), \
         patch.object(feature_pipeline.pd, "read_csv", return_value=frame) as read, \
         patch.object(pd.DataFrame, "to_csv") as write, patch("builtins.print"):
        feature_pipeline.main()
    read.assert_called_once_with(str(root / "data" / "nba_data.csv"))
    assert write.call_args.args[0] == str(root / "data" / "model_data_advanced.csv")
    pipeline.save_feature_list.assert_called_once_with(str(root / "feature_cols_advanced.json"))


def test_maintenance_passes_explicit_workspace_paths():
    profile = {"date_max": "2026-11-09"}
    with patch.object(maintenance, "run_step", return_value={}) as run, \
         patch.object(maintenance, "_date_profile", return_value=profile):
        maintenance.rebuild_training_tables()
    args = run.call_args_list[0].args[1]
    assert args[args.index("--input") + 1] == str(maintenance.DATA / "nba_data.csv")
    assert args[args.index("--output") + 1] == str(maintenance.DATA / "model_data_advanced.csv")
    assert args[args.index("--features-json") + 1] == str(maintenance.ROOT / "feature_cols_advanced.json")
