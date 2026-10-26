import numpy as np
import pandas as pd

from train_advanced_models import _get_sample_weights


def test_sample_weights_roll_forward_with_the_latest_observed_season():
    frame = pd.DataFrame(
        {
            "game_date": [
                "2024-04-01",
                "2025-04-01",
                "2026-04-01",
                "2026-10-24",
                "2027-04-01",
            ]
        }
    )
    np.testing.assert_array_equal(
        _get_sample_weights(frame),
        np.array([1.0, 1.0, 2.0, 3.0, 3.0]),
    )


def test_sample_weights_do_not_depend_on_a_hard_coded_season_label():
    frame = pd.DataFrame({"game_date": ["2030-11-01", "2031-03-01"]})
    np.testing.assert_array_equal(_get_sample_weights(frame), np.array([3.0, 3.0]))
