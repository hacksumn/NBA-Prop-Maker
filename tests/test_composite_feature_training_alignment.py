"""Regression test for the {stat}_best_estimate / edge_vs_avg training-live mismatch.

feature_pipeline.py builds the features the production models were trained on;
nba_props.py independently rebuilds a live-inference version of the same
columns. They drifted apart (different weights, and edge_vs_avg keyed off a
different base signal) — see DECISIONS.md. nba_props.py's
_align_composite_features_to_training() is a deliberate, hand-duplicated port
of feature_pipeline.py's formulas, run after all five real ingredients exist.
This test locks the two implementations together: if either side changes
without the other, it fails.
"""

import unittest

import numpy as np
import pandas as pd

from feature_pipeline import FeaturePipeline
from nba_props import _align_composite_features_to_training

OUTPUT_COLS = ["best_estimate", "synthetic_line", "edge_vs_avg", "vegas_slow", "pct_diff"]


def _sample_frame() -> pd.DataFrame:
    """Minimal frame with every ingredient _create_composite_features/_create_vegas_features
    and _align_composite_features_to_training read, for pts/trb/ast."""
    n = 6
    rng = np.random.default_rng(7)
    data = {}
    for stat, scale in [("pts", 25.0), ("trb", 8.0), ("ast", 6.0)]:
        data[f"{stat}_recency_anchor"] = rng.uniform(0.5, 1.0, n) * scale
        data[f"{stat}_defense_adj"] = rng.uniform(0.5, 1.0, n) * scale
        data[f"{stat}_situational_avg"] = rng.uniform(0.5, 1.0, n) * scale
        data[f"{stat}_season_avg"] = rng.uniform(0.5, 1.0, n) * scale
        data[f"{stat}_l20"] = rng.uniform(0.5, 1.0, n) * scale
        data[f"{stat}_l10"] = rng.uniform(0.5, 1.0, n) * scale
        data[f"{stat}_l5"] = rng.uniform(0.5, 1.0, n) * scale
        data[f"{stat}_ewma5"] = rng.uniform(0.5, 1.0, n) * scale
    return pd.DataFrame(data)


class CompositeFeatureTrainingAlignmentTests(unittest.TestCase):
    def test_matches_training_formula_with_all_ingredients_present(self):
        base = _sample_frame()

        pipeline = FeaturePipeline()
        training = pipeline._create_vegas_features(base.copy())
        training = pipeline._create_composite_features(training)

        live = _align_composite_features_to_training(base.copy())

        for stat in ["pts", "trb", "ast"]:
            for col in OUTPUT_COLS:
                name = f"{stat}_{col}"
                self.assertIn(name, training.columns, f"training missing {name}")
                self.assertIn(name, live.columns, f"live missing {name}")
                pd.testing.assert_series_equal(
                    training[name], live[name], check_exact=False, rtol=1e-9, atol=1e-9,
                    obj=name,
                )

    def test_matches_training_formula_when_defense_and_situational_missing(self):
        """Falls back correctly (recency_anchor/season_avg/l20 only) when a
        player is missing opponent-context features — e.g. early season."""
        base = _sample_frame().drop(
            columns=[c for c in _sample_frame().columns if c.endswith("_defense_adj") or c.endswith("_situational_avg")]
        )

        pipeline = FeaturePipeline()
        training = pipeline._create_vegas_features(base.copy())
        training = pipeline._create_composite_features(training)

        live = _align_composite_features_to_training(base.copy())

        for stat in ["pts", "trb", "ast"]:
            for col in OUTPUT_COLS:
                name = f"{stat}_{col}"
                pd.testing.assert_series_equal(
                    training[name], live[name], check_exact=False, rtol=1e-9, atol=1e-9,
                    obj=name,
                )


if __name__ == "__main__":
    unittest.main()
