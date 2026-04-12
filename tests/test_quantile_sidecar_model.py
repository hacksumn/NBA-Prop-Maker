import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from train_advanced_models import _make_quantile_sidecar_model


class QuantileSidecarModelTests(unittest.TestCase):
    def test_make_quantile_sidecar_uses_hist_gradient_boosting(self):
        model = _make_quantile_sidecar_model(
            "pts",
            0.5,
            {
                "max_depth": 7,
                "n_estimators": 400,
                "min_child_weight": 3,
                "learning_rate": 0.04,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
        )

        self.assertIsInstance(model, HistGradientBoostingRegressor)
        self.assertEqual(model.loss, "quantile")
        self.assertAlmostEqual(model.quantile, 0.5)
        self.assertFalse(model.early_stopping)

    def test_quantile_sidecar_model_fits_with_sample_weights(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(96, 6)), columns=[f"f{i}" for i in range(6)])
        y = pd.Series((X["f0"] * 0.8) + (X["f1"] * -0.3) + rng.normal(scale=0.2, size=len(X)))
        sample_weight = np.linspace(1.0, 2.0, len(X))

        model = _make_quantile_sidecar_model(
            "pts",
            0.75,
            {
                "max_depth": 7,
                "n_estimators": 400,
                "min_child_weight": 3,
                "learning_rate": 0.04,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
        )
        model.fit(X, y, sample_weight=sample_weight)
        preds = model.predict(X.iloc[:8])

        self.assertEqual(preds.shape, (8,))
        self.assertTrue(np.isfinite(preds).all())


if __name__ == "__main__":
    unittest.main()
