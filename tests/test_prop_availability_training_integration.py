import unittest

import pandas as pd

from scripts.build_prop_availability_training_data import (
    AVAILABILITY_MINUTES_PROP_FEATURES,
    build_training_table,
)
from train_advanced_models import (
    AdvancedPropModel,
    _validate_availability_minutes_training_features,
)


class PropAvailabilityTrainingIntegrationTests(unittest.TestCase):
    def _prop(self):
        return pd.DataFrame(
            {
                "GAME_ID": [1, 2, 3],
                "PLAYER_ID": [10, 10, 10],
                "team": ["AAA", "AAA", "AAA"],
                "game_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "mp": [20.0, 25.0, 0.0],
                "mp_l10": [18.0, 20.0, 20.0],
                "pts": [8, 12, 0],
                "trb": [4, 5, 0],
                "ast": [2, 3, 0],
            }
        )

    def _oof(self):
        return pd.DataFrame(
            {
                "game_id": [2, 3],
                "player_id": [10, 10],
                "team": ["AAA", "AAA"],
                "game_date": ["2024-01-02", "2024-01-03"],
                "available": [1, 0],
                "minutes": [25.0, 0.0],
                "availability_probability": [0.8, 0.2],
                "conditional_minutes_prediction": [24.0, 20.0],
                "expected_minutes_prediction": [19.2, 4.0],
                "prediction_split": ["fold_1", "fold_1"],
                "model_train_end": ["2024-01-01", "2024-01-01"],
            }
        )

    def test_join_is_one_to_one_chronological_and_appearance_conditional(self):
        output, summary = build_training_table(self._prop(), self._oof(), min_window_coverage=1.0)

        self.assertEqual(len(output), 1)
        self.assertEqual(int(output.iloc[0]["GAME_ID"]), 2)
        self.assertEqual(summary["excluded_before_oof_window"], 1)
        self.assertEqual(summary["excluded_nonappearances"], 1)
        self.assertEqual(summary["window_match_rate"], 1.0)
        self.assertEqual(summary["oof_chronology_violations"], 0)
        self.assertFalse(output[AVAILABILITY_MINUTES_PROP_FEATURES].isna().any().any())

    def test_duplicate_oof_key_is_rejected(self):
        oof = pd.concat([self._oof(), self._oof().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate candidate keys"):
            build_training_table(self._prop(), oof)

    def test_oof_chronology_violation_is_rejected(self):
        oof = self._oof()
        oof.loc[0, "model_train_end"] = "2024-01-02"
        with self.assertRaisesRegex(ValueError, "chronology violations"):
            build_training_table(self._prop(), oof)

    def test_features_are_registered_and_contract_validates(self):
        output, _ = build_training_table(self._prop(), self._oof(), min_window_coverage=1.0)
        trainer = AdvancedPropModel()
        available = trainer.get_feature_columns(output)
        self.assertTrue(set(AVAILABILITY_MINUTES_PROP_FEATURES).issubset(available))

        contract = _validate_availability_minutes_training_features(output, trainer=trainer)
        self.assertEqual(contract["chronology_violations"], 0)
        self.assertEqual(contract["rows"], 1)

    def test_required_contract_fails_when_features_are_absent(self):
        with self.assertRaisesRegex(ValueError, "merged OOF artifact"):
            _validate_availability_minutes_training_features(self._prop())


if __name__ == "__main__":
    unittest.main()
