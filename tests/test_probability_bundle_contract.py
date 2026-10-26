import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nba_props import CONFIG, _load_probability_calibrators, _sha256_file


class ProbabilityBundleContractTests(unittest.TestCase):
    def _write_bundle(self, model_dir: Path, *, matching: bool) -> dict:
        sidecars = {"ast": {"target": "ast"}, "blk": {"target": "blk"}}
        hashes = {}
        for target, payload in sidecars.items():
            path = model_dir / f"prob_sidecar_{target}_advanced.pkl"
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            hashes[target] = _sha256_file(path)
        if not matching:
            hashes["ast"] = "0" * 64
        with (model_dir / "probability_calibrator_advanced.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "calibrators": {"ast": object(), "blk": object()},
                    "sidecar_contract": {"sidecar_sha256": hashes},
                },
                handle,
            )
        return sidecars

    def test_matching_bundle_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            sidecars = self._write_bundle(model_dir, matching=True)
            with patch.dict(CONFIG, {"models_dir": model_dir}):
                loaded = _load_probability_calibrators(sidecars)
            self.assertEqual(set(loaded), {"ast", "blk"})

    def test_mismatched_bundle_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            sidecars = self._write_bundle(model_dir, matching=False)
            with patch.dict(CONFIG, {"models_dir": model_dir}):
                with self.assertRaisesRegex(RuntimeError, "bundle mismatch"):
                    _load_probability_calibrators(sidecars)

    def test_missing_calibrator_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(CONFIG, {"models_dir": Path(temporary)}):
                with self.assertRaises(FileNotFoundError):
                    _load_probability_calibrators({"ast": {"target": "ast"}})


if __name__ == "__main__":
    unittest.main()
