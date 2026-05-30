import json
import tempfile
import unittest
from pathlib import Path

from sparkmind_inference.policy import pi0, pi05


def _write_preprocessor(path: Path, features: dict) -> None:
    payload = {
        "steps": [
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "features": features,
                    "norm_map": {
                        "VISUAL": "IDENTITY",
                        "STATE": "MEAN_STD",
                        "ACTION": "MEAN_STD",
                    },
                },
            }
        ]
    }
    (path / "policy_preprocessor.json").write_text(json.dumps(payload), encoding="utf-8")


class PIProcessorNormalizationTest(unittest.TestCase):
    def test_empty_processor_features_are_noop_for_pi0(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            _write_preprocessor(checkpoint, features={})
            config = {}

            pi0._apply_processor_normalization_mapping(checkpoint, config)

            self.assertNotIn("normalization_mapping", config)

    def test_empty_processor_features_are_noop_for_pi05(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            _write_preprocessor(checkpoint, features={})
            config = {}

            pi05._apply_processor_normalization_mapping(checkpoint, config)

            self.assertNotIn("normalization_mapping", config)

    def test_nonempty_processor_features_enable_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            _write_preprocessor(checkpoint, features={"observation.state": {"shape": [8]}})
            config = {}

            pi0._apply_processor_normalization_mapping(checkpoint, config)

            self.assertEqual(config["normalization_mapping"]["STATE"], "MEAN_STD")


if __name__ == "__main__":
    unittest.main()
