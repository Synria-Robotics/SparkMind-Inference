import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sparkmind_inference.api import InferenceSDK
from sparkmind_inference.base import BaseInferenceEngine
from sparkmind_inference.exceptions import InferenceRuntimeError
from sparkmind_inference.robot_io import (
    load_robot_io_from_checkpoint,
    resolve_pretrained_bundle_dir,
)


ROBOT_IO = {
    "robot": "alicia_d",
    "arm_mode": "single_arm",
    "state_type": "joint_position",
    "state_joint_unit": "rad",
    "action_type": "absolute_joint_position",
    "action_joint_unit": "rad",
    "gripper_range": [0, 1000],
    "cameras": ["head", "wrist"],
}


def _make_pretrained_bundle(root: Path) -> Path:
    pretrained = root / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "config.json").write_text("{}\n", encoding="utf-8")
    (pretrained / "model.safetensors").write_bytes(b"")
    return pretrained


class DummyEngine(BaseInferenceEngine):
    def __init__(self):
        super().__init__()
        self.model_type = "dummy"
        self.required_cameras = ["head"]
        self.state_dim = 8
        self.action_dim = 7
        self.chunk_size = 50
        self.n_action_steps = 50

    def load(self, checkpoint_dir):
        return True, ""

    def unload(self):
        self.is_loaded = False

    def _predict_chunk(self, images, state):
        return np.zeros((self.n_action_steps, self.action_dim), dtype=np.float32)


class FailingEngine(DummyEngine):
    def __init__(self):
        super().__init__()
        self.model_type = "act"
        self.is_loaded = True
        self.state_dim = 8

    def _predict_chunk(self, images, state):
        raise RuntimeError("boom")

    def step(self, images, state):
        raise RuntimeError("step boom")


class RobotIOMetadataTest(unittest.TestCase):
    def test_load_robot_io_from_pretrained_and_step_checkpoint_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            pretrained = _make_pretrained_bundle(checkpoint)
            (pretrained / "robot_io.json").write_text(json.dumps(ROBOT_IO), encoding="utf-8")

            self.assertEqual(resolve_pretrained_bundle_dir(checkpoint), pretrained)
            self.assertEqual(load_robot_io_from_checkpoint(pretrained), ROBOT_IO)
            self.assertEqual(load_robot_io_from_checkpoint(checkpoint), ROBOT_IO)

    def test_missing_robot_io_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            pretrained = _make_pretrained_bundle(Path(tmp))

            self.assertIsNone(load_robot_io_from_checkpoint(pretrained))

    def test_invalid_robot_io_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            pretrained = _make_pretrained_bundle(Path(tmp))
            (pretrained / "robot_io.json").write_text('{"arm_mode": "triple_arm"}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "arm_mode"):
                load_robot_io_from_checkpoint(pretrained)

    def test_policy_metadata_exposes_robot_io(self):
        engine = DummyEngine()
        engine.robot_io = dict(ROBOT_IO)

        metadata = InferenceSDK._metadata_for(engine, "/tmp/pretrained_model")

        self.assertEqual(metadata.robot_io, ROBOT_IO)
        self.assertEqual(metadata.required_cameras, ("head",))

    def test_prediction_errors_keep_sdk_context(self):
        sdk = InferenceSDK()
        sdk._policies["act"] = FailingEngine()
        sdk._checkpoint_dirs["act"] = "/tmp/pretrained_model"
        images = {"head": np.zeros((8, 8, 3), dtype=np.uint8)}
        state = np.zeros(8, dtype=np.float32)

        with self.assertRaisesRegex(InferenceRuntimeError, "Failed to predict act action chunk"):
            sdk.predict_action_chunk("act", images=images, state=state)

        with self.assertRaisesRegex(InferenceRuntimeError, "Failed to predict act action"):
            sdk.predict_action("act", images=images, state=state)


if __name__ == "__main__":
    unittest.main()
