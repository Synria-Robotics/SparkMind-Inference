import time
import unittest

import numpy as np

from sparkmind_inference.base import ACTTemporalEnsembler, BaseInferenceEngine, SmoothingConfig


class FakeEngine(BaseInferenceEngine):
    def __init__(self, smoothing_config=None):
        super().__init__(smoothing_config)
        self.model_type = "fake"
        self.chunk_size = 5
        self.n_action_steps = 2
        self.action_dim = 2
        self.is_loaded = True
        self.calls = 0
        self._init_components()
        self.reset()

    def load(self, checkpoint_dir):
        return True, ""

    def unload(self):
        self.is_loaded = False

    def _predict_chunk(self, images, state):
        base = self.calls * 100
        self.calls += 1
        values = np.arange(base, base + self.chunk_size * self.action_dim, dtype=np.float32)
        return values.reshape(self.chunk_size, self.action_dim)


class RefTemporalEnsembler:
    def __init__(self, coeff, chunk_size):
        self.chunk_size = chunk_size
        self.weights = np.exp(-coeff * np.arange(chunk_size, dtype=np.float32))
        self.weights_cumsum = np.cumsum(self.weights)
        self.actions = None
        self.count = None

    def update(self, actions):
        actions = np.asarray(actions, dtype=np.float32)[None]
        if self.actions is None:
            self.actions = actions.copy()
            self.count = np.ones((self.chunk_size, 1), dtype=np.int64)
        else:
            self.actions *= self.weights_cumsum[self.count - 1]
            self.actions += actions[:, :-1] * self.weights[self.count]
            self.actions /= self.weights_cumsum[self.count]
            self.count = np.clip(self.count + 1, None, self.chunk_size)
            self.actions = np.concatenate([self.actions, actions[:, -1:]], axis=1)
            self.count = np.concatenate([self.count, np.ones_like(self.count[-1:])], axis=0)

        action = self.actions[:, 0].copy()[0]
        self.actions = self.actions[:, 1:]
        self.count = self.count[1:]
        return action


class LeRobotStepAlignmentTest(unittest.TestCase):
    def test_step_uses_fifo_and_does_not_skip_after_sleep(self):
        engine = FakeEngine(SmoothingConfig(control_fps=1000.0))

        first = engine.step({}, np.zeros(2, dtype=np.float32))
        time.sleep(0.02)
        second = engine.step({}, np.zeros(2, dtype=np.float32))
        third = engine.step({}, np.zeros(2, dtype=np.float32))

        np.testing.assert_array_equal(first, np.array([0, 1], dtype=np.float32))
        np.testing.assert_array_equal(second, np.array([2, 3], dtype=np.float32))
        np.testing.assert_array_equal(third, np.array([100, 101], dtype=np.float32))
        self.assertEqual(engine.calls, 2)
        self.assertEqual(engine.get_fallback_count(), 0)

    def test_reset_clears_fifo(self):
        engine = FakeEngine()

        np.testing.assert_array_equal(engine.step({}, np.zeros(2, dtype=np.float32)), [0, 1])
        engine.reset()
        np.testing.assert_array_equal(engine.step({}, np.zeros(2, dtype=np.float32)), [100, 101])

    def test_step_rejects_rtc(self):
        engine = FakeEngine(SmoothingConfig(enable_rtc=True))

        with self.assertRaisesRegex(RuntimeError, "RTC is not supported"):
            engine.step({}, np.zeros(2, dtype=np.float32))

    def test_temporal_ensembler_matches_lerobot_update(self):
        rng = np.random.default_rng(0)
        sdk = ACTTemporalEnsembler(temporal_ensemble_coeff=0.01, chunk_size=5)
        ref = RefTemporalEnsembler(coeff=0.01, chunk_size=5)

        for _ in range(20):
            chunk = rng.normal(size=(5, 3)).astype(np.float32)
            np.testing.assert_allclose(sdk.update(chunk), ref.update(chunk), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
