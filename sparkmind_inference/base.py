"""
Base inference primitives with LeRobot-style action queues.

Key Features (LeRobot Pattern):
- FIFO action queue for select_action()/step()
- Optional direct action chunk prediction
- Optional ACT temporal ensembling
"""

import copy
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TraceEvent:
    timestamp: float
    source: str
    event: str
    details: Dict[str, Any] = field(default_factory=dict)

class TraceRecorder:
    """
    Simple recorder for tracing inference events.

    Memory-safe: Limits event history to prevent unbounded growth
    during long-running inference sessions.
    """
    def __init__(self, max_events: int = 1000):
        """
        Args:
            max_events: Maximum number of events to keep (default: 1000)
                       Older events are automatically discarded.
        """
        self.events: List[TraceEvent] = []
        self._start_time = time.monotonic()
        self._lock = threading.Lock()
        self._max_events = max(1, int(max_events))

    def record(self, source: str, event: str, **details):
        with self._lock:
            self.events.append(TraceEvent(
                timestamp=time.monotonic() - self._start_time,
                source=source,
                event=event,
                details=details
            ))

            # Limit event history to prevent memory leak
            if len(self.events) > self._max_events:
                # Remove oldest 10% when limit exceeded
                remove_count = max(1, self._max_events // 10)
                self.events = self.events[remove_count:]

    def clear(self):
        with self._lock:
            self.events = []
            self._start_time = time.monotonic()

# ==================== Gripper Smoothing ====================
# ==================== Aggregate Functions ====================
# Following LeRobot pattern: configs.py

AGGREGATE_FUNCTIONS = {
    "latest_only": lambda old, new: new,
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


# ==================== Data Structures (LeRobot Pattern) ====================

@dataclass
class TimedAction:
    """Action with an execution index for the internal FIFO queue."""
    timestamp: float  # Legacy debug field; default step path does not use wall-clock scheduling
    timestep: int     # Sequential step index
    action: np.ndarray
    ensemble_count: int = 1
    
    def get_timestamp(self) -> float:
        return self.timestamp
    
    def get_timestep(self) -> int:
        return self.timestep
    
    def get_action(self) -> np.ndarray:
        return self.action


@dataclass
class SmoothingConfig:
    """Configuration for action queueing and smoothing."""
    # Control frequency
    control_fps: float = 30.0
    
    # Gripper velocity clamping (in raw action space, [0, 1000])
    gripper_max_velocity: float = 200.0
    enable_gripper_clamping: bool = False
    
    # Legacy timestamp-queue knobs. The default step path follows LeRobot v0.5.1
    # FIFO select_action() semantics and does not use these values.
    chunk_size_threshold: float = 0.5
    action_chunk_size: Optional[int] = None
    n_action_steps: Optional[int] = None
    
    # Latency estimation
    latency_ema_alpha: float = 0.2  # EMA smoothing factor
    latency_safety_margin: float = 1.5  # Multiply latency estimate by this for safety
    
    # Aggregate function for overlapping chunks
    aggregate_fn_name: str = "latest_only"  # "latest_only", "weighted_average", etc.
    
    # Fallback when queue empty
    fallback_mode: str = "repeat"  # "repeat", "hold"

    # ACT temporal ensembling. Matches LeRobot's ACT temporal ensemble
    # behavior when enabled by ACTInferenceEngine.select_action().
    enable_temporal_ensemble: bool = False
    temporal_ensemble_coeff: float = 0.01

    # Real-Time Chunking (RTC) options for VLA policies that support it
    # (SmolVLA, PI0 and PI0.5). Disabled by default.
    enable_rtc: bool = False
    rtc_prefix_attention_schedule: str = "LINEAR"
    rtc_max_guidance_weight: float = 10.0
    rtc_execution_horizon: int = 10
    rtc_inference_delay_steps: int = 0
    rtc_debug: bool = False
    rtc_debug_maxlen: int = 100
    
    @property
    def environment_dt(self) -> float:
        """Time step in seconds."""
        return 1.0 / self.control_fps


# ==================== Latency Estimator ====================

class LatencyEstimator:
    """Estimates inference latency using exponential moving average."""
    
    def __init__(self, alpha: float = 0.2, initial_value: float = 0.1):
        self.alpha = alpha
        self.value = initial_value
        self._initialized = False
        self._lock = threading.Lock()
    
    def update(self, latency: float):
        """Update estimate with new measurement."""
        with self._lock:
            if not self._initialized:
                self.value = latency
                self._initialized = True
            else:
                self.value = self.alpha * latency + (1 - self.alpha) * self.value
    
    def get_value(self) -> float:
        """Get current estimate."""
        with self._lock:
            return self.value


# ==================== Action Queue Manager (LeRobot Pattern) ====================

class TimestampedActionQueue:
    """
    Legacy indexed action queue.

    The default step/select_action path uses get_next_action(), which behaves
    like LeRobot v0.5.1 FIFO select_action(). Timestamp-based retrieval remains
    only for internal/backward-compatible debugging helpers.

    Key differences from simple deque:
    1. Actions are indexed by timestep, not FIFO order
    2. Supports aggregation when new chunk overlaps with existing actions
    3. Time-based action retrieval (skip expired actions)
    4. Thread-safe operations

    Performance optimization: maintains a sorted timestep list using bisect
    to avoid O(n log n) sorting on every get_action_for_time() call.
    """

    def __init__(self, config: SmoothingConfig):
        self.config = config
        self._queue: Dict[int, TimedAction] = {}  # timestep -> TimedAction
        self._sorted_timesteps: List[int] = []  # Sorted list of timesteps for O(log n) lookup
        self._lock = threading.Lock()
        self._latest_executed_timestep: int = -1
        self._chunk_size: int = 1
        self._aggregate_fn = get_aggregate_function(config.aggregate_fn_name)
    
    def reset(self):
        """Reset queue state for new episode."""
        with self._lock:
            self._queue.clear()
            self._sorted_timesteps.clear()
            self._latest_executed_timestep = -1
    
    def set_chunk_size(self, size: int):
        """Set expected chunk size (for threshold calculation)."""
        self._chunk_size = max(1, size)
    
    def get_queue_size(self) -> int:
        """Get number of actions in queue."""
        with self._lock:
            return len(self._queue)
    
    def get_snapshot(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return a compact snapshot of queued actions for debugging."""
        with self._lock:
            entries = []
            for timestep in self._sorted_timesteps[: max(0, limit)]:
                timed_action = self._queue[timestep]
                entries.append(
                    {
                        "timestep": int(timestep),
                        "timestamp": float(timed_action.timestamp),
                        "action_shape": tuple(timed_action.action.shape),
                        "ensemble_count": int(timed_action.ensemble_count),
                    }
                )
            return entries
    
    def add_action_chunk(self, timed_actions: List[TimedAction]):
        """
        Add new action chunk with aggregation for overlapping timesteps.

        LeRobot pattern (from robot_client.py _aggregate_action_queues):
        - Skip actions older than latest executed
        - Aggregate overlapping timesteps
        - Add new timesteps directly
        """
        import bisect

        with self._lock:
            for new_action in timed_actions:
                timestep = new_action.get_timestep()

                # Skip actions older than what we've already executed
                if timestep <= self._latest_executed_timestep:
                    continue

                # Check if this timestep already exists
                if timestep in self._queue:
                    old_action = self._queue[timestep].get_action()
                    if self.config.enable_temporal_ensemble:
                        aggregated, ensemble_count = self._aggregate_temporal_ensemble(
                            self._queue[timestep],
                            new_action,
                        )
                    else:
                        aggregated = self._aggregate_fn(old_action, new_action.get_action())
                        ensemble_count = 1
                    self._queue[timestep] = TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=timestep,
                        action=aggregated,
                        ensemble_count=ensemble_count,
                    )
                else:
                    # Add new action directly and maintain sorted timesteps
                    self._queue[timestep] = new_action
                    bisect.insort(self._sorted_timesteps, timestep)

            logger.debug(f"Queue updated: {len(self._queue)} actions, "
                        f"latest_executed={self._latest_executed_timestep}")

    def _aggregate_temporal_ensemble(
        self,
        old_action: TimedAction,
        new_action: TimedAction,
    ) -> Tuple[np.ndarray, int]:
        """ACT-style temporal ensemble for two predictions of the same timestep."""
        chunk_size = max(1, int(self._chunk_size))
        if chunk_size <= 1:
            return new_action.get_action().astype(np.float32, copy=False), 1

        # Match LeRobot ACTTemporalEnsembler: w_i = exp(-k * i), where w_0
        # is the oldest prediction and later predictions get later weights.
        old_count = max(1, int(old_action.ensemble_count))
        effective_count = min(old_count, chunk_size - 1)
        weights = np.exp(-float(self.config.temporal_ensemble_coeff) * np.arange(chunk_size, dtype=np.float32))
        old_weight_sum = float(np.sum(weights[:effective_count]))
        new_weight = float(weights[effective_count])
        denom = old_weight_sum + new_weight
        if denom <= 0.0:
            return new_action.get_action().astype(np.float32, copy=False), min(effective_count + 1, chunk_size)

        aggregated = (
            old_action.get_action().astype(np.float32, copy=False) * old_weight_sum
            + new_action.get_action().astype(np.float32, copy=False) * new_weight
        ) / denom
        return aggregated.astype(np.float32, copy=False), min(effective_count + 1, chunk_size)
    
    def get_action_for_time(self, current_time: float, t0: float) -> Optional[TimedAction]:
        """
        Get action for current time using timestamp alignment.

        LeRobot pattern: calculate which timestep we SHOULD be at,
        then get that action (or nearest future one).

        Args:
            current_time: Current wall clock time
            t0: Episode start time

        Returns:
            TimedAction for current timestep, or None if queue empty
        """
        import bisect

        with self._lock:
            if not self._queue:
                return None

            # Calculate expected timestep based on elapsed time
            elapsed = current_time - t0
            expected_timestep = int(elapsed / self.config.environment_dt)

            # Find the action to execute using binary search on sorted timesteps:
            # 1. If expected_timestep exists, use it
            # 2. If not, use the smallest timestep > latest_executed
            # 3. Skip any timesteps < expected_timestep (they're expired)

            # Use binary search to find first timestep > latest_executed
            idx = bisect.bisect_right(self._sorted_timesteps, self._latest_executed_timestep)

            if idx >= len(self._sorted_timesteps):
                return None

            # Get valid timesteps (those after latest_executed)
            valid_timesteps = self._sorted_timesteps[idx:]

            if not valid_timesteps:
                return None

            # Try to find expected_timestep or the next available one
            target_timestep = None
            for ts in valid_timesteps:
                if ts >= expected_timestep:
                    target_timestep = ts
                    break

            # If all valid timesteps are before expected, they are expired.
            if target_timestep is None:
                for ts in valid_timesteps:
                    del self._queue[ts]
                    self._sorted_timesteps.remove(ts)
                    logger.debug(f"Discarded expired action timestep {ts}")
                self._latest_executed_timestep = max(
                    self._latest_executed_timestep,
                    expected_timestep - 1,
                )
                return None

            # Get action and update state
            action = self._queue.pop(target_timestep)
            self._sorted_timesteps.remove(target_timestep)
            self._latest_executed_timestep = target_timestep

            # Clean up any expired actions we skipped
            expired = [ts for ts in list(self._sorted_timesteps) if ts < target_timestep]
            for ts in expired:
                del self._queue[ts]
                self._sorted_timesteps.remove(ts)
                logger.debug(f"Discarded expired action timestep {ts}")

            return action
    
    def get_next_action(self) -> Optional[TimedAction]:
        """
        Simple FIFO-style get (fallback when not using timestamp alignment).
        Gets the action with smallest timestep > latest_executed.
        """
        import bisect

        with self._lock:
            if not self._queue:
                return None

            # Use binary search to find first timestep > latest_executed
            idx = bisect.bisect_right(self._sorted_timesteps, self._latest_executed_timestep)

            if idx >= len(self._sorted_timesteps):
                return None

            target_timestep = self._sorted_timesteps[idx]
            action = self._queue.pop(target_timestep)
            self._sorted_timesteps.remove(target_timestep)
            self._latest_executed_timestep = target_timestep

            return action


# ==================== Gripper Smoother ====================

class GripperSmoother:
    """Velocity clamping for gripper to prevent jerky movements."""
    
    def __init__(self, config: SmoothingConfig, action_dim: int = 7):
        self.config = config
        self.action_dim = action_dim
        self._last_action: Optional[np.ndarray] = None
    
    def reset(self):
        """Reset state for new episode."""
        self._last_action = None
    
    def smooth(self, action: np.ndarray) -> np.ndarray:
        """Apply velocity clamping to gripper."""
        if not self.config.enable_gripper_clamping or self._last_action is None:
            self._last_action = action.copy()
            return action
        
        result = action.copy()
        
        # Clamp gripper (last dimension)
        if len(action) >= 7:
            gripper_idx = -1
            delta = action[gripper_idx] - self._last_action[gripper_idx]
            clamped_delta = np.clip(
                delta, 
                -self.config.gripper_max_velocity,
                self.config.gripper_max_velocity
            )
            result[gripper_idx] = self._last_action[gripper_idx] + clamped_delta
        
        self._last_action = result.copy()
        return result
    
    def get_last_action(self) -> Optional[np.ndarray]:
        """Get last action (for fallback)."""
        return self._last_action.copy() if self._last_action is not None else None


class ACTTemporalEnsembler:
    """LeRobot-style online temporal ensemble for ACT action chunks."""

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int):
        self.chunk_size = max(1, int(chunk_size))
        steps = np.arange(self.chunk_size, dtype=np.float32)
        self.ensemble_weights = np.exp(-float(temporal_ensemble_coeff) * steps).astype(np.float32)
        self.ensemble_weights_cumsum = np.cumsum(self.ensemble_weights).astype(np.float32)
        self.reset()

    def reset(self):
        self.ensembled_actions: Optional[np.ndarray] = None
        self.ensembled_actions_count: Optional[np.ndarray] = None

    def update(self, actions: np.ndarray) -> np.ndarray:
        """Update the ensemble with a new chunk and return the current action."""
        action_chunk = np.asarray(actions, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"actions must have shape (chunk_size, action_dim), got {action_chunk.shape}")
        if action_chunk.shape[0] == 0:
            raise ValueError("actions chunk must not be empty")

        if self.ensembled_actions is None or self.ensembled_actions_count is None:
            self.ensembled_actions = action_chunk.copy()
            self.ensembled_actions_count = np.ones((action_chunk.shape[0], 1), dtype=np.int64)
        else:
            overlap = min(self.ensembled_actions.shape[0], max(0, action_chunk.shape[0] - 1))
            if overlap > 0:
                counts = np.clip(self.ensembled_actions_count[:overlap], 1, self.chunk_size - 1)
                self.ensembled_actions[:overlap] *= self.ensemble_weights_cumsum[counts - 1]
                self.ensembled_actions[:overlap] += action_chunk[:overlap] * self.ensemble_weights[counts]
                self.ensembled_actions[:overlap] /= self.ensemble_weights_cumsum[counts]
                self.ensembled_actions_count[:overlap] = np.clip(counts + 1, 1, self.chunk_size)

            self.ensembled_actions = np.concatenate(
                [self.ensembled_actions[:overlap], action_chunk[overlap:].copy()],
                axis=0,
            )
            tail_count = np.ones((action_chunk.shape[0] - overlap, 1), dtype=np.int64)
            self.ensembled_actions_count = np.concatenate(
                [self.ensembled_actions_count[:overlap], tail_count],
                axis=0,
            )

        action = self.ensembled_actions[0].copy()
        self.ensembled_actions = self.ensembled_actions[1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return action


# ==================== Base Inference Engine ====================

class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference policies with LeRobot-style queue support.
    
    Key Features:
    - LeRobot v0.5.1 FIFO action queue
    - Direct action chunk prediction
    - Optional gripper velocity clamping for deployment adapters
    """
    
    def __init__(self, smoothing_config: Optional[SmoothingConfig] = None):
        self.is_loaded = False
        self.model_type: str = ""
        self.required_cameras: List[str] = []
        self.state_dim: int = 0
        self.action_dim: int = 7
        self.chunk_size: int = 1
        self.n_action_steps: int = 1
        self.requested_device: Optional[str] = None
        self.actual_device: Optional[str] = None
        self.device_warning: str = ""
        self.robot_io: Optional[Dict[str, Any]] = None
        
        # Config
        self.smoothing_config = smoothing_config or SmoothingConfig()
        
        # Components (initialized after model load)
        self._action_queue: Optional[TimestampedActionQueue] = None
        self._latency_estimator: Optional[LatencyEstimator] = None
        self._gripper_smoother: Optional[GripperSmoother] = None
        self._temporal_ensembler: Optional[ACTTemporalEnsembler] = None
        self._trace_recorder: Optional[TraceRecorder] = None
        
        # Episode state
        self._episode_start_time: float = 0.0
        self._current_timestep: int = 0
        self._fallback_count: int = 0

    def _load_robot_io_metadata(self, checkpoint_path: str | Path) -> None:
        """Load optional bundled robot I/O metadata for SDK consumers."""
        from .robot_io import load_robot_io_from_checkpoint

        self.robot_io = load_robot_io_from_checkpoint(checkpoint_path)

    def _apply_action_chunk_overrides(self, config_dict: Dict[str, Any]) -> None:
        """Apply optional user overrides for model/action chunk scheduling."""
        action_chunk_size = self.smoothing_config.action_chunk_size
        n_action_steps = self.smoothing_config.n_action_steps

        if action_chunk_size is not None:
            config_dict["chunk_size"] = int(action_chunk_size)

        chunk_size = int(config_dict.get("chunk_size", self.chunk_size or 1))
        if chunk_size < 1:
            raise ValueError("action_chunk_size/chunk_size must be >= 1")

        if n_action_steps is not None:
            n_steps = int(n_action_steps)
            if n_steps < 1:
                raise ValueError("n_action_steps must be >= 1")
            if n_steps > chunk_size:
                raise ValueError(
                    f"n_action_steps ({n_steps}) must be <= action_chunk_size/chunk_size ({chunk_size})"
                )
            config_dict["n_action_steps"] = n_steps
        elif "n_action_steps" in config_dict and int(config_dict["n_action_steps"]) > chunk_size:
            config_dict["n_action_steps"] = chunk_size
    
    def _init_components(self):
        """Initialize all components after model is loaded."""
        self._action_queue = TimestampedActionQueue(self.smoothing_config)
        self._action_queue.set_chunk_size(self.n_action_steps)
        
        self._latency_estimator = LatencyEstimator(
            alpha=self.smoothing_config.latency_ema_alpha,
            initial_value=0.1
        )
        
        self._gripper_smoother = GripperSmoother(
            self.smoothing_config,
            self.action_dim
        )
        self._temporal_ensembler = None
    
    @abstractmethod
    def load(self, checkpoint_dir: str) -> Tuple[bool, str]:
        """Load model from checkpoint directory."""
        pass
    
    @abstractmethod
    def _predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Predict action chunk from observation.
        
        Args:
            images: Dict of {camera_role: image array (H, W, 3)}
            state: Robot state array (state_dim,)
            
        Returns:
            Action chunk (n_action_steps, action_dim)
        """
        pass
    
    def reset(self):
        """Reset state for new episode."""
        # Reset all components
        if self._action_queue is not None:
            self._action_queue.reset()
        if self._gripper_smoother is not None:
            self._gripper_smoother.reset()
        if self._temporal_ensembler is not None:
            self._temporal_ensembler.reset()
        
        # Reset episode state
        self._episode_start_time = time.time()
        self._current_timestep = 0
        self._fallback_count = 0
        
        logger.debug(f"{self.model_type} inference engine reset")
    
    def predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Predict a raw action chunk without queue execution semantics.

        This is useful for offline validation and analysis tools that want the
        direct policy output instead of the single action selected by the
        control loop.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return self._predict_chunk(images, state)

    def step(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """Public alias for one control-loop step."""
        return self.select_action(images, state)

    def select_action(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Select action with LeRobot v0.5.1 FIFO queue semantics.
        
        Flow:
        1. Pop the next queued action.
        2. If the queue is empty, run synchronous chunk inference and enqueue
           the first n_action_steps actions.
        3. Return exactly one action. Wall-clock time never skips queued actions.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if self.smoothing_config.enable_rtc:
            raise RuntimeError(
                "RTC is not supported for step()/predict_action(). "
                "Use predict_action_chunk() and an RTC-aware external action queue instead."
            )

        timed_action = self._action_queue.get_next_action()
        if timed_action is None:
            if self._trace_recorder:
                self._trace_recorder.record("Engine", "Queue Empty", timestep=self._current_timestep)

            start_time = time.perf_counter()
            action_chunk = self._predict_chunk(images, state)
            elapsed = time.perf_counter() - start_time

            self._latency_estimator.update(elapsed)

            action_chunk = np.asarray(action_chunk, dtype=np.float32)
            action_chunk = action_chunk[: max(1, int(self.n_action_steps))]
            timed_actions = [
                TimedAction(
                    timestamp=0.0,
                    timestep=self._current_timestep + i,
                    action=action_chunk[i]
                )
                for i in range(len(action_chunk))
            ]
            self._action_queue.add_action_chunk(timed_actions)
            timed_action = self._action_queue.get_next_action()

            logger.debug(f"Sync inference: {elapsed*1000:.1f}ms")

        if timed_action is None:
            raise RuntimeError("Policy returned an empty action chunk")

        action = timed_action.get_action()
        self._current_timestep = timed_action.get_timestep() + 1
        
        # Apply gripper smoothing
        if self._gripper_smoother is not None:
            action = self._gripper_smoother.smooth(action)

        return action
    
    def _get_fallback_action(self, state: np.ndarray) -> np.ndarray:
        """Get fallback action when queue is empty."""
        mode = self.smoothing_config.fallback_mode
        
        if mode == "repeat" and self._gripper_smoother is not None:
            last_action = self._gripper_smoother.get_last_action()
            if last_action is not None:
                return last_action
        
        # "hold" mode or no last action: return current state
        return state[:self.action_dim].copy() if len(state) >= self.action_dim else np.zeros(self.action_dim)
    
    # ==================== Status Methods ====================
    
    def get_queue_size(self) -> int:
        """Get current action queue size."""
        if self._action_queue is not None:
            return self._action_queue.get_queue_size()
        return 0
    
    def get_fallback_count(self) -> int:
        """Get count of fallback uses."""
        return self._fallback_count
    
    def get_latency_estimate(self) -> float:
        """Get current latency estimate in seconds."""
        if self._latency_estimator is not None:
            return self._latency_estimator.get_value()
        return 0.0
    
    def get_required_cameras(self) -> List[str]:
        """Return list of required camera roles."""
        return self.required_cameras
    
    def get_state_dim(self) -> int:
        """Return expected state dimension."""
        return self.state_dim

    def get_device_status(self) -> Dict[str, Optional[str]]:
        """Return requested/actual device metadata for observability."""
        return {
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "device_warning": self.device_warning,
        }

    def get_robot_io(self) -> Optional[Dict[str, Any]]:
        """Return optional bundled robot I/O metadata."""
        return copy.deepcopy(self.robot_io) if self.robot_io is not None else None
    
    def set_control_fps(self, fps: float):
        """Update control frequency."""
        self.smoothing_config.control_fps = fps
        if self._action_queue is not None:
            self._action_queue.config.control_fps = fps
    
    def set_smoothing_config(self, config: SmoothingConfig):
        """Update smoothing configuration."""
        self.smoothing_config = config
    
    @staticmethod
    def validate_checkpoint(checkpoint_dir: str) -> Tuple[bool, str]:
        """Validate that checkpoint directory contains required files."""
        path = Path(checkpoint_dir)
        
        if not path.exists():
            return False, f"Checkpoint目录不存在: {checkpoint_dir}"
        
        required_files = ["inference_config.yaml", "model.pth", "stats.json"]
        missing = []
        for f in required_files:
            if not (path / f).exists():
                missing.append(f)
        
        if missing:
            return False, f"缺少必需文件: {', '.join(missing)}"
        
        return True, ""
    
    @abstractmethod
    def unload(self):
        """Unload model and free memory."""
        pass
    def set_trace_recorder(self, recorder: TraceRecorder):
        """Set trace recorder for observability."""
        self._trace_recorder = recorder
