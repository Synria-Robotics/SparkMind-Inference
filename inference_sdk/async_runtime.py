"""Process-local async inference runtime.

This module provides a LeRobot-style async inference runtime without a
server/client split. It owns the observation queue, background inference
thread, timestamped action queue, lifecycle state, and runtime metrics.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np

from .api import InferenceSDK, Observation, PolicyMetadata
from .base import (
    GripperSmoother,
    LatencyEstimator,
    ObservationQueue,
    SmoothingConfig,
    TimedAction,
    TimedObservation,
    TimestampedActionQueue,
    TraceEvent,
    TraceRecorder,
    get_aggregate_function,
)
from .factory import normalize_model_type

logger = logging.getLogger(__name__)


class AsyncRuntimeState(str, Enum):
    """Lifecycle states for :class:`AsyncInferenceRuntime`."""

    UNLOADED = "unloaded"
    LOADED = "loaded"
    RUNNING = "running"
    STOPPED = "stopped"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class AsyncInferenceConfig:
    """Configuration for process-local async inference scheduling."""

    control_fps: float = 30.0
    chunk_size_threshold: float = 0.5
    action_chunk_size: Optional[int] = None
    n_action_steps: Optional[int] = None
    aggregate_fn_name: str = "weighted_average"
    obs_queue_maxsize: int = 1
    fallback_mode: str = "repeat"
    latency_ema_alpha: float = 0.2
    latency_safety_margin: float = 1.5
    enable_gripper_clamping: bool = True
    gripper_max_velocity: float = 200.0
    copy_observation: bool = True
    max_consecutive_errors: int = 5
    safe_action: Optional[np.ndarray] = None
    safe_action_fn: Optional[Callable[[Optional[np.ndarray]], np.ndarray]] = None
    action_min: Optional[float | np.ndarray] = None
    action_max: Optional[float | np.ndarray] = None
    clip_action: bool = False
    trace_max_events: int = 1000
    enable_temporal_ensemble: bool = False
    temporal_ensemble_coeff: float = 0.01
    enable_rtc: bool = False
    rtc_prefix_attention_schedule: str = "LINEAR"
    rtc_max_guidance_weight: float = 10.0
    rtc_execution_horizon: int = 10
    rtc_inference_delay_steps: int = 0
    rtc_debug: bool = False
    rtc_debug_maxlen: int = 100

    @property
    def environment_dt(self) -> float:
        """Control interval in seconds."""
        return 1.0 / self.control_fps

    def to_smoothing_config(self, *, enable_async_inference: bool = False) -> SmoothingConfig:
        """Convert to the SDK's lower-level smoothing config."""
        self.validate()
        return SmoothingConfig(
            control_fps=self.control_fps,
            gripper_max_velocity=self.gripper_max_velocity,
            enable_gripper_clamping=self.enable_gripper_clamping,
            enable_async_inference=enable_async_inference,
            chunk_size_threshold=self.chunk_size_threshold,
            action_chunk_size=self.action_chunk_size,
            n_action_steps=self.n_action_steps,
            latency_ema_alpha=self.latency_ema_alpha,
            latency_safety_margin=self.latency_safety_margin,
            aggregate_fn_name=self.aggregate_fn_name,
            obs_queue_maxsize=self.obs_queue_maxsize,
            fallback_mode=self.fallback_mode,
            enable_temporal_ensemble=self.enable_temporal_ensemble,
            temporal_ensemble_coeff=self.temporal_ensemble_coeff,
            enable_rtc=self.enable_rtc,
            rtc_prefix_attention_schedule=self.rtc_prefix_attention_schedule,
            rtc_max_guidance_weight=self.rtc_max_guidance_weight,
            rtc_execution_horizon=self.rtc_execution_horizon,
            rtc_inference_delay_steps=self.rtc_inference_delay_steps,
            rtc_debug=self.rtc_debug,
            rtc_debug_maxlen=self.rtc_debug_maxlen,
        )

    def validate(self) -> None:
        """Validate scheduling and safety parameters."""
        if self.control_fps <= 0:
            raise ValueError("control_fps must be > 0")
        if not 0.0 <= self.chunk_size_threshold <= 1.0:
            raise ValueError("chunk_size_threshold must be between 0.0 and 1.0")
        if self.action_chunk_size is not None and self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be >= 1")
        if self.n_action_steps is not None and self.n_action_steps < 1:
            raise ValueError("n_action_steps must be >= 1")
        if (
            self.action_chunk_size is not None
            and self.n_action_steps is not None
            and self.n_action_steps > self.action_chunk_size
        ):
            raise ValueError("n_action_steps must be <= action_chunk_size")
        if self.obs_queue_maxsize < 1:
            raise ValueError("obs_queue_maxsize must be >= 1")
        if self.fallback_mode not in {"repeat", "hold"}:
            raise ValueError("fallback_mode must be 'repeat' or 'hold'")
        if not 0.0 < self.latency_ema_alpha <= 1.0:
            raise ValueError("latency_ema_alpha must be in (0.0, 1.0]")
        if self.latency_safety_margin < 0.0:
            raise ValueError("latency_safety_margin must be >= 0.0")
        if self.gripper_max_velocity < 0.0:
            raise ValueError("gripper_max_velocity must be >= 0.0")
        if self.temporal_ensemble_coeff < 0.0:
            raise ValueError("temporal_ensemble_coeff must be >= 0.0")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be >= 1")
        if self.trace_max_events < 1:
            raise ValueError("trace_max_events must be >= 1")
        if self.rtc_prefix_attention_schedule.strip().upper() not in {"ZEROS", "ONES", "LINEAR", "EXP"}:
            raise ValueError("rtc_prefix_attention_schedule must be one of: ZEROS, ONES, LINEAR, EXP")
        if self.rtc_max_guidance_weight <= 0.0:
            raise ValueError("rtc_max_guidance_weight must be > 0.0")
        if self.rtc_execution_horizon < 1:
            raise ValueError("rtc_execution_horizon must be >= 1")
        if self.rtc_inference_delay_steps < 0:
            raise ValueError("rtc_inference_delay_steps must be >= 0")
        if self.rtc_debug_maxlen < 1:
            raise ValueError("rtc_debug_maxlen must be >= 1")
        get_aggregate_function(self.aggregate_fn_name)


@dataclass(frozen=True)
class AsyncRuntimeStatus:
    """Runtime status for backend APIs and frontend dashboards."""

    state: str
    loaded: bool
    running: bool
    model_type: Optional[str]
    checkpoint_dir: Optional[str]
    queue_size: int
    fill_ratio: float
    latency_estimate: float
    fallback_count: int
    current_timestep: int
    last_error: Optional[str]
    worker_alive: bool
    inference_count: int
    processed_observation_count: int
    dropped_observation_count: int
    skipped_observation_count: int
    action_chunk_count: int
    queue_empty_count: int
    error_count: int
    consecutive_error_count: int
    last_inference_ms: Optional[float]
    max_inference_ms: Optional[float]
    last_observation_timestep: Optional[int]
    last_action_timestep: Optional[int]


@dataclass(frozen=True)
class AsyncStepResult:
    """Result returned by one control-loop step."""

    action: np.ndarray
    source: str
    timestep: int
    submitted_observation: bool
    queue_size: int
    latency_estimate: float
    action_timestep: Optional[int] = None


@dataclass(frozen=True)
class QueueSnapshotEntry:
    """Compact debug view of one queued action."""

    timestep: int
    timestamp: float
    action_shape: tuple[int, ...]


@dataclass
class _QueuedObservation:
    observation: TimedObservation
    generation: int
    instruction: Optional[str] = None


class AsyncInferenceRuntime:
    """Process-local async inference runtime.

    The runtime owns the async scheduling layer. The underlying policy engine is
    deliberately loaded with ``enable_async_inference=False`` so there is only
    one action queue and one background worker in the process.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._predict_lock = threading.Lock()
        self._state = AsyncRuntimeState.UNLOADED
        self._sdk: Optional[InferenceSDK] = None
        self._metadata: Optional[PolicyMetadata] = None
        self._config = AsyncInferenceConfig()
        self._model_type: Optional[str] = None
        self._checkpoint_dir: Optional[str] = None
        self._instruction: Optional[str] = None
        self._device: str = "cuda:0"
        self._strict_device: bool = False

        self._action_queue: Optional[TimestampedActionQueue] = None
        self._obs_queue: Optional[ObservationQueue] = None
        self._latency_estimator: Optional[LatencyEstimator] = None
        self._gripper_smoother: Optional[GripperSmoother] = None
        self._trace_recorder = TraceRecorder(max_events=self._config.trace_max_events)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._episode_start_time: float = 0.0
        self._current_timestep: int = 0
        self._generation: int = 0
        self._must_go_next = True
        self._last_action: Optional[np.ndarray] = None
        self._last_error: Optional[str] = None

        self._reset_metrics_locked()

    def load_policy(
        self,
        algorithm_type: str,
        checkpoint_dir: str,
        *,
        device: str = "cuda:0",
        instruction: Optional[str] = None,
        config: Optional[AsyncInferenceConfig | SmoothingConfig] = None,
        force_reload: bool = False,
        strict_device: bool = False,
    ) -> PolicyMetadata:
        """Load one policy and initialize async queues.

        A runtime instance manages one policy at a time. Loading a policy while
        the worker is running is rejected to avoid mixing old queued actions with
        a new model.
        """
        del force_reload  # Runtime policy loading always replaces the current model.

        with self._lock:
            if self._state == AsyncRuntimeState.RUNNING:
                raise RuntimeError("Stop async inference before loading another policy.")
            if self._state == AsyncRuntimeState.CLOSED:
                raise RuntimeError("Async inference runtime is closed. Create or get a new runtime.")
            if self._state == AsyncRuntimeState.ERROR:
                raise RuntimeError("Reset or close the async inference runtime before loading another policy.")

            self._close_sdk_locked()
            self._config = _normalize_config(config)
            self._trace_recorder = TraceRecorder(max_events=self._config.trace_max_events)
            self._device = device
            self._strict_device = strict_device
            self._model_type = normalize_model_type(algorithm_type)
            if self._config.enable_temporal_ensemble and self._model_type != "act":
                raise ValueError("Temporal ensemble is only supported for ACT policies.")
            self._checkpoint_dir = str(checkpoint_dir)
            self._instruction = instruction

            sdk = InferenceSDK(
                device=device,
                smoothing_config=self._config.to_smoothing_config(enable_async_inference=False),
                strict_device=strict_device,
            )
            try:
                metadata = sdk.load_policy(
                    self._model_type,
                    self._checkpoint_dir,
                    instruction=instruction,
                    force_reload=True,
                )
            except Exception:
                sdk.close()
                raise

            self._sdk = sdk
            self._metadata = metadata
            self._init_components_locked()
            self._reset_episode_locked(clear_metrics=True)
            self._state = AsyncRuntimeState.LOADED
            self._record_trace_locked("Runtime", "Policy Loaded", model_type=self._model_type)
            return metadata

    def start(self) -> None:
        """Start the background inference worker."""
        with self._lock:
            self._require_loaded_locked()
            if self._state == AsyncRuntimeState.RUNNING:
                return
            if self._state not in {AsyncRuntimeState.LOADED, AsyncRuntimeState.STOPPED}:
                raise RuntimeError(f"Cannot start async inference from state {self._state.value}.")

            if self._state == AsyncRuntimeState.STOPPED or (
                self._state == AsyncRuntimeState.LOADED and self._queue_size_locked() == 0
            ):
                self._reset_episode_locked(clear_metrics=False)
            self._stop_event.clear()
            self._state = AsyncRuntimeState.RUNNING
            self._thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="AsyncInferenceRuntimeWorker",
            )
            self._thread.start()
            self._record_trace_locked("Runtime", "Started")

    def stop(self) -> None:
        """Stop the background worker while keeping the model loaded."""
        thread: Optional[threading.Thread]
        with self._lock:
            if self._state != AsyncRuntimeState.RUNNING:
                return
            self._stop_event.set()
            thread = self._thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

        with self._lock:
            if thread is not None and thread.is_alive():
                self._set_error_locked("Async inference worker did not stop within 5 seconds.")
                return
            self._clear_observation_queue_locked()
            self._thread = None
            if self._state != AsyncRuntimeState.ERROR:
                self._state = AsyncRuntimeState.STOPPED
            self._record_trace_locked("Runtime", "Stopped")

    def reset(self, *, clear_metrics: bool = True) -> None:
        """Reset episode queues, timestep state, and optionally metrics."""
        with self._lock:
            if self._state == AsyncRuntimeState.CLOSED:
                raise RuntimeError("Async inference runtime is closed. Create or get a new runtime.")
            self._require_loaded_locked()
            self._reset_episode_locked(clear_metrics=clear_metrics)
            if self._state == AsyncRuntimeState.ERROR:
                self._state = AsyncRuntimeState.STOPPED
            self._record_trace_locked("Runtime", "Reset")

    def close(self) -> None:
        """Stop the worker and unload the model."""
        thread: Optional[threading.Thread]
        with self._lock:
            self._stop_event.set()
            thread = self._thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

        with self._lock:
            self._close_sdk_locked()
            self._action_queue = None
            self._obs_queue = None
            self._latency_estimator = None
            self._gripper_smoother = None
            self._thread = None
            self._metadata = None
            self._model_type = None
            self._checkpoint_dir = None
            self._instruction = None
            self._last_action = None
            self._state = AsyncRuntimeState.CLOSED
            self._record_trace_locked("Runtime", "Closed")

    def warmup(
        self,
        observation: Observation | Mapping[str, object] | None = None,
        *,
        images: Optional[Mapping[str, np.ndarray]] = None,
        state: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
        timestamp: Optional[float] = None,
        timestep: Optional[int] = None,
    ) -> np.ndarray:
        """Run one synchronous inference and put the chunk into the action queue."""
        images_dict, state_array, obs_instruction = self._coerce_observation_args(
            observation=observation,
            images=images,
            state=state,
            instruction=instruction,
        )

        with self._lock:
            self._require_loaded_locked()
            if self._state == AsyncRuntimeState.CLOSED:
                raise RuntimeError("Async inference runtime is closed. Create or get a new runtime.")
            observation_timestamp = time.monotonic() if timestamp is None else float(timestamp)
            if self._state in {AsyncRuntimeState.LOADED, AsyncRuntimeState.STOPPED} and self._queue_size_locked() == 0:
                initial_timestep = 0 if timestep is None else int(timestep)
                self._episode_start_time = observation_timestamp - initial_timestep * self._config.environment_dt
                self._current_timestep = initial_timestep
            elif self._episode_start_time <= 0.0:
                self._episode_start_time = observation_timestamp
            warmup_timestep = self._resolve_timestep_locked(observation_timestamp, timestep)

        action_chunk, elapsed = self._predict_chunk(images_dict, state_array, obs_instruction)

        with self._lock:
            queue_was_empty = self._queue_size_locked() == 0
            if self._state in {AsyncRuntimeState.LOADED, AsyncRuntimeState.STOPPED} and queue_was_empty:
                if timestamp is None and timestep is None:
                    action_timestamp = time.monotonic()
                    self._episode_start_time = action_timestamp
                    warmup_timestep = 0
                else:
                    action_timestamp = observation_timestamp
                    self._episode_start_time = action_timestamp - warmup_timestep * self._config.environment_dt
            else:
                action_timestamp = observation_timestamp
            self._add_action_chunk_locked(action_chunk, timestamp=action_timestamp, timestep=warmup_timestep)
            self._update_latency_metrics_locked(elapsed)
            self._record_trace_locked(
                "Runtime",
                "Warmup",
                duration_ms=elapsed * 1000.0,
                chunk_size=len(action_chunk),
            )
        return action_chunk

    def wait_until_ready(self, *, min_queue_size: int = 1, timeout: float = 5.0) -> bool:
        """Wait until the action queue contains enough actions."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._state == AsyncRuntimeState.ERROR:
                    return False
                if self._action_queue is not None and self._action_queue.get_queue_size() >= min_queue_size:
                    return True
            time.sleep(0.01)
        return False

    def submit_observation(
        self,
        observation: Observation | Mapping[str, object] | None = None,
        *,
        images: Optional[Mapping[str, np.ndarray]] = None,
        state: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
        must_go: bool = False,
        timestamp: Optional[float] = None,
        timestep: Optional[int] = None,
    ) -> bool:
        """Submit the latest observation for background inference."""
        images_dict, state_array, obs_instruction = self._coerce_observation_args(
            observation=observation,
            images=images,
            state=state,
            instruction=instruction,
        )
        return self._submit_observation_arrays(
            images=images_dict,
            state=state_array,
            instruction=obs_instruction,
            must_go=must_go,
            timestamp=timestamp,
            timestep=timestep,
        )

    def get_action(self, *, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the current action from the queue or fallback policy."""
        return self.get_action_result(state=state).action

    def get_action_result(
        self,
        *,
        state: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        timestep: Optional[int] = None,
    ) -> AsyncStepResult:
        """Return the current action with debug metadata."""
        with self._lock:
            self._require_running_locked()
            action_timestamp, action_step = self._resolve_control_time_locked(timestamp, timestep)
            action, source, action_timestep = self._get_action_locked(action_timestamp, state)
            queue_size = self._queue_size_locked()
            latency = self._latency_locked()
        return AsyncStepResult(
            action=action,
            source=source,
            timestep=action_step,
            submitted_observation=False,
            queue_size=queue_size,
            latency_estimate=latency,
            action_timestep=action_timestep,
        )

    def step(
        self,
        observation: Observation | Mapping[str, object] | None = None,
        *,
        images: Optional[Mapping[str, np.ndarray]] = None,
        state: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
        timestamp: Optional[float] = None,
        timestep: Optional[int] = None,
    ) -> AsyncStepResult:
        """Submit an observation if needed, then return one action."""
        images_dict, state_array, obs_instruction = self._coerce_observation_args(
            observation=observation,
            images=images,
            state=state,
            instruction=instruction,
        )

        submitted = False
        with self._lock:
            self._require_running_locked()
            step_timestamp, step_timestep = self._resolve_control_time_locked(timestamp, timestep)
            should_submit = self._should_submit_observation_locked()
            must_go = self._queue_size_locked() == 0 or self._must_go_next

        if should_submit:
            submitted = self._submit_observation_arrays(
                images=images_dict,
                state=state_array,
                instruction=obs_instruction,
                must_go=must_go,
                timestamp=step_timestamp,
                timestep=step_timestep,
            )

        with self._lock:
            self._require_running_locked()
            action, source, action_timestep = self._get_action_locked(step_timestamp, state_array)
            queue_size = self._queue_size_locked()
            latency = self._latency_locked()
        return AsyncStepResult(
            action=action,
            source=source,
            timestep=step_timestep,
            submitted_observation=submitted,
            queue_size=queue_size,
            latency_estimate=latency,
            action_timestep=action_timestep,
        )

    def predict_action_chunk(
        self,
        observation: Observation | Mapping[str, object] | None = None,
        *,
        images: Optional[Mapping[str, np.ndarray]] = None,
        state: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """Run direct synchronous action chunk inference."""
        images_dict, state_array, obs_instruction = self._coerce_observation_args(
            observation=observation,
            images=images,
            state=state,
            instruction=instruction,
        )
        action_chunk, _ = self._predict_chunk(images_dict, state_array, obs_instruction)
        return action_chunk

    def get_status(self) -> AsyncRuntimeStatus:
        """Return current runtime status and metrics."""
        with self._lock:
            worker_alive = self._thread is not None and self._thread.is_alive()
            return AsyncRuntimeStatus(
                state=self._state.value,
                loaded=self._sdk is not None and self._metadata is not None,
                running=self._state == AsyncRuntimeState.RUNNING and worker_alive,
                model_type=self._model_type,
                checkpoint_dir=self._checkpoint_dir,
                queue_size=self._queue_size_locked(),
                fill_ratio=self._fill_ratio_locked(),
                latency_estimate=self._latency_locked(),
                fallback_count=self._fallback_count,
                current_timestep=self._current_timestep,
                last_error=self._last_error,
                worker_alive=worker_alive,
                inference_count=self._inference_count,
                processed_observation_count=self._processed_observation_count,
                dropped_observation_count=self._dropped_observation_count,
                skipped_observation_count=self._skipped_observation_count,
                action_chunk_count=self._action_chunk_count,
                queue_empty_count=self._queue_empty_count,
                error_count=self._error_count,
                consecutive_error_count=self._consecutive_error_count,
                last_inference_ms=self._last_inference_ms,
                max_inference_ms=self._max_inference_ms,
                last_observation_timestep=self._last_observation_timestep,
                last_action_timestep=self._last_action_timestep,
            )

    def get_queue_snapshot(self, *, limit: int = 20) -> List[QueueSnapshotEntry]:
        """Return a compact debug snapshot of queued actions."""
        with self._lock:
            if self._action_queue is None:
                return []
            return [QueueSnapshotEntry(**entry) for entry in self._action_queue.get_snapshot(limit=limit)]

    def get_trace_events(self, *, limit: int = 100) -> List[TraceEvent]:
        """Return recent trace events."""
        with self._trace_recorder._lock:
            return list(self._trace_recorder.events[-limit:])

    def clear_metrics(self) -> None:
        """Reset metrics and trace events."""
        with self._lock:
            self._reset_metrics_locked()
            self._trace_recorder.clear()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                obs_queue = self._obs_queue
                if obs_queue is None:
                    time.sleep(0.01)
                    continue

                item = obs_queue.get(timeout=0.1)
                if item is None:
                    continue
                if not isinstance(item, _QueuedObservation):
                    continue

                with self._lock:
                    if self._state != AsyncRuntimeState.RUNNING:
                        continue
                    if item.generation != self._generation:
                        self._skipped_observation_count += 1
                        continue
                    should_process = item.observation.must_go or self._should_request_chunk_locked()
                    if not should_process:
                        self._skipped_observation_count += 1
                        self._record_trace_locked(
                            "Worker",
                            "Skipped Observation",
                            timestep=item.observation.timestep,
                        )
                        continue

                self._record_trace("Worker", "Start Inference", timestep=item.observation.timestep)
                action_chunk, elapsed = self._predict_chunk(
                    item.observation.images,
                    item.observation.state,
                    item.instruction,
                )

                with self._lock:
                    if item.generation != self._generation or self._state != AsyncRuntimeState.RUNNING:
                        self._skipped_observation_count += 1
                        continue
                    self._add_action_chunk_locked(
                        action_chunk,
                        timestamp=item.observation.timestamp,
                        timestep=item.observation.timestep,
                    )
                    self._processed_observation_count += 1
                    self._action_chunk_count += 1
                    self._inference_count += 1
                    self._consecutive_error_count = 0
                    self._last_error = None
                    self._update_latency_metrics_locked(elapsed)
                    self._record_trace_locked(
                        "Worker",
                        "End Inference",
                        timestep=item.observation.timestep,
                        duration_ms=elapsed * 1000.0,
                        chunk_size=len(action_chunk),
                        queue_size=self._queue_size_locked(),
                    )
            except Exception as exc:
                logger.exception("Async inference worker error")
                with self._lock:
                    self._record_error_locked(str(exc))
                    if self._consecutive_error_count >= self._config.max_consecutive_errors:
                        self._state = AsyncRuntimeState.ERROR
                        self._stop_event.set()
                        self._record_trace_locked(
                            "Worker",
                            "Stopped On Errors",
                            consecutive_errors=self._consecutive_error_count,
                        )

    def _predict_chunk(
        self,
        images: Mapping[str, np.ndarray],
        state: np.ndarray,
        instruction: Optional[str],
    ) -> tuple[np.ndarray, float]:
        with self._lock:
            self._require_loaded_locked()
            if self._sdk is None or self._model_type is None:
                raise RuntimeError("Async inference runtime has no loaded policy. Call load_policy() first.")
            self._predict_lock.acquire()
            sdk = self._sdk
            model_type = self._model_type
            default_instruction = self._instruction
            action_dim = self._action_dim_locked()

        try:
            start_time = time.monotonic()
            action_chunk = sdk.predict_action_chunk(
                model_type,
                images=images,
                state=state,
                instruction=instruction if instruction is not None else default_instruction,
            )
            elapsed = time.monotonic() - start_time
            return self._validate_action_chunk(action_chunk, action_dim), elapsed
        finally:
            self._predict_lock.release()

    def _submit_observation_arrays(
        self,
        *,
        images: Mapping[str, np.ndarray],
        state: np.ndarray,
        instruction: Optional[str],
        must_go: bool,
        timestamp: Optional[float],
        timestep: Optional[int],
    ) -> bool:
        with self._lock:
            self._require_running_locked()
            obs_timestamp = time.monotonic() if timestamp is None else float(timestamp)
            obs_timestep = self._compute_timestep_locked(obs_timestamp) if timestep is None else int(timestep)
            queue_empty = self._queue_size_locked() == 0
            obs_must_go = bool(must_go or queue_empty or self._must_go_next)
            if obs_must_go:
                self._must_go_next = False

            obs = TimedObservation(
                timestamp=obs_timestamp,
                timestep=obs_timestep,
                images=self._copy_images(images),
                state=self._copy_state(state),
                must_go=obs_must_go,
            )
            item = _QueuedObservation(
                observation=obs,
                generation=self._generation,
                instruction=instruction,
            )
            if self._obs_queue is None:
                raise RuntimeError("Observation queue is not initialized.")
            accepted, dropped = self._obs_queue.put_with_drop_info(item)
            if accepted:
                self._submitted_observation_count += 1
                self._last_observation_timestep = obs_timestep
                if dropped:
                    self._dropped_observation_count += 1
                self._record_trace_locked(
                    "Runtime",
                    "Observation Submitted",
                    timestep=obs_timestep,
                    must_go=obs_must_go,
                    dropped_old=dropped,
                )
            return accepted

    def _get_action_locked(
        self,
        timestamp: float,
        state: Optional[np.ndarray],
    ) -> tuple[np.ndarray, str, Optional[int]]:
        if self._action_queue is None:
            raise RuntimeError("Action queue is not initialized.")

        timed_action = self._action_queue.get_action_for_time(timestamp, self._episode_start_time)
        if timed_action is None:
            self._queue_empty_count += 1
            self._fallback_count += 1
            self._must_go_next = True
            action, source = self._fallback_action_locked(state)
            action_timestep = None
            self._record_trace_locked("Runtime", "Fallback Action", action_source=source)
        else:
            action = timed_action.action
            source = "queue"
            action_timestep = timed_action.timestep
            self._last_action_timestep = timed_action.timestep

        action = self._validate_action(action)
        if self._gripper_smoother is not None:
            action = self._gripper_smoother.smooth(action)
        self._last_action = action.copy()
        return action, source, action_timestep

    def _fallback_action_locked(self, state: Optional[np.ndarray]) -> tuple[np.ndarray, str]:
        action_dim = self._action_dim_locked()
        if self._config.safe_action_fn is not None:
            return np.asarray(self._config.safe_action_fn(state), dtype=np.float32).copy(), "fallback_safe_fn"
        if self._config.safe_action is not None:
            return np.asarray(self._config.safe_action, dtype=np.float32).copy(), "fallback_safe"
        if self._config.fallback_mode == "repeat" and self._last_action is not None:
            return self._last_action.copy(), "fallback_repeat"
        if self._config.fallback_mode == "hold" and state is not None and len(state) >= action_dim:
            return np.asarray(state[:action_dim], dtype=np.float32).copy(), "fallback_hold"
        return np.zeros(action_dim, dtype=np.float32), "fallback_zero"

    def _coerce_observation_args(
        self,
        *,
        observation: Observation | Mapping[str, object] | None,
        images: Optional[Mapping[str, np.ndarray]],
        state: Optional[np.ndarray],
        instruction: Optional[str],
    ) -> tuple[Dict[str, np.ndarray], np.ndarray, Optional[str]]:
        if observation is not None:
            if images is not None or state is not None:
                raise ValueError("Do not mix observation=... with images=... or state=...")
            if isinstance(observation, Observation):
                obs_images = observation.images
                obs_state = observation.state
                obs_instruction = instruction if instruction is not None else observation.instruction
            elif isinstance(observation, Mapping):
                obs_images = observation.get("images")
                obs_state = observation.get("state")
                obs_instruction = instruction if instruction is not None else observation.get("instruction")
            else:
                obs_images = getattr(observation, "images")
                obs_state = getattr(observation, "state")
                obs_instruction = instruction if instruction is not None else getattr(observation, "instruction", None)
        else:
            if images is None or state is None:
                raise ValueError("Pass either observation=... or both images=... and state=...")
            obs_images = images
            obs_state = state
            obs_instruction = instruction

        if not isinstance(obs_images, Mapping):
            raise TypeError("images must be a mapping of camera name to numpy array")

        images_dict = {str(name): np.asarray(image) for name, image in obs_images.items()}
        state_array = np.asarray(obs_state, dtype=np.float32)
        return images_dict, state_array, None if obs_instruction is None else str(obs_instruction)

    def _copy_images(self, images: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self._config.copy_observation:
            return {str(name): np.asarray(image) for name, image in images.items()}
        return {str(name): np.asarray(image).copy() for name, image in images.items()}

    def _copy_state(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        return state_array.copy() if self._config.copy_observation else state_array

    def _validate_action_chunk(self, action_chunk: np.ndarray, action_dim: Optional[int] = None) -> np.ndarray:
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"action_chunk must be 2-D, got shape {chunk.shape}")
        if action_dim is None:
            action_dim = self._action_dim_locked()
        if chunk.shape[1] != action_dim:
            raise ValueError(f"action_chunk action_dim mismatch: expected {action_dim}, got {chunk.shape[1]}")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("action_chunk contains NaN or Inf")
        if self._config.clip_action:
            chunk = self._clip_action(chunk)
        elif self._config.action_min is not None or self._config.action_max is not None:
            self._ensure_action_bounds(chunk)
        return chunk

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        action_dim = self._action_dim_locked()
        if action_array.ndim != 1 or action_array.shape[0] != action_dim:
            raise ValueError(f"action must have shape ({action_dim},), got {action_array.shape}")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action contains NaN or Inf")
        if self._config.clip_action:
            action_array = self._clip_action(action_array)
        elif self._config.action_min is not None or self._config.action_max is not None:
            self._ensure_action_bounds(action_array)
        return action_array

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        lower = self._config.action_min if self._config.action_min is not None else -np.inf
        upper = self._config.action_max if self._config.action_max is not None else np.inf
        return np.clip(action, lower, upper).astype(np.float32, copy=False)

    def _ensure_action_bounds(self, action: np.ndarray) -> None:
        if self._config.action_min is not None and np.any(action < self._config.action_min):
            raise ValueError("action is below configured action_min")
        if self._config.action_max is not None and np.any(action > self._config.action_max):
            raise ValueError("action is above configured action_max")

    def _init_components_locked(self) -> None:
        smoothing_config = self._config.to_smoothing_config(enable_async_inference=False)
        self._action_queue = TimestampedActionQueue(smoothing_config)
        self._action_queue.set_chunk_size(self._chunk_size_locked())
        self._obs_queue = ObservationQueue(maxsize=self._config.obs_queue_maxsize)
        self._latency_estimator = LatencyEstimator(
            alpha=self._config.latency_ema_alpha,
            initial_value=0.1,
        )
        self._gripper_smoother = GripperSmoother(smoothing_config, self._action_dim_locked())

    def _reset_episode_locked(self, *, clear_metrics: bool) -> None:
        self._generation += 1
        self._episode_start_time = time.monotonic()
        self._current_timestep = 0
        self._must_go_next = True
        self._last_action = None
        self._last_error = None
        if self._action_queue is not None:
            self._action_queue.reset()
        self._clear_observation_queue_locked()
        if self._gripper_smoother is not None:
            self._gripper_smoother.reset()
        if clear_metrics:
            self._reset_metrics_locked()

    def _reset_metrics_locked(self) -> None:
        self._submitted_observation_count = 0
        self._processed_observation_count = 0
        self._dropped_observation_count = 0
        self._skipped_observation_count = 0
        self._inference_count = 0
        self._action_chunk_count = 0
        self._queue_empty_count = 0
        self._fallback_count = 0
        self._error_count = 0
        self._consecutive_error_count = 0
        self._last_inference_ms: Optional[float] = None
        self._max_inference_ms: Optional[float] = None
        self._last_observation_timestep: Optional[int] = None
        self._last_action_timestep: Optional[int] = None

    def _clear_observation_queue_locked(self) -> None:
        if self._obs_queue is not None:
            self._obs_queue.clear()

    def _close_sdk_locked(self) -> None:
        with self._predict_lock:
            if self._sdk is not None:
                self._sdk.close()
                self._sdk = None

    def _add_action_chunk_locked(self, action_chunk: np.ndarray, *, timestamp: float, timestep: int) -> None:
        if self._action_queue is None:
            raise RuntimeError("Action queue is not initialized.")
        dt = self._config.environment_dt
        timed_actions = [
            TimedAction(
                timestamp=timestamp + i * dt,
                timestep=timestep + i,
                action=action_chunk[i],
            )
            for i in range(len(action_chunk))
        ]
        self._action_queue.add_action_chunk(timed_actions)

    def _should_submit_observation_locked(self) -> bool:
        if self._queue_size_locked() == 0:
            return True
        return self._should_request_chunk_locked()

    def _should_request_chunk_locked(self) -> bool:
        if self._action_queue is None or self._latency_estimator is None:
            return False
        queue_size = self._action_queue.get_queue_size()
        fill_ratio = self._action_queue.get_fill_ratio()
        if fill_ratio <= self._config.chunk_size_threshold:
            return True

        steps_during_inference = self._latency_estimator.get_steps_during_inference(self._config.control_fps)
        safety_steps = int(steps_during_inference * self._config.latency_safety_margin)
        return queue_size <= safety_steps

    def _compute_timestep_locked(self, timestamp: float) -> int:
        elapsed = max(0.0, timestamp - self._episode_start_time)
        self._current_timestep = int(elapsed / self._config.environment_dt)
        return self._current_timestep

    def _resolve_timestep_locked(self, timestamp: float, timestep: Optional[int]) -> int:
        if timestep is None:
            return self._compute_timestep_locked(timestamp)
        self._current_timestep = int(timestep)
        return self._current_timestep

    def _resolve_control_time_locked(
        self,
        timestamp: Optional[float],
        timestep: Optional[int],
    ) -> tuple[float, int]:
        if timestamp is None and timestep is None:
            resolved_timestamp = time.monotonic()
            return resolved_timestamp, self._compute_timestep_locked(resolved_timestamp)

        if timestamp is None:
            resolved_timestep = int(timestep)
            resolved_timestamp = self._episode_start_time + resolved_timestep * self._config.environment_dt
            self._current_timestep = resolved_timestep
            return resolved_timestamp, resolved_timestep

        resolved_timestamp = float(timestamp)
        if timestep is None:
            return resolved_timestamp, self._compute_timestep_locked(resolved_timestamp)

        resolved_timestep = int(timestep)
        self._current_timestep = resolved_timestep
        return resolved_timestamp, resolved_timestep

    def _update_latency_metrics_locked(self, elapsed: float) -> None:
        if self._latency_estimator is not None:
            self._latency_estimator.update(elapsed)
        elapsed_ms = elapsed * 1000.0
        self._last_inference_ms = elapsed_ms
        self._max_inference_ms = elapsed_ms if self._max_inference_ms is None else max(self._max_inference_ms, elapsed_ms)

    def _record_error_locked(self, message: str) -> None:
        self._last_error = message
        self._error_count += 1
        self._consecutive_error_count += 1
        self._record_trace_locked("Runtime", "Error", message=message)

    def _set_error_locked(self, message: str) -> None:
        self._record_error_locked(message)
        self._state = AsyncRuntimeState.ERROR

    def _record_trace(self, source: str, event: str, **details: object) -> None:
        self._trace_recorder.record(source, event, **details)

    def _record_trace_locked(self, source: str, event: str, **details: object) -> None:
        self._trace_recorder.record(source, event, **details)

    def _require_loaded_locked(self) -> None:
        if self._sdk is None or self._metadata is None:
            raise RuntimeError("Async inference runtime has no loaded policy. Call load_policy() first.")

    def _require_running_locked(self) -> None:
        self._require_loaded_locked()
        if self._state != AsyncRuntimeState.RUNNING:
            raise RuntimeError(f"Async inference runtime is not running (state={self._state.value}).")

    def _queue_size_locked(self) -> int:
        return 0 if self._action_queue is None else self._action_queue.get_queue_size()

    def _fill_ratio_locked(self) -> float:
        return 0.0 if self._action_queue is None else self._action_queue.get_fill_ratio()

    def _latency_locked(self) -> float:
        return 0.0 if self._latency_estimator is None else self._latency_estimator.get_value()

    def _action_dim_locked(self) -> int:
        return 0 if self._metadata is None else int(self._metadata.action_dim)

    def _chunk_size_locked(self) -> int:
        return 1 if self._metadata is None else max(1, int(self._metadata.n_action_steps))


_GLOBAL_RUNTIME: Optional[AsyncInferenceRuntime] = None
_GLOBAL_RUNTIME_LOCK = threading.Lock()


def get_global_async_runtime() -> AsyncInferenceRuntime:
    """Return the process-local global async inference runtime."""
    global _GLOBAL_RUNTIME
    with _GLOBAL_RUNTIME_LOCK:
        if _GLOBAL_RUNTIME is None or _GLOBAL_RUNTIME.get_status().state == AsyncRuntimeState.CLOSED.value:
            _GLOBAL_RUNTIME = AsyncInferenceRuntime()
        return _GLOBAL_RUNTIME


def load_async_policy(*args: object, **kwargs: object) -> PolicyMetadata:
    """Convenience wrapper around the global runtime's ``load_policy``."""
    return get_global_async_runtime().load_policy(*args, **kwargs)


def start_async_runtime() -> None:
    """Start the global async inference runtime."""
    get_global_async_runtime().start()


def stop_async_runtime() -> None:
    """Stop the global async inference runtime."""
    get_global_async_runtime().stop()


def async_step(*args: object, **kwargs: object) -> AsyncStepResult:
    """Run one global async runtime control step."""
    return get_global_async_runtime().step(*args, **kwargs)


def get_async_status() -> AsyncRuntimeStatus:
    """Return the global async runtime status."""
    return get_global_async_runtime().get_status()


def _normalize_config(config: Optional[AsyncInferenceConfig | SmoothingConfig]) -> AsyncInferenceConfig:
    if config is None:
        normalized = AsyncInferenceConfig()
        normalized.validate()
        return normalized
    if isinstance(config, AsyncInferenceConfig):
        config.validate()
        return config
    if isinstance(config, SmoothingConfig):
        normalized = AsyncInferenceConfig(
            control_fps=config.control_fps,
            chunk_size_threshold=config.chunk_size_threshold,
            action_chunk_size=config.action_chunk_size,
            n_action_steps=config.n_action_steps,
            aggregate_fn_name=config.aggregate_fn_name,
            obs_queue_maxsize=config.obs_queue_maxsize,
            fallback_mode=config.fallback_mode,
            latency_ema_alpha=config.latency_ema_alpha,
            latency_safety_margin=config.latency_safety_margin,
            enable_gripper_clamping=config.enable_gripper_clamping,
            gripper_max_velocity=config.gripper_max_velocity,
            enable_temporal_ensemble=config.enable_temporal_ensemble,
            temporal_ensemble_coeff=config.temporal_ensemble_coeff,
            enable_rtc=config.enable_rtc,
            rtc_prefix_attention_schedule=config.rtc_prefix_attention_schedule,
            rtc_max_guidance_weight=config.rtc_max_guidance_weight,
            rtc_execution_horizon=config.rtc_execution_horizon,
            rtc_inference_delay_steps=config.rtc_inference_delay_steps,
            rtc_debug=config.rtc_debug,
            rtc_debug_maxlen=config.rtc_debug_maxlen,
        )
        normalized.validate()
        return normalized
    raise TypeError("config must be an AsyncInferenceConfig, SmoothingConfig, or None")


__all__ = [
    "AsyncInferenceConfig",
    "AsyncInferenceRuntime",
    "AsyncRuntimeState",
    "AsyncRuntimeStatus",
    "AsyncStepResult",
    "QueueSnapshotEntry",
    "async_step",
    "get_async_status",
    "get_global_async_runtime",
    "load_async_policy",
    "start_async_runtime",
    "stop_async_runtime",
]
