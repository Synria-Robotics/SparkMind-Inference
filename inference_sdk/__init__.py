"""
Inference SDK package for ACT, SmolVLA, PI0 and PI0.5 models.

Provides optimized inference with LeRobot-style async architecture:
- Timestamp-aligned action queue (skip expired actions)
- Latency-adaptive chunk threshold
- Observation queue maxsize=1 (always use latest frame)
- Aggregate functions for overlapping chunks
- Gripper velocity clamping
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
__version__ = "0.1.0"

# Import lerobot before SparkMind to avoid vendored dependency conflicts.
try:
    import lerobot  # noqa: F401
except ImportError:
    pass
except Exception as exc:
    logger.warning("Failed to preload installed lerobot package: %s", exc)
else:
    try:
        import lerobot.policies.rtc  # noqa: F401
    except Exception as exc:
        logger.debug("Failed to preload lerobot RTC policies, continuing: %s", exc)

from .exceptions import (  # noqa: E402
    CheckpointNotFoundError,
    DeviceUnavailableError,
    InferenceRuntimeError,
    InferenceSDKError,
    MissingDependencyError,
    ModelLoadError,
    UnsupportedCheckpointFormatError,
)
from .runtime import configure_optional_import_paths  # noqa: E402

configure_optional_import_paths()

from .base import (  # noqa: E402
    AGGREGATE_FUNCTIONS,
    ACTTemporalEnsembler,
    BaseInferenceEngine,
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
from .api import InferenceSDK, Observation, PolicyMetadata, predict_action, predict_action_chunk  # noqa: E402
from .async_runtime import (  # noqa: E402
    AsyncInferenceConfig,
    AsyncInferenceRuntime,
    AsyncRuntimeState,
    AsyncRuntimeStatus,
    AsyncStepResult,
    QueueSnapshotEntry,
    async_step,
    get_async_status,
    get_global_async_runtime,
    load_async_policy,
    start_async_runtime,
    stop_async_runtime,
)
from .policy import (  # noqa: E402
    ACT_AVAILABLE,
    PI0_AVAILABLE,
    PI05_AVAILABLE,
    SMOLVLA_AVAILABLE,
    ACTInferenceEngine,
    PI0InferenceEngine,
    PI05InferenceEngine,
    SmolVLAInferenceEngine,
)
from .factory import (  # noqa: E402
    SUPPORTED_MODEL_TYPES,
    create_engine,
    create_inference_engine,
    normalize_model_type,
)

__all__ = [
    "AGGREGATE_FUNCTIONS",
    "ACTInferenceEngine",
    "ACTTemporalEnsembler",
    "ACT_AVAILABLE",
    "AsyncInferenceConfig",
    "AsyncInferenceRuntime",
    "AsyncRuntimeState",
    "AsyncRuntimeStatus",
    "AsyncStepResult",
    "BaseInferenceEngine",
    "GripperSmoother",
    "LatencyEstimator",
    "ObservationQueue",
    "PI0InferenceEngine",
    "PI0_AVAILABLE",
    "PI05InferenceEngine",
    "PI05_AVAILABLE",
    "CheckpointNotFoundError",
    "DeviceUnavailableError",
    "InferenceRuntimeError",
    "InferenceSDKError",
    "InferenceSDK",
    "MissingDependencyError",
    "ModelLoadError",
    "Observation",
    "SmolVLAInferenceEngine",
    "SMOLVLA_AVAILABLE",
    "PolicyMetadata",
    "QueueSnapshotEntry",
    "SmoothingConfig",
    "SUPPORTED_MODEL_TYPES",
    "TimedAction",
    "TimedObservation",
    "TimestampedActionQueue",
    "TraceEvent",
    "TraceRecorder",
    "UnsupportedCheckpointFormatError",
    "__version__",
    "async_step",
    "create_engine",
    "create_inference_engine",
    "get_async_status",
    "get_aggregate_function",
    "get_global_async_runtime",
    "load_async_policy",
    "normalize_model_type",
    "predict_action",
    "predict_action_chunk",
    "start_async_runtime",
    "stop_async_runtime",
]
