"""
Inference SDK package for ACT, SmolVLA, PI0 and PI0.5 models.

Provides synchronous policy inference helpers:
- High-level InferenceSDK facade
- Direct action chunk prediction
- Synchronous control-loop step execution
- Optional ACT temporal ensembling and VLA RTC
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
    SmoothingConfig,
    TimedAction,
    TimestampedActionQueue,
    TraceEvent,
    TraceRecorder,
    get_aggregate_function,
)
from .api import InferenceSDK, Observation, PolicyMetadata, predict_action, predict_action_chunk  # noqa: E402
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
    "BaseInferenceEngine",
    "GripperSmoother",
    "LatencyEstimator",
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
    "SmoothingConfig",
    "SUPPORTED_MODEL_TYPES",
    "TimedAction",
    "TimestampedActionQueue",
    "TraceEvent",
    "TraceRecorder",
    "UnsupportedCheckpointFormatError",
    "__version__",
    "create_engine",
    "create_inference_engine",
    "get_aggregate_function",
    "normalize_model_type",
    "predict_action",
    "predict_action_chunk",
]
