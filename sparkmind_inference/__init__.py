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
__version__ = "1.0.0rc7"

from .exceptions import (  # noqa: E402
    CheckpointNotFoundError,
    DeviceUnavailableError,
    InvalidObservationError,
    InferenceRuntimeError,
    InferenceSDKError,
    MissingDependencyError,
    ModelLoadError,
    UnsupportedCheckpointFormatError,
)
from .runtime import configure_optional_import_paths  # noqa: E402

configure_optional_import_paths()

from .base import SmoothingConfig  # noqa: E402
from .api import InferenceSDK, Observation, PolicyMetadata, predict_action, predict_action_chunk  # noqa: E402
from .factory import (  # noqa: E402
    create_engine,
    create_inference_engine,
)

__all__ = [
    "CheckpointNotFoundError",
    "DeviceUnavailableError",
    "InvalidObservationError",
    "InferenceRuntimeError",
    "InferenceSDKError",
    "InferenceSDK",
    "MissingDependencyError",
    "ModelLoadError",
    "Observation",
    "PolicyMetadata",
    "SmoothingConfig",
    "UnsupportedCheckpointFormatError",
    "__version__",
    "create_engine",
    "create_inference_engine",
    "predict_action",
    "predict_action_chunk",
]
