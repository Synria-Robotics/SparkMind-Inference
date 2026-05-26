"""Factory helpers for creating inference policies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .base import SmoothingConfig

if TYPE_CHECKING:
    from .base import BaseInferenceEngine

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = ("act", "smolvla", "pi0", "pi05")
MODEL_TYPE_ALIASES = {
    "act": "act",
    "smolvla": "smolvla",
    "smol_vla": "smolvla",
    "smol-vla": "smolvla",
    "pi0": "pi0",
    "pi_0": "pi0",
    "pi-0": "pi0",
    "pi05": "pi05",
    "pi0.5": "pi05",
    "pi0_5": "pi05",
    "pi0-5": "pi05",
    "pi_05": "pi05",
    "pi-05": "pi05",
}


def normalize_model_type(model_type: str) -> str:
    """Normalize a public model/algorithm type into an SDK model type."""
    key = str(model_type).strip().lower().replace(" ", "_")
    try:
        return MODEL_TYPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported: {SUPPORTED_MODEL_TYPES}"
        ) from exc


def create_engine(
    model_type: str,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    strict_device: bool = False,
) -> "BaseInferenceEngine":
    """
    Create an inference engine by model type.

    Args:
        model_type: "act", "smolvla", "pi0", or "pi05"
        device: Requested torch device string
        smoothing_config: Optional smoothing configuration
        strict_device: If True, fail instead of silently falling back
    """
    if smoothing_config is None:
        smoothing_config = SmoothingConfig(aggregate_fn_name="latest_only")

    normalized = normalize_model_type(model_type)
    if normalized == "act":
        from .policy.act import ACTInferenceEngine

        return ACTInferenceEngine(
            device=device,
            smoothing_config=smoothing_config,
            strict_device=strict_device,
        )
    if normalized == "smolvla":
        _preload_lerobot_for_vla()
        from .policy.smolvla import SmolVLAInferenceEngine

        return SmolVLAInferenceEngine(
            device=device,
            smoothing_config=smoothing_config,
            strict_device=strict_device,
        )
    if normalized == "pi0":
        _preload_lerobot_for_vla()
        from .policy.pi0 import PI0InferenceEngine

        return PI0InferenceEngine(
            device=device,
            smoothing_config=smoothing_config,
            strict_device=strict_device,
        )
    if normalized == "pi05":
        _preload_lerobot_for_vla()
        from .policy.pi05 import PI05InferenceEngine

        return PI05InferenceEngine(
            device=device,
            smoothing_config=smoothing_config,
            strict_device=strict_device,
        )
    raise ValueError(f"Unknown model type: {model_type}. Supported: {SUPPORTED_MODEL_TYPES}")


def create_inference_engine(
    model_type: str,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    strict_device: bool = False,
) -> "BaseInferenceEngine":
    """Backward-compatible alias for create_engine()."""
    return create_engine(
        model_type=model_type,
        device=device,
        smoothing_config=smoothing_config,
        strict_device=strict_device,
    )


def _preload_lerobot_for_vla() -> None:
    """Import LeRobot before SparkMind VLA modules so policy registries are ready."""
    try:
        import lerobot  # noqa: F401
    except ImportError:
        return
    except Exception as exc:
        logger.warning("Failed to preload installed lerobot package: %s", exc)
        return

    try:
        import lerobot.policies.rtc  # noqa: F401
    except Exception as exc:
        logger.debug("Failed to preload lerobot RTC policies, continuing: %s", exc)


__all__ = [
    "MODEL_TYPE_ALIASES",
    "SUPPORTED_MODEL_TYPES",
    "create_engine",
    "create_inference_engine",
    "normalize_model_type",
]
