"""Lazy access to model-specific policy implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_POLICY_EXPORTS = {
    "ACTInferenceEngine": ("inference_sdk.policy.act", "ACTInferenceEngine"),
    "ACT_AVAILABLE": ("inference_sdk.policy.act", "ACT_AVAILABLE"),
    "SmolVLAInferenceEngine": ("inference_sdk.policy.smolvla", "SmolVLAInferenceEngine"),
    "SMOLVLA_AVAILABLE": ("inference_sdk.policy.smolvla", "SMOLVLA_AVAILABLE"),
    "PI0InferenceEngine": ("inference_sdk.policy.pi0", "PI0InferenceEngine"),
    "PI0_AVAILABLE": ("inference_sdk.policy.pi0", "PI0_AVAILABLE"),
    "PI05InferenceEngine": ("inference_sdk.policy.pi05", "PI05InferenceEngine"),
    "PI05_AVAILABLE": ("inference_sdk.policy.pi05", "PI05_AVAILABLE"),
}

__all__ = sorted(_POLICY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _POLICY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
