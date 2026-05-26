"""Lazy access to model-specific policy implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_POLICY_EXPORTS = {
    "ACTInferenceEngine": ("sparkmind_inference.policy.act", "ACTInferenceEngine"),
    "ACT_AVAILABLE": ("sparkmind_inference.policy.act", "ACT_AVAILABLE"),
    "SmolVLAInferenceEngine": ("sparkmind_inference.policy.smolvla", "SmolVLAInferenceEngine"),
    "SMOLVLA_AVAILABLE": ("sparkmind_inference.policy.smolvla", "SMOLVLA_AVAILABLE"),
    "PI0InferenceEngine": ("sparkmind_inference.policy.pi0", "PI0InferenceEngine"),
    "PI0_AVAILABLE": ("sparkmind_inference.policy.pi0", "PI0_AVAILABLE"),
    "PI05InferenceEngine": ("sparkmind_inference.policy.pi05", "PI05InferenceEngine"),
    "PI05_AVAILABLE": ("sparkmind_inference.policy.pi05", "PI05_AVAILABLE"),
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
