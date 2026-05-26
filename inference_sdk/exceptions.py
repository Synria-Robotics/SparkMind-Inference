"""Exception types for the inference SDK."""


class InferenceSDKError(Exception):
    """Base exception for inference SDK failures."""


class CheckpointNotFoundError(InferenceSDKError):
    """Raised when a checkpoint path does not exist."""


class UnsupportedCheckpointFormatError(InferenceSDKError):
    """Raised when checkpoint files do not match a supported layout."""


class MissingDependencyError(InferenceSDKError):
    """Raised when optional model dependencies are unavailable."""


class DeviceUnavailableError(InferenceSDKError):
    """Raised when the requested inference device cannot be used."""


class InvalidObservationError(InferenceSDKError):
    """Raised when observation images, state, or instruction are invalid."""


class ModelLoadError(InferenceSDKError):
    """Raised when a model fails to load."""


class InferenceRuntimeError(InferenceSDKError):
    """Raised when inference execution fails at runtime."""


__all__ = [
    "InferenceSDKError",
    "CheckpointNotFoundError",
    "UnsupportedCheckpointFormatError",
    "MissingDependencyError",
    "DeviceUnavailableError",
    "InvalidObservationError",
    "ModelLoadError",
    "InferenceRuntimeError",
]
