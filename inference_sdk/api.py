"""High-level public SDK API for synchronous action and action chunk inference."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .base import BaseInferenceEngine, SmoothingConfig
from .exceptions import (
    CheckpointNotFoundError,
    InferenceRuntimeError,
    InferenceSDKError,
    InvalidObservationError,
    MissingDependencyError,
    ModelLoadError,
    UnsupportedCheckpointFormatError,
)
from .factory import create_engine, normalize_model_type


@dataclass(frozen=True)
class Observation:
    """
    Observation passed into policy inference.

    Images are expected as BGR numpy arrays with shape (H, W, 3). Keys should
    normally be camera roles such as "head" or "wrist".
    """

    images: Mapping[str, np.ndarray]
    state: np.ndarray
    instruction: Optional[str] = None


@dataclass(frozen=True)
class PolicyMetadata:
    """Loaded policy metadata useful for API consumers."""

    model_type: str
    checkpoint_dir: str
    required_cameras: tuple[str, ...]
    state_dim: int
    action_dim: int
    chunk_size: int
    n_action_steps: int
    stability: str
    requested_device: Optional[str]
    actual_device: Optional[str]
    device_warning: str


class InferenceSDK:
    """
    High-level SDK facade for loading policies and synchronous inference.

    Typical use:
        sdk.load_policy("pi0", "/path/to/checkpoint", instruction="Pick up the object.")
        action_chunk = sdk.predict_action_chunk("pi0", images=images, state=state)
        action = sdk.predict_action("pi0", images=images, state=state)
    """

    def __init__(
        self,
        device: str = "cuda:0",
        smoothing_config: Optional[SmoothingConfig] = None,
        strict_device: bool = False,
    ):
        self.device = device
        self.smoothing_config = smoothing_config
        self.strict_device = strict_device
        self._policies: Dict[str, BaseInferenceEngine] = {}
        self._checkpoint_dirs: Dict[str, str] = {}

    def load_policy(
        self,
        algorithm_type: str,
        checkpoint_dir: str,
        *,
        instruction: Optional[str] = None,
        force_reload: bool = False,
    ) -> PolicyMetadata:
        """
        Load and cache one policy by algorithm type.

        Args:
            algorithm_type: "act", "smolvla", "smol-vla", "pi0", or "pi05"
            checkpoint_dir: Checkpoint directory for the selected policy
            instruction: Optional language instruction for VLA policies
            force_reload: Reload even if this policy/checkpoint is already cached
        """
        model_type = normalize_model_type(algorithm_type)
        checkpoint_dir = str(checkpoint_dir)

        existing = self._policies.get(model_type)
        if existing is not None and not force_reload:
            if self._checkpoint_dirs.get(model_type) == checkpoint_dir:
                self._apply_instruction(existing, instruction)
                return self._metadata_for(existing, checkpoint_dir)
            raise ModelLoadError(
                f"{model_type} policy is already loaded from "
                f"{self._checkpoint_dirs[model_type]}. Use force_reload=True to replace it."
            )

        if existing is not None:
            self._policies.pop(model_type, None)
            self._checkpoint_dirs.pop(model_type, None)
            existing.unload()

        if not Path(checkpoint_dir).exists():
            raise CheckpointNotFoundError(f"Checkpoint path does not exist: {checkpoint_dir}")

        policy: BaseInferenceEngine | None = None
        try:
            policy = create_engine(
                model_type=model_type,
                device=self.device,
                smoothing_config=self._new_smoothing_config(),
                strict_device=self.strict_device,
            )
            valid, validation_error = policy.validate_checkpoint(checkpoint_dir)
            if not valid:
                raise UnsupportedCheckpointFormatError(validation_error)

            ok, error = policy.load(checkpoint_dir)
            if not ok:
                raise _load_error_for_message(model_type, error)

            self._apply_instruction(policy, instruction)
        except InferenceSDKError:
            if policy is not None:
                try:
                    policy.unload()
                except Exception:
                    pass
            raise
        except Exception as exc:
            if policy is not None:
                try:
                    policy.unload()
                except Exception:
                    pass
            raise ModelLoadError(f"Failed to load {model_type} policy: {exc}") from exc

        self._policies[model_type] = policy
        self._checkpoint_dirs[model_type] = checkpoint_dir
        return self._metadata_for(policy, checkpoint_dir)

    def predict_action_chunk(
        self,
        algorithm_type: str,
        observation: Observation | Mapping[str, Any] | None = None,
        *,
        images: Optional[Mapping[str, np.ndarray]] = None,
        state: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Predict one raw action chunk for the selected loaded policy.

        Returns:
            Numpy array with shape (n_action_steps, action_dim).
        """
        policy, normalized_images, state_array = self._prepare_policy_inputs(
            algorithm_type,
            observation=observation,
            images=images,
            state=state,
            instruction=instruction,
        )
        try:
            return policy.predict_chunk(normalized_images, state_array)
        except InferenceSDKError:
            raise
        except Exception as exc:
            raise InferenceRuntimeError(f"Failed to predict {model_type} action chunk") from exc

    def predict_action(
        self,
        algorithm_type: str,
        observation: Observation | Mapping[str, Any] | None = None,
        *,
        images: Optional[Mapping[str, np.ndarray]] = None,
        state: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Run one synchronous control-loop step for the selected loaded policy.

        This executes on the caller thread and returns exactly one action for
        the current observation. For ACT, pass a SmoothingConfig with
        enable_temporal_ensemble=True to use the online synchronous temporal
        ensemble path.

        Returns:
            Numpy array with shape (action_dim,).
        """
        policy, normalized_images, state_array = self._prepare_policy_inputs(
            algorithm_type,
            observation=observation,
            images=images,
            state=state,
            instruction=instruction,
        )
        try:
            return policy.step(normalized_images, state_array)
        except InferenceSDKError:
            raise
        except Exception as exc:
            raise InferenceRuntimeError(f"Failed to predict {model_type} action") from exc

    def get_policy_metadata(self, algorithm_type: str) -> PolicyMetadata:
        """Return metadata for a loaded policy."""
        model_type = normalize_model_type(algorithm_type)
        policy = self._get_policy(model_type)
        return self._metadata_for(policy, self._checkpoint_dirs[model_type])

    def unload_policy(self, algorithm_type: str) -> None:
        """Unload one cached policy."""
        model_type = normalize_model_type(algorithm_type)
        policy = self._policies.pop(model_type, None)
        self._checkpoint_dirs.pop(model_type, None)
        if policy is not None:
            policy.unload()

    def close(self) -> None:
        """Unload all cached policies."""
        for model_type in list(self._policies):
            self.unload_policy(model_type)

    def __enter__(self) -> "InferenceSDK":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _get_policy(self, model_type: str) -> BaseInferenceEngine:
        try:
            return self._policies[model_type]
        except KeyError as exc:
            raise InferenceRuntimeError(
                f"{model_type} policy is not loaded. Call load_policy() first."
            ) from exc

    def _prepare_policy_inputs(
        self,
        algorithm_type: str,
        observation: Observation | Mapping[str, Any] | None = None,
        *,
        images: Optional[Mapping[str, np.ndarray]] = None,
        state: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
    ) -> tuple[BaseInferenceEngine, Dict[str, np.ndarray], np.ndarray]:
        model_type = normalize_model_type(algorithm_type)
        policy = self._get_policy(model_type)
        obs = _coerce_observation(
            observation=observation,
            images=images,
            state=state,
            instruction=instruction,
        )
        self._apply_instruction(policy, obs.instruction)

        normalized_images = _normalize_camera_aliases(policy, _coerce_images(obs.images))
        state_array = _coerce_state(obs.state)
        _validate_observation(policy, normalized_images, state_array)
        return policy, normalized_images, state_array

    def _new_smoothing_config(self) -> Optional[SmoothingConfig]:
        if self.smoothing_config is None:
            return None
        return replace(self.smoothing_config)

    @staticmethod
    def _apply_instruction(
        policy: BaseInferenceEngine,
        instruction: Optional[str],
    ) -> None:
        if instruction is None:
            return

        set_instruction = getattr(policy, "set_instruction", None)
        if not callable(set_instruction):
            raise InferenceRuntimeError(f"{policy.model_type} policy does not support language instructions")

        get_instruction = getattr(policy, "get_instruction", None)
        if callable(get_instruction) and get_instruction() == instruction:
            return

        if not bool(set_instruction(instruction)):
            raise InferenceRuntimeError(f"Failed to set instruction for {policy.model_type} policy")

    @staticmethod
    def _metadata_for(policy: BaseInferenceEngine, checkpoint_dir: str) -> PolicyMetadata:
        device_status = policy.get_device_status()
        return PolicyMetadata(
            model_type=policy.model_type,
            checkpoint_dir=checkpoint_dir,
            required_cameras=tuple(policy.get_required_cameras()),
            state_dim=int(policy.state_dim),
            action_dim=int(policy.action_dim),
            chunk_size=int(policy.chunk_size),
            n_action_steps=int(policy.n_action_steps),
            stability="experimental" if policy.model_type == "pi05" else "stable",
            requested_device=device_status.get("requested_device"),
            actual_device=device_status.get("actual_device"),
            device_warning=device_status.get("device_warning") or "",
        )


def predict_action_chunk(
    algorithm_type: str,
    checkpoint_dir: str,
    observation: Observation | Mapping[str, Any] | None = None,
    *,
    images: Optional[Mapping[str, np.ndarray]] = None,
    state: Optional[np.ndarray] = None,
    instruction: Optional[str] = None,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    strict_device: bool = False,
) -> np.ndarray:
    """
    Convenience one-shot API: load one policy, run one chunk inference, unload it.

    For repeated inference, prefer InferenceSDK so model weights stay loaded.
    """
    with InferenceSDK(
        device=device,
        smoothing_config=smoothing_config,
        strict_device=strict_device,
    ) as sdk:
        sdk.load_policy(algorithm_type, checkpoint_dir, instruction=instruction)
        return sdk.predict_action_chunk(
            algorithm_type,
            observation=observation,
            images=images,
            state=state,
            instruction=instruction,
        )


def predict_action(
    algorithm_type: str,
    checkpoint_dir: str,
    observation: Observation | Mapping[str, Any] | None = None,
    *,
    images: Optional[Mapping[str, np.ndarray]] = None,
    state: Optional[np.ndarray] = None,
    instruction: Optional[str] = None,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    strict_device: bool = False,
) -> np.ndarray:
    """
    Convenience one-shot API: load one policy, run one synchronous step, unload it.

    For real-time loops, prefer InferenceSDK so model weights stay loaded.
    """
    with InferenceSDK(
        device=device,
        smoothing_config=smoothing_config,
        strict_device=strict_device,
    ) as sdk:
        sdk.load_policy(algorithm_type, checkpoint_dir, instruction=instruction)
        return sdk.predict_action(
            algorithm_type,
            observation=observation,
            images=images,
            state=state,
            instruction=instruction,
        )


def _coerce_observation(
    *,
    observation: Observation | Mapping[str, Any] | None,
    images: Optional[Mapping[str, np.ndarray]],
    state: Optional[np.ndarray],
    instruction: Optional[str],
) -> Observation:
    if observation is None:
        if images is None or state is None:
            raise InvalidObservationError("Pass either observation=... or both images=... and state=...")
        return Observation(images=images, state=state, instruction=instruction)

    if images is not None or state is not None:
        raise InvalidObservationError("Do not mix observation=... with images=... or state=...")

    if isinstance(observation, Observation):
        obs_instruction = instruction if instruction is not None else observation.instruction
        return Observation(
            images=observation.images,
            state=observation.state,
            instruction=obs_instruction,
        )

    if isinstance(observation, Mapping):
        try:
            obs_images = observation["images"]
            obs_state = observation["state"]
        except KeyError as exc:
            raise InvalidObservationError("Observation mapping must contain 'images' and 'state'") from exc
        obs_instruction = instruction if instruction is not None else observation.get("instruction")
        return Observation(images=obs_images, state=obs_state, instruction=obs_instruction)

    try:
        obs_images = getattr(observation, "images")
        obs_state = getattr(observation, "state")
    except AttributeError as exc:
        raise InvalidObservationError("Observation must be an Observation, mapping, or object with images/state") from exc
    obs_instruction = instruction if instruction is not None else getattr(observation, "instruction", None)
    return Observation(images=obs_images, state=obs_state, instruction=obs_instruction)


def _coerce_images(images: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not isinstance(images, Mapping):
        raise InvalidObservationError("images must be a mapping of camera name to numpy array")

    result: Dict[str, np.ndarray] = {}
    for camera, image in images.items():
        image_array = np.asarray(image)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise InvalidObservationError(
                f"Image for camera '{camera}' must have shape (H, W, 3), got {image_array.shape}"
            )
        result[str(camera)] = image_array
    return result


def _coerce_state(state: np.ndarray) -> np.ndarray:
    if state is None:
        raise InvalidObservationError("state is required")
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.ndim != 1:
        raise InvalidObservationError(f"state must be a 1-D array, got shape {state_array.shape}")
    return state_array


def _normalize_camera_aliases(
    policy: BaseInferenceEngine,
    images: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    normalized = dict(images)
    for role in policy.get_required_cameras():
        if role in normalized:
            continue
        for alias in (
            f"cam_{role}",
            f"observation.images.{role}",
            f"observation.images.cam_{role}",
        ):
            if alias in normalized:
                normalized[role] = normalized[alias]
                break
    return normalized


def _validate_observation(
    policy: BaseInferenceEngine,
    images: Mapping[str, np.ndarray],
    state: np.ndarray,
) -> None:
    required_cameras = tuple(policy.get_required_cameras())
    missing_cameras = [camera for camera in required_cameras if camera not in images]
    if missing_cameras:
        raise InvalidObservationError(
            f"Missing required cameras for {policy.model_type}: {missing_cameras}. "
            f"Available cameras: {list(images.keys())}"
        )

    if policy.state_dim and state.shape[0] != policy.state_dim:
        raise InvalidObservationError(
            f"Invalid state dimension for {policy.model_type}: expected {policy.state_dim}, "
            f"got {state.shape[0]}"
        )


def _load_error_for_message(model_type: str, error: str) -> InferenceSDKError:
    message = f"Failed to load {model_type} policy: {error}"
    lower_error = error.lower()
    if "依赖不可用" in error or "missing dependency" in lower_error or "import" in lower_error:
        return MissingDependencyError(message)
    if "目录格式不受支持" in error or "unsupported" in lower_error:
        return UnsupportedCheckpointFormatError(message)
    if "不存在" in error or "not exist" in lower_error or "no such file" in lower_error:
        return CheckpointNotFoundError(message)
    return ModelLoadError(message)


__all__ = [
    "InferenceSDK",
    "Observation",
    "PolicyMetadata",
    "predict_action",
    "predict_action_chunk",
]
