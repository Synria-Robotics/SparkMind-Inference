"""
ACT (Action Chunking Transformer) Inference Engine.

Implements action queue based inference following LeRobot pattern:
- Predicts chunk_size actions at once
- Maintains action queue for smooth execution
- Optional synchronous temporal ensembling
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from ..base import ACTTemporalEnsembler, BaseInferenceEngine, SmoothingConfig
from ..device import resolve_torch_device
from ..runtime import format_optional_dependency_error
from .gripper_scale import feature_gripper_stats_are_unit_scaled

logger = logging.getLogger(__name__)

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None


LEGACY_CHECKPOINT_FILES = ("inference_config.yaml", "model.pth", "stats.json")
PRETRAINED_CHECKPOINT_FILES = ("config.json", "model.safetensors")
PRETRAINED_SUBDIR_NAME = "pretrained_model"
PREPROCESSOR_CONFIG_FILENAME = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG_FILENAME = "policy_postprocessor.json"


def _has_required_files(path: Path, filenames: tuple[str, ...]) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in filenames)


def _is_legacy_act_checkpoint(path: Path) -> bool:
    return _has_required_files(path, LEGACY_CHECKPOINT_FILES)


def _is_pretrained_act_dir(path: Path) -> bool:
    return (
        _has_required_files(path, PRETRAINED_CHECKPOINT_FILES)
        and (
            (path / PREPROCESSOR_CONFIG_FILENAME).is_file()
            or (path / POSTPROCESSOR_CONFIG_FILENAME).is_file()
        )
    )


def _resolve_act_checkpoint_dir(checkpoint_dir: str) -> Path:
    path = Path(checkpoint_dir)
    if _is_pretrained_act_dir(path):
        return path

    pretrained_dir = path / PRETRAINED_SUBDIR_NAME
    if _is_pretrained_act_dir(pretrained_dir):
        return pretrained_dir

    return path


def _convert_pretrained_act_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    input_features = config_dict.get("input_features") or {}
    output_features = config_dict.get("output_features") or {}

    robot_state_feature = None
    env_state_feature = None
    image_features = []

    for key, feature in input_features.items():
        feature_type = str(feature.get("type", "")).upper()
        if key == "observation.state" and feature_type == "STATE":
            robot_state_feature = feature
        elif feature_type == "VISUAL":
            image_features.append(key)
        elif feature_type == "ENV":
            env_state_feature = feature

    action_feature = None
    for key, feature in output_features.items():
        feature_type = str(feature.get("type", "")).upper()
        if key == "action" or feature_type == "ACTION":
            action_feature = feature
            break

    if robot_state_feature is None:
        raise ValueError("导出模型缺少 observation.state 输入定义")
    if action_feature is None:
        raise ValueError("导出模型缺少 action 输出定义")

    inference_config = dict(config_dict)
    inference_config["robot_state_feature"] = robot_state_feature
    inference_config["env_state_feature"] = env_state_feature
    inference_config["action_feature"] = action_feature
    inference_config["image_features"] = image_features

    # Inference loads full checkpoint weights immediately after init.
    # Avoid any torchvision weight download during model construction.
    inference_config["pretrained_backbone_weights"] = None

    return inference_config


def _extract_stats_from_safetensors(state_path: Path) -> Dict[str, Dict[str, Any]]:
    if load_safetensors_file is None:
        raise RuntimeError("缺少 safetensors 依赖，无法读取导出模型统计")

    state_dict = load_safetensors_file(str(state_path), device="cpu")
    stats: Dict[str, Dict[str, Any]] = {}

    for key, value in state_dict.items():
        if not (key.endswith(".mean") or key.endswith(".std")):
            continue
        feature_name, stat_name = key.rsplit(".", 1)
        stats.setdefault(feature_name, {})[stat_name] = value.cpu().tolist()

    return stats


def _load_pretrained_act_stats(checkpoint_path: Path) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}

    for config_name in (PREPROCESSOR_CONFIG_FILENAME, POSTPROCESSOR_CONFIG_FILENAME):
        config_path = checkpoint_path / config_name
        if not config_path.is_file():
            continue

        with open(config_path, "r") as f:
            processor_config = json.load(f)

        for step in processor_config.get("steps", []):
            state_file = step.get("state_file")
            if not state_file:
                continue

            state_path = checkpoint_path / state_file
            if not state_path.is_file():
                raise FileNotFoundError(f"处理器状态文件不存在: {state_path}")

            step_stats = _extract_stats_from_safetensors(state_path)
            for feature_name, feature_stats in step_stats.items():
                stats.setdefault(feature_name, {}).update(feature_stats)

    if not stats:
        raise FileNotFoundError("导出模型缺少可用的预处理/后处理统计")

    return stats


def _load_act_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.is_file():
        if load_safetensors_file is None:
            raise RuntimeError("缺少 safetensors 依赖，无法读取 model.safetensors")

        try:
            state_dict = load_safetensors_file(str(safetensors_path), device=str(device))
        except Exception:
            state_dict = load_safetensors_file(str(safetensors_path), device="cpu")
        if any(key.startswith("model.") for key in state_dict):
            state_dict = {
                key.removeprefix("model."): value
                for key, value in state_dict.items()
            }
        return state_dict

    model_path = checkpoint_path / "model.pth"
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)

# Try to import SparkMind ACT model
ACT_AVAILABLE = False
ACT_IMPORT_ERROR: Exception | None = None
try:
    from omegaconf import OmegaConf
    from sparkmind.learning.IL.models.act_model import ACTModel
    ACT_AVAILABLE = True
    logger.info("SparkMind ACT model loaded successfully")
except Exception as e:
    ACT_IMPORT_ERROR = e
    logger.warning(f"SparkMind ACT model not available: {e}")


class ACTInferenceEngine(BaseInferenceEngine):
    """
    ACT (Action Chunking Transformer) inference engine.
    
    Implements action queue based inference following LeRobot pattern:
    - Predicts chunk_size actions at once
    - Maintains action queue for smooth execution
    - Supports synchronous temporal ensembling
    """
    
    def __init__(
        self,
        device: str = "cuda:0",
        smoothing_config: Optional[SmoothingConfig] = None,
        strict_device: bool = False,
    ):
        super().__init__(smoothing_config)
        self.model_type = "act"

        device_selection = resolve_torch_device(device, strict=strict_device)
        self.requested_device = device_selection.requested
        self.actual_device = device_selection.actual
        self.device_warning = device_selection.warning
        if self.device_warning:
            logger.warning(self.device_warning)
        self.device = torch.device(self.actual_device)
        
        self.model: Optional[ACTModel] = None
        self.config: Optional[Any] = None
        self.stats: Optional[Dict] = None
        self.loaded_n_action_steps: int = 1
        self._state_gripper_stats_unit_scaled = True
        self._action_gripper_stats_unit_scaled = True
        
        # Camera role mapping: image_feature -> camera_role
        # e.g., "observation.images.cam_head" -> "head"
        self._camera_key_to_role: Dict[str, str] = {}
        self._role_to_camera_key: Dict[str, str] = {}

    @staticmethod
    def validate_checkpoint(checkpoint_dir: str) -> Tuple[bool, str]:
        """Validate that checkpoint directory contains a supported ACT checkpoint format."""
        raw_path = Path(checkpoint_dir)
        if not raw_path.exists():
            return False, f"Checkpoint目录不存在: {checkpoint_dir}"

        resolved_path = _resolve_act_checkpoint_dir(checkpoint_dir)
        if _is_legacy_act_checkpoint(resolved_path) or _is_pretrained_act_dir(resolved_path):
            return True, ""

        legacy_missing = [name for name in LEGACY_CHECKPOINT_FILES if not (resolved_path / name).is_file()]
        exported_missing = [name for name in PRETRAINED_CHECKPOINT_FILES if not (resolved_path / name).is_file()]
        processor_missing = [
            name
            for name in (PREPROCESSOR_CONFIG_FILENAME, POSTPROCESSOR_CONFIG_FILENAME)
            if not (resolved_path / name).is_file()
        ]
        return (
            False,
            "ACT模型目录格式不受支持。"
            f" 旧格式需要: {', '.join(LEGACY_CHECKPOINT_FILES)}"
            f"；导出格式需要: {', '.join(PRETRAINED_CHECKPOINT_FILES)}"
            f" + 至少一个processor配置({PREPROCESSOR_CONFIG_FILENAME}/{POSTPROCESSOR_CONFIG_FILENAME})"
            f"。当前缺少 旧格式文件: {', '.join(legacy_missing) or '无'}"
            f"；导出格式文件: {', '.join(exported_missing) or '无'}"
            f"；processor配置: {', '.join(processor_missing) or '无'}",
        )
        
    def load(self, checkpoint_dir: str) -> Tuple[bool, str]:
        """Load ACT model from checkpoint."""
        if not ACT_AVAILABLE:
            return False, format_optional_dependency_error(
                "ACT 模型所需的 SparkMind / omegaconf",
                ACT_IMPORT_ERROR,
                min_python=(3, 12),
                install_hint=(
                    "如果你使用本地 SparkMind checkout，请用 Python 3.12+ 重建虚拟环境后执行 `uv pip install -e third_party/SparkMind -i https://pypi.tuna.tsinghua.edu.cn/simple`。"
                ),
            )
        
        # Validate
        valid, error = self.validate_checkpoint(checkpoint_dir)
        if not valid:
            return False, error
        
        checkpoint_path = _resolve_act_checkpoint_dir(checkpoint_dir)
        
        try:
            if _is_legacy_act_checkpoint(checkpoint_path):
                config_path = checkpoint_path / "inference_config.yaml"
                with open(config_path, "r") as f:
                    config_dict = yaml.safe_load(f)

                stats_path = checkpoint_path / "stats.json"
                with open(stats_path, "r") as f:
                    self.stats = json.load(f)
            else:
                config_path = checkpoint_path / "config.json"
                with open(config_path, "r") as f:
                    pretrained_config = json.load(f)

                config_dict = _convert_pretrained_act_config(pretrained_config)
                self.stats = _load_pretrained_act_stats(checkpoint_path)

            self._state_gripper_stats_unit_scaled = feature_gripper_stats_are_unit_scaled(
                self.stats,
                "observation.state",
            )
            self._action_gripper_stats_unit_scaled = feature_gripper_stats_are_unit_scaled(
                self.stats,
                "action",
            )
            self._apply_action_chunk_overrides(config_dict)
            self.config = OmegaConf.create(config_dict)
            
            # Parse camera requirements from image_features
            image_features = config_dict.get("image_features", [])
            self.required_cameras = []
            self._camera_key_to_role = {}
            self._role_to_camera_key = {}
            
            for key in image_features:
                # Extract role from key like "observation.images.cam_head" -> "head"
                # or "observation.images.cam_wrist" -> "wrist"
                if key.startswith("observation.images."):
                    suffix = key.replace("observation.images.", "")
                    # Handle cam_head, cam_wrist naming
                    if suffix.startswith("cam_"):
                        role = suffix[4:]  # Remove "cam_" prefix
                    else:
                        role = suffix
                    self.required_cameras.append(role)
                    self._camera_key_to_role[key] = role
                    self._role_to_camera_key[role] = key
            
            # Parse dimensions
            state_shape = config_dict.get("robot_state_feature", {}).get("shape", [7])
            action_shape = config_dict.get("action_feature", {}).get("shape", [7])
            self.state_dim = state_shape[0] if state_shape else 7
            self.action_dim = action_shape[0] if action_shape else 7
            self.chunk_size = config_dict.get("chunk_size", 100)
            self.loaded_n_action_steps = int(config_dict.get("n_action_steps", self.chunk_size))
            self.n_action_steps = self.loaded_n_action_steps

            # ACT real-robot rollouts in this codebase historically execute the full predicted
            # chunk over multiple control steps. Exported LeRobot configs commonly set
            # n_action_steps=1 for their policy wrapper API, but using only one step here
            # makes the queue degenerate and causes frequent fallback/re-query on real robots.
            if (
                self.smoothing_config.n_action_steps is None
                and self.chunk_size > 1
                and self.n_action_steps <= 1
            ):
                logger.warning(
                    "ACT checkpoint reports n_action_steps=%s with chunk_size=%s; "
                    "overriding execution to consume the full chunk for real-robot control.",
                    self.loaded_n_action_steps,
                    self.chunk_size,
                )
                self.n_action_steps = self.chunk_size
            
            # Initialize model
            self.model = ACTModel(self.config)
            self.model.to(self.device)
            self.model.eval()
            
            # Load weights
            state_dict = _load_act_state_dict(checkpoint_path, self.device)
            self.model.load_state_dict(state_dict)
            
            logger.info(f"ACT model loaded from {checkpoint_dir}")
            logger.info(f"Required cameras: {self.required_cameras}")
            logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
            logger.info(f"Chunk size: {self.chunk_size}, N action steps: {self.n_action_steps}")
            logger.info(
                "ACT gripper stats scale: state=%s action=%s",
                "[0,1]" if self._state_gripper_stats_unit_scaled else "robot-space",
                "[0,1]" if self._action_gripper_stats_unit_scaled else "robot-space",
            )
            
            self.is_loaded = True
            
            # Initialize all components (queue, latency estimator, gripper smoother)
            self._init_components()
            if self.smoothing_config.enable_temporal_ensemble:
                self._temporal_ensembler = ACTTemporalEnsembler(
                    temporal_ensemble_coeff=self.smoothing_config.temporal_ensemble_coeff,
                    chunk_size=self.n_action_steps,
                )
            self.reset()
            
            return True, ""
            
        except Exception as e:
            logger.exception("Failed to load ACT model")
            return False, f"模型加载失败: {str(e)}"
    
    def _preprocess_images(self, images: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Preprocess images for model input.
        
        Args:
            images: Dict of {role: BGR numpy array (H, W, 3)}
            
        Returns:
            Dict of {camera_key: normalized tensor (1, 3, H, W)}
        """
        processed = {}
        
        if self.stats is None:
            return processed
        
        for role, img_bgr in images.items():
            camera_key = self._role_to_camera_key.get(role)
            if camera_key is None:
                continue
            
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # HWC -> CHW, normalize to [0, 1]
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            
            # Apply ImageNet-style normalization from stats
            if camera_key in self.stats:
                cam_stats = self.stats[camera_key]
                # Stats are stored as [[[mean_r]], [[mean_g]], [[mean_b]]]
                mean = torch.tensor([
                    cam_stats["mean"][0][0][0],
                    cam_stats["mean"][1][0][0],
                    cam_stats["mean"][2][0][0]
                ]).view(3, 1, 1)
                std = torch.tensor([
                    cam_stats["std"][0][0][0],
                    cam_stats["std"][1][0][0],
                    cam_stats["std"][2][0][0]
                ]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            processed[camera_key] = img_tensor
        
        return processed
    
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Preprocess robot state for model input.
        
        Args:
            state: Robot state array (state_dim,) with gripper in [0, 1000]
            
        Returns:
            Normalized state tensor (1, state_dim)
        """
        state = state.copy()  # Don't modify original
        
        # Exported checkpoints may store gripper stats in [0, 1] or robot-space.
        if len(state) >= 7 and self._state_gripper_stats_unit_scaled:
            state[-1] = state[-1] / 1000.0
        
        state_tensor = torch.from_numpy(state).float()
        
        # Normalize using stats
        if self.stats is not None and "observation.state" in self.stats:
            stats = self.stats["observation.state"]
            mean = torch.tensor(stats["mean"])
            std = torch.tensor(stats["std"])
            state_tensor = (state_tensor - mean) / (std + 1e-6)
        
        return state_tensor.unsqueeze(0).to(self.device)
    
    def _postprocess_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess (unnormalize) action tensor.
        
        Args:
            action_tensor: Normalized action tensor (action_dim,)
            
        Returns:
            Unnormalized action array (action_dim,) with gripper scaled to [0, 1000]
        """
        action = action_tensor.cpu()
        
        # Unnormalize using stats
        if self.stats is not None and "action" in self.stats:
            stats = self.stats["action"]
            mean = torch.tensor(stats["mean"])
            std = torch.tensor(stats["std"])
            action = action * std + mean
        
        action = action.numpy()
        
        # Return robot-space actions. Only scale when checkpoint stats are unit gripper values.
        if len(action) >= 7 and self._action_gripper_stats_unit_scaled:
            action[-1] = action[-1] * 1000.0
        
        return action
    
    @torch.no_grad()
    def _predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Internal method to predict a chunk of actions.
        
        Returns:
            Action chunk (n_action_steps, action_dim) - unnormalized
        """
        if self.model is None or self.config is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess inputs
        processed_images = self._preprocess_images(images)
        processed_state = self._preprocess_state(state)
        
        # Build batch for model
        try:
            from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
        except ImportError:
            OBS_IMAGES = "observation.images"
            OBS_STATE = "observation.state"
        
        # Get image features from config
        image_features = getattr(self.config, 'image_features', [])
        
        batch = {
            OBS_STATE: processed_state,
            OBS_IMAGES: [processed_images[key] for key in image_features 
                        if key in processed_images]
        }
        
        # Forward pass
        # ACTModel returns: (actions, (mu, log_sigma_x2))
        actions_chunk, _ = self.model(batch)  # (1, chunk_size, action_dim)
        
        # Take n_action_steps actions
        actions_normalized = actions_chunk[0, :self.n_action_steps]  # (n_action_steps, action_dim)
        
        # Postprocess all actions
        actions = np.stack([
            self._postprocess_action(actions_normalized[i])
            for i in range(actions_normalized.shape[0])
        ])
        
        return actions

    def select_action(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """Select one action, optionally using LeRobot-style ACT temporal ensembling."""
        if not self.smoothing_config.enable_temporal_ensemble:
            return super().select_action(images, state)

        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        if self._temporal_ensembler is None:
            self._temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff=self.smoothing_config.temporal_ensemble_coeff,
                chunk_size=self.n_action_steps,
            )

        current_time = time.time()
        elapsed_since_reset = max(0.0, current_time - self._episode_start_time)
        self._current_timestep = int(elapsed_since_reset / self.smoothing_config.environment_dt)

        start_time = time.perf_counter()
        action_chunk = self._predict_chunk(images, state)
        elapsed = time.perf_counter() - start_time
        if self._latency_estimator is not None:
            self._latency_estimator.update(elapsed)

        action = self._temporal_ensembler.update(action_chunk)

        if self._gripper_smoother is not None:
            action = self._gripper_smoother.smooth(action)

        return action
    
    def unload(self):
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        self.reset()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("ACT model unloaded")
