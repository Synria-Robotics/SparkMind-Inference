"""
SmolVLA (Vision-Language-Action) Inference Engine.

Uses VLM (SmolVLM2-500M-Video-Instruct) combined with Flow Matching
action expert for robot control. Supports language-conditioned actions.

Implements action queue based inference:
- Predicts chunk_size actions at once using Flow Matching
- Maintains action queue for smooth execution
- Optional RTC for synchronous control-loop validation
"""

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..base import BaseInferenceEngine, SmoothingConfig
from ..device import resolve_torch_device
from ..runtime import format_optional_dependency_error, iter_model_search_roots, iter_unique_paths
from .gripper_scale import feature_gripper_stats_are_unit_scaled
from .rtc import make_rtc_config, make_rtc_processor

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
DEFAULT_SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
HF_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _has_required_files(path: Path, filenames: tuple[str, ...]) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in filenames)


def _is_legacy_smolvla_checkpoint(path: Path) -> bool:
    return _has_required_files(path, LEGACY_CHECKPOINT_FILES)


def _is_pretrained_smolvla_dir(path: Path) -> bool:
    return (
        _has_required_files(path, PRETRAINED_CHECKPOINT_FILES)
        and (
            (path / PREPROCESSOR_CONFIG_FILENAME).is_file()
            or (path / POSTPROCESSOR_CONFIG_FILENAME).is_file()
        )
    )


def _resolve_smolvla_checkpoint_dir(checkpoint_dir: str) -> Path:
    path = Path(checkpoint_dir)
    if _is_pretrained_smolvla_dir(path):
        return path

    pretrained_dir = path / PRETRAINED_SUBDIR_NAME
    if _is_pretrained_smolvla_dir(pretrained_dir):
        return pretrained_dir

    return path


def _convert_pretrained_smolvla_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
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


def _load_pretrained_smolvla_stats(checkpoint_path: Path) -> Dict[str, Dict[str, Any]]:
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


def _load_pretrained_required_image_features(checkpoint_path: Path) -> list[str]:
    config_path = checkpoint_path / PREPROCESSOR_CONFIG_FILENAME
    if not config_path.is_file():
        return []

    with open(config_path, "r") as f:
        processor_config = json.load(f)

    image_features: list[str] = []
    for step in processor_config.get("steps", []):
        if step.get("registry_name") != "rename_observations_processor":
            continue
        rename_map = step.get("config", {}).get("rename_map", {})
        for target_key in rename_map.values():
            if isinstance(target_key, str) and target_key.startswith("observation.images."):
                image_features.append(target_key)

    return image_features


def _stats_feature_dim(stats: Optional[Dict[str, Dict[str, Any]]], feature_name: str) -> Optional[int]:
    if not stats or feature_name not in stats:
        return None

    feature_stats = stats[feature_name]
    for stat_name in ("mean", "std", "min", "max"):
        values = feature_stats.get(stat_name)
        if values is None:
            continue
        if isinstance(values, list):
            return len(values)
        if hasattr(values, "shape") and values.shape:
            return int(values.shape[0])
    return None


def _load_smolvla_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
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


def _is_local_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _resolve_vlm_model_source(vlm_model_name: str, checkpoint_path: Path) -> str:
    env_override = os.environ.get("SMOLVLA_VLM_MODEL_PATH")
    candidate_paths = []

    if env_override:
        candidate_paths.append(Path(env_override).expanduser())

    configured_path = Path(vlm_model_name).expanduser()
    if configured_path.exists():
        candidate_paths.append(configured_path)

    repo_path = Path(*vlm_model_name.split("/"))
    candidate_paths.append(checkpoint_path / "vlm_model")
    for base_dir in iter_model_search_roots(checkpoint_path):
        candidate_paths.extend(
            [
                base_dir / repo_path,
                base_dir / repo_path.name,
            ]
        )

    for candidate in iter_unique_paths(candidate_paths):
        if _is_local_model_dir(candidate):
            logger.info("Using local SmolVLM assets from %s", candidate)
            return str(candidate)

    return vlm_model_name


@contextmanager
def _normalized_hf_proxy_env() -> Iterator[None]:
    """httpx rejects socks:// proxies; normalize them to socks5:// for HF clients."""
    original_values: Dict[str, str] = {}

    for key in HF_PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if isinstance(value, str) and value.startswith("socks://"):
            original_values[key] = value
            os.environ[key] = f"socks5://{value[len('socks://'):]}"
            logger.info("Normalized %s to socks5:// for Hugging Face access", key)

    try:
        yield
    finally:
        for key, value in original_values.items():
            os.environ[key] = value


def _format_smolvla_load_error(error: Exception, requested_vlm: str) -> str:
    message = str(error)

    if "Unknown scheme for proxy URL" in message and "socks://" in message:
        return (
            "模型加载失败: 检测到无效代理配置。当前 `ALL_PROXY/HTTP_PROXY` 使用了 `socks://`，"
            "请改成 `socks5://127.0.0.1:7897/`，或者临时取消 `ALL_PROXY` 后重启后端。"
        )

    if "Can't load the configuration of" in message:
        return (
            "模型加载失败: SmolVLA 基础视觉语言模型未就绪。当前需要加载 "
            f"`{requested_vlm}`。如果你要离线推理，请先下载该模型，例如:\n"
            "`source .venv/bin/activate && hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct "
            "--local-dir models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct`\n"
            "然后设置环境变量:\n"
            "`export SMOLVLA_VLM_MODEL_PATH=/absolute/path/to/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct`\n"
            "或者设置:\n"
            "`export INFERENCE_SDK_MODEL_ROOTS=/absolute/path/to/models`\n"
            "再重新启动后端。"
        )

    if "Operation not permitted" in message:
        return (
            "模型加载失败: 当前环境无法访问 Hugging Face。请检查网络/代理，"
            "或者先把 SmolVLM2 基础模型下载到本地，并通过 `SMOLVLA_VLM_MODEL_PATH` 指向它。"
        )

    return f"模型加载失败: {message}"

# Try to import SparkMind SmolVLA model
SMOLVLA_AVAILABLE = False
SMOLVLA_IMPORT_ERROR: Exception | None = None
try:
    from omegaconf import OmegaConf
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from sparkmind.learning.VLA.models.smolvla_model import VLAFlowMatching
    from transformers import AutoTokenizer
    SMOLVLA_AVAILABLE = True
    logger.info("SparkMind SmolVLA model loaded successfully")
except Exception as e:
    SMOLVLA_IMPORT_ERROR = e
    logger.warning(f"SparkMind SmolVLA model not available: {e}")


class SmolVLAInferenceEngine(BaseInferenceEngine):
    """
    SmolVLA (Vision-Language-Action) inference engine.
    
    Uses VLM (SmolVLM2-500M-Video-Instruct) combined with Flow Matching
    action expert for robot control. Supports language-conditioned actions.
    
    Implements action queue based inference:
    - Predicts chunk_size actions at once using Flow Matching
    - Maintains action queue for smooth execution
    - Optional RTC for synchronous control-loop validation
    """
    
    def __init__(
        self,
        device: str = "cuda:0",
        smoothing_config: Optional[SmoothingConfig] = None,
        strict_device: bool = False,
    ):
        super().__init__(smoothing_config)
        self.model_type = "smolvla"

        device_selection = resolve_torch_device(device, strict=strict_device)
        self.requested_device = device_selection.requested
        self.actual_device = device_selection.actual
        self.device_warning = device_selection.warning
        if self.device_warning:
            logger.warning(self.device_warning)
        self.device = torch.device(self.actual_device)
        
        self.model: Optional[Any] = None
        self.config: Optional[Any] = None
        self.config_dict: Optional[Dict] = None
        self.stats: Optional[Dict] = None
        self.tokenizer: Optional[Any] = None
        
        # Camera role mapping
        self._camera_key_to_role: Dict[str, str] = {}
        self._role_to_camera_key: Dict[str, str] = {}
        self._camera_alias_to_key: Dict[str, str] = {}
        
        # Language instruction (required for SmolVLA)
        self._instruction: str = "Pick up the object."
        self._instruction_tokens: Optional[torch.Tensor] = None
        self._instruction_attention_mask: Optional[torch.Tensor] = None
        
        # Image resize target for SmolVLA. If None, keep native resolution.
        self._image_resize: Optional[Tuple[int, int]] = (512, 512)
        self._rtc_prev_chunk_left_over: Optional[torch.Tensor] = None
        self._state_gripper_stats_unit_scaled = True
        self._action_gripper_stats_unit_scaled = True

    @staticmethod
    def validate_checkpoint(checkpoint_dir: str) -> Tuple[bool, str]:
        """Validate that checkpoint directory contains a supported SmolVLA checkpoint format."""
        raw_path = Path(checkpoint_dir)
        if not raw_path.exists():
            return False, f"Checkpoint目录不存在: {checkpoint_dir}"

        resolved_path = _resolve_smolvla_checkpoint_dir(checkpoint_dir)
        if _is_legacy_smolvla_checkpoint(resolved_path) or _is_pretrained_smolvla_dir(resolved_path):
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
            "SmolVLA模型目录格式不受支持。"
            f" 旧格式需要: {', '.join(LEGACY_CHECKPOINT_FILES)}"
            f"；导出格式需要: {', '.join(PRETRAINED_CHECKPOINT_FILES)}"
            f" + 至少一个processor配置({PREPROCESSOR_CONFIG_FILENAME}/{POSTPROCESSOR_CONFIG_FILENAME})"
            f"。当前缺少 旧格式文件: {', '.join(legacy_missing) or '无'}"
            f"；导出格式文件: {', '.join(exported_missing) or '无'}"
            f"；processor配置: {', '.join(processor_missing) or '无'}",
        )
    
    def load(self, checkpoint_dir: str) -> Tuple[bool, str]:
        """Load SmolVLA model from checkpoint."""
        if not SMOLVLA_AVAILABLE:
            return False, format_optional_dependency_error(
                "SmolVLA 模型所需的 SparkMind / transformers",
                SMOLVLA_IMPORT_ERROR,
                min_python=(3, 12),
                install_hint=(
                    "请先安装 SparkMind，例如 `uv pip install \"sparkmind[pi,libero]>=1.0.0\"` "
                    "或 `uv pip install -e \"third_party/SparkMind[pi,libero]\"`。"
                ),
            )
        
        # Validate
        valid, error = self.validate_checkpoint(checkpoint_dir)
        if not valid:
            return False, error
        
        checkpoint_path = _resolve_smolvla_checkpoint_dir(checkpoint_dir)
        vlm_model_source = DEFAULT_SMOLVLM_MODEL
        
        try:
            self._load_robot_io_metadata(checkpoint_path)

            required_image_features: list[str] = []
            if _is_legacy_smolvla_checkpoint(checkpoint_path):
                config_path = checkpoint_path / "inference_config.yaml"
                with open(config_path, "r") as f:
                    self.config_dict = yaml.safe_load(f)

                stats_path = checkpoint_path / "stats.json"
                with open(stats_path, "r") as f:
                    self.stats = json.load(f)
            else:
                config_path = checkpoint_path / "config.json"
                with open(config_path, "r") as f:
                    pretrained_config = json.load(f)

                self.config_dict = _convert_pretrained_smolvla_config(pretrained_config)
                self.stats = _load_pretrained_smolvla_stats(checkpoint_path)
                required_image_features = _load_pretrained_required_image_features(checkpoint_path)

            self._apply_action_chunk_overrides(self.config_dict)
            self._state_gripper_stats_unit_scaled = feature_gripper_stats_are_unit_scaled(
                self.stats,
                "observation.state",
            )
            self._action_gripper_stats_unit_scaled = feature_gripper_stats_are_unit_scaled(
                self.stats,
                "action",
            )
            
            # Parse camera requirements from image_features
            image_features = self.config_dict.get("image_features", [])
            required_features = required_image_features or image_features
            self.required_cameras = []
            self._camera_key_to_role = {}
            self._role_to_camera_key = {}
            self._camera_alias_to_key = {}
            
            for key in required_features:
                if key.startswith("observation.images."):
                    suffix = key.replace("observation.images.", "")
                    if suffix.startswith("cam_"):
                        role = suffix[4:]  # Remove "cam_" prefix
                    else:
                        role = suffix
                    if role not in self.required_cameras:
                        self.required_cameras.append(role)
                    self._camera_key_to_role[key] = role
                    self._role_to_camera_key[role] = key
                    self._camera_alias_to_key[role] = key
                    self._camera_alias_to_key[suffix] = key
                    self._camera_alias_to_key[key] = key
            
            # Parse dimensions
            state_shape = self.config_dict.get("robot_state_feature", {}).get("shape", [7])
            action_shape = self.config_dict.get("action_feature", {}).get("shape", [7])
            self.state_dim = state_shape[0] if state_shape else 7
            self.action_dim = action_shape[0] if action_shape else 7
            stats_state_dim = _stats_feature_dim(self.stats, "observation.state")
            if stats_state_dim is not None and stats_state_dim != self.state_dim:
                logger.warning(
                    "SmolVLA config state dim=%s but processor stats dim=%s; using stats dim",
                    self.state_dim,
                    stats_state_dim,
                )
                self.state_dim = stats_state_dim
            stats_action_dim = _stats_feature_dim(self.stats, "action")
            if stats_action_dim is not None and stats_action_dim != self.action_dim:
                logger.warning(
                    "SmolVLA config action dim=%s but processor stats dim=%s; using stats dim",
                    self.action_dim,
                    stats_action_dim,
                )
                self.action_dim = stats_action_dim
            self.chunk_size = self.config_dict.get("chunk_size", 50)
            self.n_action_steps = self.config_dict.get("n_action_steps", self.chunk_size)
            
            # Get VLM model name from config
            vlm_model_name = self.config_dict.get("vlm_model_name", DEFAULT_SMOLVLM_MODEL)
            vlm_model_source = _resolve_vlm_model_source(vlm_model_name, checkpoint_path)
            
            # Build SmolVLAConfig with only valid parameters from SmolVLAConfig dataclass
            valid_config_keys = [
                "n_obs_steps", "input_features", "output_features", "device", "use_amp", "use_peft",
                "push_to_hub", "repo_id", "private", "tags", "license", "pretrained_path",
                "chunk_size", "n_action_steps", "normalization_mapping",
                "max_state_dim", "max_action_dim", "resize_imgs_with_padding", "empty_cameras",
                "adapt_to_pi_aloha", "use_delta_joint_actions_aloha",
                "tokenizer_max_length", "num_steps", "use_cache",
                "freeze_vision_encoder", "train_expert_only", "train_state_proj",
                "optimizer_lr", "optimizer_betas", "optimizer_eps", "optimizer_weight_decay",
                "optimizer_grad_clip_norm", "scheduler_warmup_steps", "scheduler_decay_steps",
                "scheduler_decay_lr",
                "vlm_model_name", "add_image_special_tokens", "attention_mode",
                "prefix_length", "pad_language_to", "num_expert_layers", "num_vlm_layers",
                "self_attn_every_n_layers", "expert_width_multiplier",
                "min_period", "max_period", "rtc_config", "compile_model", "compile_mode"
            ]
            filtered_config = {k: v for k, v in self.config_dict.items() if k in valid_config_keys}
            filtered_config["load_vlm_weights"] = False  # Load from checkpoint instead
            filtered_config["vlm_model_name"] = vlm_model_source
            if self.smoothing_config.enable_rtc:
                filtered_config["rtc_config"] = make_rtc_config(self.smoothing_config)
            
            # Handle tuple conversion for resize_imgs_with_padding
            if "resize_imgs_with_padding" in filtered_config:
                val = filtered_config["resize_imgs_with_padding"]
                if isinstance(val, list):
                    filtered_config["resize_imgs_with_padding"] = tuple(val)
            
            self.config = SmolVLAConfig(**filtered_config)
            self._image_resize = self.config.resize_imgs_with_padding
            
            # Add rtc_config attribute if not present (required by VLAFlowMatching._rtc_enabled)
            if not hasattr(self.config, 'rtc_config'):
                self.config.rtc_config = None
            
            with _normalized_hf_proxy_env():
                # Initialize tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    vlm_model_source,
                    trust_remote_code=True,
                )
                
                # Initialize model
                self.model = VLAFlowMatching(
                    self.config,
                    rtc_processor=make_rtc_processor(self.smoothing_config),
                )

            self.model.to(self.device)
            self.model.eval()
            
            # Load weights
            state_dict = _load_smolvla_state_dict(checkpoint_path, self.device)
            self.model.load_state_dict(state_dict)
            
            logger.info(f"SmolVLA model loaded from {checkpoint_dir}")
            logger.info(f"Required cameras: {self.required_cameras}")
            logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
            logger.info(f"Chunk size: {self.chunk_size}, N action steps: {self.n_action_steps}")
            logger.info(f"VLM: {vlm_model_source}")
            logger.info(
                "SmolVLA gripper stats units: state=%s action=%s",
                "normalized" if self._state_gripper_stats_unit_scaled else "robot",
                "normalized" if self._action_gripper_stats_unit_scaled else "robot",
            )
            
            self.is_loaded = True
            
            # Initialize smoothing components
            self._init_components()
            self.reset()
            
            # Pre-tokenize default instruction
            self._tokenize_instruction(self._instruction)
            
            return True, ""
            
        except Exception as e:
            logger.exception("Failed to load SmolVLA model")
            return False, _format_smolvla_load_error(e, vlm_model_source)
    
    def set_instruction(self, instruction: str) -> bool:
        """
        Set the language instruction for SmolVLA.
        
        Args:
            instruction: Natural language instruction (e.g., "Pick up the apple")
            
        Returns:
            True if successful
        """
        if not self.is_loaded or self.tokenizer is None:
            logger.warning("Cannot set instruction: model not loaded")
            return False
        
        self._instruction = instruction
        self._tokenize_instruction(instruction)
        
        # Reset action queue when instruction changes
        if self._action_queue is not None:
            self._action_queue.reset()
        self._rtc_prev_chunk_left_over = None
        
        logger.info(f"Instruction set: {instruction}")
        return True

    def reset(self):
        super().reset()
        self._rtc_prev_chunk_left_over = None
    
    def get_instruction(self) -> str:
        """Get current instruction."""
        return self._instruction
    
    def _tokenize_instruction(self, instruction: str):
        """Tokenize instruction for model input."""
        if self.tokenizer is None:
            return

        instruction_text = instruction if instruction.endswith("\n") else f"{instruction}\n"
        tokenizer_max_length = self.config.tokenizer_max_length if self.config is not None else 64
        padding_mode = self.config.pad_language_to if self.config is not None else "max_length"
        self.tokenizer.padding_side = "right"

        tokens = self.tokenizer(
            instruction_text,
            return_tensors="pt",
            padding=padding_mode,
            max_length=tokenizer_max_length,
            truncation=True
        )
        self._instruction_tokens = tokens["input_ids"].to(self.device)
        # Convert attention mask to boolean as required by SmolVLA model
        self._instruction_attention_mask = tokens["attention_mask"].bool().to(self.device)
    
    def _resize_tensor_with_pad(self, img: torch.Tensor, target_h: int, target_w: int, pad_value: float = 0.0) -> torch.Tensor:
        if img.ndim != 4:
            raise ValueError(f"(b,c,h,w) expected, got {tuple(img.shape)}")

        cur_h, cur_w = img.shape[2:]
        ratio = max(cur_w / target_w, cur_h / target_h)
        resized_h = int(cur_h / ratio)
        resized_w = int(cur_w / ratio)
        resized = F.interpolate(img, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
        pad_h = max(0, int(target_h - resized_h))
        pad_w = max(0, int(target_w - resized_w))
        return F.pad(resized, (pad_w, 0, pad_h, 0), value=pad_value)
    
    def _preprocess_images(self, images: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Preprocess images for SmolVLA model input.
        
        SmolVLA uses different preprocessing from ACT:
        - Resize with padding to 512x512
        - Normalize to [-1, 1] range (NOT ImageNet stats)
        
        Args:
            images: Dict of {role: BGR numpy array (H, W, 3)}
            
        Returns:
            Dict of {camera_key: normalized tensor (1, 3, H, W)}
        """
        processed = {}
        
        for camera_alias, img_bgr in images.items():
            camera_key = self._camera_alias_to_key.get(camera_alias)
            if camera_key is None:
                continue
            
            # BGR -> RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # HWC -> CHW, normalize to [0, 1] then to [-1, 1]
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            if self._image_resize is not None:
                target_h, target_w = self._image_resize
                img_tensor = self._resize_tensor_with_pad(img_tensor, target_h, target_w, pad_value=0.0)
            img_tensor = img_tensor * 2.0 - 1.0  # [-1, 1] normalization

            img_tensor = img_tensor.to(self.device)
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
        state = state.copy()
        
        # Scale gripper only when checkpoint stats expect normalized gripper units.
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
        
        # Scale gripper only when checkpoint stats store normalized gripper units.
        if len(action) >= 7 and self._action_gripper_stats_unit_scaled:
            action[-1] = action[-1] * 1000.0
        
        return action
    
    @torch.no_grad()
    def _predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Internal method to predict a chunk of actions using Flow Matching.
        
        Returns:
            Action chunk (n_action_steps, action_dim) - unnormalized
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if self._instruction_tokens is None:
            raise RuntimeError("Instruction not set - call set_instruction() first")
        
        # Preprocess inputs
        processed_images = self._preprocess_images(images)
        processed_state = self._preprocess_state(state)
        
        # Get image features as list (order matters)
        image_features = self.config_dict.get("image_features", [])
        image_list = [processed_images[key] for key in image_features 
                     if key in processed_images]
        
        # Create image masks (all valid = True), one per camera
        image_masks = [torch.ones(1, dtype=torch.bool, device=self.device) 
                      for _ in image_list]
        
        # Pad state to max_state_dim (32)
        max_state_dim = self.config.max_state_dim
        padded_state = torch.zeros(1, max_state_dim, device=self.device)
        padded_state[0, :processed_state.shape[1]] = processed_state[0]
        
        # Sample actions using Flow Matching
        actions_chunk = self.model.sample_actions(
            images=image_list,
            img_masks=image_masks,
            lang_tokens=self._instruction_tokens,
            lang_masks=self._instruction_attention_mask,
            state=padded_state,
            **self._rtc_kwargs(),
        )  # (1, chunk_size, max_action_dim)
        
        self._update_rtc_left_over(actions_chunk)
        actions_normalized = actions_chunk[0, :self.n_action_steps, :self.action_dim]
        
        # Postprocess all actions
        actions = np.stack([
            self._postprocess_action(actions_normalized[i])
            for i in range(actions_normalized.shape[0])
        ])
        
        return actions

    def _rtc_kwargs(self) -> Dict[str, Any]:
        if not self.smoothing_config.enable_rtc:
            return {}
        return {
            "prev_chunk_left_over": self._rtc_prev_chunk_left_over,
            "inference_delay": int(self.smoothing_config.rtc_inference_delay_steps),
            "execution_horizon": int(self.smoothing_config.rtc_execution_horizon),
        }

    def _update_rtc_left_over(self, actions_chunk: torch.Tensor) -> None:
        if not self.smoothing_config.enable_rtc:
            return
        delay = max(0, int(self.smoothing_config.rtc_inference_delay_steps))
        chunk = actions_chunk.detach()
        self._rtc_prev_chunk_left_over = chunk[:, delay:].clone() if delay else chunk.clone()
    
    def unload(self):
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.is_loaded = False
        self.reset()
        
        self._instruction_tokens = None
        self._instruction_attention_mask = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("SmolVLA model unloaded")
