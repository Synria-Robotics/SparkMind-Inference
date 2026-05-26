"""
PI0.5 (PI05) Vision-Language-Action inference engine.

This engine follows the SparkMind / LeRobot PI05 implementation. Compared with
PI0, PI05 encodes the robot state into the language prompt after normalization
and discretization, then samples action chunks with the PI05 flow-matching
policy.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import fields
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
from .rtc import make_rtc_config, make_rtc_processor

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from safetensors import safe_open as safe_open_safetensors
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    safe_open_safetensors = None
    load_safetensors_file = None


LEGACY_CHECKPOINT_FILES = ("inference_config.yaml", "model.pth", "stats.json")
PRETRAINED_CHECKPOINT_FILES = ("config.json", "model.safetensors")
PRETRAINED_SUBDIR_NAME = "pretrained_model"
PREPROCESSOR_CONFIG_FILENAME = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG_FILENAME = "policy_postprocessor.json"
DEFAULT_PI05_TOKENIZER = "google/paligemma-3b-pt-224"
PI05_TOKENIZER_FALLBACKS = {
    "google/paligemma-3b-pt-224": ("google/paligemma2-3b-mix-224",),
}
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


def _is_legacy_pi05_checkpoint(path: Path) -> bool:
    return _has_required_files(path, LEGACY_CHECKPOINT_FILES)


def _is_pretrained_pi05_dir(path: Path) -> bool:
    return (
        _has_required_files(path, PRETRAINED_CHECKPOINT_FILES)
        and (
            (path / PREPROCESSOR_CONFIG_FILENAME).is_file()
            or (path / POSTPROCESSOR_CONFIG_FILENAME).is_file()
        )
    )


def _resolve_pi05_checkpoint_dir(checkpoint_dir: str) -> Path:
    path = Path(checkpoint_dir)
    if _is_pretrained_pi05_dir(path):
        return path

    pretrained_dir = path / PRETRAINED_SUBDIR_NAME
    if _is_pretrained_pi05_dir(pretrained_dir):
        return pretrained_dir

    return path


def _convert_pretrained_pi05_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
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
        if "." not in key:
            continue
        feature_name, stat_name = key.rsplit(".", 1)
        if stat_name not in {"mean", "std", "min", "max", "q01", "q99", "q10", "q90"}:
            continue
        stats.setdefault(feature_name, {})[stat_name] = value.cpu().tolist()

    return stats


def _extract_model_buffer_stats(checkpoint_path: Path) -> Dict[str, Dict[str, Any]]:
    if safe_open_safetensors is None:
        raise RuntimeError("缺少 safetensors 依赖，无法读取 model.safetensors 统计")

    model_path = checkpoint_path / "model.safetensors"
    if not model_path.is_file():
        return {}

    stats: Dict[str, Dict[str, Any]] = {}
    key_map = {
        "normalize_inputs.buffer_observation_state": "observation.state",
        "normalize_targets.buffer_action": "action",
        "unnormalize_outputs.buffer_action": "action",
    }
    stat_names = {"mean", "std", "min", "max", "q01", "q99", "q10", "q90"}

    with safe_open_safetensors(str(model_path), framework="pt", device="cpu") as tensors:
        for tensor_key in tensors.keys():
            prefix, _, stat_name = tensor_key.rpartition(".")
            feature_key = key_map.get(prefix)
            if feature_key is None or stat_name not in stat_names:
                continue
            stats.setdefault(feature_key, {})[stat_name] = tensors.get_tensor(tensor_key).cpu().tolist()

    return stats


def _load_external_stats_from_env(*env_names: str) -> Dict[str, Dict[str, Any]]:
    for env_name in env_names:
        stats_path = os.environ.get(env_name)
        if not stats_path:
            continue
        path = Path(stats_path).expanduser()
        if path.is_file():
            with open(path, "r") as f:
                return json.load(f)
    return {}


def _apply_processor_normalization_mapping(checkpoint_path: Path, config_dict: Dict[str, Any]) -> None:
    if config_dict.get("normalization_mapping"):
        return

    for config_name in (PREPROCESSOR_CONFIG_FILENAME, POSTPROCESSOR_CONFIG_FILENAME):
        config_path = checkpoint_path / config_name
        if not config_path.is_file():
            continue
        with open(config_path, "r") as f:
            processor_config = json.load(f)
        for step in processor_config.get("steps", []):
            registry_name = str(step.get("registry_name", ""))
            if "normalizer" not in registry_name:
                continue
            norm_map = (step.get("config") or {}).get("norm_map")
            if norm_map:
                config_dict["normalization_mapping"] = norm_map
                return


def _normalization_mode_name(value: Any) -> str:
    mode = getattr(value, "value", value)
    return str(mode).upper()


def _stats_tensor(stats: Dict[str, Any], name: str, *, like: torch.Tensor) -> torch.Tensor:
    if name not in stats:
        raise ValueError(f"PI05 normalization stats missing `{name}`")
    return torch.as_tensor(stats[name], dtype=like.dtype, device=like.device)


def _apply_feature_normalization(
    *,
    tensor: torch.Tensor,
    key: str,
    feature_type: str,
    config_dict: Optional[Dict[str, Any]],
    stats: Optional[Dict[str, Dict[str, Any]]],
    inverse: bool,
) -> torch.Tensor:
    if config_dict is None or stats is None or key not in stats:
        return tensor

    norm_map = config_dict.get("normalization_mapping") or {}
    mode = _normalization_mode_name(norm_map.get(feature_type, "IDENTITY"))
    if mode == "IDENTITY":
        return tensor

    feature_stats = stats[key]
    eps = torch.tensor(1e-8, dtype=tensor.dtype, device=tensor.device)

    if mode == "MEAN_STD":
        mean = _stats_tensor(feature_stats, "mean", like=tensor)
        std = _stats_tensor(feature_stats, "std", like=tensor)
        return tensor * std + mean if inverse else (tensor - mean) / (std + eps)

    if mode == "MIN_MAX":
        min_value = _stats_tensor(feature_stats, "min", like=tensor)
        max_value = _stats_tensor(feature_stats, "max", like=tensor)
        denom = torch.where(max_value == min_value, eps, max_value - min_value)
        return (tensor + 1.0) * denom / 2.0 + min_value if inverse else 2.0 * (tensor - min_value) / denom - 1.0

    if mode == "QUANTILES":
        low = _stats_tensor(feature_stats, "q01", like=tensor)
        high = _stats_tensor(feature_stats, "q99", like=tensor)
        denom = torch.where(high == low, eps, high - low)
        return (tensor + 1.0) * denom / 2.0 + low if inverse else 2.0 * (tensor - low) / denom - 1.0

    if mode == "QUANTILE10":
        low = _stats_tensor(feature_stats, "q10", like=tensor)
        high = _stats_tensor(feature_stats, "q90", like=tensor)
        denom = torch.where(high == low, eps, high - low)
        return (tensor + 1.0) * denom / 2.0 + low if inverse else 2.0 * (tensor - low) / denom - 1.0

    raise ValueError(f"Unsupported PI05 normalization mode: {mode}")


def _last_stat_value(feature_stats: Dict[str, Any], stat_name: str) -> Optional[float]:
    if stat_name not in feature_stats:
        return None

    values = np.asarray(feature_stats[stat_name], dtype=np.float32)
    if values.size == 0:
        return None
    return float(values.reshape(-1)[-1])


def _feature_gripper_uses_robot_units(
    stats: Optional[Dict[str, Dict[str, Any]]],
    feature_key: str,
) -> bool:
    if stats is None or feature_key not in stats:
        return False

    feature_stats = stats[feature_key]
    for stat_name in ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99"):
        value = _last_stat_value(feature_stats, stat_name)
        if value is not None and abs(value) > 1.5:
            return True
    return False


def _load_pretrained_pi05_stats(checkpoint_path: Path) -> Dict[str, Dict[str, Any]]:
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
        stats = _extract_model_buffer_stats(checkpoint_path)

    if not stats:
        stats = _load_external_stats_from_env("PI05_STATS_PATH", "PI0_STATS_PATH")

    if not stats:
        raise FileNotFoundError("导出模型缺少可用的预处理/后处理统计")

    return stats


def _is_local_model_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "tokenizer_config.json").is_file()
        or (path / "tokenizer.json").is_file()
        or (path / "config.json").is_file()
    )


def _normalize_pi05_tokenizer_name(tokenizer_name: Optional[str]) -> str:
    if not isinstance(tokenizer_name, str):
        return DEFAULT_PI05_TOKENIZER

    normalized = tokenizer_name.strip()
    if not normalized:
        return DEFAULT_PI05_TOKENIZER
    return normalized


def _candidate_pi05_tokenizer_names(requested_tokenizer: str) -> tuple[str, ...]:
    candidates = [requested_tokenizer]
    for fallback in PI05_TOKENIZER_FALLBACKS.get(requested_tokenizer, ()):
        if fallback not in candidates:
            candidates.append(fallback)
    return tuple(candidates)


def _read_pi05_tokenizer_name(checkpoint_path: Path) -> str:
    preprocessor_path = checkpoint_path / PREPROCESSOR_CONFIG_FILENAME
    if not preprocessor_path.is_file():
        return DEFAULT_PI05_TOKENIZER

    try:
        with open(preprocessor_path, "r") as f:
            processor_config = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read PI05 preprocessor config %s: %s", preprocessor_path, exc)
        return DEFAULT_PI05_TOKENIZER

    for step in processor_config.get("steps", []):
        if step.get("registry_name") != "tokenizer_processor":
            continue
        tokenizer_name = (step.get("config") or {}).get("tokenizer_name")
        if isinstance(tokenizer_name, str) and tokenizer_name.strip():
            return _normalize_pi05_tokenizer_name(tokenizer_name)

    return DEFAULT_PI05_TOKENIZER


def _add_relative_model_candidates(candidate_paths: list[Path], base_dir: Path, tokenizer_name: str) -> None:
    repo_path = Path(*tokenizer_name.split("/"))
    candidate_paths.extend(
        [
            base_dir / repo_path,
            base_dir / repo_path.name,
        ]
    )


def _resolve_pi05_tokenizer_source(checkpoint_path: Path, tokenizer_path: Optional[str] = None) -> str:
    requested_tokenizer = _normalize_pi05_tokenizer_name(_read_pi05_tokenizer_name(checkpoint_path))
    candidate_paths: list[Path] = []

    for override in (
        tokenizer_path,
        os.environ.get("PI05_TOKENIZER_PATH"),
        os.environ.get("PI0_TOKENIZER_PATH"),
    ):
        if not override:
            continue
        candidate_paths.append(Path(override).expanduser())

    candidate_paths.append(checkpoint_path / "tokenizer")

    for base_dir in iter_model_search_roots(checkpoint_path):
        for tokenizer_name in _candidate_pi05_tokenizer_names(requested_tokenizer):
            _add_relative_model_candidates(candidate_paths, base_dir, tokenizer_name)

    for candidate in iter_unique_paths(candidate_paths):
        if _is_local_model_dir(candidate):
            logger.info("Using local PI05 tokenizer assets from %s", candidate)
            return str(candidate)

    return requested_tokenizer


@contextmanager
def _normalized_hf_proxy_env() -> Iterator[None]:
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


def _format_pi05_load_error(error: Exception, requested_tokenizer: str) -> str:
    message = str(error)

    if "Unknown scheme for proxy URL" in message and "socks://" in message:
        return (
            "模型加载失败: 检测到无效代理配置。当前 `ALL_PROXY/HTTP_PROXY` 使用了 `socks://`，"
            "请改成 `socks5://127.0.0.1:7897/`，或者临时取消 `ALL_PROXY` 后重启后端。"
        )

    if "Can't load tokenizer" in message or "Can't load the tokenizer" in message:
        return (
            "模型加载失败: PI05 tokenizer 未就绪。当前需要加载 "
            f"`{requested_tokenizer}`。如果你要离线推理，请先下载该 tokenizer，例如:\n"
            f"`source .venv/bin/activate && hf download {requested_tokenizer} "
            f"--local-dir models/{requested_tokenizer}`\n"
            "然后任选其一:\n"
            f"`export PI05_TOKENIZER_PATH=/absolute/path/to/models/{requested_tokenizer}`\n"
            f"`export PI0_TOKENIZER_PATH=/absolute/path/to/models/{requested_tokenizer}`\n"
            "`export INFERENCE_SDK_MODEL_ROOTS=/absolute/path/to/models`"
        )

    if (
        "gated repo" in message.lower()
        or "Cannot access gated repo" in message
        or "401 Client Error" in message
        or ("Access to model " in message and " is restricted" in message)
    ):
        return (
            f"模型加载失败: 当前 PI05 需要的 tokenizer `{requested_tokenizer}` 是 Hugging Face 受限仓库，"
            "你当前环境既没有本地 tokenizer 文件，也没有可用的 HF 授权。\n"
            "可选解决方案:\n"
            "1. 使用有权限的 Hugging Face 账号登录并下载 tokenizer:\n"
            "`source .venv/bin/activate && hf auth login`\n"
            f"`hf download {requested_tokenizer} --local-dir models/{requested_tokenizer}`\n"
            "2. 如果这台机器不方便联网，可在另一台已获授权的机器下载后拷贝到本机。\n"
            "3. 通过 `PI05_TOKENIZER_PATH`、`PI0_TOKENIZER_PATH` 或 `INFERENCE_SDK_MODEL_ROOTS` 指向本地目录。"
        )

    if "Operation not permitted" in message:
        return (
            "模型加载失败: 当前环境无法访问 Hugging Face。请检查网络/代理，"
            "或者先把 PI05 tokenizer 下载到本地，并通过 `PI05_TOKENIZER_PATH` 指向它。"
        )

    return f"模型加载失败: {message}"


PI05_AVAILABLE = False
PI05_IMPORT_ERROR: Exception | None = None
try:
    from transformers import AutoTokenizer

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from sparkmind.learning.VLA.models.pi05_model import PI05Pytorch
    from sparkmind.learning.VLA.utils.pi_common import load_pi_core_model_weights

    PI05_AVAILABLE = True
    logger.info("SparkMind PI05 model loaded successfully")
except Exception as e:
    PI05_IMPORT_ERROR = e
    logger.warning("SparkMind PI05 model not available: %s", e)


def _coerce_policy_feature_map(feature_map: Dict[str, Any]) -> Dict[str, Any]:
    if not PI05_AVAILABLE:
        return feature_map

    coerced: Dict[str, Any] = {}
    for key, value in feature_map.items():
        if isinstance(value, PolicyFeature):
            coerced[key] = value
            continue
        if isinstance(value, dict) and "type" in value and "shape" in value:
            coerced[key] = PolicyFeature(
                type=FeatureType(str(value["type"]).upper()),
                shape=tuple(value["shape"]),
            )
        else:
            coerced[key] = value
    return coerced


def _coerce_pi05_config_dict(config_dict: Dict[str, Any], device: str) -> Dict[str, Any]:
    if not PI05_AVAILABLE:
        return config_dict

    accepted_keys = {f.name for f in fields(PI05Config)}
    filtered_config = {k: v for k, v in config_dict.items() if k in accepted_keys}
    filtered_config["device"] = device

    if "image_resolution" in filtered_config and isinstance(filtered_config["image_resolution"], list):
        filtered_config["image_resolution"] = tuple(filtered_config["image_resolution"])

    if "input_features" in filtered_config and isinstance(filtered_config["input_features"], dict):
        filtered_config["input_features"] = _coerce_policy_feature_map(filtered_config["input_features"])

    if "output_features" in filtered_config and isinstance(filtered_config["output_features"], dict):
        filtered_config["output_features"] = _coerce_policy_feature_map(filtered_config["output_features"])

    if "normalization_mapping" in filtered_config and isinstance(filtered_config["normalization_mapping"], dict):
        filtered_config["normalization_mapping"] = {
            key: value if isinstance(value, NormalizationMode) else NormalizationMode(str(value))
            for key, value in filtered_config["normalization_mapping"].items()
        }

    return filtered_config


class PI05InferenceEngine(BaseInferenceEngine):
    """PI0.5 inference engine for language-conditioned action chunks."""

    def __init__(
        self,
        device: str = "cuda:0",
        smoothing_config: Optional[SmoothingConfig] = None,
        strict_device: bool = False,
    ):
        super().__init__(smoothing_config)
        self.model_type = "pi05"

        device_selection = resolve_torch_device(device, strict=strict_device)
        self.requested_device = device_selection.requested
        self.actual_device = device_selection.actual
        self.device_warning = device_selection.warning
        if self.device_warning:
            logger.warning(self.device_warning)
        self.device = torch.device(self.actual_device)

        self.model: Optional[Any] = None
        self.config: Optional[Any] = None
        self.config_dict: Optional[Dict[str, Any]] = None
        self.stats: Optional[Dict[str, Dict[str, Any]]] = None
        self.tokenizer: Optional[Any] = None
        self.tokenizer_source: Optional[str] = None

        self._state_gripper_uses_robot_units: bool = False
        self._action_gripper_uses_robot_units: bool = False
        self._state_action_normalization: str = "sdk_stats"

        self._camera_key_to_role: Dict[str, str] = {}
        self._role_to_camera_key: Dict[str, str] = {}
        self._camera_alias_to_key: Dict[str, str] = {}

        self._instruction: str = "Pick up the object."
        self._image_resize: Optional[Tuple[int, int]] = (224, 224)
        self._max_state_dim: int = 32
        self._rtc_prev_chunk_left_over: Optional[torch.Tensor] = None

    @staticmethod
    def validate_checkpoint(checkpoint_dir: str) -> Tuple[bool, str]:
        """Validate that checkpoint directory contains a supported PI05 format."""
        raw_path = Path(checkpoint_dir)
        if not raw_path.exists():
            return False, f"Checkpoint目录不存在: {checkpoint_dir}"

        resolved_path = _resolve_pi05_checkpoint_dir(checkpoint_dir)
        if _is_legacy_pi05_checkpoint(resolved_path) or _is_pretrained_pi05_dir(resolved_path):
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
            "PI05模型目录格式不受支持。"
            f" 旧格式需要: {', '.join(LEGACY_CHECKPOINT_FILES)}"
            f"；导出格式需要: {', '.join(PRETRAINED_CHECKPOINT_FILES)}"
            f" + 至少一个processor配置({PREPROCESSOR_CONFIG_FILENAME}/{POSTPROCESSOR_CONFIG_FILENAME})"
            f"。当前缺少 旧格式文件: {', '.join(legacy_missing) or '无'}"
            f"；导出格式文件: {', '.join(exported_missing) or '无'}"
            f"；processor配置: {', '.join(processor_missing) or '无'}",
        )

    def load(self, checkpoint_dir: str, tokenizer_path: Optional[str] = None) -> Tuple[bool, str]:
        """Load PI05 model from checkpoint."""
        if not PI05_AVAILABLE:
            return False, format_optional_dependency_error(
                "PI05 模型所需的 SparkMind / transformers",
                PI05_IMPORT_ERROR,
                min_python=(3, 12),
                install_hint="请先安装本地 SparkMind checkout，例如 `uv pip install -e \"third_party/SparkMind[pi,libero]\" -i https://pypi.tuna.tsinghua.edu.cn/simple`。",
            )

        valid, error = self.validate_checkpoint(checkpoint_dir)
        if not valid:
            return False, error

        checkpoint_path = _resolve_pi05_checkpoint_dir(checkpoint_dir)
        tokenizer_source = _resolve_pi05_tokenizer_source(checkpoint_path, tokenizer_path=tokenizer_path)
        self.tokenizer_source = tokenizer_source

        try:
            self._state_action_normalization = (
                "sdk_stats" if _is_legacy_pi05_checkpoint(checkpoint_path) else "external_processor"
            )
            if _is_legacy_pi05_checkpoint(checkpoint_path):
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

                self.config_dict = _convert_pretrained_pi05_config(pretrained_config)
                _apply_processor_normalization_mapping(checkpoint_path, self.config_dict)
                self.stats = _load_pretrained_pi05_stats(checkpoint_path)

            self._apply_action_chunk_overrides(self.config_dict)

            image_features = self.config_dict.get("image_features", [])
            self.required_cameras = []
            self._camera_key_to_role = {}
            self._role_to_camera_key = {}
            self._camera_alias_to_key = {}

            for key in image_features:
                if key.startswith("observation.images."):
                    suffix = key.replace("observation.images.", "")
                    role = suffix[4:] if suffix.startswith("cam_") else suffix
                    if role not in self.required_cameras:
                        self.required_cameras.append(role)
                    self._camera_key_to_role[key] = role
                    self._role_to_camera_key[role] = key
                    self._camera_alias_to_key[role] = key
                    self._camera_alias_to_key[suffix] = key
                    self._camera_alias_to_key[key] = key

            state_shape = self.config_dict.get("robot_state_feature", {}).get("shape", [7])
            action_shape = self.config_dict.get("action_feature", {}).get("shape", [7])
            self.state_dim = state_shape[0] if state_shape else 7
            self.action_dim = action_shape[0] if action_shape else 7
            self.chunk_size = int(self.config_dict.get("chunk_size", 50))
            self.n_action_steps = int(self.config_dict.get("n_action_steps", self.chunk_size))
            self._max_state_dim = int(self.config_dict.get("max_state_dim", 32))
            self._state_gripper_uses_robot_units = _feature_gripper_uses_robot_units(
                self.stats,
                "observation.state",
            )
            self._action_gripper_uses_robot_units = _feature_gripper_uses_robot_units(self.stats, "action")

            pi05_config_kwargs = _coerce_pi05_config_dict(self.config_dict, str(self.device))
            if self.smoothing_config.enable_rtc:
                pi05_config_kwargs["rtc_config"] = make_rtc_config(self.smoothing_config)
            self.config = PI05Config(**pi05_config_kwargs)
            self._image_resize = tuple(self.config.image_resolution) if self.config.image_resolution is not None else None
            self._max_state_dim = int(self.config.max_state_dim)

            with _normalized_hf_proxy_env():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source,
                    trust_remote_code=True,
                )

            self.model = PI05Pytorch(
                self.config,
                rtc_processor=make_rtc_processor(self.smoothing_config),
            )
            self.model.to(self.device)
            self.model.eval()

            missing_keys, unexpected_keys = load_pi_core_model_weights(
                self.model,
                checkpoint_path,
                policy_cls=PI05Policy,
                config=self.model.config,
                family="PI05",
                device=self.device,
            )

            self.is_loaded = True
            self._init_components()
            self.reset()

            logger.info("PI05 model loaded from %s", checkpoint_dir)
            logger.info("Required cameras: %s", self.required_cameras)
            logger.info("State dim: %s, Action dim: %s", self.state_dim, self.action_dim)
            logger.info("Chunk size: %s, N action steps: %s", self.chunk_size, self.n_action_steps)
            logger.info("Tokenizer: %s", tokenizer_source)
            logger.info(
                "PI05 gripper stats units: state=%s action=%s",
                "robot" if self._state_gripper_uses_robot_units else "normalized",
                "robot" if self._action_gripper_uses_robot_units else "normalized",
            )
            logger.info(
                "PI05 preprocessing: %s",
                self._state_action_normalization,
            )
            logger.info(
                "PI05 weights loaded: missing=%s unexpected=%s",
                len(missing_keys),
                len(unexpected_keys),
            )

            return True, ""

        except Exception as e:
            logger.exception("Failed to load PI05 model")
            return False, _format_pi05_load_error(e, tokenizer_source)

    def set_instruction(self, instruction: str) -> bool:
        """Set the language instruction for PI05."""
        if not self.is_loaded or self.tokenizer is None:
            logger.warning("Cannot set instruction: model not loaded")
            return False

        self._instruction = instruction
        if self._action_queue is not None:
            self._action_queue.reset()
        self._rtc_prev_chunk_left_over = None

        logger.info("Instruction set: %s", instruction)
        return True

    def reset(self):
        super().reset()
        self._rtc_prev_chunk_left_over = None

    def get_instruction(self) -> str:
        """Get current instruction."""
        return self._instruction

    def _tokenize_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        tokenizer_max_length = self.config.tokenizer_max_length if self.config is not None else 200
        self.tokenizer.padding_side = "right"
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
        )
        return tokens["input_ids"].to(self.device), tokens["attention_mask"].bool().to(self.device)

    @staticmethod
    def _resize_with_pad(img: np.ndarray, target_h: int, target_w: int, pad_value: int = 0) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)

        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return padded

    def _preprocess_images(self, images: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        processed = {}

        for camera_alias, img_bgr in images.items():
            camera_key = self._camera_alias_to_key.get(camera_alias)
            if camera_key is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if self._image_resize is not None:
                target_h, target_w = self._image_resize
                img_rgb = self._resize_with_pad(img_rgb, target_h, target_w, pad_value=0)

            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor * 2.0 - 1.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            processed[camera_key] = img_tensor

        return processed

    def _preprocess_state_for_prompt(self, state: np.ndarray) -> torch.Tensor:
        state = state.copy()

        if len(state) >= 7 and not self._state_gripper_uses_robot_units:
            state[-1] = state[-1] / 1000.0

        state_tensor = torch.from_numpy(state).float()
        if self._state_action_normalization == "sdk_stats":
            state_tensor = _apply_feature_normalization(
                tensor=state_tensor,
                key="observation.state",
                feature_type="STATE",
                config_dict=self.config_dict,
                stats=self.stats,
                inverse=False,
            )

        if state_tensor.shape[-1] < self._max_state_dim:
            state_tensor = F.pad(state_tensor, (0, self._max_state_dim - state_tensor.shape[-1]))
        return state_tensor

    def _build_prompt(self, instruction: str, normalized_state: torch.Tensor) -> str:
        state_np = normalized_state.detach().cpu().numpy()
        discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        cleaned_instruction = instruction.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized_state))
        return f"Task: {cleaned_instruction}, State: {state_str};\nAction: "

    def _postprocess_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        action = action_tensor.cpu()
        if self._state_action_normalization == "sdk_stats":
            action = _apply_feature_normalization(
                tensor=action,
                key="action",
                feature_type="ACTION",
                config_dict=self.config_dict,
                stats=self.stats,
                inverse=True,
            )

        action = action.numpy()

        if len(action) >= 7 and not self._action_gripper_uses_robot_units:
            action[-1] = action[-1] * 1000.0

        return action

    @torch.no_grad()
    def _predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """Predict a chunk of unnormalized PI05 robot actions."""
        if self.model is None or self.config is None:
            raise RuntimeError("Model not loaded")

        processed_images = self._preprocess_images(images)
        normalized_state = self._preprocess_state_for_prompt(state)
        prompt = self._build_prompt(self._instruction, normalized_state)
        tokens, masks = self._tokenize_prompt(prompt)

        image_features = self.config_dict.get("image_features", [])
        present_images = [processed_images[key] for key in image_features if key in processed_images]
        if not present_images:
            raise RuntimeError("No valid images available for PI05 inference")

        image_list = []
        image_masks = []
        empty_template = present_images[0]
        for key in image_features:
            if key in processed_images:
                image_list.append(processed_images[key])
                image_masks.append(torch.ones(1, dtype=torch.bool, device=self.device))
            else:
                image_list.append(torch.ones_like(empty_template) * -1.0)
                image_masks.append(torch.zeros(1, dtype=torch.bool, device=self.device))

        actions_chunk = self.model.sample_actions(
            images=image_list,
            img_masks=image_masks,
            tokens=tokens,
            masks=masks,
            **self._rtc_kwargs(),
        )

        self._update_rtc_left_over(actions_chunk)
        actions_normalized = actions_chunk[0, :self.n_action_steps, :self.action_dim]
        actions = np.stack(
            [self._postprocess_action(actions_normalized[i]) for i in range(actions_normalized.shape[0])]
        )
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
        chunk = actions_chunk.detach()[:, :, : self.action_dim]
        self._rtc_prev_chunk_left_over = chunk[:, delay:].clone() if delay else chunk.clone()

    def unload(self):
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.tokenizer_source = None

        self.is_loaded = False
        self.reset()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        logger.info("PI05 model unloaded")
