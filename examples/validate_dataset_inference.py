#!/usr/bin/env python3
"""Run SDK inference on one or more dataset episodes and plot prediction curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)


def _ensure_cache_dirs(repo_root: Path) -> tuple[Path, Path]:
    hf_home = repo_root / ".cache" / "huggingface"
    datasets_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"
    mpl_cache = repo_root / ".cache" / "matplotlib"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    return datasets_cache, hub_cache


HF_DATASETS_CACHE, HF_HUB_CACHE = _ensure_cache_dirs(REPO_ROOT)


def _ensure_sparkmind_path(repo_root: Path) -> Path:
    candidates = (
        repo_root / "third_party" / "SparkMind",
        repo_root.parent / "SparkMind",
        repo_root / "SparkMind",
    )
    for sparkmind_root in candidates:
        if sparkmind_root.is_dir():
            sparkmind_root_str = str(sparkmind_root)
            if sparkmind_root_str not in sys.path:
                sys.path.insert(0, sparkmind_root_str)
            return sparkmind_root
    return candidates[0]


SPARKMIND_ROOT = _ensure_sparkmind_path(REPO_ROOT)

from inference_sdk import (
    SUPPORTED_MODEL_TYPES,
    SmoothingConfig,
    create_engine,
    normalize_model_type,
)


def _load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub\n"
            "Install example dependencies first, for example:\n"
            "  uv pip install -e '.[all,examples]' -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc
    return snapshot_download


def _load_dataset_class():
    try:
        from sparkmind.lerobot_compat.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: LeRobotDataset\n"
                "Install SparkMind or lerobot first, for example:\n"
                "  uv pip install -e '.[all,examples]' -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                "or:\n"
                "  uv pip install -e third_party/SparkMind -i https://pypi.tuna.tsinghua.edu.cn/simple"
            ) from exc
    return LeRobotDataset


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: matplotlib\n"
            "Install example dependencies first, for example:\n"
            "  uv pip install -e '.[examples]' -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc
    return plt


def _default_model_source() -> str:
    return str(REPO_ROOT / "models" / "ACT_pick_and_place_v2")


def _default_dataset_source() -> str:
    return str(REPO_ROOT / "data" / "lerobot" / "z18820636149" / "pick_and_place_data90")


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "run"


def _default_output_dir(model_source: str, dataset_source: str, scope: str) -> str:
    model_name = _sanitize_name(Path(model_source).name if Path(model_source).exists() else model_source)
    dataset_name = _sanitize_name(Path(dataset_source).name if Path(dataset_source).exists() else dataset_source)
    return str(REPO_ROOT / "outputs" / "validate_dataset_inference" / f"{_timestamp_tag()}_{model_name}_{dataset_name}_{scope}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate SDK inference against a local or Hub LeRobot dataset. "
            "The script compares each frame's first predicted action with the dataset action "
            "and saves per-episode point-line plots."
        )
    )
    parser.add_argument(
        "--model",
        default=_default_model_source(),
        help="Local exported model directory or Hugging Face model repo id",
    )
    parser.add_argument(
        "--model-type",
        type=str.lower,
        choices=SUPPORTED_MODEL_TYPES,
        default=None,
        help="Explicit model type override. Default: infer from model config.",
    )
    parser.add_argument(
        "--dataset",
        default=_default_dataset_source(),
        help="Local LeRobot dataset root or Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to validate. Ignored when --all-episodes is used.",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Validate every episode and generate one plot per episode.",
    )
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Inference device. Default: cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots and reports. Default: outputs/validate_dataset_inference/<timestamp>_...",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Optional explicit instruction. PI0/PI0.5/SmolVLA otherwise use the dataset task string.",
    )
    parser.add_argument(
        "--camera-map",
        action="append",
        default=None,
        metavar="POLICY_CAMERA=DATASET_CAMERA",
        help=(
            "Map a policy camera role to a dataset camera key/role. "
            "Example: camera1=image. Can be repeated."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=["auto", "raw", "step"],
        default="auto",
        help=(
            "Validation path. `raw` compares the first action from `predict_chunk()`. "
            "`step` validates control-loop execution via `step()`. "
            "`auto` uses `step()` when temporal ensembling is requested; otherwise it uses `raw`."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="Enable SDK ACT temporal ensembling. Effective only for ACT in step mode.",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="ACT temporal ensembling coefficient. Default: 0.01 when --temporal-ensemble is set.",
    )
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=None,
        help="Override checkpoint chunk_size, the model forward action horizon.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override checkpoint n_action_steps, the number of actions returned/enqueued per inference.",
    )
    parser.add_argument(
        "--enable-rtc",
        action="store_true",
        help="Enable RTC for SmolVLA/PI0/PI0.5 policies.",
    )
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=["ZEROS", "ONES", "LINEAR", "EXP", "zeros", "ones", "linear", "exp"],
        default="LINEAR",
        help="RTC prefix attention schedule. Default: LINEAR.",
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=10.0,
        help="RTC max guidance weight. Default: 10.0.",
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=10,
        help="RTC execution horizon. Default: 10.",
    )
    parser.add_argument(
        "--rtc-inference-delay-steps",
        type=int,
        default=0,
        help="Static RTC inference delay in control steps. Default: 0.",
    )
    parser.add_argument(
        "--rtc-debug",
        action="store_true",
        help="Enable RTC debug tracking in SparkMind's RTCProcessor.",
    )
    parser.add_argument(
        "--rtc-debug-maxlen",
        type=int,
        default=100,
        help="Maximum RTC debug records to keep. Default: 100.",
    )
    parser.add_argument(
        "--dataset-gripper-scale",
        choices=["auto", "normalized", "raw"],
        default="auto",
        help="How the dataset stores gripper values. Usually auto is correct.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="Video backend used by LeRobotDataset. Default: pyav.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit for the number of frames processed per episode.",
    )
    return parser.parse_args()


@dataclass
class EngineMetadata:
    required_cameras: list[str]
    state_dim: int
    action_dim: int
    chunk_size: int
    n_action_steps: int


@dataclass
class Observation:
    images: dict[str, np.ndarray]
    state: np.ndarray
    instruction: str | None = None


@dataclass
class ErrorAccumulator:
    action_dim: int
    compared_vectors: int = 0
    compared_values: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    max_abs: float = 0.0
    per_dim_abs_sum: np.ndarray | None = None
    per_dim_sq_sum: np.ndarray | None = None

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=np.float32)
        gt = np.asarray(target, dtype=np.float32)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
        if pred.ndim != 1 or pred.shape[0] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {pred.shape}")

        error = pred - gt
        abs_error = np.abs(error)
        sq_error = np.square(error)

        if self.per_dim_abs_sum is None:
            self.per_dim_abs_sum = np.zeros(self.action_dim, dtype=np.float64)
            self.per_dim_sq_sum = np.zeros(self.action_dim, dtype=np.float64)

        self.compared_vectors += 1
        self.compared_values += pred.size
        self.abs_sum += float(abs_error.sum())
        self.sq_sum += float(sq_error.sum())
        self.max_abs = max(self.max_abs, float(abs_error.max(initial=0.0)))
        self.per_dim_abs_sum += abs_error
        self.per_dim_sq_sum += sq_error

    def as_dict(self) -> dict[str, Any]:
        if self.compared_values == 0 or self.compared_vectors == 0:
            return {
                "compared_vectors": 0,
                "compared_values": 0,
            }

        assert self.per_dim_abs_sum is not None
        assert self.per_dim_sq_sum is not None

        return {
            "compared_vectors": self.compared_vectors,
            "compared_values": self.compared_values,
            "mae": self.abs_sum / self.compared_values,
            "rmse": float(np.sqrt(self.sq_sum / self.compared_values)),
            "max_abs": self.max_abs,
            "per_dim_mae": (self.per_dim_abs_sum / self.compared_vectors).tolist(),
            "per_dim_rmse": np.sqrt(self.per_dim_sq_sum / self.compared_vectors).tolist(),
        }


@dataclass
class EpisodeResult:
    episode_index: int
    task: str
    dataset_indices: np.ndarray
    frame_indices: np.ndarray
    predictions: np.ndarray
    targets: np.ndarray
    metrics: dict[str, Any]
    average_call_ms: float
    plot_path: str = ""
    csv_path: str = ""


def _normalize_local_model_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved.parent
    pretrained_dir = resolved / "pretrained_model"
    if pretrained_dir.is_dir() and (pretrained_dir / "config.json").is_file():
        return pretrained_dir
    return resolved


def _normalize_local_dataset_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.parent.name == "meta":
            return resolved.parent.parent
        return resolved.parent
    return resolved


def _resolve_model_source(source: str) -> tuple[Path, str]:
    source_path = Path(source).expanduser()
    if source_path.exists():
        model_dir = _normalize_local_model_dir(source_path)
        if not (model_dir / "config.json").is_file():
            candidate = model_dir / "pretrained_model" / "config.json"
            raise FileNotFoundError(
                f"Model directory missing config.json: {model_dir}. "
                f"If this is an exported checkpoint, expected either {model_dir / 'config.json'} "
                f"or {candidate}."
            )
        return model_dir, str(model_dir)

    snapshot_download = _load_snapshot_download()
    model_dir = Path(
        snapshot_download(
            repo_id=source,
            repo_type="model",
            cache_dir=str(HF_HUB_CACHE),
            local_files_only=False,
        )
    )
    model_dir = _normalize_local_model_dir(model_dir)
    return model_dir, source


def _resolve_dataset_source(source: str) -> tuple[Path, str]:
    source_path = Path(source).expanduser()
    if source_path.exists():
        dataset_root = _normalize_local_dataset_root(source_path)
        if not (dataset_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Dataset root missing meta/info.json: {dataset_root}")
        return dataset_root, str(dataset_root)

    snapshot_download = _load_snapshot_download()
    dataset_root = Path(
        snapshot_download(
            repo_id=source,
            repo_type="dataset",
            cache_dir=str(HF_HUB_CACHE),
            local_files_only=False,
        )
    )
    return dataset_root, source


def _infer_model_type(checkpoint_dir: Path) -> str:
    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_type = str(config.get("type", "")).strip().lower()
        if model_type:
            return normalize_model_type(model_type)

    train_config_path = checkpoint_dir / "train_config.json"
    if train_config_path.is_file():
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        model_type = str(train_config.get("policy", {}).get("type", "")).strip().lower()
        if model_type:
            return normalize_model_type(model_type)

    raise ValueError(f"Could not infer model type from {checkpoint_dir}")


def _resolve_model_type(explicit_model_type: str | None, checkpoint_dir: Path) -> str:
    inferred_model_type = _infer_model_type(checkpoint_dir)
    if explicit_model_type is None:
        return inferred_model_type

    requested_model_type = normalize_model_type(explicit_model_type)
    if requested_model_type != inferred_model_type:
        raise ValueError(
            "Explicit model type does not match checkpoint config: "
            f"requested `{requested_model_type}`, but `{checkpoint_dir}` declares "
            f"`{inferred_model_type}` in config.json/train_config.json. "
            "Remove `--model-type` to auto-detect it, or pass the declared type explicitly."
        )

    return requested_model_type


def _dataset_repo_id_for_source(source: str, dataset_root: Path) -> str:
    source_path = Path(source).expanduser()
    if source_path.exists():
        return f"local/{dataset_root.name}"
    return source


def _tensor_image_to_bgr_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim != 3 or array.shape[0] not in {1, 3}:
        raise ValueError(f"Expected CHW tensor image, got shape={array.shape}")

    array = np.transpose(array, (1, 2, 0))
    if array.dtype.kind == "f":
        array = np.clip(array, 0.0, 1.0)
        array = np.rint(array * 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)

    return array[:, :, ::-1].copy()


def _detect_dataset_gripper_mode(actions: np.ndarray, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    max_abs = float(np.max(np.abs(actions[:, -1]))) if actions.size > 0 else 0.0
    return "normalized" if max_abs <= 1.5 else "raw"


def _adapt_state_for_sdk(state: np.ndarray, gripper_mode: str) -> np.ndarray:
    result = np.asarray(state, dtype=np.float32).copy()
    if result.shape[-1] > 0 and gripper_mode == "normalized":
        result[-1] *= 1000.0
    return result


def _parse_camera_map(values: list[str] | None) -> dict[str, str]:
    if not values:
        return {}
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --camera-map value {value!r}; expected POLICY_CAMERA=DATASET_CAMERA")
        policy_camera, dataset_camera = value.split("=", 1)
        policy_camera = policy_camera.strip()
        dataset_camera = dataset_camera.strip()
        if not policy_camera or not dataset_camera:
            raise ValueError(f"Invalid --camera-map value {value!r}; expected POLICY_CAMERA=DATASET_CAMERA")
        result[policy_camera] = dataset_camera
    return result


def _adapt_prediction_for_dataset(actions: np.ndarray, gripper_mode: str) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32).copy()
    if result.ndim == 1:
        result = result[None, :]
    if result.shape[-1] > 0 and gripper_mode == "normalized":
        result[..., -1] /= 1000.0
    return result


def _resolve_image_key(item: dict[str, Any], required_camera: str) -> str:
    candidates = (
        f"observation.images.cam_{required_camera}",
        f"observation.images.{required_camera}",
        required_camera,
    )
    for key in candidates:
        if key in item:
            return key

    suffixes = (f".cam_{required_camera}", f".{required_camera}")
    for key in item.keys():
        if key.startswith("observation.images.") and key.endswith(suffixes):
            return key

    available = sorted(k for k in item.keys() if k.startswith("observation.images."))
    raise KeyError(f"Could not resolve camera '{required_camera}'. Available image keys: {available}")


def _resolve_instruction(model_type: str, explicit_instruction: str | None, item: dict[str, Any]) -> str | None:
    if explicit_instruction is not None:
        return explicit_instruction
    if model_type in {"pi0", "pi05", "smolvla"}:
        task = item.get("task")
        return str(task) if task is not None else None
    return None


def _build_observation(
    *,
    item: dict[str, Any],
    required_cameras: list[str],
    gripper_mode: str,
    instruction: str | None,
    camera_map: dict[str, str],
    state_dim: int,
) -> Observation:
    images = {
        camera: _tensor_image_to_bgr_uint8(item[_resolve_image_key(item, camera_map.get(camera, camera))])
        for camera in required_cameras
    }
    state = _adapt_state_for_sdk(item["observation.state"].detach().cpu().numpy(), gripper_mode)
    if state_dim > 0 and state.shape[0] != state_dim:
        if state.shape[0] < state_dim:
            raise ValueError(f"Dataset state dim ({state.shape[0]}) is smaller than model state dim ({state_dim})")
        state = state[:state_dim].copy()
    return Observation(images=images, state=state, instruction=instruction)


def _load_action_metadata(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    raw = dataset.hf_dataset.select_columns(["action", "episode_index"]).with_format(None)
    rows = raw[:]
    actions = np.asarray(rows["action"], dtype=np.float32)
    episode_indices = np.asarray(rows["episode_index"], dtype=np.int64)
    return actions, episode_indices


def _build_episode_to_indices(episode_indices: np.ndarray) -> dict[int, np.ndarray]:
    mapping: dict[int, list[int]] = {}
    for dataset_index, episode_index in enumerate(episode_indices.tolist()):
        mapping.setdefault(int(episode_index), []).append(dataset_index)
    return {episode: np.asarray(indices, dtype=np.int64) for episode, indices in mapping.items()}


def _select_episode_ids(args: argparse.Namespace, available_episode_ids: list[int]) -> list[int]:
    if args.all_episodes:
        return available_episode_ids
    if args.episode not in set(available_episode_ids):
        raise ValueError(f"Episode {args.episode} not found. Available episodes: {available_episode_ids[:10]}...")
    return [args.episode]


def _resolve_execution_mode(args: argparse.Namespace) -> str:
    if args.execution_mode != "auto":
        return args.execution_mode
    if args.temporal_ensemble:
        return "step"
    return "raw"


def _load_engine(model_type: str, model_dir: Path, args: argparse.Namespace, control_fps: float):
    smoothing_config = SmoothingConfig(
        control_fps=control_fps,
        aggregate_fn_name="latest_only",
        action_chunk_size=args.action_chunk_size,
        n_action_steps=args.n_action_steps,
        enable_temporal_ensemble=args.temporal_ensemble,
        temporal_ensemble_coeff=(
            0.01 if args.temporal_ensemble and args.temporal_ensemble_coeff is None
            else (args.temporal_ensemble_coeff or 0.01)
        ),
        enable_rtc=args.enable_rtc,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_inference_delay_steps=args.rtc_inference_delay_steps,
        rtc_debug=args.rtc_debug,
        rtc_debug_maxlen=args.rtc_debug_maxlen,
    )
    engine = create_engine(
        model_type=model_type,
        device=args.device,
        smoothing_config=smoothing_config,
    )

    ok, error = engine.load(str(model_dir))
    if not ok:
        raise RuntimeError(error)

    return engine


def _configure_instruction(engine: Any, instruction: str | None) -> None:
    if not instruction:
        return
    set_instruction = getattr(engine, "set_instruction", None)
    if callable(set_instruction):
        if not bool(set_instruction(instruction)):
            raise RuntimeError(f"Failed to set instruction: {instruction}")


def _get_engine_metadata(engine: Any) -> EngineMetadata:
    return EngineMetadata(
        required_cameras=list(engine.get_required_cameras()),
        state_dim=int(engine.state_dim),
        action_dim=int(engine.action_dim),
        chunk_size=int(engine.chunk_size),
        n_action_steps=int(engine.n_action_steps),
    )


def _plot_episode_curves(
    *,
    episode_result: EpisodeResult,
    model_label: str,
    dataset_label: str,
    output_path: Path,
) -> None:
    plt = _load_pyplot()

    frame_indices = episode_result.frame_indices
    predictions = episode_result.predictions
    targets = episode_result.targets
    metrics = episode_result.metrics
    action_dim = predictions.shape[1]

    ncols = 2
    nrows = math.ceil(action_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for dim in range(action_dim):
        axis = axes_flat[dim]
        axis.plot(
            frame_indices,
            targets[:, dim],
            color="#1f77b4",
            linewidth=1.2,
            marker="o",
            markersize=2.2,
            label="dataset action",
        )
        axis.plot(
            frame_indices,
            predictions[:, dim],
            color="#d62728",
            linewidth=1.2,
            marker="o",
            markersize=2.2,
            label="predicted action",
        )
        axis.set_title(
            f"action[{dim}]  mae={metrics['per_dim_mae'][dim]:.4f}  rmse={metrics['per_dim_rmse'][dim]:.4f}"
        )
        axis.set_ylabel("value")
        axis.grid(True, linestyle="--", alpha=0.35)

    for axis in axes_flat[action_dim:]:
        axis.axis("off")

    start_index = max(0, action_dim - ncols)
    for axis in axes_flat[start_index:action_dim]:
        axis.set_xlabel("frame index")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "\n".join(
            [
                f"Episode {episode_result.episode_index}  task={episode_result.task!r}",
                f"model={model_label}",
                f"dataset={dataset_label}",
                (
                    f"frames={len(frame_indices)}  mae={metrics['mae']:.6f}  "
                    f"rmse={metrics['rmse']:.6f}  max_abs={metrics['max_abs']:.6f}  "
                    f"avg_call_ms={episode_result.average_call_ms:.2f}"
                ),
            ]
        ),
        y=0.995,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_episode_csv(episode_result: EpisodeResult, output_path: Path) -> None:
    action_dim = episode_result.predictions.shape[1]
    fieldnames = ["dataset_index", "frame_index"]
    fieldnames += [f"target_action_{dim}" for dim in range(action_dim)]
    fieldnames += [f"pred_action_{dim}" for dim in range(action_dim)]
    fieldnames += [f"abs_error_{dim}" for dim in range(action_dim)]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(len(episode_result.dataset_indices)):
            row: dict[str, Any] = {
                "dataset_index": int(episode_result.dataset_indices[row_index]),
                "frame_index": int(episode_result.frame_indices[row_index]),
            }
            prediction = episode_result.predictions[row_index]
            target = episode_result.targets[row_index]
            abs_error = np.abs(prediction - target)
            for dim in range(action_dim):
                row[f"target_action_{dim}"] = float(target[dim])
                row[f"pred_action_{dim}"] = float(prediction[dim])
                row[f"abs_error_{dim}"] = float(abs_error[dim])
            writer.writerow(row)


def _run_episode_validation(
    *,
    engine: Any,
    dataset: Any,
    dataset_indices: np.ndarray,
    all_actions: np.ndarray,
    metadata: EngineMetadata,
    model_type: str,
    gripper_mode: str,
    explicit_instruction: str | None,
    execution_mode: str,
    camera_map: dict[str, str],
) -> EpisodeResult:
    action_dim = int(metadata.action_dim)
    accumulator = ErrorAccumulator(action_dim=action_dim)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frame_indices: list[int] = []
    task = ""
    total_call_ms = 0.0

    engine.reset()

    first_item = dataset[int(dataset_indices[0])]
    episode_instruction = _resolve_instruction(model_type, explicit_instruction, first_item)
    _configure_instruction(engine, episode_instruction)

    for position, dataset_index in enumerate(dataset_indices.tolist(), start=1):
        item = dataset[int(dataset_index)]
        instruction = _resolve_instruction(model_type, explicit_instruction, item) or episode_instruction
        observation = _build_observation(
            item=item,
            required_cameras=list(metadata.required_cameras),
            gripper_mode=gripper_mode,
            instruction=instruction,
            camera_map=camera_map,
            state_dim=int(metadata.state_dim),
        )

        start_time = time.perf_counter()
        if execution_mode == "step":
            predicted_first = engine.step(observation.images, observation.state)
        else:
            predicted_first = engine.predict_chunk(observation.images, observation.state)[0]
        total_call_ms += (time.perf_counter() - start_time) * 1000.0

        predicted_first = _adapt_prediction_for_dataset(predicted_first, gripper_mode)[0]
        target = all_actions[int(dataset_index)]
        accumulator.update(predicted_first, target)

        predictions.append(predicted_first)
        targets.append(target)

        frame_value = item["frame_index"]
        if hasattr(frame_value, "item"):
            frame_value = frame_value.item()
        frame_indices.append(int(frame_value))
        task = str(item.get("task", task))

        if position == 1 or position == len(dataset_indices) or position % 50 == 0:
            sample_mae = float(np.mean(np.abs(predicted_first - target)))
            print(
                f"  frame {position:04d}/{len(dataset_indices):04d} "
                f"dataset_idx={dataset_index:05d} frame_idx={frame_indices[-1]:04d} "
                f"mae={sample_mae:.6f}"
            )

    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)
    metrics = accumulator.as_dict()
    episode_value = dataset.hf_dataset[int(dataset_indices[0])]["episode_index"]
    if hasattr(episode_value, "item"):
        episode_value = episode_value.item()
    episode_index = int(episode_value)

    return EpisodeResult(
        episode_index=episode_index,
        task=task,
        dataset_indices=dataset_indices.copy(),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        predictions=predictions_array,
        targets=targets_array,
        metrics=metrics,
        average_call_ms=total_call_ms / max(1, len(dataset_indices)),
    )


def _print_run_header(
    *,
    model_type: str,
    model_dir: Path,
    model_label: str,
    dataset_root: Path,
    dataset_label: str,
    dataset: Any,
    metadata: EngineMetadata,
    gripper_mode: str,
    episode_ids: list[int],
    output_dir: Path,
    device: str,
    requested_execution_mode: str,
    resolved_execution_mode: str,
    temporal_ensemble_coeff: float | None,
    rtc_enabled: bool,
) -> None:
    print("=" * 80)
    print("SDK Episode Validation")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Model source: {model_label}")
    print(f"Model dir: {model_dir}")
    print(f"Dataset source: {dataset_label}")
    print(f"Dataset root: {dataset_root}")
    print(f"Frames: {len(dataset)}  Episodes: {dataset.num_episodes}  FPS: {dataset.fps}")
    print(f"Required cameras: {metadata.required_cameras}")
    print(f"Action dim: {metadata.action_dim}  Chunk size: {metadata.chunk_size}")
    print(f"Dataset gripper scale: {gripper_mode}")
    print(f"Device: {device}")
    print(f"Execution mode: requested={requested_execution_mode} resolved={resolved_execution_mode}")
    print(f"RTC: {'enabled' if rtc_enabled else 'disabled'}")
    print(
        "Temporal ensembling: "
        + (
            f"enabled (coeff={temporal_ensemble_coeff})"
            if temporal_ensemble_coeff is not None
            else "disabled"
        )
    )
    print(f"HF_DATASETS_CACHE: {HF_DATASETS_CACHE}")
    print(f"SparkMind root: {SPARKMIND_ROOT}")
    print(f"Episodes to validate: {episode_ids}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)


def _write_summary_csv(episode_results: list[EpisodeResult], output_path: Path) -> None:
    fieldnames = [
        "episode_index",
        "task",
        "num_frames",
        "mae",
        "rmse",
        "max_abs",
        "average_call_ms",
        "plot_path",
        "csv_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in episode_results:
            writer.writerow(
                {
                    "episode_index": result.episode_index,
                    "task": result.task,
                    "num_frames": len(result.frame_indices),
                    "mae": result.metrics.get("mae"),
                    "rmse": result.metrics.get("rmse"),
                    "max_abs": result.metrics.get("max_abs"),
                    "average_call_ms": result.average_call_ms,
                    "plot_path": result.plot_path,
                    "csv_path": result.csv_path,
                }
            )


def main() -> int:
    args = _parse_args()
    if args.temporal_ensemble_coeff is not None and not args.temporal_ensemble:
        raise ValueError("`--temporal-ensemble-coeff` requires `--temporal-ensemble`")

    temporal_ensemble_coeff = (
        0.01 if args.temporal_ensemble and args.temporal_ensemble_coeff is None
        else args.temporal_ensemble_coeff
    )
    execution_mode = _resolve_execution_mode(args)
    if args.execution_mode == "raw" and args.temporal_ensemble:
        raise ValueError("`--temporal-ensemble` requires `--execution-mode step` or `--execution-mode auto`")

    scope = "all_episodes" if args.all_episodes else f"episode_{args.episode:03d}"
    output_dir = Path(
        args.output_dir or _default_output_dir(args.model, args.dataset, scope)
    ).expanduser().resolve()
    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    model_dir, model_label = _resolve_model_source(args.model)
    dataset_root, dataset_label = _resolve_dataset_source(args.dataset)
    model_type = _resolve_model_type(args.model_type, model_dir)
    if args.temporal_ensemble and model_type != "act":
        raise ValueError("`--temporal-ensemble` is only supported for ACT")
    if args.enable_rtc and model_type not in {"smolvla", "pi0", "pi05"}:
        raise ValueError("`--enable-rtc` is only supported for SmolVLA, PI0 and PI0.5")
    if args.rtc_max_guidance_weight <= 0.0:
        raise ValueError("`--rtc-max-guidance-weight` must be > 0")
    if args.rtc_execution_horizon < 1:
        raise ValueError("`--rtc-execution-horizon` must be >= 1")
    if args.rtc_inference_delay_steps < 0:
        raise ValueError("`--rtc-inference-delay-steps` must be >= 0")
    if args.rtc_debug_maxlen < 1:
        raise ValueError("`--rtc-debug-maxlen` must be >= 1")

    dataset_repo_id = _dataset_repo_id_for_source(args.dataset, dataset_root)
    camera_map = _parse_camera_map(args.camera_map)

    LeRobotDataset = _load_dataset_class()
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=str(dataset_root),
        video_backend=args.video_backend,
    )
    all_actions, all_episode_indices = _load_action_metadata(dataset)
    gripper_mode = _detect_dataset_gripper_mode(all_actions, args.dataset_gripper_scale)

    episode_to_indices = _build_episode_to_indices(all_episode_indices)
    available_episode_ids = sorted(episode_to_indices.keys())
    episode_ids = _select_episode_ids(args, available_episode_ids)

    engine = _load_engine(model_type, model_dir, args, float(dataset.fps))

    try:
        metadata = _get_engine_metadata(engine)
        if all_actions.shape[1] != metadata.action_dim:
            raise ValueError(
                f"Dataset action dim ({all_actions.shape[1]}) does not match model action dim ({metadata.action_dim})"
            )

        _print_run_header(
            model_type=model_type,
            model_dir=model_dir,
            model_label=model_label,
            dataset_root=dataset_root,
            dataset_label=dataset_label,
            dataset=dataset,
            metadata=metadata,
            gripper_mode=gripper_mode,
            episode_ids=episode_ids,
            output_dir=output_dir,
            device=args.device,
            requested_execution_mode=args.execution_mode,
            resolved_execution_mode=execution_mode,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            rtc_enabled=args.enable_rtc,
        )

        episode_results: list[EpisodeResult] = []
        overall_accumulator = ErrorAccumulator(action_dim=int(metadata.action_dim))

        for episode_index in episode_ids:
            episode_indices = episode_to_indices[episode_index]
            if args.max_frames is not None:
                episode_indices = episode_indices[: max(0, args.max_frames)]
            if len(episode_indices) == 0:
                print(f"Episode {episode_index} skipped: no frames selected")
                print("-" * 80)
                continue
            print(f"Episode {episode_index} start: {len(episode_indices)} frames")

            result = _run_episode_validation(
                engine=engine,
                dataset=dataset,
                dataset_indices=episode_indices,
                all_actions=all_actions,
                metadata=metadata,
                model_type=model_type,
                gripper_mode=gripper_mode,
                explicit_instruction=args.instruction,
                execution_mode=execution_mode,
                camera_map=camera_map,
            )

            for prediction, target in zip(result.predictions, result.targets, strict=True):
                overall_accumulator.update(prediction, target)

            plot_path = plots_dir / f"episode_{episode_index:03d}.png"
            csv_path = csv_dir / f"episode_{episode_index:03d}.csv"
            _plot_episode_curves(
                episode_result=result,
                model_label=model_label,
                dataset_label=dataset_label,
                output_path=plot_path,
            )
            _write_episode_csv(result, csv_path)

            result.plot_path = str(plot_path)
            result.csv_path = str(csv_path)
            episode_results.append(result)

            print(
                f"Episode {episode_index} done: frames={len(result.frame_indices)} "
                f"mae={result.metrics.get('mae', float('nan')):.6f} "
                f"rmse={result.metrics.get('rmse', float('nan')):.6f} "
                f"max_abs={result.metrics.get('max_abs', float('nan')):.6f} "
                f"avg_call_ms={result.average_call_ms:.2f} "
                f"plot={plot_path.name}"
            )
            print("-" * 80)
    finally:
        engine.unload()

    overall_metrics = overall_accumulator.as_dict()
    summary = {
        "model_type": model_type,
        "model_source": model_label,
        "model_dir": str(model_dir),
        "dataset_source": dataset_label,
        "dataset_root": str(dataset_root),
        "device": args.device,
        "control_fps": float(dataset.fps),
        "dataset_gripper_scale": gripper_mode,
        "camera_map": camera_map,
        "requested_execution_mode": args.execution_mode,
        "resolved_execution_mode": execution_mode,
        "temporal_ensemble_enabled": args.temporal_ensemble,
        "temporal_ensemble_coeff": temporal_ensemble_coeff,
        "rtc_enabled": args.enable_rtc,
        "rtc_prefix_attention_schedule": args.rtc_prefix_attention_schedule,
        "rtc_max_guidance_weight": args.rtc_max_guidance_weight,
        "rtc_execution_horizon": args.rtc_execution_horizon,
        "rtc_inference_delay_steps": args.rtc_inference_delay_steps,
        "episodes": [
            {
                "episode_index": result.episode_index,
                "task": result.task,
                "num_frames": len(result.frame_indices),
                "metrics": result.metrics,
                "average_call_ms": result.average_call_ms,
                "plot_path": result.plot_path,
                "csv_path": result.csv_path,
            }
            for result in episode_results
        ],
        "overall": overall_metrics,
    }

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_summary_csv(episode_results, output_dir / "summary.csv")

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Episodes validated: {[result.episode_index for result in episode_results]}")
    print(
        f"overall mae={overall_metrics.get('mae', float('nan')):.6f} "
        f"rmse={overall_metrics.get('rmse', float('nan')):.6f} "
        f"max_abs={overall_metrics.get('max_abs', float('nan')):.6f}"
    )
    print(f"Plots dir: {plots_dir}")
    print(f"Episode CSV dir: {csv_dir}")
    print(f"Summary JSON: {summary_json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
