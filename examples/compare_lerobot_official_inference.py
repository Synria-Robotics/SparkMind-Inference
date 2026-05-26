#!/usr/bin/env python3
"""Compare SDK inference against an official LeRobot or SparkMind policy path."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_sparkmind_path(repo_root: Path) -> None:
    for candidate in (repo_root / "third_party" / "SparkMind", repo_root.parent / "SparkMind", repo_root / "SparkMind"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


_ensure_sparkmind_path(REPO_ROOT)

from inference_sdk import InferenceSDK, SUPPORTED_MODEL_TYPES, SmoothingConfig  # noqa: E402
from validate_dataset_inference import (  # noqa: E402
    _adapt_state_for_sdk,
    _parse_camera_map,
    _resolve_dataset_source,
    _resolve_image_key,
    _resolve_model_source,
    _resolve_model_type,
    _tensor_image_to_bgr_uint8,
)


@dataclass
class CompareStats:
    action_dim: int
    count: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    max_abs: float = 0.0
    per_dim_abs_sum: np.ndarray | None = None
    per_dim_sq_sum: np.ndarray | None = None

    def update(self, sdk_action: np.ndarray, official_action: np.ndarray) -> None:
        diff = np.asarray(sdk_action, dtype=np.float32) - np.asarray(official_action, dtype=np.float32)
        abs_diff = np.abs(diff)
        sq_diff = np.square(diff)
        if self.per_dim_abs_sum is None:
            self.per_dim_abs_sum = np.zeros(self.action_dim, dtype=np.float64)
            self.per_dim_sq_sum = np.zeros(self.action_dim, dtype=np.float64)
        self.count += 1
        self.abs_sum += float(abs_diff.sum())
        self.sq_sum += float(sq_diff.sum())
        self.max_abs = max(self.max_abs, float(abs_diff.max(initial=0.0)))
        self.per_dim_abs_sum += abs_diff
        self.per_dim_sq_sum += sq_diff

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        assert self.per_dim_abs_sum is not None
        assert self.per_dim_sq_sum is not None
        values = self.count * self.action_dim
        return {
            "count": self.count,
            "mae": self.abs_sum / values,
            "rmse": float(np.sqrt(self.sq_sum / values)),
            "max_abs": self.max_abs,
            "per_dim_mae": (self.per_dim_abs_sum / self.count).tolist(),
            "per_dim_rmse": np.sqrt(self.per_dim_sq_sum / self.count).tolist(),
        }


@dataclass
class OfficialRTCState:
    prev_chunk_left_over: torch.Tensor | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lerobot/smolvla_libero")
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default=None)
    parser.add_argument("--dataset", default="lerobot/libero")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--official-backend",
        choices=("lerobot", "sparkmind"),
        default="lerobot",
        help="Reference implementation. Use sparkmind for the vendored LeRobot 0.5.x PI0/PI0.5 stack.",
    )
    parser.add_argument("--compare-full-chunk", action="store_true")
    parser.add_argument(
        "--preserve-policy-state",
        action="store_true",
        help="Keep policy queues across frames. By default each frame is compared as an independent forward pass.",
    )
    parser.add_argument("--dataset-gripper-scale", choices=("normalized", "raw"), default="normalized")
    parser.add_argument(
        "--camera-map",
        action="append",
        default=None,
        help="Map SDK/official policy camera role to dataset camera key.",
    )
    parser.add_argument("--enable-rtc", action="store_true")
    parser.add_argument("--rtc-execution-horizon", type=int, default=10)
    parser.add_argument("--rtc-inference-delay-steps", type=int, default=0)
    parser.add_argument("--rtc-prefix-attention-schedule", default="LINEAR")
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--official-tokenizer-name",
        default=None,
        help="Optional tokenizer override for the official LeRobot preprocessor. Default keeps the checkpoint policy behavior.",
    )
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _default_output_dir(model: str, dataset: str, episode: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.replace("/", "_")
    dataset_name = dataset.replace("/", "_")
    return REPO_ROOT / "outputs" / "official_compare" / f"{stamp}_{model_name}_{dataset_name}_ep{episode:03d}"


def _set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _official_modules(backend: str):
    if backend == "sparkmind":
        from sparkmind.lerobot_compat.configs.policies import PreTrainedConfig
        from sparkmind.lerobot_compat.configs.types import RTCAttentionSchedule
        from sparkmind.lerobot_compat.policies.factory import make_policy, make_pre_post_processors
        from sparkmind.lerobot_compat.policies.rtc.configuration_rtc import RTCConfig
    else:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

    return PreTrainedConfig, RTCAttentionSchedule, RTCConfig, make_policy, make_pre_post_processors


def _make_official_rtc_config(args: argparse.Namespace, backend: str):
    if not args.enable_rtc:
        return None
    _, RTCAttentionSchedule, RTCConfig, _, _ = _official_modules(backend)

    return RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule(str(args.rtc_prefix_attention_schedule).upper()),
        max_guidance_weight=float(args.rtc_max_guidance_weight),
        execution_horizon=int(args.rtc_execution_horizon),
    )


def _official_preprocessor_overrides(device: str, rename_map: dict[str, str], official_tokenizer_name: str | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    if official_tokenizer_name:
        overrides["tokenizer_processor"] = {"tokenizer_name": official_tokenizer_name}
    return overrides


def _load_official_stack(model_dir: Path, dataset: Any, device: str, rename_map: dict[str, str], args: argparse.Namespace, model_type: str):
    PreTrainedConfig, _, _, make_policy, make_pre_post_processors = _official_modules(args.official_backend)

    cfg = PreTrainedConfig.from_pretrained(model_dir)
    cfg.device = device
    pretrained_path = str(model_dir)
    cfg.pretrained_path = pretrained_path
    cfg.rtc_config = _make_official_rtc_config(args, args.official_backend)

    policy = make_policy(cfg, ds_meta=dataset.meta, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(model_dir),
        dataset_stats=getattr(dataset.meta, "stats", None),
        preprocessor_overrides=_official_preprocessor_overrides(device, rename_map, args.official_tokenizer_name),
    )
    return policy, preprocessor, postprocessor


def _official_predict_chunk(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    item: dict[str, Any],
    seed: int,
    device: str,
    rtc_state: OfficialRTCState | None,
    rtc_execution_horizon: int,
    rtc_inference_delay_steps: int,
) -> np.ndarray:
    observation = {key: value for key, value in item.items() if key.startswith("observation.") or key == "task"}
    batch = preprocessor(observation)
    rtc_kwargs: dict[str, Any] = {}
    if rtc_state is not None:
        rtc_kwargs = {
            "prev_chunk_left_over": rtc_state.prev_chunk_left_over,
            "inference_delay": int(rtc_inference_delay_steps),
            "execution_horizon": int(rtc_execution_horizon),
        }
    _set_seed(seed, device)
    context = torch.enable_grad() if rtc_state is not None else torch.inference_mode()
    with context:
        normalized_chunk = policy.predict_action_chunk(batch, **rtc_kwargs)
        if rtc_state is not None:
            delay = max(0, int(rtc_inference_delay_steps))
            rtc_state.prev_chunk_left_over = normalized_chunk.detach()[:, delay:].clone() if delay else normalized_chunk.detach().clone()
        action_chunk = postprocessor(normalized_chunk)
    return action_chunk.detach().cpu().numpy()[0].astype(np.float32)


def _sdk_predict_chunk(
    *,
    sdk: InferenceSDK,
    model_type: str,
    item: dict[str, Any],
    required_cameras: list[str],
    camera_map: dict[str, str],
    gripper_mode: str,
    seed: int,
    device: str,
) -> np.ndarray:
    images = {
        camera: _tensor_image_to_bgr_uint8(item[_resolve_image_key(item, camera_map.get(camera, camera))])
        for camera in required_cameras
    }
    state = _adapt_state_for_sdk(item["observation.state"].detach().cpu().numpy(), gripper_mode)
    instruction = str(item["task"])
    _set_seed(seed, device)
    chunk = sdk.predict_action_chunk(model_type, images=images, state=state, instruction=instruction)
    chunk = np.asarray(chunk, dtype=np.float32).copy()
    if gripper_mode == "normalized" and chunk.shape[-1] > 0:
        chunk[..., -1] /= 1000.0
    return chunk


def _write_csv_header(path: Path, action_dim: int) -> None:
    fieldnames = ["frame", "dataset_index", "chunk_step", "mae", "rmse", "max_abs"]
    fieldnames += [f"official_{i}" for i in range(action_dim)]
    fieldnames += [f"sdk_{i}" for i in range(action_dim)]
    fieldnames += [f"diff_{i}" for i in range(action_dim)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def _append_csv(path: Path, frame: int, dataset_index: int, chunk_step: int, sdk_action: np.ndarray, official_action: np.ndarray) -> None:
    diff = sdk_action - official_action
    fieldnames = ["frame", "dataset_index", "chunk_step", "mae", "rmse", "max_abs"]
    fieldnames += [f"official_{i}" for i in range(official_action.shape[0])]
    fieldnames += [f"sdk_{i}" for i in range(sdk_action.shape[0])]
    fieldnames += [f"diff_{i}" for i in range(diff.shape[0])]
    row: dict[str, Any] = {
        "frame": frame,
        "dataset_index": dataset_index,
        "chunk_step": chunk_step,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs": float(np.max(np.abs(diff))),
    }
    row.update({f"official_{i}": float(value) for i, value in enumerate(official_action)})
    row.update({f"sdk_{i}": float(value) for i, value in enumerate(sdk_action)})
    row.update({f"diff_{i}": float(value) for i, value in enumerate(diff)})
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)


def main() -> int:
    args = _parse_args()
    model_dir, model_label = _resolve_model_source(args.model)
    model_type = _resolve_model_type(args.model_type, model_dir)
    dataset_root, dataset_label = _resolve_dataset_source(args.dataset)
    default_camera_map = ["camera1=image", "camera2=image2"] if model_type == "smolvla" else ["image=image", "image2=image2"]
    camera_map = _parse_camera_map(args.camera_map or default_camera_map)
    rename_map = {
        f"observation.images.{dataset_camera}": f"observation.images.{policy_camera}"
        for policy_camera, dataset_camera in camera_map.items()
    }
    output_dir = Path(args.output_dir or _default_output_dir(args.model, args.dataset, args.episode)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        dataset_label,
        root=dataset_root,
        episodes=[args.episode],
        video_backend=args.video_backend,
    )

    official_policy, official_pre, official_post = _load_official_stack(model_dir, dataset, args.device, rename_map, args, model_type)

    sdk = InferenceSDK(
        device=args.device,
        smoothing_config=SmoothingConfig(
            control_fps=float(dataset.meta.fps),
            enable_rtc=args.enable_rtc,
            rtc_execution_horizon=args.rtc_execution_horizon,
            rtc_inference_delay_steps=args.rtc_inference_delay_steps,
            rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        ),
    )
    dataset_stats_path = dataset_root / "meta" / "stats.json"
    if model_type in {"pi0", "pi05"} and dataset_stats_path.is_file():
        os.environ.setdefault("PI0_STATS_PATH", str(dataset_stats_path))
        os.environ.setdefault("PI05_STATS_PATH", str(dataset_stats_path))
    metadata = sdk.load_policy(model_type, str(model_dir))
    action_dim = int(metadata.action_dim)
    required_cameras = list(metadata.required_cameras)
    _write_csv_header(csv_path, action_dim)

    first_action_stats = CompareStats(action_dim=action_dim)
    chunk_stats = CompareStats(action_dim=action_dim)
    frame_count = min(len(dataset), args.max_frames)
    official_rtc_state = OfficialRTCState() if args.enable_rtc else None
    if args.enable_rtc and not args.preserve_policy_state:
        print("warning: --enable-rtc works best with --preserve-policy-state; resetting each frame disables RTC history.", flush=True)

    try:
        for local_index in range(frame_count):
            item = dataset[local_index]
            if not args.preserve_policy_state:
                official_policy.reset()
                sdk._get_policy(model_type).reset()
                if official_rtc_state is not None:
                    official_rtc_state.prev_chunk_left_over = None
            official_chunk = _official_predict_chunk(
                policy=official_policy,
                preprocessor=official_pre,
                postprocessor=official_post,
                item=item,
                seed=args.seed,
                device=args.device,
                rtc_state=official_rtc_state,
                rtc_execution_horizon=args.rtc_execution_horizon,
                rtc_inference_delay_steps=args.rtc_inference_delay_steps,
            )
            sdk_chunk = _sdk_predict_chunk(
                sdk=sdk,
                item=item,
                required_cameras=required_cameras,
                camera_map=camera_map,
                gripper_mode=args.dataset_gripper_scale,
                seed=args.seed,
                device=args.device,
                model_type=model_type,
            )
            if sdk_chunk.ndim != 2 or official_chunk.ndim != 2 or sdk_chunk.shape[-1] != official_chunk.shape[-1]:
                raise ValueError(f"Chunk shape mismatch: SDK {sdk_chunk.shape} vs official {official_chunk.shape}")

            dataset_index = int(item["index"])
            first_action_stats.update(sdk_chunk[0], official_chunk[0])
            _append_csv(csv_path, local_index, dataset_index, 0, sdk_chunk[0], official_chunk[0])

            if args.compare_full_chunk:
                common_steps = min(sdk_chunk.shape[0], official_chunk.shape[0])
                for chunk_step in range(common_steps):
                    chunk_stats.update(sdk_chunk[chunk_step], official_chunk[chunk_step])
                    if chunk_step != 0:
                        _append_csv(csv_path, local_index, dataset_index, chunk_step, sdk_chunk[chunk_step], official_chunk[chunk_step])

            frame_mae = float(np.mean(np.abs(sdk_chunk[0] - official_chunk[0])))
            print(
                f"frame={local_index:04d} dataset_index={dataset_index:06d} first_action_mae={frame_mae:.8f}",
                flush=True,
            )
    finally:
        sdk.close()

    summary = {
        "model": model_label,
        "model_type": model_type,
        "official_backend": args.official_backend,
        "model_dir": str(model_dir),
        "dataset": dataset_label,
        "dataset_root": str(dataset_root),
        "episode": args.episode,
        "frames": frame_count,
        "seed": args.seed,
        "device": args.device,
        "camera_map": camera_map,
        "rename_map": rename_map,
        "dataset_gripper_scale": args.dataset_gripper_scale,
        "rtc_enabled": args.enable_rtc,
        "rtc_execution_horizon": args.rtc_execution_horizon,
        "rtc_inference_delay_steps": args.rtc_inference_delay_steps,
        "compare_full_chunk": args.compare_full_chunk,
        "preserve_policy_state": args.preserve_policy_state,
        "required_cameras": required_cameras,
        "first_action": first_action_stats.as_dict(),
        "full_chunk": chunk_stats.as_dict() if args.compare_full_chunk else None,
        "csv_path": str(csv_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
