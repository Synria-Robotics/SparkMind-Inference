#!/usr/bin/env python3
"""Compare SDK SmolVLA inference against the official LeRobot policy path."""

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

from inference_sdk import InferenceSDK, SmoothingConfig  # noqa: E402
from validate_dataset_inference import (  # noqa: E402
    _adapt_state_for_sdk,
    _parse_camera_map,
    _resolve_dataset_source,
    _resolve_image_key,
    _resolve_model_source,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lerobot/smolvla_libero")
    parser.add_argument("--dataset", default="lerobot/libero")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
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
        default=["camera1=image", "camera2=image2"],
        help="Map SDK/official policy camera role to dataset camera key.",
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


def _load_official_stack(model_dir: Path, dataset: Any, device: str, rename_map: dict[str, str]):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(model_dir)
    cfg.device = device
    cfg.pretrained_path = str(model_dir)

    policy = make_policy(cfg, ds_meta=dataset.meta, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(model_dir),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
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
) -> np.ndarray:
    observation = {key: value for key, value in item.items() if key.startswith("observation.") or key == "task"}
    batch = preprocessor(observation)
    _set_seed(seed, device)
    with torch.inference_mode():
        normalized_chunk = policy.predict_action_chunk(batch)
        action_chunk = postprocessor(normalized_chunk)
    return action_chunk.detach().cpu().numpy()[0].astype(np.float32)


def _sdk_predict_chunk(
    *,
    sdk: InferenceSDK,
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
    chunk = sdk.predict_action_chunk("smolvla", images=images, state=state, instruction=instruction)
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
    dataset_root, dataset_label = _resolve_dataset_source(args.dataset)
    camera_map = _parse_camera_map(args.camera_map)
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

    official_policy, official_pre, official_post = _load_official_stack(model_dir, dataset, args.device, rename_map)

    sdk = InferenceSDK(
        device=args.device,
        smoothing_config=SmoothingConfig(control_fps=float(dataset.meta.fps)),
    )
    metadata = sdk.load_policy("smolvla", str(model_dir))
    action_dim = int(metadata.action_dim)
    required_cameras = list(metadata.required_cameras)
    _write_csv_header(csv_path, action_dim)

    first_action_stats = CompareStats(action_dim=action_dim)
    chunk_stats = CompareStats(action_dim=action_dim)
    frame_count = min(len(dataset), args.max_frames)

    try:
        for local_index in range(frame_count):
            item = dataset[local_index]
            if not args.preserve_policy_state:
                official_policy.reset()
                sdk._get_policy("smolvla").reset()
            official_chunk = _official_predict_chunk(
                policy=official_policy,
                preprocessor=official_pre,
                postprocessor=official_post,
                item=item,
                seed=args.seed,
                device=args.device,
            )
            sdk_chunk = _sdk_predict_chunk(
                sdk=sdk,
                item=item,
                required_cameras=required_cameras,
                camera_map=camera_map,
                gripper_mode=args.dataset_gripper_scale,
                seed=args.seed,
                device=args.device,
            )
            if sdk_chunk.shape != official_chunk.shape:
                raise ValueError(f"Chunk shape mismatch: SDK {sdk_chunk.shape} vs official {official_chunk.shape}")

            dataset_index = int(item["index"])
            first_action_stats.update(sdk_chunk[0], official_chunk[0])
            _append_csv(csv_path, local_index, dataset_index, 0, sdk_chunk[0], official_chunk[0])

            if args.compare_full_chunk:
                for chunk_step in range(sdk_chunk.shape[0]):
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
