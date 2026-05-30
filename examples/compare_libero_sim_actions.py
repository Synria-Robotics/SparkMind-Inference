#!/usr/bin/env python3
"""Compare SDK and official LeRobot SmolVLA actions on the same LIBERO rollout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


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

from sparkmind_inference import InferenceSDK, SmoothingConfig  # noqa: E402
from sparkmind_inference.factory import SUPPORTED_MODEL_TYPES  # noqa: E402
from validate_dataset_inference import _resolve_model_source, _resolve_model_type, _tensor_image_to_bgr_uint8  # noqa: E402


@dataclass
class ActionStats:
    action_dim: int
    count: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    max_abs: float = 0.0
    per_dim_abs_sum: np.ndarray | None = None

    def update(self, sdk_action: np.ndarray, official_action: np.ndarray) -> None:
        diff = np.asarray(sdk_action, dtype=np.float32) - np.asarray(official_action, dtype=np.float32)
        abs_diff = np.abs(diff)
        if self.per_dim_abs_sum is None:
            self.per_dim_abs_sum = np.zeros(self.action_dim, dtype=np.float64)
        self.count += 1
        self.abs_sum += float(abs_diff.sum())
        self.sq_sum += float(np.square(diff).sum())
        self.max_abs = max(self.max_abs, float(abs_diff.max(initial=0.0)))
        self.per_dim_abs_sum += abs_diff

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        assert self.per_dim_abs_sum is not None
        return {
            "count": self.count,
            "mae": self.abs_sum / (self.count * self.action_dim),
            "rmse": float(np.sqrt(self.sq_sum / (self.count * self.action_dim))),
            "max_abs": self.max_abs,
            "per_dim_mae": (self.per_dim_abs_sum / self.count).tolist(),
        }


@dataclass
class OfficialRTCState:
    prev_chunk_left_over: torch.Tensor | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lerobot/smolvla_libero")
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default=None)
    parser.add_argument("--benchmark", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument("--execute", choices=("official", "sdk"), default="official")
    parser.add_argument(
        "--sdk-selection",
        choices=("step", "fifo", "raw"),
        default="step",
        help="SDK action selection path: SDK FIFO step(), explicit FIFO chunk queue, or chunk[0] every frame.",
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
    parser.add_argument(
        "--sdk-stats-path",
        default=None,
        help="Optional LeRobot stats.json path for PI0/PI0.5 SDK normalization.",
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=80.0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _default_output_dir(model: str, benchmark: str, task_id: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "libero_sim_action_compare" / f"{stamp}_{model.replace('/', '_')}_{benchmark}_task_{task_id:03d}"


def _set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_official_rtc_config(args: argparse.Namespace):
    if not args.enable_rtc:
        return None
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

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


def _load_official_policy(model_dir: Path, env_cfg: Any, device: str, rename_map: dict[str, str], args: argparse.Namespace, model_type: str):
    # Import policy config modules so LeRobot's PreTrainedConfig registry knows
    # about Hub checkpoints with `type=pi0/pi05/smolvla`.
    import lerobot.policies.pi0.configuration_pi0  # noqa: F401
    import lerobot.policies.pi05.configuration_pi05  # noqa: F401
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(model_dir)
    cfg.device = device
    cfg.pretrained_path = str(model_dir)
    cfg.rtc_config = _make_official_rtc_config(args)
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(model_dir),
        preprocessor_overrides=_official_preprocessor_overrides(device, rename_map, args.official_tokenizer_name),
    )
    return cfg, policy, preprocessor, postprocessor


def _processed_image_key(camera: str, model_type: str) -> str | None:
    if camera in {"camera1", "cam_1"}:
        return "observation.images.image"
    if camera in {"camera2", "cam_2"}:
        return "observation.images.image2"
    if camera in {"image", "image2"}:
        return f"observation.images.{camera}"
    if camera.startswith("empty_camera"):
        return None
    return f"observation.images.{camera}"


def _sdk_inputs_from_processed_obs(
    processed_obs: dict[str, Any],
    *,
    required_cameras: list[str],
    model_type: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    images: dict[str, np.ndarray] = {}
    for camera in required_cameras:
        key = _processed_image_key(camera, model_type)
        if key is not None and key in processed_obs:
            images[camera] = _tensor_image_to_bgr_uint8(processed_obs[key][0].detach().cpu())
    state = processed_obs["observation.state"][0].detach().cpu().numpy().astype(np.float32)
    if state.shape[-1] > 0:
        state[-1] *= 1000.0
    task = processed_obs["task"][0] if isinstance(processed_obs.get("task"), list) else str(processed_obs.get("task", ""))
    return images, state, task


def _official_predict_first_action(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    observation: dict[str, Any],
    rtc_state: OfficialRTCState | None,
    rtc_execution_horizon: int,
    rtc_inference_delay_steps: int,
) -> np.ndarray:
    batch = preprocessor(observation.copy())
    with torch.inference_mode():
        if rtc_state is None:
            action = policy.select_action(batch)
        else:
            normalized_chunk = policy.predict_action_chunk(
                batch,
                prev_chunk_left_over=rtc_state.prev_chunk_left_over,
                inference_delay=int(rtc_inference_delay_steps),
                execution_horizon=int(rtc_execution_horizon),
            )
            delay = max(0, int(rtc_inference_delay_steps))
            rtc_state.prev_chunk_left_over = normalized_chunk.detach()[:, delay:].clone() if delay else normalized_chunk.detach().clone()
            action = postprocessor(normalized_chunk)[:, 0]
            action = env_postprocessor({"action": action})["action"]
            return action.detach().cpu().numpy().astype(np.float32)

        action = postprocessor(action)
        action = env_postprocessor({"action": action})["action"]
        return action.detach().cpu().numpy().astype(np.float32)


def _write_csv_header(path: Path, action_dim: int) -> None:
    fieldnames = ["step", "reward", "terminated", "truncated", "success", "mae", "rmse", "max_abs"]
    fieldnames += [f"official_{i}" for i in range(action_dim)]
    fieldnames += [f"sdk_{i}" for i in range(action_dim)]
    fieldnames += [f"diff_{i}" for i in range(action_dim)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def _append_csv(
    path: Path,
    *,
    step: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    success: bool,
    official_action: np.ndarray,
    sdk_action: np.ndarray,
) -> None:
    diff = sdk_action - official_action
    fieldnames = ["step", "reward", "terminated", "truncated", "success", "mae", "rmse", "max_abs"]
    fieldnames += [f"official_{i}" for i in range(official_action.shape[0])]
    fieldnames += [f"sdk_{i}" for i in range(sdk_action.shape[0])]
    fieldnames += [f"diff_{i}" for i in range(diff.shape[0])]
    row: dict[str, Any] = {
        "step": step,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "success": success,
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
    if args.enable_rtc and args.sdk_selection == "step":
        raise ValueError("--enable-rtc is not supported with --sdk-selection step; use fifo/raw chunk selection")
    output_dir = Path(args.output_dir or _default_output_dir(args.model, args.benchmark, args.task_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "action_comparison.csv"

    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation

    rename_map = (
        {
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        }
        if model_type == "smolvla"
        else {}
    )
    env_cfg = make_env_config(
        "libero",
        task=args.benchmark,
        task_ids=[args.task_id],
        observation_height=args.image_size,
        observation_width=args.image_size,
    )
    policy_cfg, official_policy, official_pre, official_post = _load_official_policy(model_dir, env_cfg, args.device, rename_map, args, model_type)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[args.benchmark][args.task_id]

    sdk = InferenceSDK(
        device=args.device,
        smoothing_config=SmoothingConfig(
            control_fps=float(env_cfg.fps),
            n_action_steps=policy_cfg.n_action_steps,
            enable_rtc=args.enable_rtc,
            rtc_execution_horizon=args.rtc_execution_horizon,
            rtc_inference_delay_steps=args.rtc_inference_delay_steps,
            rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        ),
    )
    if model_type in {"pi0", "pi05"} and args.sdk_stats_path:
        os.environ.setdefault("PI0_STATS_PATH", str(Path(args.sdk_stats_path).expanduser()))
        os.environ.setdefault("PI05_STATS_PATH", str(Path(args.sdk_stats_path).expanduser()))
    metadata = sdk.load_policy(model_type, str(model_dir))
    action_dim = int(metadata.action_dim)
    required_cameras = list(metadata.required_cameras)
    _write_csv_header(csv_path, action_dim)
    stats = ActionStats(action_dim=action_dim)

    video_path = output_dir / "rollout_render.mp4" if args.save_video else None
    video_writer = None

    try:
        official_policy.reset()
        sdk._get_policy(model_type).reset()
        observation, _ = env.reset(seed=[args.seed])
        sdk_fifo: deque[np.ndarray] = deque()
        official_rtc_state = OfficialRTCState() if args.enable_rtc else None
        total_reward = 0.0
        success = False
        steps_taken = 0

        for step in range(args.max_steps):
            policy_observation = preprocess_observation(observation)
            policy_observation = add_envs_task(env, policy_observation)
            policy_observation = env_pre(policy_observation)

            sdk_images, sdk_state, instruction = _sdk_inputs_from_processed_obs(
                policy_observation,
                required_cameras=required_cameras,
                model_type=model_type,
            )
            _set_seed(args.seed + step, args.device)
            official_action_np = _official_predict_first_action(
                policy=official_policy,
                preprocessor=official_pre,
                postprocessor=official_post,
                env_postprocessor=env_post,
                observation=policy_observation,
                rtc_state=official_rtc_state,
                rtc_execution_horizon=args.rtc_execution_horizon,
                rtc_inference_delay_steps=args.rtc_inference_delay_steps,
            )

            _set_seed(args.seed + step, args.device)
            if args.sdk_selection == "step":
                sdk_action = sdk.predict_action(model_type, images=sdk_images, state=sdk_state, instruction=instruction)
            elif args.sdk_selection == "raw":
                sdk_action = sdk.predict_action_chunk(model_type, images=sdk_images, state=sdk_state, instruction=instruction)[0]
            else:
                if not sdk_fifo:
                    sdk_chunk = sdk.predict_action_chunk(model_type, images=sdk_images, state=sdk_state, instruction=instruction)
                    sdk_fifo.extend(np.asarray(item, dtype=np.float32).reshape(-1) for item in sdk_chunk)
                sdk_action = sdk_fifo.popleft()
            sdk_action_np = np.asarray(sdk_action, dtype=np.float32).reshape(1, -1)
            if sdk_action_np.shape[-1] > 0:
                sdk_action_np[..., -1] /= 1000.0

            official_single = official_action_np[0]
            sdk_single = sdk_action_np[0]
            stats.update(sdk_single, official_single)

            action_to_execute = official_action_np if args.execute == "official" else sdk_action_np
            action_to_execute = np.clip(action_to_execute, -1.0, 1.0).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action_to_execute)
            step_reward = float(np.asarray(reward).reshape(-1)[0])
            total_reward += step_reward
            terminated_bool = bool(np.asarray(terminated).reshape(-1)[0])
            truncated_bool = bool(np.asarray(truncated).reshape(-1)[0])
            final_success = False
            if "final_info" in info:
                final_success = bool(info.get("final_info", {}).get("is_success", [False])[0])
            success = bool(terminated_bool or final_success)
            steps_taken = step + 1

            _append_csv(
                csv_path,
                step=step,
                reward=step_reward,
                terminated=terminated_bool,
                truncated=truncated_bool,
                success=success,
                official_action=official_single,
                sdk_action=sdk_single,
            )

            if video_path is not None:
                import cv2

                frame_rgb = env.envs[0].render()
                frame = frame_rgb[:, :, ::-1].copy()
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, args.video_fps, (frame.shape[1], frame.shape[0]))
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {video_path}")
                video_writer.write(frame)

            diff_mae = float(np.mean(np.abs(sdk_single - official_single)))
            print(
                f"step={step:04d} execute={args.execute} reward={step_reward:.3f} "
                f"success={success} sdk_selection={args.sdk_selection} action_mae={diff_mae:.6f}",
                flush=True,
            )
            if terminated_bool or truncated_bool:
                break
    finally:
        if video_writer is not None:
            video_writer.release()
        sdk.close()
        close_envs(envs)

    summary = {
        "model": model_label,
        "model_type": model_type,
        "model_dir": str(model_dir),
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "execute": args.execute,
        "sdk_selection": args.sdk_selection,
        "rtc_enabled": args.enable_rtc,
        "rtc_execution_horizon": args.rtc_execution_horizon,
        "rtc_inference_delay_steps": args.rtc_inference_delay_steps,
        "required_cameras": required_cameras,
        "seed": args.seed,
        "steps": steps_taken,
        "success": success,
        "total_reward": total_reward,
        "action": stats.as_dict(),
        "csv_path": str(csv_path),
        "video_path": str(video_path) if video_path is not None else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
