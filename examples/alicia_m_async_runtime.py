#!/usr/bin/env python3
"""Run async policy inference directly with Alicia-M-SDK.

The inference SDK owns policy loading and async action scheduling. This example
keeps Alicia-M hardware I/O in the script: connect the robot, read state, call
the runtime, and publish the selected action.

Example:
    python examples/alicia_m_async_runtime.py \
      --model-type act \
      --checkpoint-dir models/ACT_pick_and_place_v2 \
      --device cuda:0 \
      --port /dev/ttyACM0 \
      --camera head=0 \
      --camera wrist=2 \
      --fps 30
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_sdk import AsyncInferenceConfig, SUPPORTED_MODEL_TYPES, get_global_async_runtime  # noqa: E402


class OpenCVCameraReader:
    """Small OpenCV camera reader returning BGR images keyed by camera role."""

    def __init__(
        self,
        cameras: Mapping[str, str],
        *,
        width: int | None = None,
        height: int | None = None,
        warmup_frames: int = 5,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for camera capture: pip install opencv-python") from exc

        self._captures = {}
        for role, source in cameras.items():
            capture_source = int(source) if source.isdigit() else source
            cap = cv2.VideoCapture(capture_source)
            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not cap.isOpened():
                self.close()
                raise RuntimeError(f"Failed to open camera '{role}' from source {source!r}")
            self._captures[role] = cap

        for _ in range(max(0, warmup_frames)):
            self.read()

    def read(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for role, cap in self._captures.items():
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame from camera '{role}'")
            frames[role] = frame
        return frames

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()

    def __enter__(self) -> "OpenCVCameraReader":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def load_alicia_m_sdk() -> Any:
    sdk_path = os.environ.get("ALICIA_M_SDK_PATH")
    candidates = [Path(sdk_path).expanduser()] if sdk_path else []
    candidates.append(REPO_ROOT.parent / "Alicia-M-SDK")

    for candidate in candidates:
        if candidate.is_dir():
            candidate_str = str(candidate.resolve())
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    try:
        return importlib.import_module("alicia_m_sdk")
    except ImportError as exc:
        raise RuntimeError(
            "Alicia-M-SDK is not importable. Install it, place it next to this "
            "repo as ../Alicia-M-SDK, or set ALICIA_M_SDK_PATH."
        ) from exc


def create_robot(alicia_m_sdk: Any, port: str) -> Any:
    robot = alicia_m_sdk.create_robot(port=port, auto_connect=False)
    robot.connect(timeout=5.0)
    return robot


def read_robot_state(robot: Any) -> np.ndarray:
    state = robot.get_robot_state("all")
    angles = getattr(state, "angles", None) if state is not None else None
    gripper = getattr(state, "gripper", None) if state is not None else None

    if angles is None or gripper is None:
        joint_gripper = robot.get_robot_state("joint_gripper")
        if isinstance(joint_gripper, Mapping):
            angles = joint_gripper.get("angles")
            gripper = joint_gripper.get("gripper")

    if angles is None or gripper is None:
        raise RuntimeError("Alicia-M state cache is empty; wait for polling before inference.")

    result = np.asarray(list(angles) + [float(gripper)], dtype=np.float32)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise RuntimeError(f"Alicia-M state must be finite [6 joints, gripper], got {result.shape}.")
    return result


def wait_for_robot_state(robot: Any, timeout: float = 2.0) -> np.ndarray:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return read_robot_state(robot)
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.02)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Timed out waiting for Alicia-M state.")


def publish_robot_action(robot: Any, action: np.ndarray, current_state: np.ndarray) -> bool:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.shape[0] not in (6, 7):
        raise ValueError(f"Alicia-M action must contain 6 or 7 values, got {action_array.shape[0]}.")
    if not np.all(np.isfinite(action_array)):
        raise ValueError("Alicia-M action contains NaN or Inf.")

    joints = action_array[:6]
    gripper = float(action_array[6]) if action_array.shape[0] == 7 else float(current_state[6])
    gripper = float(np.clip(gripper, 0.0, 1000.0))

    return bool(
        robot.set_robot_state(
            target_joints=joints.tolist(),
            gripper_value=gripper,
            joint_format="rad",
            wait_for_completion=False,
            use_interpolation=False,
        )
    )


def hold_current(robot: Any) -> None:
    state = read_robot_state(robot)
    publish_robot_action(robot, state, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async policy inference loop for Alicia-M real robot.")
    parser.add_argument("--model-type", required=True, choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--chunk-size-threshold",
        dest="chunk_size_threshold",
        type=float,
        default=0.5,
        help="Queue refill threshold for async inference.",
    )
    parser.add_argument("--action-chunk-size", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="Enable ACT temporal ensembling. Only valid when --model-type act.",
    )
    parser.add_argument(
        "--enable-rtc",
        action="store_true",
        help="Enable RTC. Only valid for SmolVLA, PI0 and PI0.5 policies.",
    )
    parser.add_argument("--port", default="", help="Alicia-M serial port. Empty means SDK auto-discovery.")
    parser.add_argument(
        "--camera",
        action="append",
        required=True,
        metavar="ROLE=SOURCE",
        help="Camera role and OpenCV source, for example head=0 or wrist=/dev/video2.",
    )
    return parser.parse_args()


def parse_cameras(values: list[str]) -> dict[str, str]:
    cameras: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --camera value {item!r}; expected ROLE=SOURCE")
        role, source = item.split("=", 1)
        role = role.strip()
        source = source.strip()
        if not role or not source:
            raise ValueError(f"Invalid --camera value {item!r}; expected ROLE=SOURCE")
        cameras[role] = source
    return cameras


def state_dim_for_runtime(metadata: Any) -> int:
    return min(7, int(metadata.state_dim) if metadata.state_dim else 7)


def main() -> None:
    args = parse_args()
    if args.temporal_ensemble and args.model_type != "act":
        raise ValueError("--temporal-ensemble is only supported for ACT")
    if args.enable_rtc and args.model_type not in {"smolvla", "pi0", "pi05"}:
        raise ValueError("--enable-rtc is only supported for SmolVLA, PI0 and PI0.5")

    robot: Any | None = None
    camera_reader: OpenCVCameraReader | None = None
    runtime = get_global_async_runtime()
    last_status_time = 0.0
    action_dim_for_fallback = 7

    def safe_hold_action(_state: np.ndarray | None) -> np.ndarray:
        if robot is None or not robot.is_connected():
            return np.zeros(action_dim_for_fallback, dtype=np.float32)
        return read_robot_state(robot)[:action_dim_for_fallback].astype(np.float32, copy=True)

    try:
        alicia_m_sdk = load_alicia_m_sdk()
        robot = create_robot(alicia_m_sdk, args.port)
        camera_reader = OpenCVCameraReader(parse_cameras(args.camera))

        metadata = runtime.load_policy(
            algorithm_type=args.model_type,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            instruction=args.instruction,
            config=AsyncInferenceConfig(
                control_fps=args.fps,
                chunk_size_threshold=args.chunk_size_threshold,
                action_chunk_size=args.action_chunk_size,
                n_action_steps=args.n_action_steps,
                fallback_mode="hold",
                safe_action_fn=safe_hold_action,
                enable_temporal_ensemble=args.temporal_ensemble,
                enable_rtc=args.enable_rtc,
            ),
        )
        if metadata.action_dim not in (6, 7):
            raise ValueError(f"Alicia-M publisher supports action_dim 6 or 7, got {metadata.action_dim}.")
        if metadata.state_dim and metadata.state_dim not in (6, 7):
            raise ValueError(f"Alicia-M state reader supports state_dim 6 or 7, got {metadata.state_dim}.")
        action_dim_for_fallback = int(metadata.action_dim)

        print(
            "loaded model=%s action_dim=%d state_dim=%d cameras=%s"
            % (
                metadata.model_type,
                metadata.action_dim,
                metadata.state_dim,
                ",".join(metadata.required_cameras),
            ),
            flush=True,
        )

        initial_state = wait_for_robot_state(robot, timeout=2.0)
        runtime.warmup(
            images=camera_reader.read(),
            state=initial_state[: state_dim_for_runtime(metadata)],
            instruction=args.instruction,
        )
        runtime.start()
        if not runtime.wait_until_ready(min_queue_size=1, timeout=5.0):
            runtime.stop()
            raise RuntimeError("Async runtime did not produce an initial action before startup timeout.")

        while True:
            tick_start = time.monotonic()
            current_state = read_robot_state(robot)
            result = runtime.step(
                images=camera_reader.read(),
                state=current_state[: state_dim_for_runtime(metadata)],
                instruction=args.instruction,
            )
            publish_robot_action(robot, result.action, current_state)

            now = time.monotonic()
            if now - last_status_time >= 1.0:
                last_status_time = now
                status = runtime.get_status()
                print(
                    "tick=%d source=%s queue=%d latency=%.1fms fallbacks=%d errors=%d"
                    % (
                        result.timestep,
                        result.source,
                        result.queue_size,
                        status.latency_estimate * 1000.0,
                        status.fallback_count,
                        status.error_count,
                    ),
                    flush=True,
                )

            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, 1.0 / args.fps - elapsed))
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
    finally:
        runtime.stop()
        runtime.close()
        try:
            if robot is not None and robot.is_connected():
                hold_current(robot)
        except Exception as exc:
            print(f"Failed to hold Alicia-M during shutdown: {exc}", flush=True)
        if camera_reader is not None:
            camera_reader.close()
        if robot is not None and robot.is_connected():
            robot.disconnect()


if __name__ == "__main__":
    main()
