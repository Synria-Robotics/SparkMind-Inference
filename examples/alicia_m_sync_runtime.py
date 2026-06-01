#!/usr/bin/env python3
"""Run synchronous policy inference directly with Alicia-M-SDK.

The inference SDK owns policy loading and action selection. This example keeps
Alicia-M hardware I/O in the script: connect the robot, read state, call
InferenceSDK.predict_action(), and publish the selected action.

Example:
    python examples/alicia_m_sync_runtime.py \
      --model-type act \
      --checkpoint-dir /path/to/act_checkpoint \
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

from sparkmind_inference import InferenceSDK, SmoothingConfig  # noqa: E402
from sparkmind_inference.factory import SUPPORTED_MODEL_TYPES  # noqa: E402


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


def robot_state_to_policy_state(state: np.ndarray) -> np.ndarray:
    policy_state = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    if policy_state.shape[0] < 6:
        raise ValueError(f"Alicia-M state must contain at least 6 joints, got {policy_state.shape[0]}.")
    policy_state[:6] = np.degrees(policy_state[:6])
    return policy_state


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
            joint_format="deg",
            wait_for_completion=False,
            use_interpolation=False,
        )
    )


def hold_current(robot: Any) -> None:
    state = read_robot_state(robot)
    robot.set_robot_state(
        target_joints=state[:6].tolist(),
        gripper_value=float(state[6]),
        joint_format="rad",
        wait_for_completion=False,
        use_interpolation=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronous policy inference loop for Alicia-M.")
    parser.add_argument("--model-type", required=True, choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
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
        help="Deprecated for this synchronous predict_action() loop. RTC requires an external chunk action queue.",
    )
    parser.add_argument("--port", default="", help="Alicia-M serial port. Empty means SDK auto-discovery.")
    parser.add_argument(
        "--camera",
        action="append",
        required=True,
        metavar="ROLE=SOURCE",
        help="Camera role and OpenCV source, for example head=0 or wrist=/dev/video2.",
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="Print current policy state, target action and delta once per status interval.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and status logging without publishing actions to the robot.",
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


def state_dim_for_metadata(metadata: Any) -> int:
    return min(7, int(metadata.state_dim) if metadata.state_dim else 7)


def _looks_cartesian(value: Any) -> bool:
    text = str(value or "").lower()
    return any(token in text for token in ("ee", "eef", "tcp", "pose", "cartesian"))


def validate_alicia_m_robot_io(robot_io: Mapping[str, Any] | None) -> None:
    if not robot_io:
        return

    state_type = robot_io.get("state_type")
    action_type = robot_io.get("action_type")
    if _looks_cartesian(state_type) or _looks_cartesian(action_type):
        raise ValueError(
            "This Alicia-M example only supports joint-space state/action. "
            f"robot_io declares state_type={state_type!r}, action_type={action_type!r}."
        )

    gripper_range = robot_io.get("gripper_range")
    if gripper_range is not None and list(gripper_range) != [0, 1000]:
        print(
            "warning: Alicia-M example publishes gripper targets in [0, 1000], "
            f"but robot_io declares gripper_range={gripper_range!r}.",
            flush=True,
        )


def format_array(values: np.ndarray, precision: int = 2) -> str:
    return "[" + ",".join(f"{float(value):.{precision}f}" for value in values) + "]"


def main() -> None:
    args = parse_args()
    if args.temporal_ensemble and args.model_type != "act":
        raise ValueError("--temporal-ensemble is only supported for ACT")
    if args.enable_rtc:
        raise ValueError("--enable-rtc is not supported by this predict_action() loop; use chunk inference with an RTC-aware queue")

    robot: Any | None = None
    camera_reader: OpenCVCameraReader | None = None
    sdk: InferenceSDK | None = None
    last_status_time = 0.0
    tick_id = 0

    try:
        alicia_m_sdk = load_alicia_m_sdk()
        robot = create_robot(alicia_m_sdk, args.port)
        camera_reader = OpenCVCameraReader(parse_cameras(args.camera))

        sdk = InferenceSDK(
            device=args.device,
            smoothing_config=SmoothingConfig(
                control_fps=args.fps,
                action_chunk_size=args.action_chunk_size,
                n_action_steps=args.n_action_steps,
                fallback_mode="hold",
                enable_temporal_ensemble=args.temporal_ensemble,
                enable_rtc=args.enable_rtc,
            ),
        )
        metadata = sdk.load_policy(
            args.model_type,
            args.checkpoint_dir,
            instruction=args.instruction,
        )
        if metadata.action_dim not in (6, 7):
            raise ValueError(f"Alicia-M publisher supports action_dim 6 or 7, got {metadata.action_dim}.")
        if metadata.state_dim and metadata.state_dim not in (6, 7):
            raise ValueError(f"Alicia-M state reader supports state_dim 6 or 7, got {metadata.state_dim}.")
        validate_alicia_m_robot_io(metadata.robot_io)

        print(
            "loaded model=%s action_dim=%d state_dim=%d cameras=%s robot_io=%s"
            % (
                metadata.model_type,
                metadata.action_dim,
                metadata.state_dim,
                ",".join(metadata.required_cameras),
                "present" if metadata.robot_io else "not bundled",
            ),
            flush=True,
        )

        wait_for_robot_state(robot, timeout=2.0)
        while True:
            tick_start = time.monotonic()
            tick_id += 1
            current_state = read_robot_state(robot)
            current_policy_state = robot_state_to_policy_state(current_state)

            infer_start = time.perf_counter()
            raw_action = np.asarray(
                sdk.predict_action(
                    args.model_type,
                    images=camera_reader.read(),
                    state=current_policy_state[: state_dim_for_metadata(metadata)],
                    instruction=args.instruction,
                ),
                dtype=np.float32,
            ).reshape(-1)
            latency_ms = (time.perf_counter() - infer_start) * 1000.0

            published = False
            if not args.dry_run:
                published = publish_robot_action(robot, raw_action, current_state)

            now = time.monotonic()
            if now - last_status_time >= 1.0:
                last_status_time = now
                print(
                    "tick=%d source=sync latency=%.1fms publish=%s"
                    % (
                        tick_id,
                        latency_ms,
                        "dry-run" if args.dry_run else published,
                    ),
                    flush=True,
                )
                if args.debug_actions:
                    joint_dim = min(6, raw_action.shape[0], current_policy_state.shape[0])
                    joint_delta = raw_action[:joint_dim] - current_policy_state[:joint_dim]
                    gripper_target = float(raw_action[6]) if raw_action.shape[0] >= 7 else float("nan")
                    print(
                        "  current_deg=%s action_deg=%s delta_deg=%s gripper=%.1f->%.1f"
                        % (
                            format_array(current_policy_state[:joint_dim]),
                            format_array(raw_action[:joint_dim]),
                            format_array(joint_delta),
                            float(current_policy_state[6]) if current_policy_state.shape[0] >= 7 else float("nan"),
                            gripper_target,
                        ),
                        flush=True,
                    )

            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, 1.0 / args.fps - elapsed))
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
    finally:
        if sdk is not None:
            sdk.close()
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
