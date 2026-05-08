"""Template control loop for the process-local async inference runtime.

This example intentionally leaves camera and robot I/O as small placeholder
functions. Replace them with your backend's real hardware integration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_sdk import AsyncInferenceConfig, SUPPORTED_MODEL_TYPES, get_global_async_runtime


def read_camera_images() -> Dict[str, np.ndarray]:
    """Return BGR images keyed by camera role, for example {'head': image}."""
    raise NotImplementedError("Connect this function to your camera pipeline.")


def read_robot_state() -> np.ndarray:
    """Return the current robot state vector."""
    raise NotImplementedError("Connect this function to your robot state reader.")


def send_robot_action(action: np.ndarray) -> None:
    """Send one action vector to the robot controller."""
    raise NotImplementedError("Connect this function to your robot action sender.")


def sleep_until_next_tick(start_time: float, fps: float) -> None:
    elapsed = time.monotonic() - start_time
    time.sleep(max(0.0, (1.0 / fps) - elapsed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an async inference control-loop template.")
    parser.add_argument("--model-type", required=True, choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--chunk-size-threshold",
        dest="chunk_size_threshold",
        type=float,
        default=0.5,
        help=(
            "Queue refill threshold for async inference. "
            "The model action chunk size is read from the checkpoint."
        ),
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
        "--temporal-ensemble",
        action="store_true",
        help="Enable ACT temporal ensembling. Only valid when --model-type act.",
    )
    parser.add_argument(
        "--enable-rtc",
        action="store_true",
        help="Enable RTC. Only valid for SmolVLA/PI0/PI0.5 policies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temporal_ensemble and args.model_type != "act":
        raise ValueError("--temporal-ensemble is only supported for ACT")
    if args.enable_rtc and args.model_type not in {"smolvla", "pi0", "pi05"}:
        raise ValueError("--enable-rtc is only supported for SmolVLA, PI0 and PI0.5")

    runtime = get_global_async_runtime()

    runtime.load_policy(
        algorithm_type=args.model_type,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        instruction=args.instruction,
        config=AsyncInferenceConfig(
            control_fps=args.fps,
            chunk_size_threshold=args.chunk_size_threshold,
            action_chunk_size=args.action_chunk_size,
            n_action_steps=args.n_action_steps,
            enable_temporal_ensemble=args.temporal_ensemble,
            enable_rtc=args.enable_rtc,
        ),
    )

    runtime.warmup(
        images=read_camera_images(),
        state=read_robot_state(),
        instruction=args.instruction,
    )

    runtime.start()

    if not runtime.wait_until_ready(min_queue_size=1, timeout=5.0):
        runtime.stop()
        raise RuntimeError("Async runtime did not produce an initial action before startup timeout.")

    try:
        while True:
            tick_start = time.monotonic()
            result = runtime.step(
                images=read_camera_images(),
                state=read_robot_state(),
                instruction=args.instruction,
            )
            send_robot_action(result.action)
            sleep_until_next_tick(tick_start, args.fps)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
