"""Robot I/O metadata helpers for portable SparkMind/LeRobot policy bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


ROBOT_IO_FILENAME = "robot_io.json"
PRETRAINED_SUBDIR_NAME = "pretrained_model"
PRETRAINED_CHECKPOINT_FILES = ("config.json", "model.safetensors")


def _has_pretrained_files(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in PRETRAINED_CHECKPOINT_FILES)


def resolve_pretrained_bundle_dir(checkpoint_dir: str | Path) -> Path:
    """Resolve either a step checkpoint root or a direct ``pretrained_model`` path."""
    path = Path(checkpoint_dir).expanduser()
    if _has_pretrained_files(path):
        return path

    pretrained_dir = path / PRETRAINED_SUBDIR_NAME
    if _has_pretrained_files(pretrained_dir):
        return pretrained_dir

    return path


def _validate_robot_io(payload: Any, *, source: Path) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object.")

    arm_mode = payload.get("arm_mode")
    if arm_mode is not None and arm_mode not in {"single_arm", "dual_arm"}:
        raise ValueError(
            f"{source} has invalid arm_mode={arm_mode!r}; expected 'single_arm' or 'dual_arm'."
        )

    gripper_range = payload.get("gripper_range")
    if gripper_range is not None:
        if (
            not isinstance(gripper_range, list)
            or len(gripper_range) != 2
            or not all(isinstance(value, (int, float)) for value in gripper_range)
        ):
            raise ValueError(f"{source} gripper_range must be a two-number JSON array.")

    return payload


def load_robot_io_json(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate a ``robot_io.json`` file."""
    source = Path(path).expanduser()
    try:
        with source.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from exc
    return _validate_robot_io(payload, source=source)


def load_robot_io_from_checkpoint(checkpoint_dir: str | Path) -> Optional[Dict[str, Any]]:
    """Load optional ``robot_io.json`` from a checkpoint or pretrained bundle."""
    bundle_dir = resolve_pretrained_bundle_dir(checkpoint_dir)
    path = bundle_dir / ROBOT_IO_FILENAME
    if not path.is_file():
        return None
    return load_robot_io_json(path)
