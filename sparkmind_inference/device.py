import logging
from dataclasses import dataclass
from typing import Optional

import torch

from .exceptions import DeviceUnavailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    actual: str
    warning: str = ""


def _normalize_device_name(device: Optional[str]) -> str:
    if not isinstance(device, str):
        return "cuda:0"

    normalized = device.strip()
    return normalized or "cuda:0"


def _parse_cuda_index(device: str) -> Optional[int]:
    if device == "cuda":
        return 0
    if not device.startswith("cuda:"):
        return None

    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return None


def _cuda_device_count() -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception as exc:
        logger.warning("Failed to query CUDA device count: %s", exc)
        return 0


def _mps_available() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_torch_device(requested_device: Optional[str] = None, strict: bool = False) -> DeviceSelection:
    requested = _normalize_device_name(requested_device)

    if requested.startswith("cuda"):
        device_count = _cuda_device_count()
        if torch.cuda.is_available():
            requested_index = _parse_cuda_index(requested)
            if requested_index is not None and device_count > 0 and requested_index >= device_count:
                message = (
                    f"请求使用 `{requested}`，但当前只检测到 {device_count} 张 CUDA 设备。"
                )
                if strict:
                    raise DeviceUnavailableError(message)
                actual = "cuda:0"
                return DeviceSelection(
                    requested=requested,
                    actual=actual,
                    warning=f"{message} 已回退到 `{actual}`。",
                )

            return DeviceSelection(requested=requested, actual=requested)

        message = (
            f"请求使用 `{requested}`，但当前环境 CUDA 不可用。"
            f" torch.cuda.is_available()={torch.cuda.is_available()}，"
            f" torch.cuda.device_count()={device_count}。"
        )
        if strict:
            raise DeviceUnavailableError(message)

        if _mps_available():
            actual = "mps"
        else:
            actual = "cpu"

        return DeviceSelection(
            requested=requested,
            actual=actual,
            warning=f"{message} 已回退到 `{actual}`。",
        )

    if requested == "mps" and not _mps_available():
        message = "请求使用 `mps`，但当前环境 MPS 不可用。"
        if strict:
            raise DeviceUnavailableError(message)
        return DeviceSelection(
            requested=requested,
            actual="cpu",
            warning=f"{message} 已回退到 `cpu`。",
        )

    return DeviceSelection(requested=requested, actual=requested)
