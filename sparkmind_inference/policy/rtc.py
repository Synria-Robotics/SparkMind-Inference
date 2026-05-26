"""Helpers for optional LeRobot Real-Time Chunking (RTC)."""

from __future__ import annotations

from typing import Any

from ..base import SmoothingConfig


def make_rtc_config(config: SmoothingConfig) -> Any | None:
    """Build a SparkMind/LeRobot RTCConfig from SDK smoothing options."""
    if not config.enable_rtc:
        return None

    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    schedule_name = str(config.rtc_prefix_attention_schedule).strip().upper()
    try:
        schedule = RTCAttentionSchedule(schedule_name)
    except ValueError as exc:
        available = ", ".join(item.value for item in RTCAttentionSchedule)
        raise ValueError(f"rtc_prefix_attention_schedule must be one of: {available}") from exc

    return RTCConfig(
        enabled=True,
        prefix_attention_schedule=schedule,
        max_guidance_weight=float(config.rtc_max_guidance_weight),
        execution_horizon=int(config.rtc_execution_horizon),
        debug=bool(config.rtc_debug),
        debug_maxlen=int(config.rtc_debug_maxlen),
    )


def make_rtc_processor(config: SmoothingConfig) -> Any | None:
    """Build an RTCProcessor, returning None when RTC is disabled."""
    rtc_config = make_rtc_config(config)
    if rtc_config is None:
        return None

    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    return RTCProcessor(rtc_config)
