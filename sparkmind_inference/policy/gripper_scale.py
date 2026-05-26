"""Helpers for adapting gripper values between robot-space and checkpoint stats."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def feature_gripper_stats_are_unit_scaled(
    stats: Optional[dict[str, dict[str, Any]]],
    feature_name: str,
) -> bool:
    """Infer whether the last feature dimension was stored in [0, 1] scale."""
    if not stats or feature_name not in stats:
        return True

    feature_stats = stats[feature_name]
    last_dim_values: list[float] = []
    for stat_name in ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99"):
        if stat_name not in feature_stats:
            continue
        stat_array = np.asarray(feature_stats[stat_name], dtype=np.float32).reshape(-1)
        if stat_array.size:
            last_dim_values.append(abs(float(stat_array[-1])))

    if not last_dim_values:
        return True

    return max(last_dim_values) <= 1.5
