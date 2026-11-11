"""Safety metrics for scenario-based validation."""

from __future__ import annotations

import math


def time_to_collision(gap_m: float, ego_speed_mps: float, lead_speed_mps: float) -> float:
    """Seconds until the ego vehicle reaches the lead vehicle at constant speeds.

    Returns +inf when the gap is not closing. gap_m is bumper-to-bumper distance.
    """
    if gap_m < 0:
        raise ValueError("gap must be non-negative")
    closing = ego_speed_mps - lead_speed_mps
    if closing <= 0:
        return math.inf
    return gap_m / closing


def scenario_passed(min_ttc_s: float, collided: bool, ttc_threshold_s: float = 1.5) -> bool:
    """Pass if there is no collision and the minimum TTC stays above the threshold."""
    return (not collided) and min_ttc_s >= ttc_threshold_s
