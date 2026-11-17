"""ACC + AEB longitudinal controller acting on perception tracks (not on ground truth).

- In-path selection: a track is relevant if its centre, now or predicted forward to the
  moment the ego reaches it, is within (ego half width + object half width + margin) of the
  ego's centre line, i.e. if the object's body would overlap the ego's path. This is how a
  crossing pedestrian or a cutting-in car becomes relevant before it is fully in the lane.
- ACC: hold the set speed, or a time gap to the in-path lead (IDM-style law).
- FCW at TTC < fcw_ttc_s (warning only, logged); AEB: full braking when TTC < aeb_ttc_s or
  when the deceleration needed to stop short exceeds aeb_required_decel. The tracker gives no
  acceleration, so the required deceleration assumes the lead keeps its speed: against a
  braking lead the AEB fires late. If a track carries a deceleration (counterfactual runs),
  the ego must also stop within the gap plus the lead's own stopping distance.
- Available deceleration depends on the road: dry 9 m/s^2, wet (rain) 6.5 m/s^2. This is an
  explicit assumption standing in for tyre friction (the simulator's friction is not varied).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from adas_validation.perception import Track

MAX_DECEL = {"clear": 9.0, "rain": 6.5, "fog": 9.0}
OBJECT_HALF_WIDTH = {"car": 0.93, "pedestrian": 0.35}


@dataclass
class AdasConfig:
    set_speed_mps: float
    time_gap_s: float = 1.8
    standstill_gap_m: float = 4.0
    max_accel: float = 1.5
    comfort_decel: float = 3.0
    fcw_ttc_s: float = 2.6
    aeb_ttc_s: float = 1.6
    aeb_required_decel: float = 5.0
    ego_half_width_m: float = 0.93
    lateral_margin_m: float = 0.3
    ego_front_m: float = 2.3          # sensor at the front bumper: ranges are from here


@dataclass(frozen=True)
class AdasOutput:
    accel: float                       # commanded acceleration, m/s^2 (negative = brake)
    mode: str                          # "cruise", "follow", "fcw", "aeb"
    target_id: int | None
    ttc_s: float


def in_path(track: Track, ego_speed: float, cfg: AdasConfig) -> bool:
    """Is the track (now or when the ego arrives) inside the ego corridor?"""
    if track.x <= 0:
        return False
    closing = ego_speed - track.vx
    t_arrive = track.x / closing if closing > 0.1 else 3.0
    t_arrive = min(t_arrive, 3.0)
    half = cfg.ego_half_width_m + OBJECT_HALF_WIDTH.get(track.cls, 0.93) + cfg.lateral_margin_m
    for tt in (0.0, t_arrive):
        if abs(track.y + track.vy * tt) <= half:
            return True
    return False


def ttc(track: Track, ego_speed: float) -> float:
    closing = ego_speed - track.vx
    return track.x / closing if closing > 0 else math.inf


def control(ego_speed: float, tracks: list[Track], cfg: AdasConfig, weather: str,
            max_decel: float | None = None) -> AdasOutput:
    """max_decel overrides the road-dependent braking limit (counterfactual runs)."""
    max_decel = MAX_DECEL[weather] if max_decel is None else max_decel
    relevant = [tr for tr in tracks if in_path(tr, ego_speed, cfg)]
    lead = min(relevant, key=lambda tr: tr.x, default=None)
    # ACC (IDM-style): free road term plus interaction term
    a_free = cfg.max_accel * (1 - (ego_speed / max(cfg.set_speed_mps, 0.1)) ** 4)
    if lead is None:
        return AdasOutput(max(min(a_free, cfg.max_accel), -cfg.comfort_decel), "cruise",
                          None, math.inf)
    gap = max(lead.x, 0.1)
    dv = ego_speed - lead.vx
    desired = cfg.standstill_gap_m + ego_speed * cfg.time_gap_s + \
        ego_speed * dv / (2 * math.sqrt(cfg.max_accel * cfg.comfort_decel))
    a_acc = cfg.max_accel * (1 - (ego_speed / max(cfg.set_speed_mps, 0.1)) ** 4
                             - (max(desired, 0) / gap) ** 2)
    t = ttc(lead, ego_speed)
    required = dv * dv / (2 * gap) if dv > 0 else 0.0
    if lead.ax < -0.1:                 # braking lead: stop within gap + its stopping distance
        lead_stop = max(lead.vx, 0.0) ** 2 / (2 * -lead.ax)
        required = max(required, ego_speed ** 2 / (2 * (gap + lead_stop)))
    if t < cfg.aeb_ttc_s or required > cfg.aeb_required_decel:
        return AdasOutput(-max_decel, "aeb", lead.track_id, t)
    accel = max(min(a_acc, cfg.max_accel), -cfg.comfort_decel)
    return AdasOutput(accel, "fcw" if t < cfg.fcw_ttc_s else "follow", lead.track_id, t)
