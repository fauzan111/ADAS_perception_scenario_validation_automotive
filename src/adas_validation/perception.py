"""Perception error model: what the car's perception reports, given where things really are.

Scenario-based ADAS testing often replaces the full sensor stack with a statistical model of
its errors (a "phantom" or surrogate sensor): the simulator knows every object's true state,
and the model decides, per frame, whether each object is detected, how noisy its position is,
how late the output arrives, and whether ghost objects appear. This keeps thousands of
scenario runs cheap while the failure behaviour stays realistic, *if* the model is
calibrated. The parameters in configs/perception_model.yaml start as stated assumptions and
are meant to be replaced by the nuScenes measurements (recall by range, at night, in rain).

Conditions act only through this model: night shrinks the detection range, rain and fog
shrink it further and add noise. The simulator's rendering plays no part.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class TrueObject:
    """Ground truth in the ego frame: x forward, y left (metres), velocities in m/s."""
    obj_id: str
    cls: str                 # "car" or "pedestrian"
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    visible_fraction: float = 1.0   # 0 = fully occluded


@dataclass(frozen=True)
class Detection:
    obj_id: str | None       # None for a false positive (ghost)
    cls: str
    x: float
    y: float
    t: float                 # time the measurement describes (before latency)


@dataclass(frozen=True)
class ClassParams:
    r50_m: float             # range with 50 % detection probability (day, clear, unoccluded)
    softness_m: float        # width of the logistic fall-off
    p_max: float             # detection probability up close
    sigma_m: float           # position noise at 10 m
    sigma_growth: float      # extra noise per 10 m of range (fraction of sigma_m)


@dataclass
class PerceptionModel:
    classes: dict[str, ClassParams]
    range_factor: dict[str, float]            # per lighting / weather condition
    noise_factor: dict[str, float]
    latency_s: float = 0.1
    min_visible_fraction: float = 0.5         # less visible than this -> not detected
    ghost_rate_hz: float = 0.0                # false positives in the ego path, per second
    # optional per-class overrides of range_factor, e.g. {"pedestrian": {"night": 0.59}}:
    # measured night loss differs by class (cars carry lights, pedestrians do not)
    class_range_factor: dict = field(default_factory=dict)
    _queue: deque = field(default_factory=deque, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PerceptionModel:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            classes={k: ClassParams(**v) for k, v in raw["classes"].items()},
            range_factor=raw["range_factor"], noise_factor=raw["noise_factor"],
            latency_s=raw["latency_s"], min_visible_fraction=raw["min_visible_fraction"],
            ghost_rate_hz=raw["ghost_rate_hz"],
            class_range_factor=raw.get("class_range_factor", {}))

    def detection_probability(self, cls: str, rng_m: float, lighting: str, weather: str,
                              visible_fraction: float = 1.0) -> float:
        if visible_fraction < self.min_visible_fraction:
            return 0.0
        p = self.classes[cls]
        rf = {**self.range_factor, **self.class_range_factor.get(cls, {})}
        r50 = p.r50_m * rf[lighting] * rf[weather]
        return p.p_max / (1.0 + math.exp((rng_m - r50) / p.softness_m))

    def position_sigma(self, cls: str, rng_m: float, lighting: str, weather: str) -> float:
        p = self.classes[cls]
        base = p.sigma_m * (1.0 + p.sigma_growth * max(rng_m - 10.0, 0.0) / 10.0)
        return base * self.noise_factor[lighting] * self.noise_factor[weather]

    def reset(self) -> None:
        self._queue.clear()

    def step(self, t: float, dt: float, objects: list[TrueObject], lighting: str,
             weather: str, rng: np.random.Generator) -> list[Detection]:
        """Measure the scene at time t; return the frame that becomes available at t
        (the one measured latency_s ago), so the controller always acts on old data."""
        frame = []
        for o in objects:
            r = math.hypot(o.x, o.y)
            if rng.random() < self.detection_probability(o.cls, r, lighting, weather,
                                                         o.visible_fraction):
                s = self.position_sigma(o.cls, r, lighting, weather)
                frame.append(Detection(o.obj_id, o.cls, o.x + rng.normal(0, s),
                                       o.y + rng.normal(0, s * 0.5), t))
        if self.ghost_rate_hz > 0 and rng.random() < self.ghost_rate_hz * dt:
            frame.append(Detection(None, "car", rng.uniform(10, 60), rng.normal(0, 0.5), t))
        self._queue.append((t, frame))
        out: list[Detection] = []
        while self._queue and self._queue[0][0] <= t - self.latency_s + 1e-9:
            out = self._queue.popleft()[1]
        return out


@dataclass
class Track:
    track_id: int
    cls: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    hits: list = field(default_factory=list)  # 1/0 for the last frames
    last_t: float = 0.0
    obj_id: str | None = None   # ground-truth id of the first detection (metrics only)
    confirmed: bool = False
    n_updates: int = 0


class Tracker:
    """Nearest-neighbour association with an alpha-beta filter per track (ego-relative).

    - A new track takes its velocity from its first two detections (not zero: a zero relative
      velocity would make a braking car look like it moves with the ego).
    - Confirmation: 3 hits in the last 5 frames. Once confirmed, a track stays confirmed until
      it has had no detection for max_coast_s (M-of-N confirmation, time-based deletion), so
      a detector that misses every other frame does not make the obstacle flicker.
    """

    def __init__(self, gate_m: float = 3.0, alpha: float = 0.5, beta: float = 0.2,
                 max_coast_s: float = 0.6) -> None:
        self.gate_m, self.alpha, self.beta, self.max_coast_s = gate_m, alpha, beta, max_coast_s
        self.tracks: list[Track] = []
        self._next = 0

    def step(self, t: float, dt: float, detections: list[Detection]) -> list[Track]:
        for tr in self.tracks:            # predict
            tr.x += tr.vx * dt
            tr.y += tr.vy * dt
        unmatched = list(detections)
        for tr in sorted(self.tracks, key=lambda tr: not tr.confirmed):
            best = min(unmatched, key=lambda d: math.hypot(d.x - tr.x, d.y - tr.y),
                       default=None)
            if best is not None and math.hypot(best.x - tr.x, best.y - tr.y) < self.gate_m:
                unmatched.remove(best)
                rx, ry = best.x - tr.x, best.y - tr.y
                if tr.n_updates == 0:     # second detection: velocity from the two points
                    span = max(t - tr.last_t, dt)
                    tr.vx += rx / span
                    tr.vy += ry / span
                    tr.x, tr.y = best.x, best.y
                else:
                    tr.x += self.alpha * rx
                    tr.y += self.alpha * ry
                    tr.vx += self.beta * rx / dt
                    tr.vy += self.beta * ry / dt
                tr.n_updates += 1
                tr.hits.append(1)
                tr.last_t = t
            else:
                tr.hits.append(0)
            if sum(tr.hits[-5:]) >= 3:
                tr.confirmed = True
        for d in unmatched:
            self.tracks.append(Track(self._next, d.cls, d.x, d.y, hits=[1], last_t=t,
                                     obj_id=d.obj_id))
            self._next += 1
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.max_coast_s]
        return [tr for tr in self.tracks if tr.confirmed]
