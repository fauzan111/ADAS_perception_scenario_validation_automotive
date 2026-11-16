"""Run catalogue scenarios in MetaDrive with the perception error model and the ADAS in loop.

World frame: a long straight 3-lane road along +x; lane k is centred at y = 3.5 k. The ego
drives in lane 1. Other actors are kinematic (scripted exactly, so every run of a scenario
sees the same traffic and only perception noise differs between seeds). The ego is a
MetaDrive vehicle with its own physics; the ADAS acceleration request is turned into
throttle/brake by a small feedforward + feedback loop on measured acceleration.

Ground truth decides the verdict: collision from box/circle overlap of true geometry, TTC
from true gaps. Perception only decides what the controller believes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from adas_validation.adas import AdasConfig, control
from adas_validation.geometry import Box, box_circle_overlap, boxes_overlap, visible_fraction
from adas_validation.perception import PerceptionModel, Tracker, TrueObject
from adas_validation.scenarios import Scenario

LANE_W = 3.5
EGO_LANE_Y = LANE_W * 1
EGO_LEN, EGO_W = 4.515, 1.852
CAR_LEN, CAR_W = 4.515, 1.852
VAN_LEN, VAN_W = 6.0, 2.2
PED_R = 0.35
DT = 0.1


@dataclass
class Actor:
    obj_id: str
    cls: str                     # car, pedestrian, occluder
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    length: float = CAR_LEN
    width: float = CAR_W

    def box(self) -> Box:
        return Box(self.x, self.y, self.length, self.width)


@dataclass
class RunResult:
    scenario: str
    seed: int
    collided: bool
    impact_speed_kph: float
    min_ttc_s: float
    first_detect_range_m: float | None   # range when the hazard was first a confirmed track
    aeb_triggered: bool
    aeb_gap_m: float | None              # true bumper-to-hazard distance when AEB first fired
    end_reason: str
    trace: list = field(default_factory=list, repr=False)


class ScenarioScript:
    """Kinematic actors for one scenario kind; positions are a function of time and ego x."""

    def __init__(self, sc: Scenario, ego_speed: float) -> None:
        self.sc, self.v0 = sc, ego_speed
        p = sc.params
        self.actors: dict[str, Actor] = {}
        self.hazard = "hazard"
        if sc.kind == "lead_brake":
            gap = p["initial_gap_m"]
            self.actors["hazard"] = Actor("hazard", "car", EGO_LEN / 2 + gap + CAR_LEN / 2,
                                          EGO_LANE_Y, vx=ego_speed)
            self.brake_at = p.get("brake_at_s", 2.0)
            self.decel = p["lead_decel_mps2"]
        elif sc.kind == "cut_in":
            self.cutter_v = p["cutter_speed_kph"] / 3.6
            self.gap = p["cut_in_gap_m"]
            self.duration = p.get("lane_change_s", 2.5)
            lead_time = 3.0                      # seconds of approach before the cut starts
            x0 = EGO_LEN / 2 + self.gap + CAR_LEN / 2 + (ego_speed - self.cutter_v) * lead_time
            self.actors["hazard"] = Actor("hazard", "car", x0, EGO_LANE_Y + LANE_W,
                                          vx=self.cutter_v)
            self.cut_start: float | None = None
        elif sc.kind == "occluded_pedestrian":
            d = p.get("van_distance_m", 45.0)
            self.van = Actor("van", "occluder", d, EGO_LANE_Y + LANE_W, length=VAN_LEN,
                             width=VAN_W)
            self.actors["van"] = self.van
            # pedestrian waits just past the van's front corner, hidden from the ego
            self.actors["hazard"] = Actor("hazard", "pedestrian", d + VAN_LEN / 2 + 0.6,
                                          EGO_LANE_Y + LANE_W + 0.3, length=2 * PED_R,
                                          width=2 * PED_R)
            self.walk_speed = p["pedestrian_speed_mps"]
            self.trigger_m = p.get("trigger_distance_m", 25.0)
            self.walking = False
        else:
            raise ValueError(f"unknown scenario kind {sc.kind}")

    def step(self, t: float, ego_x: float) -> None:
        h = self.actors["hazard"]
        if self.sc.kind == "lead_brake":
            if t >= self.brake_at:
                h.vx = max(h.vx - self.decel * DT, 0.0)
            h.x += h.vx * DT
        elif self.sc.kind == "cut_in":
            h.x += h.vx * DT
            rear_gap = (h.x - CAR_LEN / 2) - (ego_x + EGO_LEN / 2)
            if self.cut_start is None and rear_gap <= self.gap:
                self.cut_start = t
            if self.cut_start is not None:
                f = min((t - self.cut_start) / self.duration, 1.0)
                y_new = EGO_LANE_Y + LANE_W * (1 + math.cos(math.pi * f)) / 2
                h.vy = (y_new - h.y) / DT
                h.y = y_new
        elif self.sc.kind == "occluded_pedestrian":
            if not self.walking and h.x - ego_x <= self.trigger_m:
                self.walking = True
            if self.walking and h.y > EGO_LANE_Y - LANE_W:
                h.vy = -self.walk_speed
                h.y += h.vy * DT
            else:
                h.vy = 0.0

    def occluders(self) -> list[Box]:
        return [a.box() for a in self.actors.values() if a.cls == "occluder"]


def true_ttc(ego_x: float, ego_v: float, h: Actor) -> float:
    lateral = abs(h.y - EGO_LANE_Y) <= (EGO_W + h.width) / 2
    gap = (h.x - h.length / 2) - (ego_x + EGO_LEN / 2)
    closing = ego_v - h.vx
    if not lateral or gap < 0 or closing <= 0:
        return math.inf
    return gap / closing


class LongitudinalLoop:
    """Turn an acceleration request into MetaDrive throttle/brake (measured map + PI)."""

    def __init__(self) -> None:
        self.integral = 0.0

    def __call__(self, a_cmd: float, a_meas: float) -> float:
        ff = a_cmd / 16.0 if a_cmd < 0 else (a_cmd + 0.36) / 2.77
        err = a_cmd - a_meas
        self.integral = float(np.clip(self.integral + 0.02 * err, -0.3, 0.3))
        return float(np.clip(ff + 0.05 * err + self.integral, -1.0, 1.0))


def run_scenario(sc: Scenario, seed: int, model: PerceptionModel, env=None,
                 max_time_s: float = 20.0, keep_trace: bool = False) -> RunResult:
    """One scenario run. Pass an open MetaDrive env to reuse it across runs."""
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    from metadrive.component.vehicle.vehicle_type import DefaultVehicle, LVehicle

    rng = np.random.default_rng(seed)
    v0 = sc.ego_speed_kph / 3.6
    own_env = env is None
    if own_env:
        env = make_env()
    env.reset()
    ego = env.agent
    ego.config["max_speed_km_h"] = 200
    # MetaDrive applies spawn lane and velocity only on the first reset: set them directly
    ego.set_position([ego.position[0], EGO_LANE_Y])
    ego.set_heading_theta(0.0)
    ego.set_velocity([1.0, 0.0], v0)
    x_origin = ego.position[0]
    script = ScenarioScript(sc, v0)
    bodies = {}
    for name, a in script.actors.items():
        pos = (x_origin + a.x, a.y)
        if a.cls == "pedestrian":
            bodies[name] = env.engine.spawn_object(Pedestrian, position=pos, heading_theta=0)
        else:
            cls = LVehicle if a.cls == "occluder" else DefaultVehicle
            bodies[name] = env.engine.spawn_object(cls, vehicle_config={}, position=pos,
                                                   heading=0.0)
    model.reset()
    tracker = Tracker()
    cfg = AdasConfig(set_speed_mps=v0)
    loop = LongitudinalLoop()
    hazard = script.actors[script.hazard]
    min_ttc, first_detect, aeb_gap = math.inf, None, None
    collided, impact, end = False, 0.0, "timeout"
    prev_v, trace = ego.speed, []
    stopped_for = 0.0
    for k in range(int(max_time_s / DT)):
        t = k * DT
        ego_x = ego.position[0] - x_origin
        ego_y = ego.position[1]
        ego_v = ego.speed
        script.step(t, ego_x)
        # ground truth checks
        min_ttc = min(min_ttc, true_ttc(ego_x, ego_v, hazard))
        ego_box = Box(ego_x, ego_y, EGO_LEN, EGO_W)
        hit = (box_circle_overlap(ego_box, hazard.x, hazard.y, PED_R)
               if hazard.cls == "pedestrian" else boxes_overlap(ego_box, hazard.box()))
        if hit:
            collided, impact, end = True, ego_v * 3.6, "collision"
            break
        # perception: true objects in the ego sensor frame (front bumper)
        sx, sy = ego_x + EGO_LEN / 2, ego_y
        objs = []
        for a in script.actors.values():
            if a.cls == "occluder":
                continue
            near_x = a.x - (a.length / 2 if a.cls == "car" else 0.0)
            vis = visible_fraction((sx, sy), a.box(), script.occluders())
            objs.append(TrueObject(a.obj_id, a.cls, near_x - sx, a.y - sy, a.vx, a.vy, vis))
        dets = model.step(t, DT, objs, sc.lighting, sc.weather, rng)
        tracks = tracker.step(t, DT, dets)
        for tr in tracks:
            if first_detect is None and tr.obj_id == script.hazard:
                first_detect = math.hypot(tr.x, tr.y)
        out = control(ego_v, [_relative(tr, ego_v) for tr in tracks], cfg, sc.weather)
        if out.mode == "aeb" and aeb_gap is None:
            aeb_gap = (hazard.x - hazard.length / 2) - (ego_x + EGO_LEN / 2)
        a_meas = (ego_v - prev_v) / DT
        prev_v = ego_v
        steer = -0.4 * (ego_y - EGO_LANE_Y) - 1.2 * _wrap(ego.heading_theta)
        throttle = loop(out.accel, a_meas)
        if keep_trace:
            trace.append(dict(t=round(t, 2), ego_x=ego_x, ego_y=ego_y, ego_v=ego_v, hz_x=hazard.x,
                              hz_y=hazard.y, mode=out.mode, accel_cmd=out.accel,
                              n_tracks=len(tracks), ttc_true=true_ttc(ego_x, ego_v, hazard)))
        for name, a in script.actors.items():
            bodies[name].set_position((x_origin + a.x, a.y))
            if a.cls != "pedestrian":
                bodies[name].set_velocity([1, 0], a.vx)
        _, _, _, _, info = env.step([float(np.clip(steer, -1, 1)), throttle])
        if info.get("crash_vehicle") or info.get("crash_human"):
            # physics contact can happen a hair before the box check and shove the ego aside
            collided, impact, end = True, ego_v * 3.6, "collision"
            break
        stopped_for = stopped_for + DT if ego.speed < 0.3 else 0.0
        if stopped_for >= 1.0:
            end = "stopped"
            break
        if hazard.x + hazard.length / 2 < ego_x - EGO_LEN / 2 - 5:
            end = "passed"
            break
    for b in bodies.values():
        env.engine.clear_objects([b.id])
    if own_env:
        env.close()
    return RunResult(sc.name, seed, collided, impact, min_ttc, first_detect,
                     aeb_gap is not None, aeb_gap, end, trace)


def _relative(tr, ego_v):
    """Tracker velocities are relative to the ego; the controller wants absolute vx."""
    from adas_validation.perception import Track
    return Track(tr.track_id, tr.cls, tr.x, tr.y, tr.vx + ego_v, tr.vy, tr.hits, tr.last_t,
                 tr.obj_id, tr.confirmed, tr.n_updates)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def make_env():
    from metadrive.envs.metadrive_env import MetaDriveEnv
    return MetaDriveEnv(dict(
        use_render=False, map="S" * 14, traffic_density=0.0, num_scenarios=1, start_seed=0,
        log_level=50, random_spawn_lane_index=False,
        vehicle_config=dict(spawn_lane_index=(">", ">>", 1), spawn_velocity=[20.0, 0.0]),
        decision_repeat=5, physics_world_step_size=DT / 5,
        out_of_road_done=False, crash_vehicle_done=False, crash_human_done=False,
        crash_object_done=False, horizon=100000))
