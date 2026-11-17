"""Perception error model, tracker, ADAS controller, geometry and scenario scripts."""

import math
from pathlib import Path

import numpy as np
import pytest

from adas_validation.adas import AdasConfig, control, in_path
from adas_validation.geometry import Box, box_circle_overlap, boxes_overlap, visible_fraction
from adas_validation.perception import (
    Detection,
    PerceptionModel,
    Track,
    Tracker,
    TrueObject,
)
from adas_validation.scenarios import Scenario, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "configs" / "perception_model.yaml"


# --- geometry
def test_occluder_blocks_line_of_sight_fully_and_partly():
    van = Box(20, 3.5, 6, 2.2)               # parked van, centred 3.5 m to the side
    hidden = Box(23.6, 3.8, 0.7, 0.7)        # pedestrian just past the van's front corner
    assert visible_fraction((0, 0), hidden, [van]) == 0.0
    out_in_lane = Box(23.6, 0.5, 0.7, 0.7)   # stepped into the ego lane
    assert visible_fraction((0, 0), out_in_lane, [van]) == 1.0
    edge = Box(23.6, 2.2, 0.7, 0.7)          # half out from behind the van
    assert 0.0 < visible_fraction((0, 0), edge, [van]) < 1.0


def test_overlaps():
    assert boxes_overlap(Box(0, 0, 4, 2), Box(3.9, 0, 4, 2))
    assert not boxes_overlap(Box(0, 0, 4, 2), Box(0, 2.1, 4, 2))
    assert box_circle_overlap(Box(0, 0, 4, 2), 2.2, 0, 0.35)
    assert not box_circle_overlap(Box(0, 0, 4, 2), 2.5, 0, 0.35)


# --- perception error model
def test_detection_probability_falls_with_range_night_and_occlusion():
    m = PerceptionModel.from_yaml(MODEL)
    near = m.detection_probability("car", 10, "day", "clear")
    far = m.detection_probability("car", 100, "day", "clear")
    night = m.detection_probability("car", 50, "night", "clear")
    day = m.detection_probability("car", 50, "day", "clear")
    assert near > 0.9 and near > day > night > far
    assert m.detection_probability("car", 10, "day", "clear", visible_fraction=0.2) == 0.0
    assert m.position_sigma("car", 50, "night", "fog") > m.position_sigma("car", 50, "day",
                                                                          "clear")


def test_latency_delivers_old_frames():
    m = PerceptionModel.from_yaml(MODEL)
    m.ghost_rate_hz = 0.0
    rng = np.random.default_rng(0)
    seen = []
    for k in range(5):
        t = k * 0.1
        obj = TrueObject("a", "car", 10.0 - k, 0.0)
        dets = m.step(t, 0.1, [obj], "day", "clear", rng)
        seen.append([round(d.t, 2) for d in dets])
    assert seen[0] == []                 # nothing is available before the latency has passed
    assert seen[2] == [0.1]              # at t=0.2 the controller sees the t=0.1 measurement


# --- tracker
def _det(x, y=0.0, t=0.0):
    return Detection("a", "car", x, y, t)


def test_tracker_confirms_initialises_velocity_and_survives_misses():
    tr = Tracker()
    xs = [30.0 - 2.0 * k for k in range(8)]          # closing at 20 m/s (2 m per 0.1 s)
    frames = [[_det(x)] for x in xs]
    frames[4] = []                                    # missed detections
    frames[5] = []
    confirmed_seen = []
    for k, f in enumerate(frames):
        out = tr.step(k * 0.1, 0.1, f)
        confirmed_seen.append(len(out))
    assert confirmed_seen[:2] == [0, 0]               # needs 3 hits
    assert confirmed_seen[2] == 1
    assert confirmed_seen[4] == 1 and confirmed_seen[5] == 1   # coasts through misses
    assert tr.tracks[0].vx == pytest.approx(-20.0, abs=2.0)   # from the first two points


def test_tracker_deletes_after_coasting():
    tr = Tracker(max_coast_s=0.3)
    for k in range(3):
        tr.step(k * 0.1, 0.1, [_det(20.0)])
    for k in range(3, 8):
        out = tr.step(k * 0.1, 0.1, [])
    assert out == [] and tr.tracks == []


# --- ADAS
CFG = AdasConfig(set_speed_mps=25.0)


def test_cruise_follow_and_aeb():
    assert control(20.0, [], CFG, "clear").mode == "cruise"
    lead_far = Track(0, "car", 80.0, 0.0, vx=24.0, confirmed=True)
    assert control(25.0, [lead_far], CFG, "clear").mode in ("follow", "cruise")
    close = Track(1, "car", 10.0, 0.0, vx=0.0, confirmed=True)   # stopped car 10 m ahead
    out = control(25.0, [close], CFG, "clear")
    assert out.mode == "aeb" and out.accel == -9.0
    assert control(25.0, [close], CFG, "rain").accel == -6.5    # wet road brakes less


def test_in_path_uses_object_width_and_predicted_motion():
    adjacent = Track(0, "car", 20.0, 3.5, vx=20.0)               # centred in the next lane
    overlapping = Track(1, "car", 20.0, 1.9, vx=20.0)            # body overlaps the ego path
    assert not in_path(adjacent, 25.0, CFG)
    assert in_path(overlapping, 25.0, CFG)
    walker = Track(2, "pedestrian", 12.0, 3.0, vx=0.0, vy=-1.4)  # walking towards the lane
    still = Track(3, "pedestrian", 12.0, 3.0, vx=0.0, vy=0.0)
    assert in_path(walker, 11.1, CFG)
    assert not in_path(still, 11.1, CFG)


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(30, 30)
    assert hi == 1.0 and 0.85 < lo < 0.9
    assert wilson_interval(0, 30)[0] == 0.0


# --- scenario scripts (no simulator needed)
def test_cut_in_moves_into_ego_lane_and_pedestrian_waits_for_trigger():
    from adas_validation.sim import EGO_LANE_Y, ScenarioScript

    cut = ScenarioScript(Scenario("c", "cut_in", 90, params={
        "cut_in_gap_m": 20, "cutter_speed_kph": 60, "lane_change_s": 2.5}), 25.0)
    ego_x = 0.0
    for k in range(80):
        ego_x += 25.0 * 0.1
        cut.step(k * 0.1, ego_x)
    assert cut.actors["hazard"].y == pytest.approx(EGO_LANE_Y)
    ped = ScenarioScript(Scenario("p", "occluded_pedestrian", 40, params={
        "pedestrian_speed_mps": 1.4, "van_distance_m": 45, "trigger_distance_m": 25}), 11.1)
    y0 = ped.actors["hazard"].y
    ped.step(0.0, 0.0)                                  # ego 49 m away: not walking yet
    assert ped.actors["hazard"].y == y0
    ped.step(0.1, 30.0)                                 # ego within 25 m: starts walking
    assert ped.actors["hazard"].y < y0


def test_metadrive_smoke():
    pytest.importorskip("metadrive")
    from adas_validation.sim import run_scenario

    model = PerceptionModel.from_yaml(MODEL)
    sc = Scenario("lead", "lead_brake", 90, params={"lead_decel_mps2": 4.0,
                                                   "initial_gap_m": 40, "brake_at_s": 1.0})
    r = run_scenario(sc, 0, model, max_time_s=12.0)
    assert not r.collided and r.aeb_triggered     # brakes in time (may still creep at 12 s)
    assert math.isfinite(r.min_ttc_s) and r.first_detect_range_m is not None


def test_per_class_condition_factor_overrides_shared_one():
    m = PerceptionModel.from_yaml(MODEL)
    car_before = m.detection_probability("car", 40, "night", "clear")
    m.class_range_factor = {"pedestrian": {"night": 0.3}}
    assert m.detection_probability("car", 40, "night", "clear") == car_before
    assert m.detection_probability("pedestrian", 20, "night", "clear") < \
        m.detection_probability("pedestrian", 20, "day", "clear") * 0.5


def test_measured_and_anchored_models_load():
    for name in ("perception_model_measured.yaml", "perception_model_nuscenes_anchored.yaml"):
        m = PerceptionModel.from_yaml(ROOT / "configs" / name)
        assert m.classes["car"].r50_m > 0 and "night" in m.class_range_factor["pedestrian"]
