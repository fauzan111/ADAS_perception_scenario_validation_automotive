"""Counterfactual attribution (SOTIF table) and parameter-sweep helpers."""

from pathlib import Path

import numpy as np
import pytest

from adas_validation.adas import AdasConfig, control
from adas_validation.perception import PerceptionModel, Track, Tracker, TrueObject
from adas_validation.scenarios import Scenario
from adas_validation.sotif import CAUSES, STEPS, attribute
from adas_validation.sweeps import boundary, cell_ok, make_scenario, worst_case

MODEL = Path(__file__).resolve().parents[1] / "configs" / "perception_model.yaml"


def test_attribution_takes_the_first_step_that_removes_the_collision():
    all_fail = {step: True for step, _ in STEPS}
    assert attribute(all_fail) == "design_limit"
    assert attribute({**all_fail, "detection": False}) == "detection"
    assert attribute({**all_fail, "dry": False, "instant": False}) == "friction"
    assert attribute({**all_fail, "logic": False}) == "decision"
    assert {cause for _, cause in STEPS} | {"design_limit"} == set(CAUSES)


def test_oracle_sees_everything_visible_without_noise_or_ghosts():
    m = PerceptionModel.from_yaml(MODEL)
    o = m.oracle()
    assert o.detection_probability("pedestrian", 500, "night", "fog") == 1.0
    assert o.detection_probability("car", 10, "day", "clear", visible_fraction=0.1) == 0.0
    assert o.position_sigma("car", 80, "night", "rain") == 0.0 and o.ghost_rate_hz == 0.0
    noisy = m.oracle(keep_noise=True, latency_s=0.0)
    assert noisy.position_sigma("car", 80, "night", "rain") == m.position_sigma(
        "car", 80, "night", "rain")
    assert noisy.latency_s == 0.0


def test_counterfactual_with_the_same_seed_sees_the_same_noise():
    """Random draws do not depend on what was detected, so only detection changes."""
    m = PerceptionModel.from_yaml(MODEL)
    m.latency_s = 0.0
    det_only = m.oracle(keep_noise=True)
    near = TrueObject("near", "car", 10.0, 0.0)
    far = TrueObject("far", "car", 150.0, 3.5)          # missed by the model, seen by oracle
    a = m.step(0.0, 0.1, [far, near], "night", "fog", np.random.default_rng(3))
    b = det_only.step(0.0, 0.1, [far, near], "night", "fog", np.random.default_rng(3))
    pos = {d.obj_id: (d.x, d.y) for d in b}
    assert [d.obj_id for d in a] == ["near"] and set(pos) == {"far", "near"}
    assert a[0].x == pos["near"][0] and a[0].y == pos["near"][1]


def test_tracker_can_confirm_on_first_detection():
    from adas_validation.perception import Detection
    assert Tracker(confirm_hits=1).step(0.0, 0.1, [Detection("a", "car", 20, 0, 0.0)])
    assert not Tracker().step(0.0, 0.1, [Detection("a", "car", 20, 0, 0.0)])


def test_aeb_brakes_earlier_when_it_knows_the_lead_is_braking():
    cfg = AdasConfig(set_speed_mps=30.0)
    steady = Track(0, "car", 20.0, 0.0, vx=28.0, confirmed=True)
    braking = Track(0, "car", 20.0, 0.0, vx=28.0, confirmed=True, ax=-6.0)
    assert control(30.0, [steady], cfg, "clear").mode != "aeb"
    assert control(30.0, [braking], cfg, "clear").mode == "aeb"


def test_boundary_and_worst_case():
    xs, ys = [20, 30, 40], [1.0, 2.0]
    ok = {(20, 1.0): True, (30, 1.0): True, (40, 1.0): False,
          (20, 2.0): True, (30, 2.0): False, (40, 2.0): True}   # 40 ok again: not counted
    b = boundary(ok, xs, ys)
    assert b == {1.0: 30, 2.0: 20}
    assert worst_case(b, xs) == 20
    assert worst_case({1.0: 30, 2.0: None}, xs) is None
    assert cell_ok({"pass_rate": 0.0, "ideal_collided": True}, 0.9)   # beyond the design
    assert not cell_ok({"pass_rate": 0.85, "ideal_collided": False}, 0.9)


def test_make_scenario_sets_fields_and_params():
    base = {"kind": "lead_brake", "ego_speed_kph": 90,
            "params": {"lead_decel_mps2": 6.0, "initial_gap_m": 30}}
    sc = make_scenario(base, {"ego_speed_kph": 110, "params.initial_gap_m": 15}, "night", "fog")
    assert isinstance(sc, Scenario) and sc.ego_speed_kph == 110
    assert sc.params == {"lead_decel_mps2": 6.0, "initial_gap_m": 15}
    assert base["params"]["initial_gap_m"] == 30          # base not mutated
    assert sc.lighting == "night" and sc.weather == "fog"


def test_pedestrian_trigger_by_time_scales_with_speed():
    from adas_validation.sim import EGO_LEN, ScenarioScript
    p = {"pedestrian_speed_mps": 1.4, "van_distance_m": 60, "trigger_time_s": 2.0}
    slow = ScenarioScript(Scenario("a", "occluded_pedestrian", 20, params=p), 20 / 3.6)
    fast = ScenarioScript(Scenario("b", "occluded_pedestrian", 50, params=p), 50 / 3.6)
    assert slow.trigger_m == pytest.approx(2.0 * 20 / 3.6 + EGO_LEN / 2)
    assert fast.trigger_m > slow.trigger_m
