import math
from pathlib import Path

import pytest

from adas_validation.metrics import scenario_passed, time_to_collision
from adas_validation.scenarios import Scenario, ScenarioResult, load_scenarios, pass_rate_by

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "scenarios.yaml"


def test_ttc():
    assert time_to_collision(30, 20, 10) == pytest.approx(3.0)
    assert math.isinf(time_to_collision(30, 10, 20))


def test_scenario_rules():
    assert scenario_passed(2.0, False)
    assert not scenario_passed(1.0, False)
    assert not scenario_passed(5.0, True)


def test_catalogue_loads():
    sc = load_scenarios(CONFIG)
    assert len(sc) == 5
    assert {s.kind for s in sc} == {"cut_in", "occluded_pedestrian", "lead_brake"}


def test_invalid_scenario_rejected():
    with pytest.raises(ValueError):
        Scenario(name="x", kind="cut_in", ego_speed_kph=50, weather="snowstorm")


def test_pass_rate_grouping():
    day = Scenario("a", "cut_in", 90)
    night = Scenario("b", "cut_in", 90, lighting="night")
    res = [ScenarioResult(day, True, 3.0), ScenarioResult(night, False, 0.8),
           ScenarioResult(night, True, 2.0)]
    assert pass_rate_by(res, "lighting") == {"day": 1.0, "night": 0.5}
