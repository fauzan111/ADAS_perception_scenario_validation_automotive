"""Scenario catalogue: declarative test cases loaded from YAML, plus a pass-rate report."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_LIGHTING = {"day", "dusk", "night"}
VALID_WEATHER = {"clear", "rain", "fog"}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str                      # cut_in, occluded_pedestrian, lead_brake, ...
    ego_speed_kph: float
    lighting: str = "day"
    weather: str = "clear"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lighting not in VALID_LIGHTING:
            raise ValueError(f"{self.name}: lighting must be one of {sorted(VALID_LIGHTING)}")
        if self.weather not in VALID_WEATHER:
            raise ValueError(f"{self.name}: weather must be one of {sorted(VALID_WEATHER)}")
        if self.ego_speed_kph <= 0:
            raise ValueError(f"{self.name}: ego speed must be positive")


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load the catalogue; scenarios without lighting/weather are expanded over `matrix`."""
    raw = yaml.safe_load(Path(path).read_text())
    matrix = raw.get("matrix")
    scenarios = []
    for s in raw["scenarios"]:
        if matrix and "lighting" not in s and "weather" not in s:
            for light in matrix["lighting"]:
                for weather in matrix["weather"]:
                    scenarios.append(Scenario(**{**s, "name": f"{s['name']}_{light}_{weather}",
                                                 "lighting": light, "weather": weather}))
        else:
            scenarios.append(Scenario(**s))
    names = [s.name for s in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("scenario names must be unique")
    return scenarios


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    passed: bool
    min_ttc_s: float


def pass_rate_by(results: list[ScenarioResult], key: str) -> dict[str, float]:
    """Pass rate grouped by a scenario attribute, e.g. 'kind', 'lighting' or 'weather'."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        groups[str(getattr(r.scenario, key))].append(r.passed)
    return {k: sum(v) / len(v) for k, v in sorted(groups.items())}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion (sensible at 0 % or 100 %)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
