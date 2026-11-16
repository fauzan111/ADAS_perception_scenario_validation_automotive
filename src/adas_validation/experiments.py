"""Run the scenario catalogue many times and report pass rates, margins and failures.

    uv run python -m adas_validation.experiments --seeds 30 --out results/catalogue

Each scenario is run once per seed. Traffic is scripted, so seeds differ only in perception
(detection draws, noise, ghosts). Verdict: pass = no collision (Euro NCAP style); the
minimum true TTC is reported as the safety margin and the impact speed for failures.
Outputs: runs.csv (one row per run), summary.json and report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from adas_validation.perception import PerceptionModel
from adas_validation.scenarios import Scenario, load_scenarios, wilson_interval
from adas_validation.sim import RunResult, make_env, run_scenario

ROOT = Path(__file__).resolve().parents[2]


def summarise(rows: list[dict], key) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = []
    for name, rs in sorted(groups.items()):
        passed = sum(not r["collided"] for r in rs)
        lo, hi = wilson_interval(passed, len(rs))
        impacts = [r["impact_speed_kph"] for r in rs if r["collided"]]
        ttcs = [r["min_ttc_s"] for r in rs if not r["collided"] and math.isfinite(r["min_ttc_s"])]
        det = [r["first_detect_range_m"] for r in rs if r["first_detect_range_m"] is not None]
        out.append({
            "group": name, "runs": len(rs), "passed": passed, "pass_rate": passed / len(rs),
            "ci95": [round(lo, 3), round(hi, 3)],
            "median_impact_kph": round(statistics.median(impacts), 1) if impacts else None,
            "median_min_ttc_s": round(statistics.median(ttcs), 2) if ttcs else None,
            "median_detect_range_m": round(statistics.median(det), 1) if det else None,
            "never_detected": sum(r["first_detect_range_m"] is None for r in rs),
        })
    return out


def table(rows: list[dict], title: str) -> str:
    lines = [f"### {title}", "",
             "| Group | Pass rate (95% CI) | Collisions | Median impact (km/h) "
             "| Median min TTC if passed (s) | Median first detection (m) |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['passed']}/{r['runs']} = {r['pass_rate']:.0%} "
            f"({r['ci95'][0]:.0%}-{r['ci95'][1]:.0%}) | {r['runs'] - r['passed']} | "
            f"{r['median_impact_kph'] or '-'} | {r['median_min_ttc_s'] or '-'} | "
            f"{r['median_detect_range_m'] or '-'} |")
    return "\n".join(lines) + "\n"


def run_catalogue(scenarios: list[Scenario], seeds: int, model: PerceptionModel
                  ) -> list[dict]:
    env = make_env()
    rows = []
    try:
        for sc in scenarios:
            for seed in range(seeds):
                r: RunResult = run_scenario(sc, seed, model, env)
                row = {k: v for k, v in asdict(r).items() if k != "trace"}
                row.update(kind=sc.kind, lighting=sc.lighting, weather=sc.weather,
                           ego_speed_kph=sc.ego_speed_kph)
                rows.append(row)
    finally:
        env.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", default=str(ROOT / "configs/scenarios.yaml"))
    ap.add_argument("--perception", default=str(ROOT / "configs/perception_model.yaml"))
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results/catalogue"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(args.catalogue)
    model = PerceptionModel.from_yaml(args.perception)
    t0 = time.time()
    rows = run_catalogue(scenarios, args.seeds, model)
    wall = time.time() - t0
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = {
        "scenario": summarise(rows, lambda r: r["scenario"]),
        "kind": summarise(rows, lambda r: r["kind"]),
        "lighting": summarise(rows, lambda r: r["lighting"]),
        "weather": summarise(rows, lambda r: r["weather"]),
        "kind x condition": summarise(
            rows, lambda r: f"{r['kind']} / {r['lighting']} / {r['weather']}"),
    }
    overall = summarise(rows, lambda r: "all")[0]
    (out / "summary.json").write_text(json.dumps(
        {"runs": len(rows), "seeds": args.seeds, "wall_s": round(wall, 1),
         "perception_model": args.perception, "overall": overall, "groups": groups},
        indent=2))
    report = [f"# Scenario catalogue results\n\n{len(rows)} runs ({len(scenarios)} scenarios x "
              f"{args.seeds} seeds) in {wall:.0f} s. Perception model: "
              f"`{Path(args.perception).name}` (assumed parameters until calibrated). "
              f"Pass = no collision.\n\nOverall: {overall['passed']}/{overall['runs']} = "
              f"{overall['pass_rate']:.0%} (95% CI {overall['ci95'][0]:.0%}-"
              f"{overall['ci95'][1]:.0%}).\n"]
    for title in ("kind", "lighting", "weather", "scenario"):
        report.append(table(groups[title], f"By {title}"))
    (out / "report.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
