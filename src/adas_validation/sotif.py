"""SOTIF-style failure analysis: which triggering condition causes each collision?

    uv run python -m adas_validation.sotif --perception configs/perception_model_measured.yaml \
        --out results/sotif_measured

ISO 21448 (SOTIF) asks under which conditions an intended function is unsafe without any
fault, and whether the cause is a performance insufficiency (the function cannot do what it
should here) or an insufficiency of specification (it was never asked to). Here every failed
run of the perception model under test is attributed by counterfactual reruns of the same
scenario, removing one limitation at a time:

  1. perfect detection   every object detected once at least half visible, at any range,
                         no ghosts; position noise kept (same seed as the failed run)
  2. + no position noise oracle perception (sensor latency and track confirmation kept)
  3. + dry-road braking  9 m/s^2 instead of 6.5 in rain (same as 2 in dry weather)
  4. + instant reaction  no sensor latency, tracks confirmed on their first detection
  5. + known lead decel  the AEB knows the hazard's true acceleration, so it plans to stop
                         within the gap plus a braking lead's stopping distance

The cause is the first step that removes the collision; if none does, it is a design limit:
this AEB (which brakes only for objects predicted in its path) collides even with ideal
inputs. That is not proof that no system could avoid it (a cut-in at 20 m is avoidable by
braking at the first lateral drift), only that this function cannot.
Step 1 is rerun per failed seed.
Traffic is scripted and steps 2-5 draw no random numbers that change anything, so each of
them is one deterministic run per scenario.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from adas_validation.adas import MAX_DECEL
from adas_validation.perception import PerceptionModel
from adas_validation.scenarios import Scenario, load_scenarios, wilson_interval
from adas_validation.sim import RunResult, make_env, run_scenario

ROOT = Path(__file__).resolve().parents[2]
STEPS = (("detection", "detection"), ("oracle", "noise"), ("dry", "friction"),
         ("instant", "reaction"), ("logic", "decision"))
CAUSES = {
    "detection": ("Performance insufficiency: detection range / reliability",
                  "radar fusion or better camera range in this condition; until then "
                  "lower speed / longer time gap when the condition is detected"),
    "noise": ("Performance insufficiency: position noise corrupts the motion estimate",
              "estimate lateral velocity over more frames (Kalman filter) or from radar "
              "Doppler; wider in-path margin for pedestrians near occluders"),
    "friction": ("Insufficiency of specification: speed not adapted to a wet road",
                 "detect rain and lower the set speed; braking is already at the road's limit"),
    "reaction": ("Performance insufficiency: reaction latency (sensor + confirmation)",
                 "confirm fast-closing in-path objects sooner, cut sensor latency"),
    "decision": ("Insufficiency of specification: AEB assumes the lead keeps its speed",
                 "estimate the lead's deceleration (acceleration state in the tracker, "
                 "brake-light detection, V2X) and brake earlier for a braking lead"),
    "design_limit": ("Outside the function's capability: collides even with ideal inputs",
                     "restrict the operating domain (speed, gap) to the sweep boundaries, or "
                     "extend the function (cut-in intent from lateral motion, evasive steering)"),
}
TRIGGER = {"occluded_pedestrian": "pedestrian steps out from behind a parked van",
           "lead_brake": "lead car brakes hard at a short gap",
           "cut_in": "car cuts in at a short gap"}


def attribute(collided: dict[str, bool]) -> str:
    """First counterfactual step that removes the collision, or 'design_limit'."""
    for step, cause in STEPS:
        if not collided[step]:
            return cause
    return "design_limit"


def counterfactuals(sc: Scenario, model: PerceptionModel, env) -> dict[str, RunResult]:
    """Steps 2-5 (deterministic, one run each)."""
    oracle = model.oracle()
    out = {"oracle": run_scenario(sc, 0, oracle, env)}
    out["dry"] = (run_scenario(sc, 0, oracle, env, max_decel=MAX_DECEL["clear"])
                  if MAX_DECEL[sc.weather] < MAX_DECEL["clear"] else out["oracle"])
    instant = model.oracle(latency_s=0.0)
    out["instant"] = run_scenario(sc, 0, instant, env, max_decel=MAX_DECEL["clear"],
                                  confirm_hits=1)
    out["logic"] = run_scenario(sc, 0, instant, env, max_decel=MAX_DECEL["clear"],
                                confirm_hits=1, known_lead_accel=True)
    return out


def condition_text(sc: Scenario) -> str:
    light = {"day": "by day", "night": "at night", "dusk": "at dusk"}[sc.lighting]
    weather = {"clear": "", "rain": ", rain (wet road)", "fog": ", fog"}[sc.weather]
    p, v = sc.params, f"{sc.ego_speed_kph:g} km/h"
    if sc.kind == "lead_brake":
        detail = f"{v}, gap {p['initial_gap_m']:g} m, {p['lead_decel_mps2']:g} m/s^2"
    elif sc.kind == "cut_in":
        detail = f"{v} vs {p['cutter_speed_kph']:g} km/h, gap {p['cut_in_gap_m']:g} m"
    else:
        start = (f"{p['trigger_time_s']:g} s" if "trigger_time_s" in p
                 else f"{p.get('trigger_distance_m', 25):g} m")
        detail = f"{v}, starts {start} ahead"
    return f"{TRIGGER[sc.kind]} ({detail}), {light}{weather}"


def analyse(scenarios: list[Scenario], seeds: int, model: PerceptionModel
            ) -> tuple[list[dict], list[dict]]:
    """Returns (one row per scenario, one row per failed run)."""
    env = make_env()
    rows, failures = [], []
    try:
        for sc in scenarios:
            runs = [run_scenario(sc, s, model, env) for s in range(seeds)]
            failed = [r for r in runs if r.collided]
            cf = counterfactuals(sc, model, env) if failed else {}
            chain = {k: v.collided for k, v in cf.items()}
            detect_only = model.oracle(keep_noise=True)
            causes = Counter()
            for r in failed:
                again = run_scenario(sc, r.seed, detect_only, env)
                cause = attribute({**chain, "detection": again.collided})
                causes[cause] += 1
                failures.append({"scenario": sc.name, "seed": r.seed, "cause": cause,
                                 "impact_speed_kph": round(r.impact_speed_kph, 1),
                                 "first_visible_range_m": r.first_visible_range_m,
                                 "first_detect_range_m": r.first_detect_range_m,
                                 "never_detected": r.first_detect_range_m is None})
            lo, hi = wilson_interval(len(failed), seeds)
            rows.append({
                "scenario": sc.name, "kind": sc.kind, "lighting": sc.lighting,
                "weather": sc.weather, "condition": condition_text(sc), "runs": seeds,
                "failures": len(failed), "failure_ci95": [round(lo, 3), round(hi, 3)],
                "median_impact_kph": (round(statistics.median(r.impact_speed_kph
                                                              for r in failed), 1)
                                      if failed else None),
                "never_detected": sum(r.first_detect_range_m is None for r in failed),
                "causes": dict(causes),
                "counterfactual_impact_kph": {k: round(v.impact_speed_kph, 1)
                                              for k, v in cf.items()},
                "hazard_visible_at_m": (round(runs[0].first_visible_range_m, 1)
                                        if runs[0].first_visible_range_m else None),
            })
    finally:
        env.close()
    return rows, failures


def failure_table(rows: list[dict], model_name: str) -> str:
    failed = sorted((r for r in rows if r["failures"]), key=lambda r: -r["failures"])
    total = sum(r["failures"] for r in rows)
    runs = sum(r["runs"] for r in rows)
    by_cause = Counter()
    for r in failed:
        by_cause.update(r["causes"])
    lines = [f"# SOTIF failure table: `{model_name}`", "",
             f"{total} collisions in {runs} runs. Cause = first counterfactual that removes "
             "the collision (perfect detection -> + no position noise -> + dry-road braking "
             "-> + instant reaction -> + known lead deceleration; see "
             "`adas_validation/sotif.py`).", "",
             "| Cause | SOTIF classification | Collisions | Share |", "|---|---|---:|---:|"]
    for cause, (cls, _) in CAUSES.items():
        if by_cause[cause]:
            lines.append(f"| {cause} | {cls} | {by_cause[cause]} | "
                         f"{by_cause[cause] / total:.0%} |")
    lines += ["", "| Triggering condition | Failures (95% CI) | Median impact (km/h) "
              "| Cause (collisions) | Impact with oracle perception (km/h) | Mitigation |",
              "|---|---|---:|---|---:|---|"]
    for r in failed:
        lo, hi = r["failure_ci95"]
        causes = ", ".join(f"{c} {n}" for c, n in sorted(r["causes"].items(),
                                                         key=lambda kv: -kv[1]))
        main_cause = max(r["causes"], key=r["causes"].get)
        lines.append(
            f"| {r['condition']} | {r['failures']}/{r['runs']} ({lo:.0%}-{hi:.0%}) | "
            f"{r['median_impact_kph']} | {causes} | "
            f"{r['counterfactual_impact_kph']['oracle'] or '-'} | {CAUSES[main_cause][1]} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", default=str(ROOT / "configs/scenarios.yaml"))
    ap.add_argument("--perception", default=str(ROOT / "configs/perception_model.yaml"))
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results/sotif"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, failures = analyse(load_scenarios(args.catalogue), args.seeds,
                             PerceptionModel.from_yaml(args.perception))
    if failures:
        with open(out / "failed_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(failures[0]))
            w.writeheader()
            w.writerows(failures)
    shares: dict[str, int] = defaultdict(int)
    for f in failures:
        shares[f["cause"]] += 1
    (out / "sotif.json").write_text(json.dumps(
        {"perception_model": args.perception, "seeds": args.seeds,
         "collisions_by_cause": dict(shares), "scenarios": rows,
         "causes": {k: dict(zip(("classification", "mitigation"), v, strict=True))
                    for k, v in CAUSES.items()}}, indent=2))
    table = failure_table(rows, Path(args.perception).name)
    (out / "failure_table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
