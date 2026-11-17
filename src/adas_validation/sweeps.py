"""Parameter sweeps: find the pass/fail boundary of each scenario kind, per condition.

    uv run python -m adas_validation.sweeps \
        --perception configs/perception_model_measured.yaml \
                     configs/perception_model_nuscenes_anchored.yaml \
        --labels sim-measured nuScenes-anchored --out results/sweeps

Reads configs/sweeps.yaml. Writes cells.csv (one row per grid cell), boundaries.json and one
heatmap per sweep in docs/figures/sweep_<name>.png (rows: perception models, columns:
conditions; the red line is the boundary). boundaries.json also has the worst case over y:
the speed limit covering every y.

Some cells are beyond what this AEB can do with any perception: a pedestrian who steps out
1.5 s ahead is revealed within a metre or two of the bumper. Each cell therefore also gets one
deterministic run of the same AEB with ideal inputs (oracle perception, no latency,
confirmation on the first detection, the lead's true deceleration known, braking at the real
road's limit). A cell counts against the system only if the ideal-input AEB avoids the
collision there (hatched cells: it does not). Boundary: per y,
the highest speed such that every speed up to it passes or is unavoidable.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import yaml

from adas_validation.perception import PerceptionModel
from adas_validation.scenarios import Scenario
from adas_validation.sim import make_env, run_scenario

ROOT = Path(__file__).resolve().parents[2]


def make_scenario(base: dict, assignments: dict, lighting: str, weather: str) -> Scenario:
    """Copy of the base scenario with fields set; 'params.k' sets a scenario parameter."""
    fields = {k: v for k, v in base.items() if k != "params"}
    params = dict(base.get("params", {}))
    for field, value in assignments.items():
        if field.startswith("params."):
            params[field.removeprefix("params.")] = value
        else:
            fields[field] = value
    name = "_".join([fields["kind"], lighting, weather] +
                    [f"{f.split('.')[-1]}{v}" for f, v in assignments.items()])
    return Scenario(name=name, lighting=lighting, weather=weather, params=params, **fields)


def boundary(ok: dict[tuple, bool], xs: list, ys: list) -> dict:
    """Per y, the largest x such that every x' <= x is ok (None if the smallest x is not)."""
    out = {}
    for y in ys:
        best = None
        for x in xs:
            if not ok[(x, y)]:
                break
            best = x
        out[y] = best
    return out


def worst_case(bound: dict, xs: list):
    """Largest x that passes for every y (None if some y fails even at the smallest x)."""
    if any(v is None for v in bound.values()):
        return None
    return min(bound.values(), key=xs.index)


def cell_ok(row: dict, threshold: float) -> bool:
    """Passes often enough, or the AEB collides there even with ideal inputs."""
    return row["pass_rate"] >= threshold or row["ideal_collided"]


def run_sweeps(cfg: dict, model: PerceptionModel, label: str) -> list[dict]:
    rows = []
    ideal_model = model.oracle(latency_s=0.0)
    env = make_env()
    try:
        for sw in cfg["sweeps"]:
            for lighting, weather in sw["conditions"]:
                for x in sw["x"]["values"]:
                    for y in sw["y"]["values"]:
                        sc = make_scenario(sw["base"], {sw["x"]["field"]: x,
                                                        sw["y"]["field"]: y},
                                           lighting, weather)
                        runs = [run_scenario(sc, s, model, env) for s in range(cfg["seeds"])]
                        impacts = [r.impact_speed_kph for r in runs if r.collided]
                        ideal = run_scenario(sc, 0, ideal_model, env, confirm_hits=1,
                                             known_lead_accel=True)
                        rows.append({
                            "model": label, "sweep": sw["name"], "lighting": lighting,
                            "weather": weather, "x": x, "y": y, "runs": len(runs),
                            "passed": len(runs) - len(impacts),
                            "pass_rate": round(1 - len(impacts) / len(runs), 3),
                            "median_impact_kph": (round(statistics.median(impacts), 1)
                                                  if impacts else None),
                            "ideal_collided": ideal.collided,
                            "ideal_impact_kph": round(ideal.impact_speed_kph, 1)})
            print(f"{label}: {sw['name']} done", flush=True)
    finally:
        env.close()
    return rows


def plot(sw: dict, rows: list[dict], labels: list[str], bounds: dict, threshold: float,
         path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    xs, ys, conds = sw["x"]["values"], sw["y"]["values"], sw["conditions"]
    fig, axes = plt.subplots(len(labels), len(conds), squeeze=False,
                             figsize=(2.9 * len(conds) + 0.8, 2.6 * len(labels) + 0.6),
                             sharex=True, sharey=True)
    for i, label in enumerate(labels):
        for j, (lighting, weather) in enumerate(conds):
            ax = axes[i][j]
            cell = {(r["x"], r["y"]): r for r in rows
                    if r["model"] == label and r["sweep"] == sw["name"]
                    and r["lighting"] == lighting and r["weather"] == weather}
            grid = np.array([[cell[(x, y)]["pass_rate"] for x in xs] for y in ys])
            im = ax.imshow(grid, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for a in range(len(ys)):
                for b in range(len(xs)):
                    v = grid[a, b]
                    ax.text(b, a, f"{v:.0%}"[:-1], ha="center", va="center", fontsize=6.5,
                            color="white" if v > 0.6 else "#1a1a1a")
                    if cell[(xs[b], ys[a])]["ideal_collided"]:
                        ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False,
                                               hatch="////", edgecolor="#9a9a9a", lw=0))
            bd = bounds[label][sw["name"]][f"{lighting}/{weather}"]
            pts = []
            for a, y in enumerate(ys):
                edge = (xs.index(bd[y]) if bd[y] is not None else -1) + 0.5
                pts += [(edge, a - 0.5), (edge, a + 0.5)]
            ax.plot(*zip(*pts, strict=True), color="#d62728", lw=2)
            wc = worst_case(bd, xs)
            verdict = (f"safe for every y up to {wc} km/h" if wc is not None
                       else "no tested speed is safe for every y")
            ax.set_title(f"{label}: {lighting}, {weather}\n{verdict}", fontsize=8)
            ax.set_xticks(range(len(xs)), xs, fontsize=7)
            ax.set_yticks(range(len(ys)), ys, fontsize=7)
            if i == len(labels) - 1:
                ax.set_xlabel(sw["x"]["label"], fontsize=8)
            if j == 0:
                ax.set_ylabel(sw["y"]["label"], fontsize=7)
    fig.suptitle(f"{sw['title']}\npass rate (%); hatched = collides even with ideal inputs; "
                 f"red line = boundary (pass rate >= {threshold:.0%} or hatched)",
                 fontsize=9, y=1.0 + 0.09 / len(labels))
    fig.colorbar(im, ax=axes, shrink=0.6, label="pass rate")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def parse_cell(r: dict) -> dict:
    def num(v: str) -> float | int:
        return float(v) if "." in v else int(v)

    return {**r, "x": num(r["x"]), "y": num(r["y"]), "pass_rate": float(r["pass_rate"]),
            "ideal_collided": r["ideal_collided"] == "True"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/sweeps.yaml"))
    ap.add_argument("--perception", nargs="+",
                    default=[str(ROOT / "configs/perception_model_measured.yaml")])
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=None, help="override the config")
    ap.add_argument("--out", default=str(ROOT / "results/sweeps"))
    ap.add_argument("--from-cells", action="store_true",
                    help="skip the runs: recompute boundaries and figures from <out>/cells.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seeds:
        cfg["seeds"] = args.seeds
    labels = args.labels or [Path(p).stem for p in args.perception]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    if args.from_cells:
        rows = [parse_cell(r) for r in csv.DictReader(open(out / "cells.csv"))]
    else:
        for path, label in zip(args.perception, labels, strict=True):
            rows += run_sweeps(cfg, PerceptionModel.from_yaml(path), label)
        with open(out / "cells.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    thr = cfg["pass_threshold"]
    bounds: dict = {}
    for label in labels:
        bounds[label] = {}
        for sw in cfg["sweeps"]:
            bounds[label][sw["name"]] = {}
            for lighting, weather in sw["conditions"]:
                ok = {(r["x"], r["y"]): cell_ok(r, thr) for r in rows
                      if r["model"] == label and r["sweep"] == sw["name"]
                      and r["lighting"] == lighting and r["weather"] == weather}
                bounds[label][sw["name"]][f"{lighting}/{weather}"] = boundary(
                    ok, sw["x"]["values"], sw["y"]["values"])
    wall = round(time.time() - t0)
    if args.from_cells and (out / "boundaries.json").exists():   # keep the original run time
        wall = json.loads((out / "boundaries.json").read_text()).get("wall_s", wall)
    (out / "boundaries.json").write_text(json.dumps(
        {"seeds": cfg["seeds"], "pass_threshold": thr, "wall_s": wall,
         "models": dict(zip(labels, args.perception, strict=True)),
         "worst_case_max_speed_kph": {
             lb: {sw["name"]: {c: worst_case(b, sw["x"]["values"])
                               for c, b in per[sw["name"]].items()} for sw in cfg["sweeps"]}
             for lb, per in bounds.items()},
         "boundaries": {lb: {sw: {c: {str(k): v for k, v in b.items()}
                                  for c, b in conds.items()}
                             for sw, conds in per.items()}
                        for lb, per in bounds.items()}}, indent=2))
    for sw in cfg["sweeps"]:
        plot(sw, rows, labels, bounds, thr, ROOT / "docs/figures" / f"sweep_{sw['name']}.png")
    print(json.dumps(bounds, indent=1, default=str))


if __name__ == "__main__":
    main()
