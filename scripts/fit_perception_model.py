"""Fit the perception error model to the detector calibration and compare with nuScenes.

    uv run python scripts/fit_perception_model.py results/calibration

Reads detections.jsonl from calibrate_detector.py, fits p(r) = p_max / (1 + exp((r - r50) / s))
per class and condition by maximum likelihood, and writes
  configs/perception_model_measured.yaml   (classes from day/clear; range factors measured)
  results/calibration/fit.json, docs/figures/detector_recall.png
Position noise and latency are not measured here and keep their assumed values.

Cross-check (published, real data): camera-only DETR3D car AP by distance on nuScenes val,
TransCAR (arXiv 2305.00397) Table 6. AP is not detection probability and 3D detection is
harder than 2D boxes, so this is a sanity comparison, not a calibration target.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
NUSCENES_DETR3D_CAR_AP = {"source": "TransCAR, arXiv 2305.00397, Table 6 (DETR3D, nuScenes val)",
                          "bins_m": [[0, 20], [20, 30], [30, 40], [40, 50]],
                          "ap": [0.764, 0.487, 0.288, 0.050]}
MONODIS_CLASS_AP = {"source": "nuScenes paper, arXiv 1903.11027, Table 6 (MonoDIS, test)",
                    "car": 0.478, "pedestrian": 0.370}
NUSCENES_NIGHT_RAIN_CAR_AP = {"source": "TransCAR Tables 7-8 (DETR3D, nuScenes val)",
                              "day": 0.548, "night": 0.522, "no_rain": 0.547, "rain": 0.542}


def logistic(r, r50, s, pmax):
    return pmax / (1.0 + np.exp((r - r50) / s))


def fit(ranges: np.ndarray, hits: np.ndarray) -> dict:
    """Maximum-likelihood grid search (robust, no optimiser dependencies)."""
    best = None
    for pmax in np.linspace(0.05, 1.0, 39):
        for s in np.linspace(1.0, 30.0, 59):
            r50 = np.linspace(1.0, 200.0, 400)[:, None]
            p = np.clip(logistic(ranges[None, :], r50, s, pmax), 1e-6, 1 - 1e-6)
            ll = (hits * np.log(p) + (1 - hits) * np.log(1 - p)).sum(axis=1)
            i = int(np.argmax(ll))
            if best is None or ll[i] > best[0]:
                best = (ll[i], float(r50[i, 0]), float(s), float(pmax))
    return {"r50_m": round(best[1], 1), "softness_m": round(best[2], 1),
            "p_max": round(best[3], 3), "loglik": round(float(best[0]), 2), "n": int(len(hits))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calibration", type=Path)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.calibration / "detections.jsonl")]
    meta = json.loads((args.calibration / "meta.json").read_text())
    # Positions whose object centre is outside the camera's field of view are geometry, not
    # detector recall (a car one lane over at 5 m is simply not in a 60 deg image).
    half = np.tan(np.radians(meta["hfov_deg"]) / 2)
    n_all = len(rows)
    rows = [r for r in rows if abs(r["lat_m"]) < r["range_m"] * half]
    print(f"kept {len(rows)}/{n_all} rows inside the field of view")
    data: dict = defaultdict(lambda: ([], []))
    for r in rows:
        data[(r["cls"], r["cond"])][0].append(r["range_m"])
        data[(r["cls"], r["cond"])][1].append(1.0 if r["hit"] else 0.0)
    fits, curves = {}, {}
    for (cls, cond), (rs, hs) in sorted(data.items()):
        rs, hs = np.array(rs, float), np.array(hs, float)
        fits[f"{cls}/{cond}"] = fit(rs, hs)
        u = np.unique(rs)
        curves[f"{cls}/{cond}"] = {"range_m": u.tolist(),
                                   "recall": [float(hs[rs == x].mean()) for x in u]}
    factor = {}
    for cond in ("night", "fog", "rain", "night_fog", "night_rain"):
        factor[cond] = {cls: round(fits[f"{cls}/{cond}"]["r50_m"] / fits[f"{cls}/day"]["r50_m"],
                                   3) for cls in ("car", "pedestrian")}
    mean = {c: round(float(np.mean(list(v.values()))), 3) for c, v in factor.items()}
    multiplicative = {c: {"measured": mean[f"night_{c}"],
                          "product": round(mean["night"] * mean[c], 3)} for c in ("fog", "rain")}

    base = yaml.safe_load((ROOT / "configs/perception_model.yaml").read_text())
    model = {
        "latency_s": base["latency_s"], "min_visible_fraction": base["min_visible_fraction"],
        "ghost_rate_hz": base["ghost_rate_hz"],
        "classes": {cls: {"r50_m": fits[f"{cls}/day"]["r50_m"],
                          "softness_m": fits[f"{cls}/day"]["softness_m"],
                          "p_max": fits[f"{cls}/day"]["p_max"],
                          "sigma_m": base["classes"][cls]["sigma_m"],
                          "sigma_growth": base["classes"][cls]["sigma_growth"]}
                    for cls in ("car", "pedestrian")},
        "range_factor": {"day": 1.0, "dusk": base["range_factor"]["dusk"],
                         "night": mean["night"], "clear": 1.0, "rain": mean["rain"],
                         "fog": mean["fog"]},
        "noise_factor": base["noise_factor"],
        "class_range_factor": {cls: {"night": factor["night"][cls], "rain": factor["rain"][cls],
                                     "fog": factor["fog"][cls]}
                               for cls in ("car", "pedestrian")},
    }
    header = ("# Perception error model MEASURED with scripts/calibrate_detector.py +\n"
              "# fit_perception_model.py: torchvision Faster R-CNN MobileNetV3-FPN (COCO) on\n"
              "# MetaDrive camera renders (960x540, 60 deg HFOV, 1.4 m mount), day image degraded\n"
              "# for night / fog / rain. Detection curves and per-class range factors are\n"
              "# measured; dusk, position noise, latency and ghost rate keep the assumed\n"
              "# values from perception_model.yaml. Synthetic images: see README for the gap\n"
              "# to nuScenes.\n")
    (ROOT / "configs/perception_model_measured.yaml").write_text(
        header + yaml.safe_dump(model, sort_keys=False))
    # nuScenes-anchored variant: car curve fitted (least squares) to DETR3D AP by distance,
    # pedestrian range scaled by the camera pedestrian/car AP ratio (MonoDIS); condition
    # factors from the renders. A real-world bracket for the optimistic sim-measured curves.
    mids = np.array([(a + b) / 2 for a, b in NUSCENES_DETR3D_CAR_AP["bins_m"]])
    apv = np.array(NUSCENES_DETR3D_CAR_AP["ap"])
    grid = [(float(((logistic(mids, r50, sft, pm) - apv) ** 2).sum()), r50, sft, pm)
            for r50 in np.linspace(5, 60, 111) for sft in np.linspace(1, 20, 39)
            for pm in np.linspace(0.6, 1.0, 21)]
    _, r50c, sc, pmc = (float(v) for v in min(grid))
    ratio = MONODIS_CLASS_AP["pedestrian"] / MONODIS_CLASS_AP["car"]
    anchored = json.loads(json.dumps(model))
    anchored["classes"]["car"].update(r50_m=round(r50c, 1), softness_m=round(sc, 1),
                                      p_max=round(pmc, 2))
    anchored["classes"]["pedestrian"].update(r50_m=round(r50c * ratio, 1),
                                             softness_m=round(sc * ratio, 1),
                                             p_max=round(pmc, 2))
    (ROOT / "configs/perception_model_nuscenes_anchored.yaml").write_text(
        "# Perception error model ANCHORED TO PUBLISHED nuScenes RESULTS (no nuScenes download):\n"
        "# car curve fitted to camera-only DETR3D car AP by distance (TransCAR, arXiv\n"
        "# 2305.00397, Table 6); pedestrian range x MonoDIS pedestrian/car AP ratio (nuScenes\n"
        "# paper, Table 6). AP is used as a proxy for detection probability. Condition factors\n"
        "# come from the MetaDrive renders (perception_model_measured.yaml).\n"
        + yaml.safe_dump(anchored, sort_keys=False))
    out = {"fits": fits, "range_factor_per_class": factor, "range_factor_mean": mean,
           "nuscenes_anchored": anchored["classes"],
           "multiplicative_check": multiplicative, "curves": curves,
           "nuscenes_reference": [NUSCENES_DETR3D_CAR_AP, NUSCENES_NIGHT_RAIN_CAR_AP,
                                  MONODIS_CLASS_AP]}
    (args.calibration / "fit.json").write_text(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colours = {"day": "#1f77b4", "night": "#555555", "fog": "#9467bd", "rain": "#2ca02c",
               "night_fog": "#d62728", "night_rain": "#ff7f0e"}
    for ax, cls in zip(axes, ("car", "pedestrian"), strict=True):
        for cond, col in colours.items():
            c, ft = curves[f"{cls}/{cond}"], fits[f"{cls}/{cond}"]
            ax.plot(c["range_m"], c["recall"], "o", color=col, ms=3, alpha=0.7)
            rr = np.linspace(0, max(c["range_m"]), 200)
            ax.plot(rr, logistic(rr, ft["r50_m"], ft["softness_m"], ft["p_max"]), color=col,
                    label=f"{cond} (r50 {ft['r50_m']:.0f} m)")
        if cls == "car":
            ref = NUSCENES_DETR3D_CAR_AP
            mids = [(a + b) / 2 for a, b in ref["bins_m"]]
            ax.plot(mids, ref["ap"], "s--", color="black", label="nuScenes DETR3D car AP (3D)")
        ax.set_title(f"{cls}: recall vs range (MetaDrive renders)")
        ax.set_xlabel("range from camera (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("detection rate")
    fig.tight_layout()
    (ROOT / "docs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "docs/figures/detector_recall.png", dpi=130)
    print(json.dumps({"fits": fits, "range_factor_mean": mean, "per_class": factor,
                      "multiplicative_check": multiplicative}, indent=1))


if __name__ == "__main__":
    main()
