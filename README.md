# ADAS Perception + Scenario Validation

3D object detection on nuScenes, stress-tested in simulated driving scenarios (cut-in, occluded pedestrian, night, rain, fog) with safety metrics and a table of the conditions where perception fails.

## Why it matters

Software-defined vehicle programmes such as Stellantis STLA Brain are limited by validation, not by models. Engineering consultancies staff OEM projects with engineers who can design scenarios, measure safety and explain failures. This project adds camera/LiDAR perception and validation to a portfolio that already covers fleet data, driver state and racing control.

**Relevant employers in Italy:** Stellantis, Italdesign, Teoresi, Akkodis, Reply, VisLab (Ambarella), Maserati

## What it does

- 3D / BEV detection on nuScenes-mini (mAP, NDS), broken down by range, class, night and rain
- Scenario catalogue (YAML) expanded over lighting x weather: cut-in, occluded pedestrian,
  lead-vehicle braking
- Closed loop in MetaDrive: ground truth -> **perception error model** (detection probability
  by range and condition, noise, latency, ghosts, line-of-sight occlusion) -> tracker ->
  ACC/AEB -> vehicle. The nuScenes measurements calibrate the error model.
- Safety metrics from ground truth: collisions, impact speed, minimum TTC, first-detection
  range; pass rates with Wilson 95% confidence intervals
- SOTIF-style table of triggering conditions
- Showcase run in CARLA on a cloud GPU (docs/COMPUTE.md)

## Results

First catalogue run: 18 scenarios x 30 perception seeds = 540 runs (about 20 s on a Mac).
**The perception error model still uses assumed parameters**; these numbers will change once
it is calibrated from nuScenes.

| Metric | Value |
|---|---|
| mAP / NDS (nuScenes-mini) | pending (dataset download) |
| Scenario pass rate, overall (no collision) | 368/540 = 68% (95% CI 64-72%) |
| Pass rate day vs night | 79% vs 57% |
| Pass rate clear / rain / fog | 89% / 56% / 59% |
| Hardest scenario | occluded pedestrian in rain: 0/30 by day and by night |

Full tables: `make scenarios` writes `results/catalogue/report.md`. What went wrong on the
way (and was fixed) is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
make setup        # core only
make setup-sim    # + MetaDrive (Python 3.11)
make test         # 16 tests, including a MetaDrive smoke run
make scenarios    # catalogue x 30 seeds -> results/catalogue/report.md
```

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md).

## In italiano

Percezione 3D su nuScenes e validazione basata su scenari (sorpasso stretto, pedone occluso, notte, pioggia, nebbia) con metriche di sicurezza come il time-to-collision e un'analisi delle condizioni critiche in stile SOTIF.

## Licence

Code: MIT, see [LICENSE](LICENSE). Datasets and third-party
assets keep their own licences; see [data/README.md](data/README.md).
