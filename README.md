# ADAS Perception + Scenario Validation

3D object detection on nuScenes, stress-tested in simulated driving scenarios (cut-in, occluded pedestrian, night, rain, fog) with safety metrics and a table of the conditions where perception fails.

## Why it matters

Software-defined vehicle programmes such as Stellantis STLA Brain are limited by validation, not by models. Engineering consultancies staff OEM projects with engineers who can design scenarios, measure safety and explain failures. This project adds camera/LiDAR perception and validation to a portfolio that already covers fleet data, driver state and racing control.

**Relevant employers in Italy:** Stellantis, Italdesign, Teoresi, Akkodis, Reply, VisLab (Ambarella), Maserati

## What it does

- 3D / BEV detection on nuScenes-mini (mAP, NDS)
- Declarative scenario catalogue (YAML) with validated parameters
- Safety metrics: time-to-collision, scenario pass rate by lighting and weather
- SOTIF-style table of triggering conditions
- Fast loop in MetaDrive on Mac; showcase run in CARLA on a cloud GPU (docs/COMPUTE.md)

## Results

| Metric | Value |
|---|---|
| mAP / NDS (nuScenes-mini) | TBD |
| Scenario pass rate (overall) | TBD |
| Pass rate at night vs day | TBD |
| Worst-case minimum TTC (s) | TBD |

*Filled in as the roadmap progresses.*

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
make setup        # core only
make setup-all    # + perception and simulator extras
make test
```

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md).

## In italiano

Percezione 3D su nuScenes e validazione basata su scenari (sorpasso stretto, pedone occluso, notte, pioggia, nebbia) con metriche di sicurezza come il time-to-collision e un'analisi delle condizioni critiche in stile SOTIF.

## Licence

Code: MIT, see [LICENSE](LICENSE). Datasets and third-party
assets keep their own licences; see [data/README.md](data/README.md).
