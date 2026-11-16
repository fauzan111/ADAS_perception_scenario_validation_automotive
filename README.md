# ADAS Perception + Scenario Validation

Closed-loop validation of a camera-based ACC/AEB system in simulated driving scenarios: a car cutting in, a pedestrian stepping out from behind a parked van, a lead car braking hard. Each runs by day and at night, in clear weather, rain and fog. A real detector, measured on simulator images and cross-checked against published nuScenes results, drives a perception error model. Every verdict comes from simulator ground truth.

![Detector recall vs range by condition, with nuScenes DETR3D car AP for comparison](docs/figures/detector_recall.png)

## Why it matters

Software-defined vehicle programmes such as Stellantis STLA Brain are limited by validation, not by models. Engineering consultancies staff OEM projects with engineers who can design scenarios, measure safety and explain failures. SOTIF (ISO 21448) asks exactly this: under which conditions does the intended function fail, and why?

**Relevant employers in Italy:** Stellantis, Italdesign, Teoresi, Akkodis, Reply, VisLab (Ambarella), Maserati

## How it works

```
MetaDrive (ego physics, scripted actors, 3-lane road)
   │ ground truth: positions, line-of-sight occlusion
   ▼
perception error model ── detection probability vs range, per class and condition,
   │                       noise, 0.1 s latency, ghost objects
   ▼
tracker (M-of-N confirmation) ──► ACC + AEB (width-aware path prediction,
                                  wet-road braking limit) ──► throttle / brake
verdict from ground truth: collision, impact speed, minimum TTC, first-detection range
```

**Where the perception parameters come from.** No dataset download is needed. Three models are compared:

| Model | Detection curves | Source |
|---|---|---|
| assumed | hand-set, plausible values | `configs/perception_model.yaml` |
| **sim-measured** | **measured**: torchvision Faster R-CNN (COCO) on 272 MetaDrive camera renders × 6 conditions (night lighting, Koschmieder fog and rain applied with per-pixel scene depth) | `make render-calibration fit-calibration` |
| **nuScenes-anchored** | car curve fitted to camera-only DETR3D car AP by distance on nuScenes val ([TransCAR, Table 6](https://arxiv.org/abs/2305.00397)); pedestrian range scaled by the MonoDIS pedestrian/car AP ratio ([nuScenes paper, Table 6](https://arxiv.org/abs/1903.11027)); condition factors from the renders | same |

## Results

540 runs per model (18 scenarios × 30 perception seeds, about 20 s on a Mac). Pass means no collision.

| | assumed | sim-measured | nuScenes-anchored |
|---|---:|---:|---:|
| **Overall** | 68% | 59% | 34% |
| Cut-in (20 m gap, 90 vs 60 km/h) | 100% | 100% | 79% |
| Lead car brakes at 6 m/s² (90 km/h, 30 m gap) | 74% | 35% | 9% |
| Occluded pedestrian (40 km/h) | 31% | 41% | 14% |
| **Occluded pedestrian, rain (day and night)** | **0/60** | **0/60** | **0/60** |
| Lead car brakes, night, clear | 100% | 100% | 20% |

**What the measurements show**
- **Lit cars barely lose range at night; unlit pedestrians do.** Night detection range vs day: car 0.95, pedestrian 0.58. nuScenes agrees for cars: DETR3D car AP is 52.2% at night vs 54.8% by day ([TransCAR, Table 8](https://arxiv.org/abs/2305.00397)). The perception model therefore has per-class condition factors.
- **Fog and night compound.** The range factor measured for both together is 0.28, against 0.35 predicted by multiplying the two separate effects. Night and rain roughly multiply (0.37 vs 0.35).
- **There is a sim-to-real gap.** A 2D detector on clean simulator images keeps cars out to about 66 m; camera-only 3D detection on nuScenes falls to 50% AP at about 28 m. The sim-measured model is optimistic, which is why the anchored model exists. Synthetic rain hits this detector's car class much harder than nuScenes' rain subset hits DETR3D, where much "rain" footage is wet road rather than falling rain.

**What doesn't depend on the perception model.** The occluded pedestrian in rain fails 60 of 60 under all three models. She becomes visible at about 10 m, and from 40 km/h on a wet road (6.5 m/s²) the car needs about 9.5 m plus reaction time to stop. Better perception can't fix it; the triggering condition is occlusion plus friction, so the mitigation is speed near parked vehicles. Lead-car braking, by contrast, depends entirely on detection range: with realistic camera-only range, a 30 m gap at 90 km/h sits at the edge of what the system can handle.

All findings are from simulation, with stated assumptions: position noise, latency and ghost rate are still assumed, and AP is used as a proxy for detection probability. What went wrong along the way (and was fixed) is in [docs/ROADMAP.md](docs/ROADMAP.md).

## Quickstart

You need [uv](https://docs.astral.sh/uv/). Docker is only needed for the camera calibration.

```bash
make setup-sim            # MetaDrive (Python 3.11)
make test                 # 21 tests, including a MetaDrive smoke run
make scenarios            # assumed model     -> results/catalogue/report.md
make scenarios-measured   # sim-measured model -> results/catalogue_measured/report.md
make render-calibration   # Docker: render + detect (~10 min; macOS can't render offscreen)
make fit-calibration      # fit curves -> configs/perception_model_*.yaml + the figure above
```

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). Optional later steps: run a 3D detector on nuScenes-mini yourself (needs the free registration and a 4 GB download), and run the same catalogue in CARLA on a cloud GPU.

## In italiano

Validazione a ciclo chiuso di un sistema ACC/AEB basato su telecamera in scenari simulati (MetaDrive): veicolo che si inserisce, pedone che esce da dietro un furgone, frenata del veicolo davanti. Ogni scenario è testato di giorno e di notte, con sereno, pioggia e nebbia. Il modello di errore della percezione è misurato con un rilevatore reale (Faster R-CNN) su immagini del simulatore e confrontato con i risultati pubblicati su nuScenes. Il risultato più solido: il pedone occluso con pioggia fallisce in tutti i casi, con tutti i modelli di percezione. Il limite è l'occlusione più l'aderenza, non il rilevatore. È un'analisi delle condizioni critiche in stile SOTIF (ISO 21448).

## Licence

Code: MIT, see [LICENSE](LICENSE). Detector weights: torchvision (BSD-3-Clause). MetaDrive: Apache-2.0. Datasets and third-party assets keep their own licences; see [data/README.md](data/README.md).
