# Roadmap: ADAS Perception + Scenario Validation

## Week 1-2: perception (re-planned: no dataset download)

- [x] Detector measured on simulator images: torchvision Faster R-CNN on MetaDrive renders,
      recall vs range for car / pedestrian under day, night, fog, rain and combinations
- [x] Published nuScenes results as the real-world cross-check (DETR3D AP by distance,
      night/rain; MonoDIS class AP) and a nuScenes-anchored perception model
- [ ] Optional: 3D detector on nuScenes-mini (needs registration + 4 GB download)

### Perception notes (2026-11-16)

- macOS can't render MetaDrive offscreen from a non-GUI process (simplepbr gets no buffer,
  even outside the sandbox), so rendering runs in Docker: Xvfb + Mesa llvmpipe (GL 3.3).
  `xvfb-run` hangs as PID 1 (it waits for SIGUSR1); a small entrypoint starts Xvfb itself.
  MetaDrive's depth camera needs compute shaders (not available there), so per-pixel depth
  for fog/rain comes from the scene geometry (flat road, object box, sky).
- First calibration run had three measurement bugs, found from the curves and images:
  - The ego origin is 0.595 m above the road, so the "1.5 m" camera was ~2.1 m up and all
    projected boxes sat too high (confirmed by fitting box bottoms vs 1/range: 1.98 m).
  - IoU >= 0.3 rejected correct detections of distant cars (a few pixels of projection error
    on a 20-pixel box); both classes showed the same fake cliff at 64 m. Switched to centre
    matching (nuScenes also matches on centre distance). The real limit: MetaDrive draws the
    car to 100 m, and this detector loses it at ~65-70 m (~19 px wide).
  - Night without any lights made cars undetectable at every range. Added low beams
    (inverse-square, road and objects) and taillights. Night range factor: car 0.95
    (nuScenes DETR3D night/day car AP 0.95), pedestrian 0.58.
- Positions outside the 60 deg field of view are excluded from the fit (geometry, not recall).
- Results with three perception models: see README. Robust across all three: occluded
  pedestrian in rain 0/60 (occlusion + wet-road braking). Strongly model-dependent: lead-car
  braking at 90 km/h with a 30 m gap (74% / 35% / 9%).

## Week 3-4: scenarios (started first, while the nuScenes download is pending)

- [x] MetaDrive runner for the YAML catalogue with TTC logging
- [x] Pass-rate report by kind, lighting and weather (Wilson 95% CIs)
- [ ] Failure table in SOTIF style (triggering conditions) and parameter sweeps
      (gap x speed, trigger distance x speed) to find the pass/fail boundaries
- [x] Calibrate the perception error model (sim-measured + nuScenes-anchored, see above)

### Scenario notes (2026-11-16)

- Loop: MetaDrive ego physics + scripted actors; ground truth -> perception error model
  (range/condition-dependent detection, noise, 0.1 s latency, ghosts, occlusion by line of
  sight) -> tracker -> ACC/AEB -> throttle/brake. Verdict from true geometry: pass = no
  collision; min true TTC = margin. 540 runs (18 scenarios x 30 seeds) take ~20 s.
- Bugs found by the first results, all fixed before any number was kept:
  - MetaDrive applies spawn lane and spawn velocity only on the first reset: every run
    started at 72 km/h in an outer lane, and lane keeping dragged the ego across the road.
  - Tracker: confirmed tracks dropped after two missed frames, and new tracks started at
    zero relative speed, so a braking lead looked like it moved with the ego. At night in
    rain (p_detect ~0.7 at 30 m) the ADAS kept losing the lead and re-accelerated: 0/30.
    Fix: confirmed tracks stay confirmed until they coast 0.6 s; velocity from the first
    two detections. Night-rain lead-brake impact speed went from 53 to 22 km/h.
  - In-path test used the object centre against a fixed +-1.4 m corridor, so a car
    overlapping the ego lane by ~0.9 m was ignored during a cut-in. Now ego half width +
    object half width + 0.3 m.
  - MetaDrive physics contact can shove the ego aside before the box check fires; its
    crash flags now also count as collisions.
- First catalogue (assumed perception parameters): overall 368/540 = 68%. Cut-in (20 m gap)
  100%; lead brake 74% (night + fog 13%, night + rain 37%); occluded pedestrian 31% (rain 0%
  by day and night: visible only at ~10 m, and at 6.5 m/s^2 the car needs ~9.5 m plus
  reaction time from 40 km/h; day clear 77%). Occlusion and road friction, not lighting,
  dominate the pedestrian case.

## Week 5: CARLA

- [ ] Short cloud session: ScenarioRunner batch for the same catalogue
- [ ] Compare MetaDrive vs CARLA results

## Week 6: polish

- [ ] README results, video, LinkedIn post (ADAS, validazione)

## Definition of done

- [ ] Results table filled in the README
- [ ] 2-minute demo video linked
- [ ] Tests and lint green
- [ ] LinkedIn post (Italian and English)
