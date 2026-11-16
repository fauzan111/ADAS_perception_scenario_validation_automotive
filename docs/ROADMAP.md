# Roadmap: ADAS Perception + Scenario Validation

## Week 1-2: perception

- [ ] nuScenes-mini loaded; visualise boxes in camera and BEV
- [ ] Pretrained 3D detector evaluated (mAP, NDS); per-class and per-distance breakdown

## Week 3-4: scenarios (started first, while the nuScenes download is pending)

- [x] MetaDrive runner for the YAML catalogue with TTC logging
- [x] Pass-rate report by kind, lighting and weather (Wilson 95% CIs)
- [ ] Failure table in SOTIF style (triggering conditions) and parameter sweeps
      (gap x speed, trigger distance x speed) to find the pass/fail boundaries
- [ ] Calibrate the perception error model from the nuScenes results (weeks 1-2)

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
