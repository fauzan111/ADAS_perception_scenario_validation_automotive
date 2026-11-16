"""Measure a real detector's recall vs range on MetaDrive camera images (runs in Docker).

    make render-calibration      # -> results/calibration/detections.jsonl + sample images

For each target (car seen from behind, pedestrian crossing) at each range and lateral offset,
render the ego camera, degrade the image for each condition, run a pretrained COCO detector
(torchvision Faster R-CNN MobileNetV3-FPN) and record whether it found the target: a
detection of the right class whose centre lies inside the projected ground-truth box (clipped
to the image, expanded by half its size). Centre matching, as nuScenes uses centre distance:
an IoU threshold rejected correct detections of distant cars because a few pixels of
projection error on a 20-pixel box drop the IoU below 0.3.
fit_perception_model.py then fits detection-probability curves on the host.

Conditions (applied to the rendered day image, so the scene is identical across them):
  night   ambient exposure x0.12 plus the ego's low beams (full strength to ~25 m, inverse-square
          beyond, road and objects only), glowing taillights on cars, gamma, sensor noise
  fog     Koschmieder: I = I*t + A*(1-t), t = exp(-beta*d), beta = 0.06/m (visibility ~50 m),
          with per-pixel depth computed from the scene geometry (flat road, object, sky)
  rain    streaks, blur, contrast loss and a light haze (beta = 0.012/m)
  night_fog / night_rain   both, to test whether condition effects multiply
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
CAM_FWD, CAM_H = 0.8, 1.4          # camera 0.8 m ahead of the ego origin, 1.4 m above the road
CLASSES = {"car": {"length": 4.515, "width": 1.852, "height": 1.4, "coco": {3, 6, 8}},
           "pedestrian": {"length": 0.5, "width": 0.6, "height": 1.75, "coco": {1}}}
CONDITIONS = ("day", "night", "fog", "rain", "night_fog", "night_rain")


def intrinsics(hfov_deg: float) -> tuple[float, float, float]:
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    return f, W / 2, H / 2


def project_box(rng: float, lat: float, cls: str, f: float, cx: float, cy: float):
    """Image box of the object's near face (range rng ahead of the camera, lat to the left)."""
    c = CLASSES[cls]
    x0, x1 = lat - c["width"] / 2, lat + c["width"] / 2
    us = [cx - f * x / rng for x in (x0, x1)]
    vs = [cy + f * (CAM_H - z) / rng for z in (0.0, c["height"])]
    return min(us), min(vs), max(us), max(vs)


def depth_map(box, rng: float, f: float, cy: float) -> np.ndarray:
    v = np.arange(H, dtype=float)[:, None].repeat(W, axis=1)
    d = np.full((H, W), 1e4)                       # sky
    below = v > cy + 0.5
    d[below] = CAM_H * f / (v[below] - cy)         # flat road
    u0, v0, u1, v1 = (int(round(t)) for t in box)
    d[max(v0, 0):min(v1, H), max(u0, 0):min(u1, W)] = rng
    return d


def taillights(box, cls: str, rng_m: float, f: float, cx: float, cy: float):
    """Image positions and radius of a car's two taillights (none for pedestrians)."""
    if cls != "car":
        return []
    u0, _, u1, v1 = box
    v = cy + f * (CAM_H - 0.8) / rng_m
    radius = max(1.5, 0.12 * f / rng_m)
    inset = 0.12 * (u1 - u0)
    return [(u0 + inset, v, radius), (u1 - inset, v, radius)]


def degrade(img: np.ndarray, cond: str, depth: np.ndarray, rng: np.random.Generator,
            lights=()):
    out = img.astype(float)
    if "fog" in cond:
        t = np.exp(-0.06 * depth)[..., None]
        out = out * t + 205.0 * (1 - t)
    if "rain" in cond:
        t = np.exp(-0.012 * depth)[..., None]
        out = out * t + 170.0 * (1 - t)
        out = 0.8 * out + 0.2 * out.mean()
        streaks = np.zeros((H, W), np.uint8)
        for _ in range(900):
            x, y = rng.integers(0, W), rng.integers(0, H)
            length = rng.integers(10, 25)
            cv2.line(streaks, (x, y), (x + 3, y + length), 255, 1)
        streaks = cv2.GaussianBlur(streaks, (3, 3), 0).astype(float)[..., None] / 255
        out = out * (1 - 0.35 * streaks) + 230 * 0.35 * streaks
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if "night" in cond:
        beam = np.clip((25.0 / np.maximum(depth, 1.0)) ** 2, 0.0, 1.0)
        beam[depth >= 1e4] = 0.0                      # headlights do not light the sky
        lit = np.clip(out, 0, 255) / 255 * (0.12 + 0.75 * beam[..., None])
        out = 255 * lit ** 1.2
        glow = np.zeros(out.shape[:2], np.float32)
        for u, v, r in lights:
            cv2.circle(glow, (int(round(u)), int(round(v))), int(round(r)), 1.0, -1)
        glow = cv2.GaussianBlur(glow, (0, 0), 1.5)[..., None]
        out = out * (1 - glow) + np.array([40.0, 40.0, 255.0]) * glow   # BGR red
        out = out + rng.normal(0, 4.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def clip_box(b):
    return (max(b[0], 0.0), max(b[1], 0.0), min(b[2], W - 1.0), min(b[3], H - 1.0))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/calibration")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--quick", action="store_true", help="few ranges, for a smoke test")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    import torch
    import torchvision
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    from metadrive.component.vehicle.vehicle_type import DefaultVehicle
    from metadrive.envs.metadrive_env import MetaDriveEnv

    torch.set_num_threads(4)
    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    det = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval()

    env = MetaDriveEnv(dict(
        use_render=False, image_observation=True, norm_pixel=False,
        sensors={"rgb_camera": (RGBCamera, W, H)}, map="S" * 10, traffic_density=0.0,
        num_scenarios=1, start_seed=0, log_level=50, show_interface=False))
    env.reset()
    ego = env.agent
    cam = env.engine.get_sensor("rgb_camera")
    hfov = float(cam.lens.getFov()[0])
    # The ego origin is ~0.6 m above the road (measured: a first run that mounted the camera
    # 1.5 m above the origin put it ~2.1 m up and every projected box sat too high).
    mount_z = CAM_H - float(ego.origin.getZ())
    f, cx, cy = intrinsics(hfov)
    ex, ey = ego.position
    ranges = {"car": list(range(5, 125, 5)), "pedestrian": list(range(5, 85, 5))}
    lats = {"car": [0.0, 3.5, -3.5], "pedestrian": [0.0, 1.5, -1.5, 3.0]}
    if args.quick:
        ranges = {"car": [10, 40, 80], "pedestrian": [10, 30]}
        lats = {"car": [0.0], "pedestrian": [1.5]}
    rng = np.random.default_rng(0)
    meta = {"hfov_deg": hfov, "width": W, "height": H, "camera_height_m": CAM_H,
            "ego_origin_z_m": float(ego.origin.getZ()),
            "detector": "fasterrcnn_mobilenet_v3_large_fpn (COCO)", "score": args.score}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    n = 0
    with open(out / "detections.jsonl", "w") as log:
        for cls, rs in ranges.items():
            for variant in range(2):                      # two colours / poses per position
                for lat in lats[cls]:
                    for r in rs:
                        if cls == "car":
                            centre = (ex + CAM_FWD + r + CLASSES[cls]["length"] / 2, ey + lat)
                            obj = env.engine.spawn_object(
                                DefaultVehicle, vehicle_config={}, position=centre,
                                heading=0.0, random_seed=variant * 97 + r)
                        else:
                            centre = (ex + CAM_FWD + r, ey + lat)
                            obj = env.engine.spawn_object(
                                Pedestrian, position=centre,
                                heading_theta=math.pi / 2 if variant == 0 else -math.pi / 2)
                        env.step([0.0, 0.0])
                        img = cam.perceive(to_float=False, new_parent_node=ego.origin,
                                           position=[0.0, CAM_FWD, mount_z], hpr=[0.0, 0.0, 0.0])
                        img = np.ascontiguousarray(img[..., :3])
                        env.engine.clear_objects([obj.id])
                        box = project_box(r, lat, cls, f, cx, cy)
                        depth = depth_map(box, r, f, cy)
                        lights = taillights(box, cls, r, f, cx, cy)
                        gt = clip_box(box)
                        bw, bh = gt[2] - gt[0], gt[3] - gt[1]
                        mx, my = max(0.5 * bw, 4.0), max(0.5 * bh, 4.0)
                        for cond in CONDITIONS:
                            im = degrade(img, cond, depth, rng, lights)
                            x = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1).float()
                            with torch.no_grad():
                                pred = det([x / 255.0])[0]
                            best, hit = 0.0, False
                            for b, lab, s in zip(pred["boxes"].tolist(),
                                                 pred["labels"].tolist(),
                                                 pred["scores"].tolist(), strict=True):
                                if s >= args.score and lab in CLASSES[cls]["coco"]:
                                    ux, uy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                                    if (gt[0] - mx <= ux <= gt[2] + mx
                                            and gt[1] - my <= uy <= gt[3] + my):
                                        hit = True
                                        best = max(best, iou(b, gt))
                            log.write(json.dumps({"cls": cls, "range_m": r, "lat_m": lat,
                                                  "variant": variant, "cond": cond,
                                                  "hit": hit, "iou": round(best, 3)}) + "\n")
                            if variant == 0 and lat in (0.0, 1.5) and r in (10, 30, 60):
                                vis = im.copy()
                                cv2.rectangle(vis, (int(box[0]), int(box[1])),
                                              (int(box[2]), int(box[3])),
                                              (0, 255, 0) if hit else (0, 0, 255), 2)
                                cv2.imwrite(str(out / "samples" / f"{cls}_{r}m_{cond}.jpg"), vis)
                        n += 1
                        if n % 20 == 0:
                            print(f"{n} renders", flush=True)
    env.close()
    print(f"done: {n} renders x {len(CONDITIONS)} conditions -> {out / 'detections.jsonl'}")


if __name__ == "__main__":
    main()
