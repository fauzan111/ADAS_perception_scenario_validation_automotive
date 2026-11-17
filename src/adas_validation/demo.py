"""Demo video: closed-loop runs seen through the ego camera, plus the headline results.

    make demo-record   # Docker (MetaDrive renders offscreen there): dashcam frames + traces
    make demo-video    # host: results/demo/adas_demo.mp4 and docs/media/demo_lead_brake.gif

record  runs each clip once in MetaDrive with an RGB camera on the ego and keeps the trace of
        the same run (ground truth, raw detections, confirmed tracks, ADAS mode), so the
        overlays and verdicts in the video are that run's, not a re-simulation.
stats   (host, no camera) the context numbers quoted in the video: failures over 30 seeds
        per clip and the counterfactual reruns of the pedestrian clip.
compose draws every frame (dashcam with projected detections and tracks, bird's-eye view,
        speed and ADAS-mode timeline, captions generated from the trace's events) and the
        result cards from results/sotif_* and results/sweeps, then encodes with ffmpeg
        (imageio-ffmpeg: `uv sync --extra video`).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from adas_validation.perception import PerceptionModel
from adas_validation.sweeps import make_scenario

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/demo"
CAM_W, CAM_H_PX = 960, 540
CAM_FWD, CAM_Z = 0.8, 1.4            # camera mount: ahead of the ego origin, above the road
SIZE = {"car": (1.852, 1.4), "pedestrian": (0.6, 1.75)}      # width, height (m)
MEASURED = "configs/perception_model_measured.yaml"
PED = {"kind": "occluded_pedestrian", "ego_speed_kph": 30,
       "params": {"pedestrian_speed_mps": 1.4, "van_distance_m": 60, "trigger_time_s": 2.5}}
LEAD = {"kind": "lead_brake", "ego_speed_kph": 130,
        "params": {"lead_decel_mps2": 6.0, "initial_gap_m": 30, "brake_at_s": 2.0}}
CLIPS = [
    dict(name="ped_pass", base=PED, assign={}, seed=0, system={},
         title="Pedestrian steps out from behind a parked vehicle",
         subtitle="30 km/h; the pedestrian starts walking when the car is 2.5 s away"),
    dict(name="ped_fail", base=PED, assign={"params.trigger_time_s": 1.75}, seed=0, system={},
         title="Same pedestrian, stepping out 1.75 s ahead",
         subtitle="30 km/h; starts walking when the car is 1.75 s away"),
    dict(name="lead130", base=LEAD, assign={}, seed=0, system={},
         title="Lead car brakes hard at 130 km/h",
         subtitle="30 m gap, lead brakes at 6 m/s²"),
    dict(name="lead130_known", base=LEAD, assign={}, seed=0,
         system={"known_lead_accel": True},
         title="Same run, one change: the AEB knows the lead is braking",
         subtitle="same perception; the lead's deceleration replaces the constant-speed "
                  "assumption"),
]


def clip_scenario(clip: dict):
    return make_scenario(clip["base"], clip["assign"], "day", "clear")


# ---------------------------------------------------------------- record (Docker)

def record(out: Path, names: list[str] | None) -> None:
    import cv2

    from adas_validation.sim import make_env, run_scenario

    model = PerceptionModel.from_yaml(ROOT / MEASURED)
    env = make_env(camera=(CAM_W, CAM_H_PX))
    env.reset()
    cam = env.engine.get_sensor("rgb_camera")
    hfov = float(cam.lens.getFov()[0])
    try:
        for clip in CLIPS:
            if names and clip["name"] not in names:
                continue
            d = out / clip["name"]
            shutil.rmtree(d / "frames", ignore_errors=True)
            (d / "frames").mkdir(parents=True)
            sc = clip_scenario(clip)

            def grab(env, bodies, k, d=d, sc=sc):
                if k == 0 and sc.kind == "occluded_pedestrian":
                    bodies["hazard"].set_heading_theta(-math.pi / 2)   # faces across the road
                ego = env.agent
                img = env.engine.get_sensor("rgb_camera").perceive(
                    to_float=False, new_parent_node=ego.origin,
                    position=[0.0, CAM_FWD, CAM_Z - float(ego.origin.getZ())],
                    hpr=[0.0, 0.0, 0.0])
                cv2.imwrite(str(d / "frames" / f"{k:04d}.jpg"),
                            np.ascontiguousarray(img[..., :3]), [cv2.IMWRITE_JPEG_QUALITY, 92])

            r = run_scenario(sc, clip["seed"], model, env, keep_trace=True, on_frame=grab,
                             **clip["system"])
            meta = {k: v for k, v in clip.items() if k not in ("base", "assign")}
            meta.update(kind=sc.kind, params=sc.params, ego_speed_kph=sc.ego_speed_kph,
                        hfov_deg=hfov, cam_w=CAM_W, cam_h=CAM_H_PX,
                        latency_s=model.latency_s,
                        min_visible_fraction=model.min_visible_fraction,
                        result={k: v for k, v in vars(r).items() if k != "trace"})
            (d / "run.json").write_text(json.dumps({"meta": meta, "trace": r.trace}))
            print(f"{clip['name']}: {len(r.trace)} frames, {r.end_reason}, "
                  f"impact {r.impact_speed_kph:.1f} km/h", flush=True)
    finally:
        env.close()


# ---------------------------------------------------------------- stats (host)

def stats(out: Path, seeds: int = 30) -> None:
    from adas_validation.sim import make_env, run_scenario

    model = PerceptionModel.from_yaml(ROOT / MEASURED)
    env = make_env()
    res: dict = {}
    try:
        for clip in CLIPS:
            sc = clip_scenario(clip)
            runs = [run_scenario(sc, s, model, env, **clip["system"]) for s in range(seeds)]
            res[clip["name"]] = {"seeds": seeds, "collisions": sum(r.collided for r in runs)}
        sc = clip_scenario(CLIPS[1])
        cf = {"oracle": run_scenario(sc, 0, model.oracle(), env),
              "instant": run_scenario(sc, 0, model.oracle(latency_s=0.0), env,
                                      confirm_hits=1)}
        res["ped_fail"]["counterfactual"] = {k: {"collided": r.collided,
                                                 "impact_kph": round(r.impact_speed_kph, 1)}
                                             for k, r in cf.items()}
    finally:
        env.close()
    (out / "stats.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------- compose (host)

W, H, FPS = 1920, 1080, 30
BG, PANEL, INK, MUTED, FAINT = "#121212", "#1a1a19", "#ececea", "#a3a29d", "#3a3a37"
BLUE, ORANGE, RED, GREEN, AMBER = "#3987e5", "#d95926", "#e66767", "#0ca30c", "#c98500"
MODE_COLOR = {"cruise": "#4a4a46", "follow": BLUE, "fcw": AMBER, "aeb": RED}
CAUSE_COLOR = {"detection": "#3987e5", "noise": "#d95926", "friction": "#199e70",
               "reaction": "#c98500", "decision": "#d55181", "design_limit": "#0ca30c"}
EGO_LEN, EGO_W, CAR_LEN, VAN_LEN, VAN_W, LANE = 4.515, 1.852, 4.515, 6.0, 2.2, 3.5


def gap_m(row: dict, kind: str) -> float:
    half = 0.35 if kind == "occluded_pedestrian" else CAR_LEN / 2
    return row["hz_x"] - half - (row["ego_x"] + EGO_LEN / 2)


def events(run: dict) -> list[tuple[float, str, str]]:
    """(time, key, caption) in time order, from the trace of one run."""
    m, tr = run["meta"], run["trace"]
    kind, p, res = m["kind"], m["params"], m["result"]
    ev = []
    if kind == "occluded_pedestrian":
        ev.append((0.0, "start", "A pedestrian waits behind a parked vehicle, "
                                 "out of the camera's sight."))
        walk = next((r for r in tr if abs(r["hz_y"] - tr[0]["hz_y"]) > 1e-6), None)
        if walk:
            ev.append((walk["t"], "walk", "They step into the road, still hidden."))
        vis = next((r for r in tr if r["hz_vis"] >= m["min_visible_fraction"]), None)
        if vis:
            ev.append((vis["t"], "visible",
                       f"Visible to the camera {gap_m(vis, kind):.0f} m ahead."))
        conf = next((r for r in tr if any(t[0] == "hazard" for t in r["tracks"])), None)
        if conf:
            ev.append((conf["t"], "confirmed",
                       f"Track confirmed {gap_m(conf, kind):.0f} m ahead "
                       f"(3 detections in 5 frames, {m['latency_s']:g} s latency)."))
    else:
        ev.append((0.0, "start", f"Following a car at {m['ego_speed_kph']:g} km/h, "
                                 f"{p['initial_gap_m']:g} m gap. It is tracked all along."))
        ev.append((p["brake_at_s"], "brake",
                   f"The lead car brakes at {p['lead_decel_mps2']:g} m/s²."))
        acc = next((r for r in tr if r["t"] > p["brake_at_s"] and r["mode"] != "aeb"
                    and r["accel_cmd"] <= -2.9), None)
        first_aeb = next((r["t"] for r in tr if r["mode"] == "aeb"), math.inf)
        if acc and acc["t"] < first_aeb:
            ev.append((acc["t"], "acc", "ACC brakes at its 3 m/s² comfort limit; "
                                        "the AEB sees no emergency yet."))
    aeb = next((r for r in tr if r["mode"] == "aeb"), None)
    if aeb:
        why = ""
        if kind == "lead_brake":
            why = (": it knows the lead is braking" if m["system"].get("known_lead_accel")
                   else ": it assumed the lead keeps its speed")
        ev.append((aeb["t"], "aeb", f"AEB: full braking, {gap_m(aeb, kind):.0f} m to go{why}."))
    t_end = tr[-1]["t"] + 0.1
    if res["collided"]:
        ev.append((t_end, "end", f"Collision at {res['impact_speed_kph']:.0f} km/h."))
    elif res["end_reason"] == "stopped":
        ev.append((t_end, "end", f"Stopped {gap_m(tr[-1], kind):.0f} m short."))
    else:
        ev.append((t_end, "end", "No collision: the pedestrian clears the lane."))
    return sorted(ev, key=lambda e: e[0])


def trim(run: dict) -> dict:
    """End a 'passed' run once the pedestrian has left the ego's path (plus 1 s), instead of
    showing the car accelerating back to its set speed."""
    tr = run["trace"]
    if run["meta"]["result"]["end_reason"] == "passed":
        k_aeb = next((k for k, r in enumerate(tr) if r["mode"] == "aeb"), 0)
        k_out = next((k for k in range(k_aeb, len(tr))
                      if abs(tr[k]["hz_y"] - tr[k]["ego_y"]) > 0.93 + 0.35 + 0.5), None)
        if k_out is not None:
            run["trace"] = tr[:k_out + 10]
    return run


def verdict(run: dict) -> tuple[str, str]:
    res, kind = run["meta"]["result"], run["meta"]["kind"]
    if res["collided"]:
        return f"COLLISION  {res['impact_speed_kph']:.0f} km/h", RED
    if res["end_reason"] == "stopped":
        return f"STOPPED  {gap_m(run['trace'][-1], kind):.0f} m short", GREEN
    return "NO COLLISION", GREEN


class Painter:
    """Draws 1920x1080 frames with matplotlib (one reusable figure)."""

    def __init__(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                             "axes.edgecolor": FAINT, "axes.labelcolor": MUTED,
                             "xtick.color": MUTED, "ytick.color": MUTED})
        self.plt = plt
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)

    def new(self):
        self.fig.clf()
        self.fig.set_facecolor(BG)
        return self.fig

    def axes(self, x: float, y: float, w: float, h: float):
        """Axes in pixel coordinates (origin bottom left)."""
        return self.fig.add_axes((x / W, y / H, w / W, h / H))

    def array(self) -> np.ndarray:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def text(self, x, y, s, size=20, color=INK, weight="normal", ha="left", va="baseline",
             **kw):
        self.fig.text(x / W, y / H, s, fontsize=size, color=color, weight=weight, ha=ha,
                      va=va, **kw)


def project(x_s: float, y_s: float, cls: str, hfov: float):
    """Image box (u0, v0, u1, v1) of an object at sensor-frame (x_s, y_s), or None."""
    rng = x_s + EGO_LEN / 2 - CAM_FWD
    if rng < 1.5:
        return None
    f = (CAM_W / 2) / math.tan(math.radians(hfov) / 2)
    w, h = SIZE.get(cls, SIZE["car"])
    us = [CAM_W / 2 - f * (y_s + s * w / 2) / rng for s in (-1, 1)]
    vs = [CAM_H_PX / 2 + f * (CAM_Z - z) / rng for z in (0.0, h)]
    return min(us), min(vs), max(us), max(vs)


def draw_clip_frame(pt: Painter, run: dict, k: int, frame_path: Path, evs: list,
                    index: int, n_clips: int, final: bool) -> np.ndarray:
    from matplotlib.patches import FancyBboxPatch, Rectangle

    m, tr = run["meta"], run["trace"]
    row, kind = tr[k], m["kind"]
    t = row["t"]
    fig = pt.new()
    # header
    pt.text(48, 1020, f"{index}/{n_clips}   {m['title']}", 28, weight="bold")
    pt.text(48, 982, f"{m['subtitle']}  ·  sim-measured perception  ·  by day, "
            f"clear  ·  seed {m['seed']}", 17, MUTED)
    # dashcam
    cam = pt.axes(48, 262, 1248, 702)
    import matplotlib.image as mpimg
    cam.imshow(mpimg.imread(frame_path), extent=(0, CAM_W, CAM_H_PX, 0))
    cam.set_xlim(0, CAM_W)
    cam.set_ylim(CAM_H_PX, 0)
    cam.axis("off")
    hz_cls = "pedestrian" if kind == "occluded_pedestrian" else "car"
    for obj, x, y in row["dets"]:
        b = project(x, y, hz_cls if obj == "hazard" else "car", m["hfov_deg"])
        if b:
            cam.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                    ec=ORANGE, lw=1.8, ls=(0, (3, 2))))
    for obj, x, y, _vx, _vy in row["tracks"]:
        b = project(x, y, hz_cls if obj == "hazard" else "car", m["hfov_deg"])
        if b:
            col = RED if row["mode"] == "aeb" else BLUE
            cam.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                    ec=col, lw=2.4))
            cam.text(b[0], b[1] - 6, f"track  {math.hypot(x, y):.0f} m", color="white",
                     fontsize=11, weight="bold", va="bottom",
                     bbox=dict(fc=col, ec="none", pad=2))
    caption = [c for (te, _, c) in evs if te <= t + 1e-9][-1]
    if final:
        caption = evs[-1][2]
    cam.text(CAM_W / 2, CAM_H_PX - 22, caption, ha="center", va="bottom", fontsize=17,
             color="white", bbox=dict(fc="black", ec="none", alpha=0.72, pad=8))
    if final:
        txt, col = verdict(run)
        cam.text(CAM_W / 2, CAM_H_PX * 0.42, txt, ha="center", va="center", fontsize=40,
                 weight="bold", color="white", bbox=dict(fc=col, ec="none", pad=14))
    cam.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, ec=ORANGE, ls=(0, (3, 2)),
                                  label="raw detection (noisy, 0.1 s old)"),
                        Rectangle((0, 0), 1, 1, fill=False, ec=BLUE, lw=2.4,
                                  label="confirmed track")],
               loc="upper left", fontsize=11, framealpha=0.6, facecolor="black",
               edgecolor="none", labelcolor="white")
    # bird's-eye view, true scale: lateral (world y, +y on the left) x metres ahead
    ahead = 66 if kind == "occluded_pedestrian" else 52
    road_l, road_r = LANE * 2.5, -LANE / 2
    x_left, x_right = road_l + 12.0, road_r - 5.5
    px_per_m = 702 / (ahead + 8)
    bev_w = (x_left - x_right) * px_per_m
    bev = pt.axes(1340, 262, bev_w, 702)
    bev.set_facecolor(BG)
    bev.set_xlim(x_left, x_right)
    bev.set_ylim(-8, ahead)
    for sp in bev.spines.values():
        sp.set_visible(False)
    bev.set_xticks([])
    bev.set_yticks([])
    bev.add_patch(Rectangle((road_r, -8), 3 * LANE, ahead + 8, fc="#2b2b29", ec="none",
                            zorder=0))
    for i, yl in enumerate((road_r, road_r + LANE, road_r + 2 * LANE, road_l)):
        bev.plot([yl, yl], [-8, ahead], color="#8a8882", lw=1.2,
                 ls="-" if i in (0, 3) else (0, (5, 5)), zorder=1)
    for d in range(10, ahead, 10):
        bev.plot([road_r - 0.4, road_r - 1.0], [d, d], color=FAINT, lw=1)
        bev.text(road_r - 1.2, d, f"{d} m", fontsize=9, color=MUTED, va="center", ha="left")
    ego_y = row["ego_y"]
    corridor = 0.93 + 0.3 + (0.35 if kind == "occluded_pedestrian" else 0.93)
    aeb_on = row["mode"] == "aeb"
    bev.add_patch(Rectangle((ego_y - corridor, EGO_LEN / 2), 2 * corridor, ahead,
                            fc=RED if aeb_on else "#ffffff", alpha=0.22 if aeb_on else 0.05,
                            ec="none", zorder=1))
    bev.add_patch(Rectangle((ego_y - EGO_W / 2, -EGO_LEN / 2), EGO_W, EGO_LEN, fc=INK,
                            ec="none", zorder=4))
    rel = row["ego_x"]
    lab_x = road_l + 0.6
    if kind == "occluded_pedestrian":
        vx = m["params"]["van_distance_m"]
        bev.add_patch(Rectangle((2 * LANE - VAN_W / 2, vx - rel - VAN_LEN / 2), VAN_W,
                                VAN_LEN, fc="#6b6a65", ec="none", zorder=3))
        hidden = row["hz_vis"] < m["min_visible_fraction"]
        hy = row["hz_x"] - rel
        bev.plot([row["hz_y"]], [hy], marker="o", ms=9, mfc=BG if hidden else INK, mec=INK,
                 mew=1.6, zorder=5)
        bev.text(lab_x, hy + 2.5, "pedestrian\n(hidden)" if hidden else "pedestrian",
                 fontsize=10, color=MUTED if hidden else INK, va="center", ha="right",
                 zorder=5)
        if vx - rel > -2:
            bev.text(lab_x, vx - rel - 2.5, "parked", fontsize=10, color=MUTED, va="center",
                     ha="right")
    else:
        hy = row["hz_x"] - rel
        bev.add_patch(Rectangle((row["hz_y"] - EGO_W / 2, hy - CAR_LEN / 2), EGO_W,
                                CAR_LEN, fc="none", ec=INK, lw=1.6, zorder=4))
        bev.text(lab_x, hy, f"lead\n{row['hz_vx'] * 3.6:.0f} km/h", fontsize=10, color=INK,
                 va="center", ha="right", zorder=5)
    bev.text(lab_x, 0, f"ego\n{row['ego_v'] * 3.6:.0f} km/h", fontsize=10, color=INK,
             va="center", ha="right")
    for _obj, x, y in row["dets"]:
        bev.plot([ego_y + y], [x + EGO_LEN / 2], marker="+", ms=11, mew=2, color=ORANGE,
                 zorder=6)
    for _obj, x, y, _vx, _vy in row["tracks"]:
        bev.plot([ego_y + y], [x + EGO_LEN / 2], marker="s", ms=8,
                 color=RED if aeb_on else BLUE, mec="white", mew=1, zorder=7)
    from matplotlib.lines import Line2D
    lg = pt.axes(1340 + bev_w + 24, 262, 1872 - 1340 - bev_w - 24, 702)
    lg.axis("off")
    lg.text(0, 1.0, "BIRD'S-EYE VIEW", fontsize=12, color=MUTED, weight="bold", va="top",
            transform=lg.transAxes)
    lg.text(0, 0.965, "true scale", fontsize=11, color=MUTED, va="top",
            transform=lg.transAxes)
    lg.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=9, label="ground truth"),
        Line2D([], [], marker="+", ls="", color=ORANGE, mew=2, ms=11, label="raw detection"),
        Line2D([], [], marker="s", ls="", color=BLUE, mec="white", ms=8,
               label="confirmed track"),
        Rectangle((0, 0), 1, 1, fc=RED, alpha=0.5, ec="none", label="AEB braking")],
        loc="upper left", bbox_to_anchor=(-0.04, 0.93), fontsize=11, frameon=False,
        labelcolor=INK, handletextpad=0.6, labelspacing=0.9)
    # timeline: speed with the ADAS mode underneath
    T = tr[-1]["t"] + 0.1
    tl = pt.axes(112, 70, 1184, 150)
    tl.set_facecolor(BG)
    ts = [r["t"] for r in tr[:k + 1]]
    vs = [r["ego_v"] * 3.6 for r in tr[:k + 1]]
    vmax = m["ego_speed_kph"] * 1.15
    tl.plot(ts, vs, color=INK, lw=2)
    start = 0
    for e in range(1, k + 2):           # one span per run of the same mode
        if e == k + 1 or tr[e]["mode"] != tr[start]["mode"]:
            tl.add_patch(Rectangle((tr[start]["t"], -0.16 * vmax),
                                   tr[e - 1]["t"] + 0.1 - tr[start]["t"], 0.1 * vmax,
                                   fc=MODE_COLOR[tr[start]["mode"]], ec="none"))
            start = e
    tl.set_xlim(0, T)
    tl.set_ylim(-0.18 * vmax, vmax)
    tl.set_yticks([0, round(m["ego_speed_kph"] / 2), m["ego_speed_kph"]])
    tl.tick_params(labelsize=10)
    tl.set_ylabel("speed (km/h)", fontsize=11)
    tl.set_xlabel("time (s)", fontsize=11, labelpad=2)
    for s in ("top", "right"):
        tl.spines[s].set_visible(False)
    for te, key, _ in evs:
        if key in ("walk", "brake", "aeb") and te <= t + 1e-9:
            tl.axvline(te, color=FAINT, lw=1, ls=":")
    # readouts
    mode = row["mode"]
    pt.text(1340, 186, "SPEED", 12, MUTED)
    pt.text(1340, 128, f"{row['ego_v'] * 3.6:5.1f}", 40, weight="bold", family="DejaVu Sans Mono")
    pt.text(1530, 128, "km/h", 14, MUTED)
    pt.text(1640, 186, "ADAS MODE", 12, MUTED)
    fig.patches.append(FancyBboxPatch((1640 / W, 126 / H), 210 / W, 46 / H,
                                      boxstyle="round,pad=0.002,rounding_size=0.006",
                                      transform=fig.transFigure, fc=MODE_COLOR[mode],
                                      ec="none"))
    pt.text(1745, 149, {"fcw": "FCW warning"}.get(mode, mode.upper()), 20, "white",
            weight="bold", ha="center", va="center")
    g = gap_m(row, kind)
    gap_txt = f"{g:5.1f} m" if g >= 0 else "   past"
    pt.text(1340, 82, f"true gap {gap_txt}      t = {t:4.1f} s", 15, MUTED,
            family="DejaVu Sans Mono")
    for i, (name, col) in enumerate(MODE_COLOR.items()):
        x0 = 1340 + i * 132
        fig.patches.append(Rectangle((x0 / W, 40 / H), 16 / W, 12 / H,
                                     transform=fig.transFigure, fc=col, ec="none"))
        pt.text(x0 + 22, 40, {"fcw": "FCW"}.get(name, name if name != "aeb" else "AEB"), 12,
                MUTED)
    return pt.array()


def card(pt: Painter, title: str, lines: list[str], kicker: str = "",
         title_size: int = 46) -> np.ndarray:
    """Text card; title may contain newlines."""
    import textwrap
    pt.new()
    top = 760
    title = "\n".join(textwrap.fill(part, int(1700 / (title_size * 0.85)))
                      for part in title.split("\n"))
    if kicker:
        pt.text(160, top, kicker, 20, BLUE, weight="bold")
    n_title = title.count("\n") + 1
    pt.text(160, top - 40, title, title_size, weight="bold", va="top", linespacing=1.15)
    y = top - 40 - n_title * title_size * 1.65 - 70
    for i, ln in enumerate(lines):
        pt.text(160, y - i * 52, ln, 24, MUTED)
    return pt.array()


def title_card(pt: Painter) -> np.ndarray:
    pt.new()
    pt.text(160, 700, "ADAS perception + scenario validation", 22, BLUE, weight="bold")
    pt.text(160, 610, "Does the AEB still brake in time", 58, weight="bold")
    pt.text(160, 530, "when perception is imperfect?", 58, weight="bold")
    for i, ln in enumerate(["Closed-loop ACC / AEB in the MetaDrive simulator",
                            "Perception error model measured on simulator renders, "
                            "anchored to nuScenes",
                            "SOTIF-style failure attribution  ·  parameter sweeps"]):
        pt.text(160, 410 - i * 48, ln, 24, MUTED)
    return pt.array()


def pipeline_card(pt: Painter) -> np.ndarray:
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = pt.new()
    pt.text(160, 900, "How a run works", 44, weight="bold")
    boxes = [("MetaDrive world", "ego physics, scripted\nactors, occlusion"),
             ("Perception model", "detection vs range,\nnoise, 0.1 s latency, ghosts"),
             ("Tracker", "confirm on 3 of 5\nframes, alpha-beta filter"),
             ("ACC + AEB", "path prediction,\nTTC, required decel."),
             ("Brake / throttle", "back into the\nsimulator")]
    bw, gap, y0 = 312, 37, 520
    x = 160
    for i, (head, sub) in enumerate(boxes):
        fig.patches.append(FancyBboxPatch((x / W, y0 / H), bw / W, 170 / H,
                                          boxstyle="round,pad=0,rounding_size=0.01",
                                          transform=fig.transFigure, fc=PANEL,
                                          ec=BLUE if i == 1 else FAINT, lw=2))
        pt.text(x + 20, y0 + 120, head, 20, weight="bold")
        pt.text(x + 20, y0 + 30, sub, 15, MUTED, linespacing=1.5)
        if i < len(boxes) - 1:
            fig.patches.append(FancyArrowPatch(((x + bw + 6) / W, (y0 + 85) / H),
                                               ((x + bw + gap - 6) / W, (y0 + 85) / H),
                                               transform=fig.transFigure, color=MUTED,
                                               arrowstyle="-|>", mutation_scale=18, lw=2))
        x += bw + gap
    x_last, x_first, y_low = (x - gap - bw / 2) / W, (160 + bw / 2) / W, (y0 - 50) / H
    fig.lines.append(Line2D([x_last, x_last, x_first], [(y0 - 4) / H, y_low, y_low],
                            transform=fig.transFigure, color=MUTED, lw=2))
    fig.patches.append(FancyArrowPatch((x_first, y_low), (x_first, (y0 - 4) / H),
                                       transform=fig.transFigure, color=MUTED,
                                       arrowstyle="-|>", mutation_scale=18, lw=2))
    pt.text((x_last + x_first) / 2 * W, y0 - 82, "closed loop, 10 Hz", 16, MUTED,
            ha="center")
    lines = ["Ground truth decides pass or fail. Perception only decides what the controller "
             "believes.",
             "Detection curves: a Faster R-CNN measured on 272 MetaDrive renders x 6 "
             "conditions,",
             "and a second model anchored to camera-only nuScenes results (DETR3D AP by "
             "distance)."]
    for i, ln in enumerate(lines):
        pt.text(160, 300 - i * 50, ln, 22, INK if i == 0 else MUTED)
    return pt.array()


def counterfactual_card(pt: Painter, st: dict) -> np.ndarray:
    s = st["ped_fail"]
    cf = s["counterfactual"]

    def res(r):
        return (f"collision at {r['impact_kph']:.0f} km/h" if r["collided"]
                else "no collision")
    return card(pt, "Why it fails: rerun with one\nlimitation removed at a time", [
        f"As tested:  collides in {s['collisions']} of {s['seeds']} seeds",
        f"Perfect detection, no position noise:  {res(cf['oracle'])}",
        f"+ no sensor latency, track confirmed on the first detection:  "
        f"{res(cf['instant'])}",
        "",
        "Cause: reaction latency. A performance insufficiency in SOTIF terms,",
        "not a detection problem: a better detector alone would not avoid it."],
        kicker="COUNTERFACTUAL ATTRIBUTION", title_size=40)


def sotif_card(pt: Painter) -> np.ndarray:
    from matplotlib.patches import Rectangle
    fig = pt.new()
    pt.text(160, 950, "Why each collision happens", 44, weight="bold")
    pt.text(160, 900, "Each collision is rerun with one limitation removed at a time; the "
            "cause is the first change that removes it:", 18, MUTED)
    pt.text(160, 868, "perfect detection → no position noise → dry-road braking → no latency "
            "→ known lead deceleration. None: design limit.", 18, MUTED)
    groups = [("assumed", "sotif_assumed", "catalogue"),
              ("sim-measured", "sotif_measured", "catalogue"),
              ("nuScenes-anchored", "sotif_nuscenes", "catalogue"),
              ("edge cases", "sotif_edge", "sim-measured, sweep-found")]
    ax = pt.axes(560, 250, 1200, 520)
    ax.set_facecolor(BG)
    for i, (label, d, note) in enumerate(groups):
        data = json.loads((ROOT / "results" / d / "sotif.json").read_text())
        by = data["collisions_by_cause"]
        total = sum(by.values())
        y = len(groups) - 1 - i - (0.35 if i == 3 else 0)
        left = 0.0
        for cause in CAUSE_COLOR:
            share = by.get(cause, 0) / total
            if share <= 0:
                continue
            ax.add_patch(Rectangle((left + 0.002, y - 0.3), share - 0.004, 0.6,
                                   fc=CAUSE_COLOR[cause], ec="none"))
            if share >= 0.08:
                ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center",
                        fontsize=15, color="white", weight="bold")
            left += share
        fy = 250 + 520 * (y + 0.85) / 4.2
        pt.text(530, fy + 8, label, 22, ha="right", weight="bold")
        pt.text(530, fy - 24, f"{note}, {total} collisions", 14, MUTED, ha="right")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.85, 3.35)
    ax.axis("off")
    for j, cause in enumerate(CAUSE_COLOR):
        x0 = 560 + j * 200
        fig.patches.append(Rectangle((x0 / W, 190 / H), 22 / W, 22 / H,
                                     transform=fig.transFigure, fc=CAUSE_COLOR[cause]))
        pt.text(x0 + 32, 193, cause.replace("_", " "), 17, INK)
    pt.text(160, 110, "Catalogue: 18 scenarios x 30 seeds per perception model.", 17, MUTED)
    pt.text(160, 80, "Pedestrian in rain: 0 of 60 pass under all three models; with perfect "
            "perception the car still hits at 7 km/h.", 17, MUTED)
    return pt.array()


def sweep_card(pt: Painter) -> np.ndarray:
    import matplotlib.image as mpimg
    pt.new()
    b = json.loads((ROOT / "results/sweeps/boundaries.json").read_text())
    wc = b["worst_case_max_speed_kph"]
    pt.text(160, 990, "Where it stops being safe: parameter sweeps", 40, weight="bold")
    n = sum(1 for _ in open(ROOT / "results/sweeps/cells.csv")) - 1
    pt.text(160, 945, f"{n} grid cells x {b['seeds']} seeds. Red line: highest speed that "
            f"passes (≥{b['pass_threshold']:.0%}) at every lower speed. Hatched: "
            "beyond any perception.", 18, MUTED)
    img = mpimg.imread(ROOT / "docs/figures/sweep_lead_brake.png")
    h_px = 800
    w_px = h_px * img.shape[1] / img.shape[0]
    ax = pt.axes(160, 100, w_px, h_px)
    ax.imshow(img)
    ax.axis("off")

    def s(v):
        return f"{v} km/h" if v is not None else "none"
    x = 160 + w_px + 60
    pt.text(x, 820, "Lead brakes, any gap 10-60 m", 20, weight="bold")
    pt.text(x, 780, "highest speed handled", 16, MUTED)
    rows = [("day", "day/clear"), ("rain", "day/rain"), ("night fog", "night/fog")]
    pt.text(x + 180, 735, "measured", 15, MUTED)
    pt.text(x + 330, 735, "nuScenes", 15, MUTED)
    for i, (lab, c) in enumerate(rows):
        y = 690 - i * 46
        pt.text(x, y, lab, 18)
        pt.text(x + 180, y, s(wc["sim-measured"]["lead_brake"][c]), 18, weight="bold")
        pt.text(x + 330, y, s(wc["nuScenes-anchored"]["lead_brake"][c]), 18, weight="bold")
    cut = wc["sim-measured"]["cut_in"]["day/clear"]
    pt.text(x, 480, "Cut-in, any gap 5-30 m", 20, weight="bold")
    pt.text(x, 440, f"day: {s(cut)} (both models)", 18)
    return pt.array()


def outro_card(pt: Painter) -> np.ndarray:
    pt.new()
    pt.text(160, 640, "Code, results and failure tables", 44, weight="bold")
    pt.text(160, 560, "github.com/fauzan111/ADAS_perception_scenario_validation_automotive",
            28, BLUE)
    for i, ln in enumerate(["Python · MetaDrive · PyTorch / torchvision · "
                            "Docker · pytest",
                            "Every number in this video is generated by `make` targets in "
                            "the repository."]):
        pt.text(160, 460 - i * 48, ln, 22, MUTED)
    return pt.array()


class Encoder:
    def __init__(self, path: Path, gif: Path | None = None) -> None:
        import imageio_ffmpeg
        self.ff = imageio_ffmpeg.get_ffmpeg_exe()
        raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "-"]
        self.p = subprocess.Popen([self.ff, "-y", "-loglevel", "error", *raw, "-c:v",
                                   "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt",
                                   "yuv420p", "-movflags", "+faststart", str(path)],
                                  stdin=subprocess.PIPE)
        self.gif_path, self.g = gif, None
        self.frames = 0

    def write(self, frame: np.ndarray, n: int = 1, gif: bool = False) -> None:
        b = frame.tobytes()
        for _ in range(n):
            self.p.stdin.write(b)
        if gif:
            if self.g is None:
                self.g = subprocess.Popen(
                    [self.ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                     "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-vf",
                     "fps=5,scale=640:-1:flags=lanczos,split[a][b];"
                     "[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                     str(self.gif_path)], stdin=subprocess.PIPE)
            for _ in range(n):
                self.g.stdin.write(b)
        self.frames += n

    def segment(self, frame: np.ndarray, seconds: float, fade: bool = True) -> None:
        n = int(seconds * FPS)
        f = 8 if fade else 0
        for i in range(n):
            a = min(1.0, (i + 1) / f, (n - i) / f) if f else 1.0
            self.write(frame if a >= 1 else (frame * a).astype(np.uint8))

    def close(self) -> None:
        for p in (self.p, self.g):
            if p:
                p.stdin.close()
                p.wait()


def compose(out: Path) -> None:
    pt = Painter()
    st = json.loads((out / "stats.json").read_text())
    runs = {c["name"]: json.loads((out / c["name"] / "run.json").read_text()) for c in CLIPS}
    for c in CLIPS:                      # wording from CLIPS, not from the recording
        runs[c["name"]]["meta"].update(title=c["title"], subtitle=c["subtitle"])
        trim(runs[c["name"]])
    gif = ROOT / "docs/media/demo_lead_brake.gif"
    gif.parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(out / "adas_demo.mp4", gif)
    enc.segment(title_card(pt), 5)
    enc.segment(pipeline_card(pt), 8)
    intros = {
        "ped_pass": ["The parked vehicle hides the pedestrian until the last metres.",
                     f"Over 30 seeds: {st['ped_pass']['collisions']} collisions."],
        "ped_fail": [f"Over 30 seeds: {st['ped_fail']['collisions']} collisions."],
        "lead130": ["The ACC keeps braking within comfort limits; the AEB must take over.",
                    f"Over 30 seeds: {st['lead130']['collisions']} collisions."],
        "lead130_known": ["Perception, tracker and seed unchanged.",
                          f"Over 30 seeds: {st['lead130_known']['collisions']} collisions."],
    }
    for i, clip in enumerate(CLIPS, start=1):
        run = runs[clip["name"]]
        in_gif = clip["name"].startswith("lead130")
        sub = clip["subtitle"][0].upper() + clip["subtitle"][1:]
        enc.segment(card(pt, clip["title"], [sub, *intros[clip["name"]]],
                         kicker=f"SCENARIO {i} OF {len(CLIPS)}", title_size=40), 3)
        evs = events(run)
        slow_from = next(te for te, key, _ in evs if key in ("walk", "brake")) - 0.5
        frames = out / clip["name"] / "frames"
        tr = run["trace"]
        last = None
        for k in range(len(tr)):
            last = draw_clip_frame(pt, run, k, frames / f"{k:04d}.jpg", evs, i, len(CLIPS),
                                   final=False)
            enc.write(last, 3 if tr[k]["t"] < slow_from else 6, gif=in_gif)
        final = draw_clip_frame(pt, run, len(tr) - 1, frames / f"{len(tr) - 1:04d}.jpg", evs,
                                i, len(CLIPS), final=True)
        enc.write(final, int(2.5 * FPS), gif=in_gif)
        if clip["name"] == "ped_fail":
            enc.segment(counterfactual_card(pt, st), 9)
        print(f"{clip['name']}: {len(tr)} frames composed", flush=True)
    enc.segment(sotif_card(pt), 9)
    enc.segment(sweep_card(pt), 9)
    enc.segment(outro_card(pt), 6)
    enc.close()
    print(f"{out / 'adas_demo.mp4'}: {enc.frames / FPS:.0f} s; {gif}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["record", "stats", "compose"])
    ap.add_argument("--clips", nargs="*", help="record only these clips")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    {"record": lambda: record(out, args.clips), "stats": lambda: stats(out),
     "compose": lambda: compose(out)}[args.step]()


if __name__ == "__main__":
    main()
