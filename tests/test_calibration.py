"""Geometry and image degradations of the detector calibration (scripts/calibrate_detector.py)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
spec = importlib.util.spec_from_file_location(
    "calibrate_detector", Path(__file__).resolve().parents[1] / "scripts/calibrate_detector.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)

F, CX, CY = cal.intrinsics(60.0)


def test_projection_matches_pinhole_geometry():
    assert F == pytest.approx(480 / math.tan(math.radians(30)))
    u0, v0, u1, v1 = cal.project_box(20.0, 0.0, "car", F, CX, CY)
    assert (u0 + u1) / 2 == pytest.approx(CX)                 # straight ahead: centred
    assert v1 == pytest.approx(CY + F * cal.CAM_H / 20.0)    # bottom = road contact point
    assert u1 - u0 == pytest.approx(F * 1.852 / 20.0)        # width scales with 1/range
    left = cal.project_box(20.0, 3.5, "car", F, CX, CY)
    assert left[2] < CX                                       # +lateral = left of centre


def test_depth_map_road_sky_and_object():
    box = cal.project_box(30.0, 0.0, "pedestrian", F, CX, CY)
    d = cal.depth_map(box, 30.0, F, CY)
    assert d[0, 0] >= 1e4                                     # sky
    row = int(CY + F * cal.CAM_H / 10.0)
    assert d[row, 10] == pytest.approx(10.0, rel=0.02)        # road 10 m ahead
    u, v = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    assert d[v, u] == 30.0


def test_fog_and_night_darken_or_wash_out_with_distance():
    rng = np.random.default_rng(0)
    img = np.full((cal.H, cal.W, 3), 60, np.uint8)
    box = cal.project_box(40.0, 0.0, "car", F, CX, CY)
    d = cal.depth_map(box, 40.0, F, CY)
    fog = cal.degrade(img, "fog", d, rng).astype(float)
    near_row, far_row = int(CY + F * cal.CAM_H / 8), int(CY + F * cal.CAM_H / 60)
    assert fog[near_row, 5].mean() < fog[far_row, 5].mean()  # farther = closer to airlight
    night = cal.degrade(img, "night", d, rng).astype(float)
    assert night[near_row, 5].mean() > night[far_row, 5].mean()   # headlights fall off
    assert night.mean() < img.mean()
