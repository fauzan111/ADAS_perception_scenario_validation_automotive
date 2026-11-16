"""2D geometry on the road plane: axis-aligned boxes, overlap and line-of-sight occlusion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    length: float      # along x (road direction)
    width: float       # along y

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.cx - self.length / 2, self.cx + self.length / 2,
                self.cy - self.width / 2, self.cy + self.width / 2)


def boxes_overlap(a: Box, b: Box) -> bool:
    ax0, ax1, ay0, ay1 = a.bounds
    bx0, bx1, by0, by1 = b.bounds
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def box_circle_overlap(b: Box, cx: float, cy: float, r: float) -> bool:
    x0, x1, y0, y1 = b.bounds
    nx, ny = min(max(cx, x0), x1), min(max(cy, y0), y1)
    return (cx - nx) ** 2 + (cy - ny) ** 2 < r * r


def segment_hits_box(p: tuple[float, float], q: tuple[float, float], b: Box) -> bool:
    """Liang-Barsky clipping: does the segment p-q pass through the box interior?"""
    x0, x1, y0, y1 = b.bounds
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pk == 0:
            if qk < 0:
                return False
            continue
        r = qk / pk
        if pk < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return t1 - t0 > 1e-9


def visible_fraction(sensor: tuple[float, float], target: Box, occluders: list[Box],
                     samples: int = 7) -> float:
    """Fraction of rays from the sensor to points across the target's side facing the
    sensor that no occluder blocks."""
    x0, x1, y0, y1 = target.bounds
    near_x = x0 if sensor[0] < x0 else (x1 if sensor[0] > x1 else target.cx)
    ys = [y0 + (y1 - y0) * (i + 0.5) / samples for i in range(samples)]
    blockers = [o for o in occluders if o != target]
    clear = sum(1 for y in ys
                if not any(segment_hits_box(sensor, (near_x, y), o) for o in blockers))
    return clear / samples
