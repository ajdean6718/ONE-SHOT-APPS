"""Parametric vector primitives -> BezierPath (owner: core-geometry).

All primitives are centered on (cx, cy) unless noted, CCW, closed (except line),
and use exact Bezier handles: corners use VECTOR handles, ellipse arcs use the
kappa = 0.5522847498 cubic approximation with ALIGNED handles.

Angles (``rotation``) are in radians, counter-clockwise.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .geometry import BezierPath, BezierPoint, Vec2, _check_scalars, transform_path

KAPPA = 0.5522847498307936


def _third(a: Vec2, b: Vec2) -> Vec2:
    """Point one third of the way from a to b (the position of a VECTOR handle)."""
    return (a[0] + (b[0] - a[0]) / 3.0, a[1] + (b[1] - a[1]) / 3.0)


def _polygon_path(pts: Sequence[Vec2], closed: bool = True) -> BezierPath:
    """Corner-only path: every handle is a VECTOR handle one third along the adjacent edge."""
    n = len(pts)
    out: List[BezierPoint] = []
    for i, p in enumerate(pts):
        nxt = pts[(i + 1) % n] if (closed or i + 1 < n) else p
        prv = pts[(i - 1) % n] if (closed or i > 0) else p
        out.append(BezierPoint(co=(float(p[0]), float(p[1])), handle_left=_third(p, prv),
                               handle_right=_third(p, nxt), handle_left_type="VECTOR",
                               handle_right_type="VECTOR"))
    return BezierPath(points=out, closed=closed)


def rect(width: float, height: float, cx: float = 0.0, cy: float = 0.0) -> BezierPath:
    _check_scalars(width, height, cx, cy)
    hw = abs(width) / 2.0
    hh = abs(height) / 2.0
    return _polygon_path([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)])


def rounded_rect(width: float, height: float, radius: float, cx: float = 0.0, cy: float = 0.0) -> BezierPath:
    """Radius is clamped to min(width, height)/2; radius <= 0 returns rect()."""
    _check_scalars(width, height, radius, cx, cy)
    hw = abs(width) / 2.0
    hh = abs(height) / 2.0
    r = min(float(radius), hw, hh)
    if r <= 0.0:
        return rect(width, height, cx, cy)
    k = KAPPA * r
    # (corner center, start direction on that corner's arc): going CCW starting bottom edge.
    # Each corner is an arc from A (end of previous edge) to B (start of next edge).
    corners = [
        ((cx + hw - r, cy - hh + r), (cx + hw - r, cy - hh), (cx + hw, cy - hh + r), (1.0, 0.0), (0.0, 1.0)),
        ((cx + hw - r, cy + hh - r), (cx + hw, cy + hh - r), (cx + hw - r, cy + hh), (0.0, 1.0), (-1.0, 0.0)),
        ((cx - hw + r, cy + hh - r), (cx - hw + r, cy + hh), (cx - hw, cy + hh - r), (-1.0, 0.0), (0.0, -1.0)),
        ((cx - hw + r, cy - hh + r), (cx - hw, cy - hh + r), (cx - hw + r, cy - hh), (0.0, -1.0), (1.0, 0.0)),
    ]
    # cubic segments (p0, c1, c2, p3) in CCW order
    segs: List[Tuple[Vec2, Vec2, Vec2, Vec2]] = []
    first_start = corners[0][1]
    prev_end = None
    for center, a, b, d0, d1 in corners:
        if prev_end is not None:
            segs.append((prev_end, _third(prev_end, a), _third(a, prev_end), a))
        segs.append((a, (a[0] + d0[0] * k, a[1] + d0[1] * k), (b[0] - d1[0] * k, b[1] - d1[1] * k), b))
        prev_end = b
    segs.append((prev_end, _third(prev_end, first_start), _third(first_start, prev_end), first_start))
    # drop zero-length straight edges (radius == half a side)
    segs = [s for s in segs if math.hypot(s[3][0] - s[0][0], s[3][1] - s[0][1]) > 1e-12 or s[1] != s[0] or s[2] != s[3]]
    pts: List[BezierPoint] = []
    for i, (p0, c1, _c2, _p3) in enumerate(segs):
        prev_c2 = segs[i - 1][2]
        pts.append(BezierPoint(co=p0, handle_left=prev_c2, handle_right=c1,
                               handle_left_type="ALIGNED", handle_right_type="ALIGNED"))
    # Zero-length edges were dropped above, so adjacent arcs share their junction point exactly.
    return BezierPath(points=_merge_coincident(pts), closed=True)


def _merge_coincident(pts: List[BezierPoint]) -> List[BezierPoint]:
    """Merge consecutive points with identical positions (keeping the outer handles)."""
    out: List[BezierPoint] = []
    for p in pts:
        if out and math.hypot(p.co[0] - out[-1].co[0], p.co[1] - out[-1].co[1]) <= 1e-12:
            out[-1].handle_right = p.handle_right
            out[-1].handle_right_type = p.handle_right_type
        else:
            out.append(p)
    if len(out) > 1 and math.hypot(out[0].co[0] - out[-1].co[0], out[0].co[1] - out[-1].co[1]) <= 1e-12:
        out[0].handle_left = out[-1].handle_left
        out[0].handle_left_type = out[-1].handle_left_type
        out.pop()
    return out


def ellipse(rx: float, ry: float, cx: float = 0.0, cy: float = 0.0) -> BezierPath:
    """Four-point Bezier ellipse."""
    _check_scalars(rx, ry, cx, cy)
    rx = abs(rx)
    ry = abs(ry)
    kx = KAPPA * rx
    ky = KAPPA * ry
    # right, top, left, bottom (CCW); tangent directions at those points: up, left, down, right.
    spec = [
        ((cx + rx, cy), (0.0, ky)),
        ((cx, cy + ry), (-kx, 0.0)),
        ((cx - rx, cy), (0.0, -ky)),
        ((cx, cy - ry), (kx, 0.0)),
    ]
    pts = [BezierPoint(co=co, handle_left=(co[0] - t[0], co[1] - t[1]), handle_right=(co[0] + t[0], co[1] + t[1]),
                       handle_left_type="ALIGNED", handle_right_type="ALIGNED") for co, t in spec]
    return BezierPath(points=pts, closed=True)


def _regular(sides: int, radius: float, cx: float, cy: float, rotation: float) -> List[Vec2]:
    return [(cx + radius * math.cos(math.pi / 2.0 + rotation + 2.0 * math.pi * i / sides),
             cy + radius * math.sin(math.pi / 2.0 + rotation + 2.0 * math.pi * i / sides)) for i in range(sides)]


def polygon(sides: int, radius: float, cx: float = 0.0, cy: float = 0.0, rotation: float = 0.0) -> BezierPath:
    """Regular polygon, first vertex pointing up (+Y) when rotation == 0. sides >= 3."""
    _check_scalars(radius, cx, cy, rotation)
    sides = int(sides)
    if sides < 3:
        raise ValueError("polygon needs at least 3 sides, got %d" % sides)
    return _polygon_path(_regular(sides, abs(radius), cx, cy, rotation))


def star(points: int, radius: float, inner_ratio: float = 0.5, cx: float = 0.0, cy: float = 0.0, rotation: float = 0.0) -> BezierPath:
    """Star with ``points`` outer vertices (>= 3), inner radius = radius * inner_ratio."""
    _check_scalars(radius, inner_ratio, cx, cy, rotation)
    points = int(points)
    if points < 3:
        raise ValueError("star needs at least 3 points, got %d" % points)
    radius = abs(radius)
    inner = radius * max(0.0, float(inner_ratio))
    pts: List[Vec2] = []
    for i in range(2 * points):
        r = radius if i % 2 == 0 else inner
        a = math.pi / 2.0 + rotation + math.pi * i / points
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return _polygon_path(pts)


def line(p0: Vec2, p1: Vec2) -> BezierPath:
    """Open two-point path."""
    _check_scalars(p0[0], p0[1], p1[0], p1[1])
    a = (float(p0[0]), float(p0[1]))
    b = (float(p1[0]), float(p1[1]))
    return _polygon_path([a, b], closed=False)


def from_kind(kind: str, width: float, height: float, radius: float = 0.0, sides: int = 5,
              inner_ratio: float = 0.5, cx: float = 0.0, cy: float = 0.0) -> BezierPath:
    """Dispatch on SEAM_PG_ShapeRecipe.kind: RECT, ROUNDED_RECT, ELLIPSE, POLYGON, STAR, LINE.

    POLYGON/STAR use radius = min(width, height)/2 and are scaled to fit width x height.
    LINE goes from (cx - width/2, cy) to (cx + width/2, cy).
    """
    _check_scalars(width, height, radius, inner_ratio, cx, cy)
    if kind == "RECT":
        return rect(width, height, cx, cy)
    if kind == "ROUNDED_RECT":
        return rounded_rect(width, height, radius, cx, cy)
    if kind == "ELLIPSE":
        return ellipse(abs(width) / 2.0, abs(height) / 2.0, cx, cy)
    if kind in ("POLYGON", "STAR"):
        r = min(abs(width), abs(height)) / 2.0
        if r <= 0.0:
            raise ValueError("%s needs a positive width and height" % kind)
        base = polygon(sides, r) if kind == "POLYGON" else star(sides, r, inner_ratio)
        sx = abs(width) / (2.0 * r)
        sy = abs(height) / (2.0 * r)
        return transform_path(base, (sx, 0.0, 0.0, sy, cx, cy))
    if kind == "LINE":
        return line((cx - width / 2.0, cy), (cx + width / 2.0, cy))
    raise ValueError("unsupported shape kind %r" % (kind,))
