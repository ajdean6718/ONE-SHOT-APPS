"""Stroke outlining for vector paths (owner: core-geometry).

Two outputs: a triangle mesh for viewport display (overlaps allowed), and outline
rings for export rasterization (fill with NONZERO rule).

``round_segments`` is the number of straight segments used per quarter turn (90 degrees)
of a round cap or round join, so a semicircular cap uses ``2 * round_segments`` segments.
The mesh trims the inner side of every join to the inner miter point where that is
geometrically safe, so ordinary strokes produce non-overlapping triangles.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import Vec2, _check_scalars, as_points

CAPS = ("BUTT", "ROUND", "SQUARE")
JOINS = ("MITER", "ROUND", "BEVEL")

_PARALLEL = 1e-12


class _Path:
    """Deduplicated polyline with per-segment direction data."""

    def __init__(self, polyline, closed: bool):
        a = as_points(polyline, "polyline")
        pts: List[Vec2] = []
        for x, y in a.tolist():
            if pts and abs(x - pts[-1][0]) <= 1e-12 and abs(y - pts[-1][1]) <= 1e-12:
                continue
            pts.append((x, y))
        while closed and len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) <= 1e-12 and abs(pts[0][1] - pts[-1][1]) <= 1e-12:
            pts.pop()
        self.pts = pts
        n = len(pts)
        self.closed = bool(closed) and n >= 2
        self.m = 0 if n < 2 else (n if self.closed else n - 1)
        self.d: List[Vec2] = []
        self.n: List[Vec2] = []
        self.len: List[float] = []
        for i in range(self.m):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            dx = bx - ax
            dy = by - ay
            ln = math.hypot(dx, dy)
            self.d.append((dx / ln, dy / ln))
            self.n.append((-dy / ln, dx / ln))
            self.len.append(ln)


def _rot(v: Vec2, theta: float) -> Vec2:
    c = math.cos(theta)
    s = math.sin(theta)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def _add(p: Vec2, v: Vec2, k: float = 1.0) -> Vec2:
    return (p[0] + v[0] * k, p[1] + v[1] * k)


def _area2(poly: Sequence[Vec2]) -> float:
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i - 1]
        x2, y2 = poly[i]
        s += x1 * y2 - x2 * y1
    return s


def _check_args(width, cap, join, miter_limit, round_segments):
    _check_scalars(width, miter_limit)
    if cap not in CAPS:
        raise ValueError("cap must be one of %s, got %r" % (CAPS, cap))
    if join not in JOINS:
        raise ValueError("join must be one of %s, got %r" % (JOINS, join))
    return max(1.0, float(miter_limit)), max(1, int(round_segments))


class _Join:
    """Geometry of the join at one vertex between segment ``a`` (incoming) and ``b`` (outgoing)."""

    __slots__ = ("kind", "v", "s", "tau", "dot", "o_a", "o_b", "inner", "tip", "ratio", "trim_ok", "da")

    def __init__(self, path: _Path, a: int, b: int, v: int, hw: float):
        self.v = path.pts[v]
        da = path.d[a]
        db = path.d[b]
        na = path.n[a]
        nb = path.n[b]
        self.da = da
        cross = da[0] * db[1] - da[1] * db[0]
        dot = da[0] * db[0] + da[1] * db[1]
        self.dot = dot
        self.s = 0
        self.tau = 0.0
        self.o_a = self.o_b = self.inner = self.tip = None
        self.ratio = 1.0
        self.trim_ok = False
        if abs(cross) <= _PARALLEL and dot > 0.0:
            self.kind = "straight"
            return
        if abs(cross) <= _PARALLEL:
            self.kind = "reverse"
            return
        self.kind = "turn"
        self.s = 1 if cross > 0.0 else -1
        self.tau = math.atan2(cross, dot)
        s = self.s
        vx, vy = self.v
        self.o_a = (vx - s * hw * na[0], vy - s * hw * na[1])
        self.o_b = (vx - s * hw * nb[0], vy - s * hw * nb[1])
        k = s * hw / (1.0 + dot)
        mx = (na[0] + nb[0]) * k
        my = (na[1] + nb[1]) * k
        self.inner = (vx + mx, vy + my)
        self.tip = (vx - mx, vy - my)
        self.ratio = math.sqrt(2.0 / (1.0 + dot))
        t = hw * abs(cross) / (1.0 + dot)
        self.trim_ok = t <= 0.5 * min(path.len[a], path.len[b])

    def arc(self, hw: float, na: Vec2, rseg: int) -> List[Vec2]:
        """Interior points of the outer round-join arc (excluding o_a and o_b)."""
        steps = max(1, int(math.ceil(rseg * abs(self.tau) / (math.pi / 2.0) - 1e-9)))
        out = []
        for k in range(1, steps):
            r = _rot(na, self.tau * k / steps)
            out.append((self.v[0] - self.s * hw * r[0], self.v[1] - self.s * hw * r[1]))
        return out

    def outer_chain(self, path: _Path, a: int, hw: float, join: str, miter_limit: float, rseg: int) -> List[Vec2]:
        """Outer boundary points from o_a to o_b (inclusive)."""
        chain = [self.o_a]
        if join == "MITER" and self.ratio <= miter_limit:
            chain.append(self.tip)
        elif join == "ROUND":
            chain.extend(self.arc(hw, path.n[a], rseg))
        chain.append(self.o_b)
        return chain


def _semicircle(center: Vec2, start: Vec2, hw: float, rseg: int, end: Vec2) -> List[Vec2]:
    """CCW half circle from ``start`` (a unit offset direction scaled later) to ``end``."""
    steps = max(1, 2 * rseg)
    pts = [None] * (steps + 1)
    for k in range(steps + 1):
        r = _rot(start, math.pi * k / steps)
        pts[k] = (center[0] + hw * r[0], center[1] + hw * r[1])
    return pts


def _dot_pieces(p: Vec2, hw: float, cap: str, rseg: int) -> List[List[Vec2]]:
    """A zero-length open sub-path: SVG draws a dot for round caps and a square for square caps."""
    if cap == "ROUND":
        steps = 4 * rseg
        return [[(p[0] + hw * math.cos(2 * math.pi * k / steps), p[1] + hw * math.sin(2 * math.pi * k / steps))
                 for k in range(steps)]]
    if cap == "SQUARE":
        return [[(p[0] - hw, p[1] - hw), (p[0] + hw, p[1] - hw), (p[0] + hw, p[1] + hw), (p[0] - hw, p[1] + hw)]]
    return []


def stroke_outline(polyline: Sequence[Vec2], closed: bool, width: float, cap: str = "BUTT", join: str = "MITER",
                   miter_limit: float = 4.0, round_segments: int = 8) -> List[List[Vec2]]:
    """Rings whose NONZERO union is the stroked area (segment quads + join wedges + cap pieces are fine)."""
    miter_limit, rseg = _check_args(width, cap, join, miter_limit, round_segments)
    path = _Path(polyline, closed)
    if width <= 0.0 or not path.pts:
        return []
    hw = float(width) / 2.0
    if path.m == 0:
        return _dot_pieces(path.pts[0], hw, cap, rseg)
    rings: List[List[Vec2]] = []
    pts = path.pts
    n = len(pts)

    def ccw(poly: List[Vec2]) -> List[Vec2]:
        return poly[::-1] if _area2(poly) < 0.0 else poly

    for i in range(path.m):
        a = pts[i]
        b = pts[(i + 1) % n]
        nx, ny = path.n[i]
        rings.append([(a[0] - hw * nx, a[1] - hw * ny), (b[0] - hw * nx, b[1] - hw * ny),
                      (b[0] + hw * nx, b[1] + hw * ny), (a[0] + hw * nx, a[1] + hw * ny)])
    verts = range(n) if path.closed else range(1, n - 1)
    for v in verts:
        a = (v - 1) % path.m
        b = v % path.m
        j = _Join(path, a, b, v, hw)
        if j.kind == "turn":
            rings.append(ccw([j.v] + j.outer_chain(path, a, hw, join, miter_limit, rseg)))
        elif j.kind == "reverse" and join == "ROUND":
            na = path.n[a]
            steps = max(1, 2 * rseg)
            arc = []
            for k in range(steps + 1):
                r = _rot(na, -math.pi * k / steps)
                arc.append((j.v[0] + hw * r[0], j.v[1] + hw * r[1]))
            rings.append(ccw([j.v] + arc))
    if not path.closed:
        d0, n0, p0 = path.d[0], path.n[0], pts[0]
        d1, n1, p1 = path.d[-1], path.n[-1], pts[-1]
        if cap == "SQUARE":
            ls = (p0[0] + hw * n0[0], p0[1] + hw * n0[1])
            rs = (p0[0] - hw * n0[0], p0[1] - hw * n0[1])
            rings.append([(rs[0] - hw * d0[0], rs[1] - hw * d0[1]), rs, ls, (ls[0] - hw * d0[0], ls[1] - hw * d0[1])])
            le = (p1[0] + hw * n1[0], p1[1] + hw * n1[1])
            re = (p1[0] - hw * n1[0], p1[1] - hw * n1[1])
            rings.append([re, (re[0] + hw * d1[0], re[1] + hw * d1[1]), (le[0] + hw * d1[0], le[1] + hw * d1[1]), le])
        elif cap == "ROUND":
            rings.append(ccw([p0] + _semicircle(p0, n0, hw, rseg, None)))
            rings.append(ccw([p1] + _semicircle(p1, (-n1[0], -n1[1]), hw, rseg, None)))
    return [r for r in rings if len(r) >= 3]


def stroke_mesh(polyline: Sequence[Vec2], closed: bool, width: float, cap: str = "BUTT", join: str = "MITER",
                miter_limit: float = 4.0, round_segments: int = 8) -> Tuple[List[Vec2], List[Tuple[int, int, int]]]:
    """Triangulated stroke band (segments + joins + caps). width <= 0 -> empty."""
    miter_limit, rseg = _check_args(width, cap, join, miter_limit, round_segments)
    path = _Path(polyline, closed)
    if width <= 0.0 or not path.pts:
        return [], []
    hw = float(width) / 2.0
    verts: List[Vec2] = []
    tris: List[Tuple[int, int, int]] = []
    cache: Dict[Vec2, int] = {}

    def vid(p: Vec2) -> int:
        idx = cache.get(p)
        if idx is None:
            idx = len(verts)
            cache[p] = idx
            verts.append(p)
        return idx

    def tri(a: Vec2, b: Vec2, c: Vec2) -> None:
        ar = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if ar == 0.0:
            return
        if ar < 0.0:
            b, c = c, b
        tris.append((vid(a), vid(b), vid(c)))

    def fan(pivot: Vec2, ring: Sequence[Vec2]) -> None:
        for k in range(len(ring) - 1):
            tri(pivot, ring[k], ring[k + 1])

    if path.m == 0:
        for poly in _dot_pieces(path.pts[0], hw, cap, rseg):
            fan(poly[0], poly[1:] + poly[:1])
        return verts, tris

    pts = path.pts
    n = len(pts)
    m = path.m
    # default quad corners per segment: [start_left, start_right, end_left, end_right]
    corners: List[List[Vec2]] = []
    for i in range(m):
        a = pts[i]
        b = pts[(i + 1) % n]
        nx, ny = path.n[i]
        corners.append([(a[0] + hw * nx, a[1] + hw * ny), (a[0] - hw * nx, a[1] - hw * ny),
                        (b[0] + hw * nx, b[1] + hw * ny), (b[0] - hw * nx, b[1] - hw * ny)])
    wedges: List[Tuple[Vec2, List[Vec2]]] = []
    verts_with_join = range(n) if path.closed else range(1, n - 1)
    for v in verts_with_join:
        a = (v - 1) % m
        b = v % m
        j = _Join(path, a, b, v, hw)
        if j.kind == "straight":
            continue
        if j.kind == "reverse":
            if join == "ROUND":
                na = path.n[a]
                steps = max(1, 2 * rseg)
                arc = []
                for k in range(steps + 1):
                    r = _rot(na, -math.pi * k / steps)
                    arc.append((j.v[0] + hw * r[0], j.v[1] + hw * r[1]))
                wedges.append((j.v, arc))
            continue
        chain = j.outer_chain(path, a, hw, join, miter_limit, rseg)
        if j.trim_ok:
            if j.s > 0:  # left turn: the inner side is the left side
                corners[a][2] = j.inner
                corners[b][0] = j.inner
            else:
                corners[a][3] = j.inner
                corners[b][1] = j.inner
            wedges.append((j.inner, chain))
        else:
            wedges.append((j.v, chain))
    for ls, rs, le, re in corners:
        tri(rs, re, le)
        tri(rs, le, ls)
    for pivot, chain in wedges:
        fan(pivot, chain)
    if not path.closed:
        d0, n0, p0 = path.d[0], path.n[0], pts[0]
        d1, n1, p1 = path.d[-1], path.n[-1], pts[-1]
        ls0, rs0 = corners[0][0], corners[0][1]
        le1, re1 = corners[-1][2], corners[-1][3]
        if cap == "SQUARE":
            ls_x = (ls0[0] - hw * d0[0], ls0[1] - hw * d0[1])
            rs_x = (rs0[0] - hw * d0[0], rs0[1] - hw * d0[1])
            tri(rs_x, rs0, ls0)
            tri(rs_x, ls0, ls_x)
            le_x = (le1[0] + hw * d1[0], le1[1] + hw * d1[1])
            re_x = (re1[0] + hw * d1[0], re1[1] + hw * d1[1])
            tri(re1, re_x, le_x)
            tri(re1, le_x, le1)
        elif cap == "ROUND":
            arc0 = _semicircle(p0, n0, hw, rseg, None)
            arc0[0], arc0[-1] = ls0, rs0
            fan(p0, arc0)
            arc1 = _semicircle(p1, (-n1[0], -n1[1]), hw, rseg, None)
            arc1[0], arc1[-1] = re1, le1
            fan(p1, arc1)
    return verts, tris
