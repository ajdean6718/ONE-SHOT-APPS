"""2D path geometry kernels (contract + implementation owned by core-geometry).

Conventions
-----------
* ``Vec2`` is ``(x, y)`` floats. A *ring* is a list of Vec2 **without** a repeated
  closing point. A *polyline* is an open list of Vec2.
* Positive signed area == counter-clockwise (canvas Y up).
* A normalized *Shape* is ``(outer, holes)`` with ``outer`` CCW and every hole CW.
* Affine 2D transforms are 6-tuples ``(a, b, c, d, e, f)`` mapping
  ``x' = a*x + c*y + e``, ``y' = b*x + d*y + f`` (SVG ``matrix(a b c d e f)`` order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

Vec2 = Tuple[float, float]
Ring = List[Vec2]
Affine = Tuple[float, float, float, float, float, float]
IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

HANDLE_TYPES = ("FREE", "ALIGNED", "VECTOR", "AUTO")  # Blender bezier handle types


@dataclass
class BezierPoint:
    co: Vec2
    handle_left: Vec2
    handle_right: Vec2
    handle_left_type: str = "FREE"
    handle_right_type: str = "FREE"


@dataclass
class BezierPath:
    """One spline. Segment i goes from points[i] (handle_right) to points[i+1] (handle_left)."""

    points: List[BezierPoint] = field(default_factory=list)
    closed: bool = False


@dataclass
class Issue:
    code: str  # e.g. "SELF_INTERSECTION", "DEGENERATE", "OPEN_PATH", "TOO_FEW_POINTS", "ZERO_AREA"
    message: str
    path_index: int = -1
    point: Optional[Vec2] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def as_points(points, what: str = "points") -> np.ndarray:
    """Convert a sequence of (x, y) to an ``(n, 2)`` float64 array.

    Raises ``ValueError`` for malformed input or any NaN/inf coordinate, so that
    no kernel can hang or silently produce garbage on non-finite data.
    """
    try:
        a = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s: expected a sequence of (x, y) pairs" % what) from exc
    if a.size == 0:
        return a.reshape(0, 2)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("%s: expected a sequence of (x, y) pairs, got shape %s" % (what, a.shape))
    if not np.isfinite(a).all():
        raise ValueError("%s: contains NaN or infinite coordinates" % what)
    return a


def _tolist(a: np.ndarray) -> List[Vec2]:
    return [(float(x), float(y)) for x, y in a.tolist()]


def _check_scalars(*values: float) -> None:
    for v in values:
        if not math.isfinite(v):
            raise ValueError("non-finite value %r" % (v,))


def _check_path(path: BezierPath) -> None:
    for p in path.points:
        _check_scalars(p.co[0], p.co[1], p.handle_left[0], p.handle_left[1],
                       p.handle_right[0], p.handle_right[1])


def cubic_point(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: float) -> Vec2:
    """Evaluate a cubic Bezier at ``t``."""
    _check_scalars(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], t)
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3.0 * u * u * t
    b2 = 3.0 * u * t * t
    b3 = t * t * t
    return (b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
            b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1])


_STRAIGHT_EPS = 1e-9


def _segment_is_straight(a: BezierPoint, b: BezierPoint) -> bool:
    """True when the segment a -> b is a straight line (VECTOR handles or handles on the chord)."""
    if a.handle_right_type == "VECTOR" and b.handle_left_type == "VECTOR":
        return True
    p0x, p0y = a.co
    p3x, p3y = b.co
    h0x, h0y = a.handle_right
    h1x, h1y = b.handle_left
    dx = p3x - p0x
    dy = p3y - p0y
    l2 = dx * dx + dy * dy
    scale = max(1.0, abs(p0x), abs(p0y), abs(p3x), abs(p3y))
    tol = _STRAIGHT_EPS * scale
    if l2 <= tol * tol:
        # Degenerate chord: straight only if both handles sit on the anchors.
        return (abs(h0x - p0x) <= tol and abs(h0y - p0y) <= tol
                and abs(h1x - p3x) <= tol and abs(h1y - p3y) <= tol)
    length = math.sqrt(l2)
    s_prev = -1.0
    for hx, hy in ((h0x, h0y), (h1x, h1y)):
        rx = hx - p0x
        ry = hy - p0y
        if abs(dx * ry - dy * rx) / length > tol:
            return False
        s = (rx * dx + ry * dy) / l2
        if s < -_STRAIGHT_EPS or s > 1.0 + _STRAIGHT_EPS or s < s_prev - _STRAIGHT_EPS:
            return False
        s_prev = s
    return True


def flatten_path(path: BezierPath, resolution: int = 12) -> List[Vec2]:
    """Flatten a Bezier path into a polyline/ring.

    ``resolution`` = samples per segment (like Blender ``resolution_u``); straight
    segments (both handles on the chord / VECTOR handles) emit only their end point.
    Closed paths return a ring without the duplicated first point.
    """
    pts = path.points
    n = len(pts)
    if n == 0:
        return []
    _check_path(path)
    res = max(1, int(resolution))
    out: List[Vec2] = [(float(pts[0].co[0]), float(pts[0].co[1]))]
    nseg = n if path.closed else n - 1
    if nseg <= 0:
        return out
    ts = np.arange(1, res, dtype=np.float64) / res  # interior parameters
    u = 1.0 - ts
    b0 = (u * u * u)[:, None]
    b1 = (3.0 * u * u * ts)[:, None]
    b2 = (3.0 * u * ts * ts)[:, None]
    b3 = (ts * ts * ts)[:, None]
    for i in range(nseg):
        a = pts[i]
        b = pts[(i + 1) % n]
        wraps = path.closed and i == n - 1
        if not _segment_is_straight(a, b) and res > 1:
            ctrl = np.array([a.co, a.handle_right, b.handle_left, b.co], dtype=np.float64)
            inner = b0 * ctrl[0] + b1 * ctrl[1] + b2 * ctrl[2] + b3 * ctrl[3]
            out.extend(_tolist(inner))
        if not wraps:
            out.append((float(b.co[0]), float(b.co[1])))
    return out


def flatten_paths(paths: Sequence[BezierPath], resolution: int = 12) -> List[List[Vec2]]:
    return [flatten_path(p, resolution) for p in paths]


def signed_area(ring: Sequence[Vec2]) -> float:
    n = len(ring)
    if n < 3:
        return 0.0
    a = as_points(ring, "ring")
    x = a[:, 0]
    y = a[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def is_ccw(ring: Sequence[Vec2]) -> bool:
    return signed_area(ring) > 0.0


def bbox(rings: Sequence[Sequence[Vec2]]) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy); raises ValueError on empty input."""
    chunks = [as_points(r, "ring") for r in rings]
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        raise ValueError("bbox of empty input")
    a = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    mn = a.min(axis=0)
    mx = a.max(axis=0)
    return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))


def path_length(polyline: Sequence[Vec2], closed: bool = False) -> float:
    a = as_points(polyline, "polyline")
    if len(a) < 2:
        return 0.0
    d = np.diff(a, axis=0)
    total = float(np.hypot(d[:, 0], d[:, 1]).sum())
    if closed:
        total += float(math.hypot(a[0, 0] - a[-1, 0], a[0, 1] - a[-1, 1]))
    return total


def point_in_ring(pt: Vec2, ring: Sequence[Vec2]) -> bool:
    """Even-odd point-in-polygon test (boundary counts as outside or inside consistently)."""
    px, py = float(pt[0]), float(pt[1])
    _check_scalars(px, py)
    a = as_points(ring, "ring")
    n = len(a)
    if n < 3:
        return False
    inside = False
    x1, y1 = a[-1]
    for x2, y2 in a.tolist():
        if (y2 > py) != (y1 > py):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < xint:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def winding_number(pt: Vec2, ring: Sequence[Vec2]) -> int:
    px, py = float(pt[0]), float(pt[1])
    _check_scalars(px, py)
    a = as_points(ring, "ring")
    n = len(a)
    if n < 3:
        return 0
    wn = 0
    x1, y1 = a[-1]
    for x2, y2 in a.tolist():
        if y1 <= py:
            if y2 > py and (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1) > 0.0:
                wn += 1
        elif y2 <= py and (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1) < 0.0:
            wn -= 1
        x1, y1 = x2, y2
    return wn


def transform_points(points: Sequence[Vec2], m: Affine) -> List[Vec2]:
    a = as_points(points, "points")
    _check_scalars(*m)
    if len(a) == 0:
        return []
    ma, mb, mc, md, me, mf = m
    x = a[:, 0]
    y = a[:, 1]
    out = np.empty_like(a)
    out[:, 0] = ma * x + mc * y + me
    out[:, 1] = mb * x + md * y + mf
    return _tolist(out)


def affine_multiply(m1: Affine, m2: Affine) -> Affine:
    """Return m1 ∘ m2 (apply m2 first)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + c1 * b2,
            b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2,
            b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1,
            b1 * e2 + d1 * f2 + f1)


def affine_determinant(m: Affine) -> float:
    return m[0] * m[3] - m[1] * m[2]


def _apply(m: Affine, p: Vec2) -> Vec2:
    return (m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5])


def transform_path(path: BezierPath, m: Affine) -> BezierPath:
    _check_scalars(*m)
    _check_path(path)
    pts = [BezierPoint(co=_apply(m, p.co), handle_left=_apply(m, p.handle_left),
                       handle_right=_apply(m, p.handle_right),
                       handle_left_type=p.handle_left_type, handle_right_type=p.handle_right_type)
           for p in path.points]
    return BezierPath(points=pts, closed=path.closed)


def clean_ring(ring: Sequence[Vec2], eps: float = 1e-9) -> List[Vec2]:
    """Drop consecutive duplicates (and the closing duplicate) and exactly collinear points."""
    a = as_points(ring, "ring")
    pts: List[Vec2] = []
    for x, y in a.tolist():
        if pts and abs(x - pts[-1][0]) <= eps and abs(y - pts[-1][1]) <= eps:
            continue
        pts.append((x, y))
    while len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) <= eps and abs(pts[0][1] - pts[-1][1]) <= eps:
        pts.pop()
    n = len(pts)
    if n < 3:
        return pts
    prv = [(i - 1) % n for i in range(n)]
    nxt = [(i + 1) % n for i in range(n)]
    alive = [True] * n
    count = n
    work = list(range(n))
    while work and count > 2:
        i = work.pop()
        if not alive[i]:
            continue
        ax, ay = pts[prv[i]]
        px, py = pts[i]
        bx, by = pts[nxt[i]]
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length <= eps:
            continue  # spike returning to the same point: keep, it is not collinear drift
        if abs(dx * (py - ay) - dy * (px - ax)) / length <= eps:
            alive[i] = False
            count -= 1
            p, q = prv[i], nxt[i]
            nxt[p] = q
            prv[q] = p
            work.append(p)
            work.append(q)
    return [pts[i] for i in range(n) if alive[i]]


# ---------------------------------------------------------------------------
# point location / nesting
# ---------------------------------------------------------------------------

def _locate(px: float, py: float, ring: np.ndarray, tol: float) -> int:
    """1 inside, 0 outside, -1 on the boundary (within ``tol``) of ``ring`` ((n,2) array)."""
    x = ring[:, 0]
    y = ring[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    ex = x2 - x
    ey = y2 - y
    elen = np.hypot(ex, ey)
    rx = px - x
    ry = py - y
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(elen > 0, (rx * ex + ry * ey) / np.where(elen > 0, elen * elen, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    dist = np.hypot(rx - t * ex, ry - t * ey)
    if (dist <= tol).any():
        return -1
    cond = (y > py) != (y2 > py)
    if not cond.any():
        return 0
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x + (py - y) * ex / np.where(ey != 0, ey, 1.0)
    return int(np.count_nonzero(cond & (px < xint)) & 1)


class _RingInfo:
    __slots__ = ("pts", "area", "bbox", "tol")

    def __init__(self, pts: np.ndarray):
        self.pts = pts
        x = pts[:, 0]
        y = pts[:, 1]
        self.area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
        self.bbox = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
        ext = max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1], 1e-12)
        self.tol = 1e-9 * ext


def _ring_inside(inner: _RingInfo, outer: _RingInfo) -> bool:
    """True when ``inner`` lies inside ``outer`` (boundary contacts are ignored)."""
    ib = inner.bbox
    ob = outer.bbox
    tol = outer.tol
    if ib[0] < ob[0] - tol or ib[1] < ob[1] - tol or ib[2] > ob[2] + tol or ib[3] > ob[3] + tol:
        return False
    pts = inner.pts
    n = len(pts)
    # Try vertices first, then edge midpoints, until a point off the outer boundary decides.
    step = max(1, n // 8)
    for k in range(0, n, step):
        c = _locate(float(pts[k, 0]), float(pts[k, 1]), outer.pts, tol)
        if c != -1:
            return c == 1
    mids = (pts + np.roll(pts, -1, axis=0)) * 0.5
    for k in range(0, n, step):
        c = _locate(float(mids[k, 0]), float(mids[k, 1]), outer.pts, tol)
        if c != -1:
            return c == 1
    return False


def normalize_rings(rings: Sequence[Sequence[Vec2]], fill_rule: str = "EVENODD") -> List[Tuple[Ring, List[Ring]]]:
    """Group rings into shapes by nesting depth.

    EVENODD: depth-even rings are outers, depth-odd rings are holes of their
    immediate container. NONZERO: uses winding numbers of the input orientation.
    Output orientation is normalized (outer CCW, holes CW). Degenerate rings
    (fewer than 3 points or |area| < 1e-12) are dropped.
    """
    if fill_rule not in ("EVENODD", "NONZERO"):
        raise ValueError("fill_rule must be EVENODD or NONZERO, got %r" % (fill_rule,))
    infos: List[_RingInfo] = []
    for r in rings:
        a = as_points(r, "ring")
        if len(a) >= 3 and a.shape[0] > 1 and tuple(a[0]) == tuple(a[-1]) and len(a) > 3:
            a = a[:-1]
        if len(a) < 3:
            continue
        info = _RingInfo(a)
        if abs(info.area) < 1e-12:
            continue
        infos.append(info)
    n = len(infos)
    if n == 0:
        return []
    # containers[i] = indices of rings that strictly contain ring i
    containers: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        ai = abs(infos[i].area)
        for j in range(n):
            if i == j or abs(infos[j].area) <= ai:
                continue
            if _ring_inside(infos[i], infos[j]):
                containers[i].append(j)

    def parent_of(i: int, allowed) -> int:
        best = -1
        for j in containers[i]:
            if j in allowed and (best < 0 or abs(infos[j].area) < abs(infos[best].area)):
                best = j
        return best

    role = [None] * n  # "outer" / "hole" / None (dropped)
    if fill_rule == "EVENODD":
        for i in range(n):
            role[i] = "outer" if len(containers[i]) % 2 == 0 else "hole"
        allowed_parents = {i for i in range(n) if role[i] == "outer"}
    else:
        orient = [1 if info.area > 0 else -1 for info in infos]
        for i in range(n):
            w_out = sum(orient[j] for j in containers[i])
            w_in = w_out + orient[i]
            if w_out == 0 and w_in != 0:
                role[i] = "outer"
            elif w_out != 0 and w_in == 0:
                role[i] = "hole"
        allowed_parents = {i for i in range(n) if role[i] == "outer"}

    shape_index = {}
    shapes: List[Tuple[Ring, List[Ring]]] = []
    for i in range(n):
        if role[i] == "outer":
            pts = _tolist(infos[i].pts)
            if infos[i].area < 0:
                pts.reverse()
            shape_index[i] = len(shapes)
            shapes.append((pts, []))
    for i in range(n):
        if role[i] == "hole":
            p = parent_of(i, allowed_parents)
            if p < 0:
                continue
            pts = _tolist(infos[i].pts)
            if infos[i].area > 0:
                pts.reverse()
            shapes[shape_index[p]][1].append(pts)
    return shapes


# ---------------------------------------------------------------------------
# intersections
# ---------------------------------------------------------------------------

def segment_intersection(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> Optional[Vec2]:
    """Proper intersection point of two segments, or None (touching endpoints excluded)."""
    _check_scalars(a0[0], a0[1], a1[0], a1[1], b0[0], b0[1], b1[0], b1[1])
    rx = a1[0] - a0[0]
    ry = a1[1] - a0[1]
    sx = b1[0] - b0[0]
    sy = b1[1] - b0[1]
    d1 = sx * (a0[1] - b0[1]) - sy * (a0[0] - b0[0])
    d2 = sx * (a1[1] - b0[1]) - sy * (a1[0] - b0[0])
    d3 = rx * (b0[1] - a0[1]) - ry * (b0[0] - a0[0])
    d4 = rx * (b1[1] - a0[1]) - ry * (b1[0] - a0[0])
    if not (d1 * d2 < 0.0 and d3 * d4 < 0.0):
        return None
    t = d1 / (d1 - d2)
    return (a0[0] + t * rx, a0[1] + t * ry)


_PAIR_CHUNK = 1_500_000


def _crossings(p0: np.ndarray, p1: np.ndarray, ring_id: np.ndarray, local: np.ndarray,
               ring_n: np.ndarray, ring_closed: np.ndarray, limit: int,
               same_ring: bool = True, cross_ring: bool = True):
    """Find proper crossings between segments p0[i]->p1[i].

    Uses a uniform grid with every pair tested once (in the lowest shared cell), so
    simple rings cost O(n) and the work is bounded for adversarial input by chunking.
    Returns a list of ``(i, j, x, y)`` with i < j, sorted.
    """
    m = len(p0)
    if m < 2:
        return []
    minx = np.minimum(p0[:, 0], p1[:, 0])
    maxx = np.maximum(p0[:, 0], p1[:, 0])
    miny = np.minimum(p0[:, 1], p1[:, 1])
    maxy = np.maximum(p0[:, 1], p1[:, 1])
    ox = float(minx.min())
    oy = float(miny.min())
    w = float(maxx.max()) - ox
    h = float(maxy.max()) - oy
    ext = max(w, h)
    if ext <= 0.0:
        return []
    g = int(min(512, max(1, math.sqrt(m))))
    while True:
        cs = ext / g
        ix0 = np.minimum(((minx - ox) / cs).astype(np.int64), g)
        ix1 = np.minimum(((maxx - ox) / cs).astype(np.int64), g)
        iy0 = np.minimum(((miny - oy) / cs).astype(np.int64), g)
        iy1 = np.minimum(((maxy - oy) / cs).astype(np.int64), g)
        nx = ix1 - ix0 + 1
        ny = iy1 - iy0 + 1
        counts = nx * ny
        total = int(counts.sum())
        if total <= 6_000_000 or g <= 1:
            break
        g = max(1, g // 2)
    stride = g + 1
    # Expand every segment into its cells.
    seg_ids = np.repeat(np.arange(m, dtype=np.int64), counts)
    start = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(start, counts)
    nxr = np.repeat(nx, counts)
    cx = np.repeat(ix0, counts) + within % nxr
    cy = np.repeat(iy0, counts) + within // nxr
    cell = cy * stride + cx
    order = np.argsort(cell, kind="stable")
    cell = cell[order]
    seg_ids = seg_ids[order]
    cx = cx[order]
    cy = cy[order]
    # group boundaries
    change = np.empty(total, dtype=bool)
    change[0] = True
    change[1:] = cell[1:] != cell[:-1]
    gstart = np.flatnonzero(change)
    gend = np.append(gstart[1:], total)
    group_of = np.repeat(np.arange(len(gstart)), gend - gstart)
    pos = np.arange(total, dtype=np.int64)
    partners = gend[group_of] - pos - 1  # entries after me in my cell
    hits: List[Tuple[int, int, float, float]] = []
    cum = np.cumsum(partners)
    lo = 0
    max_hits = max(limit * 8, 256)
    while lo < total:
        base = int(cum[lo - 1]) if lo > 0 else 0
        hi = int(np.searchsorted(cum, base + _PAIR_CHUNK, side="right"))
        hi = max(hi, lo + 1)
        hi = min(hi, total)
        cnt = partners[lo:hi]
        tot = int(cnt.sum())
        if tot > 0:
            first = np.repeat(np.arange(lo, hi, dtype=np.int64), cnt)
            offs = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            second = first + 1 + offs
            sa = seg_ids[first]
            sb = seg_ids[second]
            # test each pair only in the lowest shared cell
            lowx = np.maximum(ix0[sa], ix0[sb])
            lowy = np.maximum(iy0[sa], iy0[sb])
            keep = (cx[first] == lowx) & (cy[first] == lowy)
            sa = sa[keep]
            sb = sb[keep]
            hits.extend(_exact_pairs(sa, sb, p0, p1, minx, maxx, miny, maxy, ring_id, local,
                                     ring_n, ring_closed, same_ring, cross_ring))
            if len(hits) >= max_hits:
                break
        lo = hi
    hits.sort(key=lambda h: (h[0], h[1]))
    return hits


def _exact_pairs(sa, sb, p0, p1, minx, maxx, miny, maxy, ring_id, local, ring_n, ring_closed,
                 same_ring, cross_ring):
    if len(sa) == 0:
        return []
    swap = sa > sb
    a = np.where(swap, sb, sa)
    b = np.where(swap, sa, sb)
    ra = ring_id[a]
    rb = ring_id[b]
    same = ra == rb
    ok = np.where(same, same_ring, cross_ring)
    la = local[a]
    lb = local[b]
    nn = ring_n[ra]
    adjacent = same & ((np.abs(la - lb) == 1) | (ring_closed[ra] & (np.abs(la - lb) == nn - 1)))
    ok &= ~adjacent
    ok &= (minx[a] <= maxx[b]) & (minx[b] <= maxx[a]) & (miny[a] <= maxy[b]) & (miny[b] <= maxy[a])
    a = a[ok]
    b = b[ok]
    if len(a) == 0:
        return []
    a0 = p0[a]
    a1 = p1[a]
    b0 = p0[b]
    b1 = p1[b]
    rx = a1[:, 0] - a0[:, 0]
    ry = a1[:, 1] - a0[:, 1]
    sx = b1[:, 0] - b0[:, 0]
    sy = b1[:, 1] - b0[:, 1]
    d1 = sx * (a0[:, 1] - b0[:, 1]) - sy * (a0[:, 0] - b0[:, 0])
    d2 = sx * (a1[:, 1] - b0[:, 1]) - sy * (a1[:, 0] - b0[:, 0])
    d3 = rx * (b0[:, 1] - a0[:, 1]) - ry * (b0[:, 0] - a0[:, 0])
    d4 = rx * (b1[:, 1] - a0[:, 1]) - ry * (b1[:, 0] - a0[:, 0])
    proper = (d1 * d2 < 0.0) & (d3 * d4 < 0.0)
    if not proper.any():
        return []
    idx = np.flatnonzero(proper)
    t = d1[idx] / (d1[idx] - d2[idx])
    px = a0[idx, 0] + t * rx[idx]
    py = a0[idx, 1] + t * ry[idx]
    return [(int(i), int(j), float(x), float(y)) for i, j, x, y in zip(a[idx], b[idx], px, py)]


def _ring_edges(a: np.ndarray, closed: bool):
    n = len(a)
    if closed:
        return a, np.roll(a, -1, axis=0)
    return a[:-1], a[1:]


def self_intersections(ring: Sequence[Vec2], closed: bool = True, limit: int = 50) -> List[Tuple[int, int, Vec2]]:
    """Find up to ``limit`` proper intersections between non-adjacent edges.

    Must be O(n log n)-ish or bounded so a 5k-point ring answers in < 200 ms.
    Edge ``i`` runs from ``ring[i]`` to ``ring[(i + 1) % n]``; results are ``(i, j, point)`` with ``i < j``.
    """
    a = as_points(ring, "ring")
    if len(a) < 4 or limit <= 0:
        return []
    p0, p1 = _ring_edges(a, closed)
    m = len(p0)
    ring_id = np.zeros(m, dtype=np.int64)
    local = np.arange(m, dtype=np.int64)
    hits = _crossings(p0, p1, ring_id, local, np.array([m]), np.array([closed]), limit)
    return [(i, j, (x, y)) for i, j, x, y in hits[:limit]]


def validate_rings(rings: Sequence[Sequence[Vec2]], require_closed_area: bool = True) -> List[Issue]:
    """Report issues that make a filled extrusion invalid (self-intersection, zero area, < 3 pts)."""
    issues: List[Issue] = []
    arrays: List[Optional[np.ndarray]] = []
    for idx, r in enumerate(rings):
        a = as_points(r, "ring %d" % idx)
        # ignore a repeated closing point and consecutive duplicates when counting points
        keep = [0] if len(a) else []
        for k in range(1, len(a)):
            if not (a[k, 0] == a[keep[-1], 0] and a[k, 1] == a[keep[-1], 1]):
                keep.append(k)
        while len(keep) > 1 and a[keep[0], 0] == a[keep[-1], 0] and a[keep[0], 1] == a[keep[-1], 1]:
            keep.pop()
        a = a[keep] if keep else a
        min_pts = 3 if require_closed_area else 2
        if len(a) < min_pts:
            issues.append(Issue("TOO_FEW_POINTS", "ring needs at least %d distinct points, has %d" % (min_pts, len(a)), idx))
            arrays.append(None)
            continue
        if require_closed_area and abs(signed_area(a)) < 1e-12:
            issues.append(Issue("ZERO_AREA", "ring encloses no area", idx))
        arrays.append(a)
    valid = [(i, a) for i, a in enumerate(arrays) if a is not None and len(a) >= 3]
    if not valid:
        return issues
    p0s = []
    p1s = []
    rid = []
    loc = []
    ring_n = np.zeros(len(rings), dtype=np.int64)
    for i, a in valid:
        s, e = _ring_edges(a, True)
        p0s.append(s)
        p1s.append(e)
        rid.append(np.full(len(s), i, dtype=np.int64))
        loc.append(np.arange(len(s), dtype=np.int64))
        ring_n[i] = len(s)
    p0 = np.concatenate(p0s)
    p1 = np.concatenate(p1s)
    ring_id = np.concatenate(rid)
    local = np.concatenate(loc)
    closed = np.ones(len(rings), dtype=bool)
    hits = _crossings(p0, p1, ring_id, local, ring_n, closed, limit=1000)
    reported = set()
    for ea, eb, x, y in hits:
        ra = int(ring_id[ea])
        rb = int(ring_id[eb])
        key = (ra, rb)
        if key in reported:
            continue
        reported.add(key)
        if ra == rb:
            issues.append(Issue("SELF_INTERSECTION", "ring crosses itself", ra, (x, y)))
        else:
            issues.append(Issue("SELF_INTERSECTION", "rings %d and %d intersect each other" % (ra, rb), ra, (x, y)))
    return issues


# ---------------------------------------------------------------------------
# triangulation (port of the earcut algorithm with z-order hashing)
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("i", "x", "y", "prev", "next", "z", "prevz", "nextz", "steiner")

    def __init__(self, i: int, x: float, y: float):
        self.i = i
        self.x = x
        self.y = y
        self.prev = None
        self.next = None
        self.z = 0
        self.prevz = None
        self.nextz = None
        self.steiner = False


def _insert_node(i, x, y, last):
    p = _Node(i, x, y)
    if last is None:
        p.prev = p
        p.next = p
    else:
        p.next = last.next
        p.prev = last
        last.next.prev = p
        last.next = p
    return p


def _remove_node(p):
    p.next.prev = p.prev
    p.prev.next = p.next
    if p.prevz is not None:
        p.prevz.nextz = p.nextz
    if p.nextz is not None:
        p.nextz.prevz = p.prevz


def _area(p, q, r):
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)


def _equals(p1, p2):
    return p1.x == p2.x and p1.y == p2.y


def _linked_list(items, ccw: bool):
    """Build a circular list from ``[(index, x, y)]`` in the requested orientation."""
    area = 0.0
    n = len(items)
    for k in range(n):
        _, x1, y1 = items[k - 1]
        _, x2, y2 = items[k]
        area += (x1 - x2) * (y1 + y2)  # >0 when counter-clockwise (y up)
    last = None
    if ccw == (area > 0):
        for i, x, y in items:
            last = _insert_node(i, x, y, last)
    else:
        for i, x, y in reversed(items):
            last = _insert_node(i, x, y, last)
    if last is not None and _equals(last, last.next):
        _remove_node(last)
        last = last.next
    return last


def _filter_points(start, end=None):
    if start is None:
        return start
    if end is None:
        end = start
    p = start
    while True:
        again = False
        if not p.steiner and (_equals(p, p.next) or _area(p.prev, p, p.next) == 0):
            _remove_node(p)
            p = end = p.prev
            if p is p.next:
                break
            again = True
        else:
            p = p.next
        if not again and p is end:
            break
    return end


def _z_order(x, y, min_x, min_y, inv_size):
    x = int((x - min_x) * inv_size)
    y = int((y - min_y) * inv_size)
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    y = (y | (y << 8)) & 0x00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F
    y = (y | (y << 2)) & 0x33333333
    y = (y | (y << 1)) & 0x55555555
    return x | (y << 1)


def _index_curve(start, min_x, min_y, inv_size):
    nodes = []
    p = start
    while True:
        if p.z == 0:
            p.z = _z_order(p.x, p.y, min_x, min_y, inv_size)
        nodes.append(p)
        p = p.next
        if p is start:
            break
    nodes.sort(key=lambda node: node.z)
    prev = None
    for node in nodes:
        node.prevz = prev
        if prev is not None:
            prev.nextz = node
        prev = node
    prev.nextz = None


def _point_in_triangle(ax, ay, bx, by, cx, cy, px, py):
    return ((cx - px) * (ay - py) >= (ax - px) * (cy - py)
            and (ax - px) * (by - py) >= (bx - px) * (ay - py)
            and (bx - px) * (cy - py) >= (cx - px) * (by - py))


def _is_ear(ear):
    a = ear.prev
    b = ear
    c = ear.next
    if _area(a, b, c) >= 0:
        return False
    ax, ay, bx, by, cx, cy = a.x, a.y, b.x, b.y, c.x, c.y
    x0 = min(ax, bx, cx)
    y0 = min(ay, by, cy)
    x1 = max(ax, bx, cx)
    y1 = max(ay, by, cy)
    p = c.next
    while p is not a:
        if (x0 <= p.x <= x1 and y0 <= p.y <= y1
                and _point_in_triangle(ax, ay, bx, by, cx, cy, p.x, p.y)
                and _area(p.prev, p, p.next) >= 0):
            return False
        p = p.next
    return True


def _is_ear_hashed(ear, min_x, min_y, inv_size):
    a = ear.prev
    b = ear
    c = ear.next
    if _area(a, b, c) >= 0:
        return False
    ax, ay, bx, by, cx, cy = a.x, a.y, b.x, b.y, c.x, c.y
    x0 = min(ax, bx, cx)
    y0 = min(ay, by, cy)
    x1 = max(ax, bx, cx)
    y1 = max(ay, by, cy)
    min_z = _z_order(x0, y0, min_x, min_y, inv_size)
    max_z = _z_order(x1, y1, min_x, min_y, inv_size)
    p = ear.prevz
    n = ear.nextz
    while p is not None and p.z >= min_z and n is not None and n.z <= max_z:
        if (p is not a and p is not c and x0 <= p.x <= x1 and y0 <= p.y <= y1
                and _point_in_triangle(ax, ay, bx, by, cx, cy, p.x, p.y)
                and _area(p.prev, p, p.next) >= 0):
            return False
        p = p.prevz
        if (n is not a and n is not c and x0 <= n.x <= x1 and y0 <= n.y <= y1
                and _point_in_triangle(ax, ay, bx, by, cx, cy, n.x, n.y)
                and _area(n.prev, n, n.next) >= 0):
            return False
        n = n.nextz
    while p is not None and p.z >= min_z:
        if (p is not a and p is not c and x0 <= p.x <= x1 and y0 <= p.y <= y1
                and _point_in_triangle(ax, ay, bx, by, cx, cy, p.x, p.y)
                and _area(p.prev, p, p.next) >= 0):
            return False
        p = p.prevz
    while n is not None and n.z <= max_z:
        if (n is not a and n is not c and x0 <= n.x <= x1 and y0 <= n.y <= y1
                and _point_in_triangle(ax, ay, bx, by, cx, cy, n.x, n.y)
                and _area(n.prev, n, n.next) >= 0):
            return False
        n = n.nextz
    return True


def _sign(v):
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _on_segment(p, q, r):
    return (min(p.x, r.x) <= q.x <= max(p.x, r.x)) and (min(p.y, r.y) <= q.y <= max(p.y, r.y))


def _intersects(p1, q1, p2, q2):
    o1 = _sign(_area(p1, q1, p2))
    o2 = _sign(_area(p1, q1, q2))
    o3 = _sign(_area(p2, q2, p1))
    o4 = _sign(_area(p2, q2, q1))
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False


def _intersects_polygon(a, b):
    p = a
    while True:
        if (p.i != a.i and p.next.i != a.i and p.i != b.i and p.next.i != b.i
                and _intersects(p, p.next, a, b)):
            return True
        p = p.next
        if p is a:
            break
    return False


def _locally_inside(a, b):
    if _area(a.prev, a, a.next) < 0:
        return _area(a, b, a.next) >= 0 and _area(a, a.prev, b) >= 0
    return _area(a, b, a.prev) < 0 or _area(a, a.next, b) < 0


def _middle_inside(a, b):
    p = a
    inside = False
    px = (a.x + b.x) / 2
    py = (a.y + b.y) / 2
    while True:
        if ((p.y > py) != (p.next.y > py)) and p.next.y != p.y and \
                (px < (p.next.x - p.x) * (py - p.y) / (p.next.y - p.y) + p.x):
            inside = not inside
        p = p.next
        if p is a:
            break
    return inside


def _is_valid_diagonal(a, b):
    return ((a.next.i != b.i and a.prev.i != b.i and not _intersects_polygon(a, b)
             and ((_locally_inside(a, b) and _locally_inside(b, a) and _middle_inside(a, b)
                   and (_area(a.prev, a, b.prev) != 0 or _area(a, b.prev, b) != 0))
                  or (_equals(a, b) and _area(a.prev, a, a.next) > 0 and _area(b.prev, b, b.next) > 0))))


def _split_polygon(a, b):
    a2 = _Node(a.i, a.x, a.y)
    b2 = _Node(b.i, b.x, b.y)
    an = a.next
    bp = b.prev
    a.next = b
    b.prev = a
    a2.next = an
    an.prev = a2
    b2.next = a2
    a2.prev = b2
    bp.next = b2
    b2.prev = bp
    return b2


def _cure_local_intersections(start, triangles):
    p = start
    while True:
        a = p.prev
        b = p.next.next
        if (not _equals(a, b) and _intersects(a, p, p.next, b)
                and _locally_inside(a, b) and _locally_inside(b, a)):
            triangles.append((a.i, p.i, b.i))
            _remove_node(p)
            _remove_node(p.next)
            p = start = b
        p = p.next
        if p is start:
            break
    return _filter_points(p)


def _split_earcut(start, triangles, min_x, min_y, inv_size):
    a = start
    while True:
        b = a.next.next
        while b is not a.prev:
            if a.i != b.i and _is_valid_diagonal(a, b):
                c = _split_polygon(a, b)
                a = _filter_points(a, a.next)
                c = _filter_points(c, c.next)
                _earcut_linked(a, triangles, min_x, min_y, inv_size, 0)
                _earcut_linked(c, triangles, min_x, min_y, inv_size, 0)
                return
            b = b.next
        a = a.next
        if a is start:
            break


def _earcut_linked(ear, triangles, min_x, min_y, inv_size, pass_=0):
    if ear is None:
        return
    if not pass_ and inv_size:
        _index_curve(ear, min_x, min_y, inv_size)
    stop = ear
    while ear.prev is not ear.next:
        prev = ear.prev
        nxt = ear.next
        if _is_ear_hashed(ear, min_x, min_y, inv_size) if inv_size else _is_ear(ear):
            triangles.append((prev.i, ear.i, nxt.i))
            _remove_node(ear)
            ear = nxt.next
            stop = nxt.next
            continue
        ear = nxt
        if ear is stop:
            if not pass_:
                _earcut_linked(_filter_points(ear), triangles, min_x, min_y, inv_size, 1)
            elif pass_ == 1:
                ear = _cure_local_intersections(_filter_points(ear), triangles)
                _earcut_linked(ear, triangles, min_x, min_y, inv_size, 2)
            elif pass_ == 2:
                _split_earcut(ear, triangles, min_x, min_y, inv_size)
            break


def _get_leftmost(start):
    p = start
    leftmost = start
    while True:
        if p.x < leftmost.x or (p.x == leftmost.x and p.y < leftmost.y):
            leftmost = p
        p = p.next
        if p is start:
            break
    return leftmost


def _sector_contains_sector(m, p):
    return _area(m.prev, m, p.prev) < 0 and _area(p.next, m, m.next) < 0


def _find_hole_bridge(hole, outer_node):
    p = outer_node
    hx = hole.x
    hy = hole.y
    qx = -math.inf
    m = None
    while True:
        if hy <= p.y and hy >= p.next.y and p.next.y != p.y:
            x = p.x + (hy - p.y) * (p.next.x - p.x) / (p.next.y - p.y)
            if x <= hx and x > qx:
                qx = x
                m = p if p.x < p.next.x else p.next
                if x == hx:
                    return m
        p = p.next
        if p is outer_node:
            break
    if m is None:
        return None
    stop = m
    mx = m.x
    my = m.y
    tan_min = math.inf
    p = m
    while True:
        if hx >= p.x and p.x >= mx and hx != p.x and _point_in_triangle(
                hx if hy < my else qx, hy, mx, my, qx if hy < my else hx, hy, p.x, p.y):
            tan = abs(hy - p.y) / (hx - p.x)
            if _locally_inside(p, hole) and (
                    tan < tan_min or (tan == tan_min and (p.x > m.x or (p.x == m.x and _sector_contains_sector(m, p))))):
                m = p
                tan_min = tan
        p = p.next
        if p is stop:
            break
    return m


def _eliminate_hole(hole, outer_node):
    bridge = _find_hole_bridge(hole, outer_node)
    if bridge is None:
        return outer_node
    bridge_reverse = _split_polygon(bridge, hole)
    _filter_points(bridge_reverse, bridge_reverse.next)
    return _filter_points(bridge, bridge.next)


def _earcut(outer_items, hole_items):
    """``outer_items``/``hole_items[k]`` are ``[(index, x, y)]``; returns index triples."""
    outer_node = _linked_list(outer_items, True)
    triangles: List[Tuple[int, int, int]] = []
    if outer_node is None or outer_node.next is outer_node.prev:
        return triangles
    if hole_items:
        queue = []
        for items in hole_items:
            if len(items) < 1:
                continue
            lst = _linked_list(items, False)
            if lst is None:
                continue
            if lst is lst.next:
                lst.steiner = True
            queue.append(_get_leftmost(lst))
        queue.sort(key=lambda node: node.x)
        for node in queue:
            outer_node = _eliminate_hole(node, outer_node)
    min_x = min_y = inv_size = 0.0
    if len(outer_items) + sum(len(h) for h in hole_items) > 80:
        xs = [it[1] for it in outer_items]
        ys = [it[2] for it in outer_items]
        min_x = min(xs)
        min_y = min(ys)
        inv_size = max(max(xs) - min_x, max(ys) - min_y)
        inv_size = 32767.0 / inv_size if inv_size != 0 else 0.0
    _earcut_linked(outer_node, triangles, min_x, min_y, inv_size)
    return triangles


def _repair_t_junctions(order_rings: List[List[int]], tris: List[Tuple[int, int, int]],
                        pts: List[Vec2]) -> List[Tuple[int, int, int]]:
    """Earcut drops exactly collinear ring vertices; re-insert them so ring edges stay conforming.

    ``order_rings`` are index lists in material-on-the-left order (outer CCW, holes CW).
    """
    used = set()
    for t in tris:
        used.update(t)
    total = sum(len(r) for r in order_rings)
    if len(used) >= total:
        return tris
    edge_tri = {}
    for k, (a, b, c) in enumerate(tris):
        edge_tri[(a, b)] = (k, c)
        edge_tri[(b, c)] = (k, a)
        edge_tri[(c, a)] = (k, b)
    replaced = {}
    extra: List[Tuple[int, int, int]] = []
    for ring in order_rings:
        n = len(ring)
        if n < 3:
            continue
        # start at a used vertex
        s = next((k for k in range(n) if ring[k] in used), None)
        if s is None:
            continue
        k = 0
        while k < n:
            u = ring[(s + k) % n]
            j = k + 1
            run = []
            while j <= n and ring[(s + j) % n] not in used:
                run.append(ring[(s + j) % n])
                j += 1
            if run:
                w = ring[(s + j) % n]
                hit = edge_tri.get((u, w))
                if hit is not None and hit[0] not in replaced:
                    tri_k, apex = hit
                    chain = [u] + run + [w]
                    # skip coincident points (they only create zero-area fans)
                    clean = [chain[0]]
                    for v in chain[1:]:
                        if pts[v] != pts[clean[-1]]:
                            clean.append(v)
                    if len(clean) > 2:
                        replaced[tri_k] = True
                        for q in range(len(clean) - 1):
                            extra.append((clean[q], clean[q + 1], apex))
            k = max(j, k + 1)
    if not replaced:
        return tris
    return [t for k, t in enumerate(tris) if k not in replaced] + extra


def triangulate(outer: Sequence[Vec2], holes: Sequence[Sequence[Vec2]] = ()) -> Tuple[List[Vec2], List[Tuple[int, int, int]]]:
    """Triangulate a polygon with holes (ear clipping with hole bridging, earcut-style).

    Returns (vertices, triangles); vertices are outer points followed by each hole's points
    in input order; triangles are CCW index triples. Must handle collinear points,
    duplicate points and touching holes without crashing.
    """
    o = as_points(outer, "outer")
    hs = [as_points(h, "hole %d" % k) for k, h in enumerate(holes)]
    verts: List[Vec2] = _tolist(o)
    offsets = []
    for h in hs:
        offsets.append(len(verts))
        verts.extend(_tolist(h))
    if len(o) < 3:
        return verts, []
    outer_items = [(k, verts[k][0], verts[k][1]) for k in range(len(o))]
    hole_items = []
    for h, off in zip(hs, offsets):
        if len(h) >= 3:
            hole_items.append([(off + k, verts[off + k][0], verts[off + k][1]) for k in range(len(h))])
    tris = _earcut(outer_items, hole_items)
    # material-on-the-left index order for each ring, used to re-insert dropped collinear points
    def _ordered(items, ccw):
        area = 0.0
        for k in range(len(items)):
            _, x1, y1 = items[k - 1]
            _, x2, y2 = items[k]
            area += (x1 - x2) * (y1 + y2)
        idx = [it[0] for it in items]
        return idx if ccw == (area > 0) else idx[::-1]
    rings = [_ordered(outer_items, True)] + [_ordered(it, False) for it in hole_items]
    tris = _repair_t_junctions(rings, tris, verts)
    return verts, tris
