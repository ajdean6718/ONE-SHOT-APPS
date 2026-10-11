"""Freehand stroke smoothing and Bezier fitting (owner: core-geometry)."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .geometry import BezierPath, BezierPoint, Vec2, as_points, _check_scalars, _tolist

CORNER_ANGLE = math.radians(70.0)


def _dedupe(points: Sequence[Vec2], closed: bool = False) -> np.ndarray:
    a = as_points(points, "points")
    if len(a) < 2:
        return a
    keep = np.ones(len(a), dtype=bool)
    keep[1:] = np.any(a[1:] != a[:-1], axis=1)
    a = a[keep]
    if closed and len(a) > 1 and np.all(a[0] == a[-1]):
        a = a[:-1]
    return a


def resample(points: Sequence[Vec2], spacing: float) -> List[Vec2]:
    """Resample a polyline at roughly uniform arc-length ``spacing`` (keeps endpoints).

    The step is adjusted slightly so that an integer number of equal steps spans the path.
    """
    _check_scalars(spacing)
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0, got %r" % spacing)
    a = _dedupe(points)
    if len(a) == 0:
        return []
    if len(a) == 1:
        return _tolist(a)
    seg = np.hypot(*(np.diff(a, axis=0).T))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    count = max(1, int(round(total / spacing)))
    targets = np.linspace(0.0, total, count + 1)
    x = np.interp(targets, cum, a[:, 0])
    y = np.interp(targets, cum, a[:, 1])
    out = np.stack([x, y], axis=1)
    out[0] = a[0]
    out[-1] = a[-1]
    return _tolist(out)


def smooth_polyline(points: Sequence[Vec2], strength: float = 0.5, iterations: int = 2, closed: bool = False) -> List[Vec2]:
    """Laplacian/Chaikin-style smoothing; strength 0..1; endpoints fixed when open.

    Each iteration moves every free point ``strength`` of the way towards the midpoint of its two
    neighbours. The point count is preserved.
    """
    _check_scalars(strength)
    a = as_points(points, "points").copy()
    s = min(1.0, max(0.0, float(strength)))
    n = len(a)
    if n < 3 or s == 0.0 or iterations <= 0:
        return _tolist(a)
    for _ in range(int(iterations)):
        if closed:
            mid = (np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0)) * 0.5
            a = a + s * (mid - a)
        else:
            mid = (a[:-2] + a[2:]) * 0.5
            a[1:-1] = a[1:-1] + s * (mid - a[1:-1])
    return _tolist(a)


def simplify(points: Sequence[Vec2], tolerance: float) -> List[Vec2]:
    """Ramer–Douglas–Peucker."""
    _check_scalars(tolerance)
    a = as_points(points, "points")
    n = len(a)
    if n < 3 or tolerance <= 0.0:
        return _tolist(a)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        p0 = a[lo]
        p1 = a[hi]
        seg = a[lo + 1:hi]
        d = p1 - p0
        length = math.hypot(d[0], d[1])
        if length == 0.0:
            dist = np.hypot(seg[:, 0] - p0[0], seg[:, 1] - p0[1])
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p0[1]) - d[1] * (seg[:, 0] - p0[0])) / length
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            mid = lo + 1 + k
            keep[mid] = True
            stack.append((lo, mid))
            stack.append((mid, hi))
    return _tolist(a[keep])


# ---------------------------------------------------------------------------
# Schneider curve fitting
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    if n == 0.0:
        return np.zeros(2)
    return v / n


def _bez(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    u = 1.0 - t
    return ((u ** 3)[:, None] * ctrl[0] + (3 * u * u * t)[:, None] * ctrl[1]
            + (3 * u * t * t)[:, None] * ctrl[2] + (t ** 3)[:, None] * ctrl[3])


def _chord_params(d: np.ndarray) -> np.ndarray:
    seg = np.hypot(*(np.diff(d, axis=0).T))
    u = np.concatenate([[0.0], np.cumsum(seg)])
    total = u[-1]
    return u / total if total > 0 else np.linspace(0.0, 1.0, len(d))


def _generate(d: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    p0 = d[0]
    p3 = d[-1]
    b0 = (1 - u) ** 3
    b1 = 3 * u * (1 - u) ** 2
    b2 = 3 * u * u * (1 - u)
    b3 = u ** 3
    a1 = t1[None, :] * b1[:, None]
    a2 = t2[None, :] * b2[:, None]
    c00 = float((a1 * a1).sum())
    c01 = float((a1 * a2).sum())
    c11 = float((a2 * a2).sum())
    tmp = d - (p0[None, :] * (b0 + b1)[:, None] + p3[None, :] * (b2 + b3)[:, None])
    x0 = float((a1 * tmp).sum())
    x1 = float((a2 * tmp).sum())
    det = c00 * c11 - c01 * c01
    seg_len = float(np.hypot(*(p3 - p0)))
    alpha_l = alpha_r = seg_len / 3.0
    if abs(det) > 1e-12 * max(c00 * c11, 1e-300):
        al = (x0 * c11 - x1 * c01) / det
        ar = (c00 * x1 - c01 * x0) / det
        eps = 1e-6 * seg_len
        if al >= eps and ar >= eps and math.isfinite(al) and math.isfinite(ar):
            alpha_l, alpha_r = al, ar
    return np.array([p0, p0 + t1 * alpha_l, p3 + t2 * alpha_r, p3])


def _max_error(d: np.ndarray, ctrl: np.ndarray, u: np.ndarray) -> Tuple[float, int]:
    diff = _bez(ctrl, u) - d
    dist2 = (diff * diff).sum(axis=1)
    inner = dist2[1:-1]
    if len(inner) == 0:
        return 0.0, len(d) // 2
    k = int(np.argmax(inner)) + 1
    return float(dist2[k]), k


def _reparameterize(d: np.ndarray, u: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    q = _bez(ctrl, u)
    t = u
    om = 1.0 - t
    d1 = 3 * (om * om)[:, None] * (ctrl[1] - ctrl[0]) + 6 * (om * t)[:, None] * (ctrl[2] - ctrl[1]) + 3 * (t * t)[:, None] * (ctrl[3] - ctrl[2])
    d2 = 6 * om[:, None] * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t[:, None] * (ctrl[3] - 2 * ctrl[2] + ctrl[1])
    diff = q - d
    num = (diff * d1).sum(axis=1)
    den = (d1 * d1).sum(axis=1) + (diff * d2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(np.abs(den) > 1e-12, num / den, 0.0)
    return np.clip(u - step, 0.0, 1.0)


def _fit_chain(d: np.ndarray, t1: np.ndarray, t2: np.ndarray, tol: float, dir_pts: np.ndarray, offset: int) -> List[np.ndarray]:
    """Fit the points ``d`` with cubics; returns control point arrays (4x2) from first to last.

    ``dir_pts``/``offset`` allow centre tangents at interior splits: ``dir_pts[offset + k]`` is ``d[k]``.
    """
    tol2 = tol * tol
    out: List[np.ndarray] = []
    stack = [(0, len(d) - 1, t1, t2)]
    while stack:
        lo, hi, ta, tb = stack.pop()
        pts = d[lo:hi + 1]
        if len(pts) == 2:
            dist = float(np.hypot(*(pts[1] - pts[0])))
            out.append(np.array([pts[0], pts[0] + ta * dist / 3.0, pts[1] + tb * dist / 3.0, pts[1]]))
            continue
        u = _chord_params(pts)
        ctrl = _generate(pts, u, ta, tb)
        err, split = _max_error(pts, ctrl, u)
        if err < tol2:
            out.append(ctrl)
            continue
        if err < 4.0 * tol2 * 4.0:  # worth trying Newton refinement
            for _ in range(8):
                u = _reparameterize(pts, u, ctrl)
                ctrl = _generate(pts, u, ta, tb)
                err, split = _max_error(pts, ctrl, u)
                if err < tol2:
                    break
            if err < tol2:
                out.append(ctrl)
                continue
        split = min(max(split, 1), len(pts) - 2)
        mid = lo + split
        tc = _unit(d[mid - 1] - d[mid + 1])
        if not tc.any():
            tc = _unit(d[mid] - d[mid + 1])
        stack.append((mid, hi, -tc, tb))
        stack.append((lo, mid, ta, tc))
    return out


def _chord_point_index(a: np.ndarray, i: int, step: int, radius: float, closed: bool) -> int:
    """Index reached from ``i`` walking by ``step`` until the chord from ``a[i]`` is >= radius long."""
    n = len(a)
    j = i
    limit = n // 2 if closed else (n - 1 - i if step > 0 else i)
    for _ in range(max(limit, 1)):
        nj = j + step
        if closed:
            nj %= n
        elif nj < 0 or nj >= n:
            break
        j = nj
        if math.hypot(a[j, 0] - a[i, 0], a[j, 1] - a[i, 1]) >= radius:
            break
    return j


def _find_corners(a: np.ndarray, radius: float, closed: bool) -> List[int]:
    n = len(a)
    idxs = range(n) if closed else range(1, n - 1)
    turn = np.zeros(n)
    for i in idxs:
        jb = _chord_point_index(a, i, -1, radius, closed)
        jf = _chord_point_index(a, i, 1, radius, closed)
        if jb == i or jf == i:
            continue
        v1 = a[i] - a[jb]
        v2 = a[jf] - a[i]
        n1 = float(np.hypot(*v1))
        n2 = float(np.hypot(*v2))
        if n1 == 0.0 or n2 == 0.0:
            continue
        c = float(np.dot(v1, v2) / (n1 * n2))
        turn[i] = math.acos(max(-1.0, min(1.0, c)))
    flagged = turn > CORNER_ANGLE
    order = list(idxs)
    if not order or not flagged[order].any():
        return []
    if closed:
        if flagged[order].all() and len(order) <= 2:
            return [max(order, key=lambda k: turn[k])]
        # start the walk just after an unflagged sample so runs never wrap around the end
        first_clear = next((p for p, k in enumerate(order) if not flagged[k]), 0)
        order = order[first_clear:] + order[:first_clear]
    # Flagged samples belong to one corner when they are closer than one window of arc length
    # to each other; keep the sharpest of each group (dense freehand data flags several
    # neighbouring samples around a single corner, sparse polylines flag each vertex separately).
    gap = 2.0 * radius
    corners: List[int] = []
    group: List[int] = []
    last = None
    for k in order:
        if not flagged[k]:
            continue
        if last is not None and math.hypot(*(a[k] - a[last])) > gap and _arc_between(a, last, k, closed) > gap:
            corners.append(max(group, key=lambda q: turn[q]))
            group = []
        group.append(k)
        last = k
    if group:
        corners.append(max(group, key=lambda q: turn[q]))
    return sorted(corners)


def _arc_between(a: np.ndarray, i: int, j: int, closed: bool) -> float:
    """Arc length walking forward from index ``i`` to ``j`` (wrapping when closed)."""
    n = len(a)
    total = 0.0
    k = i
    while k != j:
        nk = (k + 1) % n if closed else k + 1
        if nk >= n:
            break
        total += math.hypot(a[nk, 0] - a[k, 0], a[nk, 1] - a[k, 1])
        k = nk
        if total > 1e18:
            break
    return total


def _subdivide(ctrl: np.ndarray, t: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    p0, p1, p2, p3 = ctrl
    p01 = p0 + (p1 - p0) * t
    p12 = p1 + (p2 - p1) * t
    p23 = p2 + (p3 - p2) * t
    p012 = p01 + (p12 - p01) * t
    p123 = p12 + (p23 - p12) * t
    m = p012 + (p123 - p012) * t
    return np.array([p0, p01, p012, m]), np.array([m, p123, p23, p3])


def fit_bezier(points: Sequence[Vec2], max_error: float = 1.0, closed: bool = False) -> BezierPath:
    """Schneider's algorithm: fit a piecewise cubic Bezier through ``points``.

    Corners (turn angle > 70°) split segments and get FREE handles; smooth joins get ALIGNED.
    Returns a path with >= 2 points (or a closed path with >= 3).

    The turn angle is measured over a window of about ``2 * max_error`` chord length so that noisy
    freehand samples do not create spurious corners. Every input sample ends up within
    ``max_error`` of the fitted curve (measured at its fitted parameter, hence conservative).
    """
    _check_scalars(max_error)
    if max_error <= 0.0:
        raise ValueError("max_error must be > 0, got %r" % max_error)
    a = _dedupe(points, closed)
    n = len(a)
    if n == 0:
        raise ValueError("fit_bezier needs at least one point")

    def pt(p) -> Vec2:
        return (float(p[0]), float(p[1]))

    if n == 1:
        p = pt(a[0])
        return BezierPath([BezierPoint(p, p, p, "FREE", "FREE"), BezierPoint(p, p, p, "FREE", "FREE")], False)
    if closed and n == 2:
        mid = (a[0] + a[1]) / 2.0
        a = np.array([a[0], mid, a[1]])
        n = 3
    radius = max(max_error, 1e-12)
    corners = _find_corners(a, 2.0 * radius, closed) if n >= 3 else []

    chains: List[Tuple[np.ndarray, bool, bool]] = []  # (points, start_is_corner, end_is_corner)
    if not closed:
        cuts = [0] + corners + [n - 1]
        for k in range(len(cuts) - 1):
            chains.append((a[cuts[k]:cuts[k + 1] + 1], k > 0, k < len(cuts) - 2))
    elif corners:
        corners = sorted(corners)
        for k in range(len(corners)):
            s = corners[k]
            e = corners[(k + 1) % len(corners)]
            if e > s:
                seg = a[s:e + 1]
            else:
                seg = np.concatenate([a[s:], a[:e + 1]])
            chains.append((seg, True, True))
    else:
        chains.append((np.concatenate([a, a[:1]]), False, False))

    def end_tangent(seg: np.ndarray, at_start: bool) -> np.ndarray:
        if at_start:
            j = 1
            while j < len(seg) - 1 and math.hypot(*(seg[j] - seg[0])) < radius:
                j += 1
            return _unit(seg[j] - seg[0])
        j = len(seg) - 2
        while j > 0 and math.hypot(*(seg[j] - seg[-1])) < radius:
            j -= 1
        return _unit(seg[j] - seg[-1])

    cubics: List[np.ndarray] = []
    seg_start_corner: List[bool] = []  # whether cubic k starts at a corner
    for seg, s_corner, e_corner in chains:
        if len(seg) == 1:
            continue
        if not closed or corners:
            t1 = end_tangent(seg, True)
            t2 = end_tangent(seg, False)
        else:
            # smooth closed curve: centre tangent at the shared start/end point
            jb = _chord_point_index(a, 0, -1, radius, True)
            jf = _chord_point_index(a, 0, 1, radius, True)
            tc = _unit(a[jf] - a[jb])
            t1 = tc
            t2 = -tc
        fitted = _fit_chain(seg, t1, t2, max_error, seg, 0)
        for k, c in enumerate(fitted):
            cubics.append(c)
            seg_start_corner.append(k == 0 and s_corner)
    if closed:
        while len(cubics) < 3:
            lens = [float(np.hypot(*(c[3] - c[0]))) for c in cubics]
            k = int(np.argmax(lens))
            c1, c2 = _subdivide(cubics[k])
            cubics[k:k + 1] = [c1, c2]
            seg_start_corner[k:k + 1] = [seg_start_corner[k], False]
    m = len(cubics)
    points_out: List[BezierPoint] = []
    for k in range(m):
        c = cubics[k]
        co = c[0]
        hr = c[1]
        if closed or k > 0:
            hl = cubics[(k - 1) % m][2]
        else:
            hl = 2 * co - hr
        corner = seg_start_corner[k]
        t = "FREE" if corner else "ALIGNED"
        points_out.append(BezierPoint(pt(co), pt(hl), pt(hr), t, t))
    if not closed:
        last = cubics[-1]
        co = last[3]
        hl = last[2]
        points_out.append(BezierPoint(pt(co), pt(hl), pt(2 * co - hl), "ALIGNED", "ALIGNED"))
    return BezierPath(points_out, closed)
