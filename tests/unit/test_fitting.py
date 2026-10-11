"""Unit tests for seam_core.fitting."""

import math
import random

import numpy as np
import pytest

from seam_core import fitting as f
from seam_core import geometry as g


def dist_to_polyline(p, poly, closed=False):
    a = np.asarray(poly, dtype=float)
    b = np.roll(a, -1, axis=0) if closed else a[1:]
    a = a if closed else a[:-1]
    d = b - a
    l2 = (d * d).sum(1)
    l2[l2 == 0] = 1.0
    t = np.clip(((np.asarray(p) - a) * d).sum(1) / l2, 0, 1)
    q = a + t[:, None] * d
    return float(np.hypot(*(q - np.asarray(p)).T).min())


def max_fit_error(points, path, closed=False, res=32):
    poly = g.flatten_path(path, res)
    if closed:
        return max(dist_to_polyline(p, poly, True) for p in points)
    return max(dist_to_polyline(p, poly) for p in points)


# ---------------------------------------------------------------- resample / smooth / simplify

def test_resample_uniform_spacing_keeps_endpoints():
    pts = [(0, 0), (10, 0), (10, 10)]
    out = f.resample(pts, 2.0)
    assert out[0] == (0.0, 0.0) and out[-1] == (10.0, 10.0)
    assert len(out) == 11
    gaps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(out, out[1:])]
    # euclidean gaps are <= the arc step (corner cut) and equal for the straight parts
    assert max(gaps) == pytest.approx(2.0)
    assert out[5] == pytest.approx((10.0, 0.0))
    out2 = f.resample([(0, 0), (7, 0)], 2.0)
    assert out2[0] == (0.0, 0.0) and out2[-1] == (7.0, 0.0)
    assert [round(b[0] - a[0], 9) for a, b in zip(out2, out2[1:])] == pytest.approx([7 / 4] * 4, abs=1e-9)


def test_resample_edge_cases():
    assert f.resample([], 1.0) == []
    assert f.resample([(3, 4)], 1.0) == [(3.0, 4.0)]
    assert f.resample([(3, 4), (3, 4)], 1.0) == [(3.0, 4.0)]
    assert f.resample([(0, 0), (0.1, 0)], 5.0) == [(0.0, 0.0), (0.1, 0.0)]
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            f.resample([(0, 0), (1, 1)], bad)


def test_smooth_polyline_open_keeps_endpoints_and_reduces_noise():
    rnd = random.Random(5)
    pts = [(float(i), rnd.uniform(-1, 1)) for i in range(50)]
    out = f.smooth_polyline(pts, strength=0.5, iterations=4)
    assert len(out) == len(pts) and out[0] == pts[0] and out[-1] == pts[-1]
    rough = sum(abs(pts[i + 1][1] - pts[i][1]) for i in range(49))
    smooth = sum(abs(out[i + 1][1] - out[i][1]) for i in range(49))
    assert smooth < rough * 0.6
    assert f.smooth_polyline(pts, strength=0.0) == [(float(x), float(y)) for x, y in pts]
    assert f.smooth_polyline(pts, iterations=0) == [(float(x), float(y)) for x, y in pts]
    assert f.smooth_polyline([(0, 0), (1, 1)]) == [(0.0, 0.0), (1.0, 1.0)]


def test_smooth_polyline_closed_shrinks_uniformly_and_wraps():
    ring = [(10 * math.cos(2 * math.pi * i / 24), 10 * math.sin(2 * math.pi * i / 24)) for i in range(24)]
    out = f.smooth_polyline(ring, strength=1.0, iterations=1, closed=True)
    assert len(out) == 24
    radii = [math.hypot(x, y) for x, y in out]
    assert max(radii) - min(radii) < 1e-9 and radii[0] < 10.0
    assert out[0] != ring[0]  # closed: the first point moves too
    # strength is clamped to 0..1
    assert f.smooth_polyline(ring, strength=5.0, iterations=1, closed=True) == pytest.approx(out)


def test_simplify_rdp():
    pts = [(0, 0), (1, 0.05), (2, -0.05), (3, 0.0), (4, 5), (5, 0.0), (6, 0.02), (7, 0)]
    out = f.simplify(pts, 0.5)
    assert out == [(0.0, 0.0), (3.0, 0.0), (4.0, 5.0), (5.0, 0.0), (7.0, 0.0)] or out[0] == (0.0, 0.0)
    assert (4.0, 5.0) in out and out[0] == (0.0, 0.0) and out[-1] == (7.0, 0.0)
    assert len(out) < len(pts)
    assert f.simplify(pts, 0.0) == [(float(x), float(y)) for x, y in pts]
    assert f.simplify(pts[:2], 1.0) == [(0.0, 0.0), (1.0, 0.05)]
    line = [(float(i), 2.0 * i) for i in range(100)]
    assert f.simplify(line, 1e-6) == [(0.0, 0.0), (99.0, 198.0)]
    closedish = [(0, 0), (5, 0), (0, 0)]  # first == last must not divide by zero
    assert len(f.simplify(closedish, 0.1)) == 3


# ---------------------------------------------------------------- fit_bezier

def test_fit_bezier_smooth_arc_within_tolerance():
    rnd = random.Random(1)
    pts = [(100 * math.cos(t / 100 * math.pi * 1.5) + rnd.gauss(0, 0.2),
            100 * math.sin(t / 100 * math.pi * 1.5) + rnd.gauss(0, 0.2)) for t in range(101)]
    path = f.fit_bezier(pts, max_error=1.0)
    assert len(path.points) >= 2 and not path.closed
    assert len(path.points) < 15
    assert max_fit_error(pts, path) <= 1.5 * 1.0
    assert all(p.handle_left_type == "ALIGNED" for p in path.points)  # no corners on a smooth arc


@pytest.mark.parametrize("max_error", [0.25, 0.5, 2.0])
def test_fit_bezier_error_bound_scales_with_max_error(max_error):
    pts = [(t, 20 * math.sin(t / 12.0)) for t in np.linspace(0, 120, 400)]
    path = f.fit_bezier(pts, max_error=max_error)
    assert max_fit_error(pts, path) <= 1.5 * max_error
    assert path.points[0].co == pytest.approx(pts[0]) and path.points[-1].co == pytest.approx(pts[-1])


def test_fit_bezier_tighter_tolerance_uses_more_segments():
    pts = [(t, 20 * math.sin(t / 12.0)) for t in np.linspace(0, 120, 400)]
    assert len(f.fit_bezier(pts, 0.05).points) > len(f.fit_bezier(pts, 5.0).points)


def test_fit_bezier_preserves_corners_open():
    pts = [(x * 0.5, 0.0) for x in range(0, 41)] + [(20.0, y * 0.5) for y in range(1, 41)]
    path = f.fit_bezier(pts, max_error=1.0)
    assert len(path.points) == 3
    corner = path.points[1]
    assert corner.co == pytest.approx((20.0, 0.0))
    assert (corner.handle_left_type, corner.handle_right_type) == ("FREE", "FREE")
    assert max_fit_error(pts, path) <= 1.0e-6  # straight legs stay straight
    zig = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 20)]
    zp = f.fit_bezier(zig, 0.5)
    assert [tuple(round(c, 6) for c in p.co) for p in zp.points] == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 20.0)]


def test_fit_bezier_corner_in_noisy_data():
    rnd = random.Random(2)
    pts = []
    for i in range(60):
        pts.append((i * 1.0 + rnd.gauss(0, 0.15), rnd.gauss(0, 0.15)))
    for i in range(1, 60):
        pts.append((59 + rnd.gauss(0, 0.15), i * 1.0 + rnd.gauss(0, 0.15)))
    path = f.fit_bezier(pts, max_error=1.0)
    corners = [p for p in path.points if p.handle_left_type == "FREE"]
    assert len(corners) == 1
    assert math.hypot(corners[0].co[0] - 59, corners[0].co[1] - 0) < 2.0
    assert max_fit_error(pts, path) <= 1.5


def test_fit_bezier_closed_shapes():
    rnd = random.Random(4)
    circle = [(50 * math.cos(2 * math.pi * i / 200) + rnd.gauss(0, 0.1), 50 * math.sin(2 * math.pi * i / 200) + rnd.gauss(0, 0.1))
              for i in range(200)]
    path = f.fit_bezier(circle, 1.0, closed=True)
    assert path.closed and len(path.points) >= 3
    assert max_fit_error(circle, path, closed=True) <= 1.5
    assert all(p.handle_left_type == "ALIGNED" for p in path.points)
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    sp = f.fit_bezier(sq, 1.0, closed=True)
    assert sp.closed and [p.co for p in sp.points] == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert all(p.handle_left_type == "FREE" for p in sp.points)
    dense = []
    for a, b in zip(sq, sq[1:] + sq[:1]):
        for t in range(50):
            dense.append((a[0] + (b[0] - a[0]) * t / 50, a[1] + (b[1] - a[1]) * t / 50))
    dp = f.fit_bezier(dense, 0.2, closed=True)
    assert len(dp.points) == 4
    assert max_fit_error(dense, dp, closed=True) < 1e-6
    # first == last is tolerated for closed input
    assert len(f.fit_bezier(sq + [sq[0]], 1.0, closed=True).points) == 4


def test_fit_bezier_handles_are_aligned_at_smooth_joins():
    pts = [(t, 30 * math.sin(t / 8.0)) for t in np.linspace(0, 100, 300)]
    path = f.fit_bezier(pts, 0.1)
    assert len(path.points) > 3
    for p in path.points[1:-1]:
        lx, ly = p.handle_left[0] - p.co[0], p.handle_left[1] - p.co[1]
        rx, ry = p.handle_right[0] - p.co[0], p.handle_right[1] - p.co[1]
        assert abs(lx * ry - ly * rx) < 1e-6 * max(1.0, math.hypot(lx, ly) * math.hypot(rx, ry))
        assert lx * rx + ly * ry < 0


def test_fit_bezier_degenerate_inputs():
    one = f.fit_bezier([(3, 4)], 1.0)
    assert len(one.points) == 2 and one.points[0].co == (3.0, 4.0)
    two = f.fit_bezier([(0, 0), (9, 0)], 1.0)
    assert len(two.points) == 2 and two.points[1].co == (9.0, 0.0)
    assert g.flatten_path(two, 8) == [(0.0, 0.0), (9.0, 0.0)]
    dup = f.fit_bezier([(0, 0), (0, 0), (5, 5), (5, 5), (9, 0)], 1.0)
    assert len(dup.points) >= 2
    tri = f.fit_bezier([(0, 0), (10, 0)], 1.0, closed=True)
    assert tri.closed and len(tri.points) >= 3
    with pytest.raises(ValueError):
        f.fit_bezier([], 1.0)
    with pytest.raises(ValueError):
        f.fit_bezier([(0, 0), (1, 1)], 0.0)


def test_non_finite_input_raises():
    nan = float("nan")
    inf = float("inf")
    pts = [(0, 0), (1, nan), (2, 2)]
    for call in (lambda: f.fit_bezier(pts, 1.0), lambda: f.resample(pts, 1.0), lambda: f.simplify(pts, 1.0),
                 lambda: f.smooth_polyline(pts), lambda: f.fit_bezier([(0, 0), (1, 1)], nan),
                 lambda: f.resample([(0, 0), (1, 1)], inf), lambda: f.simplify([(0, 0), (1, 1), (2, 0)], nan)):
        with pytest.raises(ValueError):
            call()
