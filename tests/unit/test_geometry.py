"""Unit tests for seam_core.geometry (pure Python + numpy)."""

import json
import math
import os
import platform
import random
import sys
import time

import pytest

from seam_core import geometry as g
from seam_core.geometry import BezierPath, BezierPoint


# ---------------------------------------------------------------- helpers

def record_perf(name, ms):
    """Append a timing to $SEAM_PERF_OUT (artifacts/perf/core-geometry.json) when set."""
    out = os.environ.get("SEAM_PERF_OUT")
    if not out:
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        with open(out) as fh:
            doc = json.load(fh)
    except Exception:
        doc = {}
    cpu = ""
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next((ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")), "")
    except OSError:
        pass
    doc["machine"] = {"vcpus": os.cpu_count(), "cpu": cpu, "platform": platform.platform()}
    label = "python-%s" % platform.python_version()
    if "bpy" in sys.modules:
        label += " (in Blender %s)" % sys.modules["bpy"].app.version_string
    run = doc.setdefault("runs", {}).setdefault(label, {})
    run[name] = round(ms, 3)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)


def tri_area(verts, tri):
    return g.signed_area([verts[i] for i in tri])


def total_area(verts, tris):
    return sum(tri_area(verts, t) for t in tris)


def circle(n, r, cx=0.0, cy=0.0, ccw=True):
    pts = [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return pts if ccw else pts[::-1]


SQ = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
HOLE_CW = [(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)]


def vpath(pts, closed=True):
    return BezierPath([BezierPoint(p, p, p, "VECTOR", "VECTOR") for p in pts], closed)


# ---------------------------------------------------------------- bezier evaluation / flattening

def test_cubic_point_endpoints_and_midpoint():
    p0, p1, p2, p3 = (0, 0), (0, 10), (10, 10), (10, 0)
    assert g.cubic_point(p0, p1, p2, p3, 0.0) == pytest.approx((0, 0))
    assert g.cubic_point(p0, p1, p2, p3, 1.0) == pytest.approx((10, 0))
    assert g.cubic_point(p0, p1, p2, p3, 0.5) == pytest.approx((5, 7.5))


def test_flatten_straight_segments_emit_only_end_points():
    ring = g.flatten_path(vpath(SQ), resolution=12)
    assert ring == [tuple(map(float, p)) for p in SQ]
    # handles on the chord (FREE type) count as straight too
    pts = [BezierPoint((0, 0), (0, 0), (3, 0), "FREE", "FREE"), BezierPoint((9, 0), (6, 0), (9, 0), "FREE", "FREE")]
    assert g.flatten_path(BezierPath(pts, False), 12) == [(0.0, 0.0), (9.0, 0.0)]


def test_flatten_curve_resolution_and_closed_ring():
    pts = [
        BezierPoint((1, 0), (1, -0.55), (1, 0.55), "ALIGNED", "ALIGNED"),
        BezierPoint((0, 1), (0.55, 1), (-0.55, 1), "ALIGNED", "ALIGNED"),
        BezierPoint((-1, 0), (-1, 0.55), (-1, -0.55), "ALIGNED", "ALIGNED"),
        BezierPoint((0, -1), (-0.55, -1), (0.55, -1), "ALIGNED", "ALIGNED"),
    ]
    ring = g.flatten_path(BezierPath(pts, True), resolution=10)
    assert len(ring) == 4 * 10  # no duplicated closing point
    assert ring[0] == (1.0, 0.0)
    assert g.signed_area(ring) == pytest.approx(math.pi, rel=0.01)
    open_pts = g.flatten_path(BezierPath(pts, False), resolution=10)
    assert len(open_pts) == 3 * 10 + 1
    assert open_pts[-1] == (0.0, -1.0)
    assert len(g.flatten_path(BezierPath(pts, True), resolution=1)) == 4


def test_flatten_degenerate_paths():
    assert g.flatten_path(BezierPath([], True)) == []
    one = BezierPath([BezierPoint((2, 3), (2, 3), (2, 3))], False)
    assert g.flatten_path(one) == [(2.0, 3.0)]
    assert g.flatten_paths([one, vpath(SQ)], 4) == [[(2.0, 3.0)], [tuple(map(float, p)) for p in SQ]]


def test_flatten_non_finite_raises():
    bad = BezierPath([BezierPoint((0, 0), (0, 0), (float("nan"), 0)), BezierPoint((1, 0), (1, 0), (1, 0))], False)
    with pytest.raises(ValueError):
        g.flatten_path(bad)
    with pytest.raises(ValueError):
        g.cubic_point((0, 0), (0, 1), (1, 1), (1, float("inf")), 0.5)


# ---------------------------------------------------------------- measures

def test_area_orientation_bbox_length():
    assert g.signed_area(SQ) == pytest.approx(100.0)
    assert g.signed_area(SQ[::-1]) == pytest.approx(-100.0)
    assert g.is_ccw(SQ) and not g.is_ccw(SQ[::-1])
    assert g.signed_area([(0, 0), (1, 1)]) == 0.0
    assert g.bbox([SQ, [(-5, 2), (3, 20)]]) == (-5.0, 0.0, 10.0, 20.0)
    with pytest.raises(ValueError):
        g.bbox([])
    with pytest.raises(ValueError):
        g.bbox([[]])
    assert g.path_length(SQ) == pytest.approx(30.0)
    assert g.path_length(SQ, closed=True) == pytest.approx(40.0)
    assert g.path_length([(0, 0)]) == 0.0


def test_point_in_ring_and_winding():
    assert g.point_in_ring((5, 5), SQ)
    assert not g.point_in_ring((15, 5), SQ)
    assert not g.point_in_ring((5, 5), SQ[:2])
    assert g.point_in_ring((5, 5), SQ[::-1])  # even-odd ignores orientation
    assert g.winding_number((5, 5), SQ) == 1
    assert g.winding_number((5, 5), SQ[::-1]) == -1
    assert g.winding_number((50, 5), SQ) == 0
    double = SQ + [(0.0, 0.0)]  # not a real double loop, sanity only
    assert g.winding_number((5, 5), double) in (1, 2)
    twice = SQ + SQ
    assert g.winding_number((5, 5), twice) == 2
    # concave (C shape): point in the notch is outside
    c_shape = [(0, 0), (10, 0), (10, 3), (3, 3), (3, 7), (10, 7), (10, 10), (0, 10)]
    assert not g.point_in_ring((7, 5), c_shape)
    assert g.point_in_ring((1, 5), c_shape)


# ---------------------------------------------------------------- affine

def test_affine_composition_order_and_determinant():
    translate = (1, 0, 0, 1, 5, 7)
    scale2 = (2, 0, 0, 2, 0, 0)
    # apply scale first then translate
    m = g.affine_multiply(translate, scale2)
    assert g.transform_points([(1, 1)], m) == [(7.0, 9.0)]
    m2 = g.affine_multiply(scale2, translate)
    assert g.transform_points([(1, 1)], m2) == [(12.0, 16.0)]
    rot90 = (0, 1, -1, 0, 0, 0)  # x' = -y, y' = x
    assert g.transform_points([(1, 0)], rot90) == [(0.0, 1.0)]
    assert g.affine_determinant(rot90) == pytest.approx(1.0)
    assert g.affine_determinant((-1, 0, 0, 1, 0, 0)) == -1.0
    assert g.affine_multiply(g.IDENTITY, m) == pytest.approx(m)
    assert g.transform_points([], m) == []
    with pytest.raises(ValueError):
        g.transform_points([(0, 0)], (1, 0, 0, 1, float("nan"), 0))


def test_transform_path_moves_handles_and_keeps_types():
    p = BezierPath([BezierPoint((0, 0), (-1, 0), (1, 0), "ALIGNED", "VECTOR")], True)
    q = g.transform_path(p, (2, 0, 0, 3, 10, 20))
    pt = q.points[0]
    assert pt.co == (10.0, 20.0) and pt.handle_left == (8.0, 20.0) and pt.handle_right == (12.0, 20.0)
    assert (pt.handle_left_type, pt.handle_right_type) == ("ALIGNED", "VECTOR")
    assert q.closed is True and p.points[0].co == (0, 0)


# ---------------------------------------------------------------- cleaning / normalising

def test_clean_ring_removes_duplicates_and_collinear_points():
    ring = [(0, 0), (0, 0), (5, 0), (10, 0), (10, 10), (10, 10), (0, 10), (0, 0)]
    cleaned = g.clean_ring(ring)
    assert cleaned == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert g.signed_area(cleaned) == pytest.approx(100.0)
    near = [(0, 0), (5, 1e-3), (10, 0), (10, 10), (0, 10)]
    assert len(g.clean_ring(near)) == 5  # not exactly collinear -> kept
    assert len(g.clean_ring([(0, 0), (1, 1), (2, 2), (3, 3)])) < 3
    assert g.clean_ring([]) == []


def test_normalize_rings_evenodd_nesting_and_orientation():
    outer = SQ[::-1]  # CW on input
    hole = [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)]  # CCW on input
    island = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
    far = [(20.0, 0.0), (24.0, 0.0), (24.0, 4.0), (20.0, 4.0)]
    shapes = g.normalize_rings([outer, hole, island, far])
    assert len(shapes) == 3
    s0 = next(s for s in shapes if abs(g.signed_area(s[0]) - 100.0) < 1e-9)
    assert len(s0[1]) == 1 and g.signed_area(s0[1][0]) == pytest.approx(-36.0)
    s1 = next(s for s in shapes if abs(g.signed_area(s[0]) - 4.0) < 1e-9)
    assert s1[1] == []
    for outer_r, holes in shapes:
        assert g.signed_area(outer_r) > 0
        assert all(g.signed_area(h) < 0 for h in holes)


def test_normalize_rings_drops_degenerate():
    assert g.normalize_rings([[(0, 0), (1, 1)], [(0, 0), (5, 0), (10, 0)], []]) == []
    assert len(g.normalize_rings([SQ, [(0, 0), (0, 0), (0, 0)]])) == 1
    with pytest.raises(ValueError):
        g.normalize_rings([SQ], fill_rule="BOGUS")


def test_normalize_rings_nonzero_uses_winding():
    big = SQ
    same_dir_inner = [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)]  # CCW inside CCW: filled (winding 2)
    s = g.normalize_rings([big, same_dir_inner], "NONZERO")
    assert len(s) == 1 and s[0][1] == []
    s_eo = g.normalize_rings([big, same_dir_inner], "EVENODD")
    assert len(s_eo) == 1 and len(s_eo[0][1]) == 1
    opposite = same_dir_inner[::-1]  # CW inside CCW: a hole under both rules
    s2 = g.normalize_rings([big, opposite], "NONZERO")
    assert len(s2) == 1 and len(s2[0][1]) == 1
    # island inside a hole is a new outer
    island = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
    s3 = g.normalize_rings([big, opposite, island], "NONZERO")
    assert len(s3) == 2


def test_normalize_rings_touching_rings_do_not_nest():
    a = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
    b = [(5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0)]  # shares an edge
    shapes = g.normalize_rings([a, b])
    assert len(shapes) == 2 and all(s[1] == [] for s in shapes)


# ---------------------------------------------------------------- intersections

def test_segment_intersection():
    assert g.segment_intersection((0, 0), (10, 10), (0, 10), (10, 0)) == pytest.approx((5, 5))
    assert g.segment_intersection((0, 0), (10, 0), (5, -5), (5, 5)) == pytest.approx((5, 0))
    assert g.segment_intersection((0, 0), (10, 0), (0, 1), (10, 1)) is None  # parallel
    assert g.segment_intersection((0, 0), (5, 5), (5, 5), (10, 0)) is None  # touching endpoints
    assert g.segment_intersection((0, 0), (10, 0), (5, 0), (5, 5)) is None  # T touch
    assert g.segment_intersection((0, 0), (10, 0), (5, 0), (15, 0)) is None  # collinear overlap
    assert g.segment_intersection((0, 0), (1, 1), (5, 0), (6, 1)) is None


FIGURE_EIGHT = [(0, 0), (10, 10), (10, 0), (0, 10)]


def test_self_intersections_figure_eight():
    hits = g.self_intersections(FIGURE_EIGHT)
    assert len(hits) == 1
    i, j, pt = hits[0]
    assert (i, j) == (0, 2) and pt == pytest.approx((5.0, 5.0))
    assert g.self_intersections(SQ) == []
    assert g.self_intersections(circle(100, 50)) == []
    # open polyline: the closing edge does not exist, so the same points need a different crossing
    assert g.self_intersections([(0, 0), (10, 10), (10, 0), (0, 10)], closed=False)[0][2] == pytest.approx((5.0, 5.0))
    assert g.self_intersections([(0, 0), (10, 0), (10, 10), (0, 10)], closed=False) == []
    assert g.self_intersections([(0, 0), (1, 1), (2, 0)]) == []


def test_self_intersections_limit_and_matches_brute_force():
    rnd = random.Random(7)
    for trial in range(30):
        n = rnd.randint(4, 40)
        pts = [(rnd.uniform(0, 100), rnd.uniform(0, 100)) for _ in range(n)]
        brute = set()
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if g.segment_intersection(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]) is not None:
                    brute.add((i, j))
        found = {(i, j) for i, j, _ in g.self_intersections(pts, limit=10 ** 6)}
        assert found == brute, trial
        assert len(g.self_intersections(pts, limit=3)) <= 3


def test_self_intersections_5k_ring_is_fast():
    n = 5000
    ring = [((100 + 20 * math.sin(7 * 2 * math.pi * i / n)) * math.cos(2 * math.pi * i / n),
             (100 + 20 * math.sin(7 * 2 * math.pi * i / n)) * math.sin(2 * math.pi * i / n)) for i in range(n)]
    t0 = time.perf_counter()
    hits = g.self_intersections(ring)
    ms = (time.perf_counter() - t0) * 1000
    assert hits == []
    assert ms < 200
    record_perf("self_intersections_5k_ring_ms", ms)
    # vertical-run worst case for x-sorting schemes: a long ladder-shaped ring
    ladder = [(0.0, float(i)) for i in range(2500)] + [(1.0, float(i)) for i in range(2499, -1, -1)]
    t0 = time.perf_counter()
    assert g.self_intersections(ladder) == []
    assert (time.perf_counter() - t0) * 1000 < 400


def test_self_intersections_star_polygon_with_many_crossings_is_bounded():
    n = 301
    step = 100  # {301/100} star: every edge crosses many others
    pts = [(100 * math.cos(2 * math.pi * ((i * step) % n) / n), 100 * math.sin(2 * math.pi * ((i * step) % n) / n))
           for i in range(n)]
    t0 = time.perf_counter()
    hits = g.self_intersections(pts, limit=20)
    assert len(hits) == 20
    assert all(i < j for i, j, _ in hits)
    assert time.perf_counter() - t0 < 2.0
    everything = g.self_intersections(pts, limit=10 ** 7)
    assert len(everything) > 1000


SKEW_EIGHT = [(0, 0), (10, 10), (10, 0), (0, 6)]  # self-crossing with non-zero net area


def test_validate_rings_reports_issue_codes():
    assert g.validate_rings([SQ, HOLE_CW]) == []
    codes = lambda rings, **kw: sorted(i.code for i in g.validate_rings(rings, **kw))
    assert codes([SKEW_EIGHT]) == ["SELF_INTERSECTION"]
    assert codes([[(0, 0), (5, 0), (10, 0)]]) == ["ZERO_AREA"]
    assert codes([[(0, 0), (1, 1)]]) == ["TOO_FEW_POINTS"]
    assert codes([[]]) == ["TOO_FEW_POINTS"]
    assert codes([[(0, 0), (0, 0), (1, 1), (1, 1)]]) == ["TOO_FEW_POINTS"]
    issues = g.validate_rings([SQ, SKEW_EIGHT])
    assert len(issues) == 1 and issues[0].path_index == 1 and issues[0].point is not None
    # a hole crossing the outer boundary is invalid too
    crossing = [(8.0, 4.0), (12.0, 4.0), (12.0, 6.0), (8.0, 6.0)]
    assert codes([SQ, crossing]) == ["SELF_INTERSECTION"]
    # open-path style validation: no area requirement
    assert codes([[(0, 0), (5, 0), (10, 0)]], require_closed_area=False) == []
    assert g.validate_rings([]) == []


def test_validate_rings_closing_duplicate_is_not_an_issue():
    assert g.validate_rings([SQ + [SQ[0]]]) == []


# ---------------------------------------------------------------- triangulation

def test_triangulate_square_with_square_hole_area():
    verts, tris = g.triangulate(SQ, [HOLE_CW])
    assert verts == [tuple(map(float, p)) for p in SQ + HOLE_CW]
    assert total_area(verts, tris) == pytest.approx(100.0 - 16.0, abs=1e-9)
    assert all(tri_area(verts, t) > 0 for t in tris)  # CCW
    assert {i for t in tris for i in t} == set(range(8))  # every ring vertex used


def test_triangulate_orientation_agnostic():
    verts, tris = g.triangulate(SQ[::-1], [HOLE_CW[::-1]])
    assert total_area(verts, tris) == pytest.approx(84.0, abs=1e-9)
    assert all(tri_area(verts, t) > 0 for t in tris)
    verts, tris = g.triangulate(SQ, [HOLE_CW[::-1]])  # hole given CCW
    assert total_area(verts, tris) == pytest.approx(84.0, abs=1e-9)


def test_triangulate_degenerate_inputs():
    assert g.triangulate([], []) == ([], [])
    v, t = g.triangulate([(0, 0), (1, 0)], [])
    assert t == [] and len(v) == 2
    v, t = g.triangulate([(0, 0), (5, 0), (10, 0)], [])  # all collinear
    assert t == []
    # duplicate and collinear points
    ring = [(0, 0), (0, 0), (5, 0), (10, 0), (10, 5), (10, 10), (10, 10), (0, 10), (0, 5)]
    v, t = g.triangulate(ring, [])
    assert total_area(v, t) == pytest.approx(100.0, abs=1e-9)
    used = {ring[i] for tr in t for i in tr}
    assert used == set(ring)  # every distinct position is still a triangle corner


def test_triangulate_collinear_vertices_stay_conforming():
    # square with many exactly collinear points on every side: walls need those vertices used
    ring = []
    for k in range(5):
        ring.append((float(k * 2), 0.0))
    for k in range(5):
        ring.append((10.0, float(k * 2)))
    for k in range(5):
        ring.append((10.0 - k * 2, 10.0))
    for k in range(5):
        ring.append((0.0, 10.0 - k * 2))
    v, t = g.triangulate(ring, [])
    assert total_area(v, t) == pytest.approx(100.0, abs=1e-9)
    assert {i for tr in t for i in tr} == set(range(len(ring)))
    # every boundary edge of the triangulation must be a ring edge (no T-junctions)
    edges = {}
    for a, b, c in t:
        for e in ((a, b), (b, c), (c, a)):
            edges[e] = edges.get(e, 0) + 1
    boundary = {e for e in edges if (e[1], e[0]) not in edges}
    n = len(ring)
    assert boundary == {(i, (i + 1) % n) for i in range(n)}


def test_triangulate_touching_holes():
    # hole whose corner touches the outer boundary vertex
    hole_at_corner = [(0.0, 0.0), (0.0, 3.0), (3.0, 3.0), (3.0, 0.0)][::1]
    v, t = g.triangulate(SQ, [hole_at_corner[::-1] if g.signed_area(hole_at_corner) > 0 else hole_at_corner])
    assert total_area(v, t) == pytest.approx(100.0 - 9.0, abs=1e-9)
    # hole touching the middle of an outer edge
    hole_edge = [(4.0, 0.0), (6.0, 3.0), (3.0, 3.0)]
    v, t = g.triangulate(SQ, [hole_edge])
    assert total_area(v, t) == pytest.approx(100.0 - abs(g.signed_area(hole_edge)), abs=1e-9)
    # two holes sharing a vertex
    h1 = [(2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (4.0, 2.0)]
    h2 = [(4.0, 4.0), (4.0, 6.0), (6.0, 6.0), (6.0, 4.0)]
    v, t = g.triangulate(SQ, [h1, h2])
    assert total_area(v, t) == pytest.approx(100.0 - 8.0, abs=1e-9)
    # hole sharing a whole edge with the outer
    h3 = [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)]
    v, t = g.triangulate(SQ, [h3])
    assert total_area(v, t) == pytest.approx(100.0 - 16.0, abs=1e-9)


def test_triangulate_concave_and_random_star_polygons():
    rnd = random.Random(3)
    for trial in range(60):
        n = rnd.randint(5, 150)
        pts = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            r = rnd.uniform(20, 100)
            pts.append((r * math.cos(ang), r * math.sin(ang)))
        v, t = g.triangulate(pts)
        assert total_area(v, t) == pytest.approx(g.signed_area(pts), rel=1e-9), trial
        assert all(tri_area(v, tr) >= -1e-9 for tr in t)


def test_triangulate_polygon_with_random_holes():
    rnd = random.Random(11)
    for trial in range(15):
        outer = circle(rnd.randint(8, 60), 100.0)
        holes = []
        centers = [(-50, -50), (0, -50), (50, -50), (-50, 0), (0, 0), (50, 0), (-50, 50), (0, 50), (50, 50)]
        for cx, cy in centers:
            if rnd.random() < 0.7:
                holes.append(circle(rnd.randint(3, 12), rnd.uniform(5, 20), cx, cy, ccw=False))
        v, t = g.triangulate(outer, holes)
        expected = g.signed_area(outer) + sum(g.signed_area(h) for h in holes)
        assert total_area(v, t) == pytest.approx(expected, rel=1e-9), trial


def test_triangulate_2k_polygon_with_20_holes_under_300ms():
    n = 2000
    outer = [((1000 + 40 * math.sin(7 * 2 * math.pi * i / n)) * math.cos(2 * math.pi * i / n),
              (1000 + 40 * math.sin(7 * 2 * math.pi * i / n)) * math.sin(2 * math.pi * i / n)) for i in range(n)]
    holes = []
    for k in range(20):
        cx = -700 + (k % 5) * 350
        cy = -450 + (k // 5) * 300
        holes.append(circle(10, 40, cx, cy, ccw=False))
    t0 = time.perf_counter()
    v, t = g.triangulate(outer, holes)
    ms = (time.perf_counter() - t0) * 1000
    expected = g.signed_area(outer) + sum(g.signed_area(h) for h in holes)
    assert total_area(v, t) == pytest.approx(expected, rel=1e-9)
    assert ms < 300
    record_perf("triangulate_2k_polygon_20_holes_ms", ms)


# ---------------------------------------------------------------- non-finite input never hangs

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_non_finite_input_raises_value_error(bad):
    ring = [(0.0, 0.0), (10.0, 0.0), (10.0, bad), (0.0, 10.0)]
    t0 = time.perf_counter()
    for call in (
        lambda: g.triangulate(ring),
        lambda: g.triangulate(SQ, [ring]),
        lambda: g.self_intersections(ring),
        lambda: g.validate_rings([ring]),
        lambda: g.signed_area(ring),
        lambda: g.bbox([ring]),
        lambda: g.clean_ring(ring),
        lambda: g.normalize_rings([ring]),
        lambda: g.transform_points(ring, g.IDENTITY),
        lambda: g.path_length(ring),
        lambda: g.point_in_ring((1, 1), ring),
        lambda: g.point_in_ring((bad, 1), SQ),
        lambda: g.winding_number((1, bad), SQ),
        lambda: g.segment_intersection((0, 0), (1, 1), (0, bad), (1, 0)),
    ):
        with pytest.raises(ValueError):
            call()
    assert time.perf_counter() - t0 < 2.0


def test_malformed_points_raise_value_error():
    with pytest.raises(ValueError):
        g.signed_area([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    with pytest.raises(ValueError):
        g.triangulate([("a", "b"), (1, 2), (3, 4)])


# ---------------------------------------------------------------- review fixes: crossing rings, duplicates, T-junctions

def test_split_self_intersections_bowtie_pentagram_and_simple():
    bow = [(0, 0), (4, 4), (4, 0), (0, 4)]
    loops = g.split_self_intersections(bow)
    assert len(loops) == 2
    assert sorted(round(g.signed_area(l), 9) for l in loops) == [-4.0, 4.0]
    assert all(not g.self_intersections(l) for l in loops)
    star = [(math.cos(math.radians(90 + 144 * k)), math.sin(math.radians(90 + 144 * k))) for k in range(5)]
    parts = g.split_self_intersections(star)
    assert len(parts) >= 2 and all(not g.self_intersections(p) for p in parts)
    sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert g.split_self_intersections(sq) == [list(map(tuple, sq))]
    # winding is preserved: the pentagram centre has winding 2 either way
    assert sum(g.winding_number((0.0, 0.05), p) for p in parts) == g.winding_number((0.0, 0.05), star) == 2
    assert g.split_self_intersections([(0, 0), (1, 1)]) == []


def test_split_self_intersections_collinear_overlap_and_touch():
    # doubled-back edge: the overlapping stretch cancels
    ring = [(0, 0), (4, 0), (4, 4), (4, 2), (0, 2)]
    loops = g.split_self_intersections(ring)
    assert sum(g.signed_area(l) for l in loops) == pytest.approx(g.signed_area(ring))
    # a vertex resting on another edge of the same ring
    touch = [(0, 0), (4, 0), (4, 4), (2, 0), (0, 4)]
    parts = g.split_self_intersections(touch)
    assert sum(g.signed_area(p) for p in parts) == pytest.approx(g.signed_area(touch))
    assert len(parts) == 2


def test_triangulate_rejects_crossing_rings_quickly():
    bow = [(0, 0), (4, 4), (4, 0), (0, 4)]
    with pytest.raises(ValueError):
        g.triangulate(bow)
    with pytest.raises(ValueError):          # a hole poking through the outer boundary
        g.triangulate([(0, 0), (10, 0), (10, 10), (0, 10)], [[(8, 5), (12, 5), (12, 6), (8, 6)][::-1]])
    rnd = random.Random(3)
    n = 5000
    ring = [(math.cos(2 * math.pi * i / n) * (100 + rnd.uniform(-30, 30)),
             math.sin(2 * math.pi * i / n) * (100 + rnd.uniform(-30, 30))) for i in range(n)]
    assert g.self_intersections(ring)
    t0 = time.perf_counter()
    with pytest.raises(ValueError):
        g.triangulate(ring)
    ms = (time.perf_counter() - t0) * 1000
    record_perf("triangulate_reject_5k_crossing_ring_ms", ms)
    assert ms < 3000   # was 10.8 s of earcut fallback passes


def test_triangulate_touching_holes_and_valid_input_still_fine():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    h = [(5, 0), (3, 3), (7, 3)]               # touches the outer edge
    verts, tris = g.triangulate(sq, [h[::-1]] if g.signed_area(h) > 0 else [h])
    total = sum(abs(g.signed_area([verts[i] for i in t])) for t in tris)
    assert total == pytest.approx(100 - abs(g.signed_area(h)))


def test_triangulate_has_no_t_junctions_when_holes_line_up_on_a_diagonal():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    h1 = [(2, 2), (4, 4), (4, 2)][::-1]
    h2 = [(4, 4), (6, 4), (6, 6)][::-1]       # shares the vertex (4, 4); (2,2),(4,4),(6,6) are collinear
    verts, tris = g.triangulate(sq, [h1, h2])
    total = sum(abs(g.signed_area([verts[i] for i in t])) for t in tris)
    assert total == pytest.approx(100 - 2 - 2)
    # every undirected edge that is not a ring edge is shared by exactly two triangles, and no vertex
    # lies strictly inside any triangle edge
    for t in tris:
        for u, v in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            for k, p in enumerate(verts):
                if k in (u, v) or p in (verts[u], verts[v]):
                    continue
                cross = (verts[v][0] - verts[u][0]) * (p[1] - verts[u][1]) - (verts[v][1] - verts[u][1]) * (p[0] - verts[u][0])
                dot = (p[0] - verts[u][0]) * (verts[v][0] - verts[u][0]) + (p[1] - verts[u][1]) * (verts[v][1] - verts[u][1])
                ln = (verts[v][0] - verts[u][0]) ** 2 + (verts[v][1] - verts[u][1]) ** 2
                assert not (abs(cross) < 1e-9 and 0 < dot < ln), (t, k)


def test_normalize_rings_duplicate_rings():
    sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
    rev = sq[::-1]
    shifted = sq[2:] + sq[:2]
    assert g.normalize_rings([sq, sq], "EVENODD") == []
    assert g.normalize_rings([sq, shifted, rev], "EVENODD") != []        # odd count survives once
    assert len(g.normalize_rings([sq, shifted, rev], "EVENODD")) == 1
    # NONZERO: same orientation merges into one outer, opposite orientation cancels
    assert len(g.normalize_rings([sq, sq], "NONZERO")) == 1
    assert g.normalize_rings([sq, rev], "NONZERO") == []
    # a hole inside a doubly-wound outer is still inside material (winding 2 + (-1) != 0)
    inner = [(1, 1), (1, 3), (3, 3), (3, 1)]
    res = g.normalize_rings([sq, sq, inner], "NONZERO")
    assert len(res) == 1 and res[0][1] == []
    res = g.normalize_rings([sq, inner], "NONZERO")
    assert len(res) == 1 and len(res[0][1]) == 1
    # EVENODD: a duplicated hole cancels and leaves a solid square
    res = g.normalize_rings([sq, inner, inner], "EVENODD")
    assert len(res) == 1 and res[0][1] == []
