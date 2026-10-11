"""Unit tests for seam_core.stroke."""

import math

import numpy as np
import pytest

from seam_core import geometry as g
from seam_core import stroke as st

W = 2.0
L = 20.0
LINE = [(0.0, 0.0), (L, 0.0)]
RIGHT_ANGLE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]


def mesh_area(verts, tris):
    return sum(abs(g.signed_area([verts[i] for i in t])) for t in tris)


def mesh_signed_area(verts, tris):
    return sum(g.signed_area([verts[i] for i in t]) for t in tris)


def nonzero_area(rings, step=0.02):
    """Area of the NONZERO union of ``rings`` by sampling cell centres of a regular grid."""
    pts = np.array([p for r in rings for p in r])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    wn = np.zeros(gx.shape, dtype=np.int32)
    for ring in rings:
        a = np.asarray(ring)
        prev = np.roll(a, 1, axis=0)
        for (xa, ya), (xb, yb) in zip(prev, a):
            up = (ya <= gy) & (yb > gy)
            down = (ya > gy) & (yb <= gy)
            cross = (xb - xa) * (gy - ya) - (gx - xa) * (yb - ya)
            wn += (up & (cross > 0)).astype(np.int32) - (down & (cross < 0)).astype(np.int32)
    return float((wn != 0).sum()) * step * step


# ---------------------------------------------------------------- caps on a straight line

@pytest.mark.parametrize("cap,expected", [
    ("BUTT", W * L),
    ("SQUARE", W * (L + W)),
    ("ROUND", W * L + math.pi * (W / 2) ** 2),
])
def test_mesh_area_by_cap(cap, expected):
    verts, tris = st.stroke_mesh(LINE, False, W, cap=cap)
    tol = 0.01 if cap == "ROUND" else 1e-9
    assert mesh_area(verts, tris) == pytest.approx(expected, rel=tol)
    assert all(g.signed_area([verts[i] for i in t]) > 0 for t in tris)  # CCW triangles
    assert mesh_signed_area(verts, tris) == pytest.approx(mesh_area(verts, tris))


@pytest.mark.parametrize("cap,expected", [
    ("BUTT", W * L),
    ("SQUARE", W * (L + W)),
    ("ROUND", W * L + math.pi * (W / 2) ** 2),
])
def test_outline_area_by_cap(cap, expected):
    rings = st.stroke_outline(LINE, False, W, cap=cap)
    assert all(g.signed_area(r) > 0 for r in rings)
    assert nonzero_area(rings) == pytest.approx(expected, rel=0.01)


def test_bounding_extent_by_cap():
    for cap, ext in (("BUTT", 0.0), ("SQUARE", W / 2), ("ROUND", W / 2)):
        verts, _ = st.stroke_mesh(LINE, False, W, cap=cap)
        xs = [v[0] for v in verts]
        assert min(xs) == pytest.approx(-ext) and max(xs) == pytest.approx(L + ext)
        assert max(v[1] for v in verts) == pytest.approx(W / 2)


# ---------------------------------------------------------------- joins on a right angle

def analytic_right_angle(join, w=W, l1=10.0, l2=10.0):
    base = w * (l1 + l2)
    return {
        "MITER": base,
        "BEVEL": base - w * w / 8.0,
        "ROUND": base - w * w / 4.0 + math.pi * w * w / 16.0,
    }[join]


@pytest.mark.parametrize("join", ["MITER", "BEVEL", "ROUND"])
@pytest.mark.parametrize("flip", [False, True])
def test_right_angle_join_area(join, flip):
    poly = [(x, -y if flip else y) for x, y in RIGHT_ANGLE]  # left and right turns
    expected = analytic_right_angle(join)
    verts, tris = st.stroke_mesh(poly, False, W, join=join)
    assert mesh_area(verts, tris) == pytest.approx(expected, rel=0.02)
    # no triangle overlaps for an ordinary corner: summed triangle area equals the union area
    rings = st.stroke_outline(poly, False, W, join=join)
    assert nonzero_area(rings) == pytest.approx(expected, rel=0.02)


def test_join_extra_area_over_bevel():
    """The outer wedge beyond the bevel triangle: miter w^2/8, round (pi/16 - 1/8) w^2."""
    bevel = st.stroke_mesh(RIGHT_ANGLE, False, W, join="BEVEL")
    miter = st.stroke_mesh(RIGHT_ANGLE, False, W, join="MITER")
    rnd = st.stroke_mesh(RIGHT_ANGLE, False, W, join="ROUND")
    base = mesh_area(*bevel)
    assert mesh_area(*miter) - base == pytest.approx(W * W / 8, rel=0.02)
    assert mesh_area(*rnd) - base == pytest.approx((math.pi / 16 - 1 / 8) * W * W, rel=0.02)


def test_miter_limit_falls_back_to_bevel():
    sharp = [(0.0, 0.0), (10.0, 0.0), (0.0, 1.0)]  # very acute turn
    mitered = st.stroke_mesh(sharp, False, W, join="MITER", miter_limit=100.0)
    limited = st.stroke_mesh(sharp, False, W, join="MITER", miter_limit=2.0)
    bevel = st.stroke_mesh(sharp, False, W, join="BEVEL")
    assert mesh_area(*mitered) > mesh_area(*limited)
    assert mesh_area(*limited) == pytest.approx(mesh_area(*bevel))
    # a right angle (ratio 1.414) stays mitered at limit 2 but bevels at limit 1.2
    assert mesh_area(*st.stroke_mesh(RIGHT_ANGLE, False, W, join="MITER", miter_limit=2.0)) == pytest.approx(W * 20)
    assert mesh_area(*st.stroke_mesh(RIGHT_ANGLE, False, W, join="MITER", miter_limit=1.2)) == pytest.approx(W * 20 - W * W / 8)
    # outlines honour the limit too
    ring_area = nonzero_area(st.stroke_outline(sharp, False, W, join="MITER", miter_limit=2.0), 0.02)
    assert ring_area == pytest.approx(nonzero_area(st.stroke_outline(sharp, False, W, join="BEVEL"), 0.02), rel=0.01)


def test_collinear_midpoint_does_not_change_area():
    straight = st.stroke_mesh([(0, 0), (10, 0), (20, 0)], False, W)
    assert mesh_area(*straight) == pytest.approx(W * L)
    assert nonzero_area(st.stroke_outline([(0, 0), (10, 0), (20, 0)], False, W)) == pytest.approx(W * L, rel=0.01)


def test_reversal_and_sharp_turn_cover_the_expected_area():
    out_and_back = [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0)]
    rings = st.stroke_outline(out_and_back, False, W, cap="BUTT", join="ROUND")
    # a round join at the reversal adds a half disc beyond x = 10
    assert nonzero_area(rings) == pytest.approx(W * 10 + math.pi * (W / 2) ** 2 / 2, rel=0.02)
    flat = st.stroke_outline(out_and_back, False, W, cap="BUTT", join="MITER")
    assert nonzero_area(flat) == pytest.approx(W * 10, rel=0.01)


# ---------------------------------------------------------------- closed paths

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


@pytest.mark.parametrize("cap", ["BUTT", "ROUND", "SQUARE"])
def test_closed_paths_have_no_caps(cap):
    verts, tris = st.stroke_mesh(SQUARE, True, W, cap=cap, join="MITER")
    # mitered square ring: outer 12x12 minus inner 8x8
    assert mesh_area(verts, tris) == pytest.approx(12 * 12 - 8 * 8)
    xs = [v[0] for v in verts]
    assert min(xs) == pytest.approx(-1.0) and max(xs) == pytest.approx(11.0)
    rings = st.stroke_outline(SQUARE, True, W, cap=cap, join="MITER")
    assert nonzero_area(rings) == pytest.approx(12 * 12 - 8 * 8, rel=0.01)


def test_closed_path_with_repeated_first_point_equals_plain_closed():
    a = st.stroke_mesh(SQUARE, True, W)
    b = st.stroke_mesh(SQUARE + [SQUARE[0]], True, W)
    assert mesh_area(*a) == pytest.approx(mesh_area(*b))


def test_closed_round_join_ring():
    circle = [(20 * math.cos(2 * math.pi * i / 64), 20 * math.sin(2 * math.pi * i / 64)) for i in range(64)]
    verts, tris = st.stroke_mesh(circle, True, 4.0, join="ROUND")
    expected = math.pi * (22 ** 2 - 18 ** 2)
    assert mesh_area(verts, tris) == pytest.approx(expected, rel=0.01)


def test_inner_side_is_trimmed_so_ordinary_strokes_do_not_overlap():
    # a gently curving open path: the union area from the outline equals the summed triangle area
    pts = [(i * 2.0, 3.0 * math.sin(i * 0.3)) for i in range(40)]
    verts, tris = st.stroke_mesh(pts, False, 1.0, join="BEVEL")
    assert mesh_area(verts, tris) == pytest.approx(nonzero_area(st.stroke_outline(pts, False, 1.0, join="BEVEL"), 0.02), rel=0.01)


# ---------------------------------------------------------------- degenerate / argument handling

def test_zero_and_negative_width_are_empty():
    assert st.stroke_mesh(LINE, False, 0.0) == ([], [])
    assert st.stroke_mesh(LINE, False, -3.0) == ([], [])
    assert st.stroke_outline(LINE, False, 0.0) == []
    assert st.stroke_outline(LINE, False, -1.0) == []


def test_degenerate_polylines():
    assert st.stroke_mesh([], False, W) == ([], [])
    assert st.stroke_outline([], True, W) == []
    assert st.stroke_mesh([(1, 1)], False, W, cap="BUTT") == ([], [])
    dot = st.stroke_mesh([(1, 1), (1, 1)], False, W, cap="ROUND")
    assert mesh_area(*dot) == pytest.approx(math.pi, rel=0.01)  # SVG draws a dot for round caps
    sq = st.stroke_mesh([(1, 1)], False, W, cap="SQUARE")
    assert mesh_area(*sq) == pytest.approx(W * W)
    assert nonzero_area(st.stroke_outline([(1, 1)], False, W, cap="ROUND")) == pytest.approx(math.pi, rel=0.02)
    # duplicate consecutive points are ignored
    a = st.stroke_mesh([(0, 0), (0, 0), (10, 0), (10, 0), (10, 10)], False, W)
    b = st.stroke_mesh(RIGHT_ANGLE, False, W)
    assert mesh_area(*a) == pytest.approx(mesh_area(*b))


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        st.stroke_mesh(LINE, False, W, cap="TRIANGLE")
    with pytest.raises(ValueError):
        st.stroke_outline(LINE, False, W, join="CURVY")
    with pytest.raises(ValueError):
        st.stroke_mesh(LINE, False, float("nan"))
    with pytest.raises(ValueError):
        st.stroke_mesh(LINE, False, W, miter_limit=float("inf"))


def test_non_finite_polyline_raises():
    for bad in (float("nan"), float("inf")):
        pts = [(0.0, 0.0), (5.0, bad), (10.0, 0.0)]
        with pytest.raises(ValueError):
            st.stroke_mesh(pts, False, W)
        with pytest.raises(ValueError):
            st.stroke_outline(pts, True, W)


def test_round_segments_controls_tessellation():
    coarse = st.stroke_mesh(LINE, False, W, cap="ROUND", round_segments=1)
    fine = st.stroke_mesh(LINE, False, W, cap="ROUND", round_segments=16)
    assert len(fine[1]) > len(coarse[1])
    assert mesh_area(*fine) == pytest.approx(W * L + math.pi * (W / 2) ** 2, rel=0.002)
    # a vertex index never refers outside the vertex list
    for verts, tris in (coarse, fine):
        assert all(0 <= i < len(verts) for t in tris for i in t)
