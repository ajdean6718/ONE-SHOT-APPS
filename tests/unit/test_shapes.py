"""Unit tests for seam_core.shapes."""

import math

import pytest

from seam_core import geometry as g
from seam_core import shapes as s


def flat(path, res=32):
    return g.flatten_path(path, res)


def test_rect_geometry_and_handles():
    p = s.rect(20, 10, cx=5, cy=-3)
    assert p.closed and len(p.points) == 4
    ring = flat(p)
    assert ring == [(-5.0, -8.0), (15.0, -8.0), (15.0, 2.0), (-5.0, 2.0)]
    assert g.is_ccw(ring) and g.signed_area(ring) == pytest.approx(200.0)
    for bp in p.points:
        assert (bp.handle_left_type, bp.handle_right_type) == ("VECTOR", "VECTOR")
    # VECTOR handles sit a third of the way along the adjacent edges
    assert p.points[0].handle_right == pytest.approx((-5.0 + 20 / 3, -8.0))
    assert p.points[0].handle_left == pytest.approx((-5.0, -8.0 + 10 / 3))


def test_rounded_rect_area_and_clamp():
    w, h, r = 40.0, 20.0, 5.0
    p = s.rounded_rect(w, h, r)
    assert len(p.points) == 8
    area = g.signed_area(flat(p, 64))
    assert area == pytest.approx(w * h - (4 - math.pi) * r * r, rel=1e-3)
    assert g.bbox([flat(p)]) == pytest.approx((-20.0, -10.0, 20.0, 10.0), abs=1e-9)
    clamped = s.rounded_rect(w, h, 999.0)  # clamps to h/2 -> stadium
    assert g.bbox([flat(clamped)]) == pytest.approx((-20.0, -10.0, 20.0, 10.0), abs=1e-9)
    stadium_area = (w - h) * h + math.pi * (h / 2) ** 2
    assert g.signed_area(flat(clamped, 64)) == pytest.approx(stadium_area, rel=1e-3)
    assert len(clamped.points) == 6  # zero-length vertical edges merge their end points
    for radius in (0.0, -3.0):
        q = s.rounded_rect(w, h, radius)
        assert len(q.points) == 4 and all(bp.handle_left_type == "VECTOR" for bp in q.points)
    circle_like = s.rounded_rect(10, 10, 50)
    assert len(circle_like.points) == 4
    assert g.signed_area(flat(circle_like, 64)) == pytest.approx(math.pi * 25, rel=1e-3)


def test_rounded_rect_is_smooth_at_arc_joins():
    p = s.rounded_rect(40, 20, 5)
    for bp in p.points:
        assert (bp.handle_left_type, bp.handle_right_type) == ("ALIGNED", "ALIGNED")
        lx, ly = bp.handle_left[0] - bp.co[0], bp.handle_left[1] - bp.co[1]
        rx, ry = bp.handle_right[0] - bp.co[0], bp.handle_right[1] - bp.co[1]
        assert abs(lx * ry - ly * rx) < 1e-9  # collinear handles
        assert lx * rx + ly * ry < 0  # pointing opposite ways


def test_ellipse_uses_kappa_handles():
    p = s.ellipse(30, 10, cx=1, cy=2)
    assert len(p.points) == 4 and p.closed
    right = p.points[0]
    assert right.co == (31.0, 2.0)
    assert right.handle_right == pytest.approx((31.0, 2.0 + s.KAPPA * 10))
    assert right.handle_left == pytest.approx((31.0, 2.0 - s.KAPPA * 10))
    top = p.points[1]
    assert top.co == (1.0, 12.0)
    assert top.handle_right == pytest.approx((1.0 - s.KAPPA * 30, 12.0))
    assert all(bp.handle_left_type == "ALIGNED" for bp in p.points)
    ring = flat(p, 64)
    assert g.is_ccw(ring)
    assert g.signed_area(ring) == pytest.approx(math.pi * 300, rel=2e-4)
    # max radial error of the kappa approximation is about 0.03%
    for x, y in ring:
        assert ((x - 1) / 30) ** 2 + ((y - 2) / 10) ** 2 == pytest.approx(1.0, abs=1e-3)


def test_polygon_orientation_and_vertices():
    p = s.polygon(6, 10, cx=2, cy=3)
    ring = flat(p)
    assert len(ring) == 6 and g.is_ccw(ring)
    assert ring[0] == pytest.approx((2.0, 13.0))  # first vertex points up
    for x, y in ring:
        assert math.hypot(x - 2, y - 3) == pytest.approx(10.0)
    assert g.signed_area(ring) == pytest.approx(3 * math.sqrt(3) / 2 * 100, rel=1e-9)
    rot = flat(s.polygon(4, 10, rotation=math.pi / 4))
    assert rot[0] == pytest.approx((-10 * math.sin(math.pi / 4), 10 * math.cos(math.pi / 4)))
    with pytest.raises(ValueError):
        s.polygon(2, 10)


def test_star_alternates_radii():
    p = s.star(5, 10, inner_ratio=0.4)
    ring = flat(p)
    assert len(ring) == 10 and g.is_ccw(ring)
    radii = [math.hypot(x, y) for x, y in ring]
    assert radii[0::2] == pytest.approx([10.0] * 5)
    assert radii[1::2] == pytest.approx([4.0] * 5)
    assert ring[0] == pytest.approx((0.0, 10.0))
    with pytest.raises(ValueError):
        s.star(2, 10)


def test_line_is_open_two_point_path():
    p = s.line((1, 2), (4, 6))
    assert not p.closed and len(p.points) == 2
    assert flat(p) == [(1.0, 2.0), (4.0, 6.0)]
    assert g.path_length(flat(p)) == pytest.approx(5.0)


def test_from_kind_dispatch_and_fit():
    assert flat(s.from_kind("RECT", 20, 10, cx=1, cy=1)) == flat(s.rect(20, 10, 1, 1))
    assert len(s.from_kind("ROUNDED_RECT", 20, 10, radius=2).points) == 8
    e = s.from_kind("ELLIPSE", 40, 20, cx=3, cy=4)
    assert g.bbox([flat(e)]) == pytest.approx((-17.0, -6.0, 23.0, 14.0), abs=1e-9)
    for kind in ("POLYGON", "STAR"):
        p = s.from_kind(kind, 40, 20, sides=6, cx=5, cy=-5)
        ring = flat(p)
        # radius = min(w, h)/2 = 10 vertically; stretched to 20 horizontally; centred on (cx, cy)
        xs = [x for x, _ in ring]
        ys = [y for _, y in ring]
        assert max(ys) - (-5) == pytest.approx(10.0)
        assert max(abs(x - 5) for x in xs) <= 20.0 + 1e-9
        assert g.is_ccw(ring)
    sq = s.from_kind("POLYGON", 10, 10, sides=4)
    assert flat(sq)[0] == pytest.approx((0.0, 5.0))
    star = s.from_kind("STAR", 10, 10, sides=5, inner_ratio=0.25)
    assert len(star.points) == 10
    line = s.from_kind("LINE", 30, 0, cx=10, cy=2)
    assert flat(line) == [(-5.0, 2.0), (25.0, 2.0)]
    for bad in ("NONE", "BLOB", ""):
        with pytest.raises(ValueError):
            s.from_kind(bad, 10, 10)
    with pytest.raises(ValueError):
        s.from_kind("POLYGON", 0, 10)


def test_non_finite_parameters_raise():
    nan = float("nan")
    for call in (lambda: s.rect(nan, 1), lambda: s.rounded_rect(1, 1, nan), lambda: s.ellipse(1, float("inf")),
                 lambda: s.polygon(5, nan), lambda: s.star(5, 1, nan), lambda: s.line((0, nan), (1, 1)),
                 lambda: s.from_kind("RECT", nan, 1)):
        with pytest.raises(ValueError):
            call()
