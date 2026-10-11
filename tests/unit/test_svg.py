import math
import os
import re

import pytest

from seam_core import svg as S
from seam_core.geometry import BezierPath, BezierPoint

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIX = os.path.join(ROOT, "resources", "fixtures", "svg")


def fixture(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as fh:
        return fh.read()


def flat(path, res=24):
    return S._flatten(path, res)


def approx_pt(a, b, tol=1e-6):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def line_path(*pts, closed=False):
    return BezierPath([BezierPoint(p, p, p, "VECTOR", "VECTOR") for p in pts], closed)


# ---- path data parser ------------------------------------------------------------------------

def test_absolute_and_relative_lines():
    p = S.parse_path_data("M 10 20 L 30 20 H 50 V 40 l 5 5 h -5 v 5 Z")[0]
    assert [pt.co for pt in p.points] == [(10, 20), (30, 20), (50, 20), (50, 40), (55, 45), (50, 45), (50, 50)]
    assert p.closed


def test_implicit_repeats_moveto_becomes_lineto():
    p = S.parse_path_data("M0 0 10 0 10 10 m 5 5 1 1 2 2")
    assert len(p) == 2
    assert [pt.co for pt in p[0].points] == [(0, 0), (10, 0), (10, 10)]
    # relative moveto starts a new subpath from the current point (10,10); then implicit relative linetos
    assert [pt.co for pt in p[1].points] == [(15, 15), (16, 16), (18, 18)]
    q = S.parse_path_data("M0 0 L1 1 2 2 3 3")[0]
    assert len(q.points) == 4
    c = S.parse_path_data("M0 0 c1 0 1 1 2 1 1 0 1 1 2 1")[0]
    assert len(c.points) == 3 and c.points[2].co == (4, 2)


def test_number_syntax_variants():
    a = S.parse_path_data("M1e1,2E+1 L-.5.5 1-2 +3.5e-1 .25")[0]
    assert [pt.co for pt in a.points] == [(10.0, 20.0), (-0.5, 0.5), (1.0, -2.0), (0.35, 0.25)]
    b = S.parse_path_data("M10,20L30,40,50,60")[0]
    assert [pt.co for pt in b.points] == [(10, 20), (30, 40), (50, 60)]
    c = S.parse_path_data("  M\n10\t20\r\nL 30\n40 ")[0]
    assert [pt.co for pt in c.points] == [(10, 20), (30, 40)]


def test_cubic_and_smooth_cubic():
    p = S.parse_path_data("M0 0 C 0 10 10 10 10 0 S 20 -10 20 0")[0]
    assert p.points[1].handle_left == (10, 10) and p.points[0].handle_right == (0, 10)
    # reflected first control of S = 2*current - previous c2 = (10,-10)
    assert p.points[1].handle_right == (10, -10)
    assert p.points[2].handle_left == (20, -10) and p.points[2].co == (20, 0)
    # S after a non-curve uses the current point as first control
    q = S.parse_path_data("M0 0 L10 0 S 20 10 30 0")[0]
    assert q.points[1].handle_right == (10, 0)
    r = S.parse_path_data("M0 0 c0 5 5 5 5 0 s5 -5 5 0")[0]
    assert r.points[1].handle_right == (5, -5)
    assert r.points[2].co == (10, 0)


def test_quadratic_and_smooth_quadratic_as_cubics():
    p = S.parse_path_data("M0 0 Q 5 10 10 0 T 20 0")[0]
    # first quad: cubic controls at 2/3 toward the quad control
    assert approx_pt(p.points[0].handle_right, (10 / 3, 20 / 3))
    assert approx_pt(p.points[1].handle_left, (10 - 10 / 3, 20 / 3))
    # T reflects the quad control (5,10) about (10,0) -> (15,-10)
    assert approx_pt(p.points[1].handle_right, (10 + 2 / 3 * 5, -2 / 3 * 10))
    assert p.points[2].co == (20, 0)
    # quad evaluates to the same curve: midpoint of Q is (5, 5)
    mid = S._cubic(p.points[0].co, p.points[0].handle_right, p.points[1].handle_left, p.points[1].co, 0.5)
    assert approx_pt(mid, (5, 5))
    t_only = S.parse_path_data("M0 0 T 10 10")[0]  # no previous quad: control = current point
    assert t_only.points[1].co == (10, 10)


def test_arc_circle_flattening_within_half_pixel():
    r, cx, cy = 50.0, 100.0, 80.0
    # full circle from two arcs, absolute
    p = S.parse_path_data("M %g %g A %g %g 0 1 0 %g %g A %g %g 0 1 0 %g %g Z" % (cx + r, cy, r, r, cx - r, cy, r, r, cx + r, cy))
    assert len(p) == 1 and p[0].closed
    pts = flat(p[0], 16)
    assert len(pts) >= 16
    worst = max(abs(math.hypot(x - cx, y - cy) - r) for x, y in pts)
    assert worst < 0.5
    # sample the dense flatten for the true max error
    dense = flat(p[0], 64)
    assert max(abs(math.hypot(x - cx, y - cy) - r) for x, y in dense) < 0.05


@pytest.mark.parametrize("fa,fs", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_arc_flag_combinations_pass_through_endpoints_and_center(fa, fs):
    # Arc of radius 10 between (0,0) and (10,10): centre is one of (0,10) or (10,0)
    p = S.parse_path_data("M0 0 A10 10 0 %d %d 10 10" % (fa, fs))[0]
    pts = flat(p, 32)
    assert approx_pt(pts[0], (0, 0)) and approx_pt(pts[-1], (10, 10), 1e-9)
    centers = [(0, 10), (10, 0)]
    best = min(max(abs(math.hypot(x - cx, y - cy) - 10) for x, y in pts) for cx, cy in centers)
    assert best < 0.02
    large = fa == 1
    # large-arc flag selects the longer arc
    length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    assert (length > 10 * math.pi / 2 + 1) == large


def test_arc_compact_flags_and_scaling_and_rotation():
    a = S.parse_path_data("M0 0a5 5 0 00 10 0")[0]  # flags '0','0' packed, then 10 0
    assert a.points[-1].co == (10, 0)
    # radii too small get scaled up so the arc still reaches the end point
    b = S.parse_path_data("M0 0 A1 1 0 0 1 10 0")[0]
    pts = flat(b, 32)
    assert approx_pt(pts[-1], (10, 0), 1e-9)
    assert max(abs(math.hypot(x - 5, y)) for x, y in pts) == pytest.approx(5.0, abs=0.02)
    # zero radius -> straight line, coincident endpoints -> omitted
    c = S.parse_path_data("M0 0 A0 5 0 0 1 10 0")[0]
    assert [pt.co for pt in c.points] == [(0, 0), (10, 0)] and c.points[0].handle_right == (0, 0)
    d = S.parse_path_data("M0 0 L5 5 A5 5 0 0 1 5 5 L10 0")[0]
    assert [pt.co for pt in d.points] == [(0, 0), (5, 5), (10, 0)]
    # rotated ellipse: x-axis rotated 90 degrees swaps the radii
    e = S.parse_path_data("M0 0 A20 10 90 0 1 20 0")[0]
    pts = flat(e, 32)
    assert approx_pt(pts[-1], (20, 0), 1e-9)


def test_close_merges_duplicate_end_and_restarts_after_z():
    p = S.parse_path_data("M0 0 L10 0 L10 10 L0 0 Z")
    assert len(p[0].points) == 3 and p[0].closed
    # drawing command right after Z starts a new subpath at the previous start
    q = S.parse_path_data("M5 5 L10 5 L10 10 Z L0 0 L1 1")
    assert len(q) == 2
    assert q[1].points[0].co == (5, 5) and not q[1].closed
    # closed bezier circle keeps incoming handle on the first point
    c = S.parse_path_data("M10 0 C10 5 5 10 0 10 C-5 10 -10 5 -10 0 C-10 -5 -5 -10 0 -10 C5 -10 10 -5 10 0 Z")[0]
    assert len(c.points) == 4 and c.points[0].handle_left == (10, -5)


def test_handle_types():
    p = S.parse_path_data("M0 0 L10 0 C10 5 15 5 20 0")[0]
    assert p.points[0].handle_right_type == "VECTOR" and p.points[1].handle_right_type == "FREE"
    assert p.points[2].handle_left_type == "FREE" and p.points[2].handle_right_type == "VECTOR"


@pytest.mark.parametrize("bad", ["L 10 10", "M 0 0 L 10", "M 0 0 X 1 2", "M 0 0 A 1 1 0 2 0 5 5", "10 10 L 5 5", "M 0 0 Z 5"])
def test_parse_errors_raise_value_error(bad):
    with pytest.raises(ValueError):
        S.parse_path_data(bad)


def test_empty_and_lonely_moveto():
    assert S.parse_path_data("") == []
    assert S.parse_path_data("M 5 5") == []


def test_format_round_trip():
    d = "M0 0 L10 0 C10 5 15 5 20 0 L20 10 Z M100 100 L110 100 L110 110"
    paths = S.parse_path_data(d)
    out = S.format_path_data(paths)
    assert out == d
    again = S.parse_path_data(out)
    assert again == paths
    assert S.format_path_data([line_path((0.123456, -0.0004), (1.0, 2.5))], precision=2) == "M0.12 0 L1 2.5"
    assert "-0 " not in S.format_path_data([line_path((-0.0001, 1), (2, 3))])


# ---- transforms ----------------------------------------------------------------------------------

def test_parse_transform_primitives():
    assert S.parse_transform("") == (1, 0, 0, 1, 0, 0)
    assert S.parse_transform("translate(10 20)") == (1, 0, 0, 1, 10, 20)
    assert S.parse_transform("translate(5)") == (1, 0, 0, 1, 5, 0)
    assert S.parse_transform("scale(2)") == (2, 0, 0, 2, 0, 0)
    assert S.parse_transform("scale(2,3)") == (2, 0, 0, 3, 0, 0)
    assert S.parse_transform("matrix(1 2 3 4 5 6)") == (1, 2, 3, 4, 5, 6)
    r = S.parse_transform("rotate(90)")
    assert approx_pt(S._apply(r, (1, 0)), (0, 1))
    r2 = S.parse_transform("rotate(90 10 10)")
    assert approx_pt(S._apply(r2, (10, 10)), (10, 10)) and approx_pt(S._apply(r2, (11, 10)), (10, 11))
    assert approx_pt(S._apply(S.parse_transform("skewX(45)"), (0, 1)), (1, 1))
    assert approx_pt(S._apply(S.parse_transform("skewY(45)"), (1, 0)), (1, 1))


def test_parse_transform_composition_order():
    # translate then scale: scale applies first to the point, translate after
    m = S.parse_transform("translate(10 0) scale(2)")
    assert approx_pt(S._apply(m, (1, 1)), (12, 2))
    m2 = S.parse_transform("scale(2) translate(10 0)")
    assert approx_pt(S._apply(m2, (1, 1)), (22, 2))
    m3 = S.parse_transform("translate(1,2),scale(3)")
    assert approx_pt(S._apply(m3, (1, 1)), (4, 5))


@pytest.mark.parametrize("bad", ["translate(", "foo(1)", "scale(a)", "rotate(1 2)", "matrix(1 2 3)", "translate(1 2 3)"])
def test_parse_transform_errors(bad):
    with pytest.raises(ValueError):
        S.parse_transform(bad)


# ---- import ---------------------------------------------------------------------------------------

def shape_by_name(res, name):
    for s in res.shapes:
        if s.name == name:
            return s
    raise KeyError(name)


def lin(c8):
    c = c8 / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def test_import_basic_shapes_fixture():
    res = S.import_svg(fixture("basic_shapes.svg"))
    assert (res.width, res.height) == (200, 100)
    assert res.warnings == []
    names = [s.name for s in res.shapes]
    assert names == ["box", "round", "dot", "oval", "rule", "zig", "tri", "inner"]
    box = shape_by_name(res, "box")
    pts = flat(box.paths[0])
    # SVG y is flipped: svg (10,10) -> canvas (10,-10)
    assert sorted(pts) == sorted([(10, -10), (50, -10), (50, -30), (10, -30)])
    assert box.style.fill == pytest.approx((lin(0x33), lin(0x66), lin(0x99), 1.0))
    assert box.style.stroke == pytest.approx((0, 0, 0, 1.0))
    assert box.style.stroke_width == 2
    rnd = shape_by_name(res, "round")
    assert rnd.style.fill[3] == pytest.approx(0.5)
    assert rnd.style.fill[:3] == pytest.approx((1.0, lin(128), 0.0))
    assert len(rnd.paths[0].points) == 8  # four straight edges + four corner arcs (one cubic each)
    xs = [p[0] for p in flat(rnd.paths[0])]
    assert min(xs) == pytest.approx(60) and max(xs) == pytest.approx(100)
    dot = shape_by_name(res, "dot")  # style attribute
    assert dot.style.fill == pytest.approx((0, 1, 0, 1))
    assert dot.style.stroke == pytest.approx((1, 0, 0, 1))
    assert dot.style.stroke_width == 3 and dot.style.stroke_cap == "ROUND" and dot.style.stroke_join == "BEVEL"
    assert dot.style.opacity == pytest.approx(0.75)
    pts = flat(dot.paths[0], 32)
    assert max(abs(math.hypot(x - 130, y + 20) - 10) for x, y in pts) < 0.05
    oval = shape_by_name(res, "oval")
    xs = [p[0] for p in flat(oval.paths[0], 32)]
    ys = [p[1] for p in flat(oval.paths[0], 32)]
    assert (min(xs), max(xs)) == pytest.approx((155, 185), abs=0.05) and (min(ys), max(ys)) == pytest.approx((-28, -12), abs=0.05)
    rule = shape_by_name(res, "rule")
    assert not rule.paths[0].closed and rule.style.stroke_cap == "SQUARE" and rule.style.stroke_width == 4
    zig = shape_by_name(res, "zig")
    assert zig.style.fill is None and len(zig.paths[0].points) == 4 and not zig.paths[0].closed
    tri = shape_by_name(res, "tri")
    assert tri.paths[0].closed and tri.style.fill_rule == "EVENODD"
    assert tri.style.fill == pytest.approx((lin(0xaa), lin(0xbb), lin(0xcc), 1))


def test_import_group_transform_scales_geometry_and_stroke():
    res = S.import_svg(fixture("basic_shapes.svg"))
    inner = shape_by_name(res, "inner")
    pts = flat(inner.paths[0])
    # translate(20 70) scale(2): (0,0)->(20,70), (10,0)->(40,70), (10,10)->(40,90); y flipped
    assert sorted(pts) == sorted([(20, -70), (40, -70), (40, -90)])
    assert inner.style.stroke_width == pytest.approx(2.0)
    assert inner.style.fill == pytest.approx((lin(128), 0, lin(128), 1))  # inherited from <g>
    assert inner.paths[0].closed


def test_import_style_attribute_overrides_presentation_attribute():
    res = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="5" height="5" fill="red" '
                       'style="fill:#0000ff; stroke : #00ff00 ;stroke-width:2px"/></svg>')
    s = res.shapes[0]
    assert s.style.fill == pytest.approx((0, 0, 1, 1)) and s.style.stroke == pytest.approx((0, 1, 0, 1)) and s.style.stroke_width == 2


def test_import_colour_forms():
    def fill_of(v):
        r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"><rect width="1" height="1" fill="%s"/></svg>' % v)
        return r.shapes[0].style.fill, r.warnings
    assert fill_of("#fff")[0] == pytest.approx((1, 1, 1, 1))
    assert fill_of("#FF0000")[0] == pytest.approx((1, 0, 0, 1))
    assert fill_of("rgb(0, 0, 255)")[0] == pytest.approx((0, 0, 1, 1))
    assert fill_of("rgb(100%, 0%, 0%)")[0] == pytest.approx((1, 0, 0, 1))
    assert fill_of("navy")[0] == pytest.approx((0, 0, lin(128), 1))
    assert fill_of("none")[0] is None
    assert fill_of("#ff000080")[0][3] == pytest.approx(128 / 255)
    val, warns = fill_of("notacolour")
    assert warns and "notacolour" in warns[0]
    cc = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"><g color="#00ff00"><rect width="1" height="1" fill="currentColor"/></g></svg>')
    assert cc.shapes[0].style.fill == pytest.approx((0, 1, 0, 1))


def test_import_fill_defaults_and_inheritance():
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><g stroke="red" stroke-width="3">'
                     '<rect width="5" height="5"/><rect width="5" height="5" stroke="none"/></g></svg>')
    a, b = r.shapes
    assert a.style.fill == pytest.approx((0, 0, 0, 1)) and a.style.stroke == pytest.approx((1, 0, 0, 1))
    assert b.style.stroke is None


def test_import_size_viewbox_and_units():
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" width="2in" height="1in" viewBox="0 0 20 10"><rect width="20" height="10"/></svg>')
    assert (r.width, r.height) == (192, 96)
    xs = [p[0] for p in flat(r.shapes[0].paths[0])]
    assert max(xs) == pytest.approx(192)  # viewBox scale 9.6
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="10 20 100 50"><rect x="10" y="20" width="100" height="50"/></svg>')
    assert (r.width, r.height) == (100, 50)
    pts = flat(r.shapes[0].paths[0])
    assert (min(p[0] for p in pts), max(p[1] for p in pts)) == pytest.approx((0, 0))
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" width="30" height="20"><rect width="1" height="1"/></svg>')
    assert (r.width, r.height) == (30, 20) and r.warnings == []
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>')
    assert r.warnings  # missing size is reported
    # preserveAspectRatio: meet, centred
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>')
    xs = [p[0] for p in flat(r.shapes[0].paths[0])]
    assert (min(xs), max(xs)) == pytest.approx((50, 150))


def test_import_gradients_fixture():
    res = S.import_svg(fixture("gradients.svg"))
    assert res.warnings == []
    a = shape_by_name(res, "a").style.fill
    assert isinstance(a, S.SvgGradient) and a.kind == "LINEAR"
    # objectBoundingBox x 0..1 over the rect 0..40, y mid = 20 (svg) -> -20 (canvas)
    assert a.start == pytest.approx((0, -0.0)) or a.start == pytest.approx((0, 0))
    assert a.end == pytest.approx((40, 0))
    assert [round(p, 3) for p, _ in a.stops] == [0.0, 0.5, 1.0]
    assert a.stops[0][1] == pytest.approx((1, 0, 0, 1))
    assert a.stops[1][1] == pytest.approx((0, 1, 0, 0.5))
    assert a.stops[2][1] == pytest.approx((0, 0, 1, 1))
    b = shape_by_name(res, "b").style.fill  # userSpaceOnUse + stops inherited through xlink:href
    assert b.kind == "LINEAR" and b.start == pytest.approx((50, 0)) and b.end == pytest.approx((50, -100))
    assert len(b.stops) == 3
    c = shape_by_name(res, "c").style.fill
    assert c.kind == "RADIAL" and c.center == pytest.approx((25, -75)) and c.radius == pytest.approx(20)
    assert c.stops[0][1] == pytest.approx((1, 1, 1, 1)) and c.stops[1][1] == pytest.approx((0, 0, 0, 1))


def test_import_gradient_transform_and_fill_opacity():
    src = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs>'
           '<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="10" y2="0" gradientTransform="translate(5 5) scale(2)">'
           '<stop offset="0" stop-color="black"/><stop offset="1" stop-color="white"/></linearGradient></defs>'
           '<rect width="50" height="50" fill="url(#g)" fill-opacity="0.5"/></svg>')
    g = S.import_svg(src).shapes[0].style.fill
    assert g.start == pytest.approx((5, -5)) and g.end == pytest.approx((25, -5))
    assert g.stops[0][1][3] == pytest.approx(0.5)


def test_import_single_stop_gradient_is_solid_and_missing_server_warns():
    src = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><defs><linearGradient id="g"><stop offset="0" stop-color="#ff0000"/>'
           '</linearGradient></defs><rect width="5" height="5" fill="url(#g)"/><rect width="5" height="5" fill="url(#nope)"/></svg>')
    r = S.import_svg(src)
    assert r.shapes[0].style.fill == pytest.approx((1, 0, 0, 1))
    assert r.shapes[1].style.fill is None and any("nope" in w for w in r.warnings)


def test_unsupported_elements_warn_never_silent():
    res = S.import_svg(fixture("unsupported.svg"))
    text = "\n".join(res.warnings)
    for needle in ("filter", "mask", "pattern", "use", "text", "dasharray", "image"):
        assert needle in text, (needle, res.warnings)
    # supported geometry is still imported
    names = {s.name for s in res.shapes}
    assert {"blurred", "masked", "patterned", "dashed"} <= names
    # the pattern fill didn't invent a colour
    assert shape_by_name(res, "patterned").style.fill is None
    # dashed stroke imported solid
    assert shape_by_name(res, "dashed").style.stroke is not None


def test_style_element_and_group_opacity_and_bad_input_warn():
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><style>.a{fill:red}</style>'
                     '<g opacity="0.5"><rect width="1" height="1"/><rect width="2" height="2"/></g></svg>')
    txt = " ".join(r.warnings)
    assert "style" in txt and "opacity" in txt
    assert r.shapes[0].style.opacity == pytest.approx(0.5)
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M 0 0 L"/><path d="M0 0 L5 5"/></svg>')
    assert len(r.shapes) == 1 and r.warnings
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><g transform="bogus(1)"><rect width="1" height="1"/></g></svg>')
    assert not r.shapes and r.warnings


def test_display_none_and_hidden_are_honoured_silently():
    r = S.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect display="none" width="1" height="1"/>'
                     '<g style="display:none"><rect width="1" height="1"/></g><rect visibility="hidden" width="1" height="1"/>'
                     '<rect width="2" height="2"/></svg>')
    assert len(r.shapes) == 1 and r.warnings == []


@pytest.mark.parametrize("bad", ["", "not xml", "<html></html>", "<svg>", '<!DOCTYPE svg [<!ENTITY a "b">]><svg xmlns="http://www.w3.org/2000/svg"/>'])
def test_import_rejects_non_svg(bad):
    with pytest.raises(ValueError):
        S.import_svg(bad)


def test_import_ignores_foreign_namespaces_and_metadata_silently():
    src = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
           'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" viewBox="0 0 10 10"><title>t</title><desc>d</desc>'
           '<metadata><x/></metadata><sodipodi:namedview id="nv"/><rect inkscape:label="Layer rect" width="1" height="1"/></svg>')
    r = S.import_svg(src)
    assert r.warnings == [] and r.shapes[0].name == "Layer rect"


# ---- clip paths ---------------------------------------------------------------------------------

def test_import_clip_fixture():
    res = S.import_svg(fixture("clip.svg"))
    assert res.warnings == []
    a = shape_by_name(res, "clipped")
    assert a.clip_paths is not None and len(a.clip_paths) == 1
    xs = [p[0] for p in flat(a.clip_paths[0])]
    ys = [p[1] for p in flat(a.clip_paths[0])]
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((0, 30, -30, 0))
    g = shape_by_name(res, "in_group")
    xs = [p[0] for p in flat(g.clip_paths[0])]
    ys = [p[1] for p in flat(g.clip_paths[0])]
    # clip is in the group's coordinate system (translate(50 50) applies to the clip too)
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((50, 80, -80, -50))


# ---- export -----------------------------------------------------------------------------------------

def rect_shape(name, x0, y0, x1, y1, style=None, clip=None):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return S.SvgShape(name, [line_path(*pts, closed=True)], style or S.SvgStyle(), clip)


def parse_export(text):
    import xml.etree.ElementTree as ET
    return ET.fromstring(text)


NS = "{http://www.w3.org/2000/svg}"


def test_export_structure_and_coordinates():
    st = S.SvgStyle(fill=(1.0, 0.0, 0.0, 1.0), stroke=(0.0, 0.0, 0.0, 1.0), stroke_width=2.5, stroke_cap="ROUND",
                    stroke_join="BEVEL", miter_limit=3.0, opacity=0.5, fill_rule="EVENODD")
    out = S.export_svg([rect_shape("My Rect", 110, 220, 150, 190, st)], (100, 250, 80, 90))
    root = parse_export(out)
    assert root.get("viewBox") == "0 0 80 90" and root.get("width") == "80" and root.get("height") == "90"
    p = root.find(NS + "path")
    assert p.get("d") == "M10 30 L50 30 L50 60 L10 60 Z"
    assert p.get("fill") == "#ff0000" and p.get("fill-rule") == "evenodd" and p.get("stroke") == "#000000"
    assert p.get("stroke-width") == "2.5" and p.get("stroke-linecap") == "round" and p.get("stroke-linejoin") == "bevel"
    assert p.get("stroke-miterlimit") == "3" and p.get("opacity") == "0.5"
    assert p.get("data-name") == "My Rect" and p.get("id") == "My_Rect"


def test_export_none_fill_alpha_and_rgb_for_non_8bit():
    st = S.SvgStyle(fill=None, stroke=(0.5, 0.5, 0.5, 0.25), stroke_width=1)
    root = parse_export(S.export_svg([rect_shape("a", 0, 0, 1, -1, st)], (0, 0, 10, 10)))
    p = root.find(NS + "path")
    assert p.get("fill") == "none" and p.get("fill-rule") is None
    assert p.get("stroke-opacity") == "0.25"
    # 0.5 linear is not an exact 8-bit sRGB value -> precise percentage form
    assert p.get("stroke").startswith("rgb(") and p.get("stroke").endswith("%)")


def test_export_gradients_defs():
    g = S.SvgGradient("LINEAR", [(0.0, (1, 0, 0, 1)), (1.0, (0, 0, 1, 0.5))], start=(0, 0), end=(40, 0))
    r = S.SvgGradient("RADIAL", [(0.0, (1, 1, 1, 1)), (1.0, (0, 0, 0, 1))], center=(20, -20), radius=15)
    items = [rect_shape("a", 0, 0, 40, -40, S.SvgStyle(fill=g)), rect_shape("b", 0, 0, 40, -40, S.SvgStyle(fill=r)),
             rect_shape("c", 0, 0, 40, -40, S.SvgStyle(fill=g))]
    root = parse_export(S.export_svg(items, (0, 0, 40, 40)))
    lg = root.find(NS + "defs").findall(NS + "linearGradient")
    assert len(lg) == 1  # identical gradients are shared
    assert lg[0].get("gradientUnits") == "userSpaceOnUse" and (lg[0].get("x1"), lg[0].get("x2")) == ("0", "40")
    stops = lg[0].findall(NS + "stop")
    assert [s.get("stop-color") for s in stops] == ["#ff0000", "#0000ff"] and stops[1].get("stop-opacity") == "0.5"
    rg = root.find(NS + "defs").find(NS + "radialGradient")
    assert (rg.get("cx"), rg.get("cy"), rg.get("r")) == ("20", "20", "15")
    paths = root.findall(NS + "path")
    assert paths[0].get("fill") == "url(#%s)" % lg[0].get("id") == paths[2].get("fill")


def test_export_clippath():
    clip = [line_path((0, 0), (30, 0), (30, -30), (0, -30), closed=True)]
    root = parse_export(S.export_svg([rect_shape("a", 0, 0, 60, -60, clip=clip)], (0, 0, 60, 60)))
    cp = root.find(NS + "defs").find(NS + "clipPath")
    assert cp is not None and cp.find(NS + "path").get("d") == "M0 0 L30 0 L30 30 L0 30 Z"
    assert root.find(NS + "path").get("clip-path") == "url(#%s)" % cp.get("id")
    # an empty clip list means "clip everything" and is still exported
    root = parse_export(S.export_svg([rect_shape("a", 0, 0, 60, -60, clip=[])], (0, 0, 60, 60)))
    assert root.find(NS + "path").get("clip-path")


def test_export_image_and_text_modes():
    import base64
    png = b"\x89PNG\r\n\x1a\nxxxx"
    img = S.SvgImage("pic", png, 10, -5, 20, 10, opacity=0.5)
    root = parse_export(S.export_svg([img], (0, 0, 100, 100)))
    im = root.find(NS + "image")
    assert (im.get("x"), im.get("y"), im.get("width"), im.get("height")) == ("10", "5", "20", "10")
    assert im.get("href") == "data:image/png;base64," + base64.b64encode(png).decode() and im.get("opacity") == "0.5"
    t = rect_shape("title", 10, -20, 60, -10)
    t.text = "Hello & <bye>"
    t.font_family = "Arial"
    t.font_size = 14
    outlines = parse_export(S.export_svg([t], (0, 0, 100, 100)))
    assert outlines.find(NS + "path") is not None and outlines.find(NS + "text") is None
    as_text = parse_export(S.export_svg([t], (0, 0, 100, 100), text_mode="TEXT"))
    tx = as_text.find(NS + "text")
    assert tx.text == "Hello & <bye>" and tx.get("font-family") == "Arial" and tx.get("font-size") == "14"
    assert (tx.get("x"), tx.get("y")) == ("10", "20")
    # shapes without text stay paths in TEXT mode
    both = parse_export(S.export_svg([rect_shape("p", 0, 0, 1, -1)], (0, 0, 10, 10), text_mode="TEXT"))
    assert both.find(NS + "path") is not None


def test_export_validation():
    with pytest.raises(ValueError):
        S.export_svg([], (0, 0, 10, 10), text_mode="BOGUS")
    with pytest.raises(ValueError):
        S.export_svg([], (0, 0, 0, 10))
    with pytest.raises(TypeError):
        S.export_svg([object()], (0, 0, 10, 10))
    assert parse_export(S.export_svg([], (0, 0, 10, 10))).get("width") == "10"


def test_export_unique_ids_for_duplicate_names():
    root = parse_export(S.export_svg([rect_shape("a", 0, 0, 1, -1), rect_shape("a", 0, 0, 1, -1), rect_shape("", 0, 0, 1, -1)], (0, 0, 5, 5)))
    ids = [p.get("id") for p in root.findall(NS + "path")]
    assert len(set(ids)) == 3 and all(re.match(r"[A-Za-z_]", i) for i in ids)


# ---- round trip ----------------------------------------------------------------------------------------

def curved_path():
    pts = [
        BezierPoint((0.0, 0.0), (-10.0, 5.0), (10.0, -5.0), "FREE", "FREE"),
        BezierPoint((50.0, -20.0), (40.0, -20.0), (60.0, -25.0), "FREE", "FREE"),
        BezierPoint((80.0, -80.0), (80.0, -80.0), (80.0, -80.0), "VECTOR", "VECTOR"),
        BezierPoint((10.0, -60.0), (30.0, -90.0), (10.0, -60.0), "FREE", "VECTOR"),
    ]
    return BezierPath(pts, True)


def assert_paths_close(a, b, tol=1e-3):
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert pa.closed == pb.closed and len(pa.points) == len(pb.points)
        for x, y in zip(pa.points, pb.points):
            for u, v in ((x.co, y.co), (x.handle_left, y.handle_left), (x.handle_right, y.handle_right)):
                assert abs(u[0] - v[0]) <= tol and abs(u[1] - v[1]) <= tol, (u, v)


def test_round_trip_geometry_style_gradient_clip():
    grad = S.SvgGradient("LINEAR", [(0.0, (1.0, 0.0, 0.0, 1.0)), (0.4, (0.0, 1.0, 0.0, 0.5)), (1.0, (0.0, 0.0, 1.0, 1.0))],
                         start=(0.0, -10.0), end=(80.0, -70.0))
    rad = S.SvgGradient("RADIAL", [(0.0, (1.0, 1.0, 1.0, 1.0)), (1.0, (0.0, 0.0, 0.0, 0.25))], center=(40.0, -40.0), radius=33.5)
    compound = [line_path((0, 0), (100, 0), (100, -100), (0, -100), closed=True),
                line_path((25, -25), (25, -75), (75, -75), (75, -25), closed=True)]
    shapes = [
        S.SvgShape("curve", [curved_path()], S.SvgStyle(fill=(0.2, 0.4, 0.6, 1.0), stroke=(1.0, 1.0, 1.0, 1.0), stroke_width=3.25,
                                                       stroke_cap="SQUARE", stroke_join="ROUND", miter_limit=2.5, opacity=0.6, fill_rule="EVENODD")),
        S.SvgShape("grad", [line_path((0, 0), (80, 0), (80, -80), (0, -80), closed=True)], S.SvgStyle(fill=grad)),
        S.SvgShape("radial", compound, S.SvgStyle(fill=rad, stroke=None, fill_rule="EVENODD")),
        S.SvgShape("open line", [line_path((5, -5), (60, -40), (70, -10))], S.SvgStyle(fill=None, stroke=(0.0, 0.0, 0.0, 0.5), stroke_width=1.5)),
        S.SvgShape("clipped", [line_path((0, 0), (50, 0), (50, -50), (0, -50), closed=True)], S.SvgStyle(fill=(0.5, 0.25, 0.125, 0.75)),
                   clip_paths=[line_path((10, -10), (40, -10), (40, -40), (10, -40), closed=True)]),
    ]
    ab = (0.0, 0.0, 120.0, 130.0)  # x_left, y_top, w, h  (canvas == artboard frame; offsets are tested separately)
    res = S.import_svg(S.export_svg(shapes, ab))
    assert res.warnings == []
    assert (res.width, res.height) == (120, 130)
    assert [s.name for s in res.shapes] == [s.name for s in shapes]
    for orig, got in zip(shapes, res.shapes):
        assert_paths_close(orig.paths, got.paths)
        if orig.clip_paths:
            assert_paths_close(orig.clip_paths, got.clip_paths)
        else:
            assert got.clip_paths is None
        so, sg = orig.style, got.style
        assert so.stroke_width == pytest.approx(sg.stroke_width) and so.stroke_cap == sg.stroke_cap
        assert so.stroke_join == sg.stroke_join and so.opacity == pytest.approx(sg.opacity, abs=1e-6)
        assert so.fill_rule == sg.fill_rule or so.fill is None
        if so.stroke is None:
            assert sg.stroke is None
        else:
            assert sg.stroke == pytest.approx(so.stroke, abs=2e-3)
            assert so.miter_limit == pytest.approx(sg.miter_limit)
        if so.fill is None:
            assert sg.fill is None
        elif isinstance(so.fill, S.SvgGradient):
            fo, fg = so.fill, sg.fill
            assert fg.kind == fo.kind
            for (po, co), (pg, cg) in zip(fo.stops, fg.stops):
                assert po == pytest.approx(pg, abs=1e-5) and cg == pytest.approx(co, abs=2e-3)
            assert len(fg.stops) == len(fo.stops)
            if fo.kind == "LINEAR":
                assert fg.start == pytest.approx(fo.start, abs=1e-3) and fg.end == pytest.approx(fo.end, abs=1e-3)
            else:
                assert fg.center == pytest.approx(fo.center, abs=1e-3) and fg.radius == pytest.approx(fo.radius, abs=1e-3)
        else:
            assert sg.fill == pytest.approx(so.fill, abs=2e-3)


def test_round_trip_geometry_with_artboard_offset():
    sh = S.SvgShape("c", [curved_path()], S.SvgStyle())
    moved = S.SvgShape("c", [S._xform_path(sh.paths[0], (1, 0, 0, 1, 300.0, 70.0))], S.SvgStyle())
    res = S.import_svg(S.export_svg([moved], (300.0, 70.0, 100, 100)))  # artboard origin at the shape origin
    assert_paths_close(sh.paths, res.shapes[0].paths)


def test_round_trip_gradient_and_clip_survive_nonzero_offsets():
    grad = S.SvgGradient("LINEAR", [(0.0, (1, 0, 0, 1)), (1.0, (0, 0, 1, 1))], start=(100, 50), end=(160, 50))
    sh = S.SvgShape("g", [line_path((100, 50), (160, 50), (160, 10), (100, 10), closed=True)], S.SvgStyle(fill=grad))
    res = S.import_svg(S.export_svg([sh], (90, 60, 100, 100)))
    g = res.shapes[0].style.fill
    assert g.start == pytest.approx((10, -10)) and g.end == pytest.approx((70, -10))


def test_import_export_import_is_stable_on_fixtures():
    for name in ("basic_shapes.svg", "gradients.svg", "clip.svg"):
        first = S.import_svg(fixture(name))
        second = S.import_svg(S.export_svg(first.shapes, (0, 0, first.width, first.height)))
        assert second.warnings == [], (name, second.warnings)
        assert len(second.shapes) == len(first.shapes)
        for a, b in zip(first.shapes, second.shapes):
            assert_paths_close(a.paths, b.paths)
            assert a.style.stroke_width == pytest.approx(b.style.stroke_width, abs=1e-4)
            if a.clip_paths is not None:
                assert_paths_close(a.clip_paths, b.clip_paths)
