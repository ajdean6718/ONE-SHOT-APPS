"""seam_core.boolean2d running on the real mathutils CDT of each Blender runtime (4.0.2 binary, 5.2.2 bpy)."""

import json
import math
import os
import platform
import time

import bpy
import pytest


def record_perf(name, ms):
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
    label = "blender-%s (python %s)" % (bpy.app.version_string, platform.python_version())
    doc.setdefault("runs", {}).setdefault(label, {})[name] = round(ms, 3)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)


@pytest.fixture
def api(seam):
    from seam_studio.seam_core import boolean2d, geometry
    assert boolean2d.available()
    return boolean2d, geometry


def sq(x, y, s):
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]


def circle(n, r, cx=0.0, cy=0.0):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def area(geometry, shapes):
    return sum(geometry.signed_area(o) + sum(geometry.signed_area(h) for h in hs) for o, hs in shapes)


def test_native_cdt_is_available_and_reports_overlap_provenance(seam):
    import mathutils
    from mathutils import Vector
    from mathutils.geometry import delaunay_2d_cdt
    verts = [Vector(p) for p in sq(0, 0, 2) + sq(1, 1, 2)]
    out_v, _e, tris, _ov, _oe, orig = delaunay_2d_cdt(verts, [], [[0, 1, 2, 3], [4, 5, 6, 7]], 1, 1e-6, True)
    assert any(sorted(o) == [0, 1] for o in orig)  # overlap triangles carry both input faces
    assert any(o == [0] for o in orig) and any(o == [1] for o in orig)
    assert mathutils.Vector((1, 2)).length > 0


@pytest.mark.parametrize("op,expected,count", [("UNION", 7.0, 1), ("DIFFERENCE", 3.0, 1), ("INTERSECT", 1.0, 1), ("EXCLUDE", 6.0, 2)])
def test_overlapping_squares_exact_areas(api, op, expected, count):
    boolean2d, geometry = api
    res = boolean2d.boolean([sq(0, 0, 2)], [sq(1, 1, 2)], op)
    assert len(res) == count
    assert area(geometry, res) == pytest.approx(expected, abs=1e-9)
    for outer, holes in res:
        assert geometry.signed_area(outer) > 0 and all(geometry.signed_area(h) < 0 for h in holes)


def test_holes_nested_touching_disjoint_identical(api):
    boolean2d, geometry = api
    outer = sq(0, 0, 10)
    hole = sq(3, 3, 4)[::-1]
    # holes preserved through a union with a far-away shape
    res = boolean2d.boolean([outer, hole], [sq(20, 20, 2)], "UNION")
    assert len(res) == 2
    holed = next(s for s in res if abs(geometry.signed_area(s[0]) - 100.0) < 1e-9)
    assert len(holed[1]) == 1 and geometry.signed_area(holed[1][0]) == pytest.approx(-16.0)
    # difference that carves into an existing hole
    diff = boolean2d.boolean([outer, hole], [sq(2, 2, 4)], "DIFFERENCE")
    assert area(geometry, diff) == pytest.approx(100.0 - 23.0)
    # disjoint
    assert boolean2d.boolean([sq(0, 0, 2)], [sq(10, 10, 2)], "INTERSECT") == []
    assert len(boolean2d.boolean([sq(0, 0, 2)], [sq(10, 10, 2)], "UNION")) == 2
    # identical
    assert area(geometry, boolean2d.boolean([sq(0, 0, 2)], [sq(0, 0, 2)], "UNION")) == pytest.approx(4.0)
    assert boolean2d.boolean([sq(0, 0, 2)], [sq(0, 0, 2)], "DIFFERENCE") == []
    # touching edge merges, touching corner stays two shapes
    merged = boolean2d.boolean([sq(0, 0, 2)], [sq(2, 0, 2)], "UNION")
    assert len(merged) == 1 and len(merged[0][0]) == 4
    assert len(boolean2d.boolean([sq(0, 0, 2)], [sq(2, 2, 2)], "UNION")) == 2
    # nested
    nested = boolean2d.boolean([outer], [sq(3, 3, 2)], "DIFFERENCE")
    assert len(nested) == 1 and len(nested[0][1]) == 1 and area(geometry, nested) == pytest.approx(96.0)


def test_nonzero_vs_evenodd(api):
    boolean2d, geometry = api
    rings = [sq(0, 0, 10), sq(3, 3, 4)]
    assert area(geometry, boolean2d.boolean(rings, [], "UNION", "EVENODD")) == pytest.approx(84.0)
    assert area(geometry, boolean2d.boolean(rings, [], "UNION", "NONZERO")) == pytest.approx(100.0)


def test_non_finite_input_is_rejected(api):
    boolean2d, _ = api
    with pytest.raises(ValueError):
        boolean2d.boolean([[(0.0, 0.0), (1.0, float("nan")), (1.0, 1.0)]], [sq(0, 0, 1)], "UNION")


def test_two_500_point_rings_under_300ms(api):
    boolean2d, geometry = api
    a = [circle(500, 100.0)]
    c = [circle(500, 100.0, 60.0, 17.0)]
    boolean2d.boolean(a, c, "UNION")  # warm-up
    timings = {}
    for op in boolean2d.OPS:
        t0 = time.perf_counter()
        res = boolean2d.boolean(a, c, op)
        timings[op] = (time.perf_counter() - t0) * 1000
        assert res
        assert timings[op] < 300
    record_perf("boolean_union_2x500_ms", timings["UNION"])
    r, d, n = 100.0, math.hypot(60.0, 17.0), 500
    lens = 2 * r * r * math.acos(d / (2 * r)) - (d / 2) * math.sqrt(4 * r * r - d * d)
    inter = boolean2d.boolean(a, c, "INTERSECT")
    assert area(geometry, inter) == pytest.approx(lens, rel=3e-3)


BOW = [(0, 0), (4, 4), (4, 0), (0, 4)]       # self-crossing: two lobes of area 4 each
BIG = [(-1, -1), (5, -1), (5, 5), (-1, 5)]


@pytest.mark.parametrize("fill", ["EVENODD", "NONZERO"])
@pytest.mark.parametrize("op,expected", [("UNION", 36.0), ("INTERSECT", 8.0), ("DIFFERENCE", 0.0), ("EXCLUDE", 28.0)])
def test_bowtie_input_is_split_not_silently_wrong(api, op, expected, fill):
    boolean2d, geometry = api
    assert geometry.self_intersections(BOW)
    res = boolean2d.boolean([BOW], [BIG], op, fill)
    assert area(geometry, res) == pytest.approx(expected, abs=1e-9)


def test_bowtie_union_with_disjoint_square_and_union_all(api):
    boolean2d, geometry = api
    far = sq(10, 0, 2)
    assert area(geometry, boolean2d.boolean([BOW], [far], "UNION")) == pytest.approx(12.0, abs=1e-9)
    assert area(geometry, boolean2d.boolean([BOW], [far], "DIFFERENCE")) == pytest.approx(8.0, abs=1e-9)
    assert area(geometry, boolean2d.union_all([[BOW], [far]])) == pytest.approx(12.0, abs=1e-9)
