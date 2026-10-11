"""Unit tests for seam_core.boolean2d (needs mathutils: the bpy venv, imported via bpy, or Blender)."""

import json
import math
import os
import platform
import random
import sys
import time

import pytest

from seam_core import boolean2d as b
from seam_core import geometry as g


def _have_mathutils():
    try:
        import mathutils.geometry  # noqa: F401
        return True
    except Exception:
        pass
    try:  # the bpy wheel registers mathutils on import
        import bpy  # noqa: F401
        import mathutils.geometry  # noqa: F401,F811
        return True
    except Exception:
        return False


if not _have_mathutils():
    pytest.skip("mathutils is not importable in this interpreter", allow_module_level=True)


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
    label = "python-%s" % platform.python_version()
    if "bpy" in sys.modules:
        label += " (in Blender %s)" % sys.modules["bpy"].app.version_string
    doc.setdefault("runs", {}).setdefault(label, {})[name] = round(ms, 3)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)


def sq(x, y, s):
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]


def area(shapes):
    return sum(g.signed_area(o) + sum(g.signed_area(h) for h in hs) for o, hs in shapes)


def circle(n, r, cx=0.0, cy=0.0):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


A = [sq(0, 0, 2)]
B = [sq(1, 1, 2)]


def test_available_in_this_environment():
    assert b.available()
    assert b.OPS == ("UNION", "DIFFERENCE", "INTERSECT", "EXCLUDE")


@pytest.mark.parametrize("op,expected,count", [("UNION", 7.0, 1), ("DIFFERENCE", 3.0, 1), ("INTERSECT", 1.0, 1), ("EXCLUDE", 6.0, 2)])
def test_overlapping_squares_exact_areas(op, expected, count):
    res = b.boolean(A, B, op)
    assert area(res) == pytest.approx(expected, abs=1e-9)
    assert len(res) == count
    for outer, holes in res:
        assert g.signed_area(outer) > 0 and all(g.signed_area(h) < 0 for h in holes)


def test_union_and_intersect_vertices_are_clean():
    union = b.boolean(A, B, "UNION")[0][0]
    assert len(union) == 8  # collinear/T vertices removed
    assert set(union) == {(0, 0), (2, 0), (2, 1), (3, 1), (3, 3), (1, 3), (1, 2), (0, 2)}
    inter = b.boolean(A, B, "INTERSECT")[0][0]
    assert sorted(inter) == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_difference_is_not_symmetric():
    ab = b.boolean(A, B, "DIFFERENCE")
    ba = b.boolean(B, A, "DIFFERENCE")
    assert area(ab) == pytest.approx(3.0) and area(ba) == pytest.approx(3.0)
    assert {tuple(sorted(s[0])) for s in ab} != {tuple(sorted(s[0])) for s in ba}


def test_holes_in_inputs_are_preserved():
    outer = sq(0, 0, 10)
    hole = sq(3, 3, 4)[::-1]
    compound = [outer, hole]
    far = [sq(20, 20, 2)]
    res = b.boolean(compound, far, "UNION")
    assert len(res) == 2
    holed = next(s for s in res if abs(g.signed_area(s[0]) - 100.0) < 1e-9)
    assert len(holed[1]) == 1 and g.signed_area(holed[1][0]) == pytest.approx(-16.0)
    # subtracting something inside the hole changes nothing; something overlapping it grows the hole
    inside_hole = [sq(4, 4, 2)]
    assert area(b.boolean(compound, inside_hole, "DIFFERENCE")) == pytest.approx(84.0)
    overlapping = [sq(2, 2, 4)]
    diff = b.boolean(compound, overlapping, "DIFFERENCE")
    expected = 100.0 - (16 + 16 - 9)  # hole (3..7)^2 united with (2..6)^2 has area 16+16-9
    assert area(diff) == pytest.approx(expected)
    assert len(diff) == 1 and len(diff[0][1]) == 1
    # intersect keeps only material
    inter = b.boolean(compound, [sq(2, 2, 6)], "INTERSECT")
    assert area(inter) == pytest.approx(36.0 - 16.0)
    assert len(inter) == 1 and len(inter[0][1]) == 1


def test_orientation_of_input_rings_is_irrelevant_for_evenodd():
    outer = sq(0, 0, 10)
    hole_ccw = sq(3, 3, 4)
    res = b.boolean([outer, hole_ccw], [], "UNION")
    assert area(res) == pytest.approx(84.0)
    res2 = b.boolean([outer[::-1], hole_ccw[::-1]], [], "UNION")
    assert area(res2) == pytest.approx(84.0)


def test_nonzero_fill_rule():
    outer = sq(0, 0, 10)
    inner_same_dir = sq(3, 3, 4)
    even = b.boolean([outer, inner_same_dir], [], "UNION", "EVENODD")
    nonzero = b.boolean([outer, inner_same_dir], [], "UNION", "NONZERO")
    assert area(even) == pytest.approx(84.0)
    assert area(nonzero) == pytest.approx(100.0)
    nonzero_hole = b.boolean([outer, inner_same_dir[::-1]], [], "UNION", "NONZERO")
    assert area(nonzero_hole) == pytest.approx(84.0)
    with pytest.raises(ValueError):
        b.boolean(A, B, "UNION", "WHATEVER")


def test_disjoint_identical_touching_nested():
    far = [sq(10, 10, 2)]
    assert b.boolean(A, far, "INTERSECT") == []
    assert area(b.boolean(A, far, "UNION")) == pytest.approx(8.0)
    assert len(b.boolean(A, far, "UNION")) == 2
    assert area(b.boolean(A, far, "DIFFERENCE")) == pytest.approx(4.0)
    assert len(b.boolean(A, far, "EXCLUDE")) == 2
    # identical
    assert area(b.boolean(A, A, "UNION")) == pytest.approx(4.0)
    assert area(b.boolean(A, A, "INTERSECT")) == pytest.approx(4.0)
    assert b.boolean(A, A, "DIFFERENCE") == []
    assert b.boolean(A, A, "EXCLUDE") == []
    # touching along an edge: union merges into one rectangle with no extra vertices
    right = [sq(2, 0, 2)]
    merged = b.boolean(A, right, "UNION")
    assert len(merged) == 1 and sorted(merged[0][0]) == [(0, 0), (0, 2), (4, 0), (4, 2)]
    assert b.boolean(A, right, "INTERSECT") == []
    assert area(b.boolean(A, right, "DIFFERENCE")) == pytest.approx(4.0)
    # touching only at a corner: two separate shapes
    corner = [sq(2, 2, 2)]
    assert len(b.boolean(A, corner, "UNION")) == 2
    assert b.boolean(A, corner, "INTERSECT") == []
    # nested: difference makes a hole, union is the big one, intersect is the small one
    big = [sq(0, 0, 10)]
    small = [sq(3, 3, 2)]
    diff = b.boolean(big, small, "DIFFERENCE")
    assert len(diff) == 1 and len(diff[0][1]) == 1 and area(diff) == pytest.approx(96.0)
    assert area(b.boolean(big, small, "UNION")) == pytest.approx(100.0)
    assert len(b.boolean(big, small, "UNION")[0][1]) == 0
    assert area(b.boolean(big, small, "INTERSECT")) == pytest.approx(4.0)
    assert b.boolean(small, big, "DIFFERENCE") == []
    ex = b.boolean(big, small, "EXCLUDE")
    assert area(ex) == pytest.approx(96.0) and len(ex[0][1]) == 1


def test_hole_touching_outer_boundary_stays_a_hole():
    outer = sq(0, 0, 10)
    hole = [(4.0, 0.0), (3.0, 3.0), (6.0, 3.0)]  # CW triangle touching the bottom edge at (4, 0)
    res = b.boolean([outer, hole], [], "UNION")
    assert area(res) == pytest.approx(100.0 - abs(g.signed_area(hole)))
    assert len(res) == 1 and len(res[0][1]) == 1


def test_empty_and_degenerate_inputs():
    assert b.boolean([], [], "UNION") == []
    assert area(b.boolean(A, [], "UNION")) == pytest.approx(4.0)
    assert area(b.boolean([], A, "UNION")) == pytest.approx(4.0)
    assert b.boolean([], A, "DIFFERENCE") == []
    assert b.boolean(A, [], "INTERSECT") == []
    assert b.boolean([[(0, 0), (1, 1)], [(0, 0), (1, 0), (2, 0)]], A, "UNION") == b.boolean([], A, "UNION")
    with pytest.raises(ValueError):
        b.boolean(A, B, "XOR")


def test_non_finite_input_raises_value_error():
    bad = [[(0.0, 0.0), (1.0, float("nan")), (1.0, 1.0)]]
    t0 = time.perf_counter()
    for call in (lambda: b.boolean(bad, A, "UNION"), lambda: b.boolean(A, bad, "UNION"),
                 lambda: b.union_all([A, bad]), lambda: b.boolean(A, [[(0, 0), (float("inf"), 0), (1, 1)]], "INTERSECT")):
        with pytest.raises(ValueError):
            call()
    assert time.perf_counter() - t0 < 2.0


def test_union_all_single_pass():
    sets = [[sq(i * 1.5, 0, 2)] for i in range(6)]  # chain of overlapping squares
    res = b.union_all(sets)
    assert len(res) == 1 and area(res) == pytest.approx(2 * (1.5 * 5 + 2))
    two_groups = b.union_all([[sq(0, 0, 2)], [sq(1, 1, 2)], [sq(10, 10, 1)]])
    assert len(two_groups) == 2 and area(two_groups) == pytest.approx(7.0 + 1.0)
    assert b.union_all([]) == []
    # a compound member keeps its own hole parity
    holed = [sq(0, 0, 10), sq(3, 3, 4)]
    res = b.union_all([holed, [sq(4, 4, 2)]])
    assert area(res) == pytest.approx(84.0 + 4.0) and len(res) == 2


def test_circles_union_matches_analytic_lens_area():
    r, d = 100.0, 60.0
    n = 500
    res = b.boolean([circle(n, r)], [circle(n, r, d, 0)], "UNION")
    lens = 2 * r * r * math.acos(d / (2 * r)) - (d / 2) * math.sqrt(4 * r * r - d * d)
    poly_area = 0.5 * n * r * r * math.sin(2 * math.pi / n)
    assert area(res) == pytest.approx(2 * poly_area - lens, rel=2e-3)
    inter = b.boolean([circle(n, r)], [circle(n, r, d, 0)], "INTERSECT")
    assert area(inter) == pytest.approx(lens, rel=2e-3)


def test_two_500_point_rings_under_300ms():
    a = [circle(500, 100.0)]
    c = [circle(500, 100.0, 60.0, 17.0)]
    b.boolean(a, c, "UNION")  # warm-up (imports)
    t0 = time.perf_counter()
    res = b.boolean(a, c, "UNION")
    ms = (time.perf_counter() - t0) * 1000
    assert res and ms < 300
    record_perf("boolean_union_2x500_ms", ms)
    for op in ("DIFFERENCE", "INTERSECT", "EXCLUDE"):
        t0 = time.perf_counter()
        b.boolean(a, c, op)
        assert (time.perf_counter() - t0) * 1000 < 300


def _star(rnd, n, cx, cy, rmin, rmax):
    pts = []
    for i in range(n):
        r = rnd.uniform(rmin, rmax)
        pts.append((cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)))
    return pts


def test_random_polygons_satisfy_set_identities():
    import random
    rnd = random.Random(5)
    for trial in range(80):
        pa = _star(rnd, rnd.randint(3, 25), 0, 0, 10, 40)
        pb = _star(rnd, rnd.randint(3, 25), rnd.uniform(-30, 30), rnd.uniform(-30, 30), 10, 40)
        aa = abs(g.signed_area(pa))
        ab = abs(g.signed_area(pb))
        union = area(b.boolean([pa], [pb], "UNION"))
        inter = area(b.boolean([pa], [pb], "INTERSECT"))
        diff = area(b.boolean([pa], [pb], "DIFFERENCE"))
        excl = area(b.boolean([pa], [pb], "EXCLUDE"))
        tol = 2e-6 * (aa + ab)  # float32 coordinates inside mathutils.Vector
        assert union + inter == pytest.approx(aa + ab, abs=tol), trial
        assert diff == pytest.approx(aa - inter, abs=tol), trial
        assert excl == pytest.approx(union - inter, abs=tol), trial


def test_grid_aligned_rectangles_match_cell_counting():
    import random
    rnd = random.Random(9)

    def rect():
        x, y, w, h = rnd.randint(0, 6), rnd.randint(0, 6), rnd.randint(1, 4), rnd.randint(1, 4)
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], {(i, j) for i in range(x, x + w) for j in range(y, y + h)}

    for trial in range(120):
        sa = [rect() for _ in range(rnd.randint(1, 2))]
        sb = [rect() for _ in range(rnd.randint(1, 2))]
        ca = set()
        for _, cells in sa:
            ca ^= cells
        cb = set()
        for _, cells in sb:
            cb ^= cells
        expected = {"UNION": len(ca | cb), "INTERSECT": len(ca & cb), "DIFFERENCE": len(ca - cb), "EXCLUDE": len(ca ^ cb)}
        for op, count in expected.items():
            got = area(b.boolean([r for r, _ in sa], [r for r, _ in sb], op))
            assert got == pytest.approx(count, abs=1e-9), (trial, op)


# ---------------------------------------------------------------- self-crossing input rings

BOW = [(0, 0), (4, 4), (4, 0), (0, 4)]       # two lobes of area 4 each, crossing at (2, 2)
BIG = [(-1, -1), (5, -1), (5, 5), (-1, 5)]   # contains both lobes


@pytest.mark.parametrize("fill", ["EVENODD", "NONZERO"])
@pytest.mark.parametrize("op,expected", [("UNION", 36), ("INTERSECT", 8), ("DIFFERENCE", 0), ("EXCLUDE", 28)])
def test_bowtie_against_enclosing_square(op, expected, fill):
    assert g.self_intersections(BOW)
    assert area(b.boolean([BOW], [BIG], op, fill)) == pytest.approx(expected)
    if op == "DIFFERENCE":
        assert area(b.boolean([BIG], [BOW], op, fill)) == pytest.approx(28)


def test_bowtie_against_disjoint_and_overlapping_shapes():
    far = [(10, 0), (12, 0), (12, 2), (10, 2)]
    assert area(b.boolean([BOW], [far], "UNION")) == pytest.approx(12)
    assert area(b.boolean([BOW], [far], "DIFFERENCE")) == pytest.approx(8.0)  # bowtie untouched
    # a square covering the left lobe's tip region only
    left = [(0, 0), (2, 0), (2, 4), (0, 4)]
    assert area(b.boolean([BOW], [left], "INTERSECT")) == pytest.approx(4.0)   # exactly the left lobe
    assert area(b.boolean([BOW], [left], "UNION")) == pytest.approx(12.0)


def test_figure_eight_and_pentagram_follow_the_fill_rule():
    eight = [(0, 0), (2, 0), (3, 1), (2, 2), (0, 2), (-1, 1)]
    star = [(math.cos(math.radians(90 + 144 * k)), math.sin(math.radians(90 + 144 * k))) for k in range(5)]
    assert g.self_intersections(star)
    full = area(b.boolean([star], [], "UNION", "NONZERO"))
    eo = area(b.boolean([star], [], "UNION", "EVENODD"))
    assert eo < full - 1e-6
    # the inner pentagon is a hole under EVENODD and filled under NONZERO
    res = b.boolean([star], [], "UNION", "EVENODD")
    assert sum(len(h) for _o, h in res) == 0 and len(res) == 5     # five separate tips
    assert len(b.boolean([star], [], "UNION", "NONZERO")) == 1
    # a ring that passes through the same vertex twice (figure eight pinched at (2, 2))
    pinch = [(0, 0), (2, 2), (4, 0), (4, 4), (2, 2), (0, 4)]
    assert area(b.boolean([pinch], [BIG], "INTERSECT")) == pytest.approx(8.0)
    assert area(b.boolean([eight], [], "UNION")) == pytest.approx(abs(g.signed_area(eight)))


def _member(pt, rings, rule):
    w = sum(g.winding_number(pt, r) for r in rings)
    return (w % 2 != 0) if rule == "EVENODD" else (w != 0)


def _in_result(pt, shapes):
    return any(g.point_in_ring(pt, o) and not any(g.point_in_ring(pt, h) for h in hs) for o, hs in shapes)


@pytest.mark.parametrize("coords", ["grid", "float"])
def test_random_self_crossing_rings_match_winding_oracle(coords):
    """Random (mostly self-crossing, on the integer grid also degenerate) rings vs a sampled winding-number oracle."""
    rng = random.Random(7)
    sels = {"UNION": lambda x, y: x or y, "INTERSECT": lambda x, y: x and y,
            "DIFFERENCE": lambda x, y: x and not y, "EXCLUDE": lambda x, y: x != y}

    def ring():
        n = rng.randint(3, 8)
        if coords == "grid":
            return [(rng.randint(0, 12), rng.randint(0, 12)) for _ in range(n)]
        return [(rng.uniform(0, 12), rng.uniform(0, 12)) for _ in range(n)]

    crossing_cases = 0
    for _case in range(70):
        ra = [ring() for _ in range(rng.randint(1, 2))]
        rb = [ring() for _ in range(rng.randint(1, 2))]
        crossing_cases += any(g.self_intersections(r) for r in ra + rb)
        samples = [(rng.uniform(-1, 13) + 7e-4, rng.uniform(-1, 13) + 3e-4) for _ in range(25)]
        for rule in ("EVENODD", "NONZERO"):
            for op, sel in sels.items():
                res = b.boolean(ra, rb, op, rule)
                for p in samples:
                    assert sel(_member(p, ra, rule), _member(p, rb, rule)) == _in_result(p, res), (op, rule, ra, rb, p)
    assert crossing_cases > 10


def test_union_all_with_self_crossing_ring():
    far = [(10, 0), (12, 0), (12, 2), (10, 2)]
    assert area(b.union_all([[BOW], [far]])) == pytest.approx(12)
