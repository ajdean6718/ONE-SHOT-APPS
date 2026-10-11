"""SWEEP, TEXT3D and REVOLVE recipes, plus the validation fallbacks used by EXTRUDE."""

import bpy
import pytest

from helpers import flush
from test_link_support import activate, extent, face_area, make_link, make_shape, mesh_arrays

LINE = ((-100, 0), (100, 0))


@pytest.mark.parametrize("profile,ey,ez", [("ROUND", 0.1, 0.1), ("SQUARE", 0.1, 0.1), ("RIBBON", 0.0, 0.1)])
def test_sweep_open_path_profiles(engine, profile, ey, ez):
    src = make_shape("Wire", rings=(LINE,), closed=False)
    link = make_link(src, "SWEEP", sweep_profile=profile, sweep_radius=0.05, orientation="FLAT", origin="SOURCE")
    assert link.state == "LIVE", link.message
    ex = extent(link.target)
    assert ex[0] == pytest.approx(2.0, abs=1e-4)
    assert ex[1] == pytest.approx(ey, abs=1e-4)
    assert ex[2] == pytest.approx(ez, abs=1e-4)
    _co, mats = mesh_arrays(link.target)
    if profile == "RIBBON":
        assert set(mats.tolist()) == {1}
    else:
        assert mats.tolist().count(0) >= 2  # two end caps
        if profile == "SQUARE":
            assert face_area(link.target, 0) == pytest.approx(2 * 0.1 * 0.1, rel=1e-3)


def test_sweep_closed_path_has_no_caps_and_upright(engine):
    src = make_shape("Ring")
    link = make_link(src, "SWEEP", sweep_radius=0.02)  # UPRIGHT/BOTTOM defaults
    ex = extent(link.target)
    # polygonal tube (profile approximated) around the 2 m square ring
    assert 2.02 < ex[0] <= 2.0401 and 2.02 < ex[2] <= 2.0401
    assert 0.03 < ex[1] <= 0.0401
    _co, mats = mesh_arrays(link.target)
    assert set(mats.tolist()) == {1}


def _make_text(body="SEAM", size=100.0):
    from seam_studio.vector import canvas
    cu = bpy.data.curves.new("Title", "FONT")
    cu.body = body
    cu.size = size
    ob = bpy.data.objects.new("Title", cu)
    canvas.add_artwork_object(bpy.context.scene, ob, role="TEXT")
    ob.seam.style.fill_color = (0.2, 0.4, 0.9, 1.0)
    return ob


def test_text3d_follows_text_edits(engine):
    txt = _make_text("SEAM")
    link = make_link(txt, "TEXT3D", depth=0.1, orientation="FLAT")
    assert link.state == "LIVE", link.message
    tgt = link.target
    assert tgt.name == "Title · Text 3D"
    ex = extent(tgt)
    assert ex[2] == pytest.approx(0.1, abs=1e-5)
    w1 = ex[0]
    assert 0.5 < w1 < 5.0  # 100 px text, a few glyphs, 0.01 m/px
    _co, mats = mesh_arrays(tgt)
    assert 0 in mats and 1 in mats
    txt.data.body = "SEAM STUDIO"
    flush()
    assert link.build_count == 2 and extent(tgt)[0] > w1 * 1.5
    txt.data.size = 50.0
    flush()
    assert link.build_count == 3


def test_text3d_rejects_shapes_and_extrude_rejects_text(engine):
    shape = make_shape("S")
    txt = _make_text()
    activate(shape)
    with pytest.raises(RuntimeError):
        bpy.ops.seam.make_3d(recipe="TEXT3D")
    activate(txt)
    with pytest.raises(RuntimeError):
        bpy.ops.seam.make_3d(recipe="EXTRUDE")
    assert len(bpy.context.scene.seam.links) == 0


def test_revolve_when_available(engine):
    from seam_studio.link import recipes
    if not recipes.revolve_available():
        pytest.skip("seam_core.mesh.revolve not implemented yet (core-geometry)")
    prof = make_shape("Vase", rings=(((0, 0), (60, 0), (40, 100), (0, 120)),), closed=False)
    link = make_link(prof, "REVOLVE", revolve_segments=16)
    assert link.state == "LIVE", link.message
    ex = extent(link.target)
    assert ex[0] == pytest.approx(1.2, abs=1e-2) and ex[1] == pytest.approx(1.2, abs=1e-2)
    assert ex[2] == pytest.approx(1.2, abs=1e-5)  # profile height 120 px * 0.01, axis along +Z (UPRIGHT)
    co, mats = mesh_arrays(link.target)
    assert co[:, 2].min() == pytest.approx(0.0, abs=1e-6)  # origin BOTTOM
    assert set(mats.tolist()) <= {0, 1}
    _assert_outward(link.target)
    # FLAT lays the axis along +Y; still a closed, outward-facing solid
    flat = make_link(prof, "REVOLVE", revolve_segments=16, orientation="FLAT")
    assert flat.state == "LIVE", flat.message
    assert extent(flat.target)[1] == pytest.approx(1.2, abs=1e-5)
    _assert_outward(flat.target)
    # about its own origin, a mirrored source revolves to the same outward-facing solid
    # (with LEFT the axis is the *visible* left edge of the mirrored drawing: a different solid)
    a = make_link(prof, "REVOLVE", revolve_segments=16, revolve_axis="SOURCE_ORIGIN")
    mirrored = make_shape("VaseM", rings=(((0, 0), (60, 0), (40, 100), (0, 120)),), closed=False)
    mirrored.scale = (-1.0, 1.0, 1.0)
    lm = make_link(mirrored, "REVOLVE", revolve_segments=16, revolve_axis="SOURCE_ORIGIN")
    assert a.state == "LIVE" and lm.state == "LIVE", (a.message, lm.message)
    _assert_outward(a.target)
    _assert_outward(lm.target)
    assert _volume(lm.target) == pytest.approx(_volume(a.target), rel=1e-6)


def _volume(obj):
    me = obj.data
    vol = 0.0
    for p in me.polygons:
        vs = [me.vertices[i].co for i in p.vertices]
        for k in range(1, len(vs) - 1):
            vol += vs[0].dot(vs[k].cross(vs[k + 1])) / 6.0
    return vol


def _assert_outward(obj):
    """Closed (convex) revolve solid: positive signed volume and every face normal pointing away from
    the centre, i.e. consistent outward normals."""
    import numpy as np
    me = obj.data
    vol = _volume(obj)
    assert vol > 0.0, vol
    co, _ = mesh_arrays(obj)
    centre = (co.max(axis=0) + co.min(axis=0)) * 0.5
    away = [np.dot(np.array(p.normal), np.array(p.center) - centre) for p in me.polygons if p.area > 1e-9]
    assert sum(1 for d in away if d < -1e-6) == 0


def test_revolve_unavailable_reports_and_disables(engine):
    from seam_studio.link import recipes
    if recipes.revolve_available():
        pytest.skip("revolve available")
    ok, why = recipes.recipe_available("REVOLVE")
    assert not ok and "Revolve" in why
    src = make_shape("Vase")
    activate(src)
    with pytest.raises(RuntimeError):
        bpy.ops.seam.make_3d(recipe="REVOLVE")


def test_validation_fallback(engine):
    from seam_studio.link import recipes
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    bow = [(0, 0), (10, 10), (10, 0), (0, 10)]
    assert recipes._ring_self_intersects(square) is None
    hit = recipes._ring_self_intersects(bow)
    assert hit == pytest.approx((5.0, 5.0))
    # touching at a shared vertex of adjacent edges is fine; collinear degenerate ring has no area
    assert recipes.validate_rings([square]) == []
    assert recipes.validate_rings([[(0, 0), (5, 0), (10, 0)]])
    assert recipes.validate_rings([bow])
    v = recipes.validate_rings_local
    assert v([square]) == [] and v([square[:2]]) and v([[(0, 0), (5, 0), (10, 0)]])
    assert "crosses itself" in v([bow])[0]
    assert "no area" in v([bow], check_self_intersection=False)[0]  # symmetric bow tie: zero net area
    # large ring is fast enough (blockwise numpy)
    import math
    import time
    big = [(math.cos(i * 2 * math.pi / 3000) * 100, math.sin(i * 2 * math.pi / 3000) * 100) for i in range(3000)]
    t = time.perf_counter()
    assert recipes._ring_self_intersects(big) is None
    assert time.perf_counter() - t < 2.0


def test_flatten_local_curves():
    from seam_studio.link import recipes
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint
    k = 0.5522847498
    pts = [((1, 0), (1, -k), (1, k)), ((0, 1), (k, 1), (-k, 1)), ((-1, 0), (-1, k), (-1, -k)), ((0, -1), (-k, -1), (k, -1))]
    circle = BezierPath([BezierPoint(c, hl, hr, "FREE", "FREE") for c, hl, hr in pts], closed=True)
    ring = recipes._flatten_local(circle, 12)
    assert len(ring) == 48
    assert all(abs(math_hypot(p) - 1.0) < 1e-3 for p in ring)
    sq = BezierPath([BezierPoint(p, p, p, "VECTOR", "VECTOR") for p in ((0, 0), (1, 0), (1, 1))], closed=True)
    assert recipes._flatten_local(sq, 12) == [(0, 0), (1, 0), (1, 1)]


def math_hypot(p):
    return (p[0] ** 2 + p[1] ** 2) ** 0.5


def _toy_revolve(profile, closed, angle=6.283185307179586, segments=32, axis_x=0.0, cap_ends=True,
                 merge_on_axis=True):
    """Minimal lathe honouring the seam_core.mesh.revolve contract (stand-in until core-geometry lands)."""
    import math
    from seam_studio.seam_core.mesh import MeshData
    n = len(profile)
    verts = []
    for s in range(segments):
        a = angle * s / segments
        for x, y in profile:
            r = abs(x - axis_x)
            verts.append((axis_x + r * math.cos(a), r * math.sin(a), y))
    faces = []
    for s in range(segments):
        s2 = (s + 1) % segments
        for i in range(n - 1):
            faces.append((s * n + i, s2 * n + i, s2 * n + i + 1, s * n + i + 1))
    return MeshData(verts=verts, faces=faces, face_materials=[1] * len(faces))


@pytest.mark.parametrize("orientation", ["UPRIGHT", "FLAT"])
def test_revolve_wiring_with_contract_stub(engine, monkeypatch, orientation):
    """Exercises axis placement, scaling and orientation of the REVOLVE path with a contract-conforming stub."""
    from seam_studio.seam_core import mesh as core_mesh
    monkeypatch.setattr(core_mesh, "revolve", _toy_revolve)
    prof = make_shape("Vase", rings=(((0, 0), (60, 0), (40, 100), (0, 120)),), closed=False)
    link = make_link(prof, "REVOLVE", revolve_segments=16, orientation=orientation, origin="BOTTOM")
    assert link.state == "LIVE", link.message
    co, mats = mesh_arrays(link.target)
    ex = co.max(axis=0) - co.min(axis=0)
    up = 2 if orientation == "UPRIGHT" else 1
    assert ex[up] == pytest.approx(1.2, abs=1e-4)  # profile height 120 px
    assert ex[0] == pytest.approx(1.2, abs=1e-2)  # radius 60 px both sides
    assert co[:, 2].min() == pytest.approx(0.0, abs=1e-6)  # BOTTOM: object stands on its origin
    assert set(mats.tolist()) == {1}


def test_fast_signature_matches_contract(engine):
    import numpy as np
    from seam_studio.link import producer
    rng = np.random.default_rng(7)
    for _ in range(5):
        co = rng.normal(size=(50, 3)).astype(np.float32) * 3
        co[0] = (0.123455, -0.000005, 2.500005)  # rounding edge cases
        totals = rng.integers(3, 6, size=20)
        corners = rng.integers(0, 50, size=int(totals.sum()))
        assert producer._signature_arrays(co, totals, corners) == producer.reference_signature(co, totals, corners)
    assert producer._signature_arrays(np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int)) == \
        producer.reference_signature(np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int))
    src = make_shape("A")
    link = make_link(src)
    me = link.target.data
    co = np.array([v.co[:] for v in me.vertices])
    totals = [p.loop_total for p in me.polygons]
    corners = [lp.vertex_index for lp in me.loops]
    assert producer.reference_signature(co, totals, corners) == link.target_signature


def _brute(ring):
    from seam_studio.link import recipes
    import numpy as np
    n = len(ring)
    p = np.asarray(ring, float)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            hit = recipes._first_crossing(p[[i]], p[[(i + 1) % n]], p[[j]], p[[(j + 1) % n]])
            if hit is not None:
                return True
    return False


def test_sweep_intersection_matches_brute_force():
    import random
    from seam_studio.link import recipes
    rnd = random.Random(3)
    for trial in range(60):
        n = rnd.randint(4, 40)
        ring = [(rnd.uniform(0, 100), rnd.uniform(0, 100)) for _ in range(n)]
        if trial % 3 == 0:  # star-shaped (simple) polygons too
            import math
            ring = sorted(ring, key=lambda p: math.atan2(p[1] - 50, p[0] - 50))
        assert (recipes._ring_self_intersects(ring) is not None) == _brute(ring), ring


def test_rings_crossing_each_other_are_invalid(engine):
    from seam_studio.link import recipes
    from test_link_support import SQUARE
    outer = [(-100, -100), (100, -100), (100, 100), (-100, 100)]
    sticking_out = [(50, -50), (50, 50), (150, 50), (150, -50)]
    inside = [(-40, -40), (-40, 40), (40, 40), (40, -40)]
    v = recipes.validate_rings_local
    assert v([outer, inside]) == []
    problems = v([outer, sticking_out])
    assert len(problems) == 1 and "paths 1 and 2 cross each other" in problems[0], problems
    assert v([outer, sticking_out], check_self_intersection=False) == []  # TEXT3D: Blender nests overlaps
    # touching rings (shared vertex, no proper crossing) are not reported
    assert v([outer, [(100, 100), (150, 100), (150, 150)]]) == []
    # end to end (core validator when merged, else the fallback): ERROR, previous geometry kept
    src = make_shape("X", rings=(SQUARE, tuple(sticking_out)))
    activate(src)
    bpy.ops.seam.make_3d(recipe="EXTRUDE")
    link = bpy.context.scene.seam.links[-1]
    assert link.state == "ERROR" and link.message.startswith("Invalid shape"), link.message


def _brute_multi(rings):
    from seam_studio.link import recipes
    import numpy as np
    edges = []
    for r, ring in enumerate(rings):
        n = len(ring)
        for i in range(n):
            edges.append((r, i, n, np.asarray(ring[i], float), np.asarray(ring[(i + 1) % n], float)))
    for x in range(len(edges)):
        for y in range(x + 1, len(edges)):
            ra, ia, na, p0, p1 = edges[x]
            rb, ib, nb, q0, q1 = edges[y]
            if ra == rb and (abs(ia - ib) in (0, 1, na - 1)):
                continue
            if recipes._first_crossing(p0[None], p1[None], q0[None], q1[None]) is not None:
                return True
    return False


def test_multi_ring_sweep_matches_brute_force():
    import math
    import random
    from seam_studio.link import recipes
    rnd = random.Random(11)
    for trial in range(60):
        rings = []
        for _ in range(rnd.randint(2, 4)):
            cx, cy, rad = rnd.uniform(0, 100), rnd.uniform(0, 100), rnd.uniform(5, 40)
            n = rnd.randint(3, 12)
            angles = sorted(rnd.uniform(0, 2 * math.pi) for _ in range(n))
            ring = [(cx + rad * math.cos(a), cy + rad * math.sin(a)) for a in angles]  # each ring simple
            if trial % 4 == 0:
                rnd.shuffle(ring)
            rings.append(ring)
        assert (recipes._rings_cross(rings) is not None) == _brute_multi(rings), rings
