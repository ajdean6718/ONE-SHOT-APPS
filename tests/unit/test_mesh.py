"""Unit tests for seam_core.mesh (extrude, revolve, sweep, orient, place_origin, scale)."""

import math
import random
from collections import Counter

import pytest

from seam_core import geometry as g
from seam_core import mesh as m
from seam_core.mesh import MAT_FACE, MAT_SIDE, MeshData

SQ = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
HOLE = [(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)]  # CW


def edge_use(mesh):
    use = Counter()
    for f in mesh.faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            use[(min(a, b), max(a, b))] += 1
    return use


def euler(mesh):
    return len(mesh.verts) - len(edge_use(mesh)) + len(mesh.faces)


def volume(mesh):
    """Signed volume by the divergence theorem (positive for outward-facing closed meshes)."""
    vol = 0.0
    for f in mesh.faces:
        v0 = mesh.verts[f[0]]
        for k in range(1, len(f) - 1):
            v1 = mesh.verts[f[k]]
            v2 = mesh.verts[f[k + 1]]
            vol += (v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
                    + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])) / 6.0
    return vol


def assert_closed_manifold(mesh):
    mesh.validate()
    assert set(edge_use(mesh).values()) == {2}


def face_area(mesh, f):
    # Newell
    nx = ny = nz = 0.0
    for i in range(len(f)):
        a = mesh.verts[f[i]]
        b = mesh.verts[f[(i + 1) % len(f)]]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


# ---------------------------------------------------------------- MeshData.validate

def test_validate_detects_problems():
    ok = MeshData(verts=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)], face_materials=[0],
                  uvs=[(0, 0), (1, 0), (0, 1)])
    ok.validate()
    MeshData(verts=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)]).validate()  # optional layers may be empty
    for bad in (
        MeshData(verts=[(0, 0, 0)], faces=[(0, 1, 2)]),  # index out of range
        MeshData(verts=[(0, 0, 0), (1, 0, 0)], faces=[(0, 1)]),  # < 3 verts
        MeshData(verts=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, -1)]),
        MeshData(verts=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)], face_materials=[0, 1]),
        MeshData(verts=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)], uvs=[(0, 0)]),
        MeshData(verts=[(0, 0, float("nan")), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)]),
    ):
        with pytest.raises(ValueError):
            bad.validate()
    MeshData().validate()


# ---------------------------------------------------------------- extrude

def test_extrude_square_with_hole_is_closed_outward_manifold():
    depth = 5.0
    mesh = m.extrude([(SQ, [HOLE])], depth)
    assert_closed_manifold(mesh)
    assert euler(mesh) == 0  # one through-hole: genus 1
    assert volume(mesh) == pytest.approx((100.0 - 16.0) * depth, abs=1e-6)
    zs = [v[2] for v in mesh.verts]
    assert min(zs) == pytest.approx(-2.5) and max(zs) == pytest.approx(2.5)


def test_extrude_materials_and_uv_counts():
    mesh = m.extrude([(SQ, [HOLE])], 2.0)
    assert len(mesh.face_materials) == len(mesh.faces)
    assert set(mesh.face_materials) == {MAT_FACE, MAT_SIDE} == {0, 1}
    for f, mat in zip(mesh.faces, mesh.face_materials):
        zs = {round(mesh.verts[i][2], 9) for i in f}
        if mat == MAT_FACE:
            assert len(zs) == 1 and len(f) == 3  # caps are flat triangles
        else:
            assert len(zs) == 2 and len(f) == 4  # walls are quads
    assert len(mesh.uvs) == sum(len(f) for f in mesh.faces)
    assert all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 for u, v in mesh.uvs)
    walls = [f for f, mat in zip(mesh.faces, mesh.face_materials) if mat == MAT_SIDE]
    assert len(walls) == 8  # 4 outer + 4 hole edges
    cap_faces = [f for f, mat in zip(mesh.faces, mesh.face_materials) if mat == MAT_FACE]
    assert len(cap_faces) == 16  # 8 triangles per cap (square ring with 8 vertices)


def test_extrude_uv_layout_caps_planar_walls_perimeter():
    mesh = m.extrude([(SQ, [])], 4.0)
    pos = 0
    for f, mat in zip(mesh.faces, mesh.face_materials):
        loops = mesh.uvs[pos:pos + len(f)]
        pos += len(f)
        if mat == MAT_FACE:
            for i, (u, v) in zip(f, loops):
                x, y, _ = mesh.verts[i]
                assert (u, v) == pytest.approx((x / 10.0, y / 10.0))
        else:
            (u0, v0), (u1, v1), (u2, v2), (u3, v3) = loops
            assert (v0, v1, v2, v3) == pytest.approx((0.0, 0.0, 1.0, 1.0))
            assert u0 == pytest.approx(u3) and u1 == pytest.approx(u2)
            assert u1 - u0 == pytest.approx(0.25)  # each side is a quarter of the perimeter


def test_extrude_orientation_independent_of_input_winding():
    ccw_ref = volume(m.extrude([(SQ, [HOLE])], 3.0))
    flipped = m.extrude([(SQ[::-1], [HOLE[::-1]])], 3.0)  # outer CW, hole CCW on input
    assert_closed_manifold(flipped)
    assert volume(flipped) == pytest.approx(ccw_ref)


def test_extrude_concave_multi_shape_and_collinear_points():
    l_shape = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    other = [(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)]
    mesh = m.extrude([(l_shape, []), (other, [])], 2.0, center=False)
    assert_closed_manifold(mesh)
    assert euler(mesh) == 4  # two disjoint spheres
    assert volume(mesh) == pytest.approx((g.signed_area(l_shape) + 100.0) * 2.0)
    assert min(v[2] for v in mesh.verts) == 0.0 and max(v[2] for v in mesh.verts) == 2.0
    # many exactly collinear points per side must still produce a watertight mesh
    dense = []
    for k in range(5):
        dense.append((float(k * 2), 0.0))
    for k in range(5):
        dense.append((10.0, float(k * 2)))
    for k in range(5):
        dense.append((10.0 - k * 2, 10.0))
    for k in range(5):
        dense.append((0.0, 10.0 - k * 2))
    dm = m.extrude([(dense, [])], 1.0)
    assert_closed_manifold(dm)
    assert volume(dm) == pytest.approx(100.0)


def test_extrude_cap_flags_and_flat():
    open_tube = m.extrude([(SQ, [])], 1.0, cap_front=False, cap_back=False)
    assert set(open_tube.face_materials) == {MAT_SIDE} and len(open_tube.faces) == 4
    only_front = m.extrude([(SQ, [])], 1.0, cap_back=False)
    caps = [f for f, mt in zip(only_front.faces, only_front.face_materials) if mt == MAT_FACE]
    assert caps and all(all(only_front.verts[i][2] == pytest.approx(0.5) for i in f) for f in caps)
    flat = m.extrude([(SQ, [HOLE])], 0.0)
    flat.validate()
    assert set(flat.face_materials) == {MAT_FACE}
    assert sum(face_area(flat, f) for f in flat.faces) == pytest.approx(84.0)
    assert m.extrude([], 1.0).faces == []
    assert m.extrude([([(0, 0), (1, 1)], [])], 1.0).faces == []  # degenerate outer skipped


def test_extrude_rejects_bad_depth():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            m.extrude([(SQ, [])], bad)
    with pytest.raises(ValueError):
        m.extrude([([(0, 0), (1, 0), (1, float("nan"))], [])], 1.0)


def test_extrude_wall_normals_point_away_from_solid():
    mesh = m.extrude([(SQ, [HOLE])], 2.0)
    cx = cy = 5.0
    for f, mat in zip(mesh.faces, mesh.face_materials):
        if mat != MAT_SIDE:
            continue
        a, b, c, d = (mesh.verts[i] for i in f)
        n = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
             (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]), 0.0)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        outward_of_outer = abs(mx - cx) > 4.0 or abs(my - cy) > 4.0  # outer ring walls sit on the 10x10 border
        radial = (mx - cx) * n[0] + (my - cy) * n[1]
        assert (radial > 0) == outward_of_outer


# ---------------------------------------------------------------- revolve

def test_revolve_offset_rectangle_volume_within_2_percent():
    r_in, r_out, h = 2.0, 3.0, 4.0
    profile = [(r_in, 0.0), (r_out, 0.0), (r_out, h), (r_in, h)]
    mesh = m.revolve(profile, True, segments=64)
    assert_closed_manifold(mesh)
    area = (r_out - r_in) * h
    centroid = (r_in + r_out) / 2
    expected = 2 * math.pi * centroid * area
    assert volume(mesh) == pytest.approx(expected, rel=0.02)
    assert euler(mesh) == 0  # torus
    assert all(v[2] in (0.0, h) for v in mesh.verts)
    assert set(mesh.face_materials) == {MAT_SIDE}  # full revolution: no caps
    assert len(mesh.uvs) == sum(len(f) for f in mesh.faces)


def test_revolve_is_independent_of_profile_winding_and_axis_offset():
    base = m.revolve([(2, 0), (3, 0), (3, 4), (2, 4)], True, segments=32)
    cw = m.revolve([(2, 0), (2, 4), (3, 4), (3, 0)], True, segments=32)
    shifted = m.revolve([(12, 0), (13, 0), (13, 4), (12, 4)], True, segments=32, axis_x=10.0)
    for mesh in (cw, shifted):
        assert_closed_manifold(mesh)
        assert volume(mesh) == pytest.approx(volume(base))


def test_revolve_on_axis_points_merge_without_degenerate_faces():
    profile = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0)]  # solid cylinder
    mesh = m.revolve(profile, True, segments=48)
    assert_closed_manifold(mesh)
    assert euler(mesh) == 2
    assert volume(mesh) == pytest.approx(math.pi * 9 * 4, rel=0.01)
    on_axis = [v for v in mesh.verts if abs(v[0]) < 1e-12 and abs(v[1]) < 1e-12]
    assert len(on_axis) == 2  # one merged vertex per axis point
    for f in mesh.faces:
        assert len(set(f)) == len(f)
        assert face_area(mesh, f) > 1e-9
        assert len(f) in (3, 4)
    tris = [f for f in mesh.faces if len(f) == 3]
    assert len(tris) == 2 * 48  # the two flat discs become triangle fans
    # a cone: apex on the axis, base at the bottom
    cone = m.revolve([(0.0, 6.0), (3.0, 0.0), (0.0, 0.0)], True, segments=64)
    assert_closed_manifold(cone)
    assert volume(cone) == pytest.approx(math.pi * 9 * 6 / 3, rel=0.01)
    # without merging, on-axis points are duplicated per column
    unmerged = m.revolve(profile, True, segments=8, merge_on_axis=False)
    unmerged.validate()
    assert len(unmerged.verts) == 4 * 8


def test_revolve_partial_angle_has_end_caps():
    profile = [(2.0, 0.0), (3.0, 0.0), (3.0, 4.0), (2.0, 4.0)]
    half = m.revolve(profile, True, angle=math.pi, segments=32)
    assert_closed_manifold(half)
    assert euler(half) == 2
    full_vol = 2 * math.pi * 2.5 * 4
    assert volume(half) == pytest.approx(full_vol / 2, rel=0.02)
    caps = [f for f, mt in zip(half.faces, half.face_materials) if mt == MAT_FACE]
    assert len(caps) == 4  # 2 triangles per end cap
    # start cap lies in the y = 0 plane, end cap in the plane at angle pi (also y ~ 0)
    for f in caps:
        assert all(abs(half.verts[i][1]) < 1e-9 for i in f)
    no_caps = m.revolve(profile, True, angle=math.pi, segments=32, cap_ends=False)
    no_caps.validate()
    assert MAT_FACE not in no_caps.face_materials
    assert set(edge_use(no_caps).values()) == {1, 2}  # open: end loops are boundary edges
    quarter = m.revolve([(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0)], True, angle=math.pi / 2, segments=16)
    assert_closed_manifold(quarter)
    assert volume(quarter) == pytest.approx(math.pi * 9 * 4 / 4, rel=0.01)


def test_revolve_open_profile_makes_open_surface_with_outward_normals():
    profile = [(2.0, 0.0), (3.0, 2.0), (2.0, 4.0)]  # vase wall drawn bottom to top
    mesh = m.revolve(profile, False, segments=32)
    mesh.validate()
    assert set(edge_use(mesh).values()) == {1, 2}
    for f in mesh.faces:
        # outward = away from the axis for this convex-outwards wall
        pts = [mesh.verts[i] for i in f]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        n = (pts[1][1] - pts[0][1]) * (pts[2][2] - pts[0][2]) - (pts[1][2] - pts[0][2]) * (pts[2][1] - pts[0][1]), \
            (pts[1][2] - pts[0][2]) * (pts[2][0] - pts[0][0]) - (pts[1][0] - pts[0][0]) * (pts[2][2] - pts[0][2])
        assert cx * n[0] + cy * n[1] > 0


def test_revolve_profile_validation():
    with pytest.raises(ValueError):
        m.revolve([(-1.0, 0.0), (1.0, 0.0), (1.0, 2.0), (-1.0, 2.0)], True)  # crosses the axis
    mirrored = m.revolve([(-2.0, 0.0), (-3.0, 0.0), (-3.0, 4.0), (-2.0, 4.0)], True, segments=32)
    assert_closed_manifold(mirrored)  # all on the negative side: mirrored, not an error
    assert volume(mirrored) == pytest.approx(volume(m.revolve([(2, 0), (3, 0), (3, 4), (2, 4)], True, segments=32)))
    with pytest.raises(ValueError):
        m.revolve([(1.0, 0.0)], False)
    with pytest.raises(ValueError):
        m.revolve([(1.0, 0.0), (2.0, 0.0)], True)
    with pytest.raises(ValueError):
        m.revolve([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], True)  # zero area
    with pytest.raises(ValueError):
        m.revolve([(1.0, 0.0), (2.0, 1.0)], False, angle=0.0)
    with pytest.raises(ValueError):
        m.revolve([(1.0, 0.0), (float("nan"), 1.0)], False)
    over = m.revolve([(2, 0), (3, 0), (3, 4), (2, 4)], True, angle=20.0, segments=16)
    assert_closed_manifold(over)  # angle clamps to a full revolution


# ---------------------------------------------------------------- sweep

def test_sweep_round_tube_volume():
    radius, length = 2.0, 100.0
    mesh = m.sweep([(0.0, 0.0), (length, 0.0)], False, radius, "ROUND", 16)
    assert_closed_manifold(mesh)
    assert volume(mesh) == pytest.approx(math.pi * radius ** 2 * length, rel=0.03)
    assert euler(mesh) == 2
    assert set(mesh.face_materials) == {MAT_SIDE, MAT_FACE}
    assert len(mesh.uvs) == sum(len(f) for f in mesh.faces)
    ys = [v[1] for v in mesh.verts]
    zs = [v[2] for v in mesh.verts]
    assert max(ys) == pytest.approx(radius) and min(zs) == pytest.approx(-radius)


def test_sweep_bent_and_closed_paths():
    bent = m.sweep([(0, 0), (50, 0), (50, 50)], False, 1.0, "ROUND", 12)
    assert_closed_manifold(bent)
    # Pappus: a mitered corner keeps the thickness, so volume tracks the polyline length
    section_area = 0.5 * 12 * math.sin(2 * math.pi / 12)  # inscribed 12-gon of radius 1
    assert volume(bent) == pytest.approx(section_area * 100.0, rel=0.03)
    ring = m.sweep([(0, 0), (40, 0), (40, 30), (0, 30)], True, 1.0, "ROUND", 12)
    assert_closed_manifold(ring)
    assert euler(ring) == 0
    assert MAT_FACE not in ring.face_materials  # closed: no caps
    assert volume(ring) > 0


def test_sweep_square_and_ribbon_profiles():
    sq = m.sweep([(0, 0), (30, 0)], False, 1.5, "SQUARE")
    assert_closed_manifold(sq)
    assert volume(sq) == pytest.approx(3.0 * 3.0 * 30.0)
    ribbon = m.sweep([(0, 0), (30, 0), (30, 10)], False, 2.0, "RIBBON")
    ribbon.validate()
    assert set(ribbon.face_materials) == {MAT_SIDE}
    assert set(edge_use(ribbon).values()) == {1, 2}  # open sheet
    assert all(abs(v[2]) < 1e-12 for v in ribbon.verts)
    area = sum(face_area(ribbon, f) for f in ribbon.faces)
    assert area == pytest.approx(4.0 * 40.0, rel=0.02)
    for f in ribbon.faces:  # faces look up (+Z)
        a, b, c = (ribbon.verts[i] for i in f[:3])
        assert (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0
    nocap = m.sweep([(0, 0), (30, 0)], False, 1.0, "ROUND", 8, cap_ends=False)
    assert MAT_FACE not in nocap.face_materials


def test_sweep_validation():
    with pytest.raises(ValueError):
        m.sweep([(0, 0), (1, 0)], False, 0.0)
    with pytest.raises(ValueError):
        m.sweep([(0, 0), (1, 0)], False, 1.0, "TRIANGLE")
    with pytest.raises(ValueError):
        m.sweep([(0, 0)], False, 1.0)
    with pytest.raises(ValueError):
        m.sweep([(0, 0), (1, 0)], True, 1.0)
    with pytest.raises(ValueError):
        m.sweep([(0, 0), (float("inf"), 0)], False, 1.0)
    # profile_segments below 3 is raised to a triangle
    tube = m.sweep([(0, 0), (5, 0)], False, 1.0, "ROUND", 1)
    assert_closed_manifold(tube)


# ---------------------------------------------------------------- orient / place_origin / scale

def test_orient_upright_and_flat():
    base = m.extrude([(SQ, [HOLE])], 2.0)
    flat = m.orient(base, "FLAT")
    assert flat.verts == base.verts and flat is not base
    up = m.orient(base, "UPRIGHT")
    assert up.faces == base.faces and up.uvs == base.uvs
    for (x, y, z), (ux, uy, uz) in zip(base.verts, up.verts):
        assert (ux, uy, uz) == (x, -z, y)
    assert volume(up) == pytest.approx(volume(base))  # proper rotation keeps outward winding
    assert max(v[2] for v in up.verts) == pytest.approx(10.0)  # drawing stands on Z
    with pytest.raises(ValueError):
        m.orient(base, "SIDEWAYS")


def test_place_origin_modes():
    mesh = MeshData(verts=[(2.0, 4.0, 6.0), (10.0, 8.0, 12.0), (4.0, 6.0, 8.0)], faces=[(0, 1, 2)])
    center = m.place_origin(mesh, "CENTER")
    assert center.verts[0] == (-4.0, -2.0, -3.0) and center.verts[1] == (4.0, 2.0, 3.0)
    bottom = m.place_origin(mesh, "BOTTOM")
    assert bottom.verts[0] == (-4.0, -2.0, 0.0) and bottom.verts[1] == (4.0, 2.0, 6.0)
    mn = m.place_origin(mesh, "MIN")
    assert mn.verts[0] == (0.0, 0.0, 0.0) and mn.verts[1] == (8.0, 4.0, 6.0)
    src = m.place_origin(mesh, "SOURCE", (1.0, 1.0, 1.0))
    assert src.verts[0] == (1.0, 3.0, 5.0)
    assert mesh.verts[0] == (2.0, 4.0, 6.0)  # input untouched
    assert m.place_origin(MeshData(), "CENTER").verts == []
    with pytest.raises(ValueError):
        m.place_origin(mesh, "MIDDLE")
    ext = m.extrude([(SQ, [])], 2.0, center=False)
    placed = m.place_origin(ext, "BOTTOM")
    assert min(v[2] for v in placed.verts) == 0.0
    assert min(v[0] for v in placed.verts) == pytest.approx(-5.0)


def test_scale_positive_and_negative():
    base = m.extrude([(SQ, [])], 2.0)
    half = m.scale(base, 0.5)
    assert volume(half) == pytest.approx(volume(base) * 0.125)
    assert half.faces == base.faces and half.uvs == base.uvs
    mirrored = m.scale(base, -1.0)
    mirrored.validate()
    assert volume(mirrored) == pytest.approx(volume(base))  # winding flipped to keep outward normals
    assert len(mirrored.uvs) == len(base.uvs)
    with pytest.raises(ValueError):
        m.scale(base, 0.0)
    with pytest.raises(ValueError):
        m.scale(base, float("nan"))


def test_mesh_signature_is_stable_and_sensitive():
    mesh = m.extrude([(SQ, [HOLE])], 2.0)
    s1 = m.mesh_signature(mesh.verts, mesh.faces)
    s2 = m.mesh_signature(list(mesh.verts), [list(f) for f in mesh.faces])
    assert s1 == s2 and len(s1) == 32
    moved = list(mesh.verts)
    moved[0] = (moved[0][0] + 1e-3, moved[0][1], moved[0][2])
    assert m.mesh_signature(moved, mesh.faces) != s1
    nudged = list(mesh.verts)
    nudged[0] = (nudged[0][0] + 1e-7, nudged[0][1], nudged[0][2])
    assert m.mesh_signature(nudged, mesh.faces) == s1  # below 5 decimals
    assert m.mesh_signature(mesh.verts, mesh.faces[::-1]) != s1


# ---------------------------------------------------------------- touching rings, crossing rings

def _pinch_report(mesh):
    """(edges not used 2 or 4 times, edges used 4 times that are not vertical wall edges)."""
    use = edge_use(mesh)
    odd = [e for e, n in use.items() if n not in (2, 4)]
    four_not_vertical = [e for e, n in use.items() if n == 4
                         and not (mesh.verts[e[0]][:2] == mesh.verts[e[1]][:2])]
    return odd, four_not_vertical


@pytest.mark.parametrize("name,holes,hole_area", [
    ("hole vertex on outer edge", [[(5.0, 0.0), (3.0, 3.0), (7.0, 3.0)]], 6.0),
    ("hole vertex on outer vertex", [[(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)]], 1.5),
    ("two holes share a vertex", [[(2.0, 2.0), (4.0, 4.0), (4.0, 2.0)], [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0)]], 4.0),
    ("hole edge touches hole vertex", [[(2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (4.0, 2.0)], [(4.0, 3.0), (6.0, 2.0), (6.0, 4.0)]], 6.0),
])
def test_extrude_touching_rings_is_watertight_with_correct_volume(name, holes, hole_area):
    ccw_holes = []
    for h in holes:
        ccw_holes.append(h if g.signed_area(h) < 0 else h[::-1])
    mesh = m.extrude([(SQ, ccw_holes)], 2.0)
    mesh.validate()
    odd, four = _pinch_report(mesh)
    assert odd == [] and four == [], name
    assert volume(mesh) == pytest.approx((100.0 - hole_area) * 2.0, abs=1e-9)
    assert min(face_area(mesh, f) for f in mesh.faces) > 0.0
    # the pinch point shares one vertex between the rings (no duplicate positions in a layer)
    layer = [(round(x, 9), round(y, 9)) for x, y, z in mesh.verts if z > 0]
    assert len(layer) == len(set(layer))


def test_extrude_corner_touching_square_holes_are_watertight():
    rng = random.Random(11)
    checked = pinched = 0
    for _ in range(80):
        n = 6
        missing = set()
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if rng.random() < 0.45 and not any((i + di, j + dj) in missing for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    missing.add((i, j))     # never edge-adjacent, so holes only meet at corners
        if not missing:
            continue
        outer = [(0.0, 0.0), (float(n), 0.0), (float(n), float(n)), (0.0, float(n))]
        holes = [[(float(i), float(j)), (float(i), j + 1.0), (i + 1.0, j + 1.0), (i + 1.0, float(j))] for i, j in sorted(missing)]
        mesh = m.extrude([(outer, holes)], 1.0)
        odd, four = _pinch_report(mesh)
        assert odd == [] and four == []
        assert volume(mesh) == pytest.approx((n * n - len(missing)) * 1.0, abs=1e-9)
        checked += 1
        pinched += any(n_ == 4 for n_ in edge_use(mesh).values())
    assert checked > 40 and pinched > 10


def test_extrude_rejects_crossing_rings():
    bow = [(0.0, 0.0), (4.0, 4.0), (4.0, 0.0), (0.0, 4.0)]
    with pytest.raises(ValueError):
        m.extrude([(bow, [])], 1.0)
