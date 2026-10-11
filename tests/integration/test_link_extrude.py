"""EXTRUDE geometry: compound shape with a hole, depth, units scale, orientation, origin, materials, UVs."""

import bpy
import pytest
from mathutils.geometry import intersect_point_tri_2d
from mathutils import Vector

from test_link_support import HOLE, SQUARE, extent, face_area, make_link, make_shape, mesh_arrays


def test_compound_shape_with_hole_flat(engine):
    src = make_shape("Logo", rings=(SQUARE, HOLE))
    link = make_link(src, "EXTRUDE", orientation="FLAT", origin="SOURCE", depth=0.2)
    tgt = link.target
    assert link.state == "LIVE" and link.build_count == 1, link.message
    assert tgt.name == "Logo · Extrude"
    assert tgt.seam.role == "RESULT" and tgt.seam.owner_id == link.id
    assert tgt.users_collection[0].name == "SEAM Results"
    # 200 px * 0.01 m/px = 2 m wide; z extent == depth
    ex = extent(tgt)
    assert ex[0] == pytest.approx(2.0, abs=1e-5) and ex[1] == pytest.approx(2.0, abs=1e-5)
    assert ex[2] == pytest.approx(0.2, abs=1e-5)
    # caps: (2^2 - 0.8^2) each, material 0; walls material 1
    assert face_area(tgt, 0) == pytest.approx(2 * (4.0 - 0.64), rel=1e-4)
    assert face_area(tgt, 1) == pytest.approx(0.2 * (8.0 + 3.2), rel=1e-4)
    # the hole is really empty: no cap triangle covers the centre
    me = tgt.data
    centre = Vector((0.0, 0.0))
    for p in me.polygons:
        if p.material_index != 0:
            continue
        vs = [me.vertices[i].co.xy for i in p.vertices]
        for k in range(1, len(vs) - 1):
            assert not intersect_point_tri_2d(centre, vs[0], vs[k], vs[k + 1])
    assert "UVMap" in me.uv_layers and len(me.uv_layers) == 1
    assert [m.name.split(".")[0] for m in me.materials] == ["Face", "Side"]
    assert tuple(me.materials[0].diffuse_color) == pytest.approx((0.8, 0.2, 0.1, 1.0), abs=1e-6)


def test_upright_bottom_origin_and_units_scale(engine):
    src = make_shape("Sq")
    link = make_link(src, "EXTRUDE", depth=0.5, units_scale=0.02)  # defaults: UPRIGHT, BOTTOM
    co, _ = mesh_arrays(link.target)
    ex = co.max(axis=0) - co.min(axis=0)
    assert ex[0] == pytest.approx(4.0, abs=1e-5)  # 200 px * 0.02
    assert ex[1] == pytest.approx(0.5, abs=1e-5)  # depth along Y
    assert ex[2] == pytest.approx(4.0, abs=1e-5)  # drawing stands on Z
    assert co[:, 2].min() == pytest.approx(0.0, abs=1e-6)  # BOTTOM origin
    assert co[:, 0].min() == pytest.approx(-2.0, abs=1e-5)
    # front cap faces -Y
    front = [p for p in link.target.data.polygons if p.material_index == 0 and p.center.y < 0]
    assert front and all(p.normal.y < -0.999 for p in front)


def test_results_are_placed_without_overlap(engine):
    a = make_link(make_shape("A"))
    b = make_link(make_shape("B"))
    assert tuple(a.target.location) == pytest.approx((0, 0, 0))
    assert tuple(b.target.location) == pytest.approx((2.5, 0, 0))


def test_source_rotation_scale_used_but_not_translation(engine):
    src = make_shape("Rot", location=(300.0, -500.0))
    src.scale = (2.0, 1.0, 1.0)
    link = make_link(src, "EXTRUDE", orientation="FLAT", origin="SOURCE")
    ex = extent(link.target)
    assert ex[0] == pytest.approx(4.0, abs=1e-5) and ex[1] == pytest.approx(2.0, abs=1e-5)
    co, _ = mesh_arrays(link.target)
    assert co[:, 0].min() == pytest.approx(-2.0, abs=1e-5)  # translation removed


def test_mirrored_source_keeps_outward_normals(engine):
    src = make_shape("Mirror", rings=(SQUARE, HOLE))
    src.scale = (-1.0, 1.0, 1.0)
    link = make_link(src, "EXTRUDE", orientation="FLAT", origin="SOURCE")
    me = link.target.data
    assert link.state == "LIVE"
    for p in me.polygons:
        c = p.center
        if p.material_index == 0:
            assert p.normal.z * c.z > 0  # caps point away from the middle
    walls_out = [p for p in me.polygons if p.material_index == 1 and abs(p.center.x) > 0.99]
    assert walls_out and all(p.normal.x * p.center.x > 0 for p in walls_out)
    assert face_area(link.target, 0) == pytest.approx(2 * 3.36, rel=1e-4)


def test_bevel_keeps_total_depth(engine):
    src = make_shape("Bev")
    link = make_link(src, "EXTRUDE", orientation="FLAT", origin="CENTER", depth=0.2, bevel_depth=0.03,
                     bevel_resolution=2)
    ex = extent(link.target)
    assert ex[2] == pytest.approx(0.2, abs=1e-5)
    assert ex[0] == pytest.approx(2.0, abs=1e-5) and ex[1] == pytest.approx(2.0, abs=1e-5)  # footprint kept
    assert len(link.target.data.polygons) > 10


def test_bevel_keeps_footprint_and_hole(engine):
    from test_link_support import HOLE, SQUARE
    src = make_shape("BevHole", rings=(SQUARE, HOLE))
    link = make_link(src, "EXTRUDE", orientation="FLAT", origin="SOURCE", depth=0.2, bevel_depth=0.05,
                     bevel_resolution=3)
    co, _ = mesh_arrays(link.target)
    assert extent(link.target) == pytest.approx((2.0, 2.0, 0.2), abs=1e-5)
    inner = co[(abs(co[:, 0]) < 0.7) & (abs(co[:, 1]) < 0.7)]
    assert len(inner) and float(abs(inner[:, :2]).max(axis=1).min()) == pytest.approx(0.4, abs=1e-5)  # hole size kept


def test_open_paths_ignored_with_message(engine):
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint
    src = make_shape("Mixed")
    sp = src.data.splines.new("BEZIER")
    sp.bezier_points.add(1)
    for bp, (x, y) in zip(sp.bezier_points, ((300, 0), (400, 50))):
        bp.co = (x, y, 0)
        bp.handle_left_type = bp.handle_right_type = "VECTOR"
    link = make_link(src)
    assert link.state == "LIVE"
    assert "open path" in link.message
    assert extent(link.target)[0] == pytest.approx(2.0, abs=1e-5)


def test_only_open_paths_is_error(engine):
    src = make_shape("Line", closed=False)
    link = make_link(src)
    assert link.state == "ERROR" and "closed" in link.message
    assert len(link.target.data.vertices) == 0
