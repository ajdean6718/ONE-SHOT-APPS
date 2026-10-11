"""Scenario 4 - extrude a compound shape with a hole and verify scale, shape, fill and geometry.

The compound is built with Add Shape + Make Compound Path (even-odd: the inner shape becomes a hole), then
Make 3D / Extrude.  Expectations are analytic: 1 canvas px = 0.01 m, areas / volumes / perimeters of the
polygon the hole was flattened to, closed 2-manifold topology, material split and colours.
"""

import math

import bpy
import numpy as np
import pytest

from accsupport import (assert_link_consistent, export_png, extent, flush, euler_characteristic, is_closed_manifold, make_3d, mesh_np,
                        mesh_volume, new_canvas, op, png_pixel, select, srgb8)

FILL = (0.9, 0.2, 0.1, 1.0)


def build_compound(hole_kind):
    new_canvas(800, 600)
    op(bpy.ops.seam.add_shape, kind="RECT", x=400, y=-300, width=400, height=300)
    outer = bpy.context.active_object
    outer.seam.style.fill_color = FILL
    if hole_kind == "RECT":
        op(bpy.ops.seam.add_shape, kind="RECT", x=400, y=-300, width=100, height=100)
    else:
        op(bpy.ops.seam.add_shape, kind="ELLIPSE", x=400, y=-300, width=120, height=120)
    hole = bpy.context.active_object
    flush()
    select(outer, hole)
    op(bpy.ops.seam.compound_make)
    flush()
    comp = bpy.context.active_object
    comp.seam.style.fill_color = FILL
    return comp


def face_stats(ob):
    """(cap_area, side_area) by material index, plus per-face data."""
    me = ob.data
    caps = sides = 0.0
    for p in me.polygons:
        if p.material_index == 0:
            caps += p.area
        else:
            sides += p.area
    return caps, sides


def cap_boundary_length(ob):
    """Length of the edges where a cap face (material 0) meets a side wall (material 1), per welded position."""
    me = ob.data
    co = np.round(mesh_np(ob), 6)
    kinds = {}
    for p in me.polygons:
        v = [tuple(co[i] + 0.0) for i in p.vertices]
        for i in range(len(v)):
            e = frozenset((v[i], v[(i + 1) % len(v)]))
            kinds.setdefault(e, set()).add(p.material_index)
    total = 0.0
    for e, k in kinds.items():
        if k == {0, 1}:
            a, b = tuple(e)
            total += float(np.linalg.norm(np.array(a) - np.array(b)))
    return total / 2.0                                          # front + back cap


@pytest.mark.parametrize("hole_kind", ["RECT", "ELLIPSE"])
def test_upright_extrude_with_hole_geometry(seam, engine, tmp_path, hole_kind):
    comp = build_compound(hole_kind)
    assert comp.seam.role == "SHAPE" and len(comp.data.splines) == 2
    link = make_3d(comp, "EXTRUDE", depth=0.5)                                    # defaults: UPRIGHT, 0.01 m per px
    res = link.target
    assert link.state == "LIVE" and link.build_count == 1 and res.seam.role == "RESULT"
    assert res.type == "MESH" and len(res.data.polygons) > 0

    # scale: 400 x 300 px at 0.01 m/px = 4 x 3 m, extruded 0.5 m along Y
    ex = extent(res)
    assert tuple(ex) == pytest.approx((4.0, 0.5, 3.0), abs=1e-5)
    lo, hi = mesh_np(res).min(axis=0), mesh_np(res).max(axis=0)
    assert tuple(lo) == pytest.approx((-2.0, -0.25, 0.0), abs=1e-5)                # origin BOTTOM: centred, sitting on z=0
    assert tuple(hi) == pytest.approx((2.0, 0.25, 3.0), abs=1e-5)

    # shape: net area of the cap = outer - hole (hole flattened to the same polygon the mesh uses)
    hole_area = 1.0 if hole_kind == "RECT" else math.pi * 0.6 * 0.6
    hole_len = 4.0 if hole_kind == "RECT" else 2 * math.pi * 0.6
    caps, sides = face_stats(res)
    tol = 1e-6 if hole_kind == "RECT" else 0.004
    assert caps / 2.0 == pytest.approx(12.0 - hole_area, rel=tol)                  # front + back cap
    assert sides == pytest.approx((14.0 + hole_len) * 0.5, rel=tol)                # outer ring + hole ring walls
    assert cap_boundary_length(res) == pytest.approx(14.0 + hole_len, rel=tol)
    assert mesh_volume(res) == pytest.approx((12.0 - hole_area) * 0.5, rel=tol)    # positive: normals point outward

    # topology: closed 2-manifold, genus 1 (one hole): V - E + F = 0
    me = res.data
    assert is_closed_manifold(res)
    assert euler_characteristic(res) == 0

    # the hole is really empty: nothing but the hole wall exists inside the hole's footprint
    co = mesh_np(res)
    cx, cz = 0.0, 1.5                                                              # hole centre (bbox centre in x / z)
    if hole_kind == "RECT":
        dist = lambda x, z: max(abs(x - cx), abs(z - cz))                          # noqa: E731  (Chebyshev: square hole)
        hole_r = 0.5
    else:
        dist = lambda x, z: math.hypot(x - cx, z - cz)                             # noqa: E731
        hole_r = 0.6
    assert min(dist(v[0], v[2]) for v in co) >= hole_r * 0.995                      # no vertex inside the hole
    caps_centres = [dist(p.center[0], p.center[2]) for p in me.polygons if p.material_index == 0]
    assert min(caps_centres) > hole_r * 0.95                                       # no cap face covers the hole
    rim = [v for v in co if abs(dist(v[0], v[2]) - hole_r) < 0.01 * hole_r + 1e-6]
    assert len(rim) >= (8 if hole_kind == "RECT" else 40)                          # the hole wall exists (front + back)
    assert_link_consistent(link)


def test_fill_materials_and_colours(seam, engine):
    comp = build_compound("RECT")
    link = make_3d(comp, "EXTRUDE", depth=0.3)
    res = link.target
    mats = list(res.data.materials)
    assert len(mats) == 2 and [m.get("seam_material_role") for m in mats] == ["FACE", "SIDE"]
    assert tuple(mats[0].diffuse_color) == pytest.approx(FILL, abs=1e-6)
    assert tuple(mats[1].diffuse_color) == pytest.approx((FILL[0] * 0.65, FILL[1] * 0.65, FILL[2] * 0.65, 1.0), abs=1e-6)
    bsdf = next(n for n in mats[0].node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    assert tuple(bsdf.inputs["Base Color"].default_value) == pytest.approx(FILL, abs=1e-6)
    # caps carry material 0, walls material 1, and the split matches the geometry: walls are axis-parallel quads
    me = res.data
    for p in me.polygons:
        if p.material_index == 1:
            assert abs(p.normal[1]) < 1e-5                                         # side walls are parallel to Y
        else:
            assert abs(abs(p.normal[1]) - 1.0) < 1e-5                              # caps face +-Y
    # changing the fill colour recolours the existing result without rebuilding the geometry
    n = link.build_count
    co = mesh_np(res).copy()
    comp.seam.style.fill_color = (0.1, 0.6, 0.2, 1.0)
    flush()
    assert tuple(res.data.materials[0].diffuse_color) == pytest.approx((0.1, 0.6, 0.2, 1.0), abs=1e-6)
    assert link.build_count == n and np.array_equal(mesh_np(res), co)


def test_scale_depth_and_orientation_parameters(seam, engine):
    comp = build_compound("RECT")
    link = make_3d(comp, "EXTRUDE", depth=0.5, orientation="FLAT")
    res = link.target
    assert tuple(extent(res)) == pytest.approx((4.0, 3.0, 0.5), abs=1e-5)         # FLAT: XY plane, extrusion along Z
    assert mesh_volume(res) == pytest.approx(11.0 * 0.5, rel=1e-6)
    link.params.units_scale = 0.02                                                 # a different scale: 8 x 6 m
    flush()
    assert tuple(extent(res)) == pytest.approx((8.0, 6.0, 0.5), abs=1e-5)
    assert mesh_volume(res) == pytest.approx(44.0 * 0.5, rel=1e-6)
    link.params.depth = 1.5
    flush()
    assert tuple(extent(res)) == pytest.approx((8.0, 6.0, 1.5), abs=1e-5)
    assert mesh_volume(res) == pytest.approx(44.0 * 1.5, rel=1e-6)
    assert is_closed_manifold(res) and euler_characteristic(res) == 0
    assert_link_consistent(link)


def test_bevel_keeps_hole_and_closes_surface(seam, engine):
    comp = build_compound("RECT")
    link = make_3d(comp, "EXTRUDE", depth=0.5, bevel_depth=0.05, bevel_resolution=3, orientation="FLAT")
    res = link.target
    assert tuple(extent(res)) == pytest.approx((4.0, 3.0, 0.5), abs=1e-4)          # bevel stays inside the bounds
    assert is_closed_manifold(res) and euler_characteristic(res) == 0
    v = mesh_volume(res)
    assert 0.0 < v < 11.0 * 0.5                                                    # bevel removes a little volume
    assert v == pytest.approx(11.0 * 0.5, rel=0.03)


def test_2d_export_shows_the_hole_and_no_3d(seam, engine, tmp_path):
    comp = build_compound("ELLIPSE")
    from seam_studio.vector import canvas
    ab = canvas.active_artboard(bpy.context.scene)
    make_3d(comp, "EXTRUDE", depth=0.5)
    img = export_png(tmp_path / "doc.png")
    assert png_pixel(img, ab, 250, -300) == (*srgb8(FILL[:3]), 255)                # the plate carries the fill colour
    assert png_pixel(img, ab, 400, -300) == (255, 255, 255, 255)                   # the hole is empty (page shows)
    assert png_pixel(img, ab, 400, -300 + 80)[:3] != (255, 255, 255)               # ring of the ellipse is filled
    # the 3D result is not drawn into the 2D export (it is far away on the page and role RESULT)
    non_white = (img[..., :3] != 255).any(axis=-1)
    ys, xs = np.nonzero(non_white)
    assert xs.min() >= 200 - 1 and xs.max() <= 600 and ys.min() >= 150 - 1 and ys.max() <= 450
