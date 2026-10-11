"""Pathfinder booleans and compound paths (vector-b)."""

import bpy
import pytest

from test_vector_b_common import add_ellipse, add_rect, require, select, spline_area


def shapes():
    from seam_studio.vector import canvas
    return canvas.artwork_objects(bpy.context.scene, roles=("SHAPE",))


@pytest.fixture
def two_squares(seam):
    require("geometry.flatten_path", "boolean2d.boolean", "boolean2d.union_all")
    scene = bpy.context.scene
    a = add_rect(scene, "A", 0, 0, 100, 100, color=(1.0, 0.0, 0.0, 1.0))
    b = add_rect(scene, "B", 50, 50, 150, 150, color=(0.0, 1.0, 0.0, 1.0))
    return a, b


@pytest.mark.parametrize("op,expected", [("UNION", 17500.0), ("INTERSECT", 2500.0), ("DIFFERENCE", 7500.0),
                                         ("EXCLUDE", 15000.0)])
def test_pathfinder_areas(two_squares, op, expected):
    a, b = two_squares
    select(a, b)
    bpy.ops.seam.pathfinder(op=op)
    res = shapes()
    assert len(res) == 1, [o.name for o in res]
    r = res[0]
    assert r.seam.role == "SHAPE" and r.type == "CURVE"
    assert spline_area(r) == pytest.approx(expected, rel=1e-4)
    assert "A" not in bpy.data.objects and "B" not in bpy.data.objects     # originals replaced
    # all handles are VECTOR (polygon result)
    assert all(bp.handle_left_type == "VECTOR" and bp.handle_right_type == "VECTOR"
               for sp in r.data.splines for bp in sp.bezier_points)
    assert all(sp.use_cyclic_u for sp in r.data.splines)


def test_result_takes_bottom_style_and_top_slot(two_squares):
    a, b = two_squares
    a.seam.style.stroke_enabled = True
    a.seam.style.stroke_width = 5.0
    top_z = b.location.z
    select(a, b)
    bpy.ops.seam.pathfinder(op="UNION")
    r = shapes()[0]
    assert tuple(r.seam.style.fill_color) == (1.0, 0.0, 0.0, 1.0)            # bottom-most = A
    assert r.seam.style.stroke_enabled and r.seam.style.stroke_width == 5.0
    assert r.location.z == pytest.approx(top_z)
    assert r.parent is not None and r.parent.seam.role == "CANVAS_ROOT"
    assert r in bpy.context.selected_objects and bpy.context.view_layer.objects.active is r


def test_difference_uses_z_order_not_selection_order(two_squares):
    a, b = two_squares                                    # A below B
    select(b, a)                                          # select top first
    bpy.ops.seam.pathfinder(op="DIFFERENCE")
    r = shapes()[0]
    assert spline_area(r) == pytest.approx(7500.0, rel=1e-4)    # A - B, not B - A
    assert tuple(r.seam.style.fill_color) == (1.0, 0.0, 0.0, 1.0)


def test_keep_originals(two_squares):
    a, b = two_squares
    select(a, b)
    bpy.ops.seam.pathfinder(op="UNION", keep_originals=True)
    names = [o.name for o in shapes()]
    assert len(names) == 3 and "A" in names and "B" in names
    result = [o for o in shapes() if o.name not in ("A", "B")][0]
    assert result.location.z > max(a.location.z, b.location.z)         # stays above the originals


def test_hole_becomes_extra_spline(seam):
    require("geometry.flatten_path", "boolean2d.boolean", "boolean2d.union_all")
    scene = bpy.context.scene
    big = add_rect(scene, "Big", 0, 0, 100, 100)
    small = add_rect(scene, "Small", 30, 30, 70, 70)
    select(big, small)
    bpy.ops.seam.pathfinder(op="DIFFERENCE")
    r = shapes()[0]
    assert len(r.data.splines) == 2
    assert spline_area(r) == pytest.approx(10000.0 - 1600.0, rel=1e-4)
    areas = sorted(_signed(sp) for sp in r.data.splines)
    assert areas[0] < 0 < areas[1]                                      # CW hole, CCW outer


def _signed(sp):
    pts = [(p.co[0], p.co[1]) for p in sp.bezier_points]
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                     for i in range(len(pts)))


def test_three_shapes_union_and_intersect(seam):
    require("geometry.flatten_path", "boolean2d.boolean", "boolean2d.union_all")
    scene = bpy.context.scene
    objs = [add_rect(scene, n, x, 0, x + 100, 100) for n, x in (("S1", 0), ("S2", 40), ("S3", 80))]
    select(*objs)
    bpy.ops.seam.pathfinder(op="INTERSECT")
    assert spline_area(shapes()[0]) == pytest.approx(20 * 100.0, rel=1e-4)       # x in [80, 100]
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    objs = [add_rect(scene, n, x, 0, x + 100, 100) for n, x in (("S1", 0), ("S2", 40), ("S3", 80))]
    select(*objs)
    bpy.ops.seam.pathfinder(op="UNION")
    assert spline_area(shapes()[0]) == pytest.approx(180 * 100.0, rel=1e-4)


def test_curved_shapes_flatten_at_resolution(seam):
    require("geometry.flatten_path", "boolean2d.boolean", "boolean2d.union_all")
    import math
    scene = bpy.context.scene
    c1 = add_ellipse(scene, "C1", 0, 0, 50, 50)
    c2 = add_ellipse(scene, "C2", 200, 0, 50, 50)
    select(c1, c2)
    bpy.ops.seam.pathfinder(op="UNION", resolution=24)
    r = shapes()[0]
    assert spline_area(r) == pytest.approx(2 * math.pi * 2500.0, rel=0.01)
    assert len(r.data.splines) == 2
    assert all(len(sp.bezier_points) >= 24 for sp in r.data.splines)


def test_transformed_objects_use_canvas_space(seam):
    require("geometry.flatten_path", "boolean2d.boolean", "boolean2d.union_all")
    import math
    scene = bpy.context.scene
    a = add_rect(scene, "A", 0, 0, 100, 100)
    b = add_rect(scene, "B", 0, 0, 100, 100)          # origin at (50, 50)
    b.scale = (0.5, 0.5, 1.0)                         # -> 50 x 50 centered on (50, 50)
    select(a, b)
    bpy.ops.seam.pathfinder(op="INTERSECT")
    assert spline_area(shapes()[0]) == pytest.approx(2500.0, rel=1e-4)
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    a = add_rect(scene, "A", -50, -5, 50, 5)          # horizontal bar centered on origin of canvas (0, 0)
    b = add_rect(scene, "B", -50, -5, 50, 5)
    b.rotation_euler = (0.0, 0.0, math.radians(90))   # vertical bar
    select(a, b)
    bpy.ops.seam.pathfinder(op="INTERSECT")
    assert spline_area(shapes()[0]) == pytest.approx(100.0, rel=1e-3)           # the 10 x 10 crossing


def test_errors_are_actionable(two_squares):
    a, b = two_squares
    select(a)
    with pytest.raises(RuntimeError, match="poll"):
        bpy.ops.seam.pathfinder(op="UNION")
    # disjoint intersect: nothing changes
    far = add_rect(bpy.context.scene, "Far", 1000, 1000, 1100, 1100)
    select(a, far)
    with pytest.raises(RuntimeError, match="produced no area"):
        bpy.ops.seam.pathfinder(op="INTERSECT")
    assert {"A", "Far"} <= {o.name for o in shapes()}
    # degenerate shape
    from test_vector_b_common import poly_path
    from seam_studio.vector import canvas
    flat = canvas.new_shape_object(bpy.context.scene, "Flat", [poly_path([(0, 0), (50, 0), (100, 0)])],
                                   location=(0, 300))
    select(a, flat)
    with pytest.raises(RuntimeError, match="'Flat' has no fillable area"):
        bpy.ops.seam.pathfinder(op="UNION")
    assert "Flat" in bpy.data.objects and "A" in bpy.data.objects


def test_text_needs_outlines_first(seam):
    require("geometry.flatten_path", "boolean2d.boolean")
    scene = bpy.context.scene
    bpy.ops.seam.add_text(body="T", x=0, y=0, size=40)
    t = bpy.context.active_object
    a = add_rect(scene, "A", 0, 0, 100, 100)
    select(a, t)
    with pytest.raises(RuntimeError, match="poll"):
        bpy.ops.seam.pathfinder(op="UNION")


def test_compound_make_and_release(seam):
    scene = bpy.context.scene
    a = add_rect(scene, "A", 0, 0, 100, 100, color=(1, 0, 0, 1))
    b = add_rect(scene, "B", 30, 30, 70, 70, color=(0, 1, 0, 1))
    select(a, b)
    bpy.ops.seam.compound_make()
    res = shapes()
    assert len(res) == 1 and len(res[0].data.splines) == 2
    c = res[0]
    assert c.seam.style.fill_rule == "EVENODD"
    assert tuple(c.seam.style.fill_color) == (1.0, 0.0, 0.0, 1.0)
    assert spline_area(c) == pytest.approx(10000.0 + 1600.0, rel=1e-6)            # both CCW; even-odd fill makes the hole
    assert "A" not in bpy.data.objects
    select(c)
    bpy.ops.seam.compound_release()
    parts = shapes()
    assert len(parts) == 2 and all(len(p.data.splines) == 1 for p in parts)
    assert sorted(round(spline_area(p)) for p in parts) == [1600, 10000]
    assert all(tuple(p.seam.style.fill_color) == (1.0, 0.0, 0.0, 1.0) for p in parts)
    with pytest.raises(RuntimeError, match="poll"):
        select(parts[0])
        bpy.ops.seam.compound_release()
