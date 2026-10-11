"""Primitives: point counts / bounds, parametric regeneration, expand-on-hand-edit, operators."""

import bpy
import pytest

from helpers import flush, save_and_reload


def _need_core():
    from seam_studio.seam_core import shapes
    try:
        shapes.rect(1, 1)
    except NotImplementedError:
        pytest.skip("core pending: seam_core.shapes")


@pytest.fixture
def doc(seam):
    _need_core()
    bpy.ops.seam.canvas_new()
    return bpy.context.scene


def _add(kind, **kw):
    assert bpy.ops.seam.add_shape(kind=kind, **kw) == {"FINISHED"}
    return bpy.context.active_object


def _npts(ob):
    return sum(len(sp.bezier_points) for sp in ob.data.splines)


def _bounds(ob):
    from seam_studio.vector import canvas
    return canvas.canvas_bounds(ob)


@pytest.mark.parametrize("kind,count", [("RECT", 4), ("ELLIPSE", 4), ("LINE", 2)])
def test_primitive_point_count_and_bbox(doc, kind, count):
    ob = _add(kind, width=200, height=80, x=300, y=-200)
    assert ob.seam.role == "SHAPE" and ob.seam.shape.kind == kind
    assert _npts(ob) == count
    assert tuple(ob.location)[:2] == pytest.approx((300, -200))
    b = _bounds(ob)
    if kind == "LINE":
        assert b == pytest.approx((200, -200, 400, -200), abs=1e-3)
        assert not ob.seam.style.fill_enabled and ob.seam.style.stroke_enabled
        assert not ob.data.splines[0].use_cyclic_u
    else:
        assert b == pytest.approx((200, -240, 400, -160), abs=1e-2)
        assert ob.data.splines[0].use_cyclic_u


def test_rounded_rect_polygon_star(doc):
    rr = _add("ROUNDED_RECT", width=200, height=100, radius=20, x=0, y=0)
    assert _npts(rr) >= 8
    assert _bounds(rr) == pytest.approx((-100, -50, 100, 50), abs=1e-2)
    # radius is clamped to half of the shorter side, not an error
    rr.seam.shape.radius = 500.0
    flush()
    assert _bounds(rr) == pytest.approx((-100, -50, 100, 50), abs=1e-2)
    poly = _add("POLYGON", width=100, height=100, sides=6, x=0, y=0)
    assert _npts(poly) == 6
    b = _bounds(poly)
    assert b[3] - b[1] == pytest.approx(100.0, abs=0.1)  # pointy-top hexagon spans the full height
    assert b[2] - b[0] <= 100.01
    star = _add("STAR", width=100, height=100, sides=5, inner_ratio=0.4, x=0, y=0)
    assert _npts(star) == 10
    assert max(abs(v) for v in _bounds(star)) <= 50.01 + 1e-3


def test_new_shapes_get_ids_layer_z_and_style_defaults(doc):
    from seam_studio.vector import canvas
    a = _add("RECT", width=10, height=10)
    b = _add("ELLIPSE", width=10, height=10)
    assert a.seam.id and b.seam.id and a.seam.id != b.seam.id
    assert b.location.z > a.location.z
    assert a.parent.seam.role == "CANVAS_ROOT"
    assert canvas.active_layer(doc) in a.users_collection
    assert [o.name for o in canvas.artwork_objects(doc)] == [a.name, b.name]
    assert len(a.seam.style.gradient_stops) == 2


def test_default_position_is_active_artboard_centre(doc):
    ab = bpy.context.scene.seam.doc.active_artboard_id
    ob = _add("RECT", width=100, height=100)
    assert tuple(ob.location)[:2] == pytest.approx((960.0, -540.0))
    bpy.ops.seam.artboard_add(width=100, height=100)  # becomes active: centre (1920+100+50, -50)
    ob2 = _add("RECT", width=10, height=10)
    assert tuple(ob2.location)[:2] == pytest.approx((2070.0, -50.0))
    assert ab


def test_parametric_regeneration_follows_parameters(doc):
    ob = _add("RECT", width=100, height=50, x=0, y=0)
    sh = ob.seam.shape
    sh.width = 300
    sh.height = 20
    res = flush()
    assert any(r.key == "shape:" + ob.seam.id and r.status == "ok" for r in res)
    assert _bounds(ob) == pytest.approx((-150, -10, 150, 10), abs=1e-2)
    assert sh.kind == "RECT"
    # change kind: RECT -> POLYGON regenerates with the new point count
    sh.kind = "POLYGON"
    sh.sides = 7
    flush()
    assert sh.kind == "POLYGON" and _npts(ob) == 7
    # nothing to do when nothing changed
    assert all(r.status == "skipped" for r in flush() if r.key == "shape:" + ob.seam.id)
    # moving the object does not touch the recipe
    ob.location.x += 25
    flush()
    assert sh.kind == "POLYGON"


def test_hand_edit_expands_shape_and_stops_regeneration(doc):
    ob = _add("RECT", width=100, height=100, x=0, y=0)
    flush()
    bp = ob.data.splines[0].bezier_points[0]
    bp.co.x -= 40.0  # an actual node edit
    ob.update_tag()
    flush()
    assert ob.seam.shape.kind == "NONE"
    assert ob.seam.shape.width == 100  # recipe values stay, they just stop owning the geometry
    x_after = ob.data.splines[0].bezier_points[0].co.x
    ob.seam.shape.width = 400  # no effect now
    flush()
    assert ob.data.splines[0].bezier_points[0].co.x == pytest.approx(x_after)
    # a free path is no longer a shape key: nothing is rebuilt
    assert not [r for r in flush() if r.key.startswith("shape:") and r.status == "ok"]


def test_edit_and_parameter_change_together_prefers_the_edit(doc):
    ob = _add("ELLIPSE", width=100, height=100, x=0, y=0)
    flush()
    ob.data.splines[0].bezier_points[1].co.y += 10.0
    ob.seam.shape.width = 500.0
    flush()
    assert ob.seam.shape.kind == "NONE"
    assert ob.data.splines[0].bezier_points[1].co.y == pytest.approx(60.0, abs=1e-3)


def test_shape_expand_operator_and_poll(doc):
    ob = _add("STAR", width=100, height=100, sides=5)
    assert bpy.ops.seam.shape_expand.poll()
    assert bpy.ops.seam.shape_expand() == {"FINISHED"}
    assert ob.seam.shape.kind == "NONE"
    assert not bpy.ops.seam.shape_expand.poll()  # nothing parametric left
    ob.seam.shape.width = 7
    flush()
    assert _npts(ob) == 10


def test_edit_mode_curve_edit_is_detected_after_leaving_edit_mode(doc):
    ob = _add("RECT", width=100, height=100, x=0, y=0)
    flush()
    bpy.ops.object.mode_set(mode="EDIT")
    ob.data.splines[0].bezier_points[2].co.x += 30  # live edit-mode data
    bpy.ops.object.mode_set(mode="OBJECT")
    flush()
    assert ob.seam.shape.kind == "NONE"


def test_shape_params_survive_save_and_reload(doc, blend_path):
    ob = _add("ROUNDED_RECT", width=120, height=60, radius=12, x=40, y=-40)
    name = ob.name
    flush()
    save_and_reload(blend_path)
    ob = bpy.data.objects[name]
    assert ob.seam.shape.kind == "ROUNDED_RECT"
    assert (ob.seam.shape.width, ob.seam.shape.height, ob.seam.shape.radius) == pytest.approx((120, 60, 12))
    from seam_studio.engine import core
    core.request_revalidate()
    core.flush()
    assert ob.seam.shape.kind == "ROUNDED_RECT"  # revalidation after load must not "expand" it
    assert _bounds(ob) == pytest.approx((-20, -70, 100, -10), abs=1e-2)


def test_undo_redo_keeps_recipe_and_geometry_consistent(doc):
    ob = _add("RECT", width=100, height=100, x=0, y=0)
    name = ob.name
    flush()
    bpy.ops.ed.undo_push(message="before")
    ob.seam.shape.width = 300
    flush()
    bpy.ops.ed.undo_push(message="after")
    bpy.ops.ed.undo()
    flush()
    ob = bpy.data.objects[name]
    assert ob.seam.shape.width == pytest.approx(100)
    assert _bounds(ob) == pytest.approx((-50, -50, 50, 50), abs=1e-2)
    assert ob.seam.shape.kind == "RECT"
    bpy.ops.ed.redo()
    flush()
    ob = bpy.data.objects[name]
    assert ob.seam.shape.width == pytest.approx(300)
    assert _bounds(ob) == pytest.approx((-150, -50, 150, 50), abs=1e-2)
