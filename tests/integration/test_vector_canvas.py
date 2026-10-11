"""Canvas operators: canvas_new, artboards (producer), layers (flags/order), adoption, id healing."""

import bpy
import pytest
from mathutils import Vector

from helpers import flush, save_and_reload


def _canvas():
    from seam_studio.vector import canvas
    return canvas


def _mesh_bounds(ob):
    xs = [v.co.x for v in ob.data.vertices]
    ys = [v.co.y for v in ob.data.vertices]
    return min(xs), min(ys), max(xs), max(ys)


def test_canvas_new_is_idempotent_and_sets_standard_view(seam):
    canvas = _canvas()
    scene = bpy.context.scene
    assert bpy.ops.seam.canvas_new(width=800, height=600) == {"FINISHED"}
    assert len(canvas.artboards(scene)) == 1 and len(canvas.layers(scene)) == 1
    ab = canvas.active_artboard(scene)
    assert (ab.seam.artboard.width, ab.seam.artboard.height) == (800, 600)
    assert scene.view_settings.view_transform == "Standard"
    assert scene.seam.doc.canvas_ready
    # native tools land in the SEAM layer and at the artboard centre
    assert canvas.active_layer(scene) == canvas.layers(scene)[0]
    assert bpy.context.view_layer.active_layer_collection.collection == canvas.layers(scene)[0]
    assert tuple(scene.cursor.location) == pytest.approx((400.0, -20000.0 - 300.0, 0.0))
    bpy.ops.seam.canvas_new(width=100, height=100)  # must not resize the existing artboard
    assert len(canvas.artboards(scene)) == 1
    assert canvas.active_artboard(scene).seam.artboard.width == 800


def test_artboard_add_places_right_and_activates(seam):
    canvas = _canvas()
    scene = bpy.context.scene
    bpy.ops.seam.canvas_new(width=1000, height=500)
    bpy.ops.seam.artboard_add(width=400, height=300)
    abs_ = canvas.artboards(scene)
    assert len(abs_) == 2
    second = canvas.active_artboard(scene)
    assert second.name == "Artboard 2"
    assert second.location.x == pytest.approx(1100.0) and second.location.y == pytest.approx(0.0)
    assert second.parent.seam.role == "CANVAS_ROOT"
    # explicit position
    bpy.ops.seam.artboard_add(auto_position=False, x=-700, y=-900, width=200, height=100, name="Side")
    side = canvas.active_artboard(scene)
    assert side.name == "Side" and tuple(side.location)[:2] == (-700.0, -900.0)
    # set active by name
    first = abs_[0]
    assert bpy.ops.seam.artboard_set_active(name=first.name) == {"FINISHED"}
    assert canvas.active_artboard(scene) == first
    with pytest.raises(RuntimeError, match="Not an artboard"):
        bpy.ops.seam.artboard_set_active(name="nope")
    # remove
    assert bpy.ops.seam.artboard_remove() == {"FINISHED"}
    assert len(canvas.artboards(scene)) == 2


def test_artboard_resize_via_props_regenerates_mesh_and_material(seam, engine):
    canvas = _canvas()
    scene = bpy.context.scene
    bpy.ops.seam.canvas_new(width=1000, height=500)
    ab = canvas.active_artboard(scene)
    flush()
    assert _mesh_bounds(ab) == (0.0, -500.0, 1000.0, 0.0)
    ab.seam.artboard.width = 640
    ab.seam.artboard.height = 480
    ab.seam.artboard.background_color = (0.25, 0.5, 0.75, 1.0)
    res = flush()
    assert any(r.key.startswith("artboard:") and r.status == "ok" for r in res)
    assert _mesh_bounds(ab) == (0.0, -480.0, 640.0, 0.0)
    mat = ab.data.materials[0]
    assert mat["seam_owner"] == ab.seam.id
    assert tuple(mat.diffuse_color)[:3] == pytest.approx((0.25, 0.5, 0.75))
    emi = mat.node_tree.nodes["SEAM_emission"].inputs["Color"].default_value
    assert tuple(emi)[:3] == pytest.approx((0.25, 0.5, 0.75))
    assert mat.node_tree.nodes["SEAM_mix"].inputs[0].default_value == pytest.approx(1.0)
    assert len(ab.data.uv_layers) == 1
    # background off -> transparent + wire
    ab.seam.artboard.background_enabled = False
    flush()
    assert mat.node_tree.nodes["SEAM_mix"].inputs[0].default_value == pytest.approx(0.0)
    assert ab.display_type == "WIRE"  # artboard planes are flat: wire shows only the frame
    # second flush without edits is idempotent
    assert all(r.status in ("skipped", "obsolete") for r in flush() if r.key.startswith("artboard:"))


def test_artboard_survives_save_and_reload(seam, blend_path):
    canvas = _canvas()
    bpy.ops.seam.canvas_new(width=640, height=480)
    flush()
    scene = save_and_reload(blend_path)
    ab = canvas.active_artboard(scene)
    assert ab is not None and (ab.seam.artboard.width, ab.seam.artboard.height) == (640, 480)
    assert _mesh_bounds(ab) == (0.0, -480.0, 640.0, 0.0)


def test_view_artboard_is_a_reported_no_op_in_background(seam):
    assert bpy.app.background
    assert bpy.ops.seam.view_artboard() == {"CANCELLED"}  # warns "needs a 3D viewport", never raises


def test_layers_add_remove_move_active(seam):
    canvas = _canvas()
    scene = bpy.context.scene
    bpy.ops.seam.canvas_new()
    a = canvas.layers(scene)[0]
    bpy.ops.seam.layer_add(name="B")
    bpy.ops.seam.layer_add()
    names = [c.name for c in canvas.layers(scene)]
    assert names == [a.name, "B", "Layer 2"] or names[:2] == [a.name, "B"]
    assert canvas.active_layer(scene).name == names[-1]
    top = canvas.layers(scene)[-1]
    # objects with z slots
    for i, col in enumerate(canvas.layers(scene)):
        canvas.new_shape_object(scene, f"S{i}", [], layer=col)
    # lock + hide state must survive reordering
    b = bpy.data.collections["B"]
    b.hide_select = True
    canvas.layer_collection_of(b).hide_viewport = True
    canvas.set_active_layer(scene, b)
    assert bpy.ops.seam.layer_move(direction="UP") == {"FINISHED"}
    assert [c.name for c in canvas.layers(scene)] == [a.name, top.name, "B"]
    assert b.hide_select and canvas.layer_collection_of(b).hide_viewport
    assert canvas.active_layer(scene) == b
    # z slots follow the layer order
    assert bpy.data.objects["S1"].location.z == pytest.approx(2 * 10.0)
    assert bpy.data.objects["S2"].location.z == pytest.approx(1 * 10.0)
    assert bpy.ops.seam.layer_move(direction="UP") == {"CANCELLED"}  # already on top
    assert bpy.ops.seam.layer_move(direction="DOWN") == {"FINISHED"}
    assert [c.name for c in canvas.layers(scene)] == [a.name, "B", top.name]
    # set active by name
    assert bpy.ops.seam.layer_set_active(name=a.name) == {"FINISHED"}
    assert canvas.active_layer(scene) == a
    # remove: artwork moves to neighbour
    canvas.set_active_layer(scene, b)
    assert bpy.ops.seam.layer_remove() == {"FINISHED"}
    assert "B" not in bpy.data.collections
    assert bpy.data.objects["S1"].users_collection[0] in canvas.layers(scene)
    assert len(canvas.layers(scene)) == 2
    bpy.ops.seam.layer_remove()
    assert len(canvas.layers(scene)) == 1
    assert not bpy.ops.seam.layer_remove.poll()


def test_layer_visibility_and_lock_are_native_flags(seam):
    canvas = _canvas()
    scene = bpy.context.scene
    bpy.ops.seam.canvas_new()
    ob = canvas.new_shape_object(scene, "S", [])
    layer = canvas.layers(scene)[0]
    lc = canvas.layer_collection_of(layer)
    lc.hide_viewport = True
    layer.hide_select = True
    bpy.context.view_layer.update()
    assert ob.hide_get() is True or ob.visible_get() is False
    assert ob.hide_select is False  # object flag itself untouched; the collection flag locks it
    assert layer.hide_select


def test_adoption_hook_adopts_curves_in_layers_preserving_world_matrix(seam, engine):
    canvas = _canvas()
    scene = bpy.context.scene
    bpy.ops.seam.canvas_new()
    layer = canvas.layers(scene)[0]
    cu = bpy.data.curves.new("Hand", "CURVE")
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(2)
    for bp, p in zip(sp.bezier_points, [(0, 0), (50, 0), (50, 50)]):
        bp.co = (p[0], p[1], 0.0)
        bp.handle_left_type = bp.handle_right_type = "VECTOR"
    ob = bpy.data.objects.new("Hand", cu)
    ob.location = (123.0, -20000.0 + 40.0, 0.0)
    layer.objects.link(ob)
    assert ob.seam.role == "NONE"  # nothing runs until the depsgraph updates
    flush()
    world_before = Vector((123.0, -20000.0 + 40.0, 0.0))
    assert ob.seam.role == "SHAPE" and ob.seam.id
    assert ob.parent.seam.role == "CANVAS_ROOT"
    assert ob.matrix_world.translation.x == pytest.approx(world_before.x, abs=1e-4)
    assert ob.matrix_world.translation.y == pytest.approx(world_before.y, abs=1e-3)
    assert ob.location.y == pytest.approx(40.0, abs=1e-3)
    assert cu.dimensions == "2D"
    # a text object in a layer is adopted as TEXT
    tcu = bpy.data.curves.new("Txt", "FONT")
    tob = bpy.data.objects.new("Txt", tcu)
    layer.objects.link(tob)
    flush()
    assert tob.seam.role == "TEXT" and tob.parent.seam.role == "CANVAS_ROOT"
    assert tob.location.z > ob.location.z
    # objects outside SEAM layers are left alone
    other = bpy.data.objects.new("Other", bpy.data.curves.new("Other", "CURVE"))
    bpy.context.scene.collection.objects.link(other)
    flush()
    assert other.seam.role == "NONE"


def test_duplicated_ids_are_reissued(seam):
    canvas = _canvas()
    scene = bpy.context.scene
    bpy.ops.seam.canvas_new()
    a = canvas.new_shape_object(scene, "A", [])
    sid = a.seam.id
    b = a.copy()
    b.data = a.data.copy()
    b.name = "A.copy"
    canvas.active_layer(scene).objects.link(b)
    assert b.seam.id == sid
    flush()
    assert a.seam.id == sid and b.seam.id and b.seam.id != sid


def test_canvas_bounds_of_evaluated_geometry(seam):
    canvas = _canvas()
    from seam_studio.seam_core import shapes
    try:
        path = shapes.rect(100, 50)
    except NotImplementedError:
        pytest.skip("core pending: shapes.rect")
    scene = bpy.context.scene
    bpy.ops.seam.canvas_new()
    ob = canvas.new_shape_object(scene, "R", [path], location=(200, -100))
    assert canvas.canvas_bounds(ob) == pytest.approx((150, -125, 250, -75), abs=1e-4)
    ob.rotation_euler.z = 3.14159265358979 / 2
    bpy.context.view_layer.update()
    assert canvas.canvas_bounds(ob) == pytest.approx((175, -150, 225, -50), abs=1e-3)
    ob.scale.x = 2.0
    bpy.context.view_layer.update()
    assert canvas.canvas_bounds(ob) == pytest.approx((175, -200, 225, 0), abs=1e-3)


def test_native_curve_add_lands_in_layer_and_is_adopted(seam):
    canvas = _canvas()
    scene = bpy.context.scene
    bpy.ops.seam.canvas_new(width=800, height=600)
    bpy.ops.curve.primitive_bezier_curve_add(location=tuple(scene.cursor.location))
    ob = bpy.context.active_object
    assert canvas.layers(scene)[0] in ob.users_collection
    flush()
    assert ob.seam.role == "SHAPE" and ob.parent.seam.role == "CANVAS_ROOT"
    assert (ob.location.x, ob.location.y) == pytest.approx((400.0, -300.0), abs=1e-3)
    assert ob.data.dimensions == "2D"
    # Shift+D duplicate (native operator): ids are re-issued
    bpy.ops.object.duplicate()
    flush()
    shapes = [o for o in scene.objects if o.seam.role == "SHAPE"]
    assert len(shapes) == 2 and len({o.seam.id for o in shapes}) == 2


# ---------------------------------------------------------------- review regressions

def test_layers_create_false_never_writes(seam):
    from seam_studio.vector import canvas
    assert canvas.layers(bpy.context.scene, create=False) == []
    assert bpy.data.collections.get(canvas.ARTWORK_NAME) is None
    assert canvas.layers(bpy.context.scene) == []  # the default still creates the container
    assert bpy.data.collections.get(canvas.ARTWORK_NAME) is not None


def test_layer_panel_index_getter_does_not_create_collections(seam):
    wm = bpy.context.window_manager
    n = len(bpy.data.collections)
    assert wm.seam_layer_index == 0
    assert len(bpy.data.collections) == n
