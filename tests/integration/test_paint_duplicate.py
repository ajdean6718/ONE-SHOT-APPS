"""Paint: Shift+D / Alt+D of a RASTER canvas object never leaves two objects bound to one raster document."""

import numpy as np

import bpy

from helpers import save_and_reload
from test_paint_support import flush, kernel, new_doc, noise, paint_pixels, pixels, rd  # noqa: F401


def scene():
    return bpy.context.scene


def raster_objects():
    return [o for o in scene().objects if o.seam.role == "RASTER"]


def duplicate_canvas(doc, linked=False):
    ob = doc.canvas
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    assert bpy.ops.object.duplicate(linked=linked) == {"FINISHED"}
    bpy.context.view_layer.update()
    flush()
    return [o for o in raster_objects() if o != ob][0]


def check_independent(orig_id, dup):
    docs = list(scene().seam.rasters)
    assert len(docs) == 2
    objs = raster_objects()
    assert len({o.seam.raster_id for o in objs}) == 2          # one document per canvas object
    d0 = rd().get_doc(scene(), orig_id)
    d1 = rd().doc_for_object(dup)
    assert d1 is not None and d1.id != d0.id and d1.canvas == dup and d0.canvas != dup
    assert d0.material != d1.material and d0.canvas.data != d1.canvas.data
    assert d1.canvas.data.materials[0] == d1.material and d0.canvas.data.materials[0] == d0.material
    for a, b in zip(d0.layers, d1.layers):
        assert a.id != b.id and a.image != b.image
    return d0, d1


def test_shift_d_makes_an_independent_document(seam):
    doc = new_doc(8, 6, "Dup")
    r = rd()
    r.add_layer(doc, "Two")
    doc = scene().seam.rasters[0]
    a, b = noise(6, 8, 1), noise(6, 8, 2)
    paint_pixels(doc.layers[0], a)
    paint_pixels(doc.layers[1], b)
    doc.layers[1].opacity, doc.layers[1].blend_mode, doc.layers[1].visible = 0.4, "MULTIPLY", False
    r.add_mask(doc, 0, "BLACK")
    doc = scene().seam.rasters[0]
    orig_id = doc.id
    dup = duplicate_canvas(doc)
    d0, d1 = check_independent(orig_id, dup)
    assert [l.name for l in d1.layers] == ["Layer 1", "Two"] and (d1.width, d1.height) == (8, 6)
    assert np.array_equal(pixels(d1.layers[0].image), a) and np.array_equal(pixels(d1.layers[1].image), b)
    assert d1.layers[0].mask is not None and not pixels(d1.layers[0].mask)[..., 0].any()
    assert abs(d1.layers[1].opacity - 0.4) < 1e-6 and d1.layers[1].blend_mode == "MULTIPLY" and not d1.layers[1].visible
    # editing the copy leaves the original alone, and the other way round
    c = noise(6, 8, 3)
    paint_pixels(d1.layers[0], c)
    assert np.array_equal(pixels(d0.layers[0].image), a)
    paint_pixels(d0.layers[1], noise(6, 8, 4))
    assert np.array_equal(pixels(d1.layers[1].image), b)
    # the copy's material shows the copy's images
    flush()
    imgs = {n.image for n in d1.material.node_tree.nodes if n.bl_idname == "ShaderNodeTexImage"}
    assert d1.layers[0].image in imgs and d0.layers[0].image not in imgs


def test_alt_d_linked_duplicate_gets_its_own_mesh_and_material(seam):
    doc = new_doc(8, 6, "Lnk")
    paint_pixels(doc.layers[0], noise(6, 8, 5))
    orig_id = doc.id
    dup = duplicate_canvas(doc, linked=True)
    d0, d1 = check_independent(orig_id, dup)
    assert np.array_equal(pixels(d1.layers[0].image), pixels(d0.layers[0].image))


def test_deleting_the_copy_leaves_the_original_working(seam, kernel):
    doc = new_doc(8, 6, "DelCopy")
    a = noise(6, 8, 6)
    paint_pixels(doc.layers[0], a)
    orig_id = doc.id
    dup = duplicate_canvas(doc)
    _d0, d1 = check_independent(orig_id, dup)
    rd().remove_doc(scene(), d1)
    flush()
    assert len(scene().seam.rasters) == 1
    d0 = scene().seam.rasters[0]
    assert d0.id == orig_id and d0.canvas is not None and np.array_equal(pixels(d0.layers[0].image), a)
    assert bpy.ops.seam.raster_fill(color=(1.0, 0.0, 0.0, 1.0)) == {"FINISHED"}


def test_duplicate_survives_save_reload_and_undo(seam, blend_path):
    doc = new_doc(8, 6, "Persist")
    a = noise(6, 8, 7)
    paint_pixels(doc.layers[0], a)
    orig_id = doc.id
    bpy.ops.ed.undo_push(message="base0")
    bpy.ops.ed.undo_push(message="base1")
    dup = duplicate_canvas(doc)
    bpy.ops.ed.undo_push(message="duplicate")
    check_independent(orig_id, dup)
    bpy.ops.ed.undo()
    flush()
    assert len(scene().seam.rasters) == 1 and len(raster_objects()) == 1
    bpy.ops.ed.redo()
    flush()
    assert len(scene().seam.rasters) == 2 and len({o.seam.raster_id for o in raster_objects()}) == 2
    save_and_reload(blend_path)
    flush()
    assert len(scene().seam.rasters) == 2
    ids = {d.id for d in scene().seam.rasters}
    assert {o.seam.raster_id for o in raster_objects()} == ids
    for d in scene().seam.rasters:
        assert np.array_equal(pixels(d.layers[0].image), a)


def test_hook_ignores_ordinary_canvases(seam):
    doc = new_doc(8, 6, "One")
    new_doc(4, 4, "Two")
    assert rd().adopt_duplicated_canvases(scene()) == 0
    assert len(scene().seam.rasters) == 2
