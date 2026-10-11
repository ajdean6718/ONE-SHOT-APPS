"""Paint: undo/redo of pixel edits (pixel journal), orphan purge, resize persistence."""

import numpy as np
import pytest

import bpy

from helpers import save_and_reload
from test_paint_support import flush, kernel, new_doc, noise, paint_pixels, pixels, rd, solid  # noqa: F401


def base_steps():
    bpy.ops.ed.undo_push(message="base0")
    bpy.ops.ed.undo_push(message="base1")


def cur():
    return bpy.context.scene.seam.rasters[0]


def test_layer_pixel_writes_undo_and_redo(seam):
    doc = new_doc(8, 6, "J")
    r = rd()
    base_steps()
    a, b = noise(6, 8, 1), noise(6, 8, 2)
    paint_pixels(doc.layers[0], a)
    bpy.ops.ed.undo_push(message="A")
    paint_pixels(cur().layers[0], b)
    bpy.ops.ed.undo_push(message="B")
    assert np.array_equal(pixels(cur().layers[0].image), b)
    bpy.ops.ed.undo()
    assert np.array_equal(pixels(cur().layers[0].image), a)
    bpy.ops.ed.undo()
    assert not pixels(cur().layers[0].image).any()  # revision 0 = transparent
    bpy.ops.ed.redo()
    assert np.array_equal(pixels(cur().layers[0].image), a)
    bpy.ops.ed.redo()
    assert np.array_equal(pixels(cur().layers[0].image), b)
    assert r._applied[(cur().layers[0].id, "img")] == cur().layers[0].rev == 2


def test_mask_pixel_writes_undo(seam):
    doc = new_doc(8, 6, "JM")
    r = rd()
    r.add_mask(doc, 0, "BLACK")
    base_steps()
    m = np.random.default_rng(5).integers(0, 256, (6, 8), dtype=np.uint8)
    r.write_mask_array(cur().layers[0], m / 255.0)
    bpy.ops.ed.undo_push(message="mask")
    assert np.array_equal(pixels(cur().layers[0].mask)[..., 0], m)
    bpy.ops.ed.undo()
    assert not pixels(cur().layers[0].mask)[..., 0].any()  # the black mask of the base step
    bpy.ops.ed.redo()
    assert np.array_equal(pixels(cur().layers[0].mask)[..., 0], m)


def test_remove_layer_and_mask_are_undoable_with_pixels(seam):
    doc = new_doc(8, 6, "JR")
    r = rd()
    r.add_layer(doc, "Two")
    a, b = noise(6, 8, 3), noise(6, 8, 4)
    paint_pixels(doc.layers[0], a)
    paint_pixels(doc.layers[1], b)
    r.add_mask(doc, 1, "BLACK")
    m = np.random.default_rng(6).integers(0, 256, (6, 8), dtype=np.uint8)
    r.write_mask_array(doc.layers[1], m / 255.0)
    base_steps()
    r.remove_layer(cur(), 1)
    bpy.ops.ed.undo_push(message="remove")
    assert len(cur().layers) == 1
    bpy.ops.ed.undo()
    d = cur()
    assert [l.name for l in d.layers] == ["Layer 1", "Two"] and d.layers[1].mask is not None
    assert np.array_equal(pixels(d.layers[1].image), b)
    assert np.array_equal(pixels(d.layers[0].image), a)
    assert np.array_equal(pixels(d.layers[1].mask)[..., 0], m)
    flush()
    assert d.material[r.MAT_SIG_KEY] == r.material_signature(d)
    assert len([n for n in d.material.node_tree.nodes if n.bl_idname == "ShaderNodeTexImage"
                and n.image == d.layers[1].image]) == 1
    bpy.ops.ed.redo()
    assert len(cur().layers) == 1
    bpy.ops.ed.undo()
    # deleting only the mask
    r.remove_mask(cur(), 1)
    bpy.ops.ed.undo_push(message="remove mask")
    assert cur().layers[1].mask is None
    bpy.ops.ed.undo()
    assert np.array_equal(pixels(cur().layers[1].mask)[..., 0], m)


def test_duplicate_layer_undo_redo_keeps_pixels(seam):
    doc = new_doc(8, 6, "JD")
    r = rd()
    a = noise(6, 8, 7)
    paint_pixels(doc.layers[0], a)
    base_steps()
    r.duplicate_layer(cur(), 0)
    bpy.ops.ed.undo_push(message="dup")
    assert np.array_equal(pixels(cur().layers[1].image), a)
    bpy.ops.ed.undo()
    assert len(cur().layers) == 1 and np.array_equal(pixels(cur().layers[0].image), a)
    bpy.ops.ed.redo()
    assert len(cur().layers) == 2 and np.array_equal(pixels(cur().layers[1].image), a)


def test_merge_down_is_undoable(seam, kernel):
    doc = new_doc(6, 5, "JMg", "WHITE")
    r = rd()
    r.add_layer(doc, "Top")
    a = noise(5, 6, 8)
    a[..., 3] = 255
    b = noise(5, 6, 9)
    paint_pixels(doc.layers[0], a)
    paint_pixels(doc.layers[1], b)
    before = r.composite_array(doc)
    base_steps()
    r.merge_down(cur(), 1)
    bpy.ops.ed.undo_push(message="merge")
    merged = pixels(cur().layers[0].image).copy()
    assert not np.array_equal(merged, a) and len(cur().layers) == 1
    bpy.ops.ed.undo()
    d = cur()
    assert len(d.layers) == 2
    assert np.array_equal(pixels(d.layers[0].image), a) and np.array_equal(pixels(d.layers[1].image), b)
    assert np.array_equal(r.composite_array(d), before)
    bpy.ops.ed.redo()
    assert np.array_equal(pixels(cur().layers[0].image), merged)


def test_fill_clear_selection_undo(seam, kernel):
    from seam_studio.paint import selection as sel
    r = rd()
    doc = new_doc(10, 8, "JF")
    base_steps()
    bpy.ops.seam.raster_select_rect(x0=2, y0=2, x1=6, y1=5)
    bpy.ops.ed.undo_push(message="select")
    bpy.ops.seam.raster_fill(color=(1.0, 0.0, 0.0, 1.0))
    bpy.ops.ed.undo_push(message="fill")
    assert pixels(cur().layers[0].image)[..., 3].sum() == 255 * 12
    bpy.ops.seam.raster_select_none()
    bpy.ops.ed.undo_push(message="deselect")
    assert sel.selection_array(cur()) is None
    bpy.ops.seam.raster_clear()
    bpy.ops.ed.undo_push(message="clear")
    assert not pixels(cur().layers[0].image)[..., 3].any()
    bpy.ops.ed.undo()  # undo clear
    assert pixels(cur().layers[0].image)[..., 3].sum() == 255 * 12
    bpy.ops.ed.undo()  # undo deselect: selection comes back with its pixels
    d = cur()
    assert d.has_selection and sel.selection_array(d).sum() == 12
    bpy.ops.ed.undo()  # undo fill
    assert not pixels(cur().layers[0].image).any()  # before the fill the layer was pristine
    assert sel.selection_array(cur()).sum() == 12
    bpy.ops.ed.undo()  # undo select
    assert not cur().has_selection and not scene_stencil()
    bpy.ops.ed.redo()
    bpy.ops.ed.redo()
    assert pixels(cur().layers[0].image)[..., 3].sum() == 255 * 12


def scene_stencil():
    return bpy.context.scene.tool_settings.image_paint.use_stencil_layer


def test_resize_undo_restores_size_and_pixels(seam, blend_path):
    doc = new_doc(8, 6, "JZ")
    r = rd()
    a = noise(6, 8, 10)
    paint_pixels(doc.layers[0], a)
    r.add_mask(doc, 0, "BLACK")
    base_steps()
    r.resize_canvas(cur(), 12, 10, "CENTER")
    bpy.ops.ed.undo_push(message="resize")
    d = cur()
    assert (d.width, d.height) == (12, 10) and tuple(d.layers[0].image.size) == (12, 10)
    bpy.ops.ed.undo()
    d = cur()
    assert (d.width, d.height) == (8, 6) and tuple(d.layers[0].image.size) == (8, 6)
    assert np.array_equal(pixels(d.layers[0].image), a)
    assert not pixels(d.layers[0].mask)[..., 0].any()
    bpy.ops.ed.redo()
    d = cur()
    assert tuple(d.layers[0].image.size) == (12, 10) and np.array_equal(pixels(d.layers[0].image)[2:8, 2:10], a)
    # the resized document survives a save/reload
    scene = save_and_reload(blend_path)
    d = scene.seam.rasters[0]
    assert (d.width, d.height) == (12, 10) and tuple(d.layers[0].image.size) == (12, 10)
    assert np.array_equal(pixels(d.layers[0].image)[2:8, 2:10], a)


def test_undo_of_document_deletion_restores_pixels(seam):
    doc = new_doc(8, 6, "JX")
    r = rd()
    a = noise(6, 8, 11)
    paint_pixels(doc.layers[0], a)
    base_steps()
    r.remove_doc(bpy.context.scene, cur())
    bpy.ops.ed.undo_push(message="delete doc")
    assert len(bpy.context.scene.seam.rasters) == 0
    bpy.ops.ed.undo()
    d = cur()
    assert d.name == "JX" and np.array_equal(pixels(d.layers[0].image), a)
    assert d.canvas is not None and d.canvas.seam.raster_id == d.id
    flush()
    assert d.material[r.MAT_SIG_KEY] == r.material_signature(d)


def test_orphan_images_are_purged_on_save(seam, blend_path):
    new_doc(8, 6, "Orph")
    assert [i.name for i in bpy.data.images if i.name.startswith("SEAM Orph")]
    bpy.context.scene.seam.rasters.remove(0)  # simulate a document that vanished without cleaning its images
    assert [i.name for i in bpy.data.images if i.name.startswith("SEAM Orph")]
    foreign = bpy.data.images.new("SEAM Gradient foreign", 4, 4)  # not owned by paint: must survive
    foreign.use_fake_user = True
    live = new_doc(4, 4, "Live")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    assert not [i.name for i in bpy.data.images if i.name.startswith("SEAM Orph")]
    assert "SEAM Gradient foreign" in bpy.data.images
    assert [i.name for i in bpy.data.images if i.name.startswith("SEAM Live")]
    assert live.layers[0].image.packed_file is not None


def test_journal_is_bounded(seam, monkeypatch):
    r = rd()
    doc = new_doc(16, 16, "JB")
    monkeypatch.setattr(r, "JOURNAL_LIMIT_BYTES", 3000)
    for i in range(12):
        paint_pixels(doc.layers[0], noise(16, 16, 100 + i))  # incompressible: ~1 KB each
    assert r._journal_bytes <= 3000 + 1100 and len(r._journal) < 12
    assert doc.layers[0].rev == 12


def test_undo_after_save_tracks_packed_revision(seam, blend_path):
    doc = new_doc(8, 6, "JS")
    r = rd()
    a, b = noise(6, 8, 21), noise(6, 8, 22)
    paint_pixels(doc.layers[0], a)
    base_steps()
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)  # packs: image is clean and records its revision
    img = cur().layers[0].image
    assert img.packed_file is not None and not img.is_dirty and img["seam_packed_rev"] == 1
    paint_pixels(cur().layers[0], b)
    bpy.ops.ed.undo_push(message="B")
    assert np.array_equal(pixels(cur().layers[0].image), b) and cur().layers[0].image.is_dirty
    bpy.ops.ed.undo()
    assert np.array_equal(pixels(cur().layers[0].image), a)
    bpy.ops.ed.redo()
    assert np.array_equal(pixels(cur().layers[0].image), b)
    # reload: the packed content and revision agree, undo history starts fresh
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    d = cur()
    assert d.layers[0].rev == 2 and np.array_equal(pixels(d.layers[0].image), b)
    assert r._content_rev(d.layers[0].image, (d.layers[0].id, "img")) == 2
