"""Paint: undoing a SEAM pixel op must not discard native brush strokes it did not write (headless).

A native stroke changes the pixels of a layer image without any SEAM revision.  Real strokes are covered by the GUI
test (test_paint_undo_gui.py); here they are simulated with ``Image.pixels.foreach_set`` (what the brush does)."""

import numpy as np

import bpy

from helpers import save_and_reload
from test_paint_support import kernel, new_doc, noise, paint_pixels, pixels, rd  # noqa: F401  (kernel = fixture)


def cur():
    return bpy.context.scene.seam.rasters[0]


def native_stroke(image, rgba8):
    """What a native brush stroke does to an image: pixels change, no SEAM revision is recorded."""
    a = np.asarray(rgba8, np.uint8)[::-1].astype(np.float32).reshape(-1) / 255.0
    image.pixels.foreach_set(a)
    image.update()


def base_steps():
    bpy.ops.ed.undo_push(message="base0")
    bpy.ops.ed.undo_push(message="base1")


def select_canvas():
    ob = cur().canvas
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def test_stroke_after_seam_write_survives_undo_of_next_seam_write(seam):
    doc = new_doc(8, 6, "N1")
    paint_pixels(doc.layers[0], noise(6, 8, 1))      # SEAM write: rev 1
    base_steps()
    stroke = noise(6, 8, 2)
    native_stroke(cur().layers[0].image, stroke)     # native stroke on top of rev 1 (the revision stays 1)
    assert cur().layers[0].rev == 1
    paint_pixels(cur().layers[0], noise(6, 8, 3))    # next SEAM write: rev 2
    bpy.ops.ed.undo_push(message="seam op")
    bpy.ops.ed.undo()
    assert np.array_equal(pixels(cur().layers[0].image), stroke)
    assert not rd().pixel_restore_warning()
    bpy.ops.ed.redo()
    assert np.array_equal(pixels(cur().layers[0].image), noise(6, 8, 3))


def test_stroke_on_fresh_layer_survives_undo_of_first_seam_write(seam):
    doc = new_doc(8, 6, "N2")
    base_steps()
    stroke = noise(6, 8, 4)
    native_stroke(doc.layers[0].image, stroke)       # rev 0 + native stroke
    paint_pixels(cur().layers[0], noise(6, 8, 5))
    bpy.ops.ed.undo_push(message="seam op")
    bpy.ops.ed.undo()
    assert np.array_equal(pixels(cur().layers[0].image), stroke)
    bpy.ops.ed.redo()
    assert np.array_equal(pixels(cur().layers[0].image), noise(6, 8, 5))


def test_stroke_survives_undo_of_a_mask_write(seam):
    doc = new_doc(8, 6, "N3")
    r = rd()
    r.add_mask(doc, 0, "WHITE")
    m1 = np.random.default_rng(7).integers(0, 256, (6, 8), dtype=np.uint8)
    r.write_mask_array(cur().layers[0], m1 / 255.0)
    base_steps()
    m_native = np.random.default_rng(8).integers(0, 256, (6, 8), dtype=np.uint8)
    a = np.empty((6, 8, 4), np.uint8)
    a[..., 0] = a[..., 1] = a[..., 2] = m_native
    a[..., 3] = 255
    native_stroke(cur().layers[0].mask, a)
    r.write_mask_array(cur().layers[0], np.zeros((6, 8), np.float32))
    bpy.ops.ed.undo_push(message="mask op")
    bpy.ops.ed.undo()
    assert np.array_equal(pixels(cur().layers[0].mask)[..., 0], m_native)


def test_operator_undo_keeps_strokes_made_after_the_previous_operator(seam, kernel):
    """Fill, stroke, Clear, then undo the Clear: the stroke made between the two operators is still there."""
    doc = new_doc(8, 6, "N4")
    base_steps()
    assert bpy.ops.seam.raster_fill(color=(0.0, 0.0, 1.0, 1.0)) == {"FINISHED"}
    bpy.ops.ed.undo_push(message="fill")
    after_fill = pixels(cur().layers[0].image).copy()
    stroke = after_fill.copy()
    stroke[2:4, 2:6] = (255, 0, 0, 255)
    native_stroke(cur().layers[0].image, stroke)
    assert bpy.ops.seam.raster_clear() == {"FINISHED"}
    bpy.ops.ed.undo_push(message="clear")
    assert not pixels(cur().layers[0].image)[..., 3].any()
    bpy.ops.ed.undo()
    assert np.array_equal(pixels(cur().layers[0].image), stroke)


# ------------------------------------------------------------------ after reopening a file (history is gone)

def test_native_strokes_survive_undo_of_fill_after_reopen(seam, blend_path):
    doc = new_doc(8, 6, "W")
    strokes = np.zeros((6, 8, 4), np.uint8)
    strokes[..., 0] = 255
    strokes[..., 3] = 255
    native_stroke(doc.layers[0].image, strokes)
    assert cur().layers[0].rev == 0
    save_and_reload(blend_path)
    assert np.array_equal(pixels(cur().layers[0].image), strokes)
    select_canvas()
    bpy.ops.ed.undo_push(message="base")
    assert bpy.ops.seam.raster_fill(color=(0.0, 0.0, 1.0, 1.0)) == {"FINISHED"}
    bpy.ops.ed.undo_push(message="fill")
    bpy.ops.ed.undo()
    assert np.array_equal(pixels(cur().layers[0].image), strokes)
    assert not rd().pixel_restore_warning()


def test_seam_written_pixels_return_on_undo_after_reopen(seam, blend_path):
    doc = new_doc(8, 6, "S")
    a = np.zeros((6, 8, 4), np.uint8)
    a[..., 1] = 255
    a[..., 3] = 255
    rd().write_layer_array_srgb8(doc.layers[0], a)
    save_and_reload(blend_path)
    select_canvas()
    bpy.ops.ed.undo_push(message="base")
    assert bpy.ops.seam.raster_fill(color=(0.0, 0.0, 1.0, 1.0)) == {"FINISHED"}
    bpy.ops.ed.undo_push(message="fill")
    bpy.ops.ed.undo()
    assert np.array_equal(pixels(cur().layers[0].image), a)
    assert not rd().pixel_restore_warning()


def test_missing_history_is_reported_with_its_real_reason(seam, blend_path, monkeypatch):
    """Undoing past the point where history starts after a reopen keeps the pixels and says history is missing -
    not that a memory limit was reached."""
    doc = new_doc(8, 6, "R")
    a = noise(6, 8, 9)
    rd().write_layer_array_srgb8(doc.layers[0], a)       # rev 1
    save_and_reload(blend_path)
    r = rd()
    r.journal_clear()                                    # nothing recorded for rev 1
    d = cur()
    d.layers[0].rev = 5                                  # a revision the (empty) journal never saw
    r.restore_pixels_after_undo()
    warn = r.pixel_restore_warning()
    assert warn and "memory limit" not in warn and "history" in warn
