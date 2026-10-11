"""Paint: pixels survive a write that skips the save_pre handlers (Blender's auto save / Recover Last Session)."""

import contextlib

import numpy as np

import bpy

from test_paint_support import flush, kernel, new_doc, noise, paint_pixels, pixels, rd  # noqa: F401


def cur():
    return bpy.context.scene.seam.rasters[0]


@contextlib.contextmanager
def no_save_pre():
    saved = list(bpy.app.handlers.save_pre)
    bpy.app.handlers.save_pre.clear()
    try:
        yield
    finally:
        bpy.app.handlers.save_pre.extend(saved)


def autosave_and_recover(path):
    """Write the file the way auto save does (no save_pre packing) and open it again (File > Recover > Auto Save)."""
    with no_save_pre():
        assert bpy.ops.wm.save_as_mainfile(filepath=path, copy=True) == {"FINISHED"}
    assert bpy.ops.wm.open_mainfile(filepath=path) == {"FINISHED"}
    flush()


def test_pixel_operator_results_survive_autosave(seam, blend_path, kernel):
    doc = new_doc(8, 6, "A1")
    assert bpy.ops.seam.raster_fill(color=(0.0, 0.0, 1.0, 1.0)) == {"FINISHED"}
    assert bpy.ops.seam.raster_layer_add(name="Two") == {"FINISHED"}
    assert bpy.ops.seam.raster_fill(color=(1.0, 0.0, 0.0, 1.0), opacity=0.5) == {"FINISHED"}
    want = [pixels(l.image).copy() for l in cur().layers]
    assert want[0][..., 2].min() == 255 and want[1][..., 0].min() > 0
    autosave_and_recover(blend_path)
    got = [pixels(l.image) for l in cur().layers]
    assert all(np.array_equal(w, g) for w, g in zip(want, got))
    assert rd().pixel_restore_warning() == ""


def test_untouched_layers_and_masks_do_not_come_back_black(seam, blend_path):
    doc = new_doc(8, 6, "A2")
    r = rd()
    r.add_mask(doc, 0, "WHITE")
    r.add_layer(cur(), "Empty")
    autosave_and_recover(blend_path)
    d = cur()
    assert not pixels(d.layers[0].image).any() and not pixels(d.layers[1].image).any()
    assert (pixels(d.layers[0].mask)[..., :3] == 255).all()
    flush()
    assert not pixels(d.composite)[..., 3].any()     # the composite shows nothing, not a black rectangle


def test_new_images_regenerate_to_their_fill_colour(seam):
    doc = new_doc(4, 4, "A3")
    r = rd()
    r.add_mask(doc, 0, "BLACK")
    d = cur()
    assert tuple(d.layers[0].image.generated_color) == (0.0, 0.0, 0.0, 0.0)
    assert tuple(d.layers[0].mask.generated_color) == (0.0, 0.0, 0.0, 1.0)


def test_native_strokes_survive_autosave_after_paint_mode_exit(seam, blend_path, kernel):
    doc = new_doc(8, 6, "A4")
    assert bpy.ops.seam.raster_paint_mode(frame_view=False) == {"FINISHED"}
    stroke = noise(6, 8, 3)
    a = stroke[::-1].astype(np.float32).reshape(-1) / 255.0
    cur().layers[0].image.pixels.foreach_set(a)          # a brush stroke
    cur().layers[0].image.update()
    assert bpy.ops.seam.raster_paint_mode(enter=False) == {"FINISHED"}
    autosave_and_recover(blend_path)
    assert np.array_equal(pixels(cur().layers[0].image), stroke)


def test_pixels_missing_from_a_recovered_file_are_reported_not_trusted(seam, blend_path):
    """Pixels written through the API are not packed until something packs them; a file written without them
    (the situation of an auto save from before the fix) says so instead of showing a black layer silently."""
    doc = new_doc(8, 6, "A5")
    paint_pixels(doc.layers[0], noise(6, 8, 4))          # rev 1, never packed
    assert cur().layers[0].image.packed_file is None
    autosave_and_recover(blend_path)
    r = rd()
    assert cur().layers[0].rev == 1 and cur().layers[0].image.packed_file is None
    assert r.pixel_restore_warning() and "without pixels" in r.pixel_restore_warning()
    paint_pixels(cur().layers[0], noise(6, 8, 5))        # writing the layer again clears the warning
    assert r.pixel_restore_warning() == ""


def test_lost_selection_is_dropped_instead_of_blocking_painting(seam, blend_path):
    from seam_studio.paint import selection as sel
    doc = new_doc(8, 6, "A6")
    sel.select_rect(doc, 1, 1, 4, 4)
    assert cur().has_selection
    autosave_and_recover(blend_path)
    assert not cur().has_selection
