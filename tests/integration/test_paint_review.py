"""Paint: review fixes (merge-down look, last layer, stencil after undo, undo-loss warning, paint-mode edits)."""

import numpy as np
import pytest

import bpy

from test_paint_support import flush, kernel, new_doc, noise, paint_pixels, pixels, rd  # noqa: F401  (kernel = fixture)


def cur():
    return bpy.context.scene.seam.rasters[0]


def opaque(h, w, seed):
    a = noise(h, w, seed)
    a[..., 3] = 255
    return a


def binary_alpha(h, w, seed):
    a = noise(h, w, seed)
    a[..., 3] = np.where(np.random.default_rng(seed + 50).random((h, w)) < 0.5, 0, 255)
    return a


def merge_error(configure, upper=None, lower_pixels=None, bg="WHITE"):
    r = rd()
    doc = new_doc(6, 5, "Mg", bg)
    r.add_layer(doc, "Top")
    paint_pixels(doc.layers[0], opaque(5, 6, 1) if lower_pixels is None else lower_pixels)
    paint_pixels(doc.layers[1], binary_alpha(5, 6, 2) if upper is None else upper)
    configure(doc)
    before = r.composite_array(doc)
    r.merge_down(doc, 1)
    return float(np.abs(r.composite_array(cur()) - before).max()), cur()


def test_merge_down_bakes_lower_opacity(seam, kernel):
    def cfg(doc):
        doc.layers[0].opacity = 0.5
    err, doc = merge_error(cfg)
    assert err < 6e-3, err
    assert doc.layers[0].opacity == 1.0 and len(doc.layers) == 1


def test_merge_down_upper_blend_over_opaque_lower(seam, kernel):
    def cfg(doc):
        doc.layers[1].opacity = 0.7
        doc.layers[1].blend_mode = "MULTIPLY"
    err, _doc = merge_error(cfg, upper=noise(5, 6, 2))
    assert err < 8e-3, err


def test_merge_down_lower_opacity_and_mask_with_binary_upper(seam, kernel):
    r = rd()

    def cfg(doc):
        doc.layers[0].opacity = 0.4
        r.add_mask(doc, 0, "WHITE")
        m = np.random.default_rng(5).integers(0, 256, (5, 6), dtype=np.uint8)
        paint_pixels(doc.layers[0], np.stack([m, m, m, np.full_like(m, 255)], -1), mask=True)
    err, _doc = merge_error(cfg)
    assert err < 8e-3, err


def test_merge_down_reports_the_approximate_cases(seam, kernel):
    # partial upper alpha over a half-opacity lower layer: the kernel model is not associative, so the look moves
    err, _doc = merge_error(lambda doc: setattr(doc.layers[0], "opacity", 0.5), upper=noise(5, 6, 2))
    assert 0.01 < err < 0.3, err
    r = rd()
    doc = new_doc(6, 5, "Mr", "WHITE")
    r.add_layer(doc, "Top")
    paint_pixels(doc.layers[0], opaque(5, 6, 1))
    paint_pixels(doc.layers[1], noise(5, 6, 2))
    doc.layers[0].opacity = 0.5
    doc.active_layer_index = 1
    assert bpy.ops.seam.raster_layer_merge_down() == {"FINISHED"}  # warns, does not refuse
    assert len(cur().layers) == 1


@pytest.mark.parametrize("hide", [0, 1])
def test_merge_down_refuses_hidden_layers(seam, kernel, hide):
    r = rd()
    doc = new_doc(6, 5, "Mh", "WHITE")
    r.add_layer(doc, "Top")
    paint_pixels(doc.layers[0], opaque(5, 6, 1))
    paint_pixels(doc.layers[1], noise(5, 6, 2))
    doc.layers[hide].visible = False
    before = (r.composite_array(doc).copy(), len(doc.layers))
    with pytest.raises(ValueError, match="hidden"):
        r.merge_down(doc, 1)
    assert len(doc.layers) == 2 and np.array_equal(r.composite_array(doc), before[0])
    doc.active_layer_index = 1
    assert not bpy.ops.seam.raster_layer_merge_down.poll()  # poll_message explains it
    assert bpy.ops.seam.raster_layer_merge_down(index=1) == {"CANCELLED"} if bpy.ops.seam.raster_layer_merge_down.poll() else True
    doc.layers[hide].visible = True
    assert bpy.ops.seam.raster_layer_merge_down.poll()


def test_remove_last_layer_is_refused(seam):
    doc = new_doc(4, 4, "Last")
    with pytest.raises(ValueError, match="at least one layer"):
        rd().remove_layer(doc, 0)
    assert len(doc.layers) == 1


def test_stencil_follows_selection_through_undo_and_redo(seam, kernel):
    def ip():  # fetch anew: a held tool-settings pointer goes stale across memfile undo
        return bpy.context.scene.tool_settings.image_paint
    new_doc(10, 8, "St")
    bpy.ops.ed.undo_push(message="base0")
    bpy.ops.ed.undo_push(message="base1")
    bpy.ops.seam.raster_select_rect(x0=1, y0=1, x1=5, y1=5)
    bpy.ops.ed.undo_push(message="select")
    assert cur().has_selection and ip().use_stencil_layer
    bpy.ops.seam.raster_select_none()
    bpy.ops.ed.undo_push(message="deselect")
    assert not cur().has_selection and not ip().use_stencil_layer
    bpy.ops.ed.undo()  # the selection is back: strokes must be confined again
    assert cur().has_selection and ip().use_stencil_layer and ip().stencil_image == cur().selection
    bpy.ops.ed.undo()  # no selection: strokes must be free again
    assert not cur().has_selection and not ip().use_stencil_layer
    bpy.ops.ed.redo()
    assert cur().has_selection and ip().use_stencil_layer
    bpy.ops.ed.redo()
    assert not cur().has_selection and not ip().use_stencil_layer


def test_evicted_journal_is_reported_not_silent(seam, monkeypatch):
    r = rd()
    doc = new_doc(16, 16, "Ev")
    for i in range(2):
        bpy.ops.ed.undo_push(message=f"b{i}")
    monkeypatch.setattr(r, "JOURNAL_LIMIT_BYTES", 2500)
    for i in range(6):
        paint_pixels(doc.layers[0], noise(16, 16, 200 + i))
        bpy.ops.ed.undo_push(message=f"p{i}")
    assert r.pixel_restore_warning() == ""
    for _ in range(5):
        bpy.ops.ed.undo()
    assert "memory limit" in r.pixel_restore_warning()
    bpy.ops.ed.redo()
    r.restore_pixels_after_undo()  # a later restore recomputes the state
    assert isinstance(r.pixel_restore_warning(), str)


def test_pixel_operators_keep_paint_mode_and_edit_pixels(seam, kernel):
    doc = new_doc(8, 6, "Pm")
    assert bpy.ops.seam.raster_paint_mode() == {"FINISHED"}
    obj = cur().canvas
    assert obj.mode == "TEXTURE_PAINT"
    assert bpy.ops.seam.raster_fill(color=(1.0, 0.0, 0.0, 1.0)) == {"FINISHED"}
    assert obj.mode == "TEXTURE_PAINT"
    assert bpy.context.scene.tool_settings.image_paint.mode == "MATERIAL"
    assert pixels(cur().layers[0].image)[..., 3].min() == 255
    assert bpy.ops.seam.raster_layer_add() == {"FINISHED"}
    assert obj.mode == "TEXTURE_PAINT" and len(cur().layers) == 2
    assert bpy.ops.seam.raster_clear() == {"FINISHED"}
    assert obj.mode == "TEXTURE_PAINT"
    bpy.ops.seam.raster_paint_mode(enter=False)
