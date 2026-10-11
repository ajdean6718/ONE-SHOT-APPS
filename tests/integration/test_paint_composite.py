"""Paint: composite, cache producer, selection, fill/clear, masks, merge, resize, PNG I/O (need the raster kernel)."""

import os

import numpy as np
import pytest

import bpy

from helpers import save_and_reload
from test_paint_support import flush, kernel, new_doc, noise, paint_pixels, pixels, rd, solid  # noqa: F401


def lin(c8):
    from seam_studio.seam_core import raster as R
    return R.srgb_to_linear(np.asarray(c8, np.float32) / 255.0)


def test_composite_empty_transparent_and_backgrounds(seam, kernel):
    r = rd()
    doc = new_doc(5, 4, "A")
    c = r.composite_array(doc)
    assert c.shape == (4, 5, 4) and c.dtype == np.float32 and not c.any()
    doc.background = "WHITE"
    assert np.allclose(r.composite_array(doc), 1.0)
    doc.background = "COLOR"
    doc.background_color = (0.2, 0.4, 0.6, 1.0)
    assert np.allclose(r.composite_array(doc)[0, 0], (0.2, 0.4, 0.6, 1.0), atol=1e-6)


def test_composite_matches_hand_computed_normal_blend(seam, kernel):
    """Two layers, NORMAL blend, opacity: result = lerp in linear light; alpha accumulates."""
    r = rd()
    doc = new_doc(4, 4, "N")
    r.add_layer(doc, "Top")
    paint_pixels(doc.layers[0], solid(4, 4, (255, 0, 0, 255)))
    paint_pixels(doc.layers[1], solid(4, 4, (0, 0, 255, 255)))
    doc.layers[1].opacity = 0.25
    c = r.composite_array(doc)
    red, blue = lin([255, 0, 0, 255])[:3], lin([0, 0, 255, 255])[:3]
    expect = red * 0.75 + blue * 0.25
    assert np.allclose(c[..., :3], expect, atol=1e-6) and np.allclose(c[..., 3], 1.0)
    # half transparent top layer over transparent bottom: colour is the top colour, alpha = f
    paint_pixels(doc.layers[0], solid(4, 4, (0, 0, 0, 0)))
    doc.layers[1].opacity = 1.0
    paint_pixels(doc.layers[1], solid(4, 4, (0, 0, 255, 128)))
    c = r.composite_array(doc)
    assert np.allclose(c[..., :3], blue, atol=1e-6) and np.allclose(c[..., 3], 128 / 255, atol=1e-6)


def test_property_changes_change_composite_exactly_as_kernel_predicts(seam, kernel):
    from seam_studio.seam_core import raster as R
    r = rd()
    doc = new_doc(7, 5, "P")
    r.add_layer(doc, "B")
    r.add_layer(doc, "C")
    arrs = [noise(5, 7, 41), noise(5, 7, 42), noise(5, 7, 43)]
    for ly, a in zip(doc.layers, arrs):
        paint_pixels(ly, a)
    r.add_mask(doc, 2, "WHITE")
    m8 = noise(5, 7, 44)[..., 0]
    mk = np.stack([m8, m8, m8, np.full_like(m8, 255)], -1)
    paint_pixels(doc.layers[2], mk, mask=True)

    def predict(opacities, modes, visible, mask_on, bg):
        specs = []
        for i, a in enumerate(arrs):
            if not visible[i]:
                continue
            specs.append(R.LayerSpec(rgba=R.from_uint8_srgb(a), opacity=opacities[i], blend_mode=modes[i],
                                     visible=True, mask=(m8.astype(np.float32) / 255.0 if (i == 2 and mask_on) else None)))
        return R.composite(specs, 5, 7, bg)

    base = r.composite_array(doc)
    assert np.array_equal(base, predict([1, 1, 1], ["NORMAL"] * 3, [True] * 3, True, None))
    doc.layers[1].opacity = 0.4
    c1 = r.composite_array(doc)
    assert not np.array_equal(c1, base)
    assert np.array_equal(c1, predict([1, 0.4, 1], ["NORMAL"] * 3, [True] * 3, True, None))
    doc.layers[1].blend_mode = "MULTIPLY"
    doc.layers[2].blend_mode = "SCREEN"
    c2 = r.composite_array(doc)
    assert np.array_equal(c2, predict([1, 0.4, 1], ["NORMAL", "MULTIPLY", "SCREEN"], [True] * 3, True, None))
    assert not np.array_equal(c2, c1)
    doc.layers[2].mask_enabled = False
    c3 = r.composite_array(doc)
    assert np.array_equal(c3, predict([1, 0.4, 1], ["NORMAL", "MULTIPLY", "SCREEN"], [True] * 3, False, None))
    assert not np.array_equal(c3, c2)
    doc.layers[0].visible = False
    c4 = r.composite_array(doc)
    assert np.array_equal(c4, predict([1, 0.4, 1], ["NORMAL", "MULTIPLY", "SCREEN"], [False, True, True], False, None))
    doc.background = "WHITE"
    c5 = r.composite_array(doc)
    assert np.array_equal(c5, predict([1, 0.4, 1], ["NORMAL", "MULTIPLY", "SCREEN"], [False, True, True], False,
                                      (1, 1, 1, 1)))
    assert np.allclose(c5[..., 3], 1.0)


def test_layer_array_roundtrip_is_bit_exact(seam, kernel):
    r = rd()
    doc = new_doc(9, 7, "RT")
    a8 = noise(7, 9, 5)
    paint_pixels(doc.layers[0], a8)
    arr = r.layer_array(doc.layers[0])
    assert arr.shape == (7, 9, 4) and arr.dtype == np.float32
    assert np.allclose(arr[..., 3], a8[..., 3] / 255.0)
    r.write_layer_array(doc.layers[0], arr)
    assert np.array_equal(pixels(doc.layers[0].image), a8)
    # row 0 is the top row: paint the top-left pixel red, check Blender's bottom-first buffer
    z = np.zeros((7, 9, 4), np.float32)
    z[0, 0] = (1, 0, 0, 1)
    r.write_layer_array(doc.layers[0], z)
    buf = np.empty(9 * 7 * 4, np.float32)
    doc.layers[0].image.pixels.foreach_get(buf)
    buf = buf.reshape(7, 9, 4)
    assert tuple(buf[6, 0]) == (1.0, 0.0, 0.0, 1.0) and not buf[0].any()


def test_cache_producer_keeps_composite_image_current(seam, engine, kernel):
    from seam_studio.seam_core import raster as R
    r = rd()
    doc = new_doc(6, 5, "Cache", "WHITE")
    flush()
    assert np.array_equal(pixels(doc.composite), R.to_uint8_srgb(r.composite_array(doc)))
    assert (pixels(doc.composite) == 255).all()
    # SEAM write path: touch + flush
    r.write_layer_array(doc.layers[0], np.stack([np.full((5, 6), v, np.float32) for v in (1, 0, 0, 1)], -1))
    res = flush()
    assert any(x.key == f"rastercache:{doc.id}" and x.status == "ok" for x in res)
    assert np.array_equal(pixels(doc.composite), R.to_uint8_srgb(r.composite_array(doc)))
    assert tuple(pixels(doc.composite)[0, 0]) == (255, 0, 0, 255)
    # idempotent
    assert all(x.status != "ok" for x in flush() if x.key.startswith("rastercache:"))
    # property change
    doc.layers[0].opacity = 0.5
    flush()
    assert np.array_equal(pixels(doc.composite), R.to_uint8_srgb(r.composite_array(doc)))
    assert not (pixels(doc.composite) == (255, 0, 0, 255)).all()
    # external edit detected through the fingerprint poll
    r.check_external_edits(doc)
    buf = np.zeros(6 * 5 * 4, np.float32)
    buf[:] = 1.0
    doc.layers[0].image.pixels.foreach_set(buf)
    assert r.check_external_edits(doc)
    flush()
    assert np.array_equal(pixels(doc.composite), R.to_uint8_srgb(r.composite_array(doc)))
    assert (pixels(doc.composite)[..., :3] == 255).all()


def test_cache_rebuilds_after_undo_and_load(seam, engine, kernel, blend_path):
    from seam_studio.seam_core import raster as R
    r = rd()
    doc = new_doc(6, 5, "U")
    bpy.ops.ed.undo_push(message="a")
    bpy.ops.ed.undo_push(message="b")
    doc.layers[0].opacity = 0.5
    paint_pixels(doc.layers[0], solid(5, 6, (10, 200, 30, 255)))
    flush()
    bpy.ops.ed.undo_push(message="c")
    doc.layers[0].opacity = 1.0
    flush()
    bpy.ops.ed.undo_push(message="d")
    c_full = pixels(r.get_doc(bpy.context.scene, doc.id).composite).copy()
    bpy.ops.ed.undo()
    flush()
    doc = bpy.context.scene.seam.rasters[0]
    assert doc.layers[0].opacity == 0.5
    assert np.array_equal(pixels(doc.composite), R.to_uint8_srgb(r.composite_array(doc)))
    assert pixels(doc.composite)[0, 0, 3] != c_full[0, 0, 3]
    bpy.ops.ed.redo()
    flush()
    doc = bpy.context.scene.seam.rasters[0]
    assert np.array_equal(pixels(doc.composite), c_full)
    scene = save_and_reload(blend_path)
    flush()
    doc = scene.seam.rasters[0]
    assert np.array_equal(pixels(doc.composite), c_full)
    assert doc.composite.packed_file is not None


def test_selection_ops_and_stencil(seam, kernel, blend_path):
    from seam_studio.paint import selection as sel
    r = rd()
    doc = new_doc(10, 8, "S")
    scene = bpy.context.scene
    ip = scene.tool_settings.image_paint
    assert sel.selection_array(doc) is None and not doc.has_selection
    assert bpy.ops.seam.raster_select_rect(x0=2, y0=1, x1=6, y1=5) == {"FINISHED"}
    doc = scene.seam.rasters[0]
    m = sel.selection_array(doc)
    assert doc.has_selection and m.shape == (8, 10)
    assert (m[1:5, 2:6] == 1).all() and m.sum() == 16
    assert ip.use_stencil_layer and ip.stencil_image == doc.selection and not ip.invert_stencil
    assert doc.selection.colorspace_settings.name == "Non-Color"
    assert doc.canvas.data.uv_layers[doc.canvas.data.uv_layer_stencil_index].name == "SEAM Stencil"
    # add / subtract / intersect
    bpy.ops.seam.raster_select_rect(x0=4, y0=3, x1=9, y1=7, mode="ADD")
    m2 = sel.selection_array(scene.seam.rasters[0])
    assert m2[3:5, 4:6].min() == 1 and m2[5:7, 6:9].min() == 1 and m2[1, 2] == 1 and m2[0, 0] == 0
    bpy.ops.seam.raster_select_rect(x0=0, y0=0, x1=5, y1=4, mode="SUBTRACT")
    m3 = sel.selection_array(scene.seam.rasters[0])
    assert m3[1, 2] == 0 and m3[1, 5] == 1 and m3[5, 8] == 1
    bpy.ops.seam.raster_select_rect(x0=0, y0=0, x1=6, y1=10, mode="INTERSECT")
    m4 = sel.selection_array(scene.seam.rasters[0])
    assert m4[..., 6:].max() == 0
    # invert
    assert bpy.ops.seam.raster_select_invert() == {"FINISHED"}
    assert np.allclose(sel.selection_array(scene.seam.rasters[0]), 1.0 - m4)
    # ellipse and lasso
    bpy.ops.seam.raster_select_ellipse(x0=0, y0=0, x1=10, y1=8)
    me = sel.selection_array(scene.seam.rasters[0])
    assert me[4, 5] == 1 and me[0, 0] == 0 and 0 < me.sum() < 80
    path = [{"name": "", "loc": p, "time": 0.0} for p in ((1, 1), (9, 1), (9, 3), (1, 3))]
    assert bpy.ops.seam.raster_select_lasso(path=path) == {"FINISHED"}
    ml = sel.selection_array(scene.seam.rasters[0])
    assert np.allclose(ml[1:3, 1:9], 1.0) and ml.sum() == pytest.approx(16.0)
    assert bpy.ops.seam.raster_select_lasso(path=path[:2]) == {"CANCELLED"}
    # all / none
    bpy.ops.seam.raster_select_all()
    assert sel.selection_array(scene.seam.rasters[0]).min() == 1
    bpy.ops.seam.raster_select_rect(x0=1, y0=1, x1=3, y1=3)
    # persistence of the selection image and stencil wiring
    scene = save_and_reload(blend_path)
    doc = scene.seam.rasters[0]
    ip = scene.tool_settings.image_paint
    ms = sel.selection_array(doc)
    assert doc.has_selection and ms.sum() == 4 and (ms[1:3, 1:3] == 1).all()
    assert ip.use_stencil_layer and ip.stencil_image == doc.selection
    assert bpy.ops.seam.raster_select_none() == {"FINISHED"}
    doc = bpy.context.scene.seam.rasters[0]
    assert not doc.has_selection and sel.selection_array(doc) is None
    assert not bpy.context.scene.tool_settings.image_paint.use_stencil_layer
    assert not bpy.ops.seam.raster_select_none.poll() and not bpy.ops.seam.raster_select_invert.poll()


def test_selection_outline_segments(seam, kernel):
    from seam_studio.paint import selection as sel
    m = np.zeros((6, 8), np.float32)
    m[2:4, 3:6] = 1
    seg = sel.outline_segments(m)
    assert seg.shape[1] == 4 and len(seg) == 2 * 3 + 2 * 2  # perimeter of a 3x2 block
    xs = np.concatenate([seg[:, 0], seg[:, 2]])
    ys = np.concatenate([seg[:, 1], seg[:, 3]])
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (3, 6, 2, 4)
    assert len(sel.outline_segments(np.zeros((4, 4), np.float32))) == 0
    noisy = (np.random.default_rng(0).random((64, 64)) > 0.5).astype(np.float32)
    box = sel.outline_segments(noisy, max_segments=100)
    assert len(box) == 4  # too complex: bounding box fallback


def test_fill_and_clear_honour_selection(seam, kernel):
    from seam_studio.paint import selection as sel
    r = rd()
    doc = new_doc(10, 8, "F")
    scene = bpy.context.scene
    bpy.ops.seam.raster_select_rect(x0=2, y0=2, x1=6, y1=5)
    assert bpy.ops.seam.raster_fill(color=(1.0, 0.0, 0.0, 1.0)) == {"FINISHED"}
    doc = scene.seam.rasters[0]
    px = pixels(doc.layers[0].image)
    assert (px[2:5, 2:6] == (255, 0, 0, 255)).all()
    assert px[..., 3].sum() == 255 * 12  # nothing outside the selection
    # erase part of it
    bpy.ops.seam.raster_select_rect(x0=0, y0=0, x1=4, y1=3)
    assert bpy.ops.seam.raster_clear() == {"FINISHED"}
    px = pixels(scene.seam.rasters[0].layers[0].image)
    assert px[2, 2:4, 3].max() == 0 and (px[3:5, 2:6, 3] == 255).all() and px[2, 4, 3] == 255
    # no selection: clear wipes the whole layer, fill covers it
    bpy.ops.seam.raster_select_none()
    bpy.ops.seam.raster_fill(color=(0.0, 1.0, 0.0, 1.0))
    assert (pixels(scene.seam.rasters[0].layers[0].image) == (0, 255, 0, 255)).all()
    bpy.ops.seam.raster_clear()
    assert not pixels(scene.seam.rasters[0].layers[0].image)[..., 3].any()
    # locked layers refuse
    scene.seam.rasters[0].layers[0].locked = True
    assert not bpy.ops.seam.raster_fill.poll() and not bpy.ops.seam.raster_clear.poll()


def test_mask_workflow(seam, kernel):
    from seam_studio.paint import selection as sel
    r = rd()
    doc = new_doc(8, 6, "M")
    scene = bpy.context.scene
    paint_pixels(doc.layers[0], solid(6, 8, (200, 100, 50, 255)))
    c0 = r.composite_array(doc)
    assert c0[..., 3].min() == 1
    # hide-all mask hides the layer
    bpy.ops.seam.raster_mask_add(fill="BLACK")
    doc = scene.seam.rasters[0]
    assert r.composite_array(doc)[..., 3].max() == 0
    # painting white into the mask through SEAM's write path reveals pixels
    reveal = np.zeros((6, 8), np.float32)
    reveal[1:3, 2:5] = 1.0
    r.write_mask_array(doc.layers[0], reveal)
    c1 = r.composite_array(doc)
    assert (c1[1:3, 2:5, 3] == 1).all() and c1[..., 3].sum() == 6
    doc.layers[0].mask_enabled = False
    assert np.array_equal(r.composite_array(doc), c0)
    doc.layers[0].mask_enabled = True
    # apply bakes into alpha and removes the mask
    assert bpy.ops.seam.raster_mask_apply() == {"FINISHED"}
    doc = scene.seam.rasters[0]
    assert doc.layers[0].mask is None
    a = pixels(doc.layers[0].image)
    assert (a[1:3, 2:5, 3] == 255).all() and a[..., 3].sum() == 255 * 6
    assert np.array_equal(r.composite_array(doc), c1)
    # mask from selection
    bpy.ops.seam.raster_select_rect(x0=0, y0=0, x1=4, y1=6)
    bpy.ops.seam.raster_mask_add(fill="SELECTION")
    doc = scene.seam.rasters[0]
    mk = r.mask_array(doc.layers[0])
    assert (mk[:, :4] == 1).all() and (mk[:, 4:] == 0).all()


def test_merge_down_preserves_the_look(seam, kernel):
    r = rd()
    doc = new_doc(6, 5, "Mg", "WHITE")
    scene = bpy.context.scene
    r.add_layer(doc, "Mid")
    r.add_layer(doc, "Top")
    paint_pixels(doc.layers[0], noise(5, 6, 31))
    mid = noise(5, 6, 32)
    mid[..., 3] = 255  # an opaque lower layer: the composite model is not associative for partial alpha
    paint_pixels(doc.layers[1], mid)
    paint_pixels(doc.layers[2], noise(5, 6, 33))
    doc.layers[2].opacity = 0.6
    doc.layers[2].blend_mode = "MULTIPLY"
    r.add_mask(doc, 2, "WHITE")
    m = np.random.default_rng(3).random((5, 6)).astype(np.float32)
    r.write_mask_array(doc.layers[2], np.rint(m * 255) / 255)
    before = r.composite_array(doc)
    assert bpy.ops.seam.raster_layer_merge_down(index=2) == {"FINISHED"}
    doc = scene.seam.rasters[0]
    assert [l.name for l in doc.layers] == ["Layer 1", "Mid"] and doc.active_layer_index == 1
    after = r.composite_array(doc)
    assert np.allclose(after, before, atol=6e-3)  # re-quantised to 8 bit
    assert doc.layers[1].mask is None
    assert not bpy.ops.seam.raster_layer_merge_down.poll() or doc.active_layer_index >= 1
    doc.active_layer_index = 0
    assert not bpy.ops.seam.raster_layer_merge_down.poll()


def test_png_export_decodes_to_expected_pixels(seam, kernel, tmp_path):
    from seam_studio.seam_core import png
    from seam_studio.seam_core import raster as R
    r = rd()
    doc = new_doc(9, 7, "Exp")
    r.add_layer(doc, "Top")
    a, b = noise(7, 9, 51), noise(7, 9, 52)
    a[..., 3] = 255
    paint_pixels(doc.layers[0], a)
    paint_pixels(doc.layers[1], b)
    doc.layers[1].opacity = 0.5
    path = str(tmp_path / "out.png")
    assert bpy.ops.seam.raster_export_png(filepath=path) == {"FINISHED"}
    got = png.read_png(path)
    expect = R.to_uint8_srgb(r.composite_array(doc))
    assert got.shape == (7, 9, 4) and np.array_equal(got, expect)
    # straight alpha: a half transparent single layer is stored un-premultiplied
    doc.layers[0].visible = False
    doc.layers[1].opacity = 1.0
    paint_pixels(doc.layers[1], solid(7, 9, (200, 40, 120, 128)))
    path2 = str(tmp_path / "half.png")
    bpy.ops.seam.raster_export_png(filepath=path2)
    got = png.read_png(path2)
    assert np.abs(got.astype(int) - np.array([200, 40, 120, 128])).max() <= 1
    # layer export is the raw layer pixels
    path3 = str(tmp_path / "layer.png")
    bpy.ops.seam.raster_export_png(filepath=path3, source="LAYER")
    assert np.array_equal(png.read_png(path3), pixels(doc.layers[1].image))
    assert os.path.getsize(path) > 50


def test_png_import_placement(seam, kernel, tmp_path):
    from seam_studio.seam_core import png
    r = rd()
    doc = new_doc(10, 8, "Imp")
    scene = bpy.context.scene
    small = noise(3, 4, 61)
    small[..., 3] = 255
    p = str(tmp_path / "small.png")
    png.write_png(p, small)
    assert bpy.ops.seam.raster_import_png(filepath=p, placement="TOP_LEFT") == {"FINISHED"}
    doc = scene.seam.rasters[0]
    assert len(doc.layers) == 2 and doc.layers[1].name == "small" and doc.active_layer_index == 1
    px = pixels(doc.layers[1].image)
    assert np.array_equal(px[:3, :4], small) and not px[3:].any() and not px[:, 4:].any()
    bpy.ops.seam.raster_import_png(filepath=p, placement="CENTER")
    px = pixels(scene.seam.rasters[0].layers[2].image)
    assert np.array_equal(px[2:5, 3:7], small)  # (8-3)/2 -> 2 (rounded), (10-4)/2 = 3
    # big image: fit-down scales to the canvas, stretch fills it
    big = np.zeros((16, 40, 4), np.uint8)
    big[...] = (255, 128, 0, 255)
    pb = str(tmp_path / "big.png")
    png.write_png(pb, big)
    bpy.ops.seam.raster_import_png(filepath=pb, placement="FIT_DOWN")
    px = pixels(scene.seam.rasters[0].layers[3].image)
    assert (px[2:6, :, 3] == 255).all() and not px[:2].any() and not px[6:].any()
    assert np.abs(px[3, 5, :3].astype(int) - (255, 128, 0)).max() <= 1
    bpy.ops.seam.raster_import_png(filepath=pb, placement="STRETCH")
    px = pixels(scene.seam.rasters[0].layers[4].image)
    assert (px[..., 3] == 255).all()
    n = len(scene.seam.rasters[0].layers)
    with pytest.raises(RuntimeError, match="not found"):  # ERROR reports surface as RuntimeError in scripts
        bpy.ops.seam.raster_import_png(filepath=str(tmp_path / "missing.png"))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(RuntimeError, match="Cannot import"):
        bpy.ops.seam.raster_import_png(filepath=str(bad))
    assert len(scene.seam.rasters[0].layers) == n


def test_resize_canvas_crop_extend_and_scale(seam, kernel):
    r = rd()
    doc = new_doc(8, 6, "Rz")
    scene = bpy.context.scene
    a = noise(6, 8, 71)
    paint_pixels(doc.layers[0], a)
    r.add_mask(doc, 0, "BLACK")
    bpy.ops.seam.raster_select_all()
    assert bpy.ops.seam.raster_resize_canvas(width=10, height=8, anchor="CENTER") == {"FINISHED"}
    doc = scene.seam.rasters[0]
    assert (doc.width, doc.height) == (10, 8) and not doc.has_selection
    px = pixels(doc.layers[0].image)
    assert px.shape == (8, 10, 4) and np.array_equal(px[1:7, 1:9], a) and not px[0].any()
    mk = pixels(doc.layers[0].mask)
    assert (mk[0] == 255).all() and (mk[1:7, 1:9, 0] == 0).all()  # new area stays visible-white
    xs = [v.co.x for v in doc.canvas.data.vertices]
    ys = [v.co.y for v in doc.canvas.data.vertices]
    assert (max(xs), min(ys)) == (10.0, -8.0)
    assert tuple(doc.composite.size) == (10, 8)
    assert bpy.ops.seam.raster_resize_canvas(width=4, height=3, anchor="TOP_LEFT") == {"FINISHED"}
    doc = scene.seam.rasters[0]
    assert np.array_equal(pixels(doc.layers[0].image), px[:3, :4])
    # scale content
    paint_pixels(doc.layers[0], solid(3, 4, (255, 0, 0, 255)))
    assert bpy.ops.seam.raster_resize_canvas(width=8, height=6, scale_content=True) == {"FINISHED"}
    doc = scene.seam.rasters[0]
    px = pixels(doc.layers[0].image)
    assert px.shape == (6, 8, 4) and (px == (255, 0, 0, 255)).all()
    flush()
    assert tuple(doc.composite.size) == (8, 6) and doc.material[r.MAT_SIG_KEY] == r.material_signature(doc)
    with pytest.raises(ValueError):
        r.resize_canvas(doc, 0, 5)


def test_resample_is_premultiplied_and_exact_on_identity(seam):
    r = rd()
    a = noise(5, 7, 81)
    assert np.array_equal(r.resample_rgba8(a, 5, 7), a) or np.abs(r.resample_rgba8(a, 5, 7).astype(int) - a).max() <= 1
    t = np.zeros((2, 2, 4), np.uint8)
    t[0, 0] = (255, 0, 0, 255)  # one opaque red pixel in a transparent field
    out = r.resample_rgba8(t, 1, 1)
    assert out[0, 0, 3] == 64 and out[0, 0, 0] == 255 and out[0, 0, 1] == 0  # no darkening of the colour
