"""Scenario 2 - raster painting.

Paint on layers, erase, opacity / blend / masks, undo + redo, reopen with editable layers.  Pixel edits go
through the SEAM operators (select + fill / clear, mask operators, layer operators) and are verified from
the PNG files the Export operator writes.  Headless Blender cannot run a native brush stroke (the 3D view
matrices are only computed when a region is drawn), so the native-stroke path is covered by writing the
layer image exactly the way a stroke does and letting SEAM detect it (see ``test_native_stroke_is_detected``);
the real brush drag lives in the GUI suite.
"""

import bpy
import numpy as np
import pytest

from accsupport import STEPS, flush, op, push, redo, save_and_reopen, srgb8, start_session, undo

W, H = 64, 48
RED, GREEN, BLUE, BLACK = (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)
CLEAR = (0, 0, 0, 0)


def doc():
    return bpy.context.scene.seam.rasters[0]


def read_png(path):
    from seam_studio.seam_core import png
    return png.read_png(str(path))


def composite(tmp_path, name="comp.png"):
    op(bpy.ops.seam.raster_export_png, filepath=str(tmp_path / name), source="COMPOSITE")
    return read_png(tmp_path / name)


def layer_png(tmp_path, name="layer.png"):
    op(bpy.ops.seam.raster_export_png, filepath=str(tmp_path / name), source="LAYER")
    return read_png(tmp_path / name)


def at(img, x, y):
    return tuple(int(v) for v in img[y, x])


def new_doc(background="WHITE"):
    start_session()
    op(bpy.ops.seam.raster_new, name="Art", width=W, height=H, background=background)
    flush()
    return doc()


def fill_rect(x0, y0, x1, y1, color):
    op(bpy.ops.seam.raster_select_rect, x0=x0, y0=y0, x1=x1, y1=y1)
    op(bpy.ops.seam.raster_fill, color=color)
    flush()


def set_layer(index):
    d = doc()
    d.active_layer_index = index
    push("active layer")


def linear_to_srgb8(rgb):
    return srgb8(rgb)


def test_layers_fill_erase_and_composite_values(seam, engine, tmp_path):
    d = new_doc()
    assert (d.width, d.height) == (W, H) and len(d.layers) == 1 and d.canvas is not None
    assert d.canvas.mode == "OBJECT" and tuple(d.canvas.dimensions)[:2] == pytest.approx((W, H))
    # base layer: whole-layer fill with no selection
    op(bpy.ops.seam.raster_fill, color=BLUE)
    flush()
    base = layer_png(tmp_path, "base.png")
    assert base.shape == (H, W, 4) and (base == (0, 0, 255, 255)).all()

    # second layer: red rectangle through a rectangular selection
    op(bpy.ops.seam.raster_layer_add, name="Red")
    assert doc().active_layer_index == 1 and doc().layers[1].name == "Red"
    fill_rect(8, 8, 40, 32, RED)
    red = layer_png(tmp_path, "red.png")
    inside = np.zeros((H, W), bool)
    inside[8:32, 8:40] = True
    assert (red[inside] == (255, 0, 0, 255)).all()
    assert (red[~inside][:, 3] == 0).all()                                  # untouched pixels stay transparent
    assert inside.sum() == 32 * 24
    comp = composite(tmp_path)
    assert (comp[inside] == (255, 0, 0, 255)).all()
    assert (comp[~inside] == (0, 0, 255, 255)).all()                        # base shows around the red layer

    # erase a hole inside the red layer: the base shows through there, nowhere else changes
    op(bpy.ops.seam.raster_select_rect, x0=16, y0=12, x1=24, y1=20)
    op(bpy.ops.seam.raster_clear)
    flush()
    red2 = layer_png(tmp_path, "red2.png")
    hole = np.zeros((H, W), bool)
    hole[12:20, 16:24] = True
    assert (red2[hole][:, 3] == 0).all() and (red2[inside & ~hole] == (255, 0, 0, 255)).all()
    comp2 = composite(tmp_path, "comp2.png")
    assert (comp2[hole] == (0, 0, 255, 255)).all() and (comp2[inside & ~hole] == (255, 0, 0, 255)).all()
    assert (comp2[inside & ~hole].shape[0]) == 32 * 24 - 64

    # third layer: green ellipse at 50 % layer opacity, drawn with an elliptical selection
    op(bpy.ops.seam.raster_layer_add, name="Green")
    op(bpy.ops.seam.raster_select_ellipse, x0=30, y0=6, x1=62, y1=38)
    op(bpy.ops.seam.raster_fill, color=GREEN)
    flush()
    gl = layer_png(tmp_path, "green.png")
    area = gl[..., 3].astype(np.float64).sum() / 255.0
    assert area == pytest.approx(np.pi * 16 * 16, rel=0.02)                   # anti-aliased disc, r = 16
    assert at(gl, 46, 22) == (0, 255, 0, 255) and at(gl, 30, 6)[3] == 0 and at(gl, 46, 4)[3] == 0
    d = doc()
    d.layers[2].opacity = 0.5
    push("opacity")
    flush()
    comp3 = composite(tmp_path, "comp3.png")
    half = srgb8((0.0, 0.5, 0.5))                                            # linear mix of green over blue
    assert at(comp3, 46, 22)[:3] == pytest.approx(half, abs=1) and at(comp3, 46, 22)[3] == 255
    assert at(comp3, 20, 28) == (255, 0, 0, 255) and at(comp3, 3, 3) == (0, 0, 255, 255)
    assert (comp3[hole] == (0, 0, 255, 255)).all()                            # earlier work is intact


def test_blend_mode_visibility_and_background(seam, engine, tmp_path):
    d = new_doc("WHITE")
    # empty document: just the white background
    assert (composite(tmp_path, "empty.png") == 255).all()
    op(bpy.ops.seam.raster_fill, color=BLUE)
    op(bpy.ops.seam.raster_layer_add, name="Top")
    op(bpy.ops.seam.raster_fill, color=(1.0, 0.5, 0.0, 1.0))
    flush()
    d = doc()
    assert at(composite(tmp_path, "n.png"), 5, 5) == (*srgb8((1.0, 0.5, 0.0)), 255)
    d.layers[1].blend_mode = "MULTIPLY"                                       # (1, .5, 0) * (0, 0, 1) = black
    push("blend")
    flush()
    assert at(composite(tmp_path, "m.png"), 5, 5) == (0, 0, 0, 255)
    d.layers[1].blend_mode = "NORMAL"
    d.layers[1].visible = False
    push("hide")
    flush()
    assert at(composite(tmp_path, "h.png"), 5, 5) == (0, 0, 255, 255)        # hidden layer contributes nothing
    d.layers[1].visible = True
    push("show")
    flush()
    assert at(composite(tmp_path, "s.png"), 5, 5) == (*srgb8((1.0, 0.5, 0.0)), 255)
    # transparent document background keeps alpha 0 where nothing is painted
    bpy.context.scene.seam.rasters[0].background = "TRANSPARENT"
    push("background")
    flush()
    op(bpy.ops.seam.raster_layer_remove, index=1)
    flush()
    assert len(doc().layers) == 1
    assert not bpy.ops.seam.raster_layer_remove.poll()                        # the last layer cannot be deleted
    op(bpy.ops.seam.raster_clear)
    flush()
    assert (composite(tmp_path, "t.png")[..., 3] == 0).all()


def test_mask_hides_without_destroying_pixels(seam, engine, tmp_path):
    d = new_doc("WHITE")
    op(bpy.ops.seam.raster_fill, color=RED)
    op(bpy.ops.seam.raster_mask_add, fill="WHITE")                            # reveal-all mask
    flush()
    assert doc().layers[0].mask is not None
    assert at(composite(tmp_path, "a.png"), 4, 4) == (255, 0, 0, 255)
    # paint the mask black in the left half: composite shows the white background there
    op(bpy.ops.seam.raster_set_paint_target, target="MASK")
    op(bpy.ops.seam.raster_select_rect, x0=0, y0=0, x1=32, y1=H)
    op(bpy.ops.seam.raster_fill, color=BLACK)
    flush()
    comp = composite(tmp_path, "b.png")
    assert (comp[:, :32] == 255).all() and (comp[:, 32:] == (255, 0, 0, 255)).all()
    # the layer pixels are untouched (mask is non-destructive)
    op(bpy.ops.seam.raster_set_paint_target, target="LAYER")
    assert (layer_png(tmp_path, "l.png") == (255, 0, 0, 255)).all()
    # disabling the mask shows everything again; enabling restores the hidden half
    doc().layers[0].mask_enabled = False
    push("mask off")
    flush()
    assert (composite(tmp_path, "c.png")[:, :32] == (255, 0, 0, 255)).all()
    doc().layers[0].mask_enabled = True
    push("mask on")
    flush()
    assert (composite(tmp_path, "d.png")[:, :32] == 255).all()
    # applying the mask bakes it into the alpha channel and removes it
    op(bpy.ops.seam.raster_mask_apply)
    flush()
    assert doc().layers[0].mask is None
    baked = layer_png(tmp_path, "baked.png")
    assert (baked[:, :32, 3] == 0).all() and (baked[:, 32:] == (255, 0, 0, 255)).all()
    assert (composite(tmp_path, "e.png") == comp).all()


def test_merge_down_and_duplicate(seam, engine, tmp_path):
    d = new_doc("WHITE")
    fill_rect(0, 0, 32, H, BLUE)
    op(bpy.ops.seam.raster_layer_add, name="Over")
    fill_rect(16, 0, 48, H, (1.0, 0.0, 0.0, 1.0))
    before = composite(tmp_path, "before.png")
    op(bpy.ops.seam.raster_layer_duplicate)
    flush()
    assert len(doc().layers) == 3 and (composite(tmp_path, "dup.png") == before).all()
    op(bpy.ops.seam.raster_layer_merge_down)
    flush()
    assert len(doc().layers) == 2
    assert (composite(tmp_path, "merged.png") == before).all()                # merging never changes the picture
    op(bpy.ops.seam.raster_layer_move, direction="DOWN")
    flush()
    swapped = composite(tmp_path, "swapped.png")
    assert at(swapped, 20, 10) == (0, 0, 255, 255) and at(before, 20, 10) == (255, 0, 0, 255)


def test_undo_redo_walks_every_edit(seam, engine, tmp_path):
    d = new_doc("WHITE")
    snaps = [(STEPS[0], composite(tmp_path, "s0.png"), 1)]                   # (undo steps so far, picture, layers)

    def snap(name):
        snaps.append((STEPS[0], composite(tmp_path, name + ".png"), len(doc().layers)))

    op(bpy.ops.seam.raster_fill, color=BLUE)
    flush()
    snap("fill")
    op(bpy.ops.seam.raster_layer_add, name="Red")
    snap("layer")
    fill_rect(10, 10, 30, 30, RED)
    snap("rect")
    op(bpy.ops.seam.raster_select_rect, x0=15, y0=15, x1=25, y1=25)
    op(bpy.ops.seam.raster_clear)
    flush()
    snap("clear")
    op(bpy.ops.seam.raster_mask_add, fill="BLACK")                            # everything hidden by a black mask
    flush()
    snap("mask")
    assert (snaps[-1][1] == snaps[1][1]).all()                                # == the plain blue picture
    assert not (snaps[3][1] == snaps[4][1]).all()
    # walk back to the start one *picture* at a time, then forward again
    for want_steps, pic, nlayers in reversed(snaps[:-1]):
        undo(STEPS[0] - want_steps)
        STEPS[0] = want_steps
        assert len(doc().layers) == nlayers
        assert (composite(tmp_path, "undo.png") == pic).all(), want_steps
    assert len(doc().layers) == 1
    for want_steps, pic, nlayers in snaps[1:]:
        redo(want_steps - STEPS[0])
        STEPS[0] = want_steps
        assert len(doc().layers) == nlayers
        assert (composite(tmp_path, "redo.png") == pic).all(), want_steps
    # a new edit after undo discards the redo branch and still works
    undo(1)
    STEPS[0] -= 1
    op(bpy.ops.seam.raster_select_all)
    op(bpy.ops.seam.raster_fill, color=GREEN)
    flush()
    assert at(composite(tmp_path, "branch.png"), 5, 5) == (0, 255, 0, 255)
    assert bpy.ops.ed.redo.poll() is False


def test_paint_mode_edits_stay_undoable(seam, engine, tmp_path):
    d = new_doc("WHITE")
    op(bpy.ops.seam.raster_fill, color=BLUE)
    flush()
    ref = composite(tmp_path, "ref.png")
    assert bpy.ops.seam.raster_paint_mode(frame_view=False) == {"FINISHED"}
    ob = doc().canvas
    assert ob.mode == "TEXTURE_PAINT" and bpy.context.view_layer.objects.active == ob
    fill_rect(0, 0, 20, 20, RED)                                              # SEAM edits keep working in paint mode
    painted = composite(tmp_path, "painted.png")
    assert at(painted, 5, 5) == (255, 0, 0, 255) and at(painted, 30, 30) == (0, 0, 255, 255)
    undo(2)                                                                    # fill + selection
    STEPS[0] -= 2
    assert (composite(tmp_path, "back.png") == ref).all()
    assert bpy.ops.seam.raster_paint_mode(enter=False) == {"FINISHED"}
    assert doc().canvas.mode == "OBJECT"


def test_native_stroke_is_detected(seam, engine, tmp_path):
    """A native brush stroke changes the layer image directly; SEAM must notice and recomposite."""
    from seam_studio.paint import rasterdoc as rd
    d = new_doc("WHITE")
    op(bpy.ops.seam.raster_fill, color=BLUE)
    flush()
    assert bpy.ops.seam.raster_paint_mode(frame_view=False) == {"FINISHED"}
    img = doc().layers[0].image
    rd.check_external_edits(doc())                                             # baseline (the paint timer does this)
    buf = np.empty(W * H * 4, np.float32)
    img.pixels.foreach_get(buf)
    px = buf.reshape(H, W, 4)                                                  # bottom row first, stored non-linear
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx + 0.5 - 32) ** 2 + (H - 1 - yy + 0.5 - 24) ** 2 <= 8 ** 2       # row 0 = bottom in Blender images
    px[disc] = (1.0, 0.0, 0.0, 1.0)
    img.pixels.foreach_set(buf)
    img.update()
    assert rd.check_external_edits(doc()) is True
    flush()
    comp = composite(tmp_path, "stroke.png")
    assert at(comp, 32, 24) == (255, 0, 0, 255) and at(comp, 3, 3) == (0, 0, 255, 255)
    red = (comp[..., 0] == 255) & (comp[..., 1] == 0)
    assert red.sum() == pytest.approx(np.pi * 64, rel=0.08)
    bpy.ops.seam.raster_paint_mode(enter=False)


def test_save_and_reopen_keep_layers_editable(seam, engine, tmp_path, blend_path):
    d = new_doc("WHITE")
    op(bpy.ops.seam.raster_fill, color=BLUE)
    op(bpy.ops.seam.raster_layer_add, name="Red")
    fill_rect(8, 8, 40, 32, RED)
    op(bpy.ops.seam.raster_mask_add, fill="WHITE")
    op(bpy.ops.seam.raster_set_paint_target, target="MASK")
    fill_rect(8, 8, 24, 32, BLACK)
    op(bpy.ops.seam.raster_set_paint_target, target="LAYER")
    doc().layers[1].opacity = 0.75
    doc().layers[1].blend_mode = "NORMAL"
    push("opacity")
    flush()
    ref = composite(tmp_path, "ref.png")
    ref_layer = layer_png(tmp_path, "ref_layer.png")
    ids = [lay.id for lay in doc().layers]

    save_and_reopen(blend_path)
    flush()
    d = doc()
    assert (d.width, d.height, len(d.layers)) == (W, H, 2)
    assert [lay.name for lay in d.layers] == ["Layer 1", "Red"] and [lay.id for lay in d.layers] == ids
    assert d.layers[1].opacity == pytest.approx(0.75) and d.layers[1].mask is not None
    assert d.layers[0].image.packed_file is not None or d.layers[0].image.filepath == ""   # pixels live in the file
    assert (composite(tmp_path, "re.png") == ref).all()
    d.active_layer_index = 1
    assert (layer_png(tmp_path, "re_layer.png") == ref_layer).all()
    # still editable: paint more, undo it, the earlier picture comes back
    # (regression: the first pixel edit after opening a file had no journal baseline and could not be undone)
    start_session()
    op(bpy.ops.seam.raster_select_rect, x0=40, y0=0, x1=64, y1=16)
    op(bpy.ops.seam.raster_fill, color=GREEN)
    flush()
    after = composite(tmp_path, "after.png")
    assert at(after, 50, 5) == (*srgb8((0.0, 0.75, 0.25)), 255)            # green at the layer's 75 % over blue
    assert at(ref, 50, 5) == (0, 0, 255, 255)
    undo(2)
    und = composite(tmp_path, "undone.png")
    assert (und == ref).all()
    # and a second round trip keeps working
    save_and_reopen(blend_path)
    flush()
    assert (composite(tmp_path, "re2.png") == ref).all()


def test_first_mask_edit_after_reopen_is_undoable(seam, engine, tmp_path, blend_path):
    """Regression (same cause as the layer case): mask pixels edited right after opening a file could not be undone."""
    new_doc("WHITE")
    op(bpy.ops.seam.raster_fill, color=RED)
    op(bpy.ops.seam.raster_mask_add, fill="WHITE")
    op(bpy.ops.seam.raster_set_paint_target, target="MASK")
    fill_rect(0, 0, 16, H, BLACK)
    flush()
    ref = composite(tmp_path, "ref.png")
    assert (ref[:, :16] == 255).all() and at(ref, 40, 10) == (255, 0, 0, 255)
    save_and_reopen(blend_path)
    flush()
    start_session()
    assert doc().edit_target == "MASK"
    op(bpy.ops.seam.raster_select_rect, x0=48, y0=0, x1=64, y1=H)
    op(bpy.ops.seam.raster_fill, color=BLACK)
    flush()
    assert (composite(tmp_path, "edited.png")[:, 48:] == 255).all()
    undo(2)
    assert np.array_equal(composite(tmp_path, "undone.png"), ref)
    redo(2)
    assert (composite(tmp_path, "redone.png")[:, 48:] == 255).all()
