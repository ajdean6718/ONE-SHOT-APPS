"""Paint: operators, poll messages, paint mode entry and paint-target selection (kernel-independent)."""

import numpy as np
import pytest

import bpy

from test_paint_support import flush, kernel, new_doc, noise, paint_pixels, pixels, rd, solid  # noqa: F401

ALL_OPS = ["raster_remove", "raster_layer_add", "raster_layer_remove", "raster_layer_duplicate", "raster_layer_move",
           "raster_layer_merge_down", "raster_mask_add", "raster_mask_apply", "raster_mask_remove",
           "raster_set_paint_target", "raster_paint_mode", "raster_fill", "raster_clear", "raster_resize_canvas",
           "raster_select_rect", "raster_select_ellipse", "raster_select_lasso", "raster_select_all",
           "raster_select_none", "raster_select_invert", "raster_import_png", "raster_export_png"]


def test_operators_registered_and_disabled_without_document(seam):
    assert hasattr(bpy.ops.seam, "raster_new")
    assert bpy.ops.seam.raster_new.poll()  # creating a document is always possible
    for name in ALL_OPS:
        op = getattr(bpy.ops.seam, name)
        assert not op.poll(), name  # each disabled operator has a poll message (see below)


def test_poll_messages_are_set():
    """Static check: every poll that can fail explains itself."""
    import inspect
    import seam_studio.paint.ops as ops
    import seam_studio.paint.selection as sel
    import seam_studio.paint.io as pio
    for mod in (ops, sel, pio):
        for cls in mod.classes:
            src = inspect.getsource(cls)
            if "def poll" in src:
                assert "poll_message_set" in src or "_need_" in src or "_doc_poll" in src, cls.__name__


def test_new_document_operator_and_layer_ops(seam):
    assert bpy.ops.seam.raster_new(name="Op Doc", width=12, height=9, background="WHITE") == {"FINISHED"}
    scene = bpy.context.scene
    doc = scene.seam.rasters[0]
    assert (doc.name, doc.width, doc.height, doc.background) == ("Op Doc", 12, 9, "WHITE")
    ob = bpy.context.view_layer.objects.active
    assert ob == doc.canvas and ob.select_get()
    assert bpy.ops.seam.raster_layer_add(name="Paint") == {"FINISHED"}
    assert bpy.ops.seam.raster_layer_add() == {"FINISHED"}
    doc = scene.seam.rasters[0]
    assert [l.name for l in doc.layers] == ["Layer 1", "Paint", "Layer 2"] and doc.active_layer_index == 2
    bpy.ops.seam.raster_layer_move(direction="DOWN")
    doc = scene.seam.rasters[0]
    assert [l.name for l in doc.layers] == ["Layer 1", "Layer 2", "Paint"] and doc.active_layer_index == 1
    assert bpy.ops.seam.raster_layer_move(direction="DOWN") == {"FINISHED"}
    assert bpy.ops.seam.raster_layer_move(direction="DOWN") == {"CANCELLED"}  # already at the bottom
    assert bpy.ops.seam.raster_layer_duplicate() == {"FINISHED"}
    doc = scene.seam.rasters[0]
    assert len(doc.layers) == 4 and doc.layers[1].name.endswith("copy")
    assert bpy.ops.seam.raster_layer_remove() == {"FINISHED"}
    assert len(scene.seam.rasters[0].layers) == 3
    # the last layer cannot be deleted
    bpy.ops.seam.raster_layer_remove(index=0)
    bpy.ops.seam.raster_layer_remove(index=0)
    assert len(scene.seam.rasters[0].layers) == 1
    assert not bpy.ops.seam.raster_layer_remove.poll()
    # mask operators
    assert bpy.ops.seam.raster_mask_add(fill="BLACK") == {"FINISHED"}
    layer = scene.seam.rasters[0].layers[0]
    assert layer.mask is not None and not pixels(layer.mask)[..., 0].any()
    assert not bpy.ops.seam.raster_mask_add.poll()  # already has one
    assert bpy.ops.seam.raster_mask_remove() == {"FINISHED"}
    assert scene.seam.rasters[0].layers[0].mask is None
    canvas_name, mat_name = scene.seam.rasters[0].canvas.name, scene.seam.rasters[0].material.name
    assert bpy.ops.seam.raster_remove() == {"FINISHED"}
    assert len(scene.seam.rasters) == 0 and canvas_name not in bpy.data.objects and mat_name not in bpy.data.materials
    assert not [i for i in bpy.data.images if i.name.startswith("SEAM Op Doc")]
    assert not bpy.ops.seam.raster_layer_add.poll()


def test_paint_target_and_slots(seam):
    doc = new_doc(8, 6)
    r = rd()
    r.add_layer(doc, "B")
    r.add_layer(doc, "C")
    r.add_mask(doc, 1, "WHITE")
    scene = bpy.context.scene
    nt = doc.material.node_tree
    # outside paint mode the composite is the active (displayed) image node
    assert nt.nodes.active.image == doc.composite
    r.set_paint_target(doc, 0, "LAYER")
    assert doc.active_layer_index == 0 and doc.edit_target == "LAYER"
    assert nt.nodes.active.image == doc.composite  # not painting yet
    assert bpy.ops.seam.raster_paint_mode() == {"FINISHED"}
    ob = doc.canvas
    assert ob.mode == "TEXTURE_PAINT" and bpy.context.view_layer.objects.active == ob
    ip = scene.tool_settings.image_paint
    assert ip.mode == "MATERIAL"
    ma = doc.material
    assert nt.nodes.active.image == doc.layers[0].image
    assert ma.texture_paint_images[ma.paint_active_slot] == doc.layers[0].image
    # switch target to another layer and to a mask while painting
    r.set_paint_target(doc, 2, "LAYER")
    assert nt.nodes.active.image == doc.layers[2].image
    assert ma.texture_paint_images[ma.paint_active_slot] == doc.layers[2].image
    assert bpy.ops.seam.raster_set_paint_target(index=1, target="MASK") == {"FINISHED"}
    assert nt.nodes.active.image == doc.layers[1].mask and doc.edit_target == "MASK"
    assert ma.texture_paint_images[ma.paint_active_slot] == doc.layers[1].mask
    assert bpy.ops.seam.raster_set_paint_target(index=0, target="MASK") == {"CANCELLED"}  # no mask on layer 0
    with pytest.raises(ValueError):
        r.set_paint_target(doc, 0, "MASK")
    # a locked layer sends native strokes into the sink image
    doc.layers[2].locked = True
    r.set_paint_target(doc, 2, "LAYER")
    assert nt.nodes.active.image == doc.sink and tuple(doc.sink.size) == (1, 1)
    assert ma.texture_paint_images[ma.paint_active_slot] == doc.sink
    doc.layers[2].locked = False
    r.set_paint_target(doc, 2, "LAYER")
    assert nt.nodes.active.image == doc.layers[2].image
    # a material rebuild while painting keeps the target
    doc.layers[2].opacity = 0.5
    flush()
    assert nt.nodes.active.image == doc.layers[2].image or doc.material.node_tree.nodes.active.image == doc.layers[2].image
    # leaving paint mode shows the composite again
    assert bpy.ops.seam.raster_paint_mode(enter=False) == {"FINISHED"}
    assert doc.canvas.mode == "OBJECT"
    assert doc.material.node_tree.nodes.active.image == doc.composite


def test_painting_state_survives_save(seam, blend_path):
    from helpers import save_and_reload
    doc = new_doc(8, 6)
    rd().add_layer(doc, "B")
    rd().set_paint_target(doc, 0, "LAYER")
    bpy.ops.seam.raster_paint_mode()
    bpy.ops.object.mode_set(mode="OBJECT")
    scene = save_and_reload(blend_path)
    doc = scene.seam.rasters[0]
    assert doc.active_layer_index == 0 and len(doc.layers) == 2
    assert doc.material.node_tree.nodes.active.image == doc.composite


def test_stroke_detection_marks_composite_dirty(seam, engine):
    """Native strokes are detected by the pixel-fingerprint poll (Python pixel writes emit no depsgraph update)."""
    doc = new_doc(8, 6)
    r = rd()
    layer = doc.layers[0]
    flush()
    assert r.check_external_edits(doc) is False  # records the baseline
    assert r.check_external_edits(doc) is False
    key = f"rastercache:{doc.id}"
    sig0 = r._cache_signature(doc)
    # simulate a native stroke: write the image behind SEAM's back
    img = layer.image
    buf = np.zeros(8 * 6 * 4, np.float32)
    buf[0:4] = (1, 0, 0, 1)
    img.pixels.foreach_set(buf)
    assert r.check_external_edits(doc) is True
    assert key in engine.pending_keys()
    assert r._cache_signature(doc) != sig0
    assert r.check_external_edits(doc) is False  # no further change
    # edits to non-target images are ignored; the sink is never part of the composite
    doc.layers[0].locked = True
    assert r.target_image(doc) == doc.sink
    assert r.check_external_edits(doc) is False


def test_fingerprint_only_target_image(seam):
    doc = new_doc(4, 4)
    r = rd()
    r.add_layer(doc, "B")
    r.set_paint_target(doc, 0)
    r.check_external_edits(doc)
    img_b = doc.layers[1].image
    img_b.pixels.foreach_set(np.ones(4 * 4 * 4, np.float32))
    assert r.check_external_edits(doc) is False  # layer 1 is not the target
    assert r.check_external_edits(doc, [img_b]) is False  # not the active layer either: foreign image


def test_draw_functions_run_with_a_recording_layout(seam):
    """Panels and the layer list draw without errors for no document, one document and a masked locked layer."""
    from seam_studio.paint import ui

    class Rec:
        def __init__(self):
            self.calls = []
            self.children = []
            self.scale_y = 1.0
            self.enabled = True

        def all_calls(self):
            out = list(self.calls)
            for ch in self.children:
                out += ch.all_calls()
            return out

        def __getattr__(self, name):
            def call(*a, **k):
                self.calls.append((name, a, k))
                if name in {"row", "column", "box", "split", "grid_flow"}:
                    ch = Rec()
                    self.children.append(ch)
                    return ch
                return _Op()
            return call

    class _Op:
        def __setattr__(self, k, v):
            object.__setattr__(self, k, v)

    class Ctx:
        def __init__(self, scene):
            self.scene = scene

    scene = bpy.context.scene

    import types

    def draw_all():
        for cls in (ui.SEAM_PT_paint_document, ui.SEAM_PT_paint_layers, ui.SEAM_PT_paint_selection,
                    ui.SEAM_PT_paint_actions):
            lay = Rec()
            cls.draw(types.SimpleNamespace(layout=lay), Ctx(scene))
            assert lay.all_calls(), cls.__name__

    draw_all()
    doc = new_doc(8, 6)
    rd().add_layer(doc, "B")
    rd().add_mask(doc, 1, "BLACK")
    draw_all()
    doc.layers[1].locked = True
    doc.background = "COLOR"
    draw_all()
    lst = types.SimpleNamespace(bitflag_filter_item=1)
    flags, order = ui.SEAM_UL_raster_layers.filter_items(lst, None, doc, "layers")
    assert order == [1, 0] and flags == [1, 1]
    row = Rec()
    ui.SEAM_UL_raster_layers.draw_item(lst, None, row, doc, doc.layers[1], 0, doc, "active_layer_index", 1)
    assert [c[0] for c in row.all_calls()].count("prop") >= 3
