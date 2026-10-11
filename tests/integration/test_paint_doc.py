"""Paint: raster document structure, layers, material node tree, persistence and undo."""

import numpy as np
import pytest

import bpy

from helpers import save_and_reload
from test_paint_support import flush, new_doc, noise, paint_pixels, pixels, rd


def image_nodes(doc):
    return [n for n in doc.material.node_tree.nodes if n.bl_idname == "ShaderNodeTexImage"]


def test_new_doc_structure(seam):
    from seam_studio.vector import canvas
    scene = bpy.context.scene
    ab = canvas.active_artboard(canvas.ensure_canvas(scene) and scene)
    doc = new_doc(16, 8, "Paint A", "TRANSPARENT")
    assert (doc.width, doc.height, doc.name) == (16, 8, "Paint A")
    assert len(doc.layers) == 1 and doc.layers[0].name == "Layer 1"
    obj = doc.canvas
    assert obj.seam.role == "RASTER" and obj.seam.raster_id == doc.id
    assert rd().doc_for_object(obj).id == doc.id and rd().get_doc(scene, doc.id).id == doc.id
    assert obj.parent.seam.role == "CANVAS_ROOT"
    assert (obj.location.x, obj.location.y) == (ab.location.x, ab.location.y)
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    assert (min(xs), max(xs), min(ys), max(ys)) == (0.0, 16.0, -8.0, 0.0)
    uv_names = [u.name for u in obj.data.uv_layers]
    assert uv_names == ["UVMap", "SEAM Stencil"] and obj.data.uv_layer_stencil_index == 1
    img = doc.layers[0].image
    assert img.name == "SEAM Paint A Layer 1" and img.use_fake_user and not img.is_float
    assert tuple(img.size) == (16, 8) and img.alpha_mode == "STRAIGHT" and img.colorspace_settings.name == "sRGB"
    assert not pixels(img).any()  # transparent
    assert obj.data.materials[0] == doc.material
    assert scene.seam.active_raster_index == 0 and rd().active_doc(scene).id == doc.id


def test_second_doc_and_names(seam):
    a = new_doc(4, 4, "Same")
    b = new_doc(4, 4, "Same")
    scene = bpy.context.scene
    assert [d.name for d in scene.seam.rasters] == ["Same", "Same 2"]
    assert scene.seam.active_raster_index == 1
    assert rd().active_doc(scene).name == "Same 2"
    assert a.id != b.id


def test_layers_crud(seam):
    doc = new_doc(8, 6)
    r = rd()
    l2 = r.add_layer(doc, "Two")
    l3 = r.add_layer(doc, "Three")
    assert [l.name for l in doc.layers] == ["Layer 1", "Two", "Three"] and doc.active_layer_index == 2
    r.add_layer(doc, "Mid", index=1)
    assert [l.name for l in doc.layers] == ["Layer 1", "Mid", "Two", "Three"] and doc.active_layer_index == 1
    assert doc.layers[1].image.name == "SEAM Doc Mid"
    # move: active layer follows
    doc.active_layer_index = 1
    r.move_layer(doc, 1, 3)
    assert [l.name for l in doc.layers] == ["Layer 1", "Two", "Three", "Mid"] and doc.active_layer_index == 3
    r.move_layer(doc, 0, 2)  # an unrelated layer passes over the active one
    assert [l.name for l in doc.layers] == ["Two", "Three", "Layer 1", "Mid"] and doc.active_layer_index == 3
    r.move_layer(doc, 3, 0)
    assert [l.name for l in doc.layers] == ["Mid", "Two", "Three", "Layer 1"] and doc.active_layer_index == 0
    # duplicate keeps pixels and properties
    px = noise(6, 8, 3)
    paint_pixels(doc.layers[1], px)
    doc.layers[1].opacity = 0.4
    doc.layers[1].blend_mode = "MULTIPLY"
    dup = r.duplicate_layer(doc, 1)
    assert dup.name == "Two copy" and doc.active_layer_index == 2
    assert np.array_equal(pixels(dup.image), px) and dup.image != doc.layers[1].image
    assert abs(dup.opacity - 0.4) < 1e-6 and dup.blend_mode == "MULTIPLY"
    # remove frees images and fixes the active index
    img = dup.image
    name = img.name
    r.remove_layer(doc, 2)
    assert name not in bpy.data.images and len(doc.layers) == 4
    assert doc.active_layer_index == 1
    with pytest.raises(IndexError):
        r.remove_layer(doc, 9)
    # every layer image has a distinct 8-bit datablock
    assert len({l.image.name for l in doc.layers}) == len(doc.layers)
    assert all(not l.image.is_float for l in doc.layers)


def test_material_one_image_node_per_layer_and_mask(seam):
    doc = new_doc(8, 6)
    r = rd()
    r.add_layer(doc, "B")
    r.add_layer(doc, "C")
    r.add_mask(doc, 1, "WHITE")
    flush()
    nodes = image_nodes(doc)
    imgs = [n.image for n in nodes]
    for ly in doc.layers:
        assert imgs.count(ly.image) == 1
    assert imgs.count(doc.layers[1].mask) == 1
    assert imgs.count(doc.composite) == 1 and imgs.count(doc.sink) == 1
    assert len(nodes) == len(doc.layers) + 1 + 2
    for n in nodes:
        assert n.interpolation == "Closest" and n.extension == "CLIP"
    mask_node = [n for n in nodes if n.image == doc.layers[1].mask][0]
    assert mask_node.image.colorspace_settings.name == "Non-Color"
    nt = doc.material.node_tree
    outs = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    assert len(outs) == 1 and outs[0].inputs["Surface"].is_linked


def test_material_follows_layer_props(seam, engine):
    doc = new_doc(8, 6)
    r = rd()
    r.add_layer(doc, "B")
    r.add_layer(doc, "C")
    flush()

    def mixes():
        return [n for n in doc.material.node_tree.nodes if n.bl_idname == "ShaderNodeMix" and n.blend_type != "MIX"]

    assert mixes() == []  # all NORMAL ("MIX") blends
    doc.layers[1].blend_mode = "MULTIPLY"
    res = flush()
    assert any(x.key.startswith("rastermat:") and x.status == "ok" for x in res)
    assert [m.blend_type for m in mixes()] == ["MULTIPLY"]
    doc.layers[2].blend_mode = "COLOR_DODGE"
    flush()
    assert sorted(m.blend_type for m in mixes()) == ["DODGE", "MULTIPLY"]
    # visibility removes the mix chain of a hidden layer
    doc.layers[1].visible = False
    flush()
    assert [m.blend_type for m in mixes()] == ["DODGE"]
    # opacity creates a multiply-by-opacity node
    n_math_before = len([n for n in doc.material.node_tree.nodes if n.bl_idname == "ShaderNodeMath"])
    doc.layers[2].opacity = 0.25
    flush()
    ops = [n for n in doc.material.node_tree.nodes if n.bl_idname == "ShaderNodeMath" and n.label == "Opacity"]
    assert len(ops) == 1 and abs(ops[0].inputs[1].default_value - 0.25) < 1e-6
    assert len([n for n in doc.material.node_tree.nodes if n.bl_idname == "ShaderNodeMath"]) > n_math_before
    assert doc.material[r.MAT_SIG_KEY] == r.material_signature(doc)
    # idempotent
    assert all(x.status != "ok" for x in flush() if x.key.startswith("rastermat:"))


def test_background_variants_build(seam):
    for bg in ("TRANSPARENT", "WHITE", "COLOR"):
        doc = new_doc(4, 4, "BG " + bg, bg)
        rd().add_layer(doc, "x")
        doc.background_color = (0.2, 0.4, 0.6, 0.5)
        rd().refresh(doc)
        out = [n for n in doc.material.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"][0]
        assert out.inputs["Surface"].is_linked


def test_save_reload_exact_pixels_and_props(seam, blend_path):
    doc = new_doc(16, 12, "Persist")
    r = rd()
    r.add_layer(doc, "Top")
    px0, px1, mk = noise(12, 16, 11), noise(12, 16, 12), noise(12, 16, 13)
    px0[:3, :, 3] = 0  # colour under zero alpha must survive too
    paint_pixels(doc.layers[0], px0)
    paint_pixels(doc.layers[1], px1)
    r.add_mask(doc, 1, "BLACK")
    mk[..., 0] = mk[..., 1] = mk[..., 2] = mk[..., 0]
    mk[..., 3] = 255
    paint_pixels(doc.layers[1], mk, mask=True)
    doc.layers[1].opacity = 0.625
    doc.layers[1].blend_mode = "SCREEN"
    doc.layers[1].locked = True
    doc.layers[0].visible = False
    doc.background = "COLOR"
    doc.background_color = (0.25, 0.5, 0.75, 1.0)
    doc.layers[1].name = "Renamed"
    flush()
    scene = save_and_reload(blend_path)
    doc = scene.seam.rasters[0]
    assert (doc.name, doc.width, doc.height, doc.background) == ("Persist", 16, 12, "COLOR")
    assert tuple(round(c, 4) for c in doc.background_color) == (0.25, 0.5, 0.75, 1.0)
    l0, l1 = doc.layers[0], doc.layers[1]
    assert (l0.name, l1.name) == ("Layer 1", "Renamed")
    assert l0.visible is False and l1.locked is True and l1.blend_mode == "SCREEN"
    assert abs(l1.opacity - 0.625) < 1e-6 and l1.mask_enabled
    assert np.array_equal(pixels(l0.image), px0)
    assert np.array_equal(pixels(l1.image), px1)
    assert np.array_equal(pixels(l1.mask), mk)
    assert l1.image.packed_file is not None and l1.image.colorspace_settings.name == "sRGB"
    assert l1.mask.colorspace_settings.name == "Non-Color"
    assert doc.canvas.seam.raster_id == doc.id and doc.canvas.seam.role == "RASTER"
    assert doc.material.node_tree is not None
    assert not l1.image.is_float


def test_save_pre_packs_dirty_images(seam, blend_path):
    doc = new_doc(8, 6, "Pack")
    paint_pixels(doc.layers[0], noise(6, 8, 5))
    img = doc.layers[0].image
    assert img.packed_file is None and img.is_dirty
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    assert img.packed_file is not None and not img.is_dirty
    # editing again and saving again re-packs
    paint_pixels(doc.layers[0], noise(6, 8, 6))
    assert img.is_dirty
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    img2 = bpy.context.scene.seam.rasters[0].layers[0].image
    assert np.array_equal(pixels(img2), noise(6, 8, 6))


def test_undo_redo_layer_property(seam):
    doc = new_doc(8, 6)
    r = rd()
    r.add_layer(doc, "B")
    flush()
    bpy.ops.ed.undo_push(message="base0")
    bpy.ops.ed.undo_push(message="base")
    doc.layers[1].opacity = 0.3
    flush()
    bpy.ops.ed.undo_push(message="opacity")
    doc.layers[1].blend_mode = "MULTIPLY"
    flush()
    bpy.ops.ed.undo_push(message="blend")
    bpy.ops.ed.undo()
    flush()
    scene = bpy.context.scene
    doc = scene.seam.rasters[0]
    assert doc.layers[1].blend_mode == "NORMAL" and abs(doc.layers[1].opacity - 0.3) < 1e-6
    assert doc.material[r.MAT_SIG_KEY] == r.material_signature(doc)
    assert not [n for n in doc.material.node_tree.nodes if n.bl_idname == "ShaderNodeMix" and n.blend_type == "MULTIPLY"]
    bpy.ops.ed.undo()
    flush()
    scene = bpy.context.scene
    doc = scene.seam.rasters[0]
    assert doc.layers[1].opacity == 1.0
    assert doc.material[r.MAT_SIG_KEY] == r.material_signature(doc)
    assert not [n for n in doc.material.node_tree.nodes if n.label == "Opacity"]
    bpy.ops.ed.redo()
    bpy.ops.ed.redo()
    flush()
    scene = bpy.context.scene
    doc = scene.seam.rasters[0]
    assert doc.layers[1].blend_mode == "MULTIPLY" and abs(doc.layers[1].opacity - 0.3) < 1e-6
    assert doc.material[r.MAT_SIG_KEY] == r.material_signature(doc)
    types = [n.blend_type for n in doc.material.node_tree.nodes if n.bl_idname == "ShaderNodeMix"]
    assert "MULTIPLY" in types
    ops = [n for n in doc.material.node_tree.nodes if n.label == "Opacity"]
    assert len(ops) == 1 and abs(ops[0].inputs[1].default_value - 0.3) < 1e-6


def test_undo_layer_operator_keeps_pixels(seam):
    doc = new_doc(8, 6)
    r = rd()
    bpy.ops.ed.undo_push(message="base")
    px = noise(6, 8, 21)
    paint_pixels(doc.layers[0], px)
    bpy.ops.ed.undo_push(message="painted")
    assert bpy.ops.seam.raster_layer_add() == {"FINISHED"}
    bpy.ops.ed.undo_push(message="add layer")  # operators do not push undo steps in background mode
    scene = bpy.context.scene
    assert len(scene.seam.rasters[0].layers) == 2
    bpy.ops.ed.undo()
    flush()
    scene = bpy.context.scene
    doc = scene.seam.rasters[0]
    assert len(doc.layers) == 1
    assert np.array_equal(pixels(doc.layers[0].image), px)
    assert len([n for n in image_nodes(doc) if n.image == doc.layers[0].image]) == 1
    bpy.ops.ed.redo()
    flush()
    scene = bpy.context.scene
    doc = scene.seam.rasters[0]
    assert len(doc.layers) == 2
    assert np.array_equal(pixels(doc.layers[0].image), px)
    assert doc.material[r.MAT_SIG_KEY] == r.material_signature(doc)
