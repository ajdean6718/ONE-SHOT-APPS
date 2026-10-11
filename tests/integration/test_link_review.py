"""Regression tests for the link/lifecycle review findings (review-findings-r1): undo colour drift, independent
copies, linked-copy scenes, results shown in a second scene, add-on re-enable, baked raster decals."""

import addon_utils
import bpy
import numpy as np
import pytest

from helpers import flush
from test_link_support import activate, extent, make_link, make_shape, move_point

U = ("EXEC_DEFAULT", True)  # operator call that pushes its undo step, like a click


def rgba(mat):
    return tuple(round(float(c), 4) for c in mat.diffuse_color)


def principled(mat):
    return next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def link0():
    return bpy.context.scene.seam.links[0]


# ------------------------------------------------------------------ undo: link material colours

def test_undo_of_a_source_fill_change_recolours_the_result(engine):
    """Fill edit pushed before the engine tick, another action, Ctrl+Z twice (tick after each): the Face/Side
    materials that the post-undo rebuild recoloured must follow the restored fill (5.2 reuses the Material)."""
    bpy.ops.ed.undo_push(message="start")
    src = make_shape("A", color=(0.8, 0.2, 0.1, 1.0))
    sname = src.name
    bpy.ops.ed.undo_push(message="shape")
    flush()
    activate(src)
    assert bpy.ops.seam.make_3d(*U, recipe="EXTRUDE", orientation="FLAT") == {"FINISHED"}
    flush()
    make_shape("B", location=(500, 0))
    bpy.ops.ed.undo_push(message="B")
    flush()
    bpy.data.objects[sname].seam.style.fill_color = (0.0, 1.0, 0.0, 1.0)
    bpy.ops.ed.undo_push(message="Fill")        # pushed on commit, before the engine timer
    flush()
    face = link0().target.data.materials[0]
    assert rgba(face) == (0.0, 1.0, 0.0, 1.0)
    make_shape("C", location=(900, 0))
    bpy.ops.ed.undo_push(message="C")
    flush()
    for _ in range(2):
        assert bpy.ops.ed.undo() == {"FINISHED"}
        flush()
    lk = link0()
    mats = {m["seam_material_role"]: m for m in lk.target.data.materials}
    assert tuple(bpy.data.objects[sname].seam.style.fill_color) == pytest.approx((0.8, 0.2, 0.1, 1.0))
    assert rgba(mats["FACE"]) == pytest.approx((0.8, 0.2, 0.1, 1.0), abs=1e-4)
    assert rgba(mats["SIDE"]) == pytest.approx((0.52, 0.13, 0.065, 1.0), abs=1e-4)
    assert tuple(principled(mats["FACE"]).inputs["Base Color"].default_value) == pytest.approx((0.8, 0.2, 0.1, 1.0), abs=1e-4)
    assert lk.state == "LIVE" and not lk.stale
    assert bpy.ops.ed.redo() == {"FINISHED"}
    flush()
    assert rgba(link0().target.data.materials[0]) == pytest.approx((0.0, 1.0, 0.0, 1.0), abs=1e-4)


def test_colour_change_recolours_without_rewriting_the_mesh(engine):
    src = make_shape("A", color=(0.8, 0.2, 0.1, 1.0))
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    n, sig = link.build_count, link.target_signature
    src.seam.style.fill_color = (0.1, 0.2, 0.9, 1.0)
    res = flush()
    assert [r.message for r in res if r.key.startswith("link") and r.status == "ok"] == ["materials re-synced"]
    assert link.build_count == n and link.target_signature == sig and link.state == "LIVE"
    assert rgba(link.target.data.materials[0]) == pytest.approx((0.1, 0.2, 0.9, 1.0), abs=1e-4)
    # a downstream mesh edit puts the link in CONFLICT; colours still follow (the mesh is never touched)
    me = link.target.data
    me.vertices[0].co.z += 0.5
    move_point(src, 0, 1, dx=40.0)
    flush()
    assert link.state == "CONFLICT"
    edited = me.vertices[0].co.z
    src.seam.style.fill_color = (0.9, 0.9, 0.1, 1.0)
    flush()
    assert link.state == "CONFLICT" and me.vertices[0].co.z == edited
    assert rgba(link.target.data.materials[0]) == pytest.approx((0.9, 0.9, 0.1, 1.0), abs=1e-4)


def test_colour_change_while_paused_waits_for_resume(engine):
    src = make_shape("A", color=(0.8, 0.2, 0.1, 1.0))
    link = make_link(src, "EXTRUDE")
    activate(link.target)
    assert bpy.ops.seam.link_pause() == {"FINISHED"}
    src.seam.style.fill_color = (0.0, 0.0, 1.0, 1.0)
    flush()
    assert link.state == "PAUSED" and link.stale
    assert rgba(link.target.data.materials[0]) == pytest.approx((0.8, 0.2, 0.1, 1.0), abs=1e-4)
    activate(link.target)
    assert bpy.ops.seam.link_resume() == {"FINISHED"}
    flush()
    assert link.state == "LIVE" and not link.stale
    assert rgba(link.target.data.materials[0]) == pytest.approx((0.0, 0.0, 1.0, 1.0), abs=1e-4)


def test_reload_resyncs_drifted_link_materials(engine, blend_path):
    from helpers import save_and_reload
    src = make_shape("A", color=(0.8, 0.2, 0.1, 1.0))
    make_link(src, "EXTRUDE")
    link0().target.data.materials[0].diffuse_color = (0.3, 0.3, 0.3, 1.0)  # drift the record cannot see
    save_and_reload(blend_path)
    flush()
    assert rgba(link0().target.data.materials[0]) == pytest.approx((0.8, 0.2, 0.1, 1.0), abs=1e-4)


# ------------------------------------------------------------------ Shift+D copy of a result

def test_shift_d_copy_of_a_result_gets_its_own_materials(engine):
    src = make_shape("S", color=(0.8, 0.2, 0.1, 1.0))
    link = make_link(src)
    tgt = link.target
    activate(tgt)
    assert bpy.ops.object.duplicate("EXEC_DEFAULT", True) == {"FINISHED"}
    copy = bpy.context.active_object
    flush()
    assert copy != tgt and copy.seam.role == "NONE" and copy.seam.owner_id == "" and copy.data != tgt.data
    assert all(m.get("seam_link_id") == "" for m in copy.data.materials)
    assert not ({m.name for m in copy.data.materials} & {m.name for m in tgt.data.materials})
    before = rgba(copy.data.materials[0])
    src.seam.style.fill_color = (0.0, 0.0, 1.0, 1.0)
    flush()
    assert rgba(copy.data.materials[0]) == before                       # the copy is independent
    assert rgba(tgt.data.materials[0]) == (0.0, 0.0, 1.0, 1.0)           # the original still follows
    copy.data.materials[0].diffuse_color = (0.2, 0.9, 0.2, 1.0)          # a colour set on the copy stays
    move_point(src, 0, 1, dx=10.0)
    flush()
    assert rgba(copy.data.materials[0]) == (0.2, 0.9, 0.2, 1.0)


def test_shift_d_copy_of_a_plane_decal_gets_its_own_image(engine):
    from seam_studio.paint import rasterdoc
    src = make_shape("S", color=(0.8, 0.2, 0.1, 1.0))
    link = make_link(src, "DECAL", decal_mode="PLANE")
    flush()
    tgt = link.target
    activate(tgt)
    assert bpy.ops.object.duplicate("EXEC_DEFAULT", True) == {"FINISHED"}
    copy = bpy.context.active_object
    flush()
    node = copy.data.materials[0].node_tree.nodes["SEAM Decal Image"]
    assert node.image is not None and node.image != link.decal_image
    before = rasterdoc.read_image_srgb8(node.image).copy()
    src.seam.style.fill_color = (0.0, 0.0, 1.0, 1.0)
    flush()
    assert np.array_equal(rasterdoc.read_image_srgb8(node.image), before)
    assert not np.array_equal(rasterdoc.read_image_srgb8(link.decal_image), before)


# ------------------------------------------------------------------ scenes

def test_linked_copy_scene_does_not_raise_a_false_conflict(engine):
    src = make_shape("S")
    make_link(src)
    flush()
    a_scene = bpy.context.scene
    bpy.ops.scene.new(type="LINK_COPY")
    b_scene = bpy.context.scene
    assert b_scene != a_scene and src.name in b_scene.objects
    move_point(src, 0, 1, dx=50)
    assert "ok" in [r.status for r in engine.flush(b_scene) if r.key.startswith("link")]
    engine.request_revalidate()
    res = engine.flush(a_scene)
    la = a_scene.seam.links[0]
    assert la.state == "LIVE", (la.message, [(r.key, r.status) for r in res])
    # a real downstream edit is still detected from either scene
    la.target.data.vertices[0].co.z += 0.25
    move_point(src, 0, 1, dx=10)
    engine.flush(a_scene)
    assert la.state == "CONFLICT"


def _show_in_scene_b(obj):
    b = bpy.data.scenes.new("B")
    b.collection.objects.link(obj)     # Object > Link/Transfer to Scene
    from seam_studio.link import lifecycle
    lifecycle.on_update(b)             # what the depsgraph handler does while a window shows scene B
    return b


@pytest.mark.parametrize("recipe,params", [("EXTRUDE", {}), ("DECAL", {"decal_mode": "PLANE"})])
def test_result_shown_in_a_second_scene_keeps_role_and_owner(engine, recipe, params):
    from seam_studio.link import decal
    src = make_shape("S")
    link = make_link(src, recipe, **params)
    flush()
    tgt, lid = link.target, link.id
    n_slots = len(tgt.data.materials)
    _show_in_scene_b(tgt)
    assert tgt.seam.role == "RESULT" and tgt.seam.owner_id == lid
    src.seam.style.fill_color = (0.0, 0.0, 1.0, 1.0)
    move_point(src, 0, 1, dx=20.0)
    flush()
    lk = link0()
    assert lk.state == "LIVE" and len(tgt.data.materials) == n_slots
    if recipe == "DECAL":
        assert decal.mode_of(lk) == "PLANE"
        activate(tgt)
        assert bpy.ops.seam.link_overwrite.poll() and bpy.ops.seam.duplicate_result.poll()


# ------------------------------------------------------------------ add-on re-enable

def test_reenabling_the_addon_revalidates_links(engine, blend_path):
    src = make_shape("S")
    link = make_link(src)
    flush()
    tname = link.target.name
    w0 = extent(link.target)[0]
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    addon_utils.disable("seam_studio", default_set=True)
    try:
        bpy.ops.wm.open_mainfile(filepath=blend_path)
        s = bpy.data.objects["S"]
        move_point(s, 0, 1, dx=200)
        move_point(s, 0, 2, dx=200)
        bpy.context.view_layer.update()
    finally:
        addon_utils.enable("seam_studio", default_set=True)
    from seam_studio.engine import core
    res = core.flush()
    assert "ok" in [r.status for r in res if r.key.startswith("link")]
    assert extent(bpy.data.objects[tname])[0] == pytest.approx(w0 + 2.0, abs=1e-5)
    assert bpy.context.scene.seam.links[0].state == "LIVE"


# ------------------------------------------------------------------ baked raster decals

def _rd():
    from seam_studio.paint import rasterdoc
    return rasterdoc


def _solid(h, w, c):
    a = np.empty((h, w, 4), np.uint8)
    a[:] = c
    return a


def _node_image(obj):
    return obj.data.materials[0].node_tree.nodes["SEAM Decal Image"].image


def _raster_decal():
    doc = _rd().new_raster_doc(bpy.context.scene, "Doc", 8, 4, "TRANSPARENT")
    _rd().write_layer_array_srgb8(doc.layers[0], _solid(4, 8, (200, 30, 20, 255)))
    flush()
    link = make_link(doc.canvas, "DECAL", decal_mode="PLANE")
    assert link.state == "LIVE" and link.decal_image == doc.composite, link.message
    return link


def test_baked_raster_decal_stops_following_the_document(engine):
    link = _raster_decal()
    tgt = link.target
    activate(tgt)
    assert bpy.ops.seam.link_detach() == {"FINISHED"}
    doc = bpy.context.scene.seam.rasters[0]
    assert link.state == "DETACHED" and _node_image(tgt) != doc.composite and link.decal_image == _node_image(tgt)
    _rd().write_layer_array_srgb8(doc.layers[0], _solid(4, 8, (0, 0, 255, 255)))
    flush()
    assert tuple(_rd().read_image_srgb8(_node_image(tgt))[0, 0]) == (200, 30, 20, 255)
    _rd().resize_canvas(doc, 16, 4)                       # replaces doc.composite
    flush()
    assert _node_image(tgt) is not None and tuple(_node_image(tgt).size) == (8, 4)
    assert _node_image(tgt).packed_file is not None       # survives save/reload


def test_removed_raster_link_keeps_its_picture(engine):
    link = _raster_decal()
    tgt = link.target
    from seam_studio.link import ops
    ops.remove_link(bpy.context.scene, link, delete_result=False)
    doc = bpy.context.scene.seam.rasters[0]
    assert _node_image(tgt) != doc.composite
    _rd().write_layer_array_srgb8(doc.layers[0], _solid(4, 8, (0, 0, 255, 255)))
    flush()
    assert tuple(_rd().read_image_srgb8(_node_image(tgt))[0, 0]) == (200, 30, 20, 255)


def test_freeze_raster_users_keeps_the_texture_when_the_document_goes(engine):
    """``decal.freeze_raster_users`` is the hook the paint module calls before it removes a composite."""
    from seam_studio.link import decal
    link = _raster_decal()
    tgt = link.target
    doc = bpy.context.scene.seam.rasters[0]
    assert decal.freeze_raster_users(doc.composite, removing_source=False) == 0   # a LIVE link keeps following
    assert decal.freeze_raster_users(doc.composite) == 1
    activate(doc.canvas)
    assert bpy.ops.seam.raster_remove() == {"FINISHED"}
    flush()
    lk = link0()
    assert lk.state == "MISSING_SOURCE"
    assert _node_image(tgt) is not None and tuple(_rd().read_image_srgb8(_node_image(tgt))[0, 0]) == (200, 30, 20, 255)
