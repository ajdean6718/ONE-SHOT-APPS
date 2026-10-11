"""Linked selection and paint-on-result (connect)."""

import bpy
import numpy as np
import pytest

from helpers import flush, save_and_reload
from test_connect_decal import add_mesh_with_uv, make_decal_pair, px
from test_link_support import SQUARE, activate, make_link, make_shape, move_point
from test_vector_b_common import add_rect, select


def selected_names():
    return sorted(o.name for o in bpy.context.selected_objects)


def only_active(obj):
    """Active object without any selection side effect other than the object itself."""
    for o in list(bpy.context.selected_objects):
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


# ------------------------------------------------------------------ linked selection

def test_sync_selects_results_of_the_active_source_and_back(engine):
    from seam_studio.link import lifecycle
    a = make_shape("A")
    la, lb = make_link(a, "EXTRUDE"), make_link(a, "SWEEP")
    only_active(a)
    got = lifecycle.sync_linked_selection(bpy.context)
    assert sorted(o.name for o in got) == sorted([la.target.name, lb.target.name])
    assert bpy.context.view_layer.objects.active == a      # the active object never changes
    assert set(selected_names()) == {a.name, la.target.name, lb.target.name}
    assert lifecycle.sync_linked_selection(bpy.context) == []   # idempotent
    # vice versa: a result pulls its source
    only_active(la.target)
    got = lifecycle.sync_linked_selection(bpy.context)
    assert got == [a] and bpy.context.view_layer.objects.active == la.target
    assert set(selected_names()) == {a.name, la.target.name}
    # unrelated objects are untouched
    other = make_shape("Other", location=(500, 0))
    only_active(other)
    assert lifecycle.sync_linked_selection(bpy.context) == [] and selected_names() == [other.name]
    # no active object: nothing happens
    bpy.context.view_layer.objects.active = None
    assert lifecycle.sync_linked_selection(bpy.context) == []


def test_sync_skips_detached_hidden_and_locked_ends(engine):
    from seam_studio.link import lifecycle
    a = make_shape("A")
    link = make_link(a, "EXTRUDE")
    tgt = link.target
    only_active(a)
    tgt.hide_set(True)
    assert lifecycle.sync_linked_selection(bpy.context) == []
    tgt.hide_set(False)
    tgt.hide_select = True
    assert lifecycle.sync_linked_selection(bpy.context) == []
    tgt.hide_select = False
    link.state = "DETACHED"
    assert lifecycle.sync_linked_selection(bpy.context) == []
    link.state = "LIVE"
    assert lifecycle.sync_linked_selection(bpy.context) == [tgt]
    # a missing source never raises
    only_active(tgt)
    link.source = None
    link.state = "MISSING_SOURCE"
    assert lifecycle.sync_linked_selection(bpy.context) == []


def test_sync_covers_decals_plane_and_material(engine):
    from seam_studio.link import lifecycle
    art = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100)
    cube = add_mesh_with_uv()
    plane_link = make_decal_pair(art, cube)       # material decal: the target is the user's cube
    activate(art)
    bpy.ops.seam.make_3d(recipe="DECAL", decal_mode="PLANE", decal_resolution=32)
    plane = bpy.context.scene.seam.links[-1].target
    only_active(art)
    got = lifecycle.sync_linked_selection(bpy.context)
    assert set(o.name for o in got) == {cube.name, plane.name}
    only_active(cube)
    assert lifecycle.sync_linked_selection(bpy.context) == [art]
    assert plane_link.target == cube


def test_sync_switch_is_a_window_manager_property(engine):
    from seam_studio.link import lifecycle
    wm = bpy.context.window_manager
    assert wm.seam_sync_link_selection is True and lifecycle.sync_enabled()
    a = make_shape("A")
    link = make_link(a, "EXTRUDE")
    only_active(a)
    wm.seam_sync_link_selection = False
    try:
        assert not lifecycle.sync_enabled()
        lifecycle._on_active_changed()                       # the msgbus callback respects the switch
        assert lifecycle._sync_pending is False
        assert link.target not in bpy.context.selected_objects
        wm.seam_sync_link_selection = True
        lifecycle._on_active_changed()                       # schedules the deferred sync
        assert lifecycle._sync_pending is True
        lifecycle._sync_deferred()                           # what the timer tick runs
        assert lifecycle._sync_pending is False and link.target in bpy.context.selected_objects
    finally:
        wm.seam_sync_link_selection = True
        lifecycle._sync_pending = False


# ------------------------------------------------------------------ paint on result

def paint_image_of(obj):
    from seam_studio.link import resultpaint
    return resultpaint.find_paint_image(obj)


def test_result_paint_setup_wires_uv_image_and_slots(engine):
    a = make_shape("A", color=(0.8, 0.2, 0.1, 1.0))
    link = make_link(a, "EXTRUDE")
    tgt = link.target
    activate(tgt)
    n_images = len(bpy.data.images)
    assert bpy.ops.seam.result_paint_setup(size=64, enter_paint=False) == {"FINISHED"}
    me = tgt.data
    assert "UVMap" in me.uv_layers and len(me.uv_layers) == 1
    img = paint_image_of(tgt)
    assert img is not None and img.name == "SEAM Paint A · Extrude" and tuple(img.size) == (64, 64)
    assert img.colorspace_settings.name == "sRGB" and not img.is_float and len(bpy.data.images) == n_images + 1
    assert np.all(np.asarray(img.pixels[:]) == 0.0)        # fully transparent
    for mat in me.materials:                                # every slot paints into the same image
        nodes = mat.node_tree.nodes
        assert nodes["SEAM Paint Image"].image == img and nodes.active == nodes["SEAM Paint Image"]
        out = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
        mix = out.inputs["Surface"].links[0].from_node
        assert mix.name == "SEAM Paint Mix" and mix.type == "MIX_SHADER"
        # the original surface (Principled with the link colour) is still the first shader
        assert mix.inputs[1].links[0].from_node.type == "BSDF_PRINCIPLED"
        assert mix.inputs[2].links[0].from_node.type == "BSDF_DIFFUSE"
        assert mix.inputs["Fac"].links[0].from_socket.name == "Alpha"
    assert tgt.active_material_index == 0
    assert tgt.mode == "OBJECT" and link.state == "LIVE"
    # entering the native mode turns the nodes into texture-paint slots with the SEAM image active
    assert bpy.ops.seam.result_paint_setup(size=64, enter_paint=True) == {"FINISHED"}
    assert tgt.mode == "TEXTURE_PAINT"
    for mat in me.materials:
        assert [t.name for t in mat.texture_paint_images] == [img.name] and mat.paint_active_slot == 0
    assert bpy.context.tool_settings.image_paint.mode == "MATERIAL"
    bpy.ops.object.mode_set(mode="OBJECT")
    # idempotent: second run reuses image and nodes
    assert bpy.ops.seam.result_paint_setup(size=256, enter_paint=False) == {"FINISHED"}
    assert paint_image_of(tgt) == img and tuple(img.size) == (64, 64) and len(bpy.data.images) == n_images + 1
    assert sum(1 for n in me.materials[0].node_tree.nodes if n.name == "SEAM Paint Image") == 1


def test_painted_result_keeps_following_the_source(engine):
    a = make_shape("A", color=(0.8, 0.2, 0.1, 1.0))
    link = make_link(a, "EXTRUDE")
    tgt = link.target
    activate(tgt)
    bpy.ops.seam.result_paint_setup(size=32, enter_paint=False)
    img = paint_image_of(tgt)
    buf = np.zeros(32 * 32 * 4, np.float32)
    buf.reshape(-1, 4)[100] = (1.0, 0.0, 0.0, 1.0)
    img.pixels.foreach_set(buf)
    uv_before = [tuple(d.uv) for d in tgt.data.uv_layers["UVMap"].data]
    # colour sync of the link materials still reaches the original Principled node
    a.seam.style.fill_color = (0.1, 0.9, 0.3, 1.0)
    flush()
    face = tgt.data.materials[0]
    bsdf = next(n for n in face.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    assert tuple(bsdf.inputs["Base Color"].default_value) == pytest.approx((0.1, 0.9, 0.3, 1.0), abs=1e-5)
    assert link.build_count == 1  # a colour-only change recolours the materials, it never rewrites the mesh
    # topology/geometry change: link stays LIVE (painting is not a mesh edit), UVs are deterministic,
    # image and nodes survive, painted texels are untouched
    move_point(a, 0, 1, dx=100.0)
    flush()
    assert link.state == "LIVE" and link.build_count == 2
    assert paint_image_of(tgt) == img
    out = np.empty(32 * 32 * 4, np.float32)
    img.pixels.foreach_get(out)
    assert np.array_equal(out, buf)
    assert face.node_tree.nodes["SEAM Paint Image"].image == img
    assert len(tgt.data.uv_layers) == 1
    move_point(a, 0, 1, dx=-100.0)
    flush()
    assert [tuple(d.uv) for d in tgt.data.uv_layers["UVMap"].data] == uv_before   # same shape => same UVs


def test_every_recipe_result_has_deterministic_uvs(engine):
    from seam_studio.link import producer
    checked = []
    for k, recipe in enumerate(("EXTRUDE", "SWEEP", "REVOLVE")):
        a = make_shape("S" + recipe, rings=(((0, 0), (100, 0), (100, 100)),), closed=recipe == "EXTRUDE",
                       location=(k * 400.0, 0))
        activate(a)
        try:
            bpy.ops.seam.make_3d(recipe=recipe)
        except RuntimeError:
            continue
        link = bpy.context.scene.seam.links[-1]
        if link.state != "LIVE":
            continue
        me = link.target.data
        assert "UVMap" in me.uv_layers, recipe
        uv = np.empty(len(me.loops) * 2, np.float32)
        me.uv_layers["UVMap"].data.foreach_get("uv", uv)
        assert uv.min() >= -1e-6 and uv.max() <= 1.0 + 1e-6, recipe
        assert link.target_data_signature == producer.read_data_signature(me)
        checked.append(recipe)
    assert "EXTRUDE" in checked and "SWEEP" in checked


def test_box_uvs_are_position_based_and_non_overlapping():
    from seam_studio.link import recipes
    # a unit cube: 6 quads, one atlas cell each
    v = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], np.float64)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    totals = np.full(6, 4, np.int32)
    corners = np.array([i for q in quads for i in q], np.int32)
    uv = recipes.box_uvs(v, totals, corners)
    assert uv.shape == (24, 2) and uv.min() > 0.0 and uv.max() < 1.0
    cells = {tuple((uv[i * 4:(i + 1) * 4].mean(axis=0) * (3, 2)).astype(int)) for i in range(6)}
    assert len(cells) == 6
    # independent of vertex numbering: permute the vertices and keep the faces
    perm = np.random.RandomState(1).permutation(8)
    inv = np.argsort(perm)
    uv2 = recipes.box_uvs(v[perm], totals, inv[corners].astype(np.int32))
    assert np.allclose(uv, uv2, atol=1e-6)
    assert recipes.box_uvs(np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0, np.int32)).shape == (0, 2)


def test_result_paint_setup_on_plain_meshes(engine):
    from seam_studio.link import resultpaint
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.active_object
    assert len(cube.data.materials) == 0
    assert bpy.ops.seam.result_paint_setup(size=16, enter_paint=False) == {"FINISHED"}
    assert [m.name for m in cube.data.materials] == ["SEAM Paint"]
    assert "UVMap" in cube.data.uv_layers
    img = resultpaint.find_paint_image(cube)
    assert tuple(img.size) == (16, 16) and cube.active_material.node_tree.nodes.active.image == img
    # no UV map at all: deterministic UVs are created
    bpy.ops.mesh.primitive_uv_sphere_add(location=(4, 0, 0))
    ball = bpy.context.active_object
    while len(ball.data.uv_layers):
        ball.data.uv_layers.remove(ball.data.uv_layers[0])
    assert bpy.ops.seam.result_paint_setup(size=16, enter_paint=False) == {"FINISHED"}
    uvl = ball.data.uv_layers["UVMap"]
    uv = np.empty(len(ball.data.loops) * 2, np.float32)
    uvl.data.foreach_get("uv", uv)
    assert uv.min() > 0.0 and uv.max() < 1.0 and len(np.unique(uv.round(4))) > 10
    # a mesh with another UV map keeps painting on that one (no extra map is added)
    bpy.ops.mesh.primitive_cube_add(location=(8, 0, 0))
    odd = bpy.context.active_object
    odd.data.uv_layers[0].name = "Custom"
    bpy.ops.seam.result_paint_setup(size=16, enter_paint=False)
    assert [u.name for u in odd.data.uv_layers] == ["Custom"]


def test_result_paint_setup_enters_texture_paint_and_blocks_builds_until_left(engine):
    a = make_shape("A")
    link = make_link(a, "EXTRUDE")
    activate(link.target)
    assert bpy.ops.seam.result_paint_setup(size=32) == {"FINISHED"}
    assert link.target.mode == "TEXTURE_PAINT"
    n = link.build_count
    move_point(a, 0, 1, dx=100.0)
    flush()
    assert link.build_count == n and link.stale and "Texture Paint" in link.message   # waits, no overwrite
    bpy.ops.object.mode_set(mode="OBJECT")
    flush()
    assert link.build_count == n + 1 and not link.stale and link.state == "LIVE"


def test_result_paint_setup_polls(engine):
    bpy.context.view_layer.objects.active = None
    assert not bpy.ops.seam.result_paint_setup.poll()
    a = make_shape("A")
    activate(a)
    assert not bpy.ops.seam.result_paint_setup.poll()        # 2D artwork cannot be painted in 3D
    link = make_link(a, "EXTRUDE")
    activate(link.target)
    assert bpy.ops.seam.result_paint_setup.poll()


def test_paint_image_is_packed_and_survives_save_reopen(engine, blend_path):
    a = make_shape("A")
    link = make_link(a, "EXTRUDE")
    lid = link.id
    activate(link.target)
    bpy.ops.seam.result_paint_setup(size=16, enter_paint=False)
    img = paint_image_of(link.target)
    buf = np.zeros(16 * 16 * 4, np.float32)
    buf.reshape(-1, 4)[37] = (0.0, 0.0, 1.0, 1.0)
    img.pixels.foreach_set(buf)
    scene = save_and_reload(blend_path)
    link = next(lk for lk in scene.seam.links if lk.id == lid)
    img = paint_image_of(link.target)
    assert img is not None and img.packed_file is not None and tuple(img.size) == (16, 16)
    out = np.empty(16 * 16 * 4, np.float32)
    img.pixels.foreach_get(out)
    assert np.allclose(out.reshape(-1, 4)[37], (0.0, 0.0, 1.0, 1.0), atol=1 / 255)
    assert link.target.data.materials[0].node_tree.nodes["SEAM Paint Image"].image == img


def test_material_decal_target_can_be_painted_too(engine):
    art = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100)
    cube = add_mesh_with_uv()
    link = make_decal_pair(art, cube)
    activate(cube)
    assert bpy.ops.seam.result_paint_setup(size=16, enter_paint=False) == {"FINISHED"}
    assert link.state == "LIVE"
    img = paint_image_of(cube)
    assert img is not None
    for mat in cube.data.materials:
        assert mat.node_tree.nodes["SEAM Paint Image"].image == img
    art.seam.style.fill_color = (0, 1, 0, 1)
    flush()
    assert link.state == "LIVE" and paint_image_of(cube) == img


def test_box_uvs_are_right_handed_from_outside_on_every_axis_and_sign():
    """u x v must point at a viewer looking at the face from outside (otherwise textures read mirrored)."""
    from seam_studio.link import recipes
    for axis in range(3):
        for sign in (1.0, -1.0):
            ua, va = [(1, 2), (0, 2), (0, 1)][axis]
            quad = []
            for (a, b) in ((0, 0), (1, 0), (1, 1), (0, 1)):  # CCW in (ua, va)
                p = [0.0, 0.0, 0.0]
                p[ua], p[va] = float(a), float(b)
                p[axis] = 0.5 * sign
                quad.append(p)
            verts = np.array(quad, dtype=np.float32)
            n = np.cross(verts[1] - verts[0], verts[2] - verts[1])
            if np.sign(n[axis]) != sign:  # wind the quad so that the geometric normal is axis * sign
                verts = verts[::-1].copy()
                n = np.cross(verts[1] - verts[0], verts[2] - verts[1])
            assert np.sign(n[axis]) == sign
            uv = recipes.box_uvs(verts, np.array([4]), np.arange(4))
            # texture right/up directions in object space from the UV gradient
            du = np.array([uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]])
            dv = np.array([uv[2][0] - uv[1][0], uv[2][1] - uv[1][1]])
            e1, e2 = verts[1] - verts[0], verts[2] - verts[1]
            M = np.array([du, dv]).T  # solve e = U*r + V*u for the object-space unit vectors of u and v
            inv = np.linalg.inv(M)
            ru = e1 * inv[0][0] + e2 * inv[0][1]
            rv = e1 * inv[1][0] + e2 * inv[1][1]
            out = np.cross(ru, rv)
            assert np.sign(out[axis]) == sign, (axis, sign, "mirrored")


def test_paint_setup_never_paints_on_materials_shared_with_other_objects(engine):
    a = add_mesh_with_uv("A", (0, 0, 0))
    mat = bpy.data.materials.new("Shared")
    if mat.node_tree is None:
        mat.use_nodes = True
    a.data.materials.append(mat)
    bpy.ops.mesh.primitive_cube_add(location=(4, 0, 0))
    b = bpy.context.active_object
    b.data.materials.append(mat)
    assert mat.users == 2
    from seam_studio.link import resultpaint
    resultpaint.setup(a)
    assert "SEAM Paint Image" not in mat.node_tree.nodes          # the shared material is untouched
    own = a.data.materials[0]
    assert own != mat and "SEAM Paint Image" in own.node_tree.nodes
    assert b.data.materials[0] == mat
    # b gets its own image, never a's
    resultpaint.setup(b)
    assert resultpaint.find_paint_image(a) != resultpaint.find_paint_image(b)
