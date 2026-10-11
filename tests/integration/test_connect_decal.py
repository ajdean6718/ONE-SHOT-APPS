"""DECAL recipe: artwork / raster layer -> live image on a plane or a mesh material (connect)."""

import math

import bpy
import numpy as np
import pytest

from helpers import flush, save_and_reload
from test_link_support import SQUARE, HOLE, activate, make_shape
from test_vector_b_common import add_rect, poly_path, select, srgb8

RED = (0.8, 0.1, 0.05, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)


def px(image):
    from seam_studio.paint import rasterdoc
    return rasterdoc.read_image_srgb8(image)


def make_decal(src, mode="PLANE", **params):
    activate(src)
    res = bpy.ops.seam.make_3d(recipe="DECAL", decal_mode=mode, **params)
    assert res == {"FINISHED"}, res
    return bpy.context.scene.seam.links[-1]


def at(link, bounds, x, y):
    """RGBA8 of the decal image at canvas point (x, y); ``bounds`` = (minx, miny, maxx, maxy) of the artwork."""
    a = px(link.decal_image)
    h, w = a.shape[:2]
    col = min(w - 1, int((x - bounds[0]) / (bounds[2] - bounds[0]) * w))
    row = min(h - 1, int((bounds[3] - y) / (bounds[3] - bounds[1]) * h))
    return tuple(int(v) for v in a[row, col])


def verts(obj):
    return np.array([tuple(v.co) for v in obj.data.vertices])


def extent(obj):
    v = verts(obj)
    return v.max(axis=0) - v.min(axis=0)


def mesh_sig(obj):
    from seam_studio.link import producer
    return producer.read_mesh_signature(obj.data)


def add_mesh_with_uv(name="Cube", location=(5.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cube_add(location=location)
    ob = bpy.context.active_object
    ob.name = name
    assert "UVMap" in ob.data.uv_layers
    return ob


# ------------------------------------------------------------------ plane mode: pixels and geometry

def test_plane_decal_pixels_size_mesh_and_material(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 200, 100, color=RED)
    link = make_decal(src, "PLANE", decal_resolution=256)
    assert link.state == "LIVE" and link.recipe == "DECAL" and link.build_count == 1, link.message
    img = link.decal_image
    assert img is not None and img.name == "SEAM Decal Art"
    assert tuple(img.size) == (256, 128)  # longer side = decal_resolution, aspect kept
    assert img.colorspace_settings.name == "sRGB" and img.alpha_mode == "STRAIGHT"
    b = (0, 0, 200, 100)
    for pt in ((100, 50), (10, 10), (190, 90), (10, 90)):
        assert at(link, b, *pt) == srgb8(RED), pt
    # square shape + hole: even-odd hole is transparent, the ring has the fill colour
    holed = make_shape("Holed", rings=(SQUARE, HOLE), color=RED, location=(500.0, 0.0))
    l2 = make_decal(holed, "PLANE", decal_resolution=128)
    bb = (500 - 100, -100, 500 + 100, 100)
    assert at(l2, bb, 500, 0)[3] == 0
    assert at(l2, bb, 500 - 80, 0) == srgb8(RED)
    # target: plane of bounds x units_scale, UPRIGHT, standing on the origin and facing -Y
    tgt = link.target
    assert tgt.seam.role == "RESULT" and tgt.seam.owner_id == link.id and tgt.type == "MESH"
    me = tgt.data
    assert len(me.polygons) == 1 and len(me.vertices) == 4
    assert tuple(extent(tgt)) == pytest.approx((2.0, 0.0, 1.0), abs=1e-6)
    v = verts(tgt)
    assert v[:, 2].min() == pytest.approx(0.0, abs=1e-6) and v[:, 0].min() == pytest.approx(-1.0, abs=1e-6)
    assert tuple(me.polygons[0].normal) == pytest.approx((0.0, -1.0, 0.0), abs=1e-5)
    uvl = me.uv_layers["UVMap"]
    uvs = sorted(tuple(round(c, 5) for c in d.uv) for d in uvl.data)
    assert uvs == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    # the bottom-left corner of the plane carries uv (0, 0), top-right (1, 1)
    for loop in me.loops:
        co = me.vertices[loop.vertex_index].co
        uv = uvl.data[loop.index].uv
        assert uv[0] == pytest.approx((co.x + 1.0) / 2.0, abs=1e-5) and uv[1] == pytest.approx(co.z / 1.0, abs=1e-5)
    # material: one SEAM slot, image node on the decal image, alpha blended
    assert [m.name for m in me.materials] == ["SEAM Decal"]
    mat = me.materials[0]
    assert mat["seam_link_id"] == link.id
    node = mat.node_tree.nodes["SEAM Decal Image"]
    assert node.image == img and node.extension == "CLIP"
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    assert bsdf.inputs["Base Color"].is_linked and bsdf.inputs["Alpha"].is_linked
    from seam_studio import compat
    if compat.rna_has(mat, "blend_method"):
        assert mat.blend_method == "BLEND"
    if compat.rna_has(mat, "surface_render_method"):
        assert mat.surface_render_method == "BLENDED"
    assert img.users >= 2  # link record + material node: survives a save


def test_plane_decal_flat_orientation_and_units_scale(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 300, color=RED)
    link = make_decal(src, "PLANE", orientation="FLAT", units_scale=0.02, decal_resolution=64)
    assert tuple(link.decal_image.size) == (21, 64)
    ex = extent(link.target)
    assert tuple(ex) == pytest.approx((2.0, 6.0, 0.0), abs=1e-5)
    assert tuple(link.target.data.polygons[0].normal) == pytest.approx((0.0, 0.0, 1.0), abs=1e-5)
    v = verts(link.target)
    assert v.mean(axis=0) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)  # centred on the object origin


def test_decal_renders_stroke_opacity_and_background_transparent(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED, stroke=(BLUE, 20.0))
    link = make_decal(src, "PLANE", decal_resolution=240)
    b = (-10, -10, 110, 110)  # the stroke grows the bounds by half its width
    assert tuple(link.decal_image.size) == (240, 240)
    assert at(link, b, 50, 50) == srgb8(RED)
    assert at(link, b, 2, 50) == srgb8(BLUE)      # inside the stroke band
    assert at(link, b, 105, 105) == srgb8(BLUE)   # the stroke corner is part of the artwork
    src.seam.style.opacity = 0.5
    flush()
    a = at(link, b, 50, 50)
    assert a[3] == pytest.approx(128, abs=1) and a[:3] == srgb8(RED)[:3]
    # no fill and no stroke -> nothing to show: the link reports it and keeps its previous result
    src.seam.style.fill_enabled = False
    src.seam.style.stroke_enabled = False
    flush()
    assert link.state == "ERROR" and "visible" in link.message


def test_decal_text_source(engine):
    bpy.ops.seam.add_text(body="Hi", x=100.0, y=-100.0, size=96.0)
    txt = bpy.context.active_object
    txt.seam.style.fill_color = RED
    flush()
    link = make_decal(txt, "PLANE", decal_resolution=256)
    assert link.state == "LIVE", link.message
    a = px(link.decal_image)
    assert max(a.shape[:2]) == 256
    opaque = a[..., 3] == 255
    assert opaque.any() and (a[..., 3] == 0).any()
    assert set(map(tuple, a[opaque][:, :3].tolist())) == {srgb8(RED)[:3]}
    before = link.build_count
    txt.data.body = "Hi there"
    flush()
    assert link.build_count == before + 1
    assert link.decal_image.size[0] > a.shape[1] or link.decal_image.size[1] != a.shape[0]


def test_decal_group_renders_every_visible_child(engine):
    scene = bpy.context.scene
    a = add_rect(scene, "A", 0, 0, 100, 100, color=RED)
    b = add_rect(scene, "B", 300, 0, 400, 100, color=BLUE)
    c = add_rect(scene, "Hidden", 150, 0, 250, 100, color=(0, 1, 0, 1))
    select(a, b, c)
    assert bpy.ops.seam.group() == {"FINISHED"}
    grp = bpy.context.active_object
    assert grp.seam.role == "GROUP"
    c.hide_set(True)
    link = make_decal(grp, "PLANE", decal_resolution=400)
    assert link.state == "LIVE", link.message
    bb = (0, 0, 400, 100)
    assert tuple(link.decal_image.size) == (400, 100)
    assert at(link, bb, 50, 50) == srgb8(RED)
    assert at(link, bb, 350, 50) == srgb8(BLUE)
    assert at(link, bb, 200, 50)[3] == 0  # the gap (and the hidden child) stay transparent
    # editing a child updates the group's decal; moving the whole group does not
    n = link.build_count
    b.seam.style.fill_color = (1.0, 1.0, 0.0, 1.0)
    flush()
    assert link.build_count == n + 1
    assert at(link, bb, 350, 50) == srgb8((1.0, 1.0, 0.0, 1.0))
    grp.location.x += 500
    flush()
    assert link.build_count == n + 1
    # moving a child relative to the group changes the artwork
    a.location.y += 50
    flush()
    assert link.build_count == n + 2


def test_decal_respects_clip_masks(engine):
    scene = bpy.context.scene
    art = add_rect(scene, "Art", 0, 0, 100, 100, color=RED)
    mask = add_rect(scene, "Mask", 50, 50, 150, 150, color=BLUE)
    select(art, mask)
    assert bpy.ops.seam.clip_make() == {"FINISHED"}
    flush()
    link = make_decal(art, "PLANE", decal_resolution=100)
    assert link.state == "LIVE", link.message
    # the visible area is art AND mask = (50,50)-(100,100): bounds of the clipped artwork items are still the art
    a = px(link.decal_image)
    h, w = a.shape[:2]
    bounds = (0, 0, 100, 100)
    assert at(link, bounds, 75, 75) == srgb8(RED)
    assert at(link, bounds, 25, 25)[3] == 0
    assert at(link, bounds, 75, 25)[3] == 0


# ------------------------------------------------------------------ live updates in place

def test_edits_update_the_same_image_in_place(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    link = make_decal(src, "PLANE", decal_resolution=128)
    img = link.decal_image
    ptr, name, n_images = img.as_pointer(), img.name, len(bpy.data.images)
    n_mats, n_meshes = len(bpy.data.materials), len(bpy.data.meshes)
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(RED)

    src.seam.style.fill_color = (0.1, 0.7, 0.2, 1.0)
    res = flush()
    assert "ok" in [r.status for r in res if r.key == "link:" + link.id]
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8((0.1, 0.7, 0.2, 1.0))
    assert link.decal_image == img and img.as_pointer() == ptr and img.name == name

    # geometry: pull the right edge out => aspect changes, image rescaled in place, plane follows
    sp = src.data.splines[0]
    for bp in sp.bezier_points:
        if bp.co.x > 0:
            bp.co.x += 100.0
            bp.handle_left.x += 100.0
            bp.handle_right.x += 100.0
    src.data.update_tag()
    flush()
    assert link.build_count == 3 and link.state == "LIVE"
    assert tuple(link.decal_image.size) == (128, 64)
    assert link.decal_image.as_pointer() == ptr
    assert tuple(extent(link.target)) == pytest.approx((2.0, 0.0, 1.0), abs=1e-5)
    assert at(link, (-50, -50, 150, 50), 140, 0) == srgb8((0.1, 0.7, 0.2, 1.0))
    # no new datablocks per edit
    assert (len(bpy.data.images), len(bpy.data.materials), len(bpy.data.meshes)) == (n_images, n_mats, n_meshes)
    # resolution parameter resizes in place as well
    link.params.decal_resolution = 512
    flush()
    assert tuple(link.decal_image.size) == (512, 256) and link.decal_image.as_pointer() == ptr
    assert len(bpy.data.images) == n_images


def test_unchanged_source_and_translation_do_not_rebuild(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    link = make_decal(src, "PLANE", decal_resolution=64)
    n = link.build_count
    src.location.x += 300
    flush()
    engine.request_revalidate()
    flush()
    assert link.build_count == n
    src.rotation_euler.z = math.radians(30)
    flush()
    assert link.build_count == n + 1


def test_pause_queues_decal_changes_until_resume(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    link = make_decal(src, "PLANE", decal_resolution=64)
    activate(link.target)
    assert bpy.ops.seam.link_pause() == {"FINISHED"}
    src.seam.style.fill_color = BLUE
    flush()
    assert link.stale and at(link, (0, 0, 100, 100), 50, 50) == srgb8(RED)
    assert bpy.ops.seam.link_resume() == {"FINISHED"}
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE)


def test_plane_edit_downstream_is_a_conflict_like_other_results(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    link = make_decal(src, "PLANE", decal_resolution=64)
    link.target.data.vertices[0].co.x -= 0.25  # sculpt-like edit
    src.seam.style.fill_color = BLUE
    flush()
    assert link.state == "CONFLICT"
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(RED)  # nothing was overwritten
    activate(link.target)
    assert bpy.ops.seam.link_overwrite() == {"FINISHED"}
    assert link.state == "LIVE"
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE)


# ------------------------------------------------------------------ raster source

@pytest.fixture
def rdoc(seam):
    from seam_studio.paint import rasterdoc
    doc = rasterdoc.new_raster_doc(bpy.context.scene, "Doc", 8, 4, "TRANSPARENT")
    return doc


def paint(layer, rgba8):
    from seam_studio.paint import rasterdoc
    rasterdoc.write_layer_array_srgb8(layer, np.asarray(rgba8, np.uint8))


def solid(h, w, rgba):
    a = np.empty((h, w, 4), np.uint8)
    a[:] = rgba
    return a


def test_raster_decal_shares_the_composite_and_follows_pixel_edits(rdoc, engine):
    from seam_studio.paint import rasterdoc
    flush()
    layer = rdoc.layers[0]
    paint(layer, solid(4, 8, (200, 30, 20, 255)))
    flush()
    link = make_decal(rdoc.canvas, "PLANE")
    assert link.state == "LIVE", link.message
    assert link.decal_image == rdoc.composite  # shared pixels, no copy
    a = px(link.decal_image)
    assert a.shape == (4, 8, 4) and tuple(a[1, 2]) == (200, 30, 20, 255)
    assert tuple(extent(link.target)) == pytest.approx((0.08, 0.0, 0.04), abs=1e-6)
    node = link.target.data.materials[0].node_tree.nodes["SEAM Decal Image"]
    assert node.image == rdoc.composite
    # a pixel edit shows on the decal after the flush
    edit = solid(4, 8, (200, 30, 20, 255))
    edit[2, 5] = (10, 220, 30, 255)
    paint(layer, edit)
    flush()
    a = px(link.decal_image)
    assert tuple(a[2, 5]) == (10, 220, 30, 255) and tuple(a[1, 2]) == (200, 30, 20, 255)
    # a second layer with blending is part of the composite as well
    rasterdoc.add_layer(rdoc, "Top")
    paint(rdoc.layers[1], solid(4, 8, (0, 0, 255, 128)))
    flush()
    assert px(link.decal_image)[0, 0, 2] > 150  # blue over red-ish: the blend shows in the shared composite
    assert link.decal_image.as_pointer() == rdoc.composite.as_pointer()
    # resizing the document rebuilds the plane (new aspect) and keeps following the composite
    rasterdoc.resize_canvas(rdoc, 16, 4)
    flush()
    assert tuple(extent(link.target)) == pytest.approx((0.16, 0.0, 0.04), abs=1e-6)
    assert link.decal_image == rdoc.composite and tuple(link.decal_image.size) == (16, 4)


def test_raster_decal_rejects_nothing_but_plain_meshes(engine, rdoc):
    from seam_studio.link import recipes
    ok, why = recipes.source_ok("DECAL", rdoc.canvas)
    assert ok, why
    cube = add_mesh_with_uv()
    ok, why = recipes.source_ok("DECAL", cube)
    assert not ok and "decal" in why.lower()


# ------------------------------------------------------------------ material mode (target mesh untouched)

def test_material_decal_never_rewrites_the_target_mesh(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    cube = add_mesh_with_uv()
    base_mat = bpy.data.materials.new("Base")
    base_mat.use_nodes = True
    cube.data.materials.append(base_mat)
    cube.data.polygons[0].select = True
    mesh_before = mesh_sig(cube)
    uv_before = [tuple(d.uv) for d in cube.data.uv_layers["UVMap"].data]
    verts_before = verts(cube).copy()
    mesh_ptr = cube.data.as_pointer()
    select(src, cube)  # source selected, the mesh is active (works in either order)
    res = bpy.ops.seam.make_3d(recipe="DECAL", decal_mode="MATERIAL", decal_resolution=64)
    assert res == {"FINISHED"}
    link = bpy.context.scene.seam.links[-1]
    assert link.state == "LIVE" and link.recipe == "DECAL", link.message
    assert link.target == cube and link.source == src
    assert cube.seam.role == "NONE"          # the user's mesh is not turned into a result
    assert len(bpy.context.scene.objects) == 3 or len(bpy.context.scene.objects) >= 3
    assert [m.name for m in cube.data.materials] == ["Base", "SEAM Decal"]
    mat = cube.data.materials[1]
    assert mat["seam_link_id"] == link.id and mat.node_tree.nodes["SEAM Decal Image"].image == link.decal_image
    # mix(base colour, decal, alpha) -> Principled base colour
    mix = mat.node_tree.nodes["SEAM Decal Mix"]
    tex = mat.node_tree.nodes["SEAM Decal Image"]
    assert any(l.from_node == tex and l.to_node == mix for l in mat.node_tree.links)
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    assert bsdf.inputs["Base Color"].is_linked and not bsdf.inputs["Alpha"].is_linked
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(RED)
    # faces were assigned to the new slot once, by the operator
    assert {p.material_index for p in cube.data.polygons} == {1}

    # the user re-assigns a face: the producer must never touch it again
    cube.data.polygons[2].material_index = 0
    n = link.build_count
    src.seam.style.fill_color = BLUE
    flush()
    src.data.update_tag()
    flush()
    assert link.build_count == n + 1 and link.state == "LIVE"
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE)
    assert cube.data.polygons[2].material_index == 0
    assert mesh_sig(cube) == mesh_before
    assert [tuple(d.uv) for d in cube.data.uv_layers["UVMap"].data] == uv_before
    assert np.array_equal(verts(cube), verts_before) and cube.data.as_pointer() == mesh_ptr
    assert cube.data.polygons[0].select  # no attribute churn
    assert [m.name for m in cube.data.materials] == ["Base", "SEAM Decal"]
    # downstream edits of the target are not conflicts (the link does not own the mesh)
    cube.data.vertices[0].co.x += 0.1
    src.seam.style.fill_color = RED
    flush()
    assert link.state == "LIVE" and at(link, (0, 0, 100, 100), 50, 50) == srgb8(RED)
    # a vertex group / extra UV map on the target is fine as well
    cube.vertex_groups.new(name="g")
    cube.data.uv_layers.new(name="Other")
    src.seam.style.fill_color = BLUE
    flush()
    assert link.state == "LIVE" and at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE)


def test_material_decal_in_edit_mode_still_updates_the_image(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    cube = add_mesh_with_uv()
    link = make_decal_pair(src, cube)
    activate(cube)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        src.seam.style.fill_color = BLUE
        flush()
        assert link.state == "LIVE" and at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def make_decal_pair(src, mesh, **params):
    select(src, mesh)
    res = bpy.ops.seam.make_3d(recipe="DECAL", decal_mode="MATERIAL", decal_resolution=64, **params)
    assert res == {"FINISHED"}, res
    return bpy.context.scene.seam.links[-1]


def test_material_decal_requirements_are_explained(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    # no mesh selected
    activate(src)
    with pytest.raises(RuntimeError, match="selected mesh"):
        bpy.ops.seam.make_3d(recipe="DECAL", decal_mode="MATERIAL")
    assert not len(bpy.context.scene.seam.links)
    # mesh without UVs
    bpy.ops.mesh.primitive_cube_add(location=(4, 0, 0))
    bare = bpy.context.active_object
    while len(bare.data.uv_layers):
        bare.data.uv_layers.remove(bare.data.uv_layers[0])
    select(src, bare)
    with pytest.raises(RuntimeError, match="no UV map"):
        bpy.ops.seam.make_3d(recipe="DECAL", decal_mode="MATERIAL")
    assert len(bare.data.materials) == 0 and not len(bpy.context.scene.seam.links)
    # a 3D result cannot be the artwork, a mesh alone cannot be the source
    select(bare)
    assert not bpy.ops.seam.make_3d.poll()
    # target explicitly by name works without selecting it
    cube = add_mesh_with_uv("Named", (8, 0, 0))
    activate(src)
    assert bpy.ops.seam.make_3d(recipe="DECAL", decal_mode="MATERIAL", target_name=cube.name) == {"FINISHED"}
    assert bpy.context.scene.seam.links[-1].target == cube


def test_material_decal_remove_and_detach_never_delete_the_mesh(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    cube = add_mesh_with_uv()
    link = make_decal_pair(src, cube)
    assert len(cube.data.materials) == 1 and link.target == cube
    # branch / overwrite / duplicate make no sense for a link that does not own the mesh
    activate(cube)
    assert not bpy.ops.seam.link_overwrite.poll()
    assert not bpy.ops.seam.link_branch.poll()
    assert not bpy.ops.seam.duplicate_result.poll()
    assert bpy.ops.seam.link_remove(delete_result=True) == {"FINISHED"}
    assert bpy.data.objects.get(cube.name) == cube        # the user's mesh survives delete_result
    assert len(cube.data.materials) == 0                   # the SEAM slot is taken away again
    assert not len(bpy.context.scene.seam.links)

    link = make_decal_pair(src, cube)
    activate(cube)
    assert bpy.ops.seam.link_detach() == {"FINISHED"}
    assert link.state == "DETACHED" and len(cube.data.materials) == 1
    assert cube.data.materials[0].get("seam_link_id") == ""   # now an ordinary material
    src.seam.style.fill_color = BLUE
    flush()
    assert link.decal_image is not None and at(link, (0, 0, 100, 100), 50, 50) == srgb8(RED)


def test_material_decal_self_heals_a_deleted_slot_and_follows_source_deletion(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    cube = add_mesh_with_uv()
    link = make_decal_pair(src, cube)
    cube.data.materials.pop(index=0)
    src.seam.style.fill_color = BLUE
    flush()
    assert [m.get("seam_link_id") for m in cube.data.materials] == [link.id]
    bpy.data.objects.remove(src, do_unlink=True)
    flush()
    assert link.state == "MISSING_SOURCE" and cube.data.materials[0] is not None


# ------------------------------------------------------------------ persistence / undo

def test_decal_survives_save_and_reopen_without_rerender(engine, blend_path):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    link = make_decal(src, "PLANE", decal_resolution=96)
    lid = link.id
    before = px(link.decal_image).copy()
    scene = save_and_reload(blend_path)
    from seam_studio.engine import core
    link = next(lk for lk in scene.seam.links if lk.id == lid)
    img = link.decal_image
    assert img is not None and img.packed_file is not None and tuple(img.size) == (96, 96)
    assert np.array_equal(px(img), before)
    assert link.target.data.materials[0].node_tree.nodes["SEAM Decal Image"].image == img
    n = link.build_count
    core.request_revalidate()
    flush()
    assert link.build_count == n and link.state == "LIVE"      # packed + signature match: nothing re-rendered
    # and the link is alive: editing the source updates the very same image
    scene.objects["Art"].seam.style.fill_color = BLUE
    flush()
    assert link.build_count == n + 1 and link.decal_image == img
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE)
    # second round trip keeps the new pixels
    scene = save_and_reload(blend_path)
    link = next(lk for lk in scene.seam.links if lk.id == lid)
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE)


def test_raster_decal_survives_save_and_reopen(rdoc, engine, blend_path):
    flush()
    paint(rdoc.layers[0], solid(4, 8, (200, 30, 20, 255)))
    flush()
    link = make_decal(rdoc.canvas, "PLANE")
    lid = link.id
    scene = save_and_reload(blend_path)
    flush()
    link = next(lk for lk in scene.seam.links if lk.id == lid)
    doc = scene.seam.rasters[0]
    assert link.state == "LIVE" and link.decal_image == doc.composite
    assert tuple(px(link.decal_image)[1, 2]) == (200, 30, 20, 255)


def test_material_decal_survives_save_and_reopen(engine, blend_path):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    cube = add_mesh_with_uv()
    link = make_decal_pair(src, cube)
    lid = link.id
    scene = save_and_reload(blend_path)
    link = next(lk for lk in scene.seam.links if lk.id == lid)
    assert link.target.data.materials[0].node_tree.nodes["SEAM Decal Image"].image == link.decal_image
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(RED)
    scene.objects["Art"].seam.style.fill_color = BLUE
    flush()
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE) and link.state == "LIVE"


def test_decal_pixels_follow_undo_and_redo(engine):
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    link = make_decal(src, "PLANE", decal_resolution=64)
    lid = link.id
    bpy.ops.ed.undo_push(message="created")
    src.seam.style.fill_color = BLUE
    flush()
    bpy.ops.ed.undo_push(message="blue")
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE)
    bpy.ops.ed.undo()
    flush()
    link = next(lk for lk in bpy.context.scene.seam.links if lk.id == lid)
    assert tuple(bpy.data.objects["Art"].seam.style.fill_color) == pytest.approx(RED, abs=1e-6)
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(RED)   # derived again from the restored source
    bpy.ops.ed.redo()
    flush()
    link = next(lk for lk in bpy.context.scene.seam.links if lk.id == lid)
    assert at(link, (0, 0, 100, 100), 50, 50) == srgb8(BLUE)


# ------------------------------------------------------------------ operator / panel surface

def test_decal_panel_rows_and_poll_messages(engine):
    from test_link_ops_ui import draw
    from seam_studio.link import ui
    src = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100, color=RED)
    activate(src)
    log = draw(ui.SEAM_PT_make_3d)
    ops = {e[2]: e[3] for e in log if e[0] == "op"}
    assert ops["Decal (Plane)"] is True and ops["Decal (Material)"] is False
    labels = " ".join(e[1] for e in log if e[0] == "label")
    assert "selected mesh" in labels
    cube = add_mesh_with_uv()
    select(src, cube)
    log = draw(ui.SEAM_PT_make_3d)
    ops = {e[2]: e[3] for e in log if e[0] == "op"}
    assert ops["Decal (Material)"] is True and ops["Decal (Plane)"] is True
    # the operator itself explains why it is unavailable for a 3D result / without selection
    bpy.context.view_layer.objects.active = None
    assert not bpy.ops.seam.make_3d.poll()
    link = make_decal(src, "PLANE", decal_resolution=32)
    activate(link.target)
    assert not bpy.ops.seam.make_3d.poll()
    log = draw(ui.SEAM_PT_live_link)
    props = [e[1] for e in log if e[0] == "prop"]
    assert "decal_resolution" in props and "units_scale" in props
    assert any(e[0] == "op" and e[1] == "seam.result_paint_setup" for e in log)
    assert any(e[0] == "label" and "Plane" in e[1] for e in log)


def test_decal_source_roles_and_group_ancestors_in_ui(engine):
    from seam_studio.link import recipes
    from seam_studio.seam_core import links as L
    shape = add_rect(bpy.context.scene, "Art", 0, 0, 100, 100)
    assert recipes.source_ok("DECAL", shape) == (True, "")
    assert recipes.recipe_available("DECAL") == (True, "")
    assert "DECAL" in recipes.EXPOSED_RECIPES
    assert L.can_link("DECAL", "GROUP")[0] and L.can_link("DECAL", "RASTER")[0]
    link = make_decal(shape, "PLANE", decal_resolution=32)
    ok, why = recipes.source_ok("DECAL", link.target)
    assert not ok and "cannot be used as a source" in why


def test_duplicate_and_relink_decal(engine):
    a = add_rect(bpy.context.scene, "A", 0, 0, 100, 100, color=RED)
    b = add_rect(bpy.context.scene, "B", 300, 0, 400, 50, color=BLUE)
    link = make_decal(a, "PLANE", decal_resolution=64)
    activate(link.target)
    assert bpy.ops.seam.duplicate_result() == {"FINISHED"}
    links = bpy.context.scene.seam.links
    assert len(links) == 2 and links[1].recipe == "DECAL" and links[1].decal_image != links[0].decal_image
    assert at(links[1], (0, 0, 100, 100), 50, 50) == srgb8(RED)
    activate(link.target)
    assert bpy.ops.seam.link_relink(source_name=b.name) == {"FINISHED"}
    assert link.source == b and tuple(link.decal_image.size) == (64, 32)
    assert at(link, (300, 0, 400, 50), 350, 25) == srgb8(BLUE)
