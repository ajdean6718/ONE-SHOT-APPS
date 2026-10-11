"""Scenario 3 - an independent native 3D object.

A plain Blender cube with a Bevel modifier and a material, lit and viewed through a camera, rendered with
Cycles (CPU, few samples) and Workbench to PNG files.  SEAM's canvas and artwork live in the same file and
must neither appear in the render nor change a single pixel of it.  All edits are native operators.
"""

import os

import bpy
import numpy as np
import pytest

from accsupport import flush, new_canvas, op, save_and_reopen, select, srgb8

RES = (96, 64)
BACKGROUND = (0.05, 0.1, 0.3)             # scene-linear world colour -> sRGB (63, 89, 149) with the Standard view


def _read_png(path):
    from seam_studio.seam_core import png
    return png.read_png(str(path))


def use_scene_collection():
    """What a user does in the Outliner before modelling: click the Scene Collection (not a SEAM layer)."""
    vl = bpy.context.view_layer
    vl.active_layer_collection = vl.layer_collection
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)


def build_native_scene():
    """Cube + bevel + material + sun + camera, all through Blender operators."""
    scene = bpy.context.scene
    assert bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0)) == {"FINISHED"}
    cube = bpy.context.active_object
    cube.name = "Native Cube"
    assert bpy.ops.object.modifier_add(type="BEVEL") == {"FINISHED"}
    bev = cube.modifiers[-1]
    bev.width = 0.15
    bev.segments = 3
    mat = bpy.data.materials.new("Native Red")
    mat.use_nodes = True
    mat.diffuse_color = (0.9, 0.05, 0.05, 1.0)           # viewport / Workbench colour
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.9, 0.05, 0.05, 1.0)
    assert bpy.ops.object.material_slot_add() == {"FINISHED"}
    cube.material_slots[0].material = mat
    assert bpy.ops.object.light_add(type="SUN", location=(3.0, -4.0, 6.0), rotation=(0.8, 0.0, 0.6)) == {"FINISHED"}
    bpy.context.active_object.data.energy = 4.0
    assert bpy.ops.object.camera_add(location=(0.0, -7.0, 2.5), rotation=(1.2, 0.0, 0.0)) == {"FINISHED"}
    scene.camera = bpy.context.active_object
    world = bpy.data.worlds.new("Native World")
    world.color = BACKGROUND
    if world.node_tree is None:
        world.use_nodes = True
    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            node.inputs["Color"].default_value = (*BACKGROUND, 1.0)
            node.inputs["Strength"].default_value = 1.0
    scene.world = world
    scene.render.resolution_x, scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    return cube


def render(path, engine):
    scene = bpy.context.scene
    scene.render.engine = engine
    if engine == "CYCLES":
        cy = scene.cycles
        cy.device = "CPU"
        cy.samples = 8
        cy.use_denoising = False
        cy.max_bounces = 2
    scene.render.filepath = str(path)
    assert bpy.ops.render.render(write_still=True) == {"FINISHED"}
    assert os.path.getsize(path) > 100
    return _read_png(path)


def evaluated_counts(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        return len(me.vertices), len(me.polygons)
    finally:
        ev.to_mesh_clear()


def assert_cube_visible(img, red_dominant=True):
    h, w = img.shape[:2]
    assert (h, w) == (RES[1], RES[0])
    rgb = img[..., :3].astype(np.int32)
    corner = rgb[:4, :4].reshape(-1, 3)
    centre = rgb[h // 2 - 6:h // 2 + 6, w // 2 - 6:w // 2 + 6].reshape(-1, 3)
    assert img[..., 3].min() == 255                                            # opaque background render
    assert tuple(corner.mean(axis=0).round()) == pytest.approx(srgb8(BACKGROUND), abs=2)   # the world colour
    assert rgb.std() > 8.0                                                    # not a flat image
    assert np.abs(centre.mean(axis=0) - corner.mean(axis=0)).max() > 40       # the cube differs from the background
    if red_dominant:
        r, g, b = centre.mean(axis=0)
        assert r > 2.0 * g and r > 2.0 * b and r > 60                         # a red material, lit
    return centre.mean(axis=0)


def test_native_modifier_material_and_both_renders(seam, engine, tmp_path):
    new_canvas(800, 600)
    op(bpy.ops.seam.add_shape, kind="RECT", x=400, y=-300, width=200, height=100)  # SEAM artwork in the same file
    flush()
    use_scene_collection()
    cube = build_native_scene()
    # the modifier is real: evaluated geometry has the bevel, the original mesh is untouched
    assert len(cube.data.vertices) == 8 and len(cube.data.polygons) == 6
    nv, nf = evaluated_counts(cube)
    assert nv > 8 and nf > 6                                                   # bevel with 3 segments adds geometry
    assert cube.modifiers[0].type == "BEVEL" and cube.modifiers[0].segments == 3
    assert cube.material_slots[0].material.name == "Native Red"
    # the cube is not a SEAM object: no role, no link, never touched by the engine
    assert cube.seam.role == "NONE" and len(bpy.context.scene.seam.links) == 0
    before = (tuple(cube.location), nv)
    flush()
    assert (tuple(cube.location), evaluated_counts(cube)[0]) == before

    wb = render(tmp_path / "workbench.png", "BLENDER_WORKBENCH")
    wb_centre = assert_cube_visible(wb)
    cy = render(tmp_path / "cycles.png", "CYCLES")
    cy_centre = assert_cube_visible(cy)
    # the two engines agree on what is where: the cube silhouette overlaps strongly
    wb_mask = np.abs(wb[..., :3].astype(np.int32) - wb[:2, :2, :3].mean(axis=(0, 1)).astype(np.int32)).max(axis=-1) > 30
    cy_mask = np.abs(cy[..., :3].astype(np.int32) - cy[:2, :2, :3].mean(axis=(0, 1)).astype(np.int32)).max(axis=-1) > 30
    inter, union = (wb_mask & cy_mask).sum(), (wb_mask | cy_mask).sum()
    assert wb_mask.sum() > 300 and cy_mask.sum() > 300 and inter / union > 0.6
    assert wb_centre.shape == cy_centre.shape == (3,)


def test_seam_content_never_reaches_the_render(seam, engine, tmp_path):
    new_canvas(800, 600)
    for i, (x, y) in enumerate([(400, -300), (200, -150), (600, -450)]):
        op(bpy.ops.seam.add_shape, kind="ELLIPSE", x=x, y=y, width=120, height=120)
        bpy.context.active_object.seam.style.fill_color = (0.0, 1.0, 0.0, 1.0)
    op(bpy.ops.seam.add_text, body="Not in 3D", x=100, y=-500, size=64)
    op(bpy.ops.seam.raster_new, name="Paint", width=32, height=32, background="WHITE")
    flush()
    use_scene_collection()
    cube = build_native_scene()
    with_seam = render(tmp_path / "with.png", "BLENDER_WORKBENCH")
    assert_cube_visible(with_seam)
    # delete every SEAM object from the file: Workbench output is bit-identical
    removed = [o for o in list(bpy.data.objects) if o != cube and o.type not in {"LIGHT", "CAMERA"}]
    assert len(removed) >= 7
    for o in removed:
        bpy.data.objects.remove(o, do_unlink=True)
    assert [o.name for o in bpy.context.scene.objects if o.type in {"MESH", "CURVE", "FONT"}] == [cube.name]
    without = render(tmp_path / "without.png", "BLENDER_WORKBENCH")
    assert np.array_equal(with_seam, without)


def test_native_scene_survives_save_reopen_and_edit(seam, engine, tmp_path, blend_path):
    new_canvas(800, 600)
    use_scene_collection()
    cube = build_native_scene()
    first = render(tmp_path / "a.png", "BLENDER_WORKBENCH")
    save_and_reopen(blend_path)
    flush()
    cube = bpy.data.objects["Native Cube"]
    assert [m.type for m in cube.modifiers] == ["BEVEL"] and cube.modifiers[0].width == pytest.approx(0.15)
    assert cube.material_slots[0].material.name == "Native Red" and bpy.context.scene.camera is not None
    again = render(tmp_path / "b.png", "BLENDER_WORKBENCH")
    assert np.array_equal(first, again)                                          # deterministic after reopen
    # native edit: move the cube in Edit-Mode-free object mode with G, the render follows
    select(cube)
    assert bpy.ops.transform.translate(value=(2.0, 0.0, 0.0)) == {"FINISHED"}
    assert cube.location.x == pytest.approx(2.0)
    moved = render(tmp_path / "c.png", "BLENDER_WORKBENCH")
    cols_first = np.nonzero((np.abs(first[..., :3].astype(int) - first[:2, :2, :3].mean(axis=(0, 1)).astype(int)).max(-1) > 30).any(axis=0))[0]
    cols_moved = np.nonzero((np.abs(moved[..., :3].astype(int) - moved[:2, :2, :3].mean(axis=(0, 1)).astype(int)).max(-1) > 30).any(axis=0))[0]
    assert cols_moved.mean() > cols_first.mean() + 8                             # shifted right in the image
    # native modifier removal changes the evaluated mesh back to 8 vertices
    assert bpy.ops.object.modifier_remove(modifier="Bevel") == {"FINISHED"}
    assert evaluated_counts(cube) == (8, 6)


@pytest.mark.xfail(strict=True, reason="BUG: after Canvas New the active collection stays a SEAM layer (hide_render) and the "
                   "3D cursor sits on the canvas, so a cube added the obvious way is neither visible nor rendered. "
                   "Fix belongs to ui/workspaces (Model workspace should activate a render-visible collection and reset "
                   "the cursor). See REQUESTS.")
def test_cube_added_without_touching_the_outliner_is_rendered(seam, engine, tmp_path):
    new_canvas(800, 600)
    cube = build_native_scene()                                                 # no use_scene_collection()
    assert not any(c.hide_render or c.name.startswith("SEAM") for c in cube.users_collection)
    assert_cube_visible(render(tmp_path / "default.png", "BLENDER_WORKBENCH"))
