"""``seam_core.raster.blend_rgb`` agrees with Blender's own Mix node (rendered by Cycles) for every BLEND_MODE.

An emission plane fills an orthographic camera; two Non-Color float images (base = Mix input 1, layer = Mix input 2)
supply one colour pair per pixel through Closest-interpolated texture nodes, so each rendered pixel is
``Mix(blend_type, fac=1, base, layer)``. The Standard view transform + disabled dither make the 8-bit PNG output the plain
sRGB encoding of the linear Mix result, which must match ``to_uint8_srgb(blend_rgb(...))`` within 2/255.
"""

import numpy as np
import pytest

import bpy

from seam_core import png, raster

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

W, H = 32, 16  # left 16x16: palette cross product, right 16x16: random pairs

PALETTE = [  # sRGB 8-bit
    (0, 0, 0), (255, 255, 255), (128, 128, 128), (64, 64, 64), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (255, 140, 0), (224, 172, 105), (20, 30, 90), (180, 150, 230), (128, 128, 0), (0, 128, 128),
]


def _pair_grid():
    rng = np.random.default_rng(1234)
    base = np.zeros((H, W, 3), dtype=np.uint8)
    layer = np.zeros((H, W, 3), dtype=np.uint8)
    pal = np.array(PALETTE, dtype=np.uint8)
    for y in range(16):
        for x in range(16):
            base[y, x] = pal[y]
            layer[y, x] = pal[x]
    base[:, 16:] = rng.integers(0, 256, size=(H, 16, 3), dtype=np.uint8)
    layer[:, 16:] = rng.integers(0, 256, size=(H, 16, 3), dtype=np.uint8)
    return base, layer


def _linear_image(name, rgb8):
    lin = raster.srgb_to_linear(rgb8.astype(np.float64) / 255.0).astype(np.float32)
    rgba = np.concatenate([lin, np.ones((H, W, 1), np.float32)], axis=-1)
    img = bpy.data.images.new(name, W, H, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(raster.array_to_blender_pixels(rgba))
    img.update()
    return img, lin


def _build_scene(mix_type, samples, fac=1.0):
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.device = "CPU"
    cy.samples = samples
    cy.use_denoising = False
    cy.pixel_filter_type = "BOX"
    cy.filter_width = 1.0
    cy.max_bounces = 0
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.dither_intensity = 0.0
    scene.render.use_motion_blur = False
    vs = scene.view_settings
    vs.view_transform = "Standard"
    vs.look = "None"
    vs.exposure = 0.0
    vs.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    ims = scene.render.image_settings
    ims.file_format = "PNG"
    ims.color_mode = "RGB"
    ims.color_depth = "8"

    # camera
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.0
    cam_data.sensor_fit = "VERTICAL"
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = (0.0, 0.0, 5.0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # plane that exactly fills the frame in X and Y (frame is 2:1 so make the plane 4 x 2)
    mesh = bpy.data.meshes.new("plane")
    mesh.from_pydata([(-2, -1, 0), (2, -1, 0), (2, 1, 0), (-2, 1, 0)], [], [(0, 1, 2, 3)])
    uv = mesh.uv_layers.new(name="UVMap")
    for loop, co in zip(mesh.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[loop.index].uv = co
    obj = bpy.data.objects.new("plane", mesh)
    scene.collection.objects.link(obj)

    base8, layer8 = _pair_grid()
    img_a, lin_a = _linear_image("base", base8)
    img_b, lin_b = _linear_image("layer", layer8)

    mat = bpy.data.materials.new("mix")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = mix_type
    mix.use_clamp = False
    mix.inputs["Fac"].default_value = fac
    for nm, im, key in (("A", img_a, "Color1"), ("B", img_b, "Color2")):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = im
        tex.interpolation = "Closest"
        tex.extension = "EXTEND"
        nt.links.new(tex.outputs["Color"], mix.inputs[key])
    nt.links.new(mix.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return scene, lin_a, lin_b


def _render_pixels(scene, tmp_path):
    path = str(tmp_path / "mix.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return png.read_png(path)


def _check(mode, samples, tmp_path):
    scene, lin_a, lin_b = _build_scene(raster.BLENDER_MIX_TYPE[mode], samples)
    got = _render_pixels(scene, tmp_path)
    assert got.shape == (H, W, 4)
    expected = raster.to_uint8_srgb(np.clip(raster.blend_rgb(mode, lin_a, lin_b), 0.0, 1.0))
    diff = np.abs(got[..., :3].astype(int) - expected.astype(int))
    worst = int(diff.max())
    if worst > 2:
        y, x, c = np.unravel_index(int(diff.argmax()), diff.shape)
        raise AssertionError("%s samples=%d: max diff %d/255 at (%d,%d) ch%d: blender=%s ours=%s base_lin=%s layer_lin=%s" % (
            mode, samples, worst, x, y, c, got[y, x, :3], expected[y, x], lin_a[y, x], lin_b[y, x]))
    # the render is not trivially black/flat: it carries the pair variation
    assert len(np.unique(got[..., :3].reshape(-1, 3), axis=0)) > 10 or mode in ("DARKEN",)
    return worst


@pytest.mark.parametrize("mode", raster.BLEND_MODES)
def test_blend_rgb_matches_blender_mix_node(mode, tmp_path):
    samples = 1 + (raster.BLEND_MODES.index(mode) % 4)  # exercise 1..4 samples across the modes
    _check(mode, samples, tmp_path)


def test_mix_type_names_are_valid_blender_enums():
    nt = bpy.data.node_groups.new("probe", "ShaderNodeTree")
    mix = nt.nodes.new("ShaderNodeMixRGB")
    valid = {i.identifier for i in mix.bl_rna.properties["blend_type"].enum_items}
    for mode, name in raster.BLENDER_MIX_TYPE.items():
        assert name in valid, (mode, name)
    bpy.data.node_groups.remove(nt)


def test_composite_normal_matches_blender_over(tmp_path):
    """composite() NORMAL with alpha*opacity factor equals Mix(MIX, fac) of the colours on an opaque base."""
    scene, lin_a, lin_b = _build_scene("MIX", 1, fac=0.37)
    got = _render_pixels(scene, tmp_path)
    base = np.concatenate([lin_a, np.ones((H, W, 1), np.float32)], axis=-1)
    layer = np.concatenate([lin_b, np.ones((H, W, 1), np.float32)], axis=-1)
    out = raster.composite([raster.LayerSpec(base), raster.LayerSpec(layer, opacity=0.37)], H, W)
    expected = raster.to_uint8_srgb(out)[..., :3]
    assert int(np.abs(got[..., :3].astype(int) - expected.astype(int)).max()) <= 2
