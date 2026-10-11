"""Paint: the canvas material (node tree) renders like seam_core.raster.composite (Cycles CPU, tiny canvases)."""

import os

import numpy as np
import pytest

import bpy

from test_paint_support import flush, kernel, new_doc, noise, paint_pixels, rd, solid  # noqa: F401


def render_canvas(doc, tmp_path):
    """Render the canvas plane top-down 1 canvas px = 1 render px; returns straight linear RGBA (H x W x 4)."""
    from mathutils import Vector
    scene = bpy.context.scene
    w, h = doc.width, doc.height
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.filter_width = 0.01
    scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = float(max(w, h))
    cam_data.clip_start, cam_data.clip_end = 0.1, 1000.0
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    centre = doc.canvas.matrix_world @ Vector((w / 2.0, -h / 2.0, 0.0))
    cam.location = (centre.x, centre.y, centre.z + 10.0)
    scene.camera = cam
    from seam_studio.vector import canvas
    canvas.artwork_collection(scene).hide_render = False
    for ob in scene.objects:  # only the canvas plane is rendered (artboards sit behind it)
        ob.hide_render = ob not in (doc.canvas, cam)
    out = str(tmp_path / "render.exr")
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(out)
    try:
        buf = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(buf)
    finally:
        bpy.data.images.remove(img)
    rgba = buf.reshape(h, w, 4)[::-1].copy()
    a = rgba[..., 3:4]
    rgb = np.where(a > 1e-6, rgba[..., :3] / np.maximum(a, 1e-6), 0.0)
    return np.concatenate([rgb, a], -1)


def has_cycles():
    return hasattr(bpy.types.Scene, "cycles") or "cycles" in getattr(bpy.context.preferences, "addons", {})


def assert_matches(doc, tmp_path, atol=0.012):
    flush()
    got = render_canvas(doc, tmp_path)
    want = rd().composite_array(doc)
    assert got.shape == want.shape
    # the display clamps colours (negative emission is black) and export clips: compare the displayable range
    got = np.concatenate([np.clip(got[..., :3], 0.0, 1.0), got[..., 3:]], -1)
    want = np.concatenate([np.clip(want[..., :3], 0.0, 1.0), want[..., 3:]], -1)
    assert np.abs(got[..., 3] - want[..., 3]).max() < atol, np.abs(got[..., 3] - want[..., 3]).max()
    vis = want[..., 3] > 0.05
    diff = np.abs(got[..., :3] - want[..., :3])[vis]
    assert diff.size == 0 or diff.max() < atol, diff.max()


def fresh_doc(w=6, h=4, name="R", bg="TRANSPARENT"):  # Blender renders need >= 4 px per side
    scene = bpy.context.scene
    if not has_cycles():
        pytest.skip("Cycles unavailable")
    return new_doc(w, h, name, bg)


def test_normal_blend_two_layers_matches_composite(seam, kernel, tmp_path):
    r = rd()
    doc = fresh_doc()
    r.add_layer(doc, "Top")
    a, b = noise(4, 6, 91), noise(4, 6, 92)
    # Blender stores 8-bit textures alpha-premultiplied, so colours under very low alpha lose precision in the
    # node tree (display only; export always uses the numpy compositor): keep alpha away from that range here
    a[..., 3] = np.where(a[..., 3] < 192, 255, a[..., 3])
    b[..., 3] = np.where(b[..., 3] < 192, 200, b[..., 3])
    paint_pixels(doc.layers[0], a)
    paint_pixels(doc.layers[1], b)
    doc.layers[1].opacity = 0.6
    assert_matches(doc, tmp_path)


def all_blend_modes():
    from seam_studio.seam_core import raster as R
    return [m for m in R.BLEND_MODES if m != "NORMAL"]


@pytest.mark.parametrize("mode", ["MULTIPLY", "SCREEN", "ADD", "OVERLAY", "DIFFERENCE", "DARKEN", "LIGHTEN", "SUBTRACT",
                                  "COLOR_DODGE", "COLOR_BURN", "EXCLUSION", "SOFT_LIGHT", "LINEAR_LIGHT", "DIVIDE",
                                  "HUE", "SATURATION", "COLOR", "VALUE"])
def test_blend_modes_match_composite(seam, kernel, tmp_path, mode):
    r = rd()
    doc = fresh_doc(5, 4, "B" + mode, "WHITE")
    r.add_layer(doc, "Mid")
    r.add_layer(doc, "Top")
    for i, ly in enumerate(doc.layers):
        px = noise(4, 5, 100 + i)
        px[..., 3] = 255  # opaque: premultiplied 8-bit storage would amplify rounding in divide / linear light
        paint_pixels(ly, px)
    doc.layers[1].blend_mode = mode
    doc.layers[2].blend_mode = mode
    doc.layers[2].opacity = 0.7
    assert_matches(doc, tmp_path)


def test_mask_visibility_and_background_color_match_composite(seam, kernel, tmp_path):
    r = rd()
    doc = fresh_doc(6, 4, "MB", "COLOR")
    doc.background_color = (0.2, 0.5, 0.8, 0.6)
    r.add_layer(doc, "Top")
    r.add_layer(doc, "Hidden")
    for i, ly in enumerate(doc.layers):
        paint_pixels(ly, noise(4, 6, 120 + i))
    r.add_mask(doc, 1, "WHITE")
    m = np.random.default_rng(7).integers(0, 256, (4, 6), dtype=np.uint8)
    paint_pixels(doc.layers[1], np.stack([m, m, m, np.full_like(m, 255)], -1), mask=True)
    doc.layers[1].blend_mode = "MULTIPLY"
    doc.layers[2].visible = False
    assert_matches(doc, tmp_path)


def test_every_kernel_blend_mode_is_covered_by_the_parity_tests(seam):
    import re
    covered = set(re.findall(r'"([A-Z_]+)"', open(__file__).read().split("@pytest.mark.parametrize")[1].split("def test_blend")[0]))
    assert set(all_blend_modes()) <= covered | {"NORMAL"}
