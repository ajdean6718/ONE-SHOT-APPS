"""Paint: the canvas material (node tree) renders like seam_core.raster.composite (Cycles CPU, tiny canvases)."""

import numpy as np
import pytest

import bpy

from test_paint_support import flush, kernel, new_doc, noise, paint_pixels, rd  # noqa: F401  (kernel = fixture)


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


@pytest.mark.parametrize("mode", ["MULTIPLY", "SCREEN", "OVERLAY", "ADD"])
@pytest.mark.parametrize("bg", ["TRANSPARENT", "COLOR"])
def test_partial_alpha_stacks_with_blend_modes_match_composite(seam, kernel, tmp_path, mode, bg):
    """Independent parity evidence (Cycles vs the numpy compositor) for partly transparent lower layers."""
    r = rd()
    doc = fresh_doc(6, 4, f"PA{mode}{bg}", bg)
    if bg == "COLOR":
        doc.background_color = (0.3, 0.6, 0.2, 0.5)
    r.add_layer(doc, "Mid")
    r.add_layer(doc, "Top")
    for i, ly in enumerate(doc.layers):
        px = noise(4, 6, 300 + i)
        px[..., 3] = np.random.default_rng(310 + i).integers(192, 256, (4, 6))  # partial, above the 8-bit premultiply range
        paint_pixels(ly, px)
    doc.layers[1].blend_mode = mode
    doc.layers[1].opacity = 0.8
    doc.layers[2].blend_mode = mode
    doc.layers[2].opacity = 0.7
    # looser than the opaque cases: Blender stores 8-bit textures alpha-premultiplied, which costs ~1 extra bit at alpha 0.75
    assert_matches(doc, tmp_path, atol=0.02)


def assert_matches_8bit(doc, tmp_path, tol=2, min_alpha=0.05):
    """Cycles render of the canvas material vs ``composite_array``, both encoded to 8-bit sRGB: within ``tol``/255."""
    from seam_studio.seam_core import raster as R
    flush()
    got = render_canvas(doc, tmp_path)
    want = rd().composite_array(doc)
    g8 = R.to_uint8_srgb(np.concatenate([np.clip(got[..., :3], 0.0, 1.0), np.clip(got[..., 3:], 0.0, 1.0)], -1)).astype(int)
    w8 = R.to_uint8_srgb(np.concatenate([np.clip(want[..., :3], 0.0, 1.0), want[..., 3:]], -1)).astype(int)
    # alpha is stored linearly: compare it as a plain 8-bit value
    ga = np.rint(np.clip(got[..., 3], 0.0, 1.0) * 255.0).astype(int)
    wa = np.rint(want[..., 3] * 255.0).astype(int)
    assert np.abs(ga - wa).max() <= tol, np.abs(ga - wa).max()
    vis = want[..., 3] > min_alpha
    d = np.abs(g8[..., :3] - w8[..., :3])[vis]
    assert d.size == 0 or d.max() <= tol, (d.max(), np.argwhere(vis)[int(d.reshape(-1, 3).max(axis=1).argmax())])


def semi_transparent(h, w, seed, lo=64, hi=224):
    """Noise with every pixel partly transparent (alpha in [lo, hi])."""
    px = noise(h, w, seed)
    px[..., 3] = np.random.default_rng(seed + 1).integers(lo, hi + 1, (h, w))
    return px


def opaque_pixels(h, w, seed):
    px = noise(h, w, seed)
    px[..., 3] = 255
    return px


def test_semi_transparent_red_blue_matches_half_half_over(seam, kernel, tmp_path):
    """The headline case: 50% red under 50% blue renders as 1/3 red + 2/3 blue at alpha 0.75 (not 0.5 / 0.5)."""
    r = rd()
    doc = fresh_doc(5, 4, "RB")
    r.add_layer(doc, "Top")
    red = np.zeros((4, 5, 4), np.uint8)
    red[...] = (255, 0, 0, 255)
    blue = np.zeros((4, 5, 4), np.uint8)
    blue[...] = (0, 0, 255, 255)
    paint_pixels(doc.layers[0], red)
    paint_pixels(doc.layers[1], blue)
    doc.layers[0].opacity = 0.5
    doc.layers[1].opacity = 0.5
    flush()
    want = rd().composite_array(doc)
    assert np.allclose(want[0, 0], [1.0 / 3.0, 0.0, 2.0 / 3.0, 0.75], atol=1e-6)
    got = render_canvas(doc, tmp_path)
    assert np.allclose(got[0, 0], want[0, 0], atol=2e-3), (got[0, 0], want[0, 0])
    assert_matches_8bit(doc, tmp_path)


@pytest.mark.parametrize("mode", ["NORMAL", "MULTIPLY", "SCREEN", "OVERLAY", "ADD"])
@pytest.mark.parametrize("bg", ["TRANSPARENT", "COLOR"])
def test_semi_transparent_layers_match_composite_8bit(seam, kernel, tmp_path, mode, bg):
    """Acceptance: viewport (Cycles render of the canvas material) == export compositor within 2/255.

    Pixels are opaque (exactly representable in Blender's premultiplied 8-bit texture storage); the layers are
    semi-transparent through opacity and an 8-bit mask, so every backdrop alpha in (0, 1) is exercised.
    """
    r = rd()
    doc = fresh_doc(6, 4, f"ST{mode}{bg}", bg)
    if bg == "COLOR":
        doc.background_color = (0.3, 0.6, 0.2, 0.5)
    r.add_layer(doc, "Mid")
    r.add_layer(doc, "Top")
    for i, ly in enumerate(doc.layers):
        paint_pixels(ly, opaque_pixels(4, 6, 500 + 7 * i))
    r.add_mask(doc, 1, "WHITE")
    r.add_mask(doc, 2, "WHITE")
    for i in (1, 2):
        m = np.random.default_rng(40 + i).integers(40, 256, (4, 6), dtype=np.uint8)
        paint_pixels(doc.layers[i], np.stack([m, m, m, np.full_like(m, 255)], -1), mask=True)
    doc.layers[0].opacity = 0.7
    doc.layers[1].blend_mode = mode
    doc.layers[1].opacity = 0.9
    doc.layers[2].blend_mode = mode
    doc.layers[2].opacity = 0.8
    assert_matches_8bit(doc, tmp_path)


@pytest.mark.parametrize("mode", ["NORMAL", "MULTIPLY", "SCREEN", "OVERLAY", "ADD"])
def test_pixel_alpha_layers_match_composite_8bit(seam, kernel, tmp_path, mode):
    """Semi-transparent *pixels* (alpha channel) on transparent layers: the anti-aliased-edge case.

    Blender keeps 8-bit textures alpha-premultiplied, so the colour of a pixel with alpha a is only known to
    about 1/(255 a): the texture node alone is off by up to 3/255 at alpha 0.25.  Alpha is therefore kept in
    [0.75, 0.88] here (2/255 for the linear modes; OVERLAY's slope amplifies the storage error to 4/255).
    """
    r = rd()
    doc = fresh_doc(6, 4, f"PX{mode}")
    r.add_layer(doc, "Mid")
    r.add_layer(doc, "Top")
    for i, ly in enumerate(doc.layers):
        paint_pixels(ly, semi_transparent(4, 6, 600 + 7 * i, lo=192, hi=224))
    for i in (1, 2):
        doc.layers[i].blend_mode = mode
    doc.layers[1].opacity = 0.9
    doc.layers[2].opacity = 0.8
    assert_matches_8bit(doc, tmp_path, tol=4 if mode == "OVERLAY" else 2)
