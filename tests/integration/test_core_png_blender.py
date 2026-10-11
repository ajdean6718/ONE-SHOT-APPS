"""seam_core.png decodes what Blender writes, on both runtimes (4.0.2 binary and the 5.2.2 bpy module)."""

import numpy as np
import pytest

import bpy

from seam_core import png, raster


def _make_image(name, w, h, alpha=True):
    img = bpy.data.images.new(name, w, h, alpha=alpha)
    img.alpha_mode = "STRAIGHT"
    return img


def _save(img, path, mode="RGBA"):
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()


def _rng_pixels(w, h, seed, alpha=True):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
    if not alpha:
        a[..., 3] = 255
    return a


def _set_pixels(img, rgba8):
    """rgba8 H x W x 4 uint8 (row 0 = top) -> image pixels, using the SEAM helper for Blender's row order."""
    arr = rgba8.astype(np.float32) / 255.0
    img.pixels.foreach_set(raster.array_to_blender_pixels(arr))
    img.update()


def test_decode_blender_saved_png_rgba_exact(tmp_path):
    w, h = 37, 23
    ref = _rng_pixels(w, h, 1)
    img = _make_image("png_rgba", w, h)
    _set_pixels(img, ref)
    path = str(tmp_path / "rgba.png")
    _save(img, path)
    out = png.read_png(path)
    assert out.shape == (h, w, 4) and out.dtype == np.uint8
    assert np.array_equal(out, ref)


def test_decode_blender_saved_png_opaque_rgb(tmp_path):
    w, h = 16, 9
    ref = _rng_pixels(w, h, 2, alpha=False)
    img = _make_image("png_rgb", w, h, alpha=False)
    _set_pixels(img, ref)
    path = str(tmp_path / "rgb.png")
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    out = png.read_png(path)
    assert out.shape == (h, w, 4)
    assert np.array_equal(out[..., :3], ref[..., :3]) and np.all(out[..., 3] == 255)


def test_blender_png_pixel_order_matches_helpers(tmp_path):
    """Top row of the file is the TOP row of the image; pixels round-trip through the Blender order helpers."""
    w, h = 5, 4
    ref = np.zeros((h, w, 4), dtype=np.uint8)
    ref[0, :, 0] = 255          # top row red
    ref[-1, :, 2] = 255         # bottom row blue
    ref[..., 3] = 255
    img = _make_image("png_order", w, h)
    _set_pixels(img, ref)
    path = str(tmp_path / "order.png")
    _save(img, path)
    out = png.read_png(path)
    assert np.array_equal(out, ref)
    back = raster.blender_pixels_to_array(np.asarray(img.pixels[:]), w, h)
    assert np.array_equal(np.rint(back * 255).astype(np.uint8), ref)


def test_decode_blender_png_straight_alpha_values(tmp_path):
    w, h = 8, 8
    ref = np.zeros((h, w, 4), dtype=np.uint8)
    ref[..., 0] = np.arange(64).reshape(8, 8) * 4
    ref[..., 1] = 255 - ref[..., 0]
    ref[..., 2] = 77
    ref[..., 3] = np.arange(64).reshape(8, 8) * 4 + 3
    img = _make_image("png_alpha", w, h)
    _set_pixels(img, ref)
    path = str(tmp_path / "alpha.png")
    _save(img, path)
    out = png.read_png(path)
    assert np.array_equal(out, ref)


def test_decode_blender_saved_png_larger_gradient_uses_filters(tmp_path):
    # smooth gradient pushes libpng to pick Sub/Up/Average/Paeth filters
    w, h = 96, 64
    yy, xx = np.mgrid[0:h, 0:w]
    ref = np.stack([xx * 255 // (w - 1), yy * 255 // (h - 1), (xx + yy) * 255 // (w + h - 2), np.full((h, w), 255)], axis=-1).astype(np.uint8)
    img = _make_image("png_grad", w, h)
    _set_pixels(img, ref)
    path = str(tmp_path / "grad.png")
    _save(img, path)
    data = open(path, "rb").read()
    out = png.decode_png(data)
    assert np.array_equal(out, ref)


def test_decode_blender_16bit_png(tmp_path):
    """Blender can write 16-bit PNGs; decode_png downconverts them to 8 bit."""
    w, h = 12, 6
    ref = _rng_pixels(w, h, 5)
    img = _make_image("png16", w, h)
    _set_pixels(img, ref)
    scene = bpy.context.scene
    s = scene.render.image_settings
    s.file_format = "PNG"
    s.color_mode = "RGBA"
    s.color_depth = "16"
    path = str(tmp_path / "deep.png")
    img.save_render(path, scene=scene)
    out = png.read_png(path)
    assert out.shape == (h, w, 4)
    # save_render applies the scene view transform; with Standard on sRGB data it is the identity within 1 level
    scene.view_settings.view_transform = "Standard"
    img.save_render(path, scene=scene)
    out = png.read_png(path)
    assert np.max(np.abs(out.astype(int) - ref.astype(int))) <= 1


def test_encode_png_is_readable_by_blender(tmp_path):
    w, h = 21, 13
    ref = _rng_pixels(w, h, 7)
    path = str(tmp_path / "ours.png")
    png.write_png(path, ref)
    img = bpy.data.images.load(path)
    try:
        assert tuple(img.size) == (w, h)
        img.alpha_mode = "STRAIGHT"
        got = raster.blender_pixels_to_array(np.asarray(img.pixels[:]), w, h)
        got8 = np.rint(got * 255).astype(np.uint8)
        # alpha-premultiplied loading can disturb low-alpha colours: compare only where alpha is high
        mask = ref[..., 3] > 200
        assert np.max(np.abs(got8[mask].astype(int) - ref[mask].astype(int))) <= 1
        assert np.array_equal(got8[..., 3], ref[..., 3])
    finally:
        bpy.data.images.remove(img)
