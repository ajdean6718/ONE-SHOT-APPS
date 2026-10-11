"""Shared helpers for the paint integration tests (not a test module)."""

import numpy as np
import pytest

import bpy


def kernel_ready() -> bool:
    """True when seam_core.raster / seam_core.png are implemented (core-raster-io merged)."""
    from seam_studio.seam_core import png
    from seam_studio.seam_core import raster as R
    try:
        R.srgb_to_linear(np.zeros((1, 1, 4), np.float32))
        R.to_uint8_srgb(np.zeros((1, 1, 4), np.float32))
        R.composite([], 1, 1, None)
        R.rect_mask(2, 2, 0, 0, 1, 1)
        png.decode_png(png.encode_png(np.zeros((1, 1, 4), np.uint8)))
    except NotImplementedError:
        return False
    return True


@pytest.fixture
def kernel(seam):
    """Skip the test while the raster kernel (core-raster-io) is not merged."""
    if not kernel_ready():
        pytest.skip("kernel pending")


def rd():
    from seam_studio.paint import rasterdoc
    return rasterdoc


def new_doc(width=8, height=6, name="Doc", background="TRANSPARENT"):
    scene = bpy.context.scene
    return rd().new_raster_doc(scene, name, width, height, background)


def flush():
    from seam_studio.engine import core
    return core.flush()


def paint_pixels(layer, rgba8, mask=False):
    """Write uint8 pixels (H x W x 4, top row first) without any kernel function."""
    a8 = np.asarray(rgba8, np.uint8)
    if mask:
        rd().write_mask_array(layer, a8[..., 0].astype(np.float32) / 255.0)
    else:
        rd().write_layer_array_srgb8(layer, a8)


def pixels(image):
    return rd().read_image_srgb8(image)


def solid(h, w, rgba):
    a = np.empty((h, w, 4), np.uint8)
    a[:] = np.asarray(rgba, np.uint8)
    return a


def noise(h, w, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8)
