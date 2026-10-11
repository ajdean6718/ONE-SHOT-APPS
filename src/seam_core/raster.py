"""Raster layer compositing, selections and pixel ops (owner: core-raster-io).

Images are numpy float32 arrays H x W x 4 (row 0 = top), straight alpha.
Layer pixels are stored 8-bit sRGB in Blender; convert with :func:`srgb_to_linear`.
Compositing is done in scene-linear to match the Blender node tree (Mix nodes).
Blend formulas must match Blender's Mix node (``ramp_blend`` in Blender source)
for the modes listed in BLEND_MODES; ``BLENDER_MIX_TYPE`` maps them to Mix ``blend_type``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

BLEND_MODES = (
    "NORMAL", "MULTIPLY", "SCREEN", "OVERLAY", "DARKEN", "LIGHTEN", "COLOR_DODGE", "COLOR_BURN",
    "ADD", "SUBTRACT", "DIFFERENCE", "EXCLUSION", "SOFT_LIGHT", "LINEAR_LIGHT", "DIVIDE",
    "HUE", "SATURATION", "COLOR", "VALUE",
)
BLENDER_MIX_TYPE = {m: m for m in BLEND_MODES}
BLENDER_MIX_TYPE.update({"NORMAL": "MIX", "COLOR_DODGE": "DODGE", "COLOR_BURN": "BURN"})

SELECTION_OPS = ("REPLACE", "ADD", "SUBTRACT", "INTERSECT")


@dataclass
class LayerSpec:
    rgba: np.ndarray                 # H x W x 4 float32, **linear**, straight alpha
    opacity: float = 1.0
    blend_mode: str = "NORMAL"
    visible: bool = True
    mask: Optional[np.ndarray] = None  # H x W float32 0..1 (white = visible)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Exact piecewise sRGB EOTF on RGB channels; alpha untouched when given H x W x 4."""
    raise NotImplementedError


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def blend_rgb(mode: str, base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Blend RGB arrays (... x 3) with full factor, Blender Mix semantics (no clamping except where Blender clamps)."""
    raise NotImplementedError


def composite(layers: Sequence[LayerSpec], height: int, width: int,
              background: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """Bottom-to-top composite. For each visible layer: f = layer.alpha * opacity * mask;
    rgb = mix(prev_rgb, blend_rgb(mode, prev_rgb, layer_rgb), f) where prev alpha > 0, and
    plain layer_rgb where prev alpha == 0; alpha = prev_a + f * (1 - prev_a).
    background (linear RGBA) is the initial state; default transparent. Returns H x W x 4 linear.
    """
    raise NotImplementedError


def to_uint8_srgb(linear_rgba: np.ndarray) -> np.ndarray:
    """Linear float straight-alpha -> uint8 sRGB straight-alpha (rounded, clipped)."""
    raise NotImplementedError


def from_uint8_srgb(rgba8: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def flood_fill_mask(rgba: np.ndarray, x: int, y: int, tolerance: float = 0.05, contiguous: bool = True) -> np.ndarray:
    """Boolean H x W mask of pixels similar (max channel diff incl. alpha <= tolerance) to (x, y)."""
    raise NotImplementedError


def fill(rgba: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float, float], opacity: float = 1.0) -> np.ndarray:
    """Return a copy with ``color`` composited (normal) where mask (float 0..1 or bool)."""
    raise NotImplementedError


def rect_mask(height: int, width: int, x0: float, y0: float, x1: float, y1: float, antialias: bool = True) -> np.ndarray:
    """Float mask 0..1; pixel coords (0,0) = top-left corner of the top-left pixel."""
    raise NotImplementedError


def ellipse_mask(height: int, width: int, x0: float, y0: float, x1: float, y1: float, antialias: bool = True) -> np.ndarray:
    raise NotImplementedError


def polygon_mask(height: int, width: int, points: Sequence[Tuple[float, float]], antialias: bool = True) -> np.ndarray:
    """Lasso selection (even-odd), supersampled AA when antialias."""
    raise NotImplementedError


def combine_selection(current: Optional[np.ndarray], new: np.ndarray, op: str) -> np.ndarray:
    raise NotImplementedError


def clear_selected(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Erase (alpha *= 1 - mask)."""
    raise NotImplementedError


def apply_mask_to_alpha(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def resize_nearest(img: np.ndarray, height: int, width: int) -> np.ndarray:
    raise NotImplementedError


def blender_pixels_to_array(flat, width: int, height: int) -> np.ndarray:
    """Blender ``Image.pixels`` (bottom row first, flat RGBA) -> H x W x 4 top-row-first float32."""
    raise NotImplementedError


def array_to_blender_pixels(arr: np.ndarray) -> np.ndarray:
    """Inverse of :func:`blender_pixels_to_array`; returns flat float32 suitable for ``pixels.foreach_set``."""
    raise NotImplementedError
