"""Raster layer compositing, selections and pixel ops (owner: core-raster-io).

Images are numpy float32 arrays H x W x 4 (row 0 = top), straight alpha.
Layer pixels are stored 8-bit sRGB in Blender; convert with :func:`srgb_to_linear`.
Compositing is done in scene-linear to match the Blender node tree (Mix nodes) and follows the W3C
*Compositing and Blending Level 1* model (source-over with a blend mode), see :func:`composite`.
Blend formulas must match Blender's Mix node (``ramp_blend`` in Blender source)
for the modes listed in BLEND_MODES; ``BLENDER_MIX_TYPE`` maps them to Mix ``blend_type``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from . import render2d

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


def _split_rgb_alpha(c):
    a = np.asarray(c)
    if a.ndim >= 1 and a.shape[-1] == 4:
        return a[..., :3], a[..., 3:4], True
    return a, None, False


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Exact piecewise sRGB EOTF on RGB channels; alpha untouched when given H x W x 4."""
    a = np.asarray(c)
    dt = a.dtype if a.dtype in (np.float32, np.float64) else np.float32
    rgb, alpha, has_a = _split_rgb_alpha(a.astype(np.float64))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, np.power(np.maximum((rgb + 0.055) / 1.055, 0.0), 2.4))
    if has_a:
        lin = np.concatenate([lin, alpha], axis=-1)
    return lin.astype(dt)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Exact piecewise sRGB OETF (inverse of :func:`srgb_to_linear`); alpha untouched when 4 channels."""
    a = np.asarray(c)
    dt = a.dtype if a.dtype in (np.float32, np.float64) else np.float32
    rgb, alpha, has_a = _split_rgb_alpha(a.astype(np.float64))
    enc = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(np.maximum(rgb, 0.0), 1.0 / 2.4) - 0.055)
    if has_a:
        enc = np.concatenate([enc, alpha], axis=-1)
    return enc.astype(dt)


# --- HSV helpers with Blender's conventions (rgb_to_hsv / hsv_to_rgb in the Mix node) -------------

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    v = cmax
    nz = cmax != 0.0
    s = np.where(nz, delta / np.where(nz, cmax, 1.0), 0.0)
    has_hue = nz & (s != 0.0)
    d = np.where(has_hue, delta, 1.0)
    cr = (cmax - r) / d
    cg = (cmax - g) / d
    cb = (cmax - b) / d
    h = np.where(r == cmax, cb - cg, np.where(g == cmax, 2.0 + cr - cb, 4.0 + cg - cr))
    h = h / 6.0
    h = np.where(h < 0.0, h + 1.0, h)
    h = np.where(has_hue, h, 0.0)
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    nr = np.clip(np.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    ng = np.clip(2.0 - np.abs(h * 6.0 - 2.0), 0.0, 1.0)
    nb = np.clip(2.0 - np.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return np.stack([((nr - 1.0) * s + 1.0) * v, ((ng - 1.0) * s + 1.0) * v, ((nb - 1.0) * s + 1.0) * v], axis=-1)


def blend_rgb(mode: str, base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Blend RGB arrays (... x 3) with full factor, Blender Mix semantics (no clamping except where Blender clamps).

    ``base`` is Mix input 1 (``col1``), ``layer`` is input 2 (``col2``); factor is 1.
    """
    if mode not in BLEND_MODES:
        raise ValueError("unknown blend mode %r" % (mode,))
    in_dt = np.result_type(np.asarray(base).dtype, np.asarray(layer).dtype)
    if in_dt not in (np.float32, np.float64):
        in_dt = np.float32
    c1 = np.asarray(base, dtype=np.float64)
    c2 = np.asarray(layer, dtype=np.float64)
    if c1.shape[-1] != 3 or c2.shape[-1] != 3:
        raise ValueError("blend_rgb expects ... x 3 arrays")
    c1, c2 = np.broadcast_arrays(c1, c2)
    if mode == "NORMAL":
        out = c2.copy()
    elif mode == "MULTIPLY":
        out = c1 * c2
    elif mode == "SCREEN":
        out = 1.0 - (1.0 - c2) * (1.0 - c1)
    elif mode == "OVERLAY":
        out = np.where(c1 < 0.5, c1 * (2.0 * c2), 1.0 - (2.0 * (1.0 - c2)) * (1.0 - c1))
    elif mode == "DARKEN":
        out = np.minimum(c1, c2)
    elif mode == "LIGHTEN":
        out = np.maximum(c1, c2)
    elif mode == "COLOR_DODGE":
        tmp = 1.0 - c2
        with np.errstate(divide="ignore", invalid="ignore"):
            q = c1 / np.where(tmp > 0.0, tmp, 1.0)
        out = np.where(c1 != 0.0, np.where(tmp <= 0.0, 1.0, np.where(q > 1.0, 1.0, q)), c1)
    elif mode == "COLOR_BURN":
        tmp = c2
        with np.errstate(divide="ignore", invalid="ignore"):
            q = 1.0 - (1.0 - c1) / np.where(tmp > 0.0, tmp, 1.0)
        out = np.where(tmp <= 0.0, 0.0, np.where(q < 0.0, 0.0, np.where(q > 1.0, 1.0, q)))
    elif mode == "ADD":
        out = c1 + c2
    elif mode == "SUBTRACT":
        out = c1 - c2
    elif mode == "DIFFERENCE":
        out = np.abs(c1 - c2)
    elif mode == "EXCLUSION":
        out = np.maximum(c1 + c2 - 2.0 * c1 * c2, 0.0)
    elif mode == "SOFT_LIGHT":
        scr = 1.0 - (1.0 - c2) * (1.0 - c1)
        out = (1.0 - c1) * c2 * c1 + c1 * scr
    elif mode == "LINEAR_LIGHT":
        out = c1 + 2.0 * (c2 - 0.5)
    elif mode == "DIVIDE":
        nz = c2 != 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(nz, c1 / np.where(nz, c2, 1.0), c1)
    else:
        hsv1 = _rgb_to_hsv(c1)
        hsv2 = _rgb_to_hsv(c2)
        if mode == "HUE":
            t = hsv1.copy()
            t[..., 0] = hsv2[..., 0]
            out = np.where((hsv2[..., 1] != 0.0)[..., None], _hsv_to_rgb(t), c1)
        elif mode == "SATURATION":
            t = hsv1.copy()
            t[..., 1] = hsv2[..., 1]
            out = np.where((hsv1[..., 1] != 0.0)[..., None], _hsv_to_rgb(t), c1)
        elif mode == "COLOR":
            t = hsv1.copy()
            t[..., 0] = hsv2[..., 0]
            t[..., 1] = hsv2[..., 1]
            out = np.where((hsv2[..., 1] != 0.0)[..., None], _hsv_to_rgb(t), c1)
        else:  # VALUE
            t = hsv1.copy()
            t[..., 2] = hsv2[..., 2]
            out = _hsv_to_rgb(t)
    return out.astype(in_dt)


def composite(layers: Sequence[LayerSpec], height: int, width: int,
              background: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """Bottom-to-top composite in scene-linear, straight alpha (W3C Compositing & Blending Level 1, source-over).

    ``background`` (linear RGBA, straight alpha) is the initial backdrop; default fully transparent.  For every
    visible layer, with backdrop colour ``Cb`` / alpha ``ab`` and layer colour ``Cs``::

        as  = clip(layer_alpha * opacity * mask, 0, 1)
        B   = blend_rgb(mode, Cb, Cs)                    # Blender Mix-node formula at full factor
        Cs' = (1 - ab) * Cs + ab * B                     # the blend only acts where there is a backdrop
        ao  = as + ab * (1 - as)
        Co  = (as * Cs' + ab * Cb * (1 - as)) / ao       # Co = 0 where ao = 0

    Equivalently ``Co = mix(Cb, Cs', as / ao)``.  For NORMAL this is exact Porter-Duff *over*; over an opaque
    backdrop (``ab = 1``) it reduces to ``mix(Cb, B, as)`` (the previous behaviour of this function).  Pixels with
    ``ao == 0`` are returned as transparent black.  Returns ``H x W x 4`` float32, linear, straight alpha.
    """
    height = int(height)
    width = int(width)
    out = np.zeros((height, width, 4), dtype=np.float64)
    if background is not None:
        out[...] = np.asarray(background, dtype=np.float64).reshape(4)
    for lyr in layers:
        if not lyr.visible:
            continue
        if lyr.blend_mode not in BLEND_MODES:
            raise ValueError("unknown blend mode %r" % (lyr.blend_mode,))
        src = np.asarray(lyr.rgba, dtype=np.float64)
        if src.shape != (height, width, 4):
            raise ValueError("layer shape %r does not match composite size %r" % (src.shape, (height, width, 4)))
        a_s = src[..., 3] * float(lyr.opacity)
        if lyr.mask is not None:
            m = np.asarray(lyr.mask, dtype=np.float64)
            if m.shape != (height, width):
                raise ValueError("mask shape %r does not match composite size" % (m.shape,))
            a_s = a_s * m
        a_s = np.clip(a_s, 0.0, 1.0)
        c_b = out[..., :3]
        a_b = out[..., 3]
        c_s = src[..., :3]
        blended = blend_rgb(lyr.blend_mode, c_b, c_s).astype(np.float64)
        c_s2 = c_s + (blended - c_s) * a_b[..., None]                     # (1 - ab) * Cs + ab * B
        a_o = a_s + a_b * (1.0 - a_s)
        covered = a_o > 0.0
        k = np.where(covered, a_s / np.where(covered, a_o, 1.0), 0.0)     # as / ao
        rgb = c_b + (c_s2 - c_b) * k[..., None]                           # mix(Cb, Cs', as / ao)
        out[..., :3] = np.where(covered[..., None], rgb, 0.0)
        out[..., 3] = a_o
    return out.astype(np.float32)


def to_uint8_srgb(linear_rgba: np.ndarray) -> np.ndarray:
    """Linear float straight-alpha -> uint8 sRGB straight-alpha (rounded, clipped)."""
    a = np.asarray(linear_rgba, dtype=np.float64)
    if a.shape[-1] == 3:
        enc = linear_to_srgb(a)
    elif a.shape[-1] == 4:
        enc = linear_to_srgb(a)
    else:
        raise ValueError("expected ... x 3 or ... x 4 array")
    return np.clip(np.rint(enc * 255.0), 0, 255).astype(np.uint8)


def from_uint8_srgb(rgba8: np.ndarray) -> np.ndarray:
    """uint8 sRGB straight-alpha -> linear float32 straight-alpha."""
    a = np.asarray(rgba8)
    if a.dtype != np.uint8:
        raise ValueError("from_uint8_srgb expects uint8")
    return srgb_to_linear(a.astype(np.float64) / 255.0).astype(np.float32)


def _similar_mask(rgba: np.ndarray, x: int, y: int, tolerance: float) -> np.ndarray:
    a = np.asarray(rgba, dtype=np.float32)
    seed = a[y, x]
    return np.max(np.abs(a - seed), axis=-1) <= tolerance + 1e-7


def _row_runs(row: np.ndarray):
    """Start/end (exclusive) of True runs in a 1-D bool array."""
    pad = np.concatenate(([False], row, [False]))
    d = np.diff(pad.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def flood_fill_mask(rgba: np.ndarray, x: int, y: int, tolerance: float = 0.05, contiguous: bool = True) -> np.ndarray:
    """Boolean H x W mask of pixels similar (max channel diff incl. alpha <= tolerance) to (x, y).

    ``contiguous`` selects the 4-connected region containing the seed; otherwise every similar pixel.
    """
    a = np.asarray(rgba)
    if a.ndim != 3 or a.shape[2] != 4:
        raise ValueError("expected H x W x 4 image")
    h, w = a.shape[:2]
    x = int(x)
    y = int(y)
    if not (0 <= x < w and 0 <= y < h):
        raise IndexError("seed (%d, %d) outside %dx%d image" % (x, y, w, h))
    sim = _similar_mask(a, x, y, float(tolerance))
    if not contiguous:
        return sim
    runs = []
    for yy in range(h):
        runs.append(_row_runs(sim[yy]))
    out = np.zeros((h, w), dtype=bool)
    visited = [np.zeros(len(r[0]), dtype=bool) for r in runs]
    s0, e0 = runs[y]
    k0 = int(np.searchsorted(e0, x, side="right"))  # first run whose end > x
    stack = [(y, k0)]
    visited[y][k0] = True
    while stack:
        yy, k = stack.pop()
        s = runs[yy][0][k]
        e = runs[yy][1][k]
        out[yy, s:e] = True
        for ny in (yy - 1, yy + 1):
            if ny < 0 or ny >= h:
                continue
            ns, ne = runs[ny]
            if len(ns) == 0:
                continue
            # runs in ny overlapping [s, e): start < e and end > s
            hi = int(np.searchsorted(ns, e, side="left"))   # runs with start < e are [0, hi)
            lo = int(np.searchsorted(ne, s, side="right"))  # runs with end > s are [lo, n)
            for kk in range(lo, hi):
                if not visited[ny][kk]:
                    visited[ny][kk] = True
                    stack.append((ny, kk))
    return out


def fill(rgba: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float, float], opacity: float = 1.0) -> np.ndarray:
    """Return a copy with ``color`` composited (normal, straight-alpha over) where mask (float 0..1 or bool)."""
    a = np.asarray(rgba, dtype=np.float32)
    m = np.asarray(mask)
    if m.shape != a.shape[:2]:
        raise ValueError("mask shape %r does not match image %r" % (m.shape, a.shape[:2]))
    col = np.asarray(color, dtype=np.float64).reshape(4)
    sa = np.clip(m.astype(np.float64) * float(opacity) * col[3], 0.0, 1.0)
    dst = a.astype(np.float64)
    da = dst[..., 3]
    oa = sa + da * (1.0 - sa)
    safe = np.where(oa > 0, oa, 1.0)
    rgb = (col[:3] * sa[..., None] + dst[..., :3] * (da * (1.0 - sa))[..., None]) / safe[..., None]
    rgb = np.where((oa > 0)[..., None], rgb, dst[..., :3])
    out = np.concatenate([rgb, oa[..., None]], axis=-1)
    return out.astype(np.float32)


def rect_mask(height: int, width: int, x0: float, y0: float, x1: float, y1: float, antialias: bool = True) -> np.ndarray:
    """Float mask 0..1; pixel coords (0,0) = top-left corner of the top-left pixel."""
    height, width = int(height), int(width)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if antialias:
        px = np.arange(width, dtype=np.float64)
        py = np.arange(height, dtype=np.float64)
        cx = np.clip(np.minimum(x1, px + 1.0) - np.maximum(x0, px), 0.0, 1.0)
        cy = np.clip(np.minimum(y1, py + 1.0) - np.maximum(y0, py), 0.0, 1.0)
        return np.outer(cy, cx).astype(np.float32)
    cx = ((np.arange(width) + 0.5) >= x0) & ((np.arange(width) + 0.5) < x1)
    cy = ((np.arange(height) + 0.5) >= y0) & ((np.arange(height) + 0.5) < y1)
    return np.outer(cy, cx).astype(np.float32)


def ellipse_mask(height: int, width: int, x0: float, y0: float, x1: float, y1: float, antialias: bool = True) -> np.ndarray:
    """Elliptical selection inscribed in the box (x0, y0)-(x1, y1) (pixel coords, same convention as rect_mask)."""
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rx, ry = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    if rx <= 0 or ry <= 0:
        return np.zeros((int(height), int(width)), dtype=np.float32)
    n = int(np.clip(np.ceil(2.0 * np.pi * max(rx, ry) / 0.5), 64, 4096))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))
    return polygon_mask(height, width, pts, antialias=antialias)


def polygon_mask(height: int, width: int, points: Sequence[Tuple[float, float]], antialias: bool = True) -> np.ndarray:
    """Lasso selection (even-odd), supersampled AA when antialias."""
    height, width = int(height), int(width)
    if len(points) < 3:
        return np.zeros((height, width), dtype=np.float32)
    view = (0.0, 0.0, float(width), float(height))  # y_top=0, y_bottom=height: canvas y == pixel y
    if antialias:
        return render2d.coverage([list(points)], height, width, view, "EVENODD", supersample=4)
    # non-AA: pixel-centre point sampling, one sub-scanline per row at the pixel centre.
    rp = render2d._rings_to_pixels([list(points)], view, height, width)
    cov = render2d._coverage_px(rp, height, width, "EVENODD", 1)
    # with one sub-scanline per row the x coverage is still fractional; threshold at centre
    return (cov >= 0.5).astype(np.float32)


def combine_selection(current: Optional[np.ndarray], new: np.ndarray, op: str) -> np.ndarray:
    """Combine selection masks: REPLACE, ADD (union: max), SUBTRACT (current * (1-new)), INTERSECT (min)."""
    if op not in SELECTION_OPS:
        raise ValueError("unknown selection op %r" % (op,))
    n = np.clip(np.asarray(new, dtype=np.float32), 0.0, 1.0)
    if current is None or op == "REPLACE":
        if current is None and op in ("SUBTRACT", "INTERSECT"):
            return np.zeros_like(n)
        return n.copy()
    c = np.clip(np.asarray(current, dtype=np.float32), 0.0, 1.0)
    if c.shape != n.shape:
        raise ValueError("selection shapes differ: %r vs %r" % (c.shape, n.shape))
    if op == "ADD":
        return np.maximum(c, n)
    if op == "SUBTRACT":
        return (c * (1.0 - n)).astype(np.float32)
    return np.minimum(c, n)


def clear_selected(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Erase (alpha *= 1 - mask)."""
    out = np.array(rgba, dtype=np.float32, copy=True)
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    out[..., 3] *= (1.0 - m)
    return out


def apply_mask_to_alpha(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply alpha by the mask (keep only the selected area)."""
    out = np.array(rgba, dtype=np.float32, copy=True)
    out[..., 3] *= np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    return out


def resize_nearest(img: np.ndarray, height: int, width: int) -> np.ndarray:
    a = np.asarray(img)
    h0, w0 = a.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("target size must be positive")
    yi = np.minimum(((np.arange(int(height)) + 0.5) * h0 / height).astype(np.int64), h0 - 1)
    xi = np.minimum(((np.arange(int(width)) + 0.5) * w0 / width).astype(np.int64), w0 - 1)
    return a[yi][:, xi].copy()


def blender_pixels_to_array(flat, width: int, height: int) -> np.ndarray:
    """Blender ``Image.pixels`` (bottom row first, flat RGBA) -> H x W x 4 top-row-first float32."""
    a = np.asarray(flat, dtype=np.float32)
    if a.size != int(width) * int(height) * 4:
        raise ValueError("pixel buffer has %d values, expected %d" % (a.size, int(width) * int(height) * 4))
    return np.ascontiguousarray(a.reshape(int(height), int(width), 4)[::-1])


def array_to_blender_pixels(arr: np.ndarray) -> np.ndarray:
    """Inverse of :func:`blender_pixels_to_array`; returns flat float32 suitable for ``pixels.foreach_set``."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3 or a.shape[2] != 4:
        raise ValueError("expected H x W x 4 array")
    return np.ascontiguousarray(a[::-1]).reshape(-1)
