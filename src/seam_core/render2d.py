"""Deterministic 2D rasterizer for PNG export (owner: core-raster-io).

Renders a list of draw items (bottom to top) into a linear float RGBA image with
supersampled anti-aliasing. Coordinates are canvas units; ``view`` maps the
rectangle (x0, y_top, x1, y_bottom) in canvas space (Y up) onto the output image
(row 0 = top). Pure numpy; 1920x1080 with ~50 simple shapes at supersample=4 should
finish in a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

RGBA = Tuple[float, float, float, float]


@dataclass
class SolidPaint:
    color: RGBA  # linear, straight alpha


@dataclass
class GradientStop:
    position: float
    color: RGBA  # linear


@dataclass
class LinearGradientPaint:
    start: Tuple[float, float]
    end: Tuple[float, float]
    stops: List[GradientStop] = field(default_factory=list)


@dataclass
class RadialGradientPaint:
    center: Tuple[float, float]
    radius: float
    stops: List[GradientStop] = field(default_factory=list)


@dataclass
class FillItem:
    rings: List[List[Tuple[float, float]]]
    paint: object  # SolidPaint | LinearGradientPaint | RadialGradientPaint
    fill_rule: str = "EVENODD"
    opacity: float = 1.0
    clip: Optional[List[List[Tuple[float, float]]]] = None  # clip rings (EVENODD) or None


@dataclass
class ImageItem:
    rgba: np.ndarray  # H x W x 4 linear straight alpha
    x: float          # canvas-space left
    y_top: float      # canvas-space top (Y up)
    width: float      # canvas units
    height: float
    opacity: float = 1.0
    clip: Optional[List[List[Tuple[float, float]]]] = None


# ---------------------------------------------------------------------------
# Scanline coverage: exact horizontal coverage per sub-scanline, sampled vertically.
# ---------------------------------------------------------------------------

def _ssy(supersample: int) -> int:
    """Vertical sub-scanlines per pixel row (x coverage is analytic, so y gets 2x the nominal samples)."""
    return max(1, int(supersample)) * 2


def _rings_to_pixels(rings, view, height, width):
    """Canvas rings -> list of (n,2) float64 pixel-space arrays (row 0 = top)."""
    vx0, vy_top, vx1, vy_bot = (float(v) for v in view)
    sx = width / (vx1 - vx0) if vx1 != vx0 else 0.0
    sy = height / (vy_top - vy_bot) if vy_top != vy_bot else 0.0
    out = []
    for ring in rings:
        if ring is None or len(ring) < 3:
            continue
        a = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
        px = (a[:, 0] - vx0) * sx
        py = (vy_top - a[:, 1]) * sy
        out.append(np.stack([px, py], axis=1))
    return out


def _coverage_px(rings_px, height: int, width: int, fill_rule: str, ssy: int) -> np.ndarray:
    """Coverage of pixel-space rings over a height x width window (float32 0..1)."""
    cov = np.zeros((height, width), dtype=np.float32)
    if height <= 0 or width <= 0 or not rings_px:
        return cov
    ax, ay, bx, by = [], [], [], []
    for r in rings_px:
        if not np.all(np.isfinite(r)):
            continue
        nxt = np.roll(r, -1, axis=0)
        ax.append(r[:, 0]); ay.append(r[:, 1]); bx.append(nxt[:, 0]); by.append(nxt[:, 1])
    if not ax:
        return cov
    ax = np.concatenate(ax); ay = np.concatenate(ay); bx = np.concatenate(bx); by = np.concatenate(by)
    keep = ay != by
    ax, ay, bx, by = ax[keep], ay[keep], bx[keep], by[keep]
    if ax.size == 0:
        return cov
    nsub = height * ssy
    ymin = np.minimum(ay, by)
    ymax = np.maximum(ay, by)
    # sub-scanline j samples y = (j + 0.5) / ssy ; edge covers ymin <= y < ymax
    j_lo = np.ceil(ymin * ssy - 0.5).astype(np.int64)
    j_hi = np.ceil(ymax * ssy - 0.5).astype(np.int64)
    j_lo = np.clip(j_lo, 0, nsub)
    j_hi = np.clip(j_hi, 0, nsub)
    cnt = np.maximum(j_hi - j_lo, 0)
    total = int(cnt.sum())
    if total == 0:
        return cov
    eidx = np.repeat(np.arange(ax.size), cnt)
    starts = np.cumsum(cnt) - cnt
    j = np.arange(total, dtype=np.int64) - np.repeat(starts, cnt) + np.repeat(j_lo, cnt)
    ys = (j + 0.5) / ssy
    ea_x, ea_y, eb_x, eb_y = ax[eidx], ay[eidx], bx[eidx], by[eidx]
    x = ea_x + (ys - ea_y) * (eb_x - ea_x) / (eb_y - ea_y)
    d = np.where(eb_y > ea_y, 1, -1).astype(np.int64)
    order = np.lexsort((x, j))
    j, x, d = j[order], x[order], d[order]
    # group boundaries
    newgrp = np.ones(total, dtype=bool)
    newgrp[1:] = j[1:] != j[:-1]
    gstart = np.flatnonzero(newgrp)
    gid = np.cumsum(newgrp) - 1
    if fill_rule == "NONZERO":
        cs = np.cumsum(d)
        base = np.where(gstart > 0, cs[gstart - 1], 0)
        wind = cs - base[gid]
        inside = wind != 0
    else:
        pos = np.arange(total) - gstart[gid]
        inside = (pos % 2) == 0
    # span k -> k+1 within the same group
    same = np.zeros(total, dtype=bool)
    same[:-1] = j[1:] == j[:-1]
    sel = inside & same
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return cov
    l = np.clip(x[idx], 0.0, float(width))
    r = np.clip(x[idx + 1], 0.0, float(width))
    ok = r > l
    idx, l, r = idx[ok], l[ok], r[ok]
    if idx.size == 0:
        return cov
    row = j[idx] // ssy
    il = np.floor(l).astype(np.int64)
    ir = np.floor(r).astype(np.int64)
    W2 = width + 2
    flat_n = height * W2
    same_px = il == ir
    # partial coverage terms
    pl_idx = row * W2 + il
    pl_val = np.where(same_px, r - l, il + 1 - l)
    diff_m = ~same_px
    pr_idx = (row * W2 + ir)[diff_m]
    pr_val = (r - ir)[diff_m]
    partial = (np.bincount(pl_idx, weights=pl_val, minlength=flat_n)
               + np.bincount(pr_idx, weights=pr_val, minlength=flat_n))
    diff = (np.bincount((row * W2 + il + 1)[diff_m], minlength=flat_n).astype(np.float64)
            - np.bincount((row * W2 + ir)[diff_m], minlength=flat_n))
    partial = partial.reshape(height, W2)
    full = np.cumsum(diff.reshape(height, W2), axis=1)
    out = (partial + full)[:, :width] / ssy
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32)


def coverage(rings: Sequence[Sequence[Tuple[float, float]]], height: int, width: int,
             view: Tuple[float, float, float, float], fill_rule: str = "EVENODD", supersample: int = 4) -> np.ndarray:
    """Float H x W coverage 0..1 of the rings.

    Horizontal coverage is analytic (exact span overlap per sub-scanline); vertical coverage is
    sampled at ``2 * supersample`` sub-scanlines per pixel row. Axis-aligned pixel-aligned shapes
    are therefore exact.
    """
    fill_rule = _check_rule(fill_rule)
    rp = _rings_to_pixels(rings, view, height, width)
    return _coverage_px(rp, int(height), int(width), fill_rule, _ssy(supersample))


def _check_rule(rule: str) -> str:
    r = str(rule).upper().replace("-", "").replace("_", "")
    if r == "EVENODD":
        return "EVENODD"
    if r == "NONZERO":
        return "NONZERO"
    raise ValueError("unknown fill rule %r" % (rule,))


# ---------------------------------------------------------------------------
# Paints
# ---------------------------------------------------------------------------

def _eval_stops(stops: Sequence[GradientStop], t: np.ndarray) -> np.ndarray:
    """Evaluate gradient stops (pad spread) at parameter array ``t`` -> t.shape + (4,) float64."""
    if not stops:
        return np.zeros(t.shape + (4,), dtype=np.float64)
    st = sorted(stops, key=lambda s: s.position)
    P = np.array([s.position for s in st], dtype=np.float64)
    C = np.array([s.color for s in st], dtype=np.float64).reshape(-1, 4)
    n = len(st)
    idx = np.searchsorted(P, t, side="right")
    i0 = np.clip(idx - 1, 0, n - 1)
    i1 = np.clip(idx, 0, n - 1)
    span = P[i1] - P[i0]
    safe = np.where(span > 0, span, 1.0)
    frac = np.where(span > 0, (t - P[i0]) / safe, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    return C[i0] + (C[i1] - C[i0]) * frac[..., None]


def _paint_colors(paint, shape, view, full_h, full_w, x0, y0):
    """Per-pixel straight RGBA (float64) for the crop starting at pixel (x0, y0) with ``shape`` (h, w);
    ``full_h``/``full_w`` are the whole output dimensions that ``view`` maps to."""
    h, w = shape
    if isinstance(paint, SolidPaint):
        col = np.asarray(paint.color, dtype=np.float64).reshape(4)
        return np.broadcast_to(col, (h, w, 4))
    vx0, vy_top, vx1, vy_bot = (float(v) for v in view)
    px = (np.arange(w) + x0 + 0.5) / full_w * (vx1 - vx0) + vx0
    py = vy_top - (np.arange(h) + y0 + 0.5) / full_h * (vy_top - vy_bot)
    X, Y = np.meshgrid(px, py)
    if isinstance(paint, LinearGradientPaint):
        sx, sy = paint.start
        ex, ey = paint.end
        dx, dy = ex - sx, ey - sy
        L2 = dx * dx + dy * dy
        if L2 <= 0:
            if not paint.stops:
                return np.zeros((h, w, 4))
            last = sorted(paint.stops, key=lambda s: s.position)[-1].color
            return np.broadcast_to(np.asarray(last, dtype=np.float64), (h, w, 4))
        t = ((X - sx) * dx + (Y - sy) * dy) / L2
        return _eval_stops(paint.stops, t)
    if isinstance(paint, RadialGradientPaint):
        cx, cy = paint.center
        if paint.radius <= 0:
            if not paint.stops:
                return np.zeros((h, w, 4))
            last = sorted(paint.stops, key=lambda s: s.position)[-1].color
            return np.broadcast_to(np.asarray(last, dtype=np.float64), (h, w, 4))
        t = np.hypot(X - cx, Y - cy) / paint.radius
        return _eval_stops(paint.stops, t)
    raise TypeError("unknown paint type %r" % (type(paint).__name__,))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _bounds_px(rings_px, height, width):
    pts = np.concatenate([r for r in rings_px]) if rings_px else np.zeros((0, 2))
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.size == 0:
        return None
    x0 = int(max(0, np.floor(pts[:, 0].min())))
    y0 = int(max(0, np.floor(pts[:, 1].min())))
    x1 = int(min(width, np.ceil(pts[:, 0].max())))
    y1 = int(min(height, np.ceil(pts[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _crop_coverage(rings_px, box, rule, ssy):
    x0, y0, x1, y1 = box
    shifted = [r - np.array([x0, y0], dtype=np.float64) for r in rings_px]
    return _coverage_px(shifted, y1 - y0, x1 - x0, rule, ssy)


def _over(dst: np.ndarray, src_rgb: np.ndarray, src_a: np.ndarray) -> None:
    """In-place straight-alpha 'normal' alpha-over of src onto dst (crop views, float32/64)."""
    da = dst[..., 3]
    out_a = src_a + da * (1.0 - src_a)
    safe = np.where(out_a > 0, out_a, 1.0)
    num = src_rgb * src_a[..., None] + dst[..., :3] * (da * (1.0 - src_a))[..., None]
    rgb = np.where((out_a > 0)[..., None], num / safe[..., None], 0.0)
    dst[..., :3] = rgb
    dst[..., 3] = out_a


def _clip_cov(clip, view, height, width, box, ssy):
    """Clip coverage cropped to ``box`` (EVENODD rings)."""
    cp = _rings_to_pixels(clip, view, height, width)
    if not cp:
        x0, y0, x1, y1 = box
        return np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    return _crop_coverage(cp, box, "EVENODD", ssy)


def _sample_image(rgba: np.ndarray, box, dst_rect, height, width):
    """Resample ``rgba`` (linear straight alpha) onto the pixel ``box`` for the destination rect
    (dx0, dy0, dx1, dy1 in output pixels). Bilinear on premultiplied values; exact when 1:1 aligned."""
    x0, y0, x1, y1 = box
    dx0, dy0, dx1, dy1 = dst_rect
    img = np.asarray(rgba, dtype=np.float64)
    ih, iw = img.shape[:2]
    # prefilter strong downscales with 2x2 premultiplied box averages
    sxr = iw / max(dx1 - dx0, 1e-9)
    syr = ih / max(dy1 - dy0, 1e-9)
    pre = np.concatenate([img[..., :3] * img[..., 3:4], img[..., 3:4]], axis=2)
    while sxr >= 2.0 and pre.shape[1] >= 2:
        n = pre.shape[1] // 2 * 2
        pre = 0.5 * (pre[:, 0:n:2] + pre[:, 1:n:2]) if n == pre.shape[1] else \
            np.concatenate([0.5 * (pre[:, 0:n:2] + pre[:, 1:n:2]), pre[:, -1:]], axis=1)
        sxr /= 2.0
    while syr >= 2.0 and pre.shape[0] >= 2:
        n = pre.shape[0] // 2 * 2
        pre = 0.5 * (pre[0:n:2] + pre[1:n:2]) if n == pre.shape[0] else \
            np.concatenate([0.5 * (pre[0:n:2] + pre[1:n:2]), pre[-1:]], axis=0)
        syr /= 2.0
    ih, iw = pre.shape[:2]
    cx = np.arange(x0, x1) + 0.5
    cy = np.arange(y0, y1) + 0.5
    u = (cx - dx0) / (dx1 - dx0) * iw - 0.5
    v = (cy - dy0) / (dy1 - dy0) * ih - 0.5
    u = np.where(np.abs(u - np.round(u)) < 1e-6, np.round(u), u)
    v = np.where(np.abs(v - np.round(v)) < 1e-6, np.round(v), v)
    u = np.clip(u, 0.0, iw - 1.0)
    v = np.clip(v, 0.0, ih - 1.0)
    iu0 = np.floor(u).astype(np.int64)
    iv0 = np.floor(v).astype(np.int64)
    iu1 = np.minimum(iu0 + 1, iw - 1)
    iv1 = np.minimum(iv0 + 1, ih - 1)
    fu = (u - iu0)[None, :, None]
    fv = (v - iv0)[:, None, None]
    p00 = pre[iv0][:, iu0]
    p01 = pre[iv0][:, iu1]
    p10 = pre[iv1][:, iu0]
    p11 = pre[iv1][:, iu1]
    top = p00 * (1 - fu) + p01 * fu
    bot = p10 * (1 - fu) + p11 * fu
    res = top * (1 - fv) + bot * fv
    a = res[..., 3]
    rgb = np.where(a[..., None] > 0, res[..., :3] / np.where(a > 0, a, 1.0)[..., None], 0.0)
    return rgb, a


def render(items: Sequence[object], width: int, height: int, view: Tuple[float, float, float, float],
           supersample: int = 4, background: Optional[RGBA] = None) -> np.ndarray:
    """Composite items with normal alpha-over in linear space; returns H x W x 4 linear float32."""
    width = int(width)
    height = int(height)
    if width < 1 or height < 1:
        raise ValueError("output size must be positive")
    vx0, vy_top, vx1, vy_bot = (float(v) for v in view)
    if vx1 == vx0 or vy_top == vy_bot:
        raise ValueError("view rectangle must have non-zero size")
    ssy = _ssy(supersample)
    canvas = np.zeros((height, width, 4), dtype=np.float64)
    if background is not None:
        canvas[...] = np.asarray(background, dtype=np.float64).reshape(4)
    sx = width / (vx1 - vx0)
    sy = height / (vy_top - vy_bot)
    for item in items:
        if isinstance(item, FillItem):
            rule = _check_rule(item.fill_rule)
            rp = _rings_to_pixels(item.rings, view, height, width)
            if not rp:
                continue
            box = _bounds_px(rp, height, width)
            if box is None:
                continue
            cov = _crop_coverage(rp, box, rule, ssy).astype(np.float64)
            if item.clip is not None:
                cov = cov * _clip_cov(item.clip, view, height, width, box, ssy)
            x0, y0, x1, y1 = box
            col = _paint_colors(item.paint, (y1 - y0, x1 - x0), view, height, width, x0, y0)
            a = col[..., 3] * float(item.opacity) * cov
            _over(canvas[y0:y1, x0:x1], col[..., :3], a)
        elif isinstance(item, ImageItem):
            dx0 = (item.x - vx0) * sx
            dx1 = (item.x + item.width - vx0) * sx
            dy0 = (vy_top - item.y_top) * sy
            dy1 = (vy_top - (item.y_top - item.height)) * sy
            if dx1 < dx0:
                dx0, dx1 = dx1, dx0
            if dy1 < dy0:
                dy0, dy1 = dy1, dy0
            if dx1 - dx0 <= 0 or dy1 - dy0 <= 0:
                continue
            rect_px = [np.array([(dx0, dy0), (dx1, dy0), (dx1, dy1), (dx0, dy1)], dtype=np.float64)]
            box = _bounds_px(rect_px, height, width)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            cov = _crop_coverage(rect_px, box, "EVENODD", ssy).astype(np.float64)
            if item.clip is not None:
                cov = cov * _clip_cov(item.clip, view, height, width, box, ssy)
            rgb, a_img = _sample_image(item.rgba, box, (dx0, dy0, dx1, dy1), height, width)
            a = a_img * float(item.opacity) * cov
            _over(canvas[y0:y1, x0:x1], rgb, a)
        else:
            raise TypeError("unknown draw item %r" % (type(item).__name__,))
    return canvas.astype(np.float32)
