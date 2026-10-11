#!/usr/bin/env python3
"""Generate the SEAM Studio splash image procedurally (original artwork, no external assets).

    python3 scripts/make_splash.py [output.png]      # default: application_templates/seam/splash.png

Needs only numpy. Draws a vector path with its anchor points and handles on a dotted canvas (left) and the same
silhouette extruded into a 3D block (right), joined by a dotted arrow: "draw in 2D, get 3D". Deterministic: the same
run always writes the same bytes.
"""

import os
import struct
import sys
import zlib

import numpy as np

W, H = 1002, 564        # 2x Blender's 501x282 splash (crisp on HiDPI)
SS = 2                  # supersampling factor
BG_TOP = np.array([0.055, 0.063, 0.094])
BG_BOT = np.array([0.105, 0.125, 0.190])
INK = np.array([0.93, 0.95, 1.00])
CYAN = np.array([0.30, 0.78, 0.98])
ACCENT = np.array([0.98, 0.52, 0.16])


def write_png(path, rgb):
    """Minimal 8-bit RGB PNG writer (zlib + struct)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def blob_points(cx, cy, r, n=160):
    """Closed smooth shape: a soft five-lobed flower (polar curve)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = r * (1.0 + 0.24 * np.cos(5 * t) + 0.05 * np.cos(10 * t + 0.6))
    return np.stack([cx + rad * np.cos(t - np.pi / 2), cy + rad * np.sin(t - np.pi / 2)], axis=1)


def fill_polygon(shape, pts):
    """Boolean mask of the polygon ``pts`` (even-odd) on an array of ``shape`` = (h, w)."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs + 0.5
    ys = ys + 0.5
    inside = np.zeros(shape, dtype=bool)
    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        if ay == by:
            continue
        crosses = (ay > ys) != (by > ys)
        xint = ax + (ys - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (xs < xint)
    return inside


def segment_distance(shape, pts, closed=True):
    """Distance of every pixel centre to the polyline ``pts``."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    px = xs + 0.5
    py = ys + 0.5
    best = np.full(shape, 1e9)
    n = len(pts)
    for i in range(n if closed else n - 1):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy + 1e-12
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
        d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
        np.minimum(best, d, out=best)
    return best


def over(img, mask_alpha, color):
    a = np.clip(mask_alpha, 0.0, 1.0)[..., None]
    img[:] = img * (1 - a) + np.asarray(color)[None, None, :] * a


def draw_disc(img, cx, cy, r, color):
    h, w, _ = img.shape
    x0, x1 = int(max(cx - r - 3, 0)), int(min(cx + r + 3, w))
    y0, y1 = int(max(cy - r - 3, 0)), int(min(cy + r + 3, h))
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy)
    sub = img[y0:y1, x0:x1]
    over(sub, np.clip(r + 0.5 - d, 0, 1), color)


def draw_square(img, cx, cy, half, color):
    h, w, _ = img.shape
    x0, x1 = int(cx - half - 1), int(cx + half + 2)
    y0, y1 = int(cy - half - 1), int(cy + half + 2)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.maximum(np.abs(xs + 0.5 - cx), np.abs(ys + 0.5 - cy))
    over(img[y0:y1, x0:x1], np.clip(half + 0.5 - d, 0, 1), color)


def render() -> np.ndarray:
    w, h = W * SS, H * SS
    s = float(SS)
    # background: vertical gradient + two soft glows
    yy = np.linspace(0, 1, h)[:, None, None]
    img = BG_TOP[None, None, :] * (1 - yy) + BG_BOT[None, None, :] * yy
    img = np.broadcast_to(img, (h, w, 3)).copy()
    gy, gx = np.mgrid[0:h, 0:w]
    glow = np.exp(-(((gx - 0.28 * w) / (0.35 * w)) ** 2 + ((gy - 0.5 * h) / (0.5 * h)) ** 2))
    img += (CYAN * 0.10)[None, None, :] * glow[..., None]
    glow2 = np.exp(-(((gx - 0.76 * w) / (0.30 * w)) ** 2 + ((gy - 0.55 * h) / (0.5 * h)) ** 2))
    img += (ACCENT * 0.09)[None, None, :] * glow2[..., None]

    # dotted canvas grid on the left half
    step = 24 * SS
    for gxp in range(int(40 * s), int(0.52 * w), step):
        for gyp in range(int(36 * s), int(h - 20 * s), step):
            img[gyp:gyp + SS, gxp:gxp + SS] = img[gyp:gyp + SS, gxp:gxp + SS] * 0.6 + INK * 0.4 * 0.35

    # artboard frame (left)
    ab_x0, ab_y0, ab_x1, ab_y1 = 70 * s, 70 * s, 470 * s, 494 * s
    ys, xs = np.mgrid[0:h, 0:w]
    inside_ab = (xs >= ab_x0) & (xs < ab_x1) & (ys >= ab_y0) & (ys < ab_y1)
    img[inside_ab] = img[inside_ab] * 0.55 + np.array([0.97, 0.97, 0.98]) * 0.45
    edge = (np.abs(xs - ab_x0) < s) | (np.abs(xs - ab_x1) < s) | (np.abs(ys - ab_y0) < s) | (np.abs(ys - ab_y1) < s)
    img[edge & (xs >= ab_x0 - s) & (xs <= ab_x1 + s) & (ys >= ab_y0 - s) & (ys <= ab_y1 + s)] = CYAN * 0.8

    # the 2D vector shape: dark fill + cyan outline, anchors and handles
    pts = blob_points(270 * s, 282 * s, 112 * s)
    bx0, by0 = int(pts[:, 0].min() - 10 * s), int(pts[:, 1].min() - 10 * s)
    bx1, by1 = int(pts[:, 0].max() + 10 * s), int(pts[:, 1].max() + 10 * s)
    local = pts - np.array([bx0, by0])
    mask = fill_polygon((by1 - by0, bx1 - bx0), local)
    dist = segment_distance((by1 - by0, bx1 - bx0), local)
    sub = img[by0:by1, bx0:bx1]
    over(sub, mask.astype(float) * 0.92, np.array([0.12, 0.16, 0.30]))
    over(sub, np.clip(2.2 * s / 2 + 0.5 - dist, 0, 1), CYAN)
    # anchors with handles on every 32nd sample of the outline
    for i in range(0, len(pts), 32):
        p = pts[i]
        prev, nxt = pts[(i - 3) % len(pts)], pts[(i + 3) % len(pts)]
        tang = (nxt - prev)
        tang = tang / (np.linalg.norm(tang) + 1e-9)
        for sign in (-1, 1):
            q = p + sign * tang * 34 * s
            n_steps = 60
            for k in range(n_steps):
                u = p + (q - p) * (k / n_steps)
                x, y = int(u[0]), int(u[1])
                img[y:y + SS, x:x + SS] = img[y:y + SS, x:x + SS] * 0.35 + INK * 0.65
            draw_disc(img, q[0], q[1], 3.2 * s, INK)
        draw_square(img, p[0], p[1], 4.6 * s, ACCENT)

    # the extruded 3D result (right): stacked offset copies, darker toward the back, bright lit top face
    c3 = blob_points(0.0, 0.0, 92 * s)
    ex0, ey0 = int(c3[:, 0].min() - 4 * s), int(c3[:, 1].min() - 4 * s)
    ex1, ey1 = int(c3[:, 0].max() + 4 * s), int(c3[:, 1].max() + 4 * s)
    local3 = c3 - np.array([ex0, ey0])
    m3 = fill_polygon((ey1 - ey0, ex1 - ex0), local3).astype(float)
    d3 = segment_distance((ey1 - ey0, ex1 - ex0), local3)
    ox, oy = 736 * s, 292 * s
    depth = int(70 * s)
    for k in range(depth, -1, -1):
        shade = 0.20 + 0.45 * (1 - k / depth)
        col = ACCENT * shade * np.array([0.95, 0.9, 0.85])
        px, py = int(ox + ex0 + k * 0.62), int(oy + ey0 - k * 0.80)
        over(img[py:py + m3.shape[0], px:px + m3.shape[1]], m3, col)
    px, py = int(ox + ex0), int(oy + ey0)
    top = img[py:py + m3.shape[0], px:px + m3.shape[1]]
    over(top, m3, ACCENT)
    # soft light gradient over the top face + crisp rim
    lx = np.linspace(0, 1, m3.shape[1])[None, :] * np.ones((m3.shape[0], 1))
    ly = np.linspace(0, 1, m3.shape[0])[:, None] * np.ones((1, m3.shape[1]))
    over(top, m3 * np.clip(0.45 * (1 - 0.5 * lx - 0.6 * ly), 0, 0.4), np.array([1.0, 0.93, 0.80]))
    over(top, np.clip(1.6 * s / 2 + 0.5 - d3, 0, 1) * m3, np.array([1.0, 0.86, 0.62]))

    # dotted arrow between them
    ax0, ax1, ay = 410 * s, 540 * s, 282 * s
    for xk in np.arange(ax0, ax1 - 14 * s, 14 * s):
        draw_disc(img, xk, ay, 2.6 * s, INK * 0.9)
    tip = np.array([ax1, ay])
    for k in range(int(12 * s)):
        spread = k * 0.55
        for sign in (-1, 1):
            x, y = int(tip[0] - k), int(tip[1] + sign * spread)
            img[y:y + SS, x:x + SS] = INK * 0.9

    # vignette + downsample
    vig = 1.0 - 0.38 * (((gx - w / 2) / (w / 2)) ** 2 + ((gy - h / 2) / (h / 2)) ** 2) ** 1.2
    img *= np.clip(vig, 0.55, 1.0)[..., None]
    img = np.clip(img, 0, 1).reshape(H, SS, W, SS, 3).mean(axis=(1, 3))
    return (img * 255 + 0.5).astype(np.uint8)


def main(argv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = argv[1] if len(argv) > 1 else os.path.join(root, "application_templates", "seam", "splash.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_png(out, render())
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv)
