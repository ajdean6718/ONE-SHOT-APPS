import json
import math
import os
import time

import numpy as np
import pytest

from seam_core import render2d as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sq(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def view_px(w, h):
    """Canvas view where canvas (x, y_up) == (px, h - py) i.e. y_top = h, y_bottom = 0."""
    return (0.0, float(h), float(w), 0.0)


RED = R.SolidPaint((1.0, 0.0, 0.0, 1.0))


def test_square_coverage_exact():
    cov = R.coverage([sq(10, 10, 30, 30)], 64, 64, view_px(64, 64))
    assert cov.shape == (64, 64) and cov.dtype == np.float32
    assert abs(cov.sum() - 400.0) / 400.0 < 0.005
    assert cov.sum() == pytest.approx(400.0, abs=1e-3)
    assert cov.max() == 1.0 and cov.min() == 0.0
    # fractional edges
    cov = R.coverage([sq(10.5, 10.25, 30.5, 30.25)], 64, 64, view_px(64, 64))
    assert abs(cov.sum() - 400.0) / 400.0 < 0.005


def test_square_coverage_scaled_view():
    # canvas 100 x 100 units mapped onto 50 x 50 px (0.5 px/unit): a 40x40 unit square covers 20x20 px
    cov = R.coverage([sq(10, 10, 50, 50)], 50, 50, (0, 100, 100, 0))
    assert cov.sum() == pytest.approx(400.0, abs=1e-3)
    assert cov[50 - 25:50 - 5, 5:25].min() == 1.0
    # non-square pixel aspect: 100 x 50 units onto 100 x 100 px
    cov = R.coverage([sq(10, 10, 30, 20)], 100, 100, (0, 50, 100, 0))
    assert cov.sum() == pytest.approx(20 * 1 * 10 * 2, abs=1e-3)


def star(cx, cy, r, n=5, step=2):
    pts = []
    for i in range(n):
        a = math.pi / 2 + 2 * math.pi * (i * step % n) / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def test_evenodd_hole_and_nonzero_star():
    # square with same-direction inner square: even-odd -> hole, nonzero -> filled
    outer = sq(10, 10, 50, 50)
    inner = sq(20, 20, 40, 40)
    eo = R.coverage([outer, inner], 64, 64, view_px(64, 64), "EVENODD")
    nz = R.coverage([outer, inner], 64, 64, view_px(64, 64), "NONZERO")
    assert eo.sum() == pytest.approx(1600 - 400, abs=1e-3)
    assert nz.sum() == pytest.approx(1600, abs=1e-3)
    # opposite winding inner is a hole for both rules
    nz2 = R.coverage([outer, list(reversed(inner))], 64, 64, view_px(64, 64), "NONZERO")
    assert nz2.sum() == pytest.approx(1200, abs=1e-3)
    # pentagram: centre pentagon is filled by nonzero but empty with even-odd
    s = star(32, 32, 28)
    eo = R.coverage([s], 64, 64, view_px(64, 64), "EVENODD")
    nz = R.coverage([s], 64, 64, view_px(64, 64), "NONZERO")
    assert nz.sum() > eo.sum() * 1.1
    assert nz[32, 32] == 1.0 and eo[32, 32] == 0.0
    # nonzero pentagram area vs analytic: outer pentagon-ish star area = whole star incl. centre
    r = 28.0
    inner_r = r * (math.sin(math.radians(18)) / math.sin(math.radians(54)))
    # centre pentagon area (circumradius inner_r)
    centre = 2.5 * inner_r ** 2 * math.sin(math.radians(72))
    assert abs((nz.sum() - eo.sum()) - centre) / centre < 0.02


def test_circle_area_and_aa_edges():
    n = 256
    pts = [(32 + 20 * math.cos(2 * math.pi * i / n), 32 + 20 * math.sin(2 * math.pi * i / n)) for i in range(n)]
    cov = R.coverage([pts], 64, 64, view_px(64, 64))
    assert abs(cov.sum() - math.pi * 400) / (math.pi * 400) < 0.005
    edge = (cov > 0) & (cov < 1)
    assert edge.sum() > 40  # anti-aliased boundary exists


def test_supersample_1_still_exact_for_aligned():
    cov = R.coverage([sq(5, 5, 25, 25)], 32, 32, view_px(32, 32), supersample=1)
    assert cov.sum() == pytest.approx(400.0, abs=1e-3)


def test_render_solid_fill_opacity_and_background():
    img = R.render([R.FillItem([sq(10, 10, 30, 30)], RED, opacity=0.5)], 64, 64, view_px(64, 64))
    assert img.shape == (64, 64, 4) and img.dtype == np.float32
    px = img[64 - 20, 20]  # canvas (20, 20) -> row 44
    assert np.allclose(px, [1, 0, 0, 0.5], atol=1e-6)
    assert np.allclose(img[0, 0], 0.0)
    bg = R.render([R.FillItem([sq(10, 10, 30, 30)], RED, opacity=0.5)], 64, 64, view_px(64, 64), background=(0, 0, 1, 1))
    assert np.allclose(bg[64 - 20, 20], [0.5, 0.0, 0.5, 1.0], atol=1e-6)
    assert np.allclose(bg[0, 0], [0, 0, 1, 1])


def test_render_item_order_and_alpha_over():
    blue = R.SolidPaint((0.0, 0.0, 1.0, 0.5))
    items = [R.FillItem([sq(0, 0, 32, 32)], RED), R.FillItem([sq(16, 16, 48, 48)], blue)]
    img = R.render(items, 64, 64, view_px(64, 64))
    assert np.allclose(img[64 - 20, 20], [0.5, 0, 0.5, 1.0], atol=1e-6)   # overlap
    assert np.allclose(img[64 - 5, 5], [1, 0, 0, 1], atol=1e-6)
    assert np.allclose(img[64 - 40, 40], [0, 0, 1, 0.5], atol=1e-6)


def test_linear_gradient_endpoints_exact():
    stops = [R.GradientStop(0.0, (1, 0, 0, 1)), R.GradientStop(1.0, (0, 0, 1, 1))]
    g = R.LinearGradientPaint((0.5, 0.0), (63.5, 0.0), stops)
    img = R.render([R.FillItem([sq(0, 0, 64, 64)], g)], 64, 64, view_px(64, 64))
    # pixel centres at x=0.5 and x=63.5 coincide with the gradient endpoints
    assert np.allclose(img[10, 0], [1, 0, 0, 1], atol=1e-6)
    assert np.allclose(img[10, 63], [0, 0, 1, 1], atol=1e-6)
    assert np.allclose(img[10, 32], [0.5, 0, 0.5, 1], atol=0.02)
    # pad outside the vector
    g2 = R.LinearGradientPaint((20.0, 0.0), (40.0, 0.0), stops)
    img2 = R.render([R.FillItem([sq(0, 0, 64, 64)], g2)], 64, 64, view_px(64, 64))
    assert np.allclose(img2[10, 0], [1, 0, 0, 1], atol=1e-6)
    assert np.allclose(img2[10, 63], [0, 0, 1, 1], atol=1e-6)


def test_radial_gradient_endpoints_exact_and_multistop():
    stops = [R.GradientStop(0.0, (1, 1, 1, 1)), R.GradientStop(0.5, (0, 1, 0, 1)), R.GradientStop(1.0, (0, 0, 0, 1))]
    g = R.RadialGradientPaint((32.5, 32.5), 20.0, stops)
    img = R.render([R.FillItem([sq(0, 0, 64, 64)], g)], 64, 64, view_px(64, 64))
    # canvas (32.5, 32.5) is the centre of pixel (x=32, row=64-33=31)
    assert np.allclose(img[31, 32], [1, 1, 1, 1], atol=1e-6)
    assert np.allclose(img[31, 63], [0, 0, 0, 1], atol=1e-6)  # far outside radius -> last stop
    # ten px to the right: t = 0.5 -> middle stop exactly green
    assert np.allclose(img[31, 42], [0, 1, 0, 1], atol=1e-5)


def test_gradient_alpha_stops_and_degenerate():
    stops = [R.GradientStop(0.0, (1, 0, 0, 0.0)), R.GradientStop(1.0, (1, 0, 0, 1.0))]
    g = R.LinearGradientPaint((0.5, 0), (63.5, 0), stops)
    img = R.render([R.FillItem([sq(0, 0, 64, 64)], g)], 64, 64, view_px(64, 64))
    assert img[5, 0, 3] == pytest.approx(0.0, abs=1e-6) and img[5, 63, 3] == pytest.approx(1.0, abs=1e-6)
    deg = R.LinearGradientPaint((5, 5), (5, 5), stops)
    img = R.render([R.FillItem([sq(0, 0, 8, 8)], deg)], 8, 8, view_px(8, 8))
    assert np.allclose(img[3, 3], [1, 0, 0, 1])
    empty = R.LinearGradientPaint((0, 0), (1, 0), [])
    img = R.render([R.FillItem([sq(0, 0, 8, 8)], empty)], 8, 8, view_px(8, 8))
    assert not img.any()


def test_clip_restricts_fill():
    clip = sq(0, 0, 32, 64)
    img = R.render([R.FillItem([sq(0, 0, 64, 64)], RED, clip=[clip])], 64, 64, view_px(64, 64))
    assert img[10, 10, 3] == 1.0 and img[10, 50, 3] == 0.0
    assert img[..., 3].sum() == pytest.approx(32 * 64, abs=1e-3)
    # empty clip -> nothing
    img = R.render([R.FillItem([sq(0, 0, 64, 64)], RED, clip=[])], 64, 64, view_px(64, 64))
    assert not img.any()


def test_image_item_integer_placement_pixel_exact():
    rng = np.random.default_rng(3)
    tex = rng.random((8, 6, 4)).astype(np.float32)
    # place at canvas x=10, y_top=40 with size 6x8 => pixel box x 10..16, rows 64-40..64-32 = 24..32
    item = R.ImageItem(tex, 10.0, 40.0, 6.0, 8.0)
    img = R.render([item], 64, 64, view_px(64, 64))
    assert np.allclose(img[24:32, 10:16], tex, atol=1e-6)
    out = img.copy()
    out[24:32, 10:16] = 0
    assert not out.any()


def test_image_item_opacity_clip_and_over():
    tex = np.tile(np.array([0, 1, 0, 1], np.float32), (4, 4, 1))
    base = R.FillItem([sq(0, 0, 16, 16)], RED)
    img = R.render([base, R.ImageItem(tex, 0, 8, 4, 4, opacity=0.5)], 16, 16, view_px(16, 16))
    assert np.allclose(img[16 - 8, 0], [0.5, 0.5, 0, 1], atol=1e-6)
    assert np.allclose(img[16 - 1, 15], [1, 0, 0, 1])
    clipped = R.render([R.ImageItem(tex, 0, 8, 4, 4, clip=[sq(0, 0, 2, 16)])], 16, 16, view_px(16, 16))
    assert clipped[..., 3].sum() == pytest.approx(2 * 4, abs=1e-3)


def test_image_item_scaled_and_straight_alpha():
    tex = np.zeros((2, 2, 4), np.float32)
    tex[0, 0] = [1, 0, 0, 1]
    tex[0, 1] = [0, 1, 0, 1]
    tex[1, 0] = [0, 0, 1, 1]
    tex[1, 1] = [1, 1, 1, 1]
    img = R.render([R.ImageItem(tex, 0, 8, 8, 8)], 8, 8, view_px(8, 8))
    assert img[..., 3].min() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(img[0, 0, :3], [1, 0, 0], atol=1e-6)
    assert np.allclose(img[7, 7, :3], [1, 1, 1], atol=1e-6)
    # fully transparent texels don't bleed colour
    t2 = np.zeros((1, 2, 4), np.float32)
    t2[0, 0] = [1, 0, 0, 1]
    t2[0, 1] = [0, 0, 1, 0]
    im = R.render([R.ImageItem(t2, 0, 1, 2, 1)], 2, 1, view_px(2, 1))
    assert np.allclose(im[0, 0], [1, 0, 0, 1], atol=1e-6)
    # downscale averages
    big = np.zeros((16, 16, 4), np.float32)
    big[::2, ::2] = 1.0
    big[1::2, 1::2] = 1.0
    sm = R.render([R.ImageItem(big, 0, 4, 4, 4)], 4, 4, view_px(4, 4))
    assert np.allclose(sm[..., 3], 0.5, atol=1e-5)


def test_render_validation_and_unknown_item():
    with pytest.raises(TypeError):
        R.render([object()], 8, 8, view_px(8, 8))
    with pytest.raises(ValueError):
        R.render([], 0, 8, view_px(8, 8))
    with pytest.raises(ValueError):
        R.coverage([sq(0, 0, 1, 1)], 8, 8, view_px(8, 8), "SPIRAL")
    # offscreen and degenerate items are harmless
    img = R.render([R.FillItem([sq(100, 100, 120, 120)], RED), R.FillItem([[(0, 0), (1, 1)]], RED),
                    R.FillItem([], RED)], 8, 8, view_px(8, 8))
    assert not img.any()


def test_y_flip_convention():
    # canvas Y up: a square near canvas y_top should land in the top rows
    img = R.render([R.FillItem([sq(0, 90, 10, 100)], RED)], 100, 100, (0, 100, 100, 0))
    assert img[0:10, 0:10, 3].min() == 1.0 and img[10:, :, 3].sum() == 0.0 and img[:, 10:, 3].sum() == 0.0


def test_perf_1080p_20_shapes():
    rng = np.random.default_rng(0)
    items = []
    for i in range(20):
        cx, cy = rng.uniform(100, 1820), rng.uniform(100, 980)
        r = rng.uniform(50, 300)
        n = 48
        pts = [(cx + r * math.cos(2 * math.pi * k / n) * (1 + 0.3 * math.sin(5 * k)), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
        if i % 3 == 0:
            paint = R.LinearGradientPaint((cx - r, cy), (cx + r, cy), [R.GradientStop(0, (1, 0, 0, 1)), R.GradientStop(1, (0, 0, 1, 1))])
        elif i % 3 == 1:
            paint = R.RadialGradientPaint((cx, cy), r, [R.GradientStop(0, (1, 1, 0, 1)), R.GradientStop(1, (0, 0, 0, 1))])
        else:
            paint = R.SolidPaint((rng.random(), rng.random(), rng.random(), 1.0))
        items.append(R.FillItem([pts], paint, opacity=0.8))
    t = time.perf_counter()
    img = R.render(items, 1920, 1080, (0, 1080, 1920, 0), supersample=4)
    dt = time.perf_counter() - t
    assert img.shape == (1080, 1920, 4)
    assert dt < 5.0, dt
    d = os.path.join(ROOT, "artifacts", "perf")
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "core-raster-io.json")
        data = {}
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
        data["render2d_1920x1080_20shapes_ss4_seconds"] = round(dt, 3)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    except OSError:
        pass
