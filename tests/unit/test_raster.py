import colorsys

import numpy as np
import pytest

from seam_core import raster as R


def rnd(shape, seed=0):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


# ---- colour space ------------------------------------------------------------------------------

def test_srgb_linear_inverse_1e6():
    x = np.linspace(0.0, 1.0, 20001, dtype=np.float64)
    assert np.max(np.abs(R.linear_to_srgb(R.srgb_to_linear(x)) - x)) < 1e-6
    assert np.max(np.abs(R.srgb_to_linear(R.linear_to_srgb(x)) - x)) < 1e-6
    x32 = x.astype(np.float32)
    assert np.max(np.abs(R.linear_to_srgb(R.srgb_to_linear(x32)) - x32)) < 1e-6


def test_srgb_known_values_and_alpha_untouched():
    assert R.srgb_to_linear(np.array([0.0, 1.0]))[0] == 0.0
    assert abs(float(R.srgb_to_linear(np.array([0.5]))[0]) - 0.21404114) < 1e-6
    assert abs(float(R.srgb_to_linear(np.array([0.04045]))[0]) - 0.04045 / 12.92) < 1e-9
    img = np.array([[[0.5, 0.5, 0.5, 0.3]]], dtype=np.float32)
    out = R.srgb_to_linear(img)
    assert out[0, 0, 3] == np.float32(0.3) and abs(out[0, 0, 0] - 0.21404114) < 1e-6
    out2 = R.linear_to_srgb(out)
    assert out2[0, 0, 3] == np.float32(0.3) and abs(out2[0, 0, 0] - 0.5) < 1e-6


def test_uint8_round_trip_all_values():
    v = np.arange(256, dtype=np.uint8)
    img = np.stack([v, v[::-1], v, v], axis=-1).reshape(16, 16, 4)
    lin = R.from_uint8_srgb(img)
    assert lin.dtype == np.float32
    back = R.to_uint8_srgb(lin)
    assert back.dtype == np.uint8 and np.array_equal(back[..., :3], img[..., :3])
    # alpha is linear: 255 -> 1.0, 128 -> ~128/255 (not gamma-decoded)
    assert abs(lin[..., 3].reshape(-1)[128] - 128 / 255) < 1e-6
    assert np.array_equal(back[..., 3], img[..., 3])


def test_to_uint8_clips_and_rounds():
    px = np.array([[[2.0, -1.0, 0.5, 1.5]]], dtype=np.float32)
    out = R.to_uint8_srgb(px)
    assert out[0, 0, 0] == 255 and out[0, 0, 1] == 0 and out[0, 0, 3] == 255


# ---- blend modes -------------------------------------------------------------------------------

def ref_blend(mode, c1, c2):
    """Independent scalar reference transcribed from Blender's Mix node (GLSL/Cycles svm)."""
    c1 = list(map(float, c1)); c2 = list(map(float, c2))
    out = list(c1)
    if mode == "NORMAL":
        return c2
    if mode == "MULTIPLY":
        return [a * b for a, b in zip(c1, c2)]
    if mode == "SCREEN":
        return [1 - (1 - b) * (1 - a) for a, b in zip(c1, c2)]
    if mode == "OVERLAY":
        return [a * (2 * b) if a < 0.5 else 1 - (2 * (1 - b)) * (1 - a) for a, b in zip(c1, c2)]
    if mode == "DARKEN":
        return [min(a, b) for a, b in zip(c1, c2)]
    if mode == "LIGHTEN":
        return [max(a, b) for a, b in zip(c1, c2)]
    if mode == "COLOR_DODGE":
        res = []
        for a, b in zip(c1, c2):
            if a != 0:
                tmp = 1 - b
                if tmp <= 0:
                    a = 1.0
                else:
                    tmp = a / tmp
                    a = 1.0 if tmp > 1 else tmp
            res.append(a)
        return res
    if mode == "COLOR_BURN":
        res = []
        for a, b in zip(c1, c2):
            tmp = b
            if tmp <= 0:
                a = 0.0
            else:
                tmp = 1 - (1 - a) / tmp
                a = 0.0 if tmp < 0 else (1.0 if tmp > 1 else tmp)
            res.append(a)
        return res
    if mode == "ADD":
        return [a + b for a, b in zip(c1, c2)]
    if mode == "SUBTRACT":
        return [a - b for a, b in zip(c1, c2)]
    if mode == "DIFFERENCE":
        return [abs(a - b) for a, b in zip(c1, c2)]
    if mode == "EXCLUSION":
        return [max(a + b - 2 * a * b, 0.0) for a, b in zip(c1, c2)]
    if mode == "SOFT_LIGHT":
        res = []
        for a, b in zip(c1, c2):
            scr = 1 - (1 - b) * (1 - a)
            res.append((1 - a) * b * a + a * scr)
        return res
    if mode == "LINEAR_LIGHT":
        return [a + 2 * (b - 0.5) for a, b in zip(c1, c2)]
    if mode == "DIVIDE":
        return [a / b if b != 0 else a for a, b in zip(c1, c2)]
    h1, s1, v1 = colorsys.rgb_to_hsv(*c1)
    h2, s2, v2 = colorsys.rgb_to_hsv(*c2)
    if mode == "HUE":
        return list(colorsys.hsv_to_rgb(h2, s1, v1)) if s2 != 0 else c1
    if mode == "SATURATION":
        return list(colorsys.hsv_to_rgb(h1, s2, v1)) if s1 != 0 else c1
    if mode == "COLOR":
        return list(colorsys.hsv_to_rgb(h2, s2, v1)) if s2 != 0 else c1
    if mode == "VALUE":
        return list(colorsys.hsv_to_rgb(h1, s1, v2))
    raise AssertionError(mode)


@pytest.mark.parametrize("mode", R.BLEND_MODES)
def test_every_mode_matches_scalar_reference(mode):
    rng = np.random.default_rng(42)
    a = rng.random((200, 3))
    b = rng.random((200, 3))
    # include edge cases: zeros, ones, equals, greys
    a[:4] = [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]
    b[:4] = [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0.8, 0.1, 0.1]]
    out = R.blend_rgb(mode, a.astype(np.float32), b.astype(np.float32))
    assert out.shape == a.shape
    ref = np.array([ref_blend(mode, x, y) for x, y in zip(a.astype(np.float32), b.astype(np.float32))])
    assert np.max(np.abs(out - ref)) < 2e-5


def test_blend_shapes_and_unknown_mode():
    a = rnd((4, 5, 3), 1)
    b = rnd((4, 5, 3), 2)
    assert R.blend_rgb("MULTIPLY", a, b).shape == (4, 5, 3)
    assert R.blend_rgb("MULTIPLY", a[0, 0], b[0, 0]).shape == (3,)
    with pytest.raises(ValueError):
        R.blend_rgb("NOPE", a, b)


def test_blender_mix_type_covers_all_modes():
    assert set(R.BLENDER_MIX_TYPE) == set(R.BLEND_MODES)
    assert R.BLENDER_MIX_TYPE["NORMAL"] == "MIX"
    assert R.BLENDER_MIX_TYPE["COLOR_DODGE"] == "DODGE" and R.BLENDER_MIX_TYPE["COLOR_BURN"] == "BURN"


def test_specific_mode_values():
    one = lambda mode, a, b: R.blend_rgb(mode, np.array(a, np.float32), np.array(b, np.float32))
    assert np.allclose(one("MULTIPLY", [0.5, 0.5, 0.5], [0.5, 0.2, 1.0]), [0.25, 0.1, 0.5])
    assert np.allclose(one("SCREEN", [0.5, 0.5, 0.5], [0.5, 0.0, 1.0]), [0.75, 0.5, 1.0])
    assert np.allclose(one("DIVIDE", [0.5, 0.5, 0.5], [0.0, 0.5, 1.0]), [0.5, 1.0, 0.5])
    assert np.allclose(one("SUBTRACT", [0.2, 0.2, 0.2], [0.5, 0.0, 0.1]), [-0.3, 0.2, 0.1])  # no clamp
    assert np.allclose(one("HUE", [0.5, 0.4, 0.3], [0.7, 0.7, 0.7]), [0.5, 0.4, 0.3])  # grey source: unchanged
    assert np.allclose(one("VALUE", [0.2, 0.0, 0.0], [0.8, 0.8, 0.8]), [0.8, 0.0, 0.0])


# ---- composite ---------------------------------------------------------------------------------

def test_composite_normal_alpha_over_hand_computed():
    base = np.zeros((1, 2, 4), np.float32)
    base[0, 0] = [1.0, 0.0, 0.0, 1.0]
    base[0, 1] = [0.0, 0.0, 0.0, 0.0]
    top = np.zeros((1, 2, 4), np.float32)
    top[0, 0] = [0.0, 1.0, 0.0, 0.5]
    top[0, 1] = [0.0, 0.0, 1.0, 0.5]
    out = R.composite([R.LayerSpec(base), R.LayerSpec(top)], 1, 2)
    assert np.allclose(out[0, 0], [0.5, 0.5, 0.0, 1.0], atol=1e-6)
    assert np.allclose(out[0, 1], [0.0, 0.0, 1.0, 0.5], atol=1e-6)


def test_composite_opacity_mask_multiply_factor():
    base = np.tile(np.array([1.0, 1.0, 1.0, 1.0], np.float32), (1, 4, 1))
    top = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (1, 4, 1))
    mask = np.array([[1.0, 0.5, 0.0, 1.0]], np.float32)
    out = R.composite([R.LayerSpec(base), R.LayerSpec(top, opacity=0.5, mask=mask)], 1, 4)
    expect = [0.5, 0.75, 1.0, 0.5]
    assert np.allclose(out[0, :, 0], expect, atol=1e-6)
    assert np.allclose(out[0, :, 3], 1.0)


def test_composite_alpha_accumulates_on_transparent():
    a = np.tile(np.array([1, 0, 0, 0.5], np.float32), (2, 2, 1))
    b = np.tile(np.array([1, 0, 0, 0.5], np.float32), (2, 2, 1))
    out = R.composite([R.LayerSpec(a), R.LayerSpec(b)], 2, 2)
    assert np.allclose(out[..., 3], 0.75)


def test_composite_multiply_blend_and_hidden_layer_and_background():
    base = np.tile(np.array([0.5, 0.5, 0.5, 1.0], np.float32), (2, 2, 1))
    top = np.tile(np.array([0.5, 0.2, 1.0, 1.0], np.float32), (2, 2, 1))
    out = R.composite([R.LayerSpec(base), R.LayerSpec(top, blend_mode="MULTIPLY")], 2, 2)
    assert np.allclose(out[0, 0], [0.25, 0.1, 0.5, 1.0], atol=1e-6)
    hidden = R.composite([R.LayerSpec(base), R.LayerSpec(top, visible=False)], 2, 2)
    assert np.allclose(hidden, base)
    bg = R.composite([R.LayerSpec(top, opacity=0.0)], 2, 2, background=(1, 1, 1, 1))
    assert np.allclose(bg[..., :3], 1.0) and np.allclose(bg[..., 3], 1.0)
    empty = R.composite([], 3, 2)
    assert empty.shape == (3, 2, 4) and not empty.any()


def test_composite_validates():
    with pytest.raises(ValueError):
        R.composite([R.LayerSpec(np.zeros((2, 2, 4), np.float32), blend_mode="BAD")], 2, 2)
    with pytest.raises(ValueError):
        R.composite([R.LayerSpec(np.zeros((3, 2, 4), np.float32))], 2, 2)


# ---- selections --------------------------------------------------------------------------------

def test_rect_mask_exact_and_aa():
    m = R.rect_mask(10, 12, 2, 3, 7, 8)
    assert m.shape == (10, 12) and m.sum() == pytest.approx(25.0)
    assert m[3:8, 2:7].min() == 1.0 and m.sum() == m[3:8, 2:7].sum()
    m = R.rect_mask(10, 10, 2.5, 2.5, 6.5, 5.0)
    assert m.sum() == pytest.approx(4.0 * 2.5)
    assert m[2, 2] == pytest.approx(0.25)
    na = R.rect_mask(10, 10, 2.4, 2.4, 6.6, 5.0, antialias=False)
    assert set(np.unique(na)) <= {0.0, 1.0}
    assert na.sum() == 5 * 3  # centres 2.5..6.5 in x (5 px), 2.5, 3.5, 4.5 in y
    assert R.rect_mask(10, 10, 7, 7, 3, 3).sum() == pytest.approx(16.0)  # reversed corners normalised


def test_ellipse_mask_area_within_1pct():
    for (w, h, box) in ((200, 150, (20, 10, 180, 140)), (64, 64, (3.3, 4.1, 59.7, 50.2)), (500, 300, (10, 10, 490, 290))):
        m = R.ellipse_mask(h, w, *box)
        rx, ry = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
        analytic = np.pi * rx * ry
        assert abs(float(m.sum()) - analytic) / analytic < 0.01
    m = R.ellipse_mask(10, 10, 0, 0, 10, 10, antialias=False)
    assert set(np.unique(m)) <= {0.0, 1.0}


def test_polygon_mask_area_within_1pct():
    tri = [(10, 10), (190, 20), (60, 140)]
    m = R.polygon_mask(160, 200, tri)
    area = 0.5 * abs((tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1]))
    assert abs(float(m.sum()) - area) / area < 0.01
    # bowtie self-overlap handled even-odd: square with inner square hole
    outer = [(0, 0), (50, 0), (50, 50), (0, 50)]
    inner = [(20, 20), (30, 20), (30, 30), (20, 30)]
    m = R.polygon_mask(50, 50, outer + [outer[0]] + inner)  # one ring - just ensure no crash
    assert m.shape == (50, 50)
    assert R.polygon_mask(5, 5, [(0, 0), (1, 1)]).sum() == 0


def test_polygon_mask_clips_to_canvas():
    m = R.polygon_mask(20, 20, [(-10, -10), (30, -10), (30, 30), (-10, 30)])
    assert m.min() == 1.0 and m.shape == (20, 20)


def test_combine_selection_ops():
    a = R.rect_mask(10, 10, 0, 0, 6, 10)
    b = R.rect_mask(10, 10, 4, 0, 10, 10)
    assert np.array_equal(R.combine_selection(None, b, "REPLACE"), b)
    assert np.array_equal(R.combine_selection(a, b, "REPLACE"), b)
    assert R.combine_selection(a, b, "ADD").sum() == 100
    assert R.combine_selection(a, b, "SUBTRACT").sum() == 40
    assert R.combine_selection(a, b, "INTERSECT").sum() == 20
    assert R.combine_selection(None, b, "ADD").sum() == b.sum()
    assert R.combine_selection(None, b, "SUBTRACT").sum() == 0
    half = np.full((10, 10), 0.5, np.float32)
    assert np.allclose(R.combine_selection(half, half, "SUBTRACT"), 0.25)
    with pytest.raises(ValueError):
        R.combine_selection(a, b, "XOR")


def test_clear_selected_and_apply_mask():
    img = np.tile(np.array([0.2, 0.4, 0.6, 1.0], np.float32), (2, 2, 1))
    m = np.array([[1.0, 0.5], [0.0, 0.25]], np.float32)
    c = R.clear_selected(img, m)
    assert np.allclose(c[..., 3], [[0.0, 0.5], [1.0, 0.75]])
    assert np.allclose(c[..., :3], img[..., :3]) and img[0, 0, 3] == 1.0  # input untouched
    k = R.apply_mask_to_alpha(img, m)
    assert np.allclose(k[..., 3], m)


# ---- fill / flood ------------------------------------------------------------------------------

def make_split():
    img = np.zeros((6, 8, 4), np.float32)
    img[..., 3] = 1.0
    img[:, 4:, :3] = 1.0   # right half white, left half black
    img[:, 2, :3] = 1.0    # a white wall inside the left half
    return img


def test_flood_fill_contiguous_vs_global():
    img = make_split()
    contig = R.flood_fill_mask(img, 0, 0, 0.01, True)
    glob = R.flood_fill_mask(img, 0, 0, 0.01, False)
    assert contig.sum() == 12  # columns 0-1 only (wall blocks), 6 rows
    assert glob.sum() == 6 * 3  # columns 0, 1, 3 are black; global sees column 3 too
    assert contig.dtype == bool and glob.dtype == bool
    assert not contig[:, 3].any() and glob[:, 3].all()
    white = R.flood_fill_mask(img, 5, 3, 0.01, True)
    # wall (col 2) is separated from the right half by black column 3
    assert white.sum() == 6 * 4 and white[:, 4:].all() and not white[:, :4].any()


def test_flood_fill_tolerance_and_alpha():
    img = np.zeros((1, 4, 4), np.float32)
    img[0, :, 3] = 1.0
    img[0, 0, :3] = 0.00
    img[0, 1, :3] = 0.04
    img[0, 2, :3] = 0.08
    img[0, 3, 3] = 0.0
    m = R.flood_fill_mask(img, 0, 0, 0.05)
    assert list(m[0]) == [True, True, False, False]
    m = R.flood_fill_mask(img, 0, 0, 0.1)
    assert list(m[0]) == [True, True, True, False]  # alpha differs by 1.0
    with pytest.raises(IndexError):
        R.flood_fill_mask(img, 9, 0)


def test_flood_fill_snake_region_connectivity():
    img = np.ones((5, 5, 4), np.float32)
    img[..., 3] = 1
    # black snake path: (0,0)-(0,4) top row then down right column
    img[0, :, :3] = 0
    img[:, 4, :3] = 0
    m = R.flood_fill_mask(img, 0, 0, 0.0)
    assert m.sum() == 5 + 4
    # diagonal neighbours are not 4-connected
    d = np.ones((3, 3, 4), np.float32)
    d[0, 0, :3] = 0
    d[1, 1, :3] = 0
    assert R.flood_fill_mask(d, 0, 0, 0.0).sum() == 1
    assert R.flood_fill_mask(d, 0, 0, 0.0, contiguous=False).sum() == 2


def test_fill_opaque_partial_and_input_unchanged():
    img = np.zeros((2, 2, 4), np.float32)
    mask = np.array([[1.0, 0.5], [0.0, 1.0]], np.float32)
    out = R.fill(img, mask, (1.0, 0.5, 0.25, 1.0))
    assert np.allclose(out[0, 0], [1.0, 0.5, 0.25, 1.0])
    assert np.allclose(out[0, 1], [1.0, 0.5, 0.25, 0.5])  # straight alpha on empty pixel
    assert np.allclose(out[1, 0], 0.0)
    assert not img.any()
    over = R.fill(np.tile(np.array([0, 0, 1, 1], np.float32), (1, 1, 1)), np.array([[True]]), (1, 0, 0, 1), opacity=0.5)
    assert np.allclose(over[0, 0], [0.5, 0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        R.fill(img, np.ones((3, 3)), (1, 1, 1, 1))


# ---- misc ---------------------------------------------------------------------------------------

def test_resize_nearest():
    a = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    up = R.resize_nearest(a, 8, 8)
    assert up.shape == (8, 8, 4) and np.array_equal(up[0, 0], a[0, 0]) and np.array_equal(up[1, 1], a[0, 0])
    assert np.array_equal(up[7, 7], a[3, 3])
    down = R.resize_nearest(a, 2, 2)
    assert np.array_equal(down[0, 0], a[1, 1]) and down.shape == (2, 2, 4)


def test_blender_pixel_order_round_trip():
    w, h = 5, 3
    arr = rnd((h, w, 4), 7)
    flat = R.array_to_blender_pixels(arr)
    assert flat.dtype == np.float32 and flat.ndim == 1 and flat.size == w * h * 4
    # Blender: first row of the buffer is the BOTTOM row of the image
    assert np.array_equal(flat[:w * 4].reshape(w, 4), arr[-1])
    back = R.blender_pixels_to_array(flat, w, h)
    assert back.dtype == np.float32 and np.array_equal(back, arr)
    assert np.array_equal(R.blender_pixels_to_array(list(flat), w, h), arr)
    with pytest.raises(ValueError):
        R.blender_pixels_to_array(flat[:-1], w, h)
