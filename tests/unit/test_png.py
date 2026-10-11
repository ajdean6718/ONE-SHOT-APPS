import struct
import zlib

import numpy as np
import pytest

from seam_core import png


def _chunk(tag, payload):
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _filter_rows(pixels_bytes, h, stride, bpp, ftypes):
    """Reference PNG filter (encoder side) used to build test files."""
    rows = [list(pixels_bytes[y * stride:(y + 1) * stride]) for y in range(h)]
    out = bytearray()
    prev = [0] * stride
    for y, row in enumerate(rows):
        ft = ftypes[y % len(ftypes)]
        out.append(ft)
        for i in range(stride):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            x = row[i]
            if ft == 0:
                v = x
            elif ft == 1:
                v = x - a
            elif ft == 2:
                v = x - b
            elif ft == 3:
                v = x - ((a + b) >> 1)
            else:
                v = x - _paeth(a, b, c)
            out.append(v & 0xFF)
        prev = row
    return bytes(out)


def _make_png(w, h, ctype, depth, pixel_bytes, ftypes=(0,), extra=()):
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    bits = channels * depth
    stride = (w * bits + 7) // 8
    bpp = max(1, bits // 8)
    ihdr = struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0)
    data = _filter_rows(pixel_bytes, h, stride, bpp, ftypes)
    parts = [b"\x89PNG\r\n\x1a\n", _chunk(b"IHDR", ihdr)]
    for tag, payload in extra:
        parts.append(_chunk(tag, payload))
    parts.append(_chunk(b"IDAT", zlib.compress(data)))
    parts.append(_chunk(b"IEND", b""))
    return b"".join(parts)


def test_round_trip_rgba_exact():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(17, 23, 4), dtype=np.uint8)
    out = png.decode_png(png.encode_png(a))
    assert out.shape == (17, 23, 4) and out.dtype == np.uint8
    assert np.array_equal(out, a)


def test_round_trip_rgb_gets_opaque_alpha():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, size=(9, 31, 3), dtype=np.uint8)
    out = png.decode_png(png.encode_png(a))
    assert np.array_equal(out[..., :3], a)
    assert np.all(out[..., 3] == 255)


def test_round_trip_1x1_and_compress_levels():
    a = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
    for lvl in (0, 1, 9):
        assert np.array_equal(png.decode_png(png.encode_png(a, lvl)), a)


def test_encode_rejects_bad_input():
    with pytest.raises(ValueError):
        png.encode_png(np.zeros((2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        png.encode_png(np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        png.encode_png(np.zeros((0, 2, 4), dtype=np.uint8))


@pytest.mark.parametrize("ft", [0, 1, 2, 3, 4])
def test_decode_each_filter_type(ft):
    rng = np.random.default_rng(10 + ft)
    a = rng.integers(0, 256, size=(11, 13, 4), dtype=np.uint8)
    data = _make_png(13, 11, 6, 8, a.tobytes(), ftypes=(ft,))
    assert np.array_equal(png.decode_png(data), a)


def test_decode_mixed_filters_per_row_rgb():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, size=(20, 7, 3), dtype=np.uint8)
    data = _make_png(7, 20, 2, 8, a.tobytes(), ftypes=(1, 2, 3, 4, 0))
    out = png.decode_png(data)
    assert np.array_equal(out[..., :3], a) and np.all(out[..., 3] == 255)


def test_decode_grey_and_grey_alpha():
    g = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    out = png.decode_png(_make_png(4, 3, 0, 8, g.tobytes(), ftypes=(4,)))
    assert np.array_equal(out[..., 0], g) and np.array_equal(out[..., 1], g) and np.all(out[..., 3] == 255)
    ga = np.stack([g, 255 - g], axis=-1)
    out = png.decode_png(_make_png(4, 3, 4, 8, ga.tobytes(), ftypes=(1, 3)))
    assert np.array_equal(out[..., 0], g) and np.array_equal(out[..., 3], 255 - g)


def test_decode_palette_with_trns():
    pal = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])
    idx = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    data = _make_png(3, 2, 3, 8, idx.tobytes(), extra=[(b"PLTE", pal), (b"tRNS", bytes([128, 255]))])
    out = png.decode_png(data)
    assert tuple(out[0, 0]) == (255, 0, 0, 128)
    assert tuple(out[0, 1]) == (0, 255, 0, 255)
    assert tuple(out[0, 2]) == (0, 0, 255, 255)  # beyond tRNS => opaque


def test_decode_palette_4bit_packed():
    pal = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90])
    # width 3, 4 bits -> 2 bytes per row; indexes 0,1,2 / 2,1,0
    rows = bytes([0x01, 0x20, 0x21, 0x00])
    out = png.decode_png(_make_png(3, 2, 3, 4, rows, extra=[(b"PLTE", pal)]))
    assert tuple(out[0, 2, :3]) == (70, 80, 90)
    assert tuple(out[1, 0, :3]) == (70, 80, 90)
    assert tuple(out[1, 2, :3]) == (10, 20, 30)


def test_decode_grey_1bit():
    rows = bytes([0b10100000, 0b01000000])
    out = png.decode_png(_make_png(3, 2, 0, 1, rows))
    assert [int(v) for v in out[0, :, 0]] == [255, 0, 255]
    assert [int(v) for v in out[1, :, 0]] == [0, 255, 0]


def test_decode_16bit_downconverts():
    w, h = 2, 1
    vals = np.array([[0x0000, 0x8000, 0xFFFF, 0xFFFF], [0xFFFF, 0x0000, 0x0000, 0x0000]], dtype=">u2")
    data = _make_png(w, h, 6, 16, vals.tobytes(), ftypes=(4,))
    out = png.decode_png(data)
    assert tuple(out[0, 0]) == (0, 128, 255, 255)
    assert tuple(out[0, 1]) == (255, 0, 0, 0)


def test_decode_errors():
    good = png.encode_png(np.zeros((2, 2, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        png.decode_png(b"not a png at all")
    with pytest.raises(ValueError):
        png.decode_png(good[:-20])  # truncated
    bad = bytearray(good)
    bad[20] ^= 0xFF  # corrupt IHDR payload -> crc mismatch
    with pytest.raises(ValueError):
        png.decode_png(bytes(bad))
    # interlaced unsupported
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 1)
    inter = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(b"\0\0\0\0\0")) + _chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="interlaced"):
        png.decode_png(inter)
    # bad filter type
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    badf = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(b"\x09\0\0\0\0")) + _chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="filter"):
        png.decode_png(badf)


def test_file_round_trip(tmp_path):
    a = np.random.default_rng(5).integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
    p = str(tmp_path / "x.png")
    png.write_png(p, a)
    assert np.array_equal(png.read_png(p), a)


def test_decode_performance_paeth_medium():
    rng = np.random.default_rng(6)
    a = rng.integers(0, 256, size=(256, 256, 4), dtype=np.uint8)
    data = _make_png(256, 256, 6, 8, a.tobytes(), ftypes=(4,))
    import time
    t = time.time()
    out = png.decode_png(data)
    assert np.array_equal(out, a)
    assert time.time() - t < 5.0
