"""Minimal PNG codec (owner: core-raster-io). Pure zlib + struct, no bpy.

Supports writing 8-bit RGBA/RGB and reading 8-bit (and 16-bit downconverted)
RGBA/RGB/grey/grey-alpha/palette non-interlaced PNGs with all five filter types.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def encode_png(rgba8: np.ndarray, compress_level: int = 6) -> bytes:
    """H x W x 4 (or x 3) uint8, row 0 = top -> PNG bytes."""
    a = np.asarray(rgba8)
    if a.dtype != np.uint8:
        raise ValueError("encode_png expects a uint8 array, got %s" % a.dtype)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("encode_png expects H x W x 3 or H x W x 4, got shape %r" % (a.shape,))
    h, w, c = a.shape
    if h < 1 or w < 1:
        raise ValueError("image must be at least 1x1")
    color_type = 6 if c == 4 else 2
    a = np.ascontiguousarray(a)
    raw = np.zeros((h, 1 + w * c), dtype=np.uint8)  # filter byte 0 (None) per row
    raw[:, 1:] = a.reshape(h, w * c)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (_SIG + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw.tobytes(), int(compress_level)))
            + _chunk(b"IEND", b""))


def _unfilter(raw: bytes, h: int, stride: int, bpp: int) -> np.ndarray:
    """Undo PNG row filters; None/Sub/Up are vectorised, Average/Paeth are per-byte."""
    expected = h * (stride + 1)
    if len(raw) < expected:
        raise ValueError("PNG image data truncated (%d < %d bytes)" % (len(raw), expected))
    buf = np.frombuffer(raw, dtype=np.uint8, count=expected).reshape(h, stride + 1)
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    npx = stride // bpp
    for y in range(h):
        ft = int(buf[y, 0])
        line = buf[y, 1:]
        if ft == 0:
            cur = line.copy()
        elif ft == 1:
            # Sub: cur[x] = line[x] + cur[x-1] per channel -> cumulative sum mod 256 over pixels
            cur = (np.cumsum(line.reshape(npx, bpp).astype(np.uint64), axis=0) & 0xFF).astype(np.uint8).reshape(stride)
        elif ft == 2:
            cur = (line.astype(np.uint16) + prev).astype(np.uint8)
        elif ft == 3:  # Average (sequential per byte; python lists are fastest here)
            ln = line.tolist()
            pv = prev.tolist()
            cl = [0] * stride
            for i in range(min(bpp, stride)):
                cl[i] = (ln[i] + (pv[i] >> 1)) & 0xFF
            for i in range(bpp, stride):
                cl[i] = (ln[i] + ((cl[i - bpp] + pv[i]) >> 1)) & 0xFF
            cur = np.array(cl, dtype=np.uint8)
        elif ft == 4:  # Paeth
            ln = line.tolist()
            pv = prev.tolist()
            cl = [0] * stride
            for i in range(min(bpp, stride)):
                cl[i] = (ln[i] + pv[i]) & 0xFF
            for i in range(bpp, stride):
                a_ = cl[i - bpp]
                b_ = pv[i]
                c_ = pv[i - bpp]
                p = a_ + b_ - c_
                pa = p - a_
                if pa < 0:
                    pa = -pa
                pb = p - b_
                if pb < 0:
                    pb = -pb
                pc = p - c_
                if pc < 0:
                    pc = -pc
                if pa <= pb and pa <= pc:
                    pr = a_
                elif pb <= pc:
                    pr = b_
                else:
                    pr = c_
                cl[i] = (ln[i] + pr) & 0xFF
            cur = np.array(cl, dtype=np.uint8)
        else:
            raise ValueError("invalid PNG filter type %d" % ft)
        out[y] = cur
        prev = cur
    return out


def _expand_bits(rows: np.ndarray, w: int, depth: int) -> np.ndarray:
    """Unpack sub-byte samples: rows H x stride -> H x w (values 0..2^depth-1)."""
    h = rows.shape[0]
    bits = np.unpackbits(rows, axis=1)  # H x (stride*8)
    per = bits.reshape(h, -1, depth)[:, :w, :]
    val = np.zeros((h, w), dtype=np.uint8)
    for k in range(depth):
        val = (val << 1) | per[:, :, k]
    return val


def decode_png(data: bytes) -> np.ndarray:
    """PNG bytes -> H x W x 4 uint8 (row 0 = top). Raises ValueError on unsupported/corrupt input."""
    data = bytes(data)
    if data[:8] != _SIG:
        raise ValueError("not a PNG file (bad signature)")
    pos = 8
    ihdr = None
    idat = []
    plte = None
    trns = None
    seen_iend = False
    while pos + 8 <= len(data):
        length, = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        if len(payload) != length or pos + 12 + length > len(data):
            raise ValueError("PNG chunk %r truncated" % tag)
        crc, = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if (zlib.crc32(tag + payload) & 0xFFFFFFFF) != crc:
            raise ValueError("PNG chunk %r has bad CRC" % tag)
        pos += 12 + length
        if tag == b"IHDR":
            if length != 13:
                raise ValueError("bad IHDR length")
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif tag == b"PLTE":
            plte = np.frombuffer(payload, dtype=np.uint8)
            if len(plte) % 3:
                raise ValueError("bad PLTE length")
            plte = plte.reshape(-1, 3)
        elif tag == b"tRNS":
            trns = payload
        elif tag == b"IDAT":
            idat.append(payload)
        elif tag == b"IEND":
            seen_iend = True
            break
    if ihdr is None:
        raise ValueError("PNG missing IHDR")
    if not idat:
        raise ValueError("PNG missing IDAT")
    if not seen_iend:
        raise ValueError("PNG missing IEND (truncated)")
    w, h, depth, ctype, comp, filt, interlace = ihdr
    if w < 1 or h < 1:
        raise ValueError("PNG has non-positive size")
    if comp != 0 or filt != 0:
        raise ValueError("unsupported PNG compression/filter method")
    if interlace != 0:
        raise ValueError("interlaced PNG is not supported")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ctype)
    if channels is None:
        raise ValueError("invalid PNG colour type %d" % ctype)
    allowed = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}[ctype]
    if depth not in allowed:
        raise ValueError("invalid bit depth %d for colour type %d" % (depth, ctype))
    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise ValueError("PNG image data corrupt: %s" % exc)
    bits_pp = channels * depth
    stride = (w * bits_pp + 7) // 8
    bpp = max(1, bits_pp // 8)
    rows = _unfilter(raw, h, stride, bpp)

    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., 3] = 255
    if depth == 16:
        # round-to-nearest from 16 to 8 bits
        full = (rows.reshape(h, w, channels, 2)[..., 0].astype(np.uint32) << 8) | rows.reshape(h, w, channels, 2)[..., 1]
        s = ((full * 255 + 32767) // 65535).astype(np.uint8)
        if ctype == 0:
            out[..., 0] = out[..., 1] = out[..., 2] = s[..., 0]
            if trns is not None and len(trns) >= 2:
                key = struct.unpack(">H", trns[:2])[0]
                out[..., 3] = np.where(full[..., 0] == key, 0, 255)
        elif ctype == 2:
            out[..., :3] = s
            if trns is not None and len(trns) >= 6:
                key = struct.unpack(">HHH", trns[:6])
                m = (full[..., 0] == key[0]) & (full[..., 1] == key[1]) & (full[..., 2] == key[2])
                out[..., 3] = np.where(m, 0, 255)
        elif ctype == 4:
            out[..., 0] = out[..., 1] = out[..., 2] = s[..., 0]
            out[..., 3] = s[..., 1]
        else:
            out[...] = s
        return out
    if depth == 8:
        px = rows.reshape(h, w, channels)
    else:
        px = _expand_bits(rows, w, depth)[..., None]
    if ctype == 3:
        if plte is None:
            raise ValueError("palette PNG missing PLTE")
        idx = px[..., 0].astype(np.int64)
        if idx.max(initial=0) >= len(plte):
            raise ValueError("palette index out of range")
        out[..., :3] = plte[idx]
        if trns is not None:
            alpha = np.full(len(plte), 255, dtype=np.uint8)
            n = min(len(trns), len(plte))
            alpha[:n] = np.frombuffer(trns[:n], dtype=np.uint8)
            out[..., 3] = alpha[idx]
        return out
    if ctype == 0:
        g = px[..., 0]
        if depth < 8:
            g_scaled = (g.astype(np.uint16) * 255 // ((1 << depth) - 1)).astype(np.uint8)
        else:
            g_scaled = g
        out[..., 0] = out[..., 1] = out[..., 2] = g_scaled
        if trns is not None and len(trns) >= 2:
            key = struct.unpack(">H", trns[:2])[0]
            out[..., 3] = np.where(g.astype(np.uint16) == key, 0, 255)
        return out
    if ctype == 2:
        out[..., :3] = px
        if trns is not None and len(trns) >= 6:
            key = struct.unpack(">HHH", trns[:6])
            m = (px[..., 0] == key[0]) & (px[..., 1] == key[1]) & (px[..., 2] == key[2])
            out[..., 3] = np.where(m, 0, 255)
        return out
    if ctype == 4:
        out[..., 0] = out[..., 1] = out[..., 2] = px[..., 0]
        out[..., 3] = px[..., 1]
        return out
    out[...] = px
    return out


def write_png(path: str, rgba8: np.ndarray) -> None:
    with open(path, "wb") as fh:
        fh.write(encode_png(rgba8))


def read_png(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        return decode_png(fh.read())
