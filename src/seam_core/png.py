"""Minimal PNG codec (owner: core-raster-io). Pure zlib + struct, no bpy.

Supports writing 8-bit RGBA/RGB and reading 8-bit (and 16-bit downconverted)
RGBA/RGB/grey/grey-alpha/palette non-interlaced PNGs with all five filter types.
"""

from __future__ import annotations

import numpy as np


def encode_png(rgba8: np.ndarray, compress_level: int = 6) -> bytes:
    """H x W x 4 (or x 3) uint8, row 0 = top -> PNG bytes."""
    raise NotImplementedError


def decode_png(data: bytes) -> np.ndarray:
    """PNG bytes -> H x W x 4 uint8 (row 0 = top). Raises ValueError on unsupported/corrupt input."""
    raise NotImplementedError


def write_png(path: str, rgba8: np.ndarray) -> None:
    with open(path, "wb") as fh:
        fh.write(encode_png(rgba8))


def read_png(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        return decode_png(fh.read())
