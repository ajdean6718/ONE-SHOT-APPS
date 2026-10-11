"""Stroke outlining for vector paths (owner: core-geometry).

Two outputs: a triangle mesh for viewport display (overlaps allowed), and outline
rings for export rasterization (fill with NONZERO rule).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .geometry import Vec2

CAPS = ("BUTT", "ROUND", "SQUARE")
JOINS = ("MITER", "ROUND", "BEVEL")


def stroke_mesh(polyline: Sequence[Vec2], closed: bool, width: float, cap: str = "BUTT", join: str = "MITER",
                miter_limit: float = 4.0, round_segments: int = 8) -> Tuple[List[Vec2], List[Tuple[int, int, int]]]:
    """Triangulated stroke band (segments + joins + caps). width <= 0 -> empty."""
    raise NotImplementedError


def stroke_outline(polyline: Sequence[Vec2], closed: bool, width: float, cap: str = "BUTT", join: str = "MITER",
                   miter_limit: float = 4.0, round_segments: int = 8) -> List[List[Vec2]]:
    """Rings whose NONZERO union is the stroked area (segment quads + join wedges + cap pieces are fine)."""
    raise NotImplementedError
