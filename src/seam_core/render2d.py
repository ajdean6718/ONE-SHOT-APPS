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


def coverage(rings: Sequence[Sequence[Tuple[float, float]]], height: int, width: int,
             view: Tuple[float, float, float, float], fill_rule: str = "EVENODD", supersample: int = 4) -> np.ndarray:
    """Float H x W coverage 0..1 of the rings."""
    raise NotImplementedError


def render(items: Sequence[object], width: int, height: int, view: Tuple[float, float, float, float],
           supersample: int = 4, background: Optional[RGBA] = None) -> np.ndarray:
    """Composite items with normal alpha-over in linear space; returns H x W x 4 linear float32."""
    raise NotImplementedError
