"""Parametric vector primitives -> BezierPath (owner: core-geometry).

All primitives are centered on (cx, cy) unless noted, CCW, closed (except line),
and use exact Bezier handles: corners use VECTOR handles, ellipse arcs use the
kappa = 0.5522847498 cubic approximation with ALIGNED handles.
"""

from __future__ import annotations

from typing import List

from .geometry import BezierPath, Vec2

KAPPA = 0.5522847498307936


def rect(width: float, height: float, cx: float = 0.0, cy: float = 0.0) -> BezierPath:
    raise NotImplementedError


def rounded_rect(width: float, height: float, radius: float, cx: float = 0.0, cy: float = 0.0) -> BezierPath:
    """Radius is clamped to min(width, height)/2; radius <= 0 returns rect()."""
    raise NotImplementedError


def ellipse(rx: float, ry: float, cx: float = 0.0, cy: float = 0.0) -> BezierPath:
    """Four-point Bezier ellipse."""
    raise NotImplementedError


def polygon(sides: int, radius: float, cx: float = 0.0, cy: float = 0.0, rotation: float = 0.0) -> BezierPath:
    """Regular polygon, first vertex pointing up (+Y) when rotation == 0. sides >= 3."""
    raise NotImplementedError


def star(points: int, radius: float, inner_ratio: float = 0.5, cx: float = 0.0, cy: float = 0.0, rotation: float = 0.0) -> BezierPath:
    """Star with ``points`` outer vertices (>= 3), inner radius = radius * inner_ratio."""
    raise NotImplementedError


def line(p0: Vec2, p1: Vec2) -> BezierPath:
    """Open two-point path."""
    raise NotImplementedError


def from_kind(kind: str, width: float, height: float, radius: float = 0.0, sides: int = 5,
              inner_ratio: float = 0.5, cx: float = 0.0, cy: float = 0.0) -> BezierPath:
    """Dispatch on SEAM_PG_ShapeRecipe.kind: RECT, ROUNDED_RECT, ELLIPSE, POLYGON, STAR, LINE.

    POLYGON/STAR use radius = min(width, height)/2 and are scaled to fit width x height.
    LINE goes from (cx - width/2, cy) to (cx + width/2, cy).
    """
    raise NotImplementedError
