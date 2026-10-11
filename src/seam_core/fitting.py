"""Freehand stroke smoothing and Bezier fitting (owner: core-geometry)."""

from __future__ import annotations

from typing import List, Sequence

from .geometry import BezierPath, Vec2


def resample(points: Sequence[Vec2], spacing: float) -> List[Vec2]:
    """Resample a polyline at roughly uniform arc-length ``spacing`` (keeps endpoints)."""
    raise NotImplementedError


def smooth_polyline(points: Sequence[Vec2], strength: float = 0.5, iterations: int = 2, closed: bool = False) -> List[Vec2]:
    """Laplacian/Chaikin-style smoothing; strength 0..1; endpoints fixed when open."""
    raise NotImplementedError


def simplify(points: Sequence[Vec2], tolerance: float) -> List[Vec2]:
    """Ramer–Douglas–Peucker."""
    raise NotImplementedError


def fit_bezier(points: Sequence[Vec2], max_error: float = 1.0, closed: bool = False) -> BezierPath:
    """Schneider's algorithm: fit a piecewise cubic Bezier through ``points``.

    Corners (turn angle > 70°) split segments and get FREE handles; smooth joins get ALIGNED.
    Returns a path with >= 2 points (or a closed path with >= 3).
    """
    raise NotImplementedError
