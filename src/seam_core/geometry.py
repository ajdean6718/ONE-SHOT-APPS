"""2D path geometry kernels (contract + implementation owned by core-geometry).

Conventions
-----------
* ``Vec2`` is ``(x, y)`` floats. A *ring* is a list of Vec2 **without** a repeated
  closing point. A *polyline* is an open list of Vec2.
* Positive signed area == counter-clockwise (canvas Y up).
* A normalized *Shape* is ``(outer, holes)`` with ``outer`` CCW and every hole CW.
* Affine 2D transforms are 6-tuples ``(a, b, c, d, e, f)`` mapping
  ``x' = a*x + c*y + e``, ``y' = b*x + d*y + f`` (SVG ``matrix(a b c d e f)`` order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Vec2 = Tuple[float, float]
Ring = List[Vec2]
Affine = Tuple[float, float, float, float, float, float]
IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

HANDLE_TYPES = ("FREE", "ALIGNED", "VECTOR", "AUTO")  # Blender bezier handle types


@dataclass
class BezierPoint:
    co: Vec2
    handle_left: Vec2
    handle_right: Vec2
    handle_left_type: str = "FREE"
    handle_right_type: str = "FREE"


@dataclass
class BezierPath:
    """One spline. Segment i goes from points[i] (handle_right) to points[i+1] (handle_left)."""

    points: List[BezierPoint] = field(default_factory=list)
    closed: bool = False


@dataclass
class Issue:
    code: str  # e.g. "SELF_INTERSECTION", "DEGENERATE", "OPEN_PATH", "TOO_FEW_POINTS", "ZERO_AREA"
    message: str
    path_index: int = -1
    point: Optional[Vec2] = None


def cubic_point(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: float) -> Vec2:
    """Evaluate a cubic Bezier at ``t``."""
    raise NotImplementedError


def flatten_path(path: BezierPath, resolution: int = 12) -> List[Vec2]:
    """Flatten a Bezier path into a polyline/ring.

    ``resolution`` = samples per segment (like Blender ``resolution_u``); straight
    segments (both handles on the chord / VECTOR handles) emit only their end point.
    Closed paths return a ring without the duplicated first point.
    """
    raise NotImplementedError


def flatten_paths(paths: Sequence[BezierPath], resolution: int = 12) -> List[List[Vec2]]:
    raise NotImplementedError


def signed_area(ring: Sequence[Vec2]) -> float:
    raise NotImplementedError


def is_ccw(ring: Sequence[Vec2]) -> bool:
    raise NotImplementedError


def bbox(rings: Sequence[Sequence[Vec2]]) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy); raises ValueError on empty input."""
    raise NotImplementedError


def path_length(polyline: Sequence[Vec2], closed: bool = False) -> float:
    raise NotImplementedError


def point_in_ring(pt: Vec2, ring: Sequence[Vec2]) -> bool:
    """Even-odd point-in-polygon test (boundary counts as outside or inside consistently)."""
    raise NotImplementedError


def winding_number(pt: Vec2, ring: Sequence[Vec2]) -> int:
    raise NotImplementedError


def transform_points(points: Sequence[Vec2], m: Affine) -> List[Vec2]:
    raise NotImplementedError


def affine_multiply(m1: Affine, m2: Affine) -> Affine:
    """Return m1 ∘ m2 (apply m2 first)."""
    raise NotImplementedError


def affine_determinant(m: Affine) -> float:
    raise NotImplementedError


def transform_path(path: BezierPath, m: Affine) -> BezierPath:
    raise NotImplementedError


def clean_ring(ring: Sequence[Vec2], eps: float = 1e-9) -> List[Vec2]:
    """Drop consecutive duplicates (and the closing duplicate) and exactly collinear points."""
    raise NotImplementedError


def normalize_rings(rings: Sequence[Sequence[Vec2]], fill_rule: str = "EVENODD") -> List[Tuple[Ring, List[Ring]]]:
    """Group rings into shapes by nesting depth.

    EVENODD: depth-even rings are outers, depth-odd rings are holes of their
    immediate container. NONZERO: uses winding numbers of the input orientation.
    Output orientation is normalized (outer CCW, holes CW). Degenerate rings
    (fewer than 3 points or |area| < 1e-12) are dropped.
    """
    raise NotImplementedError


def segment_intersection(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> Optional[Vec2]:
    """Proper intersection point of two segments, or None (touching endpoints excluded)."""
    raise NotImplementedError


def self_intersections(ring: Sequence[Vec2], closed: bool = True, limit: int = 50) -> List[Tuple[int, int, Vec2]]:
    """Find up to ``limit`` proper intersections between non-adjacent edges.

    Must be O(n log n)-ish or bounded so a 5k-point ring answers in < 200 ms.
    """
    raise NotImplementedError


def validate_rings(rings: Sequence[Sequence[Vec2]], require_closed_area: bool = True) -> List[Issue]:
    """Report issues that make a filled extrusion invalid (self-intersection, zero area, < 3 pts)."""
    raise NotImplementedError


def triangulate(outer: Sequence[Vec2], holes: Sequence[Sequence[Vec2]] = ()) -> Tuple[List[Vec2], List[Tuple[int, int, int]]]:
    """Triangulate a polygon with holes (ear clipping with hole bridging, earcut-style).

    Returns (vertices, triangles); vertices are outer points followed by each hole's points
    in input order; triangles are CCW index triples. Must handle collinear points,
    duplicate points and touching holes without crashing.
    """
    raise NotImplementedError
