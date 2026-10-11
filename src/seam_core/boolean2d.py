"""2D polygon booleans (Pathfinder) via Blender's native CDT (owner: core-geometry).

Algorithm: feed every ring of A and B as an input face to
``mathutils.geometry.delaunay_2d_cdt(verts, edges, faces, output_type=1, epsilon, need_ids=True)``.
For each output triangle, ``orig_faces`` lists the input faces that contain it
(verified: overlap triangles report ``[0, 1]`` on Blender 4.0.2 and 5.2.2).
Membership in A = parity (EVENODD) or winding (NONZERO) over A's rings; same for B.
Select triangles by op, then extract boundary edges (edges whose two sides differ in
selection) and chain them into rings; normalize (outer CCW, holes CW).
Do NOT use output_type values above 2: their numbering differs between versions.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .geometry import Ring

OPS = ("UNION", "DIFFERENCE", "INTERSECT", "EXCLUDE")


class BooleanUnavailable(RuntimeError):
    """Raised when mathutils is not importable (outside Blender)."""


def available() -> bool:
    try:
        import mathutils.geometry  # noqa: F401
    except Exception:
        return False
    return True


def boolean(rings_a: Sequence[Sequence[Tuple[float, float]]], rings_b: Sequence[Sequence[Tuple[float, float]]],
            op: str, fill_rule: str = "EVENODD", epsilon: float = 1e-6) -> List[Tuple[Ring, List[Ring]]]:
    """Return normalized shapes [(outer_ccw, [holes_cw])] for A <op> B.

    DIFFERENCE = A minus B. EXCLUDE = symmetric difference. Empty result -> [].
    """
    raise NotImplementedError


def union_all(ring_sets: Sequence[Sequence[Sequence[Tuple[float, float]]]], fill_rule: str = "EVENODD") -> List[Tuple[Ring, List[Ring]]]:
    """Union of many compound shapes in one CDT pass."""
    raise NotImplementedError
