"""Pure mesh builders for 2D->3D recipes (owner: core-geometry).

The add-on uses native Blender curve evaluation for EXTRUDE/SWEEP/TEXT3D; these
builders are the reference/fallback implementations and the REVOLVE builder.
Coordinates produced here are in the *flat* frame: profile/shape in XY, extrusion
along +Z; :func:`orient` maps to UPRIGHT (x, y, z) -> (x, -z, y)... see docstring.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .geometry import Ring, Vec2

Vec3 = Tuple[float, float, float]

MAT_FACE = 0   # caps / front faces
MAT_SIDE = 1   # walls / swept surface


@dataclass
class MeshData:
    verts: List[Vec3] = field(default_factory=list)
    faces: List[Tuple[int, ...]] = field(default_factory=list)
    face_materials: List[int] = field(default_factory=list)  # one per face (MAT_FACE / MAT_SIDE)
    uvs: List[Tuple[float, float]] = field(default_factory=list)  # optional, one per face-corner (loop) in face order

    def validate(self) -> None:
        """Raise ValueError on out-of-range indices, faces < 3 verts, or length mismatches."""
        raise NotImplementedError


def extrude(shapes: Sequence[Tuple[Ring, List[Ring]]], depth: float, cap_front: bool = True,
            cap_back: bool = True, center: bool = True) -> MeshData:
    """Extrude normalized shapes along +Z by ``depth`` (centered on z=0 when ``center``).

    Caps are triangulated (geometry.triangulate) with holes; walls are quads with
    outward normals (consistent winding). Cap faces get MAT_FACE, walls MAT_SIDE.
    UVs: caps planar from bbox (0..1), walls (perimeter length / total, z / depth).
    """
    raise NotImplementedError


def revolve(profile: Sequence[Vec2], closed: bool, angle: float = 6.283185307179586, segments: int = 32,
            axis_x: float = 0.0, cap_ends: bool = True, merge_on_axis: bool = True) -> MeshData:
    """Lathe: rotate the profile (x = radius distance from ``axis_x``, y = height) around the
    vertical axis x = axis_x. Output in the flat frame with the revolution axis along +Z
    (profile y -> z, radius -> xy plane). Points on the axis are merged; partial angles get
    end caps when the profile is closed and ``cap_ends``. Normals point outward.
    Profile points with x < axis_x are mirrored (abs distance) and reported via ValueError only
    if the profile crosses the axis.
    """
    raise NotImplementedError


def sweep(path: Sequence[Vec2], closed: bool, radius: float, profile: str = "ROUND", profile_segments: int = 8,
          cap_ends: bool = True) -> MeshData:
    """Reference tube along a planar path (rotation-minimizing frames). profile: ROUND, SQUARE, RIBBON."""
    raise NotImplementedError


def orient(mesh: MeshData, orientation: str) -> MeshData:
    """FLAT: identity. UPRIGHT: (x, y, z) -> (x, -z, y) so the drawing faces -Y and stands on Z."""
    raise NotImplementedError


def place_origin(mesh: MeshData, mode: str, source_origin: Vec3 = (0.0, 0.0, 0.0)) -> MeshData:
    """Translate verts so the object origin is: CENTER (bbox center), BOTTOM (bbox center xy, min z),
    MIN (bbox min corner), SOURCE (``source_origin`` maps to 0)."""
    raise NotImplementedError


def scale(mesh: MeshData, s: float) -> MeshData:
    raise NotImplementedError


def mesh_signature(verts: Sequence[Vec3], faces: Sequence[Sequence[int]], decimals: int = 5) -> str:
    """Stable hex digest of rounded vertex coords + face topology (used for conflict detection)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(struct.pack("<II", len(verts), len(faces)))
    q = 10 ** decimals
    for v in verts:
        h.update(struct.pack("<qqq", *(int(round(c * q)) for c in v)))
    for f in faces:
        h.update(struct.pack("<I", len(f)))
        h.update(struct.pack("<%dI" % len(f), *f))
    return h.hexdigest()
