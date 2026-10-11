"""bpy <-> seam_core conversions shared by every domain (owner: integrator).

* Curve splines are read from ``curve.splines``; in curve edit mode Blender returns the
  live edit data there (verified on 4.0.2 and 5.2.2), so no mode switching is needed.
* All returned geometry is 2D (canvas plane XY); Z is ignored.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import List, Optional, Sequence

import bpy
from mathutils import Matrix

from .seam_core import CANVAS_ORIGIN
from .seam_core.geometry import Affine, BezierPath, BezierPoint, IDENTITY


# ------------------------------------------------------------------ canvas

def canvas_root(scene=None):
    scene = scene or bpy.context.scene
    for ob in scene.objects:
        if ob.seam.role == "CANVAS_ROOT":
            return ob
    return None


def canvas_matrix(scene=None) -> Matrix:
    root = canvas_root(scene)
    if root is not None:
        return root.matrix_world.copy()
    return Matrix.Translation(CANVAS_ORIGIN)


def canvas_local_matrix(obj, scene=None) -> Matrix:
    """4x4 matrix mapping the object's local space into canvas-local space."""
    return canvas_matrix(scene).inverted_safe() @ obj.matrix_world


def matrix_to_affine(m: Matrix) -> Affine:
    """Drop Z from a 4x4 matrix -> 2D affine (a, b, c, d, e, f)."""
    return (m[0][0], m[1][0], m[0][1], m[1][1], m[0][3], m[1][3])


def object_canvas_affine(obj, scene=None) -> Affine:
    return matrix_to_affine(canvas_local_matrix(obj, scene))


def object_shape_affine(obj, scene=None) -> Affine:
    """Canvas-local 2D rotation/scale of the object with translation removed (link geometry frame)."""
    a, b, c, d, _e, _f = object_canvas_affine(obj, scene)
    return (a, b, c, d, 0.0, 0.0)


# ------------------------------------------------------------------ curves

def _v2(v) -> tuple:
    return (float(v[0]), float(v[1]))


def spline_to_path(spline) -> Optional[BezierPath]:
    if spline.type == "BEZIER":
        pts = [BezierPoint(_v2(bp.co), _v2(bp.handle_left), _v2(bp.handle_right),
                           bp.handle_left_type, bp.handle_right_type) for bp in spline.bezier_points]
    else:  # POLY / NURBS: treat control points as corners (NURBS approximated by its control polygon)
        pts = []
        for p in spline.points:
            xy = (float(p.co[0]), float(p.co[1]))
            pts.append(BezierPoint(xy, xy, xy, "VECTOR", "VECTOR"))
    if not pts:
        return None
    return BezierPath(points=pts, closed=bool(spline.use_cyclic_u))


def curve_paths(obj) -> List[BezierPath]:
    """Bezier paths of a CURVE object in object-local coordinates (live in edit mode)."""
    if obj is None or obj.type != "CURVE":
        return []
    out = []
    for sp in obj.data.splines:
        p = spline_to_path(sp)
        if p is not None:
            out.append(p)
    return out


def text_paths(obj, depsgraph=None) -> List[BezierPath]:
    """Outline paths of a FONT object (object-local), via evaluated ``to_curve``."""
    if obj is None or obj.type != "FONT":
        return []
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(depsgraph)
    cu = ev.to_curve(depsgraph)
    try:
        return [p for p in (spline_to_path(sp) for sp in cu.splines) if p is not None]
    finally:
        ev.to_curve_clear()


def object_paths(obj, depsgraph=None) -> List[BezierPath]:
    if obj is None:
        return []
    if obj.type == "CURVE":
        return curve_paths(obj)
    if obj.type == "FONT":
        return text_paths(obj, depsgraph)
    return []


def transform_bezier_paths(paths: Sequence[BezierPath], m: Affine) -> List[BezierPath]:
    if m == IDENTITY:
        return list(paths)
    a, b, c, d, e, f = m

    def t(p):
        return (a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f)

    return [BezierPath([BezierPoint(t(pt.co), t(pt.handle_left), t(pt.handle_right),
                                    pt.handle_left_type, pt.handle_right_type) for pt in path.points], path.closed)
            for path in paths]


def paths_signature(paths: Sequence[BezierPath], extra: bytes = b"", decimals: int = 5) -> str:
    h = hashlib.blake2b(digest_size=12)
    q = 10 ** decimals
    h.update(struct.pack("<I", len(paths)))
    for path in paths:
        h.update(struct.pack("<?I", path.closed, len(path.points)))
        for pt in path.points:
            for v in (pt.co, pt.handle_left, pt.handle_right):
                h.update(struct.pack("<qq", int(round(v[0] * q)), int(round(v[1] * q))))
            h.update(pt.handle_left_type.encode()[:1] + pt.handle_right_type.encode()[:1])
    h.update(extra)
    return h.hexdigest()


def write_paths(curve, paths: Sequence[BezierPath]) -> None:
    """Replace all splines of a Curve datablock with ``paths`` (z = 0)."""
    curve.splines.clear()
    for path in paths:
        if not path.points:
            continue
        sp = curve.splines.new("BEZIER")
        sp.bezier_points.add(len(path.points) - 1)
        sp.use_cyclic_u = path.closed
        for bp, pt in zip(sp.bezier_points, path.points):
            # set types first: AUTO/VECTOR recompute handles when coordinates change
            bp.handle_left_type = "FREE"
            bp.handle_right_type = "FREE"
            bp.co = (pt.co[0], pt.co[1], 0.0)
            bp.handle_left = (pt.handle_left[0], pt.handle_left[1], 0.0)
            bp.handle_right = (pt.handle_right[0], pt.handle_right[1], 0.0)
        for bp, pt in zip(sp.bezier_points, path.points):
            if pt.handle_left_type in ("ALIGNED", "VECTOR", "AUTO"):
                bp.handle_left_type = pt.handle_left_type
            if pt.handle_right_type in ("ALIGNED", "VECTOR", "AUTO"):
                bp.handle_right_type = pt.handle_right_type


def rotation_2d(obj, scene=None) -> float:
    a, b, _c, _d, _e, _f = object_canvas_affine(obj, scene)
    return math.atan2(b, a)
