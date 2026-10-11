"""Geometry builders per recipe (owner: link).

EXTRUDE / SWEEP / TEXT3D are evaluated natively: the source paths (in the link's SHAPE2D
frame, scaled to metres) are written into a temporary *unlinked* Curve object and converted
with ``Object.to_mesh()`` (holes, bevels and caps for free; identical on 4.0 and 5.2).
REVOLVE uses ``seam_core.mesh.revolve``. DECAL is built by ``decal.py`` (image + plane / material), not here.

Every builder returns a :class:`Built` in the final object frame (orientation and origin
applied, metres). Invalid input raises :class:`RecipeError` (the link goes to ERROR and the
previous geometry is kept); a missing core builder raises :class:`RecipeUnavailable`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import bpy
import numpy as np

from .. import bridge
from ..seam_core import links as L
from ..seam_core.geometry import Affine, BezierPath, BezierPoint

classes = ()

EXPOSED_RECIPES = ("EXTRUDE", "REVOLVE", "SWEEP", "TEXT3D", "DECAL")
MESH_RECIPES = ("EXTRUDE", "REVOLVE", "SWEEP", "TEXT3D")  # built here (DECAL: see decal.py)
DECAL_SOURCE_TYPES = {"SHAPE": "CURVE", "TEXT": "FONT", "GROUP": "EMPTY", "RASTER": "MESH"}
MAT_FACE = 0
MAT_SIDE = 1
TMP_NAME = "SEAM tmp link"


class RecipeError(Exception):
    """The source cannot be turned into geometry with these settings (user-facing message)."""


class RecipeUnavailable(RecipeError):
    """A required builder is not available in this build."""


@dataclass
class Built:
    verts: np.ndarray                    # (N, 3) float32, final object frame (metres)
    loop_totals: np.ndarray              # (F,) int32
    corner_verts: np.ndarray             # (L,) int32
    materials: np.ndarray                # (F,) int32  MAT_FACE / MAT_SIDE
    uvs: Optional[np.ndarray] = None     # (L, 2) float32
    messages: List[str] = field(default_factory=list)

    @property
    def loop_starts(self) -> np.ndarray:
        starts = np.zeros(len(self.loop_totals), dtype=np.int32)
        if len(self.loop_totals) > 1:
            np.cumsum(self.loop_totals[:-1], out=starts[1:])
        return starts


# ------------------------------------------------------------------ source geometry

def shape_frame_paths(obj, scene=None) -> List[BezierPath]:
    """Source paths in the link geometry frame: object-local points with the source's canvas-local
    2D rotation/scale applied and translation removed (architecture §3)."""
    paths = bridge.object_paths(obj)
    return bridge.transform_bezier_paths(paths, bridge.object_shape_affine(obj, scene))


def source_mirrored(obj, scene=None) -> bool:
    a, b, c, d, _e, _f = bridge.object_shape_affine(obj, scene)
    return (a * d - b * c) < 0.0


def _reverse_path(path: BezierPath) -> BezierPath:
    pts = [BezierPoint(p.co, p.handle_right, p.handle_left, p.handle_right_type, p.handle_left_type)
           for p in reversed(path.points)]
    return BezierPath(pts, path.closed)


def _scale_paths(paths: Sequence[BezierPath], s: float) -> List[BezierPath]:
    m: Affine = (s, 0.0, 0.0, s, 0.0, 0.0)
    return bridge.transform_bezier_paths(paths, m)


# ------------------------------------------------------------------ flattening / validation

def _straight(p0, h0, h1, p1, eps=1e-9) -> bool:
    """True when both handles lie on the chord p0-p1 (straight segment)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = math.hypot(dx, dy)
    if ln < eps:
        return math.hypot(h0[0] - p0[0], h0[1] - p0[1]) < eps and math.hypot(h1[0] - p1[0], h1[1] - p1[1]) < eps
    for h in (h0, h1):
        cross = (h[0] - p0[0]) * dy - (h[1] - p0[1]) * dx
        if abs(cross) / ln > 1e-7 * max(1.0, ln):
            return False
    return True


def _flatten_local(path: BezierPath, resolution: int) -> List[Tuple[float, float]]:
    pts = path.points
    n = len(pts)
    if n == 0:
        return []
    out = [tuple(pts[0].co)]
    segs = n if path.closed else n - 1
    for i in range(segs):
        a, b = pts[i], pts[(i + 1) % n]
        p0, h0, h1, p1 = a.co, a.handle_right, b.handle_left, b.co
        if _straight(p0, h0, h1, p1):
            out.append(tuple(p1))
            continue
        for k in range(1, resolution + 1):
            t = k / resolution
            mt = 1.0 - t
            x = mt ** 3 * p0[0] + 3 * mt * mt * t * h0[0] + 3 * mt * t * t * h1[0] + t ** 3 * p1[0]
            y = mt ** 3 * p0[1] + 3 * mt * mt * t * h0[1] + 3 * mt * t * t * h1[1] + t ** 3 * p1[1]
            out.append((x, y))
    clean = [out[0]]
    for p in out[1:]:
        if abs(p[0] - clean[-1][0]) > 1e-12 or abs(p[1] - clean[-1][1]) > 1e-12:
            clean.append(p)
    if path.closed and len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) <= 1e-12 \
            and abs(clean[0][1] - clean[-1][1]) <= 1e-12:
        clean.pop()  # closing point == first point
    return clean


def flatten(path: BezierPath, resolution: int) -> List[Tuple[float, float]]:
    """Flatten with seam_core.geometry when available, else the local reference implementation."""
    try:
        from ..seam_core import geometry
        return [tuple(p) for p in geometry.flatten_path(path, resolution)]
    except NotImplementedError:
        return _flatten_local(path, resolution)


def _signed_area(ring) -> float:
    if len(ring) < 3:
        return 0.0
    a = np.asarray(ring, dtype=np.float64)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _orient2(u, v, w):
    return (v[:, 0] - u[:, 0]) * (w[:, 1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (w[:, 0] - u[:, 0])


def _ring_self_intersects(ring) -> Optional[Tuple[float, float]]:
    """First proper intersection between non-adjacent edges of a closed ring."""
    if len(ring) < 4:
        return None
    hit = _rings_cross([ring])
    return None if hit is None else hit[0]


def _rings_cross(rings) -> Optional[Tuple[Tuple[float, float], int, int]]:
    """First proper crossing between edges of the closed ``rings`` -> (point, ring_a, ring_b).

    Covers a ring crossing itself (ring_a == ring_b, adjacent edges excluded) and two rings crossing
    each other (e.g. a hole sticking out of its outer ring). Sort-and-sweep on x (numpy): only edge
    pairs whose x-ranges overlap are tested, so typical outlines cost O(n log n); candidate pairs are
    processed in bounded chunks.
    """
    parts = [np.asarray(r, dtype=np.float64).reshape(-1, 2) for r in rings if len(r) >= 2]
    if not parts:
        return None
    a0 = np.concatenate(parts)
    a1 = np.concatenate([np.roll(p, -1, axis=0) for p in parts])
    rid = np.concatenate([np.full(len(p), k, dtype=np.int64) for k, p in enumerate(parts)])
    rlen = np.concatenate([np.full(len(p), len(p), dtype=np.int64) for p in parts])
    n = len(a0)
    mn = np.minimum(a0, a1)
    mx = np.maximum(a0, a1)
    order = np.argsort(mn[:, 0], kind="stable")
    smin_x = mn[order, 0]
    ends = np.searchsorted(smin_x, mx[order, 0], side="right")
    counts = np.maximum(ends - (np.arange(n) + 1), 0)
    csum = np.cumsum(counts)
    chunk = 1 << 21
    i0 = 0
    while i0 < n:
        base = csum[i0 - 1] if i0 else 0
        i1 = int(np.searchsorted(csum, base + chunk, side="right"))
        i1 = max(i1, i0 + 1)
        i1 = min(i1, n)
        cnt = counts[i0:i1]
        total = int(cnt.sum())
        if total:
            rel = np.repeat(np.arange(i0, i1), cnt)
            offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            a = order[rel]
            b = order[rel + 1 + offs]
            keep = (mn[a, 1] <= mx[b, 1]) & (mn[b, 1] <= mx[a, 1])
            d = np.abs(a - b)
            same = rid[a] == rid[b]
            keep &= ~same | ((d != 1) & (d != rlen[a] - 1) & (d != 0))
            a, b = a[keep], b[keep]
            if len(a):
                hit = _first_crossing(a0[a], a1[a], a0[b], a1[b])
                if hit is not None:
                    pt, k = hit
                    ra, rb = int(rid[a[k]]), int(rid[b[k]])
                    return pt, min(ra, rb), max(ra, rb)
        i0 = i1
    return None


def _first_crossing(p0, p1, q0, q1):
    d1 = _orient2(p0, p1, q0)
    d2 = _orient2(p0, p1, q1)
    d3 = _orient2(q0, q1, p0)
    d4 = _orient2(q0, q1, p1)
    scale = np.maximum(np.abs(p1 - p0).max(axis=1), np.abs(q1 - q0).max(axis=1)) ** 2
    tol = 1e-12 * np.maximum(scale, 1.0)
    hit = (((d1 > tol) & (d2 < -tol)) | ((d1 < -tol) & (d2 > tol))) & \
          (((d3 > tol) & (d4 < -tol)) | ((d3 < -tol) & (d4 > tol)))
    k = np.nonzero(hit)[0]
    if not len(k):
        return None
    k = int(k[0])
    t = d3[k] / (d3[k] - d4[k])
    pt = p0[k] + t * (p1[k] - p0[k])
    return (float(pt[0]), float(pt[1])), k


def _issue_text(message, point) -> str:
    if point is not None:
        return "%s near (%.1f, %.1f)" % (message, point[0], point[1])
    return message


def validate_rings(rings, check_self_intersection: bool = True) -> List[str]:
    """User-facing problems that make a filled extrusion invalid (empty list = valid)."""
    try:
        from ..seam_core import geometry
        issues = geometry.validate_rings(rings, require_closed_area=True)
        if not check_self_intersection:
            issues = [i for i in issues if i.code != "SELF_INTERSECTION"]
        return [_issue_text(i.message, i.point) for i in issues]
    except NotImplementedError:
        return validate_rings_local(rings, check_self_intersection)


def validate_rings_local(rings, check_self_intersection: bool = True) -> List[str]:
    """Reference validator used until ``seam_core.geometry.validate_rings`` is available."""
    problems = []
    crossing = set()
    if check_self_intersection:
        usable = [k for k, ring in enumerate(rings) if len(ring) >= 3]
        found = _rings_cross([rings[k] for k in usable])
        if found is not None:
            pt, ra, rb = found
            ra, rb = usable[ra], usable[rb]
            if ra == rb:
                problems.append(_issue_text("path %d crosses itself" % (ra + 1), pt))
            else:
                problems.append(_issue_text("paths %d and %d cross each other" % (ra + 1, rb + 1), pt))
            crossing.update((ra, rb))
    for k, ring in enumerate(rings):
        if len(ring) < 3:
            problems.append("path %d has fewer than 3 points" % (k + 1))
            continue
        if k in crossing:
            continue
        span = float(np.ptp(np.asarray(ring, dtype=np.float64), axis=0).max())
        if abs(_signed_area(ring)) <= 1e-9 * max(span, 1e-9) ** 2:
            problems.append("path %d encloses no area" % (k + 1))
    return problems


# ------------------------------------------------------------------ native curve evaluation

def _temp_pool():
    """The reused, unlinked temp curve + object (created on demand, looked up by name every time so
    undo/load never leaves us with a stale reference). Reusing them avoids adding/removing
    datablocks per build, which would force a depsgraph relations rebuild of the whole scene."""
    cu = bpy.data.curves.get(TMP_NAME)
    if cu is None or cu.get("seam_internal") != 1 or cu.library is not None:
        cu = bpy.data.curves.new(TMP_NAME, "CURVE")
        cu["seam_internal"] = 1
    ob = bpy.data.objects.get(TMP_NAME)
    if ob is None or ob.get("seam_internal") != 1 or ob.library is not None:
        ob = bpy.data.objects.new(TMP_NAME, cu)
        ob["seam_internal"] = 1
    if ob.data != cu:
        ob.data = cu
    return ob, cu


def _set(owner, attr: str, value) -> None:
    """Assign only on change: some curve RNA updates (pointers, dimensions) tag a full relations update."""
    if getattr(owner, attr) != value:
        setattr(owner, attr, value)


def _temp_curve(paths: Sequence[BezierPath], dims: str = "2D"):
    ob, cu = _temp_pool()
    _set(cu, "dimensions", dims)
    # reset every setting a recipe may change
    _set(cu, "fill_mode", "BOTH" if dims == "2D" else "FULL")
    for attr, value in (("extrude", 0.0), ("bevel_depth", 0.0), ("bevel_resolution", 0), ("bevel_mode", "ROUND"),
                        ("offset", 0.0), ("twist_mode", "MINIMUM"), ("use_fill_caps", False),
                        ("use_map_taper", False), ("taper_object", None), ("bevel_object", None)):
        _set(cu, attr, value)
    bridge.write_paths(cu, paths)
    return ob, cu


def _remove_temp(ob, cu) -> None:
    try:
        ob.to_mesh_clear()
    except Exception:
        pass
    cu.splines.clear()


def purge_temp() -> None:
    """Remove the internal temp datablocks (before saving, on unregister)."""
    ob = bpy.data.objects.get(TMP_NAME)
    if ob is not None and ob.get("seam_internal") == 1 and ob.library is None:
        bpy.data.objects.remove(ob, do_unlink=True)
    cu = bpy.data.curves.get(TMP_NAME)
    if cu is not None and cu.get("seam_internal") == 1 and cu.library is None:
        bpy.data.curves.remove(cu, do_unlink=True)


@bpy.app.handlers.persistent
def _on_save_pre(*_args):
    try:
        purge_temp()
    except Exception:
        pass


def register():
    if _on_save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(_on_save_pre)


def unregister():
    while _on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_on_save_pre)
    try:
        purge_temp()
    except Exception:
        pass


def _read_mesh(me):
    nv, nf, nl = len(me.vertices), len(me.polygons), len(me.loops)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    totals = np.empty(nf, dtype=np.int32)
    me.polygons.foreach_get("loop_total", totals)
    corners = np.empty(nl, dtype=np.int32)
    me.loops.foreach_get("vertex_index", corners)
    normals = np.empty(nf * 3, dtype=np.float32)
    me.polygons.foreach_get("normal", normals)
    centers = np.empty(nf * 3, dtype=np.float32)
    me.polygons.foreach_get("center", centers)
    uvs = None
    uvl = me.uv_layers.get("UVMap") or (me.uv_layers[0] if len(me.uv_layers) else None)
    if uvl is not None and nl:
        uvs = np.empty(nl * 2, dtype=np.float32)
        uvl.data.foreach_get("uv", uvs)
        uvs = uvs.reshape(nl, 2)
    return co.reshape(nv, 3), totals, corners, normals.reshape(nf, 3), centers.reshape(nf, 3), uvs


def _evaluate(ob, cu):
    try:
        me = ob.to_mesh()
        if me is None:
            raise RecipeError("Blender produced no geometry for this path")
        return _read_mesh(me)
    finally:
        _remove_temp(ob, cu)


# ------------------------------------------------------------------ final frame

def _orient(v: np.ndarray, orientation: str, recipe: str) -> np.ndarray:
    """Map the flat build frame (drawing in XY, thickness/axis along +Z) to the object frame."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    if recipe == "REVOLVE":
        # revolve output already stands on its axis (+Z) => UPRIGHT; FLAT lays the axis along +Y
        if orientation == "FLAT":
            return np.stack([x, z, -y], axis=1)
        return v
    if orientation == "UPRIGHT":
        return np.stack([x, -z, y], axis=1)  # drawing faces -Y and stands on Z
    return v


def _place_origin(v: np.ndarray, mode: str) -> np.ndarray:
    if not len(v) or mode == "SOURCE":
        return v
    lo, hi = v.min(axis=0), v.max(axis=0)
    if mode == "CENTER":
        off = (lo + hi) * 0.5
    elif mode == "BOTTOM":
        off = np.array([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, lo[2]])
    else:  # MIN
        off = lo
    return v - off


def _finish(recipe, params, verts, totals, corners, mats, uvs, messages) -> Built:
    v = _orient(np.asarray(verts, dtype=np.float64), params["orientation"], recipe)
    v = _place_origin(v, params["origin"])
    return Built(v.astype(np.float32), np.asarray(totals, dtype=np.int32), np.asarray(corners, dtype=np.int32),
                 np.asarray(mats, dtype=np.int32), None if uvs is None else np.asarray(uvs, dtype=np.float32),
                 messages)


# ------------------------------------------------------------------ recipes

def _closed_paths(paths, messages, what="Extrude"):
    closed = [p for p in paths if p.closed and len(p.points) >= 2]
    n_open = sum(1 for p in paths if not p.closed and len(p.points))
    if n_open:
        messages.append("%d open path%s ignored (%s needs closed paths)" % (n_open, "" if n_open == 1 else "s", what))
    return closed


def _normalize_winding(paths, mirrored: bool):
    """A mirrored source transform flips every path's winding; reverse to restore outer CCW / holes CW."""
    return [_reverse_path(p) for p in paths] if mirrored else list(paths)


def build_extrude(paths: Sequence[BezierPath], params: dict, mirrored: bool = False,
                  check_self_intersection: bool = True, what: str = "Extrude") -> Built:
    messages: List[str] = []
    closed = _closed_paths(paths, messages, what)
    if not closed:
        raise RecipeError("Nothing to extrude: the source has no closed path")
    rings = [flatten(p, params["resolution"]) for p in closed]
    problems = validate_rings(rings, check_self_intersection)
    if problems:
        raise RecipeError("Invalid shape: " + "; ".join(problems[:3]))
    s = params["units_scale"]
    ob, cu = _temp_curve(_scale_paths(_normalize_winding(closed, mirrored), s), "2D")
    depth = params["depth"]
    bevel = min(params["bevel_depth"], depth * 0.5)
    _set(cu, "fill_mode", "BOTH" if params["fill_caps"] else "NONE")
    _set(cu, "resolution_u", params["resolution"])
    _set(cu, "extrude", max(depth * 0.5 - bevel, 0.0))
    _set(cu, "bevel_depth", bevel)
    _set(cu, "bevel_resolution", params["bevel_resolution"])
    # a filled 2D bevel grows the outline by bevel_depth (outer rings out, holes in); inset by the same
    # amount so the footprint stays source size * units_scale (verified winding-independent, 4.0 + 5.2)
    _set(cu, "offset", -bevel)
    co, totals, corners, normals, centers, uvs = _evaluate(ob, cu)
    if not len(totals):
        raise RecipeError("The shape produced no faces (is it empty or degenerate?)")
    zext = float(np.abs(co[:, 2]).max()) if len(co) else 0.0
    tol = max(1e-6, zext * 1e-4)
    cap = (np.abs(normals[:, 2]) > 0.999) & (np.abs(np.abs(centers[:, 2]) - zext) <= tol)
    mats = np.where(cap, MAT_FACE, MAT_SIDE)
    return _finish("EXTRUDE", params, co, totals, corners, mats, uvs, messages)


def build_text3d(paths, params, mirrored=False) -> Built:
    if not any(p.closed for p in paths):
        raise RecipeError("The text has no visible glyphs")
    # glyph outlines may legitimately touch/overlap; Blender fills them with its nesting rules
    p = dict(params)
    p.setdefault("fill_caps", True)
    return build_extrude(paths, p, mirrored, check_self_intersection=False, what="Text 3D")


def _endpoint_frames(paths_m, resolution):
    """(point, unit tangent) at both ends of every open path (already scaled)."""
    out = []
    for path in paths_m:
        if path.closed or len(path.points) < 2:
            continue
        pts = flatten(path, resolution)
        if len(pts) < 2:
            continue
        for a, b in ((pts[0], pts[1]), (pts[-1], pts[-2])):
            t = np.array([b[0] - a[0], b[1] - a[1], 0.0])
            n = np.linalg.norm(t)
            if n > 0:
                out.append((np.array([a[0], a[1], 0.0]), t / n))
    return out


def build_sweep(paths: Sequence[BezierPath], params: dict, mirrored: bool = False) -> Built:
    usable = [p for p in paths if len(p.points) >= 2]
    if not usable:
        raise RecipeError("Nothing to sweep: the source has no path with at least 2 points")
    s = params["units_scale"]
    paths_m = _scale_paths(_normalize_winding(usable, mirrored), s)
    r = params["sweep_radius"]
    profile = params["sweep_profile"]
    if profile == "RIBBON":
        ob, cu = _temp_curve(paths_m, "2D")
        _set(cu, "fill_mode", "NONE")
        _set(cu, "extrude", r)
        _set(cu, "bevel_depth", 0.0)
    else:
        ob, cu = _temp_curve(paths_m, "3D")
        _set(cu, "fill_mode", "FULL")
        _set(cu, "twist_mode", "Z_UP")
        _set(cu, "bevel_mode", "ROUND")
        _set(cu, "use_fill_caps", bool(params["fill_caps"]))
        if profile == "SQUARE":
            _set(cu, "bevel_depth", r * math.sqrt(2.0))
            _set(cu, "bevel_resolution", 0)
            for sp in cu.splines:
                for bp in sp.bezier_points:
                    bp.tilt = math.pi / 4.0
        else:
            _set(cu, "bevel_depth", r)
            _set(cu, "bevel_resolution", params["sweep_profile_resolution"])
    _set(cu, "resolution_u", params["resolution"])
    co, totals, corners, normals, centers, uvs = _evaluate(ob, cu)
    if not len(totals):
        raise RecipeError("The sweep produced no faces")
    mats = np.full(len(totals), MAT_SIDE, dtype=np.int32)
    if profile != "RIBBON" and params["fill_caps"]:
        for p, t in _endpoint_frames(paths_m, params["resolution"]):
            par = np.abs(normals @ t) > 0.99
            near = np.linalg.norm(centers - p, axis=1) <= r * 0.75
            mats[par & near] = MAT_FACE
    return _finish("SWEEP", params, co, totals, corners, mats, uvs, [])


def revolve_available() -> bool:
    try:
        from ..seam_core import mesh
        mesh.revolve([(1.0, 0.0), (1.0, 1.0)], False, segments=3)
        return True
    except NotImplementedError:
        return False
    except Exception:
        return True


def build_revolve(paths: Sequence[BezierPath], params: dict, mirrored: bool = False) -> Built:
    from ..seam_core import mesh as core_mesh
    usable = [p for p in paths if len(p.points) >= 2]
    if not usable:
        raise RecipeError("Nothing to revolve: the source has no path with at least 2 points")
    messages = []
    if len(usable) > 1:
        messages.append("%d extra path%s ignored (Revolve uses the first path)"
                        % (len(usable) - 1, "" if len(usable) == 2 else "s"))
    path = usable[0]
    profile = flatten(path, params["resolution"])
    xs = [p[0] for p in profile]
    mode = params["revolve_axis"]
    if mode == "LEFT":
        axis = min(xs)
    elif mode == "CENTER":
        axis = 0.5 * (min(xs) + max(xs))
    else:
        axis = 0.0
    axis += params["axis_offset"]
    try:
        md = core_mesh.revolve(profile, path.closed, angle=params["revolve_angle"],
                               segments=params["revolve_segments"], axis_x=axis,
                               cap_ends=bool(params["fill_caps"]))
    except NotImplementedError:
        raise RecipeUnavailable("Revolve is not available in this build (seam_core.mesh.revolve missing)")
    except ValueError as exc:
        raise RecipeError("Invalid profile: %s" % exc)
    if not md.faces:
        raise RecipeError("The revolve produced no faces")
    s = params["units_scale"]
    verts = np.asarray(md.verts, dtype=np.float64).reshape(-1, 3)
    verts[:, 0] -= axis  # the revolution axis passes through the object origin
    verts *= s
    totals = np.array([len(f) for f in md.faces], dtype=np.int32)
    corners = np.array([i for f in md.faces for i in f], dtype=np.int32)
    mats = np.array(md.face_materials if len(md.face_materials) == len(md.faces) else [MAT_SIDE] * len(md.faces),
                    dtype=np.int32)
    uvs = np.asarray(md.uvs, dtype=np.float32) if md.uvs and len(md.uvs) == len(corners) else None
    return _finish("REVOLVE", params, verts, totals, corners, mats, uvs, messages)


def box_uvs(verts: np.ndarray, totals: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Deterministic per-corner UVs by box projection into a 3 x 2 atlas (one cell per axis and sign).

    Every face goes to the cell of its dominant normal axis/direction and is projected along that axis, all
    cells share one scale (mesh bounding box). UVs depend only on vertex positions and face winding, never on
    vertex numbering, so texels painted on a result stay in place when the source changes topology (they
    follow the surface, not the indices)."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    totals = np.asarray(totals, dtype=np.int64)
    corners = np.asarray(corners, dtype=np.int64)
    nl = len(corners)
    out = np.zeros((nl, 2), dtype=np.float32)
    if not nl or not len(totals):
        return out
    lo = verts.min(axis=0)
    span = float(max((verts.max(axis=0) - lo).max(), 1e-9))
    face_of = np.repeat(np.arange(len(totals)), totals)
    p = verts[corners]
    # Newell normal per face
    starts = np.zeros(len(totals), dtype=np.int64)
    np.cumsum(totals[:-1], out=starts[1:])
    idx = np.arange(nl)
    nxt_idx = idx + 1
    ends = starts + totals - 1
    nxt_idx[ends] = starts
    nxt = p[nxt_idx]
    nx = (p[:, 1] - nxt[:, 1]) * (p[:, 2] + nxt[:, 2])
    ny = (p[:, 2] - nxt[:, 2]) * (p[:, 0] + nxt[:, 0])
    nz = (p[:, 0] - nxt[:, 0]) * (p[:, 1] + nxt[:, 1])
    n = np.stack([np.bincount(face_of, weights=w, minlength=len(totals)) for w in (nx, ny, nz)], axis=1)
    axis = np.argmax(np.abs(n), axis=1)
    sign = np.take_along_axis(n, axis[:, None], axis=1)[:, 0] >= 0
    # (u axis, v axis) per projection axis (X: Y,Z / Y: X,Z / Z: X,Y); see the handedness note below
    uax = np.array([1, 0, 0])[axis]
    vax = np.array([2, 2, 1])[axis]
    rel = (p - lo) / span
    cu = rel[idx, uax[face_of]]
    cv = rel[idx, vax[face_of]]
    # u x v must point at the viewer: that holds for +X, -Y and +Z faces; the other three mirror u
    positive = sign[face_of]
    outward_ok = np.where(axis[face_of] == 1, ~positive, positive)
    cu = np.where(outward_ok, cu, 1.0 - cu)
    cell_x = axis[face_of].astype(np.float64)
    cell_y = np.where(sign[face_of], 0.0, 1.0)
    pad = 0.02
    out[:, 0] = ((cell_x + pad + cu * (1.0 - 2 * pad)) / 3.0).astype(np.float32)
    out[:, 1] = ((cell_y + pad + cv * (1.0 - 2 * pad)) / 2.0).astype(np.float32)
    return out


def build(recipe: str, paths: Sequence[BezierPath], params: dict, mirrored: bool = False) -> Built:
    params = L.validate_params(recipe, params)
    params.setdefault("orientation", "UPRIGHT")
    params.setdefault("origin", "BOTTOM")
    params.setdefault("fill_caps", True)
    if recipe == "EXTRUDE":
        built = build_extrude(paths, params, mirrored)
    elif recipe == "TEXT3D":
        built = build_text3d(paths, params, mirrored)
    elif recipe == "SWEEP":
        built = build_sweep(paths, params, mirrored)
    elif recipe == "REVOLVE":
        built = build_revolve(paths, params, mirrored)
    else:
        raise RecipeUnavailable("%s is built from an image, not from paths" % L.RECIPE_LABELS.get(recipe, recipe))
    if built.uvs is None and len(built.corner_verts):
        built.uvs = box_uvs(built.verts, built.loop_totals, built.corner_verts)  # every result is paintable
    return built


def recipe_available(recipe: str) -> Tuple[bool, str]:
    if recipe not in EXPOSED_RECIPES:
        return False, "%s is not available yet" % L.RECIPE_LABELS.get(recipe, recipe)
    if recipe == "DECAL":
        return True, ""
    if recipe == "REVOLVE" and not revolve_available():
        return False, "Revolve builder not available in this build"
    return True, ""


def source_ok(recipe: str, obj) -> Tuple[bool, str]:
    """Can ``obj`` feed ``recipe``? (ok, reason)."""
    if obj is None:
        return False, "Select a shape or text object"
    ok, why = recipe_available(recipe)
    if not ok:
        return ok, why
    ok, why = L.can_link(recipe, obj.seam.role)
    if not ok:
        return ok, why
    if recipe == "DECAL":
        want = DECAL_SOURCE_TYPES.get(obj.seam.role, "CURVE")
        if obj.type != want:
            return False, "The %s source has an unexpected object type (%s)" % (obj.seam.role.lower(), obj.type)
        return True, ""
    want = "FONT" if recipe == "TEXT3D" else "CURVE"
    if obj.type != want:
        return False, "%s needs a %s object" % (L.RECIPE_LABELS[recipe], "text" if want == "FONT" else "curve")
    return True, ""
