"""Pure mesh builders for 2D->3D recipes (owner: core-geometry).

The add-on uses native Blender curve evaluation for EXTRUDE/SWEEP/TEXT3D; these
builders are the reference/fallback implementations and the REVOLVE builder.
Coordinates produced here are in the *flat* frame: profile/shape in XY, extrusion
along +Z; :func:`orient` maps to UPRIGHT (x, y, z) -> (x, -z, y)... see docstring.
"""

from __future__ import annotations

import hashlib
import struct
import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .geometry import Ring, Vec2, as_points, signed_area, triangulate, _check_scalars

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
        """Raise ValueError on out-of-range indices, faces < 3 verts, or length mismatches.

        ``face_materials`` and ``uvs`` may be empty (unset); when present they must have exactly
        one entry per face / per face corner.
        """
        nv = len(self.verts)
        for k, f in enumerate(self.faces):
            if len(f) < 3:
                raise ValueError("face %d has %d vertices (need >= 3)" % (k, len(f)))
            for i in f:
                if not (0 <= i < nv):
                    raise ValueError("face %d references vertex %d out of range (0..%d)" % (k, i, nv - 1))
        if self.face_materials and len(self.face_materials) != len(self.faces):
            raise ValueError("face_materials has %d entries for %d faces" % (len(self.face_materials), len(self.faces)))
        if self.uvs:
            loops = sum(len(f) for f in self.faces)
            if len(self.uvs) != loops:
                raise ValueError("uvs has %d entries for %d face corners" % (len(self.uvs), loops))
        for v in self.verts:
            if len(v) != 3 or not all(math.isfinite(c) for c in v):
                raise ValueError("non-finite or malformed vertex %r" % (v,))


def _dedupe(points, closed: bool, what: str = "ring") -> List[Vec2]:
    a = as_points(points, what)
    out: List[Vec2] = []
    for x, y in a.tolist():
        if out and x == out[-1][0] and y == out[-1][1]:
            continue
        out.append((x, y))
    while closed and len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _oriented_ring(ring: Sequence[Vec2], ccw: bool) -> List[Vec2]:
    pts = _dedupe(ring, True)
    if len(pts) < 3:
        return []
    a = signed_area(pts)
    if a == 0.0:
        return []
    if (a > 0.0) != ccw:
        pts.reverse()
    return pts


def extrude(shapes: Sequence[Tuple[Ring, List[Ring]]], depth: float, cap_front: bool = True,
            cap_back: bool = True, center: bool = True) -> MeshData:
    """Extrude normalized shapes along +Z by ``depth`` (centered on z=0 when ``center``).

    Caps are triangulated (geometry.triangulate) with holes; walls are quads with
    outward normals (consistent winding). Cap faces get MAT_FACE, walls MAT_SIDE.
    UVs: caps planar from bbox (0..1), walls (perimeter length / total, z / depth).

    ``cap_front`` is the +Z cap, ``cap_back`` the -Z cap. ``depth == 0`` produces only the
    front cap (a flat, open surface); negative or non-finite depth raises ``ValueError``.
    Shape rings are cleaned (duplicate points dropped) and re-oriented (outer CCW, holes CW)
    defensively; shapes whose outer ring is degenerate are skipped.
    """
    d = float(depth)
    _check_scalars(d)
    if d < 0.0:
        raise ValueError("extrude depth must be >= 0, got %r" % depth)
    z0 = -d / 2.0 if center else 0.0
    z1 = z0 + d
    prepared: List[Tuple[List[Vec2], List[List[Vec2]]]] = []
    for outer, holes in shapes:
        o = _oriented_ring(outer, True)
        if not o:
            continue
        hs = []
        for h in holes:
            r = _oriented_ring(h, False)
            if r:
                hs.append(r)
        prepared.append((o, hs))
    mesh = MeshData()
    if not prepared:
        return mesh
    xs = [p[0] for o, _ in prepared for p in o]
    ys = [p[1] for o, _ in prepared for p in o]
    minx, miny = min(xs), min(ys)
    bw = (max(xs) - minx) or 1.0
    bh = (max(ys) - miny) or 1.0

    def cap_uv(x: float, y: float) -> Tuple[float, float]:
        return ((x - minx) / bw, (y - miny) / bh)

    verts = mesh.verts
    faces = mesh.faces
    mats = mesh.face_materials
    uvs = mesh.uvs
    flat = d == 0.0
    for outer, holes in prepared:
        rings = [outer] + holes
        layer_n = sum(len(r) for r in rings)
        base = len(verts)
        for z in ((z1,) if flat else (z0, z1)):
            for r in rings:
                verts.extend((x, y, z) for x, y in r)
        top = base if flat else base + layer_n
        bottom = base
        if cap_front or cap_back:
            cv, ctris = triangulate(outer, holes)
            if cap_front:
                for a, b, c in ctris:
                    faces.append((top + a, top + b, top + c))
                    mats.append(MAT_FACE)
                    uvs.extend(cap_uv(*cv[i]) for i in (a, b, c))
            if cap_back and not flat:
                for a, b, c in ctris:
                    faces.append((bottom + c, bottom + b, bottom + a))
                    mats.append(MAT_FACE)
                    uvs.extend(cap_uv(*cv[i]) for i in (c, b, a))
        if flat:
            continue
        offset = 0
        for r in rings:
            n = len(r)
            edge_len = [math.hypot(r[(k + 1) % n][0] - r[k][0], r[(k + 1) % n][1] - r[k][1]) for k in range(n)]
            total = sum(edge_len) or 1.0
            acc = 0.0
            for k in range(n):
                k2 = (k + 1) % n
                u0 = acc / total
                acc += edge_len[k]
                u1 = acc / total
                a_b = bottom + offset + k
                b_b = bottom + offset + k2
                faces.append((a_b, b_b, b_b + layer_n, a_b + layer_n))
                mats.append(MAT_SIDE)
                uvs.extend(((u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)))
            offset += n
    return mesh


def _profile_uv_lengths(pts: Sequence[Vec2], closed: bool) -> List[float]:
    """Cumulative arc length fractions per profile point (closed adds one extra entry for the wrap)."""
    n = len(pts)
    segs = n if closed else n - 1
    acc = [0.0]
    for i in range(segs):
        a = pts[i]
        b = pts[(i + 1) % n]
        acc.append(acc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = acc[-1] or 1.0
    return [v / total for v in acc]


def revolve(profile: Sequence[Vec2], closed: bool, angle: float = 6.283185307179586, segments: int = 32,
            axis_x: float = 0.0, cap_ends: bool = True, merge_on_axis: bool = True) -> MeshData:
    """Lathe: rotate the profile (x = radius distance from ``axis_x``, y = height) around the
    vertical axis x = axis_x. Output in the flat frame with the revolution axis along +Z
    (profile y -> z, radius -> xy plane). Points on the axis are merged; partial angles get
    end caps when the profile is closed and ``cap_ends``. Normals point outward.
    Profile points with x < axis_x are mirrored (abs distance) and reported via ValueError only
    if the profile crosses the axis.

    The revolution axis passes through the output origin (a profile point at ``x = axis_x + r`` ends
    up at radius ``r`` from the Z axis). Vertices start at angle 0 on +X and advance counter-clockwise.
    For an *open* profile the surface normals point to the right of the profile direction
    (draw an outer wall from bottom to top and the normals face away from the axis); closed
    profiles are reoriented automatically so the normals face outward.
    """
    _check_scalars(angle, axis_x)
    pts = _dedupe(profile, closed, "profile")
    if len(pts) < (3 if closed else 2):
        raise ValueError("revolve needs at least %d distinct profile points" % (3 if closed else 2))
    if angle <= 0.0:
        raise ValueError("revolve angle must be > 0, got %r" % angle)
    ys = [p[1] for p in pts]
    ext = max(max(p[0] for p in pts) - min(p[0] for p in pts), max(ys) - min(ys), 1e-12)
    tol = 1e-9 * max(ext, 1.0)
    rs = [p[0] - axis_x for p in pts]
    if any(r > tol for r in rs) and any(r < -tol for r in rs):
        raise ValueError("profile crosses the revolution axis")
    if all(r <= tol for r in rs):
        rs = [-r for r in rs]
    rs = [0.0 if abs(r) <= tol else r for r in rs]
    prof = list(zip(rs, ys))
    if closed:
        a = signed_area(prof)
        if a == 0.0:
            raise ValueError("closed revolve profile has zero area")
        if a < 0.0:
            prof.reverse()
    n = len(prof)
    full = angle >= 2.0 * math.pi - 1e-9
    angle = min(float(angle), 2.0 * math.pi)
    seg = max(3, int(segments)) if full else max(1, int(segments))
    ncols = seg if full else seg + 1
    uv_v = _profile_uv_lengths(prof, closed)

    mesh = MeshData()
    on_axis = [r == 0.0 for r, _ in prof]
    index = {}

    def vert(i: int, j: int) -> int:
        j = j % seg if full else j
        key = (i, 0) if (on_axis[i] and merge_on_axis) else (i, j)
        idx = index.get(key)
        if idx is None:
            r, y = prof[i]
            th = angle * j / seg
            idx = len(mesh.verts)
            mesh.verts.append((r * math.cos(th), r * math.sin(th), y))
            index[key] = idx
        return idx

    # create vertices in a stable order: column by column
    for j in range(ncols):
        for i in range(n):
            vert(i, j)
    segs = n if closed else n - 1
    for i in range(segs):
        a = i
        b = (i + 1) % n
        va = uv_v[i]
        vb = uv_v[i + 1]
        for j in range(seg):
            corners = [(a, j + 1), (b, j + 1), (b, j), (a, j)]
            ids = [vert(ii, jj) for ii, jj in corners]
            uv = [((j + 1) / seg, va), ((j + 1) / seg, vb), (j / seg, vb), (j / seg, va)]
            keep_ids = []
            keep_uv = []
            for k in range(4):
                if keep_ids and ids[k] == keep_ids[-1]:
                    continue
                keep_ids.append(ids[k])
                keep_uv.append(uv[k])
            while len(keep_ids) > 1 and keep_ids[0] == keep_ids[-1]:
                keep_ids.pop()
                keep_uv.pop()
            if len(keep_ids) < 3 or len(set(keep_ids)) < len(keep_ids):
                continue
            mesh.faces.append(tuple(keep_ids))
            mesh.face_materials.append(MAT_SIDE)
            mesh.uvs.extend(keep_uv)
    if closed and cap_ends and not full:
        cv, ctris = triangulate(prof)
        rmax = max(r for r, _ in prof) or 1.0
        ymin = min(y for _, y in prof)
        yh = (max(y for _, y in prof) - ymin) or 1.0

        def cuv(i):
            return (prof[i][0] / rmax, (prof[i][1] - ymin) / yh)

        for a, b, c in ctris:
            mesh.faces.append((vert(a, 0), vert(b, 0), vert(c, 0)))
            mesh.face_materials.append(MAT_FACE)
            mesh.uvs.extend((cuv(a), cuv(b), cuv(c)))
            mesh.faces.append((vert(c, seg), vert(b, seg), vert(a, seg)))
            mesh.face_materials.append(MAT_FACE)
            mesh.uvs.extend((cuv(c), cuv(b), cuv(a)))
    return mesh


def sweep(path: Sequence[Vec2], closed: bool, radius: float, profile: str = "ROUND", profile_segments: int = 8,
          cap_ends: bool = True) -> MeshData:
    """Reference tube along a planar path (rotation-minimizing frames). profile: ROUND, SQUARE, RIBBON.

    The path lies in the XY plane (z = 0). For a planar path the rotation-minimizing frame is
    constant relative to the plane: ``side`` is the in-plane left normal and ``up`` is +Z, so
    the cross section never twists. ROUND uses ``profile_segments`` (>= 3) ring vertices on a
    circle of ``radius``; SQUARE is a ``2 * radius`` square; RIBBON is a flat strip of width
    ``2 * radius`` in the path plane (an open surface facing +Z, no caps). Corners are mitered
    so the tube keeps its thickness (miter length is clamped). Open ROUND/SQUARE tubes get
    n-gon end caps when ``cap_ends``; closed paths never have caps.
    """
    _check_scalars(radius)
    if profile not in ("ROUND", "SQUARE", "RIBBON"):
        raise ValueError("profile must be ROUND, SQUARE or RIBBON, got %r" % (profile,))
    if radius <= 0.0:
        raise ValueError("sweep radius must be > 0, got %r" % radius)
    pts = _dedupe(path, closed, "path")
    if len(pts) < (3 if closed else 2):
        raise ValueError("sweep needs at least %d distinct path points" % (3 if closed else 2))
    n = len(pts)
    m = n if closed else n - 1
    dirs = []
    lens = []
    for i in range(m):
        a = pts[i]
        b = pts[(i + 1) % n]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        ln = math.hypot(dx, dy)
        dirs.append((dx / ln, dy / ln))
        lens.append(ln)
    normals = [(-dy, dx) for dx, dy in dirs]
    # per-vertex side vector (unit normal scaled by the miter factor)
    sides = []
    for v in range(n):
        if not closed and v == 0:
            sides.append(normals[0])
        elif not closed and v == n - 1:
            sides.append(normals[m - 1])
        else:
            na = normals[(v - 1) % m]
            nb = normals[v % m]
            dot = na[0] * nb[0] + na[1] * nb[1]
            if dot <= -0.9:
                sides.append(normals[(v - 1) % m])  # near-reversal: do not miter
            else:
                k = 1.0 / (1.0 + dot)
                sx, sy = (na[0] + nb[0]) * k, (na[1] + nb[1]) * k
                ln = math.hypot(sx, sy)
                if ln > 4.0:
                    sx, sy = sx * 4.0 / ln, sy * 4.0 / ln
                sides.append((sx, sy))
    if profile == "ROUND":
        k = max(3, int(profile_segments))
        section = [(radius * math.cos(2 * math.pi * i / k), radius * math.sin(2 * math.pi * i / k)) for i in range(k)]
        ring_closed = True
    elif profile == "SQUARE":
        section = [(radius, -radius), (radius, radius), (-radius, radius), (-radius, -radius)]
        ring_closed = True
    else:
        section = [(-radius, 0.0), (radius, 0.0)]
        ring_closed = False
    ns = len(section)
    mesh = MeshData()
    for v in range(n):
        px, py = pts[v]
        sx, sy = sides[v]
        for u, w in section:
            mesh.verts.append((px + sx * u, py + sy * u, w))
    cum = [0.0]
    for i in range(m):
        cum.append(cum[-1] + lens[i])
    total = cum[-1] or 1.0
    nseg_ring = ns if ring_closed else ns - 1
    for i in range(m):
        a = i
        b = (i + 1) % n
        v0 = cum[i] / total
        v1 = cum[i + 1] / total
        for k in range(nseg_ring):
            k2 = (k + 1) % ns
            ia, ia2 = a * ns + k, a * ns + k2
            ib, ib2 = b * ns + k, b * ns + k2
            if ring_closed:
                mesh.faces.append((ia, ia2, ib2, ib))
                mesh.uvs.extend(((k / ns, v0), ((k + 1) / ns, v0), ((k + 1) / ns, v1), (k / ns, v1)))
            else:
                mesh.faces.append((ia2, ia, ib, ib2))
                mesh.uvs.extend(((1.0, v0), (0.0, v0), (0.0, v1), (1.0, v1)))
            mesh.face_materials.append(MAT_SIDE)
    if cap_ends and not closed and ring_closed:
        s0 = list(range(ns))
        s1 = [(n - 1) * ns + k for k in range(ns)]
        ruv = [(0.5 + 0.5 * u / radius, 0.5 + 0.5 * w / radius) for u, w in section]
        mesh.faces.append(tuple(reversed(s0)))
        mesh.face_materials.append(MAT_FACE)
        mesh.uvs.extend(reversed(ruv))
        mesh.faces.append(tuple(s1))
        mesh.face_materials.append(MAT_FACE)
        mesh.uvs.extend(ruv)
    return mesh


def _copy(mesh: MeshData, verts=None, faces=None, uvs=None) -> MeshData:
    return MeshData(verts=list(mesh.verts) if verts is None else verts,
                    faces=list(mesh.faces) if faces is None else faces,
                    face_materials=list(mesh.face_materials),
                    uvs=list(mesh.uvs) if uvs is None else uvs)


def orient(mesh: MeshData, orientation: str) -> MeshData:
    """FLAT: identity. UPRIGHT: (x, y, z) -> (x, -z, y) so the drawing faces -Y and stands on Z.

    Returns a new mesh (the input is never modified). The UPRIGHT map is a proper rotation
    (+90 degrees about X) so face winding and outward normals are preserved.
    """
    if orientation == "FLAT":
        return _copy(mesh)
    if orientation == "UPRIGHT":
        return _copy(mesh, verts=[(x, -z, y) for x, y, z in mesh.verts])
    raise ValueError("orientation must be FLAT or UPRIGHT, got %r" % (orientation,))


def place_origin(mesh: MeshData, mode: str, source_origin: Vec3 = (0.0, 0.0, 0.0)) -> MeshData:
    """Translate verts so the object origin is: CENTER (bbox center), BOTTOM (bbox center xy, min z),
    MIN (bbox min corner), SOURCE (``source_origin`` maps to 0).

    Returns a new mesh. An empty mesh is returned unchanged (as a copy).
    """
    if mode not in ("CENTER", "BOTTOM", "MIN", "SOURCE"):
        raise ValueError("origin mode must be CENTER, BOTTOM, MIN or SOURCE, got %r" % (mode,))
    _check_scalars(*source_origin)
    if not mesh.verts:
        return _copy(mesh)
    a = np.asarray(mesh.verts, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("mesh contains non-finite vertices")
    lo = a.min(axis=0)
    hi = a.max(axis=0)
    if mode == "CENTER":
        o = (lo + hi) / 2.0
    elif mode == "BOTTOM":
        o = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])
    elif mode == "MIN":
        o = lo
    else:
        o = np.asarray(source_origin, dtype=np.float64)
    return _copy(mesh, verts=[(float(x - o[0]), float(y - o[1]), float(z - o[2])) for x, y, z in mesh.verts])


def scale(mesh: MeshData, s: float) -> MeshData:
    """Uniformly scale vertices about the origin; a negative factor mirrors and flips the winding."""
    _check_scalars(s)
    if s == 0.0:
        raise ValueError("scale factor must be non-zero")
    verts = [(x * s, y * s, z * s) for x, y, z in mesh.verts]
    if s > 0.0:
        return _copy(mesh, verts=verts)
    # uniform negative scale is a point reflection (improper in 3D): reverse each face to keep outward normals
    faces = [tuple(reversed(f)) for f in mesh.faces]
    uvs: List[Tuple[float, float]] = []
    if mesh.uvs:
        pos = 0
        for f in mesh.faces:
            chunk = mesh.uvs[pos:pos + len(f)]
            uvs.extend(reversed(chunk))
            pos += len(f)
    return _copy(mesh, verts=verts, faces=faces, uvs=uvs if mesh.uvs else [])


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
