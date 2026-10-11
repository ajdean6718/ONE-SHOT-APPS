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

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .geometry import (Ring, _RingInfo, _ring_inside, _tolist, as_points, clean_ring,
                       signed_area)

OPS = ("UNION", "DIFFERENCE", "INTERSECT", "EXCLUDE")


class BooleanUnavailable(RuntimeError):
    """Raised when mathutils is not importable (outside Blender)."""


def available() -> bool:
    try:
        import mathutils.geometry  # noqa: F401
    except Exception:
        return False
    return True


def _prepare_set(rings) -> List[np.ndarray]:
    """Validate rings and drop degenerate ones (consecutive duplicates removed)."""
    out: List[np.ndarray] = []
    for r in rings:
        a = as_points(r, "ring")
        if len(a) == 0:
            continue
        keep = np.ones(len(a), dtype=bool)
        keep[1:] = np.any(a[1:] != a[:-1], axis=1)
        a = a[keep]
        while len(a) > 1 and a[0, 0] == a[-1, 0] and a[0, 1] == a[-1, 1]:
            a = a[:-1]
        if len(a) < 3:
            continue
        if signed_area(a) == 0.0:
            continue
        out.append(a)
    return out


def _split_loops(ring: List[int]) -> List[List[int]]:
    """Split a closed vertex walk into simple loops at repeated vertices."""
    stack: List[int] = []
    pos: Dict[int, int] = {}
    loops: List[List[int]] = []
    for v in ring:
        k = pos.get(v)
        if k is None:
            pos[v] = len(stack)
            stack.append(v)
        else:
            loops.append(stack[k:])
            for u in stack[k + 1:]:
                del pos[u]
            del stack[k + 1:]
    loops.append(stack)
    return loops


def _trace_boundary(coords: List[Tuple[float, float]], edges: Dict[Tuple[int, int], int]) -> List[List[int]]:
    """Chain directed boundary edges (material on the left) into closed vertex walks."""
    out: Dict[int, List[int]] = {}
    for (u, v) in edges:
        out.setdefault(u, []).append(v)
    walks: List[List[int]] = []
    for start_u in sorted(out):
        while out[start_u]:
            u = start_u
            v = out[u].pop()
            walk = [u]
            prev_dir = None
            while True:
                dx = coords[v][0] - coords[u][0]
                dy = coords[v][1] - coords[u][1]
                prev_dir = (dx, dy)
                u = v
                options = out.get(u)
                if not options:
                    break
                if len(options) == 1:
                    v = options.pop()
                else:
                    # several boundary edges leave this (pinch) vertex: take the leftmost turn
                    best = 0
                    best_ang = -math.inf
                    for k, w in enumerate(options):
                        ex = coords[w][0] - coords[u][0]
                        ey = coords[w][1] - coords[u][1]
                        ang = math.atan2(prev_dir[0] * ey - prev_dir[1] * ex, prev_dir[0] * ex + prev_dir[1] * ey)
                        if ang > best_ang:
                            best_ang = ang
                            best = k
                    v = options.pop(best)
                walk.append(u)
            walks.append(walk)
    return walks


def _run(sets: Sequence[Sequence[np.ndarray]], fill_rule: str, epsilon: float,
         select: Callable[[List[bool]], bool]) -> List[Tuple[Ring, List[Ring]]]:
    if fill_rule not in ("EVENODD", "NONZERO"):
        raise ValueError("fill_rule must be EVENODD or NONZERO, got %r" % (fill_rule,))
    if not (epsilon > 0.0 and math.isfinite(epsilon)):
        raise ValueError("epsilon must be a positive finite number, got %r" % (epsilon,))
    try:
        from mathutils import Vector
        from mathutils.geometry import delaunay_2d_cdt
    except Exception as exc:  # pragma: no cover - exercised only outside Blender
        raise BooleanUnavailable("mathutils is not available: %s" % exc) from exc

    verts = []
    faces: List[List[int]] = []
    owner: List[Tuple[int, int]] = []  # (set index, orientation sign)
    for si, rings in enumerate(sets):
        for a in rings:
            base = len(verts)
            verts.extend(Vector((float(x), float(y))) for x, y in a.tolist())
            idx = list(range(base, base + len(a)))
            ccw = signed_area(a) > 0
            # delaunay_2d_cdt wants CCW faces (a CW face is treated as its complement)
            faces.append(idx if ccw else idx[::-1])
            owner.append((si, 1 if ccw else -1))
    if not faces:
        return []
    nsets = len(sets)
    out_verts, _edges, out_faces, _ov, _oe, orig_faces = delaunay_2d_cdt(verts, [], faces, 1, epsilon, True)
    coords = [(float(v[0]), float(v[1])) for v in out_verts]

    directed: Dict[Tuple[int, int], int] = {}
    for tri, ids in zip(out_faces, orig_faces):
        if len(tri) != 3:
            continue
        counts = [0] * nsets
        winding = [0] * nsets
        for fi in ids:
            si, sg = owner[fi]
            counts[si] += 1
            winding[si] += sg
        if fill_rule == "EVENODD":
            member = [(c & 1) == 1 for c in counts]
        else:
            member = [w != 0 for w in winding]
        if not select(member):
            continue
        a, b, c = tri
        ax, ay = coords[a]
        bx, by = coords[b]
        cx, cy = coords[c]
        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if cross == 0.0:
            continue
        if cross < 0.0:
            b, c = c, b
        directed[(a, b)] = directed.get((a, b), 0) + 1
        directed[(b, c)] = directed.get((b, c), 0) + 1
        directed[(c, a)] = directed.get((c, a), 0) + 1
    boundary = {e: n for e, n in directed.items() if (e[1], e[0]) not in directed}
    if not boundary:
        return []
    outers: List[List[Tuple[float, float]]] = []
    holes: List[List[Tuple[float, float]]] = []
    for walk in _trace_boundary(coords, boundary):
        for loop in _split_loops(walk):
            if len(loop) < 3:
                continue
            ring = clean_ring([coords[i] for i in loop])
            if len(ring) < 3:
                continue
            area = signed_area(ring)
            if abs(area) < 1e-12:
                continue
            (outers if area > 0 else holes).append(ring)
    outers.sort(key=lambda r: (min(p[0] for p in r), min(p[1] for p in r)))
    infos = [_RingInfo(np.asarray(r, dtype=np.float64)) for r in outers]
    shapes: List[Tuple[Ring, List[Ring]]] = [(r, []) for r in outers]
    for h in holes:
        hi = _RingInfo(np.asarray(h, dtype=np.float64))
        best = -1
        for k, oi in enumerate(infos):
            if abs(oi.area) > abs(hi.area) and _ring_inside(hi, oi):
                if best < 0 or abs(oi.area) < abs(infos[best].area):
                    best = k
        if best >= 0:
            shapes[best][1].append(h)
    for _outer, hs in shapes:
        hs.sort(key=lambda r: (min(p[0] for p in r), min(p[1] for p in r)))
    return shapes


def boolean(rings_a: Sequence[Sequence[Tuple[float, float]]], rings_b: Sequence[Sequence[Tuple[float, float]]],
            op: str, fill_rule: str = "EVENODD", epsilon: float = 1e-6) -> List[Tuple[Ring, List[Ring]]]:
    """Return normalized shapes [(outer_ccw, [holes_cw])] for A <op> B.

    DIFFERENCE = A minus B. EXCLUDE = symmetric difference. Empty result -> [].
    Raises ``BooleanUnavailable`` when ``mathutils`` cannot be imported and ``ValueError`` for
    an unknown ``op`` or non-finite coordinates.
    """
    if op not in OPS:
        raise ValueError("op must be one of %s, got %r" % (OPS, op))
    sets = [_prepare_set(rings_a), _prepare_set(rings_b)]
    selectors = {
        "UNION": lambda m: m[0] or m[1],
        "INTERSECT": lambda m: m[0] and m[1],
        "DIFFERENCE": lambda m: m[0] and not m[1],
        "EXCLUDE": lambda m: m[0] != m[1],
    }
    return _run(sets, fill_rule, epsilon, selectors[op])


def union_all(ring_sets: Sequence[Sequence[Sequence[Tuple[float, float]]]], fill_rule: str = "EVENODD") -> List[Tuple[Ring, List[Ring]]]:
    """Union of many compound shapes in one CDT pass."""
    sets = [_prepare_set(rs) for rs in ring_sets]
    return _run(sets, fill_rule, 1e-6, lambda m: any(m))
