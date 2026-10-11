"""Appearance of SHAPE / TEXT objects: derived materials, gradients and strokes (owner: vector-a).

Everything here is derived from ``obj.seam.style`` (single owner of appearance) by engine producers:

* ``MaterialProducer`` ("material", priority 20): material "SEAM <name>" in slot 0 with the node tree
  Emission (flat colour) mixed with Transparent by ``alpha * opacity``; the viewport display colour
  (Solid / Material colour) is the fill colour (gradients: the colour at the midpoint); a disabled fill is
  alpha 0 (wire display would draw the fill triangulation).
* ``GradientProducer`` ("gradient", priority 25): bakes the linear / radial gradient into an 8-bit sRGB
  image (gradient parameters live in bounding-box-relative space, so resizing a shape needs no
  re-bake) used through Texture Coordinate(Object) -> Mapping -> Image Texture over the shape bbox; the
  image node is the active node. NOTE: Solid > Texture colour does NOT map a gradient over a curve fill
  (Blender gives curve fills UV v = 0, u along the path); use Solid > Material colour (midpoint colour) or
  Material Preview / Rendered for the true gradient.
* ``StrokeProducer`` ("stroke", priority 30): child mesh object (role STROKE) built by
  ``seam_core.stroke.stroke_mesh`` from the flattened local paths. The geometry is built in the space of
  the owner's 2D scale/shear so the stroke keeps a uniform width when the owner is scaled.

Gradient pixels and undo: Blender 5.x memfile undo restores an image's properties (``seam_hash``, the owner's
stored signature) but keeps the in-memory pixel buffer, which may hold colours baked *after* the restored step.
The pixels are therefore validated in memory (``_gradient_applied``: owner id -> digest of the gradient the
buffer was checked or baked for), forgotten on undo / redo / load; the next flush compares the buffer with a
fresh bake and re-bakes only when it differs.

Gradient angle: 0 = left to right, 90 degrees = bottom to top (canvas Y up). Radial: ``gradient_center``
and ``gradient_radius`` are fractions of the bounding box (an ellipse for non-square shapes).
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np
from bpy.props import IntProperty
from mathutils import Matrix

from .. import ids
from ..bridge import canvas_local_matrix, curve_paths, paths_signature, text_paths
from ..engine import core as engine
from ..seam_core import geometry as core_geometry, stroke as core_stroke
from . import canvas

STYLED_ROLES = ("SHAPE", "TEXT")
GRADIENT_SIZE = 128
_NODE_NAMES = ("SEAM_texcoord", "SEAM_mapping", "SEAM_gradient", "SEAM_alpha")


# ------------------------------------------------------------------ colour helpers

def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_to_linear(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def _r(v, n=5):
    return round(float(v), n)


def _rgba(c):
    return tuple(_r(x) for x in c)


# ------------------------------------------------------------------ per-pass lookup index

class SceneIndex:
    """One scan of the objects, shared by every lookup of one engine pass (``engine.index_generation``).

    Replaces O(objects) scans per key (``Object.children``, owner -> strokes, mask -> clipped objects) that made a
    revalidation O(objects^2). Lists are *candidates*: lookups re-check the real predicate (and skip removed
    objects), and producers that create derived objects during a pass add them with :meth:`add`."""

    def __init__(self, scene):
        self.scene_ptr = scene.as_pointer()
        self.children: Dict[int, list] = {}
        self.strokes: Dict[str, list] = {}
        self.clipped: Dict[int, list] = {}
        self.by_id: Dict[str, object] = {}
        self._cols: Optional[Dict[int, list]] = None
        for ob in bpy.data.objects:
            p = ob.parent
            if p is not None:
                self.children.setdefault(p.as_pointer(), []).append(ob)
        for ob in scene.objects:
            self._add_scene_object(ob)

    def _add_scene_object(self, ob) -> None:
        seam = ob.seam
        if seam.id:
            self.by_id.setdefault(seam.id, ob)
        if seam.role == "STROKE" and seam.owner_id:
            self.strokes.setdefault(seam.owner_id, []).append(ob)
        m = seam.clip_source
        if m is not None and m != ob:
            self.clipped.setdefault(m.as_pointer(), []).append(ob)

    def add(self, ob) -> None:
        """Register an object created during the pass (derived child)."""
        p = ob.parent
        if p is not None:
            self.children.setdefault(p.as_pointer(), []).append(ob)
        self._add_scene_object(ob)
        self.forget_collections(ob)

    def collections(self, ob) -> list:
        """``ob.users_collection`` (an O(collections x objects) Python property) from one scan per pass."""
        if self._cols is None:
            cols: Dict[int, list] = {}
            for col in bpy.data.collections:
                for o in col.objects:
                    cols.setdefault(o.as_pointer(), []).append(col)
            for sc in bpy.data.scenes:
                for o in sc.collection.objects:
                    cols.setdefault(o.as_pointer(), []).append(sc.collection)
            self._cols = cols
        got = self._cols.get(ob.as_pointer())
        if got is None:
            return list(ob.users_collection)
        return got

    def forget_collections(self, ob) -> None:
        """The pass itself (un)linked ``ob``: answer from Blender for it from now on."""
        if self._cols is not None:
            self._cols.pop(ob.as_pointer(), None)


_index: Dict[str, object] = {"gen": None, "index": None}


def scene_index(scene) -> Optional[SceneIndex]:
    """The index of the current engine pass, or None outside the engine (then callers scan directly)."""
    gen = engine.index_generation()
    if gen is None:
        return None
    idx = _index["index"]
    if _index["gen"] != gen or idx is None or idx.scene_ptr != scene.as_pointer():
        idx = SceneIndex(scene)
        _index["gen"], _index["index"] = gen, idx
    return idx


def index_add(scene, ob) -> None:
    """Producers call this after creating a derived object during a pass."""
    idx = _index["index"] if _index["gen"] == engine.index_generation() else None
    if idx is not None and idx.scene_ptr == scene.as_pointer():
        idx.add(ob)


def _alive(objs):
    out = []
    for o in objs:
        try:
            o.name
        except ReferenceError:
            continue
        out.append(o)
    return out


def users_collection(ob, scene=None) -> list:
    idx = scene_index(scene or bpy.context.scene)
    return list(ob.users_collection) if idx is None else list(idx.collections(ob))


def find_styled(scene, sid: str):
    """``canvas.find_by_id`` through the pass index (verified) when the engine is running."""
    idx = scene_index(scene)
    if idx is not None:
        ob = idx.by_id.get(sid)
        try:
            if ob is not None and ob.seam.id == sid:
                return ob
        except ReferenceError:
            pass
    return canvas.find_by_id(sid, scene)


def children_of(obj, scene=None):
    """``obj.children`` (O(objects) in Blender) through the pass index when the engine is running."""
    idx = scene_index(scene or bpy.context.scene)
    if idx is None:
        return list(obj.children)
    return [c for c in _alive(idx.children.get(obj.as_pointer(), ())) if c.parent == obj]


# ------------------------------------------------------------------ geometry access

def owner_paths(obj, depsgraph=None):
    if obj.type == "CURVE":
        return curve_paths(obj)
    if obj.type == "FONT":
        return text_paths(obj, depsgraph)
    return []


def paths_bbox(paths, samples: int = 16):
    """Bounding box (minx, miny, maxx, maxy) of Bezier paths by sampling the cubic segments."""
    pts = []
    t = np.linspace(0.0, 1.0, samples)[:, None]
    for p in paths:
        n = len(p.points)
        segs = n if p.closed else n - 1
        if n == 1:
            pts.append(np.array([p.points[0].co]))
        for i in range(max(segs, 0)):
            a = p.points[i]
            b = p.points[(i + 1) % n]
            p0, p1, p2, p3 = (np.array(v, dtype=np.float64) for v in (a.co, a.handle_right, b.handle_left, b.co))
            u = 1.0 - t
            pts.append(u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3)
    if not pts:
        return None
    allp = np.vstack(pts)
    return (float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max()))


def local_bbox(obj, depsgraph=None):
    return paths_bbox(owner_paths(obj, depsgraph))


def _shape_objects(scene):
    return [o for o in scene.objects if o.seam.role in STYLED_ROLES]


def _data_users(id_orig):
    for ob in bpy.data.objects:
        if ob.data is id_orig and ob.seam.role in STYLED_ROLES and ob.seam.id:
            yield ob


def purge_owner_leftovers(sid: str) -> int:
    """Remove the derived datablocks of a SEAM object that no longer exists: its gradient image, its (unused)
    curve data and its SEAM material. Returns the number of datablocks removed."""
    n = 0
    img = bpy.data.images.get("SEAM Gradient " + sid)
    if img is not None and _is_gradient_image(img):
        bpy.data.images.remove(img)
        n += 1
    mats = [m for m in bpy.data.materials if m.get("seam_owner") == sid]
    for m in mats:
        for coll in (bpy.data.curves,):
            for cu in [c for c in coll if c.users == 0 and any(x is m for x in c.materials)]:
                coll.remove(cu)
                n += 1
        if m.users == 0:
            bpy.data.materials.remove(m)
            n += 1
    return n


def _is_gradient_image(img) -> bool:
    """Only images created by sync_gradient carry this marker; name prefixes are not ownership."""
    return img.get("seam_kind") == "gradient"


def _orphan_gradient_ids(scene) -> List[str]:
    live = {o.seam.id for o in _shape_objects(scene) if o.seam.id}
    out = []
    for img in bpy.data.images:
        if img.name.startswith("SEAM Gradient ") and _is_gradient_image(img):
            sid = img.name[len("SEAM Gradient "):]
            if sid not in live:
                out.append(sid)
    return out


# ------------------------------------------------------------------ materials

def _owned_material(ob) -> Optional[bpy.types.Material]:
    data = ob.data
    if data is None or not hasattr(data, "materials") or len(data.materials) == 0:
        return None
    mat = data.materials[0]
    if mat is not None and mat.get("seam_owner") == ob.seam.id:
        return mat
    return None


def ensure_material(ob) -> bpy.types.Material:
    canvas.ensure_single_user_data(ob)  # a datablock shared with another object would swap materials forever
    mat = _owned_material(ob)
    if mat is None:
        mat = bpy.data.materials.new("SEAM " + ob.name)
        mat["seam_owner"] = ob.seam.id
        if len(ob.data.materials) == 0:
            ob.data.materials.append(mat)
        else:
            ob.data.materials[0] = mat
    if mat.name != "SEAM " + ob.name:
        mat.name = "SEAM " + ob.name
    return mat


def _remove_gradient_nodes(nt) -> None:
    for name in _NODE_NAMES:
        n = nt.nodes.get(name)
        if n is not None:
            nt.nodes.remove(n)


def _ensure_gradient_nodes(mat) -> dict:
    nodes = canvas.flat_material_tree(mat)
    nt = mat.node_tree
    if not all(n in nt.nodes for n in _NODE_NAMES):
        _remove_gradient_nodes(nt)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        tc.name = "SEAM_texcoord"
        tc.location = (-700, -200)
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.name = "SEAM_mapping"
        mp.vector_type = "POINT"
        mp.location = (-500, -200)
        tx = nt.nodes.new("ShaderNodeTexImage")
        tx.name = "SEAM_gradient"
        tx.interpolation = "Linear"
        tx.extension = "EXTEND"
        tx.location = (-300, -200)
        ma = nt.nodes.new("ShaderNodeMath")
        ma.name = "SEAM_alpha"
        ma.operation = "MULTIPLY"
        ma.location = (0, 250)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], tx.inputs["Vector"])
        nt.links.new(tx.outputs["Color"], nodes["emission"].inputs["Color"])
        nt.links.new(tx.outputs["Alpha"], ma.inputs[0])
        nt.links.new(ma.outputs[0], nodes["mix"].inputs[0])
    out = dict(nodes)
    out.update({n[5:]: nt.nodes[n] for n in _NODE_NAMES})
    return out


def _stops(style) -> List[Tuple[float, Tuple[float, float, float, float]]]:
    stops = sorted(((float(s.position), tuple(float(c) for c in s.color)) for s in style.gradient_stops),
                   key=lambda s: s[0])
    if not stops:
        stops = [(0.0, tuple(style.fill_color)), (1.0, tuple(style.fill_color))]
    return stops


def gradient_display_color(style) -> Tuple[float, float, float, float]:
    """Colour shown for a gradient fill in Solid > Material colour mode: the gradient at its midpoint.

    (Solid > Texture colour cannot map a curve fill over its bounding box: Blender gives curve fills
    UV v = 0 and u along the path. Material Preview / Rendered and exports show the true gradient.)
    """
    stops = _stops(style)
    pos = [s[0] for s in stops]
    return tuple(float(np.interp(0.5, pos, [s[1][i] for s in stops])) for i in range(4))


def material_signature(ob) -> str:
    st = ob.seam.style
    mat = _owned_material(ob)
    has_grad = bool(mat is not None and mat.node_tree is not None and "SEAM_gradient" in mat.node_tree.nodes)
    flat_ok = bool(mat is not None and mat.node_tree is not None and "SEAM_mix" in mat.node_tree.nodes)
    first = _rgba(gradient_display_color(st)) if st.fill_type != "SOLID" else ()
    return repr((ob.name, bool(st.fill_enabled), st.fill_type, _rgba(st.fill_color), _r(st.opacity), first,
                 mat is not None, flat_ok, has_grad == (st.fill_type != "SOLID")))


def sync_material(ob) -> bpy.types.Material:
    """Make slot 0 of ``ob`` the SEAM material for its current style (idempotent)."""
    st = ob.seam.style
    mat = ensure_material(ob)
    gradient = st.fill_type != "SOLID"
    if gradient:
        nodes = _ensure_gradient_nodes(mat)
        nodes["alpha"].inputs[1].default_value = float(st.opacity) if st.fill_enabled else 0.0
        base = gradient_display_color(st)
        canvas.set_flat_color(mat, base, 1.0, st.fill_enabled)
        mat.diffuse_color = (base[0], base[1], base[2], base[3] * float(st.opacity) if st.fill_enabled else 0.0)
    else:
        canvas.flat_material_tree(mat)
        _remove_gradient_nodes(mat.node_tree)
        canvas.set_flat_color(mat, tuple(st.fill_color), float(st.opacity), bool(st.fill_enabled))
    if ob.display_type != "TEXTURED":
        ob.display_type = "TEXTURED"  # WIRE would draw the fill triangulation; a disabled fill is alpha 0 instead
    return mat


class MaterialProducer(engine.Producer):
    kind = "material"
    priority = 20

    def keys_for_update(self, scene, id_orig, geometry, transform):
        if isinstance(id_orig, bpy.types.Object):
            if id_orig.seam.role in STYLED_ROLES and id_orig.seam.id:
                yield "material:" + id_orig.seam.id

    def all_keys(self, scene):
        for ob in _shape_objects(scene):
            if ob.seam.id:
                yield "material:" + ob.seam.id

    def _obj(self, scene, key):
        ob = find_styled(scene, key.split(":", 1)[1])
        return ob if ob is not None and ob.seam.role in STYLED_ROLES else None

    def signature(self, scene, key):
        ob = self._obj(scene, key)
        return None if ob is None else material_signature(ob)

    def stored_signature(self, scene, key):
        ob = self._obj(scene, key)
        return None if ob is None else canvas.get_sig(ob, "material")

    def build(self, scene, key, signature):
        ob = self._obj(scene, key)
        if ob is None:
            return engine.BuildResult(key, "obsolete")
        sync_material(ob)
        canvas.set_sig(ob, "material", material_signature(ob))
        return engine.BuildResult(key, "ok")


# ------------------------------------------------------------------ gradients

def bake_gradient(fill_type: str, angle: float, center, radius: float, stops, size: int = GRADIENT_SIZE) -> np.ndarray:
    """Return a (size, size, 4) float32 array of sRGB-encoded straight RGBA (row 0 = bottom, as Blender).

    Colours are interpolated in scene-linear space; positions are in bbox-relative (u, v) with v up.
    """
    n = size
    u = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(u, u)  # uu varies along columns, vv along rows (row 0 = v 0)
    if fill_type == "RADIAL":
        r = max(float(radius), 1e-6)
        t = np.hypot(uu - center[0], vv - center[1]) / r
    else:
        dx, dy = math.cos(angle), math.sin(angle)
        denom = abs(dx) + abs(dy)
        t = 0.5 + ((uu - 0.5) * dx + (vv - 0.5) * dy) / denom
    t = np.clip(t, 0.0, 1.0)
    pos = np.array([s[0] for s in stops], dtype=np.float64)
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    out = np.empty((n, n, 4), dtype=np.float64)
    for ch in range(4):
        out[..., ch] = np.interp(t, pos, cols[:, ch])
    out[..., :3] = linear_to_srgb(out[..., :3])
    out[..., 3] = np.clip(out[..., 3], 0.0, 1.0)
    return np.round(out * 255.0).astype(np.float32) / 255.0


_gradient_applied: Dict[str, str] = {}  # owner id -> digest whose pixels are known to be in the image buffer


def gradient_digest(st) -> str:
    return hashlib.blake2b(repr(_gradient_params(st)).encode(), digest_size=8).hexdigest()


def forget_gradient_pixels() -> None:
    """Undo / redo / load: the image buffers may not match the restored properties any more."""
    _gradient_applied.clear()


def gradient_image_name(ob) -> str:
    return "SEAM Gradient " + ob.seam.id


def _gradient_params(st):
    return (st.fill_type, _r(st.gradient_angle), tuple(_r(c) for c in st.gradient_center), _r(st.gradient_radius),
            tuple((_r(p), _rgba(c)) for p, c in _stops(st)))


def gradient_signature(ob, depsgraph=None) -> str:
    st = ob.seam.style
    if st.fill_type == "SOLID":
        has_img = bpy.data.images.get(gradient_image_name(ob)) is not None
        return "solid|img" if has_img else "solid"
    bb = local_bbox(ob, depsgraph)
    mat = _owned_material(ob)
    ok = bool(mat is not None and mat.node_tree is not None and "SEAM_gradient" in mat.node_tree.nodes)
    img = bpy.data.images.get(gradient_image_name(ob))
    return repr((_gradient_params(st), tuple(_r(v, 4) for v in bb) if bb else None, ok, img is not None,
                 ob.name))


def sync_gradient(ob, depsgraph=None) -> None:
    st = ob.seam.style
    name = gradient_image_name(ob)
    img = bpy.data.images.get(name)
    if st.fill_type == "SOLID":
        if img is not None and _is_gradient_image(img):
            bpy.data.images.remove(img)
        mat = _owned_material(ob)
        if mat is not None and mat.node_tree is not None:
            _remove_gradient_nodes(mat.node_tree)
        return
    sync_material(ob)
    mat = _owned_material(ob)
    nodes = _ensure_gradient_nodes(mat)
    digest = gradient_digest(st)
    if img is not None and (tuple(img.size) != (GRADIENT_SIZE, GRADIENT_SIZE)):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(name, GRADIENT_SIZE, GRADIENT_SIZE, alpha=True, float_buffer=False)
        img.colorspace_settings.name = "sRGB"
        img["seam_kind"] = "gradient"
        img["seam_owner"] = ob.seam.id
        img.alpha_mode = "STRAIGHT"
        img["seam_hash"] = ""
    if img.get("seam_hash") != digest or not img.has_data or _gradient_applied.get(ob.seam.id) != digest:
        arr = bake_gradient(st.fill_type, st.gradient_angle, tuple(st.gradient_center), st.gradient_radius,
                            _stops(st)).ravel()
        same = False
        if img.get("seam_hash") == digest and img.has_data:  # after undo/load: verify the buffer, re-bake on mismatch
            cur = np.empty(arr.size, dtype=np.float32)
            img.pixels.foreach_get(cur)
            same = bool(np.allclose(cur, arr, atol=0.5 / 255.0))
        if not same:
            img.pixels.foreach_set(arr)
            img.update()
            img["seam_hash"] = digest
            try:
                img.pack()
            except RuntimeError:
                pass
        _gradient_applied[ob.seam.id] = digest
    # write only what differs: on Blender 5.x assigning an image-node image (even the same one) re-tags every
    # object of the scene for update, which made one node edit cost O(document) in the next depsgraph pass
    if nodes["gradient"].image != img:
        nodes["gradient"].image = img
    if mat.node_tree.nodes.active != nodes["gradient"]:
        mat.node_tree.nodes.active = nodes["gradient"]
    bb = local_bbox(ob, depsgraph)
    n = GRADIENT_SIZE
    k = (n - 1.0) / n
    if bb is not None:
        w = max(bb[2] - bb[0], 1e-6)
        h = max(bb[3] - bb[1], 1e-6)
        sx, sy = k / w, k / h
        _set_vector(nodes["mapping"].inputs["Scale"], (sx, sy, 1.0))
        _set_vector(nodes["mapping"].inputs["Location"], (0.5 / n - bb[0] * sx, 0.5 / n - bb[1] * sy, 0.0))
    _set_vector(nodes["mapping"].inputs["Rotation"], (0.0, 0.0, 0.0))


def _set_vector(socket, value) -> None:
    if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(socket.default_value, value)):
        socket.default_value = value


class GradientProducer(engine.Producer):
    kind = "gradient"
    priority = 25

    def keys_for_update(self, scene, id_orig, geometry, transform):
        if isinstance(id_orig, bpy.types.Object):
            if id_orig.seam.role in STYLED_ROLES and id_orig.seam.id and (geometry or not transform):
                yield "gradient:" + id_orig.seam.id
        elif isinstance(id_orig, bpy.types.Curve):
            for ob in _data_users(id_orig):
                yield "gradient:" + ob.seam.id

    def all_keys(self, scene):
        for ob in _shape_objects(scene):
            if ob.seam.id:
                yield "gradient:" + ob.seam.id
        for sid in _orphan_gradient_ids(scene):
            yield "gradient:" + sid

    def _obj(self, scene, key):
        ob = find_styled(scene, key.split(":", 1)[1])
        return ob if ob is not None and ob.seam.role in STYLED_ROLES else None

    def signature(self, scene, key):
        ob = self._obj(scene, key)
        if ob is None:
            sid = key.split(":", 1)[1]
            return "orphan" if bpy.data.images.get("SEAM Gradient " + sid) is not None else None
        return gradient_signature(ob)

    def stored_signature(self, scene, key):
        ob = self._obj(scene, key)
        if ob is None:
            return None
        if ob.seam.style.fill_type != "SOLID" and _gradient_applied.get(ob.seam.id) != gradient_digest(ob.seam.style):
            return None  # pixel buffer not verified since the last undo / redo / load (or never baked here)
        return canvas.get_sig(ob, "gradient")

    def build(self, scene, key, signature):
        ob = self._obj(scene, key)
        if ob is None:
            n = purge_owner_leftovers(key.split(":", 1)[1])
            return engine.BuildResult(key, "ok", f"removed {n} leftover datablock(s)") if n else \
                engine.BuildResult(key, "obsolete")
        sync_gradient(ob)
        canvas.set_sig(ob, "gradient", gradient_signature(ob))
        return engine.BuildResult(key, "ok")


# ------------------------------------------------------------------ strokes

_last_scale: Dict[str, tuple] = {}
_suspect: set = set()


def scale_matrices(owner, scene=None):
    """(S, S^-1) 2x2 numpy matrices: the rotation-free symmetric part of the owner's canvas-local linear map."""
    m = np.array(canvas_local_matrix(owner, scene))[:2, :2]
    u, s, vt = np.linalg.svd(m)
    s = np.maximum(s, 1e-9)
    smat = vt.T @ np.diag(s) @ vt
    sinv = vt.T @ np.diag(1.0 / s) @ vt
    return smat, sinv


def _scale_key(smat) -> tuple:
    return tuple(round(float(v), 4) for v in smat.ravel())


def strokes_of(scene, owner_id: str) -> List[bpy.types.Object]:
    idx = scene_index(scene)
    cands = scene.objects if idx is None else _alive(idx.strokes.get(owner_id, ()))
    return [o for o in cands if o.seam.role == "STROKE" and o.seam.owner_id == owner_id]


def stroke_enabled(st) -> bool:
    return bool(st.stroke_enabled and st.stroke_width > 0.0)


def stroke_signature(owner, scene, depsgraph=None) -> str:
    st = owner.seam.style
    if not stroke_enabled(st):
        return "none"
    smat, _ = scale_matrices(owner, scene)
    _last_scale[owner.seam.id] = _scale_key(smat)
    res = max(int(getattr(owner.data, "resolution_u", 12) or 12), 2)
    psig = paths_signature(owner_paths(owner, depsgraph))
    cols = ",".join(sorted(c.name for c in users_collection(owner, scene)))
    return repr((_rgba(st.stroke_color), _r(st.stroke_width), st.stroke_cap, st.stroke_join, _r(st.miter_limit),
                 _r(st.opacity), _scale_key(smat), res, psig, owner.name, cols,
                 owner.seam.id in _suspect, round(canvas.stroke_z(owner), 8)))


def valid_stroke_child(owner):
    """The single healthy stroke child of ``owner`` or None."""
    sid = owner.seam.id
    kids = [c for c in children_of(owner) if c.seam.role == "STROKE"]
    good = [c for c in kids if c.seam.owner_id == sid]
    if len(good) == 1 and len(kids) == 1:
        return good[0]
    return None


def _stroke_geometry(owner, depsgraph):
    st = owner.seam.style
    smat, sinv = scale_matrices(owner)
    res = max(int(getattr(owner.data, "resolution_u", 12) or 12), 2)
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for path in owner_paths(owner, depsgraph):
        poly = core_geometry.flatten_path(path, res)
        if len(poly) < 2:
            continue
        arr = np.array(poly, dtype=np.float64) @ smat.T
        pts = [(float(x), float(y)) for x, y in arr]
        v2, tris = core_stroke.stroke_mesh(pts, bool(path.closed), float(st.stroke_width), st.stroke_cap,
                                           st.stroke_join, float(st.miter_limit))
        if not v2:
            continue
        loc = np.array(v2, dtype=np.float64) @ sinv.T
        base = len(verts)
        verts.extend((float(x), float(y), 0.0) for x, y in loc)
        for a, b, c in tris:
            if a == b or b == c or a == c:
                continue
            ax, ay = loc[a]
            bx, by = loc[b]
            cx, cy = loc[c]
            area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area2) < 1e-14:
                continue
            faces.append((base + a, base + b, base + c) if area2 > 0 else (base + a, base + c, base + b))
    return verts, faces


def _delete_stroke_object(ob) -> None:
    me = ob.data if ob.type == "MESH" else None
    mats = list(me.materials) if me is not None else []
    bpy.data.objects.remove(ob, do_unlink=True)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)
    for m in mats:
        if m is not None and m.users == 0:
            bpy.data.materials.remove(m)


def remove_strokes(scene, owner_id: str, keep=None) -> int:
    n = 0
    for o in strokes_of(scene, owner_id):
        if o is not keep:
            _delete_stroke_object(o)
            n += 1
    return n


def sync_stroke(owner, scene, depsgraph=None) -> Optional[bpy.types.Object]:
    """Create / update / remove the stroke child of ``owner`` to match its style."""
    st = owner.seam.style
    sid = owner.seam.id
    if not stroke_enabled(st):
        remove_strokes(scene, sid)
        _suspect.discard(sid)
        return None
    keep = valid_stroke_child(owner)
    remove_strokes(scene, sid, keep=keep)
    _suspect.discard(sid)
    verts, faces = _stroke_geometry(owner, depsgraph)
    child = keep
    if child is None:
        me = bpy.data.meshes.new("SEAM Stroke " + owner.name)
        child = bpy.data.objects.new("SEAM Stroke " + owner.name, me)
        ids.ensure_id(child)
        child.seam.role = "STROKE"
        child.seam.owner_id = sid
        child.hide_select = True
        child.parent = owner
        child.matrix_parent_inverse = Matrix.Identity(4)
        index_add(scene, child)
        child.location = (0.0, 0.0, canvas.stroke_z(owner))
        for ax in range(3):
            child.lock_location[ax] = child.lock_rotation[ax] = child.lock_scale[ax] = True
    if child.location.z != canvas.stroke_z(owner):  # the owner may have moved into / out of a group
        child.location.z = canvas.stroke_z(owner)
    # collections follow the owner
    want = users_collection(owner, scene) or [scene.collection]
    relinked = False
    for c in users_collection(child, scene):
        if c not in want:
            c.objects.unlink(child)
            relinked = True
    for c in want:
        if child.name not in c.objects:
            c.objects.link(child)
            relinked = True
    if relinked:
        idx = scene_index(scene)
        if idx is not None:
            idx.forget_collections(child)
    if child.name != "SEAM Stroke " + owner.name:
        child.name = "SEAM Stroke " + owner.name
    me = child.data
    me.clear_geometry()
    if verts and faces:
        me.from_pydata(verts, [], faces)
        me.update()
    mat = me.materials[0] if len(me.materials) else None
    if mat is None or mat.get("seam_owner") != child.seam.id:
        mat = bpy.data.materials.new("SEAM Stroke " + owner.name)
        mat["seam_owner"] = child.seam.id
        if len(me.materials):
            me.materials[0] = mat
        else:
            me.materials.append(mat)
    if mat.name != "SEAM Stroke " + owner.name:
        mat.name = "SEAM Stroke " + owner.name
    canvas.set_flat_color(mat, tuple(st.stroke_color), float(st.opacity), True)
    return child


class StrokeProducer(engine.Producer):
    kind = "stroke"
    priority = 30

    def keys_for_update(self, scene, id_orig, geometry, transform):
        if isinstance(id_orig, bpy.types.Object):
            role = id_orig.seam.role
            if role in STYLED_ROLES and id_orig.seam.id:
                sid = id_orig.seam.id
                if transform and not geometry:
                    if not stroke_enabled(id_orig.seam.style):
                        return
                    if _last_scale.get(sid) == _scale_key(scale_matrices(id_orig, scene)[0]):
                        return
                yield "stroke:" + sid
            elif role == "STROKE" and id_orig.seam.owner_id:
                parent = id_orig.parent
                if parent is None or parent.seam.id != id_orig.seam.owner_id:
                    _suspect.add(id_orig.seam.owner_id)
                    yield "stroke:" + id_orig.seam.owner_id
        elif isinstance(id_orig, bpy.types.Curve):
            for ob in _data_users(id_orig):
                yield "stroke:" + ob.seam.id

    def all_keys(self, scene):
        owners = set()
        for ob in scene.objects:
            if ob.seam.role in STYLED_ROLES and ob.seam.id:
                owners.add(ob.seam.id)
            elif ob.seam.role == "STROKE" and ob.seam.owner_id:
                owners.add(ob.seam.owner_id)
        return ["stroke:" + i for i in owners]

    def _owner(self, scene, sid):
        ob = find_styled(scene, sid)
        return ob if ob is not None and ob.seam.role in STYLED_ROLES else None

    def signature(self, scene, key):
        sid = key.split(":", 1)[1]
        owner = self._owner(scene, sid)
        if owner is None:
            return "orphan" if strokes_of(scene, sid) else None
        return stroke_signature(owner, scene)

    def stored_signature(self, scene, key):
        sid = key.split(":", 1)[1]
        owner = self._owner(scene, sid)
        if owner is None:
            return None
        if not stroke_enabled(owner.seam.style):
            return "none" if not strokes_of(scene, sid) else None
        child = valid_stroke_child(owner)
        if child is None:
            return None
        if set(users_collection(child, scene)) != set(users_collection(owner, scene)):
            return None
        me = child.data
        if me is None or not len(me.materials) or me.materials[0] is None:
            return None
        return canvas.get_sig(child, "stroke")

    def build(self, scene, key, signature):
        sid = key.split(":", 1)[1]
        owner = self._owner(scene, sid)
        if owner is None:
            n = remove_strokes(scene, sid)
            _suspect.discard(sid)
            _last_scale.pop(sid, None)
            leftovers = purge_owner_leftovers(sid)
            return engine.BuildResult(key, "ok", f"removed {n} orphaned stroke object(s), {leftovers} leftover datablock(s)")
        child = sync_stroke(owner, scene)
        if child is not None:
            canvas.set_sig(child, "stroke", stroke_signature(owner, scene))
        return engine.BuildResult(key, "ok")


# ------------------------------------------------------------------ gradient stop operators

def _active_styled(context):
    ob = context.active_object
    return ob if ob is not None and ob.seam.role in STYLED_ROLES else None


class SEAM_OT_gradient_stop_add(bpy.types.Operator):
    bl_idname = "seam.gradient_stop_add"
    bl_label = "Add Gradient Stop"
    bl_description = "Add a gradient stop (colour interpolated at the midpoint of the largest gap)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if _active_styled(context) is None:
            cls.poll_message_set("Select a shape or text object")
            return False
        return True

    def execute(self, context):
        st = _active_styled(context).seam.style
        stops = _stops(st)
        # largest gap
        best = (0.0, 0.0, 1.0)
        pts = [(0.0, stops[0][1])] + stops + [(1.0, stops[-1][1])]
        for (p0, c0), (p1, c1) in zip(pts, pts[1:]):
            if p1 - p0 >= best[0]:
                best = (p1 - p0, p0, p1)
        mid = (best[1] + best[2]) * 0.5
        col = [float(np.interp(mid, [s[0] for s in stops], [s[1][i] for s in stops])) for i in range(4)]
        new = st.gradient_stops.add()
        new.position = mid
        new.color = col
        return {"FINISHED"}


class SEAM_OT_gradient_stop_remove(bpy.types.Operator):
    bl_idname = "seam.gradient_stop_remove"
    bl_label = "Remove Gradient Stop"
    bl_description = "Remove a gradient stop (at least two stops are kept)"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(name="Index", default=-1, description="Stop index (-1: the last one)")

    @classmethod
    def poll(cls, context):
        ob = _active_styled(context)
        if ob is None:
            cls.poll_message_set("Select a shape or text object")
            return False
        if len(ob.seam.style.gradient_stops) <= 2:
            cls.poll_message_set("A gradient needs at least two stops")
            return False
        return True

    def execute(self, context):
        st = _active_styled(context).seam.style
        i = self.index if self.index >= 0 else len(st.gradient_stops) - 1
        if not (0 <= i < len(st.gradient_stops)):
            return {"CANCELLED"}
        st.gradient_stops.remove(i)
        return {"FINISHED"}


class SEAM_OT_gradient_stops_reset(bpy.types.Operator):
    bl_idname = "seam.gradient_stops_reset"
    bl_label = "Reset Gradient Stops"
    bl_description = "Replace the stops by fill colour -> white"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if _active_styled(context) is None:
            cls.poll_message_set("Select a shape or text object")
            return False
        return True

    def execute(self, context):
        st = _active_styled(context).seam.style
        st.gradient_stops.clear()
        a = st.gradient_stops.add()
        a.position, a.color = 0.0, tuple(st.fill_color)
        b = st.gradient_stops.add()
        b.position, b.color = 1.0, (1.0, 1.0, 1.0, 1.0)
        return {"FINISHED"}


class SEAM_PT_appearance(bpy.types.Panel):
    bl_label = "Appearance"
    bl_idname = "SEAM_PT_appearance"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"
    bl_order = 30

    @classmethod
    def poll(cls, context):
        return _active_styled(context) is not None

    def draw(self, context):
        layout = self.layout
        ob = _active_styled(context)
        st = ob.seam.style
        box = layout.box()
        row = box.row()
        row.prop(st, "fill_enabled", text="Fill")
        sub = box.column()
        sub.active = st.fill_enabled
        sub.prop(st, "fill_type", text="")
        if st.fill_type == "SOLID":
            sub.prop(st, "fill_color", text="")
        else:
            if len(st.gradient_stops) == 0:
                sub.operator("seam.gradient_stops_reset", icon="ADD", text="Create Stops")
            for i, stop in enumerate(st.gradient_stops):
                r = sub.row(align=True)
                r.prop(stop, "position", text="")
                r.prop(stop, "color", text="")
                op = r.operator("seam.gradient_stop_remove", icon="X", text="")
                op.index = i
            r = sub.row(align=True)
            r.operator("seam.gradient_stop_add", icon="ADD", text="Add Stop")
            r.operator("seam.gradient_stops_reset", icon="LOOP_BACK", text="Reset")
            if st.fill_type == "LINEAR":
                sub.prop(st, "gradient_angle")
            else:
                sub.prop(st, "gradient_center")
                sub.prop(st, "gradient_radius")
        sub.prop(st, "fill_rule", text="Rule")
        box = layout.box()
        box.prop(st, "stroke_enabled", text="Stroke")
        sub = box.column()
        sub.active = st.stroke_enabled
        sub.prop(st, "stroke_color", text="")
        sub.prop(st, "stroke_width")
        sub.prop(st, "stroke_cap")
        sub.prop(st, "stroke_join")
        if st.stroke_join == "MITER":
            sub.prop(st, "miter_limit")
        layout.prop(st, "opacity", slider=True)


classes = (SEAM_OT_gradient_stop_add, SEAM_OT_gradient_stop_remove, SEAM_OT_gradient_stops_reset,
           SEAM_PT_appearance)

_producers = (MaterialProducer(), GradientProducer(), StrokeProducer())


@bpy.app.handlers.persistent
def _on_history_or_load(*_args):
    forget_gradient_pixels()


_HANDLER_LISTS = ("undo_post", "redo_post", "load_post")


def register():
    for p in _producers:
        engine.register_producer(p)
    for name in _HANDLER_LISTS:
        lst = getattr(bpy.app.handlers, name)
        if _on_history_or_load not in lst:
            lst.append(_on_history_or_load)
    forget_gradient_pixels()


def unregister():
    for p in _producers:
        engine.unregister_producer(p.kind)
    for name in _HANDLER_LISTS:
        lst = getattr(bpy.app.handlers, name)
        while _on_history_or_load in lst:
            lst.remove(_on_history_or_load)
    forget_gradient_pixels()
