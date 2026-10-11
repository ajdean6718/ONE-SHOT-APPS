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
from ..seam_core import Z_STEP, geometry as core_geometry, stroke as core_stroke
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
        ob = canvas.find_by_id(key.split(":", 1)[1], scene)
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
        if img is not None:
            bpy.data.images.remove(img)
        mat = _owned_material(ob)
        if mat is not None and mat.node_tree is not None:
            _remove_gradient_nodes(mat.node_tree)
        return
    sync_material(ob)
    mat = _owned_material(ob)
    nodes = _ensure_gradient_nodes(mat)
    params = _gradient_params(st)
    digest = hashlib.blake2b(repr(params).encode(), digest_size=8).hexdigest()
    if img is not None and (tuple(img.size) != (GRADIENT_SIZE, GRADIENT_SIZE)):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(name, GRADIENT_SIZE, GRADIENT_SIZE, alpha=True, float_buffer=False)
        img.colorspace_settings.name = "sRGB"
        img.alpha_mode = "STRAIGHT"
        img["seam_hash"] = ""
    if img.get("seam_hash") != digest or not img.has_data:
        arr = bake_gradient(st.fill_type, st.gradient_angle, tuple(st.gradient_center), st.gradient_radius,
                            _stops(st))
        img.pixels.foreach_set(arr.ravel())
        img.update()
        img["seam_hash"] = digest
        try:
            img.pack()
        except RuntimeError:
            pass
    nodes["gradient"].image = img
    mat.node_tree.nodes.active = nodes["gradient"]
    bb = local_bbox(ob, depsgraph)
    n = GRADIENT_SIZE
    k = (n - 1.0) / n
    if bb is not None:
        w = max(bb[2] - bb[0], 1e-6)
        h = max(bb[3] - bb[1], 1e-6)
        sx, sy = k / w, k / h
        nodes["mapping"].inputs["Scale"].default_value = (sx, sy, 1.0)
        nodes["mapping"].inputs["Location"].default_value = (0.5 / n - bb[0] * sx, 0.5 / n - bb[1] * sy, 0.0)
    nodes["mapping"].inputs["Rotation"].default_value = (0.0, 0.0, 0.0)


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

    def _obj(self, scene, key):
        ob = canvas.find_by_id(key.split(":", 1)[1], scene)
        return ob if ob is not None and ob.seam.role in STYLED_ROLES else None

    def signature(self, scene, key):
        ob = self._obj(scene, key)
        return None if ob is None else gradient_signature(ob)

    def stored_signature(self, scene, key):
        ob = self._obj(scene, key)
        return None if ob is None else canvas.get_sig(ob, "gradient")

    def build(self, scene, key, signature):
        ob = self._obj(scene, key)
        if ob is None:
            return engine.BuildResult(key, "obsolete")
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
    return [o for o in scene.objects if o.seam.role == "STROKE" and o.seam.owner_id == owner_id]


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
    cols = ",".join(sorted(c.name for c in owner.users_collection))
    return repr((_rgba(st.stroke_color), _r(st.stroke_width), st.stroke_cap, st.stroke_join, _r(st.miter_limit),
                 _r(st.opacity), _scale_key(smat), res, psig, owner.name, cols,
                 owner.seam.id in _suspect))


def valid_stroke_child(owner):
    """The single healthy stroke child of ``owner`` or None."""
    sid = owner.seam.id
    kids = [c for c in owner.children if c.seam.role == "STROKE"]
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
        child.location = (0.0, 0.0, Z_STEP * 0.5)
        for ax in range(3):
            child.lock_location[ax] = child.lock_rotation[ax] = child.lock_scale[ax] = True
    # collections follow the owner
    want = list(owner.users_collection) or [scene.collection]
    for c in list(child.users_collection):
        if c not in want:
            c.objects.unlink(child)
    for c in want:
        if child.name not in c.objects:
            c.objects.link(child)
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
        ob = canvas.find_by_id(sid, scene)
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
        if set(child.users_collection) != set(owner.users_collection):
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
            return engine.BuildResult(key, "ok", f"removed {n} orphaned stroke object(s)")
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


def register():
    for p in _producers:
        engine.register_producer(p)


def unregister():
    for p in _producers:
        engine.unregister_producer(p.kind)
