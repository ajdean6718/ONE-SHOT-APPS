"""Clipping masks (derived clipped geometry) (owner: vector-b).

Data model
----------
``obj.seam.clip_source = mask`` clips ``obj`` to the filled area of ``mask`` (a vector SHAPE).
Several objects may share one mask.  The mask is a normal editable object but is never drawn
by exports (like an Illustrator clipping path); the clipped object keeps its full, editable
geometry.

Display approach (viewport)
---------------------------
Blender cannot clip a curve's fill natively, so ``ClipProducer`` (engine kind ``"clip"``,
priority 40) derives the visible result instead of modifying the source:

* a mesh child ``<object> Clip`` (role ``CLIP_RESULT``, ``owner_id`` = clipped object id,
  not selectable) holds the triangulated ``fill AND mask`` area (material slot 0) and the
  ``stroke AND mask`` area (slot 1) in the clipped object's local space, so it follows the
  object's transform for free and is rebuilt only when the geometry, style, or mask changes
  (signature = geometry + transform + mask + relevant style);
* the *original* curve is switched to ``display_type = 'WIRE'`` while clipped, so the unclipped
  fill disappears but the path stays visible and node-editable (edit mode shows handles as usual);
  the mask is shown as a wire outline too.  Derived stroke children of vector-a are hidden in
  the viewport while a clip is active.  Everything is restored by ``seam.clip_release``;
* the clip result uses simple flat materials (``diffuse_color`` + Principled base colour).
  Gradient fills are approximated by the average stop colour in the viewport only - exports
  (PNG clip rings, SVG ``clipPath``) are exact.
* RASTER objects are clipped in exports only (no viewport result); documented gap.
"""

from __future__ import annotations

import struct
import warnings
from typing import List, Optional

import bpy

from .. import ids
from ..bridge import paths_signature
from ..compat import set_material_blend_alpha
from ..engine import core as engine
from ..engine.core import BuildResult, Producer
from ..seam_core import boolean2d, geometry
from ..seam_core.geometry import Affine
from . import canvas
from .export_png import (EXPORT_RESOLUTION, artwork_in_order, canvas_paths, fill_rings, gradient_stops,
                         live_canvas_affine, resolve_rings, rgba, stroke_rings)

CLIPPABLE_ROLES = ("SHAPE", "TEXT", "RASTER")
VIEWPORT_ROLES = ("SHAPE", "TEXT")
SIG_PROP = "seam_clip_sig"
_Z_STEP_LOCAL = 0.006   # child z above the clipped object (fill), stroke is slightly higher
RESOLUTION = EXPORT_RESOLUTION


# ------------------------------------------------------------------ helpers

def clip_key(obj) -> str:
    return f"clip:{ids.ensure_id(obj)}"


def clipped_by(scene, mask) -> List[bpy.types.Object]:
    return [o for o in scene.objects if o.seam.clip_source is mask and o is not mask]


def clip_child(obj):
    for ch in obj.children:
        if ch.seam.role == "CLIP_RESULT":
            return ch
    return None


def _affine_inverse(m: Affine) -> Optional[Affine]:
    a, b, c, d, e, f = m
    det = a * d - b * c
    if abs(det) < 1e-12:
        return None
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    return (ia, ib, ic, id_, -(ia * e + ic * f), -(ib * e + id_ * f))


def _style_bytes(style) -> bytes:
    vals = [1.0 if style.fill_enabled else 0.0, 1.0 if style.stroke_enabled else 0.0, style.stroke_width,
            style.miter_limit, style.opacity, *style.fill_color, *style.stroke_color]
    for _p, c in gradient_stops(style):
        vals.extend(c)
    tag = f"{style.fill_type}|{style.fill_rule}|{style.stroke_cap}|{style.stroke_join}|{len(style.gradient_stops)}"
    return tag.encode() + struct.pack(f"<{len(vals)}d", *vals)


def mask_candidate_ok(mask) -> bool:
    return mask is not None and mask.type == "CURVE" and mask.seam.role == "SHAPE"


# ------------------------------------------------------------------ set / release (used by operators and SVG import)

def _remember(obj, prop: str, value) -> None:
    if prop not in obj.keys():
        obj[prop] = value


def _mark_mask(mask) -> None:
    _remember(mask, "seam_disp0", mask.display_type)
    mask.display_type = "WIRE"


def _unmark_mask(scene, mask) -> None:
    if clipped_by(scene, mask):
        return
    disp = mask.get("seam_disp0")
    if disp:
        mask.display_type = disp
        del mask["seam_disp0"]


def apply_clip(scene, mask, objs) -> List[bpy.types.Object]:
    """Clip ``objs`` with ``mask``. Returns the objects that were clipped."""
    if not mask_candidate_ok(mask):
        raise ValueError("A clip mask must be a vector shape")
    done = []
    for o in objs:
        if o is mask or o.seam.role not in CLIPPABLE_ROLES:
            continue
        if mask.seam.clip_source is o:
            raise ValueError(f"'{o.name}' is clipped by '{mask.name}' - a mask cannot clip its own mask")
        old = o.seam.clip_source
        o.seam.clip_source = mask
        if old is not None and old is not mask:
            _unmark_mask(scene, old)
        done.append(o)
        engine.mark_dirty(clip_key(o))
    if done:
        _mark_mask(mask)
    return done


def release_clip(scene, objs) -> List[bpy.types.Object]:
    """Remove the clip from ``objs`` (and its derived result). Returns the released objects."""
    released = []
    masks = []
    for o in objs:
        if o.seam.clip_source is None and clip_child(o) is None:
            continue
        if o.seam.clip_source is not None:
            masks.append(o.seam.clip_source)
        o.seam.clip_source = None
        _cleanup(o)
        released.append(o)
    for m in set(masks):
        _unmark_mask(scene, m)
    return released


def _cleanup(obj) -> None:
    """Remove the derived result and restore display state of a no-longer-clipped object."""
    child = clip_child(obj)
    if child is not None:
        mesh = child.data
        mats = list(mesh.materials) if mesh else []
        bpy.data.objects.remove(child, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        for m in mats:
            if m is not None and m.users == 0 and m.name.startswith("SEAM Clip"):
                bpy.data.materials.remove(m)
    disp = obj.get("seam_disp0")
    if disp:
        obj.display_type = disp
        del obj["seam_disp0"]
    for ch in obj.children:
        if ch.get("seam_clip_hid"):
            ch.hide_viewport = False
            del ch["seam_clip_hid"]


# ------------------------------------------------------------------ operators

class SEAM_OT_clip_make(bpy.types.Operator):
    """Use the top-most selected shape as a clipping mask for the other selected objects"""
    bl_idname = "seam.clip_make"
    bl_label = "Make Clipping Mask"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        n = sum(1 for o in context.selected_objects if o.seam.role in CLIPPABLE_ROLES)
        if n < 2:
            cls.poll_message_set("Select the object(s) to clip plus a mask shape above them")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        sel = {o.name for o in context.selected_objects if o.seam.role in CLIPPABLE_ROLES}
        ordered = [o for o in artwork_in_order(scene, roles=CLIPPABLE_ROLES, include_hidden=True, include_masks=True)
                   if o.name in sel]
        if len(ordered) < 2:
            self.report({"ERROR"}, "Select at least two objects: the mask (top-most) and what it clips")
            return {"CANCELLED"}
        mask, others = ordered[-1], ordered[:-1]
        if not mask_candidate_ok(mask):
            self.report({"ERROR"}, f"The top-most selected object '{mask.name}' is not a vector shape; "
                        "convert text to outlines or move a shape above the objects to clip")
            return {"CANCELLED"}
        try:
            done = apply_clip(scene, mask, others)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if not done:
            self.report({"ERROR"}, "Nothing to clip")
            return {"CANCELLED"}
        self.report({"INFO"}, f"'{mask.name}' now clips {len(done)} object(s)")
        return {"FINISHED"}


class SEAM_OT_clip_release(bpy.types.Operator):
    """Release clipping: selected clipped objects, or every object clipped by a selected mask"""
    bl_idname = "seam.clip_release"
    bl_label = "Release Clipping Mask"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        for o in context.selected_objects:
            if o.seam.clip_source is not None or clipped_by(scene, o):
                return True
        cls.poll_message_set("Select a clipped object or a clipping mask")
        return False

    def execute(self, context):
        scene = context.scene
        targets: List[bpy.types.Object] = []
        for o in context.selected_objects:
            if o.seam.clip_source is not None:
                targets.append(o)
            targets.extend(clipped_by(scene, o))
        released = release_clip(scene, list(dict.fromkeys(targets)))
        if not released:
            self.report({"ERROR"}, "Nothing to release")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Released {len(released)} object(s)")
        return {"FINISHED"}


# ------------------------------------------------------------------ producer

def _avg_color(style):
    stops = gradient_stops(style)
    if style.fill_type != "SOLID" and len(stops) >= 2:
        n = len(stops)
        return tuple(sum(c[i] for _p, c in stops) / n for i in range(4))
    return rgba(style.fill_color)


def _material(mesh, slot: int, name: str, color, opacity: float):
    mat = None
    if slot < len(mesh.materials):
        mat = mesh.materials[slot]
    if mat is None:
        mat = bpy.data.materials.new(name)
        if slot < len(mesh.materials):
            mesh.materials[slot] = mat
        else:
            mesh.materials.append(mat)
    mat.diffuse_color = (color[0], color[1], color[2], color[3] * opacity)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)       # use_nodes is slated for removal in 6.0
        if not mat.use_nodes:
            mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if bsdf is not None:
        for sock, val in (("Base Color", (color[0], color[1], color[2], 1.0)), ("Alpha", color[3] * opacity)):
            if sock in bsdf.inputs:
                bsdf.inputs[sock].default_value = val
    if color[3] * opacity < 1.0:
        set_material_blend_alpha(mat)
    return mat


def _triangulate_shapes(shapes, inv: Affine, z: float):
    """Triangulate normalized shapes (canvas coords) into (verts3, faces) in object-local space."""
    a, b, c, d, e, f = inv
    verts, faces = [], []
    for outer, holes in shapes:
        pts, tris = geometry.triangulate(outer, holes)
        base = len(verts)
        verts.extend((a * x + c * y + e, b * x + d * y + f, z) for x, y in pts)
        faces.extend((base + i, base + j, base + k) for i, j, k in tris)
    return verts, faces


def _clip_shapes(rings, mask_rings):
    if not rings or not mask_rings:
        return []
    return boolean2d.boolean(rings, mask_rings, "INTERSECT", "EVENODD")


class ClipProducer(Producer):
    kind = "clip"
    priority = 40

    # -- keys
    def _keys_for_object(self, scene, ob):
        keys = []
        role = ob.seam.role
        if role in CLIPPABLE_ROLES:
            if ob.seam.clip_source is not None or clip_child(ob) is not None:
                keys.append(clip_key(ob))
            for o in clipped_by(scene, ob):
                keys.append(clip_key(o))
        elif role == "STROKE":
            owner = ids.find_object(ob.seam.owner_id, scene)
            if owner is not None and owner.seam.clip_source is not None:
                keys.append(clip_key(owner))
        return keys

    def keys_for_update(self, scene, id_orig, geometry_changed, transform):
        if isinstance(id_orig, bpy.types.Object):
            return self._keys_for_object(scene, id_orig)
        if isinstance(id_orig, bpy.types.Curve):
            keys = []
            for ob in scene.objects:
                if ob.data is id_orig:
                    keys.extend(self._keys_for_object(scene, ob))
            return keys
        return ()

    def all_keys(self, scene):
        for ob in scene.objects:
            if ob.seam.role in VIEWPORT_ROLES and (ob.seam.clip_source is not None or clip_child(ob) is not None):
                yield clip_key(ob)

    # -- signatures
    def _object(self, scene, key):
        return ids.find_object(key.split(":", 1)[1], scene)

    def signature(self, scene, key):
        obj = self._object(scene, key)
        if obj is None or obj.seam.role not in VIEWPORT_ROLES:
            return None
        mask = obj.seam.clip_source
        child = clip_child(obj)
        if mask is None:
            return "released" if child is not None else None
        if not mask_candidate_ok(mask):
            return "invalid-mask"
        extra = (paths_signature(canvas_paths(mask), extra=_style_bytes(mask.seam.style)) +
                 str(RESOLUTION)).encode()
        return paths_signature(canvas_paths(obj), extra=_style_bytes(obj.seam.style) + extra)

    def stored_signature(self, scene, key):
        obj = self._object(scene, key)
        if obj is None:
            return None
        child = clip_child(obj)
        if child is None:
            return None
        return child.get(SIG_PROP)

    # -- build
    def build(self, scene, key, signature):
        obj = self._object(scene, key)
        if obj is None:
            return BuildResult(key, "obsolete")
        mask = obj.seam.clip_source
        if mask is None or not mask_candidate_ok(mask):
            _cleanup(obj)
            return BuildResult(key, "ok", "released")
        style = obj.seam.style
        mstyle = mask.seam.style
        inv = _affine_inverse(live_canvas_affine(obj))
        if inv is None:
            return BuildResult(key, "error", f"'{obj.name}' has a degenerate scale and cannot be clipped")
        paths = canvas_paths(obj)
        mask_rings = resolve_rings(fill_rings(canvas_paths(mask), RESOLUTION), mstyle.fill_rule)
        verts, faces, mats = [], [], []
        if style.fill_enabled:
            rule = "NONZERO" if obj.type == "FONT" else style.fill_rule
            rings = resolve_rings(fill_rings(paths, RESOLUTION), rule)
            v, f = _triangulate_shapes(_clip_shapes(rings, mask_rings), inv, _Z_STEP_LOCAL)
            verts += v
            faces += [(a, b, c) for a, b, c in f]
            mats += [0] * len(f)
        if style.stroke_enabled and style.stroke_width > 0.0:
            srings = stroke_rings(paths, style, RESOLUTION)
            srings = resolve_rings(srings, "NONZERO")
            v, f = _triangulate_shapes(_clip_shapes(srings, mask_rings), inv, 2 * _Z_STEP_LOCAL)
            base = len(verts)
            verts += v
            faces += [(a + base, b + base, c + base) for a, b, c in f]
            mats += [1] * len(f)
        child = clip_child(obj)
        if child is None:
            mesh = bpy.data.meshes.new(f"{obj.name} Clip")
            child = bpy.data.objects.new(f"{obj.name} Clip", mesh)
            for col in obj.users_collection:
                col.objects.link(child)
            child.parent = obj
            child.matrix_parent_inverse.identity()
            child.hide_select = True
            ids.ensure_id(child)
            child.seam.role = "CLIP_RESULT"
        child.seam.owner_id = ids.ensure_id(obj)
        child.location = (0.0, 0.0, 0.0)
        mesh = child.data
        mesh.clear_geometry()
        if verts:
            mesh.from_pydata(verts, [], faces)
            mesh.update()
            mesh.polygons.foreach_set("material_index", mats)
        fill_c = _avg_color(style)
        _material(mesh, 0, f"SEAM Clip Fill {obj.seam.id}", fill_c, style.opacity)
        _material(mesh, 1, f"SEAM Clip Stroke {obj.seam.id}", rgba(style.stroke_color), style.opacity)
        # original: show only its path (WIRE); hide derived stroke children
        _remember(obj, "seam_disp0", obj.display_type)
        obj.display_type = "WIRE"
        for ch in obj.children:
            if ch.seam.role == "STROKE" and not ch.hide_viewport:
                ch["seam_clip_hid"] = 1
                ch.hide_viewport = True
        _mark_mask(mask)
        child[SIG_PROP] = signature
        return BuildResult(key, "ok", f"{len(faces)} triangles")


_producer = ClipProducer()


def _orphan_hook(scene, depsgraph):
    """Scene/collection membership changes (e.g. a mask was deleted): re-check clipped objects whose mask is gone."""
    for upd in depsgraph.updates:
        if isinstance(upd.id, (bpy.types.Scene, bpy.types.Collection)):
            break
    else:
        return
    for ob in scene.objects:
        if ob.seam.role in VIEWPORT_ROLES and ob.seam.clip_source is None and clip_child(ob) is not None:
            engine.mark_dirty(clip_key(ob))


classes = (SEAM_OT_clip_make, SEAM_OT_clip_release)


def register():
    engine.register_producer(_producer)
    engine.add_update_hook(_orphan_hook)


def unregister():
    engine.remove_update_hook(_orphan_hook)
    engine.unregister_producer(ClipProducer.kind)
