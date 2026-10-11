"""Engine producer for links: signatures, builds per recipe, conflict detection (owner: link).

Key = ``link:<link.id>``. Signature = recipe + params + source paths in the SHAPE2D frame
(+ fill colour when the result uses the source colour). The target object keeps its
identity, transform, modifiers and material slots: only its mesh *contents* are rewritten,
and only when the mesh still matches what the last build wrote (architecture D10).
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional

import bpy
import numpy as np

from .. import bridge, compat
from ..engine import core as engine
from ..seam_core import links as L
from ..seam_core.mesh import mesh_signature
from . import recipes

classes = ()

KIND = "link"
MISSING = "missing-source"
# Mesh attributes SEAM itself writes (or Blender adds for them); anything else is user data.
BUILTIN_ATTRIBUTES = {"position", "material_index", "sharp_face", "sharp_edge", "UVMap"}


# ------------------------------------------------------------------ lookup helpers

def key_for(link) -> str:
    return "%s:%s" % (KIND, link.id)


def find_link(scene, link_id: str):
    for link in scene.seam.links:
        if link.id == link_id:
            return link
    return None


def link_from_key(scene, key: str):
    kind, _, link_id = key.partition(":")
    if kind != KIND or not link_id:
        return None
    return find_link(scene, link_id)


def in_scene(obj, scene) -> bool:
    if obj is None:
        return False
    try:
        return scene.objects.get(obj.name) == obj
    except ReferenceError:
        return False


def params_dict(link) -> dict:
    p = link.params
    return {k: getattr(p, k) for k in L.PARAMS}


def fill_color(obj):
    style = obj.seam.style
    return tuple(round(float(c), 5) for c in style.fill_color)


# ------------------------------------------------------------------ signatures

def source_signature(scene, link) -> Optional[str]:
    src = link.source
    if not in_scene(src, scene):
        return MISSING
    paths = recipes.shape_frame_paths(src, scene)
    params = params_dict(link)
    extra = [link.recipe, L.params_signature(link.recipe, params)]
    if recipes.source_mirrored(src, scene):
        extra.append("mirrored")
    if params.get("use_source_material", True):
        extra.append(repr(fill_color(src)))
    return bridge.paths_signature(paths, extra="|".join(extra).encode())


def read_mesh_signature(me) -> str:
    nv, nf, nl = len(me.vertices), len(me.polygons), len(me.loops)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    totals = np.empty(nf, dtype=np.int32)
    me.polygons.foreach_get("loop_total", totals)
    corners = np.empty(nl, dtype=np.int32)
    me.loops.foreach_get("vertex_index", corners)
    return _signature_arrays(co.reshape(nv, 3), totals, corners)


def _signature_arrays(co: np.ndarray, totals: np.ndarray, corners: np.ndarray) -> str:
    verts = co.astype(np.float64).tolist()
    faces = np.split(corners, np.cumsum(totals)[:-1]) if len(totals) else []
    return mesh_signature(verts, [f.tolist() for f in faces])


# ------------------------------------------------------------------ conflict detection (D10)

def _has_weights(obj) -> bool:
    if not len(obj.vertex_groups):
        return False
    for v in obj.data.vertices:
        if len(v.groups):
            return True
    return False


def target_conflicts(link) -> List[str]:
    """Reasons why writing would destroy downstream edits (empty = safe to write)."""
    obj = link.target
    me = obj.data
    reasons = []
    if link.target_signature and read_mesh_signature(me) != link.target_signature:
        reasons.append("mesh was edited")
    if me.shape_keys is not None and len(me.shape_keys.key_blocks):
        reasons.append("shape keys")
    if _has_weights(obj):
        reasons.append("vertex groups")
    extra_uv = [uv.name for uv in me.uv_layers if uv.name != "UVMap"]
    if extra_uv:
        reasons.append("UV maps %s" % ", ".join(extra_uv))
    uv_names = {uv.name for uv in me.uv_layers}
    extra_attr = [a.name for a in me.attributes
                  if not a.name.startswith(".") and a.name not in BUILTIN_ATTRIBUTES and a.name not in uv_names]
    if extra_attr:
        reasons.append("attributes %s" % ", ".join(sorted(extra_attr)))
    return reasons


def target_in_edit_mode(obj) -> bool:
    try:
        return obj.mode == "EDIT" or bool(getattr(obj.data, "is_editmode", False))
    except ReferenceError:
        return False


# ------------------------------------------------------------------ writing

def write_mesh(me, built: recipes.Built) -> str:
    """Replace the mesh contents (keeps material slots); returns the new target signature."""
    me.clear_geometry()
    nv, nf, nl = len(built.verts), len(built.loop_totals), len(built.corner_verts)
    me.vertices.add(nv)
    me.vertices.foreach_set("co", built.verts.ravel())
    me.loops.add(nl)
    me.loops.foreach_set("vertex_index", built.corner_verts)
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", built.loop_starts)
    if nf:
        me.polygons.foreach_set("material_index", built.materials)
    me.update(calc_edges=True)
    if built.uvs is not None and nl:
        uv = me.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv", built.uvs.ravel())
    compat.set_shade_smooth_by_angle(me)
    return _signature_arrays(built.verts, built.loop_totals, built.corner_verts)


def update_materials(link) -> None:
    """Keep SEAM-created Face/Side materials of this link in sync with the source fill colour."""
    src, tgt = link.source, link.target
    if src is None or tgt is None or not link.params.use_source_material:
        return
    from . import ops
    rgba = tuple(src.seam.style.fill_color)
    for mat in tgt.data.materials:
        if mat is not None and mat.get("seam_link_id") == link.id:
            ops.set_material_color(mat, ops.material_color(rgba, mat.get("seam_material_role", "FACE")))


def _apply(link, event: str) -> bool:
    new, stale, rebuild = L.transition(link.state, event, link.stale)
    if link.state != new:
        link.state = new
    if link.stale != stale:
        link.stale = stale
    return rebuild


# ------------------------------------------------------------------ producer

class LinkProducer(engine.Producer):
    kind = KIND
    priority = 80

    def keys_for_update(self, scene, id_orig, geometry: bool, transform: bool) -> Iterable[str]:
        seam = getattr(scene, "seam", None)
        if seam is None or not len(seam.links):
            return ()
        out = []
        if isinstance(id_orig, bpy.types.Object):
            for link in seam.links:
                if link.source == id_orig:
                    out.append(key_for(link))
                elif link.target == id_orig and link.stale and link.state in ("LIVE", "ERROR"):
                    out.append(key_for(link))  # e.g. left edit mode while an update was waiting
        elif isinstance(id_orig, bpy.types.Curve):  # Curve and TextCurve data via their users
            for link in seam.links:
                src = link.source
                if src is not None and src.data == id_orig:
                    out.append(key_for(link))
        return out

    def all_keys(self, scene) -> Iterable[str]:
        seam = getattr(scene, "seam", None)
        if seam is None:
            return ()
        return [key_for(link) for link in seam.links if link.id]

    def signature(self, scene, key: str) -> Optional[str]:
        link = link_from_key(scene, key)
        if link is None or link.state == "DETACHED":
            return None
        if link.target is None or link.target.type != "MESH":
            return None
        return source_signature(scene, link)

    def stored_signature(self, scene, key: str) -> Optional[str]:
        link = link_from_key(scene, key)
        return None if link is None else link.source_signature

    def build(self, scene, key: str, signature: str) -> engine.BuildResult:
        t0 = time.perf_counter()
        link = link_from_key(scene, key)
        if link is None:
            return engine.BuildResult(key, "obsolete")
        if signature == MISSING:
            _apply(link, "source_missing")
            link.message = "Source was deleted; relink or detach"
            link.source_signature = signature
            return engine.BuildResult(key, "skipped", link.message)
        if link.state == "MISSING_SOURCE":
            _apply(link, "source_restored")
        if link.state == "PAUSED":
            _apply(link, "source_changed")
            link.message = "Paused: the source changed; resume to update"
            return engine.BuildResult(key, "skipped", link.message)
        force = bool(link.force_overwrite)
        if link.state == "CONFLICT" and not force:
            _apply(link, "source_changed")
            return engine.BuildResult(key, "conflict", link.message)
        target = link.target
        if target_in_edit_mode(target):
            link.stale = True
            link.message = "Waiting: the result is in Edit Mode"
            return engine.BuildResult(key, "waiting", link.message)
        if not force:
            reasons = target_conflicts(link)
            if reasons:
                _apply(link, "target_modified")
                link.message = "Result edited downstream (%s): Overwrite, Bake or Branch" % ", ".join(reasons)
                return engine.BuildResult(key, "conflict", link.message)
        try:
            built = recipes.build(link.recipe, recipes.shape_frame_paths(link.source, scene), params_dict(link),
                                  recipes.source_mirrored(link.source, scene))
        except recipes.RecipeError as exc:
            _apply(link, "build_error")
            link.message = str(exc)
            link.source_signature = signature  # do not retry until the source or params change
            return engine.BuildResult(key, "error", link.message, (time.perf_counter() - t0) * 1000.0)
        me = target.data
        if force and me.shape_keys is not None:
            target.shape_key_clear()
        link.target_signature = write_mesh(me, built)
        update_materials(link)
        ms = (time.perf_counter() - t0) * 1000.0
        link.source_signature = signature
        link.build_count += 1
        link.last_build_ms = ms
        link.message = "; ".join(built.messages)
        link.force_overwrite = False
        _apply(link, "build_ok")
        return engine.BuildResult(key, "ok", link.message, ms)


_producer: Optional[LinkProducer] = None


def register():
    global _producer
    _producer = LinkProducer()
    engine.register_producer(_producer)


def unregister():
    global _producer
    engine.unregister_producer(KIND)
    _producer = None
