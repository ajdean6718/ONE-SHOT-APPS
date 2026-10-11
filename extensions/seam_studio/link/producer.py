"""Engine producer for links: signatures, builds per recipe, conflict detection (owner: link).

Key = ``link:<link.id>``. Signature = recipe + params + source paths in the SHAPE2D frame. The target
object keeps its identity, transform, modifiers and material slots: only its mesh *contents* are
rewritten, and only when the mesh still matches what the last build wrote (architecture D10).

Link materials (the SEAM "Face"/"Side" materials tagged ``seam_link_id``) follow the source fill colour.
They are validated by *comparing the materials themselves* with the source fill (not by a stored
signature): memfile undo can keep a Material the engine recoloured after an undo while it restores the
link record, so a stored value could claim "in sync" for a stale colour. A colour-only difference never
rewrites the mesh: the build only recolours the materials (also while the link is in CONFLICT).

The signature of the mesh a build wrote is stored on the mesh itself (``MESH_SIG_PROP`` /
``MESH_DATA_SIG_PROP``) as well as on the link record. The mesh copy is authoritative: it is restored and
reused together with the geometry it describes, and it is shared by every record that drives the same
target (a "Linked Copy" scene duplicates the link records).
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
from . import decal, recipes

classes = ()

KIND = "link"
MISSING = "missing-source"
MESH_SIG_PROP = "seam_target_sig"            # mesh custom property: geometry signature the last build wrote
MESH_DATA_SIG_PROP = "seam_target_data_sig"  # mesh custom property: UV + face-material signature of that build
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
    if link.recipe == "DECAL":
        return decal.signature(scene, link, MISSING)
    paths = recipes.shape_frame_paths(src, scene)
    params = params_dict(link)
    extra = [link.recipe, L.params_signature(link.recipe, params)]
    if recipes.source_mirrored(src, scene):
        extra.append("mirrored")
    # (the fill colour is not part of the geometry signature: see materials_in_sync / update_materials)
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


def _signature_arrays(co: np.ndarray, totals: np.ndarray, corners: np.ndarray, decimals: int = 5) -> str:
    """Byte-identical, vectorized equivalent of ``seam_core.mesh.mesh_signature`` (same digest)."""
    import hashlib
    co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
    totals = np.asarray(totals, dtype=np.int64)
    corners = np.asarray(corners, dtype=np.int64)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.array([len(co), len(totals)], dtype="<u4").tobytes())
    q = 10 ** decimals
    h.update(np.round(co * q).astype("<i8").tobytes())
    if len(totals):
        starts = np.zeros(len(totals), dtype=np.int64)
        np.cumsum(totals[:-1], out=starts[1:])
        stream = np.insert(corners, starts, totals)
        h.update(stream.astype("<u4").tobytes())
    return h.hexdigest()


def reference_signature(co, totals, corners) -> str:
    """The contract implementation (slow, pure Python); used by tests to prove equivalence."""
    co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
    faces = np.split(np.asarray(corners), np.cumsum(totals)[:-1]) if len(totals) else []
    return mesh_signature(co.tolist(), [f.tolist() for f in faces])


def _data_signature_arrays(uvs, materials, decimals: int = 5) -> str:
    """Hash of what the geometry signature does not cover but downstream users edit:
    UV coordinates of "UVMap" and per-face material indices."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    mats = np.asarray(materials, dtype=np.int64).ravel()
    h.update(np.array([len(mats)], dtype="<u4").tobytes())
    h.update(mats.astype("<i4").tobytes())
    if uvs is None:
        h.update(b"no-uv")
    else:
        uv = np.asarray(uvs, dtype=np.float32).astype(np.float64).ravel()
        h.update(np.array([len(uv)], dtype="<u4").tobytes())
        h.update(np.round(uv * 10 ** decimals).astype("<i8").tobytes())
    return h.hexdigest()


def read_data_signature(me) -> str:
    nf, nl = len(me.polygons), len(me.loops)
    mats = np.empty(nf, dtype=np.int32)
    if nf:
        me.polygons.foreach_get("material_index", mats)
    uvl = me.uv_layers.get("UVMap")
    uvs = None
    if uvl is not None and nl:
        uvs = np.empty(nl * 2, dtype=np.float32)
        uvl.data.foreach_get("uv", uvs)
    return _data_signature_arrays(uvs, mats)


# ------------------------------------------------------------------ conflict detection (D10)

def _has_weights(obj) -> bool:
    if not len(obj.vertex_groups):
        return False
    for v in obj.data.vertices:
        if len(v.groups):
            return True
    return False


def stored_target_signatures(link):
    """(geometry, data) signatures of what the last build wrote into the target mesh.

    The copy on the mesh wins over the link record: it travels with the geometry through undo and is shared
    by every link record that drives this mesh (scene linked copies)."""
    me = link.target.data if link.target is not None else None
    geo = data = None
    if me is not None:
        try:
            geo, data = me.get(MESH_SIG_PROP), me.get(MESH_DATA_SIG_PROP)
        except ReferenceError:
            pass
    geo = geo if isinstance(geo, str) and geo else link.target_signature
    data = data if isinstance(data, str) and data else link.target_data_signature
    return geo, data


def store_target_signatures(link, me, geo: str, data: str) -> None:
    link.target_signature = geo
    link.target_data_signature = data
    me[MESH_SIG_PROP] = geo
    me[MESH_DATA_SIG_PROP] = data


def target_conflicts(link) -> List[str]:
    """Reasons why writing would destroy downstream edits (empty = safe to write)."""
    obj = link.target
    me = obj.data
    reasons = []
    geo_sig, data_sig = stored_target_signatures(link)
    if geo_sig and read_mesh_signature(me) != geo_sig:
        reasons.append("mesh was edited")
    elif data_sig and read_data_signature(me) != data_sig:
        reasons.append("UVs or face materials were edited")
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


def target_busy_mode(obj) -> Optional[str]:
    """Readable name of the interactive mode the target is in ("Edit Mode", "Sculpt Mode", ...),
    or None in Object Mode. Writing the mesh under an active mode session is never safe."""
    try:
        mode = obj.mode
        if mode == "OBJECT" and not getattr(obj.data, "is_editmode", False):
            return None
    except ReferenceError:
        return None
    if mode == "OBJECT":
        mode = "EDIT"
    if mode.startswith("EDIT"):
        return "Edit Mode"
    return "%s Mode" % mode.replace("_", " ").title()


def target_in_edit_mode(obj) -> bool:
    """True when the target is in any non-Object mode (kept name: used by ops and tests)."""
    return target_busy_mode(obj) is not None


# ------------------------------------------------------------------ writing

def _vertex_group_state(obj):
    return ([(g.name, g.lock_weight) for g in obj.vertex_groups], obj.vertex_groups.active_index)


def _restore_vertex_groups(obj, state) -> None:
    """``clear_geometry()`` drops the vertex-group *names* (stored on the mesh since Blender 3.0).
    Re-create them (empty, same order/lock/active) so modifiers that name a group keep their target."""
    groups, active = state
    if not groups:
        return
    for name, lock in groups:
        g = obj.vertex_groups.get(name)
        if g is None:
            g = obj.vertex_groups.new(name=name)
        if g.lock_weight != lock:
            g.lock_weight = lock
    if 0 <= active < len(obj.vertex_groups) and obj.vertex_groups.active_index != active:
        obj.vertex_groups.active_index = active


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


def _link_materials(link):
    """(material, wanted RGBA) for every SEAM-owned colour material of a non-decal link that follows the fill."""
    src, tgt = link.source, link.target
    if link.recipe == "DECAL" or src is None or tgt is None or tgt.data is None or not link.params.use_source_material:
        return []
    from . import ops
    rgba = tuple(src.seam.style.fill_color)
    return [(mat, ops.material_color(rgba, mat.get("seam_material_role", "FACE")))
            for mat in tgt.data.materials if mat is not None and mat.get("seam_link_id") == link.id]


def materials_in_sync(link) -> bool:
    """Do the link's Face/Side materials show the source fill colour right now (viewport + Principled)?"""
    from . import ops
    try:
        return all(ops.material_color_matches(mat, want) for mat, want in _link_materials(link))
    except ReferenceError:
        return True


def update_materials(link) -> None:
    """Keep SEAM-created Face/Side materials of this link in sync with the source fill colour (idempotent)."""
    from . import ops
    for mat, want in _link_materials(link):
        ops.set_material_color(mat, want)


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
                if link.source == id_orig or (link.recipe == "DECAL" and decal.depends_on(link, id_orig)):
                    out.append(key_for(link))
                elif link.target == id_orig and link.stale and link.state in ("LIVE", "ERROR"):
                    out.append(key_for(link))  # e.g. left edit mode while an update was waiting
        elif isinstance(id_orig, bpy.types.Curve):  # Curve and TextCurve data via their users
            for link in seam.links:
                src = link.source
                if src is not None and src.data == id_orig:
                    out.append(key_for(link))
                elif link.recipe == "DECAL" and decal.depends_on_data(link, id_orig):
                    out.append(key_for(link))
        elif isinstance(id_orig, bpy.types.Image):  # a raster decal follows its (shared) image
            for link in seam.links:
                if link.recipe == "DECAL" and decal.src_is_raster(link) and link.decal_image == id_orig:
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
        if link is None:
            return None
        if link.recipe == "DECAL":
            return decal.stored_signature(link)
        if not materials_in_sync(link):
            return None  # colour drift (fill edit, or a Material kept by undo): build() recolours, mesh untouched
        return link.source_signature

    def build(self, scene, key: str, signature: str) -> engine.BuildResult:
        t0 = time.perf_counter()
        link = link_from_key(scene, key)
        if link is None:
            return engine.BuildResult(key, "obsolete")
        # one-shot (set by Overwrite / Branch): consumed by the very next build attempt whatever its
        # outcome, so a failed or skipped build can never leave conflict detection disabled
        force = bool(link.force_overwrite)
        if force:
            link.force_overwrite = False
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
        if not force and link.recipe != "DECAL" and signature == link.source_signature:
            # the geometry is current: only the colour materials drifted (never touches the mesh, so it is
            # safe in CONFLICT and while the result is in Edit Mode too)
            update_materials(link)
            if link.state == "LIVE" and link.stale and link.target is not None \
                    and target_busy_mode(link.target) is None:
                _apply(link, "build_ok")
            return engine.BuildResult(key, "ok", "materials re-synced", (time.perf_counter() - t0) * 1000.0)
        if link.state == "CONFLICT" and not force:
            _apply(link, "source_changed")
            update_materials(link)  # colours never touch the protected mesh
            return engine.BuildResult(key, "conflict", link.message)
        target = link.target
        is_decal = link.recipe == "DECAL"
        # a MATERIAL decal never writes the target mesh: no mode / conflict checks, only image + material
        writes_mesh = not (is_decal and decal.mode_of(link) == "MATERIAL")
        busy = target_busy_mode(target) if writes_mesh else None
        if busy:
            link.stale = True
            link.message = "Waiting: the result is in %s" % busy
            return engine.BuildResult(key, "waiting", link.message)
        if not force and writes_mesh:
            reasons = target_conflicts(link)
            if reasons:
                _apply(link, "target_modified")
                update_materials(link)
                link.message = "Result edited downstream (%s): Overwrite, Bake or Branch" % ", ".join(reasons)
                return engine.BuildResult(key, "conflict", link.message)
        try:
            if is_decal:
                built, notes = decal.build(scene, link, signature, L.validate_params("DECAL", params_dict(link)))
            else:
                built = recipes.build(link.recipe, recipes.shape_frame_paths(link.source, scene), params_dict(link),
                                      recipes.source_mirrored(link.source, scene))
                notes = list(built.messages)
        except recipes.RecipeError as exc:
            _apply(link, "build_error")
            if not is_decal:
                update_materials(link)
            link.message = str(exc)
            link.source_signature = signature  # do not retry until the source or params change
            if is_decal:
                decal.mark_applied(link, signature)
            return engine.BuildResult(key, "error", link.message, (time.perf_counter() - t0) * 1000.0)
        if writes_mesh:
            me = target.data
            if force and me.shape_keys is not None:
                target.shape_key_clear()
            groups = _vertex_group_state(target)
            geo_sig = write_mesh(me, built)
            _restore_vertex_groups(target, groups)
            # from the arrays just written (== read_data_signature(me); asserted by the tests)
            store_target_signatures(link, me, geo_sig, _data_signature_arrays(
                built.uvs if (built.uvs is not None and len(built.corner_verts)) else None, built.materials))
            if not is_decal:
                update_materials(link)
        ms = (time.perf_counter() - t0) * 1000.0
        link.source_signature = signature
        if is_decal:
            decal.mark_applied(link, signature)
        link.build_count += 1
        link.last_build_ms = ms
        link.message = "; ".join(notes)
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
