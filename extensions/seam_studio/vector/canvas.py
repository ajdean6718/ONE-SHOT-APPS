"""Canvas root, artboards, layers and artwork placement (foundation by integrator; owner after: vector-a).

Public API (stable contract used by link, vector-b, paint, shell):
    ensure_canvas(scene) -> root Object
    artwork_collection(scene) -> Collection        (creates if missing)
    layers(scene) -> list[Collection]               bottom -> top
    active_layer(scene) -> Collection
    new_layer(scene, name) -> Collection
    artboards(scene) -> list[Object]
    active_artboard(scene) -> Object | None
    new_artboard(scene, name, x, y, width, height) -> Object   (x, y = canvas-local top-left)
    add_artwork_object(scene, obj, layer=None, role="SHAPE", on_top=True) -> obj
    new_shape_object(scene, name, paths, location=(0, 0), layer=None) -> Object
    artwork_objects(scene, roles=ARTWORK_ROLES) -> list[Object]   bottom -> top
    restack(scene) -> None
    results_collection(scene) -> Collection

Added by vector-a (additive; the foundation signatures above are unchanged):
    set_active_layer(scene, layer) / layer_collection_of(collection) / set_layer_order(scene, order)
    set_active_artboard(scene, artboard) / sync_artboard(obj)
    canvas_bounds(obj, depsgraph=None, scene=None) -> (minx, miny, maxx, maxy) | None
        canvas-local bounds of the evaluated geometry (groups: union of children, strokes excluded)
    find_by_id(seam_id, scene=None) -> Object | None   (cached lookup by SEAM id)
    get_sig / set_sig   (engine signatures stored as ID custom properties)

Operators: seam.canvas_new, seam.artboard_add / artboard_remove / artboard_set_active, seam.view_artboard,
seam.layer_add / layer_remove / layer_move / layer_set_active.
Producer: "artboard" (priority 1) regenerates the artboard plane and its background material.
Adoption hook: new CURVE/FONT objects that land in a SEAM layer get role, id, canvas parent
(world matrix preserved) and a z slot; duplicated SEAM ids are re-issued.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from mathutils import Matrix, Vector

from .. import compat, ids
from ..bridge import canvas_local_matrix, write_paths
from ..engine import core as engine
from ..seam_core import CANVAS_ORIGIN, LAYER_STEP, Z_STEP

ROOT_NAME = "SEAM Canvas"
ARTWORK_NAME = "SEAM Artwork"
RESULTS_NAME = "SEAM Results"
ARTBOARDS_NAME = "SEAM Artboards"
GUIDES_NAME = "SEAM Guides"
ARTWORK_ROLES = ("SHAPE", "TEXT", "RASTER", "GROUP")
ARTBOARD_Z = -1.0
_NON_LAYER_PREFIXES = (ARTBOARDS_NAME, GUIDES_NAME)


# ------------------------------------------------------------------ small shared helpers

def get_sig(obj, name: str) -> Optional[str]:
    """Signature stored on an ID by a producer (ID custom property ``_seam_sig_<name>``)."""
    try:
        v = obj.get("_seam_sig_" + name)
    except ReferenceError:
        return None
    return v if isinstance(v, str) else None


def set_sig(obj, name: str, value: Optional[str]) -> None:
    key = "_seam_sig_" + name
    if value is None:
        if key in obj.keys():
            del obj[key]
    elif obj.get(key) != value:
        obj[key] = value


_id_cache: Dict[str, str] = {}


def find_by_id(seam_id: str, scene=None):
    """Object with the given SEAM id (cached name lookup, verified on every hit)."""
    if not seam_id:
        return None
    name = _id_cache.get(seam_id)
    if name is not None:
        ob = bpy.data.objects.get(name)
        if ob is not None and ob.seam.id == seam_id:
            return ob
    ob = ids.find_object(seam_id, scene or bpy.context.scene)
    if ob is not None:
        _id_cache[seam_id] = ob.name
    else:
        _id_cache.pop(seam_id, None)
    return ob


def _view_layer(scene):
    try:
        if bpy.context.scene is scene:
            return bpy.context.view_layer
    except Exception:
        pass
    return scene.view_layers[0]


def _get_or_new_collection(name: str, parent) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if col.name not in parent.children:
        parent.children.link(col)
    return col


def artwork_collection(scene=None) -> bpy.types.Collection:
    scene = scene or bpy.context.scene
    col = _get_or_new_collection(ARTWORK_NAME, scene.collection)
    if not col.hide_render:  # write only when needed (writes are forbidden in draw / handler contexts)
        col.hide_render = True
    return col


def results_collection(scene=None) -> bpy.types.Collection:
    scene = scene or bpy.context.scene
    return _get_or_new_collection(RESULTS_NAME, scene.collection)


def _artboard_collection(scene) -> bpy.types.Collection:
    return _get_or_new_collection(ARTBOARDS_NAME, artwork_collection(scene))


def guides_collection(scene=None) -> bpy.types.Collection:
    scene = scene or bpy.context.scene
    return _get_or_new_collection(GUIDES_NAME, artwork_collection(scene))


def _is_layer_collection(col) -> bool:
    return not any(col.name.startswith(p) for p in _NON_LAYER_PREFIXES)


# ------------------------------------------------------------------ layers

def layers(scene=None) -> List[bpy.types.Collection]:
    scene = scene or bpy.context.scene
    art = bpy.data.collections.get(ARTWORK_NAME)
    if art is None or art.name not in scene.collection.children:
        art = artwork_collection(scene)
    return [c for c in art.children if _is_layer_collection(c)]


def new_layer(scene=None, name: str = "Layer") -> bpy.types.Collection:
    scene = scene or bpy.context.scene
    col = bpy.data.collections.new(name)
    artwork_collection(scene).children.link(col)
    return col


def layer_collection_of(collection, view_layer=None):
    """LayerCollection of ``collection`` in ``view_layer`` (default: context view layer) or None."""
    view_layer = view_layer or bpy.context.view_layer

    def walk(lc):
        if lc.collection == collection:
            return lc
        for ch in lc.children:
            r = walk(ch)
            if r is not None:
                return r
        return None

    return walk(view_layer.layer_collection)


def active_layer(scene=None) -> bpy.types.Collection:
    """Layer of the active view-layer collection if it is a SEAM layer, else the top layer."""
    scene = scene or bpy.context.scene
    lys = layers(scene)
    try:
        alc = bpy.context.view_layer.active_layer_collection
        if alc is not None and alc.collection in lys:
            return alc.collection
    except Exception:
        pass
    if not lys:
        return new_layer(scene, "Layer 1")
    return lys[-1]


def set_active_layer(scene, layer) -> bool:
    vl = _view_layer(scene)
    lc = layer_collection_of(layer, vl)
    if lc is None:
        return False
    vl.active_layer_collection = lc
    return True


def set_layer_order(scene, order: Sequence[bpy.types.Collection]) -> None:
    """Reorder SEAM layers (bottom -> top) preserving visibility/exclude flags and the active layer."""
    scene = scene or bpy.context.scene
    art = artwork_collection(scene)
    vl = _view_layer(scene)
    current = layers(scene)
    order = list(order)
    if set(c.name for c in order) != set(c.name for c in current):
        raise ValueError("order must contain exactly the current layers")
    first = next((i for i, (a, b) in enumerate(zip(current, order)) if a != b), None)
    if first is None:
        return
    moving = order[first:]
    active = None
    try:
        active = vl.active_layer_collection.collection
    except Exception:
        pass
    state = {}
    for col in moving:
        lc = layer_collection_of(col, vl)
        if lc is not None:
            state[col.name] = (lc.exclude, lc.hide_viewport)
    for col in current[first:]:
        art.children.unlink(col)
    for col in moving:
        art.children.link(col)
    for col in moving:
        lc = layer_collection_of(col, vl)
        if lc is not None and col.name in state:
            lc.exclude, lc.hide_viewport = state[col.name]
    if active is not None:
        set_active_layer(scene, active)
    restack(scene)


# ------------------------------------------------------------------ canvas root

def ensure_canvas(scene=None):
    """Create (idempotently) the canvas root, artwork collection, a first layer and a first artboard."""
    scene = scene or bpy.context.scene
    root = None
    for ob in scene.objects:
        if ob.seam.role == "CANVAS_ROOT":
            root = ob
            break
    art = artwork_collection(scene)
    if root is None:
        root = bpy.data.objects.new(ROOT_NAME, None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = 1.0
        root.location = CANVAS_ORIGIN
        root.hide_select = True
        art.objects.link(root)
        ids.ensure_id(root)
        root.seam.role = "CANVAS_ROOT"
    if not layers(scene):
        new_layer(scene, "Layer 1")
    if not artboards(scene):
        new_artboard(scene, "Artboard 1", 0.0, 0.0, 1920.0, 1080.0)
    scene.seam.doc.canvas_ready = True
    return root


def canvas_root(scene=None):
    scene = scene or bpy.context.scene
    for ob in scene.objects:
        if ob.seam.role == "CANVAS_ROOT":
            return ob
    return None


def _parent_to_canvas(obj, scene, parent=None) -> None:
    parent = parent or canvas_root(scene) or ensure_canvas(scene)
    if obj.parent is not parent:
        obj.parent = parent
        obj.matrix_parent_inverse = Matrix.Identity(4)


# ------------------------------------------------------------------ artboards

def artboards(scene=None) -> List[bpy.types.Object]:
    scene = scene or bpy.context.scene
    return [ob for ob in scene.objects if ob.seam.role == "ARTBOARD"]


def active_artboard(scene=None):
    scene = scene or bpy.context.scene
    abs_ = artboards(scene)
    want = scene.seam.doc.active_artboard_id
    for ob in abs_:
        if ob.seam.id == want:
            return ob
    return abs_[0] if abs_ else None


def set_active_artboard(scene, artboard) -> None:
    scene.seam.doc.active_artboard_id = ids.ensure_id(artboard)


def artboard_mesh(width: float, height: float, name: str = "Artboard") -> bpy.types.Mesh:
    me = bpy.data.meshes.new(name)
    me.from_pydata([(0, 0, 0), (width, 0, 0), (width, -height, 0), (0, -height, 0)], [], [(0, 3, 2, 1)])
    # UVs so a background texture could be applied later
    uv = me.uv_layers.new(name="UVMap")
    for loop, co in zip(uv.data, [(0, 1), (0, 0), (1, 0), (1, 1)]):
        loop.uv = co
    return me


def new_artboard(scene=None, name: str = "Artboard", x: float = 0.0, y: float = 0.0,
                 width: float = 1920.0, height: float = 1080.0):
    scene = scene or bpy.context.scene
    ob = bpy.data.objects.new(name, artboard_mesh(width, height, name))
    _artboard_collection(scene).objects.link(ob)
    root = canvas_root(scene)
    if root is None:
        root = ensure_canvas(scene)
    _parent_to_canvas(ob, scene, root)
    ob.location = (x, y, ARTBOARD_Z)
    ids.ensure_id(ob)
    ob.seam.role = "ARTBOARD"
    ob.seam.artboard.width = width
    ob.seam.artboard.height = height
    ob.color = (1.0, 1.0, 1.0, 1.0)
    if not scene.seam.doc.active_artboard_id:
        scene.seam.doc.active_artboard_id = ob.seam.id
    sync_artboard(ob)
    set_sig(ob, "artboard", _artboard_signature(ob))
    return ob


def artboard_rect(ob) -> Tuple[float, float, float, float]:
    """Canvas-local (minx, miny, maxx, maxy) of an artboard object."""
    ab = ob.seam.artboard
    x, y = ob.location.x, ob.location.y
    return (x, y - ab.height, x + ab.width, y)


def flat_material_tree(mat) -> dict:
    """Ensure ``mat`` has the SEAM flat tree: Emission mixed with Transparent. Returns nodes by role."""
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    need = ("SEAM_out", "SEAM_mix", "SEAM_transparent", "SEAM_emission")
    if not all(n in nt.nodes for n in need):
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.name = "SEAM_out"
        out.location = (400, 0)
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.name = "SEAM_mix"
        mix.location = (200, 0)
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr.name = "SEAM_transparent"
        tr.location = (0, 100)
        em = nt.nodes.new("ShaderNodeEmission")
        em.name = "SEAM_emission"
        em.location = (0, -100)
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return {n[5:]: nt.nodes[n] for n in need}


def set_flat_color(mat, rgba, alpha_factor: float = 1.0, enabled: bool = True) -> None:
    """Flat emission colour (scene-linear) with alpha = rgba[3] * alpha_factor (0 when disabled)."""
    nodes = flat_material_tree(mat)
    nodes["emission"].inputs["Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
    nodes["emission"].inputs["Strength"].default_value = 1.0
    alpha = (rgba[3] * alpha_factor) if enabled else 0.0
    mix_fac = nodes["mix"].inputs[0]
    if not mix_fac.is_linked:
        mix_fac.default_value = alpha
    mat.diffuse_color = (rgba[0], rgba[1], rgba[2], alpha if enabled else 0.0)
    compat.set_material_blend_alpha(mat)
    mat.use_backface_culling = False


def sync_artboard(ob) -> bool:
    """Make the plane mesh and background material of an artboard match ``ob.seam.artboard``."""
    if ob is None or ob.type != "MESH":
        return False
    ab = ob.seam.artboard
    w, h = float(ab.width), float(ab.height)
    me = ob.data
    co = [0.0, 0.0, 0.0, w, 0.0, 0.0, w, -h, 0.0, 0.0, -h, 0.0]
    if len(me.vertices) == 4:
        me.vertices.foreach_set("co", co)
        me.update()
    else:
        me.clear_geometry()
        me.from_pydata([(0, 0, 0), (w, 0, 0), (w, -h, 0), (0, -h, 0)], [], [(0, 3, 2, 1)])
        uv = me.uv_layers.new(name="UVMap")
        for loop, c in zip(uv.data, [(0, 1), (0, 0), (1, 0), (1, 1)]):
            loop.uv = c
    mat = None
    if me.materials:
        mat = me.materials[0]
        if mat is None or mat.get("seam_owner") != ob.seam.id:
            mat = None
    if mat is None:
        mat = bpy.data.materials.new("SEAM Artboard " + ob.name)
        mat["seam_owner"] = ob.seam.id
        if me.materials:
            me.materials[0] = mat
        else:
            me.materials.append(mat)
    if mat.name != "SEAM Artboard " + ob.name:
        mat.name = "SEAM Artboard " + ob.name
    set_flat_color(mat, tuple(ab.background_color), 1.0, bool(ab.background_enabled))
    ob.display_type = "TEXTURED" if ab.background_enabled else "WIRE"
    ob.color = tuple(ab.background_color)
    return True


def _artboard_signature(ob) -> str:
    ab = ob.seam.artboard
    me = ob.data if ob.type == "MESH" else None
    mat_ok = bool(me and me.materials and me.materials[0] is not None and me.materials[0].get("seam_owner") == ob.seam.id)
    parts = [round(ab.width, 4), round(ab.height, 4), bool(ab.background_enabled),
             tuple(round(c, 5) for c in ab.background_color), ob.name, mat_ok,
             len(me.vertices) if me else -1]
    return repr(parts)


class ArtboardProducer(engine.Producer):
    kind = "artboard"
    priority = 1

    def keys_for_update(self, scene, id_orig, geometry, transform):
        if isinstance(id_orig, bpy.types.Object) and id_orig.seam.role == "ARTBOARD" and id_orig.seam.id:
            yield "artboard:" + id_orig.seam.id

    def all_keys(self, scene):
        for ob in artboards(scene):
            if ob.seam.id:
                yield "artboard:" + ob.seam.id

    def _obj(self, scene, key):
        ob = find_by_id(key.split(":", 1)[1], scene)
        return ob if ob is not None and ob.seam.role == "ARTBOARD" else None

    def signature(self, scene, key):
        ob = self._obj(scene, key)
        return None if ob is None else _artboard_signature(ob)

    def stored_signature(self, scene, key):
        ob = self._obj(scene, key)
        return None if ob is None else get_sig(ob, "artboard")

    def build(self, scene, key, signature):
        ob = self._obj(scene, key)
        if ob is None:
            return engine.BuildResult(key, "obsolete")
        sync_artboard(ob)
        set_sig(ob, "artboard", _artboard_signature(ob))
        return engine.BuildResult(key, "ok")


# ------------------------------------------------------------------ artwork

def _layer_index(scene, obj) -> int:
    lys = layers(scene)
    for i, col in enumerate(lys):
        if obj.name in col.objects:
            return i
    return 0


def artwork_objects(scene=None, roles: Sequence[str] = ARTWORK_ROLES) -> List[bpy.types.Object]:
    scene = scene or bpy.context.scene
    obs = [ob for ob in scene.objects if ob.seam.role in roles]
    lys = layers(scene)
    order = {}
    for i, col in enumerate(lys):
        for ob in col.all_objects:
            order.setdefault(ob.name, i)
    return sorted(obs, key=lambda o: (order.get(o.name, 0), o.matrix_world.translation.z, o.name))


def _top_z_in_layer(scene, layer, exclude=None) -> Optional[float]:
    zs = [ob.location.z for ob in layer.objects if ob is not exclude and ob.seam.role in ARTWORK_ROLES
          and ob.parent is not None and ob.parent.seam.role == "CANVAS_ROOT"]
    return max(zs) if zs else None


def add_artwork_object(scene, obj, layer=None, role: str = "SHAPE", on_top: bool = True):
    """Link ``obj`` into a SEAM layer, parent it to the canvas root and give it a z slot."""
    scene = scene or bpy.context.scene
    ensure_canvas(scene)
    layer = layer or active_layer(scene)
    for col in list(obj.users_collection):
        if col is not layer:
            col.objects.unlink(obj)
    if obj.name not in layer.objects:
        layer.objects.link(obj)
    _parent_to_canvas(obj, scene)
    ids.ensure_id(obj)
    obj.seam.role = role
    li = layers(scene).index(layer) if layer in layers(scene) else 0
    base = li * LAYER_STEP
    if on_top:
        top = _top_z_in_layer(scene, layer, exclude=obj)
        obj.location.z = base if top is None or top < base else top + Z_STEP
    return obj


def new_shape_object(scene, name: str, paths, location=(0.0, 0.0), layer=None):
    """Create a filled 2D curve object from seam_core BezierPaths (object-local coords) on the canvas."""
    scene = scene or bpy.context.scene
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.resolution_u = 12
    write_paths(cu, paths)
    ob = bpy.data.objects.new(name, cu)
    ob.location = (location[0], location[1], 0.0)
    add_artwork_object(scene, ob, layer=layer, role="SHAPE")
    return ob


def restack(scene=None) -> None:
    """Re-assign z = layer_index * LAYER_STEP + order * Z_STEP for canvas-parented artwork, keeping order."""
    scene = scene or bpy.context.scene
    for li, col in enumerate(layers(scene)):
        obs = [ob for ob in col.objects if ob.seam.role in ARTWORK_ROLES and ob.parent is not None
               and ob.parent.seam.role == "CANVAS_ROOT"]
        obs.sort(key=lambda o: (o.location.z, o.name))
        for i, ob in enumerate(obs):
            ob.location.z = li * LAYER_STEP + i * Z_STEP


def canvas_point_to_world(scene, x: float, y: float) -> Vector:
    """World position of canvas-local (x, y). Uses the root's basis matrix so it is valid right after creation."""
    from ..bridge import canvas_matrix
    root = canvas_root(scene)
    m = root.matrix_basis if root is not None and root.parent is None else canvas_matrix(scene)
    return m @ Vector((x, y, 0.0))


def canvas_bounds(obj, depsgraph=None, scene=None):
    """Canvas-local (minx, miny, maxx, maxy) of the evaluated geometry of ``obj``.

    Empties (groups) return the union of their artwork children; derived stroke children are
    ignored (geometric bounds, like Illustrator without "preview bounds"). None when empty.
    """
    scene = scene or bpy.context.scene
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    box = None

    def merge(b):
        nonlocal box
        if b is None:
            return
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))

    if obj.type == "EMPTY":
        for ch in obj.children:
            if ch.seam.role in ARTWORK_ROLES:
                merge(canvas_bounds(ch, depsgraph, scene))
        return box
    if obj.type not in ("CURVE", "FONT", "MESH", "SURFACE"):
        return None
    ev = obj.evaluated_get(depsgraph)
    me = None
    try:
        me = ev.to_mesh()
        n = len(me.vertices)
        if n == 0:
            return None
        co = np.empty(n * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)
    finally:
        if me is not None:
            ev.to_mesh_clear()
    m = np.array(canvas_local_matrix(obj, scene))
    xy = co @ m[:2, :3].T + m[:2, 3]
    return (float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max()))


# ------------------------------------------------------------------ adoption + id healing

def _owned_stroke_parent(ob, sid) -> bool:
    for ch in ob.children:
        if ch.seam.role == "STROKE" and ch.seam.owner_id == sid:
            return True
    return False


def _heal_duplicate_ids(scene, touched) -> None:
    """Re-issue SEAM ids of duplicated (Shift+D / copy-paste) objects. The original keeps its id."""
    sids = {ob.seam.id for ob in touched if ob.seam.id}
    if not sids:
        return
    groups: Dict[str, list] = {}
    for ob in scene.objects:
        if ob.seam.id in sids:
            groups.setdefault(ob.seam.id, []).append(ob)
    for sid, obs in groups.items():
        if len(obs) < 2:
            continue
        # prefer the object that already owns a stroke child, then the one with the shortest name
        obs.sort(key=lambda o: (not _owned_stroke_parent(o, sid), len(o.name), o.name))
        for dup in obs[1:]:
            old = dup.seam.id
            dup.seam.id = ""
            ids.ensure_id(dup)
            for key in list(dup.keys()):
                if key.startswith("_seam_sig_"):
                    del dup[key]
            if dup.seam.role == "STROKE" and dup.seam.owner_id == old:
                pass  # duplicated stroke children are removed by the stroke producer's healing
            _id_cache.pop(old, None)


def _adopt(scene, ob) -> None:
    root = canvas_root(scene)
    if root is None:
        return
    layer = next((c for c in ob.users_collection if c in layers(scene)), None)
    if layer is None:
        return
    if ob.parent is None or ob.parent.seam.role not in ("CANVAS_ROOT", "GROUP"):
        if ob.parent is not None:
            mw = ob.matrix_world.copy()
        else:
            mw = ob.matrix_basis.copy()
        ob.parent = root
        ob.matrix_parent_inverse = Matrix.Identity(4)
        ob.matrix_basis = root.matrix_world.inverted_safe() @ mw
    ids.ensure_id(ob)
    ob.seam.role = "SHAPE" if ob.type == "CURVE" else "TEXT"
    if ob.type == "CURVE":
        ob.data.dimensions = "2D"
        if ob.data.fill_mode == "NONE":
            ob.data.fill_mode = "BOTH"
    if ob.parent is root:
        top = _top_z_in_layer(scene, layer, exclude=ob)
        li = layers(scene).index(layer)
        base = li * LAYER_STEP
        ob.location.z = base if top is None or top < base else top + Z_STEP
    ob.rotation_euler.x = 0.0
    ob.rotation_euler.y = 0.0
    ob.scale.z = 1.0


def adoption_hook(scene, depsgraph) -> None:
    """Engine update hook: adopt new CURVE/FONT objects in SEAM layers; re-issue duplicated ids."""
    if not scene.seam.doc.canvas_ready:
        return
    touched = []
    for upd in depsgraph.updates:
        idd = getattr(upd.id, "original", upd.id)
        if isinstance(idd, bpy.types.Object):
            touched.append(idd)
    if not touched:
        return
    lys = None
    for ob in touched:
        try:
            if ob.seam.role != "NONE" or ob.type not in ("CURVE", "FONT"):
                continue
            if lys is None:
                lys = set(c.name for c in layers(scene))
            if any(c.name in lys for c in ob.users_collection):
                _adopt(scene, ob)
        except ReferenceError:
            continue
    try:
        _heal_duplicate_ids(scene, [o for o in touched if o.seam.role != "NONE"])
    except ReferenceError:
        pass


# ------------------------------------------------------------------ operators

def _next_name(base: str, existing: Sequence[str]) -> str:
    n = 1
    names = set(existing)
    while f"{base} {n}" in names:
        n += 1
    return f"{base} {n}"


class SEAM_OT_canvas_new(bpy.types.Operator):
    bl_idname = "seam.canvas_new"
    bl_label = "New Canvas"
    bl_description = "Create the 2D canvas (layer + artboard) if the document has none, and frame it"
    bl_options = {"REGISTER", "UNDO"}

    width: FloatProperty(name="Width", min=1.0, default=1920.0, description="Width of the first artboard in px")
    height: FloatProperty(name="Height", min=1.0, default=1080.0, description="Height of the first artboard in px")

    def execute(self, context):
        scene = context.scene
        had_artboard = bool(canvas_root(scene) and artboards(scene))
        root = ensure_canvas(scene)
        if not had_artboard:
            ab = active_artboard(scene)
            if ab is not None:
                ab.seam.artboard.width = self.width
                ab.seam.artboard.height = self.height
                sync_artboard(ab)
        try:
            scene.view_settings.view_transform = "Standard"
        except (TypeError, AttributeError):
            pass
        set_active_layer(scene, active_layer(scene))  # native tools (pen, curve add) now land in a SEAM layer
        if not had_artboard:  # new objects are added at the 3D cursor: put it on the artboard, not 20000 px away
            ab = active_artboard(scene)
            if ab is not None:
                minx, miny, maxx, maxy = artboard_rect(ab)
                scene.cursor.location = canvas_point_to_world(scene, (minx + maxx) * 0.5, (miny + maxy) * 0.5)
        engine.flush(scene, evaluate=True)
        if not bpy.app.background and context.area is not None and context.area.type == "VIEW_3D":
            try:
                bpy.ops.seam.view_artboard()
            except RuntimeError:
                pass
        self.report({"INFO"}, f"Canvas ready: {len(layers(scene))} layer(s), {len(artboards(scene))} artboard(s)")
        return {"FINISHED"} if root else {"CANCELLED"}


class SEAM_OT_artboard_add(bpy.types.Operator):
    bl_idname = "seam.artboard_add"
    bl_label = "Add Artboard"
    bl_description = "Add an artboard to the right of the existing ones and make it active"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty(name="Name", default="", description="Artboard name (empty: numbered automatically)")
    width: FloatProperty(name="Width", min=1.0, default=1920.0)
    height: FloatProperty(name="Height", min=1.0, default=1080.0)
    x: FloatProperty(name="X", description="Canvas-local top-left X (used when 'Auto Position' is off)")
    y: FloatProperty(name="Y", description="Canvas-local top-left Y (used when 'Auto Position' is off)")
    auto_position: BoolProperty(name="Auto Position", default=True,
                                description="Place to the right of the last artboard with a 100 px gap")

    def execute(self, context):
        scene = context.scene
        ensure_canvas(scene)
        abs_ = artboards(scene)
        x, y = self.x, self.y
        if self.auto_position and abs_:
            rects = [artboard_rect(a) for a in abs_]
            x = max(r[2] for r in rects) + 100.0
            y = max(r[3] for r in rects)
        elif self.auto_position:
            x, y = 0.0, 0.0
        name = self.name or _next_name("Artboard", [a.name for a in abs_])
        ab = new_artboard(scene, name, x, y, self.width, self.height)
        set_active_artboard(scene, ab)
        engine.flush(scene)
        return {"FINISHED"}


class SEAM_OT_artboard_remove(bpy.types.Operator):
    bl_idname = "seam.artboard_remove"
    bl_label = "Remove Artboard"
    bl_description = "Delete the active artboard (artwork on it is kept)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if len(artboards(context.scene)) < 2:
            cls.poll_message_set("A document needs at least one artboard")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        ab = active_artboard(scene)
        if ab is None:
            return {"CANCELLED"}
        me = ab.data
        bpy.data.objects.remove(ab, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
        left = artboards(scene)
        scene.seam.doc.active_artboard_id = left[0].seam.id if left else ""
        return {"FINISHED"}


class SEAM_OT_artboard_set_active(bpy.types.Operator):
    bl_idname = "seam.artboard_set_active"
    bl_label = "Set Active Artboard"
    bl_description = "Make an artboard the active one (the selected artboard when no name is given)"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty(name="Artboard", default="", description="Artboard object name (empty: the active object)")

    def execute(self, context):
        scene = context.scene
        ab = None
        if self.name:
            ab = bpy.data.objects.get(self.name)
        elif context.active_object is not None:
            ab = context.active_object
        if ab is None or ab.seam.role != "ARTBOARD":
            self.report({"ERROR"}, "Not an artboard")
            return {"CANCELLED"}
        set_active_artboard(scene, ab)
        return {"FINISHED"}


def _find_view3d(context):
    area = context.area if context.area is not None and context.area.type == "VIEW_3D" else None
    if area is None and context.screen is not None:
        for a in context.screen.areas:
            if a.type == "VIEW_3D":
                area = a
                break
    if area is None:
        return None, None, None
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    space = area.spaces.active
    if region is None or space is None or space.type != "VIEW_3D":
        return None, None, None
    return area, region, space


def frame_rect_in_view(scene, area, region, space, rect, margin: float = 0.9) -> bool:
    """Set a top orthographic view centred on canvas-local ``rect`` so that it fits ``region``."""
    from ..bridge import canvas_matrix
    r3d = space.region_3d
    minx, miny, maxx, maxy = rect
    cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5
    prefs = bpy.context.preferences.view
    smooth = getattr(prefs, "smooth_view", None)
    try:  # proper "Top Orthographic" view, applied instantly (an animated view would overwrite the values below)
        if smooth is not None:
            prefs.smooth_view = 0
        with bpy.context.temp_override(area=area, region=region):
            bpy.ops.view3d.view_axis(type="TOP", align_active=False)
    except (RuntimeError, TypeError, AttributeError):
        pass
    finally:
        if smooth is not None:
            prefs.smooth_view = smooth
    r3d.view_perspective = "ORTHO"
    r3d.view_rotation = (1.0, 0.0, 0.0, 0.0)  # looking down -Z from +Z (top view)
    r3d.view_location = canvas_matrix(scene) @ Vector((cx, cy, 0.0))
    try:  # parenting every artwork object to the canvas root would draw long relationship lines
        space.overlay.show_relationship_lines = False
    except AttributeError:
        pass
    w = max(maxx - minx, 1e-3)
    h = max(maxy - miny, 1e-3)
    # ortho: visible extent along the larger region side = view_distance * 72 / lens (verified in the GUI
    # test: the artboard fills 90% of the viewport after this call).
    rw, rh = max(region.width, 1), max(region.height, 1)
    units_per_px = max(w / (rw * margin), h / (rh * margin))
    visible_larger = units_per_px * max(rw, rh)
    r3d.view_distance = visible_larger * space.lens / 72.0
    return True


class SEAM_OT_view_artboard(bpy.types.Operator):
    bl_idname = "seam.view_artboard"
    bl_label = "View Artboard"
    bl_description = "Top orthographic view centred on the active artboard, fitted to the viewport"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if bpy.app.background:
            self.report({"WARNING"}, "View framing needs a 3D viewport (background mode): nothing to do")
            return {"CANCELLED"}
        scene = context.scene
        area, region, space = _find_view3d(context)
        if space is None:
            self.report({"WARNING"}, "No 3D viewport to frame")
            return {"CANCELLED"}
        ab = active_artboard(scene)
        if ab is None:
            self.report({"WARNING"}, "No artboard in this document")
            return {"CANCELLED"}
        frame_rect_in_view(scene, area, region, space, artboard_rect(ab))
        area.tag_redraw()
        return {"FINISHED"}


class SEAM_OT_layer_add(bpy.types.Operator):
    bl_idname = "seam.layer_add"
    bl_label = "Add Layer"
    bl_description = "Add a layer on top of the stack and make it active"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty(name="Name", default="", description="Layer name (empty: numbered automatically)")

    def execute(self, context):
        scene = context.scene
        ensure_canvas(scene)
        lys = layers(scene)
        col = new_layer(scene, self.name or _next_name("Layer", [c.name for c in lys]))
        set_active_layer(scene, col)
        return {"FINISHED"}


class SEAM_OT_layer_remove(bpy.types.Operator):
    bl_idname = "seam.layer_remove"
    bl_label = "Remove Layer"
    bl_description = "Remove the active layer; its artwork moves to the neighbouring layer"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if len(layers(context.scene)) < 2:
            cls.poll_message_set("A document needs at least one layer")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        lys = layers(scene)
        col = active_layer(scene)
        i = lys.index(col)
        target = lys[i - 1] if i > 0 else lys[i + 1]
        for ob in list(col.objects):
            if ob.name not in target.objects:
                target.objects.link(ob)
            col.objects.unlink(ob)
        bpy.data.collections.remove(col)
        set_active_layer(scene, target)
        restack(scene)
        return {"FINISHED"}


class SEAM_OT_layer_move(bpy.types.Operator):
    bl_idname = "seam.layer_move"
    bl_label = "Move Layer"
    bl_description = "Move the active layer up or down in the stack"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(name="Direction", items=[("UP", "Up", "Towards the front"), ("DOWN", "Down", "Towards the back")],
                            default="UP")

    @classmethod
    def poll(cls, context):
        if len(layers(context.scene)) < 2:
            cls.poll_message_set("Only one layer")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        lys = layers(scene)
        col = active_layer(scene)
        i = lys.index(col)
        j = i + 1 if self.direction == "UP" else i - 1
        if j < 0 or j >= len(lys):
            self.report({"INFO"}, "Layer is already at the " + ("top" if self.direction == "UP" else "bottom"))
            return {"CANCELLED"}
        lys[i], lys[j] = lys[j], lys[i]
        set_layer_order(scene, lys)
        return {"FINISHED"}


class SEAM_OT_layer_set_active(bpy.types.Operator):
    bl_idname = "seam.layer_set_active"
    bl_label = "Set Active Layer"
    bl_description = "Make a layer the active one"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty(name="Layer", default="")

    def execute(self, context):
        col = bpy.data.collections.get(self.name)
        if col is None or col not in layers(context.scene):
            self.report({"ERROR"}, "Not a SEAM layer")
            return {"CANCELLED"}
        set_active_layer(context.scene, col)
        return {"FINISHED"}


# ------------------------------------------------------------------ UI

class SEAM_UL_layers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon="OUTLINER_COLLECTION")
        lc = layer_collection_of(item, context.view_layer)
        if lc is not None:
            row.prop(lc, "hide_viewport", text="", emboss=False)
        row.prop(item, "hide_select", text="", emboss=False)

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        flags = [self.bitflag_filter_item if _is_layer_collection(c) else 0 for c in items]
        order = list(range(len(items) - 1, -1, -1))  # top layer first
        return flags, order


class SEAM_UL_artboards(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        active = active_artboard(context.scene)
        row.label(text="", icon="CHECKMARK" if item == active else "BLANK1")
        row.prop(item, "name", text="", emboss=False)
        row.label(text=f"{item.seam.artboard.width:g} x {item.seam.artboard.height:g}")


def _layer_index_get(self):
    scene = bpy.context.scene
    art = bpy.data.collections.get(ARTWORK_NAME)
    if art is None:
        return 0
    act = active_layer(scene) if layers(scene) else None
    for i, c in enumerate(art.children):
        if c == act:
            return i
    return 0


def _layer_index_set(self, value):
    scene = bpy.context.scene
    art = bpy.data.collections.get(ARTWORK_NAME)
    if art is None or not (0 <= value < len(art.children)):
        return
    col = art.children[value]
    if _is_layer_collection(col):
        set_active_layer(scene, col)


def _artboard_index_get(self):
    scene = bpy.context.scene
    col = bpy.data.collections.get(ARTBOARDS_NAME)
    act = active_artboard(scene)
    if col is None or act is None:
        return 0
    for i, ob in enumerate(col.objects):
        if ob == act:
            return i
    return 0


def _artboard_index_set(self, value):
    scene = bpy.context.scene
    col = bpy.data.collections.get(ARTBOARDS_NAME)
    if col is None or not (0 <= value < len(col.objects)):
        return
    ob = col.objects[value]
    if ob.seam.role == "ARTBOARD":
        set_active_artboard(scene, ob)


class SEAM_PT_layers(bpy.types.Panel):
    bl_label = "Layers"
    bl_idname = "SEAM_PT_layers"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"
    bl_order = 10

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if not scene.seam.doc.canvas_ready:
            layout.label(text="No canvas yet", icon="INFO")
            layout.operator("seam.canvas_new", icon="ADD")
            return
        art = artwork_collection(scene)
        row = layout.row()
        row.template_list("SEAM_UL_layers", "", art, "children", context.window_manager, "seam_layer_index", rows=4)
        col = row.column(align=True)
        col.operator("seam.layer_add", icon="ADD", text="")
        col.operator("seam.layer_remove", icon="REMOVE", text="")
        col.separator()
        op = col.operator("seam.layer_move", icon="TRIA_UP", text="")
        op.direction = "UP"
        op = col.operator("seam.layer_move", icon="TRIA_DOWN", text="")
        op.direction = "DOWN"


class SEAM_PT_artboards(bpy.types.Panel):
    bl_label = "Artboards"
    bl_idname = "SEAM_PT_artboards"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"
    bl_order = 11

    @classmethod
    def poll(cls, context):
        return bool(context.scene.seam.doc.canvas_ready)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        abc = bpy.data.collections.get(ARTBOARDS_NAME)
        if abc is None:
            return
        row = layout.row()
        row.template_list("SEAM_UL_artboards", "", abc, "objects", context.window_manager, "seam_artboard_index", rows=3)
        col = row.column(align=True)
        col.operator("seam.artboard_add", icon="ADD", text="")
        col.operator("seam.artboard_remove", icon="REMOVE", text="")
        col.operator("seam.view_artboard", icon="VIEWZOOM", text="")
        ab = active_artboard(scene)
        if ab is not None:
            a = ab.seam.artboard
            box = layout.column(align=True)
            box.prop(a, "width")
            box.prop(a, "height")
            box.prop(a, "background_enabled")
            sub = box.row()
            sub.active = a.background_enabled
            sub.prop(a, "background_color", text="")
            box.prop(a, "export_scale")


classes = (SEAM_OT_canvas_new, SEAM_OT_artboard_add, SEAM_OT_artboard_remove, SEAM_OT_artboard_set_active,
           SEAM_OT_view_artboard, SEAM_OT_layer_add, SEAM_OT_layer_remove, SEAM_OT_layer_move,
           SEAM_OT_layer_set_active, SEAM_UL_layers, SEAM_UL_artboards, SEAM_PT_layers, SEAM_PT_artboards)

_producer = ArtboardProducer()


def register():
    bpy.types.WindowManager.seam_layer_index = IntProperty(get=_layer_index_get, set=_layer_index_set,
                                                           options={"SKIP_SAVE"})
    bpy.types.WindowManager.seam_artboard_index = IntProperty(get=_artboard_index_get, set=_artboard_index_set,
                                                              options={"SKIP_SAVE"})
    engine.register_producer(_producer)
    engine.add_update_hook(adoption_hook)


def unregister():
    engine.remove_update_hook(adoption_hook)
    engine.unregister_producer(ArtboardProducer.kind)
    for name in ("seam_layer_index", "seam_artboard_index"):
        if hasattr(bpy.types.WindowManager, name):
            delattr(bpy.types.WindowManager, name)
