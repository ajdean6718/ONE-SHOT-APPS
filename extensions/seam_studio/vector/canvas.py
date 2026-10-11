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
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import bpy
from mathutils import Matrix, Vector

from .. import ids
from ..bridge import write_paths
from ..seam_core import CANVAS_ORIGIN, LAYER_STEP, Z_STEP

ROOT_NAME = "SEAM Canvas"
ARTWORK_NAME = "SEAM Artwork"
RESULTS_NAME = "SEAM Results"
ARTBOARDS_NAME = "SEAM Artboards"
ARTWORK_ROLES = ("SHAPE", "TEXT", "RASTER", "GROUP")
ARTBOARD_Z = -1.0


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
    col.hide_render = True
    return col


def results_collection(scene=None) -> bpy.types.Collection:
    scene = scene or bpy.context.scene
    return _get_or_new_collection(RESULTS_NAME, scene.collection)


def _artboard_collection(scene) -> bpy.types.Collection:
    return _get_or_new_collection(ARTBOARDS_NAME, artwork_collection(scene))


def layers(scene=None) -> List[bpy.types.Collection]:
    scene = scene or bpy.context.scene
    art = artwork_collection(scene)
    return [c for c in art.children if c.name != ARTBOARDS_NAME and not c.name.startswith(ARTBOARDS_NAME)]


def new_layer(scene=None, name: str = "Layer") -> bpy.types.Collection:
    scene = scene or bpy.context.scene
    col = bpy.data.collections.new(name)
    artwork_collection(scene).children.link(col)
    return col


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
    return ob


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
    from ..bridge import canvas_matrix
    return canvas_matrix(scene) @ Vector((x, y, 0.0))


classes = ()
