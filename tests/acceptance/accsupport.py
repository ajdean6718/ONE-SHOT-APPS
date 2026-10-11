"""Shared helpers for the acceptance scenarios (not a test module).

Scenarios drive public operators the way the UI does: every operator is called with ``undo=True`` (so it
pushes an undo step like a click in the GUI would) and must report ``{'FINISHED'}``.  Node edits use
Blender's own Edit Mode + ``transform.translate`` (the G key).  Background mode has no timers, so
``flush()`` stands in for the engine tick.
"""

import math
import os

import bpy
import numpy as np

LAYER_FLOOR = 1e-6


# ------------------------------------------------------------------ operator driving

STEPS = [0]   # undo steps pushed since start_session(): UNDO operators + explicit push()


def op(fn, **kw):
    """Run an operator like a button press (undo step pushed); it must finish."""
    res = fn("EXEC_DEFAULT", True, **kw)
    assert res == {"FINISHED"}, "%s -> %s" % (getattr(fn, "idname", lambda: fn)(), res)
    if "UNDO" in getattr(getattr(bpy.types, fn.idname(), None), "bl_options", ()):
        STEPS[0] += 1
    return res


def push(message="edit"):
    """The undo step the UI pushes after a property edit (Python property writes push none)."""
    bpy.ops.ed.undo_push(message=message)
    STEPS[0] += 1
    return STEPS[0]


def flush():
    from seam_studio.engine import core
    return core.flush()


def start_session():
    """Initialise the (background-mode) undo system with a first step, like opening a file in the GUI."""
    bpy.ops.ed.undo_push(message="SEAM acceptance start")
    STEPS[0] = 0


def new_canvas(width=800, height=600):
    start_session()
    op(bpy.ops.seam.canvas_new, width=width, height=height)
    flush()
    from seam_studio.vector import canvas
    return canvas.active_artboard(bpy.context.scene)


def undo(n=1):
    for _ in range(n):
        assert bpy.ops.ed.undo() == {"FINISHED"}
    flush()


def redo(n=1):
    for _ in range(n):
        assert bpy.ops.ed.redo() == {"FINISHED"}
    flush()


def save_and_reopen(path):
    """Save, then open the file again (a fresh session: handlers, load_post, revalidation)."""
    assert bpy.ops.wm.save_as_mainfile(filepath=path) == {"FINISHED"}
    assert bpy.ops.wm.open_mainfile(filepath=path) == {"FINISHED"}
    return bpy.context.scene


# ------------------------------------------------------------------ selection / editing

def select(*objs, active=None):
    mode = bpy.context.mode
    if mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or (objs[-1] if objs else None)


def add_shape(kind, x, y, width, height, color=None, **kw):
    """Add a parametric shape through the Add Shape operator; ``color`` is scene-linear RGBA."""
    op(bpy.ops.seam.add_shape, kind=kind, x=x, y=y, width=width, height=height, **kw)
    ob = bpy.context.active_object
    if color is not None:
        ob.seam.style.fill_color = color
    flush()
    return ob


def move_nodes(ob, indices, dx=0.0, dy=0.0, spline=0):
    """Select control points ``indices`` of ``ob`` in Edit Mode and translate them (handles follow) with G."""
    select(ob)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for si, sp in enumerate(ob.data.splines):
            for i, bp in enumerate(sp.bezier_points):
                on = si == spline and i in indices
                bp.select_control_point = on
                bp.select_left_handle = on
                bp.select_right_handle = on
        res = bpy.ops.transform.translate("EXEC_DEFAULT", True, value=(dx, dy, 0.0))
        assert res == {"FINISHED"}, res
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    flush()


def node_positions(ob, spline=0):
    return [tuple(round(float(c), 4) for c in bp.co[:2]) for bp in ob.data.splines[spline].bezier_points]


# ------------------------------------------------------------------ geometry measurements

def mesh_np(ob):
    me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def extent(ob):
    co = mesh_np(ob)
    return co.max(axis=0) - co.min(axis=0)


def bbox(ob):
    co = mesh_np(ob)
    return co.min(axis=0), co.max(axis=0)


def mesh_volume(ob):
    """Signed volume of a closed triangulated/ngon mesh in object space (divergence theorem)."""
    me = ob.data
    co = mesh_np(ob)
    vol = 0.0
    for p in me.polygons:
        idx = list(p.vertices)
        a = co[idx[0]]
        for i in range(1, len(idx) - 1):
            b, c = co[idx[i]], co[idx[i + 1]]
            vol += float(np.dot(a, np.cross(b, c))) / 6.0
    return vol


def is_closed_manifold(ob):
    """Every edge is shared by exactly two faces."""
    me = ob.data
    counts = {}
    for p in me.polygons:
        v = list(p.vertices)
        for i in range(len(v)):
            e = (min(v[i], v[(i + 1) % len(v)]), max(v[i], v[(i + 1) % len(v)]))
            counts[e] = counts.get(e, 0) + 1
    return bool(counts) and all(c == 2 for c in counts.values())


def polygon_area_xy(ring):
    s = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def srgb8(linear):
    v = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return tuple(int(round(float(x) * 255)) for x in s)


# ------------------------------------------------------------------ 2D export

def png_pixel(img, ab, x, y, scale=1.0):
    """RGBA uint8 at canvas point (x, y) of an artboard render (``ab`` = artboard object)."""
    col = int(math.floor((x - ab.location.x) * scale))
    row = int(math.floor((ab.location.y - y) * scale))
    return tuple(int(v) for v in img[row, col])


def export_png(path, scale=1.0, transparent=False, **kw):
    """Run PNG Export (explicit scale/transparent: operator properties are remembered between calls)."""
    from seam_studio.seam_core import png
    op(bpy.ops.seam.png_export, filepath=str(path), scale=scale, transparent=transparent, **kw)
    return png.read_png(str(path))


# ------------------------------------------------------------------ links

def selected_link():
    return bpy.context.scene.seam.links[-1]


def link_by_id(link_id):
    for lk in bpy.context.scene.seam.links:
        if lk.id == link_id:
            return lk
    return None


def make_3d(src, recipe="EXTRUDE", **params):
    select(src)
    op(bpy.ops.seam.make_3d, recipe=recipe, **params)
    flush()
    return bpy.context.scene.seam.links[-1]


def results_in_scene():
    return [o for o in bpy.context.scene.objects if o.seam.role == "RESULT"]


def assert_link_consistent(link):
    """The stored signatures match the current source/mesh: nothing is silently stale."""
    from seam_studio.link import producer
    scene = bpy.context.scene
    assert link.source_signature == producer.source_signature(scene, link)
    assert producer.read_mesh_signature(link.target.data) == link.target_signature


def artifacts_dir(*parts):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, "artifacts", "acceptance", *parts)
    os.makedirs(d, exist_ok=True)
    return d
