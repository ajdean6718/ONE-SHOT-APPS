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
    if _pushes_undo(fn):
        STEPS[0] += 1
    return res


def _pushes_undo(fn):
    """True when the operator pushes an undo step: SEAM operators declare it in ``bl_options``; the native operators
    used by the scenarios (mode switch, transform, modifiers, delete, ...) all do - their flags are not introspectable."""
    name = fn.idname()
    cls = getattr(bpy.types, name, None)
    if cls is not None:
        return bool({"UNDO", "UNDO_GROUPED"} & set(getattr(cls, "bl_options", ())))
    return not name.startswith(("RENDER_", "WM_", "ED_"))


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


def move_nodes(ob, indices, dx=0.0, dy=0.0, spline=0, stay_in_edit_mode=False):
    """Tab, select control points ``indices`` of ``ob``, G (handles follow), Tab - each step pushes undo like the UI.

    ``stay_in_edit_mode`` keeps the object in Edit Mode afterwards (the caller must leave it)."""
    select(ob)
    if ob.mode != "EDIT":
        op(bpy.ops.object.mode_set, mode="EDIT")
    try:
        for si, sp in enumerate(ob.data.splines):
            for i, bp in enumerate(sp.bezier_points):
                on = si == spline and i in indices
                bp.select_control_point = on
                bp.select_left_handle = on
                bp.select_right_handle = on
        op(bpy.ops.transform.translate, value=(dx, dy, 0.0))
    except BaseException:
        bpy.ops.object.mode_set(mode="OBJECT")
        raise
    if not stay_in_edit_mode:
        op(bpy.ops.object.mode_set, mode="OBJECT")
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


def welded_topology(ob, decimals=6):
    """(V, E, F, edge_use) after welding vertices that share a position (SEAM meshes may split corners)."""
    me = ob.data
    co = mesh_np(ob)
    keys = {}
    ids = np.empty(len(co), dtype=np.int64)
    for i, c in enumerate(np.round(co, decimals)):
        ids[i] = keys.setdefault((c[0] + 0.0, c[1] + 0.0, c[2] + 0.0), len(keys))
    use = {}
    for p in me.polygons:
        v = [int(ids[i]) for i in p.vertices]
        for i in range(len(v)):
            e = (min(v[i], v[(i + 1) % len(v)]), max(v[i], v[(i + 1) % len(v)]))
            use[e] = use.get(e, 0) + 1
    return len(keys), len(use), len(me.polygons), use


def is_closed_manifold(ob):
    """Every edge of the welded mesh is shared by exactly two faces."""
    _v, _e, _f, use = welded_topology(ob)
    return bool(use) and all(c == 2 for c in use.values())


def euler_characteristic(ob):
    v, e, f, _use = welded_topology(ob)
    return v - e + f


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


# ------------------------------------------------------------------ a brand new Blender process

_CHILD_PRELUDE = '''
import json, os, sys
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
BLEND, OUT = argv[0], argv[1]
ROOT = {root!r}
for p in (ROOT + "/tests/acceptance", ROOT + "/tests", ROOT + "/src", ROOT + "/extensions"):
    if p not in sys.path:
        sys.path.insert(0, p)
import bpy, addon_utils
if addon_utils.enable("seam_studio", default_set=True, handle_error=None) is None:
    raise RuntimeError("seam_studio add-on did not enable")
RESULT = {{}}
'''

_CHILD_EPILOGUE = '''
with open(OUT, "w") as fh:
    json.dump(RESULT, fh)
sys.stdout.flush()
os._exit(0)
'''


def run_fresh_process(body, blend_path, tmp_path, timeout=300, crash=False):
    """Run ``body`` (Python source; ``BLEND`` is the file path, fill the dict ``RESULT``) in a new process of the
    same runtime - Blender binary or the ``bpy`` module - with the add-on enabled, and return RESULT."""
    import json
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(str(tmp_path), "child_%d.py" % abs(hash(body)))
    out = os.path.join(str(tmp_path), "child_out_%d.json" % abs(hash(body)))
    with open(script, "w") as fh:
        fh.write(_CHILD_PRELUDE.format(root=root) + body + _CHILD_EPILOGUE)
    binary = bpy.app.binary_path
    if binary and os.path.basename(binary).startswith("blender"):
        cmd = [binary, "-b", "--factory-startup", "--python-exit-code", "1", "--python", script, "--", str(blend_path), out]
    else:
        cmd = [sys.executable, script, str(blend_path), out]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=dict(os.environ))
    if crash:                                   # the body ends the process abruptly (os._exit(9)): no result file
        assert proc.returncode != 0 and not os.path.exists(out), (proc.returncode, proc.stdout[-2000:])
        return {"returncode": proc.returncode}
    assert proc.returncode == 0 and os.path.exists(out), "child failed (%s)\n%s\n%s" % (
        proc.returncode, proc.stdout[-3000:], proc.stderr[-3000:])
    with open(out) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ whole-project snapshot

def project_snapshot():
    """JSON-friendly description of every link, its source and its result (what a user would notice changing)."""
    import hashlib
    scene = bpy.context.scene
    out = {}
    for lk in scene.seam.links:
        t, s = lk.target, lk.source
        co = np.round(mesh_np(t), 5)
        out[lk.id] = {
            "recipe": lk.recipe, "state": lk.state, "stale": bool(lk.stale), "builds": int(lk.build_count),
            "params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in
                       ((k, getattr(lk.params, k)) for k in lk.params.bl_rna.properties.keys() if k != "rna_type")},
            "source": s.name if s else None, "target": t.name if t else None,
            "source_sig": lk.source_signature, "target_sig": lk.target_signature,
            "mesh": hashlib.blake2b(co.tobytes(), digest_size=8).hexdigest(), "verts": int(len(co)),
            "nodes": [[round(float(c), 4) for c in bp.co[:2]] for sp in s.data.splines for bp in sp.bezier_points]
            if s is not None and s.type == "CURVE" else None,
            "loc": [round(float(v), 5) for v in t.location], "rot": [round(float(v), 5) for v in t.rotation_euler],
            "scale": [round(float(v), 5) for v in t.scale],
            "modifiers": [(m.name, m.type) for m in t.modifiers],
            "materials": [(m.name if m else None, [round(float(c), 4) for c in m.diffuse_color] if m else None)
                          for m in t.data.materials],
            "role": t.seam.role, "owner_id": t.seam.owner_id,
        }
    return out
