"""Z-order, groups, align/distribute, numeric transform, guides (owner: vector-a).

Operators: seam.arrange, seam.group, seam.ungroup, seam.align, seam.distribute, seam.guide_add,
seam.guides_clear. Panel "Transform": artboard-relative X / Y (Y down) / W / H / rotation of the
active object, backed by get/set properties on ``Object`` (``seam_x``, ``seam_y``, ``seam_w``,
``seam_h``, ``seam_rot``) that edit the object transform (values shown in document units).

Align modes: LEFT, CENTER_H, RIGHT, TOP, CENTER_V, BOTTOM. Distribute axis: X | Y, by CENTERS | SPACING.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from mathutils import Matrix, Vector

from .. import ids
from ..bridge import canvas_local_matrix, rotation_2d
from ..engine import core as engine
from ..seam_core import LAYER_STEP, Z_STEP
from ..seam_core.units import px_to_unit, unit_to_px
from . import canvas

ARTWORK = canvas.ARTWORK_ROLES


# ------------------------------------------------------------------ selection helpers

def selected_artwork(context) -> List[bpy.types.Object]:
    """Selected artwork objects, without objects whose ancestor group is also selected."""
    sel = [o for o in context.selected_objects if o.seam.role in ARTWORK]
    names = {o.name for o in sel}
    out = []
    for o in sel:
        p = o.parent
        skip = False
        while p is not None:
            if p.name in names:
                skip = True
                break
            p = p.parent
        if not skip:
            out.append(o)
    return out


def _sibling_set(scene, ob) -> List[bpy.types.Object]:
    """Artwork objects stacked together with ``ob`` (same parent, same layer for canvas-level objects)."""
    parent = ob.parent
    if parent is None:
        return [ob]
    sibs = [c for c in parent.children if c.seam.role in ARTWORK]
    if parent.seam.role == "CANVAS_ROOT":
        layer = next((c for c in ob.users_collection if c in canvas.layers(scene)), None)
        if layer is not None:
            sibs = [c for c in sibs if layer in c.users_collection]
    sibs.sort(key=lambda o: (o.location.z, o.name))
    return sibs


def _stack_base(scene, ob, count: int = 1) -> Tuple[float, float]:
    """(base z, step) for the stack ``ob`` is in. Canvas-level artwork steps by Z_STEP per layer slot; the
    children of a group share the group's slot equally (``canvas.node_span``), so a child's stroke (half a
    slot above it) and a nested group's children always stay below the next sibling."""
    parent = ob.parent
    if parent is not None and parent.seam.role == "CANVAS_ROOT":
        layer = next((c for c in ob.users_collection if c in canvas.layers(scene)), None)
        li = canvas.layers(scene).index(layer) if layer is not None else 0
        return li * LAYER_STEP, Z_STEP
    return 0.0, canvas.node_span(ob)


def _compact_group(group) -> None:
    """Spread the children of a group evenly over the group's z slot (sorted by current z)."""
    kids = [c for c in group.children if c.seam.role in ARTWORK]
    kids.sort(key=lambda o: (o.location.z, o.name))
    for i, c in enumerate(kids):
        step = canvas.node_span(c)
        c.location.z = i * step
        _after_z(c)


def _after_z(o) -> None:
    """Keep derived z offsets consistent after ``o`` got a new z / a new slot size."""
    if o.seam.role == "GROUP":
        _compact_group(o)
    else:
        canvas.fix_stroke_z(o, recurse=False)


def reorder_stack(stack: List, selected: set, mode: str) -> List:
    """Pure reorder of ``stack`` (bottom -> top) for FRONT / FORWARD / BACKWARD / BACK."""
    lst = list(stack)
    n = len(lst)
    if mode == "FRONT":
        return [o for o in lst if o not in selected] + [o for o in lst if o in selected]
    if mode == "BACK":
        return [o for o in lst if o in selected] + [o for o in lst if o not in selected]
    if mode == "FORWARD":
        for i in range(n - 2, -1, -1):
            if lst[i] in selected and lst[i + 1] not in selected:
                lst[i], lst[i + 1] = lst[i + 1], lst[i]
        return lst
    if mode == "BACKWARD":
        for i in range(1, n):
            if lst[i] in selected and lst[i - 1] not in selected:
                lst[i], lst[i - 1] = lst[i - 1], lst[i]
        return lst
    raise ValueError(mode)


def assign_stack_z(scene, ordered: Sequence, base: float, step: float) -> None:
    for i, o in enumerate(ordered):
        o.location.z = base + i * step
        _after_z(o)


# ------------------------------------------------------------------ canvas-space helpers

def translate_canvas(ob, dx: float, dy: float) -> None:
    """Move ``ob`` by (dx, dy) canvas px regardless of its parent chain."""
    if dx == 0.0 and dy == 0.0:
        return
    parent = ob.parent
    if parent is None or parent.seam.role == "CANVAS_ROOT":
        ob.location.x += dx
        ob.location.y += dy
        return
    lin = canvas_local_matrix(parent).to_3x3().inverted_safe()
    d = lin @ Vector((dx, dy, 0.0))
    ob.location.x += d.x
    ob.location.y += d.y


def union_bounds(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


# ------------------------------------------------------------------ arrange

class SEAM_OT_arrange(bpy.types.Operator):
    bl_idname = "seam.arrange"
    bl_label = "Arrange"
    bl_description = "Change the stacking order of the selected artwork within its layer"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(name="Mode", items=[
        ("FRONT", "Bring to Front", "Move to the top of the layer"),
        ("FORWARD", "Bring Forward", "Move one step up"),
        ("BACKWARD", "Send Backward", "Move one step down"),
        ("BACK", "Send to Back", "Move to the bottom of the layer"),
    ], default="FRONT")

    @classmethod
    def poll(cls, context):
        if not selected_artwork(context):
            cls.poll_message_set("Select shapes, text or groups")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        sel = selected_artwork(context)
        done = set()
        for ob in sel:
            sibs = _sibling_set(scene, ob)
            key = tuple(o.name for o in sibs)
            if key in done:
                continue
            done.add(key)
            chosen = {o for o in sibs if o in sel}
            new = reorder_stack(sibs, chosen, self.mode)
            base, step = _stack_base(scene, ob, len(sibs))
            assign_stack_z(scene, new, base, step)
        return {"FINISHED"}


# ------------------------------------------------------------------ group / ungroup

def _is_pure_translation(ob) -> bool:
    return (all(abs(a) == 0.0 for a in ob.rotation_euler) and all(abs(s - 1.0) == 0.0 for s in ob.scale)
            and ob.rotation_mode != "QUATERNION" and ob.delta_location.length == 0.0)


class SEAM_OT_group(bpy.types.Operator):
    bl_idname = "seam.group"
    bl_label = "Group"
    bl_description = "Group the selected artwork under a new group (world transforms are preserved)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        sel = selected_artwork(context)
        if not sel:
            cls.poll_message_set("Select shapes, text or groups")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        sel = selected_artwork(context)
        parents = {o.parent for o in sel}
        if len(parents) != 1 or None in parents:
            self.report({"ERROR"}, "Selected objects must share one parent (do not mix grouped and ungrouped objects)")
            return {"CANCELLED"}
        parent = parents.pop()
        layer = None
        if parent.seam.role == "CANVAS_ROOT":
            layers_sel = {next((c for c in o.users_collection if c in canvas.layers(scene)), None) for o in sel}
            if len(layers_sel) != 1 or None in layers_sel:
                self.report({"ERROR"}, "Selected objects span several layers")
                return {"CANCELLED"}
            layer = layers_sel.pop()
        dg = context.evaluated_depsgraph_get()
        bounds = union_bounds([canvas.canvas_bounds(o, dg, scene) for o in sel])
        # group origin: bounds centre snapped to whole px (keeps float32 world matrices bit-exact)
        if bounds is not None:
            cx = round((bounds[0] + bounds[2]) * 0.5)
            cy = round((bounds[1] + bounds[3]) * 0.5)
        else:
            cx = cy = 0.0
        if parent.seam.role != "CANVAS_ROOT":
            # inside another group: express the centre in the parent's local space (translation only)
            pass
        gz = min(o.location.z for o in sel)
        g = bpy.data.objects.new("Group", None)
        g.empty_display_type = "PLAIN_AXES"
        g.empty_display_size = 10.0
        coll = layer or (sel[0].users_collection[0] if sel[0].users_collection else scene.collection)
        coll.objects.link(g)
        ids.ensure_id(g)
        g.seam.role = "GROUP"
        g.parent = parent
        g.matrix_parent_inverse = Matrix.Identity(4)
        if parent.seam.role == "CANVAS_ROOT":
            g.location = (cx, cy, gz)
        else:
            g.location = (0.0, 0.0, gz)  # nested: origin at the parent's origin (no rotation hazards)
            cx = cy = 0.0
        gl = Vector(g.location)
        for o in sel:
            old_mb = o.matrix_basis.copy()
            o.parent = g
            o.matrix_parent_inverse = Matrix.Identity(4)
            if _is_pure_translation(g):
                o.location = (o.location.x - gl.x, o.location.y - gl.y, o.location.z - gl.z)
            else:
                o.matrix_basis = g.matrix_basis.inverted_safe() @ old_mb
            if coll not in o.users_collection:
                coll.objects.link(o)
        for o in sel:
            canvas.fix_stroke_z(o)  # the children's slots shrank to a share of the group's slot
        canvas.select_only(context, [g], g)
        engine.flush(scene, evaluate=False)
        return {"FINISHED"}


class SEAM_OT_ungroup(bpy.types.Operator):
    bl_idname = "seam.ungroup"
    bl_label = "Ungroup"
    bl_description = "Dissolve the selected groups; children keep their world transforms"
    bl_options = {"REGISTER", "UNDO"}

    @staticmethod
    def _groups(context):
        out = []
        for o in context.selected_objects:
            g = o if o.seam.role == "GROUP" else (o.parent if o.parent is not None and o.parent.seam.role == "GROUP" else None)
            if g is not None and g not in out:
                out.append(g)
        return out

    @classmethod
    def poll(cls, context):
        if not cls._groups(context):
            cls.poll_message_set("Select a group")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        groups = self._groups(context)
        freed = []
        lost = []
        canvas.select_only(context, [])  # deselect first: removing a group leaves a None base in the view layer
        for g in groups:
            kids = [c for c in g.children if c.seam.role in ARTWORK]
            new_parent = g.parent
            gmb = g.matrix_basis.copy()
            gl = Vector(g.location)
            pure = _is_pure_translation(g)
            for c in kids:
                old_mb = c.matrix_basis.copy()
                c.parent = new_parent
                c.matrix_parent_inverse = Matrix.Identity(4)
                if pure:
                    c.location = (c.location.x + gl.x, c.location.y + gl.y, c.location.z + gl.z)
                elif not _apply_composed(c, gmb @ old_mb):
                    lost.append(c.name)
                freed.append(c)
            bpy.data.objects.remove(g, do_unlink=True)
        try:
            context.view_layer.update()
        except RuntimeError:
            pass
        for c in freed:
            canvas.fix_stroke_z(c)
        canvas.select_only(context, freed, freed[0] if freed else None)
        engine.flush(scene, evaluate=False)
        if lost:
            self.report({"WARNING"}, "Shear could not be kept for: " + ", ".join(lost[:5]) +
                        (" ..." if len(lost) > 5 else ""))
        self.report({"INFO"}, f"Ungrouped {len(groups)} group(s)")
        return {"FINISHED"}


def _apply_composed(c, composed) -> bool:
    """Set the local transform of ``c`` to ``composed`` (a 4x4 that may contain shear).

    Blender transforms are loc/rot/scale and cannot hold shear (rotated child inside a non-uniformly
    scaled group). For curve shapes the residual shear is baked into the curve points (the shape is
    expanded to a free path), so the world matrix of the geometry is unchanged. Returns False when
    shear had to be dropped (groups, text).
    """
    loc, rot, scl = composed.decompose()
    d = Matrix.LocRotScale(loc, rot, scl)
    resid = d.inverted_safe() @ composed
    shear = max(abs(resid[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3))
    if shear <= 1e-6:
        c.matrix_basis = composed
        return True
    if c.type == "CURVE" and c.seam.role == "SHAPE" and c.data is not None and c.data.users == 1:
        if c.seam.shape.kind != "NONE":
            c.seam.shape.kind = "NONE"
        c.data.transform(resid)
        c.matrix_basis = d
        return True
    c.matrix_basis = composed
    return False


# ------------------------------------------------------------------ align / distribute

def _gather_bounds(context, objs):
    dg = context.evaluated_depsgraph_get()
    items = []
    for o in objs:
        b = canvas.canvas_bounds(o, dg, context.scene)
        if b is not None:
            items.append((o, b))
    return items


def align_deltas(mode: str, boxes: Sequence[Tuple[float, float, float, float]], target) -> List[Tuple[float, float]]:
    """Per-box (dx, dy) to align ``boxes`` to the ``target`` box (minx, miny, maxx, maxy)."""
    out = []
    for b in boxes:
        dx = dy = 0.0
        if mode == "LEFT":
            dx = target[0] - b[0]
        elif mode == "RIGHT":
            dx = target[2] - b[2]
        elif mode == "CENTER_H":
            dx = (target[0] + target[2]) * 0.5 - (b[0] + b[2]) * 0.5
        elif mode == "TOP":
            dy = target[3] - b[3]
        elif mode == "BOTTOM":
            dy = target[1] - b[1]
        elif mode == "CENTER_V":
            dy = (target[1] + target[3]) * 0.5 - (b[1] + b[3]) * 0.5
        else:
            raise ValueError(mode)
        out.append((dx, dy))
    return out


def distribute_deltas(axis: str, by: str, boxes: Sequence[Tuple[float, float, float, float]]) -> List[float]:
    """Per-box shift along ``axis`` ('X' or 'Y'); the two outermost boxes stay in place."""
    i0, i1 = (0, 2) if axis == "X" else (1, 3)
    n = len(boxes)
    order = sorted(range(n), key=lambda k: ((boxes[k][i0] + boxes[k][i1]) * 0.5, k))
    deltas = [0.0] * n
    if n < 3:
        return deltas
    if by == "CENTERS":
        c_first = (boxes[order[0]][i0] + boxes[order[0]][i1]) * 0.5
        c_last = (boxes[order[-1]][i0] + boxes[order[-1]][i1]) * 0.5
        step = (c_last - c_first) / (n - 1)
        for rank, k in enumerate(order):
            cur = (boxes[k][i0] + boxes[k][i1]) * 0.5
            deltas[k] = c_first + step * rank - cur
    else:
        first, last = boxes[order[0]], boxes[order[-1]]
        span = last[i1] - first[i0]
        total = sum(boxes[k][i1] - boxes[k][i0] for k in order)
        gap = (span - total) / (n - 1)
        pos = first[i0]
        for k in order:
            deltas[k] = pos - boxes[k][i0]
            pos += (boxes[k][i1] - boxes[k][i0]) + gap
    return deltas


class SEAM_OT_align(bpy.types.Operator):
    bl_idname = "seam.align"
    bl_label = "Align"
    bl_description = "Align the selected artwork (bounds of the evaluated geometry)"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(name="Mode", items=[
        ("LEFT", "Left", ""), ("CENTER_H", "Horizontal Center", ""), ("RIGHT", "Right", ""),
        ("TOP", "Top", ""), ("CENTER_V", "Vertical Center", ""), ("BOTTOM", "Bottom", ""),
    ], default="LEFT")
    relative_to: EnumProperty(name="Relative To", items=[
        ("SELECTION", "Selection", "Bounds of the whole selection"),
        ("ARTBOARD", "Artboard", "The active artboard"),
        ("ACTIVE", "Active Object", "The active object stays where it is"),
    ], default="SELECTION")

    @classmethod
    def poll(cls, context):
        sel = selected_artwork(context)
        if not sel:
            cls.poll_message_set("Select shapes, text or groups")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        sel = selected_artwork(context)
        items = _gather_bounds(context, sel)
        if not items:
            self.report({"WARNING"}, "Selected objects have no geometry")
            return {"CANCELLED"}
        if self.relative_to == "ARTBOARD":
            ab = canvas.active_artboard(scene)
            if ab is None:
                self.report({"ERROR"}, "No artboard in this document")
                return {"CANCELLED"}
            target = canvas.artboard_rect(ab)
            moving = items
        elif self.relative_to == "ACTIVE":
            act = context.active_object
            tb = next((b for o, b in items if o == act), None)
            if tb is None:
                self.report({"ERROR"}, "The active object is not part of the selection")
                return {"CANCELLED"}
            target = tb
            moving = [(o, b) for o, b in items if o != act]
        else:
            if len(items) < 2:
                self.report({"WARNING"}, "Select at least two objects to align to the selection")
                return {"CANCELLED"}
            target = union_bounds([b for _o, b in items])
            moving = items
        deltas = align_deltas(self.mode, [b for _o, b in moving], target)
        for (o, _b), (dx, dy) in zip(moving, deltas):
            translate_canvas(o, dx, dy)
        return {"FINISHED"}


class SEAM_OT_distribute(bpy.types.Operator):
    bl_idname = "seam.distribute"
    bl_label = "Distribute"
    bl_description = "Distribute the selected artwork evenly between the outermost two objects"
    bl_options = {"REGISTER", "UNDO"}

    axis: EnumProperty(name="Axis", items=[("X", "Horizontal", "Along X"), ("Y", "Vertical", "Along Y")], default="X")
    by: EnumProperty(name="By", items=[("CENTERS", "Centers", "Equal distance between centers"),
                                       ("SPACING", "Spacing", "Equal gaps between objects")], default="CENTERS")

    @classmethod
    def poll(cls, context):
        if len(selected_artwork(context)) < 3:
            cls.poll_message_set("Select at least three objects")
            return False
        return True

    def execute(self, context):
        items = _gather_bounds(context, selected_artwork(context))
        if len(items) < 3:
            self.report({"WARNING"}, "Need at least three objects with geometry")
            return {"CANCELLED"}
        deltas = distribute_deltas(self.axis, self.by, [b for _o, b in items])
        for (o, _b), d in zip(items, deltas):
            if self.axis == "X":
                translate_canvas(o, d, 0.0)
            else:
                translate_canvas(o, 0.0, d)
        return {"FINISHED"}


# ------------------------------------------------------------------ numeric transform

def _scene():
    return bpy.context.scene


def _units(scene):
    d = scene.seam.doc
    return d.units, d.dpi


def object_box(ob):
    return canvas.canvas_bounds(ob, bpy.context.evaluated_depsgraph_get(), _scene())


_ui_cache: dict = {"on": False, "key": None, "vals": None}


@contextmanager
def ui_cache():
    """Cache ``ui_values`` while a panel draws: its five properties read the same evaluated bounds, so they
    are computed once instead of five times per redraw. Outside this context nothing is cached."""
    _ui_cache.update(on=True, key=None, vals=None)
    try:
        yield
    finally:
        _ui_cache.update(on=False, key=None, vals=None)


def ui_values(ob):
    """(x, y, w, h, rotation_deg) in canvas px, artboard-relative with Y down; None without geometry."""
    if not _ui_cache["on"]:
        return _ui_values_uncached(ob)
    try:
        key = ob.as_pointer()
    except ReferenceError:
        return None
    if _ui_cache["key"] != key:
        _ui_cache.update(key=key, vals=_ui_values_uncached(ob))
    return _ui_cache["vals"]


def _ui_values_uncached(ob):
    scene = _scene()
    b = object_box(ob)
    if b is None:
        return None
    ab = canvas.active_artboard(scene)
    ax, ay = (ab.location.x, ab.location.y) if ab is not None else (0.0, 0.0)
    return (b[0] - ax, ay - b[3], b[2] - b[0], b[3] - b[1], math.degrees(rotation_2d(ob, scene)))


def set_ui_position(ob, x: Optional[float] = None, y: Optional[float] = None) -> None:
    scene = _scene()
    b = object_box(ob)
    if b is None:
        return
    ab = canvas.active_artboard(scene)
    ax, ay = (ab.location.x, ab.location.y) if ab is not None else (0.0, 0.0)
    dx = dy = 0.0
    if x is not None:
        dx = (ax + x) - b[0]
    if y is not None:
        dy = (ay - y) - b[3]
    translate_canvas(ob, dx, dy)


def set_ui_size(ob, w: Optional[float] = None, h: Optional[float] = None) -> bool:
    """Resize the bounds of ``ob`` keeping its top-left corner. Scales along the object's own axes."""
    scene = _scene()
    b = object_box(ob)
    if b is None:
        return False
    cw, ch = b[2] - b[0], b[3] - b[1]
    th = rotation_2d(ob, scene)
    c, s = abs(math.cos(th)), abs(math.sin(th))
    sx, sy = ob.scale.x, ob.scale.y
    if abs(sx) < 1e-9 or abs(sy) < 1e-9:
        return False
    # extents along the object's own axes at current scale: A (x axis), B (y axis)
    # bounds: W = c*A + s*B ; H = s*A + c*B
    det = c * c - s * s
    if abs(det) > 1e-6:
        A = (c * cw - s * ch) / det
        B = (c * ch - s * cw) / det
    else:
        A = B = None
    nw = cw if w is None else max(w, 1e-6)
    nh = ch if h is None else max(h, 1e-6)
    if A is None or A <= 1e-9 or B <= 1e-9:
        if c > s and w is not None and cw > 1e-9:
            ob.scale.x = sx * nw / cw
        if c > s and h is not None and ch > 1e-9:
            ob.scale.y = sy * nh / ch
        if s >= c and w is not None and cw > 1e-9:
            ob.scale.y = sy * nw / cw
        if s >= c and h is not None and ch > 1e-9:
            ob.scale.x = sx * nh / ch
    else:
        nA = (c * nw - s * nh) / det
        nB = (c * nh - s * nw) / det
        if nA > 1e-9 and nB > 1e-9:
            ob.scale.x = sx * nA / A
            ob.scale.y = sy * nB / B
    nb = object_box(ob)
    if nb is not None:  # keep the top-left corner
        translate_canvas(ob, b[0] - nb[0], b[3] - nb[3])
    return True


def set_ui_rotation(ob, deg: float) -> None:
    scene = _scene()
    b = object_box(ob)
    cur = rotation_2d(ob, scene)
    ob.rotation_euler.z += math.radians(deg) - cur
    if b is not None:  # rotate about the bounds centre
        nb = object_box(ob)
        if nb is not None:
            translate_canvas(ob, (b[0] + b[2]) * 0.5 - (nb[0] + nb[2]) * 0.5, (b[1] + b[3]) * 0.5 - (nb[1] + nb[3]) * 0.5)


_NUM = {"x": 0, "y": 1, "w": 2, "h": 3, "rot": 4}


def _getter(field):
    def get(self):
        if self.seam.role not in ARTWORK:
            return 0.0
        try:
            vals = ui_values(self)
        except Exception:
            return 0.0
        if vals is None:
            return 0.0
        v = vals[_NUM[field]]
        if field == "rot":
            return v
        unit, dpi = _units(_scene())
        return px_to_unit(v, unit, dpi)
    return get


def _setter(field):
    def set_(self, value):
        if self.seam.role not in ARTWORK:
            return
        _ui_cache["key"] = None
        if field == "rot":
            set_ui_rotation(self, value)
            return
        unit, dpi = _units(_scene())
        px = unit_to_px(value, unit, dpi)
        if field in ("x", "y"):
            set_ui_position(self, x=px if field == "x" else None, y=px if field == "y" else None)
        else:
            set_ui_size(self, w=px if field == "w" else None, h=px if field == "h" else None)
    return set_


def _register_object_props():
    bpy.types.Object.seam_x = FloatProperty(name="X", description="Left edge relative to the active artboard",
                                            get=_getter("x"), set=_setter("x"), options={"SKIP_SAVE"})
    bpy.types.Object.seam_y = FloatProperty(name="Y", description="Top edge relative to the active artboard (down)",
                                            get=_getter("y"), set=_setter("y"), options={"SKIP_SAVE"})
    bpy.types.Object.seam_w = FloatProperty(name="W", description="Width of the bounds", min=0.0,
                                            get=_getter("w"), set=_setter("w"), options={"SKIP_SAVE"})
    bpy.types.Object.seam_h = FloatProperty(name="H", description="Height of the bounds", min=0.0,
                                            get=_getter("h"), set=_setter("h"), options={"SKIP_SAVE"})
    bpy.types.Object.seam_rot = FloatProperty(name="Rotation", description="Rotation on the canvas in degrees",
                                              subtype="NONE", get=_getter("rot"), set=_setter("rot"),
                                              options={"SKIP_SAVE"})


class SEAM_PT_transform(bpy.types.Panel):
    bl_label = "Transform"
    bl_idname = "SEAM_PT_transform"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"
    bl_order = 25

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.seam.role in ARTWORK

    def draw(self, context):
        with ui_cache():
            self._draw(context)

    def _draw(self, context):
        layout = self.layout
        ob = context.active_object
        unit = context.scene.seam.doc.units
        layout.label(text=f"Relative to active artboard ({unit.lower() if unit != 'PX' else 'px'})")
        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(ob, "seam_x", text="X")
        row.prop(ob, "seam_y", text="Y")
        row = col.row(align=True)
        row.prop(ob, "seam_w", text="W")
        row.prop(ob, "seam_h", text="H")
        layout.prop(ob, "seam_rot")
        row = layout.row(align=True)
        row.operator("seam.arrange", text="", icon="TRIA_UP_BAR").mode = "FRONT"
        row.operator("seam.arrange", text="", icon="TRIA_UP").mode = "FORWARD"
        row.operator("seam.arrange", text="", icon="TRIA_DOWN").mode = "BACKWARD"
        row.operator("seam.arrange", text="", icon="TRIA_DOWN_BAR").mode = "BACK"
        row = layout.row(align=True)
        row.operator("seam.group", icon="GROUP")
        row.operator("seam.ungroup")
        col = layout.column(align=True)
        row = col.row(align=True)
        for mode, icon in (("LEFT", "ALIGN_LEFT"), ("CENTER_H", "ALIGN_CENTER"), ("RIGHT", "ALIGN_RIGHT")):
            row.operator("seam.align", text="", icon=icon).mode = mode
        row = col.row(align=True)
        for mode, icon in (("TOP", "ALIGN_TOP"), ("CENTER_V", "ALIGN_MIDDLE"), ("BOTTOM", "ALIGN_BOTTOM")):
            row.operator("seam.align", text="", icon=icon).mode = mode
        row = layout.row(align=True)
        op = row.operator("seam.distribute", text="Distribute H")
        op.axis, op.by = "X", "CENTERS"
        op = row.operator("seam.distribute", text="Distribute V")
        op.axis, op.by = "Y", "CENTERS"


# ------------------------------------------------------------------ guides

def make_guide(scene, orientation: str, position: float) -> bpy.types.Object:
    """Create a guide line object at canvas-local ``position`` (x for VERTICAL, y for HORIZONTAL)."""
    canvas.ensure_canvas(scene)
    length = 100000.0
    verts = [(-length, 0.0, 0.0), (length, 0.0, 0.0)] if orientation == "HORIZONTAL" else \
            [(0.0, -length, 0.0), (0.0, length, 0.0)]
    me = bpy.data.meshes.new("Guide")
    me.from_pydata(verts, [(0, 1)], [])
    ob = bpy.data.objects.new("Guide", me)
    canvas.guides_collection(scene).objects.link(ob)
    ob.parent = canvas.canvas_root(scene)
    ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.location = (0.0, position, 5.0) if orientation == "HORIZONTAL" else (position, 0.0, 5.0)
    ids.ensure_id(ob)
    ob.seam.role = "GUIDE"
    ob["seam_orientation"] = orientation
    ob.display_type = "WIRE"
    ob.hide_render = True
    ob.show_in_front = True
    ob.color = (0.0, 0.7, 1.0, 1.0)
    return ob


class SEAM_OT_guide_add(bpy.types.Operator):
    bl_idname = "seam.guide_add"
    bl_label = "Add Guide"
    bl_description = "Add a horizontal or vertical guide line (not rendered, not exported)"
    bl_options = {"REGISTER", "UNDO"}

    orientation: EnumProperty(name="Orientation", items=[("HORIZONTAL", "Horizontal", ""), ("VERTICAL", "Vertical", "")],
                              default="HORIZONTAL")
    position: FloatProperty(name="Position", default=0.0,
                            description="Distance in px from the artboard's top-left corner (Y down for horizontal "
                                        "guides, X right for vertical ones); canvas coordinates when 'Artboard Relative' is off")
    artboard_relative: BoolProperty(name="Artboard Relative", default=True)

    def execute(self, context):
        scene = context.scene
        pos = self.position
        if self.artboard_relative:
            canvas.ensure_canvas(scene)
            ab = canvas.active_artboard(scene)
            if ab is not None:
                pos = ab.location.x + pos if self.orientation == "VERTICAL" else ab.location.y - pos
        make_guide(scene, self.orientation, pos)
        return {"FINISHED"}


class SEAM_OT_guides_clear(bpy.types.Operator):
    bl_idname = "seam.guides_clear"
    bl_label = "Clear Guides"
    bl_description = "Remove all guides"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not any(o.seam.role == "GUIDE" for o in context.scene.objects):
            cls.poll_message_set("There are no guides")
            return False
        return True

    def execute(self, context):
        n = 0
        for ob in [o for o in context.scene.objects if o.seam.role == "GUIDE"]:
            me = ob.data
            bpy.data.objects.remove(ob, do_unlink=True)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
            n += 1
        self.report({"INFO"}, f"Removed {n} guide(s)")
        return {"FINISHED"}


classes = (SEAM_OT_arrange, SEAM_OT_group, SEAM_OT_ungroup, SEAM_OT_align, SEAM_OT_distribute,
           SEAM_OT_guide_add, SEAM_OT_guides_clear, SEAM_PT_transform)


def register():
    _register_object_props()


def unregister():
    for name in ("seam_x", "seam_y", "seam_w", "seam_h", "seam_rot"):
        if hasattr(bpy.types.Object, name):
            delattr(bpy.types.Object, name)
