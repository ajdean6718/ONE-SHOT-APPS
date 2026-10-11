"""Boolean operators (union/subtract/intersect/exclude) and compound paths (owner: vector-b).

* ``seam.pathfinder(op, keep_originals, resolution)`` works on >= 2 selected SHAPE objects in canvas
  space. Every shape is flattened at ``resolution`` samples per Bezier segment, so the result is a
  polygon (VECTOR handles) - curvature is not preserved (documented gap, see architecture D11).
  UNION: all shapes merged. DIFFERENCE: bottom-most shape minus all others ("Minus Front").
  INTERSECT: area common to all. EXCLUDE: area covered by an odd number of shapes.
  The result is one compound SHAPE (holes are extra splines) that takes the style of the bottom-most
  input and the z slot of the top-most input (above everything when originals are kept).
* ``seam.compound_make`` merges the selected shapes into one compound path *without* flattening
  (exact Bezier data, even-odd fill, so overlapping parts become holes); ``seam.compound_release``
  splits a compound shape back into one shape per spline.
* Invalid input produces actionable error reports and leaves the originals untouched.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty

from ..bridge import curve_paths, write_paths
from ..seam_core import boolean2d, geometry
from ..seam_core.geometry import BezierPath, BezierPoint
from . import canvas
from .export_png import artwork_in_order, canvas_paths, copy_style, fill_rings, resolve_rings

PATHFINDER_OPS = [
    ("UNION", "Unite", "Merge all selected shapes into one outline"),
    ("DIFFERENCE", "Minus Front", "Subtract every shape in front from the bottom-most shape"),
    ("INTERSECT", "Intersect", "Keep only the area shared by all selected shapes"),
    ("EXCLUDE", "Exclude", "Keep the area covered by an odd number of shapes"),
]
_OP_LABEL = {k: v for k, v, _d in PATHFINDER_OPS}


# ------------------------------------------------------------------ shared object helpers

def selected_shapes(context) -> List[bpy.types.Object]:
    """Selected SHAPE objects of the scene, bottom -> top."""
    sel = {o.name for o in context.selected_objects if o.seam.role == "SHAPE" and o.type == "CURVE"}
    return [o for o in artwork_in_order(context.scene, roles=("SHAPE",), include_hidden=True, include_masks=True)
            if o.name in sel]


def delete_artwork(obj) -> None:
    """Delete an artwork object with its derived children, release dependants and drop orphaned data."""
    scene = bpy.context.scene
    sid = obj.seam.id
    doomed = []
    released = []
    for ob in list(bpy.data.objects):
        if ob is obj:
            continue
        if ob.seam.clip_source is obj:
            released.append(ob)
        if sid and ob.seam.owner_id == sid:
            if ob.seam.role in ("STROKE", "OUTLINE", "CLIP_RESULT"):
                doomed.append(ob)
            elif ob.seam.role == "SHAPE":      # live text outline: becomes an ordinary shape
                ob.seam.owner_id = ""
    if released:
        from .clip import release_clip      # local import: clip does not depend on this module
        release_clip(scene, released)
    for ch in obj.children_recursive:
        if ch.seam.role in ("STROKE", "OUTLINE", "CLIP_RESULT", "INTERNAL") and ch not in doomed:
            doomed.append(ch)
    for ob in doomed:
        _remove_object(ob)
    _remove_object(obj)


def _remove_object(obj) -> None:
    try:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
    except ReferenceError:
        return
    if data is not None and getattr(data, "users", 1) == 0:
        for coll in (bpy.data.curves, bpy.data.meshes):
            if data.name in coll and coll[data.name] is data:
                coll.remove(data)
                break


def paths_center(paths: Sequence[BezierPath]) -> Tuple[float, float]:
    pts = [p.co for path in paths for p in path.points]
    if not pts:
        return (0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))


def translate_paths(paths: Sequence[BezierPath], dx: float, dy: float) -> List[BezierPath]:
    def t(p):
        return (p[0] + dx, p[1] + dy)
    return [BezierPath([BezierPoint(t(pt.co), t(pt.handle_left), t(pt.handle_right), pt.handle_left_type,
                                    pt.handle_right_type) for pt in path.points], path.closed) for path in paths]


def polygon_path(ring: Sequence[Tuple[float, float]]) -> BezierPath:
    pts = [BezierPoint((float(x), float(y)), (float(x), float(y)), (float(x), float(y)), "VECTOR", "VECTOR")
           for x, y in ring]
    return BezierPath(pts, True)


def shapes_to_rings(shapes) -> List[List[Tuple[float, float]]]:
    """[(outer_ccw, [holes_cw])] -> flat ring list (outer first, then its holes)."""
    out = []
    for outer, holes in shapes:
        out.append(list(outer))
        out.extend(list(h) for h in holes)
    return out


def make_shape_from_canvas_paths(scene, name: str, paths_canvas: Sequence[BezierPath], like=None, layer=None,
                                 z: Optional[float] = None):
    """New SHAPE with origin at the bbox center of ``paths_canvas`` (canvas coords); style copied from ``like``."""
    cx, cy = paths_center(paths_canvas)
    local = translate_paths(paths_canvas, -cx, -cy)
    ob = canvas.new_shape_object(scene, name, local, location=(cx, cy), layer=layer)
    if z is not None:
        ob.location.z = z
    if like is not None:
        copy_style(like.seam.style, ob.seam.style)
        ob.data.resolution_u = like.data.resolution_u
        ob.data.fill_mode = like.data.fill_mode
    return ob


def _layer_of(scene, obj):
    for col in canvas.layers(scene):
        if obj.name in col.objects:
            return col
    return None


def select_only(context, *objs) -> None:
    """Select exactly ``objs`` (the last one becomes active)."""
    want = [o for o in objs if o is not None]
    names = {o.name for o in want}
    context.view_layer.update()      # new objects get their view-layer base here
    for o in list(context.view_layer.objects):
        if o is None:
            continue
        try:
            o.select_set(o.name in names)
        except RuntimeError:
            pass
    if want:
        context.view_layer.objects.active = want[-1]


# ------------------------------------------------------------------ boolean core

def combine(op: str, ring_sets: Sequence[Sequence[Sequence[Tuple[float, float]]]]):
    """Apply ``op`` to compound ring sets given bottom -> top; returns normalized shapes."""
    if op not in boolean2d.OPS:
        raise ValueError(f"unknown pathfinder operation {op!r}")
    sets = [list(s) for s in ring_sets]
    if len(sets) == 1:
        return boolean2d.union_all(sets, "EVENODD")
    if op == "UNION":
        return boolean2d.union_all(sets, "EVENODD")
    if op == "DIFFERENCE":
        rest = sets[1:]
        b = rest[0] if len(rest) == 1 else shapes_to_rings(boolean2d.union_all(rest, "EVENODD"))
        return boolean2d.boolean(sets[0], b, "DIFFERENCE", "EVENODD")
    acc = sets[0]
    shapes = []
    for nxt in sets[1:]:
        shapes = boolean2d.boolean(acc, nxt, op, "EVENODD")
        acc = shapes_to_rings(shapes)
        if not acc:
            return []
    return shapes


def _collect_rings(operator, objs, resolution):
    """Per-object even-odd ring sets and a list of actionable problems."""
    sets, problems = [], []
    for obj in objs:
        rings = fill_rings(canvas_paths(obj), resolution)
        good = []
        for i, ring in enumerate(rings):
            if abs(geometry.signed_area(ring)) < 1e-9:
                operator.report({"WARNING"}, f"'{obj.name}': path {i + 1} has zero area and was ignored")
                continue
            good.append(ring)
        if not good:
            problems.append(f"'{obj.name}' has no fillable area (needs a path with 3+ points that do not lie on a line)")
            sets.append([])
            continue
        try:
            issues = geometry.validate_rings(good)
        except NotImplementedError:
            issues = []
        for iss in issues:
            if iss.code == "SELF_INTERSECTION":
                operator.report({"WARNING"}, f"'{obj.name}': path crosses itself near "
                                f"({iss.point[0]:.1f}, {iss.point[1]:.1f}); the crossing is resolved by the boolean")
        sets.append(resolve_rings(good, obj.seam.style.fill_rule))
    return sets, problems


# ------------------------------------------------------------------ operators

class SEAM_OT_pathfinder(bpy.types.Operator):
    """Combine the selected shapes with a boolean operation into one compound shape"""
    bl_idname = "seam.pathfinder"
    bl_label = "Pathfinder"
    bl_options = {"REGISTER", "UNDO"}

    op: EnumProperty(name="Operation", items=PATHFINDER_OPS, default="UNION")
    keep_originals: BoolProperty(name="Keep Originals", default=False,
                                 description="Leave the source shapes in place instead of replacing them")
    resolution: IntProperty(name="Resolution", default=12, min=2, max=64,
                            description="Samples per Bezier segment when flattening curves (the result is polygonal)")

    @classmethod
    def poll(cls, context):
        n = sum(1 for o in context.selected_objects if o.seam.role == "SHAPE" and o.type == "CURVE")
        if n < 2:
            cls.poll_message_set("Select at least two vector shapes")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        objs = selected_shapes(context)
        if len(objs) < 2:
            self.report({"ERROR"}, "Select at least two vector shapes (convert text to outlines first)")
            return {"CANCELLED"}
        sets, problems = _collect_rings(self, objs, self.resolution)
        if problems:
            self.report({"ERROR"}, "Pathfinder cannot run: " + "; ".join(problems))
            return {"CANCELLED"}
        label = _OP_LABEL[self.op]
        try:
            shapes = combine(self.op, sets)
        except NotImplementedError as exc:
            self.report({"ERROR"}, f"Pathfinder unavailable: {exc}")
            return {"CANCELLED"}
        except Exception as exc:  # kernel failure: tell the user what to try
            self.report({"ERROR"}, f"{label} failed ({type(exc).__name__}: {exc}). Try a lower Resolution, "
                        "simplify self-crossing paths, or apply Pathfinder to fewer shapes")
            return {"CANCELLED"}
        if not shapes:
            what = {"INTERSECT": "the shapes do not overlap", "DIFFERENCE": "the front shapes cover the bottom shape",
                    "EXCLUDE": "the shapes cancel out", "UNION": "the shapes have no area"}[self.op]
            self.report({"ERROR"}, f"{label} produced no area ({what}); nothing was changed")
            return {"CANCELLED"}
        paths = [polygon_path(r) for r in shapes_to_rings(shapes)]
        bottom, top = objs[0], objs[-1]
        top_z = top.location.z
        layer = _layer_of(scene, top)
        res = make_shape_from_canvas_paths(scene, f"{label} {bottom.name}", paths, like=bottom, layer=layer)
        res.seam.style.fill_rule = "EVENODD"
        if not self.keep_originals:
            res.location.z = top_z
            for o in objs:
                delete_artwork(o)
        select_only(context, res)
        self.report({"INFO"}, f"{label}: {len(shapes)} shape(s), {len(paths)} path(s)")
        return {"FINISHED"}


class SEAM_OT_compound_make(bpy.types.Operator):
    """Merge the selected shapes into one compound path (overlapping parts become holes)"""
    bl_idname = "seam.compound_make"
    bl_label = "Make Compound Path"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        n = sum(1 for o in context.selected_objects if o.seam.role == "SHAPE" and o.type == "CURVE")
        if n < 2:
            cls.poll_message_set("Select at least two vector shapes")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        objs = selected_shapes(context)
        if len(objs) < 2:
            self.report({"ERROR"}, "Select at least two vector shapes")
            return {"CANCELLED"}
        paths = []
        for o in objs:
            paths.extend(canvas_paths(o))
        if not paths:
            self.report({"ERROR"}, "The selected shapes have no paths")
            return {"CANCELLED"}
        top = objs[-1]
        top_z = top.location.z
        res = make_shape_from_canvas_paths(scene, f"Compound {objs[0].name}", paths, like=objs[0],
                                           layer=_layer_of(scene, top))
        res.seam.style.fill_rule = "EVENODD"
        res.location.z = top_z
        for o in objs:
            delete_artwork(o)
        select_only(context, res)
        self.report({"INFO"}, f"Compound path with {len(paths)} sub-paths")
        return {"FINISHED"}


class SEAM_OT_compound_release(bpy.types.Operator):
    """Split the selected compound shapes into one shape per sub-path"""
    bl_idname = "seam.compound_release"
    bl_label = "Release Compound Path"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        for o in context.selected_objects:
            if o.seam.role == "SHAPE" and o.type == "CURVE" and len(o.data.splines) > 1:
                return True
        cls.poll_message_set("Select a vector shape that has more than one sub-path")
        return False

    def execute(self, context):
        scene = context.scene
        made = []
        for src in [o for o in context.selected_objects if o.seam.role == "SHAPE" and o.type == "CURVE"
                    and len(o.data.splines) > 1]:
            paths = curve_paths(src)
            layer = _layer_of(scene, src)
            base = src.name
            z0 = src.location.z
            for i, path in enumerate(paths):
                cu = bpy.data.curves.new(f"{base} {i + 1}", "CURVE")
                cu.dimensions = "2D"
                cu.fill_mode = src.data.fill_mode
                cu.resolution_u = src.data.resolution_u
                write_paths(cu, [path])
                ob = bpy.data.objects.new(cu.name, cu)
                canvas.add_artwork_object(scene, ob, layer=layer, role="SHAPE", on_top=False)
                ob.parent = src.parent
                ob.matrix_parent_inverse = src.matrix_parent_inverse.copy()
                ob.location = src.location
                ob.rotation_euler = src.rotation_euler
                ob.scale = src.scale
                ob.location.z = z0 + i * 1e-4
                copy_style(src.seam.style, ob.seam.style)
                made.append(ob)
            delete_artwork(src)
        if not made:
            self.report({"ERROR"}, "No compound shape selected")
            return {"CANCELLED"}
        canvas.restack(scene)
        select_only(context, *made)
        self.report({"INFO"}, f"Released into {len(made)} shapes")
        return {"FINISHED"}


class SEAM_MT_pathfinder(bpy.types.Menu):
    bl_idname = "SEAM_MT_pathfinder"
    bl_label = "Pathfinder"

    def draw(self, context):
        layout = self.layout
        for key, label, _desc in PATHFINDER_OPS:
            layout.operator(SEAM_OT_pathfinder.bl_idname, text=label).op = key
        layout.separator()
        layout.operator(SEAM_OT_compound_make.bl_idname)
        layout.operator(SEAM_OT_compound_release.bl_idname)


class SEAM_PT_pathfinder(bpy.types.Panel):
    bl_idname = "SEAM_PT_pathfinder"
    bl_label = "Pathfinder"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        for key, label, _desc in PATHFINDER_OPS:
            col.operator(SEAM_OT_pathfinder.bl_idname, text=label).op = key
        layout.separator()
        row = layout.row(align=True)
        row.operator(SEAM_OT_compound_make.bl_idname, text="Make Compound")
        row.operator(SEAM_OT_compound_release.bl_idname, text="Release")


classes = (SEAM_OT_pathfinder, SEAM_OT_compound_make, SEAM_OT_compound_release, SEAM_MT_pathfinder,
           SEAM_PT_pathfinder)
