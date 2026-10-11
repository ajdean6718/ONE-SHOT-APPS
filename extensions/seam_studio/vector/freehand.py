"""Freehand pencil tool with smoothing + bezier fit (owner: vector-b).

* Tool ``seam.tool_pencil`` (3D viewport, object mode) starts the modal operator ``seam.draw_freehand``
  on left-mouse press. Mouse positions are projected onto the canvas plane (z = 0 of the canvas
  root), recorded while the button is held, then smoothed (``fitting.smooth_polyline``) and fitted
  with cubic Beziers (``fitting.fit_bezier``) on release.  If the end point is within
  ``close_distance`` of the start point the path is closed.  The new shape has a stroke and no fill.
* **Tablet pressure is not supported**: the stroke has the constant width of its style
  (pen pressure is ignored; width cannot vary along a stroke).
* Exec path: ``bpy.ops.seam.draw_freehand(points="x0,y0 x1,y1 ...")`` takes canvas-local points
  (px, Y up) and commits them without any mouse interaction (scripts, tests, other tools).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import bpy
from bpy.props import FloatProperty, StringProperty
from mathutils import Vector

from ..seam_core import fitting
from . import canvas
from .pathfinder import paths_center, select_only, translate_paths

Point = Tuple[float, float]
MIN_SCREEN_STEP = 2.0     # px between recorded mouse samples


# ------------------------------------------------------------------ pure helpers

def parse_points(text: str) -> List[Point]:
    """'x,y x,y ...' (space / semicolon separated) -> list of points. Raises ValueError on bad tokens."""
    pts = []
    for tok in text.replace(";", " ").split():
        parts = tok.split(",")
        if len(parts) != 2:
            raise ValueError(f"bad point '{tok}', expected x,y")
        pts.append((float(parts[0]), float(parts[1])))
    return pts


def format_points(points: Sequence[Point]) -> str:
    return " ".join(f"{x:.4f},{y:.4f}" for x, y in points)


def dedupe(points: Sequence[Point], eps: float = 1e-6) -> List[Point]:
    out: List[Point] = []
    for p in points:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > eps:
            out.append((float(p[0]), float(p[1])))
    return out


def polyline_length(points: Sequence[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def should_close(points: Sequence[Point], close_distance: float) -> bool:
    """True when the stroke ends within ``close_distance`` of its start and is long enough to enclose something."""
    if len(points) < 4 or close_distance <= 0.0:
        return False
    gap = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    return gap <= close_distance and polyline_length(points) > 3.0 * close_distance


def fit_stroke(points: Sequence[Point], smoothing: float, max_error: float, close_distance: float):
    """Points (canvas px) -> (BezierPath, closed). Returns (None, False) when there is nothing to fit."""
    pts = dedupe(points)
    if len(pts) < 2:
        return None, False
    closed = should_close(pts, close_distance)
    if closed:
        pts = pts[:-1]
    if smoothing > 0.0 and len(pts) >= 3:
        pts = fitting.smooth_polyline(pts, smoothing, 2, closed)
    path = fitting.fit_bezier(pts, max_error, closed)
    return path, closed


def commit_stroke(context, points: Sequence[Point], smoothing: float = 0.5, max_error: float = 2.0,
                  close_distance: float = 10.0):
    """Create the freehand shape for canvas-local ``points``. Returns the object or None."""
    scene = context.scene
    path, closed = fit_stroke(points, smoothing, max_error, close_distance)
    if path is None:
        return None
    cx, cy = paths_center([path])
    local = translate_paths([path], -cx, -cy)
    ob = canvas.new_shape_object(scene, "Freehand", local, location=(cx, cy))
    ob.data.fill_mode = "NONE"
    style = ob.seam.style
    style.fill_enabled = False
    style.stroke_enabled = True
    return ob


# ------------------------------------------------------------------ operator

class SEAM_OT_draw_freehand(bpy.types.Operator):
    """Draw a smooth Bezier path by dragging. Pen pressure is not supported: the stroke keeps a constant width"""
    bl_idname = "seam.draw_freehand"
    bl_label = "Draw Freehand"
    bl_options = {"REGISTER", "UNDO"}

    points: StringProperty(name="Points", default="", options={"SKIP_SAVE", "HIDDEN"},
                           description="Explicit canvas points 'x,y x,y ...' (px); skips mouse interaction")
    smoothing: FloatProperty(name="Smoothing", default=0.5, min=0.0, max=1.0,
                             description="Amount of smoothing applied to the raw stroke before fitting")
    error: FloatProperty(name="Fit Error", default=2.0, min=0.05, max=50.0, unit="NONE",
                         description="Maximum distance between the stroke and the fitted curve (px)")
    close_distance: FloatProperty(name="Close Distance", default=12.0, min=0.0, max=200.0, unit="NONE",
                                  description="Close the path when it ends this close to its start "
                                              "(screen px while drawing, canvas px for explicit points)")

    _handle = None

    # -- exec path
    def execute(self, context):
        try:
            pts = parse_points(self.points)
        except ValueError as exc:
            self.report({"ERROR"}, f"Invalid points: {exc}")
            return {"CANCELLED"}
        return self._commit(context, pts, self.close_distance, "ERROR")

    def _commit(self, context, pts, close_distance, level="WARNING"):
        try:
            ob = commit_stroke(context, pts, self.smoothing, self.error, close_distance)
        except NotImplementedError as exc:
            self.report({"ERROR"}, f"Freehand fitting unavailable: {exc}")
            return {"CANCELLED"}
        if ob is None:
            self.report({level}, "Nothing drawn: the stroke needs at least two distinct points")
            return {"CANCELLED"}
        select_only(context, ob)
        return {"FINISHED"}

    # -- modal drawing
    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D" or context.region_data is None:
            self.report({"ERROR"}, "Freehand drawing needs a 3D viewport; use the points argument from scripts")
            return {"CANCELLED"}
        canvas.ensure_canvas(context.scene)
        self._pts: List[Point] = []
        self._world: List[Tuple[float, float, float]] = []
        self._screen: List[Point] = []
        self._scale = 1.0   # canvas px per screen px at the start of the stroke
        if not self._sample(context, event):
            self.report({"WARNING"}, "Cannot draw here: the view is edge-on to the canvas plane")
            return {"CANCELLED"}
        self._scale = self._canvas_per_screen(context, event)
        self._install_overlay(context)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            if self._screen:
                lx, ly = self._screen[-1]
                if math.hypot(event.mouse_region_x - lx, event.mouse_region_y - ly) >= MIN_SCREEN_STEP:
                    self._sample(context, event)
                    context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._sample(context, event)
            self._remove_overlay(context)
            result = self._commit(context, self._pts, self.close_distance * self._scale)
            return result
        if event.type in {"RIGHTMOUSE", "ESC"}:
            self._remove_overlay(context)
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._remove_overlay(context)

    # -- projection
    def _project(self, context, coord) -> Optional[Tuple[Point, Tuple[float, float, float]]]:
        from bpy_extras import view3d_utils
        from mathutils import geometry as mgeo
        region, rv3d = context.region, context.region_data
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        cm = canvas.canvas_root(context.scene).matrix_world
        normal = cm.to_3x3() @ Vector((0.0, 0.0, 1.0))
        hit = mgeo.intersect_line_plane(origin, origin + direction * 1000.0, cm.translation, normal)
        if hit is None:
            hit = mgeo.intersect_line_plane(origin - direction * 1000.0, origin + direction * 1000.0,
                                            cm.translation, normal)
        if hit is None:
            return None
        local = cm.inverted_safe() @ hit
        return (float(local.x), float(local.y)), (hit.x, hit.y, hit.z)

    def _sample(self, context, event) -> bool:
        coord = (event.mouse_region_x, event.mouse_region_y)
        res = self._project(context, coord)
        if res is None:
            return False
        self._pts.append(res[0])
        self._world.append(res[1])
        self._screen.append(coord)
        return True

    def _canvas_per_screen(self, context, event) -> float:
        a = self._project(context, (event.mouse_region_x, event.mouse_region_y))
        b = self._project(context, (event.mouse_region_x + 10.0, event.mouse_region_y))
        if a is None or b is None:
            return 1.0
        d = math.hypot(b[0][0] - a[0][0], b[0][1] - a[0][1]) / 10.0
        return d if d > 0.0 else 1.0

    # -- overlay (live preview of the raw stroke)
    def _install_overlay(self, context):
        if bpy.app.background:
            return
        try:
            import gpu
            from gpu_extras.batch import batch_for_shader
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        except Exception:
            return
        op = self

        def draw():
            if len(op._world) < 2:
                return
            try:
                batch = batch_for_shader(shader, "LINE_STRIP", {"pos": list(op._world)})
                gpu.state.blend_set("ALPHA")
                gpu.state.line_width_set(2.0)
                shader.bind()
                shader.uniform_float("color", (0.1, 0.55, 1.0, 0.9))
                batch.draw(shader)
                gpu.state.line_width_set(1.0)
                gpu.state.blend_set("NONE")
            except Exception:
                pass

        self._handle = bpy.types.SpaceView3D.draw_handler_add(draw, (), "WINDOW", "POST_VIEW")

    def _remove_overlay(self, context):
        if self._handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
            except (ValueError, RuntimeError):
                pass
            self._handle = None
        if context.area is not None:
            context.area.tag_redraw()


# ------------------------------------------------------------------ tool

class SEAM_TOOL_pencil(bpy.types.WorkSpaceTool):
    bl_space_type = "VIEW_3D"
    bl_context_mode = "OBJECT"
    bl_idname = "seam.tool_pencil"
    bl_label = "Pencil"
    bl_description = ("Draw freehand paths on the canvas. The stroke is smoothed and fitted with Bezier curves; "
                      "ending near the start closes the path. Pen pressure is not supported "
                      "(stroke width is constant)")
    bl_icon = "ops.gpencil.draw"
    bl_widget = None
    bl_keymap = (
        ("seam.draw_freehand", {"type": "LEFTMOUSE", "value": "PRESS"}, None),
    )

    def draw_settings(context, layout, tool):
        props = tool.operator_properties("seam.draw_freehand")
        layout.prop(props, "smoothing")
        layout.prop(props, "error")
        layout.prop(props, "close_distance")


classes = (SEAM_OT_draw_freehand,)


def register():
    try:
        bpy.utils.register_tool(SEAM_TOOL_pencil, after={"builtin.cursor"}, separator=False, group=False)
    except Exception as exc:  # tool registration must never block the add-on
        print("SEAM: pencil tool registration failed:", exc)


def unregister():
    try:
        bpy.utils.unregister_tool(SEAM_TOOL_pencil)
    except Exception:
        pass
