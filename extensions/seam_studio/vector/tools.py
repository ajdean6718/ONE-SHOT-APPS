"""Drawing tools (owner: vector-a): WorkSpaceTools bound to the modal ``seam.draw_shape``.

Tools: seam.tool_rect, _rrect, _ellipse, _polygon, _star, _line, _artboard (VIEW_3D, OBJECT mode).
Drag on the canvas: Shift = square / 45 degree line, Alt = draw from the centre. The object is
created at the first mouse move and its parameters are updated live (the shape producer rebuilds
the curve); Esc / right click cancels and removes it. ``seam.draw_shape`` also has an ``exec``
path with explicit canvas coordinates (x0, y0, x1, y1) that tests and scripts use. Tool
registration is skipped in background mode. Only tool keymaps are defined (no global keys).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector, geometry as mgeometry

from ..bridge import canvas_matrix
from ..engine import core as engine
from . import canvas, shapes

DRAW_KINDS = [
    ("RECT", "Rectangle", ""), ("ROUNDED_RECT", "Rounded Rectangle", ""), ("ELLIPSE", "Ellipse", ""),
    ("POLYGON", "Polygon", ""), ("STAR", "Star", ""), ("LINE", "Line", ""), ("ARTBOARD", "Artboard", ""),
]
MIN_SIZE = 1.0


def drag_geometry(kind: str, p0: Tuple[float, float], p1: Tuple[float, float], square: bool = False,
                  from_center: bool = False):
    """Resolve a drag into (cx, cy, width, height, rotation) in canvas px.

    Rect-like kinds: Shift forces equal sides, Alt grows from the press point. LINE: width = length,
    rotation = direction, Shift snaps the direction to 45 degree steps, Alt draws from the centre.
    """
    x0, y0 = p0
    x1, y1 = p1
    if kind == "LINE":
        dx, dy = x1 - x0, y1 - y0
        if square:
            ang = round(math.atan2(dy, dx) / (math.pi / 4)) * (math.pi / 4)
            ln = math.hypot(dx, dy)
            dx, dy = ln * math.cos(ang), ln * math.sin(ang)
        if from_center:
            cx, cy = x0, y0
            ln = 2.0 * math.hypot(dx, dy)
        else:
            cx, cy = x0 + dx * 0.5, y0 + dy * 0.5
            ln = math.hypot(dx, dy)
        return cx, cy, ln, 0.0, math.atan2(dy, dx)
    dx, dy = x1 - x0, y1 - y0
    if square:
        m = max(abs(dx), abs(dy))
        dx = math.copysign(m, dx if dx != 0 else 1.0)
        dy = math.copysign(m, dy if dy != 0 else 1.0)
    if from_center:
        return x0, y0, abs(dx) * 2.0, abs(dy) * 2.0, 0.0
    return x0 + dx * 0.5, y0 + dy * 0.5, abs(dx), abs(dy), 0.0


def screen_to_canvas(context, event_coord) -> Optional[Tuple[float, float]]:
    """Mouse position (region px) -> canvas-local (x, y) by intersecting the view ray with the canvas plane."""
    region, rv3d = context.region, context.region_data
    if region is None or rv3d is None:
        return None
    cm = canvas_matrix(context.scene)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, event_coord)
    vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, event_coord)
    plane_co = cm @ Vector((0.0, 0.0, 0.0))
    normal = cm.to_3x3() @ Vector((0.0, 0.0, 1.0))
    hit = mgeometry.intersect_line_plane(origin, origin + vec * 1000.0, plane_co, normal)
    if hit is None:
        hit = view3d_utils.region_2d_to_location_3d(region, rv3d, event_coord, plane_co)
    local = cm.inverted_safe() @ hit
    return float(local.x), float(local.y)


class SEAM_OT_draw_shape(bpy.types.Operator):
    bl_idname = "seam.draw_shape"
    bl_label = "Draw Shape"
    bl_description = "Drag on the canvas to draw a shape (Shift: square / 45 degrees, Alt: from centre)"
    bl_options = {"REGISTER", "UNDO"}

    kind: EnumProperty(name="Kind", items=DRAW_KINDS, default="RECT")
    x0: FloatProperty(name="X0", description="Start point X (canvas px) for the non-interactive path")
    y0: FloatProperty(name="Y0", description="Start point Y (canvas px) for the non-interactive path")
    x1: FloatProperty(name="X1", description="End point X (canvas px) for the non-interactive path")
    y1: FloatProperty(name="Y1", description="End point Y (canvas px) for the non-interactive path")
    square: BoolProperty(name="Square", default=False, description="Equal sides (Shift)")
    from_center: BoolProperty(name="From Center", default=False, description="Start point is the centre (Alt)")
    radius: FloatProperty(name="Corner Radius", min=0.0, default=16.0)
    sides: IntProperty(name="Sides / Points", min=3, max=200, default=5)
    inner_ratio: FloatProperty(name="Inner Ratio", min=0.01, max=1.0, default=0.5)

    _start = None
    _obj = None

    # ---- shared creation / update -------------------------------------------------
    def _apply(self, context, p0, p1, create: bool):
        scene = context.scene
        cx, cy, w, h, rot = drag_geometry(self.kind, p0, p1, self.square, self.from_center)
        w, h = max(w, MIN_SIZE), (max(h, MIN_SIZE) if self.kind != "LINE" else 0.0)
        if self.kind == "ARTBOARD":
            ab = self._obj
            if ab is None:
                ab = canvas.new_artboard(scene, canvas._next_name("Artboard", [a.name for a in canvas.artboards(scene)]),
                                         cx - w / 2, cy + h / 2, w, h)
                self._obj = ab
            ab.location.x, ab.location.y = cx - w / 2, cy + h / 2
            ab.seam.artboard.width, ab.seam.artboard.height = w, h
            return ab
        ob = self._obj
        if ob is None:
            ob = shapes.create_shape(scene, self.kind, w, h, self.radius, self.sides, self.inner_ratio, cx, cy, rot)
            self._obj = ob
            return ob
        sh = ob.seam.shape
        sh.width, sh.height = w, h
        ob.location.x, ob.location.y = cx, cy
        ob.rotation_euler.z = rot
        return ob

    def _finish(self, context, ob):
        scene = context.scene
        engine.flush(scene)
        if ob is not None and ob.seam.role == "ARTBOARD":
            canvas.set_active_artboard(scene, ob)
        if ob is not None:
            shapes.select_only(context, ob)

    # ---- exec path ------------------------------------------------------------------
    def execute(self, context):
        if (self.x0, self.y0) == (self.x1, self.y1):
            self.report({"WARNING"}, "Zero-size drag: nothing drawn")
            return {"CANCELLED"}
        canvas.ensure_canvas(context.scene)
        self._obj = None
        try:
            ob = self._apply(context, (self.x0, self.y0), (self.x1, self.y1), True)
        except NotImplementedError:
            self.report({"ERROR"}, "Shape kernels are not available in this build")
            return {"CANCELLED"}
        self._finish(context, ob)
        self._obj = None
        return {"FINISHED"}

    # ---- modal path -------------------------------------------------------------------
    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D" or context.region_data is None:
            self.report({"ERROR"}, "Draw in a 3D viewport")
            return {"CANCELLED"}
        canvas.ensure_canvas(context.scene)
        press = (event.mouse_prev_press_x - context.region.x, event.mouse_prev_press_y - context.region.y)
        start = screen_to_canvas(context, press)
        if start is None:
            return {"CANCELLED"}
        self._start = start
        self._obj = None
        self.square, self.from_center = event.shift, event.alt
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set("Drag to draw  |  Shift: square  |  Alt: from centre  |  Esc / RMB: cancel")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._cancel(context)
            return {"CANCELLED"}
        if event.type == "MOUSEMOVE":
            cur = screen_to_canvas(context, (event.mouse_region_x, event.mouse_region_y))
            if cur is not None:
                self.square, self.from_center = event.shift, event.alt
                self.x0, self.y0 = self._start
                self.x1, self.y1 = cur
                try:
                    self._apply(context, self._start, cur, False)
                except NotImplementedError:
                    self.report({"ERROR"}, "Shape kernels are not available in this build")
                    self._cancel(context)
                    return {"CANCELLED"}
                engine.flush(context.scene, evaluate=False)
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            context.workspace.status_text_set(None)
            if self._obj is None:
                return {"CANCELLED"}
            ob = self._obj
            self._obj = None
            self._finish(context, ob)
            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def _cancel(self, context):
        context.workspace.status_text_set(None)
        ob, self._obj = self._obj, None
        if ob is None:
            return
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if data is not None and data.users == 0:
            (bpy.data.meshes if isinstance(data, bpy.types.Mesh) else bpy.data.curves).remove(data)


# ------------------------------------------------------------------ workspace tools

def _make_tool(kind: str, label: str, icon: str, extra_props: Tuple[str, ...]):
    idname = {"RECT": "seam.tool_rect", "ROUNDED_RECT": "seam.tool_rrect", "ELLIPSE": "seam.tool_ellipse",
              "POLYGON": "seam.tool_polygon", "STAR": "seam.tool_star", "LINE": "seam.tool_line",
              "ARTBOARD": "seam.tool_artboard"}[kind]

    def draw_settings(context, layout, tool):
        props = tool.operator_properties("seam.draw_shape")
        for p in extra_props:
            layout.prop(props, p)

    attrs = {
        "bl_space_type": "VIEW_3D",
        "bl_context_mode": "OBJECT",
        "bl_idname": idname,
        "bl_label": label,
        "bl_description": f"Draw a {label.lower()} on the canvas (Shift: square, Alt: from centre)",
        "bl_icon": icon,
        "bl_widget": None,
        "bl_keymap": (("seam.draw_shape", {"type": "LEFTMOUSE", "value": "CLICK_DRAG"},
                       {"properties": [("kind", kind)]}),),
        "draw_settings": staticmethod(draw_settings),
    }
    return type("SEAM_TOOL_" + kind.lower(), (bpy.types.WorkSpaceTool,), attrs)


_TOOL_SPECS = (
    ("RECT", "Rectangle", "ops.mesh.primitive_cube_add_gizmo", ()),
    ("ROUNDED_RECT", "Rounded Rectangle", "ops.mesh.primitive_cube_add_gizmo", ("radius",)),
    ("ELLIPSE", "Ellipse", "ops.mesh.primitive_circle_add_gizmo", ()),
    ("POLYGON", "Polygon", "ops.mesh.primitive_cone_add_gizmo", ("sides",)),
    ("STAR", "Star", "ops.mesh.primitive_uv_sphere_add_gizmo", ("sides", "inner_ratio")),
    ("LINE", "Line", "ops.gpencil.primitive_line", ()),
    ("ARTBOARD", "Artboard", "ops.mesh.primitive_grid_add_gizmo", ()),
)
_tools = []


def register():
    if bpy.app.background:
        return  # no window manager / toolbar in background mode
    prev = None
    for kind, label, icon, extra in _TOOL_SPECS:
        tool = _make_tool(kind, label, icon, extra)
        try:
            if prev is None:
                bpy.utils.register_tool(tool, after={"builtin.cursor"}, separator=True, group=True)
            else:
                bpy.utils.register_tool(tool, after={prev.bl_idname}, group=True)
        except Exception as exc:  # a toolbar problem must not stop the add-on
            print("SEAM: tool registration failed for", kind, exc)
            continue
        _tools.append(tool)
        prev = tool


def unregister():
    while _tools:
        tool = _tools.pop()
        try:
            bpy.utils.unregister_tool(tool)
        except Exception:
            pass


classes = (SEAM_OT_draw_shape,)
