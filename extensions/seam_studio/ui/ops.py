"""Shell operators: quick starts, live toggle, start-panel dismissal (owner: shell).

The three quick starts are thin, honest orchestrators over operators owned by the domain packages
(``seam.raster_new``, ``seam.add_shape``, ``seam.make_3d`` ...). :func:`quickstart_status` tells the panel (and the
poll message) why a quick start is unavailable instead of pretending it works.
"""

from __future__ import annotations

import bpy

from . import document, prefs as prefs_mod, util, workspaces

#: operators (with the property names the quick start passes) each quick start needs
REQUIREMENTS = {
    "VECTOR": (),
    "PAINT": (("seam.raster_new", ()),),
    "DRAW3D": (("seam.add_shape", ("kind", "width", "height")), ("seam.make_3d", ("recipe",))),
}
LABELS = {"VECTOR": "New Vector Document", "PAINT": "New Paint Canvas", "DRAW3D": "Draw to 3D"}


def quickstart_status(kind: str, scene=None):
    """``(available, reason)`` for quick start ``kind`` (VECTOR | PAINT | DRAW3D)."""
    scene = scene or bpy.context.scene
    if kind == "VECTOR":
        if not document.is_pristine(scene):
            return False, "This document already has artwork. Use the artboard tools to add another artboard."
        return True, ""
    for idname, props in REQUIREMENTS[kind]:
        ok, why = util.op_available(idname, props)
        if not ok:
            feature = "Paint" if kind == "PAINT" else "Vector or 3D link"
            return False, f"{feature} tools are not installed in this build ({why})."
    return True, ""


def _run(idname: str, **kwargs):
    """Call a registered operator by id (``seam.x``) with the keyword arguments it accepts."""
    fn = util.op_callable(idname)
    have = set(util.op_property_names(idname))
    return fn(**{k: v for k, v in kwargs.items() if k in have})


def _artboard_center(scene):
    from ..vector import canvas
    ab = canvas.active_artboard(scene)
    if ab is None:
        return 960.0, -540.0, 1920.0, 1080.0
    w, h = float(ab.seam.artboard.width), float(ab.seam.artboard.height)
    return ab.location.x + w * 0.5, ab.location.y - h * 0.5, w, h


def _go(context, name: str) -> None:
    """Switch to a SEAM workspace when running in a GUI window and frame its canvas view."""
    win = context.window
    if win is not None and workspaces.activate(win, name):
        bpy.app.timers.register(lambda: (workspaces.frame_canvas_views(), None)[1], first_interval=0.1)


class _QuickStart:
    kind = ""

    @classmethod
    def poll(cls, context):
        ok, why = quickstart_status(cls.kind, context.scene)
        if not ok:
            cls.poll_message_set(why)
        return ok


class SEAM_OT_new_vector_document(_QuickStart, bpy.types.Operator):
    bl_idname = "seam.new_vector_document"
    bl_label = "New Vector Document"
    bl_description = "Create the 2D canvas with an artboard (size and units from the add-on preferences) and open the Vector workspace"
    bl_options = {"REGISTER", "UNDO"}
    kind = "VECTOR"

    def execute(self, context):
        scene = context.scene
        if util.op_available("seam.canvas_new")[0] and not document.has_canvas(scene):
            res = _run("seam.canvas_new")
            if "FINISHED" not in res:
                self.report({"ERROR"}, "Creating the canvas failed (see the Info editor)")
                return {"CANCELLED"}
            document.apply_new_document_defaults(scene, context)
        else:
            document.ensure_document(scene, context)
            document.apply_new_document_defaults(scene, context)
        _go(context, workspaces.VECTOR)
        self.report({"INFO"}, "Vector document ready: draw with the shape tools in the toolbar")
        return {"FINISHED"}


class SEAM_OT_new_paint_canvas(_QuickStart, bpy.types.Operator):
    bl_idname = "seam.new_paint_canvas"
    bl_label = "New Paint Canvas"
    bl_description = "Create a raster painting canvas the size of the active artboard and open the Paint workspace"
    bl_options = {"REGISTER", "UNDO"}
    kind = "PAINT"

    def execute(self, context):
        scene = context.scene
        document.ensure_document(scene, context)
        _cx, _cy, w, h = _artboard_center(scene)
        res = _run("seam.raster_new", width=int(w), height=int(h))
        if "FINISHED" not in res:
            self.report({"ERROR"}, "Creating the paint canvas failed (see the Info editor)")
            return {"CANCELLED"}
        _go(context, workspaces.PAINT)
        if util.op_available("seam.raster_paint_mode")[0] and util.op_callable("seam.raster_paint_mode").poll():
            _run("seam.raster_paint_mode")
        self.report({"INFO"}, "Paint canvas ready: pick a brush and paint")
        return {"FINISHED"}


class SEAM_OT_draw_to_3d(_QuickStart, bpy.types.Operator):
    bl_idname = "seam.draw_to_3d"
    bl_label = "Draw to 3D"
    bl_description = ("Draw a star on the artboard and extrude it into a live 3D result: edit the 2D shape and the "
                      "3D object follows. Opens the Split workspace")
    bl_options = {"REGISTER", "UNDO"}
    kind = "DRAW3D"

    def execute(self, context):
        scene = context.scene
        document.ensure_document(scene, context)
        cx, cy, w, h = _artboard_center(scene)
        size = min(w, h) * 0.5
        res = _run("seam.add_shape", kind="STAR", width=size, height=size, sides=5, inner_ratio=0.45, x=cx, y=cy)
        if "FINISHED" not in res:
            self.report({"ERROR"}, "Adding the star shape failed (see the Info editor)")
            return {"CANCELLED"}
        res = _run("seam.make_3d", recipe="EXTRUDE")
        if "FINISHED" not in res:
            self.report({"ERROR"}, "Creating the live 3D result failed: select a closed shape and try Make 3D")
            return {"CANCELLED"}
        _go(context, workspaces.SPLIT)
        self.report({"INFO"}, "Draw to 3D ready: edit the star's nodes and watch the 3D result update")
        return {"FINISHED"}


class SEAM_OT_start_dismiss(bpy.types.Operator):
    bl_idname = "seam.start_dismiss"
    bl_label = "Hide Start Panel"
    bl_description = "Hide the Start panel (re-enable it in Preferences > Add-ons > SEAM Studio)"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        prefs_mod.get(context).show_start_panel = False
        return {"FINISHED"}


class SEAM_OT_toggle_live(bpy.types.Operator):
    bl_idname = "seam.toggle_live"
    bl_label = "Toggle Live Updates"
    bl_description = "Pause or resume live updates of 3D links and derived geometry"
    bl_options = {"REGISTER"}

    def execute(self, context):
        eng = context.scene.seam.engine
        eng.live_enabled = not eng.live_enabled
        self.report({"INFO"}, "Live updates " + ("resumed" if eng.live_enabled else "paused"))
        return {"FINISHED"}


classes = (SEAM_OT_new_vector_document, SEAM_OT_new_paint_canvas, SEAM_OT_draw_to_3d, SEAM_OT_start_dismiss,
           SEAM_OT_toggle_live)
