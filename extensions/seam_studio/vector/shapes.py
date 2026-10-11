"""Primitive creation operators + parametric shape producer (owner: vector-a).

* ``create_shape`` builds a SHAPE curve from ``seam_core.shapes.from_kind`` centred on its
  object origin and records the recipe in ``obj.seam.shape``.
* ``ShapeProducer`` ("shape", priority 10): while ``shape.kind != NONE`` the recipe owns the
  geometry and the curve is regenerated when a parameter changes; when the curve was edited by
  hand (curve signature differs from the one stored after the last generation) the shape is
  *expanded*: ``kind`` becomes NONE and the nodes own the geometry from then on.
  The generation signature (``recipe:curve``) lives on the **Curve datablock** it describes, not on
  the object: memfile undo can keep (reuse) a Curve that the engine regenerated after an undo while it
  re-reads the Object with an older recipe. Geometry and its signature therefore always come back
  together, and a recipe that differs from the recorded one means "regenerate", never "hand edit".
* Operators: ``seam.add_shape``, ``seam.shape_expand``.
"""

from __future__ import annotations

from typing import Optional

import bpy
from bpy.props import EnumProperty, FloatProperty, IntProperty

from ..bridge import curve_paths, paths_signature, write_paths
from ..engine import core as engine
from ..seam_core import shapes as core_shapes
from . import canvas
from .props import SHAPE_KINDS

KIND_LABELS = {k: lbl for k, lbl, _d in SHAPE_KINDS}
_ADD_KINDS = [k for k in SHAPE_KINDS if k[0] != "NONE"]


def recipe_signature(shape) -> str:
    return "|".join((shape.kind, f"{shape.width:.5f}", f"{shape.height:.5f}", f"{shape.radius:.5f}",
                     str(int(shape.sides)), f"{shape.inner_ratio:.5f}"))


def curve_signature(obj) -> str:
    return paths_signature(curve_paths(obj))


def mark_generated(obj) -> None:
    """Record (on the curve datablock) that the curve currently equals what the recipe generates."""
    canvas.set_sig(obj.data, "shape", recipe_signature(obj.seam.shape) + ":" + curve_signature(obj))
    canvas.set_sig(obj, "shape", None)  # legacy location (files written before the signature moved to the curve)


def generated_signature(obj) -> Optional[str]:
    """``recipe:curve`` recorded when the curve was last generated (curve datablock; legacy: the object)."""
    sig = canvas.get_sig(obj.data, "shape") if obj.data is not None else None
    return sig if sig is not None else canvas.get_sig(obj, "shape")


def regenerate(obj) -> None:
    sh = obj.seam.shape
    path = core_shapes.from_kind(sh.kind, sh.width, sh.height, sh.radius, int(sh.sides), sh.inner_ratio, 0.0, 0.0)
    write_paths(obj.data, [path])


def default_name(kind: str) -> str:
    return KIND_LABELS.get(kind, "Shape")


def create_shape(scene, kind: str, width: float, height: float, radius: float = 0.0, sides: int = 5,
                 inner_ratio: float = 0.5, x: float = 0.0, y: float = 0.0, rotation: float = 0.0,
                 layer=None, name: Optional[str] = None):
    """Create a parametric SHAPE object centred at canvas-local (x, y). Returns the object."""
    scene = scene or bpy.context.scene
    path = core_shapes.from_kind(kind, width, height, radius, int(sides), inner_ratio, 0.0, 0.0)
    ob = canvas.new_shape_object(scene, name or default_name(kind), [path], location=(x, y), layer=layer)
    sh = ob.seam.shape
    sh.kind = kind
    sh.width = width
    sh.height = height
    sh.radius = radius
    sh.sides = int(sides)
    sh.inner_ratio = inner_ratio
    if rotation:
        ob.rotation_euler.z = rotation
    st = ob.seam.style
    if kind == "LINE":
        st.fill_enabled = False
        st.stroke_enabled = True
        st.stroke_color = (0.0, 0.0, 0.0, 1.0)
        st.stroke_width = 2.0
    _init_gradient_stops(st)
    mark_generated(ob)
    return ob


def _init_gradient_stops(style) -> None:
    if len(style.gradient_stops) == 0:
        a = style.gradient_stops.add()
        a.position = 0.0
        a.color = tuple(style.fill_color)
        b = style.gradient_stops.add()
        b.position = 1.0
        b.color = (1.0, 1.0, 1.0, 1.0)


def select_only(context, ob) -> None:
    canvas.select_only(context, [ob], ob)


def artboard_center(scene):
    ab = canvas.active_artboard(scene)
    if ab is None:
        return 0.0, 0.0
    minx, miny, maxx, maxy = canvas.artboard_rect(ab)
    return (minx + maxx) * 0.5, (miny + maxy) * 0.5


class ShapeProducer(engine.Producer):
    kind = "shape"
    priority = 10

    def keys_for_update(self, scene, id_orig, geometry, transform):
        if isinstance(id_orig, bpy.types.Object):
            if id_orig.seam.role == "SHAPE" and id_orig.seam.id and id_orig.seam.shape.kind != "NONE":
                yield "shape:" + id_orig.seam.id
        elif isinstance(id_orig, bpy.types.Curve):
            for ob in bpy.data.objects:
                if ob.data is id_orig and ob.seam.role == "SHAPE" and ob.seam.shape.kind != "NONE" and ob.seam.id:
                    yield "shape:" + ob.seam.id

    def all_keys(self, scene):
        for ob in canvas.artwork_objects(scene, ("SHAPE",)):
            if ob.seam.shape.kind != "NONE" and ob.seam.id:
                yield "shape:" + ob.seam.id

    def _obj(self, scene, key):
        ob = canvas.find_by_id(key.split(":", 1)[1], scene)
        if ob is None or ob.seam.role != "SHAPE" or ob.type != "CURVE" or ob.seam.shape.kind == "NONE":
            return None
        return ob

    def signature(self, scene, key):
        ob = self._obj(scene, key)
        if ob is None:
            return None
        return recipe_signature(ob.seam.shape) + ":" + curve_signature(ob)

    def stored_signature(self, scene, key):
        ob = self._obj(scene, key)
        return None if ob is None else generated_signature(ob)

    def build(self, scene, key, signature):
        ob = self._obj(scene, key)
        if ob is None:
            return engine.BuildResult(key, "obsolete")
        if ob.mode == "EDIT":
            # the user is editing nodes right now: leave the curve alone; judged again on leaving edit mode
            return engine.BuildResult(key, "skipped", "shape is in edit mode")
        stored = generated_signature(ob)
        cur_curve = signature.split(":", 1)[1]
        # hand edit = the curve no longer equals what was generated into *this* curve datablock; a recipe that
        # differs from the recorded one (parameter edit, or an undo that restored an older recipe) regenerates
        if stored is not None and stored.split(":", 1)[1] != cur_curve:
            expand(ob)
            return engine.BuildResult(key, "ok", "shape edited by hand: parameters released (expanded)")
        regenerate(ob)
        mark_generated(ob)
        return engine.BuildResult(key, "ok")


def expand(ob) -> None:
    """Release the recipe: the curve nodes own the geometry from now on."""
    ob.seam.shape.kind = "NONE"
    canvas.set_sig(ob, "shape", None)
    if ob.data is not None:
        canvas.set_sig(ob.data, "shape", None)


class SEAM_OT_add_shape(bpy.types.Operator):
    bl_idname = "seam.add_shape"
    bl_label = "Add Shape"
    bl_description = "Add a parametric vector shape to the active layer"
    bl_options = {"REGISTER", "UNDO"}

    kind: EnumProperty(name="Kind", items=_ADD_KINDS, default="RECT")
    width: FloatProperty(name="Width", min=0.0, default=200.0)
    height: FloatProperty(name="Height", min=0.0, default=120.0)
    radius: FloatProperty(name="Corner Radius", min=0.0, default=16.0)
    sides: IntProperty(name="Sides / Points", min=3, max=200, default=5)
    inner_ratio: FloatProperty(name="Inner Ratio", min=0.01, max=1.0, default=0.5)
    x: FloatProperty(name="X", description="Canvas-local X of the shape centre (default: active artboard centre)")
    y: FloatProperty(name="Y", description="Canvas-local Y of the shape centre (default: active artboard centre)")

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        scene = context.scene
        canvas.ensure_canvas(scene)
        x, y = self.x, self.y
        if not (self.properties.is_property_set("x") or self.properties.is_property_set("y")):
            x, y = artboard_center(scene)
        elif not self.properties.is_property_set("x"):
            x = artboard_center(scene)[0]
        elif not self.properties.is_property_set("y"):
            y = artboard_center(scene)[1]
        try:
            ob = create_shape(scene, self.kind, self.width, self.height, self.radius, self.sides,
                              self.inner_ratio, x, y)
        except NotImplementedError:
            self.report({"ERROR"}, "Shape kernels are not available in this build")
            return {"CANCELLED"}
        select_only(context, ob)
        engine.flush(scene, evaluate=True)
        return {"FINISHED"}


class SEAM_OT_shape_expand(bpy.types.Operator):
    bl_idname = "seam.shape_expand"
    bl_label = "Expand Shape"
    bl_description = "Turn the parametric shape into a free path (parameters stop controlling the geometry)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obs = [o for o in context.selected_objects if o.seam.role == "SHAPE" and o.seam.shape.kind != "NONE"]
        if not obs:
            cls.poll_message_set("Select a parametric shape")
            return False
        return True

    def execute(self, context):
        n = 0
        for ob in context.selected_objects:
            if ob.seam.role == "SHAPE" and ob.seam.shape.kind != "NONE":
                expand(ob)
                n += 1
        self.report({"INFO"}, f"Expanded {n} shape(s)")
        return {"FINISHED"}


class SEAM_PT_shape(bpy.types.Panel):
    bl_label = "Shape"
    bl_idname = "SEAM_PT_shape"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"
    bl_order = 20

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.seam.role == "SHAPE"

    def draw(self, context):
        layout = self.layout
        ob = context.active_object
        sh = ob.seam.shape
        if sh.kind == "NONE":
            layout.label(text="Free path: nodes own the geometry", icon="CURVE_BEZCURVE")
            return
        col = layout.column(align=True)
        col.prop(sh, "kind", text="Kind")
        if sh.kind != "LINE":
            col.prop(sh, "width")
            col.prop(sh, "height")
        else:
            col.prop(sh, "width", text="Length")
        if sh.kind == "ROUNDED_RECT":
            col.prop(sh, "radius")
        if sh.kind in ("POLYGON", "STAR"):
            col.prop(sh, "sides", text="Points" if sh.kind == "STAR" else "Sides")
        if sh.kind == "STAR":
            col.prop(sh, "inner_ratio")
        layout.operator("seam.shape_expand", icon="MOD_CURVE")


class SEAM_PT_create(bpy.types.Panel):
    bl_label = "Shapes"
    bl_idname = "SEAM_PT_create"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"
    bl_order = 12

    @classmethod
    def poll(cls, context):
        return bool(context.scene.seam.doc.canvas_ready)

    def draw(self, context):
        col = self.layout.column(align=True)
        icons = {"RECT": "MESH_PLANE", "ROUNDED_RECT": "MESH_PLANE", "ELLIPSE": "MESH_CIRCLE",
                 "POLYGON": "MESH_CONE", "STAR": "SOLO_ON", "LINE": "IPO_LINEAR"}
        for kind, label, _d in _ADD_KINDS:
            op = col.operator("seam.add_shape", text=label, icon=icons.get(kind, "MESH_PLANE"))
            op.kind = kind


classes = (SEAM_OT_add_shape, SEAM_OT_shape_expand, SEAM_PT_shape, SEAM_PT_create)

_producer = ShapeProducer()


def register():
    engine.register_producer(_producer)


def unregister():
    engine.unregister_producer(ShapeProducer.kind)
