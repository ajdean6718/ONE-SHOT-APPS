"""Layer/mask/canvas operators, paint mode entry (owner: paint)."""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty

from .. import compat
from ..seam_core import raster as R
from . import rasterdoc as rd
from . import selection as sel


def _luma(rgb) -> float:
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def paint_color_linear(context):
    """Current native brush colour as scene-linear RGBA (or None when no brush is available)."""
    c = None
    try:
        up = compat.unified_paint_settings(context)
        brush = compat.image_paint_brush(context)
        if up is not None and getattr(up, "use_unified_color", False):
            c = up.color
            subtype = up.bl_rna.properties["color"].subtype
        elif brush is not None:
            c = brush.color
            subtype = brush.bl_rna.properties["color"].subtype
    except Exception:
        c = None
    if c is None:
        return None
    rgb = np.array([c[0], c[1], c[2], 1.0], np.float32)
    if subtype == "COLOR_GAMMA":
        rgb = R.srgb_to_linear(rgb)
    return tuple(float(v) for v in rgb)


def _active(context):
    return rd.active_doc(context.scene)


def _need_doc(cls, context) -> bool:
    if _active(context) is None:
        cls.poll_message_set("No raster document: create one with New Raster Document")
        return False
    return True


def _need_layer_unlocked(cls, context) -> bool:
    if not _need_doc(cls, context):
        return False
    layer = rd.active_layer(_active(context))
    if layer is None:
        cls.poll_message_set("The document has no layer")
        return False
    if layer.locked:
        cls.poll_message_set("The active layer is locked")
        return False
    return True


def _focus_canvas(context, doc):
    """Make the canvas object the only selected + active object."""
    obj = rd.ensure_canvas_object(doc)
    vl = context.view_layer
    for o in list(context.selected_objects):
        o.select_set(False)
    try:
        obj.hide_set(False)
    except RuntimeError:
        pass
    obj.hide_viewport = False
    if obj.name in vl.objects:
        obj.select_set(True)
        vl.objects.active = obj
    return obj


def frame_canvas(context, doc) -> bool:
    """GUI only: top orthographic view of the canvas fitted to the 3D viewport. Returns True when applied."""
    area, region, rv3d = context.area, context.region, context.region_data
    if area is None or area.type != "VIEW_3D" or rv3d is None or region is None or doc.canvas is None:
        return False
    try:
        bpy.ops.view3d.view_axis(type="TOP")
    except RuntimeError:
        return False
    from mathutils import Vector
    w, h = doc.width, doc.height
    centre = doc.canvas.matrix_world @ Vector((w / 2.0, -h / 2.0, 0.0))
    # ortho view: the visible width is (72 / lens) * view_distance; fit the canvas with a 10 % margin
    k = 72.0 / max(1.0, area.spaces.active.lens)
    aspect = max(0.1, region.width / max(1.0, float(region.height)))
    visible_w = max(w, h * aspect) / 0.9
    rv3d.view_location = centre
    rv3d.view_distance = max(1.0, visible_w / k)
    return True


class _LayerOp:
    """Common base for operators acting on a layer of the active document (``index`` -1 = active layer)."""
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty(name="Layer", default=-1, options={"SKIP_SAVE"},
                       description="Layer index (0 = bottom); -1 uses the active layer")

    def _index(self, doc) -> int:
        return doc.active_layer_index if self.index < 0 else self.index


class SEAM_OT_raster_new(bpy.types.Operator):
    bl_idname = "seam.raster_new"
    bl_label = "New Raster Document"
    bl_description = "Create a raster canvas (plane + layers) at the top-left of the active artboard"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty(name="Name", default="Raster")
    width: IntProperty(name="Width", min=1, max=16384, default=1024, subtype="PIXEL")
    height: IntProperty(name="Height", min=1, max=16384, default=1024, subtype="PIXEL")
    background: EnumProperty(name="Background", items=[("TRANSPARENT", "Transparent", ""), ("WHITE", "White", ""),
                                                       ("COLOR", "Color", "")], default="TRANSPARENT")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        scene = context.scene
        doc = rd.new_raster_doc(scene, self.name, self.width, self.height, self.background)
        _focus_canvas(context, doc)
        self.report({"INFO"}, f"Created raster document {doc.name} ({doc.width}x{doc.height})")
        return {"FINISHED"}


class SEAM_OT_raster_remove(bpy.types.Operator):
    bl_idname = "seam.raster_remove"
    bl_label = "Delete Raster Document"
    bl_description = "Delete the active raster document with all its layers"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _need_doc(cls, context)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        doc = _active(context)
        rd.remove_doc(context.scene, doc)
        return {"FINISHED"}


class SEAM_OT_raster_layer_add(bpy.types.Operator):
    bl_idname = "seam.raster_layer_add"
    bl_label = "Add Layer"
    bl_description = "Add a transparent layer above the active layer"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty(name="Name", default="")

    @classmethod
    def poll(cls, context):
        return _need_doc(cls, context)

    def execute(self, context):
        rd.add_layer(_active(context), self.name or None)
        return {"FINISHED"}


class SEAM_OT_raster_layer_remove(_LayerOp, bpy.types.Operator):
    bl_idname = "seam.raster_layer_remove"
    bl_label = "Delete Layer"
    bl_description = "Delete the layer with its mask"

    @classmethod
    def poll(cls, context):
        if not _need_doc(cls, context):
            return False
        if len(_active(context).layers) < 2:
            cls.poll_message_set("A raster document needs at least one layer")
            return False
        return True

    def execute(self, context):
        doc = _active(context)
        i = self._index(doc)
        if not 0 <= i < len(doc.layers):
            return {"CANCELLED"}
        rd.remove_layer(doc, i)
        return {"FINISHED"}


class SEAM_OT_raster_layer_duplicate(_LayerOp, bpy.types.Operator):
    bl_idname = "seam.raster_layer_duplicate"
    bl_label = "Duplicate Layer"
    bl_description = "Duplicate the layer with its mask and properties"

    @classmethod
    def poll(cls, context):
        return _need_doc(cls, context)

    def execute(self, context):
        doc = _active(context)
        i = self._index(doc)
        if not 0 <= i < len(doc.layers):
            return {"CANCELLED"}
        rd.duplicate_layer(doc, i)
        return {"FINISHED"}


class SEAM_OT_raster_layer_move(_LayerOp, bpy.types.Operator):
    bl_idname = "seam.raster_layer_move"
    bl_label = "Move Layer"
    bl_description = "Move the layer up or down in the stack"

    direction: EnumProperty(name="Direction", items=[("UP", "Up", "Towards the top"), ("DOWN", "Down", "Towards the bottom")],
                            default="UP")

    @classmethod
    def poll(cls, context):
        return _need_doc(cls, context)

    def execute(self, context):
        doc = _active(context)
        i = self._index(doc)
        j = i + (1 if self.direction == "UP" else -1)
        if not (0 <= i < len(doc.layers) and 0 <= j < len(doc.layers)):
            self.report({"INFO"}, "Layer is already at the " + ("top" if self.direction == "UP" else "bottom"))
            return {"CANCELLED"}
        rd.move_layer(doc, i, j)
        return {"FINISHED"}


class SEAM_OT_raster_layer_merge_down(_LayerOp, bpy.types.Operator):
    bl_idname = "seam.raster_layer_merge_down"
    bl_label = "Merge Down"
    bl_description = "Merge the layer into the layer below it"

    @classmethod
    def poll(cls, context):
        if not _need_doc(cls, context):
            return False
        doc = _active(context)
        i = doc.active_layer_index
        if i < 1:
            cls.poll_message_set("There is no layer below to merge into")
            return False
        if doc.layers[i - 1].locked or doc.layers[i].locked:
            cls.poll_message_set("Locked layers cannot be merged")
            return False
        return True

    def execute(self, context):
        doc = _active(context)
        i = self._index(doc)
        if not 1 <= i < len(doc.layers):
            return {"CANCELLED"}
        if doc.layers[i - 1].locked or doc.layers[i].locked:
            self.report({"WARNING"}, "Locked layers cannot be merged")
            return {"CANCELLED"}
        rd.merge_down(doc, i)
        return {"FINISHED"}


class SEAM_OT_raster_mask_add(_LayerOp, bpy.types.Operator):
    bl_idname = "seam.raster_mask_add"
    bl_label = "Add Mask"
    bl_description = "Add a layer mask (white reveals, black hides)"

    fill: EnumProperty(name="Fill", items=[("WHITE", "Reveal All", "White mask: everything visible"),
                                           ("BLACK", "Hide All", "Black mask: everything hidden"),
                                           ("SELECTION", "From Selection", "White inside the current selection")],
                       default="WHITE")

    @classmethod
    def poll(cls, context):
        if not _need_doc(cls, context):
            return False
        layer = rd.active_layer(_active(context))
        if layer is None or layer.mask is not None:
            cls.poll_message_set("The active layer already has a mask")
            return False
        return True

    def execute(self, context):
        doc = _active(context)
        i = self._index(doc)
        if not 0 <= i < len(doc.layers) or doc.layers[i].mask is not None:
            return {"CANCELLED"}
        if self.fill == "SELECTION" and not doc.has_selection:
            self.report({"WARNING"}, "There is no selection; created a white mask")
        rd.add_mask(doc, i, self.fill)
        return {"FINISHED"}


class SEAM_OT_raster_mask_apply(_LayerOp, bpy.types.Operator):
    bl_idname = "seam.raster_mask_apply"
    bl_label = "Apply Mask"
    bl_description = "Bake the mask into the layer transparency and delete the mask"

    @classmethod
    def poll(cls, context):
        if not _need_layer_unlocked(cls, context):
            return False
        layer = rd.active_layer(_active(context))
        if layer.mask is None:
            cls.poll_message_set("The active layer has no mask")
            return False
        return True

    def execute(self, context):
        doc = _active(context)
        i = self._index(doc)
        if not 0 <= i < len(doc.layers) or doc.layers[i].mask is None:
            return {"CANCELLED"}
        rd.apply_mask(doc, i)
        return {"FINISHED"}


class SEAM_OT_raster_mask_remove(_LayerOp, bpy.types.Operator):
    bl_idname = "seam.raster_mask_remove"
    bl_label = "Delete Mask"
    bl_description = "Delete the layer mask without applying it"

    @classmethod
    def poll(cls, context):
        if not _need_doc(cls, context):
            return False
        layer = rd.active_layer(_active(context))
        if layer is None or layer.mask is None:
            cls.poll_message_set("The active layer has no mask")
            return False
        return True

    def execute(self, context):
        doc = _active(context)
        i = self._index(doc)
        if not 0 <= i < len(doc.layers) or doc.layers[i].mask is None:
            return {"CANCELLED"}
        rd.remove_mask(doc, i)
        return {"FINISHED"}


class SEAM_OT_raster_set_paint_target(_LayerOp, bpy.types.Operator):
    bl_idname = "seam.raster_set_paint_target"
    bl_label = "Set Paint Target"
    bl_description = "Choose whether native brushes paint the layer pixels or the layer mask"

    target: EnumProperty(name="Paint On", items=[("LAYER", "Layer", "Paint layer pixels"),
                                                 ("MASK", "Mask", "Paint the layer mask")], default="LAYER")

    @classmethod
    def poll(cls, context):
        return _need_doc(cls, context)

    def execute(self, context):
        doc = _active(context)
        i = self._index(doc)
        if not 0 <= i < len(doc.layers):
            return {"CANCELLED"}
        if self.target == "MASK" and doc.layers[i].mask is None:
            self.report({"WARNING"}, "This layer has no mask: add one first")
            return {"CANCELLED"}
        if doc.layers[i].locked:
            self.report({"WARNING"}, "Layer is locked: strokes are discarded until it is unlocked")
        rd.set_paint_target(doc, i, self.target)
        return {"FINISHED"}


class SEAM_OT_raster_paint_mode(bpy.types.Operator):
    bl_idname = "seam.raster_paint_mode"
    bl_label = "Paint on Canvas"
    bl_description = ("Select the raster canvas, enter Texture Paint mode (Material mode, active layer as paint slot)"
                      " and, in the 3D viewport, look at it from the top")
    bl_options = {"REGISTER", "UNDO"}

    enter: BoolProperty(name="Enter", default=True, description="False leaves Texture Paint mode")
    frame_view: BoolProperty(name="Frame View", default=True,
                             description="Switch the 3D viewport to a top orthographic view of the canvas")

    @classmethod
    def poll(cls, context):
        return _need_doc(cls, context)

    def execute(self, context):
        from . import selection
        doc = _active(context)
        obj = rd.ensure_canvas_object(doc)
        if context.object is not None and context.object.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
        if not self.enter:
            if obj.mode == "TEXTURE_PAINT":
                _focus_canvas(context, doc)
                bpy.ops.object.mode_set(mode="OBJECT")
            rd.show_composite(doc)
            return {"FINISHED"}
        obj = _focus_canvas(context, doc)
        if obj.name not in context.view_layer.objects:
            self.report({"ERROR"}, "The canvas object is not in the view layer")
            return {"CANCELLED"}
        if doc.material is None or doc.material.get(rd.MAT_SIG_KEY) != rd.material_signature(doc):
            rd.build_material(doc)
        context.tool_settings.image_paint.mode = "MATERIAL"
        try:
            bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Cannot enter Texture Paint mode: {exc}")
            return {"CANCELLED"}
        rd.sync_paint_target(doc)  # slots exist now: select the target slot
        rd.check_external_edits(doc)  # record the pixel baseline for stroke detection
        selection.sync_stencil(doc)
        layer = rd.active_layer(doc)
        if layer is not None and layer.locked:
            self.report({"WARNING"}, "The active layer is locked: strokes are discarded until it is unlocked")
        rd.start_paint_watch(doc)
        if self.frame_view:
            frame_canvas(context, doc)
        return {"FINISHED"}


class SEAM_OT_raster_fill(bpy.types.Operator):
    bl_idname = "seam.raster_fill"
    bl_label = "Fill"
    bl_description = "Fill the selection (or the whole layer when nothing is selected) with a colour"
    bl_options = {"REGISTER", "UNDO"}

    color: FloatVectorProperty(name="Color", subtype="COLOR", size=4, min=0.0, max=1.0, default=(0.0, 0.0, 0.0, 1.0),
                               description="Fill colour (scene linear, straight alpha). For masks the brightness is used")
    opacity: FloatProperty(name="Opacity", min=0.0, max=1.0, default=1.0, subtype="FACTOR")

    @classmethod
    def poll(cls, context):
        return _need_layer_unlocked(cls, context)

    def invoke(self, context, event):
        c = paint_color_linear(context)
        if c is not None:
            self.color = c
        return self.execute(context)

    def execute(self, context):
        doc = _active(context)
        layer = rd.active_layer(doc)
        mask = sel.effective_mask(doc)
        if doc.edit_target == "MASK" and layer.mask is not None:
            value = float(R.linear_to_srgb(np.array(self.color[:3], np.float32)).mean())
            cur = rd.mask_array(layer)
            f = mask * self.opacity
            rd.write_mask_array(layer, cur * (1.0 - f) + value * f)
        else:
            arr = R.fill(rd.layer_array(layer), mask, tuple(self.color), self.opacity)
            rd.write_layer_array(layer, arr)
        return {"FINISHED"}


class SEAM_OT_raster_clear(bpy.types.Operator):
    bl_idname = "seam.raster_clear"
    bl_label = "Clear"
    bl_description = "Erase the selection (or the whole layer when nothing is selected)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _need_layer_unlocked(cls, context)

    def execute(self, context):
        doc = _active(context)
        layer = rd.active_layer(doc)
        mask = sel.effective_mask(doc)
        if doc.edit_target == "MASK" and layer.mask is not None:
            rd.write_mask_array(layer, rd.mask_array(layer) * (1.0 - mask))
        else:
            rd.write_layer_array(layer, R.clear_selected(rd.layer_array(layer), mask))
        return {"FINISHED"}


class SEAM_OT_raster_resize_canvas(bpy.types.Operator):
    bl_idname = "seam.raster_resize_canvas"
    bl_label = "Resize Canvas"
    bl_description = "Change the pixel size of the document (crop/extend around an anchor, or scale the content)"
    bl_options = {"REGISTER", "UNDO"}

    width: IntProperty(name="Width", min=1, max=16384, default=1024, subtype="PIXEL")
    height: IntProperty(name="Height", min=1, max=16384, default=1024, subtype="PIXEL")
    anchor: EnumProperty(name="Anchor", items=[(a, a.replace("_", " ").title(), "") for a in rd.ANCHORS],
                         default="TOP_LEFT")
    scale_content: BoolProperty(name="Scale Content", default=False,
                                description="Resample the pixels instead of cropping/extending the canvas")

    @classmethod
    def poll(cls, context):
        return _need_doc(cls, context)

    def invoke(self, context, event):
        doc = _active(context)
        self.width, self.height = doc.width, doc.height
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        doc = _active(context)
        if doc.canvas is not None and doc.canvas.mode != "OBJECT":
            self.report({"WARNING"}, "Leave Texture Paint mode before resizing the canvas")
            return {"CANCELLED"}
        rd.resize_canvas(doc, self.width, self.height, self.anchor, self.scale_content)
        return {"FINISHED"}


classes = (SEAM_OT_raster_new, SEAM_OT_raster_remove, SEAM_OT_raster_layer_add, SEAM_OT_raster_layer_remove,
           SEAM_OT_raster_layer_duplicate, SEAM_OT_raster_layer_move, SEAM_OT_raster_layer_merge_down,
           SEAM_OT_raster_mask_add, SEAM_OT_raster_mask_apply, SEAM_OT_raster_mask_remove,
           SEAM_OT_raster_set_paint_target, SEAM_OT_raster_paint_mode, SEAM_OT_raster_fill, SEAM_OT_raster_clear,
           SEAM_OT_raster_resize_canvas)
