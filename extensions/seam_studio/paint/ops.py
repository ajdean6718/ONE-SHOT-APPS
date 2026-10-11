"""Layer/mask/canvas operators, paint mode entry (owner: paint)."""

import contextlib
import functools

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


def _enter_paint_mode(context, doc_id: str) -> None:
    """Re-enter Texture Paint mode on the canvas of document ``doc_id`` (no undo step is recorded)."""
    cur = None
    for sc in bpy.data.scenes:
        cur = rd.get_doc(sc, doc_id)
        if cur is not None:
            break
    if cur is None or cur.canvas is None or cur.canvas.mode == "TEXTURE_PAINT":
        return
    try:
        context.tool_settings.image_paint.mode = "MATERIAL"
        bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
        rd.sync_paint_target(cur)
        rd.check_external_edits(cur)
        sel.sync_stencil(cur)
        rd.start_paint_watch(cur)
    except RuntimeError:
        pass


def _reenter_paint_later(doc_id: str) -> None:
    """Re-enter paint mode after the operator returned.  Blender pushes the operator's own UNDO step right after
    ``execute`` returns, and it records scene data (a memfile step) only while the object is NOT in a paint mode;
    entering paint mode first would make it an empty image-type step.  So in the GUI the canvas stays in Object mode
    until the next timer tick (the step is pushed by then); background sessions have no timers and re-enter now."""
    if bpy.app.background:
        _enter_paint_mode(bpy.context, doc_id)
        return

    def reenter():
        try:
            override, busy = _gui_override()
            if busy:
                return 0.05
            if override is not None:
                with bpy.context.temp_override(**override):
                    _enter_paint_mode(bpy.context, doc_id)
            else:
                _enter_paint_mode(bpy.context, doc_id)
        except Exception:
            import traceback
            traceback.print_exc()
        return None

    bpy.app.timers.register(reenter, first_interval=0.0)


def _undoable_in_paint_mode(execute):
    """Decorator for ``execute`` methods that edit pixels or layers.

    Blender records no memfile undo step for anything done while an object is in Texture Paint mode, so the edit
    could not be undone.  The canvas leaves paint mode around the edit and re-enters it afterwards
    (:func:`_reenter_paint_later`); the operator's own UNDO push then is the one and only step.  The edited images
    are packed first, so that step (and the auto save written from it) holds the pixels."""
    @functools.wraps(execute)
    def wrapper(self, context):
        doc = _active(context)
        if doc is None:
            return execute(self, context)
        scene, doc_id = context.scene, doc.id
        obj = getattr(doc, "canvas", None)
        in_paint = obj is not None and obj.mode == "TEXTURE_PAINT"
        if in_paint:
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            result = execute(self, context)
            if "FINISHED" in result:
                cur = rd.get_doc(scene, doc_id)
                if cur is not None:
                    rd.pack_layer_images(cur)
        except BaseException:
            if in_paint:
                _enter_paint_mode(context, doc_id)
            raise
        if in_paint:
            _reenter_paint_later(doc_id)
        return result
    return wrapper


@contextlib.contextmanager
def _object_mode_with_undo_step(context, doc, label: str):
    """Push ONE memfile undo step named ``label`` for a change that is not made by an operator (see
    :func:`flush_property_undo`): leave paint mode, pack, push, re-enter."""
    scene, doc_id = context.scene, doc.id
    bpy.ops.object.mode_set(mode="OBJECT")
    try:
        yield
        cur = rd.get_doc(scene, doc_id)
        if cur is not None:
            rd.pack_layer_images(cur)
        bpy.ops.ed.undo_push(message=label)
    finally:
        _enter_paint_mode(context, doc_id)


# --------------------------------------------------------------------------- property edits made in paint mode
#
# Layer opacity / blend / visibility / lock / mask switch and the document background are plain properties edited
# through UI widgets.  In Texture Paint mode the widget's undo push is an image-type step that holds no scene data,
# and undoing back to the previous memfile step does not reload it (nothing newer exists), so the edit could never
# be undone.  The update callbacks (props.py) report the edit here; once the widget has settled a single undo step
# is pushed from Object mode (same dance as the operators).  A timer does it in the GUI; background sessions
# (tests, scripts) call :func:`flush_property_undo`.

PROP_UNDO_DELAY = 0.6
_prop_pending: dict = {}   # doc id -> time.monotonic() of the last edit


def note_property_edit(owner) -> None:
    """Called by the property update callbacks of layers and documents."""
    if hasattr(owner, "layers"):
        doc = owner
    else:
        _sc, doc = rd.doc_of_layer(owner)
    if doc is None or not rd.paint_mode_active(doc):
        return
    import time
    _prop_pending[doc.id] = time.monotonic()
    if not bpy.app.background and not bpy.app.timers.is_registered(_property_undo_tick):
        bpy.app.timers.register(_property_undo_tick, first_interval=PROP_UNDO_DELAY)


def _gui_override():
    wm = bpy.context.window_manager
    for win in wm.windows:
        if getattr(win, "modal_operators", None):
            return None, True  # a modal operator (a brush stroke) is running: not now
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                if region is not None:
                    return dict(window=win, area=area, region=region), False
    return None, False


def flush_property_undo(force: bool = True) -> int:
    """Push the pending single undo step for property edits made in paint mode. Returns steps pushed."""
    import time
    n = 0
    now = time.monotonic()
    for doc_id, t in list(_prop_pending.items()):
        if not force and now - t < PROP_UNDO_DELAY:
            continue
        del _prop_pending[doc_id]
        doc = None
        for sc in bpy.data.scenes:
            doc = rd.get_doc(sc, doc_id)
            if doc is not None:
                break
        if doc is None or not rd.paint_mode_active(doc):
            continue  # left paint mode meanwhile: the mode switch pushed a memfile step that holds the edit
        override = None
        if not bpy.app.background:
            override, busy = _gui_override()
            if busy:
                _prop_pending[doc_id] = t
                continue
        ctx = bpy.context
        if override is not None:
            with ctx.temp_override(**override):
                with _object_mode_with_undo_step(bpy.context, doc, "Raster Layer Settings"):
                    pass
        else:
            with _object_mode_with_undo_step(ctx, doc, "Raster Layer Settings"):
                pass
        n += 1
    return n


def _property_undo_tick():
    try:
        flush_property_undo(force=False)
    except Exception:
        import traceback
        traceback.print_exc()
        _prop_pending.clear()
    return PROP_UNDO_DELAY / 2.0 if _prop_pending else None


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


class SEAM_OT_raster_activate(bpy.types.Operator):
    bl_idname = "seam.raster_activate"
    bl_label = "Activate Raster Document"
    bl_description = "Make this raster document the active one and select its canvas"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(name="Document", min=0, default=0)

    @classmethod
    def poll(cls, context):
        return _need_doc(cls, context)

    def execute(self, context):
        rs = context.scene.seam.rasters
        if not 0 <= self.index < len(rs):
            return {"CANCELLED"}
        context.scene.seam.active_raster_index = self.index
        if context.object is not None and context.object.mode != "OBJECT":
            self.report({"INFO"}, "Leave the current mode to switch canvases")
            return {"FINISHED"}
        _focus_canvas(context, rs[self.index])
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

    @_undoable_in_paint_mode
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

    @_undoable_in_paint_mode
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

    @_undoable_in_paint_mode
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

    @_undoable_in_paint_mode
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
        if not (doc.layers[i - 1].visible and doc.layers[i].visible):
            cls.poll_message_set("Hidden layers cannot be merged: show both layers first")
            return False
        return True

    @_undoable_in_paint_mode
    def execute(self, context):
        doc = _active(context)
        i = self._index(doc)
        if not 1 <= i < len(doc.layers):
            return {"CANCELLED"}
        if doc.layers[i - 1].locked or doc.layers[i].locked:
            self.report({"WARNING"}, "Locked layers cannot be merged")
            return {"CANCELLED"}
        before = rd.composite_array(doc)
        try:
            rd.merge_down(doc, i)
        except ValueError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        err = float(np.abs(rd.composite_array(_active(context)) - before).max(initial=0.0))
        if err > 0.01:  # a non-Normal blend depends on the whole stack below the pair, so merging it can change the look
            self.report({"WARNING"}, f"Merged, but the look changed slightly (max difference {err:.2f}): "
                                     "one of the merged layers uses a blend mode other than Normal")
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

    @_undoable_in_paint_mode
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

    @_undoable_in_paint_mode
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

    @_undoable_in_paint_mode
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
            rd.pack_layer_images(doc)  # strokes into the image data that the undo step (and auto save) captures
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
        rd.pack_layer_images(doc)
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

    @_undoable_in_paint_mode
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

    @_undoable_in_paint_mode
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
        rd.pack_layer_images(_active(context))
        return {"FINISHED"}


classes = (SEAM_OT_raster_new, SEAM_OT_raster_activate, SEAM_OT_raster_remove, SEAM_OT_raster_layer_add, SEAM_OT_raster_layer_remove,
           SEAM_OT_raster_layer_duplicate, SEAM_OT_raster_layer_move, SEAM_OT_raster_layer_merge_down,
           SEAM_OT_raster_mask_add, SEAM_OT_raster_mask_apply, SEAM_OT_raster_mask_remove,
           SEAM_OT_raster_set_paint_target, SEAM_OT_raster_paint_mode, SEAM_OT_raster_fill, SEAM_OT_raster_clear,
           SEAM_OT_raster_resize_canvas)
