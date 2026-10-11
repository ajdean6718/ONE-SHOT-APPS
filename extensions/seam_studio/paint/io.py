"""PNG import as layer, PNG export (owner: paint).

Both directions use ``seam_core.png`` (pure zlib) and 8-bit sRGB straight-alpha pixels, so the file content is
exactly what is stored in the layer images / the composite.
"""

import os
import zlib

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, StringProperty

from ..seam_core import png
from ..seam_core import raster as R
from . import rasterdoc as rd

PLACEMENT_ITEMS = [
    ("FIT_DOWN", "Fit if Larger", "Scale down to fit the canvas when larger, otherwise centre at original size"),
    ("FIT", "Fit", "Scale to fit the canvas (keeps aspect ratio), centred"),
    ("CENTER", "Centre", "Original size, centred (cropped when larger)"),
    ("STRETCH", "Stretch", "Stretch to the canvas size"),
    ("TOP_LEFT", "Top Left", "Original size at the top-left corner (cropped when larger)"),
]


def place_image(a8: np.ndarray, canvas_h: int, canvas_w: int, placement: str) -> np.ndarray:
    """Return a ``canvas_h x canvas_w x 4`` uint8 array with ``a8`` placed according to ``placement``."""
    h, w = a8.shape[:2]
    if placement == "STRETCH":
        return rd.resample_rgba8(a8, canvas_h, canvas_w) if (h, w) != (canvas_h, canvas_w) else a8.copy()
    if placement in ("FIT", "FIT_DOWN"):
        scale = min(canvas_w / w, canvas_h / h)
        if placement == "FIT_DOWN":
            scale = min(scale, 1.0)
        if scale != 1.0:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            a8 = rd.resample_rgba8(a8, nh, nw)
    anchor = "TOP_LEFT" if placement == "TOP_LEFT" else "CENTER"
    return rd._fit_canvas(a8, canvas_h, canvas_w, anchor, (0, 0, 0, 0))


def import_png(doc, path: str, name: str = "", placement: str = "FIT_DOWN"):
    """Add a layer (above the active one) with the pixels of a PNG file. Returns the new layer."""
    if placement not in {p[0] for p in PLACEMENT_ITEMS}:
        raise ValueError(f"unknown placement {placement!r}")
    a8 = png.read_png(path)
    w, h = rd.doc_size(doc)
    placed = place_image(a8, h, w, placement)
    layer = rd.add_layer(doc, name or os.path.splitext(os.path.basename(path))[0])
    rd.write_layer_array_srgb8(layer, placed)
    return doc.layers[doc.active_layer_index]


def export_array_srgb8(doc, source: str = "COMPOSITE") -> np.ndarray:
    """uint8 sRGB straight-alpha pixels that :func:`export_png` writes."""
    if source == "LAYER":
        layer = rd.active_layer(doc)
        if layer is None:
            raise ValueError("no active layer")
        return rd.layer_array_srgb8(layer)
    return R.to_uint8_srgb(rd.composite_array(doc))


def export_png(doc, path: str, source: str = "COMPOSITE") -> str:
    """Write the document composite (or the active layer) as a PNG; returns the path."""
    png.write_png(path, export_array_srgb8(doc, source))
    return path


class SEAM_OT_raster_import_png(bpy.types.Operator):
    bl_idname = "seam.raster_import_png"
    bl_label = "Import PNG as Layer"
    bl_description = "Add a new raster layer from a PNG file"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.png", options={"HIDDEN"})
    placement: EnumProperty(name="Placement", items=PLACEMENT_ITEMS, default="FIT_DOWN")

    @classmethod
    def poll(cls, context):
        if rd.active_doc(context.scene) is None:
            cls.poll_message_set("No raster document: create one with New Raster Document")
            return False
        return True

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        doc = rd.active_doc(context.scene)
        path = bpy.path.abspath(self.filepath)
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}
        try:
            import_png(doc, path, placement=self.placement)
        except (ValueError, OSError, zlib.error) as exc:
            self.report({"ERROR"}, f"Cannot import PNG: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class SEAM_OT_raster_export_png(bpy.types.Operator):
    bl_idname = "seam.raster_export_png"
    bl_label = "Export PNG"
    bl_description = "Write the raster composite (or the active layer) as an 8-bit sRGB PNG with straight alpha"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.png", options={"HIDDEN"})
    source: EnumProperty(name="Export", items=[("COMPOSITE", "Composite", "All visible layers over the background"),
                                               ("LAYER", "Active Layer", "Only the pixels of the active layer")],
                         default="COMPOSITE")

    @classmethod
    def poll(cls, context):
        if rd.active_doc(context.scene) is None:
            cls.poll_message_set("No raster document: create one with New Raster Document")
            return False
        return True

    def invoke(self, context, event):
        doc = rd.active_doc(context.scene)
        if not self.filepath:
            self.filepath = f"{doc.name}.png"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        doc = rd.active_doc(context.scene)
        path = bpy.path.abspath(self.filepath)
        if not path:
            self.report({"ERROR"}, "No file path given")
            return {"CANCELLED"}
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            export_png(doc, path, self.source)
        except (ValueError, OSError) as exc:
            self.report({"ERROR"}, f"Cannot export PNG: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {path}")
        return {"FINISHED"}


classes = (SEAM_OT_raster_import_png, SEAM_OT_raster_export_png)
