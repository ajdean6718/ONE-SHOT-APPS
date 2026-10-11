"""Paint panels and layer list (owner: paint)."""

import bpy

from . import rasterdoc as rd


class SEAM_UL_raster_layers(bpy.types.UIList):
    """Layer stack, topmost layer first."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "visible", text="", icon="HIDE_OFF" if item.visible else "HIDE_ON", emboss=False)
        row.prop(item, "name", text="", emboss=False)
        if item.mask is not None:
            row.prop(item, "mask_enabled", text="", icon="MOD_MASK", emboss=False)
        row.prop(item, "locked", text="", icon="LOCKED" if item.locked else "UNLOCKED", emboss=False)

    def filter_items(self, context, data, propname):
        n = len(getattr(data, propname))
        flags = [self.bitflag_filter_item] * n
        order = list(range(n - 1, -1, -1))
        return flags, order


def _doc(context):
    return rd.active_doc(context.scene)


class SEAM_PT_paint_document(bpy.types.Panel):
    bl_idname = "SEAM_PT_paint_document"
    bl_label = "Raster Document"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        doc = _doc(context)
        if len(scene.seam.rasters) > 1:
            col = layout.column(align=True)
            for i, d in enumerate(scene.seam.rasters):
                op = col.operator("seam.raster_activate", text=d.name, icon="IMAGE_DATA",
                                  depress=(doc is not None and d.id == doc.id))
                op.index = i
        if doc is None:
            layout.operator("seam.raster_new", icon="FILE_NEW")
            layout.label(text="Create a canvas to paint on layers", icon="INFO")
            return
        col = layout.column(align=True)
        col.prop(doc, "name", text="")
        row = col.row(align=True)
        row.label(text=f"{doc.width} x {doc.height} px", icon="IMAGE_DATA")
        row.operator("seam.raster_resize_canvas", text="", icon="FULLSCREEN_ENTER")
        warn = rd.pixel_restore_warning()
        if warn:
            import textwrap
            box = layout.box()
            box.alert = True
            box.label(text="Some pixels could not be restored", icon="ERROR")
            for line in textwrap.wrap(warn, 38):
                box.label(text=line)
        layout.prop(doc, "background", text="Background")
        if doc.background == "COLOR":
            layout.prop(doc, "background_color", text="")
        in_paint = rd.paint_mode_active(doc)
        row = layout.row(align=True)
        row.scale_y = 1.3
        op = row.operator("seam.raster_paint_mode", text="Exit Paint Mode" if in_paint else "Paint on Canvas",
                          icon="OBJECT_DATAMODE" if in_paint else "BRUSH_DATA", depress=in_paint)
        op.enter = not in_paint
        row = layout.row(align=True)
        row.operator("seam.raster_new", text="New", icon="ADD")
        row.operator("seam.raster_remove", text="Delete", icon="TRASH")


class SEAM_PT_paint_layers(bpy.types.Panel):
    bl_idname = "SEAM_PT_paint_layers"
    bl_label = "Raster Layers"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"

    def draw(self, context):
        layout = self.layout
        doc = _doc(context)
        if doc is None:
            layout.label(text="No raster document", icon="INFO")
            return
        row = layout.row()
        row.template_list("SEAM_UL_raster_layers", "", doc, "layers", doc, "active_layer_index", rows=5)
        col = row.column(align=True)
        col.operator("seam.raster_layer_add", text="", icon="ADD")
        col.operator("seam.raster_layer_remove", text="", icon="REMOVE")
        col.operator("seam.raster_layer_duplicate", text="", icon="DUPLICATE")
        col.separator()
        col.operator("seam.raster_layer_move", text="", icon="TRIA_UP").direction = "UP"
        col.operator("seam.raster_layer_move", text="", icon="TRIA_DOWN").direction = "DOWN"
        layout.operator("seam.raster_layer_merge_down", text="Merge Down", icon="TRIA_DOWN_BAR")
        layer = rd.active_layer(doc)
        if layer is None:
            return
        col = layout.column(align=True)
        col.prop(layer, "opacity", text="Opacity", slider=True)
        col.prop(layer, "blend_mode", text="")
        if layer.locked:
            layout.label(text="Layer locked: strokes are discarded", icon="ERROR")
        box = layout.box()
        box.label(text="Mask", icon="MOD_MASK")
        if layer.mask is None:
            row = box.row(align=True)
            op = row.operator("seam.raster_mask_add", text="Reveal All")
            op.fill = "WHITE"
            op = row.operator("seam.raster_mask_add", text="Hide All")
            op.fill = "BLACK"
            op = box.operator("seam.raster_mask_add", text="From Selection")
            op.fill = "SELECTION"
        else:
            row = box.row(align=True)
            row.prop(layer, "mask_enabled", text="Enabled")
            row.operator("seam.raster_mask_apply", text="Apply")
            row.operator("seam.raster_mask_remove", text="Delete")
        row = layout.row(align=True)
        row.label(text="Paint on:")
        for target, text in (("LAYER", "Layer"), ("MASK", "Mask")):
            sub = row.row(align=True)
            sub.enabled = target == "LAYER" or layer.mask is not None
            op = sub.operator("seam.raster_set_paint_target", text=text,
                              depress=(doc.edit_target == target) or (target == "LAYER" and layer.mask is None))
            op.target = target


class SEAM_PT_paint_selection(bpy.types.Panel):
    bl_idname = "SEAM_PT_paint_selection"
    bl_label = "Raster Selection"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"

    def draw(self, context):
        layout = self.layout
        doc = _doc(context)
        if doc is None:
            layout.label(text="No raster document", icon="INFO")
            return
        row = layout.row(align=True)
        row.operator("seam.raster_select_rect", text="Rect", icon="SELECT_SET")
        row.operator("seam.raster_select_ellipse", text="Ellipse", icon="MESH_CIRCLE")
        row.operator("seam.raster_select_lasso", text="Lasso", icon="GREASEPENCIL")
        row = layout.row(align=True)
        row.operator("seam.raster_select_all", text="All")
        row.operator("seam.raster_select_none", text="None")
        row.operator("seam.raster_select_invert", text="Invert")
        layout.label(text="Painting inside selection" if doc.has_selection else "No selection",
                     icon="RESTRICT_SELECT_OFF" if doc.has_selection else "RESTRICT_SELECT_ON")


class SEAM_PT_paint_actions(bpy.types.Panel):
    bl_idname = "SEAM_PT_paint_actions"
    bl_label = "Raster Actions"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"

    def draw(self, context):
        layout = self.layout
        doc = _doc(context)
        if doc is None:
            layout.label(text="No raster document", icon="INFO")
            return
        row = layout.row(align=True)
        row.operator("seam.raster_fill", text="Fill", icon="SNAP_FACE")
        row.operator("seam.raster_clear", text="Clear", icon="X")
        row = layout.row(align=True)
        row.operator("seam.raster_import_png", text="Import PNG", icon="IMPORT")
        row.operator("seam.raster_export_png", text="Export PNG", icon="EXPORT")


classes = (SEAM_UL_raster_layers, SEAM_PT_paint_document, SEAM_PT_paint_layers, SEAM_PT_paint_selection,
           SEAM_PT_paint_actions)
