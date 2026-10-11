"""Raster property groups (owner: paint). Field names are a contract — see docs/architecture.md §4."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)

from ..seam_core import raster as R

BLEND_MODE_ITEMS = [(m, m.replace("_", " ").title(), "") for m in R.BLEND_MODES]


def _tag_raster(self, context):
    from ..engine import core as engine
    engine.request_revalidate()


class SEAM_PG_RasterLayer(bpy.types.PropertyGroup):
    id: StringProperty()
    name: StringProperty(name="Name", default="Layer")
    image: PointerProperty(name="Pixels", type=bpy.types.Image)
    opacity: FloatProperty(name="Opacity", min=0.0, max=1.0, default=1.0, subtype="FACTOR", update=_tag_raster)
    blend_mode: EnumProperty(name="Blend", items=BLEND_MODE_ITEMS, default="NORMAL", update=_tag_raster)
    visible: BoolProperty(name="Visible", default=True, update=_tag_raster)
    locked: BoolProperty(name="Locked", default=False)
    mask: PointerProperty(name="Mask", type=bpy.types.Image)
    mask_enabled: BoolProperty(name="Use Mask", default=True, update=_tag_raster)
    rev: IntProperty(name="Pixel Revision", default=0, description="Revision of the layer pixels written by SEAM (undo journal key)")
    mask_rev: IntProperty(name="Mask Revision", default=0, description="Revision of the mask pixels written by SEAM")


class SEAM_PG_RasterDoc(bpy.types.PropertyGroup):
    id: StringProperty()
    name: StringProperty(name="Name", default="Canvas")
    width: IntProperty(name="Width", min=1, max=16384, default=1920)
    height: IntProperty(name="Height", min=1, max=16384, default=1080)
    layers: CollectionProperty(type=SEAM_PG_RasterLayer)
    active_layer_index: IntProperty(default=0)
    canvas: PointerProperty(name="Canvas Object", type=bpy.types.Object)
    material: PointerProperty(name="Material", type=bpy.types.Material)
    composite: PointerProperty(name="Composite Cache", type=bpy.types.Image)
    sink: PointerProperty(name="Locked Paint Sink", type=bpy.types.Image,
                          description="1x1 image that receives native strokes while the active layer is locked")
    selection: PointerProperty(name="Selection Mask", type=bpy.types.Image)
    has_selection: BoolProperty(default=False)
    sel_rev: IntProperty(name="Selection Revision", default=0,
                         description="Revision of the selection pixels written by SEAM (undo journal key)")
    background: EnumProperty(name="Background", items=[("TRANSPARENT", "Transparent", ""), ("WHITE", "White", ""),
                                                       ("COLOR", "Color", "")], default="TRANSPARENT", update=_tag_raster)
    background_color: FloatVectorProperty(name="Background Color", subtype="COLOR", size=4, min=0.0, max=1.0,
                                          default=(1.0, 1.0, 1.0, 1.0), update=_tag_raster)
    edit_target: EnumProperty(name="Paint On", items=[("LAYER", "Layer", "Paint layer pixels"),
                                                      ("MASK", "Mask", "Paint the layer mask")], default="LAYER")


classes = (SEAM_PG_RasterLayer, SEAM_PG_RasterDoc)
