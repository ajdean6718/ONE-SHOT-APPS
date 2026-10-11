"""Vector property groups (owner: vector). Field names are a contract — see docs/architecture.md §4."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty)


def _tag(self, context):
    # Style edits are picked up by the engine through depsgraph ID updates on the object.
    obj = getattr(self, "id_data", None)
    if obj is not None and hasattr(obj, "update_tag"):
        obj.update_tag()


class SEAM_PG_GradientStop(bpy.types.PropertyGroup):
    position: FloatProperty(name="Position", min=0.0, max=1.0, default=0.0, update=_tag)
    color: FloatVectorProperty(name="Color", subtype="COLOR", size=4, min=0.0, max=1.0,
                               default=(1.0, 1.0, 1.0, 1.0), update=_tag)


class SEAM_PG_Style(bpy.types.PropertyGroup):
    fill_enabled: BoolProperty(name="Fill", default=True, update=_tag)
    fill_type: EnumProperty(name="Fill Type", items=[
        ("SOLID", "Solid", "Single colour"),
        ("LINEAR", "Linear Gradient", "Linear gradient across the shape bounds"),
        ("RADIAL", "Radial Gradient", "Radial gradient from the center"),
    ], default="SOLID", update=_tag)
    fill_color: FloatVectorProperty(name="Fill Color", subtype="COLOR", size=4, min=0.0, max=1.0,
                                    default=(0.8, 0.8, 0.8, 1.0), update=_tag)
    gradient_stops: CollectionProperty(type=SEAM_PG_GradientStop)
    gradient_angle: FloatProperty(name="Angle", subtype="ANGLE", default=0.0, update=_tag)
    gradient_center: FloatVectorProperty(name="Center", size=2, default=(0.5, 0.5), update=_tag,
                                         description="Bounding-box relative center (0..1)")
    gradient_radius: FloatProperty(name="Radius", min=0.0, default=0.5, update=_tag,
                                   description="Bounding-box relative radius")
    stroke_enabled: BoolProperty(name="Stroke", default=False, update=_tag)
    stroke_color: FloatVectorProperty(name="Stroke Color", subtype="COLOR", size=4, min=0.0, max=1.0,
                                      default=(0.0, 0.0, 0.0, 1.0), update=_tag)
    stroke_width: FloatProperty(name="Width", min=0.0, soft_max=200.0, default=2.0, update=_tag)
    stroke_cap: EnumProperty(name="Cap", items=[("BUTT", "Butt", ""), ("ROUND", "Round", ""),
                                                ("SQUARE", "Square", "")], default="BUTT", update=_tag)
    stroke_join: EnumProperty(name="Join", items=[("MITER", "Miter", ""), ("ROUND", "Round", ""),
                                                  ("BEVEL", "Bevel", "")], default="MITER", update=_tag)
    miter_limit: FloatProperty(name="Miter Limit", min=1.0, max=100.0, default=4.0, update=_tag)
    opacity: FloatProperty(name="Opacity", min=0.0, max=1.0, default=1.0, subtype="FACTOR", update=_tag)
    fill_rule: EnumProperty(name="Fill Rule", items=[("EVENODD", "Even-Odd", ""), ("NONZERO", "Non-Zero", "")],
                            default="EVENODD", update=_tag)


SHAPE_KINDS = [
    ("NONE", "Path", "Free-form path (nodes own the geometry)"),
    ("RECT", "Rectangle", ""),
    ("ROUNDED_RECT", "Rounded Rectangle", ""),
    ("ELLIPSE", "Ellipse", ""),
    ("POLYGON", "Polygon", ""),
    ("STAR", "Star", ""),
    ("LINE", "Line", ""),
]


class SEAM_PG_ShapeRecipe(bpy.types.PropertyGroup):
    kind: EnumProperty(name="Kind", items=SHAPE_KINDS, default="NONE")
    width: FloatProperty(name="Width", min=0.0, default=100.0, update=_tag)
    height: FloatProperty(name="Height", min=0.0, default=100.0, update=_tag)
    radius: FloatProperty(name="Corner Radius", min=0.0, default=0.0, update=_tag)
    sides: IntProperty(name="Sides", min=3, max=200, default=5, update=_tag)
    inner_ratio: FloatProperty(name="Inner Ratio", min=0.01, max=1.0, default=0.5, update=_tag)


class SEAM_PG_Artboard(bpy.types.PropertyGroup):
    width: FloatProperty(name="Width", min=1.0, default=1920.0, update=_tag)
    height: FloatProperty(name="Height", min=1.0, default=1080.0, update=_tag)
    background_enabled: BoolProperty(name="Background", default=True, update=_tag)
    background_color: FloatVectorProperty(name="Background Color", subtype="COLOR", size=4, min=0.0, max=1.0,
                                          default=(1.0, 1.0, 1.0, 1.0), update=_tag)
    export_scale: FloatProperty(name="Export Scale", min=0.01, max=16.0, default=1.0)


classes = (SEAM_PG_GradientStop, SEAM_PG_Style, SEAM_PG_ShapeRecipe, SEAM_PG_Artboard)
