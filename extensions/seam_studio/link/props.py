"""Link property groups (owner: link). Field names are a contract — see docs/architecture.md §4."""

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty,
                       StringProperty)

from ..seam_core import links as L


def _p(name):
    return L.PARAMS[name]


def _enum(name, labels=None):
    default, allowed, _ = _p(name)
    labels = labels or {}
    return [(v, labels.get(v, v.replace("_", " ").title()), "") for v in allowed], default


def _tag_link(self, context):
    # Param edits are picked up by the link producer (it compares signatures on flush).
    from ..engine import core as engine
    engine.request_revalidate()


_orientation_items, _orientation_default = _enum("orientation", {"UPRIGHT": "Upright", "FLAT": "Flat"})
_origin_items, _origin_default = _enum("origin")
_axis_items, _axis_default = _enum("revolve_axis", {"SOURCE_ORIGIN": "Source Origin"})
_profile_items, _profile_default = _enum("sweep_profile")
_decal_items, _decal_default = _enum("decal_mode")


class SEAM_PG_LinkParams(bpy.types.PropertyGroup):
    depth: FloatProperty(name="Depth", unit="LENGTH", default=_p("depth")[0], min=_p("depth")[1], soft_max=10.0, update=_tag_link)
    bevel_depth: FloatProperty(name="Bevel", unit="LENGTH", default=_p("bevel_depth")[0], min=0.0, soft_max=1.0, update=_tag_link)
    bevel_resolution: IntProperty(name="Bevel Segments", default=_p("bevel_resolution")[0], min=0, max=16, update=_tag_link)
    resolution: IntProperty(name="Curve Resolution", default=_p("resolution")[0], min=1, max=64, update=_tag_link)
    orientation: EnumProperty(name="Orientation", items=_orientation_items, default=_orientation_default, update=_tag_link)
    origin: EnumProperty(name="Origin", items=_origin_items, default=_origin_default, update=_tag_link)
    units_scale: FloatProperty(name="Scale (m per px)", default=_p("units_scale")[0], min=1e-6, soft_max=1.0,
                               precision=4, update=_tag_link)
    revolve_angle: FloatProperty(name="Angle", subtype="ANGLE", default=_p("revolve_angle")[0], min=0.01,
                                 max=6.283185307179586, update=_tag_link)
    revolve_segments: IntProperty(name="Segments", default=_p("revolve_segments")[0], min=3, max=512, update=_tag_link)
    revolve_axis: EnumProperty(name="Axis", items=_axis_items, default=_axis_default, update=_tag_link)
    axis_offset: FloatProperty(name="Axis Offset (px)", default=0.0, update=_tag_link)
    sweep_radius: FloatProperty(name="Width", unit="LENGTH", default=_p("sweep_radius")[0], min=0.0001, soft_max=1.0, update=_tag_link)
    sweep_profile: EnumProperty(name="Profile", items=_profile_items, default=_profile_default, update=_tag_link)
    sweep_profile_resolution: IntProperty(name="Profile Segments", default=_p("sweep_profile_resolution")[0], min=1, max=32, update=_tag_link)
    fill_caps: BoolProperty(name="Fill Caps", default=True, update=_tag_link)
    decal_mode: EnumProperty(name="Decal Mode", items=_decal_items, default=_decal_default, update=_tag_link)
    decal_resolution: IntProperty(name="Texture Size", default=_p("decal_resolution")[0], min=16, max=8192, update=_tag_link)
    use_source_material: BoolProperty(name="Use Source Colour", default=True, update=_tag_link)


class SEAM_PG_Link(bpy.types.PropertyGroup):
    id: StringProperty(name="Link ID")
    recipe: EnumProperty(name="Recipe", items=[(r, r.title().replace("3d", "3D"), "") for r in L.RECIPES], default="EXTRUDE")
    source: PointerProperty(name="Source", type=bpy.types.Object)
    source_id: StringProperty(name="Source SEAM ID")
    target: PointerProperty(name="Result", type=bpy.types.Object)
    target_id: StringProperty(name="Result SEAM ID")
    state: EnumProperty(name="State", items=[
        ("LIVE", "Live", "Updates when the source changes", "LINKED", 0),
        ("PAUSED", "Paused", "Source changes are queued until resumed", "PAUSE", 1),
        ("DETACHED", "Detached", "Baked; no longer follows the source", "UNLINKED", 2),
        ("ERROR", "Error", "Last update failed; previous geometry kept", "ERROR", 3),
        ("CONFLICT", "Conflict", "Result was edited downstream; choose Overwrite, Bake or Branch", "DECORATE_OVERRIDE", 4),
        ("MISSING_SOURCE", "Missing Source", "Source was deleted; relink or detach", "ORPHAN_DATA", 5),
    ], default="LIVE")
    stale: BoolProperty(name="Stale", default=False)
    message: StringProperty(name="Message")
    params: PointerProperty(type=SEAM_PG_LinkParams)
    source_signature: StringProperty()
    target_signature: StringProperty()
    # hash of the per-corner "UVMap" values and per-face material indices the last build wrote
    target_data_signature: StringProperty()
    build_count: IntProperty()
    last_build_ms: FloatProperty()
    # DECAL: the image the link keeps current (its own "SEAM Decal <source>", or a raster document composite)
    decal_image: PointerProperty(name="Decal Image", type=bpy.types.Image)
    # one-shot: the next build may discard downstream edits (set by Overwrite / Branch)
    force_overwrite: BoolProperty(default=False, options={"HIDDEN"})


classes = (SEAM_PG_LinkParams, SEAM_PG_Link)
