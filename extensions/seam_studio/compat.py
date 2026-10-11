"""Blender version shims (4.0 .. 5.2). Owner: integrator.

Only put *version differences* here. Verified facts (2026-10-11, 4.0.2 and 5.2.2 LTS):
- Unified paint settings: ``ToolSettings.unified_paint_settings`` (4.0) vs
  ``Paint.unified_paint_settings`` per paint mode (5.x).
- ``Paint.brush`` is read-only in 4.3+ (brush assets); assignable in 4.0.
- ``Mesh.use_auto_smooth`` (4.0) vs ``Mesh.set_sharp_from_angle()`` (4.1+).
- ``Material.surface_render_method`` exists in 4.2+; ``blend_method`` in both.
- ``mathutils.geometry.delaunay_2d_cdt`` output_type numbering differs above 2.
- Packed *float* images do not round-trip; 8-bit packed PNG images do.
"""

from __future__ import annotations

import bpy

VERSION = bpy.app.version  # (major, minor, patch)
IS_4_0 = VERSION < (4, 1, 0)
HAS_BRUSH_ASSETS = VERSION >= (4, 3, 0)
HAS_EXTENSIONS = VERSION >= (4, 2, 0)


def rna_has(struct_or_type, prop: str) -> bool:
    rna = getattr(struct_or_type, "bl_rna", None)
    return bool(rna) and prop in rna.properties


def unified_paint_settings(context, mode: str = "IMAGE_PAINT"):
    """Return the UnifiedPaintSettings that apply to ``mode`` (IMAGE_PAINT, SCULPT, ...)."""
    ts = context.tool_settings
    if rna_has(ts, "unified_paint_settings"):
        return ts.unified_paint_settings
    paint = {"IMAGE_PAINT": ts.image_paint, "SCULPT": getattr(ts, "sculpt", None),
             "VERTEX_PAINT": getattr(ts, "vertex_paint", None)}.get(mode, ts.image_paint)
    return getattr(paint, "unified_paint_settings", None)


def image_paint_brush(context):
    """Active image-paint brush or None (read-only access works on every version)."""
    ip = context.tool_settings.image_paint
    return getattr(ip, "brush", None)


def set_shade_smooth_by_angle(mesh, angle: float = 0.523599) -> None:
    """Mark faces smooth and sharp edges by angle in a version-independent way."""
    if len(mesh.polygons):
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if hasattr(mesh, "set_sharp_from_angle"):
        mesh.set_sharp_from_angle(angle=angle)
    elif rna_has(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = angle


def set_material_blend_alpha(material) -> None:
    """Make a material render with alpha blending in EEVEE on every version."""
    if rna_has(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    if rna_has(material, "blend_method"):
        try:
            material.blend_method = "BLEND"
        except TypeError:
            pass
    if rna_has(material, "shadow_method"):
        try:
            material.shadow_method = "NONE"
        except TypeError:
            pass


def eevee_engine_id() -> str:
    return "BLENDER_EEVEE_NEXT" if (4, 2, 0) <= VERSION < (5, 0, 0) else "BLENDER_EEVEE"


def new_mix_rgb_node(nodes, blend_type: str = "MIX"):
    """Create a colour Mix node; returns (node, factor_socket, a_socket, b_socket, result_socket)."""
    node = nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    node.blend_type = blend_type
    ins = {s.identifier: s for s in node.inputs}
    outs = {s.identifier: s for s in node.outputs}
    return node, ins["Factor_Float"], ins["A_Color"], ins["B_Color"], outs["Result_Color"]


def keyconfig_addon(context=None):
    """Add-on keyconfig or None (None in background mode)."""
    wm = (context or bpy.context).window_manager
    kcs = getattr(wm, "keyconfigs", None)
    return kcs.addon if kcs is not None else None
