"""Make 3D + Live Link panels and status indicators (owner: link).

``draw_link_status(layout, obj)`` is exported for the shell header.
"""

from __future__ import annotations

import bpy

from ..seam_core import links as L
from . import decal, ops, recipes

STATE_ICONS = {
    "LIVE": "LINKED",
    "PAUSED": "PAUSE",
    "DETACHED": "UNLINKED",
    "ERROR": "ERROR",
    "CONFLICT": "DECORATE_OVERRIDE",
    "MISSING_SOURCE": "ORPHAN_DATA",
}
STATE_LABELS = {
    "LIVE": "Live",
    "PAUSED": "Paused",
    "DETACHED": "Detached",
    "ERROR": "Error",
    "CONFLICT": "Conflict",
    "MISSING_SOURCE": "Missing Source",
}
RECIPE_ICONS = {"EXTRUDE": "MOD_SOLIDIFY", "REVOLVE": "MOD_SCREW", "SWEEP": "MOD_CURVE", "TEXT3D": "OUTLINER_OB_FONT",
                "DECAL": "TEXTURE"}


def state_icon(link) -> str:
    return STATE_ICONS.get(link.state, "QUESTION")


def state_text(link) -> str:
    text = STATE_LABELS.get(link.state, link.state)
    if link.stale and link.state in ("PAUSED", "CONFLICT", "LIVE"):
        text += " (out of date)"
    return text


def draw_link_status(layout, obj) -> None:
    """Compact status of the links of ``obj`` (source or result) for headers/status rows."""
    if obj is None:
        return
    scene = bpy.context.scene
    seam = getattr(scene, "seam", None)
    if seam is None:
        return
    links = ops.links_of(scene, obj)
    if not links:
        return
    row = layout.row(align=True)
    if len(links) == 1:
        lk = links[0]
        row.label(text="%s: %s" % (L.RECIPE_LABELS.get(lk.recipe, lk.recipe), state_text(lk)), icon=state_icon(lk))
        return
    worst = next((s for s in ("CONFLICT", "ERROR", "MISSING_SOURCE", "PAUSED") if any(lk.state == s for lk in links)),
                 "LIVE")
    row.label(text="%d links · %s" % (len(links), STATE_LABELS[worst]), icon=STATE_ICONS[worst])


class SEAM_UL_links(bpy.types.UIList):
    """Links of the active object (as source or result)."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index=0):
        row = layout.row(align=True)
        row.label(text=item.name or item.id, icon=RECIPE_ICONS.get(item.recipe, "MESH_DATA"))
        row.label(text=state_text(item), icon=state_icon(item))

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        obj = context.active_object
        flags = []
        for lk in items:
            show = obj is not None and (lk.source == obj or lk.target == obj)
            flags.append(self.bitflag_filter_item if show else 0)
        return flags, []


class _SeamPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"

    @classmethod
    def poll(cls, context):
        return getattr(context.scene, "seam", None) is not None


def decal_availability(context):
    """{"PLANE": (ok, why), "MATERIAL": (ok, why)} for the current selection (why = requirement text)."""
    src, tgt = ops.pick_decal_pair(context)
    if src is None:
        why = "Decal needs a vector shape, text, group or raster layer"
        return {"PLANE": (False, why), "MATERIAL": (False, why)}
    ok, why = recipes.source_ok("DECAL", src)
    if not ok:
        return {"PLANE": (False, why), "MATERIAL": (False, why)}
    problem = ops.material_decal_problem(src, tgt)
    return {"PLANE": (True, ""), "MATERIAL": (not problem, problem)}


class SEAM_PT_make_3d(_SeamPanel, bpy.types.Panel):
    bl_label = "Make 3D"
    bl_idname = "SEAM_PT_make_3d"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if obj is None:
            layout.label(text="Select a shape or text to make 3D", icon="INFO")
            return
        col = layout.column(align=True)
        for recipe in recipes.MESH_RECIPES:
            ok, why = recipes.source_ok(recipe, obj)
            row = col.row(align=True)
            row.enabled = ok
            op = row.operator("seam.make_3d", text=L.RECIPE_LABELS[recipe], icon=RECIPE_ICONS.get(recipe, "NONE"))
            op.recipe = recipe
            if not ok:
                sub = col.row()
                sub.enabled = False
                sub.label(text=why, icon="BLANK1")
        avail = decal_availability(context)
        col.separator()
        for mode, text in (("PLANE", "Decal (Plane)"), ("MATERIAL", "Decal (Material)")):
            ok, why = avail[mode]
            row = col.row(align=True)
            row.enabled = ok
            op = row.operator("seam.make_3d", text=text, icon=RECIPE_ICONS["DECAL"])
            op.recipe = "DECAL"
            op.decal_mode = mode
            if not ok:
                sub = col.row()
                sub.enabled = False
                sub.label(text=why, icon="BLANK1")


class SEAM_PT_live_link(_SeamPanel, bpy.types.Panel):
    bl_label = "Live Link"
    bl_idname = "SEAM_PT_live_link"

    @classmethod
    def poll(cls, context):
        if not super().poll(context):
            return False
        return bool(ops.links_of(context.scene, context.active_object)) or \
            ops.resolve_link(context) is not None

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.template_list("SEAM_UL_links", "", scene.seam, "links", scene.seam, "active_link_index", rows=2)
        link = ops.resolve_link(context)
        if link is None:
            return
        box = layout.box()
        row = box.row()
        row.label(text=state_text(link), icon=state_icon(link))
        if link.build_count:
            row.label(text="%d builds · %.1f ms" % (link.build_count, link.last_build_ms))
        if link.message:
            for line in _wrap(link.message, 48):
                box.label(text=line)
        row = box.row(align=True)
        row.label(text="Source: %s" % (link.source.name if link.source else "(missing)"), icon="CURVE_DATA")
        row = box.row(align=True)
        row.label(text="Result: %s" % (link.target.name if link.target else "(missing)"), icon="MESH_DATA")

        actions = layout.column(align=True)
        if link.state == "CONFLICT":
            r = actions.row(align=True)
            r.operator("seam.link_overwrite", text="Overwrite", icon="FILE_REFRESH")
            r.operator("seam.link_detach", text="Bake", icon="UNLINKED")
            r.operator("seam.link_branch", text="Branch", icon="DUPLICATE")
        elif link.state == "MISSING_SOURCE":
            r = actions.row(align=True)
            r.operator("seam.link_relink", text="Relink", icon="LINKED")
            r.operator("seam.link_detach", text="Bake", icon="UNLINKED")
        r = actions.row(align=True)
        if link.state == "PAUSED":
            r.operator("seam.link_resume", text="Resume", icon="PLAY")
        else:
            r.operator("seam.link_pause", text="Pause", icon="PAUSE")
        r.operator("seam.link_rebuild", text="Rebuild", icon="FILE_REFRESH")
        r = actions.row(align=True)
        r.operator("seam.edit_source", text="Edit Source", icon="EDITMODE_HLT")
        r.operator("seam.show_result", text="Show Result", icon="HIDE_OFF")
        r = actions.row(align=True)
        r.operator("seam.duplicate_result", text="Duplicate", icon="DUPLICATE")
        if link.state not in ("CONFLICT", "MISSING_SOURCE", "DETACHED"):
            r.operator("seam.link_detach", text="Bake", icon="UNLINKED")
        r.operator("seam.link_remove", text="", icon="X")
        if link.state not in ("DETACHED",) and link.state != "MISSING_SOURCE":
            actions.operator("seam.link_relink", text="Relink Source…", icon="LINKED")

        params = layout.box()
        params.label(text="%s Settings" % L.RECIPE_LABELS.get(link.recipe, link.recipe), icon="PREFERENCES")
        params.use_property_split = True
        params.enabled = link.state != "DETACHED"
        for k in param_keys(link):
            params.prop(link.params, k)
        if link.recipe == "DECAL":
            params.label(text="Mode: %s" % ("Plane" if decal.mode_of(link) == "PLANE" else "Material (mesh untouched)"),
                         icon="MESH_PLANE" if decal.mode_of(link) == "PLANE" else "MATERIAL")
            if decal.src_is_raster(link):
                params.label(text="Raster layers share the document composite", icon="INFO")

        paint = layout.column(align=True)
        paint.operator("seam.result_paint_setup", text="Paint on Result", icon="BRUSH_DATA")
        wm = context.window_manager
        if hasattr(wm, "seam_sync_link_selection"):
            paint.prop(wm, "seam_sync_link_selection", text="Sync Link Selection")


def param_keys(link):
    """Parameters worth showing for ``link`` (a decal shows only what its mode uses)."""
    keys = list(L.RECIPE_PARAMS[link.recipe])
    if link.recipe == "DECAL":
        keys = [k for k in keys if k != "decal_mode"]
        if decal.mode_of(link) == "MATERIAL":
            keys = [k for k in keys if k not in ("units_scale", "orientation")]
        if decal.src_is_raster(link):
            keys = [k for k in keys if k != "decal_resolution"]
    return keys


def _wrap(text: str, width: int):
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


classes = (SEAM_UL_links, SEAM_PT_make_3d, SEAM_PT_live_link)
