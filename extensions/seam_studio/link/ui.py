"""Make 3D + Live Link panels and status indicators (owner: link).

``draw_link_status(layout, obj)`` is exported for the shell header.
"""

from __future__ import annotations

import bpy

from ..seam_core import links as L
from . import ops, recipes

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
RECIPE_ICONS = {"EXTRUDE": "MOD_SOLIDIFY", "REVOLVE": "MOD_SCREW", "SWEEP": "MOD_CURVE", "TEXT3D": "OUTLINER_OB_FONT"}


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
        for recipe in recipes.EXPOSED_RECIPES:
            ok, why = recipes.source_ok(recipe, obj)
            row = col.row(align=True)
            row.enabled = ok
            op = row.operator("seam.make_3d", text=L.RECIPE_LABELS[recipe], icon=RECIPE_ICONS.get(recipe, "NONE"))
            op.recipe = recipe
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
        for k in L.RECIPE_PARAMS[link.recipe]:
            params.prop(link.params, k)


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
