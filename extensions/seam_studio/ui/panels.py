"""Welcome/empty states, document panel, header status, shortcuts help (owner: shell)."""

import time

import bpy

from ..seam_core.units import px_to_unit
from . import document, health, keymaps, ops, prefs as prefs_mod

CATEGORY = "SEAM"

#: region pointer -> time.monotonic() of the last Document-panel draw there. Panels of an inactive sidebar tab are
#: not drawn, so this tells scripts (``scripts/build_template.py``, GUI tests) which 3D views show the SEAM tab:
#: Blender 4.0 offers no API to read or set the active sidebar category.
drawn_regions = {}


def seam_tab_shown(region, since: float) -> bool:
    """True if the SEAM sidebar tab was drawn in ``region`` at or after ``since`` (``time.monotonic()``)."""
    return drawn_regions.get(region.as_pointer(), 0.0) >= since


class _SeamPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY


# ------------------------------------------------------------------ Start

class SEAM_PT_start(_SeamPanel, bpy.types.Panel):
    bl_idname = "SEAM_PT_start"
    bl_label = "Start"
    bl_order = 0

    @classmethod
    def poll(cls, context):
        return prefs_mod.get(context).show_start_panel and document.is_pristine(context.scene)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Draw, paint, or turn a")
        col.label(text="drawing into live 3D.")
        layout.separator(factor=0.5)
        icons = {"VECTOR": "OUTLINER_OB_CURVE", "PAINT": "BRUSH_DATA", "DRAW3D": "MESH_CUBE"}
        idnames = {"VECTOR": "seam.new_vector_document", "PAINT": "seam.new_paint_canvas", "DRAW3D": "seam.draw_to_3d"}
        for kind in ("VECTOR", "PAINT", "DRAW3D"):
            ok, why = ops.quickstart_status(kind, context.scene)
            col = layout.column(align=True)
            col.scale_y = 1.4
            col.operator(idnames[kind], text=ops.LABELS[kind], icon=icons[kind])
            if not ok:
                sub = layout.column(align=True)
                sub.scale_y = 0.8
                for line in _wrap(why, 27):
                    sub.label(text=line)
        layout.separator(factor=0.5)
        row = layout.row()
        row.scale_y = 0.9
        row.operator("seam.start_dismiss", text="Hide this panel", icon="X")


def _wrap(text: str, width: int):
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


# ------------------------------------------------------------------ Document

def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


class SEAM_PT_document(_SeamPanel, bpy.types.Panel):
    bl_idname = "SEAM_PT_document"
    bl_label = "Document"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        doc = scene.seam.doc
        eng = scene.seam.engine
        if context.region is not None:
            drawn_regions[context.region.as_pointer()] = time.monotonic()

        col = layout.column(align=True)
        col.prop(doc, "units")
        col.prop(doc, "dpi")
        ab = _active_artboard(scene)
        if ab is not None:
            w, h = ab.seam.artboard.width, ab.seam.artboard.height
            text = f"Artboard {_fmt(w)} x {_fmt(h)} px"
            if doc.units != "PX":
                unit = doc.units.lower()
                text += f" ({_fmt(px_to_unit(w, doc.units, doc.dpi))} x {_fmt(px_to_unit(h, doc.units, doc.dpi))} {unit})"
            layout.label(text=text, icon="MESH_PLANE")
        else:
            layout.label(text="No canvas yet", icon="INFO")

        row = layout.row(align=True)
        row.prop(eng, "live_enabled", text="Live Updates", toggle=True,
                 icon="PLAY" if eng.live_enabled else "PAUSE")

        rep = health.last_report
        box = layout.box()
        r = box.row()
        if rep is None:
            r.label(text="Health: not checked", icon="QUESTION")
        elif rep.problems:
            r.label(text=f"Health: {rep.summary()}", icon="ERROR")
        else:
            r.label(text="Health: no problems", icon="CHECKMARK")
        box.operator("seam.health_check", icon="VIEWZOOM")
        layout.operator("seam.workspace_repair", icon="WORKSPACE")


def _active_artboard(scene):
    from ..vector import canvas
    return canvas.active_artboard(scene)


class SEAM_PT_engine(_SeamPanel, bpy.types.Panel):
    bl_idname = "SEAM_PT_engine"
    bl_label = "Engine"
    bl_parent_id = "SEAM_PT_document"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        from ..engine import core
        layout = self.layout
        s = core.stats()
        col = layout.column(align=True)
        col.label(text=f"Builds {s['builds']}   Skipped {s['skipped']}   Errors {s['errors']}")
        col.label(text=f"Last flush {s['last_flush_ms']:.1f} ms   Slowest build {s['max_build_ms']:.1f} ms")
        col.label(text=f"Waiting {s['dirty']} queued / {s['pending']} async   Stale dropped {s['stale_dropped']}")
        for kind, k in sorted(s["by_kind"].items()):
            avg = k["ms_total"] / k["builds"] if k["builds"] else 0.0
            col.label(text=f"{kind}: {k['builds']} builds, avg {avg:.1f} ms, {k['errors']} errors")
        layout.prop(context.scene.seam.engine, "debounce_ms")
        layout.prop(context.scene.seam.engine, "async_threshold_ms")


# ------------------------------------------------------------------ Shortcuts

class SEAM_PT_shortcuts(_SeamPanel, bpy.types.Panel):
    bl_idname = "SEAM_PT_shortcuts"
    bl_label = "Shortcuts"
    bl_order = 100
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        items = keymaps.registered_items() or keymaps.available_items()
        if not items:
            layout.label(text="No SEAM shortcuts are bound.")
            return
        col = layout.column(align=True)
        for it in items:
            split = col.split(factor=0.42, align=True)
            split.label(text=keymaps.key_text(it))
            split.label(text=it.description)
        layout.label(text="Rebind in Preferences > Keymap > 3D View.")


# ------------------------------------------------------------------ 3D view header

def draw_header_status(self, context):
    """Compact live/paused switch and the active object's link state (appended to ``VIEW3D_HT_header``)."""
    scene = context.scene
    seam = getattr(scene, "seam", None)
    if seam is None or not (seam.doc.canvas_ready or len(seam.links)):
        return
    layout = self.layout
    eng = seam.engine
    row = layout.row(align=True)
    row.prop(eng, "live_enabled", text="Live" if eng.live_enabled else "Paused", toggle=True,
             icon="PLAY" if eng.live_enabled else "PAUSE")
    obj = context.active_object
    if obj is not None:
        try:
            from ..link import ui as link_ui
            fn = getattr(link_ui, "draw_link_status", None)
            if fn is not None:
                fn(layout, obj)
        except Exception as exc:  # a broken link widget must not break the header
            layout.label(text="link status error", icon="ERROR")
            print("SEAM header: draw_link_status failed:", exc)


classes = (SEAM_PT_start, SEAM_PT_document, SEAM_PT_engine, SEAM_PT_shortcuts)


def register():
    bpy.types.VIEW3D_HT_header.append(draw_header_status)


def unregister():
    try:
        bpy.types.VIEW3D_HT_header.remove(draw_header_status)
    except (ValueError, RuntimeError):
        pass
