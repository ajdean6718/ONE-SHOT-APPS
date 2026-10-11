"""SVG import/export operators (owner: vector-b).

Export (``seam.svg_export``)
    Visible artwork in z-order (layers bottom -> top, then canvas z) becomes ``svg.SvgShape`` /
    ``svg.SvgImage`` items in canvas coordinates; ``seam_core.svg.export_svg`` writes the file.
    Bezier data is exported exactly (not flattened), transforms are baked into the paths, text is
    exported as outlines (``text_mode='OUTLINES'``, default) or as ``<text>`` (``'TEXT'``).  Clips
    become ``<clipPath>`` via ``SvgShape.clip_paths``.  The artboard background rectangle is only added with ``include_background``.
Import (``seam.svg_import``)
    Creates shapes with their styles in the active layer.  The artboard comes from the SVG
    viewBox/size (the pristine default artboard is resized, otherwise a new artboard is added to
    the right of the existing ones).  Unsupported elements are reported as warnings.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from .. import ids
from ..bridge import paths_signature
from ..seam_core import png, raster as raster_core, svg
from ..seam_core.geometry import BezierPath, BezierPoint
from . import canvas
from .clip import apply_clip
from .export_png import (EXPORT_RESOLUTION, artboard_rect, artwork_in_order, canvas_paths, fill_rings,
                         gradient_geometry, gradient_stops, raster_bounds, raster_image_array, resolve_artboard,
                         rgba, rings_bounds)
from .pathfinder import paths_center, select_only, translate_paths

TEXT_MODES = [
    ("OUTLINES", "Outlines", "Export text as vector outlines (renders identically everywhere)"),
    ("TEXT", "Live Text", "Export text as <text> elements (needs the font to be installed by the viewer)"),
]


# ------------------------------------------------------------------ export

def _svg_style(obj, bounds) -> svg.SvgStyle:
    st = obj.seam.style
    fill = None
    if st.fill_enabled:
        stops = gradient_stops(st)
        if st.fill_type != "SOLID" and len(stops) >= 2 and bounds is not None:
            kind, a, b = gradient_geometry(st, bounds)
            if kind == "LINEAR":
                fill = svg.SvgGradient("LINEAR", stops=stops, start=a, end=b)
            else:
                fill = svg.SvgGradient("RADIAL", stops=stops, center=a, radius=b)
        else:
            fill = rgba(st.fill_color)
    stroke = rgba(st.stroke_color) if (st.stroke_enabled and st.stroke_width > 0.0) else None
    return svg.SvgStyle(fill=fill, stroke=stroke, stroke_width=float(st.stroke_width), stroke_cap=st.stroke_cap,
                        stroke_join=st.stroke_join, miter_limit=float(st.miter_limit), opacity=float(st.opacity),
                        fill_rule="NONZERO" if obj.type == "FONT" else st.fill_rule)


def _rect_path(x0: float, y0: float, x1: float, y1: float) -> BezierPath:
    pts = [(x0, y1), (x0, y0), (x1, y0), (x1, y1)]
    return BezierPath([BezierPoint(p, p, p, "VECTOR", "VECTOR") for p in pts], True)


def build_export_items(scene, artboard, text_mode: str = "OUTLINES", warnings: Optional[List[str]] = None,
                       include_background: bool = False):
    """Return (items, (x_left, y_top, width, height)) for ``svg.export_svg``."""
    ax, ay, aw, ah = artboard_rect(artboard)
    items: list = []
    ab = artboard.seam.artboard
    if include_background and ab.background_enabled:
        items.append(svg.SvgShape(name=f"{artboard.name} background", paths=[_rect_path(ax, ay - ah, ax + aw, ay)],
                                  style=svg.SvgStyle(fill=rgba(ab.background_color), stroke=None)))
    for obj in artwork_in_order(scene):
        if obj.seam.role == "RASTER":
            try:
                arr = raster_image_array(scene, obj)
            except (ImportError, AttributeError, NotImplementedError) as exc:
                if warnings is not None:
                    warnings.append(f"Raster '{obj.name}' skipped: raster compositor unavailable ({exc})")
                continue
            b = raster_bounds(obj)
            if arr is None or b is None:
                continue
            data = png.encode_png(raster_core.to_uint8_srgb(arr))
            items.append(svg.SvgImage(name=obj.name, png_bytes=data, x=b[0], y_top=b[3], width=b[2] - b[0],
                                      height=b[3] - b[1], opacity=float(obj.seam.style.opacity)))
            continue
        paths = canvas_paths(obj)
        if not paths:
            continue
        bounds = rings_bounds(fill_rings(paths, EXPORT_RESOLUTION))
        style = _svg_style(obj, bounds)
        if style.fill is None and style.stroke is None:
            continue
        shape = svg.SvgShape(name=obj.name, paths=paths, style=style)
        mask = obj.seam.clip_source
        if mask is not None and mask is not obj:
            shape.clip_paths = canvas_paths(mask)
        if obj.type == "FONT":
            shape.font_family = obj.data.font.name if obj.data.font else "sans-serif"
            shape.font_size = float(obj.data.size)
            if text_mode == "TEXT":
                shape.text = obj.data.body
        items.append(shape)
    return items, (ax, ay, aw, ah)


class SEAM_OT_svg_export(bpy.types.Operator, ExportHelper):
    """Export an artboard as SVG"""
    bl_idname = "seam.svg_export"
    bl_label = "Export Artboard as SVG"
    bl_options = {"REGISTER"}

    filename_ext = ".svg"
    filter_glob: StringProperty(default="*.svg", options={"HIDDEN"})
    text_mode: EnumProperty(name="Text", items=TEXT_MODES, default="OUTLINES")
    artboard: StringProperty(name="Artboard", default="",
                             description="Artboard name or SEAM id (empty = active artboard)")
    include_background: BoolProperty(name="Include Background", default=False,
                                     description="Add the artboard background colour as the first rectangle")

    def invoke(self, context, event):
        ab = resolve_artboard(context.scene, self.artboard)
        if ab is None:
            self.report({"ERROR"}, "No artboard to export")
            return {"CANCELLED"}
        if not self.filepath:
            self.filepath = ab.name + ".svg"
        return super().invoke(context, event)

    def execute(self, context):
        scene = context.scene
        ab = resolve_artboard(scene, self.artboard)
        if ab is None:
            self.report({"ERROR"}, f"Artboard '{self.artboard}' not found" if self.artboard else "No artboard to export")
            return {"CANCELLED"}
        path = bpy.path.abspath(self.filepath)
        if not path.lower().endswith(".svg"):
            path += ".svg"
        warnings: List[str] = []
        try:
            from ..engine import core as engine
            engine.flush(scene)
            items, rect = build_export_items(scene, ab, self.text_mode, warnings, self.include_background)
            text = svg.export_svg(items, rect, self.text_mode)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except NotImplementedError as exc:
            self.report({"ERROR"}, f"SVG export unavailable: {exc}")
            return {"CANCELLED"}
        except OSError as exc:
            self.report({"ERROR"}, f"SVG export failed: {exc}")
            return {"CANCELLED"}
        for w in warnings[:10]:
            self.report({"WARNING"}, w)
        self.report({"INFO"}, f"Exported {len(items)} item(s) to {os.path.basename(path)}")
        return {"FINISHED"}


# ------------------------------------------------------------------ import

def _apply_gradient(style, grad: svg.SvgGradient, bounds, warnings: List[str], name: str) -> None:
    style.fill_type = grad.kind
    style.gradient_stops.clear()
    for pos, col in grad.stops:
        s = style.gradient_stops.add()
        s.position = float(pos)
        s.color = tuple(col)
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds
    w, h = max(1e-9, x1 - x0), max(1e-9, y1 - y0)
    if grad.kind == "LINEAR":
        dx, dy = grad.end[0] - grad.start[0], grad.end[1] - grad.start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return
        style.gradient_angle = math.atan2(dy, dx)
        ux, uy = dx / length, dy / length
        expected = abs(w * ux) + abs(h * uy)
        mid = ((grad.start[0] + grad.end[0]) * 0.5, (grad.start[1] + grad.end[1]) * 0.5)
        off = math.hypot(mid[0] - (x0 + w / 2), mid[1] - (y0 + h / 2))
        if abs(length - expected) > 0.03 * expected or off > 0.03 * max(w, h):
            warnings.append(f"'{name}': gradient extent was approximated to the shape bounds")
    else:
        style.gradient_center = ((grad.center[0] - x0) / w, (grad.center[1] - y0) / h)
        style.gradient_radius = grad.radius / max(w, h)


def _apply_svg_style(obj, sty: svg.SvgStyle, paths, warnings: List[str]) -> None:
    st = obj.seam.style
    if sty.fill is None:
        st.fill_enabled = False
    elif isinstance(sty.fill, svg.SvgGradient):
        st.fill_enabled = True
        _apply_gradient(st, sty.fill, rings_bounds(fill_rings(paths, EXPORT_RESOLUTION)), warnings, obj.name)
        if sty.fill.stops:
            st.fill_color = tuple(sty.fill.stops[0][1])
    else:
        st.fill_enabled = True
        st.fill_type = "SOLID"
        st.fill_color = tuple(sty.fill)
    if sty.stroke is None:
        st.stroke_enabled = False
    else:
        st.stroke_enabled = True
        st.stroke_color = tuple(sty.stroke)
        st.stroke_width = float(sty.stroke_width)
        st.stroke_cap = sty.stroke_cap
        st.stroke_join = sty.stroke_join
        st.miter_limit = max(1.0, float(sty.miter_limit))
    st.opacity = min(1.0, max(0.0, float(sty.opacity)))
    st.fill_rule = sty.fill_rule if sty.fill_rule in ("EVENODD", "NONZERO") else "NONZERO"


def _prepare_artboard(scene, name: str, width: float, height: float):
    """Resize the pristine default artboard or add a new one; returns (artboard, x, y) of its top-left corner."""
    canvas.ensure_canvas(scene)
    abs_ = canvas.artboards(scene)
    has_art = bool(canvas.artwork_objects(scene))
    if len(abs_) == 1 and not has_art:
        ab = abs_[0]
        ab.name = name
        ab.seam.artboard.width = width
        ab.seam.artboard.height = height
        old = ab.data
        ab.data = canvas.artboard_mesh(width, height, name)
        if old is not None and old.users == 0:
            bpy.data.meshes.remove(old)
    else:
        rects = [artboard_rect(a) for a in abs_]
        right = max((r[0] + r[2] for r in rects), default=-100.0)
        top = max((r[1] for r in rects), default=0.0)
        ab = canvas.new_artboard(scene, name, right + 100.0, top, width, height)
    scene.seam.doc.active_artboard_id = ids.ensure_id(ab)
    return ab, float(ab.location.x), float(ab.location.y)


class SEAM_OT_svg_import(bpy.types.Operator, ImportHelper):
    """Import an SVG file as editable shapes on a new artboard"""
    bl_idname = "seam.svg_import"
    bl_label = "Import SVG"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".svg"
    filter_glob: StringProperty(default="*.svg", options={"HIDDEN"})

    def execute(self, context):
        scene = context.scene
        path = bpy.path.abspath(self.filepath)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            self.report({"ERROR"}, f"Cannot read '{path}': {exc}")
            return {"CANCELLED"}
        try:
            result = svg.import_svg(text)
        except NotImplementedError as exc:
            self.report({"ERROR"}, f"SVG import unavailable: {exc}")
            return {"CANCELLED"}
        except Exception as exc:  # malformed XML / unsupported structure: say which file and why
            self.report({"ERROR"}, f"Invalid SVG '{os.path.basename(path)}': {exc}")
            return {"CANCELLED"}
        warnings = list(result.warnings)
        name = os.path.splitext(os.path.basename(path))[0] or "SVG"
        ab, ax, ay = _prepare_artboard(scene, name, float(result.width), float(result.height))
        layer = canvas.active_layer(scene)
        created: List[bpy.types.Object] = []
        masks: Dict[str, bpy.types.Object] = {}
        for sh in result.shapes:
            paths = translate_paths(sh.paths, ax, ay)
            if not paths:
                continue
            cx, cy = paths_center(paths)
            ob = canvas.new_shape_object(scene, sh.name or "Path", translate_paths(paths, -cx, -cy),
                                         location=(cx, cy), layer=layer)
            _apply_svg_style(ob, sh.style, paths, warnings)
            if sh.clip_paths:
                cpaths = translate_paths(sh.clip_paths, ax, ay)
                sig = paths_signature(cpaths)
                mask = masks.get(sig)
                if mask is None:
                    mx, my = paths_center(cpaths)
                    mask = canvas.new_shape_object(scene, f"{ob.name} Clip", translate_paths(cpaths, -mx, -my),
                                                   location=(mx, my), layer=layer)
                    mask.seam.style.fill_rule = "NONZERO"
                    masks[sig] = mask
                try:
                    apply_clip(scene, mask, [ob])
                except ValueError as exc:
                    warnings.append(str(exc))
            created.append(ob)
        for w in warnings[:20]:
            self.report({"WARNING"}, w)
        if len(warnings) > 20:
            self.report({"WARNING"}, f"... and {len(warnings) - 20} more import warnings")
        if created:
            select_only(context, *created)
        self.report({"INFO"}, f"Imported {len(created)} shape(s) into artboard '{ab.name}' "
                    f"({result.width:g} x {result.height:g})")
        return {"FINISHED"}


def _menu_import(self, context):
    self.layout.operator(SEAM_OT_svg_import.bl_idname, text="SEAM SVG (.svg)")


def _menu_export(self, context):
    self.layout.operator(SEAM_OT_svg_export.bl_idname, text="SEAM Artboard (.svg)")


classes = (SEAM_OT_svg_export, SEAM_OT_svg_import)


def register():
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)
    bpy.types.TOPBAR_MT_file_export.append(_menu_export)


def unregister():
    for menu, fn in ((bpy.types.TOPBAR_MT_file_import, _menu_import),
                     (bpy.types.TOPBAR_MT_file_export, _menu_export)):
        try:
            menu.remove(fn)
        except (ValueError, AttributeError):
            pass
