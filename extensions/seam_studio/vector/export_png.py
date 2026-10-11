"""Artwork -> render items collection and PNG export (owner: vector-b).

This module is also the shared base layer of the other vector-b modules (pathfinder, clip,
text, io_svg): it knows how to read an artwork object's geometry in *canvas* coordinates,
how its style maps to paints, which objects take part in an export, and in which order.

Public API
----------
``live_canvas_matrix(obj)`` / ``live_canvas_affine(obj)``
    Object -> canvas-local transform computed from ``matrix_basis`` chains so it is correct
    immediately after setting ``location`` (``matrix_world`` is stale until the next depsgraph
    evaluation, and 20000 px of canvas offset would also cost float32 precision).
``object_paths(obj)`` / ``canvas_paths(obj)``
    Bezier paths of a SHAPE (curve) or TEXT (outline) in object-local / canvas coordinates.
``artwork_in_order(scene)``
    Visible artwork bottom -> top (layer, then canvas z), without clip masks.
``collect_render_items(scene, artboard, warnings=None)`` -> list[FillItem | ImageItem]
``render_artboard(scene, artboard, scale, transparent, ...)`` -> uint8 RGBA array (row 0 = top)
``seam.png_export(filepath, artboard, scale, transparent)``

Conventions
-----------
* Paint mapping (shared with SVG export; the viewport gradient texture of vector-a should match):
  LINEAR: direction d = (cos a, sin a) in canvas space (Y up, a = ``gradient_angle``, 0 = left to
  right); the gradient line passes through the bounding-box center and spans the bbox projection
  length ``|w cos a| + |h sin a|``.  RADIAL: center = bbox_min + ``gradient_center`` * bbox_size,
  radius = ``gradient_radius`` * max(w, h).
* Strokes are outlined in canvas pixels (``style.stroke_width`` is canvas px, independent of object scale).
* Text is filled NONZERO (font contours are wound for nonzero), shapes use ``style.fill_rule``.
* Clip masks (objects referenced by another object's ``seam.clip_source``) are never drawn
  themselves; the clipped objects get ``clip=`` rings (see ``clip.py``).
* Raster objects need ``paint.rasterdoc.composite_array(doc)``; when it is missing the raster is
  skipped and a warning is appended to ``warnings``.  The composite is stretched over the bounds of the
  raster plane (the plane owns the extent, exactly as the viewport material maps the UVs onto it; paint
  creates the plane with the aspect ratio of the document).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from bpy.props import BoolProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Vector

from ..bridge import curve_paths, matrix_to_affine, text_paths, transform_bezier_paths
from ..seam_core import geometry, png, raster as raster_core, render2d, stroke
from ..seam_core.geometry import BezierPath
from . import canvas

EXPORT_RESOLUTION = 16        # samples per Bezier segment when flattening for raster export
DRAW_ROLES = ("SHAPE", "TEXT", "RASTER")


# ------------------------------------------------------------------ transforms

def _local_step(obj) -> Matrix:
    return obj.matrix_parent_inverse @ obj.matrix_basis


def live_canvas_matrix(obj) -> Matrix:
    """4x4 matrix mapping ``obj`` local space into canvas-local space (up to the canvas root)."""
    m = Matrix.Identity(4)
    cur = obj
    while cur is not None and cur.seam.role != "CANVAS_ROOT":
        m = _local_step(cur) @ m
        cur = cur.parent
    if cur is None and obj.seam.role != "CANVAS_ROOT":
        # not parented to the canvas: interpret as world space and express relative to the root
        root = canvas.canvas_root(bpy.context.scene)
        if root is not None:
            rm = Matrix.Identity(4)
            c = root
            while c is not None:
                rm = _local_step(c) @ rm
                c = c.parent
            m = rm.inverted_safe() @ m
    return m


def live_canvas_affine(obj) -> geometry.Affine:
    return matrix_to_affine(live_canvas_matrix(obj))


def live_canvas_z(obj) -> float:
    return float(live_canvas_matrix(obj)[2][3])


# ------------------------------------------------------------------ object geometry

def object_paths(obj) -> List[BezierPath]:
    """Object-local Bezier paths of a SHAPE (curve) or TEXT (outline) object."""
    if obj is None:
        return []
    if obj.type == "CURVE":
        return curve_paths(obj)
    if obj.type == "FONT":
        return text_paths(obj)
    return []


def canvas_paths(obj) -> List[BezierPath]:
    """Bezier paths of ``obj`` in canvas-local coordinates."""
    return transform_bezier_paths(object_paths(obj), live_canvas_affine(obj))


def fill_rings(paths: Sequence[BezierPath], resolution: int = EXPORT_RESOLUTION) -> List[List[Tuple[float, float]]]:
    """Flattened rings of ``paths`` for filling (every spline with >= 3 distinct points counts as closed)."""
    rings = []
    for path in paths:
        if len(path.points) < 2:
            continue
        ring = geometry.flatten_path(path, resolution)
        ring = [(float(p[0]), float(p[1])) for p in ring]
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def stroke_polylines(paths: Sequence[BezierPath], resolution: int = EXPORT_RESOLUTION):
    """[(polyline, closed)] of ``paths``; closed paths are returned without the repeated first point."""
    out = []
    for path in paths:
        if len(path.points) < 2:
            continue
        pl = geometry.flatten_path(path, resolution)
        pl = [(float(p[0]), float(p[1])) for p in pl]
        if len(pl) >= 2:
            out.append((pl, bool(path.closed)))
    return out


def resolve_rings(rings, fill_rule: str):
    """Even-odd equivalent ring set (holes oriented) for ``rings`` interpreted with ``fill_rule``.

    EVENODD rings are returned untouched; NONZERO rings go through one boolean union pass so that
    the result can be fed into EVENODD-based kernels. Falls back to the input when the boolean
    kernel is unavailable.
    """
    from ..seam_core import boolean2d
    if fill_rule != "NONZERO" or not rings:
        return [list(r) for r in rings]
    try:
        shapes = boolean2d.union_all([rings], "NONZERO")
    except boolean2d.BooleanUnavailable:
        return [list(r) for r in rings]
    out = []
    for outer, holes in shapes:
        out.append(list(outer))
        out.extend(list(h) for h in holes)
    return out


# ------------------------------------------------------------------ style helpers

_STYLE_SKIP = {"rna_type", "name"}


def copy_style(src, dst) -> None:
    """Copy every ``SEAM_PG_Style`` field (including gradient stops) from ``src`` to ``dst``."""
    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident in _STYLE_SKIP:
            continue
        if ident == "gradient_stops":
            dst.gradient_stops.clear()
            for st in src.gradient_stops:
                n = dst.gradient_stops.add()
                n.position = st.position
                n.color = tuple(st.color)
            continue
        val = getattr(src, ident)
        if hasattr(val, "__len__") and not isinstance(val, str):
            val = tuple(val)
        setattr(dst, ident, val)


def rgba(color) -> Tuple[float, float, float, float]:
    c = tuple(float(v) for v in color)
    return (c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0)


def gradient_stops(style) -> List[Tuple[float, Tuple[float, float, float, float]]]:
    return sorted(((float(s.position), rgba(s.color)) for s in style.gradient_stops), key=lambda s: s[0])


def gradient_geometry(style, bounds: Tuple[float, float, float, float]):
    """('LINEAR', start, end) or ('RADIAL', center, radius) in canvas coords for ``bounds`` (minx, miny, maxx, maxy)."""
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    if style.fill_type == "LINEAR":
        a = float(style.gradient_angle)
        dx, dy = math.cos(a), math.sin(a)
        half = 0.5 * (abs(w * dx) + abs(h * dy))
        cx, cy = x0 + 0.5 * w, y0 + 0.5 * h
        return "LINEAR", (cx - dx * half, cy - dy * half), (cx + dx * half, cy + dy * half)
    gc = style.gradient_center
    center = (x0 + float(gc[0]) * w, y0 + float(gc[1]) * h)
    return "RADIAL", center, max(1e-6, float(style.gradient_radius) * max(w, h))


def fill_paint(style, bounds):
    """render2d paint for ``style``'s fill over ``bounds`` (solid when fewer than two gradient stops exist)."""
    stops = gradient_stops(style)
    if style.fill_type == "SOLID" or len(stops) < 2:
        return render2d.SolidPaint(rgba(style.fill_color))
    rstops = [render2d.GradientStop(p, c) for p, c in stops]
    kind, a, b = gradient_geometry(style, bounds)
    if kind == "LINEAR":
        return render2d.LinearGradientPaint(a, b, rstops)
    return render2d.RadialGradientPaint(a, b, rstops)


def rings_bounds(rings) -> Optional[Tuple[float, float, float, float]]:
    pts = [p for r in rings for p in r]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


# ------------------------------------------------------------------ artwork selection

def clip_masks(scene) -> set:
    """Names of objects used as clip source by any other object."""
    out = set()
    for ob in scene.objects:
        src = ob.seam.clip_source
        if src is not None and src is not ob:
            out.add(src.name)
    return out


def is_visible(obj) -> bool:
    try:
        if obj.hide_viewport:
            return False
        return bool(obj.visible_get())
    except (RuntimeError, ReferenceError):
        return False


def artwork_in_order(scene, roles: Sequence[str] = DRAW_ROLES, include_hidden: bool = False,
                     include_masks: bool = False) -> List[bpy.types.Object]:
    """Artwork objects bottom -> top by (layer, canvas z, name). Hidden objects and clip masks are dropped by default."""
    layer_of: Dict[str, int] = {}
    for i, col in enumerate(canvas.layers(scene)):
        for ob in col.all_objects:
            layer_of.setdefault(ob.name, i)
    masks = set() if include_masks else clip_masks(scene)
    out = []
    for ob in scene.objects:
        if ob.seam.role not in roles or ob.name in masks:
            continue
        if not include_hidden and not is_visible(ob):
            continue
        out.append((layer_of.get(ob.name, 0), live_canvas_z(ob), ob.name, ob))
    out.sort(key=lambda t: t[:3])
    return [t[3] for t in out]


def resolve_artboard(scene, ident: str = ""):
    """Artboard object by SEAM id or object name; empty -> active artboard."""
    abs_ = canvas.artboards(scene)
    if ident:
        for ob in abs_:
            if ob.seam.id == ident or ob.name == ident:
                return ob
        return None
    return canvas.active_artboard(scene)


def artboard_rect(artboard) -> Tuple[float, float, float, float]:
    """(x_left, y_top, width, height) of an artboard in canvas-local coordinates."""
    m = live_canvas_matrix(artboard)
    return (float(m[0][3]), float(m[1][3]), float(artboard.seam.artboard.width), float(artboard.seam.artboard.height))


def clip_rings_for(obj, resolution: int = EXPORT_RESOLUTION):
    """Clip rings (canvas coords, even-odd) of ``obj``'s clip mask, or None when unclipped."""
    mask = obj.seam.clip_source
    if mask is None or mask is obj:
        return None
    rings = fill_rings(canvas_paths(mask), resolution)
    return resolve_rings(rings, mask.seam.style.fill_rule)


# ------------------------------------------------------------------ render items

def _warn(warnings, msg):
    if warnings is not None:
        warnings.append(msg)


def stroke_rings(paths, style, resolution: int = EXPORT_RESOLUTION):
    """Outline rings (all CCW, NONZERO union = stroked area) for ``style``'s stroke over canvas-space ``paths``."""
    rings = []
    width = float(style.stroke_width)
    if width <= 0.0:
        return rings
    for pl, closed in stroke_polylines(paths, resolution):
        for ring in stroke.stroke_outline(pl, closed, width, style.stroke_cap, style.stroke_join,
                                          float(style.miter_limit)):
            if len(ring) < 3:
                continue
            if geometry.signed_area(ring) < 0.0:   # nonzero union needs one winding direction
                ring = list(reversed(ring))
            rings.append(ring)
    return rings


def _fill_minus_stroke(fill, fill_rule: str, stroke_rings_):
    """Fill rings with the stroked area removed (even-odd ring set), or None when that cannot be computed."""
    from ..seam_core import boolean2d
    try:
        shapes = boolean2d.boolean(resolve_rings(fill, fill_rule), resolve_rings(stroke_rings_, "NONZERO"),
                                   "DIFFERENCE", "EVENODD")
    except (boolean2d.BooleanUnavailable, ValueError):
        return None
    out = []
    for outer, holes in shapes:
        out.append(list(outer))
        out.extend(list(h) for h in holes)
    return out


def object_items(obj, resolution: int = EXPORT_RESOLUTION, warnings=None) -> list:
    """Render items (fill, then stroke) for one SHAPE or TEXT object.

    Object opacity is *group* opacity, like the SVG ``opacity`` attribute: where stroke and fill overlap
    the stroke simply replaces the fill, it is not blended on top of it.  For ``opacity < 1`` the stroked
    area is therefore cut out of the fill.  (The fill and stroke then share an anti-aliased edge, which
    is a few percent lighter than the group result on that one pixel line.)
    """
    style = obj.seam.style
    paths = canvas_paths(obj)
    if not paths:
        return []
    clip = clip_rings_for(obj, resolution)
    opacity = float(style.opacity)
    fill_items = []
    stroke_item = None
    fill_rings_ = []
    fill_rule = "NONZERO" if obj.type == "FONT" else style.fill_rule
    if style.fill_enabled:
        fill_rings_ = fill_rings(paths, resolution)
        bounds = rings_bounds(fill_rings_)
        if fill_rings_ and bounds is not None:
            fill_items.append(render2d.FillItem(rings=fill_rings_, paint=fill_paint(style, bounds),
                                                fill_rule=fill_rule, opacity=opacity, clip=clip))
    if style.stroke_enabled and style.stroke_width > 0.0:
        rings = stroke_rings(paths, style, resolution)
        if rings:
            stroke_item = render2d.FillItem(rings=rings, paint=render2d.SolidPaint(rgba(style.stroke_color)),
                                            fill_rule="NONZERO", opacity=opacity, clip=clip)
    if fill_items and stroke_item is not None and opacity < 1.0 - 1e-6:
        cut = _fill_minus_stroke(fill_rings_, fill_rule, stroke_item.rings)
        if cut is not None:
            fill_items[0].rings = cut
            fill_items[0].fill_rule = "EVENODD"
            if not cut:
                fill_items = []
    return fill_items + ([stroke_item] if stroke_item is not None else [])


def raster_image_array(scene, obj):
    """Linear H x W x 4 composite of a RASTER object's document, None if it has none (may raise ImportError/AttributeError)."""
    from ..paint import rasterdoc
    fn = getattr(rasterdoc, "composite_array")
    rid = obj.seam.raster_id
    doc = next((d for d in scene.seam.rasters if d.id == rid), None)
    if doc is None:
        return None
    return fn(doc)


def raster_bounds(obj) -> Optional[Tuple[float, float, float, float]]:
    """Canvas-space (minx, miny, maxx, maxy) of a RASTER plane."""
    m = live_canvas_matrix(obj)
    pts = [m @ Vector(c) for c in obj.bound_box]
    if not pts:
        return None
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def raster_item(scene, obj, resolution: int = EXPORT_RESOLUTION, warnings=None):
    try:
        arr = raster_image_array(scene, obj)
    except (ImportError, AttributeError) as exc:
        _warn(warnings, f"Raster '{obj.name}' skipped: raster compositor unavailable ({exc})")
        return None
    except NotImplementedError:
        _warn(warnings, f"Raster '{obj.name}' skipped: raster compositor not implemented")
        return None
    if arr is None:
        _warn(warnings, f"Raster '{obj.name}' skipped: no raster document")
        return None
    b = raster_bounds(obj)
    if b is None:
        return None
    return render2d.ImageItem(rgba=arr, x=b[0], y_top=b[3], width=b[2] - b[0], height=b[3] - b[1],
                              opacity=float(obj.seam.style.opacity), clip=clip_rings_for(obj, resolution))


def collect_render_items(scene, artboard=None, warnings=None, resolution: int = EXPORT_RESOLUTION) -> list:
    """FillItems / ImageItems for all visible artwork, bottom -> top, in canvas coordinates.

    ``artboard`` is only used to cull unclipped fills that do not touch it (None = no culling).
    The artboard background is *not* included (see ``render_artboard``).
    """
    view = None
    if artboard is not None:
        ax, ay, aw, ah = artboard_rect(artboard)
        view = (ax, ay - ah, ax + aw, ay)
    items = []
    for obj in artwork_in_order(scene):
        if obj.seam.role == "RASTER":
            it = raster_item(scene, obj, resolution, warnings)
            new = [it] if it is not None else []
        else:
            new = object_items(obj, resolution, warnings)
        for it in new:
            if view is not None and it.clip is None and isinstance(it, render2d.FillItem):
                b = rings_bounds(it.rings)
                if b is not None and (b[2] < view[0] or b[0] > view[2] or b[3] < view[1] or b[1] > view[3]):
                    continue
            items.append(it)
    return items


def background_color(artboard) -> Tuple[float, float, float, float]:
    ab = artboard.seam.artboard
    return rgba(ab.background_color) if ab.background_enabled else (1.0, 1.0, 1.0, 1.0)


MAX_PIXELS = 100_000_000


def render_artboard(scene, artboard, scale: float = 1.0, transparent: bool = False, supersample: int = 4,
                    warnings=None):
    """Render ``artboard`` to a uint8 sRGB straight-alpha array (H x W x 4, row 0 = top)."""
    ax, ay, aw, ah = artboard_rect(artboard)
    width = max(1, int(round(aw * scale)))
    height = max(1, int(round(ah * scale)))
    if width * height > MAX_PIXELS:
        raise ValueError(f"Export size {width}x{height} px is too large "
                         f"(limit {MAX_PIXELS // 1_000_000} megapixels); lower the scale")
    items = collect_render_items(scene, artboard, warnings)
    bg = None if transparent else background_color(artboard)
    img = render2d.render(items, width, height, (ax, ay, ax + aw, ay - ah), supersample=supersample, background=bg)
    return raster_core.to_uint8_srgb(img)


# ------------------------------------------------------------------ operator

class SEAM_OT_png_export(bpy.types.Operator, ExportHelper):
    """Render an artboard to a PNG file with SEAM's deterministic 2D renderer (3D objects are not rendered)"""
    bl_idname = "seam.png_export"
    bl_label = "Export Artboard as PNG"
    bl_options = {"REGISTER"}

    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png", options={"HIDDEN"})
    artboard: StringProperty(name="Artboard", default="",
                             description="Artboard name or SEAM id (empty = active artboard)")
    scale: FloatProperty(name="Scale", default=1.0, min=0.01, max=16.0,
                         description="Output pixels per canvas px")
    transparent: BoolProperty(name="Transparent", default=False,
                              description="Skip the artboard background (otherwise it is rendered; white if disabled)")

    def invoke(self, context, event):
        ab = resolve_artboard(context.scene, self.artboard)
        if ab is None:
            self.report({"ERROR"}, "No artboard to export")
            return {"CANCELLED"}
        self.scale = ab.seam.artboard.export_scale
        if not self.filepath:
            self.filepath = ab.name + ".png"
        return super().invoke(context, event)

    def execute(self, context):
        scene = context.scene
        ab = resolve_artboard(scene, self.artboard)
        if ab is None:
            self.report({"ERROR"}, f"Artboard '{self.artboard}' not found" if self.artboard else "No artboard to export")
            return {"CANCELLED"}
        path = bpy.path.abspath(self.filepath)
        if not path.lower().endswith(".png"):
            path += ".png"
        warnings: List[str] = []
        try:
            from ..engine import core as engine
            engine.flush(scene)  # derived data (outlines, clip results) must be current
            pixels = render_artboard(scene, ab, self.scale, self.transparent, warnings=warnings)
            png.write_png(path, pixels)
        except NotImplementedError as exc:
            self.report({"ERROR"}, f"PNG export unavailable: {exc}")
            return {"CANCELLED"}
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, f"PNG export failed: {exc}")
            return {"CANCELLED"}
        for w in warnings[:10]:
            self.report({"WARNING"}, w)
        self.report({"INFO"}, f"Exported {pixels.shape[1]}x{pixels.shape[0]} px to {os.path.basename(path)}")
        return {"FINISHED"}


def _menu_export(self, context):
    self.layout.operator(SEAM_OT_png_export.bl_idname, text="SEAM Artboard (.png)")


classes = (SEAM_OT_png_export,)


def register():
    bpy.types.TOPBAR_MT_file_export.append(_menu_export)


def unregister():
    try:
        bpy.types.TOPBAR_MT_file_export.remove(_menu_export)
    except (ValueError, AttributeError):
        pass
