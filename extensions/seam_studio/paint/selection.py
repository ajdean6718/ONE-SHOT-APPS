"""Selections (rect/ellipse/lasso), stencil mask sync, overlay (owner: paint).

A selection is a float mask (1 = selected) stored in the 8-bit Non-Color image ``doc.selection`` (white =
selected).  When a selection exists it is wired to the native stencil of Texture Paint
(``tool_settings.image_paint.use_stencil_layer`` / ``stencil_image`` / ``invert_stencil``, sampled through the
canvas mesh's second UV map) so native brushes only paint inside it.  Verified with real native strokes in the
4.0.2 GUI: with ``invert_stencil`` False Blender paints where the stencil is black, hence the inversion.  ``doc.has_selection == False`` means "no selection" (everything editable).
Coordinates are canvas pixels, (0, 0) = top-left corner of the top-left pixel, y down.
"""

from typing import Dict, Optional, Tuple

import bpy
import numpy as np
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty
from mathutils import Vector

from ..seam_core import raster as R
from . import rasterdoc as rd

SELECTION_OP_ITEMS = [(m, m.title(), d) for m, d in (
    ("REPLACE", "Replace the selection"), ("ADD", "Add to the selection"),
    ("SUBTRACT", "Subtract from the selection"), ("INTERSECT", "Intersect with the selection"))]



# --------------------------------------------------------------------------- data

def _ensure_image(doc):
    w, h = rd.doc_size(doc)
    img = doc.selection
    if img is not None and (int(img.size[0]), int(img.size[1])) != (w, h):
        rd._remove_image(img)
        rd._applied.pop((doc.id, "sel"), None)
        img = None
    if img is None:
        img = rd._new_image(rd._image_name(doc.name, "selection"), w, h, "Non-Color", (0, 0, 0, 1))
        doc.selection = img
    return img


def selection_array(doc) -> Optional[np.ndarray]:
    """Selection as float ``H x W`` 0..1, or None when there is no selection."""
    if not doc.has_selection or doc.selection is None:
        return None
    return rd.read_image_srgb8(doc.selection)[..., 0].astype(np.float32) / 255.0


def _write(doc, mask: np.ndarray) -> None:
    img = _ensure_image(doc)
    v = np.rint(np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    a8 = np.empty(v.shape + (4,), np.uint8)
    a8[..., 0] = a8[..., 1] = a8[..., 2] = v
    a8[..., 3] = 255
    rd.write_image_srgb8(img, a8)
    rd.commit_pixels(doc, "sel", a8)


def set_selection(doc, mask: Optional[np.ndarray], op: str = "REPLACE") -> bool:
    """Combine ``mask`` (float H x W) with the current selection using ``op``; ``None`` clears. Returns has_selection."""
    if op not in R.SELECTION_OPS:
        raise ValueError(f"unknown selection op {op!r}")
    w, h = rd.doc_size(doc)
    if mask is None:
        return clear_selection(doc)
    mask = np.asarray(mask, np.float32)
    if mask.shape != (h, w):
        raise ValueError(f"selection mask {mask.shape} does not match document {h}x{w}")
    cur = selection_array(doc)
    if op == "REPLACE":
        new = R.combine_selection(None, mask, "REPLACE")
    else:
        base = cur if cur is not None else np.zeros((h, w), np.float32)
        new = R.combine_selection(base, mask, op)
    new = np.clip(np.asarray(new, np.float32), 0.0, 1.0)
    if float(new.max(initial=0.0)) <= 0.0:
        return clear_selection(doc)
    _write(doc, new)
    doc.has_selection = True
    sync_stencil(doc)
    return True


def clear_selection(doc) -> bool:
    """Remove the selection (nothing selected == everything editable)."""
    if doc.selection is not None and doc.has_selection:
        _write(doc, np.zeros(rd.doc_size(doc)[::-1], np.float32))
    doc.has_selection = False
    sync_stencil(doc)
    return False


def select_all(doc) -> None:
    w, h = rd.doc_size(doc)
    set_selection(doc, np.ones((h, w), np.float32), "REPLACE")


def invert_selection(doc) -> None:
    cur = selection_array(doc)
    if cur is None:
        raise ValueError("no selection to invert")
    set_selection(doc, 1.0 - cur, "REPLACE")


def select_rect(doc, x0, y0, x1, y1, op: str = "REPLACE", antialias: bool = True) -> bool:
    w, h = rd.doc_size(doc)
    return set_selection(doc, R.rect_mask(h, w, x0, y0, x1, y1, antialias), op)


def select_ellipse(doc, x0, y0, x1, y1, op: str = "REPLACE", antialias: bool = True) -> bool:
    w, h = rd.doc_size(doc)
    return set_selection(doc, R.ellipse_mask(h, w, x0, y0, x1, y1, antialias), op)


def select_polygon(doc, points, op: str = "REPLACE", antialias: bool = True) -> bool:
    if len(points) < 3:
        raise ValueError("a lasso needs at least 3 points")
    w, h = rd.doc_size(doc)
    return set_selection(doc, R.polygon_mask(h, w, [(float(x), float(y)) for x, y in points], antialias), op)


def selection_bounds(doc) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) pixel bounds of the non-zero selection, or None."""
    m = selection_array(doc)
    if m is None:
        return None
    ys, xs = np.nonzero(m > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def effective_mask(doc) -> np.ndarray:
    """Selection mask, or all ones when there is no selection (what fill/clear operate on)."""
    m = selection_array(doc)
    if m is None:
        w, h = rd.doc_size(doc)
        return np.ones((h, w), np.float32)
    return m


# --------------------------------------------------------------------------- stencil

def sync_stencil(doc) -> None:
    """Wire the selection to the native image-paint stencil (or switch the stencil off)."""
    scene = rd.doc_scene(doc)
    ip = scene.tool_settings.image_paint
    if doc.has_selection and doc.selection is not None:
        ip.stencil_image = doc.selection
        # Blender's project paint treats the stencil as "paint where it is BLACK" unless invert is set
        # (measured on 4.0.2, GUI): the selection is white inside, so invert it.
        ip.invert_stencil = True
        ip.use_stencil_layer = True
        obj = doc.canvas
        if obj is not None and obj.type == "MESH":
            me = obj.data
            idx = me.uv_layers.find(rd.STENCIL_UV)
            if idx >= 0 and me.uv_layer_stencil_index != idx:
                me.uv_layer_stencil_index = idx
    else:
        cur = ip.stencil_image
        other = next((d for d in scene.seam.rasters if d.selection is not None and d.selection == cur), None)
        # switch it off unless it still serves another document that has a selection
        if cur is None or (other is not None and not other.has_selection) or cur == doc.selection:
            ip.use_stencil_layer = False
            ip.invert_stencil = False


# --------------------------------------------------------------------------- outline (pure + GUI overlay)

def outline_segments(mask: np.ndarray, threshold: float = 0.5, max_segments: int = 250_000) -> np.ndarray:
    """Boundary of ``mask >= threshold`` as unit segments ``N x 4`` float32 (x0, y0, x1, y1), pixel coords, y down.

    Falls back to the bounding box outline when the boundary is too complex to draw."""
    b = np.asarray(mask) >= threshold
    h, w = b.shape
    p = np.zeros((h + 2, w + 2), bool)
    p[1:-1, 1:-1] = b
    # horizontal edges lie on lines y = r (r = 0..h) between pixel rows r-1 and r
    hy, hx = np.nonzero(p[:-1, 1:-1] != p[1:, 1:-1])
    # vertical edges lie on lines x = c (c = 0..w) between pixel columns c-1 and c
    vy, vx = np.nonzero(p[1:-1, :-1] != p[1:-1, 1:])
    n = len(hx) + len(vx)
    if n == 0:
        return np.zeros((0, 4), np.float32)
    if n > max_segments:
        ys, xs = np.nonzero(b)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        return np.array([[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]], np.float32)
    seg = np.empty((n, 4), np.float32)
    seg[:len(hx)] = np.stack([hx, hy, hx + 1, hy], axis=1)
    seg[len(hx):] = np.stack([vx, vy, vx, vy + 1], axis=1)
    return seg


_draw_handle = None
_outline_cache: Dict[str, tuple] = {}


def _draw_outline():  # pragma: no cover - GUI only
    import gpu
    from gpu_extras.batch import batch_for_shader
    ctx = bpy.context
    try:
        scene = ctx.scene
        doc = rd.active_doc(scene)
    except Exception:
        return
    if doc is None or not doc.has_selection or doc.canvas is None or doc.canvas.name not in scene.objects:
        return
    key = (int(doc.sel_rev), doc.width, doc.height)  # sel_rev is saved and restored by undo
    cached = _outline_cache.get(doc.id)
    if cached is None or cached[0] != key:
        seg = outline_segments(selection_array(doc))
        _outline_cache[doc.id] = cached = (key, seg)
    seg = cached[1]
    if not len(seg):
        return
    m = np.array(doc.canvas.matrix_world, np.float32)
    pts = np.empty((len(seg) * 2, 4), np.float32)
    pts[0::2] = np.stack([seg[:, 0], -seg[:, 1], np.full(len(seg), 0.02), np.ones(len(seg))], axis=1)
    pts[1::2] = np.stack([seg[:, 2], -seg[:, 3], np.full(len(seg), 0.02), np.ones(len(seg))], axis=1)
    world = (pts @ m.T)[:, :3]
    parity = (((seg[:, 0] + seg[:, 1]) // 4) % 2).astype(bool)
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(1.5)
    gpu.state.depth_test_set("NONE")
    for sel, color in ((parity, (1.0, 1.0, 1.0, 1.0)), (~parity, (0.0, 0.0, 0.0, 1.0))):
        idx = np.nonzero(sel)[0]
        if not len(idx):
            continue
        verts = np.empty((len(idx) * 2, 3), np.float32)
        verts[0::2] = world[idx * 2]
        verts[1::2] = world[idx * 2 + 1]
        batch = batch_for_shader(shader, "LINES", {"pos": verts})
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")


# --------------------------------------------------------------------------- operators

def _doc_poll(cls, context, need_selection: bool = False) -> bool:
    doc = rd.active_doc(context.scene)
    if doc is None:
        cls.poll_message_set("No raster document: create one with New Raster Document")
        return False
    if need_selection and not doc.has_selection:
        cls.poll_message_set("There is no selection")
        return False
    return True


class _SelectBase:
    bl_options = {"REGISTER", "UNDO"}
    mode: EnumProperty(name="Mode", items=SELECTION_OP_ITEMS, default="REPLACE")
    antialias: BoolProperty(name="Anti-alias", default=True)

    @classmethod
    def poll(cls, context):
        return _doc_poll(cls, context)

    def _doc(self, context):
        return rd.active_doc(context.scene)


def _event_to_pixel(context, event, doc):  # pragma: no cover - GUI only
    from bpy_extras import view3d_utils
    from mathutils import geometry
    region, rv3d = context.region, context.region_data
    coord = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    mw = doc.canvas.matrix_world
    plane_co = mw.translation
    plane_no = mw.to_3x3() @ Vector((0, 0, 1))
    hit = geometry.intersect_line_plane(origin, origin + direction, plane_co, plane_no)
    if hit is None:
        return None
    local = mw.inverted() @ hit
    return float(local.x), float(-local.y)


class _DragSelect(_SelectBase):
    """Shared modal drag for the shape selection tools (GUI)."""

    _handle = None
    _p0 = None
    _p1 = None
    _pts = None

    def invoke(self, context, event):  # pragma: no cover - GUI only
        doc = self._doc(context)
        if (event is None or context.area is None or context.area.type != "VIEW_3D" or context.region_data is None
                or doc.canvas is None):
            return self.execute(context)
        self._p0 = self._p1 = None
        self._pts = []
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set("CROSSHAIR")
        context.workspace.status_text_set(f"{self.bl_label}: drag on the canvas, Esc to cancel")
        return {"RUNNING_MODAL"}

    def _finish(self, context):  # pragma: no cover - GUI only
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)

    def modal(self, context, event):  # pragma: no cover - GUI only
        doc = self._doc(context)
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._finish(context)
            return {"CANCELLED"}
        pix = _event_to_pixel(context, event, doc)
        if event.type == "LEFTMOUSE" and event.value == "PRESS" and pix is not None:
            self._p0 = self._p1 = pix
            self._pts = [pix]
        elif event.type == "MOUSEMOVE" and self._p0 is not None and pix is not None:
            self._p1 = pix
            if self._pts is not None:
                self._pts.append(pix)
        elif event.type == "LEFTMOUSE" and event.value == "RELEASE" and self._p0 is not None:
            if pix is not None:
                self._p1 = pix
            self._finish(context)
            return self._commit(context)
        return {"RUNNING_MODAL"}

    def _commit(self, context):  # pragma: no cover - GUI only
        raise NotImplementedError


class SEAM_OT_raster_select_rect(_DragSelect, bpy.types.Operator):
    bl_idname = "seam.raster_select_rect"
    bl_label = "Rectangular Selection"
    bl_description = "Select a rectangle of the raster canvas (pixel coordinates; drag in the viewport)"

    x0: FloatProperty(name="X0", default=0.0, description="Left edge in canvas pixels")
    y0: FloatProperty(name="Y0", default=0.0, description="Top edge in canvas pixels (down)")
    x1: FloatProperty(name="X1", default=64.0, description="Right edge in canvas pixels")
    y1: FloatProperty(name="Y1", default=64.0, description="Bottom edge in canvas pixels (down)")

    def _commit(self, context):  # pragma: no cover - GUI only
        self.x0, self.y0 = self._p0
        self.x1, self.y1 = self._p1
        return self.execute(context)

    def execute(self, context):
        doc = self._doc(context)
        select_rect(doc, self.x0, self.y0, self.x1, self.y1, self.mode, self.antialias)
        return {"FINISHED"}


class SEAM_OT_raster_select_ellipse(_DragSelect, bpy.types.Operator):
    bl_idname = "seam.raster_select_ellipse"
    bl_label = "Elliptical Selection"
    bl_description = "Select an ellipse inscribed in a rectangle of the raster canvas (pixel coordinates)"

    x0: FloatProperty(name="X0", default=0.0)
    y0: FloatProperty(name="Y0", default=0.0)
    x1: FloatProperty(name="X1", default=64.0)
    y1: FloatProperty(name="Y1", default=64.0)

    def _commit(self, context):  # pragma: no cover - GUI only
        self.x0, self.y0 = self._p0
        self.x1, self.y1 = self._p1
        return self.execute(context)

    def execute(self, context):
        doc = self._doc(context)
        select_ellipse(doc, self.x0, self.y0, self.x1, self.y1, self.mode, self.antialias)
        return {"FINISHED"}


class SEAM_OT_raster_select_lasso(_DragSelect, bpy.types.Operator):
    bl_idname = "seam.raster_select_lasso"
    bl_label = "Lasso Selection"
    bl_description = "Select a free-form polygon of the raster canvas (pixel coordinates; drag in the viewport)"

    path: CollectionProperty(type=bpy.types.OperatorMousePath)

    def _commit(self, context):  # pragma: no cover - GUI only
        self.path.clear()
        pts = self._pts[::max(1, len(self._pts) // 400)] if len(self._pts) > 400 else self._pts
        for p in pts:
            item = self.path.add()
            item.loc = p
        return self.execute(context)

    def execute(self, context):
        doc = self._doc(context)
        pts = [(float(p.loc[0]), float(p.loc[1])) for p in self.path]
        if len(pts) < 3:
            self.report({"WARNING"}, "Lasso needs at least 3 points")
            return {"CANCELLED"}
        select_polygon(doc, pts, self.mode, self.antialias)
        return {"FINISHED"}


class SEAM_OT_raster_select_all(bpy.types.Operator):
    bl_idname = "seam.raster_select_all"
    bl_label = "Select All"
    bl_description = "Select the whole raster canvas"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _doc_poll(cls, context)

    def execute(self, context):
        select_all(rd.active_doc(context.scene))
        return {"FINISHED"}


class SEAM_OT_raster_select_none(bpy.types.Operator):
    bl_idname = "seam.raster_select_none"
    bl_label = "Deselect"
    bl_description = "Clear the selection (everything becomes editable again)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _doc_poll(cls, context, need_selection=True)

    def execute(self, context):
        clear_selection(rd.active_doc(context.scene))
        return {"FINISHED"}


class SEAM_OT_raster_select_invert(bpy.types.Operator):
    bl_idname = "seam.raster_select_invert"
    bl_label = "Invert Selection"
    bl_description = "Swap selected and unselected pixels"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _doc_poll(cls, context, need_selection=True)

    def execute(self, context):
        invert_selection(rd.active_doc(context.scene))
        return {"FINISHED"}


classes = (SEAM_OT_raster_select_rect, SEAM_OT_raster_select_ellipse, SEAM_OT_raster_select_lasso,
           SEAM_OT_raster_select_all, SEAM_OT_raster_select_none, SEAM_OT_raster_select_invert)


def register():
    global _draw_handle
    if not bpy.app.background and _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_outline, (), "WINDOW", "POST_VIEW")


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        except Exception:
            pass
        _draw_handle = None
    _outline_cache.clear()
