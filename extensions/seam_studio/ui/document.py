"""New-document setup shared by the template, the quick-start operators and the panels (owner: shell)."""

from __future__ import annotations

from typing import Tuple

import bpy

from ..seam_core.units import UNITS
from . import prefs as prefs_mod

classes = ()

_CONTENT_ROLES = ("SHAPE", "TEXT", "RASTER", "GROUP")


def has_canvas(scene) -> bool:
    return any(ob.seam.role == "CANVAS_ROOT" for ob in scene.objects)


def content_count(scene) -> int:
    """Number of user artwork objects, raster documents and links in the scene."""
    n = sum(1 for ob in scene.objects if ob.seam.role in _CONTENT_ROLES)
    return n + len(scene.seam.rasters) + len(scene.seam.links)


def is_pristine(scene) -> bool:
    """A SEAM canvas (or nothing yet) without any artwork, raster document or link (short-circuits: cheap to poll)."""
    if len(scene.seam.rasters) or len(scene.seam.links):
        return False
    return not any(ob.seam.role in _CONTENT_ROLES for ob in scene.objects)


def _set_mesh_rect(mesh, width: float, height: float) -> None:
    """Artboard plane spans x in [0, w], y in [-h, 0] (architecture §3), vertex order of ``canvas.artboard_mesh``."""
    coords = ((0.0, 0.0, 0.0), (width, 0.0, 0.0), (width, -height, 0.0), (0.0, -height, 0.0))
    if len(mesh.vertices) != 4:
        return
    for v, co in zip(mesh.vertices, coords):
        v.co = co
    mesh.update()


def resize_artboard(artboard, width: float, height: float) -> None:
    """Set the artboard size (props + plane mesh). Idempotent."""
    ab = artboard.seam.artboard
    if ab.width != width:
        ab.width = width
    if ab.height != height:
        ab.height = height
    if artboard.type == "MESH":
        _set_mesh_rect(artboard.data, width, height)


def apply_new_document_defaults(scene, context=None) -> bool:
    """Apply preferences (artboard size, units, dpi) to a pristine document. Returns True when something changed."""
    from ..vector import canvas
    if not is_pristine(scene):
        return False
    prefs = prefs_mod.get(context)
    changed = False
    doc = scene.seam.doc
    units = prefs.default_units if prefs.default_units in UNITS else "PX"
    if doc.units != units:
        doc.units = units
        changed = True
    if abs(doc.dpi - prefs.default_dpi) > 1e-6:
        doc.dpi = prefs.default_dpi
        changed = True
    boards = canvas.artboards(scene)
    if len(boards) == 1:
        w, h = float(prefs.default_canvas_width), float(prefs.default_canvas_height)
        ab = boards[0]
        if ab.seam.artboard.width != w or ab.seam.artboard.height != h:
            resize_artboard(ab, w, h)
            changed = True
    return changed


def set_standard_view_transform(scene) -> None:
    """Colour management ``Standard`` (what you paint/draw is what you get; Filmic/AgX would shift flat colours)."""
    vs = scene.view_settings
    try:
        vs.view_transform = "Standard"
        vs.look = "None"
    except TypeError:  # an unknown enum value on an exotic OCIO config: leave the user's setting alone
        pass


def ensure_document(scene=None, context=None) -> Tuple[object, bool]:
    """Make sure the scene has the SEAM canvas. Returns ``(canvas_root, created)``; new canvases follow preferences."""
    from ..vector import canvas
    scene = scene or bpy.context.scene
    from . import health
    existed = has_canvas(scene)
    root = canvas.ensure_canvas(scene)
    health.stamp_scene(scene)
    if not existed:
        set_standard_view_transform(scene)
        apply_new_document_defaults(scene, context)
    return root, not existed
