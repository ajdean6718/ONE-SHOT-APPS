"""Document health checks and load_post migration (owner: shell).

* ``migrate_scene(scene)`` upgrades ``scene.seam`` data from an older ``schema_version``.
  Steps live in :data:`MIGRATIONS` (``from_version -> fn(scene) -> description``).
* ``health_report(scene)`` collects problems that SEAM cannot repair silently:
  missing link sources, missing / unpacked images, missing fonts, orphan derived
  objects, duplicate SEAM ids, missing canvas, files written by a newer SEAM.
* ``seam.health_check`` reports the result to the user; the load_post handler runs
  migration + check on every file load and keeps a summary for the Document panel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import bpy
from bpy.app.handlers import persistent

from .. import ids
from ..seam_core import SCHEMA_VERSION

SEVERITIES = ("ERROR", "WARNING", "INFO")
_DERIVED_ROLES = ("STROKE", "OUTLINE", "CLIP_RESULT")
_ARTWORK_ROLES = ("SHAPE", "TEXT", "RASTER", "GROUP", "ARTBOARD", "GUIDE")


@dataclass
class Issue:
    severity: str          # ERROR | WARNING | INFO
    code: str              # stable machine-readable key
    message: str           # one actionable sentence
    subject: str = ""      # name of the object / image / font concerned


@dataclass
class HealthReport:
    issues: List[Issue] = field(default_factory=list)
    migrated: List[str] = field(default_factory=list)

    def add(self, severity, code, message, subject=""):
        self.issues.append(Issue(severity, code, message, subject))

    @property
    def problems(self) -> List[Issue]:
        """Issues that need attention (everything except INFO)."""
        return [i for i in self.issues if i.severity != "INFO"]

    def by_code(self, code: str) -> List[Issue]:
        return [i for i in self.issues if i.code == code]

    def summary(self) -> str:
        n = len(self.problems)
        if n == 0:
            return "no problems"
        return f"{n} problem{'s' if n != 1 else ''}"


#: Last report computed in this session (load_post or the operator); read by the Document panel.
last_report: Optional[HealthReport] = None


# ------------------------------------------------------------------ migration

def _migrate_0_to_1(scene) -> str:
    """Pre-release documents: give managed objects a SEAM id and mark an existing canvas ready."""
    n = 0
    for ob in scene.objects:
        if ob.seam.role != "NONE" and not ob.seam.id:
            ids.ensure_id(ob)
            n += 1
    root = next((o for o in scene.objects if o.seam.role == "CANVAS_ROOT"), None)
    if root is not None:
        scene.seam.doc.canvas_ready = True
    return f"assigned {n} missing object ids" if n else "checked object ids"


MIGRATIONS: Dict[int, Callable] = {0: _migrate_0_to_1}


def migrate_scene(scene) -> List[str]:
    """Bring ``scene.seam`` up to :data:`SCHEMA_VERSION`. Returns descriptions of applied steps.

    Scenes never touched by SEAM are left alone (no property is written). A scene written by
    a newer SEAM is not modified (reported by :func:`health_report`).
    """
    seam = getattr(scene, "seam", None)
    if seam is None or not seam.is_property_set("schema_version"):
        return []
    applied = []
    version = seam.schema_version
    while version < SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is not None:
            desc = step(scene)
            applied.append(f"schema {version} -> {version + 1}: {desc}" if desc else f"schema {version} -> {version + 1}")
        version += 1
    if applied or seam.schema_version < SCHEMA_VERSION:
        seam.schema_version = SCHEMA_VERSION
    return applied


# ------------------------------------------------------------------ checks

def _abs(path: str, id_block=None) -> str:
    lib = getattr(id_block, "library", None)
    return bpy.path.abspath(path, library=lib)


def _check_links(scene, rep: HealthReport) -> None:
    for link in scene.seam.links:
        label = link.name or link.id
        if link.state == "DETACHED":
            continue
        if link.source is None:
            rep.add("WARNING", "link_missing_source",
                    f"Link '{label}': its source object is missing. Use Relink to choose a new source or Remove the link.",
                    label)
        if link.target is None:
            rep.add("WARNING", "link_missing_target",
                    f"Link '{label}': its 3D result object is missing. Remove the link or rebuild the result.", label)
        if link.state in ("ERROR", "CONFLICT") and link.source is not None:
            rep.add("INFO", "link_state", f"Link '{label}' is {link.state.lower()}: {link.message or 'see Live Link panel'}",
                    label)


def _raster_images(scene):
    for doc in scene.seam.rasters:
        for layer in doc.layers:
            yield doc, f"layer '{layer.name}'", layer.image, True
            if layer.mask is not None:
                yield doc, f"mask of '{layer.name}'", layer.mask, True
        if doc.has_selection:
            yield doc, "selection", doc.selection, True


def _check_images(scene, rep: HealthReport) -> None:
    seen = set()
    for doc, what, img, required in _raster_images(scene):
        label = f"{doc.name}: {what}"
        if img is None:
            if required:
                rep.add("ERROR", "image_missing",
                        f"Raster document '{doc.name}': the image of {what} is missing. Delete the layer or repaint it.",
                        label)
            continue
        seen.add(img.name)
        if img.packed_file is not None:
            continue
        if img.source == "FILE" and img.filepath:
            if os.path.exists(_abs(img.filepath, img)):
                rep.add("WARNING", "image_unpacked",
                        f"Raster document '{doc.name}': {what} is linked to an external file ({img.filepath}) instead of "
                        f"being packed, so the document depends on that file. Pack the image to make it self-contained.",
                        label)
            else:
                rep.add("ERROR", "image_missing",
                        f"Raster document '{doc.name}': file for {what} not found: {img.filepath}", label)
        elif img.is_dirty:
            rep.add("INFO", "image_pending_pack",
                    f"Raster document '{doc.name}': {what} has unsaved paint; it is packed when the file is saved.", label)
        # a clean generated image holds no painted pixels: nothing can be lost
    for img in bpy.data.images:
        if img.name in seen or img.type != "IMAGE" or img.source != "FILE" or not img.filepath:
            continue
        if img.packed_file is not None or not (img.users > 0 or img.use_fake_user):
            continue
        if not os.path.exists(_abs(img.filepath, img)):
            rep.add("WARNING", "image_missing_file", f"Image '{img.name}': file not found: {img.filepath}", img.name)


def _check_fonts(scene, rep: HealthReport) -> None:
    for font in bpy.data.fonts:
        if font.filepath in ("", "<builtin>") or font.packed_file is not None:
            continue
        if font.users == 0 and not font.use_fake_user:
            continue
        if not os.path.exists(_abs(font.filepath, font)):
            users = sorted(ob.name for ob in bpy.data.objects
                           if ob.type == "FONT" and font in (ob.data.font, ob.data.font_bold, ob.data.font_italic,
                                                             ob.data.font_bold_italic))
            where = f" (used by {', '.join(users[:3])}{'...' if len(users) > 3 else ''})" if users else ""
            rep.add("WARNING", "font_missing",
                    f"Font '{font.name}' not found: {font.filepath}{where}. Text falls back to the default font; "
                    f"load the font again or pack it.", font.name)


def _check_objects(scene, rep: HealthReport) -> None:
    link_ids = {l.id for l in scene.seam.links}
    has_artwork = False
    has_root = False
    for ob in scene.objects:
        role = ob.seam.role
        if role == "CANVAS_ROOT":
            has_root = True
        elif role in _ARTWORK_ROLES:
            has_artwork = True
        if role in _DERIVED_ROLES:
            owner = ids.find_object(ob.seam.owner_id, scene) if ob.seam.owner_id else None
            if owner is None:
                rep.add("WARNING", "orphan_derived",
                        f"'{ob.name}' ({role.lower().replace('_', ' ')}) lost its owner object. "
                        f"Delete it or let SEAM rebuild from its source.", ob.name)
        elif role == "RESULT":
            if not ob.seam.owner_id or ob.seam.owner_id not in link_ids:
                rep.add("WARNING", "orphan_derived",
                        f"3D result '{ob.name}' is not driven by any live link. Delete it or keep it as a normal mesh.",
                        ob.name)
    for sid, objs in ids.duplicate_groups(scene.objects).items():
        rep.add("WARNING", "duplicate_id", f"Objects share the SEAM id {sid}: {', '.join(sorted(o.name for o in objs)[:4])}.",
                sid)
    if has_artwork and not has_root:
        rep.add("ERROR", "canvas_missing", "Artwork exists but the SEAM Canvas object is missing. Run Repair Workspaces or "
                "create a new canvas.", "SEAM Canvas")


def health_report(scene=None) -> HealthReport:
    """Run every check on ``scene`` (default: current scene). Read-only."""
    scene = scene or bpy.context.scene
    rep = HealthReport()
    if scene.seam.schema_version > SCHEMA_VERSION:
        rep.add("WARNING", "schema_newer",
                f"This file was written by a newer SEAM Studio (schema {scene.seam.schema_version}, this build "
                f"understands {SCHEMA_VERSION}). Saving may drop data from newer features.", scene.name)
    _check_links(scene, rep)
    _check_images(scene, rep)
    _check_fonts(scene, rep)
    _check_objects(scene, rep)
    return rep


# ------------------------------------------------------------------ operator

class SEAM_OT_health_check(bpy.types.Operator):
    bl_idname = "seam.health_check"
    bl_label = "Health Check"
    bl_description = ("Check the document for missing link sources, missing or unpacked images, missing fonts "
                      "and orphaned derived objects")
    bl_options = {"REGISTER"}

    max_reports = 12

    def execute(self, context):
        global last_report
        rep = health_report(context.scene)
        last_report = rep
        shown = rep.problems[: self.max_reports]
        for issue in shown:
            prefix = "ERROR: " if issue.severity == "ERROR" else ""
            self.report({"WARNING"}, prefix + issue.message)  # WARNING: an ERROR report would raise in script callers
        extra = len(rep.problems) - len(shown)
        if extra > 0:
            self.report({"WARNING"}, f"...and {extra} more problems (see the system console)")
            for issue in rep.problems[self.max_reports:]:
                print(f"SEAM health [{issue.severity}] {issue.message}")
        if not rep.problems:
            self.report({"INFO"}, "Document health: no problems found")
        return {"FINISHED"}


# ------------------------------------------------------------------ load_post

@persistent
def _on_load_post(*_args):
    """Migrate every scene, then check the current one (summary kept for the Document panel)."""
    global last_report
    migrated = []
    for scene in bpy.data.scenes:
        try:
            migrated += migrate_scene(scene)
        except Exception as exc:  # never break file loading
            print("SEAM: migration failed for scene", scene.name, exc)
    try:
        rep = health_report(bpy.context.scene)
    except Exception as exc:
        print("SEAM: health check failed:", exc)
        return
    rep.migrated = migrated
    last_report = rep
    for step in migrated:
        print("SEAM migrate:", step)
    for issue in rep.problems:
        print(f"SEAM health [{issue.severity}] {issue.message}")


classes = (SEAM_OT_health_check,)


def register():
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)


def unregister():
    global last_report
    while _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    last_report = None
