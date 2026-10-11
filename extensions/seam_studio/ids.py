"""SEAM id helpers on the Blender side (owner: integrator)."""

import bpy

from .seam_core.ids import is_valid_id, new_id


def ensure_id(obj) -> str:
    """Return the object's SEAM id, assigning a new one when missing/invalid."""
    seam = obj.seam
    if not is_valid_id(seam.id):
        seam.id = new_id()
    return seam.id


def find_object(seam_id: str, scene=None):
    """Find an object by SEAM id (scene objects first, then all of bpy.data)."""
    if not seam_id:
        return None
    pool = scene.objects if scene is not None else ()
    for ob in pool:
        if ob.seam.id == seam_id:
            return ob
    for ob in bpy.data.objects:
        if ob.seam.id == seam_id:
            return ob
    return None


def duplicate_groups(objects):
    """Map seam id -> list of objects sharing it (only ids used more than once)."""
    seen = {}
    for ob in objects:
        sid = ob.seam.id
        if sid:
            seen.setdefault(sid, []).append(ob)
    return {k: v for k, v in seen.items() if len(v) > 1}
