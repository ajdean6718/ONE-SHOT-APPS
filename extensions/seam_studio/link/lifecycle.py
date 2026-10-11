"""Duplicate/delete/missing-source handling (engine update hook) (owner: link).

Runs at the start of every depsgraph update (``engine.add_update_hook``):

* deleted source  -> link goes to MISSING_SOURCE, the result is kept, the pointer is released
  (``source_id`` is kept so the link can re-point when the source comes back);
* deleted target  -> the link record is removed;
* objects sharing a SEAM id (duplicates) -> every copy gets a fresh id; a duplicated RESULT that
  no link targets becomes an ordinary independent object (role NONE, own mesh);
* a link whose source pointer is None but whose ``source_id`` is in the scene (e.g. restored by
  undo or append) re-points to it and rebuilds.

The cheap link checks run on every update; the id scan only when the object count changes.

Also (connect wave): linked selection (:func:`sync_linked_selection`, driven by a msgbus subscription on the
active object in the GUI, switchable with ``WindowManager.seam_sync_link_selection``) and the save/load
handlers of the images SEAM creates for decals and paint (packed on save so they survive a reload).
"""

from __future__ import annotations

import traceback

import bpy
from bpy.props import BoolProperty

from .. import ids
from ..engine import core as engine
from ..seam_core.ids import new_id
from . import decal, producer, resultpaint

classes = ()

_object_counts = {}


def _find_in_scene(scene, seam_id: str):
    if not seam_id:
        return None
    for ob in scene.objects:
        if ob.seam.id == seam_id:
            return ob
    return None


def check_links(scene) -> bool:
    """Apply delete / missing-source / re-point rules. Returns True when something changed."""
    links = scene.seam.links
    if not len(links):
        return False
    changed = False
    drop = []
    for i, link in enumerate(links):
        tgt = link.target
        if not producer.in_scene(tgt, scene):
            if tgt is None and link.target_id:
                found = _find_in_scene(scene, link.target_id)
                if found is not None and found.type == "MESH":
                    link.target = found
                    changed = True
                    continue
            drop.append(i)
            continue
        if link.state == "DETACHED":
            continue
        src = link.source
        if producer.in_scene(src, scene):
            if link.state == "MISSING_SOURCE":
                producer._apply(link, "source_restored")
                link.message = ""
                engine.mark_dirty(producer.key_for(link))
                changed = True
            continue
        found = _find_in_scene(scene, link.source_id) if src is None else None
        if found is not None and (found.type in ("CURVE", "FONT") or
                                  (link.recipe == "DECAL" and found.seam.role in ("GROUP", "RASTER"))):
            link.source = found
            producer._apply(link, "source_restored")
            link.message = ""
            engine.mark_dirty(producer.key_for(link))
            changed = True
            continue
        if src is not None:
            link.source = None  # release the deleted object so Blender can free it
            changed = True
        if link.state != "MISSING_SOURCE":
            producer._apply(link, "source_missing")
            link.message = "Source was deleted; relink or detach"
            changed = True
    for i in reversed(drop):
        lk = links[i]
        print("SEAM: result of link %r was deleted; removing the link" % lk.name)
        links.remove(i)
        changed = True
    if drop:
        scene.seam.active_link_index = max(0, min(scene.seam.active_link_index, len(links) - 1))
    return changed


def _pick_original(obs, referenced):
    for ob in obs:
        if ob.as_pointer() in referenced:
            return ob
    return sorted(obs, key=lambda o: (len(o.name), o.name))[0]


def dedupe_ids(scene) -> bool:
    """Give duplicated objects fresh SEAM ids; orphaned RESULT copies become independent."""
    groups = ids.duplicate_groups(scene.objects)
    links = scene.seam.links
    targets = {lk.target.as_pointer() for lk in links if lk.target is not None}
    referenced = targets | {lk.source.as_pointer() for lk in links if lk.source is not None}
    changed = False
    for _sid, obs in groups.items():
        keep = _pick_original(obs, referenced)
        for ob in obs:
            if ob == keep:
                continue
            ob.seam.id = new_id()
            changed = True
    for ob in scene.objects:
        if ob.seam.role == "RESULT" and ob.as_pointer() not in targets:
            make_independent(ob, links)
            changed = True
    return changed


def make_independent(ob, links) -> None:
    ob.seam.role = "NONE"
    ob.seam.owner_id = ""
    me = ob.data
    if me is not None and any(lk.target is not None and lk.target.data == me for lk in links):
        ob.data = me.copy()  # never let a live link rewrite the copy's mesh


def on_update(scene, depsgraph=None) -> None:
    seam = getattr(scene, "seam", None)
    if seam is None:
        return
    key = scene.as_pointer()
    n = len(scene.objects)
    if _object_counts.get(key) != n:
        _object_counts[key] = n
        dedupe_ids(scene)
    check_links(scene)


# ------------------------------------------------------------------ linked selection

def sync_enabled(context=None) -> bool:
    """The "Sync link selection" switch (a WindowManager property, so it is scene independent)."""
    try:
        wm = (context or bpy.context).window_manager
        return bool(getattr(wm, "seam_sync_link_selection", True))
    except Exception:
        return True


def sync_linked_selection(context=None) -> list:
    """Select the objects linked to the active object without changing the active object.

    The active object is a link *source*: its results (targets) are selected; it is a *result* (or the target
    mesh of a Material decal): its sources are selected. Detached links and links with a missing end are
    skipped; hidden, unselectable or out-of-view-layer objects are skipped silently. Returns the objects that
    became selected by this call."""
    ctx = context or bpy.context
    scene = ctx.scene
    seam = getattr(scene, "seam", None)
    vl = ctx.view_layer
    if seam is None or vl is None:
        return []
    active = vl.objects.active
    if active is None:
        return []
    newly = []
    for link in seam.links:
        if link.state == "DETACHED":
            continue
        if link.source == active:
            other = link.target
        elif link.target == active:
            other = link.source
        else:
            continue
        if other is None or other == active or vl.objects.get(other.name) != other:
            continue
        try:
            if other.select_get() or other.hide_get() or other.hide_select or other.hide_viewport:
                continue
            other.select_set(True)
        except (RuntimeError, ReferenceError):
            continue
        newly.append(other)
    return newly


_msgbus_owner = object()
_sync_pending = False


def _sync_deferred():
    global _sync_pending
    _sync_pending = False
    try:
        if sync_enabled():
            sync_linked_selection(bpy.context)
    except Exception:
        traceback.print_exc()
    return None


def _on_active_changed(*_args):
    """msgbus notify: run the sync from a timer tick (selection changes do not belong in a notifier)."""
    global _sync_pending
    if _sync_pending or not sync_enabled():
        return
    _sync_pending = True
    try:
        bpy.app.timers.register(_sync_deferred, first_interval=0.0)
    except Exception:
        _sync_pending = False


def subscribe_selection_sync() -> bool:
    """(Re)subscribe to active-object changes (GUI only; msgbus subscriptions are dropped on file load)."""
    if bpy.app.background:
        return False
    try:
        bpy.msgbus.clear_by_owner(_msgbus_owner)
        bpy.msgbus.subscribe_rna(key=(bpy.types.LayerObjects, "active"), owner=_msgbus_owner, args=(),
                                 notify=_on_active_changed, options={"PERSISTENT"})
        return True
    except Exception:
        traceback.print_exc()
        return False


def unsubscribe_selection_sync() -> None:
    try:
        bpy.msgbus.clear_by_owner(_msgbus_owner)
    except Exception:
        pass


# ------------------------------------------------------------------ images: pack on save, trust on load

def pack_decal_images(scene=None) -> int:
    """Pack the images SEAM owns for decals (raster decals share the paint module's composite)."""
    n = 0
    scenes = [scene] if scene is not None else list(bpy.data.scenes)
    for sc in scenes:
        for link in sc.seam.links:
            img = link.decal_image
            if link.recipe != "DECAL" or img is None or img.get(decal.IMAGE_PROP) is None or img.library is not None:
                continue
            try:
                if img.packed_file is None or img.is_dirty:
                    img.pack()
                    n += 1
            except Exception:
                traceback.print_exc()
    return n


@bpy.app.handlers.persistent
def _on_save_pre(*_args):
    try:
        pack_decal_images()
        resultpaint.pack_paint_images()
    except Exception:
        traceback.print_exc()


@bpy.app.handlers.persistent
def _on_load(*_args):
    _object_counts.clear()
    decal._applied.clear()
    try:
        for sc in bpy.data.scenes:
            decal.seed_after_load(sc)
        subscribe_selection_sync()
    except Exception:
        traceback.print_exc()


def register():
    bpy.types.WindowManager.seam_sync_link_selection = BoolProperty(
        name="Sync Link Selection", default=True,
        description="Selecting a live-link source also selects its results, and the other way round "
                    "(the active object does not change)")
    engine.add_update_hook(on_update)
    for lst, fn in ((bpy.app.handlers.load_post, _on_load), (bpy.app.handlers.save_pre, _on_save_pre)):
        if fn not in lst:
            lst.append(fn)
    subscribe_selection_sync()


def unregister():
    engine.remove_update_hook(on_update)
    for lst, fn in ((bpy.app.handlers.load_post, _on_load), (bpy.app.handlers.save_pre, _on_save_pre)):
        while fn in lst:
            lst.remove(fn)
    unsubscribe_selection_sync()
    try:
        del bpy.types.WindowManager.seam_sync_link_selection
    except AttributeError:
        pass
    _object_counts.clear()
    decal._applied.clear()
