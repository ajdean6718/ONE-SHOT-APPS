"""Undo of pixel/layer edits in Texture Paint mode (owner: paint).

Blender records no memfile undo step for anything done while an object is in a paint mode: the operator's own UNDO
push is then an image-type step that holds no scene data, so SEAM's data edits could not be undone from there.
This module provides

* :func:`undoable_in_paint_mode` - decorator for operator ``execute`` methods: the canvas leaves paint mode around the
  edit, the edited images are packed (so the undo step and the auto save written from it hold the pixels) and paint
  mode is re-entered once the operator's own UNDO push happened (one step, named after the operator);
* :func:`note_property_edit` / :func:`flush_property_undo` - the same for plain property edits made through UI widgets
  (layer opacity, blend mode, visibility, lock, mask switch, document background).
"""

import contextlib
import functools

import bpy

from . import rasterdoc as rd


def _enter_paint_mode(context, doc_id: str) -> None:
    """Re-enter Texture Paint mode on the canvas of document ``doc_id`` (no undo step is recorded)."""
    cur = None
    for sc in bpy.data.scenes:
        cur = rd.get_doc(sc, doc_id)
        if cur is not None:
            break
    if cur is None or cur.canvas is None or cur.canvas.mode == "TEXTURE_PAINT":
        return
    if context.view_layer.objects.active != cur.canvas:
        return  # the selection changed meanwhile (undo, a click): never put another object into paint mode
    from . import selection
    try:
        context.tool_settings.image_paint.mode = "MATERIAL"
        bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
        rd.sync_paint_target(cur)
        rd.check_external_edits(cur)
        selection.sync_stencil(cur)
        rd.start_paint_watch(cur)
    except RuntimeError:
        pass


def _reenter_paint_later(doc_id: str) -> None:
    """Re-enter paint mode after the operator returned.  Blender pushes the operator's own UNDO step right after
    ``execute`` returns, and it records scene data (a memfile step) only while the object is NOT in a paint mode;
    entering paint mode first would make it an empty image-type step.  So in the GUI the canvas stays in Object mode
    until the next timer tick (the step is pushed by then); background sessions have no timers and re-enter now."""
    if bpy.app.background:
        _enter_paint_mode(bpy.context, doc_id)
        return

    tries = [0]

    def reenter():
        try:
            override, busy = _gui_override()
            tries[0] += 1
            if busy and tries[0] < 40:  # a modal operator is still running (e.g. the drag that made a selection)
                return 0.05
            if override is not None:
                with bpy.context.temp_override(**override):
                    _enter_paint_mode(bpy.context, doc_id)
            else:
                _enter_paint_mode(bpy.context, doc_id)
        except Exception:
            import traceback
            traceback.print_exc()
        return None

    bpy.app.timers.register(reenter, first_interval=0.0)


def undoable_in_paint_mode(execute):
    """Decorator for ``execute`` methods that edit pixels or layers.

    Blender records no memfile undo step for anything done while an object is in Texture Paint mode, so the edit
    could not be undone.  The canvas leaves paint mode around the edit and re-enters it afterwards
    (:func:`_reenter_paint_later`); the operator's own UNDO push then is the one and only step.  The edited images
    are packed first, so that step (and the auto save written from it) holds the pixels."""
    @functools.wraps(execute)
    def wrapper(self, context):
        doc = rd.active_doc(context.scene)
        if doc is None:
            return execute(self, context)
        scene, doc_id = context.scene, doc.id
        obj = getattr(doc, "canvas", None)
        in_paint = obj is not None and obj.mode == "TEXTURE_PAINT"
        if in_paint:
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            result = execute(self, context)
            if "FINISHED" in result:
                cur = rd.get_doc(scene, doc_id)
                if cur is not None:
                    rd.pack_layer_images(cur)
        except BaseException:
            if in_paint:
                _enter_paint_mode(context, doc_id)
            raise
        if in_paint:
            if bpy.app.background and "FINISHED" in result:
                # no timers: paint mode is re-entered before the operator returns, so its own UNDO push would be an
                # image-type step without scene data.  Push the memfile step now (scripts and tests that run an
                # operator in paint mode and expect to stay in it).
                bpy.ops.ed.undo_push(message=self.bl_label)
            _reenter_paint_later(doc_id)
        return result
    return wrapper


@contextlib.contextmanager
def _object_mode_with_undo_step(context, doc, label: str):
    """Push ONE memfile undo step named ``label`` for a change that is not made by an operator (see
    :func:`flush_property_undo`): leave paint mode, pack, push, re-enter."""
    scene, doc_id = context.scene, doc.id
    bpy.ops.object.mode_set(mode="OBJECT")
    try:
        yield
        cur = rd.get_doc(scene, doc_id)
        if cur is not None:
            rd.pack_layer_images(cur)
        bpy.ops.ed.undo_push(message=label)
    finally:
        _enter_paint_mode(context, doc_id)


# --------------------------------------------------------------------------- property edits made in paint mode
#
# Layer opacity / blend / visibility / lock / mask switch and the document background are plain properties edited
# through UI widgets.  In Texture Paint mode the widget's undo push is an image-type step that holds no scene data,
# and undoing back to the previous memfile step does not reload it (nothing newer exists), so the edit could never
# be undone.  The update callbacks (props.py) report the edit here; once the widget has settled a single undo step
# is pushed from Object mode (same dance as the operators).  A timer does it in the GUI; background sessions
# (tests, scripts) call :func:`flush_property_undo`.

PROP_UNDO_DELAY = 0.6
_prop_pending: dict = {}   # doc id -> time.monotonic() of the last edit


def note_property_edit(owner) -> None:
    """Called by the property update callbacks of layers and documents."""
    if hasattr(owner, "layers"):
        doc = owner
    else:
        _sc, doc = rd.doc_of_layer(owner)
    if doc is None or not rd.paint_mode_active(doc):
        return
    import time
    _prop_pending[doc.id] = time.monotonic()
    if not bpy.app.background and not bpy.app.timers.is_registered(_property_undo_tick):
        bpy.app.timers.register(_property_undo_tick, first_interval=PROP_UNDO_DELAY)


def _gui_override():
    wm = bpy.context.window_manager
    for win in wm.windows:
        if getattr(win, "modal_operators", None):
            return None, True  # a modal operator (a brush stroke) is running: not now
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                if region is not None:
                    return dict(window=win, area=area, region=region), False
    return None, False


def flush_property_undo(force: bool = True) -> int:
    """Push the pending single undo step for property edits made in paint mode. Returns steps pushed."""
    import time
    n = 0
    now = time.monotonic()
    for doc_id, t in list(_prop_pending.items()):
        if not force and now - t < PROP_UNDO_DELAY:
            continue
        del _prop_pending[doc_id]
        doc = None
        for sc in bpy.data.scenes:
            doc = rd.get_doc(sc, doc_id)
            if doc is not None:
                break
        if doc is None or not rd.paint_mode_active(doc):
            continue  # left paint mode meanwhile: the mode switch pushed a memfile step that holds the edit
        override = None
        if not bpy.app.background:
            override, busy = _gui_override()
            if busy:
                _prop_pending[doc_id] = t
                continue
        ctx = bpy.context
        if override is not None:
            with ctx.temp_override(**override):
                with _object_mode_with_undo_step(bpy.context, doc, "Raster Layer Settings"):
                    pass
        else:
            with _object_mode_with_undo_step(ctx, doc, "Raster Layer Settings"):
                pass
        n += 1
    return n


def _property_undo_tick():
    try:
        flush_property_undo(force=False)
    except Exception:
        import traceback
        traceback.print_exc()
        _prop_pending.clear()
    return PROP_UNDO_DELAY / 2.0 if _prop_pending else None


_undoable_in_paint_mode = undoable_in_paint_mode
