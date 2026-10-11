"""Raster documents/layers/masks, node tree, composite cache producer, pack on save (owner: paint).

Data model (see docs/architecture.md D4)
----------------------------------------
* A *raster document* (``scene.seam.rasters[i]``) owns an ordered list of layers (index 0 = bottom).
  Every layer owns an 8-bit RGBA sRGB straight-alpha ``bpy.types.Image`` (named ``SEAM <doc> <layer>``)
  and optionally a mask image (8-bit, Non-Color, white = visible).
* The canvas object is a plane of ``width x height`` Blender units (1 BU = 1 px) whose origin is the
  document's top-left corner, role ``RASTER``, parented to the canvas root through ``canvas.add_artwork_object``.
  Its mesh has two identical UV maps (``UVMap`` and ``SEAM Stencil``); the second one drives the native stencil mask.
* The canvas material composites the visible layers bottom->top with Mix nodes (``compat.new_mix_rgb_node``,
  blend types from ``raster.BLENDER_MIX_TYPE``; non-normal modes blend at full factor and then mix by
  layer alpha x opacity x mask, exactly like the kernel).  Alpha accumulates with Math nodes, output = Emission mixed with Transparent (alpha blend).  The formulas are those of
  ``seam_core.raster.composite`` which is used for export.
* Extra nodes: ``SEAM Composite`` (the cached composite image, the active node while NOT painting so that
  Solid/TEXTURE shows the whole picture) and ``SEAM Sink`` (a 1x1 image that is the paint target while the
  active layer is locked, so native strokes cannot change a locked layer).

Pixel access
------------
``layer_array(layer)`` / ``write_layer_array(layer, arr)`` convert between linear float ``H x W x 4``
(row 0 = top, straight alpha) and the 8-bit sRGB image.  Writing goes through ``raster.to_uint8_srgb`` so a
read -> write cycle is bit exact.  Handles returned by ``scene.seam.rasters`` / ``doc.layers`` can be invalidated
when the same collection grows: re-fetch with :func:`get_doc` / ``doc.layers[i]`` after adding.

Derived data and undo (docs/architecture.md D7/D8)
--------------------------------------------------
* ``rastermat:<doc>`` producer (priority 55): rebuilds the canvas material when its *signature* (layer
  structure, names, opacity, blend, visibility, mask state, background) differs from the signature stored in the
  material's ID property ``seam_sig``.  The signature lives inside the material datablock, so memfile undo
  restores material and layer properties together; operators also rebuild directly so that one undo step
  holds a consistent pair.
* ``rastercache:<doc>`` producer (priority 60): keeps ``doc.composite`` (8-bit sRGB image) equal to
  ``composite_array(doc)``.  Its signature = structure signature + a per-image *revision counter*.  Revision
  counters are bumped by (a) :func:`touch_layer` (called by every SEAM write path), (b) depsgraph updates of a
  layer image (if Blender emits them for native strokes) and (c) :func:`check_external_edits`, a cheap pixel
  fingerprint poll of the *paint target image* run by a GUI timer while the canvas is in Texture Paint mode
  (``Image.pixels`` writes from Python emit no depsgraph update: measured on 4.0.2 and 5.2.2).
  The stored signature of the cache lives in a Python dict that is cleared on undo/redo/load, so the cache is
  rebuilt exactly once after each of those events (cache pixels are not undo-safe, the sources are).
* Large documents compute the composite on a worker thread (prepare/compute/apply split of the engine).
* The same GUI timer re-syncs the paint target (active image node / paint slot) because the UI list writes
  ``active_layer_index`` / ``locked`` / ``edit_target`` without an update callback.
"""

from __future__ import annotations

import hashlib
import traceback
import zlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np
from bpy.app.handlers import persistent

from .. import compat
from ..engine import core as engine
from ..seam_core import raster as R
from ..seam_core.ids import new_id
from ..vector import canvas as cnv

ROLE = "RASTER"
MAX_NAME = 56  # leave room for ".001" suffixes inside the 63-byte ID name limit of Blender 4.0

# --------------------------------------------------------------------------- lookup


def _scenes(scene=None):
    out = []
    if scene is not None:
        out.append(scene)
    try:
        ctx = bpy.context.scene
        if ctx is not None and ctx not in out:
            out.append(ctx)
    except Exception:
        pass
    for sc in bpy.data.scenes:
        if sc not in out:
            out.append(sc)
    return out


def get_doc(scene, raster_id: str):
    """Document with ``id == raster_id`` in ``scene`` (or None)."""
    if not raster_id:
        return None
    scene = scene or bpy.context.scene
    for doc in scene.seam.rasters:
        if doc.id == raster_id:
            return doc
    return None


def doc_for_object(obj):
    """Raster document of a RASTER canvas object (searches the context scene first, then all scenes)."""
    rid = getattr(getattr(obj, "seam", None), "raster_id", "") if obj is not None else ""
    if not rid:
        return None
    for sc in _scenes():
        doc = get_doc(sc, rid)
        if doc is not None:
            return doc
    return None


def doc_scene(doc):
    """Scene that owns ``doc``."""
    for sc in _scenes():
        for d in sc.seam.rasters:
            if d.id == doc.id:
                return sc
    return bpy.context.scene


def active_doc(scene=None):
    """Active raster document: the one of the active RASTER object, else ``active_raster_index``."""
    scene = scene or bpy.context.scene
    try:
        ob = bpy.context.view_layer.objects.active
    except Exception:
        ob = None
    if ob is not None and ob.seam.role == ROLE:
        doc = get_doc(scene, ob.seam.raster_id)
        if doc is not None:
            return doc
    rs = scene.seam.rasters
    if 0 <= scene.seam.active_raster_index < len(rs):
        return rs[scene.seam.active_raster_index]
    return None


def doc_index(scene, doc) -> int:
    for i, d in enumerate(scene.seam.rasters):
        if d.id == doc.id:
            return i
    return -1


def doc_of_layer(layer):
    """(scene, doc) that owns a layer (matched by layer id), or (None, None)."""
    lid = layer.id
    if not lid:
        return None, None
    for sc in _scenes():
        for doc in sc.seam.rasters:
            for ly in doc.layers:
                if ly.id == lid:
                    return sc, doc
    return None, None


def layer_index(doc, layer) -> int:
    for i, ly in enumerate(doc.layers):
        if ly.id == layer.id:
            return i
    return -1


def active_layer(doc):
    if 0 <= doc.active_layer_index < len(doc.layers):
        return doc.layers[doc.active_layer_index]
    return None


def _check_index(doc, index: int) -> int:
    if not 0 <= index < len(doc.layers):
        raise IndexError(f"layer index {index} out of range (0..{len(doc.layers) - 1})")
    return index


# --------------------------------------------------------------------------- pixel access

def _image_name(*parts: str) -> str:
    return " ".join(["SEAM"] + [p for p in parts if p])[:MAX_NAME]


def read_image_srgb8(image) -> np.ndarray:
    """8-bit RGBA ``H x W x 4`` uint8 (row 0 = top) from a Blender image (exact for 8-bit images)."""
    w, h = int(image.size[0]), int(image.size[1])
    buf = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(buf)
    buf *= 255.0
    np.rint(buf, out=buf)
    return buf.astype(np.uint8).reshape(h, w, 4)[::-1].copy()


def write_image_srgb8(image, rgba8: np.ndarray) -> None:
    """Write ``H x W x 4`` uint8 (row 0 = top) into an image of the same size."""
    a = np.asarray(rgba8, dtype=np.uint8)
    w, h = int(image.size[0]), int(image.size[1])
    if a.shape != (h, w, 4):
        raise ValueError(f"array shape {a.shape} does not match image {w}x{h}")
    buf = a[::-1].astype(np.float32).reshape(-1)
    buf /= 255.0
    image.pixels.foreach_set(buf)
    image.update()


def _new_image(name: str, w: int, h: int, colorspace: str, fill=(0, 0, 0, 0)):
    img = bpy.data.images.new(name, int(w), int(h), alpha=True, float_buffer=False)
    img.alpha_mode = "STRAIGHT"
    img.colorspace_settings.name = colorspace
    img.use_fake_user = True
    img["seam_raster"] = 1  # marker: images owned by the paint module (orphans are purged on save)
    buf = np.empty(int(w) * int(h) * 4, np.float32)
    buf.reshape(-1, 4)[:] = np.asarray(fill, np.float32)
    img.pixels.foreach_set(buf)
    return img


def _remove_image(img) -> None:
    if img is None:
        return
    try:
        bpy.data.images.remove(img)
    except (ReferenceError, RuntimeError):
        pass


def doc_size(doc) -> Tuple[int, int]:
    return int(doc.width), int(doc.height)


def layer_image(layer):
    """The pixel image of ``layer``.  If the datablock was deleted behind SEAM's back (Outliner, orphan clean-up) it is
    recreated blank and transparent, so the document stays usable instead of every pixel operator failing with a
    traceback; the console says what happened (the lost pixels can only come back through Undo)."""
    img = layer.image
    if img is not None:
        return img
    _sc, doc = doc_of_layer(layer)
    w, h = doc_size(doc)
    img = _new_image(_layer_image_name(doc, layer.name), w, h, "sRGB")
    layer.image = img
    commit_pixels(layer, "img", np.zeros((h, w, 4), np.uint8))
    touch_layer(layer)
    print(f"SEAM: the image of layer '{layer.name}' was missing; recreated it blank")
    return img


def layer_array_srgb8(layer) -> np.ndarray:
    return read_image_srgb8(layer_image(layer))


def layer_array(layer) -> np.ndarray:
    """Linear float ``H x W x 4`` (row 0 = top, straight alpha) of a layer."""
    return R.from_uint8_srgb(read_image_srgb8(layer_image(layer))).astype(np.float32, copy=False)


def write_layer_array_srgb8(layer, rgba8: np.ndarray, touch: bool = True) -> None:
    a8 = np.asarray(rgba8, np.uint8)
    layer_image(layer)
    ensure_baseline(layer, "img")
    write_image_srgb8(layer.image, a8)
    commit_pixels(layer, "img", a8)
    if touch:
        touch_layer(layer)


def write_layer_array(layer, arr: np.ndarray, touch: bool = True) -> None:
    """Store a linear float ``H x W x 4`` array (row 0 = top, straight alpha) into the layer image (8-bit sRGB)."""
    write_layer_array_srgb8(layer, R.to_uint8_srgb(np.asarray(arr, np.float32)), touch=touch)


def mask_array(layer) -> Optional[np.ndarray]:
    """Mask as float ``H x W`` 0..1 (white = visible), or None when the layer has no mask."""
    if layer.mask is None:
        return None
    return read_image_srgb8(layer.mask)[..., 0].astype(np.float32) / 255.0


def write_mask_array(layer, mask: np.ndarray, touch: bool = True) -> None:
    if layer.mask is None:
        raise ValueError("layer has no mask")
    m = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    v = np.rint(m * 255.0).astype(np.uint8)
    a8 = np.empty(v.shape + (4,), np.uint8)
    a8[..., 0] = a8[..., 1] = a8[..., 2] = v
    a8[..., 3] = 255
    ensure_baseline(layer, "mask")
    write_image_srgb8(layer.mask, a8)
    commit_pixels(layer, "mask", a8)
    if touch:
        touch_layer(layer, mask=True)


# --------------------------------------------------------------------------- pixel journal (undo of pixel edits)
#
# Blender's memfile undo restores properties but NOT the pixels of images edited from Python (measured on 4.0.2 and
# 5.2.2, even after re-packing).  Every SEAM write path therefore bumps a revision counter stored in the data
# (``layer.rev`` / ``layer.mask_rev`` / ``doc.sel_rev``, restored by undo with everything else) and stores the new
# pixels (zlib level 1) in a bounded in-memory journal.  After undo/redo the handler compares the restored
# revision with the revision of the pixels currently in the image and swaps in the journal snapshot when they
# differ.  Native brush strokes are undone by Blender's own image undo and are not touched.  Revision 0 means
# "initial content": transparent layer, white mask, empty selection.

JOURNAL_LIMIT_BYTES = 384 * 1024 * 1024
_journal: "OrderedDict[Tuple[str, str, int], Tuple[int, int, bytes]]" = OrderedDict()
_journal_bytes = 0
_applied: Dict[Tuple[str, str], int] = {}   # (owner id, kind) -> revision of the pixels held by the image
_REV_ATTR = {"img": "rev", "mask": "mask_rev", "sel": "sel_rev"}


def _holder_image(holder, kind):
    return {"img": lambda: holder.image, "mask": lambda: holder.mask, "sel": lambda: holder.selection}[kind]()


def _journal_put(key, a8: np.ndarray) -> None:
    global _journal_bytes
    a8 = np.ascontiguousarray(a8, np.uint8)
    blob = zlib.compress(a8.tobytes(), 1)
    old = _journal.pop(key, None)
    if old is not None:
        _journal_bytes -= len(old[2])
    _journal[key] = (a8.shape[1], a8.shape[0], blob)
    _journal_bytes += len(blob)
    while _journal_bytes > JOURNAL_LIMIT_BYTES and len(_journal) > 1:
        _k, v = _journal.popitem(last=False)
        _journal_bytes -= len(v[2])


def _journal_get(key) -> Optional[np.ndarray]:
    ent = _journal.get(key)
    if ent is None:
        return None
    w, h, blob = ent
    return np.frombuffer(zlib.decompress(blob), np.uint8).reshape(h, w, 4).copy()


def journal_clear() -> None:
    global _journal_bytes
    _journal.clear()
    _journal_bytes = 0


def commit_pixels(holder, kind: str, a8: np.ndarray) -> int:
    """Record that the image of ``holder`` (layer or document) now holds ``a8``: new revision + journal entry."""
    attr = _REV_ATTR[kind]
    rev = int(getattr(holder, attr)) + 1
    setattr(holder, attr, rev)
    _journal_put((holder.id, kind, rev), a8)
    _applied[(holder.id, kind)] = rev
    return rev


def ensure_baseline(holder, kind: str) -> None:
    """Journal the pixels the image holds *before* a SEAM write replaces them, when no entry exists for the current
    revision.  After a file load the journal is empty, so without this the first edit of a session could not be
    undone (the pre-edit revision had no snapshot to swap back in)."""
    attr = _REV_ATTR[kind]
    rev = int(getattr(holder, attr))
    if (holder.id, kind, rev) in _journal:
        return
    img = _holder_image(holder, kind)
    if img is not None and _applied.get((holder.id, kind), rev) == rev:
        _journal_put((holder.id, kind, rev), read_image_srgb8(img))


def snapshot_current(holder, kind: str) -> None:
    """Store the pixels currently in the image under the current revision (call before deleting an image)."""
    img = _holder_image(holder, kind)
    if img is not None:
        _journal_put((holder.id, kind, int(getattr(holder, _REV_ATTR[kind]))), read_image_srgb8(img))


def _forget_applied(owner_id: str) -> None:
    for kind in _REV_ATTR:
        _applied.pop((owner_id, kind), None)


def _default_pixels(kind: str, w: int, h: int) -> np.ndarray:
    a = np.zeros((h, w, 4), np.uint8)
    if kind == "mask":
        a[:] = 255
    elif kind == "sel":
        a[..., 3] = 255
    return a


def _pixel_holders():
    for sc in bpy.data.scenes:
        for doc in sc.seam.rasters:
            if doc.selection is not None:
                yield doc, doc, "sel"
            for ly in doc.layers:
                if ly.image is not None:
                    yield doc, ly, "img"
                if ly.mask is not None:
                    yield doc, ly, "mask"


def _content_rev(img, key, rev: int) -> Optional[int]:
    """Revision of the pixels an image holds right now, or None when it holds nothing meaningful.

    Live (dirty) buffers are SEAM's own writes (``_applied``); a clean packed image holds what was packed
    (recorded in ``seam_packed_rev``, which undo restores together with the packed data); a clean unpacked image is
    a freshly re-created, blank datablock (memfile undo re-reads images whose datablock differs between steps)."""
    if img.is_dirty:
        # a live buffer without a record was only ever edited natively (strokes): it belongs to the current revision
        return _applied.get(key, rev)
    if img.packed_file is not None:
        v = img.get("seam_packed_rev")
        return None if v is None else int(v)
    return None


_restore_lost: List[str] = []  # holders whose pixels could not be restored by the last undo/redo (journal evicted)


def pixel_restore_warning() -> str:
    """Non-empty when the last undo/redo could not restore some pixels because the journal memory cap evicted them."""
    if not _restore_lost:
        return ""
    return (f"Undo restored the settings but not the pixels of {len(_restore_lost)} image(s): "
            "the undo history memory limit was reached")


def restore_pixels_after_undo() -> int:
    """Make every SEAM image hold the pixels of the revision that undo/redo restored. Returns images restored.

    Pixels whose journal entry was evicted (memory cap) stay as they are; that is recorded in
    :func:`pixel_restore_warning` (shown in the Raster Document panel) instead of failing silently."""
    n = 0
    _restore_lost.clear()
    holders = list(_pixel_holders())
    present = {(h.id, k) for _d, h, k in holders}
    for k in [k for k in _applied if k not in present]:
        del _applied[k]  # holder vanished (undo of its creation): redo recreates a blank image that must be refilled
    for doc, holder, kind in holders:
        rev = int(getattr(holder, _REV_ATTR[kind]))
        key = (holder.id, kind)
        img = _holder_image(holder, kind)
        if _content_rev(img, key, rev) == rev:
            _applied[key] = rev
            continue
        w, h = doc_size(doc)
        a8 = _journal_get((holder.id, kind, rev))
        if a8 is None:
            if rev != 0:
                print(f"SEAM: no journal entry for {kind} revision {rev} of {holder.id}; pixels left as they are")
                _restore_lost.append(f"{holder.id}:{kind}")
                _applied[key] = rev
                continue
            a8 = _default_pixels(kind, w, h)
        if (int(img.size[0]), int(img.size[1])) != (a8.shape[1], a8.shape[0]):
            img.scale(int(a8.shape[1]), int(a8.shape[0]))
        write_image_srgb8(img, a8)
        _applied[key] = rev
        n += 1
    return n


def mark_pixels_applied() -> None:
    """After a file load the images hold the saved pixels of the saved revisions."""
    _applied.clear()
    for _doc, holder, kind in _pixel_holders():
        _applied[(holder.id, kind)] = int(getattr(holder, _REV_ATTR[kind]))


# --------------------------------------------------------------------------- revisions (stroke detection)

_revs: Dict[Tuple[str, str], int] = {}          # (layer id, "img"|"mask") -> counter
_fingerprints: Dict[Tuple[str, str], int] = {}  # (layer id, kind) -> crc32 of last seen pixels
_cache_sigs: Dict[str, str] = {}                # "rastercache:<id>" -> signature of the composite stored in doc.composite


def _bump(layer_id: str, kind: str) -> None:
    k = (layer_id, kind)
    _revs[k] = _revs.get(k, 0) + 1


def _forget_layer(layer_id: str) -> None:
    for kind in ("img", "mask"):
        _revs.pop((layer_id, kind), None)
        _fingerprints.pop((layer_id, kind), None)
    _forget_applied(layer_id)


def touch_layer(layer, mask: bool = False) -> None:
    """Declare that the pixels of ``layer`` (or its mask) changed: the composite cache will be rebuilt."""
    _bump(layer.id, "mask" if mask else "img")
    _, doc = doc_of_layer(layer)
    if doc is not None:
        engine.mark_dirty(f"rastercache:{doc.id}")


def touch_doc(doc) -> None:
    """Mark everything derived from ``doc`` dirty (composite)."""
    engine.mark_dirty(f"rastercache:{doc.id}")


def _fingerprint(image) -> int:
    w, h = int(image.size[0]), int(image.size[1])
    buf = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(buf)
    return zlib.crc32(buf.view(np.uint8))


def target_image(doc):
    """The image native brushes paint on right now (layer, mask or the locked-sink)."""
    layer = active_layer(doc)
    if layer is None:
        return None
    if layer.locked:
        return doc.sink
    if doc.edit_target == "MASK" and layer.mask is not None:
        return layer.mask
    return layer.image


def check_external_edits(doc, images: Optional[List] = None) -> bool:
    """Detect pixel changes made outside SEAM code (native strokes) on ``images`` (default: the paint target).

    Compares a CRC of the pixels with the last seen value; on change bumps the revision and marks the composite
    dirty.  Returns True when a change was detected.  The first observation of an image only records the CRC.
    """
    changed = False
    layer = active_layer(doc)
    if images is None:
        img = target_image(doc)
        images = [img] if img is not None else []
    for img in images:
        if img is None or layer is None:
            continue
        if img == layer.image:
            kind = "img"
        elif layer.mask is not None and img == layer.mask:
            kind = "mask"
        else:
            continue  # sink or foreign image: never part of the composite
        k = (layer.id, kind)
        fp = _fingerprint(img)
        old = _fingerprints.get(k)
        _fingerprints[k] = fp
        if old is not None and old != fp:
            _bump(layer.id, kind)
            changed = True
    if changed:
        engine.mark_dirty(f"rastercache:{doc.id}")
    return changed


# --------------------------------------------------------------------------- layers

def _next_layer_name(doc, base: str = "Layer") -> str:
    names = {ly.name for ly in doc.layers}
    i = 1
    while f"{base} {i}" in names:
        i += 1
    return f"{base} {i}"


def _unique_layer_name(doc, name: str) -> str:
    names = {ly.name for ly in doc.layers}
    if name not in names:
        return name
    i = 2
    while f"{name} {i}" in names:
        i += 1
    return f"{name} {i}"


def _layer_image_name(doc, layer_name: str, suffix: str = "") -> str:
    return _image_name(doc.name, layer_name, suffix)


def _append_layer(doc, name: str, image=None) -> int:
    """Append a layer entry (new transparent image unless ``image`` is given); returns its index."""
    w, h = doc_size(doc)
    n = len(doc.layers)
    item = doc.layers.add()
    item.id = new_id()
    item.name = name
    item = doc.layers[n]
    item.image = image if image is not None else _new_image(_layer_image_name(doc, name), w, h, "sRGB")
    return n


def add_layer(doc, name: Optional[str] = None, index: Optional[int] = None):
    """Add a transparent layer. ``index`` = final position (0 = bottom); default: directly above the active layer.
    The new layer becomes active.  Returns the (fresh) layer handle."""
    name = _unique_layer_name(doc, name or _next_layer_name(doc))
    if index is None:
        index = doc.active_layer_index + 1 if len(doc.layers) else 0
    index = max(0, min(int(index), len(doc.layers)))
    n = _append_layer(doc, name)
    if index != n:
        doc.layers.move(n, index)
    doc.active_layer_index = index
    refresh(doc)
    return doc.layers[index]


def remove_layer(doc, index: int) -> None:
    """Delete a layer (and its mask image). A document always keeps at least one layer."""
    _check_index(doc, index)
    if len(doc.layers) <= 1:
        raise ValueError("a raster document needs at least one layer")
    layer = doc.layers[index]
    lid = layer.id
    snapshot_current(layer, "img")  # undo restores the (blank) image datablock; the journal refills it
    if layer.mask is not None:
        snapshot_current(layer, "mask")
    for img in (layer.image, layer.mask):
        _remove_image(img)
    doc.layers.remove(index)
    _forget_layer(lid)
    act = doc.active_layer_index
    if index <= act:  # removing the active layer selects the one below it
        act -= 1
    doc.active_layer_index = max(0, min(act, len(doc.layers) - 1))
    refresh(doc)


def move_layer(doc, from_index: int, to_index: int) -> None:
    """Move a layer so that it ends at ``to_index`` (0 = bottom). The active layer follows its layer."""
    _check_index(doc, from_index)
    to_index = max(0, min(int(to_index), len(doc.layers) - 1))
    if from_index == to_index:
        return
    act = doc.active_layer_index
    doc.layers.move(from_index, to_index)
    if act == from_index:
        act = to_index
    elif from_index < act <= to_index:
        act -= 1
    elif to_index <= act < from_index:
        act += 1
    doc.active_layer_index = act
    refresh(doc)


def duplicate_layer(doc, index: int):
    """Duplicate the layer (pixels, mask and properties) directly above the original; returns the new layer."""
    _check_index(doc, index)
    src = doc.layers[index]
    w, h = doc_size(doc)
    name = _unique_layer_name(doc, f"{src.name} copy")
    img = _new_image(_layer_image_name(doc, name), w, h, "sRGB")
    write_image_srgb8(img, read_image_srgb8(layer_image(src)))
    mask_img = None
    if src.mask is not None:
        mask_img = _new_image(_layer_image_name(doc, name, "mask"), w, h, "Non-Color", (1, 1, 1, 1))
        write_image_srgb8(mask_img, read_image_srgb8(src.mask))
    props = (src.opacity, src.blend_mode, src.visible, src.locked, src.mask_enabled)
    n = _append_layer(doc, name, image=img)
    new = doc.layers[n]
    new.opacity, new.blend_mode, new.visible, new.locked, new.mask_enabled = props
    new.mask = mask_img
    commit_pixels(new, "img", read_image_srgb8(img))
    if mask_img is not None:
        commit_pixels(new, "mask", read_image_srgb8(mask_img))
    doc.layers.move(n, index + 1)
    doc.active_layer_index = index + 1
    refresh(doc)
    return doc.layers[index + 1]


def merge_down(doc, index: int):
    """Merge layer ``index`` into the layer below it.

    The result keeps the lower layer's name and blend mode.  The lower layer's mask and opacity are baked into its
    alpha first (its opacity becomes 1), and the upper layer contributes with its own opacity, blend mode and mask.
    The composite model of ``seam_core.raster`` (a straight-alpha Mix chain) is not associative, so the look of the
    document is preserved exactly (up to 8-bit rounding) only when the lower layer is opaque at full opacity with a
    NORMAL blend mode, or when the upper layer's coverage is binary; otherwise it is approximate (the operator
    measures and reports the difference).  Hidden layers have no look to preserve, so merging a pair where either
    layer is hidden raises ``ValueError``.  Returns the merged layer."""
    _check_index(doc, index)
    if index == 0:
        raise ValueError("no layer below to merge into")
    upper = doc.layers[index]
    lower = doc.layers[index - 1]
    if not (upper.visible and lower.visible):
        raise ValueError("cannot merge hidden layers: show both layers first")
    w, h = doc_size(doc)
    base = layer_array(lower)
    lm = mask_array(lower)
    if lm is not None and lower.mask_enabled:
        base = R.apply_mask_to_alpha(base, lm)
    if float(lower.opacity) != 1.0:  # bake the opacity: factor = alpha * opacity * mask, so this is exact
        base = base.copy()
        base[..., 3] = base[..., 3] * np.float32(lower.opacity)
    specs = [R.LayerSpec(rgba=base, opacity=1.0, blend_mode="NORMAL", visible=True, mask=None),
             R.LayerSpec(rgba=layer_array(upper), opacity=float(upper.opacity), blend_mode=upper.blend_mode,
                         visible=True, mask=(mask_array(upper) if upper.mask_enabled else None))]
    merged = R.composite(specs, h, w, None)
    write_layer_array(lower, merged)
    lower.opacity = 1.0
    if lower.mask is not None:
        snapshot_current(lower, "mask")
        _remove_image(lower.mask)
        lower.mask = None
        _applied.pop((lower.id, "mask"), None)
    lid = upper.id
    snapshot_current(upper, "img")
    if upper.mask is not None:
        snapshot_current(upper, "mask")
    for img in (upper.image, upper.mask):
        _remove_image(img)
    doc.layers.remove(index)
    _forget_layer(lid)
    doc.active_layer_index = index - 1
    refresh(doc)
    return doc.layers[index - 1]


def rename_layer(doc, index: int, name: str) -> None:
    _check_index(doc, index)
    doc.layers[index].name = name
    refresh(doc)


# --------------------------------------------------------------------------- masks

def add_mask(doc, index: int, fill: str = "WHITE"):
    """Add a layer mask. ``fill``: WHITE (reveal all), BLACK (hide all) or SELECTION (white inside the selection)."""
    _check_index(doc, index)
    layer = doc.layers[index]
    if layer.mask is not None:
        raise ValueError("layer already has a mask")
    if fill not in ("WHITE", "BLACK", "SELECTION"):
        raise ValueError(f"unknown mask fill {fill!r}")
    w, h = doc_size(doc)
    value = 0.0 if fill == "BLACK" else 1.0
    img = _new_image(_layer_image_name(doc, layer.name, "mask"), w, h, "Non-Color", (value, value, value, 1.0))
    layer.mask = img
    layer.mask_enabled = True
    if fill == "BLACK":
        commit_pixels(layer, "mask", read_image_srgb8(img))
    if fill == "SELECTION":
        from . import selection
        sel = selection.selection_array(doc)
        if sel is not None:
            write_mask_array(doc.layers[index], sel)
    refresh(doc)
    touch_layer(doc.layers[index], mask=True)
    return doc.layers[index]


def apply_mask(doc, index: int) -> None:
    """Bake the mask into the layer alpha and delete the mask."""
    _check_index(doc, index)
    layer = doc.layers[index]
    if layer.mask is None:
        raise ValueError("layer has no mask")
    arr = R.apply_mask_to_alpha(layer_array(layer), mask_array(layer))
    write_layer_array(layer, arr)
    remove_mask(doc, index)


def remove_mask(doc, index: int) -> None:
    """Delete the mask (the layer becomes fully visible again)."""
    _check_index(doc, index)
    layer = doc.layers[index]
    if layer.mask is None:
        return
    snapshot_current(layer, "mask")
    _remove_image(layer.mask)
    layer.mask = None
    _applied.pop((layer.id, "mask"), None)
    _revs.pop((layer.id, "mask"), None)
    _fingerprints.pop((layer.id, "mask"), None)
    if doc.edit_target == "MASK" and index == doc.active_layer_index:
        doc.edit_target = "LAYER"
    refresh(doc)
    touch_doc(doc)


# --------------------------------------------------------------------------- canvas object

STENCIL_UV = "SEAM Stencil"


def canvas_mesh(width: float, height: float, name: str = "Raster"):
    """Plane (origin = top-left, spans x in [0,w], y in [-h,0]) with two identical UV maps (render + stencil)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([(0, 0, 0), (width, 0, 0), (width, -height, 0), (0, -height, 0)], [], [(0, 3, 2, 1)])
    uvs = [(0, 1), (0, 0), (1, 0), (1, 1)]
    for uv_name in ("UVMap", STENCIL_UV):
        layer = me.uv_layers.new(name=uv_name)
        for loop, co in zip(layer.data, uvs):
            loop.uv = co
    me.uv_layers.active_index = 0
    me.uv_layers[0].active_render = True
    me.uv_layer_stencil_index = 1
    return me


def update_canvas_mesh(doc) -> None:
    """Resize the canvas plane to the document size (UVs are normalised, so unchanged)."""
    obj = doc.canvas
    if obj is None or obj.type != "MESH":
        return
    w, h = doc_size(doc)
    me = obj.data
    for v, c in zip(me.vertices, [(0, 0, 0), (w, 0, 0), (w, -h, 0), (0, -h, 0)]):
        v.co = c
    me.update()


def _unique_doc_name(scene, name: str) -> str:
    names = {d.name for d in scene.seam.rasters}
    if name not in names:
        return name
    i = 2
    while f"{name} {i}" in names:
        i += 1
    return f"{name} {i}"


def new_raster_doc(scene, name: str = "Raster", width: int = 1920, height: int = 1080,
                   background: str = "TRANSPARENT", x: Optional[float] = None, y: Optional[float] = None):
    """Create a raster document with one transparent layer and its canvas plane.

    ``x, y`` = canvas-local top-left of the plane; default = top-left of the active artboard.  Returns the
    document (fresh handle; earlier handles of other documents may be stale)."""
    scene = scene or bpy.context.scene
    width, height = int(width), int(height)
    if not (1 <= width <= 16384 and 1 <= height <= 16384):
        raise ValueError("raster size must be 1..16384 px")
    if background not in ("TRANSPARENT", "WHITE", "COLOR"):
        raise ValueError(f"unknown background {background!r}")
    cnv.ensure_canvas(scene)
    name = _unique_doc_name(scene, name)
    rs = scene.seam.rasters
    n = len(rs)
    item = rs.add()
    item.id = new_id()
    item.name = name
    item.width, item.height = width, height
    item.background = background
    doc = scene.seam.rasters[n]
    _append_layer(doc, "Layer 1")
    doc.active_layer_index = 0

    ab = cnv.active_artboard(scene)
    if x is None:
        x = ab.location.x if ab is not None else 0.0
    if y is None:
        y = ab.location.y if ab is not None else 0.0
    mesh = canvas_mesh(width, height, name)
    obj = bpy.data.objects.new(name, mesh)
    cnv.add_artwork_object(scene, obj, role=ROLE)
    obj.location.x, obj.location.y = float(x), float(y)
    obj.seam.raster_id = doc.id
    doc.canvas = obj
    ma = bpy.data.materials.new(_image_name(name, "material"))
    mesh.materials.append(ma)
    doc.material = ma
    build_material(doc)
    engine.mark_dirty(f"rastercache:{doc.id}")  # the composite image is filled by the cache producer
    scene.seam.active_raster_index = n
    return scene.seam.rasters[n]


def remove_doc(scene, doc) -> None:
    """Delete a document with its canvas object, material and all images."""
    scene = scene or doc_scene(doc)
    obj, ma = doc.canvas, doc.material
    imgs = [doc.composite, doc.selection, doc.sink]
    if doc.selection is not None:
        snapshot_current(doc, "sel")
    for ly in doc.layers:
        imgs += [ly.image, ly.mask]
        snapshot_current(ly, "img")
        if ly.mask is not None:
            snapshot_current(ly, "mask")
        _forget_layer(ly.id)
    _forget_applied(doc.id)
    rid = doc.id
    ip = scene.tool_settings.image_paint
    if ip.stencil_image is not None and ip.stencil_image in imgs:
        ip.use_stencil_layer = False
        ip.stencil_image = None
    idx = doc_index(scene, doc)
    mesh = obj.data if obj is not None and obj.type == "MESH" else None
    if obj is not None and obj.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
    scene.seam.rasters.remove(idx)
    for img in imgs:
        _remove_image(img)
    if obj is not None:
        try:
            bpy.data.objects.remove(obj)
        except (ReferenceError, RuntimeError):
            pass
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    if ma is not None:
        try:
            bpy.data.materials.remove(ma)
        except (ReferenceError, RuntimeError):
            pass
    _cache_sigs.pop(f"rastercache:{rid}", None)
    n = len(scene.seam.rasters)
    scene.seam.active_raster_index = max(0, min(scene.seam.active_raster_index, n - 1))


def ensure_canvas_object(doc):
    """Re-link the canvas object into the scene if it was removed from all collections; returns it."""
    scene = doc_scene(doc)
    obj = doc.canvas
    if obj is None:
        obj = bpy.data.objects.new(doc.name, canvas_mesh(doc.width, doc.height, doc.name))
        obj.data.materials.append(doc.material)
        doc.canvas = obj
    if obj.name not in scene.objects:
        x, y = obj.location.x, obj.location.y
        cnv.add_artwork_object(scene, obj, role=ROLE)
        obj.location.x, obj.location.y = x, y
    obj.seam.raster_id = doc.id
    return obj


# --------------------------------------------------------------------------- composite

def _background_color(doc):
    if doc.background == "WHITE":
        return (1.0, 1.0, 1.0, 1.0)
    if doc.background == "COLOR":
        c = doc.background_color
        return (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
    return None


def _layer_specs(doc) -> List:
    specs = []
    for layer in doc.layers:
        if not layer.visible or layer.image is None:
            continue
        mask = mask_array(layer) if (layer.mask is not None and layer.mask_enabled) else None
        specs.append(R.LayerSpec(rgba=layer_array(layer), opacity=float(layer.opacity),
                                 blend_mode=layer.blend_mode, visible=True, mask=mask))
    return specs


def composite_array(doc) -> np.ndarray:
    """Linear float ``H x W x 4`` composite of the visible layers over the document background."""
    w, h = doc_size(doc)
    return R.composite(_layer_specs(doc), h, w, _background_color(doc))


def _ensure_composite(doc):
    w, h = doc_size(doc)
    img = doc.composite
    if img is not None and (int(img.size[0]), int(img.size[1])) != (w, h):
        _remove_image(img)
        img = None
    if img is None:
        img = _new_image(_image_name(doc.name, "composite"), w, h, "sRGB")
        doc.composite = img
        _cache_sigs.pop(f"rastercache:{doc.id}", None)
    return img


def _ensure_sink(doc):
    if doc.sink is None:
        doc.sink = _new_image(_image_name(doc.name, "sink"), 1, 1, "sRGB")
    return doc.sink


def update_composite(doc) -> None:
    """Recompute ``doc.composite`` now (synchronously) and record the cache signature."""
    img = _ensure_composite(doc)
    write_image_srgb8(img, R.to_uint8_srgb(composite_array(doc)))
    _cache_sigs[f"rastercache:{doc.id}"] = _cache_signature(doc)


# --------------------------------------------------------------------------- material

MAT_SIG_KEY = "seam_sig"
NODE_COMPOSITE = "SEAM Composite"
NODE_SINK = "SEAM Sink"


def _f(v: float) -> str:
    return f"{float(v):.6f}"


def _structure_parts(doc) -> List[str]:
    parts = [doc.id, doc.name, str(doc.width), str(doc.height), doc.background]
    parts += [_f(c) for c in doc.background_color]
    for ly in doc.layers:
        parts += [ly.id, ly.name, ly.blend_mode, _f(ly.opacity), "V" if ly.visible else "h",
                  "M" if (ly.mask is not None and ly.mask_enabled) else "m",
                  ly.image.name if ly.image else "", ly.mask.name if ly.mask else ""]
    return parts


def _digest(parts) -> str:
    h = hashlib.blake2b(digest_size=10)
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


def material_signature(doc) -> str:
    return _digest(_structure_parts(doc))


def _cache_signature(doc) -> str:
    parts = _structure_parts(doc)
    for ly in doc.layers:
        parts.append(str(_revs.get((ly.id, "img"), 0)))
        parts.append(str(_revs.get((ly.id, "mask"), 0)))
    return _digest(parts)


def _sync_image_names(doc) -> None:
    """Keep image/material datablock names in sync with document and layer names."""
    for ly in doc.layers:
        if ly.image is not None:
            want = _layer_image_name(doc, ly.name)
            if ly.image.name != want:
                ly.image.name = want
        if ly.mask is not None:
            want = _layer_image_name(doc, ly.name, "mask")
            if ly.mask.name != want:
                ly.mask.name = want
    for img, suffix in ((doc.composite, "composite"), (doc.selection, "selection"), (doc.sink, "sink")):
        if img is not None:
            want = _image_name(doc.name, suffix)
            if img.name != want:
                img.name = want
    if doc.material is not None:
        want = _image_name(doc.name, "material")
        if doc.material.name != want:
            doc.material.name = want


def node_name_layer(layer) -> str:
    return f"SEAM Layer {layer.id}"


def node_name_mask(layer) -> str:
    return f"SEAM Mask {layer.id}"


def _tex_node(nodes, image, name, label, loc):
    n = nodes.new("ShaderNodeTexImage")
    n.name = name
    n.label = label
    n.image = image
    n.interpolation = "Closest"
    n.extension = "CLIP"
    n.location = loc
    return n


def _math(nodes, op, loc, label=""):
    m = nodes.new("ShaderNodeMath")
    m.operation = op
    m.use_clamp = False
    m.location = loc
    if label:
        m.label = label
    return m


def build_material(doc) -> None:
    """(Re)create the canvas material node tree from the layer stack and store its signature."""
    ma = doc.material
    if ma is None:
        ma = bpy.data.materials.new(_image_name(doc.name, "material"))
        doc.material = ma
        if doc.canvas is not None and doc.canvas.type == "MESH":
            doc.canvas.data.materials.append(ma)
    _sync_image_names(doc)
    _ensure_composite(doc)
    _ensure_sink(doc)
    if ma.node_tree is None:
        ma.use_nodes = True
    nt = ma.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    w = 280.0

    # ---- image nodes: one per layer (+ mask), composite (display), sink (locked)
    layer_nodes = []
    for i, ly in enumerate(doc.layers):
        nimg = _tex_node(nodes, ly.image, node_name_layer(ly), ly.name, (i * w, -i * 120.0 + 200.0))
        nmask = None
        if ly.mask is not None:
            nmask = _tex_node(nodes, ly.mask, node_name_mask(ly), ly.name + " mask", (i * w, -i * 120.0 - 160.0))
        layer_nodes.append((nimg, nmask))
    top = len(doc.layers) * w + 120.0
    _tex_node(nodes, doc.composite, NODE_COMPOSITE, "Composite (display only)", (top, 520.0))
    _tex_node(nodes, doc.sink, NODE_SINK, "Locked layer sink", (top, 760.0))

    # ---- compositing chain; state = (rgb, alpha): rgb is None/tuple(const)/socket, alpha is float/socket
    bg = _background_color(doc)
    if bg is None or bg[3] <= 0.0:
        cur_rgb, cur_a = None, 0.0
    else:
        cur_rgb, cur_a = (bg[0], bg[1], bg[2], 1.0), float(bg[3])

    for i, ly in enumerate(doc.layers):
        if not ly.visible or ly.image is None:
            continue
        nimg, nmask = layer_nodes[i]
        x = i * w + 140.0
        y = -i * 120.0 - 420.0
        f_src = nimg.outputs["Alpha"]
        if ly.opacity < 1.0:
            m = _math(nodes, "MULTIPLY", (x, y), "Opacity")
            links.new(f_src, m.inputs[0])
            m.inputs[1].default_value = float(ly.opacity)
            f_src = m.outputs[0]
        if nmask is not None and ly.mask_enabled:
            sep = nodes.new("ShaderNodeSeparateColor")
            sep.location = (x, y - 120.0)
            sep.label = "Mask value"
            links.new(nmask.outputs["Color"], sep.inputs[0])
            mm = _math(nodes, "MULTIPLY", (x + 160.0, y - 120.0), "Mask")
            links.new(f_src, mm.inputs[0])
            links.new(sep.outputs["Red"], mm.inputs[1])
            f_src = mm.outputs[0]
        layer_rgb = nimg.outputs["Color"]

        if cur_rgb is None:  # nothing underneath: rgb = layer rgb, alpha = f
            cur_rgb, cur_a = layer_rgb, f_src
            continue
        # kernel formula: rgb = mix(prev, blend(mode, prev, layer) at full factor, f).  The Mix node applies its factor
        # *inside* the blend formula (not equivalent for dodge/burn), so non-normal modes use two Mix nodes:
        # blend at factor 1, then a plain mix by f.
        mix, fac, a_in, b_in, blended = compat.new_mix_rgb_node(nodes, R.BLENDER_MIX_TYPE[ly.blend_mode])
        mix.label = ly.blend_mode.replace("_", " ").title()
        mix.location = (x + 320.0, y)
        if isinstance(cur_rgb, tuple):
            a_in.default_value = cur_rgb
        else:
            links.new(cur_rgb, a_in)
        links.new(layer_rgb, b_in)
        if ly.blend_mode == "NORMAL":
            links.new(f_src, fac)
        else:
            fac.default_value = 1.0
            fmix, ffac, fa, fb, fres = compat.new_mix_rgb_node(nodes, "MIX")
            fmix.label = "By alpha x opacity x mask"
            fmix.location = (x + 480.0, y + 120.0)
            links.new(f_src, ffac)
            if isinstance(cur_rgb, tuple):
                fa.default_value = cur_rgb
            else:
                links.new(cur_rgb, fa)
            links.new(blended, fb)
            blended = fres
        if isinstance(cur_a, float) and cur_a >= 1.0:
            cur_rgb, cur_a = blended, 1.0
        elif isinstance(cur_a, float):  # constant partial background alpha > 0
            mad = _math(nodes, "MULTIPLY_ADD", (x + 480.0, y), "Alpha")
            links.new(f_src, mad.inputs[0])
            mad.inputs[1].default_value = 1.0 - cur_a
            mad.inputs[2].default_value = cur_a
            cur_rgb, cur_a = blended, mad.outputs[0]
        else:
            gt = _math(nodes, "GREATER_THAN", (x + 320.0, y - 200.0), "Below visible?")
            links.new(cur_a, gt.inputs[0])
            gt.inputs[1].default_value = 0.0
            sel, sfac, sa, sb, sres = compat.new_mix_rgb_node(nodes, "MIX")
            sel.label = "Below ? blend : layer"
            sel.location = (x + 480.0, y)
            links.new(gt.outputs[0], sfac)
            links.new(layer_rgb, sa)
            links.new(blended, sb)
            om = _math(nodes, "SUBTRACT", (x + 320.0, y - 360.0), "1 - below")
            om.inputs[0].default_value = 1.0
            links.new(cur_a, om.inputs[1])
            mad = _math(nodes, "MULTIPLY_ADD", (x + 480.0, y - 360.0), "Alpha")
            links.new(f_src, mad.inputs[0])
            links.new(om.outputs[0], mad.inputs[1])
            links.new(cur_a, mad.inputs[2])
            cur_rgb, cur_a = sres, mad.outputs[0]

    # ---- output: Emission over Transparent, alpha = accumulated alpha
    ox = len(doc.layers) * w + 600.0
    emis = nodes.new("ShaderNodeEmission")
    emis.location = (ox, 100.0)
    emis.inputs["Strength"].default_value = 1.0
    transp = nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (ox, -60.0)
    mixsh = nodes.new("ShaderNodeMixShader")
    mixsh.location = (ox + 220.0, 0.0)
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (ox + 420.0, 0.0)
    if cur_rgb is None:
        emis.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    elif isinstance(cur_rgb, tuple):
        emis.inputs["Color"].default_value = cur_rgb
    else:
        links.new(cur_rgb, emis.inputs["Color"])
    if isinstance(cur_a, float):
        mixsh.inputs["Fac"].default_value = cur_a
    else:
        links.new(cur_a, mixsh.inputs["Fac"])
    links.new(transp.outputs[0], mixsh.inputs[1])
    links.new(emis.outputs[0], mixsh.inputs[2])
    links.new(mixsh.outputs[0], out.inputs["Surface"])
    compat.set_material_blend_alpha(ma)
    ma[MAT_SIG_KEY] = material_signature(doc)
    sync_paint_target(doc)


def material_image_nodes(doc) -> Dict[str, object]:
    """Map node name -> Image Texture node of the canvas material."""
    ma = doc.material
    if ma is None or ma.node_tree is None:
        return {}
    return {n.name: n for n in ma.node_tree.nodes if n.bl_idname == "ShaderNodeTexImage"}


# --------------------------------------------------------------------------- paint target

def set_paint_target(doc, layer_index: int, target: str = "LAYER") -> None:
    """Select which layer / mask native brushes paint on (``target`` = LAYER | MASK).

    A locked layer can be *selected* but strokes then go to the 1x1 sink image, never to the layer."""
    _check_index(doc, layer_index)
    if target not in ("LAYER", "MASK"):
        raise ValueError(f"unknown paint target {target!r}")
    layer = doc.layers[layer_index]
    if target == "MASK" and layer.mask is None:
        raise ValueError("layer has no mask")
    doc.active_layer_index = layer_index
    doc.edit_target = target
    sync_paint_target(doc)


def paint_mode_active(doc) -> bool:
    obj = doc.canvas
    return obj is not None and obj.mode == "TEXTURE_PAINT"


def paint_target_node(doc):
    """The Image Texture node that should be active for the current mode, or None."""
    ma = doc.material
    if ma is None or ma.node_tree is None:
        return None
    nodes = ma.node_tree.nodes
    if not paint_mode_active(doc):
        return nodes.get(NODE_COMPOSITE)
    img = target_image(doc)
    layer = active_layer(doc)
    if img is None or layer is None:
        return None
    if img == doc.sink:
        return nodes.get(NODE_SINK)
    if layer.mask is not None and img == layer.mask:
        return nodes.get(node_name_mask(layer))
    return nodes.get(node_name_layer(layer))


def sync_paint_target(doc) -> None:
    """Make the material's active image node match the paint target (paint mode) or the composite (otherwise)."""
    node = paint_target_node(doc)
    if node is None:
        return
    ma = doc.material
    nodes = ma.node_tree.nodes
    if nodes.active != node:  # avoid redundant writes: every node tree write re-triggers material updates
        for n in nodes:
            n.select = n == node
        nodes.active = node
    if paint_mode_active(doc):
        # slots are only valid after Blender refreshed them (entering paint mode); never touch them before
        imgs = list(ma.texture_paint_images)
        if imgs and node.image in imgs:
            idx = imgs.index(node.image)
            if ma.paint_active_slot != idx:
                ma.paint_active_slot = idx


def show_composite(doc) -> None:
    """Make Solid/TEXTURE (active image node) show the composite again (call after leaving paint mode)."""
    sync_paint_target(doc)


# --------------------------------------------------------------------------- refresh

def refresh(doc) -> None:
    """Rebuild the material now and schedule the composite cache rebuild (structure changed)."""
    build_material(doc)
    engine.mark_dirty(f"rastercache:{doc.id}")


# --------------------------------------------------------------------------- engine producers

class RasterMaterialProducer(engine.Producer):
    """Keeps the canvas material in sync with layer properties (kind ``rastermat``)."""

    kind = "rastermat"
    priority = 55

    def all_keys(self, scene):
        return [f"rastermat:{d.id}" for d in scene.seam.rasters]

    def _doc(self, scene, key):
        return get_doc(scene, key.split(":", 1)[1])

    def signature(self, scene, key):
        doc = self._doc(scene, key)
        return None if doc is None else material_signature(doc)

    def stored_signature(self, scene, key):
        doc = self._doc(scene, key)
        if doc is None or doc.material is None:
            return None
        return doc.material.get(MAT_SIG_KEY)

    def build(self, scene, key, signature):
        doc = self._doc(scene, key)
        if doc is None:
            return engine.BuildResult(key, "obsolete")
        build_material(doc)
        return engine.BuildResult(key, "ok")


ASYNC_PIXELS = 3_000_000  # pixels x layers above which the composite is computed on a worker thread


class RasterCacheProducer(engine.Producer):
    """Keeps ``doc.composite`` equal to ``composite_array(doc)`` (kind ``rastercache``)."""

    kind = "rastercache"
    priority = 60
    async_capable = True

    def keys_for_update(self, scene, id_orig, geometry, transform):
        if not isinstance(id_orig, bpy.types.Image) or not len(scene.seam.rasters):
            return ()
        keys = []
        for doc in scene.seam.rasters:
            for ly in doc.layers:
                if ly.image == id_orig:
                    _bump(ly.id, "img")
                    keys.append(f"rastercache:{doc.id}")
                elif ly.mask is not None and ly.mask == id_orig:
                    _bump(ly.id, "mask")
                    keys.append(f"rastercache:{doc.id}")
        return keys

    def all_keys(self, scene):
        return [f"rastercache:{d.id}" for d in scene.seam.rasters]

    def _doc(self, scene, key):
        return get_doc(scene, key.split(":", 1)[1])

    def signature(self, scene, key):
        doc = self._doc(scene, key)
        return None if doc is None else _cache_signature(doc)

    def stored_signature(self, scene, key):
        doc = self._doc(scene, key)
        if doc is None or doc.composite is None:
            return None
        return _cache_sigs.get(key)

    def build(self, scene, key, signature):
        doc = self._doc(scene, key)
        if doc is None:
            return engine.BuildResult(key, "obsolete")
        write_image_srgb8(_ensure_composite(doc), R.to_uint8_srgb(composite_array(doc)))
        _cache_sigs[key] = signature
        return engine.BuildResult(key, "ok")

    # async split: pixel reads on the main thread, compositing off-thread, pixel write on the main thread
    def prepare(self, scene, key):
        doc = self._doc(scene, key)
        w, h = doc_size(doc)
        return {"w": w, "h": h, "specs": _layer_specs(doc), "bg": _background_color(doc)}

    def compute(self, snap):
        return R.to_uint8_srgb(R.composite(snap["specs"], snap["h"], snap["w"], snap["bg"]))

    def apply(self, scene, key, signature, result):
        doc = self._doc(scene, key)
        if doc is None:
            return engine.BuildResult(key, "obsolete")
        write_image_srgb8(_ensure_composite(doc), result)
        _cache_sigs[key] = signature
        return engine.BuildResult(key, "ok")

    def wants_async(self, snap):
        return snap["w"] * snap["h"] * max(1, len(snap["specs"])) > ASYNC_PIXELS


# --------------------------------------------------------------------------- handlers

def pack_images(scene) -> int:
    """Pack every dirty/unpacked SEAM raster image (layers, masks, selection, composite). Returns the count.

    Packed layer/mask/selection images record the revision they hold in ``seam_packed_rev``."""
    n = 0
    for doc in scene.seam.rasters:
        items = [(doc.composite, None, None)]
        if doc.selection is not None:
            items.append((doc.selection, doc, "sel"))
        for ly in doc.layers:
            items.append((ly.image, ly, "img"))
            items.append((ly.mask, ly, "mask"))
        for img, holder, kind in items:
            if img is None:
                continue
            try:
                if img.packed_file is None or img.is_dirty:
                    img.pack()
                    if holder is not None:
                        img["seam_packed_rev"] = int(getattr(holder, _REV_ATTR[kind]))
                    n += 1
            except Exception:
                traceback.print_exc()
    return n


def purge_orphan_images() -> int:
    """Remove paint-owned images (marker ``seam_raster``) that no document references any more
    (e.g. left behind by undoing the creation of a document or layer)."""
    used = set()
    for sc in bpy.data.scenes:
        for doc in sc.seam.rasters:
            used.update(i.name for i in (doc.composite, doc.selection, doc.sink) if i is not None)
            for ly in doc.layers:
                used.update(i.name for i in (ly.image, ly.mask) if i is not None)
    n = 0
    for img in list(bpy.data.images):
        if img.get("seam_raster") and img.name not in used and img.library is None:
            _remove_image(img)
            n += 1
    return n


@persistent
def _on_save_pre(*_args):
    try:
        purge_orphan_images()
        for sc in bpy.data.scenes:
            pack_images(sc)
    except Exception:
        traceback.print_exc()


def _after_history_jump() -> None:
    # composite pixels are not undo/load safe: forget which composite is stored so it is rebuilt once
    _cache_sigs.clear()
    _fingerprints.clear()
    for sc in bpy.data.scenes:
        for doc in sc.seam.rasters:
            sync_paint_target(doc)  # the object mode may differ from the one the node selection was made in
            try:  # tool settings survive memfile undo but the selection (and its flag) does not
                from . import selection
                selection.sync_stencil(doc)
            except Exception:
                traceback.print_exc()
    _restart_paint_watch_if_needed()


@persistent
def _on_undo_redo(*_args):
    try:
        restore_pixels_after_undo()
        _after_history_jump()
    except Exception:
        traceback.print_exc()


@persistent
def _on_load_post(*_args):
    try:
        journal_clear()
        _restore_lost.clear()
        mark_pixels_applied()
        _after_history_jump()
    except Exception:
        traceback.print_exc()


_HANDLERS = (("save_pre", _on_save_pre), ("undo_post", _on_undo_redo),
             ("redo_post", _on_undo_redo), ("load_post", _on_load_post))


# --------------------------------------------------------------------------- GUI paint watch

_WATCH_MIN = 0.4
_watching = set()  # doc ids whose canvas is (or was just) in Texture Paint mode


def _watch_tick():
    """Timer: while a SEAM canvas is in Texture Paint mode, detect native strokes and keep the paint target synced;
    when the mode was left, show the composite again."""
    import time
    t0 = time.perf_counter()
    for sc in _scenes():
        for doc in list(sc.seam.rasters):
            if doc.id not in _watching:
                continue
            try:
                if paint_mode_active(doc):
                    # layer selection / lock / target edits made through UI widgets have no update callback
                    sync_paint_target(doc)
                    img = target_image(doc)
                    if img is not None and img != doc.sink and img.is_dirty:
                        check_external_edits(doc, [img])
                else:
                    _watching.discard(doc.id)
                    sync_paint_target(doc)
            except Exception:
                traceback.print_exc()
                _watching.discard(doc.id)
    if not _watching:
        return None
    return max(_WATCH_MIN, 8.0 * (time.perf_counter() - t0))


def start_paint_watch(doc) -> None:
    """Start polling ``doc``'s canvas (GUI timer; no-op in background mode)."""
    if bpy.app.background:
        return
    _watching.add(doc.id)
    if not bpy.app.timers.is_registered(_watch_tick):
        bpy.app.timers.register(_watch_tick, first_interval=_WATCH_MIN)


def _restart_paint_watch_if_needed() -> None:
    if bpy.app.background:
        return
    ob = bpy.context.view_layer.objects.active if bpy.context.view_layer else None
    if ob is not None and ob.seam.role == ROLE and ob.mode == "TEXTURE_PAINT":
        doc = doc_for_object(ob)
        if doc is not None:
            start_paint_watch(doc)


# --------------------------------------------------------------------------- resampling / canvas resize

def _halve(a: np.ndarray, axis: int) -> np.ndarray:
    n = a.shape[axis]
    if n % 2:
        last = np.take(a, [n - 1], axis=axis)
        a = np.concatenate([a, last], axis=axis)
    s0 = np.take(a, np.arange(0, a.shape[axis], 2), axis=axis)
    s1 = np.take(a, np.arange(1, a.shape[axis], 2), axis=axis)
    return (s0 + s1) * 0.5


def _linear_axis(a: np.ndarray, new_n: int, axis: int) -> np.ndarray:
    n = a.shape[axis]
    if n == new_n:
        return a
    pos = (np.arange(new_n, dtype=np.float64) + 0.5) * (n / new_n) - 0.5
    pos = np.clip(pos, 0, n - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    t = (pos - i0).astype(np.float32)
    shape = [1] * a.ndim
    shape[axis] = new_n
    t = t.reshape(shape)
    return np.take(a, i0, axis=axis) * (1.0 - t) + np.take(a, i1, axis=axis) * t


def resample_rgba8(a8: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize an ``H x W x 4`` uint8 straight-alpha image (box filter for large reductions, then bilinear).

    Interpolation happens on alpha-premultiplied values so transparent pixels do not bleed colour."""
    a = np.asarray(a8, np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    h, w = a.shape[:2]
    while h >= 2 * new_h:
        a = _halve(a, 0)
        h = a.shape[0]
    while w >= 2 * new_w:
        a = _halve(a, 1)
        w = a.shape[1]
    a = _linear_axis(a, new_h, 0)
    a = _linear_axis(a, new_w, 1)
    al = a[..., 3:4]
    rgb = np.where(al > 1e-6, a[..., :3] / np.maximum(al, 1e-6), 0.0)
    out = np.concatenate([rgb, al], axis=-1)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


ANCHORS = ("TOP_LEFT", "TOP", "TOP_RIGHT", "LEFT", "CENTER", "RIGHT", "BOTTOM_LEFT", "BOTTOM", "BOTTOM_RIGHT")


def _anchor_offset(anchor: str, old: int, new: int, axis: str) -> int:
    """Offset (in new-canvas pixels) at which the old extent starts along ``axis`` ('x' or 'y')."""
    if axis == "x":
        frac = 0.0 if "LEFT" in anchor else (1.0 if "RIGHT" in anchor else 0.5)
    else:
        frac = 0.0 if "TOP" in anchor else (1.0 if "BOTTOM" in anchor else 0.5)
    return int(round((new - old) * frac))


def _fit_canvas(a8: np.ndarray, new_h: int, new_w: int, anchor: str, fill: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = a8.shape[:2]
    out = np.empty((new_h, new_w, 4), np.uint8)
    out[:] = np.asarray(fill, np.uint8)
    ox, oy = _anchor_offset(anchor, w, new_w, "x"), _anchor_offset(anchor, h, new_h, "y")
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cw, ch = min(w - sx0, new_w - dx0), min(h - sy0, new_h - dy0)
    if cw > 0 and ch > 0:
        out[dy0:dy0 + ch, dx0:dx0 + cw] = a8[sy0:sy0 + ch, sx0:sx0 + cw]
    return out


def resize_canvas(doc, width: int, height: int, anchor: str = "TOP_LEFT", scale_content: bool = False) -> None:
    """Change the document size. ``scale_content`` resamples all pixels; otherwise the canvas is cropped/extended
    around ``anchor`` (new area transparent, mask area white).  The selection is cleared."""
    width, height = int(width), int(height)
    if not (1 <= width <= 16384 and 1 <= height <= 16384):
        raise ValueError("raster size must be 1..16384 px")
    if anchor not in ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}")
    if (width, height) == doc_size(doc):
        return

    def convert(img, fill):
        a8 = read_image_srgb8(img)
        if scale_content:
            return resample_rgba8(a8, height, width)
        return _fit_canvas(a8, height, width, anchor, fill)

    jobs = []
    for ly in doc.layers:
        jobs.append((layer_image(ly), convert(layer_image(ly), (0, 0, 0, 0))))
        if ly.mask is not None:
            jobs.append((ly.mask, convert(ly.mask, (255, 255, 255, 255))))
    for ly in doc.layers:  # undo restores the old size from these snapshots
        snapshot_current(ly, "img")
        if ly.mask is not None:
            snapshot_current(ly, "mask")
    if doc.selection is not None:
        snapshot_current(doc, "sel")
        _remove_image(doc.selection)
        doc.selection = None
        doc.has_selection = False
        _applied.pop((doc.id, "sel"), None)
    for img, a8 in jobs:
        img.scale(width, height)
        write_image_srgb8(img, a8)
    for ly in doc.layers:
        commit_pixels(ly, "img", read_image_srgb8(ly.image))
        if ly.mask is not None:
            commit_pixels(ly, "mask", read_image_srgb8(ly.mask))
    doc.width, doc.height = width, height
    update_canvas_mesh(doc)
    _remove_image(doc.composite)
    doc.composite = None
    for ly in doc.layers:
        _bump(ly.id, "img")
        _bump(ly.id, "mask")
    build_material(doc)
    engine.mark_dirty(f"rastercache:{doc.id}")
    from . import selection
    selection.sync_stencil(doc)


# --------------------------------------------------------------------------- registration

classes = ()


def register():
    engine.register_producer(RasterMaterialProducer())
    engine.register_producer(RasterCacheProducer())
    for name, fn in _HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        if fn not in lst:
            lst.append(fn)


def unregister():
    engine.unregister_producer("rastermat")
    engine.unregister_producer("rastercache")
    for name, fn in _HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        while fn in lst:
            lst.remove(fn)
    try:
        if bpy.app.timers.is_registered(_watch_tick):
            bpy.app.timers.unregister(_watch_tick)
    except Exception:
        pass
    _revs.clear()
    _fingerprints.clear()
    _cache_sigs.clear()
    _watching.clear()
    _applied.clear()
    journal_clear()
