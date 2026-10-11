"""DECAL recipe: artwork or raster layer -> live texture on a plane or on a mesh material (owner: connect).

Two shapes of the same recipe (``params.decal_mode`` is chosen when the link is created; afterwards the
mode is *derived* from who owns the target, see :func:`mode_of`):

``PLANE``
    The target is a SEAM RESULT plane mesh (artwork bounds x ``units_scale``, oriented per ``orientation``,
    UVs 0..1, centred on the object origin; UPRIGHT stands on the origin and faces -Y) with one SEAM-owned
    material that shows the decal image with alpha blending. The producer rewrites the plane through the
    regular link machinery (conflict detection included).
``MATERIAL``
    The target is an *existing user mesh*. The link adds one SEAM-owned material slot whose shader is
    ``mix(base colour, decal image, image alpha)`` evaluated over the mesh's own UVs (active render UV map).
    The producer NEVER touches that mesh: it only updates the image and the material nodes. (Assigning
    faces to the new slot is a one-time, undoable step of the ``seam.make_3d`` operator.)

Source -> image
    * SHAPE / TEXT / GROUP: the artwork (the object, or every visible SHAPE/TEXT/RASTER descendant of the
      group) is rasterized with ``seam_core.render2d`` over its own canvas bounds into the 8-bit sRGB
      straight-alpha image ``SEAM Decal <source>`` (longer side = ``decal_resolution``, transparent
      background). Rebuilds rewrite the pixels of the *same* datablock (``Image.scale`` when the size changes). The longer side is
      capped at ``MAX_RENDER_SIDE`` (4096 px; the renderer works on a float64 canvas) and the link message says so.
    * RASTER: the raster document composite (``doc.composite``, kept current by the paint producer) is shared
      directly, so pixel edits show without any rebuild; ``decal_resolution`` is ignored.

Staleness and undo
    Memfile undo restores properties but not pixels written from Python. The pixels of a vector decal are
    therefore validated by ``_applied`` (link id -> signature the image pixels were rendered for): the
    producer reports no stored signature while it differs from the link's stored one. ``_applied`` is cleared
    by the undo_post / redo_post handlers (lifecycle), because a memfile restore can bring back both the link
    and its signature together with a blank image; every vector decal then rebuilds exactly once. Load clears
    it too. The image carries ``seam_decal_sig``; after a file load a packed image with a
    matching property is trusted (no re-render).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np

from .. import bridge, compat
from ..seam_core import links as L
from ..seam_core import raster as R
from ..seam_core import render2d
from . import recipes

IMAGE_PROP = "seam_decal"          # image custom property: id of the owning link
SIG_PROP = "seam_decal_sig"        # image custom property: signature the pixels were rendered for
IMAGE_NODE = "SEAM Decal Image"
MIX_NODE = "SEAM Decal Mix"
ROLE = "DECAL"                     # value of material["seam_material_role"]
DRAW_ROLES = ("SHAPE", "TEXT", "RASTER")
SOURCE_ROLES = ("SHAPE", "TEXT", "RASTER", "GROUP")
SUPERSAMPLE = 3
MAX_RENDER_SIDE = 4096             # longer side cap of a vector decal (float64 canvas: 4096 px ~ 0.5 GB transient)
MAX_NAME = 56

_applied: Dict[str, str] = {}      # link id -> signature of the pixels currently in the image


# ------------------------------------------------------------------ mode / lookup

def is_decal(link) -> bool:
    return link.recipe == "DECAL"


def mode_of(link) -> str:
    """PLANE when the target is the RESULT plane owned by this link, MATERIAL for any other mesh."""
    tgt = link.target
    if tgt is not None and tgt.seam.role == "RESULT" and tgt.seam.owner_id == link.id:
        return "PLANE"
    return "MATERIAL"


def owns_target_object(link) -> bool:
    """True when the link created (and may delete) its target object."""
    return mode_of(link) == "PLANE"


def source_role(obj) -> str:
    return obj.seam.role if obj is not None else ""


def raster_doc(scene, src):
    from ..paint import rasterdoc
    for d in scene.seam.rasters:
        if d.id == src.seam.raster_id:
            return d
    return rasterdoc.doc_for_object(src) if hasattr(rasterdoc, "doc_for_object") else None


def _hidden(obj) -> bool:
    try:
        return bool(obj.hide_viewport or obj.hide_get())
    except (RuntimeError, ReferenceError):
        return False


def members(scene, src) -> list:
    """Artwork objects rendered for ``src`` (bottom -> top). A GROUP contributes its visible
    SHAPE/TEXT/RASTER descendants; clip masks are never drawn (unless they are the source itself)."""
    from ..vector import export_png as ex
    if src.seam.role != "GROUP":
        return [src] if src.seam.role in DRAW_ROLES else []
    cand = {o.name for o in src.children_recursive if o.seam.role in DRAW_ROLES and not _hidden(o)}
    masks = ex.clip_masks(scene)
    ordered = ex.artwork_in_order(scene, DRAW_ROLES, include_hidden=True, include_masks=True)
    return [o for o in ordered if o.name in cand and o.name not in masks]


def depends_on(link, obj) -> bool:
    """Does a change of ``obj`` (Object update) possibly change the decal? Source, group descendants, clip masks."""
    src = link.source
    if src is None or obj is None:
        return False
    if obj == src:
        return True
    if src.seam.role != "GROUP":
        return src.seam.clip_source == obj
    p = obj.parent
    while p is not None:
        if p == src:
            return True
        p = p.parent
    return any(c.seam.clip_source == obj for c in src.children_recursive)


def depends_on_data(link, data) -> bool:
    """Same for a Curve / TextCurve datablock used by a group child or clip mask."""
    src = link.source
    if src is None or src.seam.role != "GROUP":
        return False
    for c in src.children_recursive:
        if c.data == data or (c.seam.clip_source is not None and c.seam.clip_source.data == data):
            return True
    return False


# ------------------------------------------------------------------ signatures

def _r(v, n=5) -> str:
    return "%.*f" % (n, float(v))


def style_parts(style) -> List[str]:
    parts = [str(style.fill_enabled), style.fill_type, style.fill_rule, str(style.stroke_enabled),
             style.stroke_cap, style.stroke_join]
    for v in (style.gradient_angle, style.stroke_width, style.miter_limit, style.opacity, style.gradient_radius):
        parts.append(_r(v))
    for c in (style.fill_color, style.stroke_color, style.gradient_center):
        parts.extend(_r(x) for x in c)
    for stop in style.gradient_stops:
        parts.append(_r(stop.position))
        parts.extend(_r(x) for x in stop.color)
    return parts


def _affine_parts(obj, origin) -> List[str]:
    """Canvas affine with the source's own translation removed: moving the whole artwork on the canvas
    does not change the rendered image (the decal covers the artwork bounds), but rotating, scaling or
    moving members relative to each other does."""
    from ..vector import export_png as ex
    a, b, c, d, e, f = ex.live_canvas_affine(obj)
    return [_r(x, 4) for x in (a, b, c, d, e - origin[0], f - origin[1])]


def source_origin(src) -> Tuple[float, float]:
    from ..vector import export_png as ex
    a = ex.live_canvas_affine(src)
    return a[4], a[5]


def object_parts(scene, obj, origin=(0.0, 0.0)) -> List[str]:
    from ..vector import export_png as ex
    parts = [obj.seam.id, obj.seam.role, obj.type] + _affine_parts(obj, origin)
    if obj.seam.role == "RASTER":
        doc = raster_doc(scene, obj)
        parts.append(raster_revision(doc))
        b = ex.raster_bounds(obj)
        if b:
            parts.extend(_r(x, 4) for x in (b[0] - origin[0], b[1] - origin[1], b[2] - origin[0], b[3] - origin[1]))
        parts.append(_r(obj.seam.style.opacity))
        return parts
    parts.append(bridge.paths_signature(ex.object_paths(obj)))
    parts.extend(style_parts(obj.seam.style))
    mask = obj.seam.clip_source
    if mask is not None and mask is not obj:
        parts.append("clip")
        parts.append(mask.seam.id)
        parts.extend(_affine_parts(mask, origin))
        parts.append(bridge.paths_signature(ex.object_paths(mask)))
        parts.append(mask.seam.style.fill_rule)
    return parts


def raster_revision(doc) -> str:
    """Structure + pixel revision of a raster document (paint's cache signature)."""
    if doc is None:
        return "no-doc"
    from ..paint import rasterdoc
    fn = getattr(rasterdoc, "_cache_signature", None) or getattr(rasterdoc, "material_signature", None)
    revs = []
    for ly in doc.layers:
        revs.append("%s:%d:%d" % (ly.id, int(getattr(ly, "rev", 0)), int(getattr(ly, "mask_rev", 0))))
    return "%s|%s|%dx%d" % (fn(doc) if fn else "", ",".join(revs), doc.width, doc.height)


def signature(scene, link, missing: str) -> Optional[str]:
    src = link.source
    if src is None or src.name not in scene.objects:
        return missing
    mode = mode_of(link)
    params = {k: getattr(link.params, k) for k in L.PARAMS}
    used = {k: params[k] for k in L.RECIPE_PARAMS["DECAL"] if k != "decal_mode"}
    h = hashlib.blake2b(digest_size=16)

    def put(s):
        h.update(str(s).encode("utf-8", "replace"))
        h.update(b"\x1f")

    put("decal-v1")
    put(mode)
    put(L.params_signature("DECAL", dict(used, decal_mode=mode)))
    put(src.seam.role)
    origin = source_origin(src)
    if src.seam.role == "RASTER":
        # pixels are shared through doc.composite; only the extent/structure matters for the link
        put("\x1e".join(object_parts(scene, src, origin)))
    else:
        for o in members(scene, src):
            put("\x1e".join(object_parts(scene, o, origin)))
    return h.hexdigest()


def stored_signature(link) -> Optional[str]:
    """The link's stored signature, but only while the image pixels are known to match it."""
    sig = link.source_signature
    if src_is_raster(link):
        return sig
    return sig if (sig and _applied.get(link.id) == sig) else None


def src_is_raster(link) -> bool:
    return link.source is not None and link.source.seam.role == "RASTER"


def mark_applied(link, signature: str) -> None:
    _applied[link.id] = signature
    img = link.decal_image
    if img is not None and not src_is_raster(link):
        try:
            img[SIG_PROP] = signature
        except (ReferenceError, TypeError):
            pass


def forget(link_id: str) -> None:
    _applied.pop(link_id, None)


def seed_after_load(scene) -> int:
    """Trust packed decal images whose recorded signature equals the link's (no re-render after a load)."""
    n = 0
    for link in scene.seam.links:
        if link.recipe != "DECAL" or link.decal_image is None:
            continue
        img = link.decal_image
        try:
            if img.packed_file is not None and link.source_signature and img.get(SIG_PROP) == link.source_signature:
                _applied[link.id] = link.source_signature
                n += 1
        except ReferenceError:
            continue
    return n


# ------------------------------------------------------------------ rendering

def _items_bounds(items) -> Optional[Tuple[float, float, float, float]]:
    from ..vector import export_png as ex
    lo_x = lo_y = float("inf")
    hi_x = hi_y = float("-inf")
    for it in items:
        if isinstance(it, render2d.FillItem):
            b = ex.rings_bounds(it.rings)
        else:
            b = (it.x, it.y_top - it.height, it.x + it.width, it.y_top)
        if b is None:
            continue
        lo_x, lo_y = min(lo_x, b[0]), min(lo_y, b[1])
        hi_x, hi_y = max(hi_x, b[2]), max(hi_y, b[3])
    if lo_x > hi_x:
        return None
    return lo_x, lo_y, hi_x, hi_y


def image_size(width: float, height: float, longest: int) -> Tuple[int, int]:
    """Pixel size whose longer side is ``longest`` (aspect kept, at least 1 px)."""
    longest = max(1, int(longest))
    if width >= height:
        return longest, max(1, int(round(longest * height / width)))
    return max(1, int(round(longest * width / height))), longest


def render_vector(scene, link, params: dict):
    """Rasterize the source artwork -> (uint8 RGBA H x W x 4, bounds (minx, miny, maxx, maxy), messages)."""
    from ..vector import export_png as ex
    src = link.source
    mem = members(scene, src)
    warnings: List[str] = []
    items = []
    for o in mem:
        if o.seam.role == "RASTER":
            it = ex.raster_item(scene, o, warnings=warnings)
            new = [it] if it is not None else []
        else:
            new = ex.object_items(o, warnings=warnings)
        items.extend(new)
    bounds = _items_bounds(items)
    if bounds is None:
        raise recipes.RecipeError("Nothing to render: the source has no visible fill or stroke")
    w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if w <= 1e-9 or h <= 1e-9:
        raise recipes.RecipeError("The source has no area (zero width or height)")
    want = int(params["decal_resolution"])
    if want > MAX_RENDER_SIDE:
        warnings.insert(0, "Decal resolution limited to %d px (requested %d)" % (MAX_RENDER_SIDE, want))
    pw, ph = image_size(w, h, min(want, MAX_RENDER_SIDE))
    img = render2d.render(items, pw, ph, (bounds[0], bounds[3], bounds[2], bounds[1]), supersample=SUPERSAMPLE)
    return R.to_uint8_srgb(img), bounds, warnings


# ------------------------------------------------------------------ image

def image_name(src) -> str:
    return ("SEAM Decal " + src.name)[:MAX_NAME]


def _new_image(name: str, w: int, h: int):
    img = bpy.data.images.new(name, int(w), int(h), alpha=True, float_buffer=False)
    img.alpha_mode = "STRAIGHT"
    img.colorspace_settings.name = "sRGB"
    return img


def ensure_image(link, w: int, h: int):
    """The link's own decal image at ``w x h`` (created once, rescaled in place, never replaced)."""
    img = link.decal_image
    owned = img is not None and img.get(IMAGE_PROP) == link.id
    if img is None or not owned:
        img = _new_image(image_name(link.source), w, h)
        img[IMAGE_PROP] = link.id
        link.decal_image = img
        return img
    if (int(img.size[0]), int(img.size[1])) != (int(w), int(h)):
        img.scale(int(w), int(h))
    want = image_name(link.source)
    if not img.name.startswith(want[:MAX_NAME - 4]):
        img.name = want
    return img


def frozen_copy(img, name: Optional[str] = None):
    """An independent, packed 8-bit copy of ``img`` (same size, colour space and pixels). Baked results use it so they
    stop following (and cannot lose) an image someone else owns, e.g. a raster document composite."""
    w, h = int(img.size[0]), int(img.size[1])
    new = bpy.data.images.new((name or (img.name + " (baked)"))[:63], max(1, w), max(1, h), alpha=True,
                              float_buffer=False)
    new.alpha_mode = img.alpha_mode
    try:
        new.colorspace_settings.name = img.colorspace_settings.name
    except TypeError:
        pass
    if w and h:
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        new.pixels.foreach_set(buf)
        new.update()
    try:
        new.pack()
    except RuntimeError:
        pass
    return new


def freeze_material_image(mat):
    """Give a (copied) decal material its own frozen image; returns the new image or None."""
    node = mat.node_tree.nodes.get(IMAGE_NODE) if mat is not None and mat.node_tree is not None else None
    if node is None or node.image is None:
        return None
    node.image = frozen_copy(node.image)
    return node.image


def bake_image(link) -> bool:
    """Detach / remove: a decal that shares an image it does not own (raster composite) gets a frozen copy, and its
    own vector image is packed (it is no longer re-rendered). Returns True when the material now uses a copy."""
    img = link.decal_image
    if link.recipe != "DECAL" or img is None:
        return False
    if img.get(IMAGE_PROP) == link.id:
        try:
            if img.packed_file is None or img.is_dirty:
                img.pack()
        except RuntimeError:
            pass
        return False
    copy = frozen_copy(img, "SEAM Decal " + (link.source.name if link.source is not None else link.name))
    copy[IMAGE_PROP] = link.id
    for _i, mat in owned_slots(link):
        sync_image_node(mat, copy)
    link.decal_image = copy
    return True


FOLLOWING_STATES = ("LIVE", "ERROR")


def freeze_raster_users(image, removing_source: bool = True) -> int:
    """Before a raster document removes its composite ``image``, give the decals that show it a frozen copy so they
    keep their picture. ``removing_source=True`` (the document is deleted: its links become MISSING_SOURCE, which
    keeps the result) freezes every user; ``False`` (the composite is only replaced, e.g. a canvas resize) freezes
    only links that do not follow the document right now (paused / conflict / ...); following links re-point to
    the new composite on their next build. Returns the number of links frozen.

    The paint module calls it right before ``bpy.data.images.remove(doc.composite)`` (remove_doc, resize_canvas)."""
    if image is None:
        return 0
    n = 0
    for sc in bpy.data.scenes:
        seam = getattr(sc, "seam", None)
        for link in (seam.links if seam is not None else ()):
            if link.recipe != "DECAL" or link.decal_image != image:
                continue
            if not removing_source and link.state in FOLLOWING_STATES:
                continue
            if bake_image(link):
                n += 1
    return n


def share_raster_image(scene, link):
    """Point the link at the document composite (kept current by the paint producer)."""
    from ..paint import rasterdoc
    doc = raster_doc(scene, link.source)
    if doc is None:
        raise recipes.RecipeError("The raster source has no document")
    if doc.composite is None:
        rasterdoc.update_composite(doc)
    link.decal_image = doc.composite
    return doc.composite


# ------------------------------------------------------------------ plane geometry

def plane_built(width_px: float, height_px: float, params: dict) -> recipes.Built:
    """Quad (4 corners, UV 0..1, material 0) of ``width_px x height_px`` canvas px at ``units_scale``."""
    s = params["units_scale"]
    w, h = width_px * s, height_px * s
    if params["orientation"] == "UPRIGHT":
        v = [(-w / 2, 0.0, 0.0), (w / 2, 0.0, 0.0), (w / 2, 0.0, h), (-w / 2, 0.0, h)]
    else:
        v = [(-w / 2, -h / 2, 0.0), (w / 2, -h / 2, 0.0), (w / 2, h / 2, 0.0), (-w / 2, h / 2, 0.0)]
    return recipes.Built(np.asarray(v, dtype=np.float32), np.array([4], dtype=np.int32),
                         np.array([0, 1, 2, 3], dtype=np.int32), np.array([0], dtype=np.int32),
                         np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32), [])


# ------------------------------------------------------------------ materials

def _principled(nodes):
    for n in nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    return None


def _clear_nodes(mat) -> None:
    if mat.node_tree is None:
        mat.use_nodes = True
    mat.node_tree.nodes.clear()


def _new_material(name: str, link_id: str):
    mat = bpy.data.materials.new(name)
    _clear_nodes(mat)
    mat["seam_link_id"] = link_id
    mat["seam_material_role"] = ROLE
    return mat


def new_plane_material(link_id: str):
    """Principled surface showing the decal image (colour + alpha), alpha blended."""
    mat = _new_material("SEAM Decal", link_id)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)
    bsdf.inputs["Roughness"].default_value = 0.7
    tex = nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = IMAGE_NODE
    tex.location = (-300, 0)
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    compat.set_material_blend_alpha(mat)
    return mat


def new_overlay_material(link_id: str, base_rgba):
    """``mix(base colour, decal colour, decal alpha)`` into a Principled surface (opaque)."""
    mat = _new_material("SEAM Decal", link_id)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (500, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    bsdf.inputs["Roughness"].default_value = 0.6
    tex = nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = IMAGE_NODE
    tex.location = (-300, 0)
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    mix, fac, a, b, res = compat.new_mix_rgb_node(nodes, "MIX")
    mix.name = mix.label = MIX_NODE
    mix.location = (0, 0)
    a.default_value = tuple(float(c) for c in base_rgba)
    links.new(tex.outputs["Alpha"], fac)
    links.new(tex.outputs["Color"], b)
    links.new(res, bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat.diffuse_color = tuple(float(c) for c in base_rgba)
    return mat


def base_color_of(mesh):
    """Colour the first material of ``mesh`` shows (Principled Base Color, else viewport colour)."""
    for m in mesh.materials:
        if m is None:
            continue
        if m.node_tree is not None:
            p = _principled(m.node_tree.nodes)
            if p is not None:
                return tuple(float(c) for c in p.inputs["Base Color"].default_value)
        return tuple(float(c) for c in m.diffuse_color)
    return (0.8, 0.8, 0.8, 1.0)


def owned_slots(link) -> List[Tuple[int, object]]:
    tgt = link.target
    if tgt is None or tgt.data is None:
        return []
    return [(i, m) for i, m in enumerate(tgt.data.materials) if m is not None and m.get("seam_link_id") == link.id
            and m.get("seam_material_role") == ROLE]


def ensure_material(link):
    """Make sure the target has the SEAM decal material (adds a slot when it is missing); returns it."""
    owned = owned_slots(link)
    if owned:
        return owned[0][1]
    me = link.target.data
    if mode_of(link) == "PLANE":
        mat = new_plane_material(link.id)
    else:
        mat = new_overlay_material(link.id, base_color_of(me))
    me.materials.append(mat)
    return mat


def sync_image_node(mat, image) -> bool:
    """Point the decal material's image node at ``image``; returns True when it changed."""
    if mat is None or mat.node_tree is None:
        return False
    node = mat.node_tree.nodes.get(IMAGE_NODE)
    if node is None:
        return False
    if node.image != image:
        node.image = image
        return True
    return False


def assign_faces(link, mat_index: Optional[int] = None) -> int:
    """One-time helper of the creating operator: put every face on the SEAM slot. Returns face count."""
    me = link.target.data
    if mat_index is None:
        owned = owned_slots(link)
        if not owned:
            return 0
        mat_index = owned[0][0]
    n = len(me.polygons)
    if n:
        me.polygons.foreach_set("material_index", np.full(n, mat_index, dtype=np.int32))
        me.update()
    return n


def remove_owned_slots(link) -> int:
    """Remove the SEAM-owned material slots from the target (used when a MATERIAL link is removed)."""
    removed = 0
    tgt = link.target
    if tgt is None or tgt.data is None:
        return 0
    for idx, _m in reversed(owned_slots(link)):
        tgt.data.materials.pop(index=idx)
        removed += 1
    return removed


# ------------------------------------------------------------------ build

def build(scene, link, signature: str, params: dict) -> Tuple[Optional[recipes.Built], List[str]]:
    """Update the image (and return the plane geometry for PLANE mode). Raises :class:`RecipeError`."""
    from ..vector import export_png as ex
    src = link.source
    mode = mode_of(link)
    messages: List[str] = []
    if src.seam.role == "RASTER":
        doc = raster_doc(scene, src)
        img = share_raster_image(scene, link)
        b = ex.raster_bounds(src)
        if b is None or b[2] - b[0] <= 0 or b[3] - b[1] <= 0:
            raise recipes.RecipeError("The raster plane has no area")
        w, h = b[2] - b[0], b[3] - b[1]
        messages.append("Raster decal shares the document composite")
    else:
        a8, bounds, warns = render_vector(scene, link, params)
        messages.extend(warns[:3])
        h_px, w_px = a8.shape[0], a8.shape[1]
        img = ensure_image(link, w_px, h_px)
        from ..paint import rasterdoc
        rasterdoc.write_image_srgb8(img, a8)
        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    mat = ensure_material(link)
    sync_image_node(mat, img)
    if mode == "MATERIAL":
        if not len(link.target.data.uv_layers):
            messages.append("The mesh has no UV map: unwrap it to see the decal")
        return None, messages
    return plane_built(w, h, params), messages
