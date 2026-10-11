"""Native texture painting on 3D results (owner: connect).

``setup(obj)`` prepares any mesh (typically a link RESULT) for Blender's own Texture Paint:

* ensures a UV map (``UVMap``); a mesh without any UV map gets deterministic box-projected UVs
  (``recipes.box_uvs``), a mesh that already has other UV maps keeps using its active one;
* creates (or reuses) one 8-bit transparent image ``SEAM Paint <object>`` and wires it into **every**
  material of the object as an image-texture node ``SEAM Paint Image``; the original surface keeps
  working: ``Mix Shader(fac = paint alpha, original surface, Diffuse(paint colour))`` feeds the Material
  Output, so the existing base colour (including the link-synced Face / Side colours) shows wherever
  nothing is painted;
* makes the paint node the active node of each material, so it is the active texture-paint slot.

Materials that other objects also use are copied first (single-user), so the overlay stays on this object.

Painted texels follow the UVs. Results of a live link are rebuilt with *deterministic* UVs (recipe UVs, or
the box projection which only depends on vertex positions and face orientation, never on vertex numbering),
so when the source changes topology the paint stays attached to the surface where the surface still lies at
the same place of the UV layout; where the surface moved, the paint stays at its UV position. Painting never
changes the mesh, so it does not conflict with the link.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import bpy
import numpy as np

from .. import ids
from . import recipes

PAINT_PROP = "seam_paint"          # image custom property: SEAM id of the painted object
NODE_IMAGE = "SEAM Paint Image"
NODE_MIX = "SEAM Paint Mix"
NODE_DIFFUSE = "SEAM Paint Diffuse"
MAX_NAME = 56


# ------------------------------------------------------------------ UVs

def _mesh_arrays(me):
    nv, nf, nl = len(me.vertices), len(me.polygons), len(me.loops)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    totals = np.empty(nf, dtype=np.int32)
    me.polygons.foreach_get("loop_total", totals)
    corners = np.empty(nl, dtype=np.int32)
    me.loops.foreach_get("vertex_index", corners)
    return co.reshape(nv, 3), totals, corners


def ensure_uvmap(obj) -> Tuple[str, bool]:
    """(name of the UV map to paint on, created). Creates ``UVMap`` only when the mesh has no UV map."""
    me = obj.data
    if "UVMap" in me.uv_layers:
        return "UVMap", False
    if len(me.uv_layers):
        active = me.uv_layers.active or me.uv_layers[0]
        return active.name, False
    uv = me.uv_layers.new(name="UVMap")
    if len(me.loops):
        co, totals, corners = _mesh_arrays(me)
        uv.data.foreach_set("uv", recipes.box_uvs(co, totals, corners).ravel())
    return "UVMap", True


def refresh_link_signature(scene, obj) -> None:
    """After adding UVs to a link result, accept them as the result's own data (no false CONFLICT)."""
    from . import producer
    for link in scene.seam.links:
        if link.target == obj and link.state in ("LIVE", "PAUSED", "ERROR") and link.recipe != "DECAL":
            link.target_data_signature = producer.read_data_signature(obj.data)


# ------------------------------------------------------------------ image / materials

def image_name(obj) -> str:
    return ("SEAM Paint " + obj.name)[:MAX_NAME]


def find_paint_image(obj):
    """The paint image of ``obj`` already wired by :func:`setup`, or None. Only an image created for this very
    object counts, so a duplicate (or a material copied from another object) never paints on the original's image."""
    sid = obj.seam.id
    for m in obj.data.materials:
        if m is None or m.node_tree is None:
            continue
        node = m.node_tree.nodes.get(NODE_IMAGE)
        if node is not None and node.image is not None and sid and node.image.get(PAINT_PROP) == sid:
            return node.image
    return None


def unshare_materials(obj) -> int:
    """Give ``obj``'s mesh its own copy of every material that other users also use, so wiring the paint
    overlay never shows (or paints) on other objects. Returns the number of copies made."""
    me = obj.data
    n = 0
    for i, m in enumerate(me.materials):
        if m is not None and m.users - (1 if m.use_fake_user else 0) > 1 and m.library is None:
            me.materials[i] = m.copy()
            n += 1
    return n

def new_paint_image(obj, size: int):
    size = max(16, min(8192, int(size)))
    img = bpy.data.images.new(image_name(obj), size, size, alpha=True, float_buffer=False)
    img.alpha_mode = "STRAIGHT"
    img.colorspace_settings.name = "sRGB"
    img[PAINT_PROP] = ids.ensure_id(obj)
    buf = np.zeros(size * size * 4, dtype=np.float32)
    img.pixels.foreach_set(buf)
    img.update()
    return img


def _output_node(tree):
    outs = [n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"]
    for n in outs:
        if getattr(n, "is_active_output", False):
            return n
    return outs[0] if outs else None


def _wire_material(mat, image) -> bool:
    """Add the paint overlay to ``mat`` (idempotent); returns True when nodes were created."""
    if mat.node_tree is None:
        mat.use_nodes = True
    tree = mat.node_tree
    nodes, links = tree.nodes, tree.links
    created = False
    tex = nodes.get(NODE_IMAGE)
    if tex is None:
        out = _output_node(tree)
        if out is None:
            out = nodes.new("ShaderNodeOutputMaterial")
        surf = out.inputs["Surface"]
        orig = surf.links[0].from_socket if surf.is_linked else None
        if orig is None:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (out.location.x - 600, out.location.y + 300)
            bsdf.inputs["Base Color"].default_value = tuple(mat.diffuse_color)
            orig = bsdf.outputs["BSDF"]
        x, y = out.location.x, out.location.y
        tex = nodes.new("ShaderNodeTexImage")
        tex.name = tex.label = NODE_IMAGE
        tex.location = (x - 700, y - 350)
        tex.interpolation = "Linear"
        diff = nodes.new("ShaderNodeBsdfDiffuse")
        diff.name = diff.label = NODE_DIFFUSE
        diff.location = (x - 450, y - 350)
        mix = nodes.new("ShaderNodeMixShader")
        mix.name = mix.label = NODE_MIX
        mix.location = (x - 200, y)
        links.new(tex.outputs["Color"], diff.inputs["Color"])
        links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
        links.new(orig, mix.inputs[1])
        links.new(diff.outputs["BSDF"], mix.inputs[2])
        links.new(mix.outputs["Shader"], surf)
        created = True
    if tex.image != image:
        tex.image = image
    nodes.active = tex
    for n in nodes:
        n.select = n == tex
    return created


def paint_materials(obj) -> List:
    return [m for m in obj.data.materials if m is not None]


def setup(obj, size: int = 1024):
    """Prepare ``obj`` for native texture painting. Returns a dict with the UV map name, image and materials."""
    me = obj.data
    uv_name, uv_created = ensure_uvmap(obj)
    if uv_created:
        refresh_link_signature(bpy.context.scene, obj)
    if me.uv_layers.active is None or me.uv_layers.active.name != uv_name:
        me.uv_layers.active = me.uv_layers[uv_name]
    img = find_paint_image(obj)
    created_image = img is None
    if img is None:
        img = new_paint_image(obj, size)
    unshare_materials(obj)
    mats = paint_materials(obj)
    if not mats:
        mat = bpy.data.materials.new("SEAM Paint")
        if mat.node_tree is None:
            mat.use_nodes = True  # 4.x: materials start without nodes (5.x: always node based)
        me.materials.append(mat)
        mats = [mat]
    for mat in mats:
        _wire_material(mat, img)
    if obj.active_material_index >= len(mats) or obj.active_material is None:
        obj.active_material_index = 0
    # Blender derives the texture-paint slots from the image-texture nodes when Texture Paint mode is entered;
    # the paint node is the active node of every material, so it is the active slot of each of them.
    return {"uv_map": uv_name, "uv_created": uv_created, "image": img, "image_created": created_image,
            "materials": mats}


def enter_texture_paint(context, obj) -> Optional[str]:
    """Switch ``obj`` to Texture Paint mode. Returns an error text or None."""
    vl = context.view_layer
    if vl.objects.get(obj.name) != obj:
        return "The object is not in the current view layer"
    if context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
    for o in list(context.selected_objects):
        o.select_set(False)
    obj.select_set(True)
    vl.objects.active = obj
    try:
        bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
    except RuntimeError as exc:
        return str(exc).strip().splitlines()[-1] if str(exc).strip() else "Could not enter Texture Paint mode"
    return None


def pack_paint_images() -> int:
    """Pack SEAM paint images before saving (unpacked generated images would be lost)."""
    n = 0
    for img in bpy.data.images:
        if img.get(PAINT_PROP) is None or img.library is not None:
            continue
        try:
            if img.packed_file is None or img.is_dirty:
                img.pack()
                n += 1
        except Exception:
            import traceback
            traceback.print_exc()
    return n
