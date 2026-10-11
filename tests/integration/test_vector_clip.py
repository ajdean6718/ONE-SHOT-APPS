"""Clipping masks: derived viewport result, release, persistence, export (vector-b)."""

import bpy
import numpy as np
import pytest

from helpers import flush, save_and_reload
from test_vector_b_common import add_ellipse, add_rect, mesh_area, pixel, require, select, srgb8


def clip_child(obj):
    for ch in obj.children:
        if ch.seam.role == "CLIP_RESULT":
            return ch
    return None


@pytest.fixture
def clipped(seam, engine):
    require("geometry.flatten_path", "boolean2d.boolean", "geometry.triangulate")
    scene = bpy.context.scene
    a = add_rect(scene, "Art", 0, 0, 100, 100, color=(1.0, 0.0, 0.0, 1.0))
    mask = add_rect(scene, "Mask", 50, 50, 150, 150, color=(0.0, 1.0, 0.0, 1.0))
    select(a, mask)
    bpy.ops.seam.clip_make()
    flush()
    return a, mask


def test_clip_make_sets_source_and_derives_result(clipped):
    a, mask = clipped
    assert a.seam.clip_source is mask
    child = clip_child(a)
    assert child is not None and child.type == "MESH"
    assert child.seam.owner_id == a.seam.id and child.parent is a
    assert mesh_area(child) == pytest.approx(2500.0, rel=1e-4)          # 50 x 50 overlap
    assert child.hide_select
    # display approach: original path shown as wire, mask as wire, result is the visible fill
    assert a.display_type == "WIRE" and mask.display_type == "WIRE"
    assert len(child.data.materials) == 2
    mat = child.data.materials[0]
    assert tuple(mat.diffuse_color)[:3] == pytest.approx((1.0, 0.0, 0.0), abs=1e-5)
    # the mask itself is not clipped and keeps its geometry
    assert mask.seam.clip_source is None


def test_clip_result_is_live(clipped):
    a, mask = clipped
    child = clip_child(a)
    mask.location.x += 25.0                                   # overlap becomes 25 x 50
    res = flush()
    assert any(r.key.startswith("clip:") and r.status == "ok" for r in res), res
    assert mesh_area(child) == pytest.approx(25 * 50.0, rel=1e-4)
    # edit a node of the artwork
    a.data.splines[0].bezier_points[2].co.x = 200.0           # (100,100) -> (200,100) widens the quad
    flush()
    assert mesh_area(child) > 25 * 50.0
    # style change updates the viewport material
    a.seam.style.fill_color = (0.0, 0.0, 1.0, 1.0)
    flush()
    assert tuple(child.data.materials[0].diffuse_color)[:3] == pytest.approx((0.0, 0.0, 1.0), abs=1e-5)
    # unchanged -> skipped (idempotent)
    assert all(r.status == "skipped" for r in flush() if r.key.startswith("clip:"))
    # moving the clipped object moves the (child) result with it; area is unchanged by pure translation of both
    mask.location.x -= 25.0
    a.location.x += 0.0
    flush()
    assert mesh_area(child) > 0


def test_clip_follows_object_transform(clipped):
    a, mask = clipped
    child = clip_child(a)
    a.scale = (2.0, 1.0, 1.0)                                  # art becomes 200 x 100 around its center (50, 50)
    flush()
    # canvas-space overlap: art x in [-50, 150] ∩ mask x in [50,150] = 100, y in [50,100] -> 100 x 50 = 5000
    s = a.scale
    assert mesh_area(child) * s[0] * s[1] == pytest.approx(5000.0, rel=1e-3)


def test_clip_release_restores_everything(clipped):
    a, mask = clipped
    select(a)
    bpy.ops.seam.clip_release()
    assert a.seam.clip_source is None
    assert clip_child(a) is None
    assert a.display_type == "TEXTURED" and mask.display_type == "TEXTURED"
    assert not [o for o in bpy.data.objects if o.seam.role == "CLIP_RESULT"]
    assert not [m for m in bpy.data.materials if m.name.startswith("SEAM Clip")]
    # releasing through the mask works too
    select(a, mask)
    bpy.ops.seam.clip_make()
    flush()
    assert clip_child(a) is not None
    select(mask)
    bpy.ops.seam.clip_release()
    assert a.seam.clip_source is None and clip_child(a) is None
    with pytest.raises(RuntimeError, match="poll"):
        select(a)
        bpy.ops.seam.clip_release()


def test_clip_clearing_property_directly_cleans_up(clipped):
    a, mask = clipped
    a.seam.clip_source = None                       # e.g. through the properties UI
    a.update_tag()                                  # (REQUEST: props.py clip_source should carry update=_tag)
    res = flush()
    assert any(r.key.startswith("clip:") and r.status == "ok" for r in res)
    assert clip_child(a) is None and a.display_type == "TEXTURED"


def test_clip_persists_across_save_and_reload(clipped, blend_path):
    a, mask = clipped
    aname, mname = a.name, mask.name
    save_and_reload(blend_path)
    from seam_studio.engine import core
    core.request_revalidate()
    res = [r for r in core.flush() if r.key.startswith("clip:")]
    assert [r.status for r in res] == ["skipped"]
    a, mask = bpy.data.objects[aname], bpy.data.objects[mname]
    assert a.seam.clip_source is mask
    assert mesh_area(clip_child(a)) == pytest.approx(2500.0, rel=1e-4)


def test_stroke_is_clipped_too(seam, engine):
    require("geometry.flatten_path", "boolean2d.boolean", "boolean2d.union_all", "geometry.triangulate",
            "stroke.stroke_outline")
    scene = bpy.context.scene
    a = add_rect(scene, "Art", 0, 0, 100, 100, color=(1, 0, 0, 1), stroke=((0, 0, 0, 1), 10.0))
    a.seam.style.fill_enabled = False
    mask = add_rect(scene, "Mask", -100, -100, 100, 50)         # clips the stroke to y <= 50
    select(a, mask)
    bpy.ops.seam.clip_make()
    flush()
    child = clip_child(a)
    mats = [p.material_index for p in child.data.polygons]
    assert mats and set(mats) == {1}                           # only stroke faces (fill disabled)
    ys = [v.co.y for v in child.data.vertices]
    # art origin is (50, 50): local y <= 0 corresponds to canvas y <= 50
    assert max(ys) <= 1e-3


def test_clip_errors(seam):
    require("geometry.flatten_path", "boolean2d.boolean")
    scene = bpy.context.scene
    a = add_rect(scene, "A", 0, 0, 100, 100)
    bpy.ops.seam.add_text(body="T", x=0, y=0, size=30)
    t = bpy.context.active_object                               # text is created last -> top-most
    select(a, t)
    with pytest.raises(RuntimeError, match="not a vector shape"):
        bpy.ops.seam.clip_make()
    assert a.seam.clip_source is None
    select(a)
    with pytest.raises(RuntimeError, match="poll"):
        bpy.ops.seam.clip_make()


def test_text_can_be_clipped(seam, engine):
    require("geometry.flatten_path", "boolean2d.boolean", "boolean2d.union_all", "geometry.triangulate")
    scene = bpy.context.scene
    bpy.ops.seam.add_text(body="HHHH", x=0, y=0, size=100)
    t = bpy.context.active_object
    flush()
    full = None
    mask = add_rect(scene, "Mask", 0, -30, 80, 200)
    select(t, mask)
    bpy.ops.seam.clip_make()
    flush()
    child = clip_child(t)
    assert child is not None and child.type == "MESH" and len(child.data.polygons) > 0
    assert max(v.co.x for v in child.data.vertices) <= 80.01
    assert t.display_type == "WIRE"


def test_clip_export_png_and_svg(clipped, tmp_path):
    require("render2d.render", "raster.to_uint8_srgb", "png.encode_png", "png.decode_png", "svg.export_svg")
    from seam_studio.seam_core import png
    a, mask = clipped
    ab = bpy.context.scene.objects["Artboard 1"] if "Artboard 1" in bpy.context.scene.objects else None
    # Make the artboard cover the area of interest
    from seam_studio.vector import canvas
    ab = canvas.artboards(bpy.context.scene)[0]
    ab.location = (0.0, 200.0, -1.0)
    ab.seam.artboard.width, ab.seam.artboard.height = 200.0, 200.0
    path = str(tmp_path / "clip.png")
    bpy.ops.seam.png_export(filepath=path, transparent=True)
    img = png.read_png(path)
    assert img.shape == (200, 200, 4)
    red = srgb8((1.0, 0.0, 0.0))
    # inside overlap -> red; inside art but outside mask -> transparent; mask itself not drawn
    assert pixel(img, 0, 200, 1, 75, 75)[:3] == red and pixel(img, 0, 200, 1, 75, 75)[3] == 255
    assert pixel(img, 0, 200, 1, 20, 20)[3] == 0
    assert pixel(img, 0, 200, 1, 125, 125)[3] == 0                 # mask-only area stays empty
    svg_path = str(tmp_path / "clip.svg")
    bpy.ops.seam.svg_export(filepath=svg_path)
    text = open(svg_path, encoding="utf-8").read()
    assert "clip-path" in text and "<clipPath" in text


def test_clip_undo_redo(seam, engine):
    require("geometry.flatten_path", "boolean2d.boolean", "geometry.triangulate")
    scene = bpy.context.scene
    a = add_rect(scene, "Art", 0, 0, 100, 100)
    mask = add_rect(scene, "Mask", 50, 50, 150, 150)
    bpy.ops.ed.undo_push(message="shapes")
    select(a, mask)
    bpy.ops.seam.clip_make()
    flush()
    bpy.ops.ed.undo_push(message="clip")
    assert clip_child(bpy.data.objects["Art"]) is not None
    bpy.ops.ed.undo()
    flush()
    a = bpy.data.objects["Art"]
    assert a.seam.clip_source is None and clip_child(a) is None and a.display_type == "TEXTURED"
    bpy.ops.ed.redo()
    flush()
    a = bpy.data.objects["Art"]
    assert a.seam.clip_source is not None
    assert mesh_area(clip_child(a)) == pytest.approx(2500.0, rel=1e-4)


def test_deleting_the_mask_releases_the_clip(clipped):
    a, mask = clipped
    assert clip_child(a) is not None
    bpy.data.objects.remove(mask, do_unlink=True)             # e.g. the user deletes the mask in the viewport
    res = flush()
    assert any(r.key.startswith("clip:") and r.status == "ok" for r in res), res
    assert clip_child(a) is None and a.display_type == "TEXTURED"
    assert a.seam.clip_source is None


def test_pathfinder_consuming_a_mask_releases_dependants(clipped):
    a, mask = clipped
    other = add_rect(bpy.context.scene, "Other", 120, 120, 200, 200)
    select(mask, other)
    bpy.ops.seam.pathfinder(op="UNION")
    assert "Mask" not in bpy.data.objects
    a = bpy.data.objects["Art"]
    assert a.seam.clip_source is None and clip_child(a) is None and a.display_type == "TEXTURED"
