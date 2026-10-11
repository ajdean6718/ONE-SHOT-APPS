"""Scenario 1 - vector artwork.

Create shapes and text, edit nodes and typography, save/reopen, export SVG + PNG, and confirm that no 3D
object was generated on the way.  Everything goes through ``bpy.ops.seam.*`` / native Edit Mode operators.
"""

import xml.etree.ElementTree as ET

import bpy
import numpy as np
import pytest

from accsupport import (export_png, flush, move_nodes, new_canvas, node_positions, op, png_pixel, polygon_area_xy,
                        save_and_reopen, select, srgb8)

RED, GREEN, BLUE = (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
YELLOW, MAGENTA = (1.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0)
WHITE8, BLACK8 = (255, 255, 255, 255), (0, 0, 0, 255)


def _bounds(ob):
    from seam_studio.vector import canvas
    return canvas.canvas_bounds(ob)


def _svg_paths(path):
    root = ET.parse(path).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    return root, [e for e in root.iter() if e.tag == ns + "path"]


def _build_artwork():
    """Six shapes + one text on an 800x600 artboard; returns (artboard, dict of objects)."""
    ab = new_canvas(800, 600)
    art = {
        "rect": _shape("RECT", 200, -150, 200, 100, RED),
        "ellipse": _shape("ELLIPSE", 600, -150, 120, 120, BLUE),
        "star": _shape("STAR", 200, -400, 160, 160, GREEN, sides=5, inner_ratio=0.5),
        "hex": _shape("POLYGON", 700, -400, 100, 100, YELLOW, sides=6),
        "rrect": _shape("ROUNDED_RECT", 400, -520, 200, 60, MAGENTA, radius=20),
    }
    op(bpy.ops.seam.add_shape, kind="LINE", x=200, y=-580, width=300, height=0)
    line = bpy.context.active_object
    line.seam.style.stroke_color = (0.0, 0.0, 0.0, 1.0)
    line.seam.style.stroke_width = 10.0
    art["line"] = line
    op(bpy.ops.seam.add_text, body="SEAM", x=450.0, y=-420.0, size=80.0)
    art["text"] = bpy.context.active_object
    flush()
    return ab, art


def _shape(kind, x, y, w, h, color, **kw):
    op(bpy.ops.seam.add_shape, kind=kind, x=x, y=y, width=w, height=h, **kw)
    ob = bpy.context.active_object
    ob.seam.style.fill_color = color
    return ob


def _assert_no_3d(scene):
    from seam_studio.vector import canvas
    assert len(scene.seam.links) == 0
    assert [o.name for o in scene.objects if o.seam.role == "RESULT"] == []
    res = bpy.data.collections.get("SEAM Results")
    assert res is None or len(res.all_objects) == 0
    derived_ok = {"ARTBOARD", "STROKE", "OUTLINE", "CLIP_RESULT"}
    stray = [o.name for o in scene.objects if o.type == "MESH" and o.seam.role not in derived_ok]
    assert stray == []
    assert canvas.artwork_collection(scene).hide_render  # artwork never renders in 3D


def test_shapes_have_exact_geometry_and_pixels(seam, engine, tmp_path):
    ab, art = _build_artwork()
    # parametric geometry: bounds are exactly the requested box
    assert _bounds(art["rect"]) == pytest.approx((100, -200, 300, -100), abs=1e-3)
    assert _bounds(art["ellipse"]) == pytest.approx((540, -210, 660, -90), abs=0.05)
    assert _bounds(art["rrect"]) == pytest.approx((300, -550, 500, -490), abs=1e-3)
    assert len(art["star"].data.splines[0].bezier_points) == 10
    assert len(art["hex"].data.splines[0].bezier_points) == 6
    assert art["line"].seam.shape.kind == "LINE" and not art["line"].seam.style.fill_enabled

    img = export_png(tmp_path / "a.png")
    assert img.shape == (600, 800, 4) and img.dtype == np.uint8
    p = lambda x, y: png_pixel(img, ab, x, y)  # noqa: E731
    assert p(200, -150) == (255, 0, 0, 255)                       # rectangle centre
    assert p(99, -150) == WHITE8 and p(301, -150) == WHITE8       # just outside the left / right edge
    assert p(600, -150) == (0, 0, 255, 255)                       # ellipse centre
    assert p(542, -208) == WHITE8                                 # ellipse bounding-box corner is empty
    assert p(200, -400) == (0, 255, 0, 255)                       # star centre
    assert p(200, -400 + 76) == (0, 255, 0, 255)                  # top tip region is filled ...
    assert p(200 + 70, -400 + 70) == WHITE8                       # ... the star's concave side is not
    assert p(700, -400) == (255, 255, 0, 255)                     # hexagon
    assert p(400, -520) == (255, 0, 255, 255)                     # rounded rect centre
    assert p(301, -491) == WHITE8                                 # rounded rect corner is cut
    assert p(320, -520) == (255, 0, 255, 255)
    assert p(200, -580) == BLACK8 and p(200, -576) == BLACK8      # stroke band 10 px wide
    assert p(200, -570) == WHITE8 and p(200, -590) == WHITE8 and p(40, -580) == WHITE8 and p(360, -580) == WHITE8
    # filled-area accounting against the analytic shapes (anti-aliasing allows 0.5 %)
    # coverage of a pure colour over white = 1 - (the channel that the colour has at 0) / 255
    red_cov = (1.0 - img[..., 1] / 255.0)[(img[..., 0] == 255) & (img[..., 1] == img[..., 2])]
    assert red_cov.sum() == pytest.approx(200 * 100, rel=0.005)
    blue_cov = (1.0 - img[..., 0] / 255.0)[(img[..., 2] == 255) & (img[..., 0] == img[..., 1])]
    assert blue_cov.sum() == pytest.approx(np.pi * 60 * 60, rel=0.02)
    black_text = np.all(img[360:425, 440:640, :3] == 0, axis=-1)
    assert black_text.sum() > 600                                 # "SEAM" is inked in its own box
    assert _no_ink(img, rows=(300, 360), cols=(440, 640))          # nothing above the glyphs, no stray ink
    _assert_no_3d(bpy.context.scene)


def _no_ink(img, rows, cols):
    sub = img[rows[0]:rows[1], cols[0]:cols[1], :3]
    return bool(np.all(sub == 255))


def test_edit_nodes_then_everything_follows(seam, engine, tmp_path):
    ab, art = _build_artwork()
    rect = art["rect"]
    before = export_png(tmp_path / "before.png")
    assert png_pixel(before, ab, 350, -150) == WHITE8
    # drag the two right-hand nodes 100 px to the right (Edit Mode, G X 100)
    move_nodes(rect, [1, 2], dx=100.0)
    assert node_positions(rect) == [(-100.0, -50.0), (200.0, -50.0), (200.0, 50.0), (-100.0, 50.0)]
    assert rect.seam.shape.kind == "NONE"                         # nodes own the geometry now
    assert _bounds(rect) == pytest.approx((100, -200, 400, -100), abs=1e-3)
    flush()
    assert rect.seam.shape.kind == "NONE" and _bounds(rect) == pytest.approx((100, -200, 400, -100), abs=1e-3)
    after = export_png(tmp_path / "after.png")
    assert png_pixel(after, ab, 350, -150) == (255, 0, 0, 255)
    red = (1.0 - after[..., 1] / 255.0)[(after[..., 0] == 255) & (after[..., 1] == after[..., 2])]
    assert red.sum() == pytest.approx(300 * 100, rel=0.005)
    # only the edited shape changed: every other pixel is identical
    diff = np.any(before != after, axis=-1)
    ys, xs = np.nonzero(diff)
    assert xs.min() >= 300 - 1 and xs.max() <= 400 and ys.min() >= 100 - 1 and ys.max() <= 200

    # a curved edit: pull an ellipse node out with its handle (Edit Mode), area grows accordingly
    ell = art["ellipse"]
    area0 = abs(polygon_area_xy([tuple(bp.co[:2]) for bp in ell.data.splines[0].bezier_points]))
    move_nodes(ell, [0], dx=30.0)
    area1 = abs(polygon_area_xy([tuple(bp.co[:2]) for bp in ell.data.splines[0].bezier_points]))
    assert area1 > area0
    assert _bounds(ell)[2] == pytest.approx(690.0, abs=0.5)
    img = export_png(tmp_path / "ell.png")
    assert png_pixel(img, ab, 685, -150) == (0, 0, 255, 255)
    _assert_no_3d(bpy.context.scene)


def test_typography_and_text_editing(seam, engine, tmp_path):
    ab, art = _build_artwork()
    text = art["text"]
    cu = text.data
    w0 = _bounds(text)
    assert w0[0] == pytest.approx(453.2, abs=0.2) and w0[3] - w0[1] == pytest.approx(56.0, abs=0.5)
    # type into the text object (native Text Edit Mode)
    select(text)
    bpy.ops.object.mode_set(mode="EDIT")
    assert bpy.ops.font.text_insert(text="!") == {"FINISHED"}
    bpy.ops.object.mode_set(mode="OBJECT")
    flush()
    assert cu.body == "SEAM!"
    w1 = _bounds(text)
    assert w1[2] > w0[2] + 5.0                                    # one more glyph widens the text
    # typography: character spacing, size, alignment
    cu.space_character = 1.5
    flush()
    w2 = _bounds(text)
    assert w2[2] - w2[0] > (w1[2] - w1[0]) * 1.1 and w2[0] == pytest.approx(w1[0], abs=0.01)
    cu.size = 40.0
    flush()
    w3 = _bounds(text)
    assert (w3[3] - w3[1]) == pytest.approx((w2[3] - w2[1]) / 2.0, rel=0.05)
    # style toggle through the SEAM operator marks every character
    op(bpy.ops.seam.text_style, style="ITALIC")
    assert all(f.use_italic for f in cu.body_format)
    op(bpy.ops.seam.text_style, style="ITALIC")
    assert not any(f.use_italic for f in cu.body_format)
    op(bpy.ops.seam.text_style, style="BOLD", state="ON")
    assert all(f.use_bold for f in cu.body_format)
    # the SVG keeps text as real text when asked
    p = str(tmp_path / "t.svg")
    op(bpy.ops.seam.svg_export, filepath=p, text_mode="TEXT")
    root = ET.parse(p).getroot()
    txt = [e for e in root.iter() if e.tag.endswith("}text")]
    assert len(txt) == 1 and "".join(txt[0].itertext()) == "SEAM!"
    assert float(txt[0].get("font-size")) == pytest.approx(40.0, abs=0.01)
    # Text to Outline creates editable shapes that follow the text
    op(bpy.ops.seam.text_to_outline)
    flush()
    outlines = [o for o in bpy.context.scene.objects if o.seam.role == "SHAPE" and o.seam.owner_id == text.seam.id]
    assert len(outlines) == 1 and text.hide_get()
    assert len(outlines[0].data.splines) >= 6                      # S E A M ! + counters
    _assert_no_3d(bpy.context.scene)


def test_save_reopen_keeps_artwork_and_exports_identically(seam, engine, tmp_path, blend_path):
    ab, art = _build_artwork()
    move_nodes(art["rect"], [1, 2], dx=100.0)
    art["text"].data.space_character = 1.25
    art["star"].seam.style.stroke_enabled = True
    art["star"].seam.style.stroke_width = 6.0
    art["star"].seam.style.stroke_color = (0.0, 0.0, 0.0, 1.0)
    flush()
    names = sorted(o.name for o in bpy.context.scene.objects)
    rect_name, star_name, text_name = art["rect"].name, art["star"].name, art["text"].name
    nodes = node_positions(art["rect"])
    ids_before = {o.name: o.seam.id for o in bpy.context.scene.objects}
    ref_png = export_png(tmp_path / "ref.png")
    op(bpy.ops.seam.svg_export, filepath=str(tmp_path / "ref.svg"))

    scene = save_and_reopen(blend_path)
    flush()
    assert sorted(o.name for o in scene.objects) == names
    assert {o.name: o.seam.id for o in scene.objects} == ids_before
    rect = bpy.data.objects[rect_name]
    assert node_positions(rect) == nodes and rect.seam.shape.kind == "NONE"
    star = bpy.data.objects[star_name]
    assert star.seam.shape.kind == "STAR" and star.seam.style.stroke_enabled
    text = bpy.data.objects[text_name]
    assert text.data.body == "SEAM" and text.data.space_character == pytest.approx(1.25)
    new_png = export_png(tmp_path / "new.png")
    assert np.array_equal(ref_png, new_png)                        # byte-identical rendering after reopen
    op(bpy.ops.seam.svg_export, filepath=str(tmp_path / "new.svg"))
    assert open(tmp_path / "ref.svg").read() == open(tmp_path / "new.svg").read()
    # editing continues to work after reopening: the parametric star still follows its parameters
    star.seam.shape.sides = 7
    flush()
    assert len(star.data.splines[0].bezier_points) == 14
    _assert_no_3d(scene)


def test_svg_and_png_exports(seam, engine, tmp_path):
    ab, art = _build_artwork()
    move_nodes(art["rect"], [1, 2], dx=100.0)
    p = str(tmp_path / "doc.svg")
    op(bpy.ops.seam.svg_export, filepath=p)
    root, paths = _svg_paths(p)
    assert [float(v) for v in root.get("viewBox").split()] == [0.0, 0.0, 800.0, 600.0]
    assert root.get("width") == "800" and root.get("height") == "600"
    fills = sorted((e.get("fill") for e in paths if e.get("fill") not in (None, "none")))
    assert fills == sorted(["#ff0000", "#0000ff", "#00ff00", "#ffff00", "#ff00ff", "#000000"])
    stroked = [e for e in paths if e.get("stroke")]
    assert len(stroked) == 1 and stroked[0].get("stroke") == "#000000" and float(stroked[0].get("stroke-width")) == 10.0
    # the moved rectangle is in the file at its new size (SVG y is down)
    rect_path = next(e for e in paths if e.get("id") == art["rect"].name)
    nums = [float(v) for v in rect_path.get("d").replace("M", " ").replace("C", " ").replace("Z", " ").split()]
    xs, ys = nums[0::2], nums[1::2]
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((100.0, 400.0, 100.0, 200.0), abs=1e-3)

    # round trip into a fresh document: same number of shapes, colours, geometry
    bpy.ops.wm.read_homefile(use_empty=True)
    from seam_studio.engine import core
    core.reset()
    op(bpy.ops.seam.svg_import, filepath=p)
    flush()
    imported = [o for o in bpy.context.scene.objects if o.seam.role == "SHAPE"]
    assert len(imported) == 7                                      # 5 filled shapes + line + text outline
    reds = [o for o in imported if tuple(round(c, 2) for c in o.seam.style.fill_color[:3]) == (1.0, 0.0, 0.0)
            and o.seam.style.fill_enabled]
    assert len(reds) == 1
    b = _bounds(reds[0])
    from seam_studio.vector import canvas
    iab = canvas.artboards(bpy.context.scene)[0]
    assert (iab.seam.artboard.width, iab.seam.artboard.height) == (800.0, 600.0)
    assert (b[2] - b[0], b[3] - b[1]) == pytest.approx((300.0, 100.0), abs=1e-3)
    _assert_no_3d(bpy.context.scene)


def test_png_export_options(seam, engine, tmp_path):
    ab, art = _build_artwork()
    t = export_png(tmp_path / "t.png", transparent=True)
    assert png_pixel(t, ab, 10, -10)[3] == 0 and png_pixel(t, ab, 200, -150) == (255, 0, 0, 255)
    big = export_png(tmp_path / "x2.png", scale=2.0, transparent=False)
    assert big.shape == (1200, 1600, 4)
    assert png_pixel(big, ab, 200, -150, scale=2.0) == (255, 0, 0, 255)
    assert png_pixel(big, ab, 99, -150, scale=2.0) == WHITE8
    # hiding a layer's object removes it from the export; showing it brings it back
    art["rect"].hide_set(True)
    hidden = export_png(tmp_path / "h.png", transparent=False)
    assert png_pixel(hidden, ab, 200, -150) == WHITE8
    art["rect"].hide_set(False)
    assert png_pixel(export_png(tmp_path / "s.png", transparent=False), ab, 200, -150) == (255, 0, 0, 255)
    assert srgb8((0.5, 0.0, 0.0)) == (188, 0, 0)
