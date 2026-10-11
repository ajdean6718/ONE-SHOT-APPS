"""PNG / SVG export, SVG import and the acceptance scenario 1 (vector-b)."""

import os
import xml.etree.ElementTree as ET

import bpy
import numpy as np
import pytest

from helpers import flush, save_and_reload
from test_vector_b_common import (add_ellipse, add_rect, curve_bbox, pixel, require, select, spline_area,
                                  srgb8)

PNG_KERNELS = ("geometry.flatten_path", "render2d.render", "raster.to_uint8_srgb", "png.encode_png",
               "png.decode_png")
SVG_KERNELS = ("svg.export_svg", "svg.import_svg")


def setup_artboard(w=200.0, h=100.0, x=0.0, y=0.0):
    from seam_studio.vector import canvas
    scene = bpy.context.scene
    canvas.ensure_canvas(scene)
    ab = canvas.artboards(scene)[0]
    ab.location = (x, y, -1.0)
    ab.seam.artboard.width, ab.seam.artboard.height = w, h
    return ab


def export_png(tmp_path, name="out.png", **kw):
    from seam_studio.seam_core import png
    path = str(tmp_path / name)
    assert bpy.ops.seam.png_export(filepath=path, **kw) == {"FINISHED"}
    return png.read_png(path)


# ------------------------------------------------------------------ PNG

def test_png_pixels_background_and_zorder(seam, tmp_path):
    require(*PNG_KERNELS)
    setup_artboard()
    scene = bpy.context.scene
    add_rect(scene, "Red", 20, -60, 80, -20, color=(1.0, 0.0, 0.0, 1.0))
    add_rect(scene, "Half", 60, -80, 120, -40, color=(0.5, 0.0, 0.0, 1.0))      # above Red, overlapping
    add_ellipse(scene, "Blue", 150, -50, 30, 30, color=(0.0, 0.0, 1.0, 1.0))
    img = export_png(tmp_path)
    assert img.shape == (100, 200, 4)
    assert pixel(img, 0, 0, 1, 30, -30) == (255, 0, 0, 255)                     # Red only
    assert pixel(img, 0, 0, 1, 100, -70) == (*srgb8((0.5, 0, 0)), 255)          # Half only
    assert pixel(img, 0, 0, 1, 70, -50) == (*srgb8((0.5, 0, 0)), 255)          # overlap: Half on top
    assert pixel(img, 0, 0, 1, 150, -50) == (0, 0, 255, 255)                    # ellipse center
    assert pixel(img, 0, 0, 1, 150, -50 + 40)[:3] == (255, 255, 255)           # outside ellipse: white background
    assert pixel(img, 0, 0, 1, 5, -95) == (255, 255, 255, 255)
    # transparent export
    t = export_png(tmp_path, "t.png", transparent=True)
    assert pixel(t, 0, 0, 1, 5, -95)[3] == 0
    assert pixel(t, 0, 0, 1, 30, -30) == (255, 0, 0, 255)
    # scale
    big = export_png(tmp_path, "big.png", scale=2.0)
    assert big.shape == (200, 400, 4)
    assert pixel(big, 0, 0, 2, 30, -30) == (255, 0, 0, 255)
    assert pixel(big, 0, 0, 2, 150, -50) == (0, 0, 255, 255)


def test_png_ignores_3d_objects_and_hidden_artwork(seam, tmp_path):
    require(*PNG_KERNELS)
    setup_artboard()
    scene = bpy.context.scene
    add_rect(scene, "Red", 20, -60, 80, -20, color=(1.0, 0.0, 0.0, 1.0))
    hidden = add_rect(scene, "Hidden", 100, -60, 160, -20, color=(0.0, 1.0, 0.0, 1.0))
    hidden.hide_set(True)
    ref = export_png(tmp_path, "a.png")
    assert pixel(ref, 0, 0, 1, 130, -40) == (255, 255, 255, 255)
    bpy.ops.mesh.primitive_cube_add(size=50.0, location=(130.0, -20000.0 - 40.0, 0.0))      # right under the canvas
    bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
    assert any(o.type == "MESH" for o in scene.objects)
    again = export_png(tmp_path, "b.png")
    assert np.array_equal(ref, again)


def test_png_gradient_stroke_opacity(seam, tmp_path):
    require(*PNG_KERNELS, "stroke.stroke_outline")
    setup_artboard()
    scene = bpy.context.scene
    g = add_rect(scene, "Grad", 0, -40, 200, 0, color=(1, 1, 1, 1))
    st = g.seam.style
    st.fill_type = "LINEAR"
    st.gradient_angle = 0.0
    for pos, col in ((0.0, (0, 0, 0, 1)), (1.0, (1, 1, 1, 1))):
        s = st.gradient_stops.add()
        s.position, s.color = pos, col
    r = add_rect(scene, "Stroked", 20, -90, 80, -50, color=(1.0, 0.0, 0.0, 1.0), stroke=((0.0, 0.0, 0.0, 1.0), 10.0))
    o = add_rect(scene, "Faded", 120, -90, 180, -50, color=(1.0, 0.0, 0.0, 1.0))
    o.seam.style.opacity = 0.5
    img = export_png(tmp_path)
    left, mid, right = (pixel(img, 0, 0, 1, x, -20)[0] for x in (4, 100, 196))
    assert left < 60 and right > 225
    assert mid == pytest.approx(srgb8((0.5, 0.5, 0.5))[0], abs=4)               # linear-space interpolation
    assert pixel(img, 0, 0, 1, 50, -70) == (255, 0, 0, 255)                     # fill interior
    assert pixel(img, 0, 0, 1, 22, -70)[:3] == (0, 0, 0)                        # inside the 10 px stroke band (5 px in)
    assert pixel(img, 0, 0, 1, 150, -70) == (*srgb8((1.0, 0.5, 0.5)), 255)     # red at 50% over white


def test_png_text_and_live_outline_drawn_once(seam, tmp_path):
    require(*PNG_KERNELS)
    setup_artboard(400.0, 300.0)
    bpy.ops.seam.add_text(body="I", x=100.0, y=-250.0, size=250.0)
    t = bpy.context.active_object
    t.seam.style.opacity = 0.5
    ref = export_png(tmp_path, "text.png")
    ys, xs = np.nonzero(ref[..., 0] < 250)
    assert len(xs) > 200                                                        # glyph pixels exist
    cx, cy = int(xs.mean()), int(ys.mean())
    inside = tuple(ref[cy, cx])
    assert inside[:3] == pytest.approx(srgb8((0.5, 0.5, 0.5)), abs=3)           # 50 % black over white
    select(t)
    bpy.ops.seam.text_to_outline()
    again = export_png(tmp_path, "outline.png")
    assert tuple(again[cy, cx])[:3] == pytest.approx(inside[:3], abs=3)         # not drawn twice
    assert np.abs(again.astype(int) - ref.astype(int)).max() <= 3


def test_png_other_artboard_and_errors(seam, tmp_path):
    require(*PNG_KERNELS)
    from seam_studio.vector import canvas
    setup_artboard(200.0, 100.0)
    scene = bpy.context.scene
    second = canvas.new_artboard(scene, "Second", 400.0, -50.0, 100.0, 100.0)
    add_rect(scene, "InSecond", 410, -90, 450, -60, color=(0.0, 1.0, 0.0, 1.0))
    add_rect(scene, "InFirst", 20, -60, 80, -20, color=(1.0, 0.0, 0.0, 1.0))
    img = export_png(tmp_path, "second.png", artboard="Second")
    assert img.shape == (100, 100, 4)
    assert pixel(img, 400, -50, 1, 430, -75) == (0, 255, 0, 255)
    assert (img[..., 0] != 255).sum() > 0 and not ((img[..., 0] == 255) & (img[..., 1] == 0)).any()   # no red
    with pytest.raises(RuntimeError, match="not found"):
        bpy.ops.seam.png_export(filepath=str(tmp_path / "x.png"), artboard="Nope")
    setup_artboard(20000.0, 20000.0)
    with pytest.raises(RuntimeError, match="too large"):
        bpy.ops.seam.png_export(filepath=str(tmp_path / "z.png"))


# ------------------------------------------------------------------ SVG

def test_svg_export_structure_and_text_modes(seam, tmp_path):
    require(*SVG_KERNELS)
    setup_artboard(300.0, 200.0, 500.0, -300.0)
    scene = bpy.context.scene
    add_rect(scene, "R", 520, -380, 600, -320, color=(1.0, 0.0, 0.0, 1.0))
    add_ellipse(scene, "E", 700, -400, 40, 30)
    bpy.ops.seam.add_text(body="Hi", x=520.0, y=-450.0, size=60.0)
    p = str(tmp_path / "a.svg")
    assert bpy.ops.seam.svg_export(filepath=p) == {"FINISHED"}
    root = ET.parse(p).getroot()
    assert root.tag.endswith("svg")
    vb = [float(v) for v in root.get("viewBox").replace(",", " ").split()]
    assert vb == [0.0, 0.0, 300.0, 200.0]
    text = open(p, encoding="utf-8").read()
    assert text.count("<path") >= 3 and "<text" not in text                   # text exported as outlines
    p2 = str(tmp_path / "b.svg")
    bpy.ops.seam.svg_export(filepath=p2, text_mode="TEXT")
    assert "<text" in open(p2, encoding="utf-8").read()


def import_svg_into_fresh_file(path):
    bpy.ops.wm.read_homefile(use_empty=True)
    from seam_studio.engine import core
    core.reset()
    assert bpy.ops.seam.svg_import(filepath=path) == {"FINISHED"}
    from seam_studio.vector import canvas
    return canvas.artwork_objects(bpy.context.scene, roles=("SHAPE",))


def test_svg_roundtrip_geometry_and_style(seam, tmp_path):
    require(*SVG_KERNELS, "geometry.flatten_path")
    setup_artboard(300.0, 200.0, 500.0, -300.0)
    scene = bpy.context.scene
    r = add_rect(scene, "R", 520, -380, 600, -320, color=(1.0, 0.0, 0.0, 1.0))
    e = add_ellipse(scene, "E", 700, -400, 40, 30, color=(0.0, 0.0, 1.0, 1.0))
    e.seam.style.stroke_enabled = True
    e.seam.style.stroke_color = (0.0, 1.0, 0.0, 1.0)
    e.seam.style.stroke_width = 6.0
    from seam_studio.vector.export_png import canvas_paths
    want = {}
    for o in (r, e):
        pts = [p.co for path in canvas_paths(o) for p in path.points]
        xs, ys = [p[0] - 500.0 for p in pts], [p[1] + 300.0 for p in pts]
        want[o.name] = (min(xs), min(ys), max(xs), max(ys), spline_area(o))
    p = str(tmp_path / "rt.svg")
    bpy.ops.seam.svg_export(filepath=p)
    shapes = import_svg_into_fresh_file(p)
    assert len(shapes) == 2
    from seam_studio.vector import canvas
    ab = canvas.artboards(bpy.context.scene)
    assert len(ab) == 1 and (ab[0].seam.artboard.width, ab[0].seam.artboard.height) == (300.0, 200.0)
    by_color = {tuple(round(c, 2) for c in o.seam.style.fill_color[:3]): o for o in shapes}
    rr, ee = by_color[(1.0, 0.0, 0.0)], by_color[(0.0, 0.0, 1.0)]
    for o, key in ((rr, "R"), (ee, "E")):
        pts = [p.co for path in canvas_paths(o) for p in path.points]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        w = want[key]
        assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx(w[:4], abs=0.02)
        assert abs(spline_area(o)) == pytest.approx(abs(w[4]), rel=1e-3)
    assert ee.seam.style.stroke_enabled and ee.seam.style.stroke_width == pytest.approx(6.0, abs=0.01)
    assert tuple(ee.seam.style.stroke_color)[:3] == pytest.approx((0.0, 1.0, 0.0), abs=0.01)
    assert not rr.seam.style.stroke_enabled
    # z-order preserved: R below E
    assert rr.location.z < ee.location.z


HAND_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80" width="120" height="80">
  <rect x="10" y="20" width="30" height="40" fill="#ff0000"/>
  <path d="M 60 10 L 100 10 L 100 50 Z" fill="none" stroke="#0000ff" stroke-width="4"/>
  <circle cx="80" cy="60" r="10" fill="#00ff00" opacity="0.5"/>
  <text x="5" y="75">unsupported</text>
</svg>"""


def test_svg_import_hand_written(seam, tmp_path, capfd):
    require(*SVG_KERNELS, "geometry.flatten_path")
    p = str(tmp_path / "hand.svg")
    open(p, "w", encoding="utf-8").write(HAND_SVG)
    shapes = import_svg_into_fresh_file(p)
    out = capfd.readouterr()
    assert "text" in (out.out + out.err).lower()                               # unsupported element is reported
    assert len(shapes) == 3
    from seam_studio.vector import canvas
    from seam_studio.vector.export_png import canvas_paths
    ab = canvas.artboards(bpy.context.scene)[0]
    assert (ab.seam.artboard.width, ab.seam.artboard.height) == (120.0, 80.0)
    ax, ay = ab.location.x, ab.location.y

    def bounds(o):
        pts = [p.co for path in canvas_paths(o) for p in path.points]
        return (min(p[0] for p in pts) - ax, min(p[1] for p in pts) - ay, max(p[0] for p in pts) - ax,
                max(p[1] for p in pts) - ay)
    red = [o for o in shapes if tuple(round(c, 2) for c in o.seam.style.fill_color[:3]) == (1.0, 0.0, 0.0)][0]
    assert bounds(red) == pytest.approx((10.0, -60.0, 40.0, -20.0), abs=0.01)  # SVG y down -> canvas y up
    tri = [o for o in shapes if o.seam.style.stroke_enabled][0]
    assert not tri.seam.style.fill_enabled and tri.seam.style.stroke_width == pytest.approx(4.0)
    assert tuple(round(c, 2) for c in tri.seam.style.stroke_color[:3]) == (0.0, 0.0, 1.0)
    circle = [o for o in shapes if tuple(round(c, 2) for c in o.seam.style.fill_color[:3]) == (0.0, 1.0, 0.0)][0]
    assert circle.seam.style.opacity == pytest.approx(0.5)
    assert bounds(circle) == pytest.approx((70.0, -70.0, 90.0, -50.0), abs=0.2)


def test_svg_import_second_file_adds_artboard_and_errors(seam, tmp_path):
    require(*SVG_KERNELS, "geometry.flatten_path")
    p = str(tmp_path / "hand.svg")
    open(p, "w", encoding="utf-8").write(HAND_SVG)
    bpy.ops.seam.svg_import(filepath=p)
    bpy.ops.seam.svg_import(filepath=p)
    from seam_studio.vector import canvas
    abs_ = canvas.artboards(bpy.context.scene)
    assert len(abs_) == 2
    assert abs_[1].location.x >= abs_[0].location.x + 120.0                    # placed to the right
    bad = str(tmp_path / "bad.svg")
    open(bad, "w").write("this is not xml")
    with pytest.raises(RuntimeError, match="Invalid SVG"):
        bpy.ops.seam.svg_import(filepath=bad)
    with pytest.raises(RuntimeError, match="Cannot read"):
        bpy.ops.seam.svg_import(filepath=str(tmp_path / "missing.svg"))


def test_menus_registered(seam):
    from seam_studio.vector import export_png, io_svg
    export_funcs = bpy.types.TOPBAR_MT_file_export.draw._draw_funcs
    import_funcs = bpy.types.TOPBAR_MT_file_import.draw._draw_funcs
    assert export_png._menu_export in export_funcs and io_svg._menu_export in export_funcs
    assert io_svg._menu_import in import_funcs
    for ident in ("png_export", "svg_export", "svg_import"):
        assert hasattr(bpy.ops.seam, ident)


# ------------------------------------------------------------------ acceptance scenario 1

def test_acceptance_scenario_1(seam, tmp_path, blend_path):
    """Create shapes and text, edit a bezier node and typography, save/reopen, export SVG and PNG
    with no 3D objects present, check PNG pixels and SVG re-import geometry."""
    require(*PNG_KERNELS, *SVG_KERNELS)
    from seam_studio.vector import canvas
    from seam_studio.vector.export_png import canvas_paths
    from seam_studio.seam_core import png
    scene = bpy.context.scene
    ab = setup_artboard(400.0, 300.0, 0.0, 0.0)
    assert not [o for o in scene.objects if o.type == "MESH" and o.seam.role not in ("ARTBOARD",)]
    rect = add_rect(scene, "Rect", 40, -140, 160, -60, color=(1.0, 0.0, 0.0, 1.0))
    ell = add_ellipse(scene, "Ellipse", 300, -100, 60, 40, color=(0.0, 0.0, 1.0, 1.0))
    bpy.ops.seam.add_text(body="Seam", x=40.0, y=-250.0, size=60.0)
    text = bpy.context.active_object
    # edit a bezier node of the rect: move its top-right corner from (160,-60) to (200,-60)
    bp = rect.data.splines[0].bezier_points[2]
    bp.co.x += 40.0
    # edit typography
    text.data.size = 90.0
    text.data.space_character = 0.5
    text.data.align_x = "LEFT"
    bpy.ops.seam.text_style(style="BOLD", state="ON")
    names = (rect.name, ell.name, text.name)
    save_and_reload(blend_path)
    rect, ell, text = (bpy.data.objects[n] for n in names)
    assert rect.data.splines[0].bezier_points[2].co.x == pytest.approx(100.0)   # local x: canvas 200 - center 100
    assert text.data.size == pytest.approx(90.0) and text.data.space_character == pytest.approx(0.5)
    assert all(f.use_bold for f in text.data.body_format)
    assert not [o for o in bpy.context.scene.objects if o.seam.role == "RESULT"]
    # PNG
    from seam_studio.engine import core
    core.request_revalidate()
    core.flush()
    img = export_png(tmp_path, "scene1.png")
    assert img.shape == (300, 400, 4)
    assert pixel(img, 0, 0, 1, 100, -100) == (255, 0, 0, 255)                   # inside the edited rect
    assert pixel(img, 0, 0, 1, 190, -70) == (255, 0, 0, 255)                    # in the area the node edit added
    assert pixel(img, 0, 0, 1, 300, -100) == (0, 0, 255, 255)
    assert pixel(img, 0, 0, 1, 380, -20) == (255, 255, 255, 255)
    dark = (img[..., :3].sum(axis=2) < 60) & (img[..., 3] == 255)
    assert dark[200:300, 30:300].sum() > 300                                    # text glyphs are black pixels
    # SVG, re-import
    p = str(tmp_path / "scene1.svg")
    assert bpy.ops.seam.svg_export(filepath=p) == {"FINISHED"}
    shapes = import_svg_into_fresh_file(p)
    assert len(shapes) >= 3                                                     # rect, ellipse, glyph outlines
    red = [o for o in shapes if tuple(round(c, 2) for c in o.seam.style.fill_color[:3]) == (1.0, 0.0, 0.0)][0]
    pts = [q.co for path in canvas_paths(red) for q in path.points]
    assert (min(q[0] for q in pts), max(q[0] for q in pts)) == pytest.approx((40.0, 200.0), abs=0.02)
    ab2 = canvas.artboards(bpy.context.scene)[0]
    assert (ab2.seam.artboard.width, ab2.seam.artboard.height) == (400.0, 300.0)
