"""PNG / SVG export, SVG import and the acceptance scenario 1 (vector-b)."""

import os
import xml.etree.ElementTree as ET

import bpy
import numpy as np
import pytest

from helpers import save_and_reload
from test_vector_b_common import add_ellipse, add_rect, pixel, poly_path, require, select, spline_area, srgb8

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


def test_png_radial_gradient(seam, tmp_path):
    require(*PNG_KERNELS)
    setup_artboard()
    scene = bpy.context.scene
    g = add_rect(scene, "Radial", 0, -100, 200, 0, color=(1, 1, 1, 1))
    st = g.seam.style
    st.fill_type = "RADIAL"
    st.gradient_center = (0.5, 0.5)
    st.gradient_radius = 0.5                  # * max(w, h) = 100 px
    for pos, col in ((0.0, (1, 1, 1, 1)), (1.0, (0, 0, 0, 1))):
        s = st.gradient_stops.add()
        s.position, s.color = pos, col
    img = export_png(tmp_path)
    center = pixel(img, 0, 0, 1, 100, -50)[0]
    near = pixel(img, 0, 0, 1, 100 + 49, -50)[0]
    edge = pixel(img, 0, 0, 1, 100 + 95, -50)[0]
    assert center > 240 and 70 < near < 200 and edge < 90 and center > near > edge


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


# ------------------------------------------------------------------ raster items

def add_raster_plane(scene, x, y, size):
    from seam_studio.vector import canvas
    d = scene.seam.rasters.add()
    d.id = "r1"
    h = size / 2.0
    me = bpy.data.meshes.new("RasterPlane")
    me.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("RasterPlane", me)
    ob.location = (x, y, 0.0)
    canvas.add_artwork_object(scene, ob, role="RASTER")
    ob.seam.raster_id = "r1"
    return ob


def test_raster_objects_become_image_items(seam, tmp_path, monkeypatch):
    require(*PNG_KERNELS)
    from seam_studio.paint import rasterdoc
    from seam_studio.seam_core import render2d
    from seam_studio.vector import export_png
    ab = setup_artboard(200.0, 100.0)
    scene = bpy.context.scene
    add_raster_plane(scene, 100.0, -50.0, 100.0)          # covers x 50..150, y -100..0
    # without a compositor the raster is skipped with a warning (explicitly removed, whatever paint provides)
    monkeypatch.delattr(rasterdoc, "composite_array", raising=False)
    warnings = []
    items = export_png.collect_render_items(scene, ab, warnings)
    assert items == [] and warnings and "Raster" in warnings[0]
    # 20 x 20 image of four constant quadrants: sampled at quadrant centres the resampling kernel does not matter
    px = np.zeros((20, 20, 4), np.float32)
    px[:10, :10] = (0, 1, 0, 1)
    px[:10, 10:] = (1, 0, 0, 1)
    px[10:, :10] = (0, 0, 1, 1)
    px[10:, 10:] = (1, 1, 1, 1)
    monkeypatch.setattr(rasterdoc, "composite_array", lambda doc: px, raising=False)
    items = export_png.collect_render_items(scene, ab, [])
    assert len(items) == 1 and isinstance(items[0], render2d.ImageItem)
    it = items[0]
    assert (it.x, it.y_top, it.width, it.height) == pytest.approx((50.0, 0.0, 100.0, 100.0))
    img = export_png.render_artboard(scene, ab, 1.0, True)

    def near(pix, rgba):
        return all(abs(int(a) - int(b)) <= 2 for a, b in zip(pix, rgba))
    assert near(img[25, 75], (0, 255, 0, 255))              # top-left quadrant
    assert near(img[25, 125], (255, 0, 0, 255))
    assert near(img[75, 75], (0, 0, 255, 255))
    assert near(img[75, 125], (255, 255, 255, 255))
    assert img[50, 20, 3] == 0                              # outside the raster plane


FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "resources", "fixtures", "vector")


def test_svg_import_fixture_gradients_transforms_style_attribute(seam):
    require(*SVG_KERNELS, "geometry.flatten_path")
    shapes = import_svg_into_fresh_file(os.path.join(FIXTURES, "shapes.svg"))
    by_name = {o.name: o for o in shapes}
    assert {"grad-linear", "grad-radial", "moved"} <= set(by_name)
    from seam_studio.vector import canvas
    from seam_studio.vector.export_png import canvas_paths
    ab = canvas.artboards(bpy.context.scene)[0]
    ax, ay = ab.location.x, ab.location.y

    def bounds(o):
        pts = [p.co for path in canvas_paths(o) for p in path.points]
        return (min(p[0] for p in pts) - ax, min(p[1] for p in pts) - ay, max(p[0] for p in pts) - ax,
                max(p[1] for p in pts) - ay)
    lin = by_name["grad-linear"].seam.style
    assert lin.fill_enabled and lin.fill_type == "LINEAR" and len(lin.gradient_stops) == 2
    assert lin.gradient_angle == pytest.approx(0.0, abs=1e-6)
    assert [round(s.position, 3) for s in lin.gradient_stops] == [0.0, 1.0]
    assert tuple(round(c, 2) for c in lin.gradient_stops[0].color[:3]) == (0.0, 0.0, 0.0)
    assert tuple(round(c, 2) for c in lin.gradient_stops[1].color[:3]) == (1.0, 1.0, 1.0)
    rad = by_name["grad-radial"].seam.style
    assert rad.fill_type == "RADIAL" and rad.stroke_enabled and rad.stroke_width == pytest.approx(3.0)
    assert tuple(rad.gradient_center) == pytest.approx((0.5, 1.0 / 6.0 + 0.0), abs=1e-4)      # (150-120)/60, (-50+60)/60
    assert rad.gradient_radius == pytest.approx(0.5, abs=1e-4)                                    # 30 / max(60, 60)
    assert bounds(by_name["moved"]) == pytest.approx((10.0, -90.0, 30.0, -60.0), abs=0.01)         # translate(10 60), y flipped


def test_svg_gradient_roundtrip(seam, tmp_path):
    require(*SVG_KERNELS, "geometry.flatten_path")
    import math
    setup_artboard(300.0, 200.0, 0.0, 0.0)
    scene = bpy.context.scene
    lin = add_rect(scene, "Lin", 10, -150, 160, -50, color=(1, 1, 1, 1))
    rad = add_rect(scene, "Rad", 180, -120, 280, -20, color=(1, 1, 1, 1))
    for ob, kind in ((lin, "LINEAR"), (rad, "RADIAL")):
        st = ob.seam.style
        st.fill_type = kind
        for pos, col in ((0.0, (1, 0, 0, 1)), (0.4, (0, 1, 0, 1)), (1.0, (0, 0, 1, 1))):
            s = st.gradient_stops.add()
            s.position, s.color = pos, col
    lin.seam.style.gradient_angle = math.radians(30.0)
    rad.seam.style.gradient_center = (0.25, 0.75)
    rad.seam.style.gradient_radius = 0.4
    p = str(tmp_path / "grad.svg")
    bpy.ops.seam.svg_export(filepath=p)
    assert "Gradient" in open(p, encoding="utf-8").read()
    shapes = import_svg_into_fresh_file(p)
    assert len(shapes) == 2
    types = {o.seam.style.fill_type: o for o in shapes}
    li, ra = types["LINEAR"].seam.style, types["RADIAL"].seam.style
    assert li.gradient_angle == pytest.approx(math.radians(30.0), abs=1e-3)
    assert [round(s.position, 3) for s in li.gradient_stops] == [0.0, 0.4, 1.0]
    assert [tuple(round(c, 2) for c in s.color[:3]) for s in li.gradient_stops] == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                                                                       (0.0, 0.0, 1.0)]
    assert tuple(ra.gradient_center) == pytest.approx((0.25, 0.75), abs=1e-3)
    assert ra.gradient_radius == pytest.approx(0.4, abs=1e-3)


# ------------------------------------------------------------------ review fixes: opacity, clips, text, artboard

def test_group_opacity_png_matches_svg_semantics(seam, tmp_path):
    """Object opacity is group opacity: stroke over fill does not blend twice (like the SVG ``opacity`` attribute)."""
    require(*PNG_KERNELS, *SVG_KERNELS, "boolean2d.boolean")
    setup_artboard(100.0, 100.0)
    r = add_rect(bpy.context.scene, "R", 20, -80, 80, -20, color=(0, 0, 0, 1), stroke=((0, 0, 0, 1), 20.0))
    r.seam.style.opacity = 0.5
    img = export_png(tmp_path)
    outer = pixel(img, 0, 0, 1, 15, -50)        # stroke only (stroke spans x 10..30)
    inner = pixel(img, 0, 0, 1, 25, -50)        # stroke over the fill
    deep = pixel(img, 0, 0, 1, 50, -50)         # fill only
    assert outer[:3] == pytest.approx(inner[:3], abs=1)
    assert outer[:3] == pytest.approx(deep[:3], abs=1)
    assert outer[0] == pytest.approx(188, abs=2)           # 50 % black over white
    p = str(tmp_path / "g.svg")
    bpy.ops.seam.svg_export(filepath=p)
    assert 'opacity="0.5"' in open(p, encoding="utf-8").read()       # one group opacity on the single element


def test_group_opacity_cut_keeps_opaque_objects_unchanged(seam, tmp_path):
    from seam_studio.vector import export_png
    require(*PNG_KERNELS, "boolean2d.boolean")
    r = add_rect(bpy.context.scene, "R", 20, -80, 80, -20, color=(0, 0, 0, 1), stroke=((1, 0, 0, 1), 10.0))
    items = export_png.object_items(r)
    assert len(items) == 2 and len(items[0].rings) == 1                  # untouched at opacity 1
    r.seam.style.opacity = 0.5
    items = export_png.object_items(r)
    assert len(items) == 2 and items[0].fill_rule == "EVENODD"
    from seam_studio.seam_core import geometry
    assert len(items[0].rings) == 1                       # the fill shrinks by half the stroke width: 25..75
    assert abs(geometry.signed_area(items[0].rings[0])) == pytest.approx(50.0 * 50.0, rel=1e-3)
    r.seam.style.stroke_width = 200.0                     # stroke covers the whole fill -> no fill item left
    items = export_png.object_items(r)
    assert len(items) == 1 and items[0].paint.color[0] == pytest.approx(1.0)


def test_svg_clipped_raster_is_cut_by_the_mask(seam, tmp_path, monkeypatch):
    require(*PNG_KERNELS, *SVG_KERNELS)
    import base64
    import re
    from seam_studio.paint import rasterdoc
    from seam_studio.seam_core import png
    setup_artboard(200.0, 100.0)
    scene = bpy.context.scene
    ras = add_raster_plane(scene, 100.0, -50.0, 100.0)          # x 50..150, y -100..0
    mask = add_rect(scene, "M", 50, -60, 100, -10)              # left half, y -60..-10
    ras.seam.clip_source = mask
    monkeypatch.setattr(rasterdoc, "composite_array", lambda d: np.ones((20, 20, 4), np.float32), raising=False)
    p = str(tmp_path / "c.svg")
    assert bpy.ops.seam.svg_export(filepath=p) == {"FINISHED"}
    txt = open(p, encoding="utf-8").read()
    data = re.search(r'href="data:image/png;base64,([^"]+)"', txt).group(1)
    img = png.decode_png(base64.b64decode(data))
    assert img.shape[:2] == (20, 20)
    assert img[5, 5, 3] == 255                  # inside the mask (x 50..100 -> columns 0..9, y -10..-60 -> rows 2..11)
    assert img[10, 4, 3] == 255
    assert img[15, 4, 3] == 0                   # below the mask
    assert img[5, 15, 3] == 0                   # right of the mask
    assert img[0, 0, 3] == 0                    # above the mask


def test_svg_nonzero_mask_clip_rule_matches_png(seam, tmp_path):
    require(*PNG_KERNELS, *SVG_KERNELS, "boolean2d.union_all", "geometry.triangulate")
    from seam_studio.vector import canvas
    setup_artboard(100.0, 100.0)
    scene = bpy.context.scene
    art = add_rect(scene, "Art", 0, -100, 100, 0, color=(1, 0, 0, 1))
    outer = poly_path([(-40, -40), (40, -40), (40, 40), (-40, 40)])
    inner = poly_path([(-20, -20), (20, -20), (20, 20), (-20, 20)])          # same winding: NONZERO keeps it filled
    mask = canvas.new_shape_object(scene, "Mask", [outer, inner], location=(50, -50))
    mask.seam.style.fill_rule = "NONZERO"
    select(art, mask)
    bpy.ops.seam.clip_make()
    img = export_png(tmp_path)
    assert pixel(img, 0, 0, 1, 50, -50)[:3] == (255, 0, 0)                   # PNG: centre is inside the clip
    p = str(tmp_path / "n.svg")
    bpy.ops.seam.svg_export(filepath=p)
    txt = open(p, encoding="utf-8").read()
    import re
    d = re.search(r"<clipPath[^>]*><path d=\"([^\"]+)\"", txt).group(1)
    assert d.count("M") == 1                                                 # resolved to ONE ring: no evenodd hole
    shapes = import_svg_into_fresh_file(p)
    assert len(shapes) >= 1


def test_svg_evenodd_mask_keeps_exact_clip_path(seam, tmp_path):
    require(*SVG_KERNELS, "geometry.flatten_path")
    from seam_studio.vector import canvas
    from seam_studio.vector.io_svg import svg_clip_paths
    scene = bpy.context.scene
    outer = poly_path([(-40, -40), (40, -40), (40, 40), (-40, 40)])
    inner = poly_path([(-20, -20), (20, -20), (20, 20), (-20, 20)])
    mask = canvas.new_shape_object(scene, "Mask", [outer, inner], location=(50, -50))
    mask.seam.style.fill_rule = "EVENODD"
    assert len(svg_clip_paths(mask)) == 2                                    # evenodd: two sub-paths, a real hole
    ell = add_ellipse(scene, "E", 0, 0, 10, 10)
    ell.seam.style.fill_rule = "NONZERO"
    assert len(svg_clip_paths(ell)) == 1 and len(svg_clip_paths(ell)[0].points) == 4        # single spline: exact Bezier


def _text_svg(tmp_path, mode="TEXT"):
    p = str(tmp_path / "t.svg")
    assert bpy.ops.seam.svg_export(filepath=p, text_mode=mode) == {"FINISHED"}
    return open(p, encoding="utf-8").read()


def test_svg_text_mode_places_text_at_the_baseline_origin(seam, tmp_path):
    require(*SVG_KERNELS, "geometry.flatten_path")
    import re
    setup_artboard(400.0, 300.0, 100.0, -50.0)
    bpy.ops.seam.add_text(body="Hello", x=140.0, y=-120.0, size=40.0)
    txt = _text_svg(tmp_path)
    m = re.search(r"<text ([^>]*)>Hello</text>", txt)
    assert m, txt
    attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', m.group(1)))
    assert float(attrs["x"]) == pytest.approx(40.0, abs=0.01)             # 140 - artboard x 100
    assert float(attrs["y"]) == pytest.approx(70.0, abs=0.01)             # artboard top -50 minus baseline -120
    assert float(attrs["font-size"]) == pytest.approx(40.0)
    assert "sans-serif" in attrs["font-family"] and "d" not in attrs
    assert "<path" not in txt
    outlines = _text_svg(tmp_path, "OUTLINES")
    assert "<text" not in outlines and "<path" in outlines


def test_svg_text_mode_falls_back_to_outlines_with_warning(seam, tmp_path):
    require(*SVG_KERNELS, "geometry.flatten_path")
    from seam_studio.vector import io_svg
    scene = bpy.context.scene
    setup_artboard(400.0, 300.0)
    bpy.ops.seam.add_text(body="Two\nLines", x=10, y=-100, size=30)
    multi = bpy.context.active_object
    bpy.ops.seam.add_text(body="Plain", x=10, y=-200, size=30)
    plain = bpy.context.active_object
    ab = io_svg.canvas.artboards(scene)[0]
    warnings = []
    items, _ = io_svg.build_export_items(scene, ab, "TEXT", warnings)
    by_name = {i.name: i for i in items}
    assert by_name[plain.name].text == "Plain"
    assert by_name[multi.name].text is None and len(by_name[multi.name].paths) >= 2          # outlines
    assert len(warnings) == 1 and "multi-line" in warnings[0] and multi.name in warnings[0]
    plain.data.align_x = "CENTER"
    warnings = []
    items, _ = io_svg.build_export_items(scene, ab, "TEXT", warnings)
    assert {i.name: i for i in items}[plain.name].text is None and "alignment" in warnings[-1]
    plain.data.align_x = "LEFT"
    plain.rotation_euler.z = 0.3
    warnings = []
    items, _ = io_svg.build_export_items(scene, ab, "TEXT", warnings)
    assert {i.name: i for i in items}[plain.name].text is None and "rotation" in " ".join(warnings)


def test_svg_import_keeps_a_customised_default_artboard(seam, tmp_path):
    require(*SVG_KERNELS, "geometry.flatten_path")
    from seam_studio.vector import canvas
    src = os.path.join(FIXTURES, "shapes.svg")
    bpy.ops.wm.read_homefile(use_empty=True)
    from seam_studio.engine import core
    core.reset()
    scene = bpy.context.scene
    canvas.ensure_canvas(scene)
    ab0 = canvas.artboards(scene)[0]
    ab0.seam.artboard.width, ab0.seam.artboard.height = 640.0, 480.0         # user configured it
    ab0.name = "My Board"
    assert bpy.ops.seam.svg_import(filepath=src) == {"FINISHED"}
    abs_ = canvas.artboards(scene)
    assert len(abs_) == 2 and abs_[0].name == "My Board"
    assert (abs_[0].seam.artboard.width, abs_[0].seam.artboard.height) == (640.0, 480.0)
    # a pristine factory artboard is still reused
    bpy.ops.wm.read_homefile(use_empty=True)
    core.reset()
    assert bpy.ops.seam.svg_import(filepath=src) == {"FINISHED"}
    assert len(canvas.artboards(bpy.context.scene)) == 1


def test_png_export_of_a_real_paint_raster_document(seam, tmp_path):
    """RASTER objects created by paint export through ``rasterdoc.composite_array`` (skipped without paint)."""
    require(*PNG_KERNELS)
    from seam_studio.paint import rasterdoc
    if not hasattr(rasterdoc, "new_raster_doc"):
        pytest.skip("paint pending: rasterdoc.new_raster_doc")
    scene = bpy.context.scene
    setup_artboard(200.0, 100.0)
    rasterdoc.new_raster_doc(scene, "Pic", 40, 20, "WHITE", x=10.0, y=-10.0)       # plane x 10..50, y -30..-10
    img = export_png(tmp_path, transparent=True)
    assert pixel(img, 0, 0, 1, 20, -20) == (255, 255, 255, 255)           # inside the plane: white background
    assert pixel(img, 0, 0, 1, 100, -20)[3] == 0                          # outside: transparent
    plane = scene.seam.rasters[0].canvas
    plane.scale.x = 2.0                                                    # the plane owns the extent: stretch
    bpy.context.view_layer.update()
    from seam_studio.vector import export_png as ep
    item = [i for i in ep.collect_render_items(scene, canvas_ab(scene)) if hasattr(i, "rgba")][0]
    assert (item.x, item.y_top, item.width, item.height) == pytest.approx((10.0, -10.0, 80.0, 20.0))
    img = export_png(tmp_path, "wide.png", transparent=True)
    assert pixel(img, 0, 0, 1, 80, -20)[:3] == (255, 255, 255)            # x 80 is inside the stretched plane (10..90)
    assert pixel(img, 0, 0, 1, 95, -20)[3] == 0                            # beyond it: transparent


def canvas_ab(scene):
    from seam_studio.vector import canvas
    return canvas.artboards(scene)[0]
