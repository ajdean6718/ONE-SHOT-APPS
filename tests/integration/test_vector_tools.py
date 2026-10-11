"""Drawing tools: draw_shape exec path, gesture session, tool definitions, background-mode registration."""

import math

import bpy
import pytest

from helpers import flush


def _need_core():
    from seam_studio.seam_core import shapes
    try:
        shapes.rect(1, 1)
    except NotImplementedError:
        pytest.skip("core pending: seam_core.shapes")


@pytest.fixture
def doc(seam):
    _need_core()
    bpy.ops.seam.canvas_new(width=1000, height=800)
    return bpy.context.scene


def _bounds(ob):
    from seam_studio.vector import canvas
    return canvas.canvas_bounds(ob)


def _draw(kind, x0, y0, x1, y1, **kw):
    assert bpy.ops.seam.draw_shape(kind=kind, x0=x0, y0=y0, x1=x1, y1=y1, **kw) == {"FINISHED"}
    return bpy.context.active_object


def test_drag_geometry_math():
    from seam_studio.vector.tools import drag_geometry
    assert drag_geometry("RECT", (10, 10), (110, 60)) == (60, 35, 100, 50, 0.0)
    assert drag_geometry("RECT", (110, 60), (10, 10)) == (60, 35, 100, 50, 0.0)  # any drag direction
    assert drag_geometry("RECT", (0, 0), (100, 40), square=True) == (50, 50, 100, 100, 0.0)
    assert drag_geometry("RECT", (0, 0), (-100, 40), square=True) == (-50, 50, 100, 100, 0.0)
    assert drag_geometry("RECT", (50, 50), (80, 70), from_center=True) == (50, 50, 60, 40, 0.0)
    assert drag_geometry("RECT", (0, 0), (30, 10), square=True, from_center=True) == (0, 0, 60, 60, 0.0)
    cx, cy, ln, h, rot = drag_geometry("LINE", (0, 0), (30, 40))
    assert (cx, cy, ln, h) == pytest.approx((15, 20, 50, 0)) and rot == pytest.approx(math.atan2(40, 30))
    cx, cy, ln, h, rot = drag_geometry("LINE", (0, 0), (100, 30), square=True)  # snaps to 0 degrees
    assert rot == pytest.approx(0.0) and ln == pytest.approx(math.hypot(100, 30))
    cx, cy, ln, h, rot = drag_geometry("LINE", (0, 0), (30, 40), from_center=True)
    assert (cx, cy, ln) == pytest.approx((0, 0, 100))


@pytest.mark.parametrize("kind", ["RECT", "ROUNDED_RECT", "ELLIPSE", "POLYGON", "STAR"])
def test_draw_shape_exec_creates_shape_with_dragged_bounds(doc, kind):
    ob = _draw(kind, 100, -100, 300, -250)
    assert ob.seam.role == "SHAPE" and ob.seam.shape.kind == kind
    assert (ob.seam.shape.width, ob.seam.shape.height) == pytest.approx((200, 150))
    assert tuple(ob.location)[:2] == pytest.approx((200, -175))
    b = _bounds(ob)
    if kind in ("RECT", "ROUNDED_RECT", "ELLIPSE"):
        assert b == pytest.approx((100, -250, 300, -100), abs=1e-2)
    else:
        assert b[0] >= 99.9 and b[2] <= 300.1 and b[1] >= -250.1 and b[3] <= -99.9
    assert doc.objects.get(ob.name) is ob


def test_draw_shape_modifiers_and_options(doc):
    sq = _draw("RECT", 0, 0, 100, 40, square=True)
    assert _bounds(sq) == pytest.approx((0, 0, 100, 100), abs=1e-2)
    ctr = _draw("ELLIPSE", 500, -400, 550, -430, from_center=True)
    assert _bounds(ctr) == pytest.approx((450, -430, 550, -370), abs=1e-2)
    rr = _draw("ROUNDED_RECT", 0, 0, 100, 100, radius=30)
    assert rr.seam.shape.radius == 30
    star = _draw("STAR", 0, 0, 100, 100, sides=7, inner_ratio=0.3)
    assert len(star.data.splines[0].bezier_points) == 14 and star.seam.shape.inner_ratio == pytest.approx(0.3)
    # drag direction does not matter
    rev = _draw("RECT", 300, -100, 100, -250)
    assert _bounds(rev) == pytest.approx((100, -250, 300, -100), abs=1e-2)


def test_draw_line_exec_rotates_the_shape(doc):
    ob = _draw("LINE", 0, 0, 30, 40)
    assert ob.seam.shape.kind == "LINE" and ob.seam.shape.width == pytest.approx(50)
    assert ob.rotation_euler.z == pytest.approx(math.atan2(40, 30))
    assert _bounds(ob) == pytest.approx((0, 0, 30, 40), abs=1e-2)
    assert ob.seam.style.stroke_enabled and not ob.seam.style.fill_enabled


def test_draw_artboard_exec(doc):
    from seam_studio.vector import canvas
    ab = _draw("ARTBOARD", 2000, 100, 2300, -200)
    assert ab.seam.role == "ARTBOARD"
    assert (ab.seam.artboard.width, ab.seam.artboard.height) == pytest.approx((300, 300))
    assert tuple(ab.location)[:2] == pytest.approx((2000, 100))
    assert canvas.active_artboard(doc) == ab
    flush()
    xs = [v.co.x for v in ab.data.vertices]
    ys = [v.co.y for v in ab.data.vertices]
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((0, 300, -300, 0))
    assert len(canvas.artboards(doc)) == 2


def test_draw_zero_size_is_cancelled(doc):
    n = len([o for o in doc.objects])
    assert bpy.ops.seam.draw_shape(kind="RECT", x0=5, y0=5, x1=5, y1=5) == {"CANCELLED"}
    assert len(doc.objects) == n


def test_draw_gesture_session_updates_params_live_and_cancel_cleans_up(doc):
    from seam_studio.vector.tools import DrawSession
    sess = DrawSession("RECT")
    ob = sess.update(doc, (100, -100), (110, -110))
    flush()
    assert _bounds(ob) == pytest.approx((100, -110, 110, -100), abs=1e-2)
    first_id = ob.seam.id
    ob2 = sess.update(doc, (100, -100), (300, -200))
    assert ob2 is ob and ob.seam.id == first_id
    res = flush()
    assert any(r.key == "shape:" + ob.seam.id and r.status == "ok" for r in res)  # engine rebuilt from params
    assert _bounds(ob) == pytest.approx((100, -200, 300, -100), abs=1e-2)
    assert ob.seam.shape.kind == "RECT"  # live preview never expands the shape
    sess.update(doc, (100, -100), (150, -300), square=True)
    flush()
    b = _bounds(ob)  # Shift: square; the drag is 50 wide / 200 tall -> 200 x 200 about the start corner
    assert (b[2] - b[0], b[3] - b[1]) == pytest.approx((200, 200), abs=1e-2)
    assert (b[0] + b[2]) / 2 == pytest.approx(200, abs=1e-2)
    name, data = ob.name, ob.data.name
    sess.cancel()
    assert name not in bpy.data.objects and data not in bpy.data.curves
    assert sess.obj is None


def test_draw_gesture_session_artboard_and_line(doc):
    from seam_studio.vector import canvas
    from seam_studio.vector.tools import DrawSession
    s = DrawSession("ARTBOARD")
    ab = s.update(doc, (0, 0), (50, -50))
    s.update(doc, (0, 0), (400, -300))
    flush()
    assert (ab.seam.artboard.width, ab.seam.artboard.height) == pytest.approx((400, 300))
    assert max(v.co.x for v in ab.data.vertices) == pytest.approx(400)
    name = ab.name
    s.cancel()
    assert name not in bpy.data.objects and len(canvas.artboards(doc)) == 1
    ln = DrawSession("LINE")
    ob = ln.update(doc, (0, 0), (10, 0))
    ln.update(doc, (0, 0), (0, 80))
    flush()
    assert ob.seam.shape.width == pytest.approx(80) and ob.rotation_euler.z == pytest.approx(math.pi / 2)
    assert _bounds(ob) == pytest.approx((0, 0, 0, 80), abs=1e-2)


def test_screen_to_canvas_mapping_formula():
    # pure check of the plane intersection used by the tools (top ortho looking down -Z)
    from mathutils import Vector, geometry
    hit = geometry.intersect_line_plane(Vector((12, -19990, 50)), Vector((12, -19990, -950)), Vector((0, -20000, 0)), Vector((0, 0, 1)))
    assert tuple(hit) == pytest.approx((12, -19990, 0))


def test_tool_definitions_and_background_registration(seam):
    from seam_studio.vector import tools
    assert bpy.app.background
    assert tools._tools == []  # registration is skipped without a window manager
    ids = []
    for kind, label, icon, extra in tools._TOOL_SPECS:
        tool = tools._make_tool(kind, label, icon, extra)
        assert issubclass(tool, bpy.types.WorkSpaceTool)
        assert tool.bl_space_type == "VIEW_3D" and tool.bl_context_mode == "OBJECT"
        km = tool.bl_keymap[0]
        assert km[0] == "seam.draw_shape" and km[1] == {"type": "LEFTMOUSE", "value": "CLICK_DRAG"}
        assert km[2]["properties"] == [("kind", kind)]
        ids.append(tool.bl_idname)
    assert ids == ["seam.tool_rect", "seam.tool_rrect", "seam.tool_ellipse", "seam.tool_polygon", "seam.tool_star",
                   "seam.tool_line", "seam.tool_artboard"]
    tools.unregister()  # harmless
    tools.register()
