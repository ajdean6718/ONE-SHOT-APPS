"""Freehand pencil: exec path, smoothing/fit, closing, tool registration (vector-b)."""

import math

import bpy
import pytest

from test_vector_b_common import require


def sine_points(n=60, length=300.0, amp=40.0):
    return [(i * length / (n - 1), amp * math.sin(i * 2 * math.pi / (n - 1))) for i in range(n)]


def circle_points(r=50.0, n=48, cx=0.0, cy=0.0, gap=0.0):
    pts = [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]
    if gap:
        pts[-1] = (pts[-1][0] + gap, pts[-1][1])
    return pts


def draw(points, **kw):
    from seam_studio.vector.freehand import format_points
    return bpy.ops.seam.draw_freehand(points=format_points(points), **kw)


def test_parse_and_format_roundtrip(seam):
    from seam_studio.vector.freehand import format_points, parse_points
    pts = [(0.5, -2.0), (10.0, 4.25)]
    assert parse_points(format_points(pts)) == pytest.approx(pts)
    assert parse_points("1,2;3,4  5,6") == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    with pytest.raises(ValueError):
        parse_points("1,2,3")
    with pytest.raises(ValueError):
        parse_points("a,b")


def test_exec_path_creates_open_stroke_shape(seam):
    require("fitting.smooth_polyline", "fitting.fit_bezier", "geometry.flatten_path")
    from seam_studio.vector import canvas
    pts = [(x + 100.0, y - 200.0) for x, y in sine_points()]
    assert draw(pts, smoothing=0.0, error=1.5) == {"FINISHED"}
    obj = bpy.context.active_object
    assert obj.seam.role == "SHAPE" and obj.type == "CURVE"
    assert obj.parent is canvas.canvas_root(bpy.context.scene)
    assert len(obj.data.splines) == 1 and not obj.data.splines[0].use_cyclic_u
    n = len(obj.data.splines[0].bezier_points)
    assert 2 <= n < len(pts) // 2                        # fitted, not copied
    st = obj.seam.style
    assert st.stroke_enabled and not st.fill_enabled
    assert obj.data.fill_mode == "NONE"
    # location is the bbox center; first/last points land on the stroke ends
    from seam_studio.vector.export_png import canvas_paths
    path = canvas_paths(obj)[0]
    assert path.points[0].co == pytest.approx(pts[0], abs=1e-3)
    assert path.points[-1].co == pytest.approx(pts[-1], abs=1e-3)
    assert _max_deviation(obj, pts) <= 3.0               # fit error 1.5 + flattening slack


def _max_deviation(obj, pts):
    """Max distance from the sampled input points to the flattened fitted curve."""
    from seam_studio.seam_core import geometry
    from seam_studio.vector.export_png import canvas_paths
    poly = geometry.flatten_path(canvas_paths(obj)[0], 32)

    def dist(p):
        best = 1e18
        for a, b in zip(poly, poly[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L2 = dx * dx + dy * dy or 1e-12
            t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
            best = min(best, math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy))
        return best
    return max(dist(p) for p in pts)


def test_closing_when_end_near_start(seam):
    require("fitting.smooth_polyline", "fitting.fit_bezier", "geometry.flatten_path")
    assert draw(circle_points(gap=4.0), close_distance=12.0) == {"FINISHED"}
    obj = bpy.context.active_object
    assert obj.data.splines[0].use_cyclic_u
    n_closed = len(obj.data.splines[0].bezier_points)
    assert n_closed >= 3
    # the same stroke ending far from the start stays open
    pts = circle_points()[:30]
    assert draw(pts, close_distance=12.0) == {"FINISHED"}
    assert not bpy.context.active_object.data.splines[0].use_cyclic_u


def test_smoothing_reduces_jitter(seam):
    """The fitted curve follows the underlying sine more closely with more smoothing (kernel independent)."""
    require("fitting.smooth_polyline", "fitting.fit_bezier", "geometry.flatten_path")
    import random
    rnd = random.Random(3)
    base = sine_points(n=80)
    noisy = [(x, y + rnd.uniform(-3.0, 3.0)) for x, y in base]
    devs = {}
    for amount in (0.0, 0.5, 1.0):
        assert draw(noisy, smoothing=amount, error=0.3) == {"FINISHED"}
        devs[amount] = _mean_deviation(bpy.context.active_object, base[2:-2])
    assert devs[0.5] < devs[0.0]
    assert devs[1.0] < devs[0.0]
    assert devs[1.0] < 0.7 * devs[0.0]
    from seam_studio.vector import freehand
    assert [freehand.smoothing_iterations(v) for v in (0.0, 0.5, 1.0, 5.0)] == [0, 4, 8, 8]


def _mean_deviation(obj, pts):
    from seam_studio.seam_core import geometry
    from seam_studio.vector.export_png import canvas_paths
    poly = geometry.flatten_path(canvas_paths(obj)[0], 32)

    def dist(p):
        best = 1e18
        for a, b in zip(poly, poly[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L2 = dx * dx + dy * dy or 1e-12
            t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
            best = min(best, math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy))
        return best
    return sum(dist(p) for p in pts) / len(pts)


def test_errors(seam):
    require("fitting.smooth_polyline", "fitting.fit_bezier")
    with pytest.raises(RuntimeError, match="at least two distinct points"):
        draw([(1.0, 1.0), (1.0, 1.0)])
    with pytest.raises(RuntimeError, match="Invalid points"):
        bpy.ops.seam.draw_freehand(points="1,2 nonsense")
    with pytest.raises(RuntimeError, match="at least two distinct points"):
        bpy.ops.seam.draw_freehand(points="")              # background mode has no events: exec path, nothing to draw


def test_tool_and_description(seam):
    from seam_studio.vector import freehand
    tool = freehand.SEAM_TOOL_pencil
    assert tool.bl_idname == "seam.tool_pencil"
    assert "pressure" in tool.bl_description.lower() and "not supported" in tool.bl_description.lower()
    assert tool.bl_keymap[0][0] == "seam.draw_freehand"
    assert hasattr(bpy.ops.seam, "draw_freehand")
    assert "pressure" in bpy.types.SEAM_OT_draw_freehand.__doc__.lower()


def test_pencil_keymap_item_is_registered(seam):
    """The tool's press trigger is in the add-on / user key configs (an actual press cannot be simulated headless)."""
    wm = bpy.context.window_manager
    found = []
    for kc in (wm.keyconfigs.addon, wm.keyconfigs.user):
        if kc is None:
            continue
        for km in kc.keymaps:
            if "Pencil" not in km.name:
                continue
            for kmi in km.keymap_items:
                if kmi.idname == "seam.draw_freehand" and kmi.type == "LEFTMOUSE" and kmi.value == "PRESS":
                    found.append(km.name)
    if not found and bpy.app.background:
        pytest.skip("tool keymaps are not instantiated in background mode on this runtime")
    assert found, "seam.tool_pencil has no seam.draw_freehand LEFTMOUSE PRESS keymap item"
