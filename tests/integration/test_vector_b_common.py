"""Shared helpers for the vector-b integration tests (and a kernel availability report)."""

import math

import bpy
import numpy as np
import pytest

SQUARE_LIKE = "polygon"


def _probe_table():
    from seam_studio.seam_core import boolean2d, fitting, geometry, png, raster, render2d, stroke, svg
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint

    def pt(x, y):
        return BezierPoint((x, y), (x, y), (x, y), "VECTOR", "VECTOR")

    seg = BezierPath([pt(0, 0), pt(10, 0), pt(10, 10)], True)
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    sq2 = [(5, 5), (15, 5), (15, 15), (5, 15)]
    return {
        "geometry.flatten_path": lambda: geometry.flatten_path(seg, 4),
        "geometry.signed_area": lambda: geometry.signed_area(sq),
        "geometry.triangulate": lambda: geometry.triangulate(sq, []),
        "boolean2d.boolean": lambda: boolean2d.boolean([sq], [sq2], "UNION"),
        "boolean2d.union_all": lambda: boolean2d.union_all([[sq], [sq2]]),
        "fitting.smooth_polyline": lambda: fitting.smooth_polyline([(0, 0), (1, 1), (2, 0)], 0.5, 1, False),
        "fitting.fit_bezier": lambda: fitting.fit_bezier([(0, 0), (1, 1), (2, 0), (3, 1)], 1.0, False),
        "stroke.stroke_outline": lambda: stroke.stroke_outline([(0, 0), (10, 0)], False, 2.0),
        "render2d.render": lambda: render2d.render([], 2, 2, (0.0, 2.0, 2.0, 0.0)),
        "raster.to_uint8_srgb": lambda: raster.to_uint8_srgb(np.zeros((1, 1, 4), np.float32)),
        "png.encode_png": lambda: png.encode_png(np.zeros((1, 1, 4), np.uint8)),
        "png.decode_png": lambda: png.decode_png(png.encode_png(np.zeros((1, 1, 4), np.uint8))),
        "svg.export_svg": lambda: svg.export_svg([], (0.0, 0.0, 10.0, 10.0)),
        "svg.import_svg": lambda: svg.import_svg('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>'),
    }


def require(*names):
    """Skip the test when a seam_core kernel it needs is still a stub (the integrator re-runs after merging)."""
    table = _probe_table()
    for name in names:
        try:
            table[name]()
        except NotImplementedError:
            pytest.skip(f"kernel pending: {name}")


# ------------------------------------------------------------------ geometry builders

def poly_path(points):
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint
    return BezierPath([BezierPoint(tuple(p), tuple(p), tuple(p), "VECTOR", "VECTOR") for p in points], True)


def rect_points(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def add_rect(scene, name, x0, y0, x1, y1, color=(1.0, 0.0, 0.0, 1.0), stroke=None):
    """Axis-aligned rectangle SHAPE with canvas corners (x0, y0)-(x1, y1); origin at its center."""
    from seam_studio.vector import canvas
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    path = poly_path(rect_points(x0 - cx, y0 - cy, x1 - cx, y1 - cy))
    ob = canvas.new_shape_object(scene, name, [path], location=(cx, cy))
    ob.seam.style.fill_color = color
    if stroke is not None:
        ob.seam.style.stroke_enabled = True
        ob.seam.style.stroke_color = stroke[0]
        ob.seam.style.stroke_width = stroke[1]
    return ob


def add_ellipse(scene, name, cx, cy, rx, ry, color=(0.0, 0.0, 1.0, 1.0)):
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint
    from seam_studio.vector import canvas
    k = 0.5522847498307936
    pts = [BezierPoint((rx, 0), (rx, -k * ry), (rx, k * ry), "ALIGNED", "ALIGNED"),
           BezierPoint((0, ry), (k * rx, ry), (-k * rx, ry), "ALIGNED", "ALIGNED"),
           BezierPoint((-rx, 0), (-rx, k * ry), (-rx, -k * ry), "ALIGNED", "ALIGNED"),
           BezierPoint((0, -ry), (-k * rx, -ry), (k * rx, -ry), "ALIGNED", "ALIGNED")]
    ob = canvas.new_shape_object(scene, name, [BezierPath(pts, True)], location=(cx, cy))
    ob.seam.style.fill_color = color
    return ob


def select(*objs):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[-1] if objs else None


# ------------------------------------------------------------------ measurements

def spline_area(obj):
    """Net polygon area (control points, canvas space) of a curve object: outers positive, holes negative."""
    from seam_studio.vector.export_png import live_canvas_affine
    a, b, c, d, e, f = live_canvas_affine(obj)
    total = 0.0
    for sp in obj.data.splines:
        pts = [(a * p.co[0] + c * p.co[1] + e, b * p.co[0] + d * p.co[1] + f) for p in sp.bezier_points]
        s = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            s += x0 * y1 - x1 * y0
        total += 0.5 * s
    return total


def mesh_area(obj):
    me = obj.data
    verts = np.array([v.co[:2] for v in me.vertices], dtype=np.float64).reshape(-1, 2)
    area = 0.0
    for poly in me.polygons:
        idx = list(poly.vertices)
        for i in range(1, len(idx) - 1):
            p0, p1, p2 = verts[idx[0]], verts[idx[i]], verts[idx[i + 1]]
            area += abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])) * 0.5
    return area


def curve_bbox(obj):
    xs, ys = [], []
    for sp in obj.data.splines:
        for p in sp.bezier_points:
            xs.append(p.co[0])
            ys.append(p.co[1])
    return (min(xs), min(ys), max(xs), max(ys))


def pixel(img, ab_x, ab_y, scale, x, y):
    """RGBA uint8 pixel of an artboard render at canvas point (x, y)."""
    col = int(math.floor((x - ab_x) * scale))
    row = int(math.floor((ab_y - y) * scale))
    return tuple(int(v) for v in img[row, col])


def srgb8(linear):
    v = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return tuple(int(round(float(x) * 255)) for x in s)


def test_kernel_report(seam):
    """Informational: which kernels are real in this tree (never fails)."""
    import sys
    table = _probe_table()
    missing = []
    for name, fn in table.items():
        try:
            fn()
        except NotImplementedError:
            missing.append(name)
    sys.stderr.write("vector-b kernel status: pending=%s\n" % missing)
