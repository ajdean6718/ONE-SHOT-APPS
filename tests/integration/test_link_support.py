"""Shared helpers for the link integration tests (no tests here)."""

import bpy
import numpy as np

SQUARE = ((-100, -100), (100, -100), (100, 100), (-100, 100))
HOLE = ((-40, -40), (-40, 40), (40, 40), (40, -40))


def make_shape(name="Shape", rings=(SQUARE,), location=(0.0, 0.0), closed=True, color=(0.8, 0.2, 0.1, 1.0)):
    """A SEAM vector shape on the canvas (VECTOR handles => straight segments)."""
    from seam_studio.vector import canvas
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint
    paths = [BezierPath([BezierPoint(p, p, p, "VECTOR", "VECTOR") for p in ring], closed=closed) for ring in rings]
    ob = canvas.new_shape_object(bpy.context.scene, name, paths, location=location)
    ob.seam.style.fill_color = color
    return ob


def activate(obj):
    vl = bpy.context.view_layer
    for o in list(bpy.context.selected_objects):
        o.select_set(False)
    obj.select_set(True)
    vl.objects.active = obj


def make_link(src, recipe="EXTRUDE", **params):
    activate(src)
    res = bpy.ops.seam.make_3d(recipe=recipe, **params)
    assert res == {"FINISHED"}, res
    return bpy.context.scene.seam.links[-1]


def link_by_id(link_id):
    for lk in bpy.context.scene.seam.links:
        if lk.id == link_id:
            return lk
    return None


def mesh_arrays(obj):
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    mats = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", mats)
    return co.reshape(-1, 3).astype(np.float64), mats


def extent(obj):
    co, _ = mesh_arrays(obj)
    return co.max(axis=0) - co.min(axis=0)


def face_area(obj, material=None):
    total = 0.0
    for p in obj.data.polygons:
        if material is None or p.material_index == material:
            total += p.area
    return total


def move_point(obj, spline=0, index=0, dx=0.0, dy=0.0):
    bp = obj.data.splines[spline].bezier_points[index]
    bp.co.x += dx
    bp.co.y += dy
    bp.handle_left.x += dx
    bp.handle_left.y += dy
    bp.handle_right.x += dx
    bp.handle_right.y += dy
    obj.data.update_tag()
