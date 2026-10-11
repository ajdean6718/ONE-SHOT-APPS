"""Foundation canvas API: root, layers, artboards, shape placement, z stacking."""

import bpy

from helpers import save_and_reload


def test_canvas_and_shapes(seam, blend_path):
    from seam_studio.vector import canvas
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint

    scene = bpy.context.scene
    root = canvas.ensure_canvas(scene)
    assert root.seam.role == "CANVAS_ROOT"
    assert tuple(root.location) == (0.0, -20000.0, 0.0)
    assert len(canvas.layers(scene)) == 1 and len(canvas.artboards(scene)) == 1
    sq = [(0, 0), (100, 0), (100, 100), (0, 100)]
    path = BezierPath([BezierPoint(p, p, p, "VECTOR", "VECTOR") for p in sq], closed=True)
    a = canvas.new_shape_object(scene, "A", [path], location=(10, -10))
    b = canvas.new_shape_object(scene, "B", [path], location=(20, -20))
    assert a.parent is root and b.parent is root
    assert b.location.z > a.location.z
    assert [o.name for o in canvas.artwork_objects(scene)] == ["A", "B"]
    assert artwork_hidden_in_render(scene)
    canvas.ensure_canvas(scene)  # idempotent
    assert len(canvas.artboards(scene)) == 1
    save_and_reload(blend_path)
    scene = bpy.context.scene
    assert len(bpy.data.objects["A"].data.splines[0].bezier_points) == 4
    assert bpy.data.objects["A"].parent.seam.role == "CANVAS_ROOT"


def artwork_hidden_in_render(scene):
    from seam_studio.vector import canvas
    return canvas.artwork_collection(scene).hide_render
