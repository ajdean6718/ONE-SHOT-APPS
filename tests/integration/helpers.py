"""Helpers shared by integration tests (import as ``from helpers import ...``)."""

import bpy


def save_and_reload(path):
    """Save the current file, open it again, return the new context scene."""
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    return bpy.context.scene


def flush():
    from seam_studio.engine import core
    return core.flush()


def make_curve(name="shape", rings=(((-50, -50), (50, -50), (50, 50), (-50, 50)),), cyclic=True, role="SHAPE",
               collection=None):
    """Create a 2D bezier curve object with VECTOR handles from point rings (canvas px)."""
    from seam_studio import ids
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    for ring in rings:
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(ring) - 1)
        sp.use_cyclic_u = cyclic
        for bp, (x, y) in zip(sp.bezier_points, ring):
            bp.co = (x, y, 0.0)
            bp.handle_left_type = bp.handle_right_type = "VECTOR"
    ob = bpy.data.objects.new(name, cu)
    (collection or bpy.context.scene.collection).objects.link(ob)
    ids.ensure_id(ob)
    ob.seam.role = role
    return ob
