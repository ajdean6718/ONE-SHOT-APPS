"""GUI scenario 2: draw a rectangle on the canvas with the vector rectangle tool (skipped until vector tools exist)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import harness  # noqa: E402


def scenario(h):
    yield from h.wait_ready(3.0)
    from seam_studio.ui import util
    from seam_studio.ui import workspaces as W
    ok, why = util.op_available("seam.draw_shape")
    if not ok:
        h.skip(f"vector-a drawing tools are not installed in this build ({why})")
    yield from h.switch_workspace("Vector", settle=2.0)
    area = h.areas("VIEW_3D")[0]
    region = h.region(area)
    scene = bpy.context.scene
    before = {o.name for o in scene.objects}

    with bpy.context.temp_override(window=h.window, screen=h.window.screen, area=area, region=region):
        res = bpy.ops.wm.tool_set_by_id(name="seam.tool_rect")
    h.require("FINISHED" in res, "seam.tool_rect can be activated")
    yield from h.wait(0.5)

    cx, cy = h.center(area)
    upp = W._extent_per_px(area.spaces.active, region)          # canvas px per screen px
    p0, p1 = (cx - 100, cy - 60), (cx + 100, cy + 60)
    yield from h.drag(p0, p1, steps=10)
    yield from h.wait(1.0)
    h.screenshot("rect_drawn")

    new = [o for o in scene.objects if o.name not in before and o.seam.role == "SHAPE"]
    h.require(len(new) == 1, f"exactly one new shape was created ({[o.name for o in new]})")
    shape = new[0]
    h.check(shape.seam.shape.kind == "RECT", f"shape kind is RECT (is {shape.seam.shape.kind})")
    exp_w, exp_h = 200 * upp, 120 * upp
    h.check(abs(shape.seam.shape.width - exp_w) <= 0.1 * exp_w + 2,
            f"width {shape.seam.shape.width:.1f} ~ dragged {exp_w:.1f} canvas px")
    h.check(abs(shape.seam.shape.height - exp_h) <= 0.1 * exp_h + 2,
            f"height {shape.seam.shape.height:.1f} ~ dragged {exp_h:.1f} canvas px")
    h.check(shape.parent is not None and shape.parent.seam.role == "CANVAS_ROOT", "shape is parented to the canvas")

    # undo removes it again (unified undo)
    with bpy.context.temp_override(window=h.window, screen=h.window.screen, area=area, region=region):
        bpy.ops.ed.undo()
    yield from h.wait(0.5)
    h.check(not [o for o in scene.objects if o.name not in before and o.seam.role == "SHAPE"], "undo removes the shape")


harness.main(scenario, name="gui_draw_rect")
