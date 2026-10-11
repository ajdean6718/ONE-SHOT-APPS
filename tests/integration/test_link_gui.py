"""Real-UI smoke for links: Blender (GUI, not background) under Xvfb draws the Make 3D and Live Link
panels for real, the engine timer applies a live update, show_result / edit_source run with a window.

Runs the system ``blender`` binary in a subprocess (so it is the same check from both suites); skipped
when ``xvfb-run`` or the binary is missing. The add-on is found through BLENDER_USER_SCRIPTS, which
scripts/test.sh points at this tree.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

GUI_SCRIPT = textwrap.dedent(r'''
    import bpy, os, json, traceback
    import addon_utils
    out = {"log": [], "drawn": []}
    def log(*a):
        out["log"].append(" ".join(str(x) for x in a))

    if addon_utils.enable("seam_studio", default_set=True, handle_error=None) is None:
        with open(os.environ["SEAM_GUI_OUT"], "w") as fh:
            json.dump({"error": "seam_studio add-on could not be enabled"}, fh)
        os._exit(1)
    from seam_studio.vector import canvas
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint
    from seam_studio.link import ui as lui
    scene = bpy.context.scene
    sq = [(-100, -100), (100, -100), (100, 100), (-100, 100)]
    src = canvas.new_shape_object(scene, "Logo", [BezierPath([BezierPoint(p, p, p, "VECTOR", "VECTOR") for p in sq], True)])

    # Blender 4.0 cannot select a sidebar tab from Python: show the SEAM panels on the default "Item"
    # tab (re-registered once here, before anything is drawn) and record each real draw call.
    def recording(orig, name):
        def draw(self, context):
            out["drawn"].append(name)
            orig(self, context)
        return draw

    try:
        for cls in (lui.SEAM_PT_make_3d, lui.SEAM_PT_live_link):
            bpy.utils.unregister_class(cls)
            cls.bl_category = "Item"
            cls.draw = recording(cls.draw, cls.__name__)
            bpy.utils.register_class(cls)
    except Exception:
        out["error"] = traceback.format_exc()
        with open(os.environ["SEAM_GUI_OUT"], "w") as fh:
            json.dump(out, fh, indent=1)
        os._exit(1)

    state = {"i": 0}

    def view3d():
        win = bpy.context.window_manager.windows[0]
        area = next(a for a in win.screen.areas if a.type == "VIEW_3D")
        return win, area, next(r for r in area.regions if r.type == "WINDOW")

    def finish():
        with open(os.environ["SEAM_GUI_OUT"], "w") as fh:
            json.dump(out, fh, indent=1)
        os._exit(0)

    def step():
        try:
            i = state["i"]
            state["i"] += 1
            win, area, region = view3d()
            if i == 0:
                area.spaces.active.show_region_ui = True
                with bpy.context.temp_override(window=win, area=area, region=region):
                    bpy.context.view_layer.objects.active = src
                    src.select_set(True)
                    out["make_3d"] = sorted(bpy.ops.seam.make_3d(recipe="EXTRUDE", orientation="FLAT"))
                lk = scene.seam.links[0]
                out["built"] = [lk.state, lk.build_count, len(lk.target.data.vertices)]
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
                return 0.3
            if i == 1:
                bp = src.data.splines[0].bezier_points[1]
                bp.co.x += 100; bp.handle_left.x += 100; bp.handle_right.x += 100
                src.data.update_tag()
                out["builds_before_tick"] = scene.seam.links[0].build_count
                return 0.6  # no explicit flush: the engine's own timer must apply it
            if i == 2:
                lk = scene.seam.links[0]
                xs = [v.co.x for v in lk.target.data.vertices]
                out["after_tick"] = [lk.state, lk.build_count, round(max(xs) - min(xs), 4)]
                with bpy.context.temp_override(window=win, area=area, region=region):
                    bpy.context.view_layer.objects.active = lk.target
                    bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
                    out["show_result"] = sorted(bpy.ops.seam.show_result())
                    out["edit_source"] = sorted(bpy.ops.seam.edit_source())
                    out["mode_after_edit_source"] = bpy.context.mode
                    bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
                return 0.2
            out["ok"] = True
        except Exception:
            out["error"] = traceback.format_exc()
        finish()

    bpy.app.timers.register(step, first_interval=1.0)
''')


def _blender():
    return os.environ.get("SEAM_BLENDER") or shutil.which("blender")


@pytest.mark.skipif(not shutil.which("xvfb-run") or not _blender(), reason="needs xvfb-run and a blender binary")
def test_link_panels_draw_and_live_update_in_gui(tmp_path):
    script = tmp_path / "gui_link.py"
    script.write_text(GUI_SCRIPT)
    out_path = tmp_path / "gui_link.json"
    env = dict(os.environ, SEAM_GUI_OUT=str(out_path))
    proc = subprocess.run(["xvfb-run", "-a", "-s", "-screen 0 1280x800x24", _blender(), "--factory-startup",
                           "--python", str(script)],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    output = proc.stdout.decode("utf-8", "replace")
    assert out_path.exists(), output[-4000:]
    out = json.loads(out_path.read_text())
    assert "error" not in out, out["error"]
    assert out.get("ok"), out
    assert "Traceback" not in output, output[-4000:]
    assert out["make_3d"] == ["FINISHED"] and out["built"][:2] == ["LIVE", 1] and out["built"][2] > 0
    assert out["builds_before_tick"] == 1
    assert out["after_tick"] == ["LIVE", 2, 3.0]  # applied by the engine timer, no manual flush
    assert {"SEAM_PT_make_3d", "SEAM_PT_live_link"} <= set(out["drawn"]), out["drawn"]
    assert out["show_result"] == ["FINISHED"]
    assert out["edit_source"] == ["FINISHED"] and out["mode_after_edit_source"] == "EDIT_CURVE"
