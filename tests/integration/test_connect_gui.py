"""Real-UI checks for the connect features: Blender (GUI, not background) under Xvfb.

* the Make 3D panel draws the Decal rows; a plane decal is created and a live colour edit is applied by the
  engine's own timer (no flush) to the same image;
* the msgbus subscription on the active object selects the linked objects (and honours the switch);
* ``seam.result_paint_setup`` enters Texture Paint and a native brush stroke paints the SEAM paint image.

Runs the system ``blender`` binary in a subprocess; skipped when ``xvfb-run`` or the binary is missing.
"""

import json
import os
import shutil
import subprocess
import textwrap

import pytest

GUI_SCRIPT = textwrap.dedent(r'''
    import bpy, os, json, traceback
    import addon_utils
    out = {"drawn": []}

    def dump(code):
        with open(os.environ["SEAM_GUI_OUT"], "w") as fh:
            json.dump(out, fh, indent=1)
        os._exit(code)

    if addon_utils.enable("seam_studio", default_set=True, handle_error=None) is None:
        out["error"] = "seam_studio add-on could not be enabled"
        dump(1)
    import numpy as np
    from seam_studio.vector import canvas
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint
    from seam_studio.link import ui as lui, lifecycle, resultpaint
    from seam_studio.paint import rasterdoc
    scene = bpy.context.scene
    sq = [(-100, -100), (100, -100), (100, 100), (-100, 100)]
    src = canvas.new_shape_object(scene, "Logo", [BezierPath([BezierPoint(p, p, p, "VECTOR", "VECTOR") for p in sq], True)])
    src.seam.style.fill_color = (0.9, 0.1, 0.1, 1.0)
    out["subscribed"] = lifecycle.subscribe_selection_sync()

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
        dump(1)

    state = {"i": 0}

    def view3d():
        win = bpy.context.window_manager.windows[0]
        area = next(a for a in win.screen.areas if a.type == "VIEW_3D")
        return win, area, next(r for r in area.regions if r.type == "WINDOW")

    def pixel(img, x, y):
        a = rasterdoc.read_image_srgb8(img)
        return [int(v) for v in a[y, x]]

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
                    out["make_decal"] = sorted(bpy.ops.seam.make_3d(recipe="DECAL", decal_mode="PLANE", decal_resolution=64))
                lk = scene.seam.links[0]
                out["decal"] = [lk.state, lk.build_count, list(lk.decal_image.size), pixel(lk.decal_image, 32, 32)]
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
                return 0.3
            if i == 1:
                src.seam.style.fill_color = (0.1, 0.9, 0.1, 1.0)
                out["builds_before_tick"] = scene.seam.links[0].build_count
                return 0.8   # the engine timer must apply it: no explicit flush here
            if i == 2:
                lk = scene.seam.links[0]
                out["after_tick"] = [lk.state, lk.build_count, pixel(lk.decal_image, 32, 32)]
                # linked selection through the msgbus: activate the result, the source gets selected
                for o in list(bpy.context.selected_objects):
                    o.select_set(False)
                bpy.context.view_layer.objects.active = lk.target
                lk.target.select_set(True)
                src.select_set(False)
                return 0.8
            if i == 3:
                out["sync_result_to_source"] = src.select_get()
                bpy.context.window_manager.seam_sync_link_selection = False
                for o in list(bpy.context.selected_objects):
                    o.select_set(False)
                bpy.context.view_layer.objects.active = scene.seam.links[0].target
                scene.seam.links[0].target.select_set(True)
                src.select_set(False)
                bpy.context.window_manager.seam_sync_link_selection = False
                return 0.8
            if i == 4:
                out["sync_off_leaves_source"] = not src.select_get()
                bpy.context.window_manager.seam_sync_link_selection = True
                # paint on the decal plane: native Texture Paint + a real brush stroke
                tgt = scene.seam.links[0].target
                with bpy.context.temp_override(window=win, area=area, region=region):
                    for o in list(bpy.context.selected_objects):
                        o.select_set(False)
                    bpy.context.view_layer.objects.active = tgt
                    tgt.select_set(True)
                    bpy.ops.view3d.view_selected()
                    out["paint_setup"] = sorted(bpy.ops.seam.result_paint_setup(size=64))
                    out["mode"] = bpy.context.mode
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
                return 0.3
            if i == 5:
                tgt = scene.seam.links[0].target
                img = resultpaint.find_paint_image(tgt)
                before = np.asarray(img.pixels[:]).sum()
                cx, cy = region.width // 2, region.height // 2
                stroke = [{"name": "", "location": (0, 0, 0), "mouse": (cx + k * 3, cy), "mouse_event": (cx + k * 3, cy),
                           "pen_flip": 0, "is_start": k == 0, "size": 80.0, "pressure": 1.0, "time": 0.1 * k,
                           "x_tilt": 0.0, "y_tilt": 0.0} for k in range(8)]
                with bpy.context.temp_override(window=win, area=area, region=region):
                    out["stroke"] = sorted(bpy.ops.paint.image_paint(stroke=stroke))
                after = np.asarray(img.pixels[:]).sum()
                out["painted"] = [float(before), float(after)]
                bpy.ops.object.mode_set(mode="OBJECT")
                lk = scene.seam.links[0]
                out["link_after_paint"] = [lk.state]
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
                return 0.3
            out["ok"] = True
        except Exception:
            out["error"] = traceback.format_exc()
        dump(0)

    bpy.app.timers.register(step, first_interval=1.0)
''')


def _blender():
    return os.environ.get("SEAM_BLENDER") or shutil.which("blender")


@pytest.mark.skipif(not shutil.which("xvfb-run") or not _blender(), reason="needs xvfb-run and a blender binary")
def test_decal_selection_sync_and_texture_paint_in_gui(tmp_path):
    script = tmp_path / "gui_connect.py"
    script.write_text(GUI_SCRIPT)
    out_path = tmp_path / "gui_connect.json"
    env = dict(os.environ, SEAM_GUI_OUT=str(out_path))
    proc = subprocess.run(["xvfb-run", "-a", "-s", "-screen 0 1280x800x24", _blender(), "--factory-startup",
                           "--python", str(script)],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    output = proc.stdout.decode("utf-8", "replace")
    assert out_path.exists(), output[-4000:]
    out = json.loads(out_path.read_text())
    assert "error" not in out, out["error"]
    assert out.get("ok"), out
    assert "Traceback" not in output, output[-4000:]
    assert out["subscribed"] is True
    assert out["make_decal"] == ["FINISHED"]
    state, builds, size, centre = out["decal"]
    assert (state, builds, size) == ("LIVE", 1, [64, 64]) and centre[0] > 200 and centre[3] == 255
    assert out["builds_before_tick"] == 1
    state, builds, centre = out["after_tick"]
    assert (state, builds) == ("LIVE", 2) and centre[1] > 200 and centre[0] < 100   # applied by the engine timer
    assert {"SEAM_PT_make_3d"} <= set(out["drawn"]), out["drawn"]
    assert out["sync_result_to_source"] is True            # msgbus -> sync_linked_selection
    assert out["sync_off_leaves_source"] is True           # the switch is honoured
    assert out["paint_setup"] == ["FINISHED"] and out["mode"] == "PAINT_TEXTURE"
    assert out["stroke"] == ["FINISHED"]
    assert out["painted"][1] > out["painted"][0]           # a native brush stroke reached the SEAM paint image
    assert out["link_after_paint"] == ["LIVE"]
