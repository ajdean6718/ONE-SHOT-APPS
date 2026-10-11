"""GUI driver for test_paint_undo_gui.py (runs inside a real Blender window under Xvfb; not a test module).

Real brush strokes (event_simulate on the 3D viewport), SEAM operators and real Ctrl+Z / Ctrl+Shift+Z.
Scenario from env SEAM_GUI_SCENARIO: ``fill`` (undo of pixel operators vs strokes) or ``props`` (layer property edits
in paint mode); results go to the JSON file SEAM_GUI_OUT.
"""

import json
import os
import traceback

import addon_utils
import bpy
import numpy as np

SCENARIO = os.environ.get("SEAM_GUI_SCENARIO", "fill")
OUT = os.environ["SEAM_GUI_OUT"]
out = {"scenario": SCENARIO, "log": [], "m": {}}


def log(*a):
    out["log"].append(" ".join(str(x) for x in a))


def finish(code=0):
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    os._exit(code)


if addon_utils.enable("seam_studio", default_set=True, handle_error=None) is None:
    out["error"] = "seam_studio add-on could not be enabled"
    finish(1)


def run():
    from bpy_extras import view3d_utils
    from mathutils import Vector
    from seam_studio.paint import rasterdoc as rd

    win = bpy.context.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == "VIEW_3D")
    region = next(r for r in area.regions if r.type == "WINDOW")
    scene = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    ov = dict(window=win, area=area, region=region)

    def cur():
        return bpy.context.scene.seam.rasters[0]

    def a8():
        return rd.read_image_srgb8(cur().layers[0].image)

    def reds():
        x = a8()
        return int(((x[..., 0] > 200) & (x[..., 2] < 50) & (x[..., 3] > 200)).sum())

    def blues():
        x = a8()
        return int(((x[..., 2] > 200) & (x[..., 0] < 50) & (x[..., 3] > 200)).sum())

    def state(tag):
        d = cur()
        st = dict(reds=reds(), blues=blues(), opacity=round(float(d.layers[0].opacity), 3), mode=d.canvas.mode,
                  has_sel=bool(d.has_selection), rev=d.layers[0].rev, warn=rd.pixel_restore_warning())
        out["m"][tag] = st
        log(tag, st)
        return st

    with bpy.context.temp_override(**ov):
        bpy.ops.seam.raster_new(name="U", width=64, height=64, background="TRANSPARENT")
    yield 0.5
    win.event_simulate(type="ESC", value="PRESS")
    win.event_simulate(type="ESC", value="RELEASE")
    yield 0.3
    with bpy.context.temp_override(**ov):
        bpy.ops.ed.undo_push(message="created")
        bpy.ops.ed.undo_push(message="created2")

    def enter_paint():
        with bpy.context.temp_override(**ov):
            bpy.ops.seam.raster_paint_mode("EXEC_DEFAULT", True)

    enter_paint()
    yield 1.0

    def to_win(px, py):
        ob = cur().canvas        # memfile undo replaces the datablocks: never keep handles across steps
        rv3d = area.spaces.active.region_3d
        p = view3d_utils.location_3d_to_region_2d(region, rv3d, ob.matrix_world @ Vector((px, -py, 0)))
        return int(p.x + region.x), int(p.y + region.y)

    c0, c1 = to_win(0, 0), to_win(64, 64)
    sc_px = abs(c1[0] - c0[0]) / 64.0

    def setup_brush():
        ip = bpy.context.scene.tool_settings.image_paint
        ip.brush.size = max(2, int(4 * sc_px))
        ip.brush.color = (1, 0, 0)
        try:
            ip.brush.strength = 1.0
        except Exception:
            pass

    setup_brush()

    def mouse(t, v, x, y, **kw):
        win.event_simulate(type=t, value=v, x=x, y=y, **kw)

    def real_stroke(p0, p1):
        a, b = to_win(*p0), to_win(*p1)
        mouse("MOUSEMOVE", "NOTHING", *a)
        yield 0.2
        mouse("LEFTMOUSE", "PRESS", *a)
        yield 0.2
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        mouse("MOUSEMOVE", "NOTHING", *mid)
        yield 0.2
        mouse("MOUSEMOVE", "NOTHING", *b)
        yield 0.2
        mouse("LEFTMOUSE", "RELEASE", *b)
        yield 0.6

    cen = to_win(32, 32)

    def key(shift=False):
        mouse("MOUSEMOVE", "NOTHING", *cen)
        win.event_simulate(type="Z", value="PRESS", x=cen[0], y=cen[1], ctrl=True, shift=shift)
        win.event_simulate(type="Z", value="RELEASE", x=cen[0], y=cen[1], ctrl=True, shift=shift)

    def op(fn, **kw):
        with bpy.context.temp_override(**ov):
            r = fn("EXEC_DEFAULT", True, **kw)
        out["m"].setdefault("ops", []).append(sorted(r))
        return r

    state("start")
    yield from real_stroke((8, 16), (56, 16))
    state("stroke1")

    if SCENARIO == "fill":
        op(bpy.ops.seam.raster_select_rect, x0=0, y0=40, x1=64, y1=64)
        yield 0.6
        op(bpy.ops.seam.raster_fill, color=(0, 0, 1, 1))
        yield 1.0
        state("fill")
        key()
        yield 1.0
        state("undo_fill")           # ONE Ctrl+Z must undo the Fill, strokes stay
        key(shift=True)
        yield 1.0
        state("redo_fill")
        if cur().canvas.mode != "TEXTURE_PAINT":
            enter_paint()
            yield 1.0
        setup_brush()
        op(bpy.ops.seam.raster_select_none)       # the selection would stencil the second stroke away
        yield 0.6
        yield from real_stroke((8, 30), (56, 30))
        state("stroke2")
        op(bpy.ops.seam.raster_clear)
        yield 1.0
        state("clear")
        key()
        yield 1.0
        state("undo_clear")          # the Clear goes; both strokes and the fill stay
        key()
        yield 1.0
        state("undo_stroke2")
        for i in range(5):                        # walk back through the rest of the history
            key()
            yield 1.0
            state(f"undo_more_{i}")
        comp = rd.read_image_srgb8(cur().composite)
        from seam_studio.seam_core import raster as R
        want = R.to_uint8_srgb(rd.composite_array(cur()))
        out["m"]["composite_diff"] = int(np.abs(comp.astype(int) - want.astype(int)).max())

    elif SCENARIO == "props":
        cur().layers[0].opacity = 0.3             # the layer-list slider
        with bpy.context.temp_override(**ov):
            bpy.ops.ed.undo_push(message="Opacity")   # the push the slider itself makes (image type in paint mode)
        yield 1.6                                 # SEAM pushes its own memfile step once the widget settled
        state("opacity")
        if cur().canvas.mode != "TEXTURE_PAINT":
            enter_paint()
            yield 1.0
        setup_brush()
        yield from real_stroke((8, 40), (56, 40))
        state("stroke2")
        key()
        yield 1.0
        state("undo_1")                           # stroke 2 goes
        key()
        yield 1.0
        state("undo_2")                           # the opacity edit goes (it was impossible before)
        key(shift=True)
        yield 1.0
        state("redo_1")
        key(shift=True)
        yield 1.0
        state("redo_2")

    elif SCENARIO in ("autosave_memfile", "autosave_paint", "autosave_stroke"):
        import glob
        tmp = os.environ["SEAM_AUTOSAVE_DIR"]
        prefs = bpy.context.preferences
        prefs.filepaths.temporary_directory = tmp
        prefs.filepaths.use_auto_save_temporary_files = True
        prefs.filepaths.auto_save_time = 1
        if SCENARIO == "autosave_memfile":
            with bpy.context.temp_override(**ov):  # Object mode: the active undo step is a memfile step
                bpy.ops.seam.raster_paint_mode("EXEC_DEFAULT", True, enter=False)
            yield 0.5
        # (autosave_paint keeps painting mode: the operators below run from Texture Paint mode)
        if SCENARIO == "autosave_stroke":
            cur().active_layer_index = 0
            setup_brush()
            ip = bpy.context.scene.tool_settings.image_paint
            ip.brush.color = (0, 1, 0)
            yield from real_stroke((8, 50), (56, 50))
            yield 3.0                             # painting paused: SEAM packs the stroke
            with bpy.context.temp_override(**ov):
                bpy.ops.seam.raster_paint_mode("EXEC_DEFAULT", True, enter=False)
            yield 0.5
        if SCENARIO != "autosave_stroke":
            op(bpy.ops.seam.raster_fill, color=(1, 0, 0, 1), opacity=1.0)
            yield 0.5
        op(bpy.ops.seam.raster_layer_add, name="Top")
        yield 0.5
        op(bpy.ops.seam.raster_select_rect, x0=0, y0=0, x1=32, y1=64)
        yield 0.3
        op(bpy.ops.seam.raster_fill, color=(0, 0, 1, 1))
        yield 1.0
        out["m"]["mode"] = cur().canvas.mode
        out["m"]["layers"] = [[int(v) for v in rd.read_image_srgb8(ly.image)[10, 10]] for ly in cur().layers]
        x0 = rd.read_image_srgb8(cur().layers[0].image)
        out["m"]["greens"] = int(((x0[..., 1] > 200) & (x0[..., 0] < 60) & (x0[..., 3] > 200)).sum())
        out["m"]["has_sel"] = bool(cur().has_selection)
        for _ in range(100):
            if glob.glob(os.path.join(tmp, "*autosave*.blend")):
                break
            yield 1.0
        yield 2.0
        out["m"]["autosaves"] = glob.glob(os.path.join(tmp, "*autosave*.blend"))

    out["ok"] = True


gen = None


def tick():
    global gen
    if gen is None:
        gen = run()
    try:
        d = next(gen)
        return float(d) if d else 0.1
    except StopIteration:
        finish(0)
    except Exception:
        out["error"] = traceback.format_exc()
        finish(1)


bpy.app.timers.register(tick, first_interval=2.0)
