"""Build ``application_templates/seam/startup.blend`` (owner: shell). Runs inside GUI Blender.

    xvfb-run -a -s "-screen 0 1920x1080x24" blender --factory-startup --enable-event-simulate \\
        --window-geometry 0 0 1920 1080 --python scripts/build_template.py -- [--out PATH]

(``scripts/bootstrap.sh`` and ``scripts/package.py`` call this for you.) What it does, always from Blender's factory
startup so the result is identical on every run (idempotent: running it again rebuilds the same file):

1. registers the add-on from ``extensions/seam_studio`` (no installation needed),
2. removes the default cube/camera/light, creates the SEAM canvas (``document.ensure_document``: artboard,
   layer, view transform Standard),
3. creates the workspaces Vector, Paint, Model and Split by duplicating/reconfiguring Blender's native ones
   (``ui.workspaces.repair_steps``), orders them first and keeps all native workspaces behind them,
4. selects the SEAM tab of the sidebar in every SEAM 3D view (Blender 4.0 has no API for it: a simulated
   Ctrl+mouse-wheel over the sidebar cycles the tabs, 5.x sets ``Region.active_panel_category``),
5. activates Vector and writes the file compressed.

Exit code 0 only when the file was written; the reason is printed otherwise.
"""

import os
import sys
import traceback

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "extensions"))

DEFAULT_OUT = os.path.join(ROOT, "application_templates", "seam", "startup.blend")


def _arg(name, default=None):
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if name in argv:
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) else default
    return default


def fail(msg, code=2):
    print("BUILD_TEMPLATE FAILED:", msg, flush=True)
    os._exit(code)


def ensure_seam_tab(window, ws_mod, panels, max_clicks=20):
    """Select the SEAM sidebar tab in every 3D view of the current workspace.

    4.0 has no API for the active sidebar category (5.x has ``Region.active_panel_category``, set first). So click
    down the tab column at the right edge of the sidebar until the SEAM Document panel is drawn there
    (``ui.panels.seam_tab_shown`` observes the draw). The selection is stored in the file."""
    import time
    ws_mod.select_sidebar_tab(window.screen)
    for area in ws_mod.view3d_areas(window.screen):
        ui = next((r for r in area.regions if r.type == "UI"), None)
        if ui is None or ui.width < 10:
            fail(f"no visible sidebar in a 3D view of '{window.workspace.name}'")
        x, top = ui.x + ui.width - 8, ui.y + ui.height
        for k in range(max_clicks + 1):
            t = time.monotonic()
            area.tag_redraw()
            yield 6
            if panels.seam_tab_shown(ui, t):
                break
            if k == max_clicks:
                fail(f"could not select the SEAM sidebar tab in '{window.workspace.name}'")
            y = top - 12 - k * 16
            window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)
            yield 3
            window.event_simulate(type="LEFTMOUSE", value="PRESS", x=x, y=y)
            yield 2
            window.event_simulate(type="LEFTMOUSE", value="RELEASE", x=x, y=y)
            yield 3


def steps(out_path):
    import seam_studio
    from seam_studio.ui import document, workspaces as ws_mod
    seam_studio.register()
    window = bpy.context.window_manager.windows[0]
    scene = bpy.context.scene

    # 1) clean document
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)
    document.ensure_document(scene)
    scene.seam.doc.canvas_ready = True
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    yield 2

    # 2) workspaces
    log = {}
    yield from ws_mod.repair_steps(window, scene, log)
    if log.get("error"):
        fail("workspace creation failed: " + log["error"])
    print("created:", log["created"], "configured:", log["configured"], flush=True)

    # 3) sidebar tab
    from seam_studio.ui import panels
    for name in ws_mod.SEAM_WORKSPACES:
        yield from ws_mod._switch(window, bpy.data.workspaces[name])
        yield from ws_mod._settle(3)
        yield from ensure_seam_tab(window, ws_mod, panels)

    # 4) frame canvas views at the build window size, activate Vector, save
    for name in ws_mod.SEAM_WORKSPACES:
        yield from ws_mod._switch(window, bpy.data.workspaces[name])
        yield from ws_mod._settle(2)
        ws_mod.frame_canvas_views()
    yield from ws_mod._switch(window, bpy.data.workspaces[ws_mod.VECTOR])
    yield from ws_mod._settle(2)
    if bpy.context.view_layer.objects.active is not None:
        bpy.context.view_layer.objects.active = None
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True, check_existing=False)
    if not os.path.exists(out_path):
        fail("save_as_mainfile did not write " + out_path)
    print("BUILD_TEMPLATE OK", out_path, os.path.getsize(out_path), "bytes", flush=True)
    yield 1


def main():
    out = os.path.abspath(_arg("--out", DEFAULT_OUT))
    if bpy.app.background:
        fail("must run in GUI Blender (xvfb-run ... blender --factory-startup --python scripts/build_template.py)")
    gen = steps(out)

    def tick():
        try:
            n = next(gen)
        except StopIteration:
            os._exit(0)
        except Exception:
            traceback.print_exc()
            fail("exception while building (see traceback)")
        return 0.05 * max(1, n or 1)

    bpy.app.timers.register(tick, first_interval=1.0)


main()
