"""GUI scenario 3: shortcuts, header widget, quick starts, health check and workspace repair in the real UI."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import harness  # noqa: E402


def scenario(h):
    yield from h.wait_ready(3.0)
    from seam_studio import compat
    from seam_studio.ui import keymaps
    from seam_studio.ui import workspaces as W
    scene = bpy.context.scene

    # ---- keymaps registered in the live add-on keyconfig and conflict-free against the live default keyconfig
    kc_addon = compat.keyconfig_addon()
    km = kc_addon.keymaps.get("3D View")
    h.require(km is not None, "add-on keyconfig has a '3D View' keymap")
    bound = [(k.idname, k.type, k.ctrl, k.alt, k.shift) for k in km.keymap_items if k.idname.startswith("seam.")]
    for it in keymaps.available_items():
        h.check((it.idname, it.type, it.ctrl, it.alt, it.shift) in bound, f"shortcut bound: {keymaps.key_text(it)} -> {it.idname}")
    kc_default = bpy.context.window_manager.keyconfigs.default
    clashes = []
    for it in keymaps.available_items():
        for dkm in kc_default.keymaps:
            if dkm.space_type not in ("VIEW_3D", "EMPTY"):
                continue
            for k in dkm.keymap_items:
                if k.active and k.type == it.type and (k.ctrl, k.alt, k.shift) == (it.ctrl, it.alt, it.shift) and not k.oskey:
                    clashes.append(f"{keymaps.key_text(it)} vs {dkm.name}:{k.idname}")
    h.check(not clashes, f"no conflict with the live default keyconfig {clashes}")

    # ---- shortcuts switch workspaces (real key events through the window manager)
    yield from h.switch_workspace("Vector", settle=1.0)
    for key, name in (("THREE", "Model"), ("FOUR", "Split"), ("TWO", "Paint"), ("ONE", "Vector")):
        yield from h.key(key, ctrl=True, alt=True)
        yield from h.wait_until(lambda n=name: h.window.workspace.name == n, 5, f"workspace {name}")
        h.check(h.window.workspace.name == name, f"Ctrl+Alt+{key} switches to {name} (is {h.window.workspace.name})")
        yield from h.wait(0.6)

    # ---- live toggle shortcut + header widget
    eng = scene.seam.engine
    h.check(eng.live_enabled, "live updates start enabled")
    yield from h.key("L", ctrl=True, alt=True)
    yield from h.wait(0.5)
    h.check(eng.live_enabled is False, "Ctrl+Alt+L pauses live updates")
    h.redraw()
    h.screenshot("live_paused")
    yield from h.key("L", ctrl=True, alt=True)
    yield from h.wait(0.5)
    h.check(eng.live_enabled is True, "Ctrl+Alt+L resumes live updates")

    # ---- Start panel reflects the document state; quick start 'New Vector Document' does real work
    yield from h.switch_workspace("Model", settle=1.0)
    ok, why = __import__("seam_studio.ui.ops", fromlist=["x"]).quickstart_status("VECTOR", scene)
    h.check(ok, f"quick start 'New Vector Document' is available in an empty document ({why})")
    area = h.areas("VIEW_3D")[0]
    with bpy.context.temp_override(window=h.window, screen=h.window.screen, area=area, region=h.region(area)):
        res = bpy.ops.seam.new_vector_document()
    h.check("FINISHED" in res, "seam.new_vector_document runs")
    yield from h.wait_until(lambda: h.window.workspace.name == "Vector", 5, "switch to Vector")
    h.check(h.window.workspace.name == "Vector", "new vector document opens the Vector workspace")
    yield from h.wait(1.0)
    h.screenshot("after_new_vector_document")

    # ---- health check operator in the UI context
    area = h.areas("VIEW_3D")[0]
    with bpy.context.temp_override(window=h.window, screen=h.window.screen, area=area, region=h.region(area)):
        res = bpy.ops.seam.health_check()
    h.check("FINISHED" in res, "seam.health_check runs in the GUI")

    # ---- workspace repair: delete three workspaces, mangle a view, repair, check
    yield from h.switch_workspace("Vector", settle=0.5)
    for name in ("Paint", "Model", "Split"):
        bpy.data.batch_remove([bpy.data.workspaces[name]])
    h.check([n in bpy.data.workspaces for n in ("Paint", "Model", "Split")] == [False] * 3, "three SEAM workspaces removed")
    vec_area = h.areas("VIEW_3D")[0]
    r3d = vec_area.spaces.active.region_3d
    r3d.lock_rotation = False
    r3d.view_perspective = "PERSP"
    vec_area.spaces.active.overlay.show_floor = True
    yield from h.wait(0.5)
    with bpy.context.temp_override(window=h.window, screen=h.window.screen, area=vec_area, region=h.region(vec_area)):
        res = bpy.ops.seam.workspace_repair("INVOKE_DEFAULT")
    h.check("RUNNING_MODAL" in res, f"repair runs as a modal operator ({res})")
    yield from h.wait_until(lambda: not W.repair_running["active"] and
                            all(n in bpy.data.workspaces for n in W.SEAM_WORKSPACES) and
                            h.window.workspace.name == "Vector", 120, "repair to finish")
    yield from h.wait(3.0)
    h.check(all(n in bpy.data.workspaces for n in W.SEAM_WORKSPACES), "repair recreated Paint, Model and Split")
    order = []
    yield from W.tab_order_steps(h.window, order)
    h.check(order[:4] == list(W.SEAM_WORKSPACES), f"repair restored the tab order {order[:6]}")
    vec_area = h.areas("VIEW_3D", workspace="Vector")[0]
    sp = vec_area.spaces.active
    h.check(sp.region_3d.view_perspective == "ORTHO" and sp.region_3d.lock_rotation and not sp.overlay.show_floor,
            "repair restored the 2D canvas view of Vector (ortho, locked, no floor)")
    for name in ("Model", "Split"):
        yield from h.switch_workspace(name, settle=1.5)
        model = [a for a in h.areas("VIEW_3D") if a.spaces.active.region_3d.view_perspective == "PERSP"]
        h.check(len(model) == 1 and W.artwork_hidden(h.window, model[0], scene) is True,
                f"{name}: repaired 3D view hides the artwork")
    yield from h.switch_workspace("Split", settle=1.5)
    h.screenshot("repaired_split")
    h.note("repaired workspaces keep Blender's default sidebar tab: no API selects the SEAM tab (startup.blend stores it)")


harness.main(scenario, name="gui_shell", timeout=300)
