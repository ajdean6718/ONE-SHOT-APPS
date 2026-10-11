# gui-no-startup
"""GUI scenario 5: the template without startup.blend (stale or missing): workspaces are recreated at launch."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import harness  # noqa: E402


def scenario(h):
    yield from h.wait_ready(2.0)
    from seam_studio.ui import workspaces as W
    tpl = next(p for p in bpy.utils.app_template_paths() if os.path.exists(os.path.join(p, "seam", "__init__.py")))
    h.check(not os.path.exists(os.path.join(tpl, "seam", "startup.blend")), "scenario runs against a template without startup.blend")
    ok = yield from h.wait_until(lambda: all(n in bpy.data.workspaces for n in W.SEAM_WORKSPACES) and
                                 h.window.workspace.name == "Vector", 120, "automatic workspace repair")
    h.check(ok, "workspaces were recreated automatically at launch")
    yield from h.wait(3.0)
    scene = bpy.context.scene
    h.check(sorted(o.name for o in bpy.data.objects) == ["Artboard 1", "SEAM Canvas"],
            f"factory cube/camera/light removed, canvas created ({sorted(o.name for o in bpy.data.objects)})")
    order = []
    yield from W.tab_order_steps(h.window, order)
    h.check(order[:4] == list(W.SEAM_WORKSPACES), f"tab order {order[:6]}")
    for name in ("Vector", "Paint", "Model", "Split"):
        yield from h.switch_workspace(name, settle=1.5)
        v3d = h.areas("VIEW_3D")
        h.check(len(v3d) == (2 if name == "Split" else 1), f"{name}: {len(v3d)} 3D view(s)")
        for a in v3d:
            persp = a.spaces.active.region_3d.view_perspective == "PERSP"
            if persp:
                h.check(W.artwork_hidden(h.window, a, scene) is True, f"{name}: 3D view hides the artwork")
            else:
                h.check(a.spaces.active.region_3d.lock_rotation, f"{name}: canvas view locked")
    h.screenshot("fallback_vector")


harness.main(scenario, name="gui_fallback", timeout=300)
