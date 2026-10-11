"""GUI scenario 1: launch the SEAM app template; every SEAM workspace exists, is set up and draws (screenshots)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import harness  # noqa: E402


def scenario(h):
    yield from h.wait_ready(3.0)
    h.check(hasattr(bpy.types.Scene, "seam"), "SEAM add-on is enabled by the application template")
    scene = bpy.context.scene
    h.check(scene.seam.doc.canvas_ready, "the new document has its canvas")
    h.check(sorted(o.name for o in bpy.data.objects) == ["Artboard 1", "SEAM Canvas"],
            f"startup file holds only the canvas objects ({sorted(o.name for o in bpy.data.objects)})")
    from seam_studio.ui import workspaces as W
    for name in W.SEAM_WORKSPACES:
        h.check(name in bpy.data.workspaces, f"workspace '{name}' exists")
    h.require(all(n in bpy.data.workspaces for n in W.SEAM_WORKSPACES), "all SEAM workspaces present")
    h.check(h.window.workspace.name == "Vector", f"app starts in Vector (is {h.window.workspace.name})")
    first = []
    yield from W.tab_order_steps(h.window, first)
    h.check(first[:4] == list(W.SEAM_WORKSPACES), f"tab order starts with Vector, Paint, Model, Split: {first[:6]}")
    h.check(all(n in first for n in ("Layout", "Modeling", "Sculpting", "UV Editing", "Texture Paint", "Shading",
                                      "Animation", "Rendering", "Compositing", "Geometry Nodes", "Scripting")),
            "native workspaces are kept")

    for name in W.SEAM_WORKSPACES:
        yield from h.switch_workspace(name, settle=2.0)
        v3d = h.areas("VIEW_3D")
        h.check(len(v3d) == (2 if name == "Split" else 1), f"{name}: {len(v3d)} 3D view(s)")
        for area in v3d:
            r = h.region(area)
            h.check(r.width > 100 and r.height > 100, f"{name}: 3D view has a size ({r.width}x{r.height})")
        canvas_views = [a for a in v3d if a.spaces.active.region_3d.view_perspective == "ORTHO"]
        if name in ("Vector", "Paint", "Split"):
            h.check(len(canvas_views) == 1 and canvas_views[0].spaces.active.region_3d.lock_rotation,
                    f"{name}: canvas view is orthographic with locked rotation")
        if name in ("Model", "Split"):
            model = [a for a in v3d if a.spaces.active.region_3d.view_perspective == "PERSP"]
            h.check(len(model) == 1, f"{name}: one perspective 3D view")
            if model:
                from seam_studio.ui import workspaces as _W
                h.check(_W.artwork_hidden(h.window, model[0], scene) is True, f"{name}: 3D view hides 'SEAM Artwork'")
        if name == "Paint":
            h.check(len(h.areas("IMAGE_EDITOR")) >= 1, "Paint: has an Image Editor")
        # the SEAM sidebar tab is the one shown (observed through the Document panel draw)
        import time
        from seam_studio.ui import panels
        t = time.monotonic()
        for a in v3d:
            a.tag_redraw()
        yield from h.wait(0.5)
        for a in v3d:
            ui = h.region(a, "UI")
            h.check(panels.seam_tab_shown(ui, t), f"{name}: SEAM sidebar tab is selected in the 3D view at x={a.x}")
        # canvas views show the whole artboard (fitted to this window size, not to the size at build time)
        from seam_studio.ui import workspaces as _WS
        for a in canvas_views:
            vw, vh = _WS.usable_extent(a)
            _cx, _cy, aw, ah = _WS.active_artboard_rect(scene)
            h.check(vw >= aw and vh >= ah and (vw <= aw * 1.3 or vh <= ah * 1.3),
                    f"{name}: artboard {aw:.0f}x{ah:.0f} fits the usable view tightly ({vw:.0f}x{vh:.0f} usable)")
        path = h.screenshot(name.lower())
        if path:
            sd = h.image_stddev(path)
            h.check(sd > 0.03, f"{name}: screenshot is not blank (stddev {sd:.3f})")
        else:
            h.check(False, f"{name}: screenshot captured")
    yield from h.switch_workspace("Vector", settle=1.0)


harness.main(scenario, name="gui_launch")
