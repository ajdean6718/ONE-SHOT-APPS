"""GUI scenario 4: File > New with the SEAM template: new-document preferences, canvas, framing, reset."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import harness  # noqa: E402


def _new_file(h):
    with bpy.context.temp_override(window=h.window, screen=h.window.screen):
        res = bpy.ops.wm.read_homefile(app_template="seam")
    h.check("FINISHED" in res, f"File > New (seam template) runs ({res})")
    yield from h.wait(3.0)


def scenario(h):
    yield from h.wait_ready(3.0)
    from seam_studio.ui import prefs as P
    from seam_studio.ui import workspaces as W
    from seam_studio.vector import canvas

    prefs = P.get()
    prefs.default_canvas_width, prefs.default_canvas_height = 1280, 720
    prefs.default_units, prefs.default_dpi = "CM", 300.0

    # a document with content and edited views
    scene = bpy.context.scene
    ob = bpy.data.objects.new("scribble", bpy.data.curves.new("scribble", "CURVE"))
    canvas.add_artwork_object(scene, ob)
    h.check(ob.seam.role == "SHAPE", "content added to the startup document")

    yield from _new_file(h)
    scene = bpy.context.scene
    h.check(bpy.data.filepath == "", "new file is unsaved")
    h.check("scribble" not in bpy.data.objects, "File > New starts from a clean document")
    h.check(sorted(o.name for o in bpy.data.objects) == ["Artboard 1", "SEAM Canvas"], "only the canvas objects exist")
    ab = canvas.active_artboard(scene)
    h.check((ab.seam.artboard.width, ab.seam.artboard.height) == (1280, 720), "artboard follows the preferences (1280x720)")
    h.check(scene.seam.doc.units == "CM" and scene.seam.doc.dpi == 300.0, "units and dpi follow the preferences")
    h.check(scene.view_settings.view_transform == "Standard", "view transform is Standard")
    h.check(all(n in bpy.data.workspaces for n in W.SEAM_WORKSPACES), "SEAM workspaces present after File > New")
    h.check(h.window.workspace.name == "Vector", f"new file opens in Vector (is {h.window.workspace.name})")

    # first visit of each canvas workspace fits the artboard to the real window size
    for name in ("Vector", "Paint", "Split"):
        yield from h.switch_workspace(name, settle=1.5)
        yield from h.wait(1.0)
        for a in h.areas("VIEW_3D"):
            if a.spaces.active.region_3d.view_perspective != "ORTHO":
                continue
            vw, vh = W.usable_extent(a)
            _cx, _cy, aw, ah = W.active_artboard_rect(scene)
            h.check(vw >= aw and vh >= ah and (vw <= aw * 1.3 or vh <= ah * 1.3),
                    f"{name}: 1280x720 artboard fitted on first visit ({vw:.0f}x{vh:.0f} usable)")
    h.screenshot("new_file_split")
    yield from h.switch_workspace("Vector", settle=1.0)
    h.screenshot("new_file_vector")


harness.main(scenario, name="gui_newfile")
