"""Run in a background Blender: open the auto save file named by SEAM_AUTOSAVE_FILE (File > Recover > Auto Save) and
write what the layers hold to SEAM_GUI_OUT (see test_paint_undo_gui.py)."""

import json
import os

import addon_utils
import bpy

addon_utils.enable("seam_studio", default_set=True, handle_error=None)
bpy.ops.wm.open_mainfile(filepath=os.environ["SEAM_AUTOSAVE_FILE"])
from seam_studio.paint import rasterdoc as rd  # noqa: E402

doc = bpy.context.scene.seam.rasters[0]
out = {"layers": [[int(v) for v in rd.read_image_srgb8(ly.image)[10, 10]] for ly in doc.layers],
       "revs": [ly.rev for ly in doc.layers], "has_sel": bool(doc.has_selection), "warn": rd.pixel_restore_warning(),
       "greens": 0}
x0 = rd.read_image_srgb8(doc.layers[0].image)
out["greens"] = int(((x0[..., 1] > 200) & (x0[..., 0] < 60) & (x0[..., 3] > 200)).sum())
with open(os.environ["SEAM_GUI_OUT"], "w") as fh:
    json.dump(out, fh)
os._exit(0)
