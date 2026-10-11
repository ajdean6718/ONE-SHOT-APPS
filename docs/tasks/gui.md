# Task: gui (wave 2) — real-GUI scenarios, screenshot inspection, interface polish

Owned: `tests/gui/**`, `extensions/seam_studio/ui/**`, `application_templates/seam/**`, `scripts/build_template.py`, `scripts/seam-studio`, `scripts/bootstrap.sh`, `scripts/package.py`, `README.md`.
Runtime: Blender 4.0.2 GUI under Xvfb (`xvfb-run -a -s "-screen 0 1920x1080x24"`), `--enable-event-simulate`, harness in `tests/gui/harness.py`. Screenshots via `import -window root`. LOOK at every screenshot you cite (Read the PNG) and describe what is visible.
Scenarios (real input events through the real keymap/tools, not direct operator calls, except for setup):
- gui_vector: Rectangle tool drag on the canvas → shape; Tab into edit mode, select a node, G + mouse move + click → node moved; Text: add text, Tab, type characters, Tab → body changed; Appearance panel visible with values.
- gui_paint: new raster canvas → Paint workspace → brush stroke drag with the native Draw brush → active layer pixels changed under the stroke only; Ctrl+Z restores, Ctrl+Shift+Z re-applies; second layer + opacity change visible in screenshot.
- gui_model: Model workspace: add cube (Shift+A path or operator), add Bevel modifier, F12-equivalent render (Workbench/Cycles small) → non-empty render result.
- gui_split_live: Split workspace: Make 3D (Extrude) via the panel button/operator invoke on a shape; in the 2D view enter edit mode, drag a node with G → 3D result geometry follows within the same session; screenshot both views.
- gui_linked_undo: after a linked edit, Ctrl+Z / Ctrl+Shift+Z via keymap → source and result consistent each step.
- gui_packaged: `scripts/package.py` tarball → extract to a temp dir → its bootstrap + launcher with a fresh HOME/profile → template loads, SEAM workspaces present, draw→Make 3D works, no Python errors in the console log (scan for "Traceback"/"Error").
Polish from what you see: canvas-first neutral layouts, readable sidebar, consistent panel order (Create → Appearance → Transform → Make 3D/Live Link), clear link status in the header, sensible empty states, no clipped/overlapping UI at 1920×1080 and 1280×800. Fix in ui/ and the template; rebuild startup.blend with build_template.py. Record before/after screenshots in artifacts/gui/.
Report exactly which scenarios ran, their results, and what could not be executed in this environment (e.g. physical tablet pressure).

## Inputs from integration (2026-10-11 02:30)
- Screenshot `artifacts/gui/gui_draw_rect/01_rect_drawn.png` (integrator's checkout) shows: Raster Document / Raster Layers / Raster Selection / Raster Actions panels cluttering the Vector workspace sidebar with "No raster document"; panel order starts with Make 3D, Text, Pathfinder; the Appearance panel was not visible for the selected rectangle. Make panels context-aware (raster panels in Paint workspace or when a RASTER object/doc is active), order Create → Appearance → Transform → Text/Typography → Pathfinder → Make 3D → Live Link → Document, and make sure Appearance shows for SHAPE/TEXT.
- Port the paint worker's GUI probes into scenarios (copy from `/home/user/seam-wt/paint/artifacts/paint/gui-probes/`: gui_probe2 strokes+stencil, gui_probe4 modal selections+fill, gui_probe7 Ctrl+Z stroke undo, gui_probe8 composite display, gui_probe9 undo of fill/clear/layer add in paint mode).
- Connect features to exercise in GUI: linked selection sync (msgbus), seam.result_paint_setup then a native texture-paint stroke on a 3D result, PLANE decal updating after a source colour edit.
- `tests/gui/run_all.sh` currently: 5/5 pass (gui_launch, gui_shell, gui_newfile, gui_fallback, gui_draw_rect). Keep them passing.
- After template/UI changes: `scripts/bootstrap.sh --rebuild-template`, `python3 scripts/build_demos.py`, `python3 scripts/package.py`.
