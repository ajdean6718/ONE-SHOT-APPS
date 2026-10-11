# Task: fix-paint (wave 3) — raster undo/persistence findings

Owned: `extensions/seam_studio/paint/**`, `tests/integration/test_paint_*.py`, `tests/acceptance/test_s02_*.py`.
Input: `docs/tasks/review-findings-r1.json` (UNVERIFIED reviewer findings with repro steps). Yours are the raster ones:
1. Undo of a SEAM pixel op (Fill/Clear/...) erases native brush strokes painted before it on the same layer; after reopening a file it does not restore pixels at all.
2. Autosave / Recover drops raster layer pixels (recovered layers opaque black) — autosave writes memfile without running save_pre packing.
3. Raster layer property edits made in Texture Paint mode (opacity/blend/visibility) cannot be undone.
4. Every SEAM pixel/layer operator run in Texture Paint mode creates two undo steps; the first Ctrl+Z does nothing.
5. (from paint handoff) Shift+D of a RASTER canvas object binds two objects to one document — make the copy either an independent duplicate document or detach it safely; no silent sharing.
For EACH: reproduce first (headless where possible; GUI under `xvfb-run -a -s "-screen 0 1920x1080x24" blender --enable-event-simulate` with real Ctrl+Z when the bug needs paint mode). If it does not reproduce, say so with evidence and skip. If real: fix with the smallest robust change, add a regression test (integration or GUI script under tests/integration with a subprocess like test_link_gui.py), run paint + acceptance suites on both runtimes.
Native brush strokes interact with Blender's own image undo (paint mode tiles). Design rule: SEAM must never discard pixels it did not write; when unsure, prefer keeping current pixels and reporting over restoring a stale baseline.
