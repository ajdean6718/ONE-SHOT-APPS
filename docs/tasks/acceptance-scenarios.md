# Acceptance scenarios (from the product brief) — executable mapping

Each scenario must be exercised through the operators a user actually invokes (`bpy.ops.seam.*` or native ops),
on both runtimes where headless is possible, and in the GUI suite where interaction matters.

| # | Scenario | Headless test | GUI scenario |
|---|---|---|---|
| 1 | Vector artwork: create shapes/text, edit nodes + typography, save/reopen, export SVG+PNG, no 3D generated | tests/acceptance/test_s01_vector.py | gui_vector_draw.py (rect tool drag, node drag in edit mode, text typing) |
| 2 | Raster: paint on layers, erase, opacity/masks, undo/redo, reopen with editable layers | tests/acceptance/test_s02_raster.py | gui_paint_stroke.py (native brush stroke drag, Ctrl+Z/Ctrl+Shift+Z) |
| 3 | Independent native 3D object + modifier + material + real render | tests/acceptance/test_s03_native3d.py (Cycles CPU + Workbench render to PNG, non-empty pixels) | gui_model_workspace.py |
| 4 | Extrude compound shape with hole; verify scale, shape, fill, geometry | tests/acceptance/test_s04_extrude_hole.py | — |
| 5 | Move source nodes → only linked targets update; identity/materials/modifiers persist | tests/acceptance/test_s05_live_update.py | gui_split_live_update.py (edit-mode G drag in 2D view, 3D result updates, screenshot) |
| 6 | Two different results from one source with independent params | tests/acceptance/test_s06_two_results.py | — |
| 7 | Pause → edit → resume → correct; detach → later edits ignored | tests/acceptance/test_s07_pause_detach.py | — |
| 8 | Undo/redo of a linked edit is atomic across drawing and model | tests/acceptance/test_s08_undo.py | gui_linked_undo.py (Ctrl+Z / Ctrl+Shift+Z through the real keymap) |
| 9 | Save, close, reopen, continue linked editing without recreating the link | tests/acceptance/test_s09_reopen.py | — |
| 10 | Deletion, duplication, invalid geometry, stale updates, missing assets, interrupted work: no silent data loss | tests/acceptance/test_s10_robustness.py | — |
| 11 | Packaged build from a clean profile completes the main workflow with no console | scripts/package.py + tests/gui/gui_packaged_clean_profile.py | yes |

Performance: `scripts/bench.py` → `artifacts/perf/summary.json` (hardware, document size, timings for simple-shape link updates,
200-pt shape, text, revolve; drawing-tool latency where measurable).
