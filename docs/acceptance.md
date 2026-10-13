# SEAM Studio — acceptance status

Every capability from the brief is placed in exactly one category:

- **Tested** — implemented here and exercised by automated tests that drive real operators.
- **Native+tested** — Blender's own implementation, reachable inside SEAM and exercised by an integration/acceptance/GUI test.
- **Native, accessible** — Blender's own implementation, reachable in the delivered app (workspace/tool present, verified by the GUI launch scenario) but not individually exercised here.
- **Unverified** — implemented, but this environment cannot verify it (no Blender 5.x GUI, no tablet, no other OS).
- **Gap** — missing, partial or blocked; the limitation is stated exactly.

Runtimes: Blender **4.0.2** (Ubuntu package; GUI under Xvfb + headless) and official **bpy 5.2.2 LTS** (headless only — no
5.x GUI binary is obtainable because download.blender.org/GitHub are blocked by this environment's egress policy).
Evidence paths are relative to the repository root. Test counts and the commit they were measured on: see the bottom.

## 1. Acceptance scenarios

| # | Scenario | Status | Evidence |
|---|---|---|---|
| 1 | Vector artwork: create, edit nodes + typography, save/reopen, export SVG/PNG, no 3D | Tested (both runtimes) + GUI (4.0.2) | `tests/acceptance/test_s01_vector.py`; `tests/gui/gui_vector.py`, `gui_draw_rect.py` |
| 2 | Raster layers: paint, erase, opacity/masks, undo/redo, reopen | Tested (SEAM ops both runtimes; native brush strokes + real Ctrl+Z in 4.0.2 GUI) | `tests/acceptance/test_s02_raster.py`; `tests/gui/gui_paint*.py`; `tests/integration/test_paint_native_undo.py` |
| 3 | Independent native 3D object + modifier + material + real render | Native+tested (Cycles CPU + Workbench renders) | `tests/acceptance/test_s03_native3d.py`; `tests/gui/gui_model.py` |
| 4 | Extrude compound shape with hole: scale, shape, fill, geometry | Tested | `tests/acceptance/test_s04_extrude_hole.py` (analytic area/volume, closed 2-manifold, Euler χ=0) |
| 5 | Move source nodes → only linked targets update; identity/materials/modifiers persist | Tested + GUI | `tests/acceptance/test_s05_live_update.py`; `tests/gui/gui_split_live.py` |
| 6 | Two results from one source, independent parameters | Tested | `tests/acceptance/test_s06_two_results.py` |
| 7 | Pause/edit/resume; detach then later edits ignored | Tested | `tests/acceptance/test_s07_pause_detach.py` |
| 8 | Undo/redo of a linked edit atomic across drawing and model | Tested (headless both runtimes) + GUI real Ctrl+Z (4.0.2) | `tests/acceptance/test_s08_undo.py`; `tests/gui/gui_linked_undo.py` |
| 9 | Save, close, reopen, continue linked editing | Tested (incl. fresh process) | `tests/acceptance/test_s09_reopen.py` |
| 10 | Deletion, duplication, invalid geometry, stale updates, missing assets, interrupted work | Tested | `tests/acceptance/test_s10_robustness.py`, `tests/integration/test_link_lifecycle.py`, `test_link_review.py`, `test_paint_autosave.py`, `test_paint_duplicate.py` |
| 11 | Packaged build from a clean profile, main workflow, no console errors | See §6 (GUI `gui_packaged`) | `scripts/package.py`, `tests/gui/gui_packaged.py` |

## 2. Vector studio (brief §8A)

| Capability | Status | Notes / evidence |
|---|---|---|
| Artboards, pan/zoom | Tested (artboards) / Native (pan, zoom) | `test_vector_canvas.py`; 2D views are top-ortho 3D viewports |
| Selection, multi-selection | Native+tested | used throughout `test_vector_arrange.py` |
| Groups | Tested | `seam.group/ungroup`, world matrices preserved (`test_vector_arrange.py`) |
| Layers, visibility, locking | Tested | `seam.layer_*`; collections under "SEAM Artwork" |
| Z-order | Tested | `seam.arrange` |
| Snapping | Native, accessible | Blender snapping in the canvas view |
| Guides | Tested | `seam.guide_add/guides_clear` |
| Align, distribute | Tested | `seam.align`, `seam.distribute` |
| Numeric transforms | Tested | Transform panel: artboard-relative X/Y (Y down), W/H, rotation |
| Pen / Bezier drawing, editable nodes and handles | Native+tested | Blender curve edit mode (Draw/Pen tools, handles); node edits driven in tests and GUI |
| Corner/smooth conversion, add/delete/join/split nodes, open/closed paths | Native, accessible | Blender curve edit operators (V, X, F, Alt+C, subdivide) |
| Rectangle, rounded rectangle, ellipse, polygon, star, line | Tested | `seam.add_shape`, drag tools `seam.tool_*`; parametric until edited by hand |
| Freehand strokes with smoothing | Tested | `seam.draw_freehand` + Bezier fit |
| Compound paths and holes | Tested | `seam.compound_make/release`; even-odd fill |
| Boolean union / subtract / intersect / exclude | Tested | `seam.pathfinder` (native CDT). **Gap:** results are polygonal; curvature is not preserved |
| Fill / stroke, caps / joins, opacity | Tested | Appearance panel; stroke meshes |
| Linear / radial gradients | Tested | baked gradient images; exact in Material Preview/Render/export. **Gap:** Solid view shows the midpoint colour |
| Clipping / masking | Tested | `seam.clip_make/release`, honoured in PNG and SVG export |
| Duplication | Native+tested | Shift+D handled (ids, derived data) |
| Text editing, typography controls | Tested | FONT objects; Typography panel (font, size, spacing, alignment, styles, text box). Bold/italic buttons on a selection are not exercised on 4.0.2 headless (font edit-mode selection crashes there) |
| Non-destructive text-to-outline | Tested | live outline follows text; expand/revert |
| SVG import/export (explicit subset) | Tested | `seam_core/svg.py` docstring lists the subset; unsupported elements produce warnings |
| PNG export | Tested | deterministic numpy renderer |
| Full SVG fidelity, Adobe formats | Gap | not claimed |

## 3. Raster painting (brief §8B)

| Capability | Status | Notes / evidence |
|---|---|---|
| Brush, eraser, size/hardness/opacity, colour, eyedropper | Native+tested (4.0.2 GUI strokes) | Blender Texture Paint brushes on the canvas plane |
| Fill | Tested | `seam.raster_fill` (selection or layer) |
| Selections (rect / ellipse / lasso, all/none/invert) | Tested | stencil-mask confinement verified in 4.0.2 GUI; **Unverified** on 5.2 GUI |
| Layers, masks, opacity, blend modes | Tested | W3C source-over compositing with Blender Mix-node formulas; viewport material ≡ export (parity tests) |
| Undoable strokes | Tested (4.0.2 GUI, real Ctrl+Z) | SEAM pixel ops never discard native strokes (regression tests) |
| Pressure sensitivity, stabilization | Native, accessible / **Unverified** | Blender tablet pressure + stroke stabilizer; no tablet here |
| PNG create/save/export, resolution, alpha, colour space | Tested | 8-bit sRGB, straight alpha, compositing in scene-linear |
| Editable layers preserved in the project | Tested | packed 8-bit images; save/reopen, autosave/recover |
| 16/32-bit float layers | Gap | Blender does not round-trip packed float images (measured 0.25→0.0625) |
| Merge-down with non-Normal blend modes | Gap (approximate) | operator measures and warns |
| Large canvases (2048² × 4 layers) | Gap (slow) | composite ~3–9 s per edit; needs dirty-rect compositing (docs/perf.md) |

## 4. Native 3D studio (brief §8C)

| Capability | Status |
|---|---|
| Mesh editing, modifiers, materials, rendering (Cycles/Workbench) | Native+tested (`test_s03_native3d.py`, `gui_model.py`) |
| Curves, sculpting, UV editing, texture painting, lighting, cameras, animation, rigging, simulation, Geometry Nodes | Native, accessible (Layout, Modeling, Sculpting, UV Editing, Texture Paint, Shading, Animation, Rendering, Compositing, Geometry Nodes, Scripting workspaces present — `gui_launch.py`) |
| Sculpt/texture paint on a live result without losing work | Tested (conflict protection; `seam.result_paint_setup`) |

## 5. Connected creation (brief §7, §8D)

| Capability | Status | Notes |
|---|---|---|
| Closed shape → extrude (holes, bevel) | Tested | native curve evaluation |
| Profile → revolve | Tested | `seam_core.mesh.revolve` |
| Open path → sweep (width/profile) | Tested | ROUND / SQUARE / RIBBON |
| Text → 3D lettering | Tested | |
| Artwork / raster → material or decal, live texture | Tested | PLANE and MATERIAL decals; raster decals share the composite |
| Depth, bevel, resolution, orientation, origin, units, material controls | Tested | `units_scale` m/px |
| Live / paused / detached / error / conflict / missing-source states with UI indicators | Tested | Live Link panel + header status |
| Stable identity, materials, transforms, compatible modifiers across updates | Tested | |
| Dirty tracking, coalescing, deterministic recompute, cycle prevention | Tested | engine signatures; results can never be sources |
| Cancellation of stale calculations | Partial | builds are synchronous and coalesced (newest state wins); the engine's async path exists but no producer uses it |
| Unified undo/redo (one gesture ≠ many steps) | Tested | derivation-based (D8) |
| Save/reopen with links intact; relink; missing-source recovery; safe duplication | Tested | |
| No destruction of downstream sculpt/topology edits | Tested | CONFLICT → Overwrite / Bake / Branch |
| Reverse updates | Partial | generation parameters are editable from the result; editing result geometry does not flow back to the drawing (by design: conflict protection) |
| Split view, linked selection, Edit Source / Show Result, pause/resume, bake, duplicate result | Tested (+ GUI in 4.0.2) | |
| Offline, no accounts/AI dependency | Tested by construction | no network code |

## 6. Interface, packaging, platforms

| Item | Status |
|---|---|
| Vector / Paint / Model / Split workspaces, native workspaces after them | Tested in 4.0.2 GUI (`gui_launch.py`); **Unverified** on a 5.x GUI |
| Command search (F3), discoverable scoped shortcuts | Native search; SEAM keymaps conflict-checked (`test_shell_ui.py`) |
| Empty states, disabled-control explanations, health/missing-asset report, autosave/recovery | Tested (`test_shell_health.py`, `test_paint_autosave.py`) |
| Packaged build (add-on zip, app-template zip, Linux tarball, SHA256SUMS) | Tested (`test_shell_package.py`) |
| Clean-profile packaged launch + main workflow | `tests/gui/gui_packaged.py` — result recorded below |
| Blender redistribution | Not bundled (egress policy; licence notes in THIRD_PARTY_NOTICES.md) |
| Windows / macOS | **Unverified** (only Linux x86_64 here) |
| Physical pen/tablet | **Unverified** |

## Measured results

(stamped by the integrator after the final run — see below)
