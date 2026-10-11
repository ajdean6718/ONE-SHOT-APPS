# SEAM Studio architecture and contracts

Status: authoritative. Owner: integrator (director). Changes go through the integrator.

## 1. Decisions (with reasons)

| # | Decision | Why |
|---|---|---|
| D1 | Blender-native app: one add-on (`seam_studio`) + one application template (`seam`). | Preserves Blender's real modeling, sculpt, UV, texture paint, nodes, render, animation. |
| D2 | Runtime pin: **Blender 5.2.2 LTS** is the target (official `bpy` wheel, sha256 `3dfd1a9e…6d45`, headless-verified). **Blender 4.0.2** (Ubuntu noble pkg `4.0.2+dfsg-1ubuntu8`) is the only interactive build obtainable here and is the GUI-verified runtime. Minimum supported: 4.0. | `download.blender.org`, GitHub and mirrors are egress-blocked (HTTP 403). Not routed around. |
| D3 | Vector shapes are native 2D Curve objects; text is native FONT objects. SEAM adds styling, primitives, pathfinder, layers, artboards, align/distribute, export. | Reuses Blender's mature Bezier node/handle editing, pen/draw tools, snapping, transforms, undo. |
| D4 | Raster painting = native Texture Paint on a canvas plane whose material composites SEAM raster layers (8-bit sRGB images) with Mix nodes; masks are images; selections drive the native stencil mask. Export composites with numpy using the same formulas. | Real pixel painting (brushes, pressure, stabilizer, fill, eyedropper) with layers that Blender lacks natively. Float layers excluded: packed float images do not round-trip (measured 0.25→0.0625). |
| D5 | One scene. The 2D canvas lives under the `SEAM Canvas` empty at `CANVAS_ORIGIN = (0, -20000, 0)`; **1 Blender unit = 1 canvas px**. Artwork collection is `hide_render`. SEAM 3D viewports hide `SEAM Artwork` via local collections. | Split view needs both in one scene; offset keeps the canvas out of 3D views and renders; 1 BU/px keeps float precision ≈0.002 px at the offset. |
| D6 | Live links are data-driven: link records in `scene.seam.links`, targets are mesh objects whose **mesh contents** are regenerated; object transform, modifiers, material slots are never touched. | Stable identity, materials, transforms and compatible modifier stacks. |
| D7 | One engine (`seam_studio/engine/core.py`) owns change detection, dirty keys, signatures, scheduling and build execution for every derived artifact (links, strokes, gradients, raster caches, outlines, clip results). | One depsgraph handler, no competing schedulers, deterministic recomputation. |
| D8 | Undo is unified by derivation: the engine never pushes undo steps; after undo/redo/load it revalidates every key by signature and rebuilds derived data from the restored sources. | Blender's undo restores sources (memfile or edit-mode undo); derived targets follow deterministically. |
| D9 | Native curve evaluation (`Object.to_mesh()` on an unlinked temp curve, verified on 4.0.2 and 5.2.2) builds EXTRUDE / SWEEP / TEXT3D (holes, bevel, caps for free). REVOLVE uses `seam_core.mesh.revolve`. | Mature, fast (<1 ms for simple shapes), identical on both versions. |
| D10 | Downstream-edit protection: target mesh signature stored after each build; mismatch, user data layers (vertex groups, shape keys, extra attributes/UV maps), or target in edit mode ⇒ no overwrite; link goes to `CONFLICT` and the user chooses Overwrite / Bake / Branch. | Never destroy sculpting or topology-dependent edits; never reapply by vertex index. |
| D11 | Booleans use `mathutils.geometry.delaunay_2d_cdt(..., output_type=1, need_ids=True)` and per-triangle ring-membership parity. Result is polygonal (flattened at the chosen resolution). | Native, robust; overlap provenance verified (`[0,1]`) on both versions. Curvature is not preserved (documented gap). |
| D12 | Export (PNG) uses `seam_core.render2d` (numpy rasterizer, supersampled AA) + `seam_core.raster` compositor; SVG via `seam_core.svg`. Deterministic, no GPU. | Testable, identical across versions. |
| D13 | Licensing: SEAM Studio code is GPL-3.0-or-later (Blender add-on norm). Blender is not redistributed. | Blender licence; egress block. |

## 2. Repository ownership (writers)

| Path | Owner |
|---|---|
| `CLAUDE.md`, `docs/*`, `.claude/`, `extensions/seam_studio/__init__.py`, `props.py`, `compat.py`, `engine/`, `blender_manifest.toml`, `tests/conftest.py`, `tests/integration/conftest.py`, `scripts/test.sh`, `scripts/blender_pytest.py` | integrator |
| `src/seam_core/{geometry,shapes,fitting,stroke,mesh,boolean2d,units}.py`, `tests/unit/test_{geometry,shapes,fitting,stroke,mesh,boolean2d,units}*.py` | core-geometry |
| `src/seam_core/{raster,render2d,png,svg,document}.py`, `tests/unit/test_{raster,render2d,png,svg,document}*.py` | core-raster-io |
| `extensions/seam_studio/link/`, `tests/integration/test_link*.py` | link |
| `extensions/seam_studio/vector/`, `tests/integration/test_vector*.py` | vector |
| `extensions/seam_studio/paint/`, `tests/integration/test_paint*.py` | paint |
| `extensions/seam_studio/ui/`, `application_templates/`, `scripts/` (except test.sh, blender_pytest.py), `tests/gui/`, `tests/integration/test_shell_*.py`, `resources/` | shell |

## 3. Coordinate system and units

- `SEAM Canvas` (Empty, role `CANVAS_ROOT`) at `CANVAS_ORIGIN = (0.0, -20000.0, 0.0)`, identity rotation/scale. Every artwork object (roles SHAPE, TEXT, RASTER, ARTBOARD, GUIDE, GROUP) is parented to it (directly or through a GROUP) with identity `matrix_parent_inverse`, so `obj.location` is canvas-local.
- Canvas plane is local XY, viewed from +Z (top ortho). 1 BU = 1 px. Document units (px/pt/mm/cm/in at `doc.dpi`) are display conversions only (`seam_core.units`).
- Artboard object origin = artboard **top-left**; it spans x∈[0,w], y∈[-h,0]. UI shows artboard-relative X right, **Y down** (Illustrator convention): `ui_x = x - ab_x`, `ui_y = ab_y - y`.
- Z-order: `z = layer_index * LAYER_STEP + order * Z_STEP` with `LAYER_STEP = 10.0`, `Z_STEP = 0.01`; artboard backgrounds at `z = -1.0`; derived stroke child at local `z = +Z_STEP*0.5`.
- Layers are collections under `SEAM Artwork` (child of the scene collection, `hide_render=True`). Lock = `collection.hide_select`, visibility = `layer_collection.hide_viewport` / `collection.hide_viewport`.
- 3D results live in collection `SEAM Results`. New results are placed at world `(2.5 * k, 0, 0)` where k = number of existing results (no overlap). Engine-internal temp data goes in nothing (unlinked) or collection `SEAM Internal` (excluded).
- 2D→3D conversion (links): source geometry = source curve local points transformed by the source's canvas-local 2D rotation/scale (translation removed), times `params.units_scale` (m per px, default 0.01). Orientation `UPRIGHT` maps canvas (x, y) → 3D (x, 0, y) with extrusion along −Y→+Y; `FLAT` maps (x, y) → (x, y, 0) with extrusion along +Z. Mirrored (negative-determinant) source transforms are handled by re-normalizing winding (outer CCW, holes CW).

## 4. Data model (Blender side) — `extensions/seam_studio/**/props.py`

`Object.seam` (`SEAM_PG_Object`, `props.py`):
- `id: str` stable 16-hex SEAM id (`ids.ensure_id(obj)`), `role: enum {NONE, CANVAS_ROOT, ARTBOARD, GROUP, SHAPE, TEXT, RASTER, GUIDE, STROKE, OUTLINE, CLIP_RESULT, RESULT, INTERNAL}`
- `owner_id: str` (STROKE/OUTLINE/CLIP_RESULT: seam id of the owning object; RESULT: link id)
- `raster_id: str` (RASTER: id in `scene.seam.rasters`)
- `style: SEAM_PG_Style`, `shape: SEAM_PG_ShapeRecipe`, `artboard: SEAM_PG_Artboard` (vector/props.py)
- `clip_source: Object` (vector clip mask; None = unclipped)

`Scene.seam` (`SEAM_PG_Scene`, `props.py`):
- `schema_version: int` (current `seam_core.SCHEMA_VERSION`), `doc: SEAM_PG_DocSettings` (units, dpi, active artboard id, canvas_ready)
- `links: Collection[SEAM_PG_Link]`, `active_link_index` (link/props.py)
- `rasters: Collection[SEAM_PG_RasterDoc]`, `active_raster_index` (paint/props.py)
- `engine: SEAM_PG_EngineSettings` (live_enabled global switch, debounce_ms, async_threshold_ms, last stats)

`SEAM_PG_Link`: `id, name, recipe {EXTRUDE, REVOLVE, SWEEP, TEXT3D, DECAL}, source: Object, source_id, target: Object, target_id, state {LIVE, PAUSED, DETACHED, ERROR, CONFLICT, MISSING_SOURCE}, stale: bool, message, params: SEAM_PG_LinkParams, source_signature, target_signature, target_data_signature (UVMap + face materials), force_overwrite, decal_image: Image, build_count, last_build_ms`. DECAL semantics: `link/decal.py` docstring (PLANE target = generated plane mesh; MATERIAL target = existing user mesh whose geometry is never rewritten, only the SEAM material slot/image).
`SEAM_PG_LinkParams`: `depth, bevel_depth, bevel_resolution, resolution, orientation {UPRIGHT, FLAT}, origin {CENTER, BOTTOM, SOURCE, MIN}, units_scale, revolve_angle, revolve_segments, revolve_axis {LEFT, CENTER, SOURCE_ORIGIN}, axis_offset, sweep_radius, sweep_profile {ROUND, SQUARE, RIBBON}, sweep_profile_resolution, fill_caps, decal_mode {MATERIAL, PLANE}, decal_resolution, use_source_material`.

`SEAM_PG_RasterDoc`: `id, name, width, height, layers: Collection[SEAM_PG_RasterLayer], active_layer_index, canvas: Object, material: Material, composite: Image (cache), selection: Image (mask), has_selection, background {TRANSPARENT, WHITE, COLOR}, background_color, edit_target {LAYER, MASK}`.
`SEAM_PG_RasterLayer`: `id, name, image: Image, opacity, blend_mode (seam_core.raster.BLEND_MODES), visible, locked, mask: Image, mask_enabled`.

`SEAM_PG_Style`: `fill_enabled, fill_type {SOLID, LINEAR, RADIAL}, fill_color (scene-linear RGBA), gradient_stops: Collection[{position, color}], gradient_angle, gradient_center (bbox-relative), gradient_radius, stroke_enabled, stroke_color, stroke_width (px), stroke_cap {BUTT, ROUND, SQUARE}, stroke_join {MITER, ROUND, BEVEL}, miter_limit, opacity, fill_rule {EVENODD, NONZERO}`.
`SEAM_PG_ShapeRecipe`: `kind {NONE, RECT, ROUNDED_RECT, ELLIPSE, POLYGON, STAR, LINE}, width, height, radius, sides, inner_ratio` (`kind != NONE` ⇒ parameters own the geometry; manual node edits set `kind = NONE`).
`SEAM_PG_Artboard`: `width, height, background_enabled, background_color, export_scale`.

Single-owner rules: curve data owns path geometry (or the shape recipe while parametric); `style` owns appearance — materials, stroke meshes and gradient textures are derived; the link record owns recipe parameters; the target owns its transform/modifiers/material slots; raster layer images own pixels; composite caches are derived.

Colours: all style colours are scene-linear RGBA (`subtype='COLOR'`). Raster layers are 8-bit sRGB (straight alpha). Compositing happens in scene-linear (sRGB→linear, blend, →sRGB) in both the node tree and `seam_core.raster`, using the W3C Compositing Level 1 source-over model with Blender Mix-node blend formulas (backdrop-alpha aware; exact Porter–Duff for Normal). SEAM sets view transform `Standard`.

## 5. Engine contract — `extensions/seam_studio/engine/core.py`

```
class Producer:                       # subclass per derived kind
    kind: str                         # "link", "stroke", "gradient", "rastercache", "outline", "clip", "decal"
    def keys_for_update(self, scene, id_orig, geometry: bool, transform: bool) -> Iterable[str]
    def all_keys(self, scene) -> Iterable[str]
    def signature(self, scene, key) -> str | None      # None => key obsolete (drop)
    def stored_signature(self, scene, key) -> str | None
    def build(self, scene, key, signature) -> BuildResult   # main thread; must store signature on success
    # optional async split: prepare(scene,key)->snapshot (main), compute(snapshot)->result (pure, any thread), apply(scene,key,signature,result)->BuildResult (main)
engine.register_producer(p) / unregister_producer(kind)
engine.mark_dirty(key) ; engine.revalidate_all(scene) ; engine.flush(scene=None) -> list[BuildResult]
engine.suspend()  (context manager)  ; engine.stats() -> dict
BuildResult(key, status in {"ok","skipped","error","conflict","waiting"}, message="", ms=0.0)
```
Keys are `"<kind>:<id>"`. The engine installs one `depsgraph_update_post`, `undo_post`, `redo_post`, `load_post`, `save_pre` handler set. GUI: builds run in a `bpy.app.timers` tick (debounce from `scene.seam.engine.debounce_ms`, adaptive when builds are slow). Background/tests: call `engine.flush()`. Builds skip when `signature == stored_signature` (idempotent). Stale async results (signature changed meanwhile) are discarded.

## 6. seam_core contract

Signatures and docstrings in `src/seam_core/*.py` are the contract. Rings are lists of `(x, y)` tuples without a repeated closing point. Normalized shapes are `list[(outer_ccw, [holes_cw])]`. Mesh data is `seam_core.mesh.MeshData`. Images are numpy `float32` arrays `H×W×4`, row 0 = **top**, straight (non-premultiplied) alpha; linear unless the name says `srgb`.

## 7. Operator / UI naming

Operators `seam.<verb_noun>` (classes `SEAM_OT_*`), panels `SEAM_PT_*` in sidebar category `SEAM`, menus `SEAM_MT_*`, lists `SEAM_UL_*`, tools `seam.tool_*`. Every module exposes `classes = (...)` and optional `register()`/`unregister()`; `extensions/seam_studio/__init__.py` registers modules in a fixed order. Keymaps only inside SEAM tool keymaps or the `SEAM` addon keymap with documented, non-conflicting combos.

## 8. Workspaces

`Vector` (top-ortho canvas, Solid/Flat), `Paint` (canvas in Texture Paint, Material Preview + Image Editor), `Model` (perspective 3D, artwork hidden), `Split` (2D canvas left, 3D right), followed by Blender's native workspaces (Layout, Modeling, Sculpting, UV Editing, Texture Paint, Shading, Animation, Rendering, Compositing, Geometry Nodes, Scripting).

## 9. Packaging targets

- `seam_studio-<ver>.zip` — legacy add-on zip (bl_info, Blender 4.0+; also installable on 4.2+ via "Install from Disk").
- The same zip carries `blender_manifest.toml` (`blender_version_min = 4.2.0`) so it can be installed as an extension on 4.2+; the extension platform does not exist before 4.2.
- `seam_app_template-<ver>.zip` — application template (`startup.blend` is generated by `scripts/build_template.py`, not committed).
- `seam-studio-<ver>-linux-x86_64.tar.gz` — launcher, bootstrap, add-on, template, demos, docs, notices. Blender itself is not bundled.
