# Task: connect (wave 2) — artwork decals, linked selection, paint-on-result

Owned: `extensions/seam_studio/link/{recipes,producer,ops,ui,lifecycle}.py` (link/props.py: add fields only), `tests/integration/test_connect_*.py`.
Read first: link/producer.py, link/recipes.py, vector/export_png.py (`object_items`, `collect_render_items`), paint/rasterdoc.py (`composite_array`, doc.composite), seam_core.render2d.

Implement:
1. DECAL recipe (brief §8D "Artwork or raster layer to a material/decal, with live texture updates"):
   - Source SHAPE / TEXT / GROUP: rasterize only that artwork (object + group children, with styles, strokes, clips) with `render2d` over its canvas bounds into an 8-bit image "SEAM Decal ‹source›" sized so the longer side = `decal_resolution`; transparent background. Source RASTER: use the raster document composite (share pixels via `composite_array` → image, or reuse doc.composite directly if it is kept current).
   - `decal_mode = PLANE`: the link target is a plane mesh (bounds × units_scale, oriented per `orientation`, UVs 0..1) with a material using the decal image (alpha blended via compat). `decal_mode = MATERIAL`: `seam.make_3d(recipe='DECAL', decal_mode='MATERIAL')` with a selected existing mesh as target adds a SEAM-owned material slot (image over the object's UVs, alpha over a base colour) — the producer must NEVER rewrite that target's mesh; it only updates the image/material.
   - Signature covers source geometry + style + raster revision + params; rebuilds update image pixels in place (no new datablocks per edit). Images packed on save (8-bit).
2. Linked selection: `scene`-independent preference (add-on prefs or a WindowManager prop) "Sync link selection": when the active object is a link source, its results are selected too (and vice versa), without changing the active object. Implement as a pure function `sync_linked_selection(context)` + msgbus subscription on active-object change (GUI); test the function headless.
3. `seam.result_paint_setup`: on a RESULT (or any mesh) ensure "UVMap", add a SEAM paint image + texture-paint slot mixed over the existing base colour, enter Texture Paint (GUI) — native painting on 3D results. Document that painted texels follow the deterministic UVs if the source changes topology.
4. Make 3D panel: DECAL entries enabled with poll messages explaining requirements (e.g. MATERIAL mode needs a selected mesh).
Tests (both runtimes): vector decal pixels at known points have the source fill colour; editing the source colour/geometry updates the same image (same datablock, new pixels) after flush; raster decal follows a layer pixel edit; MATERIAL mode leaves target mesh signature unchanged; save/reopen keeps decal; sync_linked_selection behaviour; paint setup creates slot/UVs.
