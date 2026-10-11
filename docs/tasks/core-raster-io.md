# Task: core-raster-io (pure raster, export renderer, PNG, SVG, document snapshot)

Implement every `NotImplementedError` in `src/seam_core/{raster,render2d,png,svg,document}.py` per docstrings. Keep signatures; add helpers as needed. You may use `seam_core.geometry` dataclasses; geometry *kernels* (flatten, triangulate, ...) are being written in parallel by another worker — do not depend on them: write your own small private Bezier flattening inside render2d/svg if needed.

Owned: those files, `tests/unit/test_{raster,render2d,png,svg,document}.py`, `tests/integration/test_core_blend_parity.py`, `tests/integration/test_core_png_blender.py`, `resources/fixtures/svg/` (small ORIGINAL fixtures you write).

Acceptance:
- `scripts/test.sh unit` green on both pythons.
- png: exact encode/decode round trip (RGBA/RGB); decodes all filter types; integration test: Blender saves a PNG (`image.filepath_raw` + `file_format='PNG'` + `save()`), `decode_png` reads the same pixels (both runtimes).
- raster: sRGB↔linear inverse within 1e-6; every BLEND_MODE implemented with Blender Mix-node semantics; integration parity test renders Blender's Mix node (emission plane, orthographic camera, Cycles 1–4 samples, view transform Standard, small resolution) for every mode and several colour pairs and compares to `blend_rgb` within 2/255 in sRGB on both runtimes; composite() matches hand-computed NORMAL alpha-over; masks/opacity multiply factor; selections: rect/ellipse/polygon mask areas within 1% of analytic; combine ops; flood fill contiguous vs global; Blender pixel order helpers round-trip.
- render2d: axis-aligned square coverage exact (±0.5%), even-odd hole vs nonzero star differ correctly, linear/radial gradient endpoint colours exact, clip works, ImageItem placement pixel-exact for integer placement; 1920×1080 with 20 shapes at supersample 4 < 5 s (record timing in `artifacts/perf/core-raster-io.json`).
- svg: parser handles M/L/H/V/C/S/Q/T/A/Z abs+rel, implicit repeats, exponents, comma/space variants; arc flattening matches analytic circle within 0.5 px; transforms; style attribute; export→import round trip preserves geometry (1e-3) and fills/strokes/opacity/gradients; clipPath export; unsupported elements (filter, mask, pattern, use, text import) produce warnings, never silent drops.
- document: to_dict/from_dict round trip; migrate 0→1; validate catches duplicate ids, dangling link ids, bad recipe/state/blend mode, non-positive sizes; newer schema raises ValueError.
