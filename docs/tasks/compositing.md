# Task: compositing (wave 2) — correct alpha compositing for raster layers

Problem: `seam_core.raster.composite` (and the paint material that mirrors it) mixes colour as
`rgb = mix(prev_rgb, blend(prev_rgb, layer_rgb), f)`, ignoring the backdrop alpha. Over a semi-transparent backdrop
this is not alpha-over (e.g. 50% red backdrop + 50% blue layer gives 0.5/0.5 instead of 1/3 red, 2/3 blue), which is
visible at anti-aliased edges and with semi-transparent strokes on transparent layers.

Adopt the W3C Compositing & Blending Level 1 model ("source-over" with separable/non-separable blend modes):
  Cs' = (1 − αb)·Cs + αb·B(Cb, Cs)          (B = Blender Mix-node formula for the mode, Cb backdrop, Cs layer colour)
  αs  = layer_alpha × opacity × mask
  αo  = αs + αb·(1 − αs)
  Co  = (αs·Cs' + αb·Cb·(1 − αs)) / αo      (αo = 0 → Co = 0)
in scene-linear, straight alpha. For NORMAL this is exact Porter–Duff over; for opaque backdrops it equals the current behaviour.

Owned: `src/seam_core/raster.py` (`composite` + helpers; keep `blend_rgb` Blender-exact), `extensions/seam_studio/paint/rasterdoc.py`
(material node builder only — mirror the formula with Math/Mix nodes so viewport = export), affected tests in `tests/unit/test_raster*.py`,
`tests/integration/test_paint_*.py`, `tests/integration/test_core_blend_parity.py`; update the docstring in raster.py.
Acceptance (both runtimes): unit tests for the half-transparent red/blue case and random property tests against a float64 reference of the
formula; opaque-backdrop results unchanged vs the old implementation; the paint material parity test (Cycles render of the canvas material vs
`composite_array`) passes within 2/255 for NORMAL, MULTIPLY, SCREEN, OVERLAY, ADD with semi-transparent layers; all paint/raster tests green.
