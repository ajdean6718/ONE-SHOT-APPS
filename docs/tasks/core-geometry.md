# Task: core-geometry (pure kernels)

Implement every `NotImplementedError` in `src/seam_core/{geometry,shapes,fitting,stroke,mesh,boolean2d}.py` per docstrings. Keep signatures; you may add helpers. `mesh_signature` already exists.

Owned: those files, `tests/unit/test_{geometry,shapes,fitting,stroke,mesh,boolean2d}.py`, `tests/integration/test_core_boolean_native.py`, `artifacts/perf/core-geometry.json`.

Acceptance (all must be shown passing in your handoff):
- `scripts/test.sh unit` green on BOTH pythons (3.13 venv and /usr/bin/python3.12). Boolean unit tests `pytest.importorskip("mathutils")` (only the venv and Blender have it); the integration file runs booleans inside both Blender runtimes.
- triangulate: square with square hole → triangle areas sum to outer−hole (1e-9); collinear/duplicate points; CW input; touching hole; 2k-vertex polygon with 20 holes < 300 ms.
- extrude (shape with hole): closed 2-manifold (every edge used by exactly 2 faces), Euler V−E+F = 0, outward orientation (signed volume = area×depth within 1e-6), materials caps=0 walls=1, UV count = loop count.
- revolve: offset rectangle profile at 64 segments → volume within 2% of 2π·R·A; on-axis points merged, no degenerate faces; partial angle caps.
- sweep: ROUND tube volume within 3% of πr²L (16 segments).
- stroke: BUTT area = w·L; SQUARE = w·(L+w); ROUND ≈ w·L+π(w/2)² (1%); MITER/BEVEL/ROUND joins on a right angle have the analytic extra area within 2%; closed paths have no caps.
- boolean2d (via mathutils CDT, see module docstring): union/difference/intersect/exclude of overlapping squares exact areas; holes preserved; disjoint, identical, touching-edge, nested cases; 2×500-pt rings < 300 ms.
- fit_bezier: flattened fit stays within 1.5×max_error of the input samples; corners preserved.
- self_intersections: finds figure-eight crossing; 5k-pt simple ring < 200 ms. validate_rings reports SELF_INTERSECTION / ZERO_AREA / TOO_FEW_POINTS.
- NaN/inf input → ValueError, never a hang.
- Record timings in `artifacts/perf/core-geometry.json` (machine: 4 vCPU, Python version).
