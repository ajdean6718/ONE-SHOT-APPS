"""SEAM Studio core: pure-Python document, geometry, raster and I/O kernels.

No ``bpy`` imports anywhere in this package. ``numpy`` is allowed everywhere;
``mathutils`` only inside :mod:`seam_core.boolean2d` (imported lazily).
All intra-package imports are relative so the package works both as the
top-level ``seam_core`` (tests) and as ``seam_studio.seam_core`` (add-on).
"""

VERSION = (0, 1, 0)
VERSION_STRING = "0.1.0"
SCHEMA_VERSION = 1

CANVAS_ORIGIN = (0.0, -20000.0, 0.0)
LAYER_STEP = 10.0
Z_STEP = 0.01
