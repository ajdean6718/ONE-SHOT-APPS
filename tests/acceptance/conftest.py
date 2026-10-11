"""Acceptance suite: reuse the integration fixtures (real Blender, add-on enabled via addon_utils)."""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_INTEGRATION = os.path.join(_HERE, "..", "integration")
if _INTEGRATION not in sys.path:
    sys.path.insert(0, _INTEGRATION)  # `from helpers import ...`

_spec = importlib.util.spec_from_file_location("seam_integration_conftest", os.path.join(_INTEGRATION, "conftest.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

enable_addon = _mod.enable_addon
seam_addon = _mod.seam_addon
seam = _mod.seam
engine = _mod.engine
blend_path = _mod.blend_path
