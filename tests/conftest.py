"""Shared pytest setup: make ``seam_core`` (src/) and the add-on (extensions/) importable."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "extensions")):
    if p not in sys.path:
        sys.path.insert(0, p)

ARTIFACTS = os.path.join(ROOT, "artifacts")
