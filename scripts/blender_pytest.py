"""Run pytest inside a Blender binary: blender -b --factory-startup --python scripts/blender_pytest.py -- <pytest args>."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["tests/integration"]
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import pytest  # noqa: E402

code = pytest.main(["-p", "no:cacheprovider"] + args)
sys.stdout.flush()
os._exit(int(code))
