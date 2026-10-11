"""Integration fixtures: real Blender (4.0.2 binary or 5.2.2 bpy module), add-on enabled via addon_utils.

``scripts/test.sh`` sets BLENDER_USER_SCRIPTS to a temp dir whose ``addons/seam_studio``
symlinks to ``extensions/seam_studio``. When run directly with the bpy venv, this conftest
creates that dir itself before importing bpy.
"""

import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_user_scripts():
    scripts = os.environ.get("BLENDER_USER_SCRIPTS")
    if not scripts:
        scripts = tempfile.mkdtemp(prefix="seam-scripts-")
        os.environ["BLENDER_USER_SCRIPTS"] = scripts
        os.environ.setdefault("BLENDER_USER_CONFIG", tempfile.mkdtemp(prefix="seam-config-"))
    addons = os.path.join(scripts, "addons")
    os.makedirs(addons, exist_ok=True)
    link = os.path.join(addons, "seam_studio")
    if not os.path.exists(link):
        os.symlink(os.path.join(ROOT, "extensions", "seam_studio"), link)
    return scripts


_ensure_user_scripts()

import bpy  # noqa: E402
import addon_utils  # noqa: E402


def enable_addon():
    if "seam_studio" in sys.modules and hasattr(bpy.types.Scene, "seam"):
        return sys.modules["seam_studio"]
    mod = addon_utils.enable("seam_studio", default_set=True, handle_error=None)
    if mod is None:
        raise RuntimeError("addon_utils.enable('seam_studio') failed")
    return mod


@pytest.fixture(scope="session")
def seam_addon():
    return enable_addon()


@pytest.fixture
def seam(seam_addon):
    """Fresh empty file with the add-on enabled. Returns the add-on module."""
    bpy.ops.wm.read_homefile(use_empty=True)
    if not hasattr(bpy.types.Scene, "seam"):
        enable_addon()
    from seam_studio.engine import core as engine
    engine.reset()
    return sys.modules["seam_studio"]


@pytest.fixture
def engine(seam):
    from seam_studio.engine import core
    return core


@pytest.fixture
def blend_path(tmp_path):
    return str(tmp_path / "doc.blend")


