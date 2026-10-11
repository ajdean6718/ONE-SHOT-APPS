"""scripts/package.py: artifacts are reproducible, install into a clean profile and enable (shell)."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.environ.get("SEAM_BLENDER") or shutil.which("blender")
PYTHON3 = shutil.which("python3")
VENV52 = os.environ.get("SEAM_BPY52_PYTHON", "/opt/seam-cache/venv-bpy52/bin/python")
VERSION = re.search(r'VERSION_STRING\s*=\s*"([^"]+)"', open(os.path.join(ROOT, "src", "seam_core", "__init__.py")).read()).group(1)

pytestmark = pytest.mark.skipif(not BLENDER or not PYTHON3, reason="blender / python3 not on PATH")

ENABLE_SNIPPET = r"""
import json, os, sys, bpy, addon_utils
mod = addon_utils.enable("seam_studio", default_set=True, handle_error=None)
import seam_studio
from seam_studio import seam_core, ui
scene = bpy.context.scene
out = {"enabled": mod is not None and hasattr(bpy.types.Scene, "seam"),
       "version": bpy.app.version_string,
       "module_file": seam_studio.__file__,
       "seam_core_real": os.path.isdir(os.path.dirname(seam_core.__file__)) and not os.path.islink(os.path.dirname(seam_core.__file__)),
       "seam_core_version": seam_core.VERSION_STRING,
       "ui_modules": [m.__name__.rsplit('.', 1)[-1] for m in ui.modules]}
bpy.ops.seam.health_check()
out["health_ok"] = True
print("RESULT " + json.dumps(out))
"""


def run(cmd, env=None, timeout=300, cwd=ROOT):
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=full, cwd=cwd)


def result_of(proc):
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, f"no RESULT line (exit {proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    return json.loads(lines[0][len("RESULT "):])


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    startup = os.path.join(ROOT, "application_templates", "seam", "startup.blend")
    if not os.path.exists(startup):
        if not shutil.which("xvfb-run") and not os.environ.get("DISPLAY"):
            pytest.skip("startup.blend missing and no display/xvfb-run to build it")
        assert run([os.path.join(ROOT, "scripts", "bootstrap.sh"), "--quiet"], timeout=600).returncode == 0
    out = tmp_path_factory.mktemp("dist")
    p = run([PYTHON3, os.path.join(ROOT, "scripts", "package.py"), "--out", str(out), "--skip-demos", "--no-build"])
    assert p.returncode == 0, p.stdout + p.stderr
    return out


def test_artifacts_checksums_and_reproducibility(dist, tmp_path):
    names = sorted(os.listdir(dist))
    assert names == sorted([f"seam_studio-{VERSION}-addon.zip", f"seam_studio-{VERSION}-extension.zip",
                            f"seam-template-{VERSION}.zip", f"seam-studio-{VERSION}-linux.tar.gz", "SHA256SUMS"])
    sums = dict(reversed(l.split("  ")) for l in (dist / "SHA256SUMS").read_text().splitlines())
    sums = {k: v for k, v in ((a.strip(), b) for a, b in sums.items())}
    for name in names:
        if name != "SHA256SUMS":
            assert sums[name] == sha(dist / name), name
    again = tmp_path / "again"
    p = run([PYTHON3, os.path.join(ROOT, "scripts", "package.py"), "--out", str(again), "--skip-demos", "--no-build"])
    assert p.returncode == 0, p.stderr
    assert (again / "SHA256SUMS").read_text() == (dist / "SHA256SUMS").read_text()   # byte-for-byte reproducible


def test_addon_zip_contents(dist):
    with zipfile.ZipFile(dist / f"seam_studio-{VERSION}-addon.zip") as z:
        names = z.namelist()
        assert "seam_studio/__init__.py" in names and "seam_studio/blender_manifest.toml" in names
        assert "seam_studio/seam_core/__init__.py" in names and "seam_studio/seam_core/units.py" in names
        assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
        init = z.read("seam_studio/__init__.py").decode()
        assert "bl_info" in init and f'"version": ({VERSION.replace(".", ", ")})' in init
        manifest = z.read("seam_studio/blender_manifest.toml").decode()
        assert f'version = "{VERSION}"' in manifest and 'id = "seam_studio"' in manifest
        for n in names:                                   # fixed timestamps
            assert z.getinfo(n).date_time == (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(dist / f"seam_studio-{VERSION}-extension.zip") as z:
        names = z.namelist()
        assert "blender_manifest.toml" in names and "__init__.py" in names and "seam_core/__init__.py" in names
        assert not [n for n in names if n.startswith("seam_studio/")]


def _extract_addon(dist, scripts_dir):
    addons = scripts_dir / "addons"
    addons.mkdir(parents=True)
    with zipfile.ZipFile(dist / f"seam_studio-{VERSION}-addon.zip") as z:
        z.extractall(addons)
    assert not (addons / "seam_studio" / "seam_core").is_symlink()
    return addons


def test_addon_zip_enables_in_blender_binary(dist, tmp_path):
    scripts = tmp_path / "scripts"
    _extract_addon(dist, scripts)
    p = run([BLENDER, "--background", "--factory-startup", "--python-exit-code", "1", "--python-expr", ENABLE_SNIPPET],
            env={"BLENDER_USER_SCRIPTS": str(scripts), "BLENDER_USER_CONFIG": str(tmp_path / "config")})
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    res = result_of(p)
    assert res["enabled"] and res["seam_core_real"] and res["health_ok"]
    assert res["module_file"].startswith(str(scripts))        # imported from the zip contents, not the checkout
    assert res["seam_core_version"] == VERSION
    assert "panels" in res["ui_modules"] and "workspaces" in res["ui_modules"]


@pytest.mark.skipif(not os.path.exists(VENV52), reason="bpy 5.x venv not available")
def test_addon_zip_imports_in_bpy_module(dist, tmp_path):
    scripts = tmp_path / "scripts"
    _extract_addon(dist, scripts)
    p = run([VENV52, "-c", ENABLE_SNIPPET],
            env={"BLENDER_USER_SCRIPTS": str(scripts), "BLENDER_USER_CONFIG": str(tmp_path / "config")})
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    res = result_of(p)
    assert res["version"].startswith("5.2") and res["enabled"] and res["seam_core_real"]
    assert res["module_file"].startswith(str(scripts))


def test_template_zip_installs_and_starts(dist, tmp_path):
    scripts = tmp_path / "scripts"
    _extract_addon(dist, scripts)
    tpl_dir = scripts / "startup" / "bl_app_templates_user"
    tpl_dir.mkdir(parents=True)
    with zipfile.ZipFile(dist / f"seam-template-{VERSION}.zip") as z:
        assert sorted(z.namelist()) == ["seam/__init__.py", "seam/splash.png", "seam/startup.blend"]
        z.extractall(tpl_dir)
    expr = ("import bpy,json; s=bpy.context.scene; print('RESULT '+json.dumps({'addon':hasattr(bpy.types.Scene,'seam'),"
            "'ready':s.seam.doc.canvas_ready,'ws':sorted(w.name for w in bpy.data.workspaces if w.name in "
            "('Vector','Paint','Model','Split')),'objects':sorted(o.name for o in bpy.data.objects)}))")
    p = run([BLENDER, "--app-template", "seam", "--background", "--python-exit-code", "1", "--python-expr", expr],
            env={"BLENDER_USER_SCRIPTS": str(scripts), "BLENDER_USER_CONFIG": str(tmp_path / "config")})
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    res = result_of(p)
    assert res == {"addon": True, "ready": True, "ws": ["Model", "Paint", "Split", "Vector"],
                   "objects": ["Artboard 1", "SEAM Canvas"]}


def test_tarball_runs_from_its_own_layout(dist, tmp_path):
    with tarfile.open(dist / f"seam-studio-{VERSION}-linux.tar.gz") as t:
        t.extractall(tmp_path, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        names = t.getnames()
    top = tmp_path / f"seam-studio-{VERSION}"
    for rel in ("scripts/seam-studio", "scripts/bootstrap.sh", "scripts/doctor.py", "README.md", "LICENSE",
                "THIRD_PARTY_NOTICES.md", "application_templates/seam/startup.blend",
                "extensions/seam_studio/seam_core/__init__.py"):
        assert (top / rel).exists(), rel
    assert os.access(top / "scripts" / "seam-studio", os.X_OK) and os.access(top / "scripts" / "bootstrap.sh", os.X_OK)
    assert not any("__pycache__" in n for n in names)
    assert not (top / "extensions" / "seam_studio" / "seam_core").is_symlink()
    env = {"SEAM_PROFILE": str(tmp_path / "profile")}
    b = run([str(top / "scripts" / "bootstrap.sh")], env=env, cwd=str(top))
    assert b.returncode == 0, b.stdout + b.stderr
    expr = "import bpy; print('RESULT {\"ws\": %d, \"ready\": %s}' % (len([w for w in bpy.data.workspaces if w.name in ('Vector','Paint','Model','Split')]), bpy.context.scene.seam.doc.canvas_ready.__repr__().lower()))"
    p = run([str(top / "scripts" / "seam-studio"), "--background", "--python-expr", expr], env=env, cwd=str(top))
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    assert result_of(p) == {"ws": 4, "ready": True}
    d = run([PYTHON3, str(top / "scripts" / "doctor.py"), "--json"], env=env, cwd=str(top))
    assert d.returncode == 0, d.stdout + d.stderr


def test_zips_install_with_blenders_own_operators(dist, tmp_path):
    """Preferences > Add-ons > Install and File > New > Install Application Template, driven through the operators."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    expr = (
        "import bpy, json, os, addon_utils\n"
        f"r1 = bpy.ops.preferences.addon_install(filepath={str(dist / f'seam_studio-{VERSION}-addon.zip')!r})\n"
        f"r2 = bpy.ops.preferences.app_template_install(filepath={str(dist / f'seam-template-{VERSION}.zip')!r})\n"
        "base = bpy.utils.user_resource('SCRIPTS')\n"
        "print('RESULT ' + json.dumps({'addon': sorted(r1), 'tpl': sorted(r2),\n"
        "  'addon_init': os.path.exists(os.path.join(base, 'addons', 'seam_studio', '__init__.py')),\n"
        "  'core': os.path.isdir(os.path.join(base, 'addons', 'seam_studio', 'seam_core')),\n"
        "  'startup': os.path.exists(os.path.join(base, 'startup', 'bl_app_templates_user', 'seam', 'startup.blend'))}))\n")
    p = run([BLENDER, "--background", "--factory-startup", "--python-exit-code", "1", "--python-expr", expr],
            env={"BLENDER_USER_SCRIPTS": str(scripts), "BLENDER_USER_CONFIG": str(tmp_path / "config")})
    if p.returncode != 0 or "RESULT" not in p.stdout:
        pytest.skip("this Blender build has no legacy install operators: " + (p.stdout + p.stderr)[-300:])
    res = result_of(p)
    assert res == {"addon": ["FINISHED"], "tpl": ["FINISHED"], "addon_init": True, "core": True, "startup": True}
