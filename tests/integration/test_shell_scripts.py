"""Launcher, bootstrap, doctor and demo builder (shell). Subprocess based; skipped when Blender is unavailable."""

import json
import os
import shutil
import stat
import subprocess
import sys
import types

import bpy
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.environ.get("SEAM_BLENDER") or shutil.which("blender")
PYTHON3 = shutil.which("python3")

needs_blender = pytest.mark.skipif(not BLENDER or not PYTHON3, reason="blender / python3 not on PATH")


def run(cmd, env=None, timeout=240, cwd=ROOT):
    full = dict(os.environ)
    full.update(env or {})
    full.pop("BLENDER_USER_SCRIPTS", None) if not (env and "BLENDER_USER_SCRIPTS" in env) else None
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=full, cwd=cwd)


@pytest.fixture(scope="module")
def built_template():
    """startup.blend must exist (it is generated); build it with bootstrap when missing (needs xvfb-run)."""
    path = os.path.join(ROOT, "application_templates", "seam", "startup.blend")
    if not os.path.exists(path):
        if not shutil.which("xvfb-run") and not os.environ.get("DISPLAY"):
            pytest.skip("startup.blend missing and no display/xvfb-run to build it")
        p = run([os.path.join(ROOT, "scripts", "bootstrap.sh"), "--quiet"], timeout=600)
        assert p.returncode == 0, p.stdout + p.stderr
    assert os.path.exists(path)
    return path


# ------------------------------------------------------------------ bootstrap

@needs_blender
def test_bootstrap_is_idempotent_and_links_profile(tmp_path, built_template):
    profile = tmp_path / "profile"
    env = {"SEAM_PROFILE": str(profile)}
    boot = os.path.join(ROOT, "scripts", "bootstrap.sh")
    first = run([boot], env=env)
    assert first.returncode == 0, first.stdout + first.stderr
    addon = profile / "scripts" / "addons" / "seam_studio"
    tpl = profile / "scripts" / "startup" / "bl_app_templates_user" / "seam"
    assert addon.is_symlink() and os.path.realpath(addon) == os.path.realpath(os.path.join(ROOT, "extensions", "seam_studio"))
    assert tpl.is_symlink() and os.path.realpath(tpl) == os.path.realpath(os.path.join(ROOT, "application_templates", "seam"))
    assert (profile / "config").is_dir()
    mtime = os.path.getmtime(built_template)
    second = run([boot], env=env)
    assert second.returncode == 0 and "Ready." in second.stdout
    assert os.path.getmtime(built_template) == mtime            # existing startup.blend is not rebuilt
    assert addon.is_symlink() and tpl.is_symlink()


def _fake_blender(tmp_path, version_line):
    exe = tmp_path / "fake-blender"
    exe.write_text(f"#!/bin/sh\necho '{version_line}'\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_bootstrap_explains_missing_and_old_blender(tmp_path):
    boot = os.path.join(ROOT, "scripts", "bootstrap.sh")
    env = {"SEAM_PROFILE": str(tmp_path / "p")}
    p = run([boot, "--no-template"], env={**env, "SEAM_BLENDER": str(tmp_path / "nope")})
    assert p.returncode == 1 and "Blender not found" in p.stderr and "apt-get install blender" in p.stderr \
        and "blender.org/download" in p.stderr
    p = run([boot, "--no-template"], env={**env, "SEAM_BLENDER": _fake_blender(tmp_path, "Blender 3.6.2")})
    assert p.returncode == 1 and "too old" in p.stderr and "4.0" in p.stderr
    p = run([boot, "--no-template"], env={**env, "SEAM_BLENDER": _fake_blender(tmp_path, "gibberish")})
    assert p.returncode == 1 and "could not read the Blender version" in p.stderr
    assert not (tmp_path / "p" / "scripts").exists()            # nothing created when the check fails


# ------------------------------------------------------------------ launcher smoke

@needs_blender
def test_launcher_background_smoke(tmp_path, built_template):
    profile = tmp_path / "profile"
    env = {"SEAM_PROFILE": str(profile)}
    expr = ("import bpy,json; s=bpy.context.scene; "
            "print('SMOKE '+json.dumps({'addon':hasattr(bpy.types.Scene,'seam'),'ready':s.seam.doc.canvas_ready,"
            "'objects':sorted(o.name for o in bpy.data.objects),"
            "'ws':[w.name for w in bpy.data.workspaces if w.name in ('Vector','Paint','Model','Split')],"
            "'vt':s.view_settings.view_transform}))")
    p = run([os.path.join(ROOT, "scripts", "seam-studio"), "--background", "--python-expr", expr], env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    line = [l for l in p.stdout.splitlines() if l.startswith("SMOKE ")][0]
    d = json.loads(line[len("SMOKE "):])
    assert d["addon"] and d["ready"]
    assert d["objects"] == ["Artboard 1", "SEAM Canvas"]          # no default cube/camera/light
    assert sorted(d["ws"]) == ["Model", "Paint", "Split", "Vector"]
    assert d["vt"] == "Standard"
    # the launcher bootstrapped the missing profile itself
    assert (profile / "scripts" / "addons" / "seam_studio").is_symlink()


@needs_blender
def test_launcher_requires_a_file_argument_passthrough(tmp_path, built_template):
    """Extra arguments reach Blender: opening a .blend must keep the file's own content (no template changes)."""
    blend = tmp_path / "plain.blend"
    p = run([BLENDER, "--background", "--factory-startup", "--python-expr",
             f"import bpy; bpy.ops.wm.save_as_mainfile(filepath={str(blend)!r})"])
    assert p.returncode == 0 and blend.exists()
    expr = "import bpy; print('OBJS '+','.join(sorted(o.name for o in bpy.data.objects)))"
    p = run([os.path.join(ROOT, "scripts", "seam-studio"), "--background", str(blend), "--python-expr", expr],
            env={"SEAM_PROFILE": str(tmp_path / "profile")})
    assert p.returncode == 0, p.stdout + p.stderr
    objs = [l for l in p.stdout.splitlines() if l.startswith("OBJS ")][0][5:]
    assert objs == "Camera,Cube,Light"                              # untouched factory scene, no SEAM canvas added


# ------------------------------------------------------------------ doctor

@needs_blender
def test_doctor_reports_ok(tmp_path, built_template):
    env = {"SEAM_PROFILE": str(tmp_path / "profile")}
    run([os.path.join(ROOT, "scripts", "bootstrap.sh"), "--quiet"], env=env)
    p = run([PYTHON3, os.path.join(ROOT, "scripts", "doctor.py"), "--json"], env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = json.loads(p.stdout)
    by = {r["name"]: r for r in rows}
    assert by["blender"]["level"] == "OK"
    assert by["smoke:addon"]["level"] == "OK" and by["smoke:canvas"]["level"] == "OK"
    assert by["smoke:workspaces"]["level"] == "OK"
    assert by["numpy in Blender"]["level"] == "OK"
    assert not [r for r in rows if r["level"] == "FAIL"]


def test_doctor_fails_without_blender(tmp_path):
    p = run([PYTHON3, os.path.join(ROOT, "scripts", "doctor.py"), "--quick", "--json"],
            env={"SEAM_BLENDER": str(tmp_path / "missing"), "SEAM_PROFILE": str(tmp_path / "p")})
    assert p.returncode == 1
    rows = json.loads(p.stdout)
    assert any(r["name"] == "blender" and r["level"] == "FAIL" for r in rows)
    assert any(r["name"].startswith("profile:") and r["level"] == "FAIL" for r in rows)   # profile not bootstrapped


# ------------------------------------------------------------------ demo builder

class FakeOp:
    def __init__(self, props, fn):
        self.props, self.fn, self.calls = props, fn, []

    def __call__(self, **kw):
        self.calls.append(kw)
        self.fn(**kw)
        return {"FINISHED"}

    def poll(self):
        return True

    def get_rna_type(self):
        return types.SimpleNamespace(properties=[types.SimpleNamespace(identifier=p) for p in self.props])


@pytest.fixture
def demos_mod(seam):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib
    mod = importlib.import_module("build_demos")
    yield mod
    sys.path.remove(os.path.join(ROOT, "scripts"))


def test_demos_are_skipped_with_the_missing_operator_named(demos_mod, tmp_path, monkeypatch):
    from seam_studio.ui import util
    monkeypatch.setattr(util, "op_callable", lambda idname: None)
    names = {d["name"] for d in demos_mod.DEMOS}
    assert names == {"vector_poster", "paint_sketch", "draw_to_3d"}
    for entry in demos_mod.DEMOS:
        res = demos_mod.build_one(entry, str(tmp_path))
        assert res["status"] == "skipped" and "is not registered" in res["reason"]
    assert os.listdir(tmp_path) == []                                # nothing written for skipped demos


def test_demo_is_built_through_the_operators_that_exist(demos_mod, tmp_path, monkeypatch):
    """With stand-ins for the vector/link operators (owned by other workers) the draw_to_3d demo is built and saved."""
    from seam_studio.ui import util
    from seam_studio.vector import canvas
    from seam_studio.seam_core.geometry import BezierPath, BezierPoint

    def add_shape(kind, x, y, width, height, **_kw):
        sq = [(-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)]
        path = BezierPath([BezierPoint(p, p, p, "VECTOR", "VECTOR") for p in sq], closed=True)
        canvas.new_shape_object(bpy.context.scene, kind.title(), [path], location=(x, y))

    made = []

    def make_3d(recipe, **kw):
        src = bpy.context.view_layer.objects.active
        made.append((src.name, recipe, kw))
        res = bpy.data.objects.new("result", bpy.data.meshes.new("result"))
        bpy.context.scene.collection.objects.link(res)

    ops = {"seam.add_shape": FakeOp(("kind", "x", "y", "width", "height", "radius", "sides", "inner_ratio"), add_shape),
           "seam.make_3d": FakeOp(("recipe", "depth", "bevel_depth"), make_3d)}
    monkeypatch.setattr(util, "op_callable", lambda idname: ops.get(idname))
    entry = next(d for d in demos_mod.DEMOS if d["name"] == "draw_to_3d")
    res = demos_mod.build_one(entry, str(tmp_path))
    assert res["status"] == "built", res
    path = tmp_path / "draw_to_3d.blend"
    assert path.exists() and res["bytes"] == path.stat().st_size
    assert made == [("Star", "EXTRUDE", {"depth": 0.25, "bevel_depth": 0.015})]
    # artboard resized by the demo and the file reopens with the add-on schema
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene = bpy.context.scene
    ab = [o for o in scene.objects if o.seam.role == "ARTBOARD"][0]
    assert (ab.seam.artboard.width, ab.seam.artboard.height) == (800, 800)
    star = bpy.data.objects["Star"]
    assert star.seam.role == "SHAPE" and abs(star.seam.style.fill_color[0] - 0.871) < 0.01   # #f0882b in linear
