#!/usr/bin/env python3
"""SEAM Studio doctor: checks the environment and this checkout/installation.

    python3 scripts/doctor.py [--json] [--quick]

* ``--quick``  skip the headless Blender smoke test (the slow check, ~2 s).
* ``--json``   machine readable output.

Levels: OK, WARN (something optional is missing, e.g. GUI test tooling), FAIL (SEAM cannot run). Exit code 1 when
any FAIL. Uses only the Python standard library.
"""

import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE = os.environ.get("SEAM_PROFILE", os.path.join(ROOT, ".seam-profile"))
BLENDER = os.environ.get("SEAM_BLENDER", "blender")
MIN_BLENDER = (4, 0)
results = []


def add(level, name, detail=""):
    results.append({"level": level, "name": name, "detail": detail})


def run(cmd, timeout=60, env=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout + p.stderr
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)


def check_python():
    v = sys.version_info
    add("OK" if v >= (3, 10) else "WARN", "python", f"{v.major}.{v.minor}.{v.micro} (only the helper scripts run on it)")


def check_blender():
    exe = shutil.which(BLENDER) if os.sep not in BLENDER else (BLENDER if os.access(BLENDER, os.X_OK) else None)
    if not exe:
        add("FAIL", "blender", f"'{BLENDER}' not found. Install Blender >= 4.0 (apt-get install blender, or "
                               "https://www.blender.org/download/) or set SEAM_BLENDER=/path/to/blender")
        return None
    code, out = run([exe, "--version"], 30)
    m = re.search(r"Blender (\d+)\.(\d+)(?:\.(\d+))?", out)
    if not m:
        add("FAIL", "blender", f"{exe}: cannot read the version (exit {code}): {out.strip()[:120]}")
        return None
    ver = tuple(int(x) for x in m.groups(default="0"))
    ok = ver[:2] >= MIN_BLENDER
    add("OK" if ok else "FAIL", "blender", f"{'.'.join(map(str, ver))} at {exe}" +
        ("" if ok else f" (SEAM needs >= {MIN_BLENDER[0]}.{MIN_BLENDER[1]})"))
    return exe if ok else None


def check_layout():
    addon = os.path.join(ROOT, "extensions", "seam_studio")
    for rel, level in (("__init__.py", "FAIL"), ("blender_manifest.toml", "WARN"), ("ui/panels.py", "FAIL"),
                       ("seam_core/__init__.py", "FAIL")):
        p = os.path.join(addon, rel)
        add("OK" if os.path.exists(p) else level, f"addon:{rel}", p if os.path.exists(p) else f"missing {p}")
    template = os.path.join(ROOT, "application_templates", "seam")
    for rel, level in (("__init__.py", "FAIL"), ("startup.blend", "WARN"), ("splash.png", "WARN")):
        p = os.path.join(template, rel)
        hint = " (run scripts/bootstrap.sh)" if rel == "startup.blend" else ""
        add("OK" if os.path.exists(p) else level, f"template:{rel}", p if os.path.exists(p) else f"missing {p}{hint}")
    for rel in ("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md"):
        p = os.path.join(ROOT, rel)
        add("OK" if os.path.exists(p) else "WARN", rel, "" if os.path.exists(p) else "missing")


def check_profile():
    links = {
        os.path.join(PROFILE, "scripts", "addons", "seam_studio"): os.path.join(ROOT, "extensions", "seam_studio"),
        os.path.join(PROFILE, "scripts", "startup", "bl_app_templates_user", "seam"):
            os.path.join(ROOT, "application_templates", "seam"),
    }
    for link, target in links.items():
        name = os.path.relpath(link, PROFILE)
        if os.path.realpath(link) == os.path.realpath(target):
            add("OK", f"profile:{name}", "linked")
        else:
            add("FAIL", f"profile:{name}", f"{link} does not point to {target}; run scripts/bootstrap.sh")


SMOKE = r'''
import json, bpy
d = {"addon": hasattr(bpy.types.Scene, "seam")}
if d["addon"]:
    s = bpy.context.scene
    d["canvas_ready"] = bool(s.seam.doc.canvas_ready)
    d["objects"] = sorted(o.name for o in bpy.data.objects)
d["workspaces"] = [w.name for w in bpy.data.workspaces if w.name in ("Vector", "Paint", "Model", "Split")]
print("DOCTOR_JSON " + json.dumps(d))
'''


def check_smoke(exe):
    env = dict(os.environ, BLENDER_USER_CONFIG=os.path.join(PROFILE, "config"),
               BLENDER_USER_SCRIPTS=os.path.join(PROFILE, "scripts"))
    code, out = run([exe, "--app-template", "seam", "--background", "--python-expr", SMOKE], 120, env)
    m = re.search(r"DOCTOR_JSON (\{.*\})", out)
    if not m:
        add("FAIL", "smoke", f"Blender did not run the SEAM template headless (exit {code}): {out.strip()[-300:]}")
        return
    d = json.loads(m.group(1))
    add("OK" if d.get("addon") else "FAIL", "smoke:addon", "seam_studio enabled by the template" if d.get("addon")
        else "the add-on did not load (see 'Add-on not loaded' in the Blender output)")
    if d.get("addon"):
        add("OK" if d.get("canvas_ready") else "FAIL", "smoke:canvas", f"objects: {d.get('objects')}")
    ws = d.get("workspaces", [])
    add("OK" if len(ws) == 4 else "WARN", "smoke:workspaces",
        f"{ws}" if len(ws) == 4 else f"found {ws}; startup.blend is missing or stale: scripts/bootstrap.sh --rebuild-template")


def check_tooling(exe):
    for tool, why in (("xvfb-run", "GUI tests / template build without a display"),
                      ("import", "GUI test screenshots (ImageMagick)"), ("convert", "GUI test image checks (ImageMagick)")):
        p = shutil.which(tool)
        add("OK" if p else "WARN", f"tool:{tool}", p or f"not found; needed for {why}")
    venv = os.environ.get("SEAM_BPY52_PYTHON", "/opt/seam-cache/venv-bpy52/bin/python")
    add("OK" if os.path.exists(venv) else "WARN", "bpy 5.x runtime",
        venv if os.path.exists(venv) else "not found (optional: headless LTS test runtime)")
    if exe:
        code, out = run([exe, "--background", "--factory-startup", "--python-expr", "import numpy;print('NUMPY',numpy.__version__)"], 60)
        m = re.search(r"NUMPY (\S+)", out)
        add("OK" if m else "FAIL", "numpy in Blender", m.group(1) if m else "Blender's Python has no numpy (SEAM needs it)")


def main():
    quick = "--quick" in sys.argv
    check_python()
    exe = check_blender()
    check_layout()
    check_profile()
    if exe and not quick:
        check_smoke(exe)
    check_tooling(exe)
    if "--json" in sys.argv:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"[{r['level']:>4}] {r['name']}: {r['detail']}")
        n = {lv: sum(1 for r in results if r["level"] == lv) for lv in ("OK", "WARN", "FAIL")}
        print(f"\ndoctor: {n['OK']} ok, {n['WARN']} warnings, {n['FAIL']} failures")
    return 1 if any(r["level"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
