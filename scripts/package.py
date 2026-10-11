#!/usr/bin/env python3
"""Build the distributable artifacts in ``dist/``.

    python3 scripts/package.py [--out DIR] [--no-build] [--skip-demos]

Artifacts (``<v>`` = ``seam_core.VERSION_STRING``):

``seam_studio-<v>-addon.zip``       legacy add-on layout: ``seam_studio/`` with ``bl_info``, ``blender_manifest.toml`` and a
                                    REAL copy of ``seam_core`` (in the repository it is a symlink to ``src/seam_core``).
``seam_studio-<v>-extension.zip``   Blender 4.2+ extension layout (manifest at the zip root).
``seam-template-<v>.zip``           application template ``seam/`` (``__init__.py``, ``startup.blend``, ``splash.png``).
``seam-studio-<v>-linux.tar.gz``    launcher, bootstrap/doctor scripts, add-on, template, demos, docs, notices.
``SHA256SUMS``                      checksums of all of the above.

Builds are reproducible (sorted entries, fixed timestamps, no ``__pycache__``). ``startup.blend`` is built with
``scripts/bootstrap.sh`` when missing (needs Blender + xvfb-run); ``--no-build`` fails instead. Demos are built with
``scripts/build_demos.py`` (demos whose operators are missing are skipped there) unless ``--skip-demos``.
"""

import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDON = os.path.join(ROOT, "extensions", "seam_studio")
SEAM_CORE = os.path.join(ROOT, "src", "seam_core")
TEMPLATE = os.path.join(ROOT, "application_templates", "seam")
EPOCH = (2026, 1, 1, 0, 0, 0)  # fixed timestamp: reproducible archives
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}
SKIP_SUFFIX = (".pyc", ".pyo", ".DS_Store")
EXECUTABLE = {"seam-studio", "bootstrap.sh", "doctor.py", "build_template.py", "build_demos.py", "make_splash.py"}


def die(msg):
    print("package:", msg, file=sys.stderr)
    sys.exit(1)


def read_version():
    init = open(os.path.join(SEAM_CORE, "__init__.py")).read()
    m = re.search(r'VERSION_STRING\s*=\s*"([^"]+)"', init)
    if not m:
        die("cannot read VERSION_STRING from src/seam_core/__init__.py")
    version = m.group(1)
    manifest = open(os.path.join(ADDON, "blender_manifest.toml")).read()
    mm = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.M)
    if not mm or mm.group(1) != version:
        die(f"version mismatch: seam_core {version} vs blender_manifest.toml {mm.group(1) if mm else None}")
    bl = open(os.path.join(ADDON, "__init__.py")).read()
    mb = re.search(r'"version":\s*\((\d+),\s*(\d+),\s*(\d+)\)', bl)
    if not mb or ".".join(mb.groups()) != version:
        die(f"version mismatch: seam_core {version} vs bl_info {mb.groups() if mb else None}")
    return version


def walk_files(base, follow=True):
    """Yield ``(relative path, absolute path)`` of regular files under ``base`` (symlinked dirs are followed)."""
    for dirpath, dirnames, filenames in os.walk(base, followlinks=follow):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if fn.endswith(SKIP_SUFFIX):
                continue
            ab = os.path.join(dirpath, fn)
            if os.path.isfile(ab):
                yield os.path.relpath(ab, base).replace(os.sep, "/"), ab


def addon_entries(prefix=""):
    """``[(archive_name, absolute_path)]`` of the add-on with a real seam_core copy (symlink resolved)."""
    if not os.path.isfile(os.path.join(ADDON, "seam_core", "__init__.py")):
        die("extensions/seam_studio/seam_core is missing or a dangling symlink")
    return [(prefix + rel, ab) for rel, ab in walk_files(ADDON)]


def add_zip(z, name, data_or_path, mode=0o644):
    info = zipfile.ZipInfo(name, EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100000 | mode) << 16
    data = open(data_or_path, "rb").read() if os.path.exists(str(data_or_path)) else data_or_path
    z.writestr(info, data)


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, src in sorted(entries):
            add_zip(z, name, src)


def tar_add(t, name, src, mode=None):
    data = open(src, "rb").read()
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 1767225600  # 2026-01-01
    ti.mode = mode if mode is not None else (0o755 if os.path.basename(name) in EXECUTABLE else 0o644)
    ti.uname = ti.gname = ""
    ti.uid = ti.gid = 0
    t.addfile(ti, io.BytesIO(data))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_template_built(no_build):
    startup = os.path.join(TEMPLATE, "startup.blend")
    if os.path.exists(startup):
        return
    if no_build:
        die("application_templates/seam/startup.blend is missing (run scripts/bootstrap.sh or drop --no-build)")
    print("package: building startup.blend ...")
    rc = subprocess.call([os.path.join(ROOT, "scripts", "bootstrap.sh"), "--quiet"])
    if rc != 0 or not os.path.exists(startup):
        die("building startup.blend failed")


def ensure_demos(skip):
    demos = os.path.join(ROOT, "resources", "demos")
    if not skip:
        print("package: building demos ...")
        rc = subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "build_demos.py")])
        if rc != 0:
            die("build_demos.py failed")
    return sorted(os.path.join(demos, f) for f in os.listdir(demos) if f.endswith(".blend")) if os.path.isdir(demos) else []


def main(argv):
    out = os.path.join(ROOT, "dist")
    no_build = "--no-build" in argv
    skip_demos = "--skip-demos" in argv
    if "--out" in argv:
        out = os.path.abspath(argv[argv.index("--out") + 1])
    version = read_version()
    ensure_template_built(no_build)
    demos = ensure_demos(skip_demos)
    os.makedirs(out, exist_ok=True)
    artifacts = []

    # 1) legacy add-on zip (folder seam_studio/)
    p = os.path.join(out, f"seam_studio-{version}-addon.zip")
    write_zip(p, addon_entries("seam_studio/"))
    artifacts.append(p)

    # 2) extension zip (manifest at the root)
    p = os.path.join(out, f"seam_studio-{version}-extension.zip")
    write_zip(p, addon_entries(""))
    artifacts.append(p)

    # 3) application template zip
    p = os.path.join(out, f"seam-template-{version}.zip")
    tpl = [("seam/" + rel, ab) for rel, ab in walk_files(TEMPLATE)
           if rel in ("__init__.py", "startup.blend", "splash.png")]
    if not any(n.endswith("startup.blend") for n, _ in tpl):
        die("startup.blend missing from the template")
    write_zip(p, tpl)
    artifacts.append(p)

    # 4) linux tarball (same layout as the repository so scripts/seam-studio and bootstrap.sh work unchanged)
    p = os.path.join(out, f"seam-studio-{version}-linux.tar.gz")
    top = f"seam-studio-{version}/"
    import gzip
    with open(p, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for name in ("seam-studio", "bootstrap.sh", "doctor.py", "build_template.py", "build_demos.py", "make_splash.py"):
            tar_add(t, top + "scripts/" + name, os.path.join(ROOT, "scripts", name))
        for rel, ab in addon_entries("extensions/seam_studio/"):
            tar_add(t, top + rel, ab)
        for rel, ab in walk_files(TEMPLATE):
            if rel in ("__init__.py", "startup.blend", "splash.png"):
                tar_add(t, top + "application_templates/seam/" + rel, ab)
        for ab in demos:
            tar_add(t, top + "resources/demos/" + os.path.basename(ab), ab)
        for rel in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
            tar_add(t, top + rel, os.path.join(ROOT, rel))
        for rel in ("architecture.md", "acceptance.md"):
            src = os.path.join(ROOT, "docs", rel)
            if os.path.exists(src):
                tar_add(t, top + "docs/" + rel, src)
    artifacts.append(p)

    # checksums
    sums = os.path.join(out, "SHA256SUMS")
    with open(sums, "w") as fh:
        for a in sorted(artifacts):
            fh.write(f"{sha256(a)}  {os.path.basename(a)}\n")
    print(f"package: {len(artifacts)} artifacts in {out}")
    for a in sorted(artifacts):
        print(f"  {os.path.getsize(a):>10,}  {os.path.basename(a)}")
    print("  SHA256SUMS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
