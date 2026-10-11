#!/usr/bin/env python3
"""Build the demo documents in ``resources/demos/`` headlessly through SEAM's public operators.

    python3 scripts/build_demos.py [--out DIR] [--only NAME ...] [--list]

Runs inside Blender (or the ``bpy`` module): when started with a plain Python it re-launches itself in
``$SEAM_BLENDER`` / ``blender``. Every demo declares the operators it needs; demos whose operators are not
registered at run time are SKIPPED with the missing operator named (nothing is faked), the rest are built,
flushed through the engine and saved. A JSON summary goes to ``artifacts/demos.json``; exit code 0 unless a demo
that could be built failed.
"""

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _relaunch():
    exe = os.environ.get("SEAM_BLENDER") or shutil.which("blender")
    if not exe:
        sys.exit("build_demos: needs Blender (set SEAM_BLENDER) or the bpy module")
    cmd = [exe, "--background", "--factory-startup", "--python-exit-code", "1", "--python", os.path.abspath(__file__),
           "--", *sys.argv[1:]]
    sys.exit(subprocess.call(cmd))


try:
    import bpy  # noqa: F401
except ImportError:
    _relaunch()

sys.path.insert(0, os.path.join(ROOT, "extensions"))

DEMOS = []   # filled by @demo


def demo(name, title, requires, optional=()):
    """Register a demo builder. ``requires``/``optional``: ``[(operator id, (property names the demo passes))]``."""
    def deco(fn):
        DEMOS.append({"name": name, "title": title, "requires": list(requires), "optional": list(optional), "build": fn})
        return fn
    return deco


# ------------------------------------------------------------------ helpers (operator adapter)

class Ctx:
    """Thin adapter: calls registered operators with the properties they accept and tracks new objects."""

    def __init__(self, scene):
        import seam_studio  # noqa: F401
        from seam_studio.ui import util
        self.util = util
        self.scene = scene

    def has(self, idname):
        return self.util.op_available(idname)[0]

    def call(self, idname, **kwargs):
        fn = self.util.op_callable(idname)
        if fn is None:
            raise RuntimeError(f"operator {idname} is not registered")
        have = set(self.util.op_property_names(idname))
        res = fn(**{k: v for k, v in kwargs.items() if k in have})
        if "FINISHED" not in res:
            raise RuntimeError(f"{idname} returned {sorted(res)}")
        return res

    def new_objects(self, before):
        return [o for o in self.scene.objects if o.name not in before]

    def select_only(self, obj):
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    def add_shape(self, kind, name, x, y, w, h, fill, **extra):
        before = {o.name for o in self.scene.objects}
        self.call("seam.add_shape", kind=kind, x=x, y=y, width=w, height=h, **extra)
        new = [o for o in self.new_objects(before) if o.seam.role == "SHAPE"]
        if not new:
            raise RuntimeError("seam.add_shape created no SHAPE object")
        obj = new[-1]
        obj.name = name
        st = obj.seam.style
        st.fill_enabled = True
        st.fill_type = "SOLID"
        st.fill_color = fill
        self.select_only(obj)
        return obj


def _srgb(hex_rgb, alpha=1.0):
    """#rrggbb -> scene-linear RGBA (style colours are scene-linear)."""
    v = [int(hex_rgb[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in v]
    return (lin[0], lin[1], lin[2], alpha)


# ------------------------------------------------------------------ demos

_SHAPE = ("seam.add_shape", ("kind", "x", "y", "width", "height"))


@demo("vector_poster", "Vector poster: shapes, colours and text on an artboard",
      requires=[_SHAPE], optional=[("seam.add_text", ("body",))])
def build_vector_poster(ctx):
    from seam_studio.vector import canvas
    from seam_studio.ui import document
    scene = ctx.scene
    ab = canvas.active_artboard(scene)
    document.resize_artboard(ab, 1200, 800)
    ctx.add_shape("ROUNDED_RECT", "Card", 600, -400, 1000, 640, _srgb("#f4efe6"), radius=36)
    ctx.add_shape("ELLIPSE", "Sun", 380, -330, 300, 300, _srgb("#f0882b"))
    ctx.add_shape("STAR", "Star", 820, -330, 300, 300, _srgb("#2f6fdd"), sides=5, inner_ratio=0.45)
    ctx.add_shape("RECT", "Ground", 600, -600, 1000, 120, _srgb("#233047"))
    if ctx.has("seam.add_text"):
        before = {o.name for o in scene.objects}
        ctx.call("seam.add_text", body="SEAM Studio", x=270, y=-640, size=72)
        for o in ctx.new_objects(before):
            if o.seam.role == "TEXT":
                o.seam.style.fill_color = _srgb("#f4efe6")


@demo("paint_sketch", "Paint canvas with two layers", requires=[("seam.raster_new", ("name", "width", "height"))],
      optional=[("seam.raster_layer_add", ())])
def build_paint_sketch(ctx):
    ctx.call("seam.raster_new", name="Sketch", width=1024, height=768, background="WHITE")
    if ctx.has("seam.raster_layer_add"):
        ctx.call("seam.raster_layer_add")
    try:  # optional pixel content through the public raster API (operators paint with brushes in the GUI)
        import numpy as np
        from seam_studio.paint import rasterdoc
        doc = ctx.scene.seam.rasters[0]
        arr = np.zeros((doc.height, doc.width, 4), dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, doc.width, dtype=np.float32)[None, :]
        arr[..., 0] = ramp
        arr[..., 1] = 0.35
        arr[..., 2] = 1.0 - ramp
        arr[..., 3] = 1.0
        rasterdoc.write_layer_array(doc.layers[len(doc.layers) - 1], arr)
    except (ImportError, AttributeError, NotImplementedError) as exc:
        print("  (paint_sketch: layer content skipped:", exc, ")")


@demo("draw_to_3d", "Draw to 3D: a star shape with a live extruded result",
      requires=[_SHAPE, ("seam.make_3d", ("recipe",))])
def build_draw_to_3d(ctx):
    from seam_studio.ui import document
    document.resize_artboard(__import__("seam_studio.vector.canvas", fromlist=["x"]).active_artboard(ctx.scene), 800, 800)
    star = ctx.add_shape("STAR", "Star", 400, -400, 440, 440, _srgb("#f0882b"), sides=5, inner_ratio=0.45)
    ctx.select_only(star)
    ctx.call("seam.make_3d", recipe="EXTRUDE", depth=0.25, bevel_depth=0.015)


# ------------------------------------------------------------------ driver

def build_one(entry, out_dir):
    import bpy
    import seam_studio
    from seam_studio.engine import core as engine
    from seam_studio.ui import document, util
    missing = []
    for idname, props in entry["requires"]:
        ok, why = util.op_available(idname, props)
        if not ok:
            missing.append(why)
    if missing:
        return {"name": entry["name"], "status": "skipped", "reason": "; ".join(missing)}
    bpy.ops.wm.read_homefile(use_empty=True)
    engine.reset()
    scene = bpy.context.scene
    document.ensure_document(scene)
    ctx = Ctx(scene)
    entry["build"](ctx)
    engine.flush()
    path = os.path.join(out_dir, entry["name"] + ".blend")
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True, check_existing=False)
    return {"name": entry["name"], "status": "built", "path": path, "bytes": os.path.getsize(path),
            "objects": len(scene.objects)}


def main(argv):
    out_dir = os.path.join(ROOT, "resources", "demos")
    only = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out_dir = os.path.abspath(argv[i + 1])
            i += 2
        elif argv[i] == "--only":
            only.append(argv[i + 1])
            i += 2
        elif argv[i] == "--list":
            for d in DEMOS:
                print(f"{d['name']}: {d['title']}  (needs: {', '.join(i for i, _ in d['requires'])})")
            return 0
        else:
            i += 1
    import addon_utils  # noqa: F401
    import seam_studio
    seam_studio.register()
    results = []
    failed = 0
    for entry in DEMOS:
        if only and entry["name"] not in only:
            continue
        try:
            res = build_one(entry, out_dir)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            res = {"name": entry["name"], "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            failed += 1
        results.append(res)
        extra = res.get("reason") or res.get("path", "")
        print(f"demo {res['name']}: {res['status'].upper()} {extra}")
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    with open(os.path.join(ROOT, "artifacts", "demos.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    built = sum(1 for r in results if r["status"] == "built")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"demos: {built} built, {skipped} skipped (missing operators), {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(args))
