#!/usr/bin/env python3
"""SEAM Studio benchmarks - headless, on both runtimes.

Run on one runtime:
    /opt/seam-cache/venv-bpy52/bin/python scripts/bench.py [--quick] [--only link,doc,...]
    blender -b --factory-startup --python scripts/bench.py -- [--quick] [--only link,doc,...]
Run on both and rebuild the summary + docs/perf.md:
    python3 scripts/bench.py --all

What is measured (wall-clock, ``time.perf_counter``; every number in the output is a measurement, nothing is
extrapolated):
  link      link update latency: edit a source node, ``engine.flush()`` (this includes the ``view_layer.update()``
            the background engine performs) and the build time the engine itself reports, for a 4-point square,
            a 200-point polygon, 3D text, a revolve and a sweep; >= 20 edits each (median / p95 / min / max).
  doc       document-size scaling with 10 / 100 / 500 shapes: initial flush, flush with no change, flush after one
            node edit, a bare ``view_layer.update()`` for reference, .blend save and open time.
  png       PNG export of a 1920 x 1080 artboard (100 shapes, strokes and gradients) through ``seam.png_export``.
  raster    compositing 4 layers of 2048 x 2048 (numpy kernel and the in-Blender document path).
  svg       SVG export (outlines / text) and import through ``seam.svg_export`` / ``seam.svg_import``.

Outputs: ``artifacts/perf/run-<runtime>.json`` per runtime and ``artifacts/perf/summary.json`` (hardware, versions,
all runs).  ``--all`` also regenerates ``docs/perf.md`` from the summary.
"""

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF_DIR = os.path.join(ROOT, "artifacts", "perf")
VENV52 = os.environ.get("SEAM_BPY52_PYTHON", "/opt/seam-cache/venv-bpy52/bin/python")
BLENDER40 = os.environ.get("SEAM_BLENDER", "/usr/bin/blender")
ALL_SECTIONS = ("link", "doc", "png", "raster", "svg")


# ------------------------------------------------------------------ small utilities

def stats(values):
    """median / p95 (nearest rank) / min / max / mean of a list of milliseconds."""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    return {"n": n, "median_ms": round(statistics.median(xs), 3), "p95_ms": round(xs[max(0, math.ceil(0.95 * n) - 1)], 3),
            "min_ms": round(xs[0], 3), "max_ms": round(xs[-1], 3), "mean_ms": round(sum(xs) / n, 3)}


def timed(fn, reps, warmup=0):
    out = []
    for i in range(warmup + reps):
        t0 = time.perf_counter()
        fn(i)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            out.append(dt)
    return out


def parse_args(argv):
    opts = {"quick": False, "only": list(ALL_SECTIONS), "out": None, "all": False, "doc_only": False}
    if "--" in argv:                       # blender -b --python bench.py -- <args>
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--quick":
            opts["quick"] = True
        elif a == "--all":
            opts["all"] = True
        elif a == "--only":
            i += 1
            opts["only"] = [s for s in argv[i].split(",") if s]
        elif a == "--out":
            i += 1
            opts["out"] = argv[i]
        elif a == "--write-doc":
            opts["doc_only"] = True
        else:
            raise SystemExit("unknown argument %r" % a)
        i += 1
    for s in opts["only"]:
        if s not in ALL_SECTIONS:
            raise SystemExit("unknown section %r (choose from %s)" % (s, ", ".join(ALL_SECTIONS)))
    return opts


# ------------------------------------------------------------------ hardware / versions

def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def hardware():
    cpu = ""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem_kb = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal"):
            mem_kb = int(line.split()[1])
            break
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    return {"cpu_model": cpu, "logical_cpus": os.cpu_count(), "usable_cpus": affinity,
            "memory_gib": round(mem_kb / 1024.0 / 1024.0, 2), "platform": platform.platform(),
            "machine": platform.machine(), "load_avg_start": [round(x, 2) for x in os.getloadavg()]}


def repo_commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


# ------------------------------------------------------------------ bootstrap (runs inside Blender / bpy)

def bootstrap():
    """Make the add-on importable and enabled exactly like the test runner does."""
    scripts = os.environ.get("BLENDER_USER_SCRIPTS")
    if not scripts:
        scripts = tempfile.mkdtemp(prefix="seam-bench-scripts-")
        os.environ["BLENDER_USER_SCRIPTS"] = scripts
        os.environ.setdefault("BLENDER_USER_CONFIG", tempfile.mkdtemp(prefix="seam-bench-config-"))
    addons = os.path.join(scripts, "addons")
    os.makedirs(addons, exist_ok=True)
    link = os.path.join(addons, "seam_studio")
    if not os.path.exists(link):
        os.symlink(os.path.join(ROOT, "extensions", "seam_studio"), link)
    for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "extensions")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import bpy
    import addon_utils
    t0 = time.perf_counter()
    if "seam_studio" not in sys.modules or not hasattr(bpy.types.Scene, "seam"):
        if addon_utils.enable("seam_studio", default_set=True, handle_error=None) is None:
            import seam_studio                 # user-scripts dir not scanned (Blender started without it): register directly
            seam_studio.register()
    return (time.perf_counter() - t0) * 1000.0


def fresh_file():
    import bpy
    from seam_studio.engine import core
    bpy.ops.wm.read_homefile(use_empty=True)
    if not hasattr(bpy.types.Scene, "seam"):
        import addon_utils
        addon_utils.enable("seam_studio", default_set=True, handle_error=None)
    core.reset()
    return core


def new_canvas(width=1920, height=1080):
    import bpy
    fresh_file()
    bpy.ops.seam.canvas_new(width=width, height=height)


def flush():
    from seam_studio.engine import core
    return core.flush()


# ------------------------------------------------------------------ geometry builders

def _pt(x, y, kind="VECTOR", hl=None, hr=None):
    from seam_studio.seam_core.geometry import BezierPoint
    return BezierPoint((x, y), hl or (x, y), hr or (x, y), kind, kind)


def polygon_path(points, closed=True):
    from seam_studio.seam_core.geometry import BezierPath
    return BezierPath([_pt(x, y) for x, y in points], closed=closed)


def circle_polygon(n, r):
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def ellipse_path(rx, ry):
    from seam_studio.seam_core.geometry import BezierPath
    k = 0.5522847498307936
    return BezierPath([_pt(rx, 0, "ALIGNED", (rx, -k * ry), (rx, k * ry)), _pt(0, ry, "ALIGNED", (k * rx, ry), (-k * rx, ry)),
                       _pt(-rx, 0, "ALIGNED", (-rx, k * ry), (-rx, -k * ry)),
                       _pt(0, -ry, "ALIGNED", (-k * rx, -ry), (k * rx, -ry))], closed=True)


def new_shape(name, path, x, y, color=(0.8, 0.3, 0.1, 1.0)):
    import bpy
    from seam_studio.vector import canvas
    ob = canvas.new_shape_object(bpy.context.scene, name, [path], location=(x, y))
    ob.seam.style.fill_color = color
    return ob


def activate(obj):
    import bpy
    for o in list(bpy.context.selected_objects):
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def nudge(obj, k):
    """Move node 1 by +-1 px alternately (handles follow): a real source edit that changes the signature."""
    d = 1.0 if k % 2 == 0 else -1.0
    sp = obj.data.splines[0]
    bp = sp.bezier_points[1 % len(sp.bezier_points)]
    for attr in ("co", "handle_left", "handle_right"):
        getattr(bp, attr).x += d
    obj.data.update_tag()


def showcase(n_shapes, seed=7):
    """n shapes with fills (all), strokes (every 2nd) and gradients (every 5th) on a 1920 x 1080 artboard."""
    import random
    rng = random.Random(seed)
    objs = []
    for i in range(n_shapes):
        x, y = rng.uniform(100, 1820), -rng.uniform(100, 980)
        if i % 3 == 0:
            path = ellipse_path(rng.uniform(20, 80), rng.uniform(20, 80))
        elif i % 3 == 1:
            w, h = rng.uniform(30, 150), rng.uniform(30, 150)
            path = polygon_path([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
        else:
            path = polygon_path(circle_polygon(8, rng.uniform(30, 80)))
        ob = new_shape("Shape %d" % i, path, x, y, (rng.random(), rng.random(), rng.random(), 1.0))
        if i % 2 == 0:
            ob.seam.style.stroke_enabled = True
            ob.seam.style.stroke_width = 4.0
            ob.seam.style.stroke_color = (0.0, 0.0, 0.0, 1.0)
        if i % 5 == 0:
            ob.seam.style.fill_type = "LINEAR"
        objs.append(ob)
    return objs


# ------------------------------------------------------------------ benchmark: link latency

def link_case(kind):
    import bpy
    new_canvas()
    if kind == "square4":
        src = new_shape("Square", polygon_path([(-100, -100), (100, -100), (100, 100), (-100, 100)]), 960, -540)
        params = dict(recipe="EXTRUDE", depth=0.3, orientation="FLAT")
    elif kind == "polygon200":
        src = new_shape("Poly200", polygon_path(circle_polygon(200, 150)), 960, -540)
        params = dict(recipe="EXTRUDE", depth=0.3, orientation="FLAT")
    elif kind == "text":
        bpy.ops.seam.add_text(body="SEAM Studio", x=300.0, y=-540.0, size=120.0)
        src = bpy.context.active_object
        params = dict(recipe="TEXT3D", depth=0.2, orientation="FLAT")
    elif kind == "revolve":
        prof = [(0, -150), (90, -150), (110, -90), (60, 0), (50, 60), (95, 120), (80, 150), (0, 150)]
        src = new_shape("Vase", polygon_path(prof), 960, -540)
        params = dict(recipe="REVOLVE", revolve_segments=32, resolution=6)
    elif kind == "sweep":
        pts = []
        for i in range(12):
            x = i * 60.0
            pts.append(_pt(x, 80 * math.sin(i * 0.7), "ALIGNED", (x - 20, 80 * math.sin(i * 0.7) - 15),
                           (x + 20, 80 * math.sin(i * 0.7) + 15)))
        from seam_studio.seam_core.geometry import BezierPath
        src = new_shape("Wave", BezierPath(pts, closed=False), 400, -540)
        src.seam.style.fill_enabled = False
        params = dict(recipe="SWEEP", sweep_radius=0.05, sweep_profile_resolution=6, resolution=8)
    else:
        raise ValueError(kind)
    flush()
    activate(src)
    assert bpy.ops.seam.make_3d(**params) == {"FINISHED"}
    flush()
    link = bpy.context.scene.seam.links[-1]
    assert link.state == "LIVE", (kind, link.state, link.message)
    return src, link


def bench_link(quick):
    import bpy
    edits = 20 if quick else 40
    out = {}
    for kind in ("square4", "polygon200", "text", "revolve", "sweep"):
        src, link = link_case(kind)
        n_points = sum(len(sp.bezier_points) for sp in src.data.splines) if src.type == "CURVE" else None
        e2e, build = [], []

        def edit_and_flush(k):
            if kind == "text":
                src.data.body = "SEAM Studio" + ("!" if k % 2 == 0 else "")
            else:
                nudge(src, k)
            t0 = time.perf_counter()
            res = flush()
            ms = (time.perf_counter() - t0) * 1000.0
            built = [r for r in res if r.key == "link:" + link.id and r.status == "ok"]
            assert built, "edit %d did not rebuild the link (%s)" % (k, [(r.key, r.status) for r in res])
            e2e.append(ms)
            build.append(built[-1].ms)
        for k in range(3):                                   # warm-up (not recorded)
            edit_and_flush(k)
        e2e.clear()
        build.clear()
        for k in range(edits):
            edit_and_flush(k + 3)
        me = link.target.data
        out[kind] = {"edits": edits, "source_points": n_points, "result_vertices": len(me.vertices),
                     "result_faces": len(me.polygons), "flush_after_edit": stats(e2e), "engine_build": stats(build)}
    return out


# ------------------------------------------------------------------ benchmark: document size

def bench_doc(quick):
    import bpy
    reps = 10 if quick else 20
    sizes = (10, 100) if quick else (10, 100, 500)
    out = {}
    for n in sizes:
        new_canvas()
        t0 = time.perf_counter()
        objs = showcase(n)
        create_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        flush()
        initial_ms = (time.perf_counter() - t0) * 1000.0
        flush()
        no_change = timed(lambda i: flush(), reps)
        depsgraph_only = timed(lambda i: bpy.context.view_layer.update(), reps)
        target = objs[(n // 2) // 10 * 10]          # a stroked shape with a gradient: the edit rebuilds derived data
        changed = []
        built_keys = []
        for k in range(reps + 2):
            nudge(target, k)
            t0 = time.perf_counter()
            res = flush()
            ms = (time.perf_counter() - t0) * 1000.0
            if k >= 2:
                changed.append(ms)
                built_keys.append(sum(1 for r in res if r.status == "ok"))
        path = os.path.join(tempfile.mkdtemp(prefix="seam-bench-"), "doc.blend")
        t0 = time.perf_counter()
        bpy.ops.wm.save_as_mainfile(filepath=path)
        save_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        bpy.ops.wm.open_mainfile(filepath=path)
        open_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        flush()
        post_open_flush_ms = (time.perf_counter() - t0) * 1000.0
        out[str(n)] = {"shapes": n, "scene_objects": len(bpy.context.scene.objects), "create_ms": round(create_ms, 1),
                       "initial_flush_ms": round(initial_ms, 1), "flush_no_change": stats(no_change),
                       "flush_one_node_edit": stats(changed), "builds_per_edit_median": statistics.median(built_keys),
                       "view_layer_update_only": stats(depsgraph_only), "blend_save_ms": round(save_ms, 1),
                       "blend_open_ms": round(open_ms, 1), "flush_after_open_ms": round(post_open_flush_ms, 1),
                       "blend_bytes": os.path.getsize(path)}
    return out


# ------------------------------------------------------------------ benchmark: PNG export

def bench_png(quick):
    import bpy
    from seam_studio.seam_core import png
    from seam_studio.vector import canvas, export_png
    reps = 3 if quick else 7
    new_canvas(1920, 1080)
    showcase(100)
    flush()
    tmp = tempfile.mkdtemp(prefix="seam-bench-")
    path = os.path.join(tmp, "out.png")
    ab = canvas.active_artboard(bpy.context.scene)
    op_ms = timed(lambda i: bpy.ops.seam.png_export(filepath=path, scale=1.0, transparent=False), reps, warmup=1)
    img = png.read_png(path)
    render_ms = timed(lambda i: export_png.render_artboard(bpy.context.scene, ab, 1.0, False), reps, warmup=1)
    bpy.ops.seam.png_export(filepath=path, scale=2.0, transparent=False)
    big = png.read_png(path)
    big_ms = timed(lambda i: bpy.ops.seam.png_export(filepath=path, scale=2.0, transparent=False), 2 if quick else 3)
    return {"artboard": [1920, 1080], "shapes": 100, "scale1_operator": stats(op_ms), "scale1_render_only": stats(render_ms),
            "scale1_bytes": os.path.getsize(os.path.join(tmp, "out.png")), "scale1_shape": list(img.shape),
            "scale2_operator": stats(big_ms), "scale2_shape": list(big.shape)}


# ------------------------------------------------------------------ benchmark: raster composite

def bench_raster(quick):
    import bpy
    import numpy as np
    from seam_studio.seam_core import png, raster as R
    from seam_studio.paint import rasterdoc as rd
    reps = 2 if quick else 5
    reps_doc = 2 if quick else 4
    size = 1024 if quick else 2048              # quick mode is a smoke run, not a measurement of the 2048 case
    rng = np.random.default_rng(3)
    specs = []
    for mode, opacity, masked in (("NORMAL", 1.0, False), ("MULTIPLY", 0.8, False), ("SCREEN", 0.6, True), ("NORMAL", 0.9, False)):
        rgba = rng.random((size, size, 4), dtype=np.float32)
        mask = rng.random((size, size), dtype=np.float32) if masked else None
        specs.append(R.LayerSpec(rgba=rgba, opacity=opacity, blend_mode=mode, visible=True, mask=mask))
    kernel = timed(lambda i: R.composite(specs, size, size, (1.0, 1.0, 1.0, 1.0)), reps, warmup=1)
    comp = R.composite(specs, size, size, (1.0, 1.0, 1.0, 1.0))
    to8 = timed(lambda i: R.to_uint8_srgb(comp), reps, warmup=1)
    a8 = R.to_uint8_srgb(comp)
    tmp = tempfile.mkdtemp(prefix="seam-bench-")
    enc = timed(lambda i: png.write_png(os.path.join(tmp, "k.png"), a8), 2 if quick else 3)
    out = {"size": [size, size], "layers": 4, "kernel_composite": stats(kernel), "to_uint8_srgb": stats(to8),
           "png_write": stats(enc), "png_bytes": os.path.getsize(os.path.join(tmp, "k.png"))}

    # the in-Blender document path
    fresh_file()
    t0 = time.perf_counter()
    bpy.ops.seam.raster_new(name="Bench", width=size, height=size, background="WHITE")
    out["raster_new_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    colors = ((1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1))
    fill_ms, add_ms = [], []
    for i, c in enumerate(colors):
        if i:
            t0 = time.perf_counter()
            bpy.ops.seam.raster_layer_add(name="L%d" % i)
            add_ms.append((time.perf_counter() - t0) * 1000.0)
        bpy.ops.seam.raster_select_rect(x0=100 * i, y0=100 * i, x1=100 * i + size * 0.6, y1=100 * i + size * 0.6)
        t0 = time.perf_counter()
        bpy.ops.seam.raster_fill(color=c)
        fill_ms.append((time.perf_counter() - t0) * 1000.0)
    flush()
    doc = bpy.context.scene.seam.rasters[0]
    doc.layers[1].blend_mode = "MULTIPLY"
    doc.layers[2].opacity = 0.7
    flush()
    comp_ms = timed(lambda i: rd.update_composite(doc), reps_doc, warmup=1)
    after_edit = []
    for k in range(reps_doc + 1):
        bpy.ops.seam.raster_select_rect(x0=10 * k, y0=10 * k, x1=10 * k + size * 0.2, y1=10 * k + size * 0.2)
        bpy.ops.seam.raster_fill(color=colors[k % 4])
        t0 = time.perf_counter()
        flush()
        if k:
            after_edit.append((time.perf_counter() - t0) * 1000.0)
    path = os.path.join(tmp, "doc.png")
    exp = timed(lambda i: bpy.ops.seam.raster_export_png(filepath=path, source="COMPOSITE"), reps_doc, warmup=1)
    out.update({"layer_add_ms": stats(add_ms), "fill_operator_60pct_selection": stats(fill_ms),
                "update_composite_4_layers": stats(comp_ms), "flush_after_fill_20pct": stats(after_edit),
                "export_composite_png_operator": stats(exp), "export_bytes": os.path.getsize(path)})
    return out


# ------------------------------------------------------------------ benchmark: SVG

def bench_svg(quick):
    import bpy
    from seam_studio.engine import core
    reps = 3 if quick else 7
    new_canvas(1920, 1080)
    showcase(200)
    for i in range(3):
        bpy.ops.seam.add_text(body="Benchmark text %d" % i, x=100.0 + 500 * i, y=-80.0, size=64.0)
    flush()
    tmp = tempfile.mkdtemp(prefix="seam-bench-")
    path = os.path.join(tmp, "doc.svg")
    outlines = timed(lambda i: bpy.ops.seam.svg_export(filepath=path, text_mode="OUTLINES"), reps, warmup=1)
    size_outlines = os.path.getsize(path)
    as_text = timed(lambda i: bpy.ops.seam.svg_export(filepath=path, text_mode="TEXT"), reps, warmup=1)
    size_text = os.path.getsize(path)
    bpy.ops.seam.svg_export(filepath=path, text_mode="OUTLINES")
    imp = []
    imported = 0
    for i in range(reps + 1):
        fresh_file()
        t0 = time.perf_counter()
        bpy.ops.seam.svg_import(filepath=path)
        ms = (time.perf_counter() - t0) * 1000.0
        flush()
        if i:
            imp.append(ms)
        imported = sum(1 for o in bpy.context.scene.objects if o.seam.role == "SHAPE")
    return {"shapes": 200, "texts": 3, "export_outlines": stats(outlines), "export_outlines_bytes": size_outlines,
            "export_text": stats(as_text), "export_text_bytes": size_text, "import": stats(imp), "imported_shapes": imported}


# ------------------------------------------------------------------ driver

SECTION_FUNCS = {"link": bench_link, "doc": bench_doc, "png": bench_png, "raster": bench_raster, "svg": bench_svg}


def runtime_info():
    import bpy
    import numpy
    binary = bool(bpy.app.binary_path)
    return {"id": ("blender-" if binary else "bpy-") + bpy.app.version_string.split()[0],
            "kind": "blender binary (background)" if binary else "bpy module", "blender_version": bpy.app.version_string,
            "blender_build_hash": getattr(bpy.app, "build_hash", b"").decode() if isinstance(getattr(bpy.app, "build_hash", b""), bytes)
            else str(getattr(bpy.app, "build_hash", "")),
            "python": platform.python_version(), "numpy": numpy.__version__}


def run_in_runtime(opts):
    import gc
    addon_ms = bootstrap()
    info = runtime_info()
    result = {"runtime": info, "quick": opts["quick"], "addon_enable_ms": round(addon_ms, 1),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": repo_commit(), "hardware": hardware(),
              "sections": {}, "failures": {}}
    for name in opts["only"]:
        gc.collect()
        t0 = time.perf_counter()
        try:
            result["sections"][name] = SECTION_FUNCS[name](opts["quick"])
        except Exception as exc:               # a failing section is reported, never hidden
            import traceback
            traceback.print_exc()
            result["failures"][name] = "%s: %s" % (type(exc).__name__, exc)
        print("bench %-7s %6.1f s" % (name, time.perf_counter() - t0), flush=True)
    result["hardware"]["load_avg_end"] = [round(x, 2) for x in os.getloadavg()]
    out = opts["out"] or os.path.join(PERF_DIR, "run-%s.json" % info["id"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print("wrote", out, flush=True)
    return result


def merge_summary():
    runs = {}
    for fn in sorted(os.listdir(PERF_DIR)):
        if fn.startswith("run-") and fn.endswith(".json"):
            with open(os.path.join(PERF_DIR, fn)) as fh:
                data = json.load(fh)
            runs[data["runtime"]["id"]] = data
    if not runs:
        raise SystemExit("no run-*.json in %s" % PERF_DIR)
    first = next(iter(runs.values()))
    summary = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": repo_commit(), "hardware": first["hardware"],
               "runs": runs}
    path = os.path.join(PERF_DIR, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", path)
    return summary


def run_all(opts):
    os.makedirs(PERF_DIR, exist_ok=True)
    scripts = tempfile.mkdtemp(prefix="seam-bench-scripts-")
    os.makedirs(os.path.join(scripts, "addons"))
    os.symlink(os.path.join(ROOT, "extensions", "seam_studio"), os.path.join(scripts, "addons", "seam_studio"))
    env = dict(os.environ, BLENDER_USER_SCRIPTS=scripts, BLENDER_USER_CONFIG=tempfile.mkdtemp(prefix="seam-bench-config-"))
    extra = (["--quick"] if opts["quick"] else []) + ["--only", ",".join(opts["only"])]
    rc = 0
    cmds = [("bpy 5.2", [VENV52, os.path.abspath(__file__)] + extra),
            ("blender 4.0", [BLENDER40, "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                             os.path.abspath(__file__), "--"] + extra)]
    for label, cmd in cmds:
        print("== %s" % label, flush=True)
        rc = subprocess.run(cmd, env=env).returncode or rc
    summary = merge_summary()
    write_doc(summary)
    return rc


# ------------------------------------------------------------------ docs/perf.md

def _row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _fmt(s):
    return "%.2f / %.2f" % (s["median_ms"], s["p95_ms"])


def write_doc(summary):
    runs = summary["runs"]
    ids = sorted(runs)
    hw = summary["hardware"]
    L = []
    L.append("# SEAM Studio performance (measured)\n")
    L.append("Every number below was measured by `scripts/bench.py` on the machine described here and copied from "
             "`artifacts/perf/summary.json`; nothing is extrapolated. Times are wall-clock milliseconds, shown as "
             "**median / p95** over the stated number of repetitions. Regenerate with `python3 scripts/bench.py --all`.\n")
    L.append("## Machine and runtimes\n")
    L.append(_row(["", "value"]))
    L.append(_row(["---", "---"]))
    L.append(_row(["CPU", "%s (%s logical CPUs, %s usable)" % (hw["cpu_model"], hw["logical_cpus"], hw["usable_cpus"])]))
    L.append(_row(["Memory", "%s GiB" % hw["memory_gib"]]))
    L.append(_row(["OS", hw["platform"]]))
    L.append(_row(["Repository commit", summary["commit"]]))
    L.append(_row(["Generated", summary["generated"]]))
    for rid in ids:
        r = runs[rid]
        L.append(_row([rid, "%s, %s, Python %s, numpy %s, add-on enable %.0f ms; load average at start/end %s / %s%s" % (
            r["runtime"]["kind"], r["runtime"]["blender_version"], r["runtime"]["python"], r["runtime"]["numpy"],
            r["addon_enable_ms"], r["hardware"]["load_avg_start"], r["hardware"].get("load_avg_end"),
            " (quick mode)" if r["quick"] else "")]))
    L.append("\nThe machine has 4 CPUs that were shared with other processes while measuring (see the load averages); "
             "treat small differences between runtimes as noise.\n")
    hdr = ["", *ids]

    def table(title, rows):
        L.append("### %s\n" % title)
        L.append(_row(hdr))
        L.append(_row(["---"] * len(hdr)))
        for label, getter in rows:
            cells = []
            for rid in ids:
                try:
                    cells.append(getter(runs[rid]["sections"]))
                except KeyError:
                    cells.append("n/a")
            L.append(_row([label, *cells]))
        L.append("")

    L.append("## Link update latency\n")
    L.append("An edit moves one source node (as a Grab in Edit Mode does) and the engine is flushed. "
             "*flush after edit* is the whole `engine.flush()` call (it includes the `view_layer.update()` that the "
             "background engine performs, i.e. depsgraph evaluation of the scene); *engine build* is the build time "
             "the engine reports for the link only. Repetitions per case: see `edits` in the JSON "
             "(%s).\n" % ", ".join("%s: %s" % (rid, runs[rid]["sections"]["link"]["square4"]["edits"]) for rid in ids
                                   if "link" in runs[rid]["sections"]))
    rows = []
    for kind, label in (("square4", "4-point square (extrude)"), ("polygon200", "200-point polygon (extrude)"),
                        ("text", "text 'SEAM Studio' (Text 3D)"), ("revolve", "8-point profile, 32 segments (revolve)"),
                        ("sweep", "12-node open path (sweep)")):
        rows.append((label + " - flush after edit", lambda s, k=kind: _fmt(s["link"][k]["flush_after_edit"])))
        rows.append((label + " - engine build", lambda s, k=kind: _fmt(s["link"][k]["engine_build"])))
        rows.append((label + " - result size", lambda s, k=kind: "%d vertices, %d faces" % (
            s["link"][k]["result_vertices"], s["link"][k]["result_faces"])))
    table("median / p95 ms", rows)

    L.append("## Document size scaling\n")
    L.append("Shapes are fills (every 2nd also stroked, every 5th with a gradient) on a 1920 x 1080 artboard, no links. "
             "*flush, no change* is `engine.flush()` with nothing dirty; *flush, one node edit* edits one shape and "
             "flushes; *view_layer.update only* is the bare depsgraph evaluation that every flush includes.\n")
    sizes = sorted({int(n) for rid in ids for n in runs[rid]["sections"].get("doc", {})})
    rows = []
    for n in sizes:
        n = str(n)
        rows += [("%s shapes - flush, no change" % n, lambda s, n=n: _fmt(s["doc"][n]["flush_no_change"])),
                 ("%s shapes - flush, one node edit" % n, lambda s, n=n: _fmt(s["doc"][n]["flush_one_node_edit"])),
                 ("%s shapes - view_layer.update only" % n, lambda s, n=n: _fmt(s["doc"][n]["view_layer_update_only"])),
                 ("%s shapes - builds per edit (median)" % n, lambda s, n=n: s["doc"][n]["builds_per_edit_median"]),
                 ("%s shapes - first flush after creation (ms)" % n, lambda s, n=n: s["doc"][n]["initial_flush_ms"]),
                 ("%s shapes - save / open .blend (ms)" % n, lambda s, n=n: "%s / %s" % (s["doc"][n]["blend_save_ms"],
                                                                                         s["doc"][n]["blend_open_ms"])),
                 ("%s shapes - .blend size (bytes)" % n, lambda s, n=n: s["doc"][n]["blend_bytes"])]
    table("median / p95 ms unless stated", rows)

    L.append("## PNG export, 1920 x 1080\n")
    table("seam.png_export, 100 shapes (strokes, gradients)", [
        ("operator, scale 1 (1920 x 1080)", lambda s: _fmt(s["png"]["scale1_operator"])),
        ("render only (no file write), scale 1", lambda s: _fmt(s["png"]["scale1_render_only"])),
        ("file size, scale 1 (bytes)", lambda s: s["png"]["scale1_bytes"]),
        ("operator, scale 2 (3840 x 2160)", lambda s: _fmt(s["png"]["scale2_operator"]))])

    L.append("## Raster compositing, 2048 x 2048, 4 layers\n")
    table("median / p95 ms", [
        ("numpy kernel `raster.composite` (NORMAL, MULTIPLY 0.8, SCREEN 0.6 + mask, NORMAL 0.9)",
         lambda s: _fmt(s["raster"]["kernel_composite"])),
        ("`to_uint8_srgb` of the result", lambda s: _fmt(s["raster"]["to_uint8_srgb"])),
        ("`png.write_png` of the result", lambda s: _fmt(s["raster"]["png_write"])),
        ("document: `raster_new` 2048 x 2048 (ms, single run)", lambda s: s["raster"]["raster_new_ms"]),
        ("document: `raster_fill` on a selection of 60 % of the width and height", lambda s: _fmt(s["raster"]["fill_operator_60pct_selection"])),
        ("document: `update_composite` (4 layers)", lambda s: _fmt(s["raster"]["update_composite_4_layers"])),
        ("document: `engine.flush()` after a fill of 20 % of the width and height", lambda s: _fmt(s["raster"]["flush_after_fill_20pct"])),
        ("document: `raster_export_png` composite", lambda s: _fmt(s["raster"]["export_composite_png_operator"]))])

    L.append("## SVG, 200 shapes + 3 text objects\n")
    table("median / p95 ms", [
        ("export, text as outlines", lambda s: _fmt(s["svg"]["export_outlines"])),
        ("export, text as text", lambda s: _fmt(s["svg"]["export_text"])),
        ("import of the exported file", lambda s: _fmt(s["svg"]["import"])),
        ("exported file size, outlines (bytes)", lambda s: s["svg"]["export_outlines_bytes"]),
        ("shapes after import", lambda s: s["svg"]["imported_shapes"])])

    fails = {rid: runs[rid]["failures"] for rid in ids if runs[rid]["failures"]}
    if fails:
        L.append("## Failed sections\n")
        for rid, f in fails.items():
            for k, v in f.items():
                L.append("- %s / %s: %s" % (rid, k, v))
        L.append("")
    L.append("## Method notes\n")
    L.append("- Sources are created with the internal shape helpers (fast); everything that is timed as an operator "
             "(`seam.make_3d`, `seam.png_export`, `seam.svg_export`, `seam.svg_import`, `seam.raster_*`) goes through "
             "`bpy.ops`.\n"
             "- Edits are Python-side node moves followed by `update_tag()`; this is the same data change a Grab in Edit "
             "Mode produces, without the GUI's own redraw cost, which is not measured here.\n"
             "- Background mode has no timers, so `engine.flush()` stands in for the engine tick (debounce, which is a "
             "configurable delay, is not included).\n"
             "- Not measured: GPU viewport drawing, interactive drawing-tool latency, native sculpt/paint stroke speed "
             "(all need the GUI).\n")
    path = os.path.join(ROOT, "docs", "perf.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print("wrote", path)


def main():
    opts = parse_args(sys.argv)
    if opts["all"]:
        raise SystemExit(run_all(opts))
    if opts["doc_only"]:
        write_doc(merge_summary())
        return
    run_in_runtime(opts)
    if opts["out"] is None:
        merge_summary()
    if _in_blender_binary():
        sys.stdout.flush()
        os._exit(0)


def _in_blender_binary():
    try:
        import bpy
        return bool(bpy.app.binary_path)
    except Exception:
        return False


if __name__ == "__main__":
    main()
