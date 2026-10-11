"""Perf: source edit -> flush -> target updated. Writes artifacts/perf/link.json (one entry per runtime)."""

import json
import math
import os
import statistics
import time

import bpy
import pytest

from helpers import flush
from test_link_support import make_link, make_shape, move_point

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "artifacts", "perf", "link.json")
RUNS = 25


def _star(n=200, r1=100.0, r2=70.0):
    return tuple((math.cos(2 * math.pi * i / n) * (r1 if i % 2 == 0 else r2),
                  math.sin(2 * math.pi * i / n) * (r1 if i % 2 == 0 else r2)) for i in range(n))


def _measure(src, link, smooth=False):
    from seam_studio.link import producer
    times, builds = [], []
    for i in range(RUNS):
        dx = 1.0 if i % 2 == 0 else -1.0
        t0 = time.perf_counter()
        move_point(src, 0, 0, dx=dx)
        flush()
        times.append((time.perf_counter() - t0) * 1000.0)
        builds.append(link.last_build_ms)
        assert link.state == "LIVE", link.message
        assert link.target_signature == producer.read_mesh_signature(link.target.data)
    assert link.build_count == RUNS + 1
    return {
        "edit_to_updated_ms_median": round(statistics.median(times), 3),
        "edit_to_updated_ms_max": round(max(times), 3),
        "build_ms_median": round(statistics.median(builds), 3),
        "verts": len(link.target.data.vertices),
        "faces": len(link.target.data.polygons),
        "runs": RUNS,
    }


def _record(entry):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    data = {}
    if os.path.exists(OUT):
        try:
            with open(OUT) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
    data["blender_%d.%d.%d" % bpy.app.version] = entry
    data["target_simple_ms"] = 100.0
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def test_link_update_latency(engine):
    simple_src = make_shape("Square")
    simple = _measure(simple_src, make_link(simple_src))

    star_src = make_shape("Star200", rings=(_star(),))
    star = _measure(star_src, make_link(star_src))

    blob = tuple((math.cos(2 * math.pi * i / 200) * (100 + 8 * math.sin(10 * math.pi * i / 200)),
                  math.sin(2 * math.pi * i / 200) * (100 + 8 * math.sin(10 * math.pi * i / 200))) for i in range(200))
    blob_src = make_shape("Blob200", rings=(blob,))
    for bp in blob_src.data.splines[0].bezier_points:
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    blob_src.data.update_tag()
    flush()
    blob_link = make_link(blob_src)
    blob = _measure(blob_src, blob_link)

    entry = {"square_4pt": simple, "star_200pt_straight": star, "blob_200pt_curved_res12": blob,
             "python": ".".join(map(str, __import__("sys").version_info[:3]))}
    _record(entry)
    print("LINK PERF", json.dumps(entry))
    assert simple["edit_to_updated_ms_median"] < 100.0
    assert simple["edit_to_updated_ms_max"] < 100.0
