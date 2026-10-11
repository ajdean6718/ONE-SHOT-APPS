"""scripts/bench.py runs on this runtime and produces the documented summary (quick mode, small sections)."""

import json
import os
import subprocess
import sys

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_bench(out, sections):
    script = os.path.join(ROOT, "scripts", "bench.py")
    args = ["--quick", "--only", sections, "--out", str(out)]
    binary = bpy.app.binary_path
    if binary and os.path.basename(binary).startswith("blender"):
        cmd = [binary, "-b", "--factory-startup", "--python-exit-code", "1", "--python", script, "--"] + args
    else:
        cmd = [sys.executable, script] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    with open(out) as fh:
        return json.load(fh)


def test_bench_quick_run_produces_measurements(seam, tmp_path):
    data = run_bench(tmp_path / "run.json", "link,doc,svg")
    assert data["failures"] == {}
    rt = data["runtime"]
    assert rt["blender_version"] == bpy.app.version_string and rt["numpy"] and rt["python"]
    assert data["hardware"]["logical_cpus"] >= 1 and data["hardware"]["memory_gib"] > 0
    link = data["sections"]["link"]
    assert set(link) == {"square4", "polygon200", "text", "revolve", "sweep"}
    for kind, row in link.items():
        st = row["flush_after_edit"]
        assert row["edits"] >= 20 and st["n"] == row["edits"], kind
        assert 0.0 < st["min_ms"] <= st["median_ms"] <= st["p95_ms"] <= st["max_ms"], kind
        assert 0.0 < row["engine_build"]["median_ms"] <= st["median_ms"] * 1.5 + 1.0, kind
        assert row["result_vertices"] > 0 and row["result_faces"] > 0
    assert link["polygon200"]["source_points"] == 200 and link["square4"]["source_points"] == 4
    assert link["polygon200"]["result_vertices"] > link["square4"]["result_vertices"]
    doc = data["sections"]["doc"]
    assert set(doc) >= {"10", "100"}
    assert doc["100"]["scene_objects"] > doc["10"]["scene_objects"]
    assert doc["100"]["flush_one_node_edit"]["median_ms"] > doc["100"]["flush_no_change"]["median_ms"]
    assert doc["100"]["builds_per_edit_median"] >= 1
    svg = data["sections"]["svg"]
    assert svg["imported_shapes"] >= 200 and svg["export_outlines_bytes"] > 10000
