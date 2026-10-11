"""Real-GUI checks for raster undo: real brush strokes, SEAM operators, Ctrl+Z / Ctrl+Shift+Z in Texture Paint mode.

Runs the system ``blender`` binary under Xvfb in a subprocess (so it is the same check from both suites); skipped
when ``xvfb-run`` or the binary is missing.  The add-on is found through BLENDER_USER_SCRIPTS (scripts/test.sh points
it at this tree; conftest does it for a plain pytest run).
"""

import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "gui_paint_undo.py")


def _blender():
    return os.environ.get("SEAM_BLENDER") or shutil.which("blender")


def run_gui(tmp_path, scenario):
    out_path = tmp_path / f"gui_paint_{scenario}.json"
    env = dict(os.environ, SEAM_GUI_OUT=str(out_path), SEAM_GUI_SCENARIO=scenario)
    proc = subprocess.run(["xvfb-run", "-a", "-s", "-screen 0 1600x900x24", _blender(), "--factory-startup",
                           "--enable-event-simulate", "--window-geometry", "0", "0", "1600", "900",
                           "--python", SCRIPT],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=150)
    output = proc.stdout.decode("utf-8", "replace")
    assert out_path.exists(), output[-4000:]
    out = json.loads(out_path.read_text())
    assert "error" not in out, out["error"]
    assert out.get("ok"), out
    assert "Traceback" not in output, output[-4000:]
    return out["m"]


needs_gui = pytest.mark.skipif(not shutil.which("xvfb-run") or not _blender(), reason="needs xvfb-run and a blender binary")


@needs_gui
def test_pixel_operator_is_one_undo_step_and_keeps_strokes(tmp_path):
    m = run_gui(tmp_path, "fill")
    r1 = m["stroke1"]["reds"]
    assert r1 > 100 and m["stroke1"]["blues"] == 0
    assert m["fill"]["blues"] > 500 and m["fill"]["reds"] == r1
    # finding 10: the first Ctrl+Z undoes the Fill (it used to do nothing: an empty second step), strokes stay
    assert m["undo_fill"]["blues"] == 0 and m["undo_fill"]["reds"] == r1
    assert m["redo_fill"]["blues"] == m["fill"]["blues"] and m["redo_fill"]["reds"] == r1
    # finding 7: a stroke made after the Fill survives the undo of the next SEAM operator
    r2 = m["stroke2"]["reds"]
    assert r2 > r1 and m["clear"]["blues"] == 0 and m["clear"]["reds"] == 0
    assert m["undo_clear"]["reds"] == r2 and m["undo_clear"]["blues"] == m["fill"]["blues"]
    assert m["undo_stroke2"]["reds"] == r1 and m["undo_stroke2"]["blues"] == m["fill"]["blues"]
    # selection changes made in paint mode are undoable too (they used to leave an empty image step behind)
    assert m["undo_fill"]["has_sel"] and not m["undo_clear"]["has_sel"] and not m["stroke2"]["has_sel"]
    # further back: the selection and mode steps change nothing, then the Fill goes - strokes are never lost on the way
    walk = [m[f"undo_more_{i}"] for i in range(5)]
    assert any(w["blues"] == 0 and w["reds"] == r1 for w in walk), walk
    for w in walk:
        assert w["reds"] in (0, r1) and w["blues"] in (0, m["fill"]["blues"]), walk
    for k in ("undo_fill", "redo_fill", "undo_clear", "undo_stroke2") + tuple(f"undo_more_{i}" for i in range(5)):
        assert m[k]["warn"] == "", (k, m[k])
    assert m["composite_diff"] == 0
    assert all(r == ["FINISHED"] for r in m["ops"])


@needs_gui
def test_layer_property_edit_in_paint_mode_is_undoable(tmp_path):
    m = run_gui(tmp_path, "props")
    r1 = m["stroke1"]["reds"]
    assert m["opacity"]["opacity"] == pytest.approx(0.3) and m["stroke2"]["opacity"] == pytest.approx(0.3)
    assert m["undo_1"]["reds"] == r1 and m["undo_1"]["opacity"] == pytest.approx(0.3)      # stroke 2 undone
    assert m["undo_2"]["reds"] == r1 and m["undo_2"]["opacity"] == pytest.approx(1.0)      # the opacity edit undone
    assert m["redo_2"]["opacity"] == pytest.approx(0.3)


def run_autosave(tmp_path, scenario):
    """Real auto save (the shortest interval is one minute): returns what the recovered file holds."""
    import glob
    tmp = tmp_path / "autosave"
    tmp.mkdir()
    env = dict(os.environ, SEAM_GUI_OUT=str(tmp_path / "gui.json"), SEAM_GUI_SCENARIO=scenario, SEAM_AUTOSAVE_DIR=str(tmp))
    proc = subprocess.run(["xvfb-run", "-a", "-s", "-screen 0 1600x900x24", _blender(), "--factory-startup",
                           "--enable-event-simulate", "--window-geometry", "0", "0", "1600", "900",
                           "--python", SCRIPT],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    output = proc.stdout.decode("utf-8", "replace")
    out = json.loads((tmp_path / "gui.json").read_text())
    assert "error" not in out, out["error"]
    assert out["m"]["autosaves"], output[-3000:]
    env2 = dict(os.environ, SEAM_GUI_OUT=str(tmp_path / "check.json"),
                SEAM_AUTOSAVE_FILE=sorted(glob.glob(str(tmp / "*autosave*.blend")))[-1])
    subprocess.run([_blender(), "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                    os.path.join(HERE, "gui_paint_autosave_check.py")], env=env2,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120, check=True)
    return out["m"], json.loads((tmp_path / "check.json").read_text())


slow = pytest.mark.skipif(not os.environ.get("SEAM_SLOW_GUI"), reason="waits for Blender's auto save timer (about 70 s); set SEAM_SLOW_GUI=1")


@needs_gui
@slow
@pytest.mark.parametrize("scenario", ["autosave_memfile", "autosave_paint", "autosave_stroke"])
def test_autosave_keeps_raster_pixels(tmp_path, scenario):
    """Blender's auto save writes the last undo (memfile) step without running save_pre: SEAM packs its images
    before it pushes that step.  memfile = operators in Object mode; paint = operators run from Texture Paint mode;
    stroke = a native brush stroke, then leaving paint mode (the stroke is part of the next memfile step)."""
    live, recovered = run_autosave(tmp_path, scenario)
    assert recovered["layers"] == live["layers"], (live, recovered)       # not opaque black, nothing lost
    assert recovered["layers"][-1][:3] == [0, 0, 255]
    assert recovered["greens"] == live["greens"] and (live["greens"] > 100) == (scenario == "autosave_stroke")
    assert recovered["warn"] == ""
    assert recovered["has_sel"] == live["has_sel"]
