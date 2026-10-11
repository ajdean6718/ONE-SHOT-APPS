"""Scenario 9 - save, close, reopen and keep editing the links without recreating them.

Two flavours: in the same process (Save, load an unrelated file = Close, Open) and in a brand new Blender
process of the same runtime (what a user gets after quitting and starting again).
"""

import json

import bpy
import numpy as np
import pytest

from accsupport import (STEPS, assert_link_consistent, extent, flush, make_3d, mesh_np, mesh_volume, move_nodes,
                        new_canvas, op, project_snapshot, run_fresh_process, save_and_reopen, select, undo)


def build_project():
    """A compound plate (extruded, live) and a ring profile (revolved, paused), both dressed downstream."""
    new_canvas(1200, 800)
    op(bpy.ops.seam.add_shape, kind="RECT", x=300, y=-300, width=300, height=200)
    outer = bpy.context.active_object
    outer.seam.style.fill_color = (0.2, 0.5, 0.8, 1.0)
    op(bpy.ops.seam.add_shape, kind="ELLIPSE", x=300, y=-300, width=100, height=100)
    hole = bpy.context.active_object
    flush()
    select(outer, hole)
    op(bpy.ops.seam.compound_make)
    plate = bpy.context.active_object
    plate.seam.style.fill_color = (0.2, 0.5, 0.8, 1.0)
    op(bpy.ops.seam.add_shape, kind="RECT", x=800, y=-300, width=60, height=240)
    prof = bpy.context.active_object
    prof.seam.style.fill_color = (0.9, 0.7, 0.1, 1.0)
    flush()
    a = make_3d(plate, "EXTRUDE", depth=0.4, orientation="UPRIGHT")
    b = make_3d(prof, "REVOLVE", revolve_segments=24, resolution=6)
    # downstream work on the results
    for tgt in (a.target, b.target):
        select(tgt)
        assert bpy.ops.transform.rotate(value=0.3, orient_axis="Z") == {"FINISHED"}
        assert bpy.ops.object.modifier_add(type="BEVEL") == {"FINISHED"}
    user = bpy.data.materials.new("UserShiny")
    user.diffuse_color = (0.9, 0.9, 0.9, 1.0)
    a.target.material_slots[1].material = user
    select(b.target)
    op(bpy.ops.seam.link_pause)
    flush()
    return a.id, b.id, plate.name, prof.name


def test_reopen_in_the_same_process_restores_everything_and_keeps_editing(seam, engine, tmp_path, blend_path):
    a_id, b_id, plate_name, prof_name = build_project()
    before = project_snapshot()
    assert before[a_id]["state"] == "LIVE" and before[b_id]["state"] == "PAUSED"
    assert before[a_id]["modifiers"] == [("Bevel", "BEVEL")]

    assert bpy.ops.wm.save_as_mainfile(filepath=blend_path) == {"FINISHED"}
    bpy.ops.wm.read_homefile(use_empty=True)                                          # "close": an unrelated empty file
    assert len(bpy.context.scene.seam.links) == 0
    assert bpy.ops.wm.open_mainfile(filepath=blend_path) == {"FINISHED"}
    results = flush()

    # nothing was recreated and nothing had to be rebuilt: the file loads exactly as saved
    assert not [r for r in results if r.key.startswith("link:") and r.status == "ok"], results
    after = project_snapshot()
    assert after == before
    for lid in (a_id, b_id):
        lk = next(l for l in bpy.context.scene.seam.links if l.id == lid)
        assert lk.source is not None and lk.target is not None
    assert_link_consistent(next(l for l in bpy.context.scene.seam.links if l.id == a_id))

    # continue: edit the live link's source - only that result changes
    a = next(l for l in bpy.context.scene.seam.links if l.id == a_id)
    b = next(l for l in bpy.context.scene.seam.links if l.id == b_id)
    b_mesh = mesh_np(b.target).copy()
    n = a.build_count
    ex0 = tuple(extent(a.target))
    move_nodes(a.source, [1, 2], dx=100.0)                                            # widen the plate
    assert a.build_count == n + 1 and a.state == "LIVE"
    assert extent(a.target)[0] > ex0[0] + 0.5
    assert np.array_equal(mesh_np(b.target), b_mesh) and b.state == "PAUSED"
    assert_link_consistent(a)
    # the paused one still holds its queued state and catches up on resume
    move_nodes(b.source, [1, 2], dx=40.0)
    assert b.stale and b.build_count == before[b_id]["builds"]
    select(b.target)
    op(bpy.ops.seam.link_resume)
    flush()
    assert b.state == "LIVE" and not b.stale and b.build_count == before[b_id]["builds"] + 1
    assert_link_consistent(b)
    # downstream dressing survived the whole trip
    assert [m.type for m in a.target.modifiers] == ["BEVEL"]
    assert a.target.material_slots[1].material.name == "UserShiny"
    # and a second save / reopen cycle keeps it all
    snap2 = project_snapshot()
    save_and_reopen(blend_path)
    flush()
    assert project_snapshot() == snap2


def test_undo_works_in_the_reopened_session(seam, engine, blend_path):
    a_id, b_id, plate_name, prof_name = build_project()
    save_and_reopen(blend_path)
    flush()
    from accsupport import start_session
    start_session()                                                                   # fresh undo history after opening
    a = next(l for l in bpy.context.scene.seam.links if l.id == a_id)
    src_name = a.source.name
    snap = project_snapshot()[a_id]
    move_nodes(bpy.data.objects[src_name], [1, 2], dx=100.0)
    moved = project_snapshot()[a_id]
    assert moved["mesh"] != snap["mesh"]
    undo(3)
    STEPS[0] = 0
    back = project_snapshot()[a_id]
    assert back["nodes"] == snap["nodes"] and back["mesh"] == snap["mesh"] and back["state"] == "LIVE"
    assert_link_consistent(next(l for l in bpy.context.scene.seam.links if l.id == a_id))


CHILD_REOPEN = '''
from accsupport import project_snapshot, flush, move_nodes, extent, assert_link_consistent
assert bpy.ops.wm.open_mainfile(filepath=BLEND) == {"FINISHED"}
results = flush()
RESULT["rebuilt"] = [r.key for r in results if r.key.startswith("link:") and r.status == "ok"]
RESULT["snapshot"] = project_snapshot()
links = list(bpy.context.scene.seam.links)
a = next(l for l in links if l.recipe == "EXTRUDE")
n = a.build_count
move_nodes(a.source, [1, 2], dx=100.0)
RESULT["edit_builds"] = a.build_count - n
RESULT["edit_state"] = a.state
RESULT["edit_extent_x"] = float(extent(a.target)[0])
assert_link_consistent(a)
assert bpy.ops.wm.save_as_mainfile(filepath=BLEND) == {"FINISHED"}
RESULT["after_edit"] = project_snapshot()
'''


def test_reopen_in_a_brand_new_process(seam, engine, tmp_path, blend_path):
    a_id, b_id, plate_name, prof_name = build_project()
    ex0 = float(extent(next(l for l in bpy.context.scene.seam.links if l.id == a_id).target)[0])
    before = json.loads(json.dumps(project_snapshot()))                                 # as the child reports it
    assert bpy.ops.wm.save_as_mainfile(filepath=blend_path) == {"FINISHED"}

    res = run_fresh_process(CHILD_REOPEN, blend_path, tmp_path)
    assert res["rebuilt"] == []                                                       # a fresh process rebuilds nothing
    assert res["snapshot"] == before                                                  # identical project, same link ids
    assert res["edit_builds"] == 1 and res["edit_state"] == "LIVE"
    assert res["edit_extent_x"] == pytest.approx(ex0 + 1.0, abs=1e-4)                 # +100 px = +1 m on the widened side

    # the file saved by the other process opens here with the edit in place and the link still live
    assert bpy.ops.wm.open_mainfile(filepath=blend_path) == {"FINISHED"}
    flush()
    a = next(l for l in bpy.context.scene.seam.links if l.id == a_id)
    assert a.state == "LIVE" and a.build_count == before[a_id]["builds"] + 1
    assert extent(a.target)[0] == pytest.approx(ex0 + 1.0, abs=1e-4)
    assert json.loads(json.dumps(project_snapshot())) == res["after_edit"]
    assert_link_consistent(a)
    move_nodes(a.source, [0, 3], dx=-50.0)
    assert a.build_count == before[a_id]["builds"] + 2
    assert mesh_volume(a.target) > 0
