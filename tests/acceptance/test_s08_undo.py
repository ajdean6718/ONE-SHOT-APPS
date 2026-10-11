"""Scenario 8 - undo / redo of a linked edit is atomic across the drawing and the 3D model.

Every operator runs with its undo step, exactly like the UI.  After each undo or redo the 2D source and the 3D
result must describe the same shape *immediately* (no rebuild needed to make them agree), and the engine must
find nothing stale.  Objects are looked up by name after every history jump (memfile undo recreates them).
"""

import bpy
import numpy as np
import pytest

from accsupport import (STEPS, assert_link_consistent, extent, flush, make_3d, mesh_np, move_nodes, new_canvas,
                        node_positions, op, push, redo, select, undo)

SCALE = 0.01


def obj(name):
    return bpy.data.objects[name]


def link_of(link_id):
    return next(lk for lk in bpy.context.scene.seam.links if lk.id == link_id)


def build(**params):
    new_canvas(1000, 800)
    op(bpy.ops.seam.add_shape, kind="RECT", x=300, y=-300, width=200, height=200)
    src = bpy.context.active_object
    src.seam.style.fill_color = (0.8, 0.3, 0.1, 1.0)
    flush()
    params.setdefault("orientation", "FLAT")
    params.setdefault("depth", 0.3)
    link = make_3d(src, "EXTRUDE", **params)
    return src.name, link.id, link.target.name


def observe(src_name, link_id, tgt_name):
    """Everything an artist can see: source nodes, link state, result size, result colour."""
    lk = link_of(link_id)
    src, tgt = obj(src_name), obj(tgt_name)
    return {
        "nodes": node_positions(src),
        "state": lk.state,
        "depth": round(lk.params.depth, 6),
        "extent": tuple(round(float(v), 5) for v in extent(tgt)),
        "fill": tuple(round(float(c), 4) for c in src.seam.style.fill_color),
        "mode": bpy.context.mode,
    }


def expected_extent(state):
    xs = [p[0] for p in state["nodes"]]
    ys = [p[1] for p in state["nodes"]]
    return ((max(xs) - min(xs)) * SCALE, (max(ys) - min(ys)) * SCALE, state["depth"])


def check_agrees(src_name, link_id, tgt_name, *, live=True):
    """2D and 3D describe the same shape right now, without flushing the engine."""
    st = observe(src_name, link_id, tgt_name)
    if live and st["state"] == "LIVE":
        assert st["extent"] == pytest.approx(expected_extent(st), abs=2e-5), st
    return st


def test_one_undo_reverts_drawing_and_model_together(seam, engine):
    src, lid, tgt = build()
    before = observe(src, lid, tgt)
    move_nodes(obj(src), [1, 2], dx=100.0)                                           # Tab, G, Tab = 3 undo steps
    after = observe(src, lid, tgt)
    assert before["extent"][0] == pytest.approx(2.0) and after["extent"][0] == pytest.approx(3.0)

    for _ in range(3):                                                               # undo the whole edit
        assert bpy.ops.ed.undo() == {"FINISHED"}
        # no flush here: memfile undo restores curve and mesh together, they must already agree
        mid = observe(src, lid, tgt) if bpy.context.mode == "OBJECT" else None
        if mid is not None:
            assert mid["extent"] == pytest.approx(expected_extent(mid), abs=2e-5)
    STEPS[0] -= 3
    flush()
    assert observe(src, lid, tgt) == before
    assert_link_consistent(link_of(lid))
    assert link_of(lid).build_count >= 1

    for _ in range(3):
        assert bpy.ops.ed.redo() == {"FINISHED"}
    flush()
    STEPS[0] += 3
    assert observe(src, lid, tgt) == after
    assert_link_consistent(link_of(lid))


def test_walk_back_and_forward_through_a_history(seam, engine):
    src, lid, tgt = build()
    history = [(STEPS[0], observe(src, lid, tgt), "made")]

    def mark(label):
        flush()
        history.append((STEPS[0], check_agrees(src, lid, tgt), label))

    move_nodes(obj(src), [1, 2], dx=100.0)
    mark("widen")
    move_nodes(obj(src), [2, 3], dy=-50.0)
    mark("heighten")
    link_of(lid).params.depth = 0.9
    push("depth")
    mark("depth")
    select(obj(tgt))
    op(bpy.ops.seam.link_pause)
    mark("pause")
    move_nodes(obj(src), [0, 3], dx=-40.0)                                           # edit while paused
    mark("edit while paused")
    select(obj(tgt))
    op(bpy.ops.seam.link_resume)
    mark("resume")
    obj(src).seam.style.fill_color = (0.1, 0.7, 0.2, 1.0)
    push("colour")
    mark("colour")
    assert history[-1][1]["fill"] == pytest.approx((0.1, 0.7, 0.2, 1.0), abs=1e-4)
    assert [h[1]["state"] for h in history] == ["LIVE", "LIVE", "LIVE", "LIVE", "PAUSED", "PAUSED", "LIVE", "LIVE"]

    # back, one recorded state at a time
    for want_steps, want, label in reversed(history[:-1]):
        undo(STEPS[0] - want_steps)
        STEPS[0] = want_steps
        got = observe(src, lid, tgt)
        assert got == want, (label, got, want)
        if want["state"] == "LIVE":
            assert_link_consistent(link_of(lid))
    # forward again
    for want_steps, want, label in history[1:]:
        redo(want_steps - STEPS[0])
        STEPS[0] = want_steps
        got = observe(src, lid, tgt)
        assert got == want, (label, got, want)
        if want["state"] == "LIVE":
            assert_link_consistent(link_of(lid))
    # and live editing still works at the very end of the history
    move_nodes(obj(src), [1, 2], dx=10.0)
    st = check_agrees(src, lid, tgt)
    assert st["state"] == "LIVE" and link_of(lid).build_count >= 2


def test_undo_make_3d_removes_link_and_result_as_a_unit(seam, engine):
    new_canvas(1000, 800)
    op(bpy.ops.seam.add_shape, kind="RECT", x=300, y=-300, width=200, height=200)
    src = bpy.context.active_object
    src_name = src.name
    flush()
    n_objects = len(bpy.context.scene.objects)
    link = make_3d(src, "EXTRUDE", depth=0.4, orientation="FLAT")
    lid, tname, mesh_name = link.id, link.target.name, link.target.data.name
    sig = link.target_signature
    undo(1)
    STEPS[0] -= 1
    assert len(bpy.context.scene.seam.links) == 0
    assert tname not in bpy.context.scene.objects and len(bpy.context.scene.objects) == n_objects
    assert not any(o.seam.role == "RESULT" for o in bpy.context.scene.objects)
    redo(1)
    STEPS[0] += 1
    lk = link_of(lid)
    assert lk.state == "LIVE" and lk.target.name == tname and lk.target.data.name == mesh_name
    assert lk.target_signature == sig and tuple(extent(lk.target)) == pytest.approx((2.0, 2.0, 0.4), abs=1e-5)
    assert_link_consistent(lk)
    # the restored link keeps following its source
    move_nodes(obj(src_name), [1, 2], dx=100.0)
    assert extent(link_of(lid).target)[0] == pytest.approx(3.0, abs=1e-5)


def test_undo_inside_edit_mode_keeps_both_sides_in_step(seam, engine):
    src, lid, tgt = build()
    move_nodes(obj(src), [1, 2], dx=100.0, stay_in_edit_mode=True)
    assert bpy.context.mode == "EDIT_CURVE"
    assert extent(obj(tgt))[0] == pytest.approx(3.0, abs=1e-5)
    op(bpy.ops.transform.translate, value=(0.0, 100.0, 0.0))                         # a second grab, same edit session
    flush()
    assert extent(obj(tgt))[1] == pytest.approx(3.0, abs=1e-5)
    undo(1)                                                                          # Ctrl+Z inside Edit Mode
    STEPS[0] -= 1
    assert bpy.context.mode == "EDIT_CURVE"
    xs = [p[0] for p in node_positions(obj(src))]
    assert max(xs) - min(xs) == pytest.approx(300.0)
    assert tuple(extent(obj(tgt))[:2]) == pytest.approx((3.0, 2.0), abs=1e-5)         # result followed the undo live
    redo(1)
    STEPS[0] += 1
    assert extent(obj(tgt))[1] == pytest.approx(3.0, abs=1e-5)
    op(bpy.ops.object.mode_set, mode="OBJECT")
    flush()
    assert_link_consistent(link_of(lid))
    st = check_agrees(src, lid, tgt)
    assert st["state"] == "LIVE"


def test_undo_detach_makes_the_result_follow_again(seam, engine):
    src, lid, tgt = build()
    select(obj(tgt))
    op(bpy.ops.seam.link_detach)
    flush()
    assert link_of(lid).state == "DETACHED"
    move_nodes(obj(src), [1, 2], dx=100.0)
    assert extent(obj(tgt))[0] == pytest.approx(2.0, abs=1e-5)                       # ignored while detached
    undo(3)                                                                          # the edit
    undo(1)                                                                          # the detach
    STEPS[0] -= 4
    assert link_of(lid).state == "LIVE"
    move_nodes(obj(src), [1, 2], dx=100.0)
    assert extent(obj(tgt))[0] == pytest.approx(3.0, abs=1e-5) and link_of(lid).state == "LIVE"
    assert_link_consistent(link_of(lid))


def test_redo_branch_is_discarded_by_a_new_edit(seam, engine):
    src, lid, tgt = build()
    move_nodes(obj(src), [1, 2], dx=100.0)
    undo(3)
    STEPS[0] -= 3
    move_nodes(obj(src), [1, 2], dx=-50.0)                                           # a different edit after undo
    assert extent(obj(tgt))[0] == pytest.approx(1.5, abs=1e-5)
    assert bpy.ops.ed.redo.poll() is False                                           # the old branch is gone
    undo(3)
    STEPS[0] -= 3
    assert extent(obj(tgt))[0] == pytest.approx(2.0, abs=1e-5)
    assert np.allclose([p[0] for p in node_positions(obj(src))], [-100, 100, 100, -100])


def colour(name):
    return tuple(round(float(c), 4) for c in obj(name).data.materials[0].diffuse_color)


def recolour_and_undo(src, tgt):
    """Change the source fill, let the engine sync the result, undo: the result must follow the 2D fill back."""
    first = colour(tgt)
    obj(src).seam.style.fill_color = (0.1, 0.7, 0.2, 1.0)
    push("colour")
    flush()                                                                          # the debounced engine tick
    assert colour(tgt) == pytest.approx((0.1, 0.7, 0.2, 1.0), abs=1e-4)
    undo(1)
    STEPS[0] -= 1
    assert tuple(obj(src).seam.style.fill_color) == pytest.approx(first, abs=1e-4)
    return first


def test_undo_of_a_fill_colour_change_recolours_the_result(seam, engine):
    src, lid, tgt = build()
    first = recolour_and_undo(src, tgt)
    assert colour(tgt) == pytest.approx(first, abs=1e-4)
    assert first == pytest.approx((0.8, 0.3, 0.1, 1.0), abs=1e-4)
    redo(1)
    STEPS[0] += 1
    assert colour(tgt) == pytest.approx((0.1, 0.7, 0.2, 1.0), abs=1e-4)


def test_undo_of_a_fill_colour_change_after_pause_resume(seam, engine):
    src, lid, tgt = build()
    select(obj(tgt))
    op(bpy.ops.seam.link_pause)
    select(obj(tgt))
    op(bpy.ops.seam.link_resume)
    flush()
    first = recolour_and_undo(src, tgt)
    assert colour(tgt) == pytest.approx(first, abs=1e-4)
