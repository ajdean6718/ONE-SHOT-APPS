"""Scenario 7 - pause, edit, resume gives the right result; detach freezes the result for good."""

import bpy
import numpy as np
import pytest

from accsupport import (assert_link_consistent, extent, flush, make_3d, mesh_np, mesh_volume, move_nodes, new_canvas,
                        op, save_and_reopen, select)


def build_link(recipe="EXTRUDE", **params):
    new_canvas(1000, 800)
    op(bpy.ops.seam.add_shape, kind="RECT", x=300, y=-300, width=200, height=200)
    src = bpy.context.active_object
    src.seam.style.fill_color = (0.8, 0.3, 0.1, 1.0)
    flush()
    params.setdefault("orientation", "FLAT")
    params.setdefault("depth", 0.3)
    link = make_3d(src, recipe, **params)
    return src, link


def fresh_copy(src, recipe="EXTRUDE", **params):
    """A brand-new result of the source's current state (the reference for 'correct')."""
    params.setdefault("orientation", "FLAT")
    params.setdefault("depth", 0.3)
    return make_3d(src, recipe, **params)


def test_pause_edit_resume_is_correct(seam, engine):
    src, link = build_link()
    tgt = link.target
    select(tgt)
    op(bpy.ops.seam.link_pause)
    flush()
    assert link.state == "PAUSED" and not link.stale
    frozen = mesh_np(tgt).copy()
    n = link.build_count

    # three separate edits while paused: nothing is applied, the link says it is behind
    move_nodes(src, [1, 2], dx=100.0)
    move_nodes(src, [2, 3], dy=100.0)
    src.seam.style.fill_color = (0.1, 0.8, 0.3, 1.0)
    flush()
    assert link.state == "PAUSED" and link.stale and link.build_count == n
    assert np.array_equal(mesh_np(tgt), frozen)
    assert tuple(tgt.data.materials[0].diffuse_color) == pytest.approx((0.8, 0.3, 0.1, 1.0), abs=1e-6)
    assert link.message                                                          # the UI can explain why it is stale

    # a parameter change while paused is queued as well
    link.params.depth = 0.6
    flush()
    assert np.array_equal(mesh_np(tgt), frozen) and link.build_count == n

    select(tgt)
    op(bpy.ops.seam.link_resume)
    flush()
    assert link.state == "LIVE" and not link.stale and link.build_count == n + 1     # one catch-up build, not three
    assert tuple(extent(tgt)) == pytest.approx((3.0, 3.0, 0.6), abs=1e-5)
    assert mesh_volume(tgt) == pytest.approx(3.0 * 3.0 * 0.6, rel=1e-6)
    assert tuple(tgt.data.materials[0].diffuse_color) == pytest.approx((0.1, 0.8, 0.3, 1.0), abs=1e-6)
    assert_link_consistent(link)
    ref = fresh_copy(src, depth=0.6)
    assert np.allclose(mesh_np(tgt), mesh_np(ref.target), atol=1e-6)                 # equals a from-scratch build
    # and live updating continues after the resume
    move_nodes(src, [0, 3], dx=-100.0)
    assert extent(tgt)[0] == pytest.approx(4.0, abs=1e-5) and link.build_count == n + 2


def test_pause_then_undoing_the_edit_resumes_cleanly(seam, engine):
    src, link = build_link()
    select(link.target)
    op(bpy.ops.seam.link_pause)
    co = mesh_np(link.target).copy()
    move_nodes(src, [1, 2], dx=100.0)
    move_nodes(src, [1, 2], dx=-100.0)                                               # edited back to the original
    select(link.target)
    op(bpy.ops.seam.link_resume)
    flush()
    assert link.state == "LIVE" and np.allclose(mesh_np(link.target), co, atol=1e-6)
    assert_link_consistent(link)


def test_paused_state_survives_save_and_reopen(seam, engine, blend_path):
    src, link = build_link()
    link_id, src_name, tgt_name = link.id, src.name, link.target.name
    select(link.target)
    op(bpy.ops.seam.link_pause)
    move_nodes(src, [1, 2], dx=100.0)
    save_and_reopen(blend_path)
    flush()
    link = next(lk for lk in bpy.context.scene.seam.links if lk.id == link_id)
    tgt = bpy.data.objects[tgt_name]
    assert link.state == "PAUSED" and link.stale
    assert tuple(extent(tgt)) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)             # still the old shape
    select(tgt)
    op(bpy.ops.seam.link_resume)
    flush()
    assert link.state == "LIVE" and tuple(extent(tgt)) == pytest.approx((3.0, 2.0, 0.3), abs=1e-5)
    assert bpy.data.objects[src_name] == link.source


def test_detach_freezes_the_result(seam, engine):
    src, link = build_link()
    tgt = link.target
    select(tgt)                                                                      # modifier work must survive detach
    assert bpy.ops.object.modifier_add(type="SOLIDIFY") == {"FINISHED"}
    select(tgt)
    op(bpy.ops.seam.link_detach)
    flush()
    assert link.state == "DETACHED"
    baked = mesh_np(tgt).copy()
    n = link.build_count
    sig = link.target_signature
    # every later change to the source is ignored: geometry, parameters, colour
    move_nodes(src, [1, 2], dx=100.0)
    src.seam.style.fill_color = (0.0, 0.0, 1.0, 1.0)
    link.params.depth = 2.0
    flush()
    assert link.state == "DETACHED" and link.build_count == n and link.target_signature == sig
    assert np.array_equal(mesh_np(tgt), baked)
    assert tuple(extent(tgt)) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)
    assert tuple(tgt.data.materials[0].diffuse_color) == pytest.approx((0.8, 0.3, 0.1, 1.0), abs=1e-6)
    assert [m.type for m in tgt.modifiers] == ["SOLIDIFY"]
    # a detached link offers no pause / resume / rebuild; the result is an ordinary mesh the user may edit
    select(tgt)
    assert not bpy.ops.seam.link_pause.poll() and not bpy.ops.seam.link_resume.poll()
    assert not bpy.ops.seam.link_rebuild.poll()
    select(tgt)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    assert bpy.ops.transform.translate(value=(0.0, 0.0, 1.0)) == {"FINISHED"}
    bpy.ops.object.mode_set(mode="OBJECT")
    flush()
    assert link.state == "DETACHED" and mesh_np(tgt)[:, 2].min() == pytest.approx(baked[:, 2].min() + 1.0, abs=1e-5)


def test_detach_while_paused_keeps_the_last_applied_geometry(seam, engine):
    src, link = build_link()
    tgt = link.target
    select(tgt)
    op(bpy.ops.seam.link_pause)
    move_nodes(src, [1, 2], dx=100.0)
    assert link.stale
    op(bpy.ops.seam.link_detach)
    flush()
    assert link.state == "DETACHED"
    assert tuple(extent(tgt)) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)             # the queued change is NOT applied


def test_deleting_the_source_after_detach_is_harmless(seam, engine):
    src, link = build_link()
    tgt = link.target
    baked = mesh_np(tgt).copy()
    select(tgt)
    op(bpy.ops.seam.link_detach)
    select(src)
    assert bpy.ops.object.delete() == {"FINISHED"}
    flush()
    assert link.state == "DETACHED" and tgt.name in bpy.context.scene.objects
    assert np.array_equal(mesh_np(tgt), baked)
    select(tgt)
    op(bpy.ops.seam.link_remove)                                                     # the dead link can be cleaned up
    flush()
    assert len(bpy.context.scene.seam.links) == 0 and np.array_equal(mesh_np(tgt), baked)
