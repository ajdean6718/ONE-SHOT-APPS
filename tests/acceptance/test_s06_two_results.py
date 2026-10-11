"""Scenario 6 - two different 3D results from one source, each with independent parameters."""

import math

import bpy
import numpy as np
import pytest

from accsupport import (STEPS, assert_link_consistent, extent, flush, make_3d, mesh_np, mesh_volume, new_canvas, op,
                        results_in_scene, select, undo)


def source():
    new_canvas(1000, 800)
    op(bpy.ops.seam.add_shape, kind="RECT", x=300, y=-300, width=100, height=200)
    ob = bpy.context.active_object
    ob.seam.style.fill_color = (0.3, 0.5, 0.9, 1.0)
    flush()
    return ob


def prism_volume(segments, radius, height):
    return 0.5 * segments * radius * radius * math.sin(2 * math.pi / segments) * height


def test_extrude_and_revolve_from_the_same_source(seam, engine):
    src = source()
    ext = make_3d(src, "EXTRUDE", depth=0.2, orientation="FLAT")
    rev = make_3d(src, "REVOLVE", revolve_segments=64, orientation="UPRIGHT")
    t1, t2 = ext.target, rev.target
    assert ext.source == rev.source == src and t1 != t2 and t1.data != t2.data
    assert ext.id != rev.id and ext.recipe == "EXTRUDE" and rev.recipe == "REVOLVE"
    # extrude: the 100 x 200 px profile is a 1 x 2 m plate, 0.2 m thick
    assert tuple(extent(t1)) == pytest.approx((1.0, 2.0, 0.2), abs=1e-5)
    assert mesh_volume(t1) == pytest.approx(1.0 * 2.0 * 0.2, rel=1e-6)
    # revolve: the same profile lathed around its left edge is a cylinder r = 1 m, h = 2 m (64-gon)
    ex = extent(t2)
    assert tuple(ex) == pytest.approx((2.0, 2.0, 2.0), abs=0.01)                    # diameter 2 m, height 2 m
    assert mesh_volume(t2) == pytest.approx(prism_volume(64, 1.0, 2.0), rel=0.002)
    # results sit side by side on the 3D stage (no overlap) and are plain meshes with their own materials
    xs = sorted(o.location.x for o in results_in_scene())
    assert xs == pytest.approx([0.0, 2.5])
    mats1, mats2 = {m.name for m in t1.data.materials}, {m.name for m in t2.data.materials}
    assert mats1.isdisjoint(mats2) and len(mats1) == len(mats2) == 2


def test_parameters_are_independent(seam, engine):
    src = source()
    a = make_3d(src, "EXTRUDE", depth=0.2, orientation="FLAT")
    op(bpy.ops.seam.duplicate_result)                                               # second result, same settings
    flush()
    b = bpy.context.scene.seam.links[-1]
    assert a.id != b.id and a.target != b.target and b.source == src
    assert {k: getattr(a.params, k) for k in ("depth", "orientation", "units_scale")} == \
           {k: getattr(b.params, k) for k in ("depth", "orientation", "units_scale")}
    assert np.array_equal(mesh_np(a.target), mesh_np(b.target))
    sig_a, n_a, co_a = a.target_signature, a.build_count, mesh_np(a.target).copy()

    # change B only
    b.params.depth = 0.8
    b.params.units_scale = 0.02
    b.params.orientation = "UPRIGHT"
    flush()
    assert tuple(extent(b.target)) == pytest.approx((2.0, 0.8, 4.0), abs=1e-5)       # 2 x (0.8) x 4 m
    assert (a.target_signature, a.build_count) == (sig_a, n_a) and np.array_equal(mesh_np(a.target), co_a)
    assert tuple(extent(a.target)) == pytest.approx((1.0, 2.0, 0.2), abs=1e-5)

    # change A only
    b_sig, b_n, b_co = b.target_signature, b.build_count, mesh_np(b.target).copy()
    a.params.depth = 0.5
    a.params.bevel_depth = 0.05
    flush()
    assert a.build_count == n_a + 1 and extent(a.target)[2] == pytest.approx(0.5, abs=1e-4)
    assert (b.target_signature, b.build_count) == (b_sig, b_n) and np.array_equal(mesh_np(b.target), b_co)
    assert_link_consistent(a)
    assert_link_consistent(b)


def test_source_edit_updates_both_in_their_own_frames(seam, engine):
    from accsupport import move_nodes
    src = source()
    a = make_3d(src, "EXTRUDE", depth=0.2, orientation="FLAT")
    b = make_3d(src, "EXTRUDE", depth=0.4, orientation="UPRIGHT", units_scale=0.02)
    move_nodes(src, [1, 2], dx=100.0)                                               # widen the profile 100 -> 200 px
    assert tuple(extent(a.target)) == pytest.approx((2.0, 2.0, 0.2), abs=1e-5)
    assert tuple(extent(b.target)) == pytest.approx((4.0, 0.4, 4.0), abs=1e-5)
    assert mesh_volume(a.target) == pytest.approx(2.0 * 2.0 * 0.2, rel=1e-6)
    assert mesh_volume(b.target) == pytest.approx(4.0 * 4.0 * 0.4, rel=1e-6)
    assert a.build_count == 2 and b.build_count == 2


def test_pausing_one_result_leaves_the_other_live(seam, engine):
    from accsupport import move_nodes
    src = source()
    a = make_3d(src, "EXTRUDE", depth=0.2, orientation="FLAT")
    b = make_3d(src, "EXTRUDE", depth=0.2, orientation="FLAT")
    select(a.target)
    op(bpy.ops.seam.link_pause)
    flush()
    assert a.state == "PAUSED" and b.state == "LIVE"
    co_a = mesh_np(a.target).copy()
    move_nodes(src, [1, 2], dx=100.0)
    assert np.array_equal(mesh_np(a.target), co_a) and a.stale                      # paused: queued, not applied
    assert tuple(extent(b.target)) == pytest.approx((2.0, 2.0, 0.2), abs=1e-5)       # live one followed
    select(a.target)
    op(bpy.ops.seam.link_resume)
    flush()
    assert a.state == "LIVE" and not a.stale
    # the caught-up result equals a result built from scratch with the same parameters
    c = make_3d(src, "EXTRUDE", depth=0.2, orientation="FLAT")
    assert np.allclose(mesh_np(a.target), mesh_np(c.target), atol=1e-6)
    assert np.allclose(mesh_np(b.target), mesh_np(c.target), atol=1e-6)


def test_deleting_or_unlinking_one_result_keeps_the_other(seam, engine):
    src = source()
    a = make_3d(src, "EXTRUDE", depth=0.2, orientation="FLAT")
    b = make_3d(src, "REVOLVE")
    a_id, b_id, a_name, b_name = a.id, b.id, a.target.name, b.target.name
    # unlink B: its result stays as an ordinary object, A is untouched
    select(b.target)
    op(bpy.ops.seam.link_remove)
    flush()
    assert [lk.id for lk in bpy.context.scene.seam.links] == [a_id]
    assert bpy.data.objects[b_name].seam.role == "NONE" and bpy.data.objects[b_name].type == "MESH"
    # delete A's result with the native delete operator: the link goes away, the source stays
    select(bpy.data.objects[a_name])
    assert bpy.ops.object.delete() == {"FINISHED"}
    flush()
    assert len(bpy.context.scene.seam.links) == 0
    assert bpy.data.objects.get(src.name) is not None and bpy.data.objects.get(b_name) is not None


def test_undo_removes_only_the_latest_result(seam, engine):
    src = source()
    make_3d(src, "EXTRUDE", depth=0.2, orientation="FLAT")
    first = bpy.context.scene.seam.links[0]
    first_id, first_name = first.id, first.target.name
    make_3d(src, "REVOLVE")
    assert len(bpy.context.scene.seam.links) == 2
    undo(1)
    STEPS[0] -= 1
    links = list(bpy.context.scene.seam.links)
    assert [lk.id for lk in links] == [first_id] and links[0].target.name == first_name
    assert len(results_in_scene()) == 1
    assert_link_consistent(links[0])
