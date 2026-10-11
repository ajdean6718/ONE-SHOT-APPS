"""Scenario 5 - moving source nodes updates only the linked targets; identity, materials, modifiers persist.

Source nodes are moved with Blender's own Edit Mode + Grab (``transform.translate``); the results are
dressed with native modifiers, a user material and object transforms first, and none of that may change.
"""

import bpy
import numpy as np
import pytest

from accsupport import (assert_link_consistent, extent, flush, make_3d, mesh_np, mesh_volume, move_nodes, new_canvas,
                        op, select)


def setup_two_links():
    new_canvas(1000, 800)
    shapes = {}
    for name, x, y in (("A", 200, -300), ("B", 600, -300), ("C", 400, -100)):
        op(bpy.ops.seam.add_shape, kind="RECT", x=x, y=y, width=200, height=200)
        ob = bpy.context.active_object
        ob.name = name
        ob.seam.style.fill_color = (0.2, 0.6, 0.9, 1.0)
        shapes[name] = ob
    flush()
    la = make_3d(shapes["A"], "EXTRUDE", depth=0.3, orientation="FLAT")
    lb = make_3d(shapes["B"], "EXTRUDE", depth=0.3, orientation="FLAT")
    return shapes, la, lb


def dress(target):
    """Downstream work on a result that live updates must never undo: transform, modifiers, user material."""
    select(target)
    assert bpy.ops.transform.translate(value=(5.0, 1.0, 2.0)) == {"FINISHED"}
    assert bpy.ops.transform.rotate(value=0.5, orient_axis="Z") == {"FINISHED"}
    assert bpy.ops.transform.resize(value=(1.5, 1.5, 1.5)) == {"FINISHED"}
    assert bpy.ops.object.modifier_add(type="BEVEL") == {"FINISHED"}
    target.modifiers[-1].width = 0.01
    assert bpy.ops.object.modifier_add(type="ARRAY") == {"FINISHED"}
    target.modifiers[-1].count = 3
    user_mat = bpy.data.materials.new("UserPaint")
    user_mat.diffuse_color = (0.1, 0.9, 0.1, 1.0)
    target.material_slots[1].material = user_mat
    return user_mat


def snapshot(target):
    return {
        "ptr": (target.name, target.as_pointer(), target.data.as_pointer()),
        "matrix": [tuple(r) for r in target.matrix_world],
        "mods": [(m.name, m.type) for m in target.modifiers],
        "array": target.modifiers["Array"].count if "Array" in target.modifiers else None,
        "mats": [m.name for m in target.data.materials],
        "collections": sorted(c.name for c in target.users_collection),
    }


def assert_same(a, b):
    assert a["ptr"] == b["ptr"] and a["mods"] == b["mods"] and a["array"] == b["array"]
    assert a["mats"] == b["mats"] and a["collections"] == b["collections"]
    assert np.allclose(a["matrix"], b["matrix"], atol=1e-6)


def test_only_the_linked_target_updates(seam, engine):
    shapes, la, lb = setup_two_links()
    ta, tb = la.target, lb.target
    user_mat = dress(ta)
    before = snapshot(ta)
    tb_co = mesh_np(tb).copy()
    tb_sig, tb_count = lb.target_signature, lb.build_count
    count_a = la.build_count
    assert tuple(extent(ta)) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)

    # Edit Mode, select the two right-hand nodes of A, grab 100 px to the right
    move_nodes(shapes["A"], [1, 2], dx=100.0)

    assert la.build_count == count_a + 1 and la.state == "LIVE"
    assert tuple(extent(ta)) == pytest.approx((3.0, 2.0, 0.3), abs=1e-5)
    assert mesh_volume(ta) == pytest.approx(3.0 * 2.0 * 0.3, rel=1e-6)
    assert_link_consistent(la)
    # identity, transform, modifiers and the user's material are exactly as the user left them
    assert_same(before, snapshot(ta))
    assert ta.material_slots[1].material == user_mat and tuple(user_mat.diffuse_color) == pytest.approx((0.1, 0.9, 0.1, 1.0))
    # the other result and its link did not rebuild or change by a single vertex
    assert lb.build_count == tb_count and lb.target_signature == tb_sig and lb.state == "LIVE"
    assert np.array_equal(mesh_np(tb), tb_co)
    assert tuple(extent(tb)) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)

    # editing the unlinked shape C triggers nothing at all
    stamp = (la.build_count, lb.build_count, la.target_signature, lb.target_signature)
    move_nodes(shapes["C"], [0, 1], dy=40.0)
    assert (la.build_count, lb.build_count, la.target_signature, lb.target_signature) == stamp

    # now B moves; A keeps its edited shape
    move_nodes(shapes["B"], [2, 3], dy=100.0)
    assert lb.build_count == tb_count + 1 and la.build_count == count_a + 1
    assert tuple(extent(tb)) == pytest.approx((2.0, 3.0, 0.3), abs=1e-5)
    assert tuple(extent(ta)) == pytest.approx((3.0, 2.0, 0.3), abs=1e-5)
    assert_same(before, snapshot(ta))


def test_result_follows_the_drag_while_still_in_edit_mode(seam, engine):
    shapes, la, lb = setup_two_links()
    a, ta = shapes["A"], la.target
    n = la.build_count
    move_nodes(a, [1, 2], dx=50.0, stay_in_edit_mode=True)
    try:
        assert a.mode == "EDIT"
        assert la.build_count == n + 1 and tuple(extent(ta)) == pytest.approx((2.5, 2.0, 0.3), abs=1e-5)
        # a second grab in the same Edit Mode session keeps updating
        assert bpy.ops.transform.translate(value=(0.0, 50.0, 0.0)) == {"FINISHED"}
        flush()
        assert la.build_count == n + 2 and tuple(extent(ta)) == pytest.approx((2.5, 2.5, 0.3), abs=1e-5)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    flush()
    assert la.state == "LIVE" and la.build_count == n + 2                             # leaving edit mode adds nothing
    assert_link_consistent(la)


def test_handle_edits_change_the_curve_and_the_mesh(seam, engine):
    new_canvas(1000, 800)
    op(bpy.ops.seam.add_shape, kind="ELLIPSE", x=300, y=-300, width=200, height=200)
    ell = bpy.context.active_object
    flush()
    link = make_3d(ell, "EXTRUDE", depth=0.2, orientation="FLAT")
    v0 = mesh_volume(link.target)
    assert v0 == pytest.approx(np.pi * 1.0 * 1.0 * 0.2, rel=0.01)                      # r = 1 m
    move_nodes(ell, [0], dx=40.0)                                                    # pull the right-most node outwards
    assert extent(link.target)[0] == pytest.approx(2.4, abs=0.02)
    v1 = mesh_volume(link.target)
    assert v1 > v0 * 1.05
    assert_link_consistent(link)
    n = link.build_count
    # four consecutive grabs of the left-most node (net +20 px) leave a consistent final state
    for dx in (10.0, -25.0, 5.0, 30.0):
        move_nodes(ell, [2], dx=dx)
    assert link.build_count >= n + 1
    assert extent(link.target)[0] == pytest.approx(2.4 - 0.2, abs=0.02)             # left node moved +20 px net
    assert_link_consistent(link)


def test_source_object_transform_rules(seam, engine):
    shapes, la, lb = setup_two_links()
    a, ta = shapes["A"], la.target
    n = la.build_count
    ta_loc = tuple(ta.location)
    select(a)
    assert bpy.ops.transform.translate(value=(300.0, -120.0, 0.0)) == {"FINISHED"}   # moving the drawing around the page
    flush()
    assert la.build_count == n and tuple(ta.location) == ta_loc
    assert tuple(extent(ta)) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)
    assert bpy.ops.transform.resize(value=(2.0, 1.0, 1.0)) == {"FINISHED"}            # a stretch changes the shape
    flush()
    assert la.build_count == n + 1 and tuple(extent(ta)) == pytest.approx((4.0, 2.0, 0.3), abs=1e-5)
    assert bpy.ops.transform.rotate(value=np.pi / 2, orient_axis="Z") == {"FINISHED"}
    flush()
    assert la.build_count == n + 2 and tuple(extent(ta)) == pytest.approx((2.0, 4.0, 0.3), abs=1e-4)
    assert_link_consistent(la)


def test_style_edits_recolour_without_touching_geometry(seam, engine):
    shapes, la, lb = setup_two_links()
    a, ta = shapes["A"], la.target
    co = mesh_np(ta).copy()
    a.seam.style.fill_color = (0.9, 0.1, 0.1, 1.0)
    flush()
    assert tuple(ta.data.materials[0].diffuse_color) == pytest.approx((0.9, 0.1, 0.1, 1.0), abs=1e-6)
    assert np.array_equal(mesh_np(ta), co)
    assert tuple(lb.target.data.materials[0].diffuse_color) == pytest.approx((0.2, 0.6, 0.9, 1.0), abs=1e-6)
    assert_link_consistent(la)
    assert_link_consistent(lb)
