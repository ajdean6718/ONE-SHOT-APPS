"""Live updates: editing source nodes rebuilds only the linked results; target identity persists."""

import bpy
import pytest

from helpers import flush
from test_link_support import extent, make_link, make_shape, move_point


def test_edit_updates_only_linked_target_and_keeps_identity(engine):
    a = make_shape("A")
    b = make_shape("B", location=(400, 0))
    la = make_link(a, "EXTRUDE", orientation="FLAT", origin="SOURCE")
    lb = make_link(b, "EXTRUDE", orientation="FLAT", origin="SOURCE")
    ta, tb = la.target, lb.target
    # downstream setup the link must never touch
    ta.location = (5.0, 1.0, 2.0)
    ta.rotation_euler = (0.1, 0.2, 0.3)
    ta.scale = (1.5, 1.5, 1.5)
    bev = ta.modifiers.new("Bevel", "BEVEL")
    bev.width = 0.01
    arr = ta.modifiers.new("Array", "ARRAY")
    arr.count = 3
    user_mat = bpy.data.materials.new("UserPaint")
    ta.data.materials[1] = user_mat  # user replaced the side material
    ident = (ta.name, ta.as_pointer(), ta.data.as_pointer())
    mats = [m.name for m in ta.data.materials]
    flush()
    count_a, count_b = la.build_count, lb.build_count

    move_point(a, 0, 1, dx=100.0)  # (100,-100) -> (200,-100)
    results = flush()

    keys = {r.key: r.status for r in results}
    assert keys.get("link:" + la.id) == "ok"
    assert ("link:" + lb.id) not in keys or keys["link:" + lb.id] == "skipped"
    assert la.build_count == count_a + 1 and lb.build_count == count_b
    assert extent(ta)[0] == pytest.approx(3.0, abs=1e-5)
    assert extent(tb)[0] == pytest.approx(2.0, abs=1e-5)
    # identity, transform, modifiers, material slots untouched
    assert (ta.name, ta.as_pointer(), ta.data.as_pointer()) == ident
    assert bpy.data.objects.get(ident[0]) == ta
    assert tuple(ta.location) == pytest.approx((5.0, 1.0, 2.0))
    assert tuple(ta.rotation_euler) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(ta.scale) == pytest.approx((1.5, 1.5, 1.5))
    assert [m.name for m in ta.modifiers] == ["Bevel", "Array"] and ta.modifiers["Array"].count == 3
    assert [m.name for m in ta.data.materials] == mats
    assert la.state == "LIVE"


def test_unchanged_source_is_not_rebuilt(engine):
    a = make_shape("A")
    la = make_link(a)
    n = la.build_count
    a.location.x += 50  # translation is not part of the link geometry
    flush()
    assert la.build_count == n
    engine.request_revalidate()
    flush()
    assert la.build_count == n


def test_rotation_change_rebuilds(engine):
    a = make_shape("A")
    la = make_link(a, "EXTRUDE", orientation="FLAT")
    a.rotation_euler.z = 0.7853981633974483
    flush()
    assert la.build_count == 2
    assert extent(la.target)[0] == pytest.approx(2.0 * 2 ** 0.5, abs=1e-4)


def test_fill_colour_syncs_link_materials_only(engine):
    a = make_shape("A", color=(0.1, 0.2, 0.3, 1.0))
    la = make_link(a)
    face = la.target.data.materials[0]
    a.seam.style.fill_color = (0.9, 0.5, 0.1, 1.0)
    flush()
    assert tuple(face.diffuse_color) == pytest.approx((0.9, 0.5, 0.1, 1.0), abs=1e-6)


def test_param_change_rebuilds(engine):
    a = make_shape("A")
    la = make_link(a, "EXTRUDE", orientation="FLAT")
    la.params.depth = 0.75
    flush()
    assert extent(la.target)[2] == pytest.approx(0.75, abs=1e-5)


def test_edit_mode_source_updates_live(engine):
    a = make_shape("A")
    la = make_link(a, "EXTRUDE", orientation="FLAT")
    from test_link_support import activate
    activate(a)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        move_point(a, 0, 2, dx=50.0, dy=50.0)  # (100,100) -> (150,150)
        flush()
        assert la.build_count == 2
        ex = extent(la.target)
        assert ex[0] == pytest.approx(2.5, abs=1e-5) and ex[1] == pytest.approx(2.5, abs=1e-5)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
