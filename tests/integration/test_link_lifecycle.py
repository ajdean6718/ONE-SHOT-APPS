"""Lifecycle: delete source/target, duplicate source/target, re-point after undo-like restores, relink."""

import bpy
import pytest

from helpers import flush
from test_link_support import activate, extent, link_by_id, make_link, make_shape, move_point


def test_delete_source_keeps_target_and_goes_missing(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    link_id, tgt_name, sid = link.id, link.target.name, src.seam.id
    activate(src)
    assert bpy.ops.object.delete() == {"FINISHED"}
    flush()
    link = link_by_id(link_id)
    assert link.state == "MISSING_SOURCE" and link.source is None and link.source_id == sid
    tgt = bpy.data.objects[tgt_name]
    assert link.target == tgt and len(tgt.data.vertices) > 0
    assert "deleted" in link.message
    # relink to another shape
    other = make_shape("B", rings=(((0, 0), (300, 0), (300, 100), (0, 100)),))
    activate(tgt)
    assert bpy.ops.seam.link_relink(source_name=other.name) == {"FINISHED"}
    assert link.state == "LIVE" and link.source == other and link.source_id == other.seam.id
    assert extent(tgt)[0] == pytest.approx(3.0, abs=1e-5)


def test_missing_source_can_be_baked(engine):
    src = make_shape("A")
    link = make_link(src)
    tgt = link.target
    bpy.data.objects.remove(src, do_unlink=True)
    flush()
    assert link.state == "MISSING_SOURCE"
    activate(tgt)
    assert bpy.ops.seam.link_detach() == {"FINISHED"}
    assert link.state == "DETACHED"


def test_delete_target_removes_link(engine):
    src = make_shape("A")
    a = make_link(src)
    b = make_link(src)
    b_id = b.id
    activate(a.target)
    assert bpy.ops.object.delete() == {"FINISHED"}
    flush()
    links = bpy.context.scene.seam.links
    assert [lk.id for lk in links] == [b_id]
    bpy.data.objects.remove(links[0].target, do_unlink=True)
    flush()
    assert len(bpy.context.scene.seam.links) == 0
    move_point(src, 0, 0, dx=5)
    flush()  # nothing to do, no errors


def test_duplicate_source_gets_new_id_and_is_not_linked(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    activate(src)
    assert bpy.ops.object.duplicate() == {"FINISHED"}
    copy = bpy.context.active_object
    assert copy != src
    flush()
    assert copy.seam.id and copy.seam.id != src.seam.id and src.seam.id == link.source_id
    assert link.source == src
    n = link.build_count
    move_point(copy, 0, 1, dx=100.0)
    flush()
    assert link.build_count == n
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link.build_count == n + 1


@pytest.mark.parametrize("linked", [False, True])
def test_duplicate_target_becomes_independent(engine, linked):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    tgt = link.target
    activate(tgt)
    assert bpy.ops.object.duplicate(linked=linked) == {"FINISHED"}
    copy = bpy.context.active_object
    flush()
    assert copy != tgt
    assert copy.seam.role == "NONE" and copy.seam.owner_id == "" and copy.seam.id != tgt.seam.id
    assert tgt.seam.role == "RESULT" and tgt.seam.id == link.target_id
    assert copy.data != tgt.data  # even a linked duplicate gets its own mesh
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert extent(tgt)[0] == pytest.approx(3.0, abs=1e-5)
    assert extent(copy)[0] == pytest.approx(2.0, abs=1e-5)
    assert len(bpy.context.scene.seam.links) == 1


def test_source_pointer_restored_by_id(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    link.source = None  # e.g. restored data without the pointer
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link.source == src and link.state == "LIVE"
    assert extent(link.target)[0] == pytest.approx(3.0, abs=1e-5)


def test_undo_of_source_delete_restores_live_link(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    link_id, src_name = link.id, src.name
    bpy.ops.ed.undo_push(message="before delete")
    activate(src)
    bpy.ops.object.delete()
    bpy.ops.ed.undo_push(message="delete")
    flush()
    assert link_by_id(link_id).state == "MISSING_SOURCE"
    bpy.ops.ed.undo()
    flush()
    link = link_by_id(link_id)
    src = bpy.data.objects[src_name]
    assert link.source == src and link.state == "LIVE"
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link_by_id(link_id).build_count == 2
