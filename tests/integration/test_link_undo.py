"""Undo/redo of a linked edit keeps source and result consistent both ways (D8)."""

import bpy
import pytest

from helpers import flush
from test_link_support import extent, link_by_id, make_link, make_shape, move_point


def _consistent(link_id):
    from seam_studio.link import producer
    scene = bpy.context.scene
    link = link_by_id(link_id)
    assert link.source_signature == producer.source_signature(scene, link)
    assert producer.read_mesh_signature(link.target.data) == link.target_signature
    return link


def test_undo_redo_linked_edit(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    link_id, src_name = link.id, src.name
    bpy.ops.ed.undo_push(message="make 3d")
    move_point(src, 0, 1, dx=100.0)
    flush()
    bpy.ops.ed.undo_push(message="edit source")
    assert extent(link.target)[0] == pytest.approx(3.0, abs=1e-5)

    assert bpy.ops.ed.undo() == {"FINISHED"}
    flush()
    link = _consistent(link_id)
    src = bpy.data.objects[src_name]
    assert src.data.splines[0].bezier_points[1].co.x == pytest.approx(100.0)
    assert extent(link.target)[0] == pytest.approx(2.0, abs=1e-5)
    assert link.state == "LIVE"

    assert bpy.ops.ed.redo() == {"FINISHED"}
    flush()
    link = _consistent(link_id)
    assert bpy.data.objects[src_name].data.splines[0].bezier_points[1].co.x == pytest.approx(200.0)
    assert extent(link.target)[0] == pytest.approx(3.0, abs=1e-5)

    # editing after undo/redo keeps working
    move_point(bpy.data.objects[src_name], 0, 1, dx=100.0)
    flush()
    link = _consistent(link_id)
    assert extent(link.target)[0] == pytest.approx(4.0, abs=1e-5)


def test_undo_make_3d_removes_link_and_result(engine):
    src = make_shape("A")
    bpy.ops.ed.undo_push(message="shape")
    link = make_link(src)
    tgt_name = link.target.name
    bpy.ops.ed.undo_push(message="make 3d")
    bpy.ops.ed.undo()
    flush()
    assert len(bpy.context.scene.seam.links) == 0
    assert tgt_name not in bpy.context.scene.objects
    bpy.ops.ed.redo()
    flush()
    assert len(bpy.context.scene.seam.links) == 1
    assert bpy.context.scene.seam.links[0].target.name == tgt_name


def test_undo_param_change(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT", depth=0.2)
    link_id = link.id
    bpy.ops.ed.undo_push(message="make")
    link.params.depth = 0.6
    flush()
    bpy.ops.ed.undo_push(message="depth")
    assert extent(link.target)[2] == pytest.approx(0.6, abs=1e-5)
    bpy.ops.ed.undo()
    flush()
    link = _consistent(link_id)
    assert link.params.depth == pytest.approx(0.2) and extent(link.target)[2] == pytest.approx(0.2, abs=1e-5)
