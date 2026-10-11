"""Scenario 10 - deletion, duplication, invalid geometry, stale updates, missing assets, interrupted work.

The rule for every test here is *no silent data loss*: whatever goes wrong, the user's last good result is still
there, the app says what is wrong (link state + message, operator error) and the path back to a good state works.
"""

import os
import shutil

import bpy
import numpy as np
import pytest

from accsupport import (STEPS, assert_link_consistent, export_png, extent, flush, make_3d, mesh_np, mesh_volume,
                        move_nodes, new_canvas, node_positions, op, png_pixel, project_snapshot, push,
                        run_fresh_process, save_and_reopen, select, start_session, undo)

# ====================================================================== helpers


def build(width=200, height=200, **params):
    new_canvas(1000, 800)
    op(bpy.ops.seam.add_shape, kind="RECT", x=300, y=-300, width=width, height=height)
    src = bpy.context.active_object
    src.seam.style.fill_color = (0.8, 0.3, 0.1, 1.0)
    flush()
    params.setdefault("depth", 0.3)
    params.setdefault("orientation", "FLAT")
    link = make_3d(src, "EXTRUDE", **params)
    return src, link


def obj(name):
    return bpy.data.objects[name]


def link_of(link_id):
    return next(lk for lk in bpy.context.scene.seam.links if lk.id == link_id)


def delete_selected():
    """X / Delete in the 3D view (native operator, with its undo step)."""
    res = bpy.ops.object.delete("EXEC_DEFAULT", True)
    assert res == {"FINISHED"}
    STEPS[0] += 1
    flush()


def png_composite(tmp_path, name="c.png"):
    from seam_studio.seam_core import png
    op(bpy.ops.seam.raster_export_png, filepath=str(tmp_path / name), source="COMPOSITE")
    return png.read_png(str(tmp_path / name))


def raster_doc(width=32, height=24, background="WHITE"):
    start_session()
    op(bpy.ops.seam.raster_new, name="Art", width=width, height=height, background=background)
    flush()
    return bpy.context.scene.seam.rasters[0]


# ====================================================================== deletion

def test_deleting_the_source_keeps_the_result_and_allows_relink_or_bake(seam, engine):
    src, link = build()
    op(bpy.ops.seam.add_shape, kind="RECT", x=700, y=-300, width=300, height=100)
    other = bpy.context.active_object
    other.seam.style.fill_color = (0.2, 0.9, 0.2, 1.0)
    flush()
    lid, tname = link.id, link.target.name
    co = mesh_np(link.target).copy()
    sig = link.target_signature

    select(src)
    delete_selected()
    link = link_of(lid)
    assert link.state == "MISSING_SOURCE" and link.source is None
    assert "deleted" in link.message.lower()
    assert np.array_equal(mesh_np(obj(tname)), co) and link.target_signature == sig       # the result is untouched
    assert obj(tname).seam.role == "RESULT"

    # relink to another shape: the same result object now follows it
    select(obj(tname), obj(other.name), active=obj(tname))
    op(bpy.ops.seam.link_relink, source_name=other.name)
    flush()
    link = link_of(lid)
    assert link.state == "LIVE" and link.source == obj(other.name) and obj(tname) == link.target
    assert tuple(extent(obj(tname))) == pytest.approx((3.0, 1.0, 0.3), abs=1e-5)
    assert_link_consistent(link)

    # a second deleted source can be baked instead
    select(obj(other.name))
    delete_selected()
    link = link_of(lid)
    assert link.state == "MISSING_SOURCE"
    select(obj(tname))
    op(bpy.ops.seam.link_detach)
    flush()
    assert link_of(lid).state == "DETACHED" and tuple(extent(obj(tname))) == pytest.approx((3.0, 1.0, 0.3), abs=1e-5)


def test_undoing_the_deletion_of_the_source_reconnects_it(seam, engine):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    co = mesh_np(link.target).copy()
    select(src)
    delete_selected()
    assert link_of(lid).state == "MISSING_SOURCE"
    undo(1)
    STEPS[0] -= 1
    link = link_of(lid)
    assert link.state == "LIVE" and link.source == obj(sname) and link.message in ("", None)
    assert np.array_equal(mesh_np(obj(tname)), co)
    assert_link_consistent(link)
    move_nodes(obj(sname), [1, 2], dx=100.0)
    assert extent(link_of(lid).target)[0] == pytest.approx(3.0, abs=1e-5)


def test_deleting_the_result_removes_only_the_link_and_undo_restores_it(seam, engine):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    sig = link.target_signature
    select(link.target)
    delete_selected()
    assert len(bpy.context.scene.seam.links) == 0 and tname not in bpy.context.scene.objects
    assert sname in bpy.context.scene.objects and node_positions(obj(sname))[0] == (-100.0, -100.0)
    undo(1)
    STEPS[0] -= 1
    link = link_of(lid)
    assert link.state == "LIVE" and link.target.name == tname and link.target_signature == sig
    assert_link_consistent(link)


def test_removing_a_layer_or_an_artboard_keeps_artwork_and_links(seam, engine):
    from seam_studio.vector import canvas
    src, link = build()
    lid, sname = link.id, src.name
    op(bpy.ops.seam.layer_add, name="Top")
    op(bpy.ops.seam.add_shape, kind="ELLIPSE", x=600, y=-300, width=100, height=100)
    top_shape = bpy.context.active_object.name
    flush()
    before = sorted(o.name for o in canvas.artwork_objects(bpy.context.scene))
    assert len(canvas.layers(bpy.context.scene)) == 2
    op(bpy.ops.seam.layer_remove)                                                # removes the active ("Top") layer
    flush()
    assert len(canvas.layers(bpy.context.scene)) == 1
    assert sorted(o.name for o in canvas.artwork_objects(bpy.context.scene)) == before
    assert top_shape in bpy.context.scene.objects
    assert link_of(lid).state == "LIVE" and link_of(lid).source == obj(sname)
    # artboards: removing one keeps the artwork that sat on it
    op(bpy.ops.seam.artboard_add, width=500, height=500)
    op(bpy.ops.seam.artboard_remove)
    flush()
    assert sorted(o.name for o in canvas.artwork_objects(bpy.context.scene)) == before
    assert len(canvas.artboards(bpy.context.scene)) == 1
    assert_link_consistent(link_of(lid))


def test_raster_layer_and_document_deletion_are_undoable(seam, engine, tmp_path):
    d = raster_doc()
    op(bpy.ops.seam.raster_fill, color=(1.0, 0.0, 0.0, 1.0))
    op(bpy.ops.seam.raster_layer_add, name="Blue")
    op(bpy.ops.seam.raster_fill, color=(0.0, 0.0, 1.0, 1.0))
    flush()
    ref = png_composite(tmp_path, "ref.png")
    assert tuple(ref[3, 3]) == (0, 0, 255, 255)
    op(bpy.ops.seam.raster_layer_remove)
    flush()
    assert tuple(png_composite(tmp_path, "gone.png")[3, 3]) == (255, 0, 0, 255)
    undo(1)
    STEPS[0] -= 1
    assert len(bpy.context.scene.seam.rasters[0].layers) == 2
    assert np.array_equal(png_composite(tmp_path, "back.png"), ref)
    # delete the whole document, then bring it back with Undo: every pixel and both layers return
    op(bpy.ops.seam.raster_remove)
    flush()
    assert len(bpy.context.scene.seam.rasters) == 0 and not any(i.name.startswith("SEAM Art") for i in bpy.data.images)
    undo(1)
    STEPS[0] -= 1
    assert len(bpy.context.scene.seam.rasters) == 1 and len(bpy.context.scene.seam.rasters[0].layers) == 2
    assert np.array_equal(png_composite(tmp_path, "doc_back.png"), ref)


# ====================================================================== duplication

def test_shift_d_of_a_source_is_an_independent_unlinked_copy(seam, engine):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    co, n = mesh_np(link.target).copy(), link.build_count
    select(src)
    assert bpy.ops.object.duplicate("EXEC_DEFAULT", True, linked=False) == {"FINISHED"}
    flush()
    dup = bpy.context.active_object
    assert dup.name != sname and dup.seam.role == "SHAPE" and dup.seam.id and dup.seam.id != obj(sname).seam.id
    assert dup.data != obj(sname).data                                             # own curve data
    assert len(bpy.context.scene.seam.links) == 1 and link_of(lid).source == obj(sname)
    # editing the copy leaves the original, its link and its result alone
    move_nodes(dup, [1, 2], dx=300.0)
    assert node_positions(obj(sname))[1] == (100.0, -100.0)
    assert link_of(lid).build_count == n and np.array_equal(mesh_np(obj(tname)), co)
    # the copy can be linked on its own
    link2 = make_3d(dup, "EXTRUDE", depth=0.3, orientation="FLAT")
    assert link2.id != lid and link2.source == dup and tuple(extent(link2.target)) == pytest.approx((5.0, 2.0, 0.3), abs=1e-5)
    assert tuple(extent(link_of(lid).target)) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)


def test_alt_d_linked_duplicate_does_not_share_the_curve(seam, engine):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    co, n = mesh_np(link.target).copy(), link.build_count
    select(src)
    assert bpy.ops.object.duplicate("EXEC_DEFAULT", True, linked=True) == {"FINISHED"}
    flush()
    dup = bpy.context.active_object
    assert dup.data != obj(sname).data and dup.seam.id != obj(sname).seam.id
    move_nodes(dup, [0, 3], dx=-200.0)
    assert node_positions(obj(sname))[0] == (-100.0, -100.0) and node_positions(dup)[0] == (-300.0, -100.0)
    assert link_of(lid).build_count == n and np.array_equal(mesh_np(obj(tname)), co)
    assert_link_consistent(link_of(lid))


def test_duplicating_a_result_makes_an_independent_object(seam, engine):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    select(link.target)
    assert bpy.ops.object.duplicate("EXEC_DEFAULT", True, linked=False) == {"FINISHED"}
    flush()
    dup = bpy.context.active_object
    assert dup.seam.role == "NONE" and dup.seam.owner_id == "" and dup.data != obj(tname).data
    assert len(bpy.context.scene.seam.links) == 1 and link_of(lid).target == obj(tname)
    frozen = mesh_np(dup).copy()
    # the copy has its own materials: recolouring the source recolours the live result only
    live_mats = {m.name for m in obj(tname).data.materials}
    assert len(dup.data.materials) >= 1 and not ({m.name for m in dup.data.materials} & live_mats)
    assert all(not m.get("seam_link_id") for m in dup.data.materials)
    dup_colours = [tuple(m.diffuse_color) for m in dup.data.materials]
    obj(sname).seam.style.fill_color = (0.1, 0.9, 0.1, 1.0)
    flush()
    assert tuple(obj(tname).data.materials[0].diffuse_color) == pytest.approx((0.1, 0.9, 0.1, 1.0), abs=1e-6)
    assert [tuple(m.diffuse_color) for m in dup.data.materials] == dup_colours
    move_nodes(obj(sname), [1, 2], dx=100.0)
    assert extent(link_of(lid).target)[0] == pytest.approx(3.0, abs=1e-5)
    assert np.array_equal(mesh_np(dup), frozen) and extent(dup)[0] == pytest.approx(2.0, abs=1e-5)
    # Alt+D of the result: same, and the mesh is not shared with the live one
    select(obj(tname))
    assert bpy.ops.object.duplicate("EXEC_DEFAULT", True, linked=True) == {"FINISHED"}
    flush()
    alt = bpy.context.active_object
    assert alt.seam.role == "NONE" and alt.data != obj(tname).data
    assert not ({m.name for m in alt.data.materials} & {m.name for m in obj(tname).data.materials})
    move_nodes(obj(sname), [0, 3], dx=-100.0)
    assert extent(link_of(lid).target)[0] == pytest.approx(4.0, abs=1e-5) and extent(alt)[0] == pytest.approx(3.0, abs=1e-5)
    assert_link_consistent(link_of(lid))


def test_duplicating_source_and_result_together_creates_no_second_link(seam, engine):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    select(obj(sname), obj(tname))
    assert bpy.ops.object.duplicate("EXEC_DEFAULT", True, linked=False) == {"FINISHED"}
    flush()
    new = [o for o in bpy.context.scene.objects if o.name not in (sname, tname) and o.seam.role in ("SHAPE", "NONE")
           and o.type in ("CURVE", "MESH") and o.seam.role != "STROKE" and "Stroke" not in o.name]
    assert len(new) == 2 and len(bpy.context.scene.seam.links) == 1
    ids = [o.seam.id for o in bpy.context.scene.objects if o.seam.id]
    assert len(ids) == len(set(ids))                                                # every object has a unique id
    assert link_of(lid).source == obj(sname) and link_of(lid).target == obj(tname)
    assert_link_consistent(link_of(lid))


def test_duplicating_a_raster_canvas_keeps_the_document(seam, engine, tmp_path):
    d = raster_doc()
    op(bpy.ops.seam.raster_fill, color=(1.0, 0.0, 0.0, 1.0))
    flush()
    ref = png_composite(tmp_path, "ref.png")
    select(d.canvas)
    assert bpy.ops.object.duplicate("EXEC_DEFAULT", True, linked=False) == {"FINISHED"}
    flush()
    assert len(bpy.context.scene.seam.rasters) == 1 and len(bpy.context.scene.seam.rasters[0].layers) == 1
    assert np.array_equal(png_composite(tmp_path, "dup.png"), ref)
    op(bpy.ops.seam.raster_fill, color=(0.0, 1.0, 0.0, 1.0))                      # still paintable
    flush()
    assert tuple(png_composite(tmp_path, "painted.png")[5, 5]) == (0, 255, 0, 255)


# ====================================================================== invalid geometry

def bowtie(src):
    """Drag the two right-hand nodes across each other: the outline now crosses itself."""
    move_nodes(src, [1], dy=200.0)
    move_nodes(src, [2], dy=-200.0)


def test_self_intersecting_source_keeps_the_last_good_result_and_recovers(seam, engine, tmp_path):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    move_nodes(src, [1], dy=200.0)                                                 # still a valid (thin) outline
    good = mesh_np(obj(tname)).copy()
    good_sig = link_of(lid).target_signature
    n = link_of(lid).build_count
    move_nodes(src, [2], dy=-200.0)                                                # now a bow tie
    link = link_of(lid)
    assert link.state == "ERROR"
    assert "crosses itself" in link.message or "Invalid" in link.message
    assert np.array_equal(mesh_np(obj(tname)), good) and link.target_signature == good_sig   # last good mesh kept
    assert link.build_count == n and len(obj(tname).data.polygons) > 0
    # nothing is retried forever: another flush leaves it in the same state
    flush()
    assert link_of(lid).state == "ERROR" and link_of(lid).build_count == n
    # 2D is unaffected by the bad outline: PNG and SVG export still work
    img = export_png(tmp_path / "bad.png")
    assert img.shape[0] == 800 and (img[..., :3] != 255).any()
    op(bpy.ops.seam.svg_export, filepath=str(tmp_path / "bad.svg"))
    assert os.path.getsize(tmp_path / "bad.svg") > 100
    # the path back: move the nodes back, the link returns to LIVE and the mesh is rebuilt correctly
    move_nodes(obj(sname), [2], dy=200.0)
    move_nodes(obj(sname), [1], dy=-200.0)
    link = link_of(lid)
    assert link.state == "LIVE" and link.message == ""
    assert tuple(extent(obj(tname))) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)
    assert mesh_volume(obj(tname)) == pytest.approx(2.0 * 2.0 * 0.3, rel=1e-6)
    assert_link_consistent(link)


def test_error_state_survives_save_and_reopen_and_can_be_fixed(seam, engine, blend_path):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    bowtie(src)
    good = mesh_np(link_of(lid).target).copy()
    assert link_of(lid).state == "ERROR"
    save_and_reopen(blend_path)
    flush()
    link = link_of(lid)
    assert link.state == "ERROR" and link.message and np.array_equal(mesh_np(obj(tname)), good)
    move_nodes(obj(sname), [2], dy=200.0)
    move_nodes(obj(sname), [1], dy=-200.0)
    assert link_of(lid).state == "LIVE" and extent(obj(tname))[0] == pytest.approx(2.0, abs=1e-5)


def test_degenerate_and_open_sources_report_instead_of_breaking(seam, engine):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    co = mesh_np(link.target).copy()
    for bp in obj(sname).data.splines[0].bezier_points:                            # collapse every node onto one point
        bp.co = (0.0, 0.0, 0.0)
        bp.handle_left = bp.handle_right = (0.0, 0.0, 0.0)
    obj(sname).data.update_tag()
    flush()
    link = link_of(lid)
    assert link.state == "ERROR" and "distinct points" in link.message
    assert np.array_equal(mesh_np(obj(tname)), co)
    # an open line cannot be extruded: the link is created in ERROR with a clear message, no crash
    op(bpy.ops.seam.add_shape, kind="LINE", x=300, y=-600, width=200, height=0)
    line = bpy.context.active_object
    flush()
    select(line)
    op(bpy.ops.seam.make_3d, recipe="EXTRUDE")
    flush()
    lk = bpy.context.scene.seam.links[-1]
    assert lk.state == "ERROR" and "closed path" in lk.message
    assert len(lk.target.data.polygons) == 0
    # a revolve of the broken rectangle fails the same way, and removes cleanly
    select(obj(sname))
    op(bpy.ops.seam.make_3d, recipe="REVOLVE")
    flush()
    assert bpy.context.scene.seam.links[-1].state == "ERROR"


def test_pathfinder_on_unusable_input_changes_nothing(seam, engine):
    new_canvas(1000, 800)
    op(bpy.ops.seam.add_shape, kind="RECT", x=300, y=-300, width=200, height=200)
    a = bpy.context.active_object
    bowtie(a)
    op(bpy.ops.seam.add_shape, kind="RECT", x=350, y=-300, width=100, height=100)
    b = bpy.context.active_object
    names = sorted(o.name for o in bpy.context.scene.objects)
    nodes_a = node_positions(obj(a.name))
    select(obj(a.name), obj(b.name))
    with pytest.raises(RuntimeError, match="Pathfinder cannot run"):
        bpy.ops.seam.pathfinder(op="UNION")
    assert sorted(o.name for o in bpy.context.scene.objects) == names               # no shape was consumed or created
    assert node_positions(obj(a.name)) == nodes_a


# ====================================================================== stale updates

def test_result_in_edit_mode_waits_and_catches_up_when_left(seam, engine):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    select(obj(tname))
    op(bpy.ops.object.mode_set, mode="EDIT")
    co = mesh_np(obj(tname)).copy()
    move_nodes_with_target_in_edit_mode(sname)
    link = link_of(lid)
    assert link.state == "LIVE" and link.stale and "Edit Mode" in link.message      # waiting, and it says why
    assert np.array_equal(mesh_np(obj(tname)), co)
    select(obj(tname))
    op(bpy.ops.object.mode_set, mode="OBJECT")
    flush()
    link = link_of(lid)
    assert not link.stale and link.state == "LIVE"
    assert extent(obj(tname))[0] == pytest.approx(3.0, abs=1e-5)
    assert_link_consistent(link)


def move_nodes_with_target_in_edit_mode(source_name):
    """Edit the source's nodes directly while the result sits in Edit Mode (the 3D view is in the other window)."""
    bp = obj(source_name).data.splines[0].bezier_points[1]
    for attr in ("co", "handle_left", "handle_right"):
        getattr(bp, attr).x += 100.0
    obj(source_name).data.update_tag()
    flush()


def edit_result_mesh(tname):
    """The user pushes the whole result up in Edit Mode (G Z 0.5)."""
    select(obj(tname))
    op(bpy.ops.object.mode_set, mode="EDIT")
    assert bpy.ops.mesh.select_all(action="SELECT") == {"FINISHED"}
    op(bpy.ops.transform.translate, value=(0.0, 0.0, 0.5))
    op(bpy.ops.object.mode_set, mode="OBJECT")
    flush()
    return mesh_np(obj(tname)).copy()


@pytest.mark.parametrize("resolution", ["OVERWRITE", "BAKE", "BRANCH"])
def test_downstream_edit_is_never_overwritten_and_every_resolution_keeps_data(seam, engine, resolution):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    edited = edit_result_mesh(tname)
    n = link_of(lid).build_count
    assert link_of(lid).state == "LIVE"                                             # nothing noticed until the next build
    move_nodes(obj(sname), [1, 2], dx=100.0)
    link = link_of(lid)
    assert link.state == "CONFLICT" and "edited" in link.message
    assert np.array_equal(mesh_np(obj(tname)), edited) and link.build_count == n       # user's work is intact
    flush()
    move_nodes(obj(sname), [0, 3], dx=-50.0)                                        # further source edits: still no overwrite
    assert link_of(lid).state == "CONFLICT" and np.array_equal(mesh_np(obj(tname)), edited)

    select(obj(tname))
    if resolution == "OVERWRITE":
        op(bpy.ops.seam.link_overwrite)
        flush()
        link = link_of(lid)
        assert link.state == "LIVE" and tuple(extent(obj(tname))) == pytest.approx((3.5, 2.0, 0.3), abs=1e-5)
        assert mesh_np(obj(tname))[:, 2].min() == pytest.approx(0.0, abs=1e-5)       # the pushed-up edit was dropped on purpose
        assert_link_consistent(link)
    elif resolution == "BAKE":
        op(bpy.ops.seam.link_detach)
        flush()
        assert link_of(lid).state == "DETACHED" and np.array_equal(mesh_np(obj(tname)), edited)
        move_nodes(obj(sname), [1, 2], dx=100.0)
        assert np.array_equal(mesh_np(obj(tname)), edited)                          # and it stays that way
    else:
        op(bpy.ops.seam.link_branch)
        flush()
        branch = [o for o in bpy.context.scene.objects if "(edited)" in o.name]
        assert len(branch) == 1 and np.array_equal(mesh_np(branch[0]), edited)       # the edited mesh lives on, independent
        assert branch[0].seam.role == "NONE"
        link = link_of(lid)
        assert link.state == "LIVE" and tuple(extent(obj(tname))) == pytest.approx((3.5, 2.0, 0.3), abs=1e-5)
        assert_link_consistent(link)


def test_shape_keys_block_an_overwrite_until_the_user_decides(seam, engine):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    select(obj(tname))
    op(bpy.ops.object.shape_key_add, from_mix=False)
    op(bpy.ops.object.shape_key_add, from_mix=False)
    keys = [k.name for k in obj(tname).data.shape_keys.key_blocks]
    co = mesh_np(obj(tname)).copy()
    move_nodes(obj(sname), [1, 2], dx=100.0)
    link = link_of(lid)
    assert link.state == "CONFLICT" and "shape keys" in link.message
    assert np.array_equal(mesh_np(obj(tname)), co) and [k.name for k in obj(tname).data.shape_keys.key_blocks] == keys


def test_edits_made_while_the_engine_is_suspended_apply_once(seam, engine):
    src, link = build()
    lid, sname = link.id, src.name
    n = link.build_count
    with engine.suspend():
        for _ in range(3):
            bp = obj(sname).data.splines[0].bezier_points[1]
            for attr in ("co", "handle_left", "handle_right"):
                getattr(bp, attr).x += 50.0
            obj(sname).data.update_tag()
        assert link_of(lid).build_count == n
    flush()
    link = link_of(lid)
    assert link.build_count == n + 1 and extent(link.target)[0] == pytest.approx(3.5, abs=1e-5)
    assert_link_consistent(link)


def test_turning_live_updates_off_defers_builds_without_losing_them(seam, engine):
    src, link = build()
    lid, sname = link.id, src.name
    co = mesh_np(link.target).copy()
    bpy.context.scene.seam.engine.live_enabled = False
    move_nodes(obj(sname), [1, 2], dx=100.0)
    link = link_of(lid)
    assert link.build_count == 1 and np.array_equal(mesh_np(link.target), co)
    bpy.context.scene.seam.engine.live_enabled = True
    flush()
    link = link_of(lid)
    assert link.build_count == 2 and extent(link.target)[0] == pytest.approx(3.0, abs=1e-5)
    assert_link_consistent(link)


# ====================================================================== missing assets

def write_png(path, rgba8):
    from seam_studio.seam_core import png
    png.write_png(str(path), np.asarray(rgba8, np.uint8))


def test_missing_or_corrupt_png_import_changes_nothing(seam, engine, tmp_path):
    d = raster_doc()
    n_layers, n_images = len(d.layers), len(bpy.data.images)
    with pytest.raises(RuntimeError, match="not found"):
        bpy.ops.seam.raster_import_png(filepath=str(tmp_path / "nope.png"))
    (tmp_path / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\nthis is not a png")
    with pytest.raises(RuntimeError, match="Cannot import PNG"):
        bpy.ops.seam.raster_import_png(filepath=str(tmp_path / "bad.png"))
    (tmp_path / "empty.png").write_bytes(b"")
    with pytest.raises(RuntimeError):
        bpy.ops.seam.raster_import_png(filepath=str(tmp_path / "empty.png"))
    assert len(bpy.context.scene.seam.rasters[0].layers) == n_layers and len(bpy.data.images) == n_images


def test_png_source_file_deleted_after_import_loses_nothing(seam, engine, tmp_path, blend_path):
    d = raster_doc(32, 24, "WHITE")
    pic = np.zeros((10, 10, 4), np.uint8)
    pic[..., 0], pic[..., 3] = 255, 255
    pic[:5, :5] = (0, 255, 0, 255)
    write_png(tmp_path / "pic.png", pic)
    op(bpy.ops.seam.raster_import_png, filepath=str(tmp_path / "pic.png"), placement="FIT_DOWN")
    flush()
    ref = png_composite(tmp_path, "ref.png")
    assert (ref[..., 3] == 255).all() and (ref[..., :3] == (255, 0, 0)).all(axis=-1).sum() >= 50
    os.remove(tmp_path / "pic.png")                                                # the file the layer came from is gone
    save_and_reopen(blend_path)
    flush()
    assert np.array_equal(png_composite(tmp_path, "re.png"), ref)                   # pixels live inside the .blend
    cached = np.array(bpy.context.scene.seam.rasters[0].composite.pixels[:])
    assert cached.size == 32 * 24 * 4 and cached.max() > 0.0


def break_layer_image(index=0):
    """Delete the layer's Image datablock behind SEAM's back (Outliner > Blender File > Images > Delete)."""
    layer = bpy.context.scene.seam.rasters[0].layers[index]
    bpy.data.images.remove(layer.image)
    assert bpy.context.scene.seam.rasters[0].layers[index].image is None


PIXEL_TOOLS = {
    "fill": lambda: op(bpy.ops.seam.raster_fill, color=(0.0, 1.0, 0.0, 1.0)),
    "select_and_fill": lambda: (op(bpy.ops.seam.raster_select_rect, x0=0, y0=0, x1=8, y1=8),
                                op(bpy.ops.seam.raster_fill, color=(0.0, 1.0, 0.0, 1.0))),
    "clear": lambda: op(bpy.ops.seam.raster_clear),
    "duplicate": lambda: op(bpy.ops.seam.raster_layer_duplicate),
    "mask_add_and_apply": lambda: (op(bpy.ops.seam.raster_mask_add, fill="WHITE"), op(bpy.ops.seam.raster_mask_apply)),
    "resize": lambda: op(bpy.ops.seam.raster_resize_canvas, width=40, height=30, scale_content=False),
    "export_layer": None,                      # handled in the test: it needs a path
}


@pytest.mark.parametrize("tool", sorted(PIXEL_TOOLS))
def test_a_deleted_layer_image_is_recreated_and_the_pixel_tools_keep_working(seam, engine, tmp_path, tool):
    """Regression: deleting a layer's Image in the Outliner made Fill, Clear, Duplicate, Resize and Export fail with an
    AttributeError traceback.  The layer image is now recreated blank (transparent) and the tool runs."""
    d = raster_doc()
    op(bpy.ops.seam.raster_fill, color=(1.0, 0.0, 0.0, 1.0))
    op(bpy.ops.seam.raster_layer_add, name="Two")
    op(bpy.ops.seam.raster_fill, color=(0.0, 0.0, 1.0, 1.0))
    bpy.context.scene.seam.rasters[0].active_layer_index = 0
    flush()
    break_layer_image(0)
    flush()
    comp = png_composite(tmp_path, "broken.png")                                    # the document still composites
    assert tuple(comp[3, 3]) == (0, 0, 255, 255) and tuple(comp[20, 30]) == (0, 0, 255, 255)
    if tool == "export_layer":
        op(bpy.ops.seam.raster_export_png, filepath=str(tmp_path / "layer.png"), source="LAYER")
    else:
        PIXEL_TOOLS[tool]()
    flush()
    d = bpy.context.scene.seam.rasters[0]
    assert all(ly.image is not None and tuple(ly.image.size) == (d.width, d.height) for ly in d.layers)
    if tool in ("fill", "select_and_fill"):
        from seam_studio.seam_core import png
        op(bpy.ops.seam.raster_export_png, filepath=str(tmp_path / "l0.png"), source="LAYER")
        lay = png.read_png(str(tmp_path / "l0.png"))
        assert tuple(lay[3, 3]) == (0, 255, 0, 255)
        assert (lay[..., 3] == 255).all() if tool == "fill" else (lay[..., 3] > 0).sum() == 64


def test_merging_into_a_layer_whose_image_was_deleted(seam, engine, tmp_path):
    d = raster_doc()
    op(bpy.ops.seam.raster_fill, color=(1.0, 0.0, 0.0, 1.0))
    op(bpy.ops.seam.raster_layer_add, name="Two")
    op(bpy.ops.seam.raster_select_rect, x0=0, y0=0, x1=16, y1=24)
    op(bpy.ops.seam.raster_fill, color=(0.0, 0.0, 1.0, 1.0))
    flush()
    break_layer_image(0)
    op(bpy.ops.seam.raster_layer_merge_down)
    flush()
    d = bpy.context.scene.seam.rasters[0]
    comp = png_composite(tmp_path, "merged.png")
    assert len(d.layers) == 1
    assert tuple(comp[3, 3]) == (0, 0, 255, 255)                                   # the upper layer's blue half survived
    assert tuple(comp[3, 20]) == (255, 255, 255, 255)                              # the lost red is now just the white page


def test_missing_font_file_falls_back_and_keeps_the_text(seam, engine, tmp_path, blend_path):
    from seam_studio.vector import canvas
    new_canvas(800, 600)
    ttf = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(ttf):
        pytest.skip("DejaVu Bold not installed")
    mine = str(tmp_path / "Custom.ttf")
    shutil.copy(ttf, mine)
    op(bpy.ops.seam.add_text, body="Hello", x=100, y=-200, size=60)
    t = bpy.context.active_object
    t.data.font = bpy.data.fonts.load(mine)
    flush()
    name = t.name
    ref_w = canvas.canvas_bounds(t)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    os.remove(mine)                                                                 # the user's font file disappears
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    flush()
    t = bpy.data.objects[name]
    assert t.data.body == "Hello" and t.data.font is not None
    assert t.data.font.filepath.endswith("Custom.ttf")                              # still points at the missing file ...
    b = canvas.canvas_bounds(t)
    assert b[2] > b[0] and b[3] > b[1] and ref_w[2] > ref_w[0]                      # ... and the text still has outlines
    op(bpy.ops.seam.svg_export, filepath=str(tmp_path / "t.svg"))                   # export works with the fallback font
    assert os.path.getsize(tmp_path / "t.svg") > 200
    shutil.copy(ttf, mine)                                                          # restoring the file restores the look
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    flush()
    t = bpy.data.objects[name]
    assert t.data.font.filepath.endswith("Custom.ttf") and os.path.exists(bpy.path.abspath(t.data.font.filepath))
    b2 = canvas.canvas_bounds(t)
    assert (b2[2] - b2[0]) == pytest.approx(ref_w[2] - ref_w[0], rel=0.01)           # same glyph widths as before the loss


def test_svg_import_problems_do_not_leave_partial_results(seam, engine, tmp_path):
    new_canvas(800, 600)
    before = sorted(o.name for o in bpy.context.scene.objects)
    (tmp_path / "broken.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><rect')
    with pytest.raises(RuntimeError, match="Invalid SVG"):
        bpy.ops.seam.svg_import(filepath=str(tmp_path / "broken.svg"))
    with pytest.raises(RuntimeError):
        bpy.ops.seam.svg_import(filepath=str(tmp_path / "does_not_exist.svg"))
    assert sorted(o.name for o in bpy.context.scene.objects) == before
    (tmp_path / "ref.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 100 100">'
        '<image xlink:href="missing.png" x="0" y="0" width="50" height="50"/>'
        '<rect x="10" y="10" width="30" height="30" fill="#ff0000"/></svg>')
    op(bpy.ops.seam.svg_import, filepath=str(tmp_path / "ref.svg"))
    flush()
    shapes = [o for o in bpy.context.scene.objects if o.seam.role == "SHAPE"]
    assert len(shapes) == 1 and tuple(round(c, 2) for c in shapes[0].seam.style.fill_color[:3]) == (1.0, 0.0, 0.0)


# ====================================================================== interrupted work

def test_saving_with_an_update_still_pending_stores_a_consistent_file(seam, engine, blend_path):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    bp = obj(sname).data.splines[0].bezier_points[1]                                # edit, but never reach the engine tick
    for attr in ("co", "handle_left", "handle_right"):
        getattr(bp, attr).x += 100.0
    obj(sname).data.update_tag()
    assert link_of(lid).build_count == 1
    assert bpy.ops.wm.save_as_mainfile(filepath=blend_path) == {"FINISHED"}
    assert bpy.ops.wm.open_mainfile(filepath=blend_path) == {"FINISHED"}
    assert tuple(extent(obj(tname))) == pytest.approx((3.0, 2.0, 0.3), abs=1e-5)       # already in step before any flush
    flush()
    assert_link_consistent(link_of(lid))
    assert node_positions(obj(sname))[1] == (200.0, -100.0)


def test_saving_while_the_source_is_in_edit_mode(seam, engine, blend_path):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    move_nodes(obj(sname), [1, 2], dx=100.0, stay_in_edit_mode=True)
    assert bpy.context.mode == "EDIT_CURVE"
    assert bpy.ops.wm.save_as_mainfile(filepath=blend_path) == {"FINISHED"}
    assert bpy.ops.wm.open_mainfile(filepath=blend_path) == {"FINISHED"}
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    flush()
    assert node_positions(obj(sname))[1] == (200.0, -100.0)                          # the edit made it into the file
    assert tuple(extent(obj(tname))) == pytest.approx((3.0, 2.0, 0.3), abs=1e-5)
    assert_link_consistent(link_of(lid))


def test_saving_while_the_result_is_in_edit_mode(seam, engine, blend_path):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    select(obj(tname))
    op(bpy.ops.object.mode_set, mode="EDIT")
    move_nodes_with_target_in_edit_mode(sname)
    assert link_of(lid).stale
    assert bpy.ops.wm.save_as_mainfile(filepath=blend_path) == {"FINISHED"}
    assert bpy.ops.wm.open_mainfile(filepath=blend_path) == {"FINISHED"}
    flush()
    link = link_of(lid)
    assert link.stale and node_positions(obj(sname))[1] == (200.0, -100.0)           # the pending update is remembered
    if bpy.context.mode != "OBJECT":
        select(obj(tname))
        bpy.ops.object.mode_set(mode="OBJECT")
    flush()
    link = link_of(lid)
    assert not link.stale and tuple(extent(obj(tname))) == pytest.approx((3.0, 2.0, 0.3), abs=1e-5)
    assert_link_consistent(link)


def test_recovery_file_and_save_backup_restore_the_last_good_state(seam, engine, tmp_path, blend_path):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    # 1) an auto-save style copy can be recovered
    autosave = str(tmp_path / "autosave.blend")
    assert bpy.ops.wm.save_as_mainfile(filepath=autosave, copy=True) == {"FINISHED"}
    bpy.ops.wm.read_homefile(use_empty=True)
    assert len(bpy.context.scene.seam.links) == 0
    assert bpy.ops.wm.recover_auto_save(filepath=autosave) == {"FINISHED"}
    flush()
    link = link_of(lid)
    assert link.state == "LIVE" and tuple(extent(obj(tname))) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)
    assert_link_consistent(link)
    # 2) overwriting a save keeps the previous one as .blend1; a damaged main file is refused and the backup opens
    assert bpy.ops.wm.save_as_mainfile(filepath=blend_path) == {"FINISHED"}
    move_nodes(obj(sname), [1, 2], dx=100.0)
    assert bpy.ops.wm.save_as_mainfile(filepath=blend_path) == {"FINISHED"}
    assert os.path.exists(blend_path + "1")
    data = open(blend_path, "rb").read()
    with open(blend_path, "wb") as fh:
        fh.write(data[: len(data) // 2])                                            # simulate a torn write
    with pytest.raises(RuntimeError, match="Loading"):
        bpy.ops.wm.open_mainfile(filepath=blend_path)
    assert len(bpy.context.scene.seam.links) == 1                                    # the open session is untouched
    assert bpy.ops.wm.open_mainfile(filepath=blend_path + "1") == {"FINISHED"}
    flush()
    link = link_of(lid)
    assert tuple(extent(obj(tname))) == pytest.approx((2.0, 2.0, 0.3), abs=1e-5)     # the state before the second save
    assert node_positions(obj(sname))[1] == (100.0, -100.0)
    assert_link_consistent(link)


CHILD_CRASH = '''
from accsupport import flush, move_nodes
assert bpy.ops.wm.open_mainfile(filepath=BLEND) == {"FINISHED"}
flush()
link = bpy.context.scene.seam.links[0]
move_nodes(link.source, [1, 2], dx=100.0)
assert bpy.ops.wm.save_as_mainfile(filepath=BLEND) == {"FINISHED"}                  # saved: source +100 px
move_nodes(link.source, [1, 2], dx=100.0)                                           # not saved
sys.stdout.flush()
os._exit(9)                                                                          # the process dies here
'''


def test_a_process_that_dies_after_a_save_leaves_the_saved_state(seam, engine, tmp_path, blend_path):
    src, link = build()
    lid, sname, tname = link.id, src.name, link.target.name
    assert bpy.ops.wm.save_as_mainfile(filepath=blend_path) == {"FINISHED"}
    run_fresh_process(CHILD_CRASH, blend_path, tmp_path, crash=True)
    assert bpy.ops.wm.open_mainfile(filepath=blend_path) == {"FINISHED"}
    flush()
    link = link_of(lid)
    assert link.state == "LIVE" and tuple(extent(obj(tname))) == pytest.approx((3.0, 2.0, 0.3), abs=1e-5)
    assert node_positions(obj(sname))[1] == (200.0, -100.0)                          # exactly the saved edit, not the lost one
    assert_link_consistent(link)


def test_a_native_stroke_that_was_never_noticed_survives_save_and_reopen(seam, engine, tmp_path, blend_path):
    """The paint timer did not get to see the stroke (process ended / background): the pixels must still be in the
    file, in the exported picture and in the canvas the user sees after reopening."""
    from seam_studio.paint import rasterdoc as rd
    d = raster_doc()
    op(bpy.ops.seam.raster_fill, color=(0.0, 0.0, 1.0, 1.0))
    flush()
    img = d.layers[0].image
    buf = np.empty(32 * 24 * 4, np.float32)
    img.pixels.foreach_get(buf)
    buf.reshape(24, 32, 4)[:8, :8] = (1.0, 0.0, 0.0, 1.0)                           # bottom-left 8 x 8 (image rows are bottom-up)
    img.pixels.foreach_set(buf)
    img.update()
    save_and_reopen(blend_path)
    flush()
    comp = png_composite(tmp_path, "re.png")
    assert (comp[16:24, 0:8] == (255, 0, 0, 255)).all() and tuple(comp[2, 2]) == (0, 0, 255, 255)
    shown = rd.read_image_srgb8(bpy.context.scene.seam.rasters[0].composite)
    assert np.array_equal(shown, comp)                                                # what the canvas shows == the export
