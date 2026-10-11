"""Link states driven through real operators: pause/resume, detach, invalid geometry, downstream conflicts."""

import bpy
import pytest

from helpers import flush
from test_link_support import activate, extent, make_link, make_shape, mesh_arrays, move_point


def _sig(link):
    from seam_studio.link import producer
    return producer.read_mesh_signature(link.target.data)


def test_pause_edit_resume(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    activate(link.target)
    assert bpy.ops.seam.link_pause() == {"FINISHED"}
    assert link.state == "PAUSED"
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link.state == "PAUSED" and link.stale and link.build_count == 1
    assert extent(link.target)[0] == pytest.approx(2.0, abs=1e-5)
    assert not bpy.ops.seam.link_pause.poll()  # already paused
    assert bpy.ops.seam.link_resume() == {"FINISHED"}
    assert link.state == "LIVE" and not link.stale and link.build_count == 2
    assert extent(link.target)[0] == pytest.approx(3.0, abs=1e-5)
    assert not bpy.ops.seam.link_resume.poll()


def test_resume_without_changes_does_not_rebuild(engine):
    src = make_shape("A")
    link = make_link(src)
    activate(src)
    bpy.ops.seam.link_pause()
    bpy.ops.seam.link_resume()
    assert link.state == "LIVE" and link.build_count == 1


def test_detach_ignores_later_edits(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    activate(src)
    assert bpy.ops.seam.link_detach() == {"FINISHED"}
    assert link.state == "DETACHED"
    sig = _sig(link)
    move_point(src, 0, 1, dx=100.0)
    src.seam.style.fill_color = (0, 1, 0, 1)
    flush()
    engine.request_revalidate()
    flush()
    assert link.state == "DETACHED" and link.build_count == 1 and _sig(link) == sig
    assert tuple(link.target.data.materials[0].diffuse_color) != pytest.approx((0, 1, 0, 1))
    assert not bpy.ops.seam.link_detach.poll() and not bpy.ops.seam.link_resume.poll()


def test_invalid_geometry_keeps_previous_result(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    sig = _sig(link)
    pts = src.data.splines[0].bezier_points
    # swap two corners -> bow tie (self-intersecting)
    for bp, co in ((pts[1], (100, 100)), (pts[2], (100, -100))):
        bp.co = (*co, 0)
        bp.handle_left = bp.handle_right = (*co, 0)
    src.data.update_tag()
    res = [r for r in flush() if r.key.endswith(link.id)]
    assert res and res[-1].status == "error"
    assert link.state == "ERROR" and link.message.startswith("Invalid shape"), link.message
    assert _sig(link) == sig  # previous geometry kept
    # fix it again -> back to LIVE
    for bp, co in ((pts[1], (100, -100)), (pts[2], (100, 100))):
        bp.co = (*co, 0)
        bp.handle_left = bp.handle_right = (*co, 0)
    move_point(src, 0, 2, dx=50.0)
    flush()
    assert link.state == "LIVE" and link.message == ""
    assert extent(link.target)[0] == pytest.approx(2.5, abs=1e-5)


def test_zero_area_is_error(engine):
    src = make_shape("Flat", rings=(((0, 0), (100, 0), (200, 0)),))
    link = make_link(src)
    assert link.state == "ERROR" and link.message.startswith("Invalid shape"), link.message


def _downstream_edit(link):
    me = link.target.data
    me.vertices[0].co.z += 0.25
    me.update()


def test_conflict_then_overwrite(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    _downstream_edit(link)
    edited = _sig(link)
    move_point(src, 0, 1, dx=100.0)
    res = [r for r in flush() if r.key.endswith(link.id)]
    assert res[-1].status == "conflict"
    assert link.state == "CONFLICT" and "edited" in link.message
    assert _sig(link) == edited  # downstream edit preserved
    move_point(src, 0, 1, dx=10.0)  # more source edits still never overwrite
    flush()
    assert link.state == "CONFLICT" and _sig(link) == edited
    activate(link.target)
    assert not bpy.ops.seam.link_rebuild.poll()
    assert bpy.ops.seam.link_overwrite() == {"FINISHED"}
    assert link.state == "LIVE" and not link.force_overwrite
    assert extent(link.target)[0] == pytest.approx(3.1, abs=1e-5)
    assert _sig(link) == link.target_signature


def test_conflict_then_bake(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    _downstream_edit(link)
    edited = _sig(link)
    move_point(src, 0, 1, dx=100.0)
    flush()
    activate(link.target)
    assert bpy.ops.seam.link_detach() == {"FINISHED"}
    assert link.state == "DETACHED" and _sig(link) == edited
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert _sig(link) == edited


def test_conflict_then_branch(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    tgt = link.target
    tgt.modifiers.new("Bevel", "BEVEL")
    _downstream_edit(link)
    edited = _sig(link)
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link.state == "CONFLICT"
    activate(tgt)
    n_before = len(bpy.data.objects)
    assert bpy.ops.seam.link_branch() == {"FINISHED"}
    from seam_studio.link import producer
    copy = bpy.data.objects[tgt.name + " (edited)"]
    assert len(bpy.data.objects) == n_before + 1
    assert producer.read_mesh_signature(copy.data) == edited  # edits kept on the branch
    assert copy.seam.role == "NONE" and copy.seam.id != tgt.seam.id and copy.data != tgt.data
    assert [m.name for m in copy.modifiers] == ["Bevel"]
    assert link.state == "LIVE" and link.target == tgt
    assert extent(tgt)[0] == pytest.approx(3.0, abs=1e-5)
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert producer.read_mesh_signature(copy.data) == edited  # branch is independent
    assert extent(tgt)[0] == pytest.approx(4.0, abs=1e-5)


@pytest.mark.parametrize("kind", ["shape_keys", "vertex_group", "uv_map", "attribute"])
def test_user_data_layers_conflict(engine, kind):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    tgt = link.target
    if kind == "shape_keys":
        tgt.shape_key_add(name="Basis")
        tgt.shape_key_add(name="Smile")
    elif kind == "vertex_group":
        vg = tgt.vertex_groups.new(name="pin")
        vg.add([0, 1], 1.0, "REPLACE")
    elif kind == "uv_map":
        tgt.data.uv_layers.new(name="Lightmap")
    else:
        tgt.data.attributes.new("wear", "FLOAT", "POINT")
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link.state == "CONFLICT", (kind, link.message)
    assert extent(tgt)[0] == pytest.approx(2.0, abs=1e-5)
    activate(tgt)
    assert bpy.ops.seam.link_overwrite() == {"FINISHED"}
    assert link.state == "LIVE" and extent(tgt)[0] == pytest.approx(3.0, abs=1e-5)
    assert tgt.data.shape_keys is None
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link.state == "LIVE" and extent(tgt)[0] == pytest.approx(4.0, abs=1e-5)


def _bow_tie(src, swap=True):
    pts = src.data.splines[0].bezier_points
    corners = ((100, 100), (100, -100)) if swap else ((100, -100), (100, 100))
    for bp, co in zip((pts[1], pts[2]), corners):
        bp.co = (*co, 0)
        bp.handle_left = bp.handle_right = (*co, 0)
    src.data.update_tag()


def test_vertex_group_names_survive_rebuild(engine):
    """Empty user vertex groups hold no per-vertex data, so they do not block a rebuild; their names
    (stored on the mesh, wiped by clear_geometry) are re-created so modifier references keep working."""
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    tgt = link.target
    tgt.vertex_groups.new(name="pin")
    locked = tgt.vertex_groups.new(name="mask")
    locked.lock_weight = True
    tgt.vertex_groups.active_index = 0
    disp = tgt.modifiers.new("Disp", "DISPLACE")
    disp.vertex_group = "pin"
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link.state == "LIVE", link.message
    assert extent(tgt)[0] == pytest.approx(3.0, abs=1e-5)
    assert [g.name for g in tgt.vertex_groups] == ["pin", "mask"]
    assert tgt.vertex_groups["mask"].lock_weight and tgt.vertex_groups.active_index == 0
    assert disp.vertex_group == "pin" and tgt.vertex_groups.get(disp.vertex_group) is not None
    move_point(src, 0, 1, dx=100.0)  # and it keeps following (restored empty groups are no conflict)
    flush()
    assert link.state == "LIVE" and extent(tgt)[0] == pytest.approx(4.0, abs=1e-5)


def test_overwrite_keeps_vertex_group_names(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    tgt = link.target
    vg = tgt.vertex_groups.new(name="pin")
    vg.add([0, 1], 1.0, "REPLACE")
    disp = tgt.modifiers.new("Disp", "DISPLACE")
    disp.vertex_group = "pin"
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link.state == "CONFLICT" and "vertex groups" in link.message
    activate(tgt)
    assert bpy.ops.seam.link_overwrite() == {"FINISHED"}
    assert link.state == "LIVE"
    assert [g.name for g in tgt.vertex_groups] == ["pin"] and disp.vertex_group == "pin"
    assert not any(len(v.groups) for v in tgt.data.vertices)  # weights were the discarded edit


def test_force_overwrite_is_consumed_by_failed_build(engine):
    """Overwrite while the source is invalid: the build fails; the one-shot flag must not survive it,
    so a later downstream edit is still protected (D10)."""
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    _downstream_edit(link)
    _bow_tie(src)
    flush()
    assert link.state == "CONFLICT"
    activate(link.target)
    bpy.ops.seam.link_overwrite()
    assert link.state == "ERROR" and not link.force_overwrite
    _downstream_edit(link)  # user edits the result again
    edited = _sig(link)
    _bow_tie(src, swap=False)  # source valid again
    move_point(src, 0, 2, dx=50.0)
    flush()
    assert link.state == "CONFLICT", link.message
    assert _sig(link) == edited


def test_force_overwrite_is_consumed_when_paused(engine):
    from seam_studio.link import producer
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    link.force_overwrite = True
    link.state = "PAUSED"
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert not link.force_overwrite
    producer._apply(link, "resume")
    _downstream_edit(link)
    move_point(src, 0, 1, dx=10.0)
    flush()
    assert link.state == "CONFLICT"


@pytest.mark.parametrize("kind", ["uv", "material_index"])
def test_uv_and_face_material_edits_conflict(engine, kind):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    me = link.target.data
    if kind == "uv":
        me.uv_layers["UVMap"].data[0].uv = (0.123, 0.456)
    else:
        for p in me.polygons[:2]:
            p.material_index = 1 - p.material_index
    me.update()
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert link.state == "CONFLICT" and "UVs or face materials" in link.message, link.message
    assert extent(link.target)[0] == pytest.approx(2.0, abs=1e-5)
    activate(link.target)
    assert bpy.ops.seam.link_overwrite() == {"FINISHED"}
    assert link.state == "LIVE" and extent(link.target)[0] == pytest.approx(3.0, abs=1e-5)
    move_point(src, 0, 1, dx=100.0)  # the rebuilt result's own UVs/materials are no conflict
    flush()
    assert link.state == "LIVE" and extent(link.target)[0] == pytest.approx(4.0, abs=1e-5)


def test_data_signature_matches_after_every_recipe_build(engine):
    from seam_studio.link import producer
    src = make_shape("A", rings=(((-100, -100), (100, -100), (100, 100), (-100, 100)),
                                 ((-40, -40), (-40, 40), (40, 40), (40, -40))))
    link = make_link(src, "EXTRUDE", bevel_depth=0.02, bevel_resolution=2)
    assert link.target_data_signature == producer.read_data_signature(link.target.data)
    sweep = make_link(make_shape("S", rings=(((0, 0), (100, 0), (100, 100)),), closed=False), "SWEEP")
    assert sweep.state == "LIVE"
    assert sweep.target_data_signature == producer.read_data_signature(sweep.target.data)


def test_target_in_edit_mode_waits(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    activate(link.target)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        move_point(src, 0, 1, dx=100.0)
        res = [r for r in flush() if r.key.endswith(link.id)]
        assert res[-1].status == "waiting"
        assert link.state == "LIVE" and link.stale and "Edit Mode" in link.message
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    flush()  # leaving edit mode re-triggers the waiting build (mesh unchanged => no conflict)
    assert link.state == "LIVE" and not link.stale
    assert extent(link.target)[0] == pytest.approx(3.0, abs=1e-5)


@pytest.mark.parametrize("mode", ["SCULPT", "VERTEX_PAINT", "WEIGHT_PAINT"])
def test_target_in_other_modes_waits(engine, mode):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", orientation="FLAT")
    activate(link.target)
    bpy.ops.object.mode_set(mode=mode)
    try:
        assert link.target.mode == mode
        move_point(src, 0, 1, dx=100.0)
        res = [r for r in flush() if r.key.endswith(link.id)]
        assert res[-1].status == "waiting"
        assert link.state == "LIVE" and link.stale and "Mode" in link.message, link.message
        assert extent(link.target)[0] == pytest.approx(2.0, abs=1e-5)
        with pytest.raises(RuntimeError, match="Leave .*Mode on the result first"):
            bpy.ops.seam.link_overwrite()  # refuses under an active mode session
        assert not link.force_overwrite
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    flush()  # back in Object Mode: the waiting build runs
    if mode == "VERTEX_PAINT":
        # entering Vertex Paint adds a colour attribute (paint layer = downstream data): protected
        assert link.state == "CONFLICT" and "attributes" in link.message, link.message
        assert extent(link.target)[0] == pytest.approx(2.0, abs=1e-5)
        return
    assert link.state == "LIVE" and not link.stale, link.message
    assert extent(link.target)[0] == pytest.approx(3.0, abs=1e-5)


def test_live_updates_off_reports_instead_of_fake_success(engine):
    scene = bpy.context.scene
    scene.seam.engine.live_enabled = False
    try:
        src = make_shape("A")
        activate(src)
        assert bpy.ops.seam.make_3d(recipe="EXTRUDE", orientation="FLAT") == {"FINISHED"}
        link = scene.seam.links[-1]
        assert link.build_count == 0 and link.stale and "Live updates are off" in link.message
        activate(link.target)
        assert bpy.ops.seam.link_rebuild() == {"CANCELLED"}
        assert link.build_count == 0 and "Live updates are off" in link.message
    finally:
        scene.seam.engine.live_enabled = True
    flush()
    assert link.state == "LIVE" and not link.stale and link.build_count == 1 and link.message == ""
    assert extent(link.target)[0] == pytest.approx(2.0, abs=1e-5)
