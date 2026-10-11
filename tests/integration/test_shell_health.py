"""Schema migration and document health checks (shell)."""

import os
import shutil

import bpy
import pytest

from helpers import make_curve, save_and_reload


def _health():
    from seam_studio.ui import health
    return health


def _bad_link(scene, name="Link A", state="LIVE"):
    link = scene.seam.links.add()
    link.id = "a" * 16
    link.name = name
    link.state = state
    return link


# ------------------------------------------------------------------ migration

def test_untouched_scene_is_not_stamped(seam):
    scene = bpy.context.scene
    assert not scene.seam.is_property_set("schema_version")
    assert _health().migrate_scene(scene) == []
    assert not scene.seam.is_property_set("schema_version")  # opening foreign files must not dirty them


def test_migrate_0_to_1_assigns_ids_and_marks_canvas(seam):
    from seam_studio.vector import canvas
    scene = bpy.context.scene
    canvas.ensure_canvas(scene)
    ob = make_curve("legacy")
    ob.seam.id = ""          # legacy object: role set, id missing
    scene.seam.doc.canvas_ready = False
    scene.seam.schema_version = 0
    applied = _health().migrate_scene(scene)
    assert applied and applied[0].startswith("schema 0 -> 1")
    assert scene.seam.schema_version == 1
    assert len(ob.seam.id) == 16
    assert scene.seam.doc.canvas_ready is True
    assert _health().migrate_scene(scene) == []  # idempotent


def test_migration_chain_runs_each_step_in_order(seam, monkeypatch):
    health = _health()
    scene = bpy.context.scene
    calls = []
    monkeypatch.setattr(health, "SCHEMA_VERSION", 3)
    monkeypatch.setitem(health.MIGRATIONS, 1, lambda s: calls.append(1) or "one")
    monkeypatch.setitem(health.MIGRATIONS, 2, lambda s: calls.append(2) or "two")
    scene.seam.schema_version = 1
    applied = health.migrate_scene(scene)
    assert calls == [1, 2]
    assert applied == ["schema 1 -> 2: one", "schema 2 -> 3: two"]
    assert scene.seam.schema_version == 3


def test_newer_schema_is_kept_and_reported(seam):
    scene = bpy.context.scene
    scene.seam.schema_version = 99
    assert _health().migrate_scene(scene) == []
    assert scene.seam.schema_version == 99
    rep = _health().health_report(scene)
    assert [i.code for i in rep.issues] == ["schema_newer"]


def test_load_post_migrates_on_open(seam, blend_path):
    health = _health()
    scene = bpy.context.scene
    scene.seam.schema_version = 0
    ob = make_curve("legacy")
    ob.seam.id = ""
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.wm.open_mainfile(filepath=blend_path)   # persistent load_post handler runs
    scene = bpy.context.scene
    assert scene.seam.schema_version == 1
    assert len(bpy.data.objects["legacy"].seam.id) == 16
    assert health.last_report is not None and health.last_report.migrated


# ------------------------------------------------------------------ report

def test_clean_document_has_no_problems(seam):
    from seam_studio.vector import canvas
    canvas.ensure_canvas(bpy.context.scene)
    make_curve("ok")
    rep = _health().health_report()
    assert rep.problems == [] and rep.summary() == "no problems"


def test_missing_link_source_and_target(seam):
    scene = bpy.context.scene
    _bad_link(scene, "Gear")
    rep = _health().health_report(scene)
    codes = sorted(i.code for i in rep.problems)
    assert codes == ["link_missing_source", "link_missing_target"]
    assert "Gear" in rep.problems[0].message
    # detached links are intentionally source-less
    scene.seam.links[0].state = "DETACHED"
    assert _health().health_report(scene).problems == []


def test_unpacked_and_missing_images(seam, tmp_path):
    scene = bpy.context.scene
    doc = scene.seam.rasters.add()
    doc.id, doc.name = "d" * 16, "Canvas"

    def layer(name, image=None):
        lay = doc.layers.add()
        lay.name = name
        lay.image = image
        return lay

    ext = tmp_path / "tex.png"
    src = bpy.data.images.new("tmp", 4, 4)
    src.filepath_raw = str(ext)
    src.file_format = "PNG"
    src.save()
    packed = bpy.data.images.new("SEAM Canvas Packed", 8, 8)
    packed.pixels[0] = 0.5
    packed.pack()
    painted = bpy.data.images.new("SEAM Canvas Painted", 8, 8)
    painted.pixels[0] = 0.5                      # dirty, not packed yet (packs on save)
    blank = bpy.data.images.new("SEAM Canvas Blank", 8, 8)   # clean generated image: nothing to lose
    external = bpy.data.images.load(str(ext))
    layer("Packed", packed)
    layer("Painted", painted)
    layer("Blank", blank)
    layer("Linked", external)
    layer("Gone")                                 # pointer lost
    rep = _health().health_report(scene)
    problems = {i.code: i for i in rep.problems}
    assert set(problems) == {"image_unpacked", "image_missing"}
    assert "Linked" in problems["image_unpacked"].message
    assert "Gone" in problems["image_missing"].message
    assert [i.code for i in rep.issues if i.severity == "INFO"] == ["image_pending_pack"]
    # the external file disappears -> an error, not just a warning
    os.remove(ext)
    rep = _health().health_report(scene)
    assert sorted(i.code for i in rep.problems) == ["image_missing", "image_missing"]
    # generic image with a missing file (not a SEAM layer)
    stray = bpy.data.images.new("tmp2", 4, 4)
    stray_path = tmp_path / "stray.png"
    stray.filepath_raw = str(stray_path)
    stray.file_format = "PNG"
    stray.save()
    loaded = bpy.data.images.load(str(stray_path))
    loaded.use_fake_user = True
    assert not any(i.subject == loaded.name for i in _health().health_report(scene).issues)
    os.remove(stray_path)
    assert any(i.code == "image_missing_file" and i.subject == loaded.name for i in _health().health_report(scene).problems)


def test_missing_font(seam, tmp_path):
    fonts = [f for f in os.listdir(os.path.join(bpy.utils.system_resource("DATAFILES"), "fonts"))]
    src = os.path.join(bpy.utils.system_resource("DATAFILES"), "fonts", fonts[0])
    dst = tmp_path / ("temp_font" + os.path.splitext(fonts[0])[1])
    shutil.copy(src, dst)
    font = bpy.data.fonts.load(str(dst))
    font.use_fake_user = True
    assert _health().health_report().problems == []
    os.remove(dst)
    rep = _health().health_report()
    assert [i.code for i in rep.problems] == ["font_missing"]
    assert font.name in rep.problems[0].message


def test_orphan_derived_objects_and_duplicate_ids(seam):
    from seam_studio.vector import canvas
    scene = bpy.context.scene
    canvas.ensure_canvas(scene)
    owner = make_curve("owner")
    stroke = make_curve("owner stroke", role="STROKE")
    stroke.seam.owner_id = owner.seam.id
    result = make_curve("res", role="RESULT")
    result.seam.owner_id = "deadbeefdeadbeef"
    assert [i.code for i in _health().health_report(scene).problems] == ["orphan_derived"]  # only the RESULT
    stroke.seam.owner_id = "0123456789abcdef"
    codes = [i.code for i in _health().health_report(scene).problems]
    assert codes.count("orphan_derived") == 2
    dup = make_curve("dup")
    dup.seam.id = owner.seam.id
    assert "duplicate_id" in [i.code for i in _health().health_report(scene).problems]


def test_artwork_without_canvas_is_an_error(seam):
    make_curve("floating")
    rep = _health().health_report()
    assert [(i.severity, i.code) for i in rep.problems] == [("ERROR", "canvas_missing")]


def test_health_check_operator_reports(seam):
    health = _health()
    scene = bpy.context.scene
    assert bpy.ops.seam.health_check() == {"FINISHED"}
    assert health.last_report.problems == []
    _bad_link(scene, "Gear")
    assert bpy.ops.seam.health_check() == {"FINISHED"}   # problems are warnings, not exceptions
    assert health.last_report.summary() == "2 problems"
