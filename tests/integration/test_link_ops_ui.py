"""Operators (polls, show/edit, rebuild, remove, relink guards) and the sidebar UI drawing paths."""

import bpy
import pytest

from helpers import flush
from test_link_support import activate, extent, make_link, make_shape, move_point


class FakeLayout:
    """Records UI calls so panel draw code can run headless."""

    def __init__(self, log=None):
        self.log = [] if log is None else log
        self.enabled = True
        self.use_property_split = False

    def _sub(self, *a, **k):
        return FakeLayout(self.log)

    row = column = box = split = _sub

    def label(self, text="", icon="NONE", **k):
        self.log.append(("label", text, icon, self.enabled))

    def operator(self, idname, text="", icon="NONE", **k):
        self.log.append(("op", idname, text, self.enabled))
        return type("OpProps", (), {})()

    def prop(self, data, attr, **k):
        assert hasattr(data, attr), attr
        self.log.append(("prop", attr, self.enabled))

    def prop_search(self, *a, **k):
        self.log.append(("prop_search",))

    def template_list(self, listtype, list_id, dataptr, propname, active_dataptr, active_propname, **k):
        assert hasattr(dataptr, propname) and hasattr(active_dataptr, active_propname)
        self.log.append(("list", listtype))

    def separator(self, **k):
        pass


def draw(panel_cls):
    fake = type("P", (), {})()
    fake.layout = FakeLayout()
    if hasattr(panel_cls, "poll") and not panel_cls.poll(bpy.context):
        return None
    panel_cls.draw(fake, bpy.context)
    return fake.layout.log


def test_classes_registered(seam):
    for name in ("SEAM_PT_make_3d", "SEAM_PT_live_link", "SEAM_UL_links"):
        assert hasattr(bpy.types, name)
    for op in ("make_3d", "link_pause", "link_resume", "link_detach", "link_overwrite", "link_branch",
               "link_relink", "link_rebuild", "link_remove", "edit_source", "show_result", "duplicate_result"):
        assert hasattr(bpy.ops.seam, op)
    from seam_studio.engine import core
    assert "link" in core.producers()
    assert core.producers()["link"].priority == 80


def test_make_3d_poll(engine):
    bpy.context.view_layer.objects.active = None
    assert not bpy.ops.seam.make_3d.poll()
    src = make_shape("A")
    activate(src)
    assert bpy.ops.seam.make_3d.poll()
    link = make_link(src)
    activate(link.target)
    assert not bpy.ops.seam.make_3d.poll()  # results can never be sources
    assert not bpy.ops.seam.link_resume.poll()
    assert bpy.ops.seam.link_pause.poll()


def test_make_3d_panel_explains_disabled_recipes(engine):
    from seam_studio.link import ui
    src = make_shape("A")
    activate(src)
    log = draw(ui.SEAM_PT_make_3d)
    ops = {e[2]: e[3] for e in log if e[0] == "op"}
    assert ops["Extrude"] is True and ops["Sweep"] is True
    assert ops["Text 3D"] is False
    labels = [e[1] for e in log if e[0] == "label"]
    assert any("text" in t.lower() for t in labels)
    bpy.context.view_layer.objects.active = None
    log = draw(ui.SEAM_PT_make_3d)
    assert log and log[0][0] == "label"


@pytest.mark.parametrize("state", ["LIVE", "PAUSED", "CONFLICT", "MISSING_SOURCE", "ERROR", "DETACHED"])
def test_live_link_panel_draws_every_state(engine, state):
    from seam_studio.link import ui
    src = make_shape("A")
    link = make_link(src)
    link.state = state
    link.message = "Something to say about this link that is long enough to wrap onto two lines"
    activate(link.target)
    log = draw(ui.SEAM_PT_live_link)
    assert log is not None
    opnames = {e[1] for e in log if e[0] == "op"}
    if state == "CONFLICT":
        assert {"seam.link_overwrite", "seam.link_branch", "seam.link_detach"} <= opnames
    if state == "MISSING_SOURCE":
        assert "seam.link_relink" in opnames
    if state == "PAUSED":
        assert "seam.link_resume" in opnames
    assert ("list", "SEAM_UL_links") in log
    assert any(e[0] == "prop" and e[1] == "depth" for e in log)


def test_draw_link_status(engine):
    from seam_studio.link import ui
    src = make_shape("A")
    make_link(src)
    lay = FakeLayout()
    ui.draw_link_status(lay, src)
    assert lay.log and "Extrude: Live" in lay.log[0][1]
    make_link(src)
    lay = FakeLayout()
    ui.draw_link_status(lay, src)
    assert "2 links" in lay.log[0][1]
    lay = FakeLayout()
    ui.draw_link_status(lay, None)
    assert lay.log == []


def test_ui_list_filters_to_active_object(engine):
    from seam_studio.link import ui
    a, b = make_shape("A"), make_shape("B")
    make_link(a)
    make_link(b)
    activate(a)
    flags, order = ui.SEAM_UL_links.filter_items(type("U", (), {"bitflag_filter_item": 1 << 30})(),
                                                 bpy.context, bpy.context.scene.seam, "links")
    assert [bool(f) for f in flags] == [True, False] and order == []


def test_show_result_and_edit_source_select(engine):
    src = make_shape("A")
    link = make_link(src)
    activate(src)
    assert bpy.ops.seam.show_result() == {"FINISHED"}
    assert bpy.context.view_layer.objects.active == link.target
    assert list(bpy.context.selected_objects) == [link.target]
    assert bpy.ops.seam.edit_source(edit_mode=False) == {"FINISHED"}
    assert bpy.context.view_layer.objects.active == src and src.select_get()
    assert bpy.ops.seam.edit_source(edit_mode=True) == {"FINISHED"}  # headless: stays usable


def test_rebuild_and_remove(engine):
    src = make_shape("A")
    link = make_link(src)
    activate(src)
    assert bpy.ops.seam.link_rebuild() == {"FINISHED"}
    assert link.build_count == 2
    tgt = link.target
    assert bpy.ops.seam.link_remove() == {"FINISHED"}
    assert len(bpy.context.scene.seam.links) == 0
    assert tgt.seam.role == "NONE" and tgt.name in bpy.data.objects
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert extent(tgt)[0] == pytest.approx(2.0, abs=1e-5)  # independent now
    link = make_link(src)
    name = link.target.name
    activate(src)
    assert bpy.ops.seam.link_remove(delete_result=True) == {"FINISHED"}
    assert name not in bpy.data.objects


def test_relink_guards(engine):
    a = make_shape("A")
    link = make_link(a)
    other = make_link(make_shape("B"))
    activate(link.target)
    with pytest.raises(RuntimeError):
        bpy.ops.seam.link_relink(source_name=other.target.name)  # a result can't be a source
    with pytest.raises(RuntimeError):
        bpy.ops.seam.link_relink()  # nothing selected to relink to
    c = make_shape("C", rings=(((0, 0), (50, 0), (50, 50), (0, 50)),))
    activate(link.target)
    c.select_set(True)
    assert bpy.ops.seam.link_relink() == {"FINISHED"}
    assert link.source == c and extent(link.target)[0] == pytest.approx(0.5, abs=1e-5)


def test_make_3d_by_name_and_error_reports(engine):
    src = make_shape("Named")
    bpy.context.view_layer.objects.active = src
    assert bpy.ops.seam.make_3d(recipe="EXTRUDE", source_name="Named") == {"FINISHED"}
    assert bpy.context.scene.seam.links[0].source == src
