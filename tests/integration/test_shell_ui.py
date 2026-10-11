"""Preferences, new-document defaults, quick starts, panel polls and keymaps (shell)."""

import types

import bpy
import pytest

from helpers import make_curve


def _prefs():
    from seam_studio.ui import prefs
    return prefs.get()


@pytest.fixture
def prefs_reset(seam):
    p = _prefs()
    saved = {k: getattr(p, k) for k in ("show_start_panel", "default_canvas_width", "default_canvas_height",
                                       "default_units", "default_dpi")}
    yield p
    for k, v in saved.items():
        setattr(p, k, v)


class FakeOp:
    """Stand-in for a domain operator that is not installed: records calls, returns FINISHED."""

    def __init__(self, props, result=None):
        self.props = props
        self.calls = []
        self.result = result or {"FINISHED"}

    def __call__(self, **kw):
        self.calls.append(kw)
        return self.result

    def poll(self):
        return True

    def get_rna_type(self):
        return types.SimpleNamespace(properties=[types.SimpleNamespace(identifier=p) for p in self.props])


@pytest.fixture
def fake_ops(monkeypatch):
    """Replace operator lookup by a registry the test controls (domain operators are other workers' code)."""
    from seam_studio.ui import util
    registry = {}
    real = util.op_callable
    monkeypatch.setattr(util, "op_callable", lambda idname: registry.get(idname) if idname.split(".")[1] in
                        {k.split(".")[1] for k in ("seam.raster_new", "seam.raster_paint_mode", "seam.add_shape",
                                                    "seam.make_3d", "seam.canvas_new")} else real(idname))
    return registry


# ------------------------------------------------------------------ preferences + document defaults

def test_preferences_registered_with_defaults(seam):
    from seam_studio.ui import prefs, util
    assert util.ADDON_ID in bpy.context.preferences.addons
    p = prefs.get()
    assert isinstance(p, prefs.SEAM_AP_Preferences)
    assert (p.show_start_panel, p.default_canvas_width, p.default_canvas_height, p.default_units, p.default_dpi) == \
        (True, 1920, 1080, "PX", 96.0)


def test_defaults_shim_without_addon(seam):
    from seam_studio.ui import prefs
    shim = prefs._DefaultPrefs()
    assert shim.default_canvas_width == 1920 and shim.show_start_panel is True


def test_new_document_follows_preferences(prefs_reset):
    from seam_studio.ui import document
    from seam_studio.vector import canvas
    p = prefs_reset
    p.default_canvas_width, p.default_canvas_height, p.default_units, p.default_dpi = 800, 600, "MM", 300.0
    scene = bpy.context.scene
    root, created = document.ensure_document(scene)
    assert created and root.seam.role == "CANVAS_ROOT"
    ab = canvas.active_artboard(scene)
    assert (ab.seam.artboard.width, ab.seam.artboard.height) == (800, 600)
    xs = sorted({round(v.co.x) for v in ab.data.vertices})
    ys = sorted({round(v.co.y) for v in ab.data.vertices})
    assert xs == [0, 800] and ys == [-600, 0]
    assert scene.seam.doc.units == "MM" and scene.seam.doc.dpi == 300.0
    assert scene.view_settings.view_transform == "Standard"
    # a second call finds the canvas and changes nothing
    scene.view_settings.view_transform = "Raw"
    assert document.ensure_document(scene)[1] is False
    assert scene.view_settings.view_transform == "Raw"


def test_defaults_never_touch_a_document_with_artwork(prefs_reset):
    from seam_studio.ui import document
    from seam_studio.vector import canvas
    scene = bpy.context.scene
    canvas.ensure_canvas(scene)
    make_curve("art")
    prefs_reset.default_canvas_width = 640
    assert document.is_pristine(scene) is False
    assert document.apply_new_document_defaults(scene) is False
    assert canvas.active_artboard(scene).seam.artboard.width == 1920


# ------------------------------------------------------------------ start panel + quick starts

def test_start_panel_poll_follows_document_state(prefs_reset):
    from seam_studio.ui import panels
    from seam_studio.vector import canvas
    ctx = bpy.context
    assert panels.SEAM_PT_start.poll(ctx) is True            # empty file
    canvas.ensure_canvas(ctx.scene)
    assert panels.SEAM_PT_start.poll(ctx) is True            # canvas only: still empty
    make_curve("art")
    assert panels.SEAM_PT_start.poll(ctx) is False           # has artwork
    bpy.data.objects.remove(bpy.data.objects["art"])
    assert panels.SEAM_PT_start.poll(ctx) is True
    prefs_reset.show_start_panel = False
    assert panels.SEAM_PT_start.poll(ctx) is False
    assert bpy.ops.seam.start_dismiss() == {"FINISHED"}


def test_vector_quickstart_creates_canvas_with_prefs(prefs_reset):
    from seam_studio.vector import canvas
    prefs_reset.default_canvas_width, prefs_reset.default_canvas_height = 1024, 768
    assert bpy.ops.seam.new_vector_document() == {"FINISHED"}
    ab = canvas.active_artboard(bpy.context.scene)
    assert (ab.seam.artboard.width, ab.seam.artboard.height) == (1024, 768)
    # not available once there is artwork, and says why
    make_curve("art")
    assert bpy.ops.seam.new_vector_document.poll() is False
    from seam_studio.ui import ops
    ok, why = ops.quickstart_status("VECTOR")
    assert not ok and "already has artwork" in why


def test_paint_quickstart_reports_missing_tools(seam, monkeypatch):
    from seam_studio.ui import ops, util
    monkeypatch.setattr(util, "op_callable", lambda idname: None)
    ok, why = ops.quickstart_status("PAINT")
    assert not ok and "Paint tools are not installed" in why and "seam.raster_new" in why
    assert bpy.ops.seam.new_paint_canvas.poll() is False
    ok, why = ops.quickstart_status("DRAW3D")
    assert not ok and "seam.add_shape" in why
    assert bpy.ops.seam.draw_to_3d.poll() is False


def test_paint_quickstart_dispatches_to_raster_new(seam, fake_ops):
    fake_ops["seam.raster_new"] = FakeOp(("name", "width", "height", "background"))
    fake_ops["seam.raster_paint_mode"] = FakeOp(())
    assert bpy.ops.seam.new_paint_canvas.poll() is True
    assert bpy.ops.seam.new_paint_canvas() == {"FINISHED"}
    assert fake_ops["seam.raster_new"].calls == [{"width": 1920, "height": 1080}]
    assert fake_ops["seam.raster_paint_mode"].calls == [{}]
    assert bpy.context.scene.seam.doc.canvas_ready  # the canvas exists first


def test_paint_quickstart_failure_is_an_error_not_success(seam, fake_ops):
    fake_ops["seam.raster_new"] = FakeOp(("width", "height"), result={"CANCELLED"})
    with pytest.raises(RuntimeError, match="Creating the paint canvas failed"):
        bpy.ops.seam.new_paint_canvas()


def test_draw_to_3d_quickstart_sequence(seam, fake_ops):
    fake_ops["seam.add_shape"] = FakeOp(("kind", "width", "height", "radius", "sides", "inner_ratio", "x", "y"))
    fake_ops["seam.make_3d"] = FakeOp(("recipe", "depth"))
    assert bpy.ops.seam.draw_to_3d() == {"FINISHED"}
    add = fake_ops["seam.add_shape"].calls[0]
    assert add["kind"] == "STAR" and add["sides"] == 5 and add["width"] == add["height"] == 540.0
    assert (add["x"], add["y"]) == (960.0, -540.0)           # artboard centre in canvas space
    assert fake_ops["seam.make_3d"].calls == [{"recipe": "EXTRUDE"}]


class _AddsStar(FakeOp):
    """add_shape stand-in that really creates an object, like the vector operator does."""

    def __call__(self, **kw):
        from helpers import make_curve
        super().__call__(**kw)
        make_curve("Star")
        return {"FINISHED"}


class _PollFails(FakeOp):
    def poll(self):
        return False


def test_draw_to_3d_failure_leaves_the_document_untouched(seam, fake_ops):
    from seam_studio.ui import document
    fake_ops["seam.add_shape"] = _AddsStar(("kind", "width", "height"))
    fake_ops["seam.make_3d"] = FakeOp(("recipe",), result={"CANCELLED"})
    document.ensure_document(bpy.context.scene)
    before = sorted(o.name for o in bpy.context.scene.objects)
    with pytest.raises(RuntimeError, match="Creating the live 3D result failed"):
        bpy.ops.seam.draw_to_3d()
    assert sorted(o.name for o in bpy.context.scene.objects) == before     # no half-made star left behind
    assert "Star" not in bpy.data.curves
    assert document.is_pristine(bpy.context.scene)                          # the Start panel stays


def test_draw_to_3d_poll_failure_is_a_clean_error(seam, fake_ops):
    fake_ops["seam.add_shape"] = _AddsStar(("kind", "width", "height"))
    fake_ops["seam.make_3d"] = _PollFails(("recipe",))                      # e.g. nothing selected
    with pytest.raises(RuntimeError, match="cannot run here"):
        bpy.ops.seam.draw_to_3d()
    assert not any(o.name.startswith("Star") for o in bpy.context.scene.objects)


def test_paint_quickstart_failure_removes_partial_documents(seam, fake_ops):
    class Half(FakeOp):
        def __call__(self, **kw):
            doc = bpy.context.scene.seam.rasters.add()
            doc.id, doc.name = "f" * 16, "Half"
            doc.layers.add().image = bpy.data.images.new("Half layer", 4, 4)
            return {"CANCELLED"}
    fake_ops["seam.raster_new"] = Half(("width", "height"))
    with pytest.raises(RuntimeError, match="Creating the paint canvas failed"):
        bpy.ops.seam.new_paint_canvas()
    assert len(bpy.context.scene.seam.rasters) == 0 and "Half layer" not in bpy.data.images


def test_draw_to_3d_needs_the_properties_it_passes(seam, fake_ops):
    fake_ops["seam.add_shape"] = FakeOp(("kind", "width"))    # no 'height': incompatible build
    fake_ops["seam.make_3d"] = FakeOp(("recipe",))
    from seam_studio.ui import ops
    ok, why = ops.quickstart_status("DRAW3D")
    assert not ok and "lacks properties: height" in why


def test_toggle_live_operator(seam):
    eng = bpy.context.scene.seam.engine
    assert eng.live_enabled
    bpy.ops.seam.toggle_live()
    assert eng.live_enabled is False
    bpy.ops.seam.toggle_live()
    assert eng.live_enabled is True


# ------------------------------------------------------------------ keymaps

def test_keymap_items_do_not_conflict_with_blender_defaults(seam):
    from seam_studio.ui import keymaps
    assert keymaps.find_conflicts() == []


def test_conflict_detector_detects_conflicts(seam):
    from seam_studio.ui import keymaps
    undo = keymaps.KeyItem("seam.fake", "Z", ctrl=True)               # Ctrl+Z is Undo in Screen keymap
    assert any("Ctrl Z" in c for c in keymaps.find_conflicts([undo]))
    both = [keymaps.KeyItem("a.x", "L", ctrl=True, alt=True), keymaps.KeyItem("b.y", "L", ctrl=True, alt=True)]
    assert any("used by both" in c for c in keymaps.find_conflicts(both))
    g = keymaps.KeyItem("seam.fake", "G", ctrl=True)                  # Ctrl+G in Object Mode
    assert keymaps.find_conflicts([g])


def test_keymap_registered_in_addon_keyconfig(seam):
    from seam_studio import compat
    from seam_studio.ui import keymaps
    kc = compat.keyconfig_addon()
    assert kc is not None
    km = kc.keymaps["3D View"]
    wanted = keymaps.available_items()
    assert len(wanted) >= 5                                            # 4 workspaces + live toggle (+ vector ops when installed)
    got = [(kmi.idname, kmi.type, kmi.ctrl, kmi.alt, kmi.shift) for kmi in km.keymap_items
           if kmi.idname.startswith("seam.")]
    for it in wanted:
        assert (it.idname, it.type, it.ctrl, it.alt, it.shift) in got
    assert [d.idname for d in keymaps.registered_items()] == [d.idname for d in wanted]
    ws = [kmi for kmi in km.keymap_items if kmi.idname == "seam.switch_workspace"]
    assert sorted(k.properties.name for k in ws) == ["Model", "Paint", "Split", "Vector"]
