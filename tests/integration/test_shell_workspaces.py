"""Viewport configuration of the SEAM workspaces (the structural, event-loop dependent part runs in tests/gui)."""

import math

import bpy
import pytest

from seam_studio.ui import workspaces as W


@pytest.fixture
def win(seam):
    wm = bpy.context.window_manager
    if not len(wm.windows):
        pytest.skip("no window available in this runtime")
    return wm.windows[0]


def _layout_area(win):
    screen = [s for s in bpy.data.screens if s.name == "Layout"][0]
    area = W.view3d_areas(screen)[0]
    return screen, area


def test_canvas_space_and_region_configuration(seam, win):
    from seam_studio.vector import canvas
    _screen, area = _layout_area(win)
    scene = bpy.context.scene
    canvas.ensure_canvas(scene)
    space = area.spaces.active
    W.configure_canvas_space(space)
    W.configure_canvas_region(space, center=(960.0, -20540.0))
    sh, ov, r3d = space.shading, space.overlay, space.region_3d
    assert sh.type == "SOLID" and sh.light == "FLAT" and sh.color_type == "TEXTURE"
    assert tuple(sh.background_color) == pytest.approx(W.CANVAS_BACKGROUND) and sh.background_type == "VIEWPORT"
    assert not (ov.show_floor or ov.show_axis_x or ov.show_axis_y or ov.show_axis_z or ov.show_relationship_lines)
    assert r3d.view_perspective == "ORTHO" and r3d.lock_rotation is True
    assert tuple(r3d.view_rotation) == pytest.approx((1.0, 0.0, 0.0, 0.0))     # top view
    assert tuple(r3d.view_location)[:2] == pytest.approx((960.0, -20540.0))
    assert space.clip_end >= 1e5
    W.configure_canvas_space(space, material_preview=True)
    assert space.shading.type == "MATERIAL"


def test_model_space_configuration(seam, win):
    _screen, area = _layout_area(win)
    space = area.spaces.active
    W.configure_canvas_space(space)
    W.configure_canvas_region(space)
    W.configure_model_space(space)
    r3d = space.region_3d
    assert r3d.view_perspective == "PERSP" and r3d.lock_rotation is False
    assert space.overlay.show_floor and space.overlay.show_axis_x
    assert space.shading.light == "STUDIO"


def test_frame_artboard_fits_and_centres(seam, win):
    from seam_studio.vector import canvas
    _screen, area = _layout_area(win)
    scene = bpy.context.scene
    canvas.ensure_canvas(scene)
    space = area.spaces.active
    W.configure_canvas_space(space)
    W.configure_canvas_region(space)
    assert W.frame_artboard(area, scene, margin=1.0) is True
    region = W.window_region(area)
    cx, cy, w, h = W.active_artboard_rect(scene)
    assert (cx, cy, w, h) == (960.0, -20540.0, 1920.0, 1080.0)
    uw, uh = W.usable_extent(area)
    # the usable part (not covered by floating toolbar / sidebar / header) shows the artboard, tight in one dimension
    assert uw >= w - 1e-3 and uh >= h - 1e-3
    assert math.isclose(uw, w, rel_tol=1e-6) or math.isclose(uh, h, rel_tol=1e-6)
    # ... and its centre is the artboard centre
    pw, ph, sx, sy = W.usable_rect(area)
    upp = W._extent_per_px(space, region)
    loc = space.region_3d.view_location
    assert (loc.x + sx * upp, loc.y + sy * upp) == pytest.approx((cx, cy), abs=1e-3)
    # larger artboard -> zoomed out proportionally
    dist1 = space.region_3d.view_distance
    canvas.active_artboard(scene).seam.artboard.width = 3840
    W.frame_artboard(area, scene, margin=1.0)
    assert space.region_3d.view_distance > dist1


def test_usable_rect_accounts_for_floating_regions(seam, win):
    _screen, area = _layout_area(win)
    space = area.spaces.active
    space.show_region_ui = False
    space.show_region_toolbar = False
    w0, h0, sx0, sy0 = W.usable_rect(area)
    space.show_region_ui = True
    space.show_region_toolbar = True
    w1, h1, sx1, sy1 = W.usable_rect(area)
    region = W.window_region(area)
    assert w0 <= region.width and h0 <= region.height
    assert w1 <= w0 and h1 <= h0          # showing the sidebar and toolbar can only shrink the usable rect


def test_fit_distance_needs_a_laid_out_region(seam):
    class R:
        width = 0
        height = 0

    class S:
        lens = 50.0

    assert W.fit_distance(S(), R(), 100, 100) is None
    assert W.fit_distance(S(), None, 100, 100) is None


def test_artwork_hidden_only_in_the_configured_view(seam, win):
    from seam_studio.vector import canvas
    scene = bpy.context.scene
    canvas.ensure_canvas(scene)
    ob = canvas.new_shape_object(scene, "A", [], location=(0, 0))
    screen = [s for s in bpy.data.screens if s.name == "Layout"][0]
    area = W.view3d_areas(screen)[0]
    other = [s for s in bpy.data.screens if s.name != "Layout" and W.view3d_areas(s)][0]
    other_area = W.view3d_areas(other)[0]
    assert W.artwork_hidden(win, area, scene) is False
    assert W.hide_artwork_locally(win, area, scene) is True
    assert W.artwork_hidden(win, area, scene) is True
    assert W.hide_artwork_locally(win, area, scene) is True      # idempotent: does not toggle back
    assert W.artwork_hidden(win, area, scene) is True
    assert area.spaces.active.use_local_collections is True
    # the artwork object still exists and stays visible globally (other views unaffected)
    assert other_area.spaces.active.use_local_collections is False
    assert ob.name in bpy.context.scene.objects
    # new layers created later are hidden too
    layer2 = canvas.new_layer(scene, "Layer 2")
    ob2 = bpy.data.objects.new("B", None)
    layer2.objects.link(ob2)
    bpy.context.view_layer.update()
    assert ob2.name not in W._visible_names(win, area)
    W.show_artwork_locally(area)
    assert W.artwork_hidden(win, area, scene) is False
    assert "B" in W._visible_names(win, area)


def test_hide_without_artwork_collection_is_a_noop(seam, win):
    screen, area = _layout_area(win)
    assert W.hide_artwork_locally(win, area, bpy.context.scene) is False
    assert W.hide_artwork_locally(None, area, bpy.context.scene) is False


def test_configure_screen_roles(seam, win):
    from seam_studio.vector import canvas
    scene = bpy.context.scene
    canvas.ensure_canvas(scene)
    screen, area = _layout_area(win)
    # canvas workspaces configure the first view only, Model the 3D one
    done = W.configure_screen(W.VECTOR, screen, win, scene)
    assert list(done.values())[0].startswith("canvas")
    assert area.spaces.active.region_3d.lock_rotation is True
    done = W.configure_screen(W.MODEL, screen, win, scene)
    assert list(done.values())[0] == "model (artwork hidden)"
    assert area.spaces.active.region_3d.lock_rotation is False
    # Split: left canvas, right 3D
    with bpy.context.temp_override(window=win, screen=screen, area=area, region=W.window_region(area)):
        bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
    assert len(W.view3d_areas(screen)) == 2
    roles = [r for _a, r in W.roles_for(W.SPLIT, screen)]
    assert roles == ["canvas", "model"]
    done = W.configure_screen(W.SPLIT, screen, win, scene)
    assert sorted(v.split(" ")[0] for v in done.values()) == ["canvas", "model"]
    assert W.roles_for("Layout", screen) == []


def test_repair_operator_poll_follows_window(seam):
    # no context window (Blender binary in background mode) -> poll fails with an explanation, never half works
    has_window = bpy.context.window is not None
    assert bpy.ops.seam.workspace_repair.poll() is has_window
    assert bpy.ops.seam.switch_workspace.poll() is has_window
