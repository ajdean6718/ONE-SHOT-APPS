"""Workspace helpers: create/repair the SEAM workspaces and configure their viewports (owner: shell).

The four SEAM workspaces (architecture §8) are built from Blender's native ones:

* ``Vector``  - copy of Layout: one 2D canvas viewport (top ortho, rotation locked, Solid/Flat texture colours,
  neutral background, floor/axes hidden), timeline area removed.
* ``Paint``   - copy of Texture Paint: canvas viewport in Material Preview + Image Editor, Texture Paint mode.
* ``Model``   - copy of Layout: perspective 3D viewport with local collections hiding ``SEAM Artwork``.
* ``Split``   - copy of Layout: 2D canvas on the left, 3D viewport on the right.

Everything that only touches viewport properties works without a window (tests); creating workspaces and
local-collection visibility need a GUI window and say so when it is missing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Quaternion, Vector

from ..seam_core import CANVAS_ORIGIN

VECTOR, PAINT, MODEL, SPLIT = "Vector", "Paint", "Model", "Split"
SEAM_WORKSPACES = (VECTOR, PAINT, MODEL, SPLIT)
ARTWORK_COLLECTION = "SEAM Artwork"

#: native workspace each SEAM workspace is duplicated from
SOURCES = {VECTOR: "Layout", PAINT: "Texture Paint", MODEL: "Layout", SPLIT: "Layout"}
#: neutral mid-dark grey canvas surround
CANVAS_BACKGROUND = (0.18, 0.18, 0.19)
ORTHO_SENSOR = 72.0  # ortho extent along the larger region side = view_distance * 72 / lens (measured, 4.0 + 5.2)


# ------------------------------------------------------------------ discovery

def view3d_areas(screen) -> list:
    """VIEW_3D areas of a screen sorted left-to-right then top-to-bottom."""
    areas = [a for a in screen.areas if a.type == "VIEW_3D"]
    return sorted(areas, key=lambda a: (a.x, -a.y))


def window_region(area):
    for r in area.regions:
        if r.type == "WINDOW":
            return r
    return None


def workspace_screen(ws):
    """The screen of a workspace (first of ``ws.screens``)."""
    return ws.screens[0] if len(ws.screens) else None


def roles_for(workspace_name: str, screen) -> List[Tuple[object, str]]:
    """``[(area, role)]`` with role ``"canvas"`` / ``"model"`` for the 3D views of a SEAM workspace."""
    areas = view3d_areas(screen)
    if workspace_name in (VECTOR, PAINT):
        return [(a, "canvas") for a in areas[:1]]
    if workspace_name == MODEL:
        return [(a, "model") for a in areas[:1]]
    if workspace_name == SPLIT:
        out = []
        if len(areas) >= 1:
            out.append((areas[0], "canvas"))
        if len(areas) >= 2:
            out.append((areas[-1], "model"))
        return out
    return []


# ------------------------------------------------------------------ viewport configuration

def configure_canvas_space(space, material_preview: bool = False) -> None:
    """2D canvas look: flat unlit texture colours, neutral background, no floor / axes / relationship lines."""
    sh = space.shading
    if material_preview:
        sh.type = "MATERIAL"
    else:
        sh.type = "SOLID"
    # Solid settings are configured in both cases so toggling the shading mode keeps a sane 2D look.
    sh.light = "FLAT"
    sh.color_type = "TEXTURE"
    sh.show_shadows = False
    sh.show_cavity = False
    sh.show_backface_culling = False
    sh.show_xray = False
    sh.background_type = "VIEWPORT"
    sh.background_color = CANVAS_BACKGROUND
    ov = space.overlay
    ov.show_floor = False
    ov.show_axis_x = False
    ov.show_axis_y = False
    ov.show_axis_z = False
    ov.show_ortho_grid = False
    ov.show_relationship_lines = False
    ov.show_text = True
    space.clip_start = 0.01
    space.clip_end = 100000.0  # ortho views of a large canvas sit far from the view centre
    space.show_region_ui = True
    space.show_region_toolbar = True


def configure_canvas_region(space, center=None) -> None:
    """Top orthographic view, rotation locked (the canvas is a plane; orbiting would only disorient)."""
    r3d = space.region_3d
    if r3d is None:
        return
    r3d.view_perspective = "ORTHO"
    r3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))  # top view
    if center is not None:
        r3d.view_location = Vector((center[0], center[1], 0.0))
    r3d.lock_rotation = True


def configure_model_space(space) -> None:
    """3D look for modelling: perspective, regular solid shading with floor and axes."""
    sh = space.shading
    sh.type = "SOLID"
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.background_type = "THEME"
    ov = space.overlay
    ov.show_floor = True
    ov.show_axis_x = True
    ov.show_axis_y = True
    ov.show_ortho_grid = True
    ov.show_relationship_lines = True
    space.clip_start = 0.01
    space.clip_end = 1000.0
    space.show_region_ui = True
    space.show_region_toolbar = True
    r3d = space.region_3d
    if r3d is not None:
        r3d.lock_rotation = False
        r3d.view_perspective = "PERSP"
        # a 3/4 view of the world origin, where new 3D results are placed
        r3d.view_rotation = Quaternion((0.8205, 0.4247, 0.1759, 0.3399))
        r3d.view_location = Vector((0.0, 0.0, 0.5))
        r3d.view_distance = 9.0


def active_artboard_rect(scene) -> Tuple[float, float, float, float]:
    """(center_x, center_y, width, height) of the active artboard in WORLD space (canvas origin applied)."""
    from ..vector import canvas
    ab = canvas.active_artboard(scene)
    ox, oy, _oz = CANVAS_ORIGIN
    if ab is None:
        return (ox + 960.0, oy - 540.0, 1920.0, 1080.0)
    w = float(ab.seam.artboard.width)
    h = float(ab.seam.artboard.height)
    return (ox + ab.location.x + w * 0.5, oy + ab.location.y - h * 0.5, w, h)


def usable_rect(area) -> Tuple[float, float, float, float]:
    """Pixels of a 3D view that artwork can use: ``(width, height, shift_x, shift_y)``.

    With region overlap (the default in 3D views) the toolbar, sidebar and headers float over the window region;
    the artwork should be centred in what remains. ``shift_*`` is the offset of that rect's centre from the window
    region's centre in pixels (x right, y up)."""
    space = area.spaces.active
    win = window_region(area)
    if win is None:
        return (0.0, 0.0, 0.0, 0.0)

    def size(rtype, shown, axis):
        if not shown:
            return 0.0
        r = next((r for r in area.regions if r.type == rtype), None)
        return float(getattr(r, axis)) if r is not None and r.width > 1 and r.height > 1 else 0.0

    tw = size("TOOLS", space.show_region_toolbar, "width")
    uw = size("UI", space.show_region_ui, "width")
    hh = size("HEADER", True, "height") + size("TOOL_HEADER", True, "height")
    w, h = float(win.width), float(win.height)
    overlapped_x = w >= area.width - 1
    overlapped_y = h >= area.height - 1
    if overlapped_x:
        sx = (tw - uw) / 2.0
        w = max(w - tw - uw, 50.0)
    else:
        sx = 0.0
    if overlapped_y:
        sy = -hh / 2.0
        h = max(h - hh, 50.0)
    else:
        sy = 0.0
    return (w, h, sx, sy)


def _extent_per_px(space, region) -> float:
    """World units per pixel of an ortho region at the current ``view_distance``."""
    major = space.region_3d.view_distance * ORTHO_SENSOR / space.lens
    return major / max(region.width, region.height)


def fit_distance(space, region, width: float, height: float, margin: float = 1.15,
                 usable: Optional[Tuple[float, float]] = None) -> Optional[float]:
    """Ortho ``view_distance`` that shows ``width`` x ``height`` world units in ``usable`` pixels of ``region``
    (default: the whole region). None when the region has no size yet."""
    if region is None or region.width < 2 or region.height < 2:
        return None
    uw, uh = usable if usable is not None else (region.width, region.height)
    major = max(region.width, region.height)
    # world units per pixel needed so that the artboard (plus margin) fits the usable pixels
    upp = max(width * margin / uw, height * margin / uh)
    return upp * major * space.lens / ORTHO_SENSOR


def visible_extent(space, region) -> Tuple[float, float]:
    """World-unit width/height currently visible in an ortho ``region`` (analytic, matches ``window_matrix``)."""
    upp = _extent_per_px(space, region)
    return (upp * region.width, upp * region.height)


def usable_extent(area) -> Tuple[float, float]:
    """World-unit size of the part of a canvas view not covered by the floating toolbar/sidebar/headers."""
    space = area.spaces.active
    region = window_region(area)
    uw, uh, _sx, _sy = usable_rect(area)
    upp = _extent_per_px(space, region)
    return (upp * uw, upp * uh)


def frame_artboard(area, scene, margin: float = 1.15) -> bool:
    """Fit the active artboard into the usable part of a canvas viewport and centre it there.

    False when the region has no size yet (not laid out)."""
    space = area.spaces.active
    region = window_region(area)
    r3d = space.region_3d
    cx, cy, w, h = active_artboard_rect(scene)
    uw, uh, sx, sy = usable_rect(area)
    dist = fit_distance(space, region, w, h, margin, usable=(uw, uh) if uw > 0 else None)
    if r3d is None or dist is None:
        return False
    r3d.view_distance = dist
    upp = _extent_per_px(space, region)
    # the window-region centre must sit (sx, sy) pixels away from the artboard centre, opposite to the usable shift
    r3d.view_location = Vector((cx - sx * upp, cy - sy * upp, 0.0))
    return True


# ------------------------------------------------------------------ local collections (artwork hidden in 3D)

def _layer_collection_index(view_layer, name: str) -> int:
    """1-based depth-first index of a layer collection as used by ``object.hide_collection`` (0 = not found)."""
    order = []

    def walk(lc):
        for child in lc.children:
            order.append(child.name)
            walk(child)

    walk(view_layer.layer_collection)
    return order.index(name) + 1 if name in order else 0


def _visible_names(window, area) -> set:
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=window_region(area)):
        return {o.name for o in bpy.context.visible_objects}


def _probe_object(scene):
    from ..vector import canvas
    return canvas.canvas_root(scene)


def artwork_hidden(window, area, scene=None) -> Optional[bool]:
    """True when ``SEAM Artwork`` objects are hidden in this viewport, False when visible, None when unknown."""
    scene = scene or bpy.context.scene
    if not area.spaces.active.use_local_collections:
        return False
    probe = _probe_object(scene)
    if probe is None or window is None:
        return None
    return probe.name not in _visible_names(window, area)


def hide_artwork_locally(window, area, scene=None) -> bool:
    """Hide the ``SEAM Artwork`` collection in this 3D view only (local collections). Idempotent.

    Needs a window (operator override). Returns True when the artwork is hidden afterwards.
    """
    scene = scene or bpy.context.scene
    if window is None:
        return False
    view_layer = window.view_layer
    idx = _layer_collection_index(view_layer, ARTWORK_COLLECTION)
    if idx == 0:
        return False
    space = area.spaces.active
    space.use_local_collections = True
    if artwork_hidden(window, area, scene) is True:
        return True
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=window_region(area),
                                  scene=scene):
        bpy.ops.object.hide_collection(collection_index=idx, toggle=True)
    return artwork_hidden(window, area, scene) is True


def show_artwork_locally(area) -> None:
    """Undo :func:`hide_artwork_locally` for this view (turns local collections off)."""
    area.spaces.active.use_local_collections = False


# ------------------------------------------------------------------ per workspace configuration

def configure_screen(name: str, screen, window=None, scene=None) -> Dict[str, str]:
    """Apply the view configuration of SEAM workspace ``name`` to every 3D view of ``screen``.

    Returns ``{area_key: description}`` for the configured views.
    """
    scene = scene or bpy.context.scene
    done = {}
    for area, role in roles_for(name, screen):
        space = area.spaces.active
        key = f"{name}:{area.x},{area.y}"
        if role == "canvas":
            configure_canvas_space(space, material_preview=(name == PAINT))
            cx, cy, _w, _h = active_artboard_rect(scene)
            configure_canvas_region(space, center=(cx, cy))
            fit = frame_artboard(area, scene)
            done[key] = "canvas" + ("" if fit else " (frame pending)")
        else:
            configure_model_space(space)
            if window is not None:
                hidden = hide_artwork_locally(window, area, scene)
                done[key] = "model" + (" (artwork hidden)" if hidden else " (artwork visible)")
            else:
                done[key] = "model (no window: artwork not hidden)"
    return done


def frame_canvas_views(window_manager=None, scene=None) -> int:
    """Fit every canvas viewport of every SEAM workspace in every window; returns how many were framed."""
    wm = window_manager or bpy.context.window_manager
    scene = scene or bpy.context.scene
    n = 0
    for win in wm.windows:
        ws = win.workspace
        if ws.name not in SEAM_WORKSPACES:
            continue
        for area, role in roles_for(ws.name, win.screen):
            if role == "canvas" and frame_artboard(area, scene):
                n += 1
    return n


# ------------------------------------------------------------------ creating workspaces (GUI only)
#
# ``window.workspace = ws`` is applied by the window manager on its next event-loop iteration (verified on 4.0.2
# and 5.2.2), so every structural step runs as a *generator*: each ``yield`` returns control to the event loop.
# :func:`run_steps` drives such a generator with ``bpy.app.timers`` (scripts) and ``SEAM_OT_workspace_repair``
# drives it with a modal timer. Headless runs have no event loop; there only the pure viewport configuration works.

class WorkspaceError(RuntimeError):
    pass


def _override(window, **extra):
    return bpy.context.temp_override(window=window, screen=window.screen, workspace=window.workspace, **extra)


def _close_area(window, area) -> bool:
    if bpy.app.background:
        return False  # screen.area_close never returns without an event loop
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=area.regions[0]):
        return "FINISHED" in bpy.ops.screen.area_close()


def _split_area(window, area, factor: float = 0.5):
    before = {a.as_pointer() for a in window.screen.areas}
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=window_region(area)):
        bpy.ops.screen.area_split(direction="VERTICAL", factor=factor)
    return [a for a in window.screen.areas if a.as_pointer() not in before]


def _source_workspace(source_name: str, new_name: str):
    src = bpy.data.workspaces.get(source_name)
    if src is None:  # fall back to any native workspace that has a 3D viewport
        for ws in bpy.data.workspaces:
            scr = workspace_screen(ws)
            if ws.name not in SEAM_WORKSPACES and scr is not None and view3d_areas(scr):
                return ws
        raise WorkspaceError(f"no workspace to duplicate for '{new_name}'")
    return src


def duplicate_workspace(window, src, new_name: str):
    """Duplicate workspace ``src`` (which must already be the window's active one) as ``new_name``."""
    before = {w.name for w in bpy.data.workspaces}
    with _override(window):
        bpy.ops.workspace.duplicate()
    created = [w for w in bpy.data.workspaces if w.name not in before]
    if not created:
        raise WorkspaceError(f"duplicating '{src.name}' did not create a workspace")
    created[0].name = new_name
    return created[0]


def _switch(window, ws, max_ticks: int = 200):
    """Generator: make ``ws`` the window's workspace and wait until the window manager applied it."""
    window.workspace = ws
    ticks = 0
    while window.workspace.name != ws.name:
        ticks += 1
        if ticks > max_ticks:
            raise WorkspaceError(f"window did not switch to workspace '{ws.name}'")
        yield 1


def _settle(ticks: int = 3):
    """Generator: let the UI lay out areas/regions (they have size 0 until drawn)."""
    for _ in range(ticks):
        yield 1


def _prepare_screen(window, name: str) -> None:
    """Structural changes of a freshly duplicated workspace (remove timeline, split for Split)."""
    if name in (VECTOR, MODEL, SPLIT):
        for area in list(window.screen.areas):
            if area.type == "DOPESHEET_EDITOR" and area.ui_type == "TIMELINE":
                _close_area(window, area)
    if name == SPLIT and len(view3d_areas(window.screen)) < 2:
        _split_area(window, view3d_areas(window.screen)[0], 0.5)


def repair_steps(window, scene, log: dict):
    """Generator that creates missing SEAM workspaces, orders them first and configures their views.

    ``log`` receives ``created`` (names), ``configured`` ({view: description}) and ``error`` (message or None).
    """
    from ..vector import canvas
    log.update(created=[], configured={}, error=None)
    original = window.workspace
    try:
        canvas.ensure_canvas(scene)
        for name in SEAM_WORKSPACES:
            if name in bpy.data.workspaces:
                continue
            src = _source_workspace(SOURCES[name], name)
            yield from _switch(window, src)
            new = duplicate_workspace(window, src, name)
            yield from _switch(window, new)
            _prepare_screen(window, name)
            yield from _settle()
            if name == PAINT:
                new.object_mode = "TEXTURE_PAINT"
            log["created"].append(name)
        for name in reversed(SEAM_WORKSPACES):
            yield from _switch(window, bpy.data.workspaces[name])
            with _override(window):
                bpy.ops.workspace.reorder_to_front()
        for name in SEAM_WORKSPACES:
            yield from _switch(window, bpy.data.workspaces[name])
            yield from _settle(2)
            log["configured"].update(configure_screen(name, window.screen, window, scene))
    except Exception as exc:  # report, then fall through to restore the user's workspace
        import traceback
        traceback.print_exc()
        log["error"] = f"{type(exc).__name__}: {exc}"
    if original is not None and original.name in bpy.data.workspaces:
        try:
            yield from _switch(window, original)
        except WorkspaceError:
            pass


def run_steps(gen, on_done=None, interval: float = 0.05):
    """Drive a step generator with ``bpy.app.timers`` (one ``yield`` per timer tick)."""
    state = {"gen": gen}

    def tick():
        try:
            n = next(state["gen"])
        except StopIteration:
            if on_done:
                on_done()
            return None
        return interval * max(1, n or 1)

    bpy.app.timers.register(tick, first_interval=interval)


def tab_order_steps(window, out: list, start: str = VECTOR):
    """Generator filling ``out`` with workspace names in tab order beginning at ``start`` (restores the window).

    ``bpy.data.workspaces`` is alphabetical and ``screen.workspace_cycle`` refuses Python calls, so the order is
    observed by simulating the default keymap's Ctrl+Page Down ("next workspace"); needs ``--enable-event-simulate``.
    """
    original = window.workspace
    ws0 = bpy.data.workspaces.get(start) or original
    yield from _switch(window, ws0)
    out.append(ws0.name)
    area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
    region = window_region(area)
    x, y = region.x + region.width // 2, region.y + region.height // 2
    for _ in range(len(bpy.data.workspaces)):
        current = window.workspace.name
        window.event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)
        yield 2
        window.event_simulate(type="PAGE_DOWN", value="PRESS", ctrl=True, x=x, y=y)
        window.event_simulate(type="PAGE_DOWN", value="RELEASE", ctrl=True, x=x, y=y)
        for _tick in range(60):
            yield 1
            if window.workspace.name != current:
                break
        else:
            raise WorkspaceError("Ctrl+Page Down did not switch workspace (is --enable-event-simulate set?)")
        yield 2
        name = window.workspace.name
        if name == out[0]:
            break
        out.append(name)
        area = next((a for a in window.screen.areas if a.type != "EMPTY"), area)
        region = area.regions[-1] if area.regions else region
        x, y = region.x + max(region.width, 2) // 2, region.y + max(region.height, 2) // 2
    yield from _switch(window, original)


def activate(window, name: str) -> bool:
    """Ask the window to show workspace ``name`` (applied on the next event-loop iteration)."""
    ws = bpy.data.workspaces.get(name)
    if ws is None or window is None:
        return False
    window.workspace = ws
    return True


# ------------------------------------------------------------------ sidebar tab

def sidebar_category(region) -> Optional[str]:
    """Active sidebar tab name where Blender exposes it (5.x ``Region.active_panel_category``, read-only), else None.

    Neither 4.0 nor 5.2 lets scripts *set* the tab: ``scripts/build_template.py`` selects the SEAM tab by clicking it
    (``--enable-event-simulate``) and stores the choice in ``startup.blend``; ``ui.panels.seam_tab_shown`` observes it.
    """
    return getattr(region, "active_panel_category", None)


# ------------------------------------------------------------------ operators

#: True while a workspace repair (modal or timer driven) is switching workspaces; tests wait for it to clear.
repair_running = {"active": False, "count": 0}


class SEAM_OT_workspace_repair(bpy.types.Operator):
    bl_idname = "seam.workspace_repair"
    bl_label = "Repair SEAM Workspaces"
    bl_description = ("Recreate missing Vector, Paint, Model and Split workspaces, put them first and restore their "
                      "viewport setup (2D canvas views are top ortho and locked, 3D views hide the artwork). "
                      "The window flips through the workspaces while it works")
    bl_options = {"REGISTER"}

    _timer = None
    _gen = None
    _log = None

    @classmethod
    def poll(cls, context):
        if context.window is None:
            cls.poll_message_set("Needs a Blender window (not available in background mode)")
            return False
        return True

    def invoke(self, context, event):
        self._log = {}
        self._gen = repair_steps(context.window, context.scene, self._log)
        repair_running["active"] = True
        repair_running["count"] += 1
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        # Script / EXEC_DEFAULT path: same steps, driven asynchronously by timers (no modal handler available).
        log = {}
        window, scene = context.window, context.scene

        def done():
            repair_running["active"] = False
            if log.get("error"):
                print("SEAM workspace repair failed:", log["error"])
            frame_canvas_views()

        repair_running["active"] = True
        repair_running["count"] += 1
        run_steps(repair_steps(window, scene, log), on_done=done)
        self.report({"INFO"}, "Repairing SEAM workspaces...")
        return {"FINISHED"}

    def _finish(self, context):
        repair_running["active"] = False
        context.window_manager.event_timer_remove(self._timer)
        self._timer = None
        if self._log.get("error"):
            self.report({"ERROR"}, "Workspace repair failed: " + self._log["error"])
            return {"CANCELLED"}
        created = self._log.get("created", [])
        msg = ("Created " + ", ".join(created) + "; " if created else "") + \
            f"configured {len(self._log.get('configured', {}))} viewport(s)"
        self.report({"INFO"}, msg)
        return {"FINISHED"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            next(self._gen)
        except StopIteration:
            bpy.app.timers.register(lambda: (frame_canvas_views(), None)[1], first_interval=0.2)
            return self._finish(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        repair_running["active"] = False
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


class SEAM_OT_switch_workspace(bpy.types.Operator):
    bl_idname = "seam.switch_workspace"
    bl_label = "Switch SEAM Workspace"
    bl_description = "Switch to one of the SEAM workspaces (Vector, Paint, Model, Split)"
    bl_options = {"INTERNAL"}

    name: bpy.props.EnumProperty(name="Workspace", items=[(n, n, "") for n in SEAM_WORKSPACES], default=VECTOR)

    @classmethod
    def poll(cls, context):
        if context.window is None:
            cls.poll_message_set("Needs a Blender window")
            return False
        return True

    @classmethod
    def description(cls, context, properties):
        return f"Switch to the {properties.name} workspace"

    def execute(self, context):
        if self.name not in bpy.data.workspaces:
            self.report({"ERROR"}, f"Workspace '{self.name}' does not exist. Run 'Repair SEAM Workspaces'.")
            return {"CANCELLED"}
        context.window.workspace = bpy.data.workspaces[self.name]
        return {"FINISHED"}


classes = (SEAM_OT_workspace_repair, SEAM_OT_switch_workspace)
