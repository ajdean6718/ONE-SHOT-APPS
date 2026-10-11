"""Operators: make_3d, pause/resume, detach, branch, overwrite, relink, rebuild, remove, edit_source,
show_result, duplicate_result (owner: link). All are REGISTER/UNDO with poll + poll_message_set."""

from __future__ import annotations

from typing import List, Optional, Tuple

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

from .. import ids
from ..engine import core as engine
from ..seam_core import links as L
from ..seam_core.ids import new_id
from . import producer, recipes

RESULT_SPACING = 2.5
SIDE_SHADE = 0.65


# ------------------------------------------------------------------ materials

def material_color(rgba, role: str):
    r, g, b, a = (float(c) for c in rgba)
    if role == "SIDE":
        return (r * SIDE_SHADE, g * SIDE_SHADE, b * SIDE_SHADE, a)
    return (r, g, b, a)


def set_material_color(mat, rgba) -> None:
    rgba = tuple(float(c) for c in rgba)
    if tuple(round(c, 6) for c in mat.diffuse_color) != tuple(round(c, 6) for c in rgba):
        mat.diffuse_color = rgba
    tree = mat.node_tree
    if tree is None:
        return
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            sock = node.inputs.get("Base Color")
            if sock is not None and tuple(round(c, 6) for c in sock.default_value) != tuple(round(c, 6) for c in rgba):
                sock.default_value = rgba


def new_link_material(name: str, link_id: str, role: str, rgba):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True  # 4.x: materials start without nodes (5.x: always node based)
    mat["seam_link_id"] = link_id
    mat["seam_material_role"] = role
    set_material_color(mat, material_color(rgba, role))
    return mat


# ------------------------------------------------------------------ lookup

def links_of(scene, obj) -> List:
    if obj is None:
        return []
    return [lk for lk in scene.seam.links if lk.source == obj or lk.target == obj]


def resolve_link(context, link_id: str = ""):
    scene = context.scene
    seam = getattr(scene, "seam", None)
    if seam is None:
        return None
    if link_id:
        return producer.find_link(scene, link_id)
    links = seam.links
    obj = context.active_object
    idx = seam.active_link_index
    active = links[idx] if 0 <= idx < len(links) else None
    if obj is not None:
        for lk in links:
            if lk.target == obj:
                return lk
        cands = [lk for lk in links if lk.source == obj]
        if cands:
            return active if active is not None and active.source == obj else cands[0]
        if obj.seam.role == "RESULT" and obj.seam.owner_id:
            return producer.find_link(scene, obj.seam.owner_id)
        return None
    return active


def set_active_link(scene, link) -> None:
    for i, lk in enumerate(scene.seam.links):
        if lk.id == link.id:
            scene.seam.active_link_index = i
            return


LIVE_OFF_MESSAGE = "Live updates are off: turn them on to build the result"


def live_enabled(scene) -> bool:
    eng = getattr(getattr(scene, "seam", None), "engine", None)
    return True if eng is None else bool(eng.live_enabled)


def rebuild_now(scene, link) -> Optional[engine.BuildResult]:
    """Build ``link`` immediately (inside the operator, so the undo step holds the result).

    With the global live switch off nothing is built: the key stays dirty (it builds as soon as
    live updates are turned back on), the link is marked stale and a "waiting" result is returned
    so callers report it instead of claiming success."""
    key = producer.key_for(link)
    engine.mark_dirty(key)
    if not live_enabled(scene):
        link.stale = True
        link.message = LIVE_OFF_MESSAGE
        return engine.BuildResult(key, "waiting", LIVE_OFF_MESSAGE)
    results = engine.flush(scene)
    mine = [r for r in results if r.key == key]
    return mine[-1] if mine else None


def _report_build(op, link, res) -> None:
    if res is None:
        return
    if res.status == "error":
        op.report({"WARNING"}, "%s: %s" % (link.name, res.message))
    elif res.status in ("conflict", "waiting"):
        op.report({"WARNING"}, "%s: %s" % (link.name, res.message))
    elif res.message:
        op.report({"INFO"}, "%s: %s" % (link.name, res.message))


def count_results(scene) -> int:
    return sum(1 for ob in scene.objects if ob.seam.role == "RESULT")


def create_link(context, source, recipe: str, params: dict) -> Tuple[object, object]:
    """Create the target mesh object + link record for ``source``; does not build."""
    from ..vector import canvas
    scene = context.scene
    params = L.validate_params(recipe, params)
    link_id = new_id()
    label = L.RECIPE_LABELS.get(recipe, recipe.title())
    name = "%s · %s" % (source.name, label)
    me = bpy.data.meshes.new(name)
    rgba = tuple(source.seam.style.fill_color) if params.get("use_source_material", True) else (0.8, 0.8, 0.8, 1.0)
    me.materials.append(new_link_material("Face", link_id, "FACE", rgba))
    me.materials.append(new_link_material("Side", link_id, "SIDE", rgba))
    if not params.get("use_source_material", True):
        for m in me.materials:
            m["seam_link_id"] = ""  # user-owned colours: never synced
    target = bpy.data.objects.new(name, me)
    target.location = (RESULT_SPACING * count_results(scene), 0.0, 0.0)
    canvas.results_collection(scene).objects.link(target)
    ids.ensure_id(target)
    target.seam.role = "RESULT"
    target.seam.owner_id = link_id
    link = scene.seam.links.add()
    link.id = link_id
    link.name = name
    link.recipe = recipe
    link.source = source
    link.source_id = ids.ensure_id(source)
    link.target = target
    link.target_id = target.seam.id
    for k, v in params.items():
        setattr(link.params, k, v)
    link.state = "LIVE"
    set_active_link(scene, link)
    return link, target


# ------------------------------------------------------------------ workspaces / view

def switch_workspace(context, name: str) -> bool:
    win = context.window
    ws = bpy.data.workspaces.get(name)
    if win is None or ws is None:
        return False
    if win.workspace != ws:
        win.workspace = ws
    return True


def frame_selected(context, workspace_name: str = "") -> bool:
    if bpy.app.background or context.window is None:
        return False
    screens = []
    ws = bpy.data.workspaces.get(workspace_name) if workspace_name else None
    if ws is not None:
        screens.extend(ws.screens)
    if context.window.screen is not None:
        screens.append(context.window.screen)
    for screen in screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is None:
                continue
            try:
                with context.temp_override(window=context.window, screen=screen, area=area, region=region):
                    bpy.ops.view3d.view_selected()
                return True
            except Exception:
                continue
    return False


def select_only(context, obj) -> bool:
    vl = context.view_layer
    if vl.objects.get(obj.name) != obj:
        return False
    for ob in list(context.selected_objects):
        ob.select_set(False)
    try:
        obj.hide_set(False)
    except RuntimeError:
        pass
    obj.select_set(True)
    vl.objects.active = obj
    return True


# ------------------------------------------------------------------ operator params

def _f(name, label, **kw):
    d, lo, hi = L.PARAMS[name]
    return FloatProperty(name=label, default=d, min=lo, max=hi, **kw)


def _i(name, label):
    d, lo, hi = L.PARAMS[name]
    return IntProperty(name=label, default=d, min=lo, max=hi)


def _e(name, label):
    d, allowed, _ = L.PARAMS[name]
    return EnumProperty(name=label, items=[(v, v.replace("_", " ").title(), "") for v in allowed], default=d)


class _LinkParamsMixin:
    depth: _f("depth", "Depth", unit="LENGTH")
    bevel_depth: _f("bevel_depth", "Bevel", unit="LENGTH")
    bevel_resolution: _i("bevel_resolution", "Bevel Segments")
    resolution: _i("resolution", "Curve Resolution")
    orientation: _e("orientation", "Orientation")
    origin: _e("origin", "Origin")
    units_scale: _f("units_scale", "Scale (m per px)", precision=4)
    revolve_angle: _f("revolve_angle", "Angle", subtype="ANGLE")
    revolve_segments: _i("revolve_segments", "Segments")
    revolve_axis: _e("revolve_axis", "Axis")
    axis_offset: _f("axis_offset", "Axis Offset (px)")
    sweep_radius: _f("sweep_radius", "Width", unit="LENGTH")
    sweep_profile: _e("sweep_profile", "Profile")
    sweep_profile_resolution: _i("sweep_profile_resolution", "Profile Segments")
    fill_caps: BoolProperty(name="Fill Caps", default=True)
    use_source_material: BoolProperty(name="Use Source Colour", default=True)

    def collect_params(self, recipe: str) -> dict:
        return {k: getattr(self, k) for k in L.RECIPE_PARAMS[recipe] if hasattr(self, k)}

    def draw_params(self, layout, recipe: str) -> None:
        for k in L.RECIPE_PARAMS[recipe]:
            if hasattr(self, k):
                layout.prop(self, k)


class _LinkOp:
    """Base for operators acting on one link (``link_id`` or the link of the active object)."""

    bl_options = {"REGISTER", "UNDO"}
    allowed_states: Tuple[str, ...] = tuple(L.STATES)
    state_hint = ""

    link_id: StringProperty(name="Link", default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        if getattr(context.scene, "seam", None) is None:
            return False
        link = resolve_link(context)
        if link is None:
            cls.poll_message_set("Select a live-link source or 3D result")
            return False
        if link.state not in cls.allowed_states:
            cls.poll_message_set(cls.state_hint or "Not available while the link is %s"
                                 % link.state.replace("_", " ").lower())
            return False
        return cls.extra_poll(context, link)

    @classmethod
    def extra_poll(cls, context, link) -> bool:
        return True

    def get_link(self, context):
        link = resolve_link(context, self.link_id)
        if link is None:
            self.report({"ERROR"}, "Link not found")
            return None
        if link.state not in self.allowed_states:
            self.report({"ERROR"}, self.state_hint or "Not available while the link is %s" % link.state.lower())
            return None
        return link


# ------------------------------------------------------------------ operators

class SEAM_OT_make_3d(_LinkParamsMixin, bpy.types.Operator):
    """Create a live 3D result from the active shape or text"""

    bl_idname = "seam.make_3d"
    bl_label = "Make 3D"
    bl_options = {"REGISTER", "UNDO"}

    recipe: EnumProperty(name="Recipe", items=[
        ("EXTRUDE", "Extrude", "Extrude the filled shape (holes kept)"),
        ("REVOLVE", "Revolve", "Lathe the path around a vertical axis"),
        ("SWEEP", "Sweep", "Sweep a profile along the path"),
        ("TEXT3D", "Text 3D", "Extrude text outlines"),
    ], default="EXTRUDE")
    source_name: StringProperty(name="Source", default="", options={"SKIP_SAVE"},
                                description="Source object name (default: active object)")

    @classmethod
    def poll(cls, context):
        if getattr(context.scene, "seam", None) is None:
            return False
        obj = context.active_object
        if obj is None:
            cls.poll_message_set("Select a vector shape or text object")
            return False
        if obj.seam.role in L.DERIVED_ROLES:
            cls.poll_message_set("Generated objects cannot be used as a source")
            return False
        if obj.seam.role not in ("SHAPE", "TEXT"):
            cls.poll_message_set("Make 3D needs a SEAM vector shape or text object")
            return False
        return True

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "recipe")
        self.draw_params(layout, self.recipe)

    def execute(self, context):
        scene = context.scene
        src = bpy.data.objects.get(self.source_name) if self.source_name else context.active_object
        ok, why = recipes.source_ok(self.recipe, src)
        if not ok:
            self.report({"ERROR"}, why)
            return {"CANCELLED"}
        params = L.default_params(self.recipe)
        params.update(self.collect_params(self.recipe))
        try:
            params = L.validate_params(self.recipe, params)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        link, _target = create_link(context, src, self.recipe, params)
        res = rebuild_now(scene, link)
        _report_build(self, link, res)
        if res is not None and res.status == "ok":
            self.report({"INFO"}, "Created %s (live)" % link.name)
        return {"FINISHED"}


class SEAM_OT_link_pause(_LinkOp, bpy.types.Operator):
    """Stop following the source until resumed (changes are queued)"""

    bl_idname = "seam.link_pause"
    bl_label = "Pause Link"
    allowed_states = ("LIVE", "ERROR")
    state_hint = "Only live links can be paused"

    def execute(self, context):
        link = self.get_link(context)
        if link is None:
            return {"CANCELLED"}
        producer._apply(link, "pause")
        link.message = "Paused"
        return {"FINISHED"}


class SEAM_OT_link_resume(_LinkOp, bpy.types.Operator):
    """Follow the source again and apply queued changes"""

    bl_idname = "seam.link_resume"
    bl_label = "Resume Link"
    allowed_states = ("PAUSED",)
    state_hint = "The link is not paused"

    def execute(self, context):
        link = self.get_link(context)
        if link is None:
            return {"CANCELLED"}
        rebuild = producer._apply(link, "resume")
        link.message = ""
        if rebuild or link.source_signature != producer.source_signature(context.scene, link):
            _report_build(self, link, rebuild_now(context.scene, link))
        return {"FINISHED"}


class SEAM_OT_link_detach(_LinkOp, bpy.types.Operator):
    """Bake the result: keep its geometry and stop following the source"""

    bl_idname = "seam.link_detach"
    bl_label = "Bake (Detach)"
    allowed_states = ("LIVE", "PAUSED", "ERROR", "CONFLICT", "MISSING_SOURCE")
    state_hint = "The link is already detached"

    def execute(self, context):
        link = self.get_link(context)
        if link is None:
            return {"CANCELLED"}
        producer._apply(link, "detach")
        link.message = "Baked: no longer follows the source"
        for mat in (link.target.data.materials if link.target is not None else ()):
            if mat is not None and mat.get("seam_link_id") == link.id:
                mat["seam_link_id"] = ""
        self.report({"INFO"}, "%s detached" % link.name)
        return {"FINISHED"}


class SEAM_OT_link_overwrite(_LinkOp, bpy.types.Operator):
    """Discard downstream edits on the result and rebuild it from the source"""

    bl_idname = "seam.link_overwrite"
    bl_label = "Overwrite Result"
    allowed_states = ("CONFLICT", "LIVE", "ERROR")
    state_hint = "Overwrite needs a live or conflicting link"

    @classmethod
    def extra_poll(cls, context, link):
        if link.target is None or link.source is None:
            cls.poll_message_set("Source or result missing")
            return False
        return True

    def execute(self, context):
        link = self.get_link(context)
        if link is None:
            return {"CANCELLED"}
        busy = producer.target_busy_mode(link.target)
        if busy:
            self.report({"ERROR"}, "Leave %s on the result first" % busy)
            return {"CANCELLED"}
        producer._apply(link, "overwrite")
        link.force_overwrite = True
        link.source_signature = ""
        _report_build(self, link, rebuild_now(context.scene, link))
        return {"FINISHED"}


class SEAM_OT_link_branch(_LinkOp, bpy.types.Operator):
    """Keep the edited result as an independent copy and rebuild the linked result from the source"""

    bl_idname = "seam.link_branch"
    bl_label = "Branch Result"
    allowed_states = ("CONFLICT", "LIVE", "ERROR")
    state_hint = "Branch needs a live or conflicting link"

    @classmethod
    def extra_poll(cls, context, link):
        if link.target is None or link.source is None:
            cls.poll_message_set("Source or result missing")
            return False
        return True

    def execute(self, context):
        link = self.get_link(context)
        if link is None:
            return {"CANCELLED"}
        tgt = link.target
        busy = producer.target_busy_mode(tgt)
        if busy:
            self.report({"ERROR"}, "Leave %s on the result first" % busy)
            return {"CANCELLED"}
        copy = branch_copy(link)
        producer._apply(link, "branch")
        link.force_overwrite = True
        link.source_signature = ""
        _report_build(self, link, rebuild_now(context.scene, link))
        self.report({"INFO"}, "Kept the edited result as '%s'" % copy.name)
        return {"FINISHED"}


def branch_copy(link):
    tgt = link.target
    copy = tgt.copy()
    copy.data = tgt.data.copy()
    copy.name = "%s (edited)" % tgt.name
    for col in tgt.users_collection:
        col.objects.link(copy)
    copy.seam.id = new_id()
    copy.seam.role = "NONE"
    copy.seam.owner_id = ""
    for i, mat in enumerate(copy.data.materials):
        if mat is not None and mat.get("seam_link_id") == link.id:
            m2 = mat.copy()
            m2["seam_link_id"] = ""
            copy.data.materials[i] = m2
    return copy


class SEAM_OT_link_relink(_LinkOp, bpy.types.Operator):
    """Use another shape/text object as this link's source"""

    bl_idname = "seam.link_relink"
    bl_label = "Relink Source"
    allowed_states = ("LIVE", "PAUSED", "ERROR", "CONFLICT", "MISSING_SOURCE")
    state_hint = "Detached links cannot be relinked"

    source_name: StringProperty(name="New Source", default="", options={"SKIP_SAVE"})

    def _candidate(self, context, link):
        if self.source_name:
            return bpy.data.objects.get(self.source_name)
        for ob in context.selected_objects:
            if ob != link.target and ob != link.source and recipes.source_ok(link.recipe, ob)[0]:
                return ob
        return None

    def invoke(self, context, event):
        link = resolve_link(context, self.link_id)
        if link is not None and not self.source_name and self._candidate(context, link) is None:
            return context.window_manager.invoke_props_dialog(self)
        return self.execute(context)

    def draw(self, context):
        self.layout.prop_search(self, "source_name", context.scene, "objects", text="New Source")

    def execute(self, context):
        link = self.get_link(context)
        if link is None:
            return {"CANCELLED"}
        src = self._candidate(context, link)
        if src is None:
            self.report({"ERROR"}, "Choose a new source (select it together with the result, or name it)")
            return {"CANCELLED"}
        ok, why = recipes.source_ok(link.recipe, src)
        if not ok:
            self.report({"ERROR"}, why)
            return {"CANCELLED"}
        sid = ids.ensure_id(src)
        edges = [(lk.source_id, lk.target_id) for lk in context.scene.seam.links if lk.id != link.id]
        if L.would_create_cycle(edges, sid, link.target_id):
            self.report({"ERROR"}, "That source depends on this result (cycle)")
            return {"CANCELLED"}
        link.source = src
        link.source_id = sid
        producer._apply(link, "relink")
        link.source_signature = ""
        link.message = ""
        _report_build(self, link, rebuild_now(context.scene, link))
        return {"FINISHED"}


class SEAM_OT_link_rebuild(_LinkOp, bpy.types.Operator):
    """Rebuild the result from the source now"""

    bl_idname = "seam.link_rebuild"
    bl_label = "Rebuild"
    allowed_states = ("LIVE", "ERROR")
    state_hint = "Resume, resolve the conflict or relink first"

    def execute(self, context):
        link = self.get_link(context)
        if link is None:
            return {"CANCELLED"}
        link.source_signature = ""
        res = rebuild_now(context.scene, link)
        _report_build(self, link, res)
        return {"FINISHED"} if res is None or res.status in ("ok", "skipped") else {"CANCELLED"}


class SEAM_OT_link_remove(_LinkOp, bpy.types.Operator):
    """Remove the live link (the result stays as an ordinary object unless deleted)"""

    bl_idname = "seam.link_remove"
    bl_label = "Remove Link"

    delete_result: BoolProperty(name="Delete Result", default=False)

    def execute(self, context):
        link = self.get_link(context)
        if link is None:
            return {"CANCELLED"}
        remove_link(context.scene, link, delete_result=self.delete_result)
        return {"FINISHED"}


def remove_link(scene, link, delete_result: bool = False) -> None:
    tgt = link.target
    link_id = link.id
    links = scene.seam.links
    idx = next(i for i, lk in enumerate(links) if lk.id == link_id)
    links.remove(idx)
    scene.seam.active_link_index = max(0, min(scene.seam.active_link_index, len(links) - 1))
    if tgt is not None:
        if delete_result:
            bpy.data.objects.remove(tgt, do_unlink=True)
        else:
            if tgt.seam.owner_id == link_id:
                tgt.seam.role = "NONE"
                tgt.seam.owner_id = ""
            for mat in tgt.data.materials:
                if mat is not None and mat.get("seam_link_id") == link_id:
                    mat["seam_link_id"] = ""


class SEAM_OT_edit_source(_LinkOp, bpy.types.Operator):
    """Select the link's source and edit its path"""

    bl_idname = "seam.edit_source"
    bl_label = "Edit Source"
    allowed_states = ("LIVE", "PAUSED", "ERROR", "CONFLICT", "DETACHED")
    state_hint = "The source is missing"

    edit_mode: BoolProperty(name="Enter Edit Mode", default=True)

    @classmethod
    def extra_poll(cls, context, link):
        if link.source is None:
            cls.poll_message_set("The source is missing")
            return False
        return True

    def execute(self, context):
        link = self.get_link(context)
        if link is None or link.source is None:
            return {"CANCELLED"}
        src = link.source
        if context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
        switch_workspace(context, "Vector")
        if not select_only(context, src):
            self.report({"ERROR"}, "The source is not in the current view layer")
            return {"CANCELLED"}
        if self.edit_mode and context.window is not None:
            try:
                bpy.ops.object.mode_set(mode="EDIT")
            except RuntimeError:
                pass
        return {"FINISHED"}


class SEAM_OT_show_result(_LinkOp, bpy.types.Operator):
    """Select the 3D result and show it in the Model workspace"""

    bl_idname = "seam.show_result"
    bl_label = "Show Result"

    @classmethod
    def extra_poll(cls, context, link):
        if link.target is None:
            cls.poll_message_set("The result object is missing")
            return False
        return True

    def execute(self, context):
        link = self.get_link(context)
        if link is None or link.target is None:
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
        switched = switch_workspace(context, "Model")
        if not select_only(context, link.target):
            self.report({"ERROR"}, "The result is not in the current view layer")
            return {"CANCELLED"}
        frame_selected(context, "Model" if switched else "")
        return {"FINISHED"}


class SEAM_OT_duplicate_result(_LinkOp, bpy.types.Operator):
    """Create a second, independent live result from the same source with the same settings"""

    bl_idname = "seam.duplicate_result"
    bl_label = "Duplicate Result"
    allowed_states = ("LIVE", "PAUSED", "ERROR", "CONFLICT", "DETACHED")
    state_hint = "The source is missing"

    @classmethod
    def extra_poll(cls, context, link):
        if link.source is None:
            cls.poll_message_set("The source is missing")
            return False
        return True

    def execute(self, context):
        link = self.get_link(context)
        if link is None or link.source is None:
            return {"CANCELLED"}
        params = {k: getattr(link.params, k) for k in L.RECIPE_PARAMS[link.recipe]}
        new, _tgt = create_link(context, link.source, link.recipe, params)
        _report_build(self, new, rebuild_now(context.scene, new))
        return {"FINISHED"}


classes = (
    SEAM_OT_make_3d,
    SEAM_OT_link_pause,
    SEAM_OT_link_resume,
    SEAM_OT_link_detach,
    SEAM_OT_link_overwrite,
    SEAM_OT_link_branch,
    SEAM_OT_link_relink,
    SEAM_OT_link_rebuild,
    SEAM_OT_link_remove,
    SEAM_OT_edit_source,
    SEAM_OT_show_result,
    SEAM_OT_duplicate_result,
)
