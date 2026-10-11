"""Text creation, typography panel, live outlines (owner: vector-b).

Text is a native FONT object (role ``TEXT``) so Blender's text editing, fonts and text boxes keep
working. SEAM adds:

* ``seam.add_text(body, x, y, size)`` - (x, y) are canvas-local coordinates of the text origin
  (left end of the first baseline for the default alignment), ``size`` is the em size in px.
* Typography panel (sidebar > SEAM) with font list (``SEAM_MT_fonts`` + ``seam.font_load``),
  size, character / word / line spacing, horizontal and vertical alignment, text-box width/height
  (0 = unbounded) and Bold / Italic / Underline / Small Caps.  Style buttons act on the selection
  while the text is in edit mode and on the whole body otherwise.
* ``seam.text_to_outline``: creates a SHAPE curve (``owner_id`` = text id) whose splines are the
  text's glyph contours. ``OutlineProducer`` (engine kind ``"outline"``, priority 5) regenerates the
  outline whenever the text's evaluated outline changes (body, font, size, spacing ...); the text
  object is hidden but kept (non-destructive).  Editing the outline's nodes is allowed; the next
  change of the text replaces them.  ``seam.outline_expand`` stops following (and deletes the hidden
  text unless ``keep_text``), ``seam.outline_revert`` shows the text again and deletes the outline.
* The outline keeps its own transform: it starts from the text's placement and can then be moved,
  rotated and scaled independently.
"""

from __future__ import annotations

from typing import List, Optional

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from .. import ids
from ..bridge import paths_signature, text_paths, write_paths
from ..engine import core as engine
from ..engine.core import BuildResult, Producer
from . import canvas
from .export_png import copy_style
from .pathfinder import delete_artwork, select_only

SIG_PROP = "seam_outline_sig"
FONT_EXTENSIONS = ("*.ttf", "*.otf", "*.ttc", "*.otc", "*.pfb", "*.pfm", "*.woff", "*.woff2")

_STYLE_ATTR = {"BOLD": "use_bold", "ITALIC": "use_italic", "UNDERLINE": "use_underline",
               "SMALL_CAPS": "use_small_caps"}
_SELECT_ATTR = {"BOLD": "is_select_bold", "ITALIC": "is_select_italic", "UNDERLINE": "is_select_underline",
                "SMALL_CAPS": "is_select_smallcaps"}
_OPS_STYLE = {"BOLD": "BOLD", "ITALIC": "ITALIC", "UNDERLINE": "UNDERLINE", "SMALL_CAPS": "SMALL_CAPS"}


# ------------------------------------------------------------------ helpers

def is_text(obj) -> bool:
    return obj is not None and obj.type == "FONT" and obj.seam.role == "TEXT"


def outlines_of(scene, text) -> List[bpy.types.Object]:
    sid = text.seam.id
    if not sid:
        return []
    return [o for o in scene.objects if o.seam.role == "SHAPE" and o.seam.owner_id == sid and o.type == "CURVE"]


def outline_owner(scene, outline):
    """TEXT object a live outline follows, or None."""
    if outline.seam.role != "SHAPE" or not outline.seam.owner_id:
        return None
    owner = ids.find_object(outline.seam.owner_id, scene)
    return owner if is_text(owner) else None


def style_state(obj, style: str) -> bool:
    """Whether ``style`` is active for the selection (edit mode) or the whole body."""
    cu = obj.data
    if obj.mode == "EDIT":
        return bool(getattr(cu, _SELECT_ATTR[style]))
    fmt = cu.body_format
    return len(fmt) > 0 and all(getattr(f, _STYLE_ATTR[style]) for f in fmt)


def set_body_style(obj, style: str, value: bool) -> None:
    attr = _STYLE_ATTR[style]
    for f in obj.data.body_format:
        setattr(f, attr, value)
    obj.data.update_tag()


def create_text(scene, body: str, x: float, y: float, size: float, name: str = "Text", layer=None):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    ob = bpy.data.objects.new(name, cu)
    ob.location = (x, y, 0.0)
    canvas.add_artwork_object(scene, ob, layer=layer, role="TEXT")
    ob.seam.style.fill_color = (0.0, 0.0, 0.0, 1.0)
    ob.seam.style.stroke_enabled = False
    return ob


# ------------------------------------------------------------------ operators

class SEAM_OT_add_text(bpy.types.Operator):
    """Add an editable text object to the canvas"""
    bl_idname = "seam.add_text"
    bl_label = "Add Text"
    bl_options = {"REGISTER", "UNDO"}

    body: StringProperty(name="Text", default="Text")
    x: FloatProperty(name="X", default=0.0, unit="NONE", description="Canvas X of the text origin (px)")
    y: FloatProperty(name="Y", default=0.0, unit="NONE", description="Canvas Y of the text origin (px, Y up)")
    size: FloatProperty(name="Size", default=48.0, min=1.0, soft_max=500.0, unit="NONE",
                        description="Font size in px")
    place_on_artboard: BoolProperty(name="Place on Artboard", default=False, options={"HIDDEN", "SKIP_SAVE"},
                                    description="Ignore X/Y and place near the top-left of the active artboard")

    def invoke(self, context, event):
        self.place_on_artboard = True
        return self.execute(context)

    def execute(self, context):
        scene = context.scene
        if not self.body:
            self.report({"ERROR"}, "Text cannot be empty")
            return {"CANCELLED"}
        x, y = self.x, self.y
        if self.place_on_artboard:
            from .export_png import artboard_rect
            canvas.ensure_canvas(scene)
            ab = canvas.active_artboard(scene)
            if ab is not None:
                ax, ay, _w, _h = artboard_rect(ab)
                x, y = ax + 40.0, ay - 40.0 - self.size
        ob = create_text(scene, self.body, x, y, self.size, name="Text")
        select_only(context, ob)
        return {"FINISHED"}


class SEAM_OT_font_load(bpy.types.Operator, ImportHelper):
    """Load a font file (TTF/OTF/...) and use it for the active text"""
    bl_idname = "seam.font_load"
    bl_label = "Load Font"
    bl_options = {"REGISTER", "UNDO"}

    filter_glob: StringProperty(default=";".join(FONT_EXTENSIONS), options={"HIDDEN"})
    slot: EnumProperty(name="Use As", items=[
        ("REGULAR", "Regular", "Regular font of the active text"),
        ("BOLD", "Bold", "Font used for bold characters"),
        ("ITALIC", "Italic", "Font used for italic characters"),
        ("BOLD_ITALIC", "Bold Italic", "Font used for bold italic characters"),
        ("NONE", "Only Load", "Load the font without assigning it"),
    ], default="REGULAR")

    def execute(self, context):
        path = bpy.path.abspath(self.filepath)
        try:
            font = bpy.data.fonts.load(path, check_existing=True)
        except (RuntimeError, OSError) as exc:
            self.report({"ERROR"}, f"Could not load font '{path}': {exc}")
            return {"CANCELLED"}
        ob = context.active_object
        if self.slot != "NONE" and is_text(ob):
            attr = {"REGULAR": "font", "BOLD": "font_bold", "ITALIC": "font_italic",
                    "BOLD_ITALIC": "font_bold_italic"}[self.slot]
            setattr(ob.data, attr, font)
        self.report({"INFO"}, f"Loaded font {font.name}")
        return {"FINISHED"}


class SEAM_OT_font_set(bpy.types.Operator):
    """Use a loaded font for the active text"""
    bl_idname = "seam.font_set"
    bl_label = "Set Font"
    bl_options = {"REGISTER", "UNDO"}

    font: StringProperty(name="Font")

    @classmethod
    def poll(cls, context):
        if not is_text(context.active_object):
            cls.poll_message_set("Select a text object")
            return False
        return True

    def execute(self, context):
        font = bpy.data.fonts.get(self.font)
        if font is None:
            self.report({"ERROR"}, f"Font '{self.font}' is not loaded")
            return {"CANCELLED"}
        context.active_object.data.font = font
        return {"FINISHED"}


class SEAM_MT_fonts(bpy.types.Menu):
    bl_idname = "SEAM_MT_fonts"
    bl_label = "Fonts"

    def draw(self, context):
        layout = self.layout
        for font in bpy.data.fonts:
            layout.operator(SEAM_OT_font_set.bl_idname, text=font.name).font = font.name
        layout.separator()
        layout.operator(SEAM_OT_font_load.bl_idname, text="Load Font...", icon="FILEBROWSER")


class SEAM_OT_text_style(bpy.types.Operator):
    """Toggle Bold / Italic / Underline / Small Caps for the selected characters (edit mode) or the whole text"""
    bl_idname = "seam.text_style"
    bl_label = "Text Style"
    bl_options = {"REGISTER", "UNDO"}

    style: EnumProperty(name="Style", items=[("BOLD", "Bold", ""), ("ITALIC", "Italic", ""),
                                             ("UNDERLINE", "Underline", ""), ("SMALL_CAPS", "Small Caps", "")],
                        default="BOLD")
    state: EnumProperty(name="State", items=[("TOGGLE", "Toggle", ""), ("ON", "On", ""), ("OFF", "Off", "")],
                        default="TOGGLE")

    @classmethod
    def poll(cls, context):
        if not is_text(context.active_object):
            cls.poll_message_set("Select a text object")
            return False
        return True

    def execute(self, context):
        ob = context.active_object
        if ob.mode == "EDIT":
            current = style_state(ob, self.style)
            want = (not current) if self.state == "TOGGLE" else self.state == "ON"
            if want == current:
                return {"FINISHED"}
            bpy.ops.font.style_toggle(style=_OPS_STYLE[self.style])
            return {"FINISHED"}
        current = style_state(ob, self.style)
        want = (not current) if self.state == "TOGGLE" else self.state == "ON"
        set_body_style(ob, self.style, want)
        return {"FINISHED"}


# ------------------------------------------------------------------ outlines

def create_outline(scene, text):
    """Create the live outline shape for ``text`` (does not hide the text)."""
    paths = text_paths(text)
    if not paths:
        raise ValueError(f"'{text.name}' has no glyph outlines (empty text, or the font has no glyphs for it)")
    cu = bpy.data.curves.new(f"{text.name} Outline", "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.resolution_u = 12
    write_paths(cu, paths)
    ob = bpy.data.objects.new(cu.name, cu)
    layer = next((c for c in canvas.layers(scene) if text.name in c.objects), None)
    canvas.add_artwork_object(scene, ob, layer=layer, role="SHAPE", on_top=False)
    ob.parent = text.parent
    ob.matrix_parent_inverse = text.matrix_parent_inverse.copy()
    ob.location = text.location
    ob.rotation_mode = text.rotation_mode
    ob.rotation_euler = text.rotation_euler
    ob.scale = text.scale
    ob.seam.owner_id = ids.ensure_id(text)
    copy_style(text.seam.style, ob.seam.style)
    ob.seam.style.fill_rule = "NONZERO"
    ob[SIG_PROP] = paths_signature(paths)
    return ob


class SEAM_OT_text_to_outline(bpy.types.Operator):
    """Create editable outline shapes from the selected text. The text stays (hidden) and the outline follows its changes"""
    bl_idname = "seam.text_to_outline"
    bl_label = "Text to Outline"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not any(is_text(o) for o in context.selected_objects):
            cls.poll_message_set("Select a text object")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        made = []
        problems = []
        for text in [o for o in context.selected_objects if is_text(o)]:
            if outlines_of(scene, text):
                problems.append(f"'{text.name}' already has an outline")
                continue
            try:
                outline = create_outline(scene, text)
            except ValueError as exc:
                problems.append(str(exc))
                continue
            text.hide_set(True)
            made.append(outline)
        if not made:
            self.report({"ERROR"}, "; ".join(problems) or "Select a text object")
            return {"CANCELLED"}
        for msg in problems:
            self.report({"WARNING"}, msg)
        for o in context.view_layer.objects:
            try:
                o.select_set(o in made)
            except RuntimeError:
                pass
        context.view_layer.objects.active = made[-1]
        engine.mark_dirty(f"outline:{made[-1].seam.id}")
        self.report({"INFO"}, f"Created {len(made)} live outline(s)")
        return {"FINISHED"}


def _selected_outlines(context):
    return [o for o in context.selected_objects if outline_owner(context.scene, o) is not None]


class SEAM_OT_outline_expand(bpy.types.Operator):
    """Make the selected outline independent: it stops following its text"""
    bl_idname = "seam.outline_expand"
    bl_label = "Expand Outline"
    bl_options = {"REGISTER", "UNDO"}

    keep_text: BoolProperty(name="Keep Text", default=False,
                            description="Keep the hidden source text object instead of deleting it")

    @classmethod
    def poll(cls, context):
        if not _selected_outlines(context):
            cls.poll_message_set("Select a live text outline")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        outlines = _selected_outlines(context)
        if not outlines:
            self.report({"ERROR"}, "Select a live text outline")
            return {"CANCELLED"}
        for o in outlines:
            text = outline_owner(scene, o)
            o.seam.owner_id = ""
            if SIG_PROP in o.keys():
                del o[SIG_PROP]
            if text is not None and not self.keep_text:
                delete_artwork(text)
        self.report({"INFO"}, f"Expanded {len(outlines)} outline(s)")
        return {"FINISHED"}


class SEAM_OT_outline_revert(bpy.types.Operator):
    """Show the source text again and delete the selected outline"""
    bl_idname = "seam.outline_revert"
    bl_label = "Revert Outline to Text"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if not _selected_outlines(context):
            cls.poll_message_set("Select a live text outline")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        outlines = _selected_outlines(context)
        if not outlines:
            self.report({"ERROR"}, "Select a live text outline")
            return {"CANCELLED"}
        last = None
        for o in outlines:
            text = outline_owner(scene, o)
            if text is None:
                continue
            delete_artwork(o)
            text.hide_set(False)
            last = text
        if last is not None:
            select_only(context, last)
        return {"FINISHED"}


# ------------------------------------------------------------------ producer

class OutlineProducer(Producer):
    kind = "outline"
    priority = 5

    def _keys_for_object(self, scene, ob):
        keys = []
        if is_text(ob):
            keys.extend(f"outline:{o.seam.id}" for o in outlines_of(scene, ob))
        elif ob.seam.role == "SHAPE" and ob.seam.owner_id and outline_owner(scene, ob) is not None:
            keys.append(f"outline:{ob.seam.id}")
        return keys

    def keys_for_update(self, scene, id_orig, geometry, transform):
        if isinstance(id_orig, bpy.types.Object):
            return self._keys_for_object(scene, id_orig)
        if isinstance(id_orig, bpy.types.Curve):
            keys = []
            for ob in scene.objects:
                if ob.data is id_orig:
                    keys.extend(self._keys_for_object(scene, ob))
            return keys
        if isinstance(id_orig, bpy.types.VectorFont):
            keys = []
            for ob in scene.objects:
                if ob.type == "FONT" and is_text(ob):
                    keys.extend(self._keys_for_object(scene, ob))
            return keys
        return ()

    def all_keys(self, scene):
        for ob in scene.objects:
            if ob.seam.role == "SHAPE" and ob.seam.owner_id and outline_owner(scene, ob) is not None:
                yield f"outline:{ob.seam.id}"

    def _pair(self, scene, key):
        outline = ids.find_object(key.split(":", 1)[1], scene)
        if outline is None:
            return None, None
        return outline, outline_owner(scene, outline)

    def signature(self, scene, key):
        outline, text = self._pair(scene, key)
        if outline is None or text is None:
            return None
        return paths_signature(text_paths(text))

    def stored_signature(self, scene, key):
        outline, _text = self._pair(scene, key)
        return outline.get(SIG_PROP) if outline is not None else None

    def build(self, scene, key, signature):
        outline, text = self._pair(scene, key)
        if outline is None or text is None:
            return BuildResult(key, "obsolete")
        if outline.mode != "OBJECT":
            return BuildResult(key, "waiting", "outline is being edited")
        paths = text_paths(text)
        write_paths(outline.data, paths)
        outline[SIG_PROP] = signature
        return BuildResult(key, "ok", f"{len(paths)} contours")


_producer = OutlineProducer()


# ------------------------------------------------------------------ UI

class SEAM_PT_text(bpy.types.Panel):
    bl_idname = "SEAM_PT_text"
    bl_label = "Text"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"

    def draw(self, context):
        layout = self.layout
        layout.operator(SEAM_OT_add_text.bl_idname, text="Add Text", icon="FONT_DATA")
        col = layout.column(align=True)
        col.operator(SEAM_OT_text_to_outline.bl_idname, text="Create Outlines")
        row = col.row(align=True)
        row.operator(SEAM_OT_outline_expand.bl_idname, text="Expand")
        row.operator(SEAM_OT_outline_revert.bl_idname, text="Revert to Text")


class SEAM_PT_typography(bpy.types.Panel):
    bl_idname = "SEAM_PT_typography"
    bl_label = "Typography"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"

    @classmethod
    def poll(cls, context):
        return is_text(context.active_object)

    def draw(self, context):
        layout = self.layout
        ob = context.active_object
        cu = ob.data
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(cu, "body", text="Text")
        layout.menu(SEAM_MT_fonts.bl_idname, text=cu.font.name if cu.font else "Font", icon="FONT_DATA")
        col = layout.column(align=True)
        col.prop(cu, "size", text="Size")
        col.prop(cu, "space_character", text="Character")
        col.prop(cu, "space_word", text="Word")
        col.prop(cu, "space_line", text="Line")
        col = layout.column(align=True)
        col.prop(cu, "align_x", text="Align")
        col.prop(cu, "align_y", text="Vertical")
        row = layout.row(align=True)
        for style, label in (("BOLD", "B"), ("ITALIC", "I"), ("UNDERLINE", "U"), ("SMALL_CAPS", "Sc")):
            op = row.operator(SEAM_OT_text_style.bl_idname, text=label, depress=style_state(ob, style))
            op.style = style
            op.state = "TOGGLE"
        box = cu.text_boxes[0] if len(cu.text_boxes) else None
        if box is not None:
            col = layout.column(align=True)
            col.prop(box, "width", text="Box Width")
            col.prop(box, "height", text="Box Height")


class SEAM_PT_typography_fonts(bpy.types.Panel):
    bl_idname = "SEAM_PT_typography_fonts"
    bl_label = "Font Variants"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SEAM"
    bl_parent_id = "SEAM_PT_typography"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        cu = context.active_object.data
        layout.use_property_split = True
        layout.prop(cu, "font_bold", text="Bold")
        layout.prop(cu, "font_italic", text="Italic")
        layout.prop(cu, "font_bold_italic", text="Bold Italic")
        layout.operator(SEAM_OT_font_load.bl_idname, text="Load Font...", icon="FILEBROWSER")


classes = (SEAM_OT_add_text, SEAM_OT_font_load, SEAM_OT_font_set, SEAM_MT_fonts, SEAM_OT_text_style,
           SEAM_OT_text_to_outline, SEAM_OT_outline_expand, SEAM_OT_outline_revert,
           SEAM_PT_text, SEAM_PT_typography, SEAM_PT_typography_fonts)


def register():
    engine.register_producer(_producer)


def unregister():
    engine.unregister_producer(OutlineProducer.kind)
