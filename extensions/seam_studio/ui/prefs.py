"""Add-on preferences (owner: shell).

``get(context)`` always returns an object with the preference attributes: the real
:class:`SEAM_AP_Preferences` when the add-on is enabled, otherwise a defaults shim
(background scripts, preferences not yet available).
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty

from ..seam_core.units import UNITS
from .util import ADDON_ID

DEFAULTS = {
    "show_start_panel": True,
    "default_canvas_width": 1920,
    "default_canvas_height": 1080,
    "default_units": "PX",
    "default_dpi": 96.0,
}


class _DefaultPrefs:
    def __init__(self):
        self.__dict__.update(DEFAULTS)


class SEAM_AP_Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    show_start_panel: BoolProperty(
        name="Show Start Panel", default=DEFAULTS["show_start_panel"],
        description="Show the Start panel (New Vector Document, New Paint Canvas, Draw to 3D) in the SEAM sidebar "
                    "while the document is empty")
    default_canvas_width: IntProperty(
        name="Canvas Width", min=1, max=16384, default=DEFAULTS["default_canvas_width"], subtype="PIXEL",
        description="Width in px of the artboard created for new documents")
    default_canvas_height: IntProperty(
        name="Canvas Height", min=1, max=16384, default=DEFAULTS["default_canvas_height"], subtype="PIXEL",
        description="Height in px of the artboard created for new documents")
    default_units: EnumProperty(
        name="Units", default=DEFAULTS["default_units"],
        items=[(u, "px" if u == "PX" else u.lower(), "") for u in UNITS],
        description="Display units of new documents (the canvas itself is always 1 unit = 1 px)")
    default_dpi: FloatProperty(
        name="DPI", min=1.0, max=4800.0, default=DEFAULTS["default_dpi"],
        description="Resolution used to convert px to physical units in new documents")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(heading="Interface")
        col.prop(self, "show_start_panel")
        col = layout.column(heading="New Documents")
        col.prop(self, "default_canvas_width")
        col.prop(self, "default_canvas_height")
        col.prop(self, "default_units")
        col.prop(self, "default_dpi")


def get(context=None):
    """Preferences of this add-on or a defaults shim with the same attribute names."""
    context = context or bpy.context
    try:
        addon = context.preferences.addons.get(ADDON_ID)
        if addon is not None and addon.preferences is not None:
            return addon.preferences
    except Exception:
        pass
    return _DefaultPrefs()


classes = (SEAM_AP_Preferences,)
