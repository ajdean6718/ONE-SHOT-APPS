"""Add-on keymaps: documented, non-conflicting combinations (owner: shell).

Items live in the add-on keyconfig under the ``3D View`` keymap (the only way an add-on keymap becomes active in
the viewport). An item is bound only when its operator is registered (domain add-on modules register before
``ui``), so a missing operator never leaves a dead shortcut.

Combinations avoid Blender's default keymaps for the same editor: :func:`find_conflicts` compares
:data:`KEY_ITEMS` against Blender's default keymap data (``keymap_data/blender_default.py``) for the 3D View and the
global (Window / Screen) keymaps and against each other. A test asserts the list is empty.
"""

from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import bpy

from .. import compat
from . import workspaces
from .util import op_available

KEYMAP_NAME = "3D View"
SPACE_TYPE = "VIEW_3D"
REGION_TYPE = "WINDOW"


class KeyItem(NamedTuple):
    idname: str
    type: str
    value: str = "PRESS"
    ctrl: bool = False
    alt: bool = False
    shift: bool = False
    props: Tuple[Tuple[str, object], ...] = ()
    description: str = ""


def _ws_items():
    keys = ("ONE", "TWO", "THREE", "FOUR")
    return [KeyItem("seam.switch_workspace", key, ctrl=True, alt=True, props=(("name", name),),
                    description=f"Switch to the {name} workspace")
            for key, name in zip(keys, workspaces.SEAM_WORKSPACES)]


#: Every shortcut SEAM defines (bound when the operator exists).
KEY_ITEMS: List[KeyItem] = _ws_items() + [
    KeyItem("seam.toggle_live", "L", ctrl=True, alt=True, description="Pause / resume live updates"),
    KeyItem("seam.view_artboard", "ZERO", ctrl=True, alt=True, description="Fit the active artboard in the view"),
    KeyItem("seam.group", "J", ctrl=True, alt=True, description="Group selected artwork"),
    KeyItem("seam.ungroup", "J", ctrl=True, alt=True, shift=True, description="Ungroup selected artwork"),
]

_registered: List[Tuple[object, object, KeyItem]] = []  # (keymap, keymap_item, definition)

_MODS = ("ctrl", "alt", "shift")


def key_text(item: KeyItem) -> str:
    """Human readable combination, e.g. ``Ctrl Alt 1``."""
    names = {"ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4", "ZERO": "0"}
    parts = [m.capitalize() for m in _MODS if getattr(item, m)]
    parts.append(names.get(item.type, item.type.replace("_", " ").title()))
    return " ".join(parts)


def available_items() -> List[KeyItem]:
    """Items whose operator is registered right now."""
    return [it for it in KEY_ITEMS if op_available(it.idname)[0]]


def registered_items() -> List[KeyItem]:
    """Definitions currently bound in the add-on keyconfig."""
    return [d for _km, _kmi, d in _registered]


# ------------------------------------------------------------------ conflict detection

def _default_keymap_data():
    """Blender's default keymap definition (pure data, works headless on 4.0 and 5.x)."""
    path = None
    for base in (bpy.utils.system_resource("SCRIPTS"),):
        cand = os.path.join(base, "presets", "keyconfig", "keymap_data", "blender_default.py")
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        raise RuntimeError("Blender default keymap data not found")
    module = bpy.utils.execfile(path)
    return module.generate_keymaps(module.Params())


def _flatten(data) -> List[Tuple[str, str, str, dict]]:
    """``[(keymap_name, space_type, region_type, item_properties)]`` from keymap data."""
    out = []
    for name, space, content in data:
        for entry in content.get("items", ()):
            # entry = (idname, key_dict, props_dict_or_None)
            key = entry[1]
            out.append((name, space.get("space_type", "EMPTY"), space.get("region_type", "WINDOW"), key))
    return out


def _same_combo(item: KeyItem, key: dict) -> bool:
    if key.get("type") != item.type:
        return False
    if bool(key.get("ctrl")) != item.ctrl or bool(key.get("alt")) != item.alt or bool(key.get("shift")) != item.shift:
        return False
    if key.get("oskey") or key.get("any"):
        return bool(key.get("any"))
    if key.get("key_modifier") not in (None, "NONE"):
        return False
    return True


def find_conflicts(items: Optional[List[KeyItem]] = None, default_data=None) -> List[str]:
    """Descriptions of shortcut clashes between ``items`` and Blender's defaults for the 3D View / global keymaps,
    or between items themselves. Empty list = no conflicts."""
    items = KEY_ITEMS if items is None else items
    data = default_data if default_data is not None else _default_keymap_data()
    flat = _flatten(data)
    out = []
    for it in items:
        for name, space, _region, key in flat:
            if space not in (SPACE_TYPE, "EMPTY"):
                continue
            if _same_combo(it, key):
                out.append(f"{key_text(it)} ({it.idname}) conflicts with Blender default in keymap '{name}'")
    seen: Dict[Tuple, KeyItem] = {}
    for it in items:
        combo = (it.type, it.ctrl, it.alt, it.shift)
        if combo in seen:
            out.append(f"{key_text(it)} is used by both {seen[combo].idname} and {it.idname}")
        seen[combo] = it
    return out


# ------------------------------------------------------------------ registration

def register():
    kc = compat.keyconfig_addon()
    if kc is None:
        return
    km = kc.keymaps.new(name=KEYMAP_NAME, space_type=SPACE_TYPE, region_type=REGION_TYPE)
    for it in available_items():
        kmi = km.keymap_items.new(it.idname, it.type, it.value, ctrl=it.ctrl, alt=it.alt, shift=it.shift)
        for prop, value in it.props:
            setattr(kmi.properties, prop, value)
        _registered.append((km, kmi, it))


def unregister():
    while _registered:
        km, kmi, _it = _registered.pop()
        try:
            km.keymap_items.remove(kmi)
        except (RuntimeError, ReferenceError):
            pass


classes = ()
