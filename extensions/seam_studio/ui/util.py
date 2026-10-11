"""Small helpers shared by the shell modules (owner: shell). No classes."""

from __future__ import annotations

import bpy

classes = ()

#: Module name this add-on is registered under ("seam_studio" or "bl_ext.<repo>.seam_studio").
ADDON_ID = __package__.rpartition(".")[0]


def op_callable(idname: str):
    """Return the ``bpy.ops`` callable for ``"seam.some_op"`` or None when it is not registered.

    (``getattr(bpy.ops.x, name)`` succeeds for unknown names; ``get_rna_type`` raises ``KeyError`` for them.)
    """
    try:
        category, name = idname.split(".", 1)
        fn = getattr(getattr(bpy.ops, category), name)
        fn.get_rna_type()
        return fn
    except (AttributeError, KeyError, ValueError):
        return None


def op_property_names(idname: str):
    """Property names of a registered operator (empty tuple if unknown)."""
    fn = op_callable(idname)
    if fn is None:
        return ()
    try:
        return tuple(p.identifier for p in fn.get_rna_type().properties if p.identifier != "rna_type")
    except Exception:
        return ()


def op_available(idname: str, props=()):
    """``(ok, reason)``: operator registered and accepting every name in ``props``."""
    if op_callable(idname) is None:
        return False, f"operator {idname} is not registered"
    have = set(op_property_names(idname))
    missing = [p for p in props if p not in have]
    if missing:
        return False, f"operator {idname} lacks properties: {', '.join(missing)}"
    return True, ""


def window_for(context=None):
    """A window to use for operator overrides, or None (background mode / no GUI)."""
    context = context or bpy.context
    win = getattr(context, "window", None)
    if win is not None:
        return win
    wm = bpy.context.window_manager
    return wm.windows[0] if len(wm.windows) else None
