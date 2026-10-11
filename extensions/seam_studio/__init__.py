"""SEAM Studio — Illustrator/Photoshop-class 2D authoring with live links into Blender's native 3D.

Registration order (owner: integrator): domain property groups -> root property groups ->
engine -> domain modules (link, vector, paint, ui). Domain packages list their
submodules in ``<package>.modules``.
"""

bl_info = {
    "name": "SEAM Studio",
    "author": "SEAM Studio contributors",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "Workspaces: Vector, Paint, Model, Split · Sidebar > SEAM",
    "description": "Vector drawing, raster painting and live non-destructive 2D-to-3D links",
    "category": "Interface",
    "doc_url": "",
    "support": "COMMUNITY",
}

import bpy

from . import compat, ids, props  # noqa: F401
from . import engine, link, paint, ui, vector
from .engine import props as engine_props
from .link import props as link_props
from .paint import props as paint_props
from .ui import props as ui_props
from .vector import props as vector_props

_PROP_MODULES = (vector_props, link_props, paint_props, engine_props, ui_props)
_DOMAINS = (link, vector, paint, ui)
_registered = []


def _register_classes(module):
    for cls in getattr(module, "classes", ()):
        bpy.utils.register_class(cls)
        _registered.append(cls)


def _domain_modules():
    for pkg in _DOMAINS:
        for mod in getattr(pkg, "modules", ()):
            yield mod


def register():
    for mod in _PROP_MODULES:
        _register_classes(mod)
    _register_classes(props)
    props.register()
    engine.register()
    for mod in _domain_modules():
        _register_classes(mod)
        if hasattr(mod, "register"):
            mod.register()


def unregister():
    for mod in reversed(list(_domain_modules())):
        if hasattr(mod, "unregister"):
            try:
                mod.unregister()
            except Exception as exc:  # keep unregistering the rest
                print("SEAM: unregister failed for", mod.__name__, exc)
    engine.unregister()
    props.unregister()
    while _registered:
        cls = _registered.pop()
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
