"""SEAM ui package. Owner adds submodules (each exposing `classes` and optional
register()/unregister()) to `modules` in registration order. Props are registered by
the top-level package before the root property groups."""

from . import props  # noqa: F401

from . import prefs, health, panels, workspaces, keymaps  # noqa: E402

modules = (prefs, health, panels, workspaces, keymaps,)
