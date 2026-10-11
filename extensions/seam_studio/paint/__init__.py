"""SEAM paint package. Owner adds submodules (each exposing `classes` and optional
register()/unregister()) to `modules` in registration order. Props are registered by
the top-level package before the root property groups."""

from . import props  # noqa: F401  (ui has no props; see ui/props.py)

from . import rasterdoc, ops, selection, io, ui  # noqa: E402

modules = (rasterdoc, ops, selection, io, ui,)
