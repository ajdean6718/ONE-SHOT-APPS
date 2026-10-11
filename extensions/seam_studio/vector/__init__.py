"""SEAM vector package. Owner adds submodules (each exposing `classes` and optional
register()/unregister()) to `modules` in registration order. Props are registered by
the top-level package before the root property groups."""

from . import props  # noqa: F401  (ui has no props; see ui/props.py)

from . import canvas, shapes, style, arrange, tools, text, pathfinder, clip, freehand, io_svg, export_png  # noqa: E402

modules = (canvas, shapes, style, arrange, tools, text, pathfinder, clip, freehand, io_svg, export_png,)
