"""SVG interoperability for an explicit, tested subset (owner: core-raster-io).

Export subset: <svg viewBox width height>, <g> (flattened transforms), <path d> with
M/L/C/Z absolute commands, fill (solid / linearGradient / radialGradient via <defs>),
fill-opacity, fill-rule, stroke, stroke-width, stroke-linecap, stroke-linejoin,
stroke-miterlimit, stroke-opacity, opacity, <clipPath>, <image href="data:image/png;base64,...">,
<text> (only when text_mode == "TEXT"; default is outlines as paths).

Import subset: path (M m L l H h V v C c S s Q q T t A a Z z), rect (rx/ry), circle,
ellipse, line, polyline, polygon, g, transform (matrix/translate/scale/rotate/skewX/skewY),
presentation attributes and ``style=""`` declarations, fill/stroke colours (#rgb,
#rrggbb, rgb(), named basic colours, none), linear/radialGradient stops (gradientUnits
objectBoundingBox and userSpaceOnUse), viewBox/width/height. Unsupported elements are
reported in ``ImportResult.warnings`` (never silently dropped).
Colours in the document model are linear RGBA; SVG colours are sRGB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .geometry import Affine, BezierPath

SUPPORTED_EXPORT = ("path", "g", "linearGradient", "radialGradient", "clipPath", "image", "text")
SUPPORTED_IMPORT = ("path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "g",
                    "linearGradient", "radialGradient", "defs", "svg", "title", "desc")


@dataclass
class SvgStyle:
    fill: Optional[object] = (0.0, 0.0, 0.0, 1.0)  # linear RGBA tuple | SvgGradient | None
    stroke: Optional[Tuple[float, float, float, float]] = None
    stroke_width: float = 1.0
    stroke_cap: str = "BUTT"
    stroke_join: str = "MITER"
    miter_limit: float = 4.0
    opacity: float = 1.0
    fill_rule: str = "NONZERO"


@dataclass
class SvgGradient:
    kind: str  # "LINEAR" | "RADIAL"
    stops: List[Tuple[float, Tuple[float, float, float, float]]] = field(default_factory=list)
    # LINEAR: start/end; RADIAL: center/radius — in user space after transforms
    start: Tuple[float, float] = (0.0, 0.0)
    end: Tuple[float, float] = (1.0, 0.0)
    center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0


@dataclass
class SvgShape:
    """One drawable: compound path(s) in canvas coordinates (Y up, SVG Y flipped on import)."""
    name: str
    paths: List[BezierPath]
    style: SvgStyle
    clip_paths: Optional[List[BezierPath]] = None
    text: Optional[str] = None           # export only: original text when text_mode == "TEXT"
    font_family: str = "sans-serif"
    font_size: float = 12.0


@dataclass
class SvgImage:
    name: str
    png_bytes: bytes
    x: float
    y_top: float
    width: float
    height: float
    opacity: float = 1.0


@dataclass
class ImportResult:
    shapes: List[SvgShape]
    width: float
    height: float
    warnings: List[str] = field(default_factory=list)


def parse_path_data(d: str) -> List[BezierPath]:
    """Parse SVG path data (SVG coordinates, Y down) into cubic Bezier paths (arcs/quads converted)."""
    raise NotImplementedError


def format_path_data(paths: List[BezierPath], precision: int = 3) -> str:
    raise NotImplementedError


def parse_transform(value: str) -> Affine:
    raise NotImplementedError


def export_svg(items: List[object], artboard: Tuple[float, float, float, float], text_mode: str = "OUTLINES") -> str:
    """items: SvgShape | SvgImage bottom-to-top, in canvas coordinates (Y up).
    artboard: (x_left, y_top, width, height) in canvas units; output viewBox is 0 0 width height with Y down."""
    raise NotImplementedError


def import_svg(text: str) -> ImportResult:
    """Parse SVG text into shapes in canvas coordinates (Y up, origin = SVG top-left, so y = -svg_y)."""
    raise NotImplementedError
