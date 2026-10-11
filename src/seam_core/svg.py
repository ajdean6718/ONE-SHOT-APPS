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

import base64
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as _xml_escape
from xml.sax.saxutils import quoteattr as _xml_quoteattr

from .geometry import Affine, BezierPath, BezierPoint, IDENTITY

SUPPORTED_EXPORT = ("path", "g", "linearGradient", "radialGradient", "clipPath", "image", "text")
SUPPORTED_IMPORT = ("path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "g", "a",
                    "linearGradient", "radialGradient", "clipPath", "defs", "svg", "title", "desc", "metadata", "stop")


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




# ---------------------------------------------------------------------------
# Small affine / bezier helpers (private; geometry kernels are owned by another worker)
# ---------------------------------------------------------------------------

def _mul(m1: Affine, m2: Affine) -> Affine:
    """m1 o m2 (apply m2 first)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def _apply(m: Affine, p: Tuple[float, float]) -> Tuple[float, float]:
    return (m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5])


def _det(m: Affine) -> float:
    return m[0] * m[3] - m[1] * m[2]


def _scale_factors(m: Affine) -> Tuple[float, float]:
    return math.hypot(m[0], m[1]), math.hypot(m[2], m[3])


def _xform_path(path: BezierPath, m: Affine) -> BezierPath:
    pts = []
    for p in path.points:
        pts.append(BezierPoint(_apply(m, p.co), _apply(m, p.handle_left), _apply(m, p.handle_right),
                               p.handle_left_type, p.handle_right_type))
    return BezierPath(pts, path.closed)


def _cubic(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
            u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1])


def _flatten(path: BezierPath, resolution: int = 16) -> List[Tuple[float, float]]:
    """Flatten a bezier path to a polyline (closed paths: no repeated closing point)."""
    pts = path.points
    n = len(pts)
    if n == 0:
        return []
    out = [pts[0].co]
    nseg = n if path.closed else n - 1
    for i in range(nseg):
        a = pts[i]
        b = pts[(i + 1) % n]
        straight = a.handle_right == a.co and b.handle_left == b.co
        if straight:
            if not (path.closed and i == nseg - 1):
                out.append(b.co)
        else:
            for k in range(1, resolution + 1):
                if path.closed and i == nseg - 1 and k == resolution:
                    break
                out.append(_cubic(a.co, a.handle_right, b.handle_left, b.co, k / resolution))
    return out


def _bbox_of(paths: Sequence[BezierPath]) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for p in paths:
        for x, y in _flatten(p, 24):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Path data parsing
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARG_COUNT = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
_EPS = 1e-9


def _arc_to_cubics(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2):
    """SVG elliptical arc -> list of cubic segments ((c1x,c1y),(c2x,c2y),(x,y)) (SVG implementation notes F.6)."""
    if x1 == x2 and y1 == y2:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [((x1, y1), (x2, y2), (x2, y2))]  # straight line expressed as a cubic with handles on ends
    phi = math.radians(phi_deg % 360.0)
    cp, sp = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cp * dx2 + sp * dy2
    y1p = -sp * dx2 + cp * dy2
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        sq = math.sqrt(lam)
        rx *= sq
        ry *= sq
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = 0.0 if den == 0 else math.sqrt(max(0.0, num / den))
    if fa == fs:
        coef = -coef
    cxp = coef * (rx * y1p / ry)
    cyp = coef * -(ry * x1p / rx)
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0

    def ang(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        ln = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot / ln))) if ln else 0.0
        return -a if ux * vy - uy * vx < 0 else a

    th1 = ang(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not fs and dth > 0:
        dth -= 2.0 * math.pi
    elif fs and dth < 0:
        dth += 2.0 * math.pi
    nseg = max(1, int(math.ceil(abs(dth) / (math.pi / 2.0) - 1e-9)))
    delta = dth / nseg
    t = 4.0 / 3.0 * math.tan(delta / 4.0)
    segs = []
    a0 = th1
    for i in range(nseg):
        a1 = a0 + delta
        c0, s0 = math.cos(a0), math.sin(a0)
        c1_, s1_ = math.cos(a1), math.sin(a1)
        # unit-circle control points -> ellipse -> rotate -> translate
        p1 = (c0 - t * s0, s0 + t * c0)
        p2 = (c1_ + t * s1_, s1_ - t * c1_)
        p3 = (c1_, s1_)

        def tr(p):
            x = rx * p[0]
            y = ry * p[1]
            return (cp * x - sp * y + cx, sp * x + cp * y + cy)

        e = tr(p3) if i < nseg - 1 else (x2, y2)
        segs.append((tr(p1), tr(p2), e))
        a0 = a1
    return segs


class _PathBuilder:
    def __init__(self):
        self.paths: List[BezierPath] = []
        self.cur: Optional[BezierPath] = None
        self.start = (0.0, 0.0)
        self.pos = (0.0, 0.0)

    def _flush(self):
        if self.cur is not None and len(self.cur.points) >= 2:
            self.paths.append(self.cur)
        self.cur = None

    def move_to(self, p):
        self._flush()
        self.cur = BezierPath([BezierPoint(p, p, p)], False)
        self.start = p
        self.pos = p

    def _ensure(self):
        if self.cur is None:
            p = self.pos
            self.cur = BezierPath([BezierPoint(p, p, p)], False)
            self.start = p

    def line_to(self, p):
        self._ensure()
        self.cur.points.append(BezierPoint(p, p, p))
        self.pos = p

    def cubic_to(self, c1, c2, p):
        self._ensure()
        self.cur.points[-1].handle_right = c1
        self.cur.points.append(BezierPoint(p, c2, p))
        self.pos = p

    def close(self):
        if self.cur is None:
            return
        pts = self.cur.points
        if len(pts) > 1 and math.hypot(pts[-1].co[0] - pts[0].co[0], pts[-1].co[1] - pts[0].co[1]) < _EPS:
            pts[0].handle_left = pts[-1].handle_left
            pts.pop()
        self.cur.closed = True
        self._flush()
        self.pos = self.start

    def finish(self) -> List[BezierPath]:
        self._flush()
        for path in self.paths:
            for pt in path.points:
                pt.handle_left_type = "VECTOR" if pt.handle_left == pt.co else "FREE"
                pt.handle_right_type = "VECTOR" if pt.handle_right == pt.co else "FREE"
        return self.paths


def parse_path_data(d: str) -> List[BezierPath]:
    """Parse SVG path data (SVG coordinates, Y down) into cubic Bezier paths (arcs/quads converted).

    Supports M m L l H h V v C c S s Q q T t A a Z z with implicit command repetition, exponents,
    comma/space separators and packed arc flags. Raises ValueError with the offending offset on bad data.
    """
    if d is None:
        raise ValueError("missing path data")
    s = d
    n = len(s)
    i = 0
    b = _PathBuilder()
    cmd = None
    prev_cmd = None
    last_c2 = None   # reflected control for S
    last_q = None    # reflected control for T
    first = True

    def skip_ws(j):
        while j < n and s[j] in " \t\r\n\f":
            j += 1
        return j

    def skip_sep(j):
        j = skip_ws(j)
        if j < n and s[j] == ",":
            j = skip_ws(j + 1)
        return j

    def read_num(j):
        m = _NUM_RE.match(s, j)
        if not m:
            raise ValueError("expected number at offset %d in path data" % j)
        return float(m.group(0)), m.end()

    def read_flag(j):
        if j < n and s[j] in "01":
            return int(s[j]), j + 1
        raise ValueError("expected arc flag (0 or 1) at offset %d in path data" % j)

    while True:
        i = skip_ws(i)
        if i >= n:
            break
        ch = s[i]
        if ch.isalpha():
            if ch.upper() not in _ARG_COUNT:
                raise ValueError("unknown path command %r at offset %d" % (ch, i))
            cmd = ch
            i += 1
            if first and cmd.upper() != "M":
                raise ValueError("path data must begin with a moveto command")
            if cmd.upper() == "Z":
                b.close()
                prev_cmd = cmd
                last_c2 = last_q = None
                first = False
                continue
            i = skip_ws(i)
        elif cmd is None:
            raise ValueError("path data must begin with a command, found %r at offset %d" % (ch, i))
        elif cmd.upper() == "Z":
            raise ValueError("unexpected number after closepath at offset %d" % i)
        first = False
        this = cmd
        up = cmd.upper()
        rel = cmd.islower()
        args = []
        for k in range(_ARG_COUNT[up]):
            i = skip_sep(i) if k else i
            if up == "A" and k in (3, 4):
                v, i = read_flag(i)
            else:
                v, i = read_num(i)
            args.append(v)
        cx, cy = b.pos
        ox, oy = (cx, cy) if rel else (0.0, 0.0)
        new_c2 = new_q = None
        if up == "M":
            p = (args[0] + ox, args[1] + oy)
            b.move_to(p)
            cmd = "l" if rel else "L"   # implicit repeats are linetos
        elif up == "L":
            b.line_to((args[0] + ox, args[1] + oy))
        elif up == "H":
            b.line_to((args[0] + ox, cy))
        elif up == "V":
            b.line_to((cx, args[0] + oy))
        elif up == "C":
            c1 = (args[0] + ox, args[1] + oy)
            c2 = (args[2] + ox, args[3] + oy)
            p = (args[4] + ox, args[5] + oy)
            b.cubic_to(c1, c2, p)
            new_c2 = c2
        elif up == "S":
            if prev_cmd is not None and prev_cmd.upper() in ("C", "S") and last_c2 is not None:
                c1 = (2 * cx - last_c2[0], 2 * cy - last_c2[1])
            else:
                c1 = (cx, cy)
            c2 = (args[0] + ox, args[1] + oy)
            p = (args[2] + ox, args[3] + oy)
            b.cubic_to(c1, c2, p)
            new_c2 = c2
        elif up == "Q":
            q = (args[0] + ox, args[1] + oy)
            p = (args[2] + ox, args[3] + oy)
            b.cubic_to((cx + 2.0 / 3.0 * (q[0] - cx), cy + 2.0 / 3.0 * (q[1] - cy)),
                       (p[0] + 2.0 / 3.0 * (q[0] - p[0]), p[1] + 2.0 / 3.0 * (q[1] - p[1])), p)
            new_q = q
        elif up == "T":
            if prev_cmd is not None and prev_cmd.upper() in ("Q", "T") and last_q is not None:
                q = (2 * cx - last_q[0], 2 * cy - last_q[1])
            else:
                q = (cx, cy)
            p = (args[0] + ox, args[1] + oy)
            b.cubic_to((cx + 2.0 / 3.0 * (q[0] - cx), cy + 2.0 / 3.0 * (q[1] - cy)),
                       (p[0] + 2.0 / 3.0 * (q[0] - p[0]), p[1] + 2.0 / 3.0 * (q[1] - p[1])), p)
            new_q = q
        elif up == "A":
            p = (args[5] + ox, args[6] + oy)
            segs = _arc_to_cubics(cx, cy, args[0], args[1], args[2], int(args[3]), int(args[4]), p[0], p[1])
            if not segs:
                pass  # endpoints coincide: the arc is omitted
            elif args[0] == 0 or args[1] == 0:
                b.line_to(p)
            else:
                for c1, c2, e in segs:
                    b.cubic_to(c1, c2, e)
        prev_cmd = this
        last_c2 = new_c2
        last_q = new_q
        i = skip_sep(i)  # optional comma between repeated argument sets
    return b.finish()


def _fmt_num(v: float, precision: int) -> str:
    r = round(float(v), precision)
    if r == 0:
        r = 0.0
    t = ("%.*f" % (precision, r)).rstrip("0").rstrip(".")
    return t if t not in ("", "-") else "0"


def format_path_data(paths: List[BezierPath], precision: int = 3) -> str:
    """Absolute M/L/C/Z path data. Straight segments (handles on their end points) become L, others C."""
    parts: List[str] = []
    f = lambda v: _fmt_num(v, precision)  # noqa: E731
    for path in paths:
        pts = path.points
        if not pts:
            continue
        parts.append("M%s %s" % (f(pts[0].co[0]), f(pts[0].co[1])))
        n = len(pts)
        nseg = n if path.closed else n - 1
        for i in range(nseg):
            a = pts[i]
            b = pts[(i + 1) % n]
            last_closing = path.closed and i == nseg - 1
            straight = a.handle_right == a.co and b.handle_left == b.co
            if straight:
                if not last_closing:
                    parts.append("L%s %s" % (f(b.co[0]), f(b.co[1])))
            else:
                parts.append("C%s %s %s %s %s %s" % (f(a.handle_right[0]), f(a.handle_right[1]),
                                                     f(b.handle_left[0]), f(b.handle_left[1]),
                                                     f(b.co[0]), f(b.co[1])))
        if path.closed:
            parts.append("Z")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_TRANSFORM_RE = re.compile(r"\s*([a-zA-Z]+)\s*\(([^)]*)\)\s*,?")


def parse_transform(value: str) -> Affine:
    """Parse an SVG ``transform`` list into one affine (a, b, c, d, e, f). Raises ValueError on bad syntax."""
    if value is None:
        return IDENTITY
    s = value.strip()
    if not s:
        return IDENTITY
    m_total: Affine = IDENTITY
    pos = 0
    while pos < len(s):
        m = _TRANSFORM_RE.match(s, pos)
        if not m or m.start() != pos:
            raise ValueError("invalid transform %r" % value)
        name = m.group(1)
        raw = m.group(2).replace(",", " ").split()
        try:
            a = [float(x) for x in raw]
        except ValueError:
            raise ValueError("invalid number in transform %r" % value)
        if name == "matrix":
            if len(a) != 6:
                raise ValueError("matrix() needs 6 numbers in %r" % value)
            t: Affine = (a[0], a[1], a[2], a[3], a[4], a[5])
        elif name == "translate":
            if len(a) not in (1, 2):
                raise ValueError("translate() needs 1 or 2 numbers in %r" % value)
            t = (1.0, 0.0, 0.0, 1.0, a[0], a[1] if len(a) == 2 else 0.0)
        elif name == "scale":
            if len(a) not in (1, 2):
                raise ValueError("scale() needs 1 or 2 numbers in %r" % value)
            t = (a[0], 0.0, 0.0, a[1] if len(a) == 2 else a[0], 0.0, 0.0)
        elif name == "rotate":
            if len(a) not in (1, 3):
                raise ValueError("rotate() needs 1 or 3 numbers in %r" % value)
            r = math.radians(a[0])
            c, sn = math.cos(r), math.sin(r)
            t = (c, sn, -sn, c, 0.0, 0.0)
            if len(a) == 3:
                t = _mul(_mul((1.0, 0.0, 0.0, 1.0, a[1], a[2]), t), (1.0, 0.0, 0.0, 1.0, -a[1], -a[2]))
        elif name == "skewX":
            if len(a) != 1:
                raise ValueError("skewX() needs 1 number in %r" % value)
            t = (1.0, 0.0, math.tan(math.radians(a[0])), 1.0, 0.0, 0.0)
        elif name == "skewY":
            if len(a) != 1:
                raise ValueError("skewY() needs 1 number in %r" % value)
            t = (1.0, math.tan(math.radians(a[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            raise ValueError("unknown transform function %r" % name)
        m_total = _mul(m_total, t)
        pos = m.end()
    return m_total


# ---------------------------------------------------------------------------
# Colours (SVG colours are sRGB; the document model is scene-linear)
# ---------------------------------------------------------------------------

def _s2l(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _l2s(c: float) -> float:
    return c * 12.92 if c <= 0.0031308 else 1.055 * (max(c, 0.0) ** (1.0 / 2.4)) - 0.055


_NAMED = {
    "black": (0, 0, 0), "silver": (192, 192, 192), "gray": (128, 128, 128), "grey": (128, 128, 128),
    "white": (255, 255, 255), "maroon": (128, 0, 0), "red": (255, 0, 0), "purple": (128, 0, 128),
    "fuchsia": (255, 0, 255), "magenta": (255, 0, 255), "green": (0, 128, 0), "lime": (0, 255, 0),
    "olive": (128, 128, 0), "yellow": (255, 255, 0), "navy": (0, 0, 128), "blue": (0, 0, 255),
    "teal": (0, 128, 128), "aqua": (0, 255, 255), "cyan": (0, 255, 255), "orange": (255, 165, 0),
    "pink": (255, 192, 203), "brown": (165, 42, 42), "gold": (255, 215, 0), "indigo": (75, 0, 130),
    "violet": (238, 130, 238), "crimson": (220, 20, 60), "coral": (255, 127, 80), "salmon": (250, 128, 114),
    "khaki": (240, 230, 140), "beige": (245, 245, 220), "ivory": (255, 255, 240), "tan": (210, 180, 140),
    "turquoise": (64, 224, 208), "lavender": (230, 230, 250), "orchid": (218, 112, 214),
    "plum": (221, 160, 221), "chocolate": (210, 105, 30), "tomato": (255, 99, 71), "skyblue": (135, 206, 235),
    "steelblue": (70, 130, 180), "royalblue": (65, 105, 225), "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144), "lightgray": (211, 211, 211), "lightgrey": (211, 211, 211),
    "darkgray": (169, 169, 169), "darkgrey": (169, 169, 169), "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105), "gainsboro": (220, 220, 220), "whitesmoke": (245, 245, 245),
    "darkred": (139, 0, 0), "darkgreen": (0, 100, 0), "darkblue": (0, 0, 139), "lightblue": (173, 216, 230),
    "lightgreen": (144, 238, 144), "limegreen": (50, 205, 50), "forestgreen": (34, 139, 34),
    "seagreen": (46, 139, 87), "firebrick": (178, 34, 34), "hotpink": (255, 105, 180),
    "deeppink": (255, 20, 147), "darkorange": (255, 140, 0), "orangered": (255, 69, 0),
    "dodgerblue": (30, 144, 255), "cornflowerblue": (100, 149, 237), "midnightblue": (25, 25, 112),
    "darkviolet": (148, 0, 211), "mediumpurple": (147, 112, 219), "sienna": (160, 82, 45),
    "peru": (205, 133, 63), "wheat": (245, 222, 179), "linen": (250, 240, 230), "snow": (255, 250, 250),
    "azure": (240, 255, 255), "mintcream": (245, 255, 250), "aliceblue": (240, 248, 255),
    "goldenrod": (218, 165, 32), "darkcyan": (0, 139, 139), "darkmagenta": (139, 0, 139),
    "chartreuse": (127, 255, 0), "springgreen": (0, 255, 127), "aquamarine": (127, 255, 212),
    "lightyellow": (255, 255, 224), "lightpink": (255, 182, 193), "lightcoral": (240, 128, 128),
    "rosybrown": (188, 143, 143), "thistle": (216, 191, 216), "moccasin": (255, 228, 181),
    "bisque": (255, 228, 196), "mistyrose": (255, 228, 225), "lemonchiffon": (255, 250, 205),
    "darkslategray": (47, 79, 79), "darkslategrey": (47, 79, 79), "olivedrab": (107, 142, 35),
    "yellowgreen": (154, 205, 50), "darkkhaki": (189, 183, 107), "slateblue": (106, 90, 205),
    "mediumblue": (0, 0, 205), "darkturquoise": (0, 206, 209), "cadetblue": (95, 158, 160),
}


def _parse_color(value: str, current: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)):
    """CSS colour -> sRGB (r, g, b, a) floats 0..1. 'none' -> None. Raises ValueError when unparsable."""
    v = value.strip()
    lo = v.lower()
    if lo == "none":
        return None
    if lo == "currentcolor":
        return current
    if lo == "transparent":
        return (0.0, 0.0, 0.0, 0.0)
    if lo.startswith("#"):
        h = lo[1:]
        if not re.fullmatch(r"[0-9a-f]+", h) or len(h) not in (3, 4, 6, 8):
            raise ValueError("bad hex colour %r" % value)
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h)
        vals = [int(h[i:i + 2], 16) / 255.0 for i in range(0, len(h), 2)]
        return (vals[0], vals[1], vals[2], vals[3] if len(vals) == 4 else 1.0)
    m = re.fullmatch(r"(rgba?)\(([^)]*)\)", lo)
    if m:
        parts = [p for p in re.split(r"[\s,/]+", m.group(2).strip()) if p]
        if len(parts) not in (3, 4):
            raise ValueError("bad rgb() colour %r" % value)
        comps = []
        for p in parts[:3]:
            comps.append(float(p[:-1]) / 100.0 if p.endswith("%") else float(p) / 255.0)
        a = 1.0
        if len(parts) == 4:
            a = float(parts[3][:-1]) / 100.0 if parts[3].endswith("%") else float(parts[3])
        return tuple(min(1.0, max(0.0, c)) for c in comps) + (min(1.0, max(0.0, a)),)
    if lo in _NAMED:
        r, g, b = _NAMED[lo]
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)
    raise ValueError("unsupported colour %r" % value)


def _srgb_to_linear_rgba(c):
    return (_s2l(c[0]), _s2l(c[1]), _s2l(c[2]), c[3])


def _color_to_svg(c) -> Tuple[str, float]:
    """Linear RGBA -> (css colour, alpha). Uses #rrggbb when 8-bit exact (within 1e-4 linear), else rgb(%)."""
    s = [min(1.0, max(0.0, _l2s(c[i]))) for i in range(3)]
    q = [int(round(v * 255.0)) for v in s]
    if all(abs(_s2l(qi / 255.0) - min(1.0, max(0.0, c[i]))) < 1e-4 for i, qi in enumerate(q)):
        return "#%02x%02x%02x" % tuple(q), c[3]
    return "rgb(%s%%,%s%%,%s%%)" % tuple(_fmt_num(v * 100.0, 4) for v in s), c[3]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _svg_xf(artboard) -> Affine:
    """canvas (x, y up) -> svg (x - x_left, y_top - y)."""
    x_left, y_top = artboard[0], artboard[1]
    return (1.0, 0.0, 0.0, -1.0, -x_left, y_top)


def _sanitize_id(name: str, used: set, fallback: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.:-]", "_", name or "") or fallback
    if not re.match(r"[A-Za-z_]", base):
        base = "_" + base
    cand = base
    k = 2
    while cand in used:
        cand = "%s_%d" % (base, k)
        k += 1
    used.add(cand)
    return cand


def _num_attr(v: float) -> str:
    return _fmt_num(v, 6)


def export_svg(items: List[object], artboard: Tuple[float, float, float, float], text_mode: str = "OUTLINES") -> str:
    """items: SvgShape | SvgImage bottom-to-top, in canvas coordinates (Y up).
    artboard: (x_left, y_top, width, height) in canvas units; output viewBox is 0 0 width height with Y down."""
    if text_mode not in ("OUTLINES", "TEXT"):
        raise ValueError("text_mode must be 'OUTLINES' or 'TEXT', got %r" % (text_mode,))
    if len(artboard) != 4:
        raise ValueError("artboard must be (x_left, y_top, width, height)")
    x_left, y_top, aw, ah = (float(v) for v in artboard)
    if aw <= 0 or ah <= 0:
        raise ValueError("artboard size must be positive")
    xf = _svg_xf((x_left, y_top))
    PRECISION = 4
    used_ids: set = set()
    defs: List[str] = []
    body: List[str] = []
    grad_cache: Dict[str, str] = {}
    clip_cache: Dict[str, str] = {}

    def add_gradient(g: SvgGradient) -> str:
        if g.kind == "LINEAR":
            s = _apply(xf, g.start)
            e = _apply(xf, g.end)
            geo = 'x1="%s" y1="%s" x2="%s" y2="%s"' % tuple(_fmt_num(v, PRECISION) for v in (s[0], s[1], e[0], e[1]))
            tag = "linearGradient"
        elif g.kind == "RADIAL":
            c = _apply(xf, g.center)
            geo = 'cx="%s" cy="%s" r="%s"' % (_fmt_num(c[0], PRECISION), _fmt_num(c[1], PRECISION), _fmt_num(g.radius, PRECISION))
            tag = "radialGradient"
        else:
            raise ValueError("unknown gradient kind %r" % (g.kind,))
        stops = []
        for pos, col in g.stops:
            css, alpha = _color_to_svg(col)
            st = '<stop offset="%s" stop-color="%s"' % (_fmt_num(pos, 6), css)
            if alpha < 1.0:
                st += ' stop-opacity="%s"' % _num_attr(alpha)
            stops.append(st + "/>")
        key = tag + geo + "".join(stops)
        if key in grad_cache:
            return grad_cache[key]
        gid = _sanitize_id("grad", used_ids, "grad")
        defs.append('<%s id="%s" gradientUnits="userSpaceOnUse" %s>%s</%s>' % (tag, gid, geo, "".join(stops), tag))
        grad_cache[key] = gid
        return gid

    def path_d(paths: List[BezierPath]) -> str:
        return format_path_data([_xform_path(p, xf) for p in paths], PRECISION)

    def add_clip(paths: List[BezierPath]) -> str:
        d = path_d(paths)
        if d in clip_cache:
            return clip_cache[d]
        cid = _sanitize_id("clip", used_ids, "clip")
        defs.append('<clipPath id="%s"><path d=%s clip-rule="evenodd"/></clipPath>' % (cid, _xml_quoteattr(d)))
        clip_cache[d] = cid
        return cid

    for idx, it in enumerate(items):
        if isinstance(it, SvgShape):
            st = it.style
            attrs: List[str] = []
            sid = _sanitize_id(it.name, used_ids, "shape%d" % (idx + 1))
            attrs.append('id="%s"' % sid)
            if it.name and it.name != sid:
                attrs.append("data-name=%s" % _xml_quoteattr(it.name))
            as_text = text_mode == "TEXT" and it.text is not None
            if as_text:
                bb = _bbox_of([_xform_path(p, xf) for p in it.paths])
                tx, ty = (bb[0], bb[3]) if bb else (0.0, 0.0)
                attrs.append('x="%s" y="%s"' % (_fmt_num(tx, PRECISION), _fmt_num(ty, PRECISION)))
                attrs.append("font-family=%s" % _xml_quoteattr(it.font_family))
                attrs.append('font-size="%s"' % _num_attr(it.font_size))
            else:
                attrs.append("d=%s" % _xml_quoteattr(path_d(it.paths)))
            # fill
            fill = st.fill
            if fill is None:
                attrs.append('fill="none"')
            elif isinstance(fill, SvgGradient):
                attrs.append('fill="url(#%s)"' % add_gradient(fill))
            else:
                css, alpha = _color_to_svg(fill)
                attrs.append('fill="%s"' % css)
                if alpha < 1.0:
                    attrs.append('fill-opacity="%s"' % _num_attr(alpha))
            if fill is not None:
                attrs.append('fill-rule="%s"' % ("evenodd" if st.fill_rule == "EVENODD" else "nonzero"))
            # stroke
            if st.stroke is not None:
                css, alpha = _color_to_svg(st.stroke)
                attrs.append('stroke="%s"' % css)
                if alpha < 1.0:
                    attrs.append('stroke-opacity="%s"' % _num_attr(alpha))
                attrs.append('stroke-width="%s"' % _num_attr(st.stroke_width))
                attrs.append('stroke-linecap="%s"' % {"BUTT": "butt", "ROUND": "round", "SQUARE": "square"}.get(st.stroke_cap, "butt"))
                attrs.append('stroke-linejoin="%s"' % {"MITER": "miter", "ROUND": "round", "BEVEL": "bevel"}.get(st.stroke_join, "miter"))
                attrs.append('stroke-miterlimit="%s"' % _num_attr(st.miter_limit))
            if st.opacity < 1.0:
                attrs.append('opacity="%s"' % _num_attr(st.opacity))
            if it.clip_paths is not None:
                attrs.append('clip-path="url(#%s)"' % add_clip(it.clip_paths))
            if as_text:
                body.append("<text %s>%s</text>" % (" ".join(attrs), _xml_escape(it.text)))
            else:
                body.append("<path %s/>" % " ".join(attrs))
        elif isinstance(it, SvgImage):
            iid = _sanitize_id(it.name, used_ids, "image%d" % (idx + 1))
            tl = _apply(xf, (it.x, it.y_top))
            data = base64.b64encode(it.png_bytes).decode("ascii")
            attrs = ['id="%s"' % iid]
            if it.name and it.name != iid:
                attrs.append("data-name=%s" % _xml_quoteattr(it.name))
            attrs.append('x="%s" y="%s" width="%s" height="%s"' % (
                _fmt_num(tl[0], PRECISION), _fmt_num(tl[1], PRECISION), _fmt_num(it.width, PRECISION), _fmt_num(it.height, PRECISION)))
            attrs.append('preserveAspectRatio="none"')
            if it.opacity < 1.0:
                attrs.append('opacity="%s"' % _num_attr(it.opacity))
            attrs.append('href="data:image/png;base64,%s"' % data)
            body.append("<image %s/>" % " ".join(attrs))
        else:
            raise TypeError("export_svg items must be SvgShape or SvgImage, got %s" % type(it).__name__)

    W, H = _fmt_num(aw, PRECISION), _fmt_num(ah, PRECISION)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           'width="%s" height="%s" viewBox="0 0 %s %s">' % (W, H, W, H)]
    if defs:
        out.append("<defs>")
        out.extend(defs)
        out.append("</defs>")
    out.extend(body)
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
_SHAPE_TAGS = ("path", "rect", "circle", "ellipse", "line", "polyline", "polygon")
_GROUP_TAGS = ("g", "a", "svg")
_QUIET_TAGS = ("title", "desc", "metadata")
_INHERITED = ("fill", "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-opacity", "stroke-linecap",
              "stroke-linejoin", "stroke-miterlimit", "color", "visibility", "stroke-dasharray")
_PRESENTATION = _INHERITED + ("opacity", "display", "clip-path", "stop-color", "stop-opacity", "filter", "mask",
                              "marker", "marker-start", "marker-mid", "marker-end", "mix-blend-mode",
                              "vector-effect", "clip-rule")
_UNITS = {"px": 1.0, "pt": 96.0 / 72.0, "pc": 16.0, "mm": 96.0 / 25.4, "cm": 96.0 / 2.54, "in": 96.0,
          "em": 16.0, "ex": 8.0, "": 1.0}
_LEN_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(px|pt|pc|mm|cm|in|em|ex|%)?\s*$")
_FLIP: Affine = (1.0, 0.0, 0.0, -1.0, 0.0, 0.0)


def _split_tag(tag) -> Tuple[str, str]:
    if not isinstance(tag, str):
        return "", ""
    if tag.startswith("{"):
        ns, _, local = tag[1:].partition("}")
        return ns, local
    return "", tag


def _parse_len(value: str, ref: Optional[float] = None) -> float:
    m = _LEN_RE.match(value)
    if not m:
        raise ValueError("invalid length %r" % value)
    v = float(m.group(1))
    unit = m.group(2) or ""
    if unit == "%":
        if ref is None:
            raise ValueError("percentage length %r not allowed here" % value)
        return v * ref / 100.0
    return v * _UNITS[unit]


def _parse_style_attr(style: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for decl in style.split(";"):
        if ":" in decl:
            k, _, v = decl.partition(":")
            k = k.strip().lower()
            v = v.strip()
            if k:
                out[k] = v
    return out


class _Ctx:
    __slots__ = ("ctm", "style", "opacity", "clip")

    def __init__(self, ctm, style, opacity, clip):
        self.ctm = ctm
        self.style = style
        self.opacity = opacity
        self.clip = clip


class _Importer:
    def __init__(self, root: ET.Element):
        self.root = root
        self.warnings: List[str] = []
        self.shapes: List[SvgShape] = []
        self.ids: Dict[str, ET.Element] = {}
        self.counter = 0
        self.vp = (100.0, 100.0)  # viewport for percentage lengths
        for el in root.iter():
            i = el.get("id")
            if i and i not in self.ids:
                self.ids[i] = el

    # -- diagnostics ----------------------------------------------------------------------------
    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def label(self, el: ET.Element) -> str:
        _, tag = _split_tag(el.tag)
        i = el.get("id")
        return "<%s%s>" % (tag, " id=%r" % i if i else "")

    # -- attribute helpers ------------------------------------------------------------------------
    def props(self, el: ET.Element) -> Dict[str, str]:
        out = {}
        for k in _PRESENTATION:
            if k in el.attrib:
                out[k] = el.attrib[k].strip()
        st = el.get("style")
        if st:
            out.update({k: v for k, v in _parse_style_attr(st).items()})
        return out

    def href(self, el: ET.Element) -> Optional[str]:
        h = el.get(_XLINK_HREF) or el.get("href")
        return h.strip() if h else None

    def length(self, el: ET.Element, name: str, default: float = 0.0, axis: str = "x") -> float:
        v = el.get(name)
        if v is None:
            return default
        ref = self.vp[0] if axis == "x" else self.vp[1] if axis == "y" else math.sqrt((self.vp[0] ** 2 + self.vp[1] ** 2) / 2.0)
        try:
            return _parse_len(v, ref)
        except ValueError:
            self.warn("%s: invalid %s=%r, using %s" % (self.label(el), name, v, default))
            return default

    # -- traversal --------------------------------------------------------------------------------
    def run(self) -> ImportResult:
        root = self.root
        ns, tag = _split_tag(root.tag)
        vb = None
        if root.get("viewBox"):
            try:
                nums = [float(x) for x in root.get("viewBox").replace(",", " ").split()]
                if len(nums) != 4 or nums[2] <= 0 or nums[3] <= 0:
                    raise ValueError
                vb = tuple(nums)
            except ValueError:
                self.warn("<svg>: invalid viewBox %r ignored" % root.get("viewBox"))

        def dim(name):
            v = root.get(name)
            if v is None:
                return None
            try:
                if v.strip().endswith("%"):
                    return None
                return _parse_len(v)
            except ValueError:
                self.warn("<svg>: invalid %s=%r ignored" % (name, v))
                return None

        w, h = dim("width"), dim("height")
        if vb:
            if w is None and h is None:
                w, h = vb[2], vb[3]
            elif w is None:
                w = h * vb[2] / vb[3]
            elif h is None:
                h = w * vb[3] / vb[2]
        else:
            if w is None or h is None:
                self.warn("<svg>: no viewBox and missing width/height; assuming %s x %s px" % (
                    "100" if w is None else _fmt_num(w, 3), "100" if h is None else _fmt_num(h, 3)))
                w = 100.0 if w is None else w
                h = 100.0 if h is None else h
        ctm: Affine = IDENTITY
        if vb:
            sx, sy = w / vb[2], h / vb[3]
            par = (root.get("preserveAspectRatio") or "xMidYMid meet").split()
            align = par[0] if par else "xMidYMid"
            slice_ = len(par) > 1 and par[1] == "slice"
            if align == "none":
                ctm = (sx, 0.0, 0.0, sy, -vb[0] * sx, -vb[1] * sy)
            else:
                s = max(sx, sy) if slice_ else min(sx, sy)
                fx = {"xMin": 0.0, "xMid": 0.5, "xMax": 1.0}.get(align[:4], 0.5)
                fy = {"YMin": 0.0, "YMid": 0.5, "YMax": 1.0}.get(align[4:], 0.5)
                ctm = (s, 0.0, 0.0, s, -vb[0] * s + (w - vb[2] * s) * fx, -vb[1] * s + (h - vb[3] * s) * fy)
            self.vp = (vb[2], vb[3])
        else:
            self.vp = (w, h)
        if ns != _SVG_NS and ns != "":
            self.warn("root element is in namespace %r, expected the SVG namespace" % ns)
        ctx = _Ctx(ctm, {}, 1.0, None)
        root_props = self.props(root)
        ctx = self.enter(root, ctx, root_props, is_root=True)
        if ctx is not None:
            self.walk(root, ctx)
        return ImportResult(self.shapes, float(w), float(h), self.warnings)

    def enter(self, el: ET.Element, ctx: _Ctx, props: Dict[str, str], is_root: bool = False) -> Optional[_Ctx]:
        """Build the child context for ``el`` (transform, inherited style, opacity, clip) or None if not rendered."""
        if props.get("display", "").lower() == "none":
            return None
        ctm = ctx.ctm
        if not is_root:
            tr = el.get("transform")
            if tr:
                try:
                    ctm = _mul(ctm, parse_transform(tr))
                except ValueError as exc:
                    self.warn("%s: %s; element skipped" % (self.label(el), exc))
                    return None
        style = dict(ctx.style)
        for k in _INHERITED:
            if k in props and props[k].lower() != "inherit":
                style[k] = props[k]
        opacity = ctx.opacity
        if "opacity" in props:
            try:
                o = float(props["opacity"][:-1]) / 100.0 if props["opacity"].endswith("%") else float(props["opacity"])
                opacity *= min(1.0, max(0.0, o))
            except ValueError:
                self.warn("%s: invalid opacity %r ignored" % (self.label(el), props["opacity"]))
        clip = ctx.clip
        cp = props.get("clip-path")
        if cp and cp.lower() != "none":
            own = self.resolve_clip(el, cp, ctm)
            if own is not None:
                if clip is not None and not is_root:
                    self.warn("%s: nested clip paths are not intersected; the innermost clip is used" % self.label(el))
                clip = own
        for attr in ("filter", "mask"):
            if props.get(attr, "none").lower() != "none":
                self.warn("%s: %s attribute is not supported and was ignored" % (self.label(el), attr))
        if any(props.get(a, "none").lower() != "none" for a in ("marker", "marker-start", "marker-mid", "marker-end")):
            self.warn("%s: markers are not supported and were ignored" % self.label(el))
        if props.get("mix-blend-mode", "normal").lower() != "normal":
            self.warn("%s: mix-blend-mode is not supported and was ignored" % self.label(el))
        return _Ctx(ctm, style, opacity, clip)

    def walk(self, parent: ET.Element, ctx: _Ctx) -> None:
        kids = [c for c in parent if isinstance(c.tag, str)]
        for el in kids:
            ns, tag = _split_tag(el.tag)
            if ns != _SVG_NS and ns != "":
                continue  # foreign namespaces (inkscape, sodipodi, ...) are editor metadata
            if tag in _QUIET_TAGS or tag in ("linearGradient", "radialGradient", "clipPath", "stop"):
                continue
            if tag == "defs":
                self.scan_defs(el)
                continue
            if tag in _SHAPE_TAGS:
                self.shape(el, tag, ctx)
            elif tag in _GROUP_TAGS:
                props = self.props(el)
                child = self.enter(el, ctx, props)
                if child is None:
                    continue
                if tag == "svg":
                    tx, ty = self.length(el, "x"), self.length(el, "y", axis="y")
                    child.ctm = _mul(child.ctm, (1.0, 0.0, 0.0, 1.0, tx, ty))
                    if el.get("viewBox"):
                        self.warn("%s: nested <svg> viewBox/viewport clipping is not supported; treated as a group" % self.label(el))
                if tag == "g" and "opacity" in props and child.opacity != ctx.opacity and len([k for k in el if isinstance(k.tag, str)]) > 1:
                    self.warn("%s: group opacity is applied to each child separately" % self.label(el))
                self.walk(el, child)
            else:
                self.unsupported(el, tag)

    def unsupported(self, el: ET.Element, tag: str) -> None:
        if tag == "use":
            self.warn("%s: <use> is not supported on import; referenced geometry %r was ignored" % (self.label(el), self.href(el)))
        elif tag in ("text", "tspan", "textPath"):
            txt = "".join(el.itertext()).strip()
            self.warn("%s: text is not supported on import (%r ignored); convert text to outlines first" % (self.label(el), txt[:40]))
        elif tag == "image":
            self.warn("%s: <image> is not supported on import and was ignored" % self.label(el))
        elif tag == "style":
            self.warn("<style> element is not supported; CSS rules and class selectors were ignored")
        else:
            self.warn("%s: <%s> is not supported on import and was ignored" % (self.label(el), tag))

    def scan_defs(self, defs: ET.Element) -> None:
        for el in defs:
            ns, tag = _split_tag(el.tag)
            if ns != _SVG_NS and ns != "":
                continue
            if tag in _QUIET_TAGS or tag in ("linearGradient", "radialGradient", "clipPath") or tag in _SHAPE_TAGS or tag in ("g", "a", "defs"):
                if tag == "defs":
                    self.scan_defs(el)
                continue
            self.unsupported(el, tag)

    # -- clip paths -------------------------------------------------------------------------------
    def resolve_clip(self, el: ET.Element, value: str, ctm: Affine) -> Optional[List[BezierPath]]:
        m = re.match(r"^url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)", value)
        if not m:
            self.warn("%s: unsupported clip-path %r ignored" % (self.label(el), value))
            return None
        cp = self.ids.get(m.group(1))
        if cp is None or _split_tag(cp.tag)[1] != "clipPath":
            self.warn("%s: clip-path references missing clipPath #%s; ignored" % (self.label(el), m.group(1)))
            return None
        if cp.get("clipPathUnits") == "objectBoundingBox":
            self.warn("%s: clipPathUnits=objectBoundingBox is not supported; clip ignored" % self.label(el))
            return None
        m_clip = ctm
        if cp.get("transform"):
            try:
                m_clip = _mul(m_clip, parse_transform(cp.get("transform")))
            except ValueError as exc:
                self.warn("%s: %s; clip ignored" % (self.label(cp), exc))
                return None
        paths: List[BezierPath] = []
        for child in cp:
            ns, tag = _split_tag(child.tag)
            if not isinstance(child.tag, str) or (ns != _SVG_NS and ns != ""):
                continue
            if tag in _SHAPE_TAGS:
                if self.props(child).get("display", "").lower() == "none":
                    continue
                local = self.geometry(child, tag)
                mm = m_clip
                if child.get("transform"):
                    try:
                        mm = _mul(mm, parse_transform(child.get("transform")))
                    except ValueError as exc:
                        self.warn("%s: %s; skipped" % (self.label(child), exc))
                        continue
                paths.extend(_xform_path(p, mm) for p in local)
            elif tag in _QUIET_TAGS:
                continue
            else:
                self.warn("%s: <%s> inside clipPath is not supported and was ignored" % (self.label(cp), tag))
        return paths

    # -- geometry ---------------------------------------------------------------------------------
    def geometry(self, el: ET.Element, tag: str) -> List[BezierPath]:
        L = self.label(el)
        try:
            if tag == "path":
                d = el.get("d")
                if d is None:
                    self.warn("%s: missing 'd' attribute; skipped" % L)
                    return []
                return parse_path_data(d)
            if tag == "rect":
                x, y = self.length(el, "x"), self.length(el, "y", axis="y")
                w, h = self.length(el, "width"), self.length(el, "height", axis="y")
                if w <= 0 or h <= 0:
                    self.warn("%s: zero or negative size; not rendered" % L)
                    return []
                rx_s, ry_s = el.get("rx"), el.get("ry")
                rx = self.length(el, "rx") if rx_s is not None else None
                ry = self.length(el, "ry", axis="y") if ry_s is not None else None
                if rx is None and ry is None:
                    rx = ry = 0.0
                elif rx is None:
                    rx = ry
                elif ry is None:
                    ry = rx
                rx, ry = min(max(rx, 0.0), w / 2.0), min(max(ry, 0.0), h / 2.0)
                f = lambda v: repr(float(v))  # noqa: E731
                if rx > 0 and ry > 0:
                    d = ("M%s %s H%s A%s %s 0 0 1 %s %s V%s A%s %s 0 0 1 %s %s H%s A%s %s 0 0 1 %s %s V%s A%s %s 0 0 1 %s %s Z" % (
                        f(x + rx), f(y), f(x + w - rx), f(rx), f(ry), f(x + w), f(y + ry), f(y + h - ry),
                        f(rx), f(ry), f(x + w - rx), f(y + h), f(x + rx), f(rx), f(ry), f(x), f(y + h - ry),
                        f(y + ry), f(rx), f(ry), f(x + rx), f(y)))
                else:
                    d = "M%s %s H%s V%s H%s Z" % (f(x), f(y), f(x + w), f(y + h), f(x))
                return parse_path_data(d)
            if tag in ("circle", "ellipse"):
                cx, cy = self.length(el, "cx"), self.length(el, "cy", axis="y")
                if tag == "circle":
                    rx = ry = self.length(el, "r", axis="d")
                else:
                    rx, ry = self.length(el, "rx"), self.length(el, "ry", axis="y")
                if rx <= 0 or ry <= 0:
                    self.warn("%s: zero or negative radius; not rendered" % L)
                    return []
                f = lambda v: repr(float(v))  # noqa: E731
                d = "M%s %s A%s %s 0 1 0 %s %s A%s %s 0 1 0 %s %s Z" % (
                    f(cx + rx), f(cy), f(rx), f(ry), f(cx - rx), f(cy), f(rx), f(ry), f(cx + rx), f(cy))
                return parse_path_data(d)
            if tag == "line":
                x1, y1 = self.length(el, "x1"), self.length(el, "y1", axis="y")
                x2, y2 = self.length(el, "x2"), self.length(el, "y2", axis="y")
                return parse_path_data("M%r %r L%r %r" % (x1, y1, x2, y2))
            if tag in ("polyline", "polygon"):
                nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", el.get("points") or "")]
                if len(nums) % 2:
                    self.warn("%s: odd number of coordinates in points; last value dropped" % L)
                    nums = nums[:-1]
                if len(nums) < 4:
                    self.warn("%s: fewer than two points; not rendered" % L)
                    return []
                pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
                d = "M" + " L".join("%r %r" % p for p in pts) + (" Z" if tag == "polygon" else "")
                return parse_path_data(d)
        except ValueError as exc:
            self.warn("%s: %s; skipped" % (L, exc))
            return []
        return []

    # -- shapes -----------------------------------------------------------------------------------
    def shape(self, el: ET.Element, tag: str, ctx: _Ctx) -> None:
        props = self.props(el)
        c = self.enter(el, ctx, props)
        if c is None:
            return
        if c.style.get("visibility", "visible").lower() in ("hidden", "collapse"):
            return
        local = self.geometry(el, tag)
        if not local:
            return
        L = self.label(el)
        style = self.compute_style(el, c, local)
        ctm = c.ctm
        sx, sy = _scale_factors(ctm)
        if abs(sx - sy) > 0.01 * max(sx, sy, 1e-12) or abs(ctm[0] * ctm[2] + ctm[1] * ctm[3]) > 0.01 * sx * sy:
            if style.stroke is not None:
                self.warn("%s: non-uniform scale or skew; stroke width is approximated" % L)
        paths = [_xform_path(p, _mul(_FLIP, ctm)) for p in local]
        clip = None
        if c.clip is not None:
            clip = [_xform_path(p, _FLIP) for p in c.clip]
        if isinstance(style.fill, SvgGradient):
            g = style.fill
            g.start = _apply(_FLIP, g.start)
            g.end = _apply(_FLIP, g.end)
            g.center = _apply(_FLIP, g.center)
        self.counter += 1
        name = el.get("data-name") or el.get(_INKSCAPE_LABEL) or el.get("id") or "%s_%d" % (tag, self.counter)
        self.shapes.append(SvgShape(name=name, paths=paths, style=style, clip_paths=clip))

    def compute_style(self, el: ET.Element, c: _Ctx, local: List[BezierPath]) -> SvgStyle:
        L = self.label(el)
        st = c.style
        try:
            current = _parse_color(st.get("color", "black"))
        except ValueError:
            current = (0.0, 0.0, 0.0, 1.0)
        if current is None:
            current = (0.0, 0.0, 0.0, 1.0)

        def opacity_of(key: str) -> float:
            v = st.get(key)
            if v is None:
                return 1.0
            try:
                return min(1.0, max(0.0, float(v[:-1]) / 100.0 if v.endswith("%") else float(v)))
            except ValueError:
                self.warn("%s: invalid %s %r ignored" % (L, key, v))
                return 1.0

        fill_op = opacity_of("fill-opacity")
        stroke_op = opacity_of("stroke-opacity")
        fill = self.resolve_paint(el, st.get("fill", "black"), current, fill_op, local, c.ctm, "fill")
        stroke = self.resolve_paint(el, st.get("stroke", "none"), current, stroke_op, local, c.ctm, "stroke")
        if isinstance(stroke, SvgGradient):
            self.warn("%s: gradient strokes are not supported; using the last gradient stop colour" % L)
            stroke = stroke.stops[-1][1] if stroke.stops else None
        scale = math.sqrt(abs(_det(c.ctm)))
        sw = 1.0
        if "stroke-width" in st:
            try:
                sw = _parse_len(st["stroke-width"], math.sqrt((self.vp[0] ** 2 + self.vp[1] ** 2) / 2.0))
            except ValueError:
                self.warn("%s: invalid stroke-width %r; using 1" % (L, st["stroke-width"]))
        cap = {"butt": "BUTT", "round": "ROUND", "square": "SQUARE"}.get(st.get("stroke-linecap", "butt").lower())
        if cap is None:
            self.warn("%s: unknown stroke-linecap %r; using butt" % (L, st.get("stroke-linecap")))
            cap = "BUTT"
        join_raw = st.get("stroke-linejoin", "miter").lower()
        join = {"miter": "MITER", "round": "ROUND", "bevel": "BEVEL"}.get(join_raw)
        if join is None:
            if join_raw in ("miter-clip", "arcs"):
                self.warn("%s: stroke-linejoin %r approximated by miter" % (L, join_raw))
            else:
                self.warn("%s: unknown stroke-linejoin %r; using miter" % (L, join_raw))
            join = "MITER"
        ml = 4.0
        if "stroke-miterlimit" in st:
            try:
                ml = max(1.0, float(st["stroke-miterlimit"]))
            except ValueError:
                self.warn("%s: invalid stroke-miterlimit %r; using 4" % (L, st["stroke-miterlimit"]))
        rule = {"nonzero": "NONZERO", "evenodd": "EVENODD"}.get(st.get("fill-rule", "nonzero").lower())
        if rule is None:
            self.warn("%s: unknown fill-rule %r; using nonzero" % (L, st.get("fill-rule")))
            rule = "NONZERO"
        da = st.get("stroke-dasharray", "none").strip().lower()
        if da not in ("none", "") and stroke is not None:
            self.warn("%s: stroke-dasharray is not supported; stroke drawn solid" % L)
        return SvgStyle(fill=fill, stroke=stroke, stroke_width=sw * scale, stroke_cap=cap, stroke_join=join,
                        miter_limit=ml, opacity=c.opacity, fill_rule=rule)

    def resolve_paint(self, el, value, current, alpha, local, ctm, what):
        """-> linear RGBA tuple | SvgGradient (svg space) | None."""
        L = self.label(el)
        v = value.strip()
        fallback = None
        m = re.match(r"^url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)\s*(.*)$", v)
        if m:
            ref = self.ids.get(m.group(1))
            rest = m.group(2).strip()
            if rest:
                try:
                    fallback = _parse_color(rest, current)
                except ValueError:
                    fallback = None
            tag = _split_tag(ref.tag)[1] if ref is not None else ""
            if tag in ("linearGradient", "radialGradient"):
                g = self.build_gradient(el, ref, local, ctm, alpha)
                return g
            if ref is None:
                self.warn("%s: %s references missing paint server #%s" % (L, what, m.group(1)))
            else:
                self.warn("%s: %s paint server <%s> is not supported%s" % (L, what, tag, "; using fallback colour" if fallback else ""))
            if fallback is None:
                return None
            c = fallback
        else:
            try:
                c = _parse_color(v, current)
            except ValueError as exc:
                self.warn("%s: %s: %s; paint ignored" % (L, what, exc))
                return None if what == "stroke" else (0.0, 0.0, 0.0, alpha)
            if c is None:
                return None
        lin = _srgb_to_linear_rgba(c)
        return (lin[0], lin[1], lin[2], lin[3] * alpha)

    def gradient_chain(self, grad: ET.Element) -> List[ET.Element]:
        chain, seen = [], set()
        el = grad
        while el is not None and id(el) not in seen:
            chain.append(el)
            seen.add(id(el))
            h = self.href(el)
            nxt = self.ids.get(h[1:]) if h and h.startswith("#") else None
            if nxt is not None and _split_tag(nxt.tag)[1] not in ("linearGradient", "radialGradient"):
                nxt = None
            el = nxt
        return chain

    def build_gradient(self, shape_el, grad, local, ctm, alpha):
        L = self.label(shape_el)
        chain = self.gradient_chain(grad)

        def gattr(name, default=None):
            for e in chain:
                if name in e.attrib:
                    return e.attrib[name]
            return default

        stops_el = []
        for e in chain:
            stops_el = [s for s in e if isinstance(s.tag, str) and _split_tag(s.tag)[1] == "stop"]
            if stops_el:
                break
        stops = []
        last = 0.0
        for s in stops_el:
            p = self.props(s)
            try:
                off_raw = s.get("offset", "0").strip()
                off = float(off_raw[:-1]) / 100.0 if off_raw.endswith("%") else float(off_raw)
            except ValueError:
                self.warn("%s: invalid stop offset %r; using 0" % (self.label(grad), s.get("offset")))
                off = 0.0
            off = min(1.0, max(0.0, off))
            off = max(off, last)
            last = off
            try:
                cur = _parse_color(self.props(shape_el).get("color", "black"))
            except ValueError:
                cur = (0.0, 0.0, 0.0, 1.0)
            try:
                col = _parse_color(p.get("stop-color", "black"), cur or (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 0.0)
            except ValueError as exc:
                self.warn("%s: %s; stop colour set to black" % (self.label(grad), exc))
                col = (0.0, 0.0, 0.0, 1.0)
            try:
                so = p.get("stop-opacity")
                so = 1.0 if so is None else (float(so[:-1]) / 100.0 if so.endswith("%") else float(so))
            except ValueError:
                so = 1.0
            lin = _srgb_to_linear_rgba(col)
            stops.append((off, (lin[0], lin[1], lin[2], lin[3] * min(1.0, max(0.0, so)) * alpha)))
        if not stops:
            self.warn("%s: gradient #%s has no stops; paint ignored" % (L, grad.get("id")))
            return None
        if len(stops) == 1:
            return stops[0][1]
        spread = gattr("spreadMethod", "pad")
        if spread != "pad":
            self.warn("%s: gradient spreadMethod=%s is not supported; using pad" % (L, spread))
        units = gattr("gradientUnits", "objectBoundingBox")
        try:
            gt = parse_transform(gattr("gradientTransform", ""))
        except ValueError as exc:
            self.warn("%s: %s; gradientTransform ignored" % (self.label(grad), exc))
            gt = IDENTITY
        bbox_units = units != "userSpaceOnUse"
        if bbox_units:
            bb = _bbox_of(local)
            if bb is None or bb[2] - bb[0] <= 0 or bb[3] - bb[1] <= 0:
                self.warn("%s: gradient on a zero-size bounding box; paint ignored" % L)
                return None
            bbm: Affine = (bb[2] - bb[0], 0.0, 0.0, bb[3] - bb[1], bb[0], bb[1])
            M = _mul(_mul(ctm, bbm), gt)
        else:
            M = _mul(ctm, gt)

        def coord(name, default, axis):
            raw = gattr(name, default)
            raw = str(raw).strip()
            try:
                if bbox_units:
                    return float(raw[:-1]) / 100.0 if raw.endswith("%") else float(raw)
                ref = self.vp[0] if axis == "x" else self.vp[1] if axis == "y" else math.sqrt((self.vp[0] ** 2 + self.vp[1] ** 2) / 2.0)
                return _parse_len(raw, ref)
            except ValueError:
                self.warn("%s: invalid gradient %s=%r; using default" % (self.label(grad), name, raw))
                return coord(name, default, axis) if raw != str(default) else 0.0

        sx, sy = _scale_factors(M)
        aniso = abs(sx - sy) > 0.01 * max(sx, sy, 1e-12) or abs(M[0] * M[2] + M[1] * M[3]) > 0.01 * sx * sy
        kind = _split_tag(grad.tag)[1]
        if kind == "linearGradient":
            x1, y1 = coord("x1", "0%", "x"), coord("y1", "0%", "y")
            x2, y2 = coord("x2", "100%", "x"), coord("y2", "0%", "y")
            if aniso and abs(x2 - x1) > 1e-9 and abs(y2 - y1) > 1e-9:
                self.warn("%s: gradient under non-uniform scale/skew is approximated" % L)
            return SvgGradient("LINEAR", stops, start=_apply(M, (x1, y1)), end=_apply(M, (x2, y2)))
        cx, cy = coord("cx", "50%", "x"), coord("cy", "50%", "y")
        r = coord("r", "50%", "d")
        fx, fy = gattr("fx"), gattr("fy")
        if fx is not None or fy is not None:
            if (fx is not None and abs(coord("fx", fx, "x") - cx) > 1e-9) or (fy is not None and abs(coord("fy", fy, "y") - cy) > 1e-9):
                self.warn("%s: radial gradient focal point is not supported; using the centre" % L)
        if aniso:
            self.warn("%s: radial gradient under non-uniform scale is approximated by a circle" % L)
        if r <= 0:
            return stops[-1][1]
        return SvgGradient("RADIAL", stops, center=_apply(M, (cx, cy)), radius=r * math.sqrt(abs(_det(M))))


def import_svg(text: str) -> ImportResult:
    """Parse SVG text into shapes in canvas coordinates (Y up, origin = SVG top-left, so y = -svg_y)."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if "<!ENTITY" in text:
        raise ValueError("SVG with entity declarations is not supported")
    try:
        root = ET.fromstring(text.lstrip("﻿").strip())
    except ET.ParseError as exc:
        raise ValueError("invalid SVG XML: %s" % exc)
    ns, tag = _split_tag(root.tag)
    if tag != "svg":
        raise ValueError("not an SVG document (root element is <%s>)" % tag)
    return _Importer(root).run()
