"""Versioned JSON snapshot of SEAM document metadata (owner: core-raster-io).

The .blend file is the project; this snapshot is a portable, validated view of
SEAM metadata (artboards, items, styles, raster docs, links) used for diagnostics,
migration tests and the "Document Health" report. ``migrate`` upgrades older
snapshots step by step; ``validate`` returns a list of human-readable problems.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Dict, List

from . import SCHEMA_VERSION
from .raster import BLEND_MODES
from .units import UNITS

LINK_RECIPES = ("EXTRUDE", "REVOLVE", "SWEEP", "TEXT3D", "DECAL")
LINK_STATES = ("LIVE", "PAUSED", "DETACHED", "ERROR", "CONFLICT", "MISSING_SOURCE")
ITEM_ROLES = ("NONE", "CANVAS_ROOT", "ARTBOARD", "GROUP", "SHAPE", "TEXT", "RASTER", "GUIDE", "STROKE",
              "OUTLINE", "CLIP_RESULT", "RESULT", "INTERNAL")


@dataclass
class LinkRecord:
    id: str
    recipe: str
    source_id: str
    target_id: str
    state: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemRecord:
    id: str
    role: str
    name: str
    layer: str = ""
    style: Dict[str, Any] = field(default_factory=dict)
    shape: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RasterRecord:
    id: str
    name: str
    width: int
    height: int
    layers: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentSnapshot:
    schema_version: int = SCHEMA_VERSION
    units: str = "PX"
    dpi: float = 96.0
    artboards: List[Dict[str, Any]] = field(default_factory=list)
    items: List[ItemRecord] = field(default_factory=list)
    rasters: List[RasterRecord] = field(default_factory=list)
    links: List[LinkRecord] = field(default_factory=list)


def to_dict(doc: DocumentSnapshot) -> Dict[str, Any]:
    """Plain-JSON-able dict (deep copy; safe to mutate)."""
    return copy.deepcopy(asdict(doc))


def _build(cls, data: Any, where: str):
    if not isinstance(data, dict):
        raise ValueError("%s must be an object, got %s" % (where, type(data).__name__))
    names = {f.name for f in fields(cls)}
    required = [f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError("%s is missing required field(s): %s" % (where, ", ".join(missing)))
    return cls(**{k: copy.deepcopy(v) for k, v in data.items() if k in names})


def _list_of(data: Dict[str, Any], key: str) -> List[Any]:
    v = data.get(key, [])
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError("%r must be a list" % key)
    return v


def from_dict(data: Dict[str, Any]) -> DocumentSnapshot:
    """Accepts any supported schema version (migrates first). Raises ValueError if newer than SCHEMA_VERSION."""
    d = migrate(data)
    return DocumentSnapshot(
        schema_version=d["schema_version"],
        units=d.get("units", "PX"),
        dpi=d.get("dpi", 96.0),
        artboards=copy.deepcopy(_list_of(d, "artboards")),
        items=[_build(ItemRecord, x, "items[%d]" % i) for i, x in enumerate(_list_of(d, "items"))],
        rasters=[_build(RasterRecord, x, "rasters[%d]" % i) for i, x in enumerate(_list_of(d, "rasters"))],
        links=[_build(LinkRecord, x, "links[%d]" % i) for i, x in enumerate(_list_of(d, "links"))],
    )


def _migrate_0_to_1(d: Dict[str, Any]) -> Dict[str, Any]:
    for link in d.get("links") or []:
        if isinstance(link, dict) and "kind" in link:
            kind = link.pop("kind")
            link.setdefault("recipe", kind)
    d["schema_version"] = 1
    return d


_MIGRATIONS = {0: _migrate_0_to_1}


def migrate(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy upgraded to SCHEMA_VERSION. Version 0 (pre-release) -> 1 renames 'kind'->'recipe' in links."""
    if not isinstance(data, dict):
        raise ValueError("document snapshot must be an object")
    d = copy.deepcopy(data)
    ver = d.get("schema_version", 0)
    if isinstance(ver, bool) or not isinstance(ver, int) or ver < 0:
        raise ValueError("invalid schema_version %r" % (ver,))
    if ver > SCHEMA_VERSION:
        raise ValueError("snapshot schema_version %d is newer than supported version %d" % (ver, SCHEMA_VERSION))
    while ver < SCHEMA_VERSION:
        d = _MIGRATIONS[ver](d)
        ver = d["schema_version"]
    return d


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 and value == value


def validate(doc: DocumentSnapshot) -> List[str]:
    """Problems: duplicate ids, links whose source/target ids are missing from items, unknown recipes/states,
    raster layers with invalid blend modes, non-positive sizes."""
    problems: List[str] = []
    if doc.schema_version != SCHEMA_VERSION:
        problems.append("schema_version %r does not match current version %d" % (doc.schema_version, SCHEMA_VERSION))
    if doc.units not in UNITS:
        problems.append("unknown document units %r" % (doc.units,))
    if not _positive(doc.dpi):
        problems.append("document dpi must be positive, got %r" % (doc.dpi,))

    seen: Dict[str, str] = {}

    def note_id(value: Any, kind: str) -> None:
        if not isinstance(value, str) or not value:
            problems.append("%s has an empty or non-string id %r" % (kind, value))
            return
        if value in seen:
            problems.append("duplicate id %r (%s and %s)" % (value, seen[value], kind))
        else:
            seen[value] = kind

    for i, ab in enumerate(doc.artboards):
        if not isinstance(ab, dict):
            problems.append("artboards[%d] is not an object" % i)
            continue
        note_id(ab.get("id"), "artboard")
        for dim in ("width", "height"):
            if dim in ab and not _positive(ab[dim]):
                problems.append("artboard %r has non-positive %s %r" % (ab.get("id"), dim, ab[dim]))
        if "export_scale" in ab and not _positive(ab["export_scale"]):
            problems.append("artboard %r has non-positive export_scale %r" % (ab.get("id"), ab["export_scale"]))

    item_ids = set()
    for it in doc.items:
        note_id(it.id, "item")
        item_ids.add(it.id)
        if it.role not in ITEM_ROLES:
            problems.append("item %r has unknown role %r" % (it.id, it.role))

    for r in doc.rasters:
        note_id(r.id, "raster")
        if not _positive(r.width) or not _positive(r.height):
            problems.append("raster %r has non-positive size %rx%r" % (r.id, r.width, r.height))
        layer_ids = set()
        for j, layer in enumerate(r.layers):
            lid = layer.get("id") if isinstance(layer, dict) else None
            if not isinstance(layer, dict):
                problems.append("raster %r layer %d is not an object" % (r.id, j))
                continue
            if lid in layer_ids:
                problems.append("duplicate layer id %r in raster %r" % (lid, r.id))
            layer_ids.add(lid)
            bm = layer.get("blend_mode", "NORMAL")
            if bm not in BLEND_MODES:
                problems.append("raster %r layer %r has invalid blend mode %r" % (r.id, lid, bm))
            op = layer.get("opacity", 1.0)
            if not isinstance(op, (int, float)) or isinstance(op, bool) or not (0.0 <= op <= 1.0):
                problems.append("raster %r layer %r has opacity %r outside 0..1" % (r.id, lid, op))

    for ln in doc.links:
        note_id(ln.id, "link")
        if ln.recipe not in LINK_RECIPES:
            problems.append("link %r has unknown recipe %r" % (ln.id, ln.recipe))
        if ln.state not in LINK_STATES:
            problems.append("link %r has unknown state %r" % (ln.id, ln.state))
        if ln.source_id not in item_ids and ln.state != "MISSING_SOURCE":
            problems.append("link %r references missing source id %r" % (ln.id, ln.source_id))
        if ln.target_id not in item_ids:
            problems.append("link %r references missing target id %r" % (ln.id, ln.target_id))
    return problems
