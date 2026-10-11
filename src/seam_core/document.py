"""Versioned JSON snapshot of SEAM document metadata (owner: core-raster-io).

The .blend file is the project; this snapshot is a portable, validated view of
SEAM metadata (artboards, items, styles, raster docs, links) used for diagnostics,
migration tests and the "Document Health" report. ``migrate`` upgrades older
snapshots step by step; ``validate`` returns a list of human-readable problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import SCHEMA_VERSION


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
    raise NotImplementedError


def from_dict(data: Dict[str, Any]) -> DocumentSnapshot:
    """Accepts any supported schema version (migrates first). Raises ValueError if newer than SCHEMA_VERSION."""
    raise NotImplementedError


def migrate(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy upgraded to SCHEMA_VERSION. Version 0 (pre-release) -> 1 renames 'kind'->'recipe' in links."""
    raise NotImplementedError


def validate(doc: DocumentSnapshot) -> List[str]:
    """Problems: duplicate ids, links whose source/target ids are missing from items, unknown recipes/states,
    raster layers with invalid blend modes, non-positive sizes."""
    raise NotImplementedError
