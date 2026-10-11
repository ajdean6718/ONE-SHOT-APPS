"""Pure link model: recipes, parameters, state machine, cycle checks (owner: link).

The Blender side stores links in ``scene.seam.links`` (SEAM_PG_Link); this module
holds the rules so they are unit-testable without Blender.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Tuple

RECIPES = ("EXTRUDE", "REVOLVE", "SWEEP", "TEXT3D", "DECAL")
STATES = ("LIVE", "PAUSED", "DETACHED", "ERROR", "CONFLICT", "MISSING_SOURCE")

# Allowed source roles per recipe (SEAM_PG_Object.role)
SOURCE_ROLES = {
    "EXTRUDE": ("SHAPE",),
    "REVOLVE": ("SHAPE",),
    "SWEEP": ("SHAPE",),
    "TEXT3D": ("TEXT",),
    "DECAL": ("SHAPE", "TEXT", "RASTER", "GROUP"),
}

# name: (default, min, max)  — enum params use a tuple of allowed values as "min" and None as "max"
PARAMS: Dict[str, Tuple[Any, Any, Any]] = {
    "depth": (0.2, 0.0, 100.0),               # metres
    "bevel_depth": (0.0, 0.0, 10.0),          # metres
    "bevel_resolution": (2, 0, 16),
    "resolution": (12, 1, 64),                # curve samples per segment
    "orientation": ("UPRIGHT", ("UPRIGHT", "FLAT"), None),
    "origin": ("BOTTOM", ("CENTER", "BOTTOM", "SOURCE", "MIN"), None),
    "units_scale": (0.01, 1e-6, 1000.0),      # metres per canvas px
    "revolve_angle": (6.283185307179586, 0.01, 6.283185307179586),
    "revolve_segments": (32, 3, 512),
    "revolve_axis": ("LEFT", ("LEFT", "CENTER", "SOURCE_ORIGIN"), None),
    "axis_offset": (0.0, -1e6, 1e6),          # canvas px, added to the axis position
    "sweep_radius": (0.05, 0.0001, 100.0),    # metres
    "sweep_profile": ("ROUND", ("ROUND", "SQUARE", "RIBBON"), None),
    "sweep_profile_resolution": (4, 1, 32),
    "fill_caps": (True, None, None),
    "decal_mode": ("PLANE", ("MATERIAL", "PLANE"), None),
    "decal_resolution": (1024, 16, 8192),     # px of the longer side
    "use_source_material": (True, None, None),
}

# Parameters each recipe actually uses (others are ignored for its signature).
RECIPE_PARAMS = {
    "EXTRUDE": ("depth", "bevel_depth", "bevel_resolution", "resolution", "orientation", "origin", "units_scale", "fill_caps", "use_source_material"),
    "REVOLVE": ("resolution", "orientation", "origin", "units_scale", "revolve_angle", "revolve_segments", "revolve_axis", "axis_offset", "fill_caps", "use_source_material"),
    "SWEEP": ("resolution", "orientation", "origin", "units_scale", "sweep_radius", "sweep_profile", "sweep_profile_resolution", "fill_caps", "use_source_material"),
    "TEXT3D": ("depth", "bevel_depth", "bevel_resolution", "resolution", "orientation", "origin", "units_scale", "use_source_material"),
    "DECAL": ("decal_mode", "decal_resolution", "units_scale", "orientation"),
}


def default_params(recipe: str) -> Dict[str, Any]:
    raise NotImplementedError


def validate_params(recipe: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return clamped/validated params for ``recipe``; raise ValueError for unknown recipe or bad enum."""
    raise NotImplementedError


def params_signature(recipe: str, params: Mapping[str, Any]) -> str:
    """Stable digest of only the params the recipe uses."""
    used = {k: params.get(k, PARAMS[k][0]) for k in RECIPE_PARAMS[recipe]}
    blob = json.dumps({"recipe": recipe, "params": used}, sort_keys=True, default=float).encode()
    return hashlib.blake2b(blob, digest_size=12).hexdigest()


def can_link(recipe: str, source_role: str, target_role: str = "RESULT") -> Tuple[bool, str]:
    """(ok, reason). Results/strokes/internal objects can never be sources (cycle prevention)."""
    raise NotImplementedError


def would_create_cycle(edges: Iterable[Tuple[str, str]], source_id: str, target_id: str) -> bool:
    """True if adding source->target to the directed graph ``edges`` creates a cycle (or self-loop)."""
    raise NotImplementedError


# --- state machine -------------------------------------------------------
# Events: "source_changed", "pause", "resume", "detach", "build_ok", "build_error",
#         "target_modified", "source_missing", "source_restored", "overwrite", "branch", "relink"
def transition(state: str, event: str, stale: bool = False) -> Tuple[str, bool, bool]:
    """Return (new_state, new_stale, should_rebuild).

    Rules: LIVE+source_changed -> (LIVE, stale, rebuild). PAUSED+source_changed -> (PAUSED, True, no).
    resume -> LIVE, rebuild if stale. detach -> DETACHED (terminal: every later event keeps DETACHED
    and never rebuilds). build_error -> ERROR (keeps last good geometry; source_changed from ERROR
    rebuilds). target_modified -> CONFLICT (no rebuild until overwrite/branch -> LIVE + rebuild).
    source_missing -> MISSING_SOURCE; source_restored/relink -> LIVE + rebuild.
    Unknown events raise ValueError.
    """
    raise NotImplementedError
