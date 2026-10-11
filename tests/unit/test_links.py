"""Pure link model: params, can_link, cycle checks and every state-machine rule."""

import pytest

from seam_core import links as L


# ------------------------------------------------------------------ params

@pytest.mark.parametrize("recipe", L.RECIPES)
def test_default_params_cover_recipe_params(recipe):
    d = L.default_params(recipe)
    assert set(d) == set(L.RECIPE_PARAMS[recipe])
    for k, v in d.items():
        assert v == L.PARAMS[k][0]
    d["depth"] = 99  # fresh dict every call
    assert L.default_params(recipe).get("depth") != 99


def test_default_params_unknown_recipe():
    with pytest.raises(ValueError):
        L.default_params("LOFT")


def test_validate_params_clamps_and_coerces():
    p = L.validate_params("EXTRUDE", {"depth": -1.0, "resolution": 1000, "bevel_resolution": 2.6,
                                      "units_scale": 0.0, "fill_caps": 0})
    assert p["depth"] == 0.0
    assert p["resolution"] == 64 and isinstance(p["resolution"], int)
    assert p["bevel_resolution"] == 3
    assert p["units_scale"] == pytest.approx(1e-6)
    assert p["fill_caps"] is False
    assert p["orientation"] == "UPRIGHT"  # default filled in
    assert "revolve_angle" not in p  # only recipe params


def test_validate_params_errors():
    with pytest.raises(ValueError):
        L.validate_params("NOPE", {})
    with pytest.raises(ValueError):
        L.validate_params("EXTRUDE", {"orientation": "SIDEWAYS"})
    with pytest.raises(ValueError):
        L.validate_params("EXTRUDE", {"depth": "deep"})
    with pytest.raises(ValueError):
        L.validate_params("EXTRUDE", {"depth": float("nan")})
    with pytest.raises(ValueError):
        L.validate_params("EXTRUDE", {"fill_caps": 0.5})


def test_validate_params_ignores_unused_and_accepts_none():
    p = L.validate_params("SWEEP", {"depth": 5.0, "sweep_profile": "SQUARE"})
    assert "depth" not in p and p["sweep_profile"] == "SQUARE"
    assert L.validate_params("REVOLVE", None) == L.default_params("REVOLVE")


def test_params_signature_only_uses_recipe_params():
    a = L.params_signature("SWEEP", {"depth": 1.0})
    b = L.params_signature("SWEEP", {"depth": 2.0})
    assert a == b
    assert L.params_signature("EXTRUDE", {"depth": 1.0}) != L.params_signature("EXTRUDE", {"depth": 2.0})
    assert L.params_signature("EXTRUDE", {}) != L.params_signature("TEXT3D", {})


# ------------------------------------------------------------------ can_link / cycles

def test_can_link_roles():
    assert L.can_link("EXTRUDE", "SHAPE") == (True, "")
    assert L.can_link("TEXT3D", "TEXT")[0]
    ok, why = L.can_link("TEXT3D", "SHAPE")
    assert not ok and "text" in why.lower()
    ok, why = L.can_link("EXTRUDE", "TEXT")
    assert not ok and "shape" in why.lower()
    assert L.can_link("DECAL", "RASTER")[0]


@pytest.mark.parametrize("role", L.DERIVED_ROLES)
@pytest.mark.parametrize("recipe", L.RECIPES)
def test_derived_objects_never_sources(recipe, role):
    ok, why = L.can_link(recipe, role)
    assert not ok and why


def test_can_link_unknown_recipe_and_bad_target():
    assert not L.can_link("LOFT", "SHAPE")[0]
    assert not L.can_link("EXTRUDE", "SHAPE", target_role="SHAPE")[0]


def test_would_create_cycle():
    edges = [("a", "b"), ("b", "c")]
    assert L.would_create_cycle(edges, "x", "x")
    assert L.would_create_cycle(edges, "c", "a")
    assert L.would_create_cycle(edges, "b", "a")
    assert not L.would_create_cycle(edges, "a", "c")
    assert not L.would_create_cycle(edges, "a", "d")
    assert not L.would_create_cycle([], "a", "b")
    # diamond + unrelated loop does not confuse traversal
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("x", "y"), ("y", "x")]
    assert L.would_create_cycle(edges, "d", "a")
    assert not L.would_create_cycle(edges, "d", "e")


# ------------------------------------------------------------------ state machine

T = L.transition


def test_live_source_changed_rebuilds():
    assert T("LIVE", "source_changed") == ("LIVE", False, True)
    assert T("LIVE", "source_changed", True) == ("LIVE", True, True)


def test_paused_source_changed_queues():
    assert T("PAUSED", "source_changed") == ("PAUSED", True, False)


def test_pause_and_resume():
    assert T("LIVE", "pause") == ("PAUSED", False, False)
    assert T("ERROR", "pause", True) == ("PAUSED", True, False)
    assert T("PAUSED", "pause", True) == ("PAUSED", True, False)
    assert T("PAUSED", "resume", True) == ("LIVE", True, True)
    assert T("PAUSED", "resume", False) == ("LIVE", False, False)
    assert T("LIVE", "resume") == ("LIVE", False, False)
    assert T("CONFLICT", "pause", True) == ("CONFLICT", True, False)


@pytest.mark.parametrize("event", L.EVENTS)
def test_detached_is_terminal(event):
    new, _stale, rebuild = T("DETACHED", event, True)
    assert new == "DETACHED" and rebuild is False


@pytest.mark.parametrize("state", [s for s in L.STATES if s != "DETACHED"])
def test_detach_from_any_state(state):
    assert T(state, "detach", True) == ("DETACHED", False, False)


def test_build_results():
    assert T("LIVE", "build_ok", True) == ("LIVE", False, False)
    assert T("ERROR", "build_ok", True) == ("LIVE", False, False)
    assert T("PAUSED", "build_ok", True) == ("PAUSED", False, False)
    assert T("LIVE", "build_error") == ("ERROR", False, False)
    assert T("ERROR", "source_changed") == ("ERROR", False, True)


def test_conflict_rules():
    assert T("LIVE", "target_modified") == ("CONFLICT", True, False)
    assert T("CONFLICT", "source_changed") == ("CONFLICT", True, False)
    assert T("CONFLICT", "resume", True) == ("CONFLICT", True, False)
    assert T("CONFLICT", "overwrite", True) == ("LIVE", True, True)
    assert T("CONFLICT", "branch", True) == ("LIVE", True, True)
    assert T("MISSING_SOURCE", "target_modified") == ("MISSING_SOURCE", False, False)


def test_missing_source_rules():
    for s in ("LIVE", "PAUSED", "ERROR", "CONFLICT"):
        assert T(s, "source_missing")[0] == "MISSING_SOURCE"
        assert T(s, "source_missing")[2] is False
    assert T("MISSING_SOURCE", "source_changed") == ("MISSING_SOURCE", True, False)
    assert T("MISSING_SOURCE", "source_restored") == ("LIVE", True, True)
    assert T("MISSING_SOURCE", "relink") == ("LIVE", True, True)
    assert T("ERROR", "relink") == ("LIVE", True, True)


def test_unknown_event_and_state():
    with pytest.raises(ValueError):
        T("LIVE", "explode")
    with pytest.raises(ValueError):
        T("ZOMBIE", "pause")


@pytest.mark.parametrize("state", L.STATES)
@pytest.mark.parametrize("event", L.EVENTS)
def test_transition_total(state, event):
    new, stale, rebuild = T(state, event, False)
    assert new in L.STATES and isinstance(stale, bool) and isinstance(rebuild, bool)
    if rebuild:
        assert new in ("LIVE", "ERROR")
