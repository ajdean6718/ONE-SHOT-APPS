"""Two results from one source with independent parameters."""

import bpy
import pytest

from helpers import flush
from test_link_support import activate, extent, make_link, make_shape, move_point


def test_two_results_independent_params(engine):
    src = make_shape("A")
    thin = make_link(src, "EXTRUDE", orientation="FLAT", depth=0.1)
    thick = make_link(src, "EXTRUDE", orientation="UPRIGHT", depth=1.0)
    assert thin.target != thick.target and thin.id != thick.id
    assert extent(thin.target)[2] == pytest.approx(0.1, abs=1e-5)
    assert extent(thick.target)[1] == pytest.approx(1.0, abs=1e-5)
    move_point(src, 0, 1, dx=100.0)
    flush()
    assert extent(thin.target)[0] == pytest.approx(3.0, abs=1e-5)
    assert extent(thick.target)[0] == pytest.approx(3.0, abs=1e-5)
    assert extent(thin.target)[2] == pytest.approx(0.1, abs=1e-5)
    assert extent(thick.target)[1] == pytest.approx(1.0, abs=1e-5)
    n_thin, n_thick = thin.build_count, thick.build_count
    thick.params.depth = 0.4
    flush()
    assert thick.build_count == n_thick + 1 and thin.build_count == n_thin
    assert extent(thick.target)[1] == pytest.approx(0.4, abs=1e-5)
    assert extent(thin.target)[2] == pytest.approx(0.1, abs=1e-5)


def test_duplicate_result_copies_recipe_and_params(engine):
    src = make_shape("A")
    first = make_link(src, "EXTRUDE", orientation="FLAT", depth=0.35, origin="MIN")
    activate(first.target)
    assert bpy.ops.seam.duplicate_result() == {"FINISHED"}
    links = bpy.context.scene.seam.links
    assert len(links) == 2
    second = links[1]
    assert second.source == src and second.target != first.target
    assert second.params.depth == pytest.approx(0.35) and second.params.origin == "MIN"
    assert second.state == "LIVE" and second.build_count == 1
    assert tuple(second.target.location) == pytest.approx((2.5, 0, 0))
    second.params.depth = 0.9
    flush()
    assert first.params.depth == pytest.approx(0.35)
    assert extent(first.target)[2] == pytest.approx(0.35, abs=1e-5)


def test_redo_panel_params_via_operator(engine):
    src = make_shape("A")
    link = make_link(src, "EXTRUDE", depth=0.05, units_scale=0.001, orientation="FLAT", origin="CENTER")
    assert link.params.units_scale == pytest.approx(0.001)
    ex = extent(link.target)
    assert ex[0] == pytest.approx(0.2, abs=1e-6) and ex[2] == pytest.approx(0.05, abs=1e-6)
