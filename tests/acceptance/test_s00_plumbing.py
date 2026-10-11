"""Acceptance plumbing: fixtures from the integration conftest work here."""

import bpy


def test_fixture_plumbing(seam, engine):
    assert hasattr(bpy.types.Scene, "seam")
    assert engine.flush() == [] or isinstance(engine.flush(), list)
