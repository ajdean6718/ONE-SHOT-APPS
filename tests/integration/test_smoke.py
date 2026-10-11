"""The add-on enables through addon_utils, exposes its schema, and survives save/reload."""

import bpy

from helpers import save_and_reload


def test_addon_registers_schema(seam):
    assert hasattr(bpy.types.Scene, "seam")
    assert hasattr(bpy.types.Object, "seam")
    scene = bpy.context.scene
    assert scene.seam.schema_version >= 1
    assert scene.seam.engine.live_enabled is True


def test_object_props_roundtrip(seam, blend_path):
    from seam_studio import ids
    cu = bpy.data.curves.new("c", "CURVE")
    ob = bpy.data.objects.new("c", cu)
    bpy.context.scene.collection.objects.link(ob)
    sid = ids.ensure_id(ob)
    ob.seam.role = "SHAPE"
    ob.seam.style.fill_color = (0.1, 0.2, 0.3, 1.0)
    scene = save_and_reload(blend_path)
    ob2 = bpy.data.objects["c"]
    assert ob2.seam.id == sid and ob2.seam.role == "SHAPE"
    assert abs(ob2.seam.style.fill_color[2] - 0.3) < 1e-6
    assert scene.seam.schema_version >= 1
