"""Save / close / reopen: links keep working without re-linking."""

import bpy
import pytest

from helpers import flush, save_and_reload
from test_link_support import extent, link_by_id, make_link, make_shape, move_point


def test_save_reopen_then_edit(engine, blend_path):
    src = make_shape("Logo")
    link = make_link(src, "EXTRUDE", orientation="FLAT", depth=0.3)
    link_id, tgt_name, src_name = link.id, link.target.name, src.name
    sig = link.target_signature
    save_and_reload(blend_path)
    results = flush()  # load_post revalidates: nothing changed => no rebuild
    assert all(r.status != "ok" for r in results)
    link = link_by_id(link_id)
    assert "SEAM tmp link" not in bpy.data.objects and "SEAM tmp link" not in bpy.data.curves  # purged on save
    assert link is not None and link.state == "LIVE" and link.build_count == 1
    assert link.target == bpy.data.objects[tgt_name] and link.source == bpy.data.objects[src_name]
    assert link.target_signature == sig
    move_point(link.source, 0, 0, dx=-100.0)
    flush()
    link = link_by_id(link_id)
    assert link.build_count == 2 and link.state == "LIVE", link.message
    assert extent(link.target)[0] == pytest.approx(3.0, abs=1e-5)
    assert extent(link.target)[2] == pytest.approx(0.3, abs=1e-5)
    # and again after a second round trip
    save_and_reload(blend_path)
    flush()
    link = link_by_id(link_id)
    move_point(link.source, 0, 2, dy=100.0)
    flush()
    link = link_by_id(link_id)
    assert link.build_count == 3 and extent(link.target)[1] == pytest.approx(3.0, abs=1e-5)
