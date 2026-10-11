import copy
import json

import pytest

from seam_core import SCHEMA_VERSION
from seam_core import document as D


def good_doc():
    return D.DocumentSnapshot(
        units="MM", dpi=300.0,
        artboards=[{"id": "ab1", "name": "A4", "width": 794, "height": 1123, "export_scale": 1.0}],
        items=[
            D.ItemRecord("s1", "SHAPE", "Rect", layer="Layer 1", style={"fill_type": "SOLID", "opacity": 1.0}, shape={"kind": "RECT"}),
            D.ItemRecord("r1", "RESULT", "Extrusion"),
        ],
        rasters=[D.RasterRecord("ra1", "Paint", 512, 256, layers=[
            {"id": "l1", "name": "Bg", "opacity": 1.0, "blend_mode": "NORMAL", "visible": True},
            {"id": "l2", "name": "Top", "opacity": 0.5, "blend_mode": "MULTIPLY", "visible": True},
        ])],
        links=[D.LinkRecord("k1", "EXTRUDE", "s1", "r1", "LIVE", {"depth": 0.2})],
    )


def test_to_dict_from_dict_round_trip():
    doc = good_doc()
    d = D.to_dict(doc)
    assert d["schema_version"] == SCHEMA_VERSION
    assert json.loads(json.dumps(d)) == d  # JSON-able
    back = D.from_dict(d)
    assert back == doc
    assert D.to_dict(back) == d
    # to_dict returns an independent copy
    d["items"][0]["name"] = "changed"
    assert doc.items[0].name == "Rect"


def test_from_dict_does_not_alias_input():
    d = D.to_dict(good_doc())
    doc = D.from_dict(d)
    d["rasters"][0]["layers"][0]["name"] = "mutated"
    d["links"][0]["params"]["depth"] = 99
    assert doc.rasters[0].layers[0]["name"] == "Bg" and doc.links[0].params["depth"] == 0.2


def test_valid_document_has_no_problems():
    assert D.validate(good_doc()) == []
    assert D.validate(D.DocumentSnapshot()) == []


def test_migrate_0_to_1_renames_kind():
    old = {"schema_version": 0, "units": "PX", "dpi": 96.0,
           "items": [{"id": "a", "role": "SHAPE", "name": "A"}, {"id": "b", "role": "RESULT", "name": "B"}],
           "links": [{"id": "k", "kind": "REVOLVE", "source_id": "a", "target_id": "b", "state": "LIVE"}]}
    snapshot = copy.deepcopy(old)
    m = D.migrate(old)
    assert old == snapshot  # input untouched
    assert m["schema_version"] == 1
    assert m["links"][0]["recipe"] == "REVOLVE" and "kind" not in m["links"][0]
    doc = D.from_dict(old)
    assert doc.schema_version == 1 and doc.links[0].recipe == "REVOLVE"
    assert D.validate(doc) == []
    # missing schema_version means version 0
    no_ver = {k: v for k, v in old.items() if k != "schema_version"}
    assert D.migrate(no_ver)["links"][0]["recipe"] == "REVOLVE"


def test_migrate_current_version_is_identity_copy():
    d = D.to_dict(good_doc())
    m = D.migrate(d)
    assert m == d and m is not d


def test_newer_schema_raises():
    d = D.to_dict(good_doc())
    d["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer"):
        D.from_dict(d)
    with pytest.raises(ValueError):
        D.migrate(d)


def test_from_dict_rejects_malformed():
    with pytest.raises(ValueError):
        D.from_dict([])
    with pytest.raises(ValueError, match="missing required"):
        D.from_dict({"schema_version": 1, "items": [{"id": "a"}]})
    with pytest.raises(ValueError):
        D.from_dict({"schema_version": 1, "items": "nope"})
    with pytest.raises(ValueError):
        D.from_dict({"schema_version": "x"})
    # unknown extra keys are ignored (forward compatible within the same version)
    doc = D.from_dict({"schema_version": 1, "items": [{"id": "a", "role": "SHAPE", "name": "A", "future": 1}]})
    assert doc.items[0].id == "a"


def test_validate_duplicate_ids():
    doc = good_doc()
    doc.items.append(D.ItemRecord("s1", "SHAPE", "Dup"))
    probs = D.validate(doc)
    assert any("duplicate id 's1'" in p for p in probs)
    doc = good_doc()
    doc.links.append(D.LinkRecord("s1", "EXTRUDE", "s1", "r1", "LIVE"))  # link id collides with item id
    assert any("duplicate id 's1'" in p for p in D.validate(doc))
    doc = good_doc()
    doc.rasters[0].layers[1]["id"] = "l1"
    assert any("duplicate layer id" in p for p in D.validate(doc))


def test_validate_dangling_links():
    doc = good_doc()
    doc.links[0].source_id = "ghost"
    doc.links.append(D.LinkRecord("k2", "SWEEP", "s1", "nowhere", "LIVE"))
    probs = D.validate(doc)
    assert any("missing source id 'ghost'" in p for p in probs)
    assert any("missing target id 'nowhere'" in p for p in probs)
    # a link already known to be MISSING_SOURCE is not reported for its source
    doc = good_doc()
    doc.links[0].source_id = "ghost"
    doc.links[0].state = "MISSING_SOURCE"
    assert D.validate(doc) == []


def test_validate_bad_recipe_state_blend():
    doc = good_doc()
    doc.links[0].recipe = "LOFT"
    doc.links[0].state = "BROKEN"
    doc.rasters[0].layers[1]["blend_mode"] = "PLUS_DARKER"
    probs = D.validate(doc)
    assert any("unknown recipe 'LOFT'" in p for p in probs)
    assert any("unknown state 'BROKEN'" in p for p in probs)
    assert any("invalid blend mode 'PLUS_DARKER'" in p for p in probs)


def test_validate_every_known_recipe_state_blend_ok():
    from seam_core.raster import BLEND_MODES
    doc = good_doc()
    for i, r in enumerate(D.LINK_RECIPES):
        for j, st in enumerate(D.LINK_STATES):
            doc.links.append(D.LinkRecord("x%d_%d" % (i, j), r, "s1", "r1", st))
    doc.rasters[0].layers = [{"id": "m%d" % i, "blend_mode": m, "opacity": 1.0} for i, m in enumerate(BLEND_MODES)]
    assert D.validate(doc) == []


def test_validate_non_positive_sizes():
    doc = good_doc()
    doc.rasters[0].width = 0
    doc.artboards[0]["height"] = -5
    doc.dpi = 0
    probs = D.validate(doc)
    assert any("non-positive size" in p for p in probs)
    assert any("non-positive height" in p for p in probs)
    assert any("dpi" in p for p in probs)


def test_validate_misc_problems():
    doc = good_doc()
    doc.units = "FURLONG"
    doc.items[0].role = "WIDGET"
    doc.rasters[0].layers[0]["opacity"] = 1.5
    doc.schema_version = SCHEMA_VERSION + 3
    probs = D.validate(doc)
    assert len(probs) == 4
