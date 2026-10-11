"""Typography, text objects and live outlines (vector-b). No seam_core kernels needed."""

import os

import bpy
import pytest

from helpers import flush, save_and_reload
from test_vector_b_common import curve_bbox, select

DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def add_text(body="Hello", x=100.0, y=-100.0, size=48.0):
    bpy.ops.seam.add_text(body=body, x=x, y=y, size=size)
    return bpy.context.active_object


def outline_of(text):
    for o in bpy.context.scene.objects:
        if o.seam.role == "SHAPE" and o.seam.owner_id == text.seam.id:
            return o
    return None


def test_add_text_object(seam):
    from seam_studio.vector import canvas
    ob = add_text("Hello", 120.0, -80.0, 64.0)
    assert ob.type == "FONT" and ob.seam.role == "TEXT"
    assert ob.data.body == "Hello" and ob.data.size == pytest.approx(64.0)
    assert tuple(ob.location)[:2] == pytest.approx((120.0, -80.0))
    assert ob.parent is canvas.canvas_root(bpy.context.scene)
    assert list(canvas.artwork_objects(bpy.context.scene)) == [ob]
    assert tuple(ob.seam.style.fill_color) == (0.0, 0.0, 0.0, 1.0)
    # empty text is refused with an explanation
    with pytest.raises(RuntimeError, match="empty"):
        bpy.ops.seam.add_text(body="", x=0, y=0, size=10)


def test_typography_properties_persist(seam, blend_path):
    ob = add_text("Typography test")
    cu = ob.data
    cu.size = 80.0
    cu.space_character = 1.25
    cu.space_word = 2.0
    cu.space_line = 1.4
    cu.align_x = "CENTER"
    cu.align_y = "TOP"
    cu.text_boxes[0].width = 3.0
    cu.text_boxes[0].height = 2.0
    name = ob.name
    save_and_reload(blend_path)
    cu = bpy.data.objects[name].data
    assert cu.size == pytest.approx(80.0)
    assert cu.space_character == pytest.approx(1.25)
    assert cu.space_word == pytest.approx(2.0)
    assert cu.space_line == pytest.approx(1.4)
    assert (cu.align_x, cu.align_y) == ("CENTER", "TOP")
    assert cu.text_boxes[0].width == pytest.approx(3.0) and cu.text_boxes[0].height == pytest.approx(2.0)


def test_style_buttons_whole_body(seam):
    ob = add_text("Styled text")
    n = len(ob.data.body)
    assert len(ob.data.body_format) == n
    for style, attr in (("BOLD", "use_bold"), ("ITALIC", "use_italic"), ("UNDERLINE", "use_underline"),
                        ("SMALL_CAPS", "use_small_caps")):
        bpy.ops.seam.text_style(style=style)
        assert all(getattr(f, attr) for f in ob.data.body_format), style
        bpy.ops.seam.text_style(style=style)           # toggles back
        assert not any(getattr(f, attr) for f in ob.data.body_format), style
    bpy.ops.seam.text_style(style="BOLD", state="ON")
    bpy.ops.seam.text_style(style="BOLD", state="ON")
    assert all(f.use_bold for f in ob.data.body_format)
    bpy.ops.seam.text_style(style="BOLD", state="OFF")
    assert not any(f.use_bold for f in ob.data.body_format)


def test_underline_changes_outline_geometry(seam):
    ob = add_text("Under")
    plain = len(_glyph_paths(ob))
    bpy.ops.seam.text_style(style="UNDERLINE", state="ON")
    assert len(_glyph_paths(ob)) > plain          # the underline bar is extra geometry


def _glyph_paths(ob):
    from seam_studio import bridge
    return bridge.text_paths(ob)


@pytest.mark.skipif(bpy.app.version < (4, 1, 0) and bpy.app.background,
                    reason="Blender 4.0 background crashes in font edit-mode selection operators")
def test_style_buttons_edit_selection(seam):
    ob = add_text("Hello World")
    select(ob)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        with bpy.context.temp_override(edit_object=ob, active_object=ob, object=ob):
            bpy.ops.font.select_all()
            bpy.ops.seam.text_style(style="BOLD", state="ON")
            assert ob.data.is_select_bold
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    assert all(f.use_bold for f in ob.data.body_format)


def test_font_load_and_assign(seam):
    if not os.path.exists(DEJAVU):
        pytest.skip("DejaVuSans.ttf not installed")
    ob = add_text("Hamburg")
    builtin_width = _width(ob)
    bpy.ops.seam.font_load(filepath=DEJAVU)
    assert ob.data.font is not None and ob.data.font.filepath.endswith("DejaVuSans.ttf")
    assert _width(ob) != pytest.approx(builtin_width)
    assert ob.data.font.name in [f.name for f in bpy.data.fonts]
    # assign a different loaded font through the font list operator
    other = bpy.data.fonts.get("Bfont Regular") or bpy.data.fonts[0]
    bpy.ops.seam.font_set(font=other.name)
    assert ob.data.font is other
    with pytest.raises(RuntimeError, match="not loaded"):
        bpy.ops.seam.font_set(font="No Such Font")
    # a non-font file is rejected with a message, nothing is assigned
    with pytest.raises(RuntimeError, match="Could not load font"):
        bpy.ops.seam.font_load(filepath=os.path.abspath(__file__))


def _width(ob):
    from seam_studio import bridge
    xs = [p.co[0] for path in bridge.text_paths(ob) for p in path.points]
    return max(xs) - min(xs)


def test_text_to_outline_is_non_destructive_and_live(seam, engine):
    ob = add_text("Hi", size=100.0)
    select(ob)
    bpy.ops.seam.text_to_outline()
    out = outline_of(ob)
    assert out is not None and out.type == "CURVE" and out.seam.role == "SHAPE"
    assert out.seam.owner_id == ob.seam.id
    assert ob.name in bpy.data.objects                      # text kept
    assert not ob.visible_get()                             # ... but hidden
    assert tuple(out.location) == pytest.approx(tuple(ob.location))
    n0 = len(out.data.splines)
    assert n0 == len(_glyph_paths(ob)) and n0 >= 2          # H, i (+ dot)
    flush()
    w0 = curve_bbox(out)[2] - curve_bbox(out)[0]
    # change the text -> outline follows through the engine
    ob.data.body = "Hi there"
    res = flush()
    assert any(r.key.startswith("outline:") and r.status == "ok" for r in res), res
    w1 = curve_bbox(out)[2] - curve_bbox(out)[0]
    assert w1 > 2 * w0
    assert len(out.data.splines) == len(_glyph_paths(ob))
    # size change scales it
    ob.data.size = 200.0
    flush()
    w2 = curve_bbox(out)[2] - curve_bbox(out)[0]
    assert w2 == pytest.approx(2 * w1, rel=0.02)
    # spacing change
    ob.data.space_character = 2.0
    flush()
    assert curve_bbox(out)[2] - curve_bbox(out)[0] > w2
    # an unchanged text does not rebuild
    assert [r.status for r in flush() if r.key.startswith("outline:")] in ([], ["skipped"])
    # outline can be moved independently from the (hidden) text
    out.location.x += 500.0
    flush()
    assert out.location.x == pytest.approx(ob.location.x + 500.0)


def test_outline_survives_save_and_undo_like_revalidation(seam, engine, blend_path):
    ob = add_text("Persist", size=60.0)
    select(ob)
    bpy.ops.seam.text_to_outline()
    out = outline_of(ob)
    flush()
    oname, tname = out.name, ob.name
    save_and_reload(blend_path)
    from seam_studio.engine import core
    core.request_revalidate()
    res = core.flush()
    assert [r.status for r in res if r.key.startswith("outline:")] == ["skipped"]
    out, ob = bpy.data.objects[oname], bpy.data.objects[tname]
    assert out.seam.owner_id == ob.seam.id
    ob.data.body = "Persisted!"
    flush()
    assert len(out.data.splines) == len(_glyph_paths(ob))


def test_expand_and_revert(seam, engine):
    from seam_studio.vector import canvas
    a = add_text("Ex", size=50.0)
    select(a)
    bpy.ops.seam.text_to_outline()
    out_a = outline_of(a)
    select(out_a)
    bpy.ops.seam.outline_expand()
    assert out_a.seam.owner_id == "" and out_a.seam.role == "SHAPE"
    assert "Text" not in [o.name for o in bpy.data.objects if o.type == "FONT"]   # hidden source removed
    n = len(out_a.data.splines)
    flush()
    assert len(out_a.data.splines) == n                        # stopped following

    b = add_text("Keep", size=50.0)
    select(b)
    bpy.ops.seam.text_to_outline()
    out_b = outline_of(b)
    select(out_b)
    bpy.ops.seam.outline_expand(keep_text=True)
    assert b.name in bpy.data.objects and out_b.seam.owner_id == ""

    c = add_text("Rev", size=50.0)
    select(c)
    bpy.ops.seam.text_to_outline()
    out_c = outline_of(c)
    cname, oname = c.name, out_c.name
    select(out_c)
    bpy.ops.seam.outline_revert()
    assert oname not in bpy.data.objects
    assert bpy.data.objects[cname].visible_get()
    assert bpy.data.objects[cname] in canvas.artwork_objects(bpy.context.scene)


def test_text_to_outline_errors(seam):
    ob = add_text(" ", size=50.0)         # only a blank: no glyph outlines
    select(ob)
    with pytest.raises(RuntimeError, match="no glyph outlines"):
        bpy.ops.seam.text_to_outline()
    assert outline_of(ob) is None and ob.visible_get()
    # twice on the same text is refused
    ob.data.body = "Twice"
    select(ob)
    bpy.ops.seam.text_to_outline()
    ob.hide_set(False)
    select(ob)
    with pytest.raises(RuntimeError, match="already has an outline"):
        bpy.ops.seam.text_to_outline()
