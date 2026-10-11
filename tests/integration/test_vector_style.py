"""Appearance: materials, gradients, strokes (derived data follows style edits after engine.flush())."""

import math

import bpy
import numpy as np
import pytest

from helpers import flush, save_and_reload


def _need_core(stroke=False):
    from seam_studio.seam_core import geometry, shapes, stroke as st
    try:
        shapes.rect(1, 1)
        geometry.flatten_path(shapes.rect(1, 1))
        if stroke:
            st.stroke_mesh([(0, 0), (1, 0)], False, 1.0)
    except NotImplementedError as exc:
        pytest.skip("core pending: " + str(exc))


@pytest.fixture
def doc(seam):
    _need_core()
    bpy.ops.seam.canvas_new()
    return bpy.context.scene


def _rect(w=100, h=60, x=0, y=0):
    assert bpy.ops.seam.add_shape(kind="RECT", width=w, height=h, x=x, y=y) == {"FINISHED"}
    ob = bpy.context.active_object
    flush()
    return ob


def _mat(ob):
    return ob.data.materials[0]


def _mix_alpha(mat):
    return mat.node_tree.nodes["SEAM_mix"].inputs[0].default_value


def _emission(mat):
    return tuple(mat.node_tree.nodes["SEAM_emission"].inputs["Color"].default_value)


def _stroke_child(ob):
    kids = [c for c in ob.children if c.seam.role == "STROKE"]
    return kids[0] if kids else None


def _mesh_area(child):
    me = child.data
    mw = child.matrix_world
    tot = 0.0
    for p in me.polygons:
        vs = [mw @ me.vertices[i].co for i in p.vertices]
        a = 0.0
        for i in range(len(vs)):
            a += vs[i].x * vs[(i + 1) % len(vs)].y - vs[(i + 1) % len(vs)].x * vs[i].y
        tot += abs(a) * 0.5
    return tot


# ---------------------------------------------------------------- material

def test_material_is_created_and_follows_style_edits(doc):
    ob = _rect()
    mat = _mat(ob)
    assert mat.name == "SEAM " + ob.name and mat["seam_owner"] == ob.seam.id
    assert _emission(mat)[:3] == pytest.approx(tuple(ob.seam.style.fill_color)[:3])
    assert _mix_alpha(mat) == pytest.approx(1.0)
    st = ob.seam.style
    st.fill_color = (1.0, 0.25, 0.0, 0.5)
    st.opacity = 0.5
    flush()
    assert _emission(mat)[:3] == pytest.approx((1.0, 0.25, 0.0))
    assert _mix_alpha(mat) == pytest.approx(0.25)
    assert tuple(mat.diffuse_color) == pytest.approx((1.0, 0.25, 0.0, 0.25))
    # blending is set so alpha works in EEVEE
    assert getattr(mat, "blend_method", "BLEND") in ("BLEND", "HASHED")
    st.fill_enabled = False
    flush()
    assert _mix_alpha(mat) == pytest.approx(0.0)
    assert mat.diffuse_color[3] == pytest.approx(0.0)  # invisible in Solid view too (no wire triangulation)
    assert ob.display_type == "TEXTURED"
    st.fill_enabled = True
    flush()
    assert ob.display_type == "TEXTURED" and _mix_alpha(mat) == pytest.approx(0.25)
    # renaming keeps the material in sync
    ob.name = "Renamed"
    flush()
    assert mat.name == "SEAM Renamed"


def test_material_slot_healed_when_replaced_or_removed(doc):
    ob = _rect()
    ob.data.materials[0] = bpy.data.materials.new("User material")
    flush()
    assert _mat(ob)["seam_owner"] == ob.seam.id
    ob.data.materials.clear()
    flush()
    assert len(ob.data.materials) == 1 and _mat(ob)["seam_owner"] == ob.seam.id


def test_material_is_idempotent(doc):
    ob = _rect()
    flush()
    res = flush()
    assert all(r.status in ("skipped", "obsolete") for r in res)
    n = len(bpy.data.materials)
    flush()
    ob.seam.style.fill_color = (0, 1, 0, 1)
    flush()
    assert len(bpy.data.materials) == n


# ---------------------------------------------------------------- gradients

def _gradient_image(ob):
    img = bpy.data.images.get("SEAM Gradient " + ob.seam.id)
    assert img is not None
    arr = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(arr)
    return img, arr.reshape(img.size[1], img.size[0], 4)


def _srgb_decode(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _set_stops(st, stops):
    st.gradient_stops.clear()
    for pos, col in stops:
        s = st.gradient_stops.add()
        s.position = pos
        s.color = col


def test_linear_gradient_end_pixels_equal_stop_colours(doc):
    ob = _rect(200, 100)
    st = ob.seam.style
    red, blue = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    _set_stops(st, [(0.0, red), (1.0, blue)])
    st.fill_type = "LINEAR"
    st.gradient_angle = 0.0
    flush()
    img, px = _gradient_image(ob)
    assert img.size[0] == img.size[1] >= 64
    assert img.colorspace_settings.name == "sRGB" and img.alpha_mode == "STRAIGHT"
    lin_left = _srgb_decode(px[px.shape[0] // 2, 0, :3])
    lin_right = _srgb_decode(px[px.shape[0] // 2, -1, :3])
    assert lin_left == pytest.approx(red[:3], abs=2e-3)
    assert lin_right == pytest.approx(blue[:3], abs=2e-3)
    assert px[0, 0, 3] == pytest.approx(1.0) and px[0, -1, 3] == pytest.approx(1.0)
    # the middle is the linear-space mix (0.5, 0, 0.5)
    mid = _srgb_decode(px[px.shape[0] // 2, px.shape[1] // 2, :3])
    assert mid == pytest.approx((0.5, 0.0, 0.5), abs=0.02)
    # vertical variation: none for angle 0
    assert np.allclose(px[0], px[-1])
    # material wiring: image is the active node and feeds the emission
    mat = _mat(ob)
    mid = _srgb_decode(px[px.shape[0] // 2, px.shape[1] // 2, :3])
    assert tuple(mat.diffuse_color)[:3] == pytest.approx((0.5, 0.0, 0.5), abs=1e-4)  # Solid view: midpoint colour
    tex = mat.node_tree.nodes["SEAM_gradient"]
    assert tex.image == img and mat.node_tree.nodes.active == tex
    link = [l for l in mat.node_tree.links if l.to_node.name == "SEAM_emission" and l.to_socket.name == "Color"]
    assert link and link[0].from_node == tex
    assert tex.extension == "EXTEND"


def test_gradient_angle_and_stop_alpha(doc):
    ob = _rect()
    st = ob.seam.style
    _set_stops(st, [(0.0, (1, 1, 1, 0.0)), (1.0, (1, 1, 1, 1.0))])
    st.fill_type = "LINEAR"
    st.gradient_angle = math.pi / 2  # bottom -> top
    flush()
    _img, px = _gradient_image(ob)
    assert px[0, 5, 3] == pytest.approx(0.0, abs=1 / 255) and px[-1, 5, 3] == pytest.approx(1.0, abs=1 / 255)
    assert np.allclose(px[:, 0], px[:, -1])
    assert _mat(ob).node_tree.nodes["SEAM_alpha"].operation == "MULTIPLY"
    st.opacity = 0.4
    flush()
    assert _mat(ob).node_tree.nodes["SEAM_alpha"].inputs[1].default_value == pytest.approx(0.4)


def test_radial_gradient_centre_and_edge(doc):
    ob = _rect(100, 100)
    st = ob.seam.style
    _set_stops(st, [(0.0, (0, 1, 0, 1)), (1.0, (0, 0, 0, 1))])
    st.fill_type = "RADIAL"
    st.gradient_center = (0.5, 0.5)
    st.gradient_radius = 0.5
    flush()
    img, px = _gradient_image(ob)
    n = img.size[0]
    c = _srgb_decode(px[n // 2, n // 2, :3])
    assert c[1] > 0.95 and c[0] < 0.01
    edge = _srgb_decode(px[n // 2, 0, :3])
    assert edge[1] < 0.05
    # off-centre radial: the brightest pixel moves
    st.gradient_center = (0.1, 0.1)
    flush()
    _img, px = _gradient_image(ob)
    g = px[..., 1]
    iy, ix = np.unravel_index(np.argmax(g), g.shape)
    assert ix <= 0.2 * n and iy <= 0.2 * n


def test_gradient_mapping_covers_bounds_and_survives_resize(doc):
    ob = _rect(200, 100, x=500, y=-300)
    st = ob.seam.style
    st.fill_type = "LINEAR"
    flush()
    mp = _mat(ob).node_tree.nodes["SEAM_mapping"]
    assert mp.vector_type == "POINT"
    n = 128
    k = (n - 1) / n
    sx = mp.inputs["Scale"].default_value
    loc = mp.inputs["Location"].default_value
    assert sx[0] == pytest.approx(k / 200, rel=1e-4) and sx[1] == pytest.approx(k / 100, rel=1e-4)
    # object-local x = -100 (left edge) maps to the centre of texel 0, +100 to the centre of the last texel
    assert -100 * sx[0] + loc[0] == pytest.approx(0.5 / n, abs=1e-5)
    assert 100 * sx[0] + loc[0] == pytest.approx(1 - 0.5 / n, abs=1e-5)
    ob.seam.shape.width = 400
    flush()
    mp = _mat(ob).node_tree.nodes["SEAM_mapping"]
    assert mp.inputs["Scale"].default_value[0] == pytest.approx(k / 400, rel=1e-4)


def test_switching_back_to_solid_removes_gradient(doc):
    ob = _rect()
    ob.seam.style.fill_type = "RADIAL"
    flush()
    assert bpy.data.images.get("SEAM Gradient " + ob.seam.id) is not None
    ob.seam.style.fill_type = "SOLID"
    ob.seam.style.fill_color = (0.1, 0.2, 0.3, 1.0)
    flush()
    assert bpy.data.images.get("SEAM Gradient " + ob.seam.id) is None
    assert "SEAM_gradient" not in _mat(ob).node_tree.nodes
    assert _emission(_mat(ob))[:3] == pytest.approx((0.1, 0.2, 0.3))
    assert _mix_alpha(_mat(ob)) == pytest.approx(1.0)


def test_gradient_image_survives_save_and_reload(doc, blend_path):
    ob = _rect()
    st = ob.seam.style
    _set_stops(st, [(0.0, (1, 0, 0, 1)), (1.0, (0, 1, 0, 1))])
    st.fill_type = "LINEAR"
    flush()
    name = ob.name
    _img, before = _gradient_image(ob)
    save_and_reload(blend_path)
    from seam_studio.engine import core
    core.request_revalidate()
    flush()
    ob = bpy.data.objects[name]
    img, after = _gradient_image(ob)
    assert np.abs(after[0, 0, :3] - before[0, 0, :3]).max() <= 1 / 255 + 1e-6
    assert np.abs(after[0, -1, :3] - before[0, -1, :3]).max() <= 1 / 255 + 1e-6
    assert _mat(ob).node_tree.nodes["SEAM_gradient"].image == img


def test_gradient_stop_operators(doc):
    ob = _rect()
    st = ob.seam.style
    assert len(st.gradient_stops) == 2
    assert bpy.ops.seam.gradient_stop_add() == {"FINISHED"}
    assert len(st.gradient_stops) == 3 and st.gradient_stops[2].position == pytest.approx(0.5)
    assert bpy.ops.seam.gradient_stop_remove(index=2) == {"FINISHED"}
    assert not bpy.ops.seam.gradient_stop_remove.poll()  # two stops are the minimum
    st.gradient_stops.clear()
    assert bpy.ops.seam.gradient_stops_reset() == {"FINISHED"}
    assert len(st.gradient_stops) == 2


# ---------------------------------------------------------------- strokes

def test_stroke_child_created_with_style_and_removed(doc):
    _need_core(stroke=True)
    ob = _rect(100, 100)
    st = ob.seam.style
    assert _stroke_child(ob) is None
    st.stroke_enabled = True
    st.stroke_width = 4.0
    st.stroke_color = (0.0, 1.0, 0.0, 1.0)
    flush()
    ch = _stroke_child(ob)
    assert ch is not None and ch.seam.role == "STROKE" and ch.seam.owner_id == ob.seam.id
    assert ch.hide_select and ch.parent == ob
    assert ch.matrix_parent_inverse == ch.matrix_parent_inverse.Identity(4)
    from seam_studio.seam_core import Z_STEP
    assert ch.location.z == pytest.approx(Z_STEP * 0.5)
    assert set(ch.users_collection) == set(ob.users_collection)
    assert len(ch.data.polygons) > 0
    cm = ch.data.materials[0]
    assert _emission(cm)[:3] == pytest.approx((0, 1, 0))
    assert cm is not _mat(ob)
    st.stroke_color = (1.0, 0.0, 0.0, 1.0)
    st.opacity = 0.5
    flush()
    assert _emission(ch.data.materials[0])[:3] == pytest.approx((1, 0, 0))
    assert _mix_alpha(ch.data.materials[0]) == pytest.approx(0.5)
    st.stroke_enabled = False
    flush()
    assert _stroke_child(ob) is None
    assert not [o for o in bpy.data.objects if o.seam.role == "STROKE"]
    assert not [m for m in bpy.data.materials if m.name.startswith("SEAM Stroke") and m.users == 0]


def test_stroke_mesh_area_matches_analytic_band(doc):
    _need_core(stroke=True)
    ob = _rect(100, 100)
    st = ob.seam.style
    st.stroke_enabled = True
    st.stroke_width = 4.0
    st.stroke_join = "MITER"
    flush()
    ch = _stroke_child(ob)
    area = _mesh_area(ch)
    analytic = 4 * 100 * 4.0  # perimeter x width for mitred square corners
    assert area == pytest.approx(analytic, rel=0.06)
    # the mesh extends half a stroke width beyond the path
    xs = [v.co.x for v in ch.data.vertices]
    assert max(xs) == pytest.approx(52.0, abs=0.6) and min(xs) == pytest.approx(-52.0, abs=0.6)
    st.stroke_width = 10.0
    flush()
    assert _mesh_area(_stroke_child(ob)) == pytest.approx(4 * 100 * 10.0, rel=0.06)


def test_stroke_of_open_line_with_caps(doc):
    _need_core(stroke=True)
    assert bpy.ops.seam.add_shape(kind="LINE", width=100, height=0, x=0, y=0) == {"FINISHED"}
    ob = bpy.context.active_object
    ob.seam.style.stroke_width = 10.0
    flush()
    ch = _stroke_child(ob)
    assert ch is not None
    assert _mesh_area(ch) == pytest.approx(1000.0, rel=0.05)
    ob.seam.style.stroke_cap = "SQUARE"
    flush()
    assert _mesh_area(_stroke_child(ob)) == pytest.approx(1100.0, rel=0.12)


def test_stroke_width_is_uniform_under_non_uniform_scale(doc):
    _need_core(stroke=True)
    ob = _rect(100, 100)
    st = ob.seam.style
    st.stroke_enabled = True
    st.stroke_width = 4.0
    flush()
    ob.scale = (3.0, 0.5, 1.0)
    flush()
    ch = _stroke_child(ob)
    # world-space band: perimeter (2*300 + 2*50) * 4, still 4 px thick
    assert _mesh_area(ch) == pytest.approx((2 * 300 + 2 * 50) * 4.0, rel=0.08)
    mw = ch.matrix_world
    xs = [(mw @ v.co).x for v in ch.data.vertices]
    ys = [(mw @ v.co).y for v in ch.data.vertices]
    assert (max(xs) - min(xs)) == pytest.approx(300 + 4, abs=1.0)
    assert (max(ys) - min(ys)) == pytest.approx(50 + 4, abs=1.0)
    # translation does not rebuild the stroke
    ob.location.x += 10
    res = flush()
    assert not [r for r in res if r.key == "stroke:" + ob.seam.id and r.status == "ok"]


def test_stroke_follows_geometry_edits_and_transform(doc):
    _need_core(stroke=True)
    ob = _rect(100, 100)
    ob.seam.style.stroke_enabled = True
    flush()
    ch = _stroke_child(ob)
    a0 = _mesh_area(ch)
    ob.seam.shape.width = 300
    flush()
    ch = _stroke_child(ob)
    assert _mesh_area(ch) > a0 * 1.5
    ob.location = (50, 60, ob.location.z)
    flush()
    mw = ch.matrix_world
    assert (mw.translation.x, mw.translation.y - (-20000.0)) == pytest.approx((50, 60), abs=1e-2)


def test_stroke_heals_missing_duplicated_and_orphaned_children(doc):
    _need_core(stroke=True)
    ob = _rect(100, 100)
    ob.seam.style.stroke_enabled = True
    flush()
    child = _stroke_child(ob)
    # missing
    bpy.data.objects.remove(child, do_unlink=True)
    from seam_studio.engine import core
    core.request_revalidate()
    flush()
    assert _stroke_child(ob) is not None
    # duplicated
    first = _stroke_child(ob)
    dup = first.copy()
    dup.data = first.data.copy()
    for c in first.users_collection:
        c.objects.link(dup)
    assert len([o for o in bpy.data.objects if o.seam.role == "STROKE"]) == 2
    core.request_revalidate()
    flush()
    strokes = [o for o in bpy.data.objects if o.seam.role == "STROKE"]
    assert len(strokes) == 1 and strokes[0].parent == ob
    # orphaned: owner deleted
    bpy.data.objects.remove(ob, do_unlink=True)
    core.request_revalidate()
    flush()
    assert not [o for o in bpy.data.objects if o.seam.role == "STROKE"]


def test_duplicated_owner_gets_its_own_stroke(doc):
    _need_core(stroke=True)
    ob = _rect(100, 100)
    ob.seam.style.stroke_enabled = True
    flush()
    cp = ob.copy()
    cp.data = ob.data.copy()
    cp.name = "Copy"
    for c in ob.users_collection:
        c.objects.link(cp)
    flush()
    flush()
    assert cp.seam.id != ob.seam.id
    assert _stroke_child(cp) is not None and _stroke_child(ob) is not None
    assert _stroke_child(cp) != _stroke_child(ob)
    assert _mat(cp)["seam_owner"] == cp.seam.id and _mat(ob)["seam_owner"] == ob.seam.id
    assert len([o for o in bpy.data.objects if o.seam.role == "STROKE"]) == 2


def test_stroke_follows_owner_layer_move(doc):
    _need_core(stroke=True)
    from seam_studio.vector import canvas
    bpy.ops.seam.layer_add(name="Top")
    ob = _rect(50, 50)
    ob.seam.style.stroke_enabled = True
    flush()
    lower = canvas.layers(doc)[0]
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    lower.objects.link(ob)
    flush()
    assert set(_stroke_child(ob).users_collection) == {lower}


def test_stroke_survives_save_and_reload(doc, blend_path):
    _need_core(stroke=True)
    ob = _rect(100, 100)
    ob.seam.style.stroke_enabled = True
    flush()
    name = ob.name
    area = _mesh_area(_stroke_child(ob))
    save_and_reload(blend_path)
    from seam_studio.engine import core
    core.request_revalidate()
    res = flush()
    ob = bpy.data.objects[name]
    assert not [r for r in res if r.status == "ok" and r.key.startswith(("stroke:", "material:", "gradient:"))]
    assert _mesh_area(_stroke_child(ob)) == pytest.approx(area)


# ---------------------------------------------------------------- text

def _text_object(scene, body="Hi"):
    from seam_studio.vector import canvas
    cu = bpy.data.curves.new("Txt", "FONT")
    cu.body = body
    cu.size = 40.0
    ob = bpy.data.objects.new("Txt", cu)
    canvas.add_artwork_object(scene, ob, role="TEXT")
    return ob


def test_text_object_gets_material_and_stroke(doc):
    _need_core(stroke=True)
    ob = _text_object(doc, "Hi")
    ob.seam.style.fill_color = (0.0, 0.0, 1.0, 1.0)
    ob.seam.style.stroke_enabled = True
    ob.seam.style.stroke_width = 2.0
    flush()
    assert _mat(ob)["seam_owner"] == ob.seam.id
    assert _emission(_mat(ob))[:3] == pytest.approx((0, 0, 1))
    ch = _stroke_child(ob)
    assert ch is not None and len(ch.data.polygons) > 0
    ob.seam.style.fill_type = "LINEAR"
    flush()
    assert bpy.data.images.get("SEAM Gradient " + ob.seam.id) is not None
