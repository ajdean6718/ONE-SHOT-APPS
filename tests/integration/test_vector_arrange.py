"""Arrange / group / align / distribute / numeric transform / guides."""

import math

import bpy
import pytest
from mathutils import Matrix

from helpers import flush


def _need_core():
    from seam_studio.seam_core import shapes
    try:
        shapes.rect(1, 1)
    except NotImplementedError:
        pytest.skip("core pending: seam_core.shapes")


@pytest.fixture
def doc(seam):
    _need_core()
    bpy.ops.seam.canvas_new(width=1000, height=800)
    return bpy.context.scene


def _add(kind="RECT", w=100, h=100, x=0, y=0, name=None):
    assert bpy.ops.seam.add_shape(kind=kind, width=w, height=h, x=x, y=y) == {"FINISHED"}
    ob = bpy.context.active_object
    if name:
        ob.name = name
    return ob


def _select(*obs):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[-1]


def _bounds(ob):
    from seam_studio.vector import canvas
    return canvas.canvas_bounds(ob)


def _stack(scene):
    from seam_studio.vector import canvas
    return [o.name for o in canvas.artwork_objects(scene)]


# ---------------------------------------------------------------- z-order

def test_pure_reorder_logic():
    from seam_studio.vector.arrange import reorder_stack
    s = ["a", "b", "c", "d"]
    assert reorder_stack(s, {"b"}, "FRONT") == ["a", "c", "d", "b"]
    assert reorder_stack(s, {"c"}, "BACK") == ["c", "a", "b", "d"]
    assert reorder_stack(s, {"b"}, "FORWARD") == ["a", "c", "b", "d"]
    assert reorder_stack(s, {"c"}, "BACKWARD") == ["a", "c", "b", "d"]
    assert reorder_stack(s, {"b", "c"}, "FORWARD") == ["a", "d", "b", "c"]
    assert reorder_stack(s, {"d"}, "FORWARD") == s
    assert reorder_stack(s, {"a"}, "BACKWARD") == s
    assert reorder_stack(s, {"a", "c"}, "FRONT") == ["b", "d", "a", "c"]


def test_arrange_operators_change_z_slots(doc):
    a, b, c, d = (_add(name=n, x=i * 10) for i, n in enumerate("ABCD"))
    zs = lambda: [o.location.z for o in (a, b, c, d)]
    assert _stack(doc) == ["A", "B", "C", "D"]
    _select(b)
    bpy.ops.seam.arrange(mode="FRONT")
    assert _stack(doc) == ["A", "C", "D", "B"]
    bpy.ops.seam.arrange(mode="BACKWARD")
    assert _stack(doc) == ["A", "C", "B", "D"]
    bpy.ops.seam.arrange(mode="BACK")
    assert _stack(doc) == ["B", "A", "C", "D"]
    bpy.ops.seam.arrange(mode="FORWARD")
    assert _stack(doc) == ["A", "B", "C", "D"]
    # z slots are contiguous steps from the layer base
    z = sorted(zs())
    assert z == pytest.approx([0.0, 0.01, 0.02, 0.03])
    _select(a, c)
    bpy.ops.seam.arrange(mode="FRONT")
    assert _stack(doc) == ["B", "D", "A", "C"]


def test_arrange_stays_within_layer(doc):
    low = _add(name="Low")
    bpy.ops.seam.layer_add(name="Upper")
    up1 = _add(name="Up1")
    up2 = _add(name="Up2")
    _select(up1)
    bpy.ops.seam.arrange(mode="FRONT")
    assert up1.location.z > up2.location.z >= 10.0
    assert low.location.z < 10.0
    _select(low)
    bpy.ops.seam.arrange(mode="FRONT")
    assert low.location.z < 10.0  # still below the upper layer
    assert _stack(doc) == ["Low", "Up2", "Up1"]


def test_arrange_poll_message(doc):
    assert not bpy.ops.seam.arrange.poll()
    _add()
    assert bpy.ops.seam.arrange.poll()


# ---------------------------------------------------------------- group / ungroup

def _worlds(obs):
    return {o.name: o.matrix_world.copy() for o in obs}


def _same(w1, w2, tol=1e-6):
    for n, m in w1.items():
        for r in range(4):
            for c in range(4):
                assert m[r][c] == pytest.approx(w2[n][r][c], abs=tol), (n, r, c)


def test_group_and_ungroup_preserve_world_matrices_exactly(doc):
    a = _add(name="A", x=100.5, y=-50.25)
    b = _add("ELLIPSE", 80, 40, x=350.0, y=-120.0, name="B")
    c = _add(name="C", x=700, y=-300)
    b.rotation_euler.z = 0.7
    b.scale = (1.5, 0.75, 1.0)
    bpy.context.view_layer.update()
    before = _worlds([a, b, c])
    _select(a, b)
    assert bpy.ops.seam.group() == {"FINISHED"}
    g = bpy.context.active_object
    assert g.seam.role == "GROUP" and g.type == "EMPTY" and g.parent.seam.role == "CANVAS_ROOT"
    assert a.parent == g and b.parent == g and c.parent != g
    assert g.matrix_parent_inverse == Matrix.Identity(4)
    assert a.matrix_parent_inverse == Matrix.Identity(4)
    bpy.context.view_layer.update()
    _same(before, _worlds([a, b, c]))
    assert g.users_collection[0] == a.users_collection[0]
    # group origin is on whole pixels (bit-exact world matrices)
    assert g.location.x == round(g.location.x) and g.location.y == round(g.location.y)
    _select(g)
    gname = g.name
    assert bpy.ops.seam.ungroup() == {"FINISHED"}
    assert gname not in bpy.data.objects
    assert a.parent.seam.role == "CANVAS_ROOT" and b.parent.seam.role == "CANVAS_ROOT"
    bpy.context.view_layer.update()
    _same(before, _worlds([a, b, c]))


def test_group_after_moving_group_children_follow_and_ungroup_keeps_world(doc):
    a = _add(name="A", x=0, y=0)
    b = _add(name="B", x=200, y=0)
    _select(a, b)
    bpy.ops.seam.group()
    g = bpy.context.active_object
    g.location.x += 300
    g.rotation_euler.z = math.radians(90)
    bpy.context.view_layer.update()
    moved = _worlds([a, b])
    _select(b)  # a child selected: ungroup dissolves its group
    assert bpy.ops.seam.ungroup() == {"FINISHED"}
    bpy.context.view_layer.update()
    _same(moved, _worlds([a, b]), tol=1e-4)


def test_nested_groups_and_group_bounds(doc):
    from seam_studio.vector import canvas
    a = _add(name="A", x=0, y=0, w=100, h=100)
    b = _add(name="B", x=300, y=0, w=100, h=100)
    c = _add(name="C", x=0, y=-500, w=100, h=100)
    bpy.context.view_layer.update()
    before = _worlds([a, b, c])
    _select(a, b)
    bpy.ops.seam.group()
    g1 = bpy.context.active_object
    assert canvas.canvas_bounds(g1) == pytest.approx((-50, -50, 350, 50), abs=1e-3)
    _select(g1, c)
    assert bpy.ops.seam.group() == {"FINISHED"}
    g2 = bpy.context.active_object
    assert g1.parent == g2 and c.parent == g2
    bpy.context.view_layer.update()
    _same(before, _worlds([a, b, c]), tol=1e-5)
    assert canvas.canvas_bounds(g2) == pytest.approx((-50, -550, 350, 50), abs=1e-3)


def test_group_rejects_mixed_parents_and_layers(doc):
    a = _add(name="A")
    b = _add(name="B", x=200)
    _select(a, b)
    bpy.ops.seam.group()
    c = _add(name="C", x=400)
    _select(a, c)  # a inside a group, c on the canvas
    with pytest.raises(RuntimeError, match="share one parent"):
        bpy.ops.seam.group()
    bpy.ops.seam.layer_add(name="Other")
    d = _add(name="D")
    _select(c, d)
    with pytest.raises(RuntimeError, match="several layers"):
        bpy.ops.seam.group()


def test_arrange_group_as_one_unit(doc):
    a = _add(name="A", x=0)
    b = _add(name="B", x=100)
    c = _add(name="C", x=200)
    _select(a, b)
    bpy.ops.seam.group()
    g = bpy.context.active_object
    _select(g)
    bpy.ops.seam.arrange(mode="FRONT")
    assert g.location.z > c.location.z
    bpy.ops.seam.arrange(mode="BACK")
    assert g.location.z < c.location.z


# ---------------------------------------------------------------- align / distribute

def test_align_delta_math():
    from seam_studio.vector.arrange import align_deltas, distribute_deltas
    boxes = [(0, 0, 10, 10), (20, 5, 50, 25)]
    t = (0, 0, 50, 25)
    assert align_deltas("LEFT", boxes, t) == [(0, 0), (-20, 0)]
    assert align_deltas("RIGHT", boxes, t) == [(40, 0), (0, 0)]
    assert align_deltas("CENTER_H", boxes, t) == [(20, 0), (-10, 0)]
    assert align_deltas("TOP", boxes, t) == [(0, 15), (0, 0)]
    assert align_deltas("BOTTOM", boxes, t) == [(0, 0), (0, -5)]
    assert align_deltas("CENTER_V", boxes, t) == [(0, 7.5), (0, -2.5)]
    b3 = [(0, 0, 10, 10), (12, 0, 22, 10), (100, 0, 110, 10)]
    # centres 5, 17, 105 -> the middle one goes to 55: delta 38
    assert distribute_deltas("X", "CENTERS", b3) == [0.0, 38.0, 0.0]
    # span 110, sizes 30 -> gap 40: the middle box starts at 50 -> delta 38
    assert distribute_deltas("X", "SPACING", b3) == [0.0, 38.0, 0.0]
    # unequal sizes: boxes 10, 40, 10 wide over span 210 -> gap 75: middle box starts at 85
    assert distribute_deltas("X", "SPACING", [(0, 0, 10, 10), (30, 0, 70, 10), (200, 0, 210, 10)]) == [0.0, 55.0, 0.0]


def test_align_selection_artboard_active(doc):
    a = _add(w=100, h=50, x=100, y=-100, name="A")      # x 50..150
    b = _add(w=40, h=40, x=400, y=-300, name="B")       # x 380..420
    c = _add(w=60, h=20, x=250, y=-50, name="C")        # x 220..280
    _select(a, b, c)
    assert bpy.ops.seam.align(mode="LEFT", relative_to="SELECTION") == {"FINISHED"}
    assert [_bounds(o)[0] for o in (a, b, c)] == pytest.approx([50, 50, 50], abs=1e-3)
    bpy.ops.seam.align(mode="RIGHT", relative_to="SELECTION")
    assert [_bounds(o)[2] for o in (a, b, c)] == pytest.approx([150, 150, 150], abs=1e-3)
    bpy.ops.seam.align(mode="CENTER_H", relative_to="ARTBOARD")
    assert [(_bounds(o)[0] + _bounds(o)[2]) / 2 for o in (a, b, c)] == pytest.approx([500, 500, 500], abs=1e-3)
    bpy.ops.seam.align(mode="TOP", relative_to="ARTBOARD")
    assert [_bounds(o)[3] for o in (a, b, c)] == pytest.approx([0, 0, 0], abs=1e-3)
    bpy.ops.seam.align(mode="BOTTOM", relative_to="ARTBOARD")
    assert [_bounds(o)[1] for o in (a, b, c)] == pytest.approx([-800, -800, -800], abs=1e-3)
    bpy.ops.seam.align(mode="CENTER_V", relative_to="ARTBOARD")
    assert [(_bounds(o)[1] + _bounds(o)[3]) / 2 for o in (a, b, c)] == pytest.approx([-400] * 3, abs=1e-3)
    # active object stays, others align to it
    b.location.x += 200
    _select(a, c, b)
    before = _bounds(b)
    bpy.ops.seam.align(mode="LEFT", relative_to="ACTIVE")
    assert _bounds(b) == pytest.approx(before, abs=1e-3)
    assert _bounds(a)[0] == pytest.approx(before[0], abs=1e-3) and _bounds(c)[0] == pytest.approx(before[0], abs=1e-3)


def test_align_uses_evaluated_geometry_of_rotated_scaled_objects(doc):
    a = _add(w=100, h=40, x=300, y=-300, name="A")
    a.rotation_euler.z = math.radians(90)
    a.scale.x = 2.0
    b = _add(w=10, h=10, x=0, y=0, name="B")
    bpy.context.view_layer.update()
    # A is now 40 wide (rotated) and 200 tall
    assert _bounds(a) == pytest.approx((280, -400, 320, -200), abs=1e-2)
    _select(a, b)
    bpy.ops.seam.align(mode="LEFT", relative_to="SELECTION")
    assert _bounds(b)[0] == pytest.approx(-5, abs=1e-3) and _bounds(a)[0] == pytest.approx(-5, abs=1e-2)


def test_align_poll_and_single_selection_message(doc):
    a = _add()
    _select(a)
    before = _bounds(a)
    assert bpy.ops.seam.align(mode="LEFT", relative_to="SELECTION") == {"CANCELLED"}
    assert _bounds(a) == pytest.approx(before)
    assert bpy.ops.seam.align(mode="LEFT", relative_to="ARTBOARD") == {"FINISHED"}
    assert _bounds(a)[0] == pytest.approx(0.0, abs=1e-3)


def test_distribute_centers_and_spacing(doc):
    a = _add(w=20, h=20, x=0, y=0, name="A")        # centre 0
    b = _add(w=60, h=20, x=30, y=0, name="B")       # centre 30
    c = _add(w=20, h=20, x=100, y=0, name="C")      # centre 100
    d = _add(w=40, h=20, x=400, y=0, name="D")      # centre 400
    _select(a, b, c, d)
    assert bpy.ops.seam.distribute(axis="X", by="CENTERS") == {"FINISHED"}
    ctr = lambda o: (_bounds(o)[0] + _bounds(o)[2]) / 2
    assert [ctr(o) for o in (a, b, c, d)] == pytest.approx([0, 400 / 3, 800 / 3, 400], abs=1e-3)
    assert bpy.ops.seam.distribute(axis="X", by="SPACING") == {"FINISHED"}
    gaps = [_bounds(n)[0] - _bounds(p)[2] for p, n in ((a, b), (b, c), (c, d))]
    assert gaps[0] == pytest.approx(gaps[1], abs=1e-3) and gaps[1] == pytest.approx(gaps[2], abs=1e-3)
    assert _bounds(a)[0] == pytest.approx(-10, abs=1e-3) and _bounds(d)[2] == pytest.approx(420, abs=1e-3)
    # vertical
    for i, o in enumerate((a, b, c, d)):
        o.location.y = (0, -10, -250, -600)[i]
    bpy.context.view_layer.update()
    bpy.ops.seam.distribute(axis="Y", by="CENTERS")
    cy = lambda o: (_bounds(o)[1] + _bounds(o)[3]) / 2
    assert [cy(o) for o in (a, b, c, d)] == pytest.approx([0, -200, -400, -600], abs=1e-3)


def test_distribute_needs_three(doc):
    a, b = _add(x=0), _add(x=100)
    _select(a, b)
    assert not bpy.ops.seam.distribute.poll()


# ---------------------------------------------------------------- numeric transform

def test_numeric_transform_round_trip_artboard_relative_y_down(doc):
    ob = _add(w=100, h=50, x=300, y=-200, name="N")
    bpy.context.view_layer.update()
    # artboard at (0, 0) top-left: x = 250, y = 175 (down), w = 100, h = 50
    assert (ob.seam_x, ob.seam_y, ob.seam_w, ob.seam_h) == pytest.approx((250, 175, 100, 50), abs=1e-3)
    ob.seam_x = 40
    ob.seam_y = 10
    assert (ob.seam_x, ob.seam_y) == pytest.approx((40, 10), abs=1e-3)
    assert _bounds(ob) == pytest.approx((40, -60, 140, -10), abs=1e-3)
    ob.seam_w = 250
    ob.seam_h = 20
    assert (ob.seam_w, ob.seam_h) == pytest.approx((250, 20), abs=1e-3)
    assert (ob.seam_x, ob.seam_y) == pytest.approx((40, 10), abs=1e-3)  # top-left kept
    assert ob.scale.x == pytest.approx(2.5) and ob.scale.y == pytest.approx(0.4)
    ob.seam_rot = 30
    assert ob.seam_rot == pytest.approx(30, abs=1e-3)
    assert ob.rotation_euler.z == pytest.approx(math.radians(30), abs=1e-5)
    ob.seam_rot = 0
    assert (ob.seam_x, ob.seam_y, ob.seam_w, ob.seam_h) == pytest.approx((40, 10, 250, 20), abs=1e-2)


def test_numeric_transform_rotated_object_size_round_trip(doc):
    ob = _add(w=100, h=40, x=500, y=-400, name="R")
    ob.seam_rot = 25
    w0, h0 = ob.seam_w, ob.seam_h
    ob.seam_w = w0 + 60
    assert ob.seam_w == pytest.approx(w0 + 60, abs=1e-3)
    ob.seam_h = h0 + 10
    assert ob.seam_h == pytest.approx(h0 + 10, abs=1e-3)
    assert ob.seam_w == pytest.approx(w0 + 60, abs=1e-3)
    assert ob.seam_rot == pytest.approx(25, abs=1e-3)


def test_numeric_transform_follows_active_artboard_and_units(doc):
    ob = _add(w=100, h=100, x=2200, y=-300, name="U")
    bpy.ops.seam.artboard_add(width=500, height=500)  # active: x=1100
    bpy.context.view_layer.update()
    assert ob.seam_x == pytest.approx(2150 - 1100, abs=1e-3)
    doc.seam.doc.units = "IN"
    doc.seam.doc.dpi = 96.0
    assert ob.seam_w == pytest.approx(100 / 96.0, abs=1e-5)
    ob.seam_w = 2.0
    doc.seam.doc.units = "PX"
    assert ob.seam_w == pytest.approx(192.0, abs=1e-3)


def test_numeric_transform_for_group(doc):
    a = _add(w=100, h=100, x=0, y=0, name="A")
    b = _add(w=100, h=100, x=300, y=0, name="B")
    _select(a, b)
    bpy.ops.seam.group()
    g = bpy.context.active_object
    bpy.context.view_layer.update()
    assert (g.seam_w, g.seam_h) == pytest.approx((400, 100), abs=1e-3)
    g.seam_x = 100
    assert _bounds(a)[0] == pytest.approx(100, abs=1e-3)
    assert _bounds(b)[0] == pytest.approx(400, abs=1e-3)


# ---------------------------------------------------------------- guides

def test_guides_add_and_clear(doc):
    bpy.ops.seam.artboard_add(width=500, height=500)  # active artboard at x=1100 (top-left)
    assert bpy.ops.seam.guide_add(orientation="VERTICAL", position=120) == {"FINISHED"}
    assert bpy.ops.seam.guide_add(orientation="HORIZONTAL", position=30) == {"FINISHED"}
    assert bpy.ops.seam.guide_add(orientation="HORIZONTAL", position=-77, artboard_relative=False) == {"FINISHED"}
    guides = [o for o in doc.objects if o.seam.role == "GUIDE"]
    assert len(guides) == 3
    v = next(g for g in guides if g["seam_orientation"] == "VERTICAL")
    assert v.location.x == pytest.approx(1100 + 120)
    h = [g for g in guides if g["seam_orientation"] == "HORIZONTAL"]
    assert sorted(g.location.y for g in h) == pytest.approx([-77, -30])  # Y down from the artboard top (y=0)
    assert all(g.hide_render and g.parent.seam.role == "CANVAS_ROOT" and len(g.data.edges) == 1 for g in guides)
    # guides are not artwork: never aligned / stacked / counted
    from seam_studio.vector import canvas
    assert not [o for o in canvas.artwork_objects(doc) if o.seam.role == "GUIDE"]
    assert all(c not in canvas.layers(doc) for c in guides[0].users_collection)
    assert bpy.ops.seam.guides_clear() == {"FINISHED"}
    assert not [o for o in doc.objects if o.seam.role == "GUIDE"]
    assert not bpy.ops.seam.guides_clear.poll()


# ---------------------------------------------------------------- review regressions

def _strokes_of(ob):
    return [c for c in ob.children if c.seam.role == "STROKE"]


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("extra", [0, 1])
def test_ungroup_works_for_any_child_count_and_base_order(doc, n, extra):
    """A group of ONE shape (view-layer None base after removal) used to raise AttributeError mid-operator."""
    kids = [_add(w=50, h=50, x=i * 100) for i in range(n)]
    for _ in range(extra):
        _add(w=10, h=10, x=900)
    flush()
    before = [k.matrix_world.copy() for k in kids]
    _select(*kids)
    assert bpy.ops.seam.group() == {"FINISHED"}
    g = bpy.context.active_object
    assert g.seam.role == "GROUP"
    gname = g.name
    _select(g)
    assert bpy.ops.seam.ungroup() == {"FINISHED"}
    flush()
    assert gname not in bpy.data.objects
    for k, m in zip(kids, before):
        assert k.parent.seam.role == "CANVAS_ROOT"
        assert max(abs(k.matrix_world[i][j] - m[i][j]) for i in range(4) for j in range(4)) < 1e-6
        assert k.select_get()
    assert bpy.context.view_layer.objects.active in kids


def test_group_of_one_without_any_other_artwork_then_ungroup_undoably(doc):
    ob = _add()
    _select(ob)
    assert bpy.ops.seam.group() == {"FINISHED"}
    flush()
    assert bpy.ops.seam.ungroup() == {"FINISHED"}
    assert ob.parent.seam.role == "CANVAS_ROOT"


def test_group_children_strokes_stay_below_next_sibling_and_next_object(doc):
    obs = []
    for i in range(3):
        ob = _add(w=100, h=100, x=i * 20)
        ob.seam.style.stroke_enabled = True
        obs.append(ob)
    flush()
    top = _add(w=100, h=100)
    flush()
    _select(*obs)
    assert bpy.ops.seam.group() == {"FINISHED"}
    g = bpy.context.active_object
    flush()
    _select(g)
    bpy.ops.seam.arrange(mode="BACK")  # compacts the group's children into its slot
    flush()

    def wz(o):
        return o.matrix_world.translation.z

    rows = [(wz(o), wz(_strokes_of(o)[0])) for o in obs]
    for i in range(2):
        assert rows[i][0] < rows[i][1] < rows[i + 1][0], rows
    assert rows[2][1] < wz(top), rows
    assert wz(g) <= rows[0][0] < rows[2][1] < wz(g) + 0.01 + 1e-9


def test_nested_group_strokes_stay_inside_their_slot(doc):
    a = _add(w=50, h=50, x=0)
    b = _add(w=50, h=50, x=100)
    c = _add(w=50, h=50, x=200)
    for o in (a, b, c):
        o.seam.style.stroke_enabled = True
    flush()
    _select(a, b)
    bpy.ops.seam.group()
    inner = bpy.context.active_object
    _select(inner, c)
    bpy.ops.seam.group()
    outer = bpy.context.active_object
    flush()
    _select(outer)
    bpy.ops.seam.arrange(mode="FRONT")
    flush()
    zs = sorted([o.matrix_world.translation.z for o in (a, b, c)]
                + [_strokes_of(o)[0].matrix_world.translation.z for o in (a, b, c)])
    z0 = outer.matrix_world.translation.z
    assert zs[0] >= z0 - 1e-9 and zs[-1] < z0 + 0.01
    # every stroke sits between its owner and the next higher fill
    fills = sorted((o.matrix_world.translation.z, o) for o in (a, b, c))
    for (z, o), nxt in zip(fills, fills[1:]):
        assert z < _strokes_of(o)[0].matrix_world.translation.z < nxt[0]


def test_ungroup_nonuniform_scaled_group_keeps_world_matrix_of_rotated_shape(doc):
    """Shear (rotated child in a non-uniformly scaled group) is baked into the curve points."""
    ob = _add(w=100, h=40, x=0, name="rot")
    other = _add(w=100, h=40, x=500)
    ob.rotation_euler.z = math.radians(30)
    flush()
    _select(ob, other)
    assert bpy.ops.seam.group() == {"FINISHED"}
    g = ob.parent
    g.scale = (3.0, 1.0, 1.0)
    flush()
    import numpy as np
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    before = np.array([tuple(ob.matrix_world @ v.co) for v in me.vertices])
    ev.to_mesh_clear()
    _select(g)
    assert bpy.ops.seam.ungroup() == {"FINISHED"}
    flush()
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    after = np.array([tuple(ob.matrix_world @ v.co) for v in me.vertices])
    ev.to_mesh_clear()
    assert before.shape == after.shape
    assert np.abs(before - after).max() < 1e-3
    assert ob.seam.shape.kind == "NONE"  # the sheared geometry no longer matches a recipe


def test_transform_panel_cache_is_scoped_to_draw(doc):
    from seam_studio.vector import arrange
    ob = _add(w=100, h=60)
    flush()
    with arrange.ui_cache():
        a = arrange.ui_values(ob)
        b = arrange.ui_values(ob)
        assert a is b
    ob.seam.shape.width = 200.0
    flush()
    assert arrange.ui_values(ob)[2] == pytest.approx(200.0)
