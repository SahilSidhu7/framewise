"""Schema v3 layers: Photoshop blend modes, masks, mattes, solid and adjustment
layers, anchor point and bezier easing. Rendered with solids only, so no media
files are needed and every expected pixel can be worked out by hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aieditor_engine.core import builtin, video_commands  # noqa: F401  (register commands)
from aieditor_engine.core.bus import CommandBus
from aieditor_engine.core.commands import CommandError
from aieditor_engine.core.project import (
    BLEND_MODES,
    Clip,
    EditOp,
    Keyframe,
    Mask,
    Project,
    ProjectKind,
    Track,
    cubic_bezier,
    keyframe_value,
    new_id,
)
from aieditor_engine.core.store import migrate
from aieditor_engine.imaging.color import linear_to_srgb, srgb_to_linear
from aieditor_engine.video.compose import RenderSource, render_frame
from aieditor_engine.video.layers import blend_rgb, composite
from aieditor_engine.video.vdsl import apply_video_program, timeline_to_dsl

W, H = 160, 120


def _solid(color: str, **kw) -> Clip:
    return Clip(id=new_id("clp"), source="solid", color=color, start=0, in_point=0, out_point=4, **kw)


def _project(*layers: Clip | tuple[Clip, dict]) -> Project:
    p = Project(name="L", kind=ProjectKind.VIDEO)
    p.settings.width, p.settings.height = W, H
    for i, layer in enumerate(layers):
        clip, track_kw = layer if isinstance(layer, tuple) else (layer, {})
        p.tracks.append(Track(id=new_id("trk"), kind="video" if i == 0 else "overlay", clips=[clip], **track_kw))
    return p


def _frame(p: Project, tmp_path: Path, t: float = 1.0) -> np.ndarray:
    return render_frame(p, t, W, H, RenderSource(tmp_path)).astype(np.float32)


def _u8(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)


# ------------------------------------------------------------------ blend


CB, CS = 0.25, 0.75  # base and source values (display-encoded)
SEPARABLE = {
    "normal": CS,
    "darken": 0.25,
    "multiply": CB * CS,
    "color_burn": 1 - (1 - CB) / CS,
    "linear_burn": 0.0,
    "lighten": 0.75,
    "screen": CB + CS - CB * CS,
    "color_dodge": CB / (1 - CS),
    "add": 1.0,
    "overlay": 2 * CB * CS,  # base <= 0.5
    "soft_light": CB + (2 * CS - 1) * ((((16 * CB - 12) * CB + 4) * CB) - CB),  # W3C formula, CB <= 0.25 branch
    "hard_light": 1 - (1 - CB) * (1 - (2 * CS - 1)),
    "vivid_light": min(1, CB / (1 - (2 * CS - 1))),
    "linear_light": CB + 2 * CS - 1,
    "pin_light": max(CB, 2 * CS - 1),
    "hard_mix": 1.0,
    "difference": 0.5,
    "exclusion": CB + CS - 2 * CB * CS,
    "subtract": 0.0,
    "divide": CB / CS,
}


@pytest.mark.parametrize("mode", list(SEPARABLE))
def test_separable_blend_modes_match_the_formulas(mode):
    cb = np.full((1, 1, 3), CB, np.float32)
    cs = np.full((1, 1, 3), CS, np.float32)
    got = float(np.clip(blend_rgb(cb, cs, mode), 0, 1)[0, 0, 0])
    assert got == pytest.approx(min(1.0, max(0.0, SEPARABLE[mode])), abs=1e-5), mode


def test_non_separable_modes_keep_the_right_parts():
    lum = lambda c: float(np.ravel(c[..., 0] * 0.3 + c[..., 1] * 0.59 + c[..., 2] * 0.11)[0])  # noqa: E731
    base = np.array([[[0.6, 0.3, 0.2]]], np.float32)
    src = np.array([[[0.2, 0.4, 0.7]]], np.float32)
    assert lum(blend_rgb(base, src, "luminosity")) == pytest.approx(lum(src), abs=1e-4)
    assert lum(blend_rgb(base, src, "color")) == pytest.approx(lum(base), abs=1e-4)
    assert lum(blend_rgb(base, src, "hue")) == pytest.approx(lum(base), abs=1e-4)
    sat = lambda c: float(c.max() - c.min())  # noqa: E731
    assert sat(blend_rgb(base, src, "saturation")) == pytest.approx(sat(src), abs=1e-4)
    assert (blend_rgb(base, src, "darker_color") == src).all() and (blend_rgb(base, src, "lighter_color") == base).all()


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_every_mode_composites_in_range_and_respects_opacity(mode):
    rng = np.random.default_rng(1)
    canvas = rng.random((8, 8, 3), dtype=np.float32)
    layer = rng.random((8, 8, 3), dtype=np.float32)
    out = composite(canvas.copy(), layer * 0.0, 0.0, mode)  # fully transparent: canvas unchanged
    assert np.allclose(out, canvas, atol=1e-4), mode
    full = composite(canvas.copy(), layer, 1.0, mode)
    assert np.isfinite(full).all() and full.min() >= -1e-4, mode


def test_blend_through_the_renderer(tmp_path):
    base, top = "#405060", "#c08020"
    for mode in ("multiply", "difference", "soft_light", "color_dodge", "luminosity"):
        got = _frame(_project(_solid(base), _solid(top, blend=mode)), tmp_path)[60, 80]
        cb = linear_to_srgb(srgb_to_linear(_u8(base) / 255))
        cs = linear_to_srgb(srgb_to_linear(_u8(top) / 255))
        if mode == "multiply":  # linear-light mode
            want = linear_to_srgb(srgb_to_linear(cb) * srgb_to_linear(cs)) * 255
        else:
            want = np.clip(blend_rgb(cb[None, None], cs[None, None], mode)[0, 0], 0, 1) * 255
        assert np.abs(got - want).max() <= 2.5, (mode, got, want)


# ------------------------------------------------------------------ masks


def test_rect_and_ellipse_masks(tmp_path):
    red, blue = "#ff0000", "#0000ff"
    p = _project(_solid(red), _solid(blue, masks=[Mask.rect(0.25, 0.25, 0.75, 0.75)]))
    f = _frame(p, tmp_path)
    assert f[60, 80, 2] > 200 and f[60, 80, 0] < 30  # inside: blue
    assert f[10, 10, 0] > 200 and f[10, 10, 2] < 30  # outside: red
    inv = _project(_solid(red), _solid(blue, masks=[Mask.rect(0.25, 0.25, 0.75, 0.75, inverted=True)]))
    g = _frame(inv, tmp_path)
    assert g[60, 80, 0] > 200 and g[10, 10, 2] > 200
    soft = _project(_solid("#000000"), _solid("#ffffff", masks=[Mask.ellipse(0.5, 0.5, 0.3, 0.3, feather=0.1)]))
    s = _frame(soft, tmp_path)[60]
    assert s[80, 0] > 250 and s[2, 0] < 5 and 40 < s[128, 0] < 215  # solid centre, soft edge at the rim, black outside


def test_mask_modes_combine_like_after_effects(tmp_path):
    left = Mask.rect(0.0, 0.0, 0.6, 1.0)
    right = Mask.rect(0.4, 0.0, 1.0, 1.0)

    def white_at(*masks: Mask) -> list[bool]:
        f = _frame(_project(_solid("#000000"), _solid("#ffffff", masks=list(masks))), tmp_path)
        return [bool(f[60, x, 0] > 128) for x in (20, 80, 140)]  # left, overlap, right

    assert white_at(left, right) == [True, True, True]
    assert white_at(left, right.model_copy(update={"mode": "subtract"})) == [True, False, False]
    assert white_at(left, right.model_copy(update={"mode": "intersect"})) == [False, True, False]
    assert white_at(left, right.model_copy(update={"mode": "difference"})) == [True, False, True]


def test_mask_moves_with_the_layer(tmp_path):
    clip = _solid("#ffffff", masks=[Mask.rect(0.4, 0.4, 0.6, 0.6)])
    clip.transform.x = 0.25  # layer moves right by 40 px; its mask goes with it
    f = _frame(_project(_solid("#000000"), clip), tmp_path)
    assert f[60, 80, 0] < 20 and f[60, 120, 0] > 230


# ------------------------------------------------------------------ mattes


def test_track_mattes(tmp_path):
    fill = _solid("#00ff00")
    matte = _solid("#ffffff", masks=[Mask.rect(0.0, 0.0, 0.5, 1.0)])  # left half
    for mode, left_green in (("alpha", True), ("alpha_inverted", False), ("luma", True), ("luma_inverted", False)):
        p = _project(_solid("#000000"), (fill.model_copy(deep=True), {"matte": mode}), matte.model_copy(deep=True))
        f = _frame(p, tmp_path)
        green_left, green_right = f[60, 30, 1] > 200, f[60, 130, 1] > 200
        assert (green_left, green_right) == (left_green, not left_green), mode
        assert f[60, 30, 0] < 20  # the white matte itself is not drawn
    gray = _project(_solid("#000000"), (_solid("#ffffff"), {"matte": "luma"}), _solid("#808080"))
    v = _frame(gray, tmp_path)[60, 80, 0]
    assert 100 < v < 200  # a mid-grey luma matte lets about half through


# --------------------------------------------------------- solid / adjust


def test_adjustment_layer_changes_only_what_is_below_inside_its_mask(tmp_path):
    adj = Clip(id=new_id("clp"), source="adjustment", start=0, in_point=0, out_point=4,
               ops=[EditOp(id="o1", op="black_white", params={})], masks=[Mask.rect(0.5, 0.0, 1.0, 1.0)])
    p = _project(_solid("#d03020"), adj)
    f = _frame(p, tmp_path)
    left, right = f[60, 30], f[60, 130]
    assert left[0] - left[2] > 100  # untouched: still red
    assert np.ptp(right) < 6  # adjusted: grey
    half = adj.model_copy(update={"opacity": 0.5, "masks": []})
    g = _frame(_project(_solid("#d03020"), half), tmp_path)[60, 80]
    assert 10 < g[0] - g[2] < left[0] - left[2]


# -------------------------------------------------------- anchor / easing


def test_anchor_is_the_pivot_for_rotation(tmp_path):
    box = _solid("#ffffff", masks=[Mask.rect(0.5, 0.4, 0.7, 0.6)])  # right of centre
    turned = box.model_copy(deep=True)
    turned.transform.rotation = 180  # about the centre: lands left of centre
    f = _frame(_project(_solid("#000000"), turned), tmp_path)
    assert f[60, 50, 0] > 230 and f[60, 100, 0] < 20
    pivot = box.model_copy(deep=True)
    pivot.transform.rotation = 180
    pivot.transform.anchor_x = 0.1  # pivot at x=96: the box spins in place
    g = _frame(_project(_solid("#000000"), pivot), tmp_path)
    assert g[60, 96, 0] > 230 and g[60, 50, 0] < 20


def test_bezier_easing():
    assert cubic_bezier(0, 0, 1, 1, 0.3) == pytest.approx(0.3, abs=1e-4)
    assert cubic_bezier(0.42, 0, 1, 1, 0.5) < 0.4  # ease-in starts slow
    assert cubic_bezier(0, 0, 0.58, 1, 0.5) > 0.6  # ease-out starts fast
    keys = [Keyframe(t=0, value=0, ease="bezier", bezier=(0.42, 0, 0.58, 1)), Keyframe(t=2, value=10)]
    vals = [keyframe_value(keys, t) for t in (0.5, 1.0, 1.5)]
    assert vals[0] < 2.5 and vals[1] == pytest.approx(5, abs=0.01) and vals[2] > 7.5


# ---------------------------------------------------------- DSL / commands


def test_dsl_solid_mask_anchor_roundtrip():
    p = Project(name="D", kind=ProjectKind.VIDEO)
    apply_video_program(p, "aspect 16:9\nsolid #101820 at=1 dur=3 blend=soft_light opacity=0.5 anchor=0.1,-0.2 "
                           "mask=!ellipse:0.5,0.5,0.3,0.4 feather=0.05")
    clip = next(c for t in p.tracks for c in t.clips)
    assert clip.source == "solid" and clip.blend == "soft_light" and clip.transform.anchor_x == 0.1
    assert clip.masks[0].inverted and clip.masks[0].feather == 0.05 and len(clip.masks[0].points) == 4
    text = timeline_to_dsl(p)
    assert "solid #101820" in text and "mask=!path:" in text
    q = Project(name="D2", kind=ProjectKind.VIDEO)
    apply_video_program(q, text)
    clip2 = next(c for t in q.tracks for c in t.clips)
    a = np.array([[q.x, q.y, q.in_x, q.in_y, q.out_x, q.out_y] for q in clip.masks[0].points])
    b = np.array([[q.x, q.y, q.in_x, q.in_y, q.out_x, q.out_y] for q in clip2.masks[0].points])
    assert np.abs(a - b).max() < 1e-3 and clip2.blend == clip.blend and clip2.masks[0].inverted
    for bad in ("solid red", "solid #101820 mask=ellipse:0.5,0.5,0,0.4", "solid #101820 feather=0.1", "solid #101820 blend=nope"):
        with pytest.raises(Exception):
            apply_video_program(Project(name="x", kind=ProjectKind.VIDEO), bad)


def test_layer_commands_with_undo():
    bus = CommandBus(Project(name="C", kind=ProjectKind.VIDEO))
    res = bus.execute("layer.add", {"source": "solid", "color": "#336699", "duration": 3})
    cid, tid = res["clip_id"], res["track_id"]
    bus.execute("clip.set_masks", {"clip_id": cid, "masks": [Mask.ellipse(0.5, 0.5, 0.2, 0.2).model_dump()]})
    bus.execute("clip.set_layer", {"clip_id": cid, "blend": "color_dodge", "anchor_x": 0.2, "color": "#ffffff"})
    bus.execute("track.set", {"track_id": tid, "matte": "luma"})
    _, clip = bus.project.find_clip(cid)
    assert clip.blend == "color_dodge" and clip.color == "#ffffff" and len(clip.masks) == 1
    assert bus.project.find_track(tid).matte == "luma"
    bus.execute("clip.set_keyframes", {"clip_id": cid, "param": "anchor_x", "keys": [{"t": 0, "value": 0, "ease": "bezier", "bezier": [0.3, 0, 0.7, 1]}, {"t": 1, "value": 0.3}]})
    with pytest.raises(CommandError):
        bus.execute("clip.set_keyframes", {"clip_id": cid, "param": "x", "keys": [{"t": 0, "value": 0, "ease": "bezier", "bezier": [1.5, 0, 0.7, 1]}]})
    with pytest.raises(CommandError):
        bus.execute("layer.add", {"source": "media"})
    for _ in range(5):
        bus.undo()
    assert bus.project.tracks == []
    bus.execute("layer.add", {"source": "adjustment", "start": 1, "duration": 2})
    clip = bus.project.tracks[0].clips[0]
    bus.execute("clip.trim", {"clip_id": clip.id, "out_point": 10})  # no media: any length
    assert bus.project.tracks[0].clips[0].out_point == 10


def test_v2_project_upgrades_to_v3():
    raw = {"schema_version": 2, "name": "old", "kind": "video",
           "tracks": [{"id": "t1", "kind": "video", "clips": [{"id": "c1", "media_id": "m1", "start": 0, "in_point": 0, "out_point": 2}]}]}
    p = Project.model_validate(migrate(raw))
    c = p.tracks[0].clips[0]
    assert p.schema_version == 3 and c.source == "media" and c.masks == [] and p.tracks[0].matte is None and not c.is_layered
