"""GPU compositor parity: every frame the GPU renders must match the CPU
reference (video/compose.py) to within one 8-bit level, apart from a few
antialiased edge pixels. Skipped on machines without OpenGL 3.3."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from aieditor_engine.core.project import (
    BLEND_MODES,
    Clip,
    EditOp,
    Keyframe,
    Mask,
    MediaItem,
    MediaKind,
    Project,
    ProjectKind,
    Track,
    Transition,
    new_id,
)
from aieditor_engine.gpu import compositor as G
from aieditor_engine.video.compose import RenderSource, render_frame

pytestmark = pytest.mark.skipif(G.get() is None, reason="no OpenGL 3.3 GPU")

W, H = 256, 144


def _video(path: Path, hue: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h, fps = 320, 240, 25
    yy, xx = np.mgrid[0:h, 0:w]
    with av.open(str(path), "w") as out:
        v = out.add_stream("mpeg4", rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        for i in range(50):
            img = np.stack([(xx * 255 // w + hue) % 256, yy * 255 // h, np.full_like(xx, (hue + i * 3) % 256)], -1).astype(np.uint8)
            img[:, (i * 5) % w : (i * 5) % w + 16] = 240
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts, f.time_base = i, Fraction(1, fps)
            out.mux(v.encode(f))
        out.mux(v.encode(None))
    return path


@pytest.fixture
def media(tmp_path: Path) -> dict[str, MediaItem]:
    a, b = _video(tmp_path / "a.mp4", 0), _video(tmp_path / "b.mp4", 120)
    photo = tmp_path / "p.png"
    yy, xx = np.mgrid[0:300, 0:400]
    Image.fromarray(np.stack([xx * 255 // 400, yy * 255 // 300, 255 - xx * 255 // 400], -1).astype(np.uint8)).save(photo)
    return {
        "a": MediaItem(id="a", name="a", path=str(a), kind=MediaKind.VIDEO, width=320, height=240, duration=2.0, fps=25),
        "b": MediaItem(id="b", name="b", path=str(b), kind=MediaKind.VIDEO, width=320, height=240, duration=2.0, fps=25),
        "p": MediaItem(id="p", name="p", path=str(photo), kind=MediaKind.IMAGE, width=400, height=300),
    }


def _project(media: dict[str, MediaItem], *tracks: list[Clip], **settings) -> Project:
    p = Project(name="g", kind=ProjectKind.VIDEO, media=dict(media))
    p.settings.width, p.settings.height = W, H
    for k, v in settings.items():
        setattr(p.settings, k, v)
    for i, clips in enumerate(tracks):
        p.tracks.append(Track(id=new_id("trk"), kind="video" if i == 0 else "overlay", clips=clips))
    return p


def _clip(mid: str, start: float = 0.0, a: float = 0.0, b: float = 2.0, **kw) -> Clip:
    return Clip(id=new_id("clp"), media_id=mid, start=start, in_point=a, out_point=b, **kw)


def _same(p: Project, tmp_path: Path, times: tuple[float, ...] = (0.3, 1.0, 1.7), edge_share: float = 0.01) -> None:
    src = RenderSource(tmp_path)
    for t in times:
        gpu = render_frame(p, t, W, H, src, use_gpu=True).astype(int)
        cpu = render_frame(p, t, W, H, src, use_gpu=False).astype(int)
        d = np.abs(gpu - cpu).max(-1)
        assert (d > 1).mean() <= edge_share, (t, d.max(), (d > 1).mean())
        assert np.median(d) <= 1


def test_plain_video_and_photo_are_identical(media, tmp_path):
    _same(_project(media, [_clip("a")]), tmp_path, edge_share=0.0)
    _same(_project(media, [_clip("p", b=2.0)]), tmp_path, edge_share=0.0)


def test_transforms_crop_motion_keyframes(media, tmp_path):
    pip = _clip("b")
    pip.transform.scale, pip.transform.rotation, pip.transform.x = 0.45, 20, 0.2
    pip.crop.left = 0.1
    moving = _clip("p", motion="zoom_in")
    moving.keyframes["opacity"] = [Keyframe(t=0, value=0.2), Keyframe(t=2, value=1, ease="bezier", bezier=(0.3, 0, 0.7, 1))]
    _same(_project(media, [_clip("a")], [pip], [moving]), tmp_path)
    big = _clip("b")
    big.transform.scale, big.transform.anchor_x = 1.6, -0.2
    _same(_project(media, [_clip("a")], [big]), tmp_path)


@pytest.mark.parametrize("kind", ["crossfade", "fade_black", "fade_white", "wipe_left", "wipe_right", "slide_left", "slide_up", "zoom", "blur"])
def test_transitions(media, tmp_path, kind):
    first = _clip("a", 0, 0, 1.0)
    second = _clip("b", 1.0, 0.5, 2.0, transition=Transition(kind=kind, duration=0.8))
    _same(_project(media, [first, second]), tmp_path, times=(0.7, 0.9, 1.0, 1.1, 1.3), edge_share=0.02)


def test_blend_modes_masks_and_mattes_on_video(media, tmp_path):
    for mode in BLEND_MODES[::3]:
        top = _clip("b", opacity=0.8, blend=mode, masks=[Mask.ellipse(0.5, 0.5, 0.35, 0.4, feather=0.04)])
        _same(_project(media, [_clip("a")], [top]), tmp_path, times=(1.0,))
    p = _project(media, [_clip("a")], [_clip("p")], [_clip("b")])
    p.tracks[1].matte = "luma_inverted"  # the photo is cut by the brightness of clip b
    _same(p, tmp_path, times=(0.5, 1.5))


def test_grades_effects_and_adjustment_layers(media, tmp_path):
    look = _project(media, [_clip("a")], [_clip("b", opacity=0.5)])
    look.grade = [EditOp(id="g1", op="look", params={"name": "teal_orange_cinematic"})]
    _same(look, tmp_path)  # baked LUT on the GPU
    grain = look.model_copy(deep=True)
    grain.grade.append(EditOp(id="g2", op="grain", params={"amount": 0.3}))
    _same(grain, tmp_path)  # spatial effect: finished on the CPU
    fx = _project(media, [_clip("a", ops=[EditOp(id="o1", op="blur", params={"radius": 4})])])
    _same(fx, tmp_path)  # clip effects run on the CPU, then composite on the GPU
    adj = Clip(id=new_id("clp"), source="adjustment", start=0, in_point=0, out_point=2,
               ops=[EditOp(id="o2", op="black_white", params={})], masks=[Mask.rect(0, 0, 0.5, 1)])
    _same(_project(media, [_clip("a")], [adj]), tmp_path)


def test_grade_effects_run_on_the_gpu_or_fall_back(media, tmp_path):
    look = [EditOp(id="g1", op="look", params={"name": "teal_orange_cinematic"}), EditOp(id="g2", op="grain", params={"amount": 0.4})]
    plan = G.prepare(_project(media, [_clip("a")]), 1.0, W, H, RenderSource(tmp_path), 30, [o.model_dump(mode="json") for o in look])
    assert plan.post is not None and [k for k, *_ in plan.post] == [1, 2]  # vignette + grain in the shader
    bloom = _project(media, [_clip("a")])
    bloom.grade = [EditOp(id="g3", op="bloom", params={"amount": 0.5})]
    plan = G.prepare(bloom, 1.0, W, H, RenderSource(tmp_path), 30, [o.model_dump(mode="json") for o in bloom.grade])
    assert plan.post is None  # no GPU bloom yet: the CPU finishes the frame
    _same(bloom, tmp_path, times=(1.0,))


def test_letterbox_fades_and_end_card_still_apply(media, tmp_path):
    p = _project(media, [_clip("a")], letterbox=2.39, fade_in=0.5, fade_out=0.5, end_card="Bye")
    _same(p, tmp_path, times=(0.2, 1.0, 2.5))


def test_turning_the_gpu_off(monkeypatch):
    monkeypatch.setenv("FRAMEWISE_GPU", "0")
    assert G.get() is None


def test_shaders_build_for_webgl2_too():
    for name in ("layer", "composite", "transition", "matte", "finish"):
        src = G.shader_source(name + ".frag", "#version 300 es\nprecision highp float;\n")
        assert src.startswith("#version 300 es") and "void main()" in src and "// needs:" in src
