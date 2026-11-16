"""Reel finishing: word-timed caption highlight, caption placement off faces / burned-in text,
progress bar and end card."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from aieditor_engine.core.project import Caption, MediaItem, MediaKind, Project, ProjectKind
from aieditor_engine.video import placement as P
from aieditor_engine.video.captions import group_words_timed
from aieditor_engine.video.compose import RenderSource, render_frame
from aieditor_engine.video.export import ExportSpec, export_video
from aieditor_engine.video.text import STYLES, caption_box, draw_text
from aieditor_engine.video.vdsl import apply_video_program, timeline_to_dsl

from test_video import _video

YELLOW = np.array(STYLES["bold_pop"].highlight)


def _yellow(frame: np.ndarray) -> int:
    return int((np.abs(frame.astype(int) - YELLOW).sum(axis=2) < 60).sum())


def test_word_times_and_highlight() -> None:
    words = [{"start": 1.0, "end": 1.3, "text": "hello"}, {"start": 1.4, "end": 1.8, "text": "big"},
             {"start": 1.9, "end": 2.4, "text": "world."}]
    (s, e, text, rel), = group_words_timed(words, "bold_pop")
    assert (s, text, rel) == (1.0, "hello big world.", [0.0, 0.4, 0.9])

    def frame(age: float, style: str = "bold_pop", w: list[float] | None = rel) -> np.ndarray:
        f = np.full((640, 360, 3), 60, np.uint8)
        draw_text(f, text, style, None, age, 2.0, words=w)
        return f

    plain = frame(0.5, w=None)
    assert _yellow(plain) == 0 and plain.max() > 200  # white text, no highlight without word times
    first, second = frame(0.2), frame(0.5)
    assert _yellow(first) > 20 and _yellow(second) > 20
    ys, xs = np.nonzero(np.abs(first.astype(int) - YELLOW).sum(axis=2) < 60)
    ys2, xs2 = np.nonzero(np.abs(second.astype(int) - YELLOW).sum(axis=2) < 60)
    assert xs.mean() < xs2.mean()  # the highlight moves to the next word
    # same layout with or without highlight (no jitter): non-highlighted pixels match
    mask = (np.abs(first.astype(int) - YELLOW).sum(axis=2) >= 60) & (np.abs(second.astype(int) - YELLOW).sum(axis=2) >= 60)
    assert np.abs(first[mask].astype(int) - second[mask].astype(int)).mean() < 3
    assert _yellow(frame(0.5, style="clean")) == 0  # styles without a highlight colour ignore word times


def _portrait_face(face_y: float, title: bool = False) -> np.ndarray:
    """A grey 9:16 frame with a skin-coloured head (and optionally a burned-in white name title)."""
    img = Image.new("RGB", (360, 640), (70, 75, 80))
    d = ImageDraw.Draw(img)
    cy = int(640 * face_y)
    d.ellipse((120, cy - 85, 240, cy + 85), fill=(224, 172, 140))
    d.rectangle((150, cy + 70, 210, cy + 140), fill=(214, 160, 128))  # neck
    if title:
        d.text((40, 470), "DR. JANE EXAMPLE  DIRECTOR OF THINGS", fill=(255, 255, 255))
        d.text((40, 490), "DR. JANE EXAMPLE  DIRECTOR OF THINGS", fill=(255, 255, 255))
    return np.asarray(img)


def test_detectors() -> None:
    frame = _portrait_face(0.55)
    (x0, y0, x1, y1), = P.faces(frame)
    assert 0.3 < x0 < 0.4 and 0.6 < x1 < 0.72
    assert 0.38 < y0 < 0.45 and 0.62 < y1 < 0.78  # head only, not the neck down to 0.77+
    assert P._boxes(P.text_mask(frame)) == []
    boxes = P._boxes(P.text_mask(_portrait_face(0.3, title=True)))
    assert boxes and all(0.7 < b[1] and b[3] < 0.8 for b in boxes)
    hands = np.asarray(Image.new("RGB", (360, 640), (70, 75, 80)))
    hands = hands.copy()
    hands[520:600, 100:200] = (224, 172, 140)  # skin low in the frame: hands, not a head
    assert P.faces(hands) == []


def test_choose_moves_off_face_and_text() -> None:
    cap = Caption(id="c", start=0, end=1, text="hello world", style="bold_pop")
    w, h = 1080, 1920
    own = caption_box(cap.text, cap.style, None, w, h)
    face = ((0.3, own[1] / h - 0.05, 0.7, own[3] / h + 0.02), P.FACE_WEIGHT)
    assert P.choose(cap, w, h, [], None)[0] == "center_low"
    assert P.choose(cap, w, h, [face], None)[0] != "center_low"
    title = ((0.05, 0.7, 0.95, 0.8), P.TEXT_WEIGHT)
    pos, cost = P.choose(cap, w, h, [face, title], None)
    assert pos == "high" and cost < 0.5
    # stickiness: an almost free spot does not beat staying where the last caption was
    small = ((0.45, own[1] / h, 0.47, own[1] / h + 0.01), P.TEXT_WEIGHT)
    assert P.choose(cap, w, h, [small], "center_low")[0] == "center_low"


def _face_video(path: Path) -> Path:
    frame = _portrait_face(0.55)
    with av.open(str(path), "w") as out:
        v = out.add_stream("mpeg4", rate=10)
        v.width, v.height, v.pix_fmt = 360, 640, "yuv420p"
        for i in range(20):
            f = av.VideoFrame.from_ndarray(frame, format="rgb24")
            f.pts = i
            out.mux(v.encode(f))
        out.mux(v.encode(None))
    return path


def _face_project(tmp_path: Path) -> Project:
    p = Project(name="r", kind=ProjectKind.VIDEO)
    p.media["v"] = MediaItem(id="v", name="v", path=str(_face_video(tmp_path / "face.mp4")), kind=MediaKind.VIDEO,
                             width=360, height=640, duration=2.0, fps=10)
    apply_video_program(p, "aspect 9:16\nclip 1")
    p.captions = [Caption(id="a", start=0.1, end=0.9, text="right on the chin", style="bold_pop"),
                  Caption(id="b", start=1.0, end=1.9, text="hand placed", style="bold_pop", position="center_low")]
    return p


def test_place_captions_and_export_copy(tmp_path: Path) -> None:
    p = _face_project(tmp_path)
    res = P.place_captions(p.model_copy(deep=True), RenderSource(tmp_path))
    assert res == {"checked": 1, "moved": 1}  # the hand-placed caption is left alone
    q = p.model_copy(deep=True)
    P.place_captions(q, RenderSource(tmp_path))
    assert q.captions[0].position in ("bottom", "high") and q.captions[0].auto_position
    assert q.captions[1].position == "center_low" and not q.captions[1].auto_position

    out = tmp_path / "o.mp4"
    export_video(p, tmp_path, ExportSpec(out, 360, 640, 10, "small", audio=False, workers=1))
    assert p.captions[0].position is None  # the caller's project is not changed
    with av.open(str(out)) as c:
        f = next(f for f in c.decode(video=0) if f.time >= 0.5).to_ndarray(format="rgb24")
    lo, hi = caption_box("right on the chin", "bold_pop", "center_low", 360, 640)[1::2]
    chin = f[lo:hi, 120:240].astype(int)
    assert np.abs(chin - np.array([224, 172, 140])).sum(axis=2).mean() < 120  # the chin is still visible there


def test_progress_bar_end_card_and_dsl(tmp_path: Path) -> None:
    clip = _video(tmp_path / "m" / "grey.mp4", (90, 90, 90), tone=None)
    p = Project(name="e", kind=ProjectKind.VIDEO)
    p.media["g"] = MediaItem(id="g", name="g", path=str(clip), kind=MediaKind.VIDEO, width=320, height=240, duration=2.0, fps=25)
    apply_video_program(p, 'aspect 16:9\nclip 1\nprogress_bar top\nend_card "Follow for part 2" dur=1\ntitle "Hi" at=0 dur=1 pos=high')
    assert p.content_end == 2.0 and p.duration == 3.0
    dsl = timeline_to_dsl(p)
    assert "progress_bar top" in dsl and 'end_card "Follow for part 2" dur=1' in dsl and "pos=high" in dsl
    src = RenderSource(tmp_path)
    early, late = render_frame(p, 0.5, 320, 180, src), render_frame(p, 1.5, 320, 180, src)
    bar_h = 3
    assert early[:bar_h, :60].mean() > 200 and early[:bar_h, 200:].mean() < 80  # 25% filled
    assert late[:bar_h, :200].mean() > 200
    card = render_frame(p, 2.6, 320, 180, src)
    assert card[:bar_h].mean() < 120  # no bar on the end card
    assert card[60:120].max() > 200 and card.mean() < 90  # bright text on a darkened background
    apply_video_program(p, "progress_bar off\nend_card off", replace=False)
    assert p.duration == 2.0 and "progress_bar" not in timeline_to_dsl(p)


def test_caption_set_position_and_words() -> None:
    from aieditor_engine.core.bus import CommandBus

    p = Project(name="c", kind=ProjectKind.VIDEO)
    p.captions = [Caption(id="a", start=0, end=1, text="two words", words=[0.0, 0.4], position="high", auto_position=True)]
    bus = CommandBus(p)
    bus.execute("caption.set", {"caption_id": "a", "style": "boxed"})
    cap = bus.project.captions[0]
    assert (cap.position, cap.auto_position, cap.words) == ("high", True, [0.0, 0.4])  # untouched
    bus.execute("caption.set", {"caption_id": "a", "position": "top"})
    assert (bus.project.captions[0].position, bus.project.captions[0].auto_position) == ("top", False)
    bus.execute("caption.set", {"caption_id": "a", "position": None})
    assert bus.project.captions[0].position is None  # back to auto
    bus.execute("caption.set", {"caption_id": "a", "text": "now three words"})
    assert bus.project.captions[0].words is None
