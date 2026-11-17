"""Subject selection: auto and click-to-pick masks, the subject edit, video tracking, API, planner words."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from aieditor_engine.agent.planner import plan_rules
from aieditor_engine.api.server import create_app
from aieditor_engine.imaging.ops import OPS, RenderContext
from aieditor_engine.imaging.subject import mask_box, parse_point, subject_mask
from aieditor_engine.video.track import track_subject
from aieditor_engine.video.vdsl import VideoDslError, apply_video_program


def scene(w: int = 480, h: int = 320, cx: float = 0.62, cy: float = 0.55, r: float = 0.18) -> np.ndarray:
    """Soft blurry background (gradient + low-contrast blobs) and one sharp, colourful round subject."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([90 + 40 * xx / w, 110 + 20 * yy / h, 120 + 0 * xx], axis=2)
    img = Image.fromarray(bg.astype(np.uint8))
    d = ImageDraw.Draw(img)
    rr = int(min(w, h) * r)
    x, y = int(cx * w), int(cy * h)
    d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(220, 60, 40))
    for k in range(6):  # texture on the subject so it is "in focus"
        d.line((x - rr + 4, y - rr + 10 + k * rr // 3, x + rr - 4, y - rr + 10 + k * rr // 3), fill=(250, 220, 60), width=2)
    return np.asarray(img)


def test_parse_point() -> None:
    assert parse_point("auto") is None and parse_point(None) is None
    assert parse_point("0.25,0.75") == (0.25, 0.75)
    assert parse_point("1.4,-1") == (1.0, 0.0)
    with pytest.raises(ValueError):
        parse_point("left")


def test_auto_and_point_masks_find_the_subject() -> None:
    img = scene()
    for point in (None, (0.62, 0.55)):
        m = subject_mask(img, point)
        assert m.shape == img.shape[:2] and m.dtype == np.float32
        x0, y0, x1, y1 = mask_box(m)  # type: ignore[misc]
        assert x0 < 0.62 < x1 and y0 < 0.55 < y1
        assert 0.03 < (m > 0.5).mean() < 0.35
        assert m[int(0.55 * 320), int(0.62 * 480)] > 0.9 and m[10, 10] < 0.1


def test_subject_op_blurs_and_dims_only_the_background() -> None:
    img = scene().astype(np.float32) / 255
    lin = img ** 2.2
    spec = OPS["subject"]
    p = spec.params_model(bg_blur=30, bg_light=-1.0, point="0.62,0.55")
    out = spec.fn(lin, p, RenderContext())
    sub = (slice(150, 200), slice(280, 320))  # inside the subject
    bgr = (slice(10, 60), slice(10, 60))      # background corner
    assert np.abs(out[sub] - lin[sub]).mean() < 0.03
    assert out[bgr].mean() < lin[bgr].mean() * 0.6
    noop = spec.params_model()
    assert spec.fn(lin, noop, RenderContext()) is lin
    with pytest.raises(ValueError):
        spec.params_model(point="middle")


def _moving_video(path: Path, frames: int = 40, fps: int = 10) -> Path:
    with av.open(str(path), "w") as out:
        v = out.add_stream("mpeg4", rate=fps)
        v.width, v.height, v.pix_fmt = 480, 320, "yuv420p"
        v.bit_rate = 3_000_000
        for i in range(frames):
            img = scene(cx=0.3 + 0.4 * i / (frames - 1))
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts, f.time_base = i, Fraction(1, fps)
            out.mux(v.encode(f))
        out.mux(v.encode(None))
    return path


def test_track_follows_a_moving_subject(tmp_path: Path) -> None:
    path = _moving_video(tmp_path / "move.mp4")
    tr = track_subject(path, t=0.0, x=0.3, y=0.55)
    assert tr["method"] == "manual" and len(tr["t"]) == len(tr["box"]) >= 20
    first, last = tr["box"][0], tr["box"][-1]
    cx0, cx1 = (first[0] + first[2]) / 2, (last[0] + last[2]) / 2
    assert abs(cx0 - 0.3) < 0.08 and abs(cx1 - 0.7) < 0.1  # followed left -> right
    assert tr["x"][-1] > tr["x"][0] + 0.25


def test_api_pick_follow_and_clear(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "m" / "move.mp4") if (tmp_path / "m").mkdir() is None else None
    photo = tmp_path / "m" / "photo.png"
    Image.fromarray(scene()).save(photo)
    c = TestClient(create_app(token="t"))
    h = {"X-Engine-Token": "t"}
    assert c.post("/projects", headers=h, json={"path": str(tmp_path / "p"), "name": "s", "kind": "video"}).status_code == 200
    ids = c.post("/media/import", headers=h, json={"paths": [str(video), str(photo)]}).json()["imported"]
    vid, pid = ids
    r = c.get(f"/media/{pid}/subject", headers=h, params={"point": "0.62,0.55"}).json()
    assert r["kind"] == "photo" and r["outline"] and 0.02 < r["coverage"] < 0.4
    fr = c.get(f"/media/{vid}/frame", headers=h, params={"t": 2.0, "max": 320})
    assert fr.status_code == 200 and fr.headers["content-type"] == "image/jpeg" and len(fr.content) > 1000
    r = c.post(f"/media/{vid}/subject", headers=h, json={"x": 0.3, "y": 0.55, "t": 0}).json()
    assert r["kind"] == "video" and r["pick"]["x"] == 0.3 and r["track"] and r["box"]
    c.post("/commands/project.set_settings", headers=h, json={"params": {"width": 1080, "height": 1920}})
    track = c.post("/commands/track.add", headers=h, json={"params": {"kind": "video"}}).json()["result"]["track_id"]
    clip = c.post("/commands/clip.add", headers=h, json={"params": {"track_id": track, "media_id": vid}}).json()["result"]["clip_id"]
    r = c.post(f"/clips/{clip}/follow_subject", headers=h)
    assert r.status_code == 200 and r.json()["method"] == "manual" and r.json()["keys"] >= 2
    xs = [k["value"] for k in c.get("/project", headers=h).json()["project"]["tracks"][0]["clips"][0]["keyframes"]["x"]]
    assert xs[0] > xs[-1]  # subject moves right -> picture shifts left
    assert c.post("/undo", headers=h).json()["ok"]
    assert c.delete(f"/media/{vid}/subject", headers=h).json()["pick"] is None


def test_planner_words_and_video_guard() -> None:
    assert "subject bg_blur" in plan_rules("blur the background", {"notes": []}, "", 1)[0].program
    splash = plan_rules("colour splash, background black and white", {"notes": []}, "", 1)[0].program
    assert "bg_saturation=-100" in splash and "bw_" not in splash
    from aieditor_engine.core.project import Project, ProjectKind

    with pytest.raises(VideoDslError, match="for photos"):
        apply_video_program(Project(name="v", kind=ProjectKind.VIDEO), "subject bg_blur=10")


@pytest.fixture(autouse=True)
def _grabcut_only(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these tests deterministic even when the optional model is installed on the machine."""
    if "uses_model" not in request.keywords:
        from aieditor_engine.imaging import segment

        monkeypatch.setattr(segment, "available", lambda: False)


def test_model_mask_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a model: a click keeps the clicked object; auto keeps the main subject; a click elsewhere falls back."""
    from aieditor_engine.imaging import segment, subject

    h, w = 300, 400
    prob = np.zeros((h, w), np.float32)
    prob[60:240, 40:200] = 0.95   # big object (left)
    prob[100:160, 280:340] = 0.9  # small object (right)
    monkeypatch.setattr(segment, "subject_probability", lambda img: prob)
    img = np.full((h, w, 3), 120, np.uint8)
    auto = subject._model_mask(img, None)
    assert auto[150, 100] > 0.9 and auto[130, 310] < 0.1  # small one dropped (< 35% of the main)
    right = subject._model_mask(img, (310 / w, 130 / h))
    assert right[130, 310] > 0.85 and right[150, 100] < 0.1
    assert subject._model_mask(img, (0.95, 0.95)) is None  # nothing there: GrabCut takes over
    monkeypatch.setattr(segment, "subject_probability", lambda img: np.zeros((h, w), np.float32))
    assert subject._model_mask(img, None) is None


@pytest.mark.uses_model
@pytest.mark.skipif(__import__("aieditor_engine.models", fromlist=["path_of"]).path_of("subject") is None,
                    reason="subject model not installed (framewise models install subject)")
def test_real_model_finds_the_subject() -> None:
    img = scene(640, 427)
    m = subject_mask(img, None)
    x0, y0, x1, y1 = mask_box(m)  # type: ignore[misc]
    assert x0 < 0.62 < x1 and y0 < 0.55 < y1 and m[10, 10] < 0.2


def test_model_routes() -> None:
    c = TestClient(create_app(token="t"))
    h = {"X-Engine-Token": "t"}
    models = c.get("/models", headers=h).json()
    sub = next(m for m in models if m["name"] == "subject")
    assert sub["licence"] == "Apache-2.0" and sub["size_mb"] > 100 and sub["md5"] and "installed" in sub and "fits" in sub
    hq = next(m for m in models if m["name"] == "subject-hq")
    assert hq["licence"] == "MIT" and hq["min_memory_gb"] >= 12
    assert c.post("/models/nope/install", headers=h).status_code == 404
