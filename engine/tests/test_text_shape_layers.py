"""Text and shape layers: drawn in layer space with their own alpha, then
placed, masked and blended like any layer; the same in preview and export."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aieditor_engine.api.server import create_app
from aieditor_engine.core import builtin, video_commands  # noqa: F401  (register commands)
from aieditor_engine.core.bus import CommandBus
from aieditor_engine.core.commands import CommandError
from aieditor_engine.core.project import Clip, Project, ProjectKind, ShapeStyle, TextStyle, Track, new_id
from aieditor_engine.video.compose import RenderSource, render_frame
from aieditor_engine.video.vector import vector_bounds, vector_raster

W, H = 320, 180


def _project(*clips: Clip) -> Project:
    p = Project(name="T", kind=ProjectKind.VIDEO)
    p.settings.width, p.settings.height = W, H
    bg = Clip(id=new_id("clp"), source="solid", color="#000000", start=0, in_point=0, out_point=4)
    p.tracks.append(Track(id=new_id("trk"), kind="video", clips=[bg]))
    for c in clips:
        p.tracks.append(Track(id=new_id("trk"), kind="overlay", clips=[c]))
    return p


def _shape(**kw) -> Clip:
    return Clip(id=new_id("clp"), source="shape", start=0, in_point=0, out_point=4, shape=ShapeStyle(**kw))


def _text(**kw) -> Clip:
    return Clip(id=new_id("clp"), source="text", start=0, in_point=0, out_point=4, text=TextStyle(**kw))


def _frame(p: Project, tmp_path: Path) -> np.ndarray:
    return render_frame(p, 1.0, W, H, RenderSource(tmp_path), use_gpu=False).astype(np.float32)


def test_shape_rectangle_ellipse_and_outline(tmp_path):
    rect = _shape(width=0.5, height=0.5, fill="#ff0000")
    f = _frame(_project(rect), tmp_path)
    assert np.abs(f[90, 160] - [255, 0, 0]).max() <= 2  # centre
    assert f[10, 10].max() <= 2  # outside: background shows
    assert np.abs(f[90, 82] - [255, 0, 0]).max() <= 2 and f[90, 76].max() <= 2  # edge at x = 80

    ell = _shape(kind="ellipse", width=0.5, height=0.5, fill="#00ff00")
    f = _frame(_project(ell), tmp_path)
    assert f[90, 160, 1] > 250 and f[50, 85].max() <= 2  # box corner is outside the ellipse

    ring = _shape(width=0.5, height=0.5, fill=None, stroke_color="#ffffff", stroke_width=0.05)
    f = _frame(_project(ring), tmp_path)
    assert f[90, 160].max() <= 2 and f[90, 84].min() > 250  # hollow, with a 9 px outline inside the edge


def test_shape_moves_scales_and_is_masked_like_a_layer(tmp_path):
    c = _shape(width=0.25, height=0.25, fill="#ffffff")
    c.transform.x, c.transform.scale = 0.25, 2.0
    f = _frame(_project(c), tmp_path)
    # 80 x 45 px box doubled about the canvas centre, then moved right by 80 px: x 160..320
    assert f[90, 170].min() > 250 and f[90, 150].max() <= 2
    c.blend, c.opacity = "multiply", 1.0
    f = _frame(_project(c), tmp_path)
    assert f[90, 200].max() <= 2  # white multiplied over black


def test_text_draws_inside_its_bounds_with_an_outline(tmp_path):
    c = _text(text="HELLO\nWORLD", size=0.15, color="#ffff00", stroke_width=0.1, stroke_color="#0000ff")
    rgb, alpha = vector_raster(c, W, H)
    x0, y0, x1, y1 = vector_bounds(c, W, H)
    assert 0.2 < x0 < 0.5 < x1 < 0.8 and 0.1 < y0 < 0.5 < y1 < 0.9
    ys, xs = np.nonzero(alpha > 0.5)
    assert xs.min() >= x0 * W - 2 and xs.max() <= x1 * W + 2 and ys.min() >= y0 * H - 2 and ys.max() <= y1 * H + 2
    f = _frame(_project(c), tmp_path)
    inside = alpha > 0.99
    lit = f[inside]
    assert (lit[:, 0] > 200).any() and (lit[:, 2] > 200).any()  # yellow fill and blue outline both drawn
    assert f[~(alpha > 0)].max() <= 2  # nothing outside the glyphs
    # size is relative: twice the canvas -> twice the box
    b2 = vector_bounds(c, W * 2, H * 2)
    assert abs((b2[2] - b2[0]) - (x1 - x0)) < 0.02


def test_commands():
    bus = CommandBus(Project(name="C", kind=ProjectKind.VIDEO))
    t = bus.execute("layer.add", {"source": "text", "text": "Hi there"})
    s = bus.execute("layer.add", {"source": "shape", "shape": "ellipse", "color": "#ff0000"})
    _, tc = bus.project.find_clip(t["clip_id"])
    assert tc.text.text == "Hi there" and tc.text.color == "#ffffff" and bus.project.find_track(t["track_id"]).name == "Hi there"
    bus.execute("clip.set_text", {"clip_id": t["clip_id"], "size": 0.2, "font": "serif", "align": "left"})
    _, tc = bus.project.find_clip(t["clip_id"])
    assert tc.text.size == 0.2 and tc.text.font == "serif" and tc.text.text == "Hi there"
    bus.execute("clip.set_shape", {"clip_id": s["clip_id"], "fill": "none", "stroke_width": 0.02})
    _, sc = bus.project.find_clip(s["clip_id"])
    assert sc.shape.kind == "ellipse" and sc.shape.fill is None
    with pytest.raises(CommandError):
        bus.execute("clip.set_shape", {"clip_id": s["clip_id"], "stroke_width": 0})  # nothing left to draw
    with pytest.raises(CommandError):
        bus.execute("clip.set_text", {"clip_id": s["clip_id"], "text": "x"})
    bus.undo()
    _, sc = bus.project.find_clip(s["clip_id"])
    assert sc.shape.fill == "#ff0000"


def test_photo_text_layer_and_raster_endpoints(tmp_path: Path, image_file: Path) -> None:
    token = "t"
    hdr = {"X-Engine-Token": token}
    with TestClient(create_app(token)) as client:
        client.post("/projects", headers=hdr, json={"path": str(tmp_path / "p"), "name": "Doc", "kind": "photo"})
        photo = client.post("/media/import", headers=hdr, json={"paths": [str(image_file)]}).json()["imported"][0]
        r = client.post("/commands/layer.add", headers=hdr, json={"params": {"source": "text", "text": "SALE", "doc": photo}})
        cid = r.json()["result"]["clip_id"]
        client.post("/commands/clip.set_text", headers=hdr, json={"params": {"clip_id": cid, "size": 0.3, "color": "#000000"}})
        shown = np.asarray(Image.open(io.BytesIO(client.get(f"/media/{photo}/render?token={token}&max=320").content)).convert("RGB"), np.float32)
        assert shown[120, 160].max() < 60 or shown[110:130, 140:180].min() < 60  # black letters in the middle
        assert np.abs(shown[10, 10] - [200, 120, 40]).max() < 6
        png = client.get(f"/layers/{cid}/raster?token={token}&w=320&h=240")
        assert png.status_code == 200 and png.headers["content-type"] == "image/png"
        rgba = np.asarray(Image.open(io.BytesIO(png.content)))
        assert rgba.shape == (240, 320, 4) and rgba[..., 3].max() == 255 and rgba[0, 0, 3] == 0
        b = client.get(f"/layers/{cid}/bounds", headers=hdr, params={"w": 320, "h": 240}).json()["bounds"]
        assert 0 < b[0] < 0.5 < b[2] < 1
        assert client.get("/layers/nope/bounds", headers=hdr, params={"w": 2, "h": 2}).status_code == 404
