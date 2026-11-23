"""Photo documents: layers over the edited photo (images, solids, masks,
blend modes), rendered the same in preview and export."""

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
from aieditor_engine.core.project import MediaItem, MediaKind, Project, ProjectKind

TOKEN = "test-token"
H = {"X-Engine-Token": TOKEN}


def _photo_project() -> Project:
    p = Project(name="P", kind=ProjectKind.PHOTO)
    p.media["m1"] = MediaItem(id="m1", path="a.png", name="a", kind=MediaKind.IMAGE, width=64, height=48)
    p.media["m2"] = MediaItem(id="m2", path="b.png", name="b", kind=MediaKind.IMAGE, width=32, height=32)
    return p


def test_layer_commands_route_into_the_photo_stack():
    bus = CommandBus(_photo_project())
    solid = bus.execute("layer.add", {"source": "solid", "color": "#ff0000", "doc": "m1"})
    img = bus.execute("layer.add", {"source": "media", "media_id": "m2", "doc": "m1"})
    extra = bus.execute("track.add", {"kind": "overlay", "doc": "m1"})
    p = lambda: bus.project  # noqa: E731
    assert p().tracks == [] and [t.id for t in p().layers["m1"]] == [solid["track_id"], img["track_id"], extra["track_id"]]
    _, clip = p().find_clip(img["clip_id"])
    assert clip.media_id == "m2" and clip.fit == "fit" and clip.transform.scale == 0.5
    # the ordinary layer commands find clips and tracks in a photo stack
    bus.execute("clip.set_layer", {"clip_id": solid["clip_id"], "blend": "multiply", "opacity": 0.5})
    bus.execute("track.set", {"track_id": extra["track_id"], "index": 0})
    assert p().layers["m1"][0].id == extra["track_id"] and p().tracks == []
    bus.execute("track.remove", {"track_id": extra["track_id"]})
    assert len(p().layers["m1"]) == 2
    with pytest.raises(CommandError):
        bus.execute("layer.add", {"source": "media", "media_id": "m2"})  # timeline media goes through clip.add
    with pytest.raises(CommandError):
        bus.execute("layer.add", {"source": "solid", "doc": "nope"})
    with pytest.raises(CommandError):
        bus.execute("track.add", {"kind": "audio", "doc": "m1"})
    bus.execute("media.remove", {"media_id": "m2"})
    assert all(c.media_id != "m2" for t in p().layers["m1"] for c in t.clips)
    bus.execute("media.remove", {"media_id": "m1"})
    assert "m1" not in p().layers
    for _ in range(9):
        bus.undo()
    assert bus.project.layers == {}


@pytest.fixture
def client():
    with TestClient(create_app(TOKEN)) as c:
        yield c


def _pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)


def test_layers_render_in_preview_and_export(client: TestClient, tmp_path: Path, image_file: Path) -> None:
    logo = tmp_path / "src" / "logo.png"
    Image.new("RGB", (100, 100), (20, 200, 60)).save(logo)
    client.post("/projects", headers=H, json={"path": str(tmp_path / "p"), "name": "Doc", "kind": "photo"})
    photo, other = client.post("/media/import", headers=H, json={"paths": [str(image_file), str(logo)]}).json()["imported"]
    run = lambda name, **params: client.post(f"/commands/{name}", headers=H, json={"params": params})  # noqa: E731

    plain = _pixels(client.get(f"/media/{photo}/render?token={TOKEN}&max=320").content)
    r = run("layer.add", source="media", media_id=other, doc=photo)
    assert r.status_code == 200, r.text
    layer = r.json()["result"]["clip_id"]
    run("clip.set_layer", clip_id=layer, x=0.25)  # right half
    shown = _pixels(client.get(f"/media/{photo}/render?token={TOKEN}&max=320").content)
    h, w = shown.shape[:2]
    assert (w, h) == (320, 240)
    # the green square sits at the centre-right; the left edge is still the photo
    assert np.abs(shown[h // 2, int(w * 0.75)] - [20, 200, 60]).max() < 12
    assert np.abs(shown[h // 2, 5] - plain[h // 2, 5]).max() < 6
    # before/after shows the photo alone
    before = _pixels(client.get(f"/media/{photo}/render?token={TOKEN}&max=320&bypass=1").content)
    assert np.abs(before[h // 2, int(w * 0.75)] - plain[h // 2, int(w * 0.75)]).max() < 6

    # a solid in multiply at half opacity over everything, cut to the left half by a mask
    r = run("layer.add", source="solid", color="#000000", doc=photo)
    solid = r.json()["result"]["clip_id"]
    run("clip.set_layer", clip_id=solid, opacity=0.5)
    mask = {"id": "m", "points": [{"x": 0, "y": 0}, {"x": 0.5, "y": 0}, {"x": 0.5, "y": 1}, {"x": 0, "y": 1}]}
    run("clip.set_masks", clip_id=solid, masks=[mask])
    out = client.post(f"/media/{photo}/export", headers=H, json={"path": str(tmp_path / "out.png"), "format": "png"})
    assert out.status_code == 200, out.text
    full = np.asarray(Image.open(tmp_path / "out.png").convert("RGB"), dtype=np.float32)
    assert full.shape[:2] == (480, 640)
    src = np.array([200, 120, 40], np.float32)
    assert (full[240, 20] < src * 0.85).all()  # darkened (left half, under the mask)
    assert np.abs(full[20, 620] - src).max() < 4  # untouched (right edge, outside mask and logo)
    assert np.abs(full[240, 480] - [20, 200, 60]).max() < 12  # the logo, full resolution
