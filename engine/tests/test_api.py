from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aieditor_engine.api.server import create_app

TOKEN = "test-token"
H = {"X-Engine-Token": TOKEN}


@pytest.fixture
def client():
    with TestClient(create_app(TOKEN)) as c:
        yield c


def test_health_is_public(client: TestClient) -> None:
    assert client.get("/health").json()["ok"]


def test_token_required(client: TestClient) -> None:
    assert client.get("/commands").status_code == 401
    assert client.get("/commands", headers={"X-Engine-Token": "wrong"}).status_code == 401
    assert client.get("/commands", headers=H).status_code == 200
    assert client.get(f"/commands?token={TOKEN}").status_code == 200


def test_no_project_open(client: TestClient) -> None:
    assert client.get("/project", headers=H).status_code == 409


def test_full_flow(client: TestClient, tmp_path: Path, image_file: Path, video_file: Path) -> None:
    root = tmp_path / "proj"
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        # Socket connected before any project exists must still get events.
        assert ws.receive_json() == {"type": "hello", "revision": None}
        r = client.post("/projects", headers=H, json={"path": str(root), "name": "Demo"})
        assert r.status_code == 200, r.text
        assert r.json()["project"]["name"] == "Demo"
        assert ws.receive_json()["type"] == "project_opened"

        r = client.post(
            "/media/import", headers=H,
            json={"paths": [str(image_file), str(video_file), str(tmp_path / "nope.mp4")]},
        )
        body = r.json()
        assert len(body["imported"]) == 2
        assert len(body["failed"]) == 1
        image_id, video_id = body["imported"]
        assert ws.receive_json()["type"] == "command"  # media.add for the image

    engine = client.app.state.engine
    engine.session.wait_for_jobs(timeout=60)

    snap = client.get("/project", headers=H).json()
    media = snap["project"]["media"]
    assert media[video_id]["proxy_status"] == "ready"
    assert media[image_id]["proxy_status"] == "none"

    assert client.get(f"/media/{image_id}/thumb?token={TOKEN}").headers["content-type"] == "image/jpeg"
    assert client.get(f"/media/{video_id}/proxy", headers=H).status_code == 200
    assert client.get(f"/media/{image_id}/proxy", headers=H).status_code == 404

    track = client.post("/commands/track.add", headers=H, json={}).json()["result"]["track_id"]
    r = client.post(
        "/commands/clip.add", headers=H,
        json={"params": {"track_id": track, "media_id": video_id, "out_point": 1.0}, "source": "ai"},
    )
    assert r.status_code == 200, r.text
    bad = client.post("/commands/clip.add", headers=H, json={"params": {"track_id": "nope", "media_id": video_id}})
    assert bad.status_code == 422

    assert client.post("/undo", headers=H).json()["ok"]
    assert client.get("/project", headers=H).json()["project"]["tracks"][0]["clips"] == []
    assert client.post("/redo", headers=H).json()["ok"]

    # Autosaved to disk, reopenable.
    r = client.post("/projects/open", headers=H, json={"path": str(root)})
    assert r.status_code == 200
    assert len(r.json()["project"]["tracks"][0]["clips"]) == 1
    assert (root / "logs" / "trajectory.jsonl").read_text().count("\n") >= 4


def test_photo_editing_flow(client: TestClient, tmp_path: Path, image_file: Path) -> None:
    client.post("/projects", headers=H, json={"path": str(tmp_path / "p"), "name": "Photos", "kind": "photo"})
    media_id = client.post("/media/import", headers=H, json={"paths": [str(image_file)]}).json()["imported"][0]

    assert any(o["name"] == "adjust" for o in client.get("/ops", headers=H).json())
    assert len(client.get("/looks", headers=H).json()) >= 15

    r = client.post("/commands/photo.apply_look", headers=H, json={"params": {"media_id": media_id, "name": "noir", "strength": 0.8}})
    assert r.status_code == 200, r.text
    r = client.post(
        "/commands/photo.apply_program", headers=H,
        json={"params": {"media_id": media_id, "program": "look vintage_70s s=0.6\nvignette 0.3", "mode": "replace"}, "source": "ai"},
    )
    assert r.status_code == 200, r.text
    assert client.get(f"/media/{media_id}/dsl", headers=H).json()["program"] == "look vintage_70s s=0.6\nvignette 0.3"

    bad = client.post("/commands/photo.apply_program", headers=H, json={"params": {"media_id": media_id, "program": "sparkle 1"}})
    assert bad.status_code == 422 and "unknown op" in bad.json()["detail"]

    for q in ("", "&bypass=1", "&look=noir&s=0.5"):
        r = client.get(f"/media/{media_id}/render?token={TOKEN}&max=256{q}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"

    r = client.post(f"/media/{media_id}/preview", headers=H, json={"ops": [{"op": "adjust", "params": {"exposure": 1}}], "max": 200})
    assert r.status_code == 200 and r.content[:2] == bytes([0xFF, 0xD8])
    bad = client.post(f"/media/{media_id}/preview", headers=H, json={"ops": [{"op": "adjust", "params": {"exposure": 99}}]})
    assert bad.status_code == 422
    r = client.get(f"/media/{media_id}/render?token={TOKEN}&max=200&program=look%20noir")
    assert r.status_code == 200

    ai = client.post("/ai/edit", headers=H, json={"media_id": media_id, "prompt": "make it vintage"})
    assert ai.status_code == 200 and ai.json()["variations"]
    assert client.get("/ai/status", headers=H).json()["state"] in ("ready", "idle", "not_installed", "error")

    a = client.get(f"/media/{media_id}/analysis", headers=H).json()
    assert a["size"] == [640, 480] and "notes" in a

    out = client.post(f"/media/{media_id}/export", headers=H, json={"path": str(tmp_path / "export" / "final"), "format": "png"})
    assert out.status_code == 200, out.text
    assert Path(out.json()["path"]).suffix == ".png" and Path(out.json()["path"]).is_file()

    assert client.get("/media/nope/render", headers=H).status_code == 404
    client.post("/undo", headers=H)
    assert client.get(f"/media/{media_id}/dsl", headers=H).json()["program"] == "look noir s=0.8"


def test_open_missing(client: TestClient, tmp_path: Path) -> None:
    assert client.post("/projects/open", headers=H, json={"path": str(tmp_path / "x")}).status_code == 404
