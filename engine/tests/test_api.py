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


def test_open_missing(client: TestClient, tmp_path: Path) -> None:
    assert client.post("/projects/open", headers=H, json={"path": str(tmp_path / "x")}).status_code == 404
