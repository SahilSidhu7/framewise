"""Phase D: automation engine, public /v1 API, webhooks, keys, CLI, watch folders, MCP."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import av
import pytest
from fastapi.testclient import TestClient

from aieditor_engine.agent import llm as llm_mod
from aieditor_engine.api.server import create_app
from aieditor_engine.automation import webhooks
from aieditor_engine.automation.jobs import JobError, JobManager, JobRequest
from aieditor_engine.automation.keys import KeyStore
from aieditor_engine.automation.mcp import McpServer
from aieditor_engine.automation.watch import FolderWatcher
from aieditor_engine.cli import main as cli_main
from aieditor_engine.session import Session

PHOTO_PROGRAM = "adjust exposure=0.3 contrast=10"


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Rules only: deterministic and fast."""
    monkeypatch.setattr(llm_mod.LLM, "available", lambda: False)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


class _Hook:
    """A tiny webhook receiver; the first `fail` calls answer 500."""

    def __init__(self, fail: int = 0) -> None:
        self.calls: list[tuple[dict, bytes]] = []
        self.fail = fail
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.calls.append(({k.lower(): v for k, v in self.headers.items()}, body))
                code = 500 if outer.fail > 0 else 200
                outer.fail -= 1
                self.send_response(code)
                self.end_headers()

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/hook"

    def wait(self, n: int = 1, timeout: float = 10) -> None:
        end = time.time() + timeout
        while len(self.calls) < n and time.time() < end:
            time.sleep(0.05)
        assert len(self.calls) >= n, "webhook not received"


def test_keys_are_hashed_scoped_and_revocable(home):
    (home.parent / "k").mkdir(parents=True, exist_ok=True)
    store = KeyStore(home.parent / "k")
    key, rec = store.create("n8n", ["jobs"])
    assert key.startswith("aie_") and "hash" not in rec
    raw = (home.parent / "k" / "keys.json").read_text()
    assert key not in raw  # only the hash is stored
    assert store.verify(key)["scopes"] == ["jobs"]
    assert store.verify(key + "x") is None and store.verify("nope") is None
    assert store.revoke(rec["id"]) and store.verify(key) is None
    with pytest.raises(ValueError):
        store.create("bad", ["root"])
    assert store.webhook_secret() == store.webhook_secret()


def test_webhook_signature_and_retry():
    body = b'{"a":1}'
    sig = webhooks.sign("s3cret", body)
    assert webhooks.verify_signature("s3cret", body, sig)
    assert not webhooks.verify_signature("other", body, sig)
    assert not webhooks.verify_signature("s3cret", body + b" ", sig)
    old = webhooks.sign("s3cret", body, int(time.time()) - 3600)
    assert not webhooks.verify_signature("s3cret", body, old)

    hook = _Hook(fail=1)
    results = []
    webhooks.deliver(hook.url, "s3cret", "job.done", {"job": {"id": "j_1"}}, results.append, delays=(0.05, 0.05)).join(5)
    assert len(hook.calls) == 2 and results[0]["delivered"] and results[0]["attempts"] == 2
    headers, raw = hook.calls[-1]
    assert headers["x-aieditor-event"] == "job.done" and headers["x-aieditor-job"] == "j_1"
    assert webhooks.verify_signature("s3cret", raw, headers["x-aieditor-signature"])


def _api(home: Path):
    m = JobManager(home, workers=1)
    key, _ = m.keys.create("n8n", ["jobs"])
    local, _ = m.keys.create("local", ["jobs", "local"])
    admin, _ = m.keys.create("admin", ["jobs", "admin"])
    return m, {"Authorization": f"Bearer {key}"}, {"X-API-Key": local}, {"Authorization": f"Bearer {admin}"}


def test_api_upload_job_wait_download_and_webhook(home, video_file, tmp_path):
    m, auth, _, admin = _api(home)
    hook = _Hook()
    with TestClient(create_app("tok", automation=m)) as c:
        assert c.get("/v1/me", headers=auth).json()["name"] == "n8n"
        assert {r["name"] for r in c.get("/v1/recipes", headers=auth).json()} >= {"reel", "captions", "photo_edit"}
        assert c.get("/v1/options", headers=auth).json()["video"]["styles"]
        up = c.put("/v1/files", params={"name": "clip.mp4"}, content=video_file.read_bytes(), headers=auth)
        assert up.status_code == 201, up.text
        fid = up.json()["file_id"]
        r = c.post("/v1/jobs", headers=auth, json={
            "recipe": "render", "inputs": [fid], "options": {"program": "aspect 1:1\nclip 1 0-1.5\nfade_out 0.3"},
            "output": {"quality": "draft"}, "webhook": {"url": hook.url, "secret": "hook-secret-1"},
        })
        assert r.status_code == 202, r.text
        jid = r.json()["id"]
        job = c.get(f"/v1/jobs/{jid}/wait", params={"timeout": 120}, headers=auth).json()
        assert job["state"] == "done", job
        assert job["outputs"][0]["name"] == "render.mp4"
        res = c.get(f"/v1/jobs/{jid}/result", headers=auth).json()
        f = c.get(res["outputs"][0]["url"], headers=auth)
        assert f.status_code == 200
        dst = tmp_path / "got.mp4"
        dst.write_bytes(f.content)
        with av.open(str(dst)) as cont:
            s = cont.streams.video[0]
            assert s.codec_context.width == s.codec_context.height
        hook.wait()
        headers, raw = hook.calls[0]
        assert webhooks.verify_signature("hook-secret-1", raw, headers["x-aieditor-signature"])
        assert json.loads(raw)["job"]["state"] == "done"
        # the desktop app's engine token works too, and admins see every job
        assert any(j["id"] == jid for j in c.get("/v1/jobs", headers={"X-Engine-Token": "tok"}).json())
        assert any(j["id"] == jid for j in c.get("/v1/jobs", headers=admin).json())
        assert c.delete(f"/v1/jobs/{jid}", headers=auth).json() == {"deleted": True}
        assert c.get(f"/v1/jobs/{jid}", headers=auth).status_code == 404


def test_api_auth_scopes_and_validation(home, image_file, tmp_path):
    m, auth, local, admin = _api(home)
    with TestClient(create_app("tok", automation=m)) as c:
        assert c.get("/v1/jobs").status_code == 401
        assert c.get("/v1/jobs", headers={"Authorization": "Bearer aie_dead_beef"}).status_code == 401
        assert c.get("/v1/keys", headers=auth).status_code == 403
        body = {"recipe": "photo_edit", "inputs": [str(image_file)]}
        assert c.post("/v1/jobs", headers=auth, json=body).status_code == 403  # local path needs `local`
        assert c.post("/v1/jobs", headers=local, json={**body, "output_dir": str(tmp_path)}).status_code == 202
        assert c.post("/v1/jobs", headers=local, json={**body, "recipe": "nope"}).status_code == 422
        assert c.post("/v1/jobs", headers=local, json={**body, "recipe": "video_edit"}).status_code == 422  # needs prompt
        assert c.post("/v1/jobs", headers=local, json={**body, "options": {"quality": 5}}).status_code == 422
        assert c.post("/v1/jobs", headers=auth, json={"recipe": "reel", "inputs": ["f_0123456789abcdef"]}).status_code == 404
        assert c.post("/v1/jobs", headers=auth, json={"recipe": "reel", "inputs": ["ftp://x/y.mp4"]}).status_code in (403, 422)
        # jobs are private to their key
        jid = c.post("/v1/jobs", headers=local, json=body).json()["id"]
        assert c.get(f"/v1/jobs/{jid}", headers=auth).status_code == 404
        assert c.get(f"/v1/jobs/{jid}", headers=admin).status_code == 200
        # admins manage keys
        new = c.post("/v1/keys", headers=admin, json={"name": "zapier"}).json()
        assert new["key"].startswith("aie_") and new["scopes"] == ["jobs"]
        assert c.get("/v1/me", headers={"X-API-Key": new["key"]}).status_code == 200
        assert c.delete(f"/v1/keys/{new['id']}", headers=admin).status_code == 204
        assert c.get("/v1/me", headers={"X-API-Key": new["key"]}).status_code == 401
        assert c.get("/v1/webhooks/secret", headers=admin).json()["secret"].startswith("whsec_")
        m.wait(jid, 60)


def test_reel_and_highlights_recipes(home, video_file):
    m = JobManager(home, start=False)
    job = m.run_now(JobRequest(recipe="reel", inputs=[str(video_file)], output={"platform": "tiktok", "quality": "draft"},
                               options={"count": 1}))
    assert job["state"] == "done", job["error"]
    names = [o["name"] for o in job["outputs"]]
    assert names == ["reel_1.mp4", "reels.json"]
    out = m.home / "jobs" / job["id"] / "out"
    with av.open(str(out / "reel_1.mp4")) as c:
        s = c.streams.video[0]
        assert s.codec_context.height > s.codec_context.width  # vertical
    reels = json.loads((out / "reels.json").read_text())
    assert reels[0]["program"].startswith("#") and "aspect 9:16" in reels[0]["program"]

    job = m.run_now(JobRequest(recipe="highlights", inputs=[str(video_file)], options={"count": 1, "length": 3},
                               output={"quality": "draft"}))
    assert job["state"] == "done", job["error"]
    moments = json.loads((m.home / "jobs" / job["id"] / "out" / "highlights.json").read_text())
    assert moments and moments[0]["file"] == "highlight_1.mp4" and moments[0]["end"] > moments[0]["start"]
    # analyses are shared across jobs: the second job reused the first one's
    assert list((m.home / "cache" / "analysis").glob("*.json"))


def test_photo_edit_render_and_shared_failures(home, image_file, video_file):
    m = JobManager(home, start=False)
    job = m.run_now(JobRequest(recipe="photo_edit", inputs=[str(image_file)], prompt="warmer and brighter",
                               style="food", output={"platform": "instagram_post", "versions": 2}))
    assert job["state"] == "done", job["error"]
    photos = [o for o in job["outputs"] if o["kind"] == "photo"]
    assert [p["name"] for p in photos] == ["photo_edit1.jpg", "photo_edit2.jpg"]
    edits = json.loads((m.home / "jobs" / job["id"] / "out" / "edits.json").read_text())
    assert any("crop" in e["program"] for e in edits)  # 4:5 output enforces the crop

    job = m.run_now(JobRequest(recipe="render", inputs=[str(image_file)], options={"program": PHOTO_PROGRAM, "format": "png"}))
    assert job["state"] == "done" and job["outputs"][0]["name"] == "photo_render.png"

    job = m.run_now(JobRequest(recipe="reel", inputs=[str(image_file)]))
    assert job["state"] == "failed" and "video" in job["error"]
    job = m.run_now(JobRequest(recipe="render", inputs=[str(video_file)], options={"program": "clip 9 0-1"}))
    assert job["state"] == "failed" and "program failed" in job["error"]
    with pytest.raises(JobError):
        m.submit(JobRequest(recipe="photo_edit", inputs=[str(image_file)]))  # local paths need allow_local


def test_captions_recipe_writes_srt_and_burns(home, video_file, monkeypatch):
    from aieditor_engine.agent import whisper

    words = [{"start": 0.1, "end": 0.5, "text": "Hello"}, {"start": 0.55, "end": 1.0, "text": "world."},
             {"start": 1.2, "end": 1.8, "text": "Again."}]
    fake = {"language": "en", "words": words, "segments": [{"start": 0.1, "end": 1.0, "text": "Hello world."},
                                                           {"start": 1.2, "end": 1.8, "text": "Again."}]}
    monkeypatch.setattr(whisper, "whisper_status", lambda: {"available": True})
    monkeypatch.setattr(Session, "transcript", lambda self, mid, compute=True: fake)
    m = JobManager(home, start=False)
    job = m.run_now(JobRequest(recipe="captions", inputs=[str(video_file)], options={"style": "boxed"},
                               output={"quality": "draft"}))
    assert job["state"] == "done", job["error"]
    out = m.home / "jobs" / job["id"] / "out"
    srt = (out / "clip.srt").read_text()
    assert srt.startswith("1\n00:00:00,100 --> 00:00:01,000\nHello world.")
    assert (out / "clip_captioned.mp4").stat().st_size > 1000

    monkeypatch.setattr(whisper, "whisper_status", lambda: {"available": False})
    job = m.run_now(JobRequest(recipe="captions", inputs=[str(video_file)]))
    assert job["state"] == "failed" and "whisper" in job["error"]


def test_jobs_survive_restart_and_cancel(home, image_file):
    m = JobManager(home, start=False)
    queued = m.submit(JobRequest(recipe="render", inputs=[str(image_file)], options={"program": PHOTO_PROGRAM}), allow_local=True)
    running = m.submit(JobRequest(recipe="render", inputs=[str(image_file)], options={"program": PHOTO_PROGRAM}), allow_local=True)
    m._jobs[running["id"]]["state"] = "running"
    m._save(m._jobs[running["id"]])

    m2 = JobManager(home, workers=1)  # restart
    assert m2.wait(queued["id"], 60)["state"] == "done"
    j = m2.get(running["id"])
    assert j["state"] == "failed" and "interrupted" in j["error"]

    m3 = JobManager(home, start=False)
    j = m3.submit(JobRequest(recipe="render", inputs=[str(image_file)], options={"program": PHOTO_PROGRAM}), allow_local=True)
    assert m3.cancel(j["id"])["state"] == "cancelled"
    with pytest.raises(JobError):
        m3.delete(m3.submit(JobRequest(recipe="render", inputs=[str(image_file)], options={"program": PHOTO_PROGRAM}),
                            allow_local=True)["id"])


def test_watch_folder_processes_new_files_once(home, image_file, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "aieditor-watch.yaml").write_text(f"recipe: render\noptions: {{program: '{PHOTO_PROGRAM}'}}\n")
    m = JobManager(home, start=False)
    w = FolderWatcher(m, inbox)
    (inbox / "a.png").write_bytes(image_file.read_bytes())
    (inbox / "notes.txt").write_text("ignored")
    assert w.scan_once() == []  # first sighting: wait until the file stops growing
    [jid] = w.scan_once()
    m._run(jid)
    w.scan_once()
    assert (inbox / "output" / "a" / "a_render.jpg").is_file()
    assert w.scan_once() == [] and FolderWatcher(m, inbox).scan_once() == []  # remembered across restarts


def test_mcp_server_lists_and_runs(home, image_file):
    m = JobManager(home, workers=1)
    srv = McpServer(m)
    init = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    assert init["result"]["serverInfo"]["name"] == "aieditor"
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert {t["name"] for t in tools} == {"list_recipes", "list_styles", "run_recipe", "get_job", "cancel_job"}
    res = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "run_recipe", "arguments": {
        "recipe": "render_program", "files": [str(image_file)], "options": {"program": PHOTO_PROGRAM}}}})["result"]
    job = json.loads(res["content"][0]["text"])
    assert not res["isError"] and job["state"] == "done" and Path(job["outputs"][0]["path"]).is_file()
    bad = srv.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "run_recipe", "arguments": {
        "recipe": "nope", "files": [str(image_file)]}}})["result"]
    assert bad["isError"]
    assert srv.handle({"jsonrpc": "2.0", "id": 5, "method": "bogus"})["error"]["code"] == -32601
    m.shutdown()


def test_cli_one_shot_and_keys(home, image_file, tmp_path, capsys):
    out = tmp_path / "results"
    code = cli_main(["--home", str(home), "render", str(image_file), "--set", f"program={PHOTO_PROGRAM}", "--out", str(out), "--quiet"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["state"] == "done"
    assert (out / "photo_render.jpg").is_file() and summary["files"] == [str(out / "photo_render.jpg")]
    assert not list((home / "jobs").iterdir())  # one-shot jobs clean up unless --keep

    assert cli_main(["--home", str(home), "keys", "create", "ci", "--scope", "jobs", "--scope", "local"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["scopes"] == ["jobs", "local"] and KeyStore(home).verify(rec["key"])
    assert cli_main(["--home", str(home), "keys", "revoke", rec["id"]]) == 0
    assert cli_main(["--home", str(home), "reel", str(image_file), "--out", str(out), "--quiet"]) == 1
