"""Job manager: queue, run, persist and report automation jobs.

Layout under the automation home:
    keys.json, webhook_secret
    uploads/<file_id>/<name>             files sent with PUT /v1/files
    cache/                               analyses + transcripts shared by all jobs
    jobs/<job_id>/job.json               state (survives restarts)
                 /in/                    inputs downloaded from URLs
                 /project/               the throwaway project the recipe edits
                 /out/                   results (what the API serves)

Jobs run one at a time by default (a render already uses every core). Jobs
that were queued when the engine stopped are queued again on start; a job that
was running is marked failed. Finished jobs and uploads are deleted after
`keep_days`.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import secrets
import shutil
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ..agent.options import OutputOptions
from . import recipes as R
from .home import automation_home
from .keys import KeyStore
from .webhooks import deliver

log = logging.getLogger(__name__)

MAX_DOWNLOAD = 4 * 1024**3
FINAL = ("done", "failed", "cancelled")
_SAFE = re.compile(r"[^\w.\- ]+")


class JobError(ValueError):
    """The request cannot be accepted (bad input reference, missing scope...)."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def safe_name(name: str, default: str = "file") -> str:
    name = _SAFE.sub("_", Path(name.replace("\\", "/")).name).strip(" .")
    return name[:120] or default


class InputRef(BaseModel):
    """One input file: an uploaded file id, an http(s) URL, or (scope `local`) a path on this computer."""

    file_id: str | None = None
    url: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def _one(self) -> InputRef:
        if sum(x is not None for x in (self.file_id, self.url, self.path)) != 1:
            raise ValueError("give exactly one of file_id, url, path")
        return self


class WebhookSpec(BaseModel):
    url: str = Field(description="Called with POST when the job finishes")
    secret: str | None = Field(None, min_length=8, description="HMAC secret; default is the engine's webhook secret")

    @field_validator("url")
    @classmethod
    def _http(cls, v: str) -> str:
        if urllib.parse.urlsplit(v).scheme not in ("http", "https"):
            raise ValueError("webhook url must be http or https")
        return v


class JobRequest(BaseModel):
    recipe: str | None = Field(None, description="reel, highlights, captions, video_edit, photo_edit, render_program, transcribe")
    prompt: str | None = Field(None, max_length=500)
    inputs: list[InputRef | str] = Field(min_length=1, max_length=50,
                                         description="file ids (f_...), URLs, or local paths; numbered 1..n in this order")
    style: str | None = Field(None, description="Editing style id (GET /v1/options)")
    output: OutputOptions | None = None
    options: dict[str, Any] = Field(default_factory=dict, description="Recipe options (GET /v1/recipes)")
    webhook: WebhookSpec | str | None = None
    output_dir: str | None = Field(None, description="Also copy results into this folder (scope `local`)")
    label: str | None = Field(None, max_length=120)

    def refs(self) -> list[InputRef]:
        out = []
        for x in self.inputs:
            if isinstance(x, InputRef):
                out.append(x)
            elif x.startswith("f_"):
                out.append(InputRef(file_id=x))
            elif x.startswith(("http://", "https://")):
                out.append(InputRef(url=x))
            else:
                out.append(InputRef(path=x))
        return out

    def hook(self) -> WebhookSpec | None:
        return WebhookSpec(url=self.webhook) if isinstance(self.webhook, str) else self.webhook


class JobManager:
    def __init__(self, home: Path | str | None = None, *, workers: int = 1, keep_days: float = 7,
                 allow_urls: bool = True, start: bool = True) -> None:
        self.home = automation_home(home)
        self.keys = KeyStore(self.home)
        self.keep_days = keep_days
        self.allow_urls = allow_urls
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._workers: list[threading.Thread] = []
        self._stopping = False
        self._cleanup()
        self._load()
        if start:
            for i in range(max(1, workers)):
                t = threading.Thread(target=self._worker, name=f"automation-{i}", daemon=True)
                t.start()
                self._workers.append(t)

    # ------------------------------------------------------------ uploads
    def reserve_upload(self, name: str) -> tuple[str, Path]:
        fid = "f_" + secrets.token_hex(8)
        d = self.home / "uploads" / fid
        d.mkdir(parents=True)
        return fid, d / safe_name(name)

    def discard_upload(self, fid: str) -> None:
        shutil.rmtree(self.home / "uploads" / fid, ignore_errors=True)

    def finish_upload(self, fid: str, dst: Path, size: int) -> dict[str, Any]:
        if size == 0:
            self.discard_upload(fid)
            raise JobError("empty upload")
        return {"file_id": fid, "name": dst.name, "size": size}

    def upload(self, name: str, chunks: Iterable[bytes], limit: int = MAX_DOWNLOAD) -> dict[str, Any]:
        fid, dst = self.reserve_upload(name)
        size = 0
        try:
            with dst.open("wb") as f:
                for chunk in chunks:
                    size += len(chunk)
                    if size > limit:
                        raise JobError("file is too large", 413)
                    f.write(chunk)
        except BaseException:
            self.discard_upload(fid)
            raise
        return self.finish_upload(fid, dst, size)

    def upload_path(self, fid: str) -> Path:
        if not re.fullmatch(r"f_[0-9a-f]{16}", fid):
            raise JobError(f"bad file id {fid!r}")
        d = self.home / "uploads" / fid
        files = [p for p in d.iterdir() if p.is_file()] if d.is_dir() else []
        if not files:
            raise JobError(f"no uploaded file {fid} (uploads are kept for {self.keep_days:g} days)", 404)
        return files[0]

    # --------------------------------------------------------------- jobs
    def submit(self, req: JobRequest, *, allow_local: bool = False, owner: str | None = None) -> dict[str, Any]:
        refs = req.refs()
        names = []
        for r in refs:
            if r.path is not None:
                if not allow_local:
                    raise JobError("local paths need an API key with the `local` scope; upload the file instead", 403)
                p = Path(r.path).expanduser()
                if not p.is_file():
                    raise JobError(f"input not found: {p}")
                names.append(p.name)
            elif r.file_id is not None:
                names.append(self.upload_path(r.file_id).name)
            else:
                if not self.allow_urls:
                    raise JobError("URL inputs are switched off on this engine")
                if urllib.parse.urlsplit(r.url or "").scheme not in ("http", "https"):
                    raise JobError("input URLs must be http or https")
                names.append(safe_name(urllib.parse.urlsplit(r.url or "").path, "download"))
        if req.output_dir and not allow_local:
            raise JobError("output_dir needs an API key with the `local` scope", 403)
        try:
            recipe = R.resolve(req.recipe or R.guess_recipe(req.prompt, [Path(n) for n in names]))
            R.parse_options(recipe, req.options)
        except R.RecipeError as exc:
            raise JobError(str(exc)) from exc
        if recipe.needs_prompt and not req.prompt:
            raise JobError(f"recipe {recipe.name} needs a prompt")
        req.hook()  # validates a plain-string webhook url
        jid = "j_" + secrets.token_hex(6)
        job = {
            "id": jid, "recipe": recipe.name, "label": req.label or f"{recipe.title}: {', '.join(names)[:80]}",
            "state": "queued", "progress": 0.0, "stage": "Waiting", "error": None, "result": None, "outputs": [],
            "inputs": names, "created": time.time(), "started": None, "finished": None, "owner": owner,
            "request": req.model_dump(mode="json"), "webhook": None,
        }
        (self.home / "jobs" / jid).mkdir(parents=True)
        with self._lock:
            self._jobs[jid] = job
            self._cancel[jid] = threading.Event()
            self._save(job)
        self._queue.put(jid)
        self._emit(job)
        return self.public(job)

    def run_now(self, req: JobRequest, *, allow_local: bool = True,
                on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        """Submit and wait (CLI / MCP / watch folders without a background worker)."""
        job = self.submit(req, allow_local=allow_local)
        if not self._workers:
            unsubscribe = self.subscribe(lambda j: on_progress(self.public(j)) if on_progress and j["id"] == job["id"] else None)
            try:
                self._run(job["id"])
            finally:
                unsubscribe()
            return self.get(job["id"])
        return self.wait(job["id"], on_progress=on_progress)

    def get(self, jid: str, owner: str | None = None) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(jid)
            if job is None or (owner is not None and job.get("owner") != owner):
                raise JobError(f"no job {jid}", 404)
            return self.public(job)

    def list(self, owner: str | None = None, state: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if (owner is None or j.get("owner") == owner) and (state is None or j["state"] == state)]
        jobs.sort(key=lambda j: j["created"], reverse=True)
        return [self.public(j) for j in jobs[:limit]]

    def cancel(self, jid: str, owner: str | None = None) -> dict[str, Any]:
        self.get(jid, owner)
        with self._lock:
            job = self._jobs[jid]
            if job["state"] in FINAL:
                return self.public(job)
            self._cancel[jid].set()
            if job["state"] == "queued":
                self._finish(job, "cancelled")
        return self.get(jid)

    def delete(self, jid: str, owner: str | None = None) -> bool:
        """Remove a finished job and its files (cancel first if it is still running)."""
        self.get(jid, owner)
        with self._lock:
            if self._jobs[jid]["state"] not in FINAL:
                raise JobError("cancel the job first", 409)
            self._jobs.pop(jid)
            self._cancel.pop(jid, None)
        shutil.rmtree(self.home / "jobs" / jid, ignore_errors=True)
        return True

    def wait(self, jid: str, timeout: float | None = None, on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        end = None if timeout is None else time.time() + timeout
        last = None
        with self._changed:
            while True:
                job = self._jobs[jid]
                snap = (job["state"], round(job["progress"], 3), job["stage"])
                if on_progress and snap != last:
                    on_progress(self.public(job))
                    last = snap
                if job["state"] in FINAL:
                    return self.public(job)
                left = None if end is None else end - time.time()
                if left is not None and left <= 0:
                    return self.public(job)
                self._changed.wait(timeout=min(1.0, left) if left is not None else 1.0)

    def output_path(self, jid: str, name: str, owner: str | None = None) -> Path:
        job = self.get(jid, owner)
        if name not in {o["name"] for o in job["outputs"]}:
            raise JobError(f"job {jid} has no output {name!r}", 404)
        return self.home / "jobs" / jid / "out" / name

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def shutdown(self) -> None:
        self._stopping = True
        for ev in list(self._cancel.values()):
            ev.set()
        for _ in self._workers:
            self._queue.put(None)

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in job.items() if k != "request"}
        out["progress"] = round(job["progress"], 3)
        out["outputs"] = [{**o, "url": f"/v1/jobs/{job['id']}/files/{urllib.parse.quote(o['name'])}"} for o in job["outputs"]]
        req = job.get("request") or {}
        out["prompt"], out["style"], out["output"], out["options"] = req.get("prompt"), req.get("style"), req.get("output"), req.get("options")
        return out

    # ------------------------------------------------------------ running
    def _worker(self) -> None:
        while not self._stopping:
            jid = self._queue.get()
            if jid is None:
                return
            try:
                self._run(jid)
            except Exception:  # noqa: BLE001 - a broken job must never kill the worker
                log.exception("job %s crashed", jid)

    def _run(self, jid: str) -> None:
        from ..session import Session

        with self._lock:
            job = self._jobs.get(jid)
            if job is None or job["state"] != "queued":
                return
            cancel = self._cancel.setdefault(jid, threading.Event())
            job["state"], job["started"], job["stage"] = "running", time.time(), "Preparing inputs"
            self._save(job)
        self._emit(job)
        req = JobRequest.model_validate(job["request"])
        d = self.home / "jobs" / jid
        out = d / "out"
        out.mkdir(parents=True, exist_ok=True)
        session = None
        last_save = [0.0]

        def report(frac: float, stage: str) -> None:
            with self._lock:
                job["progress"], job["stage"] = max(job["progress"], min(0.999, frac)), stage
                if time.time() - last_save[0] > 2:
                    last_save[0] = time.time()
                    self._save(job)
            self._emit(job)

        try:
            files = [self._fetch(r, d / "in", cancel) for r in req.refs()]
            recipe = R.resolve(job["recipe"])
            options = R.parse_options(recipe, req.options)
            session = Session.create(d / "project", job["label"][:60] or jid, R.project_kind(recipe, files),
                                     make_proxies=False, shared_cache=self.home / "cache")
            imp = session.import_media(files, source="ai")
            if imp["failed"]:
                bad = imp["failed"][0]
                raise R.RecipeError(f"cannot read {Path(bad['path']).name}: {bad['error']}")
            ctx = R.JobContext(session=session, inputs=files, out=out, options=options, prompt=req.prompt,
                               style=req.style, output=req.output, report=report, cancelled=cancel.is_set)
            result = recipe.run(ctx)
            if req.output_dir:
                result["copied_to"] = R.copy_outputs(out, Path(req.output_dir).expanduser(), ctx.outputs)
            with self._lock:
                job["outputs"], job["result"] = ctx.outputs, result
            self._finish(job, "done")
        except R.Cancelled:
            self._finish(job, "cancelled")
        except (R.RecipeError, JobError) as exc:
            self._finish(job, "failed", str(exc))
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            log.exception("job %s failed", jid)
            self._finish(job, "failed", f"{type(exc).__name__}: {exc}")
        finally:
            if session is not None:
                session.close(wait=True)
            shutil.rmtree(d / "in", ignore_errors=True)

    def _fetch(self, ref: InputRef, folder: Path, cancel: threading.Event) -> Path:
        if ref.file_id:
            return self.upload_path(ref.file_id)
        if ref.path:
            p = Path(ref.path).expanduser()
            if not p.is_file():
                raise JobError(f"input not found: {p}")
            return p
        folder.mkdir(parents=True, exist_ok=True)
        url = ref.url or ""
        dst = folder / f"{len(list(folder.iterdir())) + 1}_{safe_name(urllib.parse.urlsplit(url).path, 'download')}"
        req = urllib.request.Request(url, headers={"User-Agent": "AIEditor/1"})
        size = 0
        with urllib.request.urlopen(req, timeout=60) as resp, dst.open("wb") as f:  # noqa: S310 - scheme checked in submit
            while chunk := resp.read(1 << 20):
                if cancel.is_set():
                    raise R.Cancelled()
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise JobError("input download is too large")
                f.write(chunk)
        if not dst.suffix:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0]
            ext = {"video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov", "image/jpeg": ".jpg",
                   "image/png": ".png", "audio/mpeg": ".mp3", "audio/wav": ".wav"}.get(ctype)
            if ext:
                dst = dst.rename(dst.with_suffix(ext))
        return dst

    def _finish(self, job: dict[str, Any], state: str, error: str | None = None) -> None:
        with self._lock:
            job["state"], job["error"], job["finished"] = state, error, time.time()
            if state == "done":
                job["progress"], job["stage"] = 1.0, "Done"
            else:
                job["stage"] = {"failed": "Failed", "cancelled": "Cancelled"}[state]
            self._cancel.pop(job["id"], None)
            self._save(job)
        self._emit(job)
        hook = JobRequest.model_validate(job["request"]).hook()
        if hook:
            def result(r: dict[str, Any]) -> None:
                with self._lock:
                    job["webhook"] = {"url": hook.url, **r}
                    self._save(job)

            deliver(hook.url, hook.secret or self.keys.webhook_secret(), f"job.{state}", {"job": self.public(job)}, result)

    def _emit(self, job: dict[str, Any]) -> None:
        with self._changed:
            self._changed.notify_all()
        for fn in list(self._listeners):
            try:
                fn(job)
            except Exception:  # noqa: BLE001
                log.exception("job listener failed")

    # -------------------------------------------------------- persistence
    def _save(self, job: dict[str, Any]) -> None:
        p = self.home / "jobs" / job["id"] / "job.json"
        if not p.parent.is_dir():
            return
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job), encoding="utf-8")
        tmp.replace(p)

    def _load(self) -> None:
        for p in sorted((self.home / "jobs").glob("*/job.json")):
            try:
                job = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("state") == "running":
                job.update(state="failed", error="interrupted: the engine stopped while this job was running",
                           stage="Failed", finished=time.time())
                self._save(job)
            self._jobs[job["id"]] = job
            if job["state"] == "queued":
                self._cancel[job["id"]] = threading.Event()
                self._queue.put(job["id"])

    def _cleanup(self) -> None:
        cutoff = time.time() - self.keep_days * 86400
        for d in list((self.home / "jobs").iterdir()) + list((self.home / "uploads").iterdir()):
            try:
                meta = d / "job.json"
                if meta.is_file():
                    job = json.loads(meta.read_text(encoding="utf-8"))
                    if job.get("state") in FINAL and (job.get("finished") or 0) < cutoff:
                        shutil.rmtree(d, ignore_errors=True)
                elif d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except (OSError, ValueError):
                continue
