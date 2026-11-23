"""Public automation API (/v1) for n8n, Zapier/Make, scripts and agents.

Auth: `Authorization: Bearer aie_...` (or `X-API-Key`). The desktop app's
engine token is also accepted and has every scope. Keys are created with
`framewise keys create NAME` or POST /v1/keys (admin scope).

Typical flow:
    PUT  /v1/files?name=talk.mp4        (raw bytes)          -> {"file_id": "f_..."}
    POST /v1/jobs {"recipe": "reel", "inputs": ["f_..."], "output": {"platform": "tiktok"}, "webhook": "https://..."}
    GET  /v1/jobs/{id}/wait?timeout=60   (or receive the webhook)
    GET  /v1/jobs/{id}/files/reel_1.mp4
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..agent.options import catalog as option_catalog
from ..automation import recipes as R
from ..automation.jobs import MAX_DOWNLOAD, JobError, JobManager, JobRequest
from ..automation.keys import DEFAULT_SCOPES, KeyError_


@dataclass
class Caller:
    key_id: str | None  # None = desktop app (engine token)
    name: str
    scopes: tuple[str, ...]

    @property
    def owner(self) -> str | None:
        """Jobs are private to the key that made them; admin keys and the app see all."""
        return None if (self.key_id is None or "admin" in self.scopes) else self.key_id


class NewKeyBody(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    scopes: list[str] = Field(default_factory=lambda: list(DEFAULT_SCOPES))


def _err(exc: JobError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


def register(app: FastAPI, manager: JobManager, engine_token: str | None) -> None:
    def caller(request: Request) -> Caller:
        auth = request.headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "").strip()
        tok = request.headers.get("x-engine-token", "")
        if engine_token and tok and secrets.compare_digest(tok, engine_token):
            return Caller(None, "desktop app", ("jobs", "local", "admin"))
        rec = manager.keys.verify(key) if key else None
        if rec is None:
            raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return Caller(rec["id"], rec["name"], tuple(rec["scopes"]))

    def need(scope: str):  # noqa: ANN202 - FastAPI dependency factory
        def check(c: Caller = Depends(caller)) -> Caller:
            if scope not in c.scopes:
                raise HTTPException(403, f"this API key lacks the `{scope}` scope")
            return c

        return check

    jobs_scope = Depends(need("jobs"))
    admin_scope = Depends(need("admin"))
    tags = ["automation"]

    @app.get("/v1", tags=tags)
    def v1_info() -> dict[str, Any]:
        return {"name": "Framewise automation API", "version": __version__, "docs": "/docs", "openapi": "/openapi.json",
                "recipes": list(R.RECIPES)}

    @app.get("/v1/me", tags=tags)
    def me(c: Caller = Depends(caller)) -> dict[str, Any]:
        """Who am I (used by the n8n credential test)."""
        return {"key_id": c.key_id, "name": c.name, "scopes": list(c.scopes)}

    @app.get("/v1/recipes", tags=tags)
    def recipes(_: Caller = jobs_scope) -> list[dict[str, Any]]:
        return R.catalog()

    @app.get("/v1/options", tags=tags)
    def options(_: Caller = jobs_scope) -> dict[str, Any]:
        """Editing styles, desired outputs and lengths (same as the app's prompt chips)."""
        return option_catalog(None)

    # ------------------------------------------------------------- files
    @app.put("/v1/files", tags=tags, status_code=201)
    @app.post("/v1/files", tags=tags, status_code=201, include_in_schema=False)
    async def upload(request: Request, name: str = Query(..., min_length=1, max_length=200), _: Caller = jobs_scope) -> dict[str, Any]:
        """Upload one input file as the raw request body (n8n: Body Content Type = n8n Binary File)."""
        fid, dst = manager.reserve_upload(name)
        size = 0
        try:
            with dst.open("wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD:
                        raise JobError("file is too large", 413)
                    f.write(chunk)
            return manager.finish_upload(fid, dst, size)
        except JobError as exc:
            manager.discard_upload(fid)
            raise _err(exc) from exc
        except BaseException:
            manager.discard_upload(fid)
            raise

    # -------------------------------------------------------------- jobs
    @app.post("/v1/jobs", tags=tags, status_code=202)
    def create_job(body: JobRequest, c: Caller = jobs_scope) -> dict[str, Any]:
        try:
            return manager.submit(body, allow_local="local" in c.scopes, owner=c.key_id)
        except JobError as exc:
            raise _err(exc) from exc

    @app.get("/v1/jobs", tags=tags)
    def list_jobs(state: str | None = None, limit: int = Query(50, ge=1, le=500), c: Caller = jobs_scope) -> list[dict[str, Any]]:
        return manager.list(c.owner, state, limit)

    @app.get("/v1/jobs/{job_id}", tags=tags)
    def get_job(job_id: str, c: Caller = jobs_scope) -> dict[str, Any]:
        try:
            return manager.get(job_id, c.owner)
        except JobError as exc:
            raise _err(exc) from exc

    @app.get("/v1/jobs/{job_id}/wait", tags=tags)
    def wait_job(job_id: str, timeout: float = Query(30, ge=0, le=120), c: Caller = jobs_scope) -> dict[str, Any]:
        """Long-poll: returns when the job finishes or after `timeout` seconds (check `state`)."""
        try:
            manager.get(job_id, c.owner)
            return manager.wait(job_id, timeout)
        except JobError as exc:
            raise _err(exc) from exc

    @app.get("/v1/jobs/{job_id}/result", tags=tags)
    def job_result(job_id: str, c: Caller = jobs_scope) -> dict[str, Any]:
        try:
            job = manager.get(job_id, c.owner)
        except JobError as exc:
            raise _err(exc) from exc
        if job["state"] != "done":
            raise HTTPException(409, f"job is {job['state']}" + (f": {job['error']}" if job["error"] else ""))
        return {"id": job_id, "result": job["result"], "outputs": job["outputs"]}

    @app.get("/v1/jobs/{job_id}/files/{name}", tags=tags)
    def job_file(job_id: str, name: str, c: Caller = jobs_scope) -> FileResponse:
        try:
            path = manager.output_path(job_id, name, c.owner)
        except JobError as exc:
            raise _err(exc) from exc
        return FileResponse(path, filename=name)

    @app.post("/v1/jobs/{job_id}/cancel", tags=tags)
    def cancel_job(job_id: str, c: Caller = jobs_scope) -> dict[str, Any]:
        try:
            return manager.cancel(job_id, c.owner)
        except JobError as exc:
            raise _err(exc) from exc

    @app.delete("/v1/jobs/{job_id}", tags=tags)
    def delete_job(job_id: str, c: Caller = jobs_scope) -> dict[str, Any]:
        """Cancels a queued/running job; deletes a finished job and its files."""
        try:
            job = manager.get(job_id, c.owner)
            if job["state"] in ("queued", "running"):
                return {"cancelled": True, "job": manager.cancel(job_id, c.owner)}
            return {"deleted": manager.delete(job_id, c.owner)}
        except JobError as exc:
            raise _err(exc) from exc

    # --------------------------------------------------------- admin
    @app.get("/v1/keys", tags=tags)
    def list_keys(_: Caller = admin_scope) -> list[dict[str, Any]]:
        return manager.keys.list()

    @app.post("/v1/keys", tags=tags, status_code=201)
    def create_key(body: NewKeyBody, _: Caller = admin_scope) -> dict[str, Any]:
        """The full key is returned once; only its hash is stored."""
        try:
            key, rec = manager.keys.create(body.name, body.scopes)
        except KeyError_ as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"key": key, **rec}

    @app.delete("/v1/keys/{key_id}", tags=tags)
    def revoke_key(key_id: str, _: Caller = admin_scope) -> Response:
        if not manager.keys.revoke(key_id):
            raise HTTPException(404, "no such key")
        return Response(status_code=204)

    @app.get("/v1/webhooks/secret", tags=tags)
    def webhook_secret(_: Caller = admin_scope) -> dict[str, str]:
        """Secret that signs webhooks for jobs that do not set their own."""
        return {"secret": manager.keys.webhook_secret(), "header": "X-Framewise-Signature",
                "scheme": "t=<unix>,v1=hex(HMAC_SHA256(secret, '<t>.' + raw_body))"}
