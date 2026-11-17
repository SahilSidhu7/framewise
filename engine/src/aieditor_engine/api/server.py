"""Local HTTP + WebSocket API used by the desktop UI (and later the agent).

Security: the engine listens on 127.0.0.1 only, and every request must carry
the per-launch token (header `X-Engine-Token`, or `?token=` for <img>/<video>
tags and WebSockets). This stops web pages in the user's browser from driving
the engine through localhost.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..agent.llm import LLM, LLMUnavailable
from ..agent.options import OptionError, OutputOptions
from ..agent.planner import plan
from ..automation.jobs import JobManager
from ..core.commands import CommandError, list_commands
from ..core.project import ProjectKind
from ..dsl import DslError, compile_program, to_dsl
from ..imaging.ops import OPS
from ..imaging.recipes import load_recipes
from ..imaging.render import export_image, render_preview
from ..session import Session
from . import video_routes

ALLOWED_ORIGINS = [
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",  # Vite dev server
    "http://127.0.0.1:1420",
]


class CreateProjectBody(BaseModel):
    path: str
    name: str = Field(min_length=1, max_length=200)
    kind: ProjectKind = ProjectKind.VIDEO


class LicenceBody(BaseModel):
    key: str = Field(max_length=4000)


class OpenProjectBody(BaseModel):
    path: str


class CommandBody(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)
    source: str = "user"


class ImportBody(BaseModel):
    paths: list[str] = Field(min_length=1)


class ExportBody(BaseModel):
    path: str
    format: Literal["jpeg", "png", "tiff", "webp"] = "jpeg"
    quality: int = Field(95, ge=50, le=100)
    bit_depth: Literal[8, 16] = 8
    long_edge: int | None = Field(None, ge=64, le=20000)


class PreviewBody(BaseModel):
    ops: list[dict[str, Any]] = Field(default_factory=list, max_length=64)
    max: int = Field(1200, ge=64, le=4096)


class AiEditBody(BaseModel):
    media_id: str
    prompt: str = Field(min_length=1, max_length=500)
    variations: int = Field(3, ge=1, le=5)
    style: str | None = Field(None, description="Photo style id from /ai/options, or auto")
    output: OutputOptions | None = None


def _safe_start() -> None:
    try:
        LLM.ensure_started()
    except LLMUnavailable:
        pass  # status endpoint reports the error


class EngineState:
    """Holds the open session and fans its events out to every WebSocket.

    Listeners live here, not on a session's bus, so a socket connected before
    a project is opened (or across project switches) keeps receiving events.
    """

    def __init__(self) -> None:
        self.session: Session | None = None
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._unsubscribe_bus: Callable[[], None] = lambda: None

    def require(self) -> Session:
        if self.session is None:
            raise HTTPException(409, "no project open")
        return self.session

    def replace(self, session: Session) -> None:
        if self.session is not None:
            self._unsubscribe_bus()  # a still-running job of the old project must not leak events
            self.session.close(wait=False)
        self.session = session
        self._unsubscribe_bus = session.bus.subscribe(self.broadcast)
        self.broadcast({"type": "project_opened", "revision": session.bus.revision})

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def broadcast(self, event: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            listener(event)


def create_app(token: str | None = None, automation: JobManager | None = None) -> FastAPI:
    """automation: when given, the public /v1 job API is mounted (headless `framewise serve`)."""
    token = token or secrets.token_urlsafe(32)
    state = EngineState()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if state.session is not None:
            state.session.close(wait=False)
        if automation is not None:
            automation.shutdown()

    app = FastAPI(title="Framewise Engine", version=__version__, lifespan=lifespan)
    app.state.token = token
    app.state.engine = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def auth(request: Request, token_q: str | None = Query(None, alias="token")) -> None:
        supplied = request.headers.get("x-engine-token") or token_q or ""
        if not secrets.compare_digest(supplied, token):
            raise HTTPException(401, "bad or missing engine token")

    secured = [Depends(auth)]

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError) -> Response:
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": str(exc.args[0]) if exc.args else "not found"}, status_code=404)

    # ---------------------------------------------------------------- meta
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.get("/commands", dependencies=secured)
    def commands() -> list[dict[str, Any]]:
        return list_commands()

    # ------------------------------------------------------------- project
    @app.post("/projects", dependencies=secured)
    def create_project(body: CreateProjectBody) -> dict[str, Any]:
        root = Path(body.path).expanduser()
        try:
            session = Session.create(root, body.name, body.kind)
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        state.replace(session)
        return session.bus.snapshot()

    @app.post("/projects/open", dependencies=secured)
    def open_project(body: OpenProjectBody) -> dict[str, Any]:
        root = Path(body.path).expanduser()
        try:
            session = Session.open(root)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"no project at {root}") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        state.replace(session)
        return session.bus.snapshot()

    @app.get("/project", dependencies=secured)
    def get_project() -> dict[str, Any]:
        return state.require().bus.snapshot()

    @app.post("/project/save", dependencies=secured)
    def save_project() -> dict[str, bool]:
        state.require().save()
        return {"ok": True}

    # ------------------------------------------------------------ commands
    @app.post("/commands/{name}", dependencies=secured)
    def run_command(name: str, body: CommandBody) -> dict[str, Any]:
        session = state.require()
        if body.source not in ("user", "ai"):
            raise HTTPException(422, "source must be 'user' or 'ai'")
        try:
            result = session.bus.execute(name, body.params, source=body.source)  # type: ignore[arg-type]
        except CommandError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"result": result, "revision": session.bus.revision}

    @app.post("/undo", dependencies=secured)
    def undo() -> dict[str, Any]:
        session = state.require()
        return {"ok": session.bus.undo(), "revision": session.bus.revision}

    @app.post("/redo", dependencies=secured)
    def redo() -> dict[str, Any]:
        session = state.require()
        return {"ok": session.bus.redo(), "revision": session.bus.revision}

    # --------------------------------------------------------------- media
    @app.post("/media/import", dependencies=secured)
    def import_media(body: ImportBody) -> dict[str, Any]:
        return state.require().import_media([Path(p).expanduser() for p in body.paths])

    @app.get("/media/{media_id}/thumb", dependencies=secured)
    def media_thumb(media_id: str) -> FileResponse:
        return _derived_file(state.require(), media_id, "thumb_path", "image/jpeg")

    @app.get("/media/{media_id}/proxy", dependencies=secured)
    def media_proxy(media_id: str) -> FileResponse:
        return _derived_file(state.require(), media_id, "proxy_path", "video/mp4")

    # --------------------------------------------------------------- photo
    @app.get("/ops", dependencies=secured)
    def ops() -> list[dict[str, Any]]:
        return [spec.schema() for spec in OPS.values()]

    @app.get("/looks", dependencies=secured)
    def looks() -> list[dict[str, Any]]:
        return [r.summary() for r in load_recipes().values()]

    @app.get("/media/{media_id}/render", dependencies=secured)
    def render(
        media_id: str,
        max: int = Query(1600, ge=64, le=4096),
        bypass: bool = False,
        look: str | None = None,
        s: float = Query(1.0, ge=0, le=1.5),
        program: str | None = Query(None, max_length=4000),
    ) -> Response:
        """Preview JPEG. bypass=1 -> original (before/after); look=name -> try a look on top;
        program=<Edit DSL> -> preview an unapplied program (AI variations)."""
        session = state.require()
        path, stack = session.photo_stack(media_id)
        if bypass:
            stack = []
        if program is not None:
            try:
                stack = compile_program(program)
            except DslError as exc:
                raise HTTPException(422, str(exc)) from exc
        if look:
            stack = [o for o in stack if o["op"] != "look"] + [{"op": "look", "params": {"name": look, "strength": s}}]
        try:
            data = render_preview(path, stack, max_side=max)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/media/{media_id}/preview", dependencies=secured)
    def preview(media_id: str, body: PreviewBody) -> Response:
        """Render an arbitrary (uncommitted) stack, e.g. while a slider is being dragged."""
        path, _ = state.require().photo_stack(media_id)
        try:
            data = render_preview(path, body.ops, max_side=body.max)
        except (ValueError, KeyError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return Response(data, media_type="image/jpeg")

    @app.get("/media/{media_id}/analysis", dependencies=secured)
    def analysis(media_id: str) -> dict[str, Any]:
        return state.require().analysis(media_id)

    @app.get("/media/{media_id}/dsl", dependencies=secured)
    def media_dsl(media_id: str) -> dict[str, str]:
        _, stack = state.require().photo_stack(media_id)
        return {"program": to_dsl(stack)}

    @app.post("/media/{media_id}/export", dependencies=secured)
    def export(media_id: str, body: ExportBody) -> dict[str, Any]:
        session = state.require()
        path, stack = session.photo_stack(media_id)
        out = Path(body.path).expanduser()
        if out.suffix == "":
            out = out.with_suffix({"jpeg": ".jpg", "png": ".png", "tiff": ".tif", "webp": ".webp"}[body.format])
        try:
            return export_image(path, stack, out, body.format, body.quality, body.bit_depth, body.long_edge)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    # ------------------------------------------------------------------ AI
    @app.get("/ai/status", dependencies=secured)
    def ai_status() -> dict[str, Any]:
        return LLM.status()

    @app.post("/ai/warmup", dependencies=secured)
    def ai_warmup() -> dict[str, Any]:
        """Start the local model in the background so the first request is fast."""
        if LLM.available():
            threading.Thread(target=_safe_start, daemon=True).start()
        return LLM.status()

    @app.post("/ai/edit", dependencies=secured)
    def ai_edit(body: AiEditBody) -> dict[str, Any]:
        """Plan edits for a photo. Returns variations (Edit DSL); nothing is applied until
        the client runs photo.apply_program with source='ai'."""
        session = state.require()
        _, stack = session.photo_stack(body.media_id)
        try:
            return plan(body.prompt, session.analysis(body.media_id), to_dsl(stack), body.variations, body.style, body.output)
        except OptionError as exc:
            raise HTTPException(422, str(exc)) from exc

    # ------------------------------------------------------------- licence
    from ..licence import LicenceError, LicenceStore

    licence_store = automation.licence if automation is not None and automation.licence else LicenceStore()

    @app.get("/licence", dependencies=secured)
    def licence_status() -> dict[str, Any]:
        """Free / trial / pro, for the app's About and Pro screens."""
        return licence_store.status()

    @app.post("/licence/activate", dependencies=secured)
    def licence_activate(body: LicenceBody) -> dict[str, Any]:
        try:
            return licence_store.activate(body.key)
        except LicenceError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/licence/trial", dependencies=secured)
    def licence_trial() -> dict[str, Any]:
        try:
            return licence_store.start_trial()
        except LicenceError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/licence", dependencies=secured)
    def licence_deactivate() -> dict[str, Any]:
        return licence_store.deactivate()

    # ----------------------------------------------------- onboarding / system
    from ..system import AppSettings, benchmark, system_info

    app_settings = AppSettings()

    @app.get("/system", dependencies=secured)
    def system() -> dict[str, Any]:
        """Processor, memory, disk, hardware encoder and which AI parts are installed."""
        return system_info()

    @app.post("/system/benchmark", dependencies=secured)
    def system_benchmark() -> dict[str, Any]:
        """About one second of real export work; estimates a 30 s 1080p reel on this PC."""
        return benchmark()

    @app.get("/app/settings", dependencies=secured)
    def get_app_settings() -> dict[str, Any]:
        return app_settings.get()

    @app.post("/app/settings", dependencies=secured)
    def set_app_settings(body: dict[str, Any]) -> dict[str, Any]:
        try:
            return app_settings.update(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    video_routes.register(app, state.require, secured)
    if automation is not None:
        from . import public_routes

        app.state.automation = automation
        public_routes.register(app, automation, token)

    # -------------------------------------------------------------- events
    @app.websocket("/ws")
    async def events(ws: WebSocket) -> None:
        supplied = ws.query_params.get("token", "")
        if not secrets.compare_digest(supplied, token):
            await ws.close(code=4401)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        # Events fire on worker threads; hop onto this socket's event loop.
        unsubscribe = state.subscribe(lambda e: loop.call_soon_threadsafe(queue.put_nowait, e))
        session = state.session
        await ws.send_json({"type": "hello", "revision": session.bus.revision if session else None})
        try:
            while True:
                await ws.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            unsubscribe()

    return app


def _derived_file(session: Session, media_id: str, attr: str, media_type: str) -> FileResponse:
    item = session.bus.project.media.get(media_id)
    rel = getattr(item, attr, None) if item else None
    if not rel:
        raise HTTPException(404, "not available")
    try:
        path = session.paths.resolve(rel)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, "not available")
    return FileResponse(path, media_type=media_type)
