"""Local HTTP + WebSocket API used by the desktop UI (and later the agent).

Security: the engine listens on 127.0.0.1 only, and every request must carry
the per-launch token (header `X-Engine-Token`, or `?token=` for <img>/<video>
tags and WebSockets). This stops web pages in the user's browser from driving
the engine through localhost.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..core.commands import CommandError, list_commands
from ..core.project import ProjectKind
from ..session import Session

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


class OpenProjectBody(BaseModel):
    path: str


class CommandBody(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)
    source: str = "user"


class ImportBody(BaseModel):
    paths: list[str] = Field(min_length=1)


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


def create_app(token: str | None = None) -> FastAPI:
    token = token or secrets.token_urlsafe(32)
    state = EngineState()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if state.session is not None:
            state.session.close(wait=False)

    app = FastAPI(title="AIEditor Engine", version=__version__, lifespan=lifespan)
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
