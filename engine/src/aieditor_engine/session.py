"""An open project: bus + storage + background media jobs."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .core import builtin, photo_commands  # noqa: F401  (register commands)
from .core.bus import CommandBus, Source
from .core.project import MediaKind, Project, ProjectKind, ProxyStatus
from .core.store import ProjectPaths, create_project, load_project, save_project
from .imaging.render import SOURCES, analyze
from .media.probe import ProbeError, make_thumbnail, probe
from .media.proxy import ProxyError, make_proxy

log = logging.getLogger(__name__)


class Session:
    def __init__(self, root: Path, project: Project, *, proxy_workers: int = 1) -> None:
        self.paths = ProjectPaths(root)
        self.paths.ensure()
        self.bus = CommandBus(project, log_path=self.paths.trajectory_log)
        self.bus.subscribe(self._autosave)
        self._jobs = ThreadPoolExecutor(max_workers=proxy_workers, thread_name_prefix="proxy")
        self._pending: list[Future] = []
        self._analysis: dict[str, dict[str, Any]] = {}

    # ----------------------------------------------------------- lifecycle
    @classmethod
    def create(cls, root: Path, name: str, kind: ProjectKind = ProjectKind.VIDEO) -> Session:
        return cls(root, create_project(root, name, kind))

    @classmethod
    def open(cls, root: Path) -> Session:
        session = cls(root, load_project(root))
        session._resume_proxies()
        return session

    def save(self) -> None:
        with self.bus.lock:
            save_project(self.paths.root, self.bus.project)
            self.bus.dirty = False

    def close(self, wait: bool = True) -> None:
        self._jobs.shutdown(wait=wait, cancel_futures=not wait)
        if self.bus.dirty:
            self.save()

    def wait_for_jobs(self, timeout: float | None = None) -> None:
        for fut in list(self._pending):
            fut.result(timeout=timeout)

    # -------------------------------------------------------------- import
    def import_media(self, files: list[Path], source: Source = "user") -> dict[str, Any]:
        """Probe files, add them to the bin, and queue proxies. Bad files are
        reported individually; one failure does not abort the batch."""
        imported: list[str] = []
        failed: list[dict[str, str]] = []
        for file in files:
            try:
                info = probe(file)
            except ProbeError as exc:
                failed.append({"path": str(file), "error": str(exc)})
                continue
            result = self.bus.execute("media.add", info, source=source)
            media_id = result["media_id"]
            imported.append(media_id)
            self._make_thumb(media_id)
            if info["kind"] == MediaKind.VIDEO:
                self._queue_proxy(media_id)
        return {"imported": imported, "failed": failed}

    # --------------------------------------------------------------- photo
    def photo_stack(self, media_id: str) -> tuple[Path, list[dict[str, Any]]]:
        """Source path + a JSON copy of the edit stack (safe to use off-lock)."""
        with self.bus.lock:
            item = self.bus.project.media.get(media_id)
            if item is None or item.kind != MediaKind.IMAGE:
                raise KeyError(f"photo not found: {media_id}")
            stack = [o.model_dump(mode="json") for o in self.bus.project.edits.get(media_id, [])]
        return Path(item.path), stack

    def analysis(self, media_id: str) -> dict[str, Any]:
        """L1 measurements of the *original* photo, cached per media."""
        if media_id not in self._analysis:
            path, _ = self.photo_stack(media_id)
            self._analysis[media_id] = analyze(SOURCES.get(path, 1024))
        return self._analysis[media_id]

    # ---------------------------------------------------------------- jobs
    def _make_thumb(self, media_id: str) -> None:
        item = self.bus.project.media[media_id]
        rel = f"cache/thumbs/{media_id}.jpg"
        try:
            if make_thumbnail(Path(item.path), item.kind, self.paths.root / rel):
                self.bus.patch_media(media_id, thumb_path=rel)
        except Exception:  # thumbnails are cosmetic; never block an import
            log.exception("thumbnail failed for %s", item.path)

    def _queue_proxy(self, media_id: str) -> None:
        self._pending.append(self._jobs.submit(self._proxy_job, media_id))

    def _proxy_job(self, media_id: str) -> None:
        item = self.bus.project.media.get(media_id)
        if item is None:
            return
        rel = f"cache/proxies/{media_id}.mp4"
        last = [-1]

        def progress(frac: float) -> None:
            pct = int(frac * 100)
            if pct >= last[0] + 5:  # throttle events to every 5%
                last[0] = pct
                self.bus.emit({"type": "proxy_progress", "media_id": media_id, "progress": frac})

        try:
            encoder = make_proxy(Path(item.path), self.paths.root / rel, progress)
        except (ProxyError, OSError) as exc:
            log.warning("proxy failed for %s: %s", item.path, exc)
            self.bus.patch_media(media_id, proxy_status=ProxyStatus.FAILED)
            return
        log.info("proxy ready for %s (%s)", item.name, encoder)
        self.bus.patch_media(media_id, proxy_path=rel, proxy_status=ProxyStatus.READY)

    def _resume_proxies(self) -> None:
        for item in self.bus.project.media.values():
            if item.kind == MediaKind.VIDEO and item.proxy_status in (ProxyStatus.PENDING, ProxyStatus.FAILED):
                self._queue_proxy(item.id)

    def _autosave(self, _event: dict[str, Any]) -> None:
        if self.bus.dirty:
            self.save()
