"""An open project: bus + storage + background media jobs."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .core import builtin, photo_commands, video_commands  # noqa: F401  (register commands)
from .core.bus import CommandBus, Source
from .core.project import MediaKind, Project, ProjectKind, ProxyStatus
from .core.store import ProjectPaths, create_project, load_project, save_project
from .imaging.render import SOURCES, analyze
from .media.probe import ProbeError, make_thumbnail, probe
from .media.proxy import ProxyError, make_proxy

log = logging.getLogger(__name__)


ANALYSIS_VERSION = 3


class Session:
    def __init__(self, root: Path, project: Project, *, proxy_workers: int = 1) -> None:
        self.paths = ProjectPaths(root)
        self.paths.ensure()
        self.bus = CommandBus(project, log_path=self.paths.trajectory_log)
        self.bus.subscribe(self._autosave)
        self._jobs = ThreadPoolExecutor(max_workers=proxy_workers, thread_name_prefix="proxy")
        self._pending: list[Future] = []
        self._analysis: dict[str, dict[str, Any]] = {}
        self._video_lock = threading.Lock()
        self._render_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render-job")
        self.jobs: dict[str, dict[str, Any]] = {}
        self._cancel: dict[str, threading.Event] = {}

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
        for ev in list(self._cancel.values()):
            ev.set()
        self._jobs.shutdown(wait=wait, cancel_futures=not wait)
        self._render_pool.shutdown(wait=wait, cancel_futures=not wait)
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

    # --------------------------------------------------------------- video
    def _cache_json(self, kind: str, media_id: str) -> Path:
        d = self.paths.root / "cache" / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{media_id}.json"

    def _cached(self, kind: str, media_id: str, compute: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
        """Per-media analysis, cached in memory and in cache/<kind>/<id>.json (keyed on file mtime)."""
        item = self.bus.project.media.get(media_id)
        if item is None:
            raise KeyError(f"media not found: {media_id}")
        key = f"{kind}:{media_id}"
        if key in self._analysis:
            return self._analysis[key]
        path = self._cache_json(kind, media_id)
        src = Path(item.path)
        # bump ANALYSIS_VERSION when an analyzer changes so stale cache files are recomputed
        stamp = f"{src.stat().st_mtime_ns if src.exists() else 0}:v{ANALYSIS_VERSION}"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("_stamp") == stamp:
                    self._analysis[key] = data
                    return data
            except (OSError, ValueError):
                pass
        data = compute(src)
        data["_stamp"] = stamp
        path.write_text(json.dumps(data), encoding="utf-8")
        self._analysis[key] = data
        return data

    def video_analysis(self, media_id: str) -> dict[str, Any]:
        from .video.analysis import analyze_audio_track, analyze_video

        def compute(path: Path) -> dict[str, Any]:
            data = analyze_video(path)
            data["audio"] = analyze_audio_track(path)
            return data

        return self._cached("analysis", media_id, compute)

    def subject_track(self, media_id: str) -> dict[str, Any] | None:
        """Where the subject is over time (for auto-reframe); None for non-videos."""
        from .video.reframe import subject_track

        item = self.bus.project.media.get(media_id)
        if item is None or item.kind != MediaKind.VIDEO:
            return None
        with self._video_lock:
            return self._cached("subjects", media_id, subject_track)

    @property
    def styles_dir(self) -> Path:
        """Styles saved for this project ("edit like this")."""
        return self.paths.root / "styles"

    def music_analysis(self, media_id: str) -> dict[str, Any]:
        from .video.analysis import analyze_music

        return self._cached("music", media_id, analyze_music)

    def transcript(self, media_id: str, compute: bool = True) -> dict[str, Any] | None:
        """Speech transcript of a video/audio file. compute=False only reads the cache."""
        from .agent.whisper import transcribe

        if compute:
            return self._cached("transcripts", media_id, transcribe)
        key = f"transcripts:{media_id}"
        if key in self._analysis:
            return self._analysis[key]
        path = self._cache_json("transcripts", media_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        self._analysis[key] = data
        return data

    def transcripts_for_timeline(self, project: Project | None = None) -> dict[str, dict[str, Any]]:
        """Cached transcripts of every video on the timeline (no new speech recognition)."""
        project = project or self.bus.project
        out: dict[str, dict[str, Any]] = {}
        for media_id, item in project.media.items():
            if item.kind == MediaKind.VIDEO:
                tr = self.transcript(media_id, compute=False)
                if tr:
                    out[media_id] = {k: v for k, v in tr.items() if k in ("words", "segments")}
        return out

    def video_context(self, transcribe_speech: bool = False) -> Any:
        """Everything the video planner needs (analyses are cached per media)."""
        from .agent.video_planner import MediaInfo, VideoContext
        from .agent.whisper import WhisperUnavailable, whisper_status
        from .video.vdsl import timeline_to_dsl

        with self.bus.lock:
            project = self.bus.project.model_copy(deep=True)
        asr = transcribe_speech and whisper_status()["available"]
        infos = []
        with self._video_lock:
            for n, (mid, item) in enumerate(project.media.items(), start=1):
                if item.kind == MediaKind.VIDEO:
                    a = self.video_analysis(mid)
                    tr = self.transcript(mid, compute=False)
                    if tr is None and asr and a.get("audio", {}).get("speech_likely"):
                        try:
                            tr = self.transcript(mid)
                        except WhisperUnavailable as exc:
                            log.warning("transcription failed: %s", exc)
                    infos.append(MediaInfo(n, mid, item.name, "video", item.duration or a["duration"],
                                           a.get("orientation", "landscape"), a, a.get("audio", {}), tr))
                elif item.kind == MediaKind.IMAGE:
                    portrait = (item.height or 0) > (item.width or 0) * 1.05
                    infos.append(MediaInfo(n, mid, item.name, "image", 3.0, "portrait" if portrait else "landscape"))
                else:
                    mu = self.music_analysis(mid)
                    infos.append(MediaInfo(n, mid, item.name, "audio", item.duration or mu["duration"], music=mu))
        has_edit = any(t.clips for t in project.tracks)
        return VideoContext(infos, timeline_to_dsl(project) if has_edit else "", subject_of=self.subject_track)

    # ---------------------------------------------------------- render jobs
    def start_job(self, kind: str, fn: Callable[..., Any], *, label: str = "") -> str:
        """Run fn(progress, cancelled) on the render thread. Emits job_progress /
        job_done / job_failed / job_cancelled events. A new preview cancels older ones."""
        if kind == "preview":
            for jid, job in self.jobs.items():
                if job["kind"] == "preview" and job["state"] in ("queued", "running") and jid in self._cancel:
                    self._cancel[jid].set()
        job_id = uuid.uuid4().hex[:10]
        cancel = threading.Event()
        self._cancel[job_id] = cancel
        job: dict[str, Any] = {"id": job_id, "kind": kind, "label": label, "state": "queued", "progress": 0.0,
                               "result": None, "error": None, "revision": self.bus.revision, "started": time.time()}
        self.jobs[job_id] = job
        last = [0.0]

        def progress(frac: float) -> None:
            job["progress"] = frac
            now = time.time()
            if now - last[0] > 0.25 or frac >= 1:
                last[0] = now
                self.bus.emit({"type": "job_progress", "job": job_id, "kind": kind, "progress": round(frac, 4)})

        def run() -> None:
            from .video.export import Cancelled

            if cancel.is_set():
                job["state"] = "cancelled"
                self._cancel.pop(job_id, None)
                return
            job["state"] = "running"
            try:
                job["result"] = fn(progress, cancel.is_set)
                job["state"] = "done"
            except Cancelled:
                job["state"] = "cancelled"
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                job["state"], job["error"] = "failed", str(exc)
                log.exception("job %s failed", job_id)
            finally:
                self._cancel.pop(job_id, None)
            self.bus.emit({"type": f"job_{job['state']}", "job": job_id, "kind": kind,
                           "result": job["result"], "error": job["error"]})
            finished = [k for k, j in self.jobs.items() if j["state"] in ("done", "failed", "cancelled")]
            for old in finished[:-20]:
                self.jobs.pop(old, None)

        self._render_pool.submit(run)
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        ev = self._cancel.get(job_id)
        if ev is not None:
            ev.set()
        return ev is not None

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
        try:  # warm the AI's measurements so the first request is fast
            self.video_analysis(media_id)
        except Exception:  # noqa: BLE001 - analysis is optional here; it is retried on demand
            log.exception("analysis failed for %s", item.path)

    def _resume_proxies(self) -> None:
        for item in self.bus.project.media.values():
            if item.kind == MediaKind.VIDEO and item.proxy_status in (ProxyStatus.PENDING, ProxyStatus.FAILED):
                self._queue_proxy(item.id)

    def _autosave(self, _event: dict[str, Any]) -> None:
        if self.bus.dirty:
            self.save()
