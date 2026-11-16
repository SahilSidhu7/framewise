"""Watch folders: drop a file in, get the edit out. No code.

Every new media file that appears in the folder (and has stopped growing) is
sent through a recipe; results land in `<folder>/output/<file name>/`.
The recipe comes from `aieditor-watch.yaml` in the folder, e.g.

    recipe: reel
    style: energetic_reel
    output: {platform: tiktok, length: 30}
    options: {count: 3}

or from the command line (`aieditor watch DIR --recipe captions`).
Processed files are remembered in `.aieditor-watch.json`, so restarting the
watcher does not redo them.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import yaml

from .jobs import JobError, JobManager, JobRequest

log = logging.getLogger(__name__)

MEDIA_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mts", ".m2ts", ".jpg", ".jpeg", ".png", ".webp",
              ".heic", ".tif", ".tiff", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
CONFIG = "aieditor-watch.yaml"
STATE = ".aieditor-watch.json"
OUTPUT = "output"


class FolderWatcher:
    def __init__(self, manager: JobManager, folder: Path, defaults: dict[str, Any] | None = None, interval: float = 2.0) -> None:
        self.manager = manager
        self.folder = folder.expanduser().resolve()
        if not self.folder.is_dir():
            raise FileNotFoundError(f"not a folder: {self.folder}")
        self.defaults = defaults or {}
        self.interval = interval
        self._seen_sizes: dict[str, tuple[int, int]] = {}
        self._state_path = self.folder / STATE
        self.done: dict[str, str] = self._load_state()
        self.jobs: dict[str, str] = {}  # file name -> job id

    def _load_state(self) -> dict[str, str]:
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps(self.done, indent=1), encoding="utf-8")

    def config(self) -> dict[str, Any]:
        cfg = dict(self.defaults)
        path = self.folder / CONFIG
        if path.is_file():
            try:
                cfg.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            except yaml.YAMLError as exc:
                log.warning("bad %s: %s", path, exc)
        cfg.setdefault("recipe", "reel")
        return cfg

    def scan_once(self) -> list[str]:
        """Submit every new, finished file. Returns the new job ids."""
        submitted = []
        for f in sorted(self.folder.iterdir()):
            if not f.is_file() or f.name.startswith(".") or f.suffix.lower() not in MEDIA_EXTS:
                continue
            st = f.stat()
            sig = (st.st_size, st.st_mtime_ns)
            stamp = f"{st.st_size}:{st.st_mtime_ns}"
            if self.done.get(f.name) == stamp or f.name in self.jobs:
                continue
            if self._seen_sizes.get(f.name) != sig:  # still being copied: wait one more round
                self._seen_sizes[f.name] = sig
                continue
            cfg = self.config()
            body = {k: cfg[k] for k in ("recipe", "prompt", "style", "output", "options", "webhook") if cfg.get(k) is not None}
            body.update(inputs=[str(f)], output_dir=str(self.folder / OUTPUT / f.stem), label=f"watch: {f.name}")
            try:
                job = self.manager.submit(JobRequest.model_validate(body), allow_local=True)
            except (JobError, ValueError) as exc:
                log.warning("watch folder: cannot start %s: %s", f.name, exc)
                self.done[f.name] = stamp
                self._save_state()
                continue
            self.jobs[f.name] = job["id"]
            submitted.append(job["id"])
        self._collect()
        return submitted

    def _collect(self) -> None:
        for name, jid in list(self.jobs.items()):
            try:
                job = self.manager.get(jid)
            except JobError:
                self.jobs.pop(name)
                continue
            if job["state"] in ("done", "failed", "cancelled"):
                f = self.folder / name
                if f.exists():
                    st = f.stat()
                    self.done[name] = f"{st.st_size}:{st.st_mtime_ns}"
                self.jobs.pop(name)
                self._save_state()
                if job["state"] == "failed":
                    err = self.folder / OUTPUT / Path(name).stem / "error.txt"
                    err.parent.mkdir(parents=True, exist_ok=True)
                    err.write_text(job["error"] or "failed", encoding="utf-8")
                log.info("watch folder: %s -> %s", name, job["state"])

    def run(self, stop: threading.Event) -> None:
        log.info("watching %s (recipe %s)", self.folder, self.config()["recipe"])
        while not stop.is_set():
            try:
                self.scan_once()
            except OSError as exc:
                log.warning("watch folder scan failed: %s", exc)
            stop.wait(self.interval)
