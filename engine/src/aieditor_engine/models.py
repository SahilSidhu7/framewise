"""Optional models that Framewise downloads only when the user asks.

    framewise models                    # list, with what is installed
    framewise models install subject    # download one (shows progress)

Files go to <data folder>/models (see paths.py). Every download is checked against a known
MD5 before it is used; a partial or corrupt file is never picked up.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .paths import data_root

log = logging.getLogger(__name__)
UA = {"User-Agent": "Framewise-model-download/0.1"}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    title: str
    purpose: str
    file: str
    url: str
    md5: str
    size_mb: int
    licence: str
    source: str
    min_memory_gb: int = 0
    prep: str = "isnet"  # input preparation: "isnet" or "birefnet"


CATALOG: dict[str, ModelInfo] = {
    "subject": ModelInfo(
        name="subject",
        title="Subject cut-out (ISNet)",
        purpose="Sharp subject outlines for photos and for picking subjects in video. About 0.7 s per photo, 1.2 GB of memory.",
        file="isnet-general-use.onnx",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        md5="fc16ebd8b0c10d971d3513d564d01e29",
        size_mb=170,
        licence="Apache-2.0",
        source="https://github.com/xuebinqin/DIS",
        min_memory_gb=8,
        prep="isnet",
    ),
    "subject-hq": ModelInfo(
        name="subject-hq",
        title="Subject cut-out, high quality (BiRefNet lite)",
        purpose="Finest edges (hair, whole bodies in busy scenes). About 8 s per photo and 6.5 GB of memory while it works.",
        file="BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        md5="4fab47adc4ff364be1713e97b7e66334",
        size_mb=213,
        licence="MIT",
        source="https://github.com/ZhengPeng7/BiRefNet",
        min_memory_gb=16,
        prep="birefnet",
    ),
}

_progress: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def models_dir() -> Path:
    d = data_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def path_of(name: str) -> Path | None:
    """The installed file for a catalog model, or None."""
    info = CATALOG.get(name)
    if info is None:
        return None
    p = models_dir() / info.file
    return p if p.is_file() else None


def status() -> list[dict[str, Any]]:
    from .system import memory_gb

    ram = memory_gb()
    out = []
    for name, info in CATALOG.items():
        with _lock:
            prog = dict(_progress.get(name, {}))
        fits = ram is None or ram >= info.min_memory_gb
        out.append({**asdict(info), "installed": path_of(name) is not None, "download": prog or None,
                    "memory_gb": ram, "fits": fits})
    return out


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against the publisher's checksum, not security
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def install(name: str, on_progress: Callable[[float], None] | None = None) -> Path:
    """Download and verify a catalog model (blocking). Returns its path."""
    info = CATALOG.get(name)
    if info is None:
        raise KeyError(f"unknown model `{name}`; choose one of {', '.join(CATALOG)}")
    done = path_of(name)
    if done is not None:
        return done
    dst = models_dir() / info.file
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(info.url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or info.size_mb * 1_048_576)
        got, last = 0, 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if on_progress and time.time() - last > 0.25:
                last = time.time()
                on_progress(min(0.999, got / max(total, 1)))
    if _md5(tmp) != info.md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{info.title}: the download was corrupted (checksum mismatch); please try again")
    tmp.replace(dst)
    if on_progress:
        on_progress(1.0)
    log.info("installed model %s at %s", name, dst)
    return dst


def install_in_background(name: str) -> dict[str, Any]:
    """Start a download in a thread (for the app); progress shows up in status()."""
    if name not in CATALOG:
        raise KeyError(name)
    with _lock:
        cur = _progress.get(name)
        if cur and cur.get("state") == "running":
            return cur

        _progress[name] = {"state": "running", "progress": 0.0, "error": None}

    def set_progress(p: float) -> None:
        with _lock:
            _progress[name]["progress"] = round(p, 3)

    def work() -> None:
        try:
            install(name, set_progress)
            with _lock:
                _progress[name].update(state="done", progress=1.0)
        except Exception as exc:  # noqa: BLE001 - reported to the app
            log.exception("model download failed")
            with _lock:
                _progress[name].update(state="failed", error=str(exc))

    threading.Thread(target=work, name=f"model-{name}", daemon=True).start()
    with _lock:
        return dict(_progress[name])
