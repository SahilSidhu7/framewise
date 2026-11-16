"""First-run system check and the app's own small settings file.

`system_info()` answers "what can this PC do?" in plain terms for onboarding: processor,
memory, free disk space, whether the NVIDIA hardware encoder works, and which optional AI
parts are installed. `benchmark()` times the real per-frame work of a 1080p export (colour
grade + captions) for about a second and turns it into an estimate a person understands:
how long a 30-second reel takes on this machine.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .licence import config_dir

REEL_SECONDS = 30
REEL_FPS = 30


# ---------------------------------------------------------------- hardware
def memory_gb() -> float | None:
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MemoryStatus()
            st.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
                return round(st.ullTotalPhys / 1024**3, 1)
            return None
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3, 1)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "Unknown processor"


_encoder_cache: dict[str, bool] = {}


def hardware_encoder() -> str | None:
    """'nvenc' when the NVIDIA encoder works, else None (exports then use the CPU)."""
    if "nvenc" not in _encoder_cache:
        from .video.export import encoder_works

        _encoder_cache["nvenc"] = encoder_works("h264_nvenc", 640, 360)
    return "nvenc" if _encoder_cache["nvenc"] else None


def ai_parts() -> dict[str, Any]:
    from .agent.llm import LLM
    from .agent.whisper import whisper_status
    from .video.reframe import _face_model

    llm = LLM.status()
    speech = whisper_status()
    return {
        "editing_model": {"ready": llm["state"] != "not_installed", "name": llm.get("model")},
        "speech": {"ready": bool(speech["available"]), "name": speech.get("model")},
        "face_model": {"ready": _face_model() is not None},
    }


def system_info() -> dict[str, Any]:
    folder = config_dir()
    free = shutil.disk_usage(folder).free / 1024**3
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_name(),
        "cores": os.cpu_count() or 1,
        "memory_gb": memory_gb(),
        "disk_free_gb": round(free, 1),
        "hardware_encoder": hardware_encoder(),
        "ai": ai_parts(),
        "data_folder": str(folder),
    }


# ---------------------------------------------------------------- benchmark
_bench_lock = threading.Lock()


def benchmark(seconds: float = 1.0) -> dict[str, Any]:
    """Time the per-frame export work at 1080x1920 and estimate a 30 s reel."""
    from .video.fastgrade import GradePlan
    from .video.text import draw_text

    with _bench_lock:
        rng = np.random.default_rng(0)
        src = (rng.random((1920, 1080, 3), dtype=np.float32) ** 2.2).astype(np.float32)
        plan = GradePlan([{"op": "look", "params": {"name": "vivid_pop", "strength": 0.75}}])
        plan.finish(src[:64, :64], 0)  # build the LUT outside the timed loop
        frames, start = 0, time.perf_counter()
        while frames < 3 or (time.perf_counter() - start < seconds and frames < 60):
            out = plan.finish(src, frames)
            draw_text(out if out.flags.writeable else out.copy(), "WORD BY WORD CAPTIONS", "bold_pop", None, 0.5, 1.0,
                      words=[0.0, 0.3, 0.6, 0.9])
            frames += 1
        per_frame = (time.perf_counter() - start) / frames
    workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    # decode + encode roughly double the per-frame cost; segments run in parallel
    reel_seconds = REEL_SECONDS * REEL_FPS * per_frame * 2.0 / workers * 1.15
    speed = (REEL_SECONDS / reel_seconds) if reel_seconds else 0.0
    tier = "fast" if speed >= 0.5 else "good" if speed >= 0.12 else "basic"
    minutes = reel_seconds / 60
    when = "under a minute" if minutes < 1 else f"about {max(1, round(minutes))} minute{'s' if round(minutes) > 1 else ''}"
    return {
        "ms_per_frame": round(per_frame * 1000, 1),
        "workers": workers,
        "reel_seconds": round(reel_seconds),
        "tier": tier,
        "summary": {
            "fast": f"Fast. A 30-second reel exports in {when}.",
            "good": f"Good. A 30-second reel exports in {when}.",
            "basic": f"Works, but slowly: a 30-second reel takes {when}. Use Draft quality while you experiment.",
        }[tier],
    }


# ---------------------------------------------------------------- app settings
SETTINGS_DEFAULTS: dict[str, Any] = {"onboarded": False, "focus": None}
SETTINGS_KEYS = {"onboarded": bool, "focus": (str, type(None))}
FOCUS = ("photos", "videos", "reels", "automation")


class AppSettings:
    """app.json in the config folder (survives reinstalls; shared with the CLI)."""

    def __init__(self, folder: Path | None = None) -> None:
        self.path = Path(folder or config_dir()) / "app.json"
        self._lock = threading.Lock()

    def get(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        return {k: data.get(k, v) for k, v in SETTINGS_DEFAULTS.items()}

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        for k, v in changes.items():
            if k not in SETTINGS_KEYS or not isinstance(v, SETTINGS_KEYS[k]):
                raise ValueError(f"unknown or invalid setting `{k}`")
            if k == "focus" and v is not None and v not in FOCUS:
                raise ValueError(f"focus must be one of {', '.join(FOCUS)}")
        with self._lock:
            data = {**self.get(), **changes}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        return data
