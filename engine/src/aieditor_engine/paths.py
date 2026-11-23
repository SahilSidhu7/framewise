"""Where Framewise keeps its data, and the settings environment variables.

Data folder: %LOCALAPPDATA%/Framewise (Windows), ~/Library/Application Support/Framewise
(macOS), $XDG_DATA_HOME/Framewise (Linux). It holds models/, runtime/ (whisper, llama),
automation/ and app.json.

The product used to be called AIEditor. On first use, data found in the old
%LOCALAPPDATA%/AIEditor folder (models, runtimes, jobs, settings) is moved
into the new folder, so nothing has to be downloaded again. Program
files of an old install are left where they are.

Environment variables are read as FRAMEWISE_<NAME>, falling back to the old
AIEDITOR_<NAME> spelling.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
from pathlib import Path

log = logging.getLogger(__name__)

APP = "Framewise"
LEGACY = "AIEditor"
MOVABLE = ("models", "runtime", "automation", "dl", "app.json")
_lock = threading.Lock()
_migrated = False


def env(name: str) -> str | None:
    """FRAMEWISE_<name>, or the legacy AIEDITOR_<name>."""
    return os.environ.get(f"FRAMEWISE_{name}") or os.environ.get(f"AIEDITOR_{name}") or None


def _base() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def migrate_legacy(new: Path, old: Path) -> list[str]:
    """Move data entries from the old AIEditor folder into the new one (never overwriting)."""
    moved: list[str] = []
    if not old.is_dir() or old.resolve() == new.resolve():
        return moved
    new.mkdir(parents=True, exist_ok=True)
    for name in MOVABLE:
        src, dst = old / name, new / name
        if not src.exists() or dst.exists():
            continue
        try:
            shutil.move(str(src), str(dst))
            moved.append(name)
        except OSError as exc:  # in use by a running old engine: leave it, try again next start
            log.warning("could not move %s from the old AIEditor folder: %s", name, exc)
    if moved:
        log.info("moved %s from %s to %s", ", ".join(moved), old, new)
    return moved


def data_root() -> Path:
    """The per-user data folder (created on first use; old AIEditor data is moved in once)."""
    global _migrated
    root = _base() / APP
    with _lock:
        if not _migrated:
            _migrated = True
            try:
                migrate_legacy(root, _base() / LEGACY)
            except OSError as exc:
                log.warning("data migration skipped: %s", exc)
        root.mkdir(parents=True, exist_ok=True)
    return root


def config_dir() -> Path:
    """FRAMEWISE_CONFIG, else the data folder (paths.data_root); shared by the app and the CLI."""
    base = Path(env("CONFIG")).expanduser() if env("CONFIG") else data_root()
    base.mkdir(parents=True, exist_ok=True)
    return base
