"""Where automation keeps its state: keys, uploads, jobs and the shared analysis cache."""

from __future__ import annotations

import os
from pathlib import Path


def automation_home(override: Path | str | None = None) -> Path:
    """--home, else FRAMEWISE_HOME, else <data folder>/automation."""
    from ..paths import data_root, env

    if override:
        base = Path(override)
    elif env("HOME"):
        base = Path(env("HOME"))  # type: ignore[arg-type]
    else:
        base = data_root() / "automation"
    base = base.expanduser()
    for sub in ("jobs", "uploads", "cache"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base
