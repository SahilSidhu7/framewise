"""Where automation keeps its state: keys, uploads, jobs and the shared analysis cache."""

from __future__ import annotations

import os
from pathlib import Path


def automation_home(override: Path | str | None = None) -> Path:
    """AIEDITOR_HOME, else %LOCALAPPDATA%/AIEditor/automation, else ~/.aieditor/automation."""
    if override:
        base = Path(override)
    elif os.environ.get("AIEDITOR_HOME"):
        base = Path(os.environ["AIEDITOR_HOME"])
    elif os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]) / "AIEditor" / "automation"
    else:
        base = Path.home() / ".aieditor" / "automation"
    base = base.expanduser()
    for sub in ("jobs", "uploads", "cache"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base
