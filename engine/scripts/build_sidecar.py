"""Freeze the engine into a single binary for the Tauri bundle.

Output: app/src-tauri/binaries/framewise-engine-<rust target triple>[.exe]
(Tauri's externalBin naming), picked up by `tauri build --config
src-tauri/tauri.bundle.conf.json`.

Usage (from engine/):  uv run --group build python scripts/build_sidecar.py

Licensing note: the PyAV wheel bundles an FFmpeg that includes GPL parts
(libx264). Before a public release, swap in an LGPL-only FFmpeg build (see PLAN.md §2).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
WORK = ENGINE / "build"
BIN_DIR = ENGINE.parent / "app" / "src-tauri" / "binaries"


def target_triple() -> str:
    out = subprocess.run(["rustc", "-vV"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("could not read host triple from `rustc -vV`")


def main() -> None:
    triple = target_triple()
    WORK.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm", "--clean", "--onefile",
            "--name", "framewise-engine",
            "--collect-submodules", "aieditor_engine",
            "--collect-data", "aieditor_engine",  # looks (recipes/*.yaml)
            "--collect-all", "av",
            "--collect-submodules", "uvicorn",
            "--exclude-module", "tkinter", "--exclude-module", "matplotlib", "--exclude-module", "pytest",
            str(ENGINE / "scripts" / "sidecar_entry.py"),
        ],
        cwd=WORK,
        check=True,
    )
    ext = ".exe" if sys.platform == "win32" else ""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    dest = BIN_DIR / f"framewise-engine-{triple}{ext}"
    shutil.copy2(WORK / "dist" / f"framewise-engine{ext}", dest)
    print(f"sidecar -> {dest}")


if __name__ == "__main__":
    main()
