"""Helper processes bound to the engine die with it, even when the engine is killed."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from aieditor_engine.__main__ import _alive


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are Windows-only")
def test_helper_dies_when_engine_is_killed(tmp_path) -> None:
    pid_file = tmp_path / "child.pid"
    engine = subprocess.Popen([sys.executable, "-c", (
        "import subprocess, sys, time\n"
        "from aieditor_engine.procs import bind_to_engine\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "assert bind_to_engine(c)\n"
        f"open(r'{pid_file}', 'w').write(str(c.pid))\n"
        "time.sleep(120)\n"
    )])
    try:
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text():
                break
            time.sleep(0.1)
        child = int(pid_file.read_text())
        assert _alive(child)
        engine.kill()  # like Task Manager's End task: no cleanup code runs
        engine.wait(10)
        for _ in range(50):
            if not _alive(child):
                break
            time.sleep(0.1)
        assert not _alive(child)
    finally:
        if engine.poll() is None:
            engine.kill()
