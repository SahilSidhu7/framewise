"""The desktop engine exits when the app that started it is gone (no orphaned engines)."""

from __future__ import annotations

import os
import subprocess
import sys
import time

from aieditor_engine.__main__ import _alive


def test_alive() -> None:
    assert _alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not _alive(p.pid)


def test_engine_exits_when_parent_dies() -> None:
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
    child = subprocess.Popen([sys.executable, "-c",
                              f"from aieditor_engine.__main__ import watch_parent; watch_parent({parent.pid}, 0.2);"
                              "import time; time.sleep(60)"])
    try:
        parent.wait()
        start = time.time()
        child.wait(timeout=10)
        assert time.time() - start < 5
    finally:
        if child.poll() is None:
            child.kill()
