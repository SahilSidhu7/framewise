"""Helper processes (llama-server, whisper) must never outlive the engine.

`bind_to_engine(popen)` puts a child process into a Windows job object created with
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. The engine holds the only handle to that job; when the
engine exits for any reason (closed, crashed, killed from Task Manager) Windows closes the
handle and ends every process in the job. On other systems this is a no-op (the child is
started in the engine's process group and `atexit` stops it on a normal exit).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading

log = logging.getLogger(__name__)
_job = None
_lock = threading.Lock()


def _create_job():  # noqa: ANN202
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # 9 = ExtendedLimitInformation
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    return job


def bind_to_engine(proc: subprocess.Popen) -> bool:
    """Make `proc` end when this engine process ends. Returns True if bound."""
    global _job
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    try:
        with _lock:
            if _job is None:
                _job = _create_job()
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        handle = wintypes.HANDLE(int(proc._handle))  # type: ignore[attr-defined]
        if not k32.AssignProcessToJobObject(_job, handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        return True
    except (OSError, AttributeError, ValueError) as exc:
        log.warning("could not tie helper process %s to the engine: %s", proc.pid, exc)
        return False
