"""Subject cut-out with an optional model on onnxruntime: ISNet (Apache-2.0, the default; ~0.7 s,
1.2 GB) or BiRefNet lite (MIT, high quality; ~8 s, 6.5 GB peak, used when installed and memory allows).

    prob = subject_probability(rgb_u8)   # float32 0..1 per pixel, or None if the model isn't installed

Input preparation follows the reference implementation: resize to 1024x1024, scale by the
image maximum, ImageNet mean/std, NCHW; output passes through a sigmoid and is min-max
normalised. One inference takes about 1-3 s on a CPU, so results are cached by image content.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict

import cv2
import numpy as np

log = logging.getLogger(__name__)

SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_session = None
_session_path = None
_lock = threading.Lock()
_cache: OrderedDict[str, np.ndarray] = OrderedDict()


def _choose() -> tuple[str, object] | None:
    """The best installed model that fits the memory available right now."""
    from ..models import CATALOG, path_of

    free = _free_gb()
    for name in ("subject-hq", "subject"):
        if path_of(name) is None:
            continue
        need = 7.0 if name == "subject-hq" else 1.5
        if free is not None and free < need:
            log.info("skipping %s: only %.1f GB of memory available", name, free)
            continue
        return name, CATALOG[name]
    return None


def _get_session(path):  # noqa: ANN001, ANN202
    global _session, _session_path
    with _lock:
        if _session is None or _session_path != path:
            _session = _open(path, gpu=True)
            _session_path = path
        return _session


def _open(path, gpu: bool):  # noqa: ANN001, ANN202
    """CPU by default (about 11 s per new photo on an 8-core laptop, then cached).

    GPU through DirectML is opt-in (FRAMEWISE_SUBJECT_GPU=1 with the onnxruntime-directml build):
    it halves the time, but on shared or busy GPUs it ran out of memory and left the process
    short of memory, so it is not the default. DirectML needs memory patterns off, sequential
    execution and only basic graph fusion (full fusion fails on this model)."""
    import onnxruntime as ort

    from ..paths import env

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
    if gpu and env("SUBJECT_GPU") == "1" and "DmlExecutionProvider" in ort.get_available_providers():
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        try:
            return ort.InferenceSession(str(path), sess_options=opts, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        except Exception:  # noqa: BLE001
            log.warning("GPU (DirectML) unavailable for the subject model; using the CPU")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
    # no memory arena: with it, one run kept ~9 GB committed (17 GB peak); without, memory returns to
    # ~0.8 GB after each run (6.5 GB peak during it) and the run is faster
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def available() -> bool:
    from ..models import path_of

    return path_of("subject") is not None or path_of("subject-hq") is not None


def _free_gb() -> float | None:
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = Status()
        st.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return st.ullAvailPageFile / 2**30  # memory Windows can still commit (RAM + page file)
    except (AttributeError, OSError):
        pass
    return None


def subject_probability(rgb_u8: np.ndarray) -> np.ndarray | None:
    """Per-pixel probability that a pixel belongs to the main subject(s), at the input size."""
    if not available():
        return None
    h, w = rgb_u8.shape[:2]
    small = rgb_u8
    if max(h, w) > 1400:  # the model sees 1024 px anyway; hash and resize a smaller copy
        s = 1400 / max(h, w)
        small = cv2.resize(rgb_u8, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    key = hashlib.blake2b(cv2.resize(small, (256, 256), interpolation=cv2.INTER_AREA).tobytes(), digest_size=16).hexdigest()
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
    if hit is None:
        choice = _choose()
        if choice is None:
            log.warning("no subject model fits the memory available; using the built-in subject finder")
            return None
        name, info = choice
        from ..models import path_of

        sess = _get_session(path_of(name))
        x = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        x = x / max(float(x.max()), 1e-6)
        if info.prep == "birefnet":
            x = (x - MEAN) / STD
        else:
            x = x - 0.5
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        try:
            out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
        except Exception:  # noqa: BLE001 - a GPU driver hiccup: retry once on the CPU, else fall back to GrabCut
            log.warning("subject model failed on the current device; retrying on the CPU")
            global _session
            try:
                with _lock:
                    _session = _open(_session_path, gpu=False)
                    sess = _session
                out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
            except Exception:  # noqa: BLE001
                log.exception("subject model failed")
                return None
        pred = 1.0 / (1.0 + np.exp(-out)) if info.prep == "birefnet" else out
        lo, hi = float(pred.min()), float(pred.max())
        hit = ((pred - lo) / max(hi - lo, 1e-6)).astype(np.float32)
        with _lock:
            _cache[key] = hit
            while len(_cache) > 12:
                _cache.popitem(last=False)
    return cv2.resize(hit, (w, h), interpolation=cv2.INTER_LINEAR)
