"""Tighten speech: cut silences and filler words using word timings."""

from __future__ import annotations

import re
from typing import Any

FILLERS = re.compile(r"^(u+m+|u+h+|e+r+m*|a+h+|h+m+|m+h+m+)[.,!?…]*$", re.IGNORECASE)


def keep_ranges(
    transcript: dict[str, Any] | None, a: float, b: float, *, max_gap: float = 0.45, pad: float = 0.12,
    fillers: bool = True, min_len: float = 0.3,
) -> list[tuple[float, float]]:
    """Source ranges inside [a, b] worth keeping: runs of words with pauses
    longer than max_gap (and filler words) removed. Without word timings the
    whole range is kept."""
    words = []
    for w in (transcript or {}).get("words", []):
        s, e = float(w["start"]), float(w["end"])
        text = str(w.get("text", "")).strip()
        if not text or (s + e) / 2 < a or (s + e) / 2 > b:
            continue
        if fillers and FILLERS.match(text):
            continue
        words.append((s, e))
    if not words:
        return [(a, b)]
    runs: list[list[float]] = [[words[0][0], words[0][1]]]
    for s, e in words[1:]:
        if s - runs[-1][1] > max_gap:
            runs.append([s, e])
        else:
            runs[-1][1] = max(runs[-1][1], e)
    out: list[tuple[float, float]] = []
    for s, e in runs:
        s, e = max(a, s - pad), min(b, e + pad)
        if out and s <= out[-1][1] + 0.05:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        elif e - s >= min_len:
            out.append((s, e))
    return out or [(a, b)]


def saved_seconds(ranges: list[tuple[float, float]], a: float, b: float) -> float:
    return max(0.0, (b - a) - sum(e - s for s, e in ranges))
