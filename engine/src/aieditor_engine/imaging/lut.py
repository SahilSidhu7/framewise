"""Adobe/Resolve .cube 3D LUT parsing and trilinear application."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_cube(path: Path) -> np.ndarray:
    """Return a (N, N, N, 3) float32 table indexed [b, g, r] (cube files vary red fastest)."""
    size = None
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0].upper()
        if key == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif key in ("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE"):
            if key == "LUT_1D_SIZE":
                raise ValueError("1D LUTs are not supported")
            continue
        else:
            rows.append([float(v) for v in line.split()[:3]])
    if size is None or len(rows) != size**3:
        raise ValueError(f"malformed .cube file: {path.name}")
    return np.asarray(rows, dtype=np.float32).reshape(size, size, size, 3)


def apply_lut3d(img: np.ndarray, table: np.ndarray) -> np.ndarray:
    n = table.shape[0] - 1
    pos = np.clip(img, 0, 1) * n
    i0 = np.clip(np.floor(pos).astype(np.int32), 0, n - 1)
    f = (pos - i0).astype(np.float32)
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]

    def t(db: int, dg: int, dr: int) -> np.ndarray:
        return table[b0 + db, g0 + dg, r0 + dr]

    c00 = t(0, 0, 0) * (1 - fr) + t(0, 0, 1) * fr
    c01 = t(0, 1, 0) * (1 - fr) + t(0, 1, 1) * fr
    c10 = t(1, 0, 0) * (1 - fr) + t(1, 0, 1) * fr
    c11 = t(1, 1, 0) * (1 - fr) + t(1, 1, 1) * fr
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    return (c0 * (1 - fb) + c1 * fb).astype(np.float32)


def write_cube(path: Path, table: np.ndarray, title: str = "AIEditor") -> None:
    n = table.shape[0]
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {n}"]
    for b in range(n):
        for g in range(n):
            for r in range(n):
                lines.append("{:.6f} {:.6f} {:.6f}".format(*table[b, g, r]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
