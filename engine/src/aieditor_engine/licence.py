"""Offline Pro licences: Ed25519-signed keys checked on this computer, no licence server.

A key looks like ``AIE1-<payload>.<signature>`` (both base64url). The payload is JSON:
``{"v": 1, "id": "L-...", "plan": "pro"|"studio", "to": "name", "seats": 1, "iat": unix, "exp": unix|null}``.
The engine only holds the public key, so it can check keys but not make them; keys are made with
``tools/licence_issuer.py`` and the private key that never leaves the seller's machine.

Free tier: the whole desktop editor, and one-shot automation (CLI / MCP) with limits. Pro (or a
14-day trial) unlocks the headless API (`aieditor serve`), watch folders, batches, webhooks and
several reels per video.

Ed25519 is implemented here in pure Python (RFC 8032, section 6) so the engine needs no crypto
dependency. Verification is not secret-dependent, so the lack of constant-time maths does not
matter; signing is only done by the issuer on the seller's own machine.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Public half of the AIEditor licence signing key. The private half is kept outside the repository.
PUBLIC_KEY = "8853ce4b93fb60e9a88db36493717416bdb83f5991e4a4988b5e2d46cefef349"

PREFIX = "AIE1-"
TRIAL_DAYS = 14
PLANS = ("pro", "studio")
GRACE_DAYS = 0

# ---------------------------------------------------------------- Ed25519 (RFC 8032)
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _sha512(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()


def _add(p: tuple[int, int, int, int], q: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    a = (p[1] - p[0]) * (q[1] - q[0]) % _P
    b = (p[1] + p[0]) * (q[1] + q[0]) % _P
    c = 2 * p[3] * q[3] * _D % _P
    d = 2 * p[2] * q[2] % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _mul(s: int, p: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            q = _add(q, p)
        p = _add(p, p)
        s >>= 1
    return q


def _equal(p: tuple[int, int, int, int], q: tuple[int, int, int, int]) -> bool:
    return (p[0] * q[2] - q[0] * p[2]) % _P == 0 and (p[1] * q[2] - q[1] * p[2]) % _P == 0


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _I % _P
    if (x * x - x2) % _P:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * pow(5, _P - 2, _P) % _P
_GX = _recover_x(_GY, 0) or 0
_G = (_GX, _GY, 1, _GX * _GY % _P)


def _compress(p: tuple[int, int, int, int]) -> bytes:
    zinv = pow(p[2], _P - 2, _P)
    x, y = p[0] * zinv % _P, p[1] * zinv % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(s: bytes) -> tuple[int, int, int, int] | None:
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _expand(seed: bytes) -> tuple[int, bytes]:
    if len(seed) != 32:
        raise ValueError("Ed25519 seed must be 32 bytes")
    h = _sha512(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(seed: bytes) -> bytes:
    return _compress(_mul(_expand(seed)[0], _G))


def sign(seed: bytes, msg: bytes) -> bytes:
    a, prefix = _expand(seed)
    pub = _compress(_mul(a, _G))
    r = int.from_bytes(_sha512(prefix + msg), "little") % _L
    big_r = _compress(_mul(r, _G))
    h = int.from_bytes(_sha512(big_r + pub + msg), "little") % _L
    return big_r + int.to_bytes((r + h * a) % _L, 32, "little")


def verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    if len(pub) != 32 or len(sig) != 64:
        return False
    a = _decompress(pub)
    r = _decompress(sig[:32])
    if a is None or r is None:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(_sha512(sig[:32] + pub + msg), "little") % _L
    return _equal(_mul(s, _G), _add(r, _mul(h, a)))


# ---------------------------------------------------------------- keys
class LicenceError(ValueError):
    pass


class LicenceRequired(PermissionError):
    """A Pro feature was used without a Pro licence or trial."""

    def __init__(self, feature: str) -> None:
        self.feature = feature
        super().__init__(
            f"{FEATURES.get(feature, feature)} needs AIEditor Pro. Start a free {TRIAL_DAYS}-day trial with "
            "`aieditor licence trial`, or activate a key with `aieditor licence activate KEY`.")


FEATURES = {
    "serve": "The automation API (aieditor serve)",
    "watch": "Watch folders",
    "batch": "Editing several photos in one job",
    "webhook": "Webhooks",
    "reels": "More than one reel per video",
}


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


@dataclass(frozen=True)
class Licence:
    id: str
    plan: str
    to: str
    seats: int
    issued: int
    expires: int | None

    def expired(self, now: float | None = None) -> bool:
        return self.expires is not None and (now or time.time()) > self.expires + GRACE_DAYS * 86400


def make_key(seed: bytes, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return PREFIX + _b64e(body) + "." + _b64e(sign(seed, body))


def parse(key: str, public: str | None = None) -> Licence:
    """Check a key's signature and fields. Raises LicenceError; does not check expiry."""
    key = "".join(key.split())  # tolerate line breaks and spaces from copy/paste
    if not key.startswith(PREFIX) or key.count(".") != 1:
        raise LicenceError("not an AIEditor licence key")
    body_s, sig_s = key[len(PREFIX):].split(".")
    try:
        body, sig = _b64d(body_s), _b64d(sig_s)
    except (binascii.Error, ValueError) as exc:
        raise LicenceError("licence key is damaged") from exc
    if not verify(bytes.fromhex(public or PUBLIC_KEY), body, sig):
        raise LicenceError("licence key is not valid (signature check failed)")
    try:
        d = json.loads(body)
        lic = Licence(id=str(d["id"]), plan=str(d["plan"]), to=str(d.get("to", "")), seats=int(d.get("seats", 1)),
                      issued=int(d["iat"]), expires=None if d.get("exp") is None else int(d["exp"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise LicenceError("licence key has unreadable data") from exc
    if d.get("v") != 1 or lic.plan not in PLANS:
        raise LicenceError("licence key is for a different version of AIEditor")
    return lic


# ---------------------------------------------------------------- store
def config_dir() -> Path:
    """AIEDITOR_CONFIG, else %LOCALAPPDATA%/AIEditor, else ~/.aieditor (shared by the app and the CLI)."""
    if os.environ.get("AIEDITOR_CONFIG"):
        base = Path(os.environ["AIEDITOR_CONFIG"])
    elif os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]) / "AIEditor"
    else:
        base = Path.home() / ".aieditor"
    base = base.expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base


class LicenceStore:
    """licence.json in the config folder: the activated key and when the trial started.

    AIEDITOR_LICENSE (or AIEDITOR_LICENCE) can hold a key instead, for servers and containers.
    """

    def __init__(self, folder: Path | str | None = None, public: str | None = None) -> None:
        self.path = Path(folder or config_dir()) / "licence.json"
        self.public = public
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def activate(self, key: str) -> dict[str, Any]:
        lic = parse(key, self.public)
        if lic.expired():
            raise LicenceError("this licence key has expired")
        with self._lock:
            data = self._read()
            data["key"] = "".join(key.split())
            data["activated"] = int(time.time())
            self._write(data)
        return self.status()

    def deactivate(self) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            data.pop("key", None)
            self._write(data)
        return self.status()

    def start_trial(self) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            if "trial_started" in data:
                st = self.status()
                if st["tier"] != "trial":
                    raise LicenceError("the free trial has already been used on this computer")
                return st
            data["trial_started"] = int(time.time())
            self._write(data)
        return self.status()

    def status(self, now: float | None = None) -> dict[str, Any]:
        now = now or time.time()
        data = self._read()
        key = os.environ.get("AIEDITOR_LICENSE") or os.environ.get("AIEDITOR_LICENCE") or data.get("key")
        out: dict[str, Any] = {"tier": "free", "pro": False, "licence": None, "trial_days_left": None, "problem": None}
        if key:
            try:
                lic = parse(key, self.public)
                out["licence"] = asdict(lic)
                if lic.expired(now):
                    out["problem"] = "licence expired"
                else:
                    out.update(tier=lic.plan, pro=True)
                    return out
            except LicenceError as exc:
                out["problem"] = str(exc)
        if "trial_started" in data:
            left = TRIAL_DAYS - (now - float(data["trial_started"])) / 86400
            out["trial_days_left"] = max(0, int(left + 0.999))
            if left > 0:
                out.update(tier="trial", pro=True)
        return out

    def is_pro(self) -> bool:
        return bool(self.status()["pro"])

    def require(self, feature: str) -> None:
        if not self.is_pro():
            raise LicenceRequired(feature)
