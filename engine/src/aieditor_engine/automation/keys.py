"""Named API keys for the public API.

A key looks like `aie_<id>_<secret>`. Only a SHA-256 hash of the whole key is
stored (keys.json in the automation home), so a copied keys file cannot be
used to call the API. The full key is shown once, when it is created.

Scopes:
    jobs   create, read, cancel jobs; upload inputs; download results
    local  read input files from, and write results to, paths on this computer
    admin  manage keys and read the webhook secret
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any

SCOPES = ("jobs", "local", "admin")
DEFAULT_SCOPES = ("jobs",)


class KeyError_(ValueError):
    pass


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KeyStore:
    def __init__(self, home: Path) -> None:
        self.path = home / "keys.json"
        self.secret_path = home / "webhook_secret"
        self._lock = threading.Lock()
        self._keys: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get("keys", [])
        except (OSError, ValueError):
            return []

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"keys": self._keys}, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def create(self, name: str, scopes: list[str] | tuple[str, ...] = DEFAULT_SCOPES) -> tuple[str, dict[str, Any]]:
        """Returns (full key, public record). The full key is not stored."""
        bad = [s for s in scopes if s not in SCOPES]
        if bad:
            raise KeyError_(f"unknown scope {bad[0]!r}; use {', '.join(SCOPES)}")
        name = name.strip()[:60] or "key"
        kid = secrets.token_hex(4)
        key = f"aie_{kid}_{secrets.token_urlsafe(24)}"
        rec = {"id": kid, "name": name, "hash": _hash(key), "scopes": sorted(set(scopes)),
               "created": round(time.time()), "last_used": None}
        with self._lock:
            self._keys.append(rec)
            self._save()
        return key, self.public(rec)

    @staticmethod
    def public(rec: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in rec.items() if k != "hash"}

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.public(r) for r in self._keys]

    def revoke(self, kid: str) -> bool:
        with self._lock:
            n = len(self._keys)
            self._keys = [r for r in self._keys if r["id"] != kid]
            if len(self._keys) != n:
                self._save()
                return True
        return False

    def verify(self, key: str) -> dict[str, Any] | None:
        """The key's record if valid, else None. Constant-time on the hash."""
        if not key.startswith("aie_") or key.count("_") < 2:
            return None
        kid = key.split("_", 2)[1]
        h = _hash(key)
        with self._lock:
            for rec in self._keys:
                if rec["id"] == kid and secrets.compare_digest(rec["hash"], h):
                    now = round(time.time())
                    if not rec.get("last_used") or now - rec["last_used"] > 60:
                        rec["last_used"] = now
                        self._save()
                    return self.public(rec)
        return None

    def webhook_secret(self) -> str:
        """Secret used to sign webhooks when a job does not bring its own. Created on first use."""
        with self._lock:
            try:
                s = self.secret_path.read_text(encoding="utf-8").strip()
                if s:
                    return s
            except OSError:
                pass
            s = "whsec_" + secrets.token_urlsafe(32)
            self.secret_path.write_text(s, encoding="utf-8")
            return s
