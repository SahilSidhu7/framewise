"""Signed webhooks for finished jobs.

Each delivery is a POST with a JSON body and these headers:
    X-AIEditor-Event:     job.done | job.failed | job.cancelled
    X-AIEditor-Job:       the job id
    X-AIEditor-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256(secret, "<t>.<raw body>")>

Receivers should recompute the HMAC over the raw body, compare in constant
time and reject timestamps older than a few minutes (see verify_signature).
Failed deliveries are retried after 2, 10 and 30 seconds.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

RETRY_DELAYS = (2.0, 10.0, 30.0)
TIMEOUT = 10.0


def sign(secret: str, body: bytes, ts: int | None = None) -> str:
    ts = int(time.time()) if ts is None else ts
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify_signature(secret: str, body: bytes, header: str, tolerance: float = 300) -> bool:
    """Receiver side check (also used by the tests and documented for n8n Code nodes)."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
    except (ValueError, KeyError):
        return False
    if abs(time.time() - ts) > tolerance:
        return False
    expected = sign(secret, body, ts).split("v1=", 1)[1]
    return hmac.compare_digest(expected, parts.get("v1", ""))


def deliver(url: str, secret: str, event: str, payload: dict[str, Any],
            on_result: Callable[[dict[str, Any]], None] | None = None, delays: tuple[float, ...] = RETRY_DELAYS) -> threading.Thread:
    """POST in a background thread with retries. on_result gets {delivered, attempts, status, error}."""
    body = json.dumps({"event": event, **payload}, separators=(",", ":")).encode()

    def run() -> None:
        attempts, status, error = 0, None, None
        for wait in (0.0, *delays):
            if wait:
                time.sleep(wait)
            attempts += 1
            req = urllib.request.Request(url, data=body, method="POST", headers={
                "Content-Type": "application/json", "User-Agent": "AIEditor-Webhook/1",
                "X-AIEditor-Event": event, "X-AIEditor-Job": str(payload.get("job", {}).get("id", "")),
                "X-AIEditor-Signature": sign(secret, body),
            })
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - http(s) checked by caller
                    status = resp.status
                if 200 <= status < 300:
                    error = None
                    break
                error = f"HTTP {status}"
            except urllib.error.HTTPError as exc:
                status, error = exc.code, f"HTTP {exc.code}"
                if 400 <= exc.code < 500 and exc.code not in (408, 429):
                    break  # the receiver rejected it; retrying will not help
            except (urllib.error.URLError, OSError) as exc:
                error = str(getattr(exc, "reason", exc))
            log.info("webhook %s attempt %d failed: %s", url, attempts, error)
        if on_result:
            on_result({"delivered": error is None, "attempts": attempts, "status": status, "error": error})

    t = threading.Thread(target=run, name="webhook", daemon=True)
    t.start()
    return t
