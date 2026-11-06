"""Engine entry point.

The desktop app launches this as a sidecar. On startup it prints one JSON line
to stdout, `{"event": "ready", "port": ..., "token": ...}`, which the app reads
to connect. The token never touches disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import socket
import sys

import uvicorn

from .api.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aieditor-engine")
    parser.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    port = args.port or _free_port()
    token = os.environ.get("AIEDITOR_ENGINE_TOKEN") or secrets.token_urlsafe(32)
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr)

    print(json.dumps({"event": "ready", "port": port, "token": token}), flush=True)
    # Host is fixed to loopback on purpose: the engine must never be reachable from the network.
    uvicorn.run(create_app(token), host="127.0.0.1", port=port, log_level=args.log_level)


if __name__ == "__main__":
    main()
