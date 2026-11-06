"""Local LLM runtime: a managed llama.cpp `llama-server` child process.

Located via env vars or the per-user app data folder:
  AIEDITOR_LLAMA_SERVER  path to llama-server(.exe)
  AIEDITOR_MODEL         path to a .gguf model
default root: %LOCALAPPDATA%/AIEditor (Windows) or ~/.local/share/AIEditor
  runtime/llama-*/llama-server(.exe)
  models/*.gguf   (first match of MODEL_PREFERENCE wins)

The server binds 127.0.0.1 on a random free port and starts lazily on the
first AI request. Generation is grammar-constrained (GBNF) so output always
parses as Edit DSL.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MODEL_PREFERENCE = ("EditLM", "Qwen3-1.7B", "Qwen3-0.6B", "Qwen3-4B")


def app_data_root() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "AIEditor"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AIEditor"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "AIEditor"


def find_server() -> Path | None:
    env = os.environ.get("AIEDITOR_LLAMA_SERVER")
    if env:
        return Path(env) if Path(env).is_file() else None
    exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    candidates = sorted((app_data_root() / "runtime").glob(f"llama-*/**/{exe}"), reverse=True)
    return candidates[0] if candidates else None


def find_model() -> Path | None:
    env = os.environ.get("AIEDITOR_MODEL")
    if env:
        return Path(env) if Path(env).is_file() else None
    models = list((app_data_root() / "models").glob("*.gguf"))
    for prefix in MODEL_PREFERENCE:
        for m in sorted(models):
            if m.name.lower().startswith(prefix.lower()):
                return m
    return models[0] if models else None


class LLMUnavailable(RuntimeError):
    pass


class LlamaServer:
    def __init__(self, ctx_size: int = 8192) -> None:
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._lock = threading.Lock()
        self._ctx = ctx_size
        self.error: str | None = None
        self.model: Path | None = None
        atexit.register(self.stop)

    # ---------------------------------------------------------------- state
    def status(self) -> dict[str, Any]:
        server, model = find_server(), find_model()
        if self._proc is not None and self._proc.poll() is None and self._port:
            state = "ready"
        elif server is None or model is None:
            state = "not_installed"
        elif self.error:
            state = "error"
        else:
            state = "idle"
        return {
            "state": state,
            "model": (self.model or model).name if (self.model or model) else None,
            "error": self.error,
            "server": str(server) if server else None,
        }

    def available(self) -> bool:
        return find_server() is not None and find_model() is not None

    # ------------------------------------------------------------ lifecycle
    def ensure_started(self, timeout: float = 120.0) -> int:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._port:
                return self._port
            server, model = find_server(), find_model()
            if server is None or model is None:
                raise LLMUnavailable("local AI model is not installed")
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            args = [
                str(server), "-m", str(model),
                "--host", "127.0.0.1", "--port", str(port),
                "-c", str(self._ctx), "-ngl", "99", "--jinja",
                "--no-webui", "-np", "3",
            ]
            flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
            log.info("starting llama-server: %s", " ".join(args))
            self._proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                creationflags=flags, cwd=str(server.parent),
            )
            stderr_tail: list[str] = []
            threading.Thread(target=self._drain, args=(self._proc, stderr_tail), daemon=True).start()
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    self.error = "llama-server exited: " + " | ".join(stderr_tail[-5:])
                    self._proc = None
                    raise LLMUnavailable(self.error)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            self._port, self.model, self.error = port, model, None
                            return port
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    pass
                time.sleep(0.4)
            self.stop()
            self.error = "llama-server did not become ready in time"
            raise LLMUnavailable(self.error)

    @staticmethod
    def _drain(proc: subprocess.Popen, tail: list[str]) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            tail.append(line)
            del tail[:-50]
            log.debug("llama: %s", line)

    def stop(self) -> None:
        proc, self._proc, self._port = self._proc, None, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ----------------------------------------------------------- generation
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        grammar: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 400,
        seed: int | None = None,
        timeout: float = 120.0,
    ) -> str:
        port = self.ensure_started()
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if grammar:
            body["grammar"] = grammar
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMUnavailable(f"generation failed: {exc}") from exc
        return data["choices"][0]["message"]["content"] or ""


LLM = LlamaServer()
