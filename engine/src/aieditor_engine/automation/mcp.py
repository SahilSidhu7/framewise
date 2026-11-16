"""MCP server (Model Context Protocol, stdio) so AI agents can drive the editor.

Add to an MCP client config, e.g.
    {"mcpServers": {"aieditor": {"command": "aieditor", "args": ["mcp"]}}}

Tools: list_recipes, list_styles, run_recipe, get_job, cancel_job. Files are
local paths (the agent runs on this computer); results are local paths too.
Implemented directly on JSON-RPC 2.0 (newline-delimited) to avoid a dependency.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import IO, Any

from .. import __version__
from .jobs import JobError, JobManager, JobRequest

log = logging.getLogger(__name__)
PROTOCOL = "2025-06-18"

_OUTPUT = {
    "type": "object",
    "description": "Desired output",
    "properties": {
        "platform": {"type": "string", "description": "tiktok, instagram_reel, youtube_short, story, youtube, square_post, portrait_post, movie, instagram_post"},
        "length": {"type": "number", "description": "seconds"},
        "versions": {"type": "integer", "minimum": 1, "maximum": 5},
        "captions": {"type": "boolean"},
        "music": {"type": "boolean"},
        "quality": {"type": "string", "enum": ["draft", "standard", "high"]},
    },
}

TOOLS = [
    {"name": "list_recipes", "description": "List what the editor can do (recipes) and each recipe's options.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_styles", "description": "List editing styles (cinematic, energetic_reel, vlog, ...) and output platforms.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "run_recipe",
     "description": "Edit local video/photo files. Recipes: reel (long video -> vertical shorts), highlights, captions, "
                    "video_edit (needs prompt), photo_edit, render_program (options.program = Edit DSL), transcribe. "
                    "Returns the job with the paths of the result files.",
     "inputSchema": {"type": "object", "required": ["files"], "properties": {
         "recipe": {"type": "string"},
         "files": {"type": "array", "items": {"type": "string"}, "description": "absolute paths, numbered 1..n"},
         "prompt": {"type": "string"},
         "style": {"type": "string"},
         "output": _OUTPUT,
         "options": {"type": "object", "description": "recipe options, see list_recipes"},
         "output_dir": {"type": "string", "description": "copy the results here"},
         "wait": {"type": "boolean", "default": True, "description": "false = return the job id immediately"},
         "timeout": {"type": "number", "default": 1800},
     }}},
    {"name": "get_job", "description": "State, progress and results of a job.",
     "inputSchema": {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}}},
    {"name": "cancel_job", "description": "Cancel a queued or running job.",
     "inputSchema": {"type": "object", "required": ["job_id"], "properties": {"job_id": {"type": "string"}}}},
]


class McpServer:
    def __init__(self, manager: JobManager) -> None:
        self.manager = manager

    def _with_paths(self, job: dict[str, Any]) -> dict[str, Any]:
        d = self.manager.home / "jobs" / job["id"] / "out"
        job = dict(job)
        job["outputs"] = [{**{k: v for k, v in o.items() if k != "url"}, "path": str(d / o["name"])} for o in job["outputs"]]
        return job

    def call(self, name: str, a: dict[str, Any]) -> Any:
        from ..agent.options import catalog as options
        from . import recipes as R

        if name == "list_recipes":
            return R.catalog()
        if name == "list_styles":
            return options(None)
        if name == "run_recipe":
            body = {k: a[k] for k in ("recipe", "prompt", "style", "output", "options", "output_dir") if a.get(k) is not None}
            body["inputs"] = list(a.get("files") or [])
            job = self.manager.submit(JobRequest.model_validate(body), allow_local=True, owner=None)
            if a.get("wait", True):
                job = self.manager.wait(job["id"], float(a.get("timeout") or 1800))
            return self._with_paths(job)
        if name == "get_job":
            return self._with_paths(self.manager.get(str(a.get("job_id"))))
        if name == "cancel_job":
            return self._with_paths(self.manager.cancel(str(a.get("job_id"))))
        raise JobError(f"unknown tool {name}", 404)

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:  # notification
            return None
        try:
            if method == "initialize":
                result: Any = {"protocolVersion": params.get("protocolVersion") or PROTOCOL,
                               "capabilities": {"tools": {"listChanged": False}},
                               "serverInfo": {"name": "aieditor", "version": __version__}}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                try:
                    data = self.call(str(params.get("name")), params.get("arguments") or {})
                    result = {"content": [{"type": "text", "text": json.dumps(data, indent=1, default=str)}],
                              "isError": isinstance(data, dict) and data.get("state") == "failed"}
                except (JobError, ValueError) as exc:
                    result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        except Exception as exc:  # noqa: BLE001 - never crash the stdio loop
            log.exception("mcp request failed")
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, inp: IO[str], out: IO[str]) -> None:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                out.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
                out.flush()
                continue
            reply = self.handle(msg)
            if reply is not None:
                out.write(json.dumps(reply) + "\n")
                out.flush()


def serve_stdio(home: str | Path | None = None) -> int:
    manager = JobManager(home, workers=1)
    try:
        McpServer(manager).serve(sys.stdin, sys.stdout)
    finally:
        manager.shutdown()
    return 0
