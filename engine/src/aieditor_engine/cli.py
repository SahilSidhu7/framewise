"""`framewise` command line: headless server, one-shot recipes, keys, watch folders, MCP.

    framewise serve                         # API on http://127.0.0.1:8765 (docs at /docs)
    framewise keys create n8n               # prints a key once
    framewise reel talk.mp4 --platform tiktok --set count=3 --out shorts/
    framewise edit a.mp4 b.mp4 song.mp3 --prompt "fun travel recap" --style travel
    framewise photo *.jpg --style food --platform instagram_post
    framewise captions talk.mp4 --set style=boxed
    framewise watch ~/Videos/inbox --recipe reel
    framewise mcp                           # MCP server on stdio for AI agents
    framewise licence trial | activate KEY  # Pro: API, watch folders, batches, webhooks, several reels

One-shot commands run in this process (no server needed) and print a JSON
summary with the output paths on stdout; progress goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

SHORTCUTS = {"reel": "reel", "highlights": "highlights", "captions": "captions", "edit": "video_edit",
             "photo": "photo_edit", "render": "render_program", "transcribe": "transcribe"}
DEFAULT_PORT = 8765


def _value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _job_args(p: argparse.ArgumentParser, files: bool = True) -> None:
    if files:
        p.add_argument("files", nargs="+", help="input files, numbered 1..n in this order")
    p.add_argument("--prompt", help="what you want, in plain words")
    p.add_argument("--style", help="editing style id (framewise styles)")
    p.add_argument("--platform", help="tiktok, instagram_reel, youtube_short, youtube, instagram_post, ...")
    p.add_argument("--length", type=float, help="target length in seconds")
    p.add_argument("--versions", type=int, help="how many versions (1-5)")
    p.add_argument("--captions", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--music", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--quality", choices=["draft", "standard", "high"])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="recipe option, e.g. count=3 (repeatable)")
    p.add_argument("--program", type=Path, help="DSL file for the render recipe")
    p.add_argument("--webhook", help="URL to POST when the job finishes")


def _request(args: argparse.Namespace, recipe: str | None, files: list[str]) -> dict[str, Any]:
    output = {k: getattr(args, k) for k in ("platform", "length", "versions", "captions", "music", "quality")
              if getattr(args, k, None) is not None}
    options: dict[str, Any] = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        options[k.strip()] = _value(v)
    if getattr(args, "program", None):
        options["program"] = args.program.read_text(encoding="utf-8")
    body: dict[str, Any] = {"inputs": [str(Path(f).expanduser().resolve()) for f in files], "options": options}
    for k in ("prompt", "style", "webhook"):
        if getattr(args, k, None):
            body[k] = getattr(args, k)
    if recipe:
        body["recipe"] = recipe
    if output:
        body["output"] = output
    return body


def _progress(job: dict[str, Any]) -> None:
    pct = int(job["progress"] * 100)
    sys.stderr.write(f"\r[{pct:3d}%] {job['stage'][:60]:<60}")
    if job["state"] in ("done", "failed", "cancelled"):
        sys.stderr.write("\n")
    sys.stderr.flush()


def cmd_run(args: argparse.Namespace, recipe: str | None) -> int:
    from .automation.jobs import JobError, JobManager, JobRequest

    from .licence import LicenceStore

    manager = JobManager(args.home, start=False, licence=LicenceStore())
    try:
        req = JobRequest.model_validate(_request(args, recipe, args.files))
        job = manager.run_now(req, on_progress=None if args.quiet else _progress)
    except (JobError, ValueError) as exc:
        print(json.dumps({"state": "failed", "error": str(exc)}), file=sys.stdout)
        return 2
    files = []
    if job["state"] == "done":
        out = Path(args.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for o in job["outputs"]:
            dst = out / o["name"]
            shutil.copy2(manager.home / "jobs" / job["id"] / "out" / o["name"], dst)
            files.append(str(dst))
        if not args.keep:
            manager.delete(job["id"])
    summary = {"state": job["state"], "error": job["error"], "result": job["result"], "files": files}
    if job.get("notes"):
        summary["notes"] = job["notes"]
        for n in job["notes"]:
            print(n, file=sys.stderr)
    print(json.dumps(summary, indent=1))
    return 0 if job["state"] == "done" else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.server import create_app
    from .automation.jobs import JobManager
    from .automation.watch import FolderWatcher
    from .licence import LicenceStore

    licence = LicenceStore()
    if not _pro_or_explain(licence, "serve"):
        return 3
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not args.allow_lan:
        print("Refusing to listen on a network address without --allow-lan.", file=sys.stderr)
        return 2
    manager = JobManager(args.home, workers=args.workers, keep_days=args.keep_days, licence=licence, pro_feature="serve")
    if not loopback:
        print("WARNING: the API is reachable from your network. Anyone with an API key can run jobs on this computer.\n"
              "Use keys without the `local` scope for other machines, and never expose this port to the internet\n"
              "directly (use a tunnel with its own authentication).", file=sys.stderr)
    if not manager.keys.list():
        key, _ = manager.keys.create("default", ["jobs", "local", "admin"])
        print(f"Created your first API key (shown once, keep it safe):\n  {key}\n", file=sys.stderr)
    stop = threading.Event()
    for folder in args.watch or []:
        w = FolderWatcher(manager, Path(folder), {"recipe": args.watch_recipe} if args.watch_recipe else None)
        threading.Thread(target=w.run, args=(stop,), daemon=True, name="watch").start()
    print(f"Framewise automation API on http://{args.host}:{args.port}  (docs: /docs, data: {manager.home})", file=sys.stderr)
    try:
        uvicorn.run(create_app(automation=manager), host=args.host, port=args.port, log_level=args.log_level)
    finally:
        stop.set()
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from .automation.jobs import JobManager
    from .automation.watch import FolderWatcher
    from .licence import LicenceStore

    licence = LicenceStore()
    if not _pro_or_explain(licence, "watch"):
        return 3
    manager = JobManager(args.home, workers=1, licence=licence, pro_feature="watch")
    body = _request(args, args.recipe, [])
    body.pop("inputs")
    w = FolderWatcher(manager, Path(args.folder), body, interval=args.interval)
    print(f"Watching {w.folder} (recipe {w.config()['recipe']}); results go to {w.folder / 'output'}. Ctrl+C to stop.", file=sys.stderr)
    stop = threading.Event()
    try:
        w.run(stop)
    except KeyboardInterrupt:
        stop.set()
    manager.shutdown()
    return 0


def _pro_or_explain(licence: Any, feature: str) -> bool:
    from .licence import LicenceRequired

    try:
        licence.require(feature)
    except LicenceRequired as exc:
        print(exc, file=sys.stderr)
        return False
    st = licence.status()
    if st["tier"] == "trial":
        print(f"Framewise Pro trial: {st['trial_days_left']} day(s) left.", file=sys.stderr)
    return True


def cmd_licence(args: argparse.Namespace) -> int:
    from .licence import LicenceError, LicenceStore

    store = LicenceStore()
    try:
        if args.action == "activate":
            if not args.key:
                print("usage: framewise licence activate KEY", file=sys.stderr)
                return 2
            st = store.activate(args.key)
        elif args.action == "trial":
            st = store.start_trial()
        elif args.action == "deactivate":
            st = store.deactivate()
        else:
            st = store.status()
    except LicenceError as exc:
        print(json.dumps({"error": str(exc), **store.status()}, indent=1))
        return 1
    print(json.dumps(st, indent=1))
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    from .automation.keys import KeyError_, KeyStore
    from .automation.home import automation_home

    store = KeyStore(automation_home(args.home))
    if args.action == "create":
        try:
            key, rec = store.create(args.name or "key", args.scope or ["jobs"])
        except KeyError_ as exc:
            print(exc, file=sys.stderr)
            return 2
        print(json.dumps({"key": key, **rec}, indent=1))
        print("The key is shown only once.", file=sys.stderr)
    elif args.action == "list":
        print(json.dumps(store.list(), indent=1))
    elif args.action == "revoke":
        if not args.name or not store.revoke(args.name):
            print("no such key id", file=sys.stderr)
            return 1
        print("revoked")
    elif args.action == "secret":
        print(store.webhook_secret())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="framewise", description="Framewise automation: headless API, recipes, watch folders, MCP.")
    parser.add_argument("--home", help="automation data folder (default: %%LOCALAPPDATA%%/Framewise/automation or $AIEDITOR_HOME)")
    parser.add_argument("--log-level", default="warning")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the automation API (headless)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--allow-lan", action="store_true", help="allow a non-loopback --host")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--keep-days", type=float, default=7, help="delete finished jobs and uploads after this many days")
    s.add_argument("--watch", action="append", metavar="DIR", help="also watch this folder (repeatable)")
    s.add_argument("--watch-recipe", help="recipe for watched folders without framewise-watch.yaml")

    r = sub.add_parser("run", help="run any recipe once")
    r.add_argument("recipe")
    _job_args(r)
    for name, recipe in SHORTCUTS.items():
        sp = sub.add_parser(name, help=f"run the {recipe} recipe once")
        _job_args(sp)
    for sp in [r, *(sub.choices[n] for n in SHORTCUTS)]:
        sp.add_argument("--out", default="framewise-output", help="folder for the results")
        sp.add_argument("--keep", action="store_true", help="keep the job in the automation folder too")
        sp.add_argument("--quiet", action="store_true")

    w = sub.add_parser("watch", help="process every file dropped into a folder")
    w.add_argument("folder")
    w.add_argument("--recipe", default=None)
    w.add_argument("--interval", type=float, default=2.0)
    _job_args(w, files=False)

    k = sub.add_parser("keys", help="manage API keys")
    k.add_argument("action", choices=["create", "list", "revoke", "secret"])
    k.add_argument("name", nargs="?", help="key name (create) or key id (revoke)")
    k.add_argument("--scope", action="append", choices=["jobs", "local", "admin"])

    sub.add_parser("recipes", help="list recipes and their options")
    sub.add_parser("styles", help="list editing styles and outputs")
    sub.add_parser("mcp", help="MCP server on stdio (for AI agents)")
    lc = sub.add_parser("licence", aliases=["license"], help="Pro licence: status, activate KEY, trial, deactivate")
    lc.add_argument("action", nargs="?", default="status", choices=["status", "activate", "trial", "deactivate"])
    lc.add_argument("key", nargs="?")

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr)
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "run":
        return cmd_run(args, args.recipe)
    if args.cmd in SHORTCUTS:
        return cmd_run(args, SHORTCUTS[args.cmd])
    if args.cmd == "watch":
        return cmd_watch(args)
    if args.cmd == "keys":
        return cmd_keys(args)
    if args.cmd in ("licence", "license"):
        return cmd_licence(args)
    if args.cmd == "recipes":
        from .automation.recipes import catalog

        print(json.dumps(catalog(), indent=1))
        return 0
    if args.cmd == "styles":
        from .agent.options import catalog as options

        print(json.dumps(options(None), indent=1))
        return 0
    if args.cmd == "mcp":
        from .automation.mcp import serve_stdio

        return serve_stdio(args.home)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
