# AIEditor

A local, beginner-first AI photo and video editor. See [PLAN.md](PLAN.md) for the full design and roadmap.

**Status: M0, foundations.** The engine core, the local API, media import with preview proxies, and the desktop shell are done. Editing tools start in M1.

## Layout
```
engine/   Python engine: project model, command bus (undo/redo + trajectory log), media probe/proxies, FastAPI
app/      Tauri 2 + React desktop app (launches the engine as a child process)
.github/  CI: engine tests on 3 OSes, frontend build, packaged app bundles
```

## Develop

### Engine
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. Run these from `engine/`:
```bash
uv sync
uv run pytest -q
uv run python -m aieditor_engine --port 8765   # prints {"event":"ready","port":...,"token":...}
```

### App in the browser (no Rust needed)
1. Start the engine as shown above.
2. Run these from `app/`:
   ```bash
   npm install
   npm run dev
   ```
3. Open `http://localhost:1420/?port=8765&token=<token from the ready line>`.

### Desktop app
Requires a working Rust toolchain ([rustup](https://rustup.rs)). From `app/`:
```bash
npm run tauri dev
```
In debug builds the shell starts `engine/.venv`'s Python automatically. To use a different Python, set `AIEDITOR_ENGINE_PYTHON`.

### Release bundle
```bash
cd engine && uv run --group build python scripts/build_sidecar.py
cd ../app && npx tauri build --config src-tauri/tauri.bundle.conf.json
```

## Engine API (localhost only, token required)
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness check (public) |
| GET | `/commands` | every command with its JSON schema (the GUI, and later EditLM, build on this) |
| POST | `/projects` · `/projects/open` | create or open a project folder |
| GET | `/project` | snapshot: project, revision, undo/redo state |
| POST | `/commands/{name}` | run a command `{params, source: "user"\|"ai"}` |
| POST | `/undo` · `/redo` | history |
| POST | `/media/import` | probe files, add them to the bin, queue thumbnails and proxies |
| GET | `/media/{id}/thumb` · `/media/{id}/proxy` | derived files |
| WS | `/ws` | live events: command, undo, redo, media_updated, proxy_progress, project_opened |

Every command, from a person or the AI, is appended to `logs/trajectory.jsonl` in the project folder. That log is EditLM training data.
