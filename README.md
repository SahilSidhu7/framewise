# AIEditor

A local, beginner-first AI photo and video editor. See [PLAN.md](PLAN.md) for the full design and roadmap.

**Status: working photo and video editor with local AI (M0, M1, first AI milestone, video v1).**

- **Photo engine:** a 32-bit float, linear-light pipeline with 23 ops:
  - tone, color, HSL, curves, split-tone grade, LUTs
  - sharpen, denoise, dehaze, clarity
  - grain, vignette, bloom/halation, light leaks, dust, paper texture
  - crop, rotate, auto-straighten, flip

  Dithered 8/16-bit export to JPEG/PNG/TIFF/WebP. Preview and export use the same code path.
- **18 looks:** YAML recipes with a strength knob, for example `ancient_sepia`, `teal_orange_cinematic`, `aesthetic_soft`, `vintage_70s` and `noir`.
- **Edit DSL:** the compact language the AI writes, e.g. `look vintage_70s s=0.7 | grain 0.3 | crop 4:5`.
- **EditLM v0:** local Qwen3-1.7B through llama.cpp (Vulkan). How it plans:
  1. A rule layer reads the request and the photo's measurements, then narrows a grammar. The model must answer within it.
  2. The model writes 3 variations.
  3. An Edit Linter enforces the playbook rules on what it wrote.

  With no model installed, the rule planner answers alone. Every AI edit is a single undo step.
- **Video engine:**
  - timeline compositor with transitions, speed, Ken Burns motion, per-clip and whole-video looks
  - cinematic bars, fades, captions and titles (6 styles, bundled OFL fonts)
  - audio mix with crossfades, music ducking under speech and -14 LUFS loudness
  - color grades baked into a 3D LUT per export; frames render in parallel (1080p about 17 fps on an RTX 4060 laptop)
  - export to H.264 + AAC (NVENC/QSV/AMF first, software fallback), tagged BT.709
- **Video AI director:** the model picks the intent (aspect, pace, transitions, look, captions, title, music). Code then chooses the best moments, cuts on the music beat, avoids jump cuts and keeps whole sentences.
  - Auto-captions use local whisper.cpp.
- **Beginner UI:**
  - AI prompt and one-click chips
  - looks gallery rendered on your photo
  - live sliders
  - crop, rotate and straighten
  - edit list with on/off toggles
  - before/after
  - export dialog

## Local AI model setup
The engine looks in `%LOCALAPPDATA%\AIEditor` (or the `AIEDITOR_LLAMA_SERVER` and `AIEDITOR_MODEL` env vars) for:
```
runtime/llama-<build>/llama-server.exe   # llama.cpp release zip, e.g. llama-bXXXXX-bin-win-vulkan-x64.zip
models/Qwen3-1.7B-Q8_0.gguf              # huggingface.co/Qwen/Qwen3-1.7B-GGUF (Apache-2.0)
runtime/whisper-<build>/Release/whisper-cli.exe   # whisper.cpp release zip (MIT), for auto-captions
models/ggml-base.bin                     # huggingface.co/ggerganov/whisper.cpp (MIT)
```
The model starts automatically when the AI tab opens, and runs only on 127.0.0.1.

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
