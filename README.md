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
Output: `app/src-tauri/target/release/bundle/nsis/AIEditor_<version>_x64-setup.exe` (about 105 MB, per-user install,
no admin rights). The engine is one frozen binary (`aieditor-engine.exe`, starts in about 3 s) that the app launches;
called with a command (`aieditor-engine.exe reel ...` or `... cli ...`) it is the `aieditor` command line, and the
installer drops an `aieditor.cmd` launcher into `%LOCALAPPDATA%\Microsoft\WindowsApps` (already on PATH) so
`aieditor` works in any terminal. Uninstalling removes the program files and that launcher, never the user's
models, licence or jobs. The installer is not code-signed yet, so Windows SmartScreen will warn.

## Automation (n8n, scripts, AI agents)
```bash
aieditor serve                                   # headless API on 127.0.0.1:8765, docs at /docs
aieditor reel talk.mp4 --platform tiktok --set count=3 --out shorts/
aieditor watch ~/Videos/inbox --recipe captions  # drop files in, get results out
aieditor mcp                                     # MCP server for AI agents
```
Public job API under `/v1` (API keys, uploads, jobs, long-poll, signed webhooks), recipes `reel`, `highlights`,
`captions`, `video_edit`, `photo_edit`, `render_program`, `transcribe`. See [integrations/README.md](integrations/README.md)
for the HTTP API, webhooks, the n8n node and template, watch folders and MCP.

**Free vs Pro.** The desktop editor is free. One-shot `aieditor` commands and MCP are free with limits (one reel per
video, one photo per job, no webhooks). Pro unlocks `aieditor serve`, watch folders, batches, webhooks and up to 10
reels per video. `aieditor licence trial` starts a 14-day trial once; `aieditor licence activate KEY` adds a key
(or set `AIEDITOR_LICENSE` on servers and in containers). Keys are Ed25519-signed and checked offline; the app also
exposes `GET/POST/DELETE /licence`. Keys are made with `tools/licence_issuer.py` and a private key that stays
outside this repository.

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
| GET | `/ai/options` | editing styles and desired outputs (platform, length, captions, music) for the prompt options |
| POST | `/ai/edit` · `/ai/video` | plan photo / video edits from a prompt plus optional `style` and `output`; returns variations, applies nothing |
| POST | `/ai/reels` | best moments of one long video as ready short clips: sentence-aligned, pauses and fillers cut, speaker kept in frame |
| POST | `/ai/styles/from_video` | "edit like this": measure a reference video and save it as a reusable style |

Every command, from a person or the AI, is appended to `logs/trajectory.jsonl` in the project folder. That log is EditLM training data.

## Website and architecture
- `site/`: the public landing page, a static site on Cloudflare Pages (project `aieditor`, https://aieditor-1me.pages.dev).
  Deploy with `npx wrangler pages deploy site --project-name aieditor --branch main`.
- `docs/architecture.html`: low-level design tour (components, data model, flows, rendering, caption placement,
  jobs, licensing, installer) with Mermaid diagrams.
