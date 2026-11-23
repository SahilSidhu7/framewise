# Contributing to Framewise

Framewise is free and open source under the [GNU AGPL-3.0](LICENSE). Contributions are welcome: bug reports, fixes,
new looks and recipes, docs, and larger features (layers and compositing are next on the roadmap in [PLAN.md](PLAN.md)).

By sending a pull request you agree that your contribution is licensed under AGPL-3.0, the same as the rest of the project.

## Repository layout
```
engine/   Python engine: project model, command bus, media pipeline, local API (FastAPI)
app/      Tauri 2 + React desktop app (launches the engine as a child process)
site/     Static landing page (Cloudflare Pages)
demo/     Scripts that build the example images on the site
docs/     Design notes and the third-party licence audit
```

## Engine (`engine/`)
Requires Python 3.12 or newer. Run everything from `engine/`.

### With a plain venv
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
pip install pytest httpx          # test dependencies (the "dev" group in pyproject.toml)
pytest -q
python -m aieditor_engine --port 8765   # prints {"event":"ready","port":...,"token":...}
```

### With uv (what CI uses)
```bash
uv sync
uv run pytest -q
```

Tests marked `uses_model` run the optional subject model only when it is installed; they are skipped otherwise.
The local AI models (llama.cpp, Qwen3, whisper.cpp) are optional for development; see the README for where they go.

## App (`app/`)
Requires Node.js 20+ and npm.

### In the browser (no Rust needed)
1. Start the engine as above.
2. From `app/`:
   ```bash
   npm install
   npm run dev
   ```
3. Open `http://localhost:1420/?port=8765&token=<token from the engine's ready line>`.

### As the desktop app (Tauri)
Requires a Rust toolchain ([rustup](https://rustup.rs)) and the
[Tauri prerequisites](https://tauri.app/start/prerequisites/) for your OS. From `app/`:
```bash
npm run tauri dev
```
In debug builds the shell starts `engine/.venv`'s Python automatically. Set `AIEDITOR_ENGINE_PYTHON` to use another one.

Before sending a change to the app, run:
```bash
npm run typecheck
npm run build
```

## Pull requests
- Keep each pull request to one change, and say what it does and how you checked it.
- Add or update tests in `engine/tests/` for engine changes.
- Every edit goes through the command bus so it can be undone; follow the existing commands as examples.
- New dependencies and models must have a licence compatible with AGPL-3.0. Add them to
  [docs/licences.md](docs/licences.md) in the same pull request.
- Do not commit media you do not have the rights to. Stock used in demos must be CC0 or CC BY, with credits.

## Reporting bugs
Open an issue with what you did, what you expected, what happened, your OS and GPU, and the engine log if there is one.
