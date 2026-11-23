# Third-party licences

Framewise is licensed under the **GNU AGPL-3.0** ([LICENSE](../LICENSE)). This page audits every runtime dependency
and every model the plan uses, and says whether each one can ship with an AGPL-3.0 project.

Audited 2026-11-23 against `engine/pyproject.toml`, `app/package.json`, `app/src-tauri/Cargo.toml`, the bundled
assets in `engine/src/aieditor_engine/fonts/`, and [PLAN.md §9](../PLAN.md). Python versions are the ones installed in
`engine/.venv` on that date; licences were read from each package's metadata.

**Compatible** means the licence allows the component to be combined with, or distributed alongside, AGPL-3.0 code.
Permissive licences (MIT, BSD, ISC, Apache-2.0) are one-way compatible: they can be used inside an AGPL project, and the
combined work is AGPL.

## Summary
- **No runtime dependency is incompatible.**
- **Needs attention:** PyAV's binary wheels bundle an FFmpeg build that includes **x264 and x265 (GPL-2.0-or-later)**.
  That is compatible with AGPL-3.0 (the "or later" lets it be used under GPL-3.0, and AGPL-3.0 §13 allows combining with
  GPL-3.0), but it means the shipped installer is a GPL/AGPL combination, and H.264/HEVC encoding may need patent
  licences in some countries. Keep the FFmpeg licence and source offer in `THIRD_PARTY_LICENSES` when packaging.
- **Not compatible, kept out:** the non-commercial models PLAN.md already excludes (listed at the end).
- **Check before use:** NVIDIA API models (training judge) are under per-model service terms. They are never shipped;
  check that their terms allow using outputs as training data before EditLM training.

## Engine (Python) runtime dependencies
| Package | Version checked | Licence | AGPL-compatible |
|---|---|---|---|
| fastapi | 0.162.0 | MIT | Yes |
| uvicorn[standard] | 0.54.0 | BSD-3-Clause (extras: uvloop MIT/Apache-2.0, httptools MIT, websockets BSD-3-Clause, watchfiles MIT, python-dotenv BSD-3-Clause) | Yes |
| pydantic (+ pydantic-core) | 2.14.3 | MIT | Yes |
| av (PyAV) | 19.0.2 | BSD-3-Clause | Yes |
| ↳ FFmpeg libs bundled in the PyAV wheel (`av.libs/`) | avcodec 63 | LGPL-2.1-or-later, built with GPL parts | Yes, see note |
| ↳ libx264, libx265 (in that FFmpeg build) | | GPL-2.0-or-later | **Yes, with note** (makes the binary GPL; patent caveat) |
| ↳ libvpx, libdav1d, SVT-AV1, libopus, libwebp, zlib, libvmaf, libvpl | | BSD-2/3-Clause, zlib, MIT | Yes |
| ↳ libgcc_s, libstdc++, libwinpthread (MinGW runtime) | | GPL-3.0 with GCC Runtime Library Exception / MIT+ZPL | Yes |
| pillow | 13.0.0 | MIT-CMU (HPND) | Yes |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (wheel bundles OpenBLAS, BSD-3-Clause) | Yes |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 (wheel bundles third-party libs, see its `LICENSE-3RD-PARTY.txt`; all permissive, LGPL-2.1 (FFmpeg) or MPL-2.0) | Yes |
| pyyaml | 6.0.3 | MIT | Yes |
| onnxruntime | 1.32.0 | MIT | Yes |

Development and build only (not shipped as code): pytest (MIT), httpx (BSD-3-Clause), hatchling (MIT),
pyinstaller (GPL-2.0 with a bootloader exception that allows any licence for the bundled program).

## App (JavaScript and Rust) runtime dependencies
| Package | Licence | AGPL-compatible |
|---|---|---|
| react | MIT | Yes |
| react-dom | MIT | Yes |
| @tauri-apps/api | Apache-2.0 OR MIT | Yes |
| @tauri-apps/plugin-dialog | Apache-2.0 OR MIT | Yes |
| tauri (Rust crate) | Apache-2.0 OR MIT | Yes |
| tauri-plugin-dialog (Rust crate) | Apache-2.0 OR MIT | Yes |
| serde, serde_json (Rust crates) | MIT OR Apache-2.0 | Yes |

Development only: @tauri-apps/cli, vite, @vitejs/plugin-react (MIT or Apache-2.0/MIT), typescript (Apache-2.0),
@types/react, @types/react-dom (MIT).

## Bundled assets and external runtimes
| Component | Licence | AGPL-compatible |
|---|---|---|
| Inter, Montserrat, Playfair Display fonts (`engine/src/aieditor_engine/fonts/`) | SIL OFL-1.1 | Yes, as separate font files shipped alongside (OFL text included next to each font) |
| llama.cpp (`llama-server`, downloaded at setup) | MIT | Yes |
| whisper.cpp (`whisper-cli`, downloaded at setup) | MIT | Yes |

## Models (PLAN.md §9)
Model weights are downloaded separately by the model manager, not compiled into the program.

| Role | Model | Licence | AGPL-compatible |
|---|---|---|---|
| EditLM base | Qwen3-1.7B / Qwen3-0.6B | Apache-2.0 | Yes |
| EditLM base (vision) | Qwen3-VL-2B | Apache-2.0 | Yes |
| Lite alternative | SmolVLM2-500M | Apache-2.0 | Yes |
| Visual encoder | SigLIP2 | Apache-2.0 | Yes |
| Speech | Whisper (OpenAI weights, `ggml-base.bin`) / faster-whisper | MIT | Yes |
| Background / subject | BiRefNet | MIT | Yes |
| Segmentation | SAM 2.1 | Apache-2.0 | Yes |
| Depth | Depth-Anything-V2 **Small** | Apache-2.0 | Yes (Base/Large are not; see below) |
| Shot detection | TransNetV2 | MIT | Yes |
| Faces | MediaPipe | Apache-2.0 | Yes |
| Inpaint | LaMa | Apache-2.0 | Yes |
| Upscale | Real-ESRGAN | BSD-3-Clause | Yes |
| Frame interpolation | RIFE | MIT | Yes |
| Beats | librosa (library) | ISC | Yes |
| Generative | FLUX.1-schnell | Apache-2.0 | Yes |
| Training teacher | Qwen3-VL-32B / Qwen3-VL-235B | Apache-2.0 | Yes (used for training, not shipped) |
| Training judge | NVIDIA API models | Per-model service terms | **Check** (not shipped; confirm output-use terms) |

### Not compatible: excluded
These have non-commercial or otherwise restrictive licences. They must not be bundled, downloaded by default or used to
produce shipped weights.

| Model | Why |
|---|---|
| RMBG-2.0 | CC BY-NC 4.0 (non-commercial) |
| 4x-UltraSharp | CC BY-NC-SA 4.0 (non-commercial) |
| Depth-Anything-V2 Base / Large | CC BY-NC 4.0 (non-commercial) |
| InsightFace models | Non-commercial research only |
| madmom pretrained models | CC BY-NC-SA 4.0 (non-commercial) |
| SDXL-Turbo | Stability AI non-commercial research licence |
| Qwen2.5-VL-3B | Qwen Research licence (non-commercial) |

## Rules for new dependencies
1. Prefer MIT, BSD, ISC, Apache-2.0, MPL-2.0, LGPL or GPL-3.0-compatible licences.
2. Never add anything under a non-commercial, "research only", SSPL or custom licence that forbids redistribution.
3. GPL-2.0-**only** code cannot be combined with AGPL-3.0; check for "or later".
4. Add the row to this file in the same pull request, and re-run this audit before each release.
