# AIEditor: plan for a local AI photo and video editor

**Product:** a public, beginner-first photo and video editor. A user with no editing experience drops in media and types a request, for example:
- "make a collage"
- "give this an aesthetic / ancient look"
- "combine these clips into a cinematic video with captions"

The app delivers a polished result, rendered at the best quality possible.

**AI:** **EditLM**, a purpose-built small model (≤ 2B parameters) trained only to drive this editor's tools. Everything runs **locally**. No media leaves the machine.

**Dev / reference machine:** RTX 4060 Laptop, 8 GB VRAM (Standard tier).

---

## 1. Core principle: one command bus

The GUI and the AI call **exactly the same command API**.

```
 User clicks "Brightness +10"   ─┐
                                 ├──►  Command Bus  ──►  Project State (JSON doc)  ──►  Renderer
 EditLM emits DSL → compiler   ──┘        │                    │
                                     Undo/Redo stack      Preview / Export
                                     Trajectory log (training data)
```

- Every edit is a typed, serializable **Command**. Edits are non-destructive.
- Every AI edit appears in history and can be undone and tweaked in the UI.
- The command log doubles as **training data**. Editors working in the app produce exact expert trajectories.
- No pixel-level GUI clicking or screen scraping.

---

## 2. System architecture

```
┌──────────────────────────── Desktop App (Tauri + React) ───────────────────────────┐
│ SIMPLE MODE: drop media · prompt / templates · 3 variations · more/less · export   │
│ ADVANCED MODE: media bin · viewer · timeline · layers · inspector · chat           │
└───────────────────────────────┬────────────────────────────────────────────────────┘
                                │ WebSocket / HTTP (localhost)
┌───────────────────────────────▼──────────── Engine (frozen Python sidecar) ────────┐
│ Command Bus · Project Model (versioned) · Undo · Autosave · Trajectory recorder    │
│ Tool Registry (schemas → GUI forms, DSL grammar, plugin SDK)                       │
│ Tools: photo · collage · timeline · color · text/captions · audio · fx/vfx         │
│ Recipes (YAML looks/macros + LUTs) · Edit Linter                                   │
│ Renderer: float linear pipeline (GLSL/moderngl, CPU fallback) · OCIO · export      │
│ Perception (ONNX Runtime) · EditLM runtime (llama.cpp / ORT GenAI) · Model Manager │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Stack
| Layer | Choice | Why |
|---|---|---|
| UI shell | **Tauri 2 + React + TypeScript** | Light, native, small installer. |
| Engine | **Python 3.12 + FastAPI**, frozen with PyInstaller/Nuitka | Fast development. Ships as one sidecar binary. |
| Effects / compositing | **GLSL via moderngl (OpenGL 4.1)**, numpy/OpenCV CPU fallback | Works across vendors and on Windows, macOS and Linux. |
| ML inference | **ONNX Runtime**: DirectML (Windows: NVIDIA/AMD/Intel), CUDA/TensorRT, CoreML, CPU | **No PyTorch in the shipped app.** Torch is used in `training/` only. |
| LLM runtime | **llama.cpp** (CUDA / Vulkan / Metal / CPU builds, chosen at runtime) or ORT GenAI | Small quantized models on any hardware. |
| Video I/O | **FFmpeg 7, LGPL build only** + PyAV | No GPL x264/x265 in the shipped app. |
| Encoders | Hardware: NVENC, AMF, QSV, VideoToolbox, Media Foundation. Software fallback: SVT-AV1 (BSD), openh264 (Cisco binary). | Codec patents are handled by OS/hardware encoders. |
| Color | OpenColorIO, `.cube` LUTs | Accurate, color-managed output. |
| Text | Skia-python, ASS/SRT export | Animated captions. |
| Project | Folder: `project.json` (versioned schema + migrations), `media/`, `cache/` | Portable. |

---

## 3. The editing software (tool catalog)

Each tool = a Python function + a Pydantic schema. The schema drives the GUI form, the EditLM DSL grammar and validation. Aim for **60 to 80 coarse tools** with sensible defaults.

**3.1 Project / media:** `import_media`, `get_media_info`, `create_project(photo|video|collage, aspect, fps, res)`, `export(preset)`

**3.2 Photo:**
- **Adjust:** `adjust(exposure, contrast, highlights, shadows, whites, blacks, temperature, tint, vibrance, saturation, clarity, dehaze)`
- **Color:** `curves`, `hsl`, `color_grade`, `apply_lut`
- **Texture and effects:** `sharpen`, `denoise`, `grain`, `vignette`, `blur`, `bloom`, `chromatic_aberration`, `light_leak`, `dust_scratches`, `paper_texture`
- **Geometry:** `crop`, `smart_crop`, `rotate`, `straighten(auto)`, `perspective`, `resize`
- **AI:** `remove_background`, `select_subject|sky → mask`, `remove_object`, `upscale`, `face_retouch`, `depth_blur`
- **Layers:** `add_layer`, `blend_mode`, `opacity`, `mask`, `transform`
- **Collage:** `make_collage(images, layout=auto|grid|masonry|polaroid|filmstrip, gap, bg, aspect)`

**3.3 Video / timeline:**
- **Clips:** `add_clip`, `trim`, `split`, `ripple_delete`, `move`, `set_speed`, `reverse`, `freeze_frame`
- **Assembly:** `auto_assemble(mode=story|montage|beat_sync, target_duration)`, `cut_to_beats`, `remove_silences`, `select_best_shots`
- **Transitions:** `add_transition(cut|crossfade|dip|whip|zoom|glitch)`
- **Motion:** `keyframe`, `ken_burns`, `stabilize`, `reframe(9:16, track=subject)`
- **Color:** per clip or adjustment track, `match_color(ref)`
- **Audio:** `add_music`, `duck_under_voice`, `normalize_loudness`, `denoise_audio`, `fade`, `beat_detect`
- **Captions:** `transcribe`, `style_captions(tiktok_bold|cinematic_lower|karaoke)`, `translate_captions`, `add_title`

**3.4 VFX:** `letterbox`, `film_look(grain, halation, gate_weave, flicker)`, `vhs/glitch`, `light_leaks`, `lens_flare`, `particles`, `fog(depth-aware)`, `sky_replace`, `chroma_key`, `rotoscope(SAM2) → mask track`, `background_blur_video`, `slow_mo(RIFE)`, `speed_ramp`, `camera_shake`, `zoom_punch`, `overlay`, `parallax_3d`

**3.5 Recipes:** named, parameterized YAML macros that expand to many commands. Ship 40+ (`aesthetic_soft`, `ancient_sepia`, `vintage_70s`, `teal_orange_cinematic`, `noir`, `moody_dark`, `dreamy_glow`, `vhs_90s`, `pastel_symmetric`…).

```yaml
recipe: ancient_sepia
params: {strength: 0.8, era: "1900s|1970s"}
steps:
  - adjust: {contrast: -10, saturation: -40*strength, temperature: 15}
  - apply_lut: {name: sepia_warm, intensity: strength}
  - grain: {amount: 0.35, size: 1.4}
  - dust_scratches: {amount: 0.3}
  - vignette: {amount: 0.45}
  - paper_texture: {when: era == "1900s"}
```

---

## 4. EditLM: the custom model

### 4.1 How the model "sees": a hybrid, measurements first
Most decisions don't need raw pixels. Numbers are more precise and much cheaper for a small model. Each layer is used only when needed:

| Layer | What it gives | How | Cost |
|---|---|---|---|
| L1 Measurements | exposure, histogram, white balance, contrast, noise, sharpness, horizon tilt, palette, motion per shot | OpenCV / numpy | ms, exact |
| L2 Structure | shot cuts, faces, subject box/mask, depth, transcript, beats, silences | TransNetV2, MediaPipe, BiRefNet, Depth-Anything-V2-S, whisper, librosa | cached once |
| L3 Semantics | scene type, mood, style, aesthetic score, best-shot rank | SigLIP2 zero-shot + small trained heads | fast |
| L4 Visual tokens | overall "feel" the numbers miss | SigLIP2 embedding → small projector → **16 to 64 tokens per image/shot** | cheap |
| L5 Full look | ambiguous requests, final critique | low-res frames through a full VLM path | rare |

L1 to L3 compile into a compact text **Media Brief**. L4 adds a few visual tokens.

### 4.2 Base model: decided by a bake-off at M3
All candidates are Apache 2.0 and ≤ 2B:
- **A:** Qwen3-1.7B (text only) reading the Media Brief.
- **B:** Qwen3-1.7B + SigLIP2 projector adapter (L4 tokens). **Expected winner.**
- **C:** Qwen3-VL-2B reading frames + brief.

Pick on the eval harness: quality, latency, VRAM. **Lite build:** Qwen3-0.6B (+ projector) or SmolVLM2-500M.

### 4.3 Task modes (one model, selected by a task token)
1. `PLAN`: prompt + brief (+ visual tokens) → edit program
2. `CRITIQUE`: preview + prompt + linter report → score + fixes
3. `SELECT`: rank shots or photos for a goal
4. `DESCRIBE`: tag shots or photos for the brief

### 4.4 Output: a compact Edit DSL with a one-line rationale
Grammar-constrained (always parses), token-efficient, compiled into Command Bus commands:
```
# moody request → cool shadows, low sat, keep skin warm
look moody_dark s=0.7 | hsl orange s=+5 | vignette 0.3
clip 3 trim 1.2-6.0 | xfade 0.5 | captions cinematic_lower
```

### 4.5 Division of labour (why 2B is enough)
- **EditLM decides *what*:** intent, recipe, shots, order, strength.
- **Specialist models measure** (section 4.1).
- **Deterministic code decides *how*:** exact beat cut points, crop composition, color math.
- **Parameter auto-tuning:** after a recipe is chosen, a quick search over its parameters maximizes aesthetic score + prompt similarity − linter penalties.

### 4.6 Agent loop
```
analyze (cached) → PLAN (×3 variations) → compile + execute → render preview
   → lint + CRITIQUE → fix (≤ 2 rounds) → show variations → user picks / "more/less" → refine
```

### 4.7 Footprint
2B at Q4 ≈ 1.5 GB; 0.6B ≈ 0.4 GB. Runs on CPU or integrated graphics (Lite), fast on any GPU. A Model Manager keeps an LRU of models within the VRAM budget and loads them lazily.

---

## 5. Training EditLM

### 5.1 Teaching editing craft

**A. Editing Playbook** (`training/playbook/*.md`, written by us so it's license-clean). About 300 rules. Each has: rule, why it matters, a good and bad example, and the DSL that applies it.
- **Photo:**
  - correct first, then grade
  - protect highlights, don't crush blacks
  - keep skin tones on the skin-tone line
  - rule of thirds, level horizon, headroom
  - restrained saturation; subtle grain and vignette
  - the look must match the mood of the request
- **Collage:** balance, consistent color, a hero image, even gutters, never cut faces.
- **Video:**
  - shot variety (wide → medium → close)
  - cut on action, match cuts, J/L cuts
  - no jump cuts on the same framing
  - pacing per genre: reel 0.5 to 2 s shots, cinematic 3 to 6 s
  - cut on the beat, few transitions
  - strongest shot first, end on a resolve
  - 180° rule, consistent grade
- **Captions:** ≤ 2 lines and about 42 characters, safe zones, high contrast, synced to speech.
- **Audio:** duck music under voice, -14 LUFS, no clipping, fades.

**Three uses of the playbook:**
1. **Knowledge data:** Q&A and "explain this edit" examples, so the model learns the *why*.
2. **Rationale line** before each DSL program (cheap guided reasoning).
3. **Edit Linter** (`engine/lint/`): rules as automatic checks:
   - shots under 0.4 s
   - clipped or crushed histogram
   - skin hue off
   - captions outside the safe zone
   - too many transition types
   - loudness off target

   The linter feeds RL rewards, CRITIQUE and pre-export warnings.

**B. Good-example datasets** (check each license before use):

| Source | Content | Use |
|---|---|---|
| **Pro-editor recordings (main)** | Freelance editors edit CC0/own media *inside our editor*. The trajectory recorder logs exact expert DSL. About 2k photo + 500 video projects. | SFT gold + eval |
| Team exemplars | About 200 per genre: travel reel, vlog, wedding, product, birthday collage, trailer, portrait, landscape | few-shot retrieval + SFT |
| Opt-in beta users | edits kept vs undone (anonymized, consent only) | DPO preference pairs |
| CC0/CC-BY media (Pexels, Pixabay, Open Images, Wikimedia) | raw material | inverse edits, freelancer jobs |
| MIT-Adobe FiveK, PPR10K, AVA, AVE, MovieNet | expert retouches, aesthetics, cut annotations | **research/eval only** unless the license allows commercial training. Never ship weights trained on them without clearance. |

**C. Curriculum (training order):**
1. Knowledge (playbook Q&A).
2. Single-tool tasks ("brighten slightly" → `adjust exposure=+0.3`).
3. Looks and recipes.
4. Multi-step photo work and collages.
5. Video assembly: select, order, beat cuts, transitions, captions.
6. Full projects with self-critique.

### 5.2 Synthetic data (license-clean)
1. **Inverse edits (photo):**
   - Take a clean image, apply a random recipe with random parameters.
   - A teacher writes the matching natural-language request.
   - The model learns (image, request) → DSL.
   - Reverse direction too: degraded image + "fix this" → correction.
2. **Video trajectories:** an open teacher (Qwen3-VL-32B/235B, Apache, on rented GPU) plans edits for real clip sets. Every program is **executed in the engine**, and failures or low scores are dropped.
3. **Critique pairs:** good and broken renders paired with fixes.

Target: about **300k SFT examples** plus the human gold set.

### 5.3 Training stages
1. **SFT:** full fine-tune of the ≤ 2B on a rented A100/H100, about 1 to 2 days (a few hundred USD). The SigLIP projector trains together with it. LoRA experiments run on the RTX 4060.
2. **RL with verifiable rewards (GRPO).** The reward combines:
   - the DSL parses and executes
   - aesthetic gain
   - CLIP prompt similarity
   - judge score
   - minus linter penalties and step count
3. **Optimization with the NVIDIA free API** (section 5.4).
4. **Quantize** (GGUF Q4/Q5, ONNX int4) and check that eval quality holds.

### 5.4 Post-fine-tune optimization with the NVIDIA free API (build.nvidia.com / NIM)
The hosted API runs inference only. **It cannot edit our weights directly.** It acts as a strong external **teacher and judge**, and the weight updates happen on our own or rented GPUs:
1. **Judge / reward:**
   - A Nemotron reward model and large vision models on the API (e.g. Llama 3.2 90B Vision, Qwen-VL) score EditLM's renders.
   - Scores become GRPO rewards and DPO chosen/rejected pairs.
2. **Hard-case teacher:** where EditLM fails in eval, the API model writes the corrected program, which is added to the SFT data.
3. **Weight-level selection and merging:**
   - Train several LoRA / full-tune variants, per task (photo look, collage, video assembly, critique) and per hyperparameter setting.
   - The NVIDIA judge ranks them on held-out tasks.
   - **Merge the best with mergekit (TIES/DARE)**, re-evaluate, and keep the merge only if it beats the best single variant.
4. **Budget and terms:**
   - The free key is rate- and credit-limited. Use it on sampled subsets only (thousands of calls), and cache every response.
   - Bulk data generation stays on the open teacher.
   - Check each API model's license before training on its outputs.
   - The key lives in `.env` / OS keychain, never in the repo or the shipped app. **The shipped app makes no API calls.**

### 5.5 Eval harness (CI)
- 300 fixed tasks.
- Metrics: validity %, task success (judge), aesthetic delta, linter violations, latency per tier.
- A model ships only when it beats the previous one.

`training/`: `playbook/`, `datagen/`, `teacher/`, `sft/`, `rl/`, `merge/`, `eval/`, `export/`.

---

## 6. Beginner-first UX
- **Simple mode is the default.** Drop media, type a request, or pick a template card ("Aesthetic", "Vintage", "Collage", "Cinematic trailer", "Reel with captions").
- **Three variations per request.** Pick one. No tool names to learn.
- **"More / less" sliders** (recipe strength) and chips: "brighter", "shorter", "add music", "warmer".
- Before/after slider, one-click undo per AI step, plain-language "what I did" list.
- **One-click export presets:** Instagram post/reel, TikTok, YouTube 1080p/4K, print. Aspect, bitrate and loudness are set automatically.
- **Advanced mode:** the full editor (timeline, layers, inspector).
- Keyboard-first, screen-reader labels, i18n-ready strings.

---

## 7. Best-quality rendering
- 32-bit float linear-light pipeline, color managed with OpenColorIO (sRGB / Rec.709 / P3 / Rec.2020).
- 16-bit photo export; 10-bit HDR (HLG/PQ) video when the source is HDR.
- Proxies (540p/720p) for preview only. **The final render uses the originals at source resolution.**
- Lanczos / high-quality scaling, output dithering (no banding), correct motion blur and frame blending on speed changes.
- Optional **Enhance** pass at export: AI upscale (Real-ESRGAN), frame interpolation (RIFE), denoise.
- High-quality encoder presets with 2-pass where supported, plus AV1 option.
- Audio: 48 kHz, loudness-normalized per platform.
- **Pre-export QA:** linter + CRITIQUE + checks for black frames, clipping and caption overflow. The user is warned before render.
- Frame cache keyed by hash of (clip, time, effect stack). One code path for preview and export.

---

## 8. Hardware tiers
| Tier | Hardware | EditLM | Extras |
|---|---|---|---|
| Lite | CPU or integrated GPU, 16 GB RAM | 0.6B / 500M | perception on CPU, slower export |
| Standard | 6 to 8 GB VRAM or Apple M-series 16 GB | ≤ 2B | all features |
| Pro | 12 GB+ VRAM | ≤ 2B | generative fill / sky (FLUX.1-schnell) |

At first run, hardware detection and a quick benchmark pick the tier. Models are downloaded to match.

---

## 9. Models and licenses
Before release, re-verify every license and keep a `THIRD_PARTY_LICENSES` file.

| Role | Model | License |
|---|---|---|
| EditLM base | Qwen3-1.7B / 0.6B, Qwen3-VL-2B | Apache 2.0 |
| Lite alt | SmolVLM2-500M | Apache 2.0 |
| Visual encoder | SigLIP2 | Apache 2.0 |
| Speech | faster-whisper / Whisper | MIT |
| Background / subject | BiRefNet | MIT |
| Segmentation | SAM 2.1 | Apache 2.0 |
| Depth | Depth-Anything-V2 **Small** | Apache 2.0 |
| Shots | TransNetV2 | MIT |
| Faces | MediaPipe | Apache 2.0 |
| Inpaint | LaMa | Apache 2.0 |
| Upscale | Real-ESRGAN | BSD-3 |
| Interpolation | RIFE | MIT |
| Beats | librosa | ISC |
| Generative (Pro) | FLUX.1-schnell | Apache 2.0 |
| Training teacher | Qwen3-VL-32B/235B (open), NVIDIA API models (judge) | Apache / per-model terms |

**Excluded (non-commercial):** RMBG-2, 4x-UltraSharp, Depth-Anything-V2 Base/Large, InsightFace models, madmom pretrained models, SDXL-Turbo, Qwen2.5-VL-3B.

---

## 10. Product layer
- **Install and update:**
  - Windows: NSIS/MSI, EV code signing.
  - macOS: notarized dmg.
  - Linux: AppImage.
  - Signed Tauri auto-updater with a beta channel.
- **Model delivery:** small installer. First-run downloader with resume, SHA-256 checks and a mirror fallback. Models are managed in Settings.
- **Onboarding:** hardware benchmark → tier, then a guided demo edit.
- **Privacy:** fully local. Telemetry and crash reports are opt-in (self-hosted Sentry). Clear privacy statement.
- **Trust:** C2PA content credentials on AI-assisted edits. No face-swap or identity-manipulation tools. Usage policy.
- **Projects:** versioned schema + migrations, autosave, crash recovery, OpenTimelineIO export.
- **Plugin SDK:** third-party tools and recipes register with the same schema, so EditLM can use them automatically.
- **Monetization:**
  - **Free:** full manual editor + AI.
  - **Pro:** generative features, 4K/ProRes export, premium recipe packs.
  - Offline license key.
- **Platforms:** Windows first, macOS Apple Silicon second, Linux third.

---

## 11. Repository layout
```
AIEditor/
├─ app/                 # Tauri + React (simple/, advanced/, viewer/, timeline/, chat/)
├─ engine/
│  ├─ core/             # project model, command bus, undo, autosave, trajectory recorder
│  ├─ tools/            # photo/ video/ fx/ text/ audio/ collage/
│  ├─ recipes/          # *.yaml + LUTs
│  ├─ dsl/              # grammar, parser, compiler → commands
│  ├─ lint/             # Edit Linter rules
│  ├─ render/           # frame graph, compositor (GLSL), OCIO, proxy, export
│  ├─ perception/       # ONNX wrappers + analysis cache + media brief
│  ├─ agent/            # EditLM runtime, variations, critique loop, param auto-tune
│  ├─ models/           # model manager, downloader, hardware tiers
│  ├─ api/              # FastAPI + WebSocket
│  └─ tests/            # unit + golden-image tests
├─ training/            # playbook/ datagen/ teacher/ sft/ rl/ merge/ eval/ export/
└─ assets/              # fonts (OFL), textures, overlays, LUTs, music (license-clean)
```

---

## 12. Roadmap
| # | Milestone | Weeks |
|---|---|---|
| M0 | Foundations: command bus, project model, FastAPI, Tauri shell, media import + proxies, CI for 3 platforms, packaging pipeline, backend abstraction | 3 |
| M1 | Photo engine: float pipeline, OCIO, 20 tools, 15 recipes, DSL + compiler | 3 |
| M2 | Perception specialists + Media Brief + aesthetic scorer | 2 |
| M3 | **EditLM v0:** eval harness + **bake-off A/B/C** with off-the-shelf models + DSL grammar | 3 |
| M4 | Simple-mode UI, variations, templates, collage | 3 |
| M5 | Editing Playbook + Edit Linter + trajectory recorder; hire editors for gold set; datagen + teacher + execution filtering (photo) | 5 |
| M6 | **EditLM v1:** projector + SFT on photo; NVIDIA-judge DPO; LoRA variants + mergekit; quantize | 3 |
| M7 | **Photo public beta:** installer, downloader, onboarding, opt-in crash reports | 2 |
| M8 | Video engine: timeline, transitions, audio, captions, export quality | 5 |
| M9 | Video datagen + **EditLM v2:** SFT + GRPO with NVIDIA-judge rewards, hard-case distillation, merge | 5 |
| M10 | VFX pack + Enhance export pass | 4 |
| M11 | **1.0:** license audit, signing, QA hardware matrix (NVIDIA, AMD, Intel Arc, iGPU, M1/M3), docs, launch | 3 |

**About 41 weeks solo**, plus a freelancer budget for the gold set (roughly $3k to $8k, scalable). The first AI photo demo (off-the-shelf model) arrives at M4, about 14 weeks in.

---

## 13. QA
- Golden-image tests per tool on fixed inputs.
- Per-tier performance benchmarks.
- The agent eval harness runs in CI (CPU, smallest model).
- A beta channel for early users.

---

## 14. Risks and mitigations
| Risk | Mitigation |
|---|---|
| A 2B model's taste is weak | recipes, param auto-tuning, playbook + linter, variations, critique loop, expert gold set |
| Synthetic data bias | human gold set, opt-in user preferences, RL with judge rewards |
| Training cost | off-the-shelf baseline first; train only once the data pipeline is proven; LoRA experiments locally |
| GPU / driver fragmentation | ONNX Runtime providers, llama.cpp backends, CPU fallbacks, QA hardware matrix |
| Installer size | no torch in the shipped app; models downloaded per tier |
| Model / dataset licensing | license table, research-only data kept out of shipped weights, audit before 1.0 |
| Codec patents | LGPL FFmpeg + OS/hardware encoders |
| Misuse (deepfakes) | no face-swap tools, C2PA credentials |
| Preview too slow for VFX | proxies, frame cache, render queue, draft quality mode |
| Competition (CapCut, Photoshop AI) | position as local, private, free AI editing for beginners |

---

## 15. Next steps (M0)
1. Scaffold `engine/`: FastAPI, Pydantic project model, command bus + undo, trajectory log.
2. Scaffold `app/`: Tauri 2 + React, talking to the engine over WebSocket.
3. Media import + proxy generation (FFmpeg).
4. Set up GitHub Actions to build Windows/macOS/Linux, plus a packaging smoke test.
