# apic

A fully local, on-device voice pipeline in Rust for Apple Silicon.
Built as a devrel benchmark showcase for the apfel project.

## What this is

apic is a voice pipeline:
  mic → STT → LLM → TTS → speaker

Every stage runs locally on Apple Silicon. Nothing leaves the machine.
The goal is to benchmark each stage's latency with criterion and publish
the results as a devrel artifact.

## Hardware and OS requirements
- Apple Silicon (M1 or later)
- macOS Tahoe 26.x
- Apple Intelligence enabled in System Settings

## Stack

### LLM — apfel
- Apple's on-device 3B parameter model exposed as OpenAI-compatible HTTP
- Start with: `apfel --serve`
- Base URL: http://127.0.0.1:11434/v1
- Model name: apple-foundationmodel
- API key: "unused" (no auth needed)
- Context window: 4096 tokens (input + output combined — important constraint)
- Runs on Neural Engine via FoundationModels.framework

### STT — whisper-rs
- Rust bindings for whisper.cpp
- Model file: models/ggml-small.en.bin (~150MB, downloaded separately)
- Runs on CPU/GPU
- Input: raw f32 audio samples at 16kHz mono
- Output: transcribed text string

### TTS — voice-tts + voice-g2p
- Kokoro 82M parameter neural TTS model
- Runs via mlx-rs on Apple Silicon Neural Engine
- voice-g2p converts text → phonemes
- voice-tts generates audio from phonemes
- Model auto-downloaded from HuggingFace Hub on first run
- Output: 24kHz audio samples

### Audio I/O — cpal + rodio
- cpal: low-level cross-platform audio I/O
- rodio: higher-level playback built on cpal
- Mic capture: cpal input stream at 16kHz mono f32
- Speaker playback: rodio sink

## Module structure

src/
  main.rs     — pipeline orchestration, ties all modules together
  audio.rs    — mic capture via cpal, speaker playback via rodio
  stt.rs      — whisper-rs STT, takes f32 audio buffer, returns String
  llm.rs      — async-openai client pointing at apfel, takes String, returns String
  tts.rs      — voice-g2p + voice-tts, takes String, returns audio samples
  
benches/
  pipeline.rs — criterion benchmarks for each stage and end-to-end

models/
  ggml-small.en.bin — whisper model (not committed to git)

## Build rules
- Always run `cargo clippy` before marking any module done
- Always run `cargo test` before /ship
- Each module must have at least one unit test
- Benchmark every stage individually in benches/pipeline.rs
- Use anyhow::Result for all error handling
- Build modules in this order: llm.rs → stt.rs → tts.rs → audio.rs → main.rs

## Key constraints
- apfel context window is 4096 tokens — keep LLM prompts short
- whisper-rs expects 16kHz mono f32 samples
- voice-tts outputs 24kHz — rodio must resample if needed
- cpal buffer sizes affect latency — document the tradeoff in benchmarks

# gstack

Available skills:

- `/office-hours` — Office hours discussion
- `/plan-ceo-review` — Plan a CEO review
- `/plan-eng-review` — Plan an engineering review
- `/plan-design-review` — Plan a design review
- `/design-consultation` — Design consultation
- `/review` — Code review
- `/ship` — Ship a change
- `/land-and-deploy` — Land and deploy
- `/canary` — Canary deploy
- `/benchmark` — Run benchmarks
- `/browse` — Browse the web
- `/qa` — QA testing
- `/qa-only` — QA only (no code changes)
- `/design-review` — Design review
- `/setup-browser-cookies` — Set up browser cookies
- `/setup-deploy` — Set up deployment
- `/retro` — Retrospective
- `/investigate` — Investigate an issue
- `/document-release` — Document a release
- `/codex` — Codex
- `/cso` — CSO review
- `/careful` — Careful mode
- `/freeze` — Freeze deployments
- `/guard` — Guard mode
- `/unfreeze` — Unfreeze deployments
- `/gstack-upgrade` — Upgrade gstack

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
