# apic

A fully local, on-device voice pipeline in Rust for Apple Silicon. Built as a devrel benchmark for [apfel](https://github.com/LachPawel/apfel).

```
mic → whisper-rs (STT) → apfel (LLM) → Kokoro (TTS) → speaker
```

Everything runs on your Mac. Nothing leaves the machine. No API keys. No cloud.

## Benchmark results

Measured on Apple Silicon with `cargo bench`:

| Stage | Input | Latency |
|-------|-------|---------|
| STT (whisper small.en) | 2 s audio | **661 ms** |
| TTS (Kokoro 82M) | one sentence | **146 ms** |
| LLM (apfel 3B) | full round-trip | run with `apfel --serve` |

STT runs CPU-only in this build. Enable Metal for lower latency: set `use_gpu: true` in `WhisperContextParameters` in `src/stt.rs`.

Reproduce: `cargo bench --bench pipeline`

Full results: [benches/RESULTS.md](benches/RESULTS.md)

## Requirements

- Apple Silicon (M1 or later)
- macOS Tahoe 26.x
- Apple Intelligence enabled in System Settings
- Xcode (not just Command Line Tools) — the Metal toolchain is required to build mlx-rs
- [apfel](https://github.com/LachPawel/apfel) for the LLM stage

## Setup

**1. Install Xcode and accept the license**

```bash
# Open Xcode at least once, or:
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
xcodebuild -downloadComponent MetalToolchain
```

**2. Download the Whisper model**

```bash
mkdir -p models
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin \
     -o models/ggml-small.en.bin
```

The Kokoro TTS model (~82 MB) downloads automatically from HuggingFace on first run.

**3. Start apfel**

```bash
apfel --serve
```

This starts the on-device LLM server at `http://127.0.0.1:11434/v1`.

**4. Build and run**

```bash
cargo run --release
```

You'll see:
```
apic ready. Speak after the prompt (4s window).
Press Ctrl+C to quit.

Listening...
```

Speak during the 4-second window. apic transcribes your speech, sends it to apfel, synthesizes the response with Kokoro, and plays it back.

## Running benchmarks

```bash
# STT and TTS (no apfel needed)
cargo bench --bench pipeline -- stt_latency_2s_sine
cargo bench --bench pipeline -- tts_per_sentence

# LLM and end-to-end (requires apfel --serve)
cargo bench --bench pipeline -- llm_round_trip
cargo bench --bench pipeline -- e2e_first_audio_from_fixed_buffer
```

Criterion writes HTML reports to `target/criterion/`.

## Running tests

```bash
cargo test
```

Tests cover: LLM client (wiremock), STT short-buffer guard, STT silence detection, TTS synthesis.

## Architecture

```
src/
  lib.rs      public module exports for benchmark access
  main.rs     pipeline loop: record → STT → LLM → TTS → play
  audio.rs    mic capture (cpal), speaker playback (rodio)
  stt.rs      whisper-rs transcription, dedicated std::thread
  llm.rs      async-openai client → apfel
  tts.rs      voice-g2p + voice-tts synthesis, dedicated std::thread
benches/
  pipeline.rs criterion benchmarks, per-stage and end-to-end
  RESULTS.md  benchmark results (update after each run)
models/
  ggml-small.en.bin  whisper model (not in git, download separately)
```

**Thread model.** `WhisperContext` (whisper-rs) and `KokoroModel` (voice-tts via mlx-rs) are not `Send` — they hold Metal/C++ state tied to the thread they were created on. Both run on dedicated `std::thread`s and communicate with the async main loop via `std::sync::mpsc` channels and `tokio::sync::oneshot` reply channels.

## Known limitations

- **No barge-in.** The pipeline finishes speaking before listening again. Interruption is not supported in v0.1.
- **Fixed 4-second recording window.** VAD (voice activity detection) is not implemented. You have 4 seconds per turn; silence at the end is fine.
- **Whisper runs CPU-only** by default. Set `use_gpu: true` in `stt.rs` for Metal acceleration.
- **No conversation history.** Each turn is independent. The LLM does not remember previous turns.

## Stack versions

| Crate | Version | Role |
|-------|---------|------|
| async-openai | 0.34 | LLM client (apfel) |
| whisper-rs | 0.16 | STT |
| voice-g2p | 0.2.2 | Text → phonemes |
| voice-tts | 0.2.1 | Phonemes → audio |
| cpal | 0.15 | Mic capture |
| rodio | 0.22 | Audio playback |
| criterion | 0.5 | Benchmarks |

## License

MIT
