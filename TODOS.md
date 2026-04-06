# TODOS

## bench fixture (captured 2026-04-06)

Create `benches/fixtures/utterance.f32` — a 2-second pre-recorded utterance as raw
f32 little-endian samples at 16kHz mono (32000 f32 values, ~128KB).

**Why:** `bench_e2e_first_audio` uses this file as a deterministic input. Without it,
the benchmark panics at startup. Whisper accuracy doesn't matter here — the benchmark
measures latency, not transcription quality.

**How:** Generate a 440Hz sine wave inline rather than committing a binary file:
```rust
// In benches/pipeline.rs
let fixed_buf: Vec<f32> = (0..32000)
    .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin())
    .collect();
```
Or record real speech and export as raw f32le from Audacity/ffmpeg:
```
ffmpeg -i utterance.wav -ar 16000 -ac 1 -f f32le benches/fixtures/utterance.f32
```

**Where to start:** Replace the `include_f32_fixture!("fixtures/utterance.f32")` stub
in `bench_e2e_first_audio` with the inline sine wave generation above. Simplest approach,
no binary file in the repo.

**Depends on:** Nothing — can be done at any time.
