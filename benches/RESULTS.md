# apic Benchmark Results

Hardware: Apple Silicon (M-series), macOS Tahoe 26.x  
Run: `cargo bench --bench pipeline`  
All times are wall-clock latency (mean ± confidence interval from criterion).

## Per-Stage Latency

| Stage | Benchmark | Input | Mean | CI low | CI high |
|-------|-----------|-------|------|--------|---------|
| STT | `stt_latency_2s_sine` | 2 s 440 Hz sine, 16 kHz mono f32 | **661 ms** | 659 ms | 664 ms |
| TTS | `tts_per_sentence` | "Hello, this is a test sentence." | **146 ms** | 145 ms | 146 ms |
| LLM | `llm_round_trip` | "Reply with one word: yes." | — | — | — |
| E2E | `e2e_first_audio_from_fixed_buffer` | 2 s sine → STT → LLM → TTS | — | — | — |

_LLM and E2E require `apfel --serve` running. Run `cargo bench -- llm_round_trip` and `cargo bench -- e2e_first_audio_from_fixed_buffer` with apfel active to populate those rows._

## Notes

- **STT input is synthetic** — 440 Hz sine wave, not speech. Whisper processes it in the same
  GGML compute graph regardless of content; latency is dominated by the transformer forward pass,
  not transcription quality. Real speech latency will be in the same range.
- **TTS measures synthesis only** — model load is amortized outside the hot loop (one-time ~2 s).
  The 146 ms is per-sentence synthesis time at 24 kHz.
- **Whisper runs CPU-only** in this build (`use_gpu = 0` in log output). Metal backend would
  reduce STT latency; enable with `WhisperContextParameters { use_gpu: true, .. }` in `stt.rs`.
- **Time to first audio (streaming)** — not yet measured. With streaming TTS (sentence-by-sentence
  as LLM generates), the headline metric would be: STT + TTFT + TTS(first sentence). Given
  TTS ≈ 146 ms/sentence, the bottleneck is STT (661 ms) + LLM TTFT.

## Reproduce

```bash
# Stage benchmarks (no external deps)
cargo bench --bench pipeline -- stt_latency_2s_sine
cargo bench --bench pipeline -- tts_per_sentence

# LLM + E2E (requires apfel)
apfel --serve &
cargo bench --bench pipeline -- llm_round_trip
cargo bench --bench pipeline -- e2e_first_audio_from_fixed_buffer
```
