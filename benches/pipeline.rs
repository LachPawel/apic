// Criterion benchmarks for each pipeline stage and end-to-end.
//
// Run: cargo bench
//
// Requires:
//   - models/ggml-small.en.bin  (whisper model)
//   - apfel running on 127.0.0.1:11434  (llm_round_trip, e2e)
//   - HuggingFace Hub access or cached Kokoro model  (tts_per_sentence, e2e)
//
// Benchmarks that need external services are skipped gracefully if the
// service is unavailable (they panic at setup, not during the hot loop).

use apic::{llm, stt, tts};
use criterion::{criterion_group, criterion_main, Criterion};
use std::cell::RefCell;
use std::path::Path;

// ---------------------------------------------------------------------------
// STT — whisper-rs transcription latency
// ---------------------------------------------------------------------------

fn bench_stt(c: &mut Criterion) {
    let model_path = Path::new("models/ggml-small.en.bin");
    if !model_path.exists() {
        eprintln!("[bench_stt] SKIP — model not found at {}", model_path.display());
        return;
    }

    let ctx = stt::init_whisper(model_path).expect("whisper model load failed");

    // 2-second 440 Hz sine wave. Deterministic, no binary fixture committed.
    // Whisper produces garbage transcription from a sine wave; we measure latency only.
    let buf: Vec<f32> = (0..32_000)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16_000.0).sin())
        .collect();

    c.bench_function("stt_latency_2s_sine", |b| {
        b.iter(|| stt::transcribe(&ctx, &buf).unwrap())
    });
}

// ---------------------------------------------------------------------------
// TTS — Kokoro synthesis latency per sentence
// ---------------------------------------------------------------------------

fn bench_tts(c: &mut Criterion) {
    // init_model downloads from HuggingFace on first run (~82 MB, cached after).
    let model = match tts::init_model() {
        Ok(m) => RefCell::new(m),
        Err(e) => {
            eprintln!("[bench_tts] SKIP — TTS model load failed: {e}");
            return;
        }
    };

    c.bench_function("tts_per_sentence", |b| {
        b.iter(|| {
            // RefCell lets us take &mut KokoroModel inside an FnMut closure.
            // synthesize_with_model takes &mut KokoroModel; borrow_mut() is
            // released as soon as the call returns (temporaries live for the call).
            tts::synthesize_with_model(
                &mut model.borrow_mut(),
                "Hello, this is a test sentence.",
            )
            .unwrap()
        })
    });
}

// ---------------------------------------------------------------------------
// LLM — full round-trip latency (TTFT proxy; no streaming in current client)
// ---------------------------------------------------------------------------

fn bench_llm_round_trip(c: &mut Criterion) {
    let client = llm::LlmClient::new();

    // Probe liveness — skip gracefully if apfel isn't running.
    let rt = tokio::runtime::Runtime::new().unwrap();
    if rt
        .block_on(async { client.send_message("ping").await })
        .is_err()
    {
        eprintln!("[bench_llm] SKIP — apfel not reachable at http://127.0.0.1:11434");
        return;
    }

    c.bench_function("llm_round_trip", |b| {
        b.iter(|| {
            rt.block_on(async {
                client
                    .send_message("Reply with one word: yes.")
                    .await
                    .unwrap()
            })
        })
    });
}

// ---------------------------------------------------------------------------
// E2E — STT + LLM + TTS in sequence (time to first synthesized audio)
// ---------------------------------------------------------------------------

fn bench_e2e_first_audio(c: &mut Criterion) {
    let model_path = Path::new("models/ggml-small.en.bin");
    if !model_path.exists() {
        eprintln!("[bench_e2e] SKIP — whisper model not found");
        return;
    }

    let stt_ctx = stt::init_whisper(model_path).expect("whisper init");
    let tts_model = match tts::init_model() {
        Ok(m) => RefCell::new(m),
        Err(e) => {
            eprintln!("[bench_e2e] SKIP — TTS model load failed: {e}");
            return;
        }
    };
    let llm_client = llm::LlmClient::new();
    let rt = tokio::runtime::Runtime::new().unwrap();

    if rt
        .block_on(async { llm_client.send_message("ping").await })
        .is_err()
    {
        eprintln!("[bench_e2e] SKIP — apfel not reachable");
        return;
    }

    // Fixed synthetic utterance: 2 s 440 Hz sine at 16 kHz.
    let fixed_buf: Vec<f32> = (0..32_000)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16_000.0).sin())
        .collect();

    c.bench_function("e2e_first_audio_from_fixed_buffer", |b| {
        b.iter(|| {
            // STT (synchronous)
            let transcript = stt::transcribe(&stt_ctx, &fixed_buf).unwrap();
            // Fall back to a known prompt when whisper returns empty for sine input
            let prompt = if transcript.trim().is_empty() {
                "Say hello briefly.".to_string()
            } else {
                transcript
            };

            // LLM (async — block current thread)
            let response = rt.block_on(async { llm_client.send_message(&prompt).await.unwrap() });

            // TTS (synchronous)
            tts::synthesize_with_model(&mut tts_model.borrow_mut(), &response).unwrap()
        })
    });
}

// ---------------------------------------------------------------------------

criterion_group!(benches, bench_stt, bench_tts, bench_llm_round_trip, bench_e2e_first_audio);
criterion_main!(benches);
