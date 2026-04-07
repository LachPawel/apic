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

use aic_sdk::{Model, Processor, ProcessorConfig};
use apic::{llm, stt, tts};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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
// Quail VF — per-chunk enhancement latency and AIC on/off comparison
// ---------------------------------------------------------------------------
//
// Benchmarks:
//   quail_vf_chunk_latency/<model>  — wall time for one 10ms chunk through Quail VF
//   stt_with_aic_vs_raw             — STT latency with and without AIC preprocessing
//
// Model files must be present in models/. Multiple files are benchmarked in
// one run so you can compare quail-vf-1.1 vs quail-vf-2.0 side-by-side.
//
// Run:
//   cargo bench --bench pipeline -- quail_vf_chunk_latency
//   cargo bench --bench pipeline -- stt_with_aic_vs_raw

fn bench_quail_vf_chunk(c: &mut Criterion) {
    // Discover all .aicmodel files in models/.
    let model_files: Vec<std::path::PathBuf> = std::fs::read_dir("models")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "aicmodel"))
        .collect();

    if model_files.is_empty() {
        eprintln!("[bench_quail_vf] SKIP — no .aicmodel files in models/. \
            Download from https://artifacts.ai-coustics.io/");
        return;
    }

    let license_key = std::env::var("AIC_SDK_LICENSE").unwrap_or_default();
    if license_key.is_empty() {
        eprintln!("[bench_quail_vf] SKIP — AIC_SDK_LICENSE not set");
        return;
    }

    let mut group = c.benchmark_group("quail_vf_chunk_latency");

    for model_path in &model_files {
        let name = model_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let model = match Model::from_file(model_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[bench_quail_vf] SKIP {name}: {e}");
                continue;
            }
        };
        let mut processor = match Processor::new(&model, &license_key) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[bench_quail_vf] SKIP {name}: processor create: {e}");
                continue;
            }
        };
        let config = ProcessorConfig::optimal(&model);
        if let Err(e) = processor.initialize(&config) {
            eprintln!("[bench_quail_vf] SKIP {name}: init: {e}");
            continue;
        }

        // One 10ms chunk of 440Hz sine at 16kHz.
        let mut chunk: Vec<f32> = (0..160)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16_000.0).sin())
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(&name),
            &name,
            |b, _| {
                b.iter(|| {
                    // Reset chunk so each iteration is identical input.
                    for (i, s) in chunk.iter_mut().enumerate() {
                        *s = (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16_000.0).sin();
                    }
                    processor.process_sequential(&mut chunk).unwrap();
                    processor.vad_context().is_speech_detected()
                })
            },
        );
    }

    group.finish();
}

// Compare STT latency on the same 2s sine buffer, with and without Quail VF
// preprocessing. Both paths produce the same Whisper input; this measures the
// overhead AIC adds to the hot path, not transcription accuracy.
fn bench_stt_with_aic_vs_raw(c: &mut Criterion) {
    let whisper_path = Path::new("models/ggml-small.en.bin");
    if !whisper_path.exists() {
        eprintln!("[bench_stt_aic] SKIP — whisper model not found");
        return;
    }
    let ctx = stt::init_whisper(whisper_path).expect("whisper init");

    let license_key = std::env::var("AIC_SDK_LICENSE").unwrap_or_default();
    let quail_path = Path::new("models/quail-vf-2-0-l-16khz.aicmodel");
    let aic_available = !license_key.is_empty() && quail_path.exists();

    // 2s 440Hz sine, pre-split into 160-frame chunks for AIC processing.
    let raw_buf: Vec<f32> = (0..32_000)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16_000.0).sin())
        .collect();

    let mut group = c.benchmark_group("stt_with_aic_vs_raw");

    // Baseline: raw audio straight to STT.
    group.bench_function("raw", |b| {
        b.iter(|| stt::transcribe(&ctx, &raw_buf).unwrap())
    });

    // With AIC: run through Quail VF first, then STT.
    if aic_available {
        let model = Model::from_file(quail_path).expect("quail model");
        let mut processor = Processor::new(&model, &license_key).expect("processor");
        processor
            .initialize(&ProcessorConfig::optimal(&model))
            .expect("init");

        group.bench_function("aic_enhanced", |b| {
            b.iter(|| {
                // Enhance in 160-frame chunks.
                let mut enhanced = raw_buf.clone();
                for chunk in enhanced.chunks_exact_mut(160) {
                    processor.process_sequential(chunk).unwrap();
                }
                stt::transcribe(&ctx, &enhanced).unwrap()
            })
        });
    } else {
        eprintln!(
            "[bench_stt_aic] aic_enhanced skipped — set AIC_SDK_LICENSE and \
             download models/quail-vf-2-0-l-16khz.aicmodel"
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_stt,
    bench_tts,
    bench_llm_round_trip,
    bench_e2e_first_audio,
    bench_quail_vf_chunk,
    bench_stt_with_aic_vs_raw,
);
criterion_main!(benches);
