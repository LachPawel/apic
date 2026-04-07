use aic_sdk::{Model, Processor, ProcessorConfig, VadParameter};
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rodio::buffer::SamplesBuffer;
use rodio::{DeviceSinkBuilder, Player};
use std::collections::VecDeque;
use std::num::NonZero;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const TARGET_HZ: u32 = 16_000;

/// Number of 16kHz mono samples per Quail VF processing chunk (10ms).
pub const FRAMES_PER_CHUNK: usize = 160;

/// After this many consecutive silent chunks, the accumulated speech is
/// committed as a complete utterance. 20 chunks × 10ms = 200ms hold.
const SILENCE_CHUNKS_TO_COMMIT: usize = 20;

/// Run the continuous capture + VAD loop. Intended to be called from a
/// dedicated `std::thread::spawn` closure; it never returns.
///
/// Each time the VAD detects the end of an utterance, the voiced audio
/// (16kHz mono f32) is sent via `utterance_tx`. If speech is detected
/// while `is_speaking` is true, a `()` is sent via `barge_in_tx`.
pub fn run_capture_loop(
    vad_model_path: std::path::PathBuf,
    license_key: String,
    utterance_tx: tokio::sync::mpsc::Sender<Vec<f32>>,
    is_speaking: Arc<AtomicBool>,
    _barge_in_tx: tokio::sync::mpsc::Sender<()>,
) {
    // ── Quail VF initialisation ───────────────────────────────────────────
    let model =
        Model::from_file(&vad_model_path).expect("Quail VF: model load failed. Download from \
            https://artifacts.ai-coustics.io/models/quail-vf-2-0-l-16khz/v2/\
            quail_vf_2_0_l_16khz_d42jls1e_v18.aicmodel");

    let mut processor =
        Processor::new(&model, &license_key).expect("Quail VF: processor creation failed \
            (check AIC_SDK_LICENSE env var)");

    let config = ProcessorConfig::optimal(&model);
    processor
        .initialize(&config)
        .expect("Quail VF: processor initialization failed");

    let vad = processor.vad_context();
    // Hold for 400ms after last voiced frame before committing an utterance.
    vad.set_parameter(VadParameter::SpeechHoldDuration, 0.4)
        .expect("VAD: SpeechHoldDuration");
    // Require 80ms of continuous speech before triggering — suppresses clicks.
    vad.set_parameter(VadParameter::MinimumSpeechDuration, 0.08)
        .expect("VAD: MinimumSpeechDuration");

    // ── cpal mic setup ────────────────────────────────────────────────────
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("cpal: no default input device");
    let supported = device
        .default_input_config()
        .expect("cpal: cannot get default input config");

    let channels = supported.channels();
    let device_hz = supported.sample_rate().0;

    let stream_config = cpal::StreamConfig {
        channels,
        sample_rate: supported.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    // Bounded channel from cpal callback → our processing loop.
    // 256 × native-size callback ≈ several seconds of headroom.
    let (raw_tx, raw_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(256);
    let raw_tx_i16 = raw_tx.clone();

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let _ = raw_tx.try_send(data.to_vec());
                },
                |e| eprintln!("[audio] cpal error: {e}"),
                None,
            )
            .expect("cpal: failed to build F32 input stream"),
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let converted: Vec<f32> =
                        data.iter().map(|&s| s as f32 / 32_768.0).collect();
                    let _ = raw_tx_i16.try_send(converted);
                },
                |e| eprintln!("[audio] cpal error: {e}"),
                None,
            )
            .expect("cpal: failed to build I16 input stream"),
        fmt => panic!("[audio] unsupported sample format: {fmt:?}"),
    };

    stream.play().expect("cpal: failed to start input stream");

    eprintln!("[audio] capture started ({device_hz} Hz, {channels} ch)");

    // How many raw interleaved samples correspond to one FRAMES_PER_CHUNK
    // mono 16kHz chunk? We collect this many from the ring buffer before
    // running one Quail VF + VAD pass.
    let device_chunk_frames =
        (FRAMES_PER_CHUNK as u64 * device_hz as u64 / TARGET_HZ as u64) as usize;
    let raw_chunk_samples = device_chunk_frames * channels as usize;

    // ── Processing state ─────────────────────────────────────────────────
    let mut pending: VecDeque<f32> = VecDeque::new();
    let mut voiced_buf: Vec<f32> = Vec::new();
    let mut in_speech = false;
    let mut silence_chunks: usize = 0;
    // After playback stops, ignore mic for this many chunks (~250ms) so room
    // echo doesn't trigger a false utterance.
    const ECHO_COOLDOWN_CHUNKS: usize = 25;
    let mut echo_cooldown: usize = 0;
    let mut was_speaking = false;

    // ── Main loop ─────────────────────────────────────────────────────────
    for raw in &raw_rx {
        pending.extend(raw.iter().copied());

        while pending.len() >= raw_chunk_samples {
            // Drain one device-rate chunk.
            let chunk: Vec<f32> = pending.drain(..raw_chunk_samples).collect();

            // Mix interleaved multi-channel audio down to mono.
            let mono: Vec<f32> = if channels == 1 {
                chunk
            } else {
                chunk
                    .chunks_exact(channels as usize)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                    .collect()
            };

            // Resample from device rate to 16kHz.
            let mut chunk_16k: Vec<f32> = if device_hz == TARGET_HZ {
                mono
            } else {
                resample_linear(&mono, device_hz, TARGET_HZ)
            };

            // Pad or trim to exactly FRAMES_PER_CHUNK so Quail VF is happy.
            chunk_16k.resize(FRAMES_PER_CHUNK, 0.0);

            // Quail VF: enhance in-place, then query VAD.
            if let Err(e) = processor.process_sequential(&mut chunk_16k) {
                eprintln!("[audio] Quail VF process error: {e}");
                continue;
            }
            let speech = vad.is_speech_detected();

            let speaking_now = is_speaking.load(Ordering::Relaxed);

            // Detect the falling edge of is_speaking to start the cooldown.
            if was_speaking && !speaking_now {
                echo_cooldown = ECHO_COOLDOWN_CHUNKS;
                // Discard anything accumulated while the speaker was active.
                voiced_buf.clear();
                in_speech = false;
                silence_chunks = 0;
            }
            was_speaking = speaking_now;

            // While the assistant is speaking or the echo cooldown is active,
            // keep running Quail VF (it needs continuous audio to stay warm)
            // but don't feed anything into the VAD state machine.
            if speaking_now {
                continue;
            }
            if echo_cooldown > 0 {
                echo_cooldown -= 1;
                continue;
            }

            // VAD state machine.
            if speech {
                in_speech = true;
                silence_chunks = 0;
                voiced_buf.extend_from_slice(&chunk_16k);
            } else if in_speech {
                silence_chunks += 1;
                voiced_buf.extend_from_slice(&chunk_16k);

                if silence_chunks >= SILENCE_CHUNKS_TO_COMMIT {
                    // Trim the trailing silence, then send the utterance.
                    let trim = silence_chunks * FRAMES_PER_CHUNK;
                    let end = voiced_buf.len().saturating_sub(trim);
                    if end > 0 {
                        let utterance = voiced_buf[..end].to_vec();
                        // blocking_send: OK to park briefly while main loop
                        // catches up; the capture loop just accumulates more.
                        let _ = utterance_tx.blocking_send(utterance);
                    }
                    voiced_buf.clear();
                    in_speech = false;
                    silence_chunks = 0;
                }
            }
        }
    }
}

/// Linear-interpolation resample from `in_hz` to `out_hz` (mono f32).
fn resample_linear(input: &[f32], in_hz: u32, out_hz: u32) -> Vec<f32> {
    let out_len = (input.len() as f64 * out_hz as f64 / in_hz as f64) as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * in_hz as f64 / out_hz as f64;
            let idx = src as usize;
            let frac = (src - idx as f64) as f32;
            let a = input.get(idx).copied().unwrap_or(0.0);
            let b = input.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

/// Play `samples` through the default output device.
///
/// Blocks until playback completes. Rodio resamples from `sample_rate` to the
/// device's native rate automatically, so 24kHz TTS output plays without any
/// manual conversion.
pub fn play(samples: &[f32], sample_rate: u32) -> Result<()> {
    let handle = DeviceSinkBuilder::open_default_sink()
        .map_err(|e| anyhow::anyhow!("Failed to open output sink: {e}"))?;

    let player = Player::connect_new(handle.mixer());

    let source = SamplesBuffer::new(
        NonZero::new(1u16).unwrap(),
        NonZero::new(sample_rate).unwrap(),
        samples.to_vec(),
    );

    player.append(source);
    player.sleep_until_end();

    Ok(())
}
