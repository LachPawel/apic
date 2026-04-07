use apic::{audio, llm, stt, tts};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::oneshot;

const WHISPER_MODEL: &str = "models/ggml-small.en.bin";
const QUAIL_MODEL: &str = "models/quail-vf-2-0-l-16khz.aicmodel";

#[tokio::main]
async fn main() -> Result<()> {
    // ── License key ──────────────────────────────────────────────────────
    // Load .env if present; ignore if missing.
    let _ = dotenvy::dotenv();
    let license_key = std::env::var("AIC_SDK_LICENSE").map_err(|_| {
        anyhow::anyhow!(
            "AIC_SDK_LICENSE env var is required. \
             Set it in .env or export it. \
             Get your key at https://developers.ai-coustics.com"
        )
    })?;

    // ── STT thread ────────────────────────────────────────────────────────
    // WhisperContext is not Send — lives on a dedicated std::thread.
    let (stt_tx, stt_rx) =
        std::sync::mpsc::channel::<(Vec<f32>, oneshot::Sender<Result<String>>)>();
    let whisper_path = PathBuf::from(WHISPER_MODEL);
    std::thread::spawn(move || stt::run_transcribe_loop(&whisper_path, stt_rx));

    // ── TTS thread ────────────────────────────────────────────────────────
    // KokoroModel is not Send (Metal thread-local) — lives on a dedicated std::thread.
    let (tts_tx, tts_rx) =
        std::sync::mpsc::channel::<(String, oneshot::Sender<Vec<f32>>)>();
    std::thread::spawn(move || tts::run_synthesis_loop(tts_rx));

    // ── Capture / VAD thread ─────────────────────────────────────────────
    // Quail VF lives in its own std::thread; sends complete utterances to main.
    let (utterance_tx, mut utterance_rx) = tokio::sync::mpsc::channel::<Vec<f32>>(4);
    let (barge_in_tx, mut barge_in_rx) = tokio::sync::mpsc::channel::<()>(4);
    let is_speaking = Arc::new(AtomicBool::new(false));

    {
        let is_speaking = Arc::clone(&is_speaking);
        let quail_path = PathBuf::from(QUAIL_MODEL);
        std::thread::spawn(move || {
            audio::run_capture_loop(
                quail_path,
                license_key,
                utterance_tx,
                is_speaking,
                barge_in_tx,
            );
        });
    }

    // Arc so it can be moved into tokio::spawn without a reference.
    let client = Arc::new(llm::LlmClient::new());

    eprintln!("apic ready. Listening... (Ctrl+C to quit)\n");

    loop {
        // 1. Wait for VAD to deliver a complete utterance.
        let samples = match utterance_rx.recv().await {
            Some(s) => s,
            None => {
                eprintln!("[main] capture loop closed — exiting");
                break;
            }
        };

        // 2. Transcribe on the STT thread (Metal/GPU via whisper-rs).
        let (stt_reply_tx, stt_reply_rx) = oneshot::channel();
        stt_tx
            .send((samples, stt_reply_tx))
            .map_err(|_| {
                anyhow::anyhow!("STT thread died — is the model at {WHISPER_MODEL}?")
            })?;

        let transcript = match stt_reply_rx.await? {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[stt] error: {e}");
                continue;
            }
        };

        if transcript.trim().is_empty() {
            eprintln!("(silence — try again)\n");
            continue;
        }
        eprintln!("You: {transcript}");

        // 3. Stream LLM response; TTS + play each sentence as it arrives.
        let (sentence_tx, mut sentence_rx) =
            tokio::sync::mpsc::unbounded_channel::<String>();

        // Spawn the streaming task so it runs concurrently with our TTS loop.
        let client_clone = Arc::clone(&client);
        let transcript_clone = transcript.clone();
        tokio::spawn(async move {
            if let Err(e) = client_clone
                .stream_sentences(&transcript_clone, sentence_tx)
                .await
            {
                eprintln!("[llm] stream error: {e}");
            }
        });

        // Drain stale barge-in signals from the previous turn.
        while barge_in_rx.try_recv().is_ok() {}
        is_speaking.store(false, Ordering::Relaxed);

        let mut interrupted = false;

        while let Some(sentence) = sentence_rx.recv().await {
            if sentence.trim().is_empty() {
                continue;
            }
            eprintln!("apic: {sentence}");

            // TTS on the Kokoro thread.
            let (tts_reply_tx, tts_reply_rx) = oneshot::channel();
            tts_tx
                .send((sentence, tts_reply_tx))
                .map_err(|_| anyhow::anyhow!("TTS thread died"))?;

            let audio_samples = tts_reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("TTS reply channel closed"))?;

            // Play, flagging is_speaking so the capture loop can detect barge-in.
            is_speaking.store(true, Ordering::Relaxed);
            if let Err(e) =
                tokio::task::spawn_blocking(move || audio::play(&audio_samples, 24_000)).await?
            {
                eprintln!("[audio] playback error: {e}");
            }
            is_speaking.store(false, Ordering::Relaxed);

            // Check barge-in between sentences.
            if barge_in_rx.try_recv().is_ok() {
                eprintln!("[barge-in] interrupting\n");
                interrupted = true;
                break;
            }
        }

        is_speaking.store(false, Ordering::Relaxed);

        if interrupted {
            // Dropping sentence_rx causes the LLM stream task to see a closed
            // channel on its next send() and stop naturally.
            drop(sentence_rx);
        }

        eprintln!();
    }

    Ok(())
}
