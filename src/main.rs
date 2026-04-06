mod audio;
mod llm;
mod stt;
mod tts;

use anyhow::Result;
use std::path::PathBuf;
use tokio::sync::oneshot;

const RECORD_SECS: f32 = 4.0;
const MODEL_PATH: &str = "models/ggml-small.en.bin";

#[tokio::main]
async fn main() -> Result<()> {
    // STT: WhisperContext is not Send (raw pointer) — runs on a dedicated thread.
    let (stt_tx, stt_rx) =
        std::sync::mpsc::channel::<(Vec<f32>, oneshot::Sender<Result<String>>)>();
    let model_path = PathBuf::from(MODEL_PATH);
    std::thread::spawn(move || stt::run_transcribe_loop(&model_path, stt_rx));

    // TTS: KokoroModel is not Send (Metal thread-local) — runs on a dedicated thread.
    let (tts_tx, tts_rx) =
        std::sync::mpsc::channel::<(String, oneshot::Sender<Vec<f32>>)>();
    std::thread::spawn(move || tts::run_synthesis_loop(tts_rx));

    let client = llm::LlmClient::new();

    eprintln!("apic ready. Speak after the prompt ({}s window).", RECORD_SECS);
    eprintln!("Press Ctrl+C to quit.\n");

    loop {
        // 1. Capture mic audio on a blocking thread (record() is blocking).
        eprint!("Listening... ");
        let samples = match tokio::task::spawn_blocking(|| audio::record(RECORD_SECS))
            .await?
        {
            Ok(s) => {
                eprintln!("done ({} samples)", s.len());
                s
            }
            Err(e) => {
                eprintln!("mic error: {e}");
                continue;
            }
        };

        // 2. Transcribe on the dedicated STT thread.
        let (stt_reply_tx, stt_reply_rx) = oneshot::channel();
        // Fatal: if the STT thread died the model is gone, nothing to recover.
        stt_tx
            .send((samples, stt_reply_tx))
            .map_err(|_| anyhow::anyhow!("STT thread died — is the model at {MODEL_PATH}?"))?;
        let transcript = match stt_reply_rx.await? {
            Ok(t) => t,
            Err(e) => {
                eprintln!("STT error: {e}");
                continue;
            }
        };

        if transcript.trim().is_empty() {
            eprintln!("(silence — try again)\n");
            continue;
        }
        eprintln!("You: {transcript}");

        // 3. Send to LLM.
        eprint!("Thinking... ");
        let response = match client.send_message(&transcript).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("LLM error: {e}");
                continue;
            }
        };
        eprintln!("\napic: {response}");

        // 4. Synthesize on the dedicated TTS thread.
        eprint!("Synthesizing... ");
        let (tts_reply_tx, tts_reply_rx) = oneshot::channel();
        // Fatal: TTS thread owns the model, can't recover if it died.
        tts_tx
            .send((response, tts_reply_tx))
            .map_err(|_| anyhow::anyhow!("TTS thread died"))?;
        let audio_samples = tts_reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("TTS reply channel closed"))?;
        eprintln!("done ({} samples at 24kHz)", audio_samples.len());

        // 5. Play back on a blocking thread (play() blocks until done).
        eprint!("Playing... ");
        if let Err(e) =
            tokio::task::spawn_blocking(move || audio::play(&audio_samples, 24_000)).await?
        {
            eprintln!("playback error: {e}");
            continue;
        }
        eprintln!("done\n");
    }
}
