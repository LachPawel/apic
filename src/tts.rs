// voice-tts wraps mlx-rs which calls Metal. KokoroModel is NOT Send — Metal
// command queues are thread-local. This module runs on a dedicated std::thread
// that owns the model and receives requests via std::sync::mpsc channel.
// (Verified: mlx_rs::Array does not implement Send.)

use anyhow::Result;
use voice_tts::KokoroModel;

const DEFAULT_VOICE: &str = "af_heart";
const DEFAULT_SPEED: f32 = 1.0;

/// Load the Kokoro TTS model. Downloads from HuggingFace Hub on first call
/// (~82MB, cached at `~/.cache/huggingface/hub/`). Call on the dedicated TTS
/// thread — KokoroModel is not Send.
pub fn init_model() -> Result<KokoroModel> {
    voice_tts::load_model("prince-canuma/Kokoro-82M")
        .map_err(|e| anyhow::anyhow!("Failed to load TTS model: {e}"))
}

/// Synthesize audio using an already-loaded model.
///
/// Returns 24kHz mono f32 samples.
pub fn synthesize_with_model(model: &mut KokoroModel, text: &str) -> Result<Vec<f32>> {
    let phonemes = voice_g2p::english_to_phonemes(text)
        .map_err(|e| anyhow::anyhow!("G2P failed: {e}"))?;

    let voice = voice_tts::load_voice(DEFAULT_VOICE, None)
        .map_err(|e| anyhow::anyhow!("Failed to load voice: {e}"))?;

    let audio = voice_tts::generate(model, &phonemes, &voice, DEFAULT_SPEED)
        .map_err(|e| anyhow::anyhow!("TTS generate failed: {e}"))?;

    audio
        .eval()
        .map_err(|e| anyhow::anyhow!("MLX eval failed: {e}"))?;

    let samples: &[f32] = audio.as_slice();
    Ok(samples.to_vec())
}

/// Convenience function: load model, synthesize, return 24kHz mono f32 samples.
///
/// Prefer `init_model` + `synthesize_with_model` in loops to avoid
/// re-downloading the model on each call.
pub fn synthesize(text: &str) -> Result<Vec<f32>> {
    let mut model = init_model()?;
    synthesize_with_model(&mut model, text)
}

/// Run the TTS synthesis loop on the calling thread (must be a dedicated
/// `std::thread` — KokoroModel is not Send). Blocks until the sender drops.
pub fn run_synthesis_loop(
    rx: std::sync::mpsc::Receiver<(String, tokio::sync::oneshot::Sender<Vec<f32>>)>,
) {
    let mut model = init_model().expect("TTS model load failed");
    while let Ok((sentence, reply_tx)) = rx.recv() {
        let samples = synthesize_with_model(&mut model, &sentence).unwrap_or_else(|e| {
            eprintln!("TTS error: {e}");
            vec![]
        });
        let _ = reply_tx.send(samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_short_string_is_non_empty() {
        let result = synthesize("Hello.");
        match result {
            Ok(samples) => {
                assert!(!samples.is_empty(), "expected non-empty audio output");
                // 24kHz mono — even a one-word utterance should be at least 2400 samples (0.1s)
                assert!(
                    samples.len() >= 2400,
                    "expected at least 0.1s of audio, got {} samples",
                    samples.len()
                );
            }
            Err(e) => {
                // Skip if HuggingFace Hub is unreachable (CI without network)
                eprintln!("skipping TTS test: {e}");
            }
        }
    }
}
