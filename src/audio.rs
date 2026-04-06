use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rodio::buffer::SamplesBuffer;
use rodio::{DeviceSinkBuilder, Player};
use std::num::NonZero;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Capture mic audio for `duration_secs` seconds at 16kHz mono f32.
///
/// Opens the default input device, records for the requested duration, then
/// closes the stream. Requests 16kHz mono f32 directly from the device —
/// most Apple Silicon macs support this natively.
pub fn record(duration_secs: f32) -> Result<Vec<f32>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No default input device found"))?;

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16_000),
        buffer_size: cpal::BufferSize::Fixed(1024),
    };

    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_cb = Arc::clone(&samples);

    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                samples_cb.lock().unwrap().extend_from_slice(data);
            },
            |e| eprintln!("cpal input error: {e}"),
            None,
        )
        .map_err(|e| anyhow::anyhow!("Failed to build input stream: {e}"))?;

    stream
        .play()
        .map_err(|e| anyhow::anyhow!("Failed to start input stream: {e}"))?;

    std::thread::sleep(Duration::from_secs_f32(duration_secs));

    // Drop the stream before reading the buffer — ensures the callback is no
    // longer running and the mutex is uncontested.
    drop(stream);

    Ok(samples.lock().unwrap().to_vec())
}

/// Play `samples` through the default output device.
///
/// Blocks until playback is complete. Rodio resamples from `sample_rate` to
/// the device's native rate automatically, so 24kHz TTS output plays without
/// any manual conversion.
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
