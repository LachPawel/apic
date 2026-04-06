use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rodio::buffer::SamplesBuffer;
use rodio::{DeviceSinkBuilder, Player};
use std::num::NonZero;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const TARGET_HZ: u32 = 16_000;

/// Capture mic audio for `duration_secs` seconds, returned as 16kHz mono f32.
///
/// Records at the device's native rate/format, then converts to mono and
/// resamples to 16kHz via linear interpolation. This avoids the
/// "stream config not supported" error that comes from forcing 16kHz on
/// devices whose native rate is 44100 or 48000 Hz.
pub fn record(duration_secs: f32) -> Result<Vec<f32>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No default input device found"))?;

    let supported = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Cannot get default input config: {e}"))?;

    let channels = supported.channels();
    let device_hz = supported.sample_rate().0;

    let config = cpal::StreamConfig {
        channels,
        sample_rate: supported.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let raw: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let raw_cb = Arc::clone(&raw);

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                raw_cb.lock().unwrap().extend_from_slice(data);
            },
            |e| eprintln!("cpal error: {e}"),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                raw_cb
                    .lock()
                    .unwrap()
                    .extend(data.iter().map(|&s| s as f32 / 32_768.0));
            },
            |e| eprintln!("cpal error: {e}"),
            None,
        )?,
        fmt => {
            return Err(anyhow::anyhow!(
                "Unsupported sample format {fmt:?}. Expected F32 or I16."
            ))
        }
    };

    stream
        .play()
        .map_err(|e| anyhow::anyhow!("Failed to start input stream: {e}"))?;

    std::thread::sleep(Duration::from_secs_f32(duration_secs));
    drop(stream); // stops callbacks before we read the buffer

    let raw = raw.lock().unwrap().to_vec();

    // Mix down to mono.
    let mono: Vec<f32> = if channels == 1 {
        raw
    } else {
        raw.chunks_exact(channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // Resample to 16kHz if needed.
    if device_hz == TARGET_HZ {
        return Ok(mono);
    }

    let out_len = (mono.len() as f64 * TARGET_HZ as f64 / device_hz as f64) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * device_hz as f64 / TARGET_HZ as f64;
        let idx = src as usize;
        let frac = (src - idx as f64) as f32;
        let a = mono.get(idx).copied().unwrap_or(0.0);
        let b = mono.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }

    Ok(out)
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
