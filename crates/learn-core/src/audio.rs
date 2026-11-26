use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::models::AudioDemoSpec;

const SAMPLE_RATE: u32 = 44_100;

/// Synthesizes a clean 2-bar procedural audio preview (16-bit mono PCM `.wav`) in pure Rust.
/// When `dry_before` is true, renders the flat/unprocessed "Before" reference so the learner
/// can A/B compare the raw sound against the processed/target effect.
pub fn synthesize_audio_wav(spec: &AudioDemoSpec, dry_before: bool, out_path: &Path) -> Result<PathBuf> {
    let bpm = spec.bpm.clamp(70.0, 170.0);
    let base_hz = spec.base_hz.clamp(45.0, 880.0);
    let step_sec = 60.0 / bpm / 4.0; // 16th note duration
    let total_steps = 32usize; // 2 bars of 16th notes (~3.5-4s)
    let total_samples = ((total_steps as f32) * step_sec * (SAMPLE_RATE as f32)) as usize;

    // Simple 1-bar repeating synth motif (scale degrees in semitones relative to base_hz)
    let motif_pattern: [Option<f32>; 16] = [
        Some(0.0),
        None,
        Some(0.0),
        Some(3.0),
        None,
        Some(7.0),
        None,
        Some(5.0),
        Some(0.0),
        None,
        Some(3.0),
        None,
        Some(7.0),
        None,
        Some(10.0),
        Some(7.0),
    ];

    let mut samples = vec![0.0f32; total_samples];

    // 1. Synth motif with 1-pole low-pass filter sweep
    let mut synth_phase = 0.0f32;
    let mut lp_state = 0.0f32;
    let cutoff_start = spec.cutoff_start_hz.clamp(120.0, 12_000.0);
    let cutoff_end = if dry_before {
        cutoff_start
    } else {
        spec.cutoff_end_hz.clamp(120.0, 14_000.0)
    };

    for i in 0..total_samples {
        let t = (i as f32) / (SAMPLE_RATE as f32);
        let progress = (i as f32) / (total_samples.max(1) as f32);
        let step_f = t / step_sec;
        let step_idx = (step_f as usize) % 16;
        let step_pos = step_f.fract();

        let note_semis = motif_pattern[step_idx];
        let mut raw_synth = 0.0f32;
        if let Some(semis) = note_semis {
            let freq = base_hz * 2.0f32.powf(semis / 12.0);
            synth_phase = (synth_phase + freq / (SAMPLE_RATE as f32)).fract();
            // Warm sawtooth + subtle sub sine
            let saw = 2.0 * synth_phase - 1.0;
            let sub = (2.0 * PI * synth_phase).sin();
            let env = (step_pos * 18.0).min(1.0) * (-step_pos * 4.5).exp();
            raw_synth = (0.7 * saw + 0.3 * sub) * env * 0.42;
        }

        // Exponential filter cutoff interpolation
        let cutoff = cutoff_start * (cutoff_end / cutoff_start).powf(progress);
        let rc = 1.0 / (2.0 * PI * cutoff);
        let dt = 1.0 / (SAMPLE_RATE as f32);
        let alpha = (dt / (rc + dt)).clamp(0.005, 0.99);
        lp_state += alpha * (raw_synth - lp_state);
        samples[i] += lp_state;
    }

    // 2. Optional 4-on-the-floor kick + 16th closed hi-hats with accent & microtiming nudge
    if spec.include_drums || spec.accent_step.is_some() {
        let mut lcg: u32 = 0x1234_5678;
        let mut next_noise = || -> f32 {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((lcg >> 16) as f32 / 32768.0) - 1.0
        };

        for bar in 0..2usize {
            for step in 0..16usize {
                let global_step = bar * 16 + step;
                let base_start_sec = (global_step as f32) * step_sec;

                // Kick on steps 0, 4, 8, 12
                if step % 4 == 0 {
                    let kick_start = (base_start_sec * (SAMPLE_RATE as f32)) as usize;
                    let kick_len = (0.18 * (SAMPLE_RATE as f32)) as usize;
                    let mut k_phase = 0.0f32;
                    for k in 0..kick_len {
                        let idx = kick_start + k;
                        if idx >= total_samples {
                            break;
                        }
                        let kt = (k as f32) / (SAMPLE_RATE as f32);
                        let freq = 48.0 + 115.0 * (-kt * 32.0).exp();
                        k_phase += 2.0 * PI * freq / (SAMPLE_RATE as f32);
                        let env = (-kt * 14.0).exp();
                        samples[idx] += k_phase.sin() * env * 0.48;
                    }
                }

                // Steady 16th closed hi-hat with optional accent + microtiming nudge
                let is_target_step = spec
                    .accent_step
                    .map(|s| s.clamp(1, 16) as usize - 1 == step)
                    .unwrap_or(false);
                let nudge_sec = if !dry_before && is_target_step {
                    (spec.nudge_ms.clamp(-45.0, 45.0)) / 1000.0
                } else {
                    0.0
                };
                let hat_start_sec = (base_start_sec + nudge_sec).max(0.0);
                let hat_start = (hat_start_sec * (SAMPLE_RATE as f32)) as usize;
                let hat_len = (0.045 * (SAMPLE_RATE as f32)) as usize;
                let hat_gain = if !dry_before && is_target_step {
                    0.26
                } else if step % 4 == 2 {
                    0.12
                } else {
                    0.08
                };

                let mut prev_n = 0.0f32;
                for h in 0..hat_len {
                    let idx = hat_start + h;
                    if idx >= total_samples {
                        break;
                    }
                    let ht = (h as f32) / (SAMPLE_RATE as f32);
                    let n = next_noise();
                    // Simple high-pass difference for crisp metallic hat transient
                    let hp = n - prev_n * 0.85;
                    prev_n = n;
                    let env = (-ht * 75.0).exp();
                    samples[idx] += hp * env * hat_gain;
                }
            }
        }
    }

    // Soft-clip & write standard 16-bit mono PCM WAV
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let data_bytes = (samples.len() * 2) as u32;
    let riff_size = 36 + data_bytes;
    let mut wav = Vec::with_capacity(44 + data_bytes as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());

    for s in samples {
        let shaped = (s * 0.92).tanh();
        let pcm = (shaped * i16::MAX as f32) as i16;
        wav.extend_from_slice(&pcm.to_le_bytes());
    }

    std::fs::write(out_path, &wav)
        .with_context(|| format!("write synthesized WAV to {}", out_path.display()))?;
    Ok(out_path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn synthesizes_valid_pcm_wav_for_filter_and_groove_examples() {
        let spec = AudioDemoSpec {
            bpm: 130.0,
            base_hz: 110.0,
            cutoff_start_hz: 300.0,
            cutoff_end_hz: 3600.0,
            accent_step: Some(16),
            nudge_ms: 14.0,
            include_drums: true,
            before_label: Some("Flat / Muted".into()),
            after_label: Some("Filter Sweep + Laid-back Hat".into()),
        };

        let path = std::env::temp_dir().join(format!("learn-audio-test-{}.wav", Uuid::new_v4()));
        synthesize_audio_wav(&spec, false, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert!(bytes.len() > 100_000);
        let _ = std::fs::remove_file(&path);
    }
}
