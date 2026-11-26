use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::models::AudioDemoSpec;

const SAMPLE_RATE: u32 = 44_100;

/// Synthesizes a clean 2-bar procedural audio preview (16-bit mono PCM `.wav`) in pure Rust.
pub fn synthesize_audio_wav(spec: &AudioDemoSpec, dry_before: bool, out_path: &Path) -> Result<PathBuf> {
    synthesize_audio_wav_bars(spec, dry_before, 2, out_path)
}

/// Synthesizes an exact `total_bars`-long sample-accurate procedural `.wav` (e.g. 16 bars for
/// gapless looped playback, or 2 bars for a single preview pass).
pub fn synthesize_audio_wav_bars(
    spec: &AudioDemoSpec,
    dry_before: bool,
    total_bars: usize,
    out_path: &Path,
) -> Result<PathBuf> {
    let bpm = spec.bpm.clamp(70.0, 170.0);
    let base_hz = spec.base_hz.clamp(45.0, 880.0);
    let voice = spec.voice.trim().to_lowercase();
    let total_bars = total_bars.clamp(1, 32);

    let active_levels: &[f32] = if !dry_before && !spec.after_step_levels.is_empty() {
        &spec.after_step_levels
    } else {
        &spec.step_levels
    };

    // Match the exact step grid shown visually in the UI (4, 8, or 16 steps per bar).
    let steps_per_bar = match active_levels.len() {
        4 => 4usize,
        8 => 8usize,
        _ => 16usize,
    };
    let bar_sec = (60.0 / bpm) * 4.0;
    let step_sec = bar_sec / (steps_per_bar as f32);
    let total_samples = ((total_bars as f32) * bar_sec * (SAMPLE_RATE as f32)) as usize;
    let mut samples = vec![0.0f32; total_samples];

    let mut lcg: u32 = 0x1234_5678;
    let mut next_noise = || -> f32 {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((lcg >> 16) as f32 / 32768.0) - 1.0
    };

    match voice.as_str() {
        "rim_click" => {
            // Crisp rim clicks on the step grid driven directly by `active_levels` + optional pulse kick
            for bar in 0..total_bars {
                for step in 0..steps_per_bar {
                    let level = active_levels
                        .get(step)
                        .copied()
                        .unwrap_or_else(|| {
                            let is_acc = spec
                                .accent_step
                                .map(|s| s.saturating_sub(1) as usize % steps_per_bar == step)
                                .unwrap_or(false);
                            if is_acc { 1.0 } else { 0.45 }
                        })
                        .clamp(0.0, 1.0);

                    if level <= 0.02 {
                        continue;
                    }
                    let is_acc = spec
                        .accent_step
                        .map(|s| s.saturating_sub(1) as usize % steps_per_bar == step)
                        .unwrap_or(false);
                    let nudge_sec = if !dry_before && is_acc {
                        spec.nudge_ms.clamp(-45.0, 45.0) / 1000.0
                    } else {
                        0.0
                    };
                    let start_sec = ((bar * steps_per_bar + step) as f32 * step_sec + nudge_sec).max(0.0);
                    let start_idx = (start_sec * SAMPLE_RATE as f32) as usize;
                    let click_len = (0.045 * SAMPLE_RATE as f32) as usize;

                    for k in 0..click_len {
                        let idx = start_idx + k;
                        if idx >= total_samples {
                            break;
                        }
                        let kt = k as f32 / SAMPLE_RATE as f32;
                        let body = (2.0 * PI * 920.0 * kt).sin() * (-kt * 110.0).exp();
                        let snap = (2.0 * PI * 2450.0 * kt).sin() * (-kt * 210.0).exp();
                        let transient = next_noise() * (-kt * 280.0).exp() * 0.35;
                        samples[idx] += (body * 0.65 + snap * 0.45 + transient) * level * 0.65;
                    }
                }
            }
            if spec.include_drums {
                add_four_on_floor_kick(&mut samples, bar_sec, total_bars);
            }
        }
        "kick_bass" => {
            // Punchy 4-on-the-floor kick + sub bassline (with clean ducking envelope in "after" mode)
            add_four_on_floor_kick(&mut samples, bar_sec, total_bars);
            let beat_sec = bar_sec / 4.0;
            let mut phase = 0.0f32;
            for i in 0..total_samples {
                let t = i as f32 / SAMPLE_RATE as f32;
                let beat_pos = (t / beat_sec).fract();
                // Dry/before = bass clashes straight through kick; After = clean sidechain/ducked pocket
                let env = if dry_before {
                    0.85
                } else {
                    (beat_pos * 4.5).clamp(0.0, 1.0).powf(1.6) * (1.0 - beat_pos * 0.25)
                };
                phase = (phase + base_hz / SAMPLE_RATE as f32).fract();
                let bass = (2.0 * PI * phase).sin() * 0.75 + (2.0 * phase - 1.0) * 0.25;
                samples[i] += bass * env * 0.40;
            }
        }
        "delay" => {
            // Sparse stab motif + dotted-eighth rhythmic feedback delay in "after" mode
            let stab_sec = bar_sec / 2.0; // 2 stabs per bar
            for bar in 0..total_bars {
                for hit in 0..2usize {
                    let start = (((bar * 2 + hit) as f32) * stab_sec * SAMPLE_RATE as f32) as usize;
                    let len = (0.14 * SAMPLE_RATE as f32) as usize;
                    let freq = if hit == 0 { base_hz * 2.0 } else { base_hz * 2.5 };
                    for k in 0..len {
                        let idx = start + k;
                        if idx >= total_samples {
                            break;
                        }
                        let kt = k as f32 / SAMPLE_RATE as f32;
                        let env = (kt * 80.0).min(1.0) * (-kt * 24.0).exp();
                        let tone = (2.0 * PI * freq * kt).sin() * 0.6
                            + (2.0 * PI * freq * 1.5 * kt).sin() * 0.3;
                        samples[idx] += tone * env * 0.55;
                    }
                }
            }
            if !dry_before {
                // Dotted-8th (3 sixteenth notes) feedback delay
                let delay_samples = ((bar_sec / 16.0 * 3.0) * SAMPLE_RATE as f32) as usize;
                let feedback = 0.52f32;
                let mut lp = 0.0f32;
                for i in delay_samples..total_samples {
                    let delayed = samples[i - delay_samples] * feedback;
                    lp += 0.28 * (delayed - lp);
                    samples[i] += lp;
                }
            }
            if spec.include_drums {
                add_four_on_floor_kick(&mut samples, bar_sec, total_bars);
            }
        }
        _ => {
            // Default: hypnotic 1-bar repeating synth motif with low-pass filter sweep + optional drums
            let motif_pattern: [Option<f32>; 16] = [
                Some(0.0), None, Some(0.0), Some(3.0),
                None, Some(7.0), None, Some(5.0),
                Some(0.0), None, Some(3.0), None,
                Some(7.0), None, Some(10.0), Some(7.0),
            ];
            let sixteenth_sec = bar_sec / 16.0;
            let mut synth_phase = 0.0f32;
            let mut lp_state = 0.0f32;
            let cutoff_start = spec.cutoff_start_hz.clamp(120.0, 12_000.0);
            let cutoff_end = if dry_before {
                cutoff_start
            } else {
                spec.cutoff_end_hz.clamp(120.0, 14_000.0)
            };

            let cycle_samples = ((2.0 * bar_sec * SAMPLE_RATE as f32) as usize).max(1);
            for i in 0..total_samples {
                let t = i as f32 / SAMPLE_RATE as f32;
                let progress = (i % cycle_samples) as f32 / (cycle_samples as f32);
                let step_f = t / sixteenth_sec;
                let step_idx = (step_f as usize) % 16;
                let step_pos = step_f.fract();

                let mut raw_synth = 0.0f32;
                if let Some(semis) = motif_pattern[step_idx] {
                    let freq = base_hz * 2.0f32.powf(semis / 12.0);
                    synth_phase = (synth_phase + freq / SAMPLE_RATE as f32).fract();
                    let saw = 2.0 * synth_phase - 1.0;
                    let sub = (2.0 * PI * synth_phase).sin();
                    let step_vel = active_levels.get(step_idx).copied().unwrap_or(0.85).clamp(0.1, 1.0);
                    let env = (step_pos * 18.0).min(1.0) * (-step_pos * 4.5).exp();
                    raw_synth = (0.7 * saw + 0.3 * sub) * env * step_vel * 0.44;
                }

                let cutoff = cutoff_start * (cutoff_end / cutoff_start).powf(progress);
                let rc = 1.0 / (2.0 * PI * cutoff);
                let dt = 1.0 / SAMPLE_RATE as f32;
                let alpha = (dt / (rc + dt)).clamp(0.005, 0.99);
                lp_state += alpha * (raw_synth - lp_state);
                samples[i] += lp_state;
            }

            if spec.include_drums || spec.accent_step.is_some() {
                add_four_on_floor_kick(&mut samples, bar_sec, total_bars);
                let sixteenth_sec = bar_sec / 16.0;
                for bar in 0..total_bars {
                    for step in 0..16usize {
                        let global_step = bar * 16 + step;
                        let base_start_sec = global_step as f32 * sixteenth_sec;
                        let is_target_step = spec
                            .accent_step
                            .map(|s| s.clamp(1, 16) as usize - 1 == step)
                            .unwrap_or(false);
                        let nudge_sec = if !dry_before && is_target_step {
                            spec.nudge_ms.clamp(-45.0, 45.0) / 1000.0
                        } else {
                            0.0
                        };
                        let hat_start = ((base_start_sec + nudge_sec).max(0.0) * SAMPLE_RATE as f32) as usize;
                        let hat_len = (0.045 * SAMPLE_RATE as f32) as usize;
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
                            let ht = h as f32 / SAMPLE_RATE as f32;
                            let n = next_noise();
                            let hp = n - prev_n * 0.85;
                            prev_n = n;
                            let env = (-ht * 75.0).exp();
                            samples[idx] += hp * env * hat_gain;
                        }
                    }
                }
            }
        }
    }

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
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
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

fn add_four_on_floor_kick(samples: &mut [f32], bar_sec: f32, total_bars: usize) {
    let beat_sec = bar_sec / 4.0;
    let kick_len = (0.18 * SAMPLE_RATE as f32) as usize;
    for bar in 0..total_bars {
        for beat in 0..4usize {
            let start = (((bar * 4 + beat) as f32) * beat_sec * SAMPLE_RATE as f32) as usize;
            let mut k_phase = 0.0f32;
            for k in 0..kick_len {
                let idx = start + k;
                if idx >= samples.len() {
                    break;
                }
                let kt = k as f32 / SAMPLE_RATE as f32;
                let freq = 48.0 + 115.0 * (-kt * 32.0).exp();
                k_phase += 2.0 * PI * freq / SAMPLE_RATE as f32;
                let env = (-kt * 14.0).exp();
                samples[idx] += k_phase.sin() * env * 0.48;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn synthesizes_valid_pcm_wav_for_rim_click_and_filter_examples() {
        for voice in ["synth", "rim_click", "kick_bass", "delay"] {
            let spec = AudioDemoSpec {
                voice: voice.into(),
                bpm: 130.0,
                base_hz: 110.0,
                cutoff_start_hz: 300.0,
                cutoff_end_hz: 3600.0,
                step_levels: vec![0.45, 0.45, 1.0, 0.45],
                after_step_levels: vec![0.45, 0.45, 0.45, 1.0],
                step_labels: vec!["1".into(), "2".into(), "3".into(), "4".into()],
                accent_step: Some(4),
                nudge_ms: 12.0,
                include_drums: true,
                before_label: Some("Bars 1-7 (Beat 3 accent)".into()),
                after_label: Some("Bar 8 (Beat 4 accent)".into()),
            };

            let path = std::env::temp_dir().join(format!("learn-audio-{voice}-{}.wav", Uuid::new_v4()));
            synthesize_audio_wav(&spec, false, &path).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(&bytes[0..4], b"RIFF");
            assert_eq!(&bytes[8..12], b"WAVE");
            assert!(bytes.len() > 100_000);
            let _ = std::fs::remove_file(&path);
        }
    }
}
