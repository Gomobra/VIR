//! Anggaran token per aset (ADR-009 §2–§4) — dihitung **sebelum** encoder
//! dijalankan supaya prefix bisa dijadwalkan terhadap `P` (F2.3) dan `L` (F1.1).

use vir_common::VirError;

use crate::asset::ModalAsset;
use crate::encoder::EncoderSpec;
use crate::kind::ModalKind;

/// Ukuran patch bawaan encoder visi (ADR-009 §2: 224/16 → 196, 384/16 → 576).
pub const DEFAULT_PATCH: u32 = 16;

/// Laju keluaran bawaan encoder ucapan (posisi per detik).
pub const SPEECH_POSITIONS_PER_SEC: f32 = 50.0;

/// Laju kode SNAC 24 kHz per level (Hz): stride `[4, 2, 1]` (ADR-009 §3).
pub const SNAC24K_RATES: [f32; 3] = [12.0, 23.0, 47.0];

/// Bit per kode SNAC (codebook 4096).
pub const SNAC_CODEBOOK_BITS: u32 = 12;

/// Token prefix untuk gambar: `N = (H/p)·(W/p)`.
pub fn image_tokens(height: u32, width: u32, patch: u32) -> Result<usize, VirError> {
    if patch == 0 {
        return Err(VirError::Config("ukuran patch harus > 0".into()));
    }
    if height % patch != 0 || width % patch != 0 {
        return Err(VirError::Config(format!(
            "dimensi {height}x{width} tidak habis dibagi patch {patch}"
        )));
    }
    let h = u64::from(height / patch);
    let w = u64::from(width / patch);
    Ok((h * w) as usize)
}

/// Token prefix audio (jalur fitur): `durasi × posisi per detik`.
pub fn audio_understanding_tokens(
    duration_ms: u32,
    positions_per_sec: f32,
) -> Result<usize, VirError> {
    if duration_ms == 0 {
        return Err(VirError::Config("durasi audio 0".into()));
    }
    if positions_per_sec <= 0.0 {
        return Err(VirError::Config("posisi per detik harus > 0".into()));
    }
    let secs = f64::from(duration_ms) / 1_000.0;
    Ok((secs * f64::from(positions_per_sec)).round() as usize)
}

/// Jumlah kode generatif: `Σ_k round(durasi × laju_k)`.
pub fn audio_codes(duration_ms: u32, level_rates: &[f32]) -> Result<usize, VirError> {
    if duration_ms == 0 {
        return Err(VirError::Config("durasi audio 0".into()));
    }
    if level_rates.is_empty() {
        return Err(VirError::Config("daftar laju level kosong".into()));
    }
    let secs = f64::from(duration_ms) / 1_000.0;
    let mut total = 0usize;
    for rate in level_rates {
        if *rate <= 0.0 {
            return Err(VirError::Config("laju level harus > 0".into()));
        }
        total += (secs * f64::from(*rate)).round() as usize;
    }
    Ok(total)
}

/// Bitrate efektif `bps = Σ_k laju_k × bit_per_kode` (pemeriksaan ADR-009 §3).
pub fn audio_bitrate(level_rates: &[f32], bits_per_code: u32) -> f32 {
    level_rates
        .iter()
        .fold(0.0_f32, |acc, r| acc + r * bits_per_code as f32)
}

/// Kode SNAC 24 kHz untuk `duration_ms` milidetik.
pub fn snac24k_codes(duration_ms: u32) -> Result<usize, VirError> {
    audio_codes(duration_ms, &SNAC24K_RATES)
}

/// Bitrate SNAC 24 kHz (harus ≈ 984 bps).
pub fn snac24k_bitrate() -> f32 {
    audio_bitrate(&SNAC24K_RATES, SNAC_CODEBOOK_BITS)
}

/// Token video: frame × token per frame (ADR-009 §4).
pub fn video_tokens(frames: u32, per_frame: usize) -> Result<usize, VirError> {
    if frames == 0 {
        return Err(VirError::Config("video tanpa frame".into()));
    }
    Ok(u64::from(frames) as usize * per_frame)
}

/// Anggaran token jalur fitur untuk satu aset, memakai spesifikasi encoder.
pub fn prefix_budget(spec: &EncoderSpec, asset: &ModalAsset) -> Result<usize, VirError> {
    if spec.kind != asset.kind && !(spec.kind == ModalKind::Image && asset.kind == ModalKind::Video)
    {
        return Err(VirError::Config(format!(
            "spec {} tidak cocok aset {}",
            spec.kind, asset.kind
        )));
    }
    match asset.kind {
        ModalKind::Image => {
            let per_frame = image_tokens(asset.height, asset.width, spec.patch)?;
            Ok(per_frame)
        }
        ModalKind::Video => {
            let per_frame = image_tokens(asset.height, asset.width, spec.patch)?;
            video_tokens(asset.frames, per_frame)
        }
        ModalKind::Audio => {
            // `EncoderSpec::patch` = hop (sample per frame), `stride` = faktor
            // penurunan laju frame keluaran.
            if spec.patch == 0 {
                return Err(VirError::Config("hop encoder audio 0".into()));
            }
            let stride = if spec.stride == 0 {
                1
            } else {
                u64::from(spec.stride)
            };
            let total = u64::from(asset.duration_ms) * u64::from(asset.sample_rate) / 1_000;
            let frames = total / u64::from(spec.patch);
            Ok((frames.div_ceil(stride)) as usize)
        }
        ModalKind::Text => Err(VirError::Config(
            "teks tidak memakai encoder modalitas".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gambar_224_dan_384() -> Result<(), VirError> {
        assert_eq!(image_tokens(224, 224, 16)?, 196);
        assert_eq!(image_tokens(384, 384, 16)?, 576);
        assert!(image_tokens(225, 224, 16).is_err());
        assert!(image_tokens(224, 224, 0).is_err());
        Ok(())
    }

    #[test]
    fn audio_10_detik() -> Result<(), VirError> {
        assert_eq!(
            audio_understanding_tokens(10_000, SPEECH_POSITIONS_PER_SEC)?,
            500
        );
        assert_eq!(snac24k_codes(10_000)?, 820);
        assert!((snac24k_bitrate() - 984.0).abs() < 1e-3);
        assert!(snac24k_codes(0).is_err());
        Ok(())
    }

    #[test]
    fn video_dan_prefix_budget() -> Result<(), VirError> {
        let vision = EncoderSpec::vision("siglip", "2026-01", DEFAULT_PATCH);
        let image = ModalAsset::image("a.png", 384, 384);
        assert_eq!(prefix_budget(&vision, &image)?, 576);

        let mut vid = ModalAsset::video("v.mp4", 224, 224, 4_000, 10);
        assert_eq!(vid.frames, 40);
        assert_eq!(prefix_budget(&vision, &vid)?, 40 * 196);
        vid.frames = 0;
        assert!(prefix_budget(&vision, &vid).is_err());

        let speech = EncoderSpec::speech("whisper", "2026-01", 160, 2, 16_000);
        let audio = ModalAsset::audio("a.wav", 5_000, 16_000, 1);
        assert_eq!(prefix_budget(&speech, &audio)?, 250);
        assert!(prefix_budget(&vision, &audio).is_err());
        Ok(())
    }
}
