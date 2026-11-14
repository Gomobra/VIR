//! Aset modalitas: referensi file + metadata yang dibutuhkan anggaran token
//! (ADR-009 §6). Byte aset **tidak** disimpan di sini; hanya path, dimensi, dan
//! sidik jari opsional, supaya shard data tetap kecil dan bisa di-dedup.

use serde::{Deserialize, Serialize};
use vir_common::VirError;

use crate::kind::ModalKind;

/// Satu aset (gambar/audio/video) yang dirujuk sebuah sampel berpasangan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalAsset {
    pub kind: ModalKind,
    /// Path relatif terhadap akar dataset.
    pub path: String,
    /// Sidik jari SHA-256 (hex) bila sudah dihitung.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    #[serde(default)]
    pub duration_ms: u32,
    #[serde(default)]
    pub sample_rate: u32,
    #[serde(default)]
    pub channels: u16,
    /// Jumlah frame (khusus video).
    #[serde(default)]
    pub frames: u32,
    /// Frame per detik (khusus video).
    #[serde(default)]
    pub fps: u16,
}

impl ModalAsset {
    pub fn image(path: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            kind: ModalKind::Image,
            path: path.into(),
            sha256: None,
            width,
            height,
            duration_ms: 0,
            sample_rate: 0,
            channels: 0,
            frames: 0,
            fps: 0,
        }
    }

    pub fn audio(
        path: impl Into<String>,
        duration_ms: u32,
        sample_rate: u32,
        channels: u16,
    ) -> Self {
        Self {
            kind: ModalKind::Audio,
            path: path.into(),
            sha256: None,
            width: 0,
            height: 0,
            duration_ms,
            sample_rate,
            channels,
            frames: 0,
            fps: 0,
        }
    }

    pub fn video(
        path: impl Into<String>,
        width: u32,
        height: u32,
        duration_ms: u32,
        fps: u16,
    ) -> Self {
        let frames = u64::from(duration_ms) * u64::from(fps) / 1_000;
        Self {
            kind: ModalKind::Video,
            path: path.into(),
            sha256: None,
            width,
            height,
            duration_ms,
            sample_rate: 0,
            channels: 0,
            frames: frames.min(u64::from(u32::MAX)) as u32,
            fps,
        }
    }

    pub fn with_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.sha256 = Some(sha256.into());
        self
    }

    /// Detik (presisi `f32`; cukup untuk anggaran token).
    pub fn duration_s(&self) -> f32 {
        self.duration_ms as f32 / 1_000.0
    }

    /// Cek konsistensi field dengan jenis modalitas.
    pub fn validate(&self) -> Result<(), VirError> {
        if self.path.trim().is_empty() {
            return Err(VirError::Config("path aset kosong".into()));
        }
        if self.sha256.as_deref().is_some_and(|sha| sha.len() != 64) {
            return Err(VirError::Config(format!(
                "sha256 harus 64 hex: {}",
                self.path
            )));
        }
        match self.kind {
            ModalKind::Text => Err(VirError::Config(
                "teks bukan aset; pakai segmen teks".into(),
            )),
            ModalKind::Image => {
                self.require(self.width > 0 && self.height > 0, "dimensi gambar")?;
                self.reject(self.duration_ms > 0, "gambar tidak punya durasi")?;
                self.reject(self.frames > 0, "gambar tidak punya frame")?;
                Ok(())
            }
            ModalKind::Audio => {
                self.require(self.duration_ms > 0, "durasi audio")?;
                self.require(self.sample_rate > 0, "sample rate audio")?;
                self.require(self.channels > 0, "jumlah kanal audio")?;
                self.reject(self.width > 0, "audio tidak punya lebar")?;
                Ok(())
            }
            ModalKind::Video => {
                self.require(self.width > 0 && self.height > 0, "dimensi video")?;
                self.require(self.duration_ms > 0, "durasi video")?;
                self.require(self.fps > 0, "fps video")?;
                self.require(self.frames > 0, "frame video")?;
                Ok(())
            }
        }
    }

    fn require(&self, ok: bool, field: &str) -> Result<(), VirError> {
        if ok {
            Ok(())
        } else {
            Err(VirError::Config(format!("{field} tidak valid: {}", self.path)))
        }
    }

    fn reject(&self, bad: bool, reason: &str) -> Result<(), VirError> {
        if bad {
            Err(VirError::Config(format!("{reason}: {}", self.path)))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gambar_valid() -> Result<(), VirError> {
        let a = ModalAsset::image("img/a.webp", 384, 384).with_sha256("a".repeat(64));
        a.validate()
    }

    #[test]
    fn audio_valid_dan_durasi_detik() -> Result<(), VirError> {
        let a = ModalAsset::audio("aud/a.wav", 10_000, 24_000, 1);
        a.validate()?;
        assert!((a.duration_s() - 10.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn video_menghitung_frame() -> Result<(), VirError> {
        let a = ModalAsset::video("vid/a.mp4", 640, 360, 2_000, 30);
        assert_eq!(a.frames, 60);
        a.validate()
    }

    #[test]
    fn tolak_field_salah() -> Result<(), VirError> {
        assert!(ModalAsset::image("", 4, 4).validate().is_err());
        assert!(ModalAsset::image("a.png", 0, 4).validate().is_err());
        assert!(ModalAsset::audio("a.wav", 0, 16_000, 1).validate().is_err());
        assert!(ModalAsset::audio("a.wav", 1_000, 0, 1).validate().is_err());
        let mut bad = ModalAsset::audio("a.wav", 1_000, 16_000, 1);
        bad.sha256 = Some("abc".into());
        assert!(bad.validate().is_err());
        let text = ModalAsset {
            kind: ModalKind::Text,
            ..ModalAsset::image("a.png", 4, 4)
        };
        assert!(text.validate().is_err());
        Ok(())
    }
}