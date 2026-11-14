//! Trait encoder modalitas (ADR-009 §1 jalur fitur, §7 pemisahan tensor).
//!
//! Crate ini **tidak** memuat tensor: implementor (nanti di `vir-nn`/
//! `vir-model`) memilih tipe `Output` sendiri, mis. `Tensor<3>` fitur atau
//! `Vec<Vec<u32>>` kode. Yang dijamin trait ini: spesifikasi encoder dapat
//! direproduksi, anggaran token bisa dihitung lebih dulu, dan encoder
//! pemahaman **beku** secara bawaan.

use serde::{Deserialize, Serialize};
use vir_common::VirError;

use crate::asset::ModalAsset;
use crate::budget::{self, DEFAULT_PATCH, SPEECH_POSITIONS_PER_SEC};
use crate::kind::ModalKind;

/// Spesifikasi encoder beku: keluarga, revisi, dan parameter yang menentukan
/// anggaran token. Disimpan bersama checkpoint agar eksperimen dapat diulang.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncoderSpec {
    pub kind: ModalKind,
    /// Keluarga model (mis. `siglip`, `whisper`). Verifikasi di F6.1/F6.2.
    pub family: String,
    /// Revisi/tag bobot yang dipakai.
    pub revision: String,
    /// Encoder pemahaman dibekukan (ADR-009 §1).
    pub frozen: bool,
    /// Gambar/video: ukuran patch `p`. Audio: hop dalam sample.
    pub patch: u32,
    /// Audio: faktor stride encoder (frame keluaran per frame masuk).
    pub stride: u32,
    /// Audio: sample rate input.
    pub sample_rate: u32,
}

impl EncoderSpec {
    /// Encoder visi (jalur fitur gambar/video).
    pub fn vision(family: impl Into<String>, revision: impl Into<String>, patch: u32) -> Self {
        Self {
            kind: ModalKind::Image,
            family: family.into(),
            revision: revision.into(),
            frozen: true,
            patch,
            stride: 1,
            sample_rate: 0,
        }
    }

    /// Encoder ucapan (jalur fitur audio).
    pub fn speech(
        family: impl Into<String>,
        revision: impl Into<String>,
        hop: u32,
        stride: u32,
        sample_rate: u32,
    ) -> Self {
        Self {
            kind: ModalKind::Audio,
            family: family.into(),
            revision: revision.into(),
            frozen: true,
            patch: hop,
            stride,
            sample_rate,
        }
    }

    pub fn validate(&self) -> Result<(), VirError> {
        if self.family.trim().is_empty() || self.revision.trim().is_empty() {
            return Err(VirError::Config(
                "family/revision encoder wajib diisi".into(),
            ));
        }
        match self.kind {
            ModalKind::Image => {
                if self.patch == 0 {
                    return Err(VirError::Config("patch encoder visi 0".into()));
                }
                Ok(())
            }
            ModalKind::Audio => {
                if self.patch == 0 {
                    return Err(VirError::Config("hop encoder audio 0".into()));
                }
                if self.sample_rate == 0 {
                    return Err(VirError::Config("sample rate encoder audio 0".into()));
                }
                Ok(())
            }
            ModalKind::Video => Err(VirError::Config(
                "video memakai encoder gambar per frame (ADR-009 §4)".into(),
            )),
            ModalKind::Text => Err(VirError::Config("teks tidak punya encoder".into())),
        }
    }

    /// Posisi keluaran per detik untuk jalur fitur audio.
    pub fn positions_per_sec(&self) -> f32 {
        if self.kind != ModalKind::Audio || self.patch == 0 || self.stride == 0 {
            return SPEECH_POSITIONS_PER_SEC;
        }
        f64::from(self.sample_rate) as f32 / (f64::from(self.patch) * f64::from(self.stride)) as f32
    }

    /// Label singkat untuk log dan manifest.
    pub fn describe(&self) -> String {
        format!(
            "{}/{} patch={} stride={} frozen={}",
            self.family, self.revision, self.patch, self.stride, self.frozen
        )
    }
}

/// Encoder satu modalitas. `Output` ditentukan implementor (tensor/kode).
pub trait ModalEncoder {
    type Output;

    /// Spesifikasi encoder (keluarga, revisi, parameter token).
    fn spec(&self) -> &EncoderSpec;

    /// Anggaran token prefix untuk aset ini, dihitung tanpa menjalankan encoder.
    fn token_budget(&self, asset: &ModalAsset) -> Result<usize, VirError>;

    /// Bangun representasi aset (jalur fitur: vektor konektor; jalur kode: ID).
    fn encode(&self, asset: &ModalAsset) -> Result<Self::Output, VirError>;

    /// Encoder pemahaman beku secara bawaan (ADR-009 §1).
    fn is_frozen(&self) -> bool {
        self.spec().frozen
    }

    fn name(&self) -> String {
        self.spec().describe()
    }
}

/// Bantu implementor: anggaran token bawaan dari spesifikasi + aset.
pub fn budget_of(encoder: &impl ModalEncoder, asset: &ModalAsset) -> Result<usize, VirError> {
    encoder.token_budget(asset)
}

/// Anggaran jalur fitur langsung dari spesifikasi (tanpa implementor).
pub fn spec_budget(spec: &EncoderSpec, asset: &ModalAsset) -> Result<usize, VirError> {
    spec.validate()?;
    budget::prefix_budget(spec, asset)
}

/// Patch bawaan yang dipakai bila spesifikasi tidak menyebut angka.
pub const fn default_patch() -> u32 {
    DEFAULT_PATCH
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyVision(EncoderSpec);

    impl ModalEncoder for DummyVision {
        type Output = usize;

        fn spec(&self) -> &EncoderSpec {
            &self.0
        }

        fn token_budget(&self, asset: &ModalAsset) -> Result<usize, VirError> {
            spec_budget(self.spec(), asset)
        }

        fn encode(&self, asset: &ModalAsset) -> Result<usize, VirError> {
            self.token_budget(asset)
        }
    }

    #[test]
    fn encoder_gambar_beku_dan_anggaran() -> Result<(), VirError> {
        let enc = DummyVision(EncoderSpec::vision(
            "siglip",
            "2026-01",
            default_patch(),
        ));
        assert!(enc.is_frozen());
        assert!(enc.name().contains("siglip/2026-01"));
        let asset = ModalAsset::image("a.png", 384, 384);
        assert_eq!(enc.token_budget(&asset)?, 576);
        assert_eq!(budget_of(&enc, &asset)?, 576);
        assert_eq!(enc.encode(&asset)?, 576);
        Ok(())
    }

    #[test]
    fn encoder_ucapan_posisi_per_detik() -> Result<(), VirError> {
        let spec = EncoderSpec::speech("whisper", "2026-01", 160, 2, 16_000);
        assert_eq!(spec.positions_per_sec(), 50.0);
        assert_eq!(spec_budget(&spec, &ModalAsset::audio("a.wav", 10_000, 16_000, 1))?, 500);
        assert!(EncoderSpec::vision("x", "", 16).validate().is_err());
        Ok(())
    }
}