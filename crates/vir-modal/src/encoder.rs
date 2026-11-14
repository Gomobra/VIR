//! Trait encoder modalitas (ADR-009 §1 jalur fitur, §7 pemisahan tensor),
//! spesifikasi encoder beku, dan prefix bersama (F6.1/F6.2).
//!
//! Crate ini **tidak** memuat tensor: encoder beku menghasilkan satu vektor
//! fitur per posisi, lalu [`crate::connector::Connector`] yang terlatih
//! memetakannya menjadi prefix. Yang dijamin modul ini: spesifikasi encoder
//! dapat direproduksi, anggaran token bisa dihitung **sebelum** encoder
//! dijalankan (PRD F6.1 langkah 2), dan encoder pemahaman **beku** secara
//! bawaan (ADR-009 §1).
//!
//! Encoder gambar ([`crate::vision::ImageEncoder`]) dan audio
//! ([`crate::speech::AudioEncoder`]) mengimplementasikan trait ini; keduanya
//! menghasilkan [`Prefix`] yang sama bentuknya.

use vir_common::VirError;

use crate::asset::ModalAsset;
use crate::budget::{self, DEFAULT_PATCH, SPEECH_POSITIONS_PER_SEC};

use crate::kind::ModalKind;

/// Spesifikasi encoder beku: keluarga, revisi, dan parameter yang menentukan
/// anggaran token. Disimpan bersama checkpoint agar eksperimen dapat diulang.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

    /// Encoder visi SigLIP-style 224/16 (196 token) — kandidat F6.1.
    pub fn siglip(patch: u32) -> Self {
        Self::vision("siglip", "webli-en-patch16", patch)
    }

    /// Encoder ucapan Whisper-style 16 kHz, hop 160, stride 2 (50 posisi/s).
    pub fn whisper() -> Self {
        Self::speech("whisper", "large-v3-turbo", 160, 2, 16_000)
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

    /// Jumlah frame keluaran untuk `duration_ms` audio pada `sample_rate` aset.
    pub fn frames(&self, duration_ms: u32, sample_rate: u32) -> Result<usize, VirError> {
        if self.kind != ModalKind::Audio {
            return Err(VirError::Config("frames hanya untuk encoder audio".into()));
        }
        if duration_ms == 0 || sample_rate == 0 {
            return Err(VirError::Config("durasi/sample rate audio 0".into()));
        }
        if self.patch == 0 || self.stride == 0 {
            return Err(VirError::Config("hop/stride encoder audio 0".into()));
        }
        let samples = u64::from(duration_ms) * u64::from(sample_rate) / 1_000;
        let frames = samples / u64::from(self.patch);
        Ok(frames.div_ceil(u64::from(self.stride)) as usize)
    }

    /// Label singkat untuk log dan manifest.
    pub fn describe(&self) -> String {
        format!(
            "{}/{} patch={} stride={} frozen={}",
            self.family, self.revision, self.patch, self.stride, self.frozen
        )
    }
}

/// Prefix hasil konektor: `tokens` baris berdimensi `d_model` (row-major).
///
/// Ini keluaran bersama jalur fitur gambar ([`crate::vision::ImageEncoder`])
/// dan audio ([`crate::speech::AudioEncoder`]): ADR-009 §1 menjanjikan satu
/// bentuk prefix untuk semua encoder beku.
#[derive(Debug, Clone, PartialEq)]
pub struct Prefix {
    pub rows: Vec<f32>,
    pub tokens: usize,
    pub d_model: usize,
}

impl Prefix {
    pub fn new(rows: Vec<f32>, tokens: usize, d_model: usize) -> Self {
        Self {
            rows,
            tokens,
            d_model,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    pub fn len(&self) -> usize {
        self.tokens
    }

    /// Baris ke-`i`, atau `None` bila di luar jangkauan.
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        if i >= self.tokens {
            return None;
        }
        Some(&self.rows[i * self.d_model..(i + 1) * self.d_model])
    }

    /// Total elemen (untuk alokasi buffer di sisi tensor).
    pub fn elements(&self) -> usize {
        self.tokens * self.d_model
    }
}

/// Encoder satu modalitas. `Output` ditentukan implementor.
pub trait ModalEncoder {
    type Output;

    /// Spesifikasi encoder (keluarga, revisi, parameter token).
    fn spec(&self) -> &EncoderSpec;

    /// Anggaran token prefix untuk aset ini, dihitung tanpa menjalankan encoder.
    fn token_budget(&self, asset: &ModalAsset) -> Result<usize, VirError>;

    /// Panjang data mentah yang dibutuhkan aset (piksel `H·W·c` atau PCM mono).
    fn raw_len(&self, asset: &ModalAsset) -> Result<usize, VirError>;

    /// Bangun prefix dari data mentah aset (jalur fitur).
    fn encode(&self, asset: &ModalAsset, data: &[f32]) -> Result<Self::Output, VirError>;

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
