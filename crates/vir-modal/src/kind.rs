//! Modalitas yang dikenal VIR (PRD F6.0).
//!
//! `ModalKind` adalah kunci utama seluruh crate: ia menentukan jalur
//! representasi (kode vs fitur), penanda vocab, dan rumus anggaran token.
//! Lihat ADR-009 §1–§5.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use vir_common::VirError;

/// Jenis modalitas. `Text` disertakan agar satu ruang ID (ADR-009 §5) bisa
/// menjelaskan seluruh urutan, termasuk teks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModalKind {
    Text,
    Image,
    Audio,
    Video,
}

impl ModalKind {
    /// Urutan alokasi blok vocab: teks, lalu gambar, audio, video.
    pub const ALL: [ModalKind; 4] = [
        ModalKind::Text,
        ModalKind::Image,
        ModalKind::Audio,
        ModalKind::Video,
    ];

    /// Modalitas yang punya penanda `<x>`/`</x>` di vocab.
    pub const MARKED: [ModalKind; 3] = [ModalKind::Image, ModalKind::Audio, ModalKind::Video];

    pub const fn as_str(self) -> &'static str {
        match self {
            ModalKind::Text => "text",
            ModalKind::Image => "image",
            ModalKind::Audio => "audio",
            ModalKind::Video => "video",
        }
    }

    /// `true` bila modalitas dapat **dihasilkan** lewat jalur kode diskret
    /// (ADR-009 §1). Video hanya untuk pemahaman di F6.5.
    pub const fn is_generative(self) -> bool {
        matches!(self, ModalKind::Image | ModalKind::Audio)
    }

    /// `true` bila aset modalitas ini butuh dimensi spasial (H, W) untuk
    /// menghitung anggaran token.
    pub const fn is_spatial(self) -> bool {
        matches!(self, ModalKind::Image | ModalKind::Video)
    }

    /// `true` bila aset modalitas ini punya durasi waktu.
    pub const fn is_temporal(self) -> bool {
        matches!(self, ModalKind::Audio | ModalKind::Video)
    }

    /// Pasangan penanda awal/akhir, atau `None` untuk teks.
    pub const fn markers(self) -> Option<(&'static str, &'static str)> {
        match self {
            ModalKind::Image => Some(("<img>", "</img>")),
            ModalKind::Audio => Some(("<aud>", "</aud>")),
            ModalKind::Video => Some(("<vid>", "</vid>")),
            ModalKind::Text => None,
        }
    }
}

impl fmt::Display for ModalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ModalKind {
    type Err = VirError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "text" => Ok(ModalKind::Text),
            "image" | "img" => Ok(ModalKind::Image),
            "audio" | "aud" | "speech" => Ok(ModalKind::Audio),
            "video" | "vid" => Ok(ModalKind::Video),
            other => Err(VirError::Config(format!(
                "modalitas tidak dikenal: {other}"
            ))),
        }
    }
}
