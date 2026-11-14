//! Format data multimodal berpasangan (ADR-009 §6): satu baris JSONL per sampel,
//! `segments` sebagai urutan asli (interleaved), lisensi wajib.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use vir_common::VirError;

use crate::asset::ModalAsset;
use crate::kind::ModalKind;

/// Nilai lisensi yang menolak data dari latihan secara bawaan.
pub const UNKNOWN_LICENSE: &str = "unknown";

/// Peran satu segmen dalam percakapan multimodal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SegmentRole {
    System,
    User,
    Assistant,
    Image,
    Audio,
    Video,
    Time,
}

impl SegmentRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            SegmentRole::System => "system",
            SegmentRole::User => "user",
            SegmentRole::Assistant => "assistant",
            SegmentRole::Image => "image",
            SegmentRole::Audio => "audio",
            SegmentRole::Video => "video",
            SegmentRole::Time => "time",
        }
    }

    /// Modalitas yang wajib disertai aset untuk peran ini. Peran `time` adalah
    /// penanda waktu di dalam segmen video, bukan pemilik aset.
    pub const fn modality(self) -> Option<ModalKind> {
        match self {
            SegmentRole::Image => Some(ModalKind::Image),
            SegmentRole::Audio => Some(ModalKind::Audio),
            SegmentRole::Video => Some(ModalKind::Video),
            SegmentRole::System
            | SegmentRole::User
            | SegmentRole::Assistant
            | SegmentRole::Time => None,
        }
    }

    /// `true` bila segmen ini membawa aset biner.
    pub const fn has_asset(self) -> bool {
        self.modality().is_some()
    }

    /// `true` bila segmen ini adalah penanda waktu (butuh `meta["t"]`).
    pub const fn is_time(self) -> bool {
        matches!(self, SegmentRole::Time)
    }
}

/// Satu bagian urutan: teks, aset, atau penanda waktu.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub role: SegmentRole,
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset: Option<ModalAsset>,
    /// Metadata bebas (mis. `{"t": 12}` untuk peran `time`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub meta: BTreeMap<String, String>,
}

impl Segment {
    pub fn text(role: SegmentRole, text: impl Into<String>) -> Self {
        Self {
            role,
            text: text.into(),
            asset: None,
            meta: BTreeMap::new(),
        }
    }

    pub fn asset(role: SegmentRole, asset: ModalAsset) -> Self {
        Self {
            role,
            text: String::new(),
            asset: Some(asset),
            meta: BTreeMap::new(),
        }
    }

    /// Waktu dalam detik untuk peran `time` (dari `meta["t"]`).
    pub fn time_s(&self) -> Option<f32> {
        self.meta.get("t").and_then(|v| v.parse::<f32>().ok())
    }

    pub fn validate(&self) -> Result<(), VirError> {
        if self.role.is_time() {
            if self.asset.is_some() {
                return Err(VirError::Config(
                    "segmen time tidak boleh menyertakan aset".into(),
                ));
            }
            if self.time_s().is_none() {
                return Err(VirError::Config(
                    "segmen time wajib punya meta[\"t\"] (detik)".into(),
                ));
            }
            return Ok(());
        }
        let want = self.role.modality();
        match (&self.asset, want) {
            (Some(asset), Some(kind)) => {
                if asset.kind != kind {
                    return Err(VirError::Config(format!(
                        "peran {} menuntut aset {}, bukan {}",
                        self.role.as_str(),
                        kind,
                        asset.kind
                    )));
                }
                asset.validate()
            }
            (None, Some(kind)) => Err(VirError::Config(format!(
                "peran {} wajib menyertakan aset {}",
                self.role.as_str(),
                kind
            ))),
            (Some(_), None) => Err(VirError::Config(format!(
                "peran {} tidak boleh menyertakan aset",
                self.role.as_str()
            ))),
            (None, None) => {
                if self.text.trim().is_empty() {
                    return Err(VirError::Config(format!(
                        "segmen {} kosong",
                        self.role.as_str()
                    )));
                }
                Ok(())
            }
        }
    }
}

/// Satu sampel berpasangan (teks + satu atau lebih aset) beserta lisensinya.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedSample {
    pub id: String,
    #[serde(default)]
    pub segments: Vec<Segment>,
    pub license: String,
    #[serde(default)]
    pub source: String,
}

impl PairedSample {
    pub fn new(
        id: impl Into<String>,
        license: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            segments: Vec::new(),
            license: license.into(),
            source: source.into(),
        }
    }

    pub fn push(&mut self, segment: Segment) {
        self.segments.push(segment);
    }

    /// Gabungan teks saja (untuk evaluasi/analitik, bukan untuk melatih urutan).
    pub fn text(&self) -> String {
        let mut out = String::new();
        for s in &self.segments {
            if !s.text.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&s.text);
            }
        }
        out
    }

    pub fn assets(&self) -> impl Iterator<Item = &ModalAsset> {
        self.segments.iter().filter_map(|s| s.asset.as_ref())
    }

    /// Jumlah aset per modalitas (untuk pemeriksaan anggaran konteks).
    pub fn count_kind(&self, kind: ModalKind) -> usize {
        self.assets().filter(|a| a.kind == kind).count()
    }

    /// Validasi penuh: struktur, aset, dan lisensi.
    ///
    /// `allow_unknown_license` hanya untuk riset; loader latih memanggil dengan
    /// `false` sehingga data berlisensi `unknown` ditolak.
    pub fn validate(&self, allow_unknown_license: bool) -> Result<(), VirError> {
        if self.id.trim().is_empty() {
            return Err(VirError::Config("id sampel kosong".into()));
        }
        if self.license.trim().is_empty() {
            return Err(VirError::Config(format!("lisensi kosong: {}", self.id)));
        }
        if !allow_unknown_license && self.license.trim().eq_ignore_ascii_case(UNKNOWN_LICENSE) {
            return Err(VirError::Config(format!(
                "lisensi {UNKNOWN_LICENSE} ditolak: {}",
                self.id
            )));
        }
        if self.segments.is_empty() {
            return Err(VirError::Config(format!(
                "sampel tanpa segmen: {}",
                self.id
            )));
        }
        for s in &self.segments {
            s.validate()?;
        }
        Ok(())
    }

    /// Baris JSONL satu sampel (tanpa newline).
    pub fn to_json_line(&self) -> Result<String, VirError> {
        serde_json::to_string(self).map_err(|e| VirError::Internal(format!("json: {e}")))
    }

    pub fn from_json_line(line: &str) -> Result<Self, VirError> {
        serde_json::from_str(line).map_err(|e| VirError::Internal(format!("json: {e}")))
    }
}

/// Muat berkas JSONL menjadi sampel tervalidasi (satu baris = satu sampel).
pub fn read_jsonl(input: &str, allow_unknown_license: bool) -> Result<Vec<PairedSample>, VirError> {
    let text = std::fs::read_to_string(input)?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let sample = PairedSample::from_json_line(line)
            .map_err(|e| VirError::Internal(format!("baris {}: {e}", i + 1)))?;
        sample.validate(allow_unknown_license)?;
        out.push(sample);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PairedSample {
        let mut s = PairedSample::new("coco-000123", "CC-BY-4.0", "coco");
        s.push(Segment::text(SegmentRole::User, "Jelaskan gambar ini."));
        s.push(Segment::asset(
            SegmentRole::Image,
            ModalAsset::image("img/000123.webp", 384, 384),
        ));
        s.push(Segment::text(
            SegmentRole::Assistant,
            "Seekor kucing di atas kertas.",
        ));
        s
    }

    #[test]
    fn jsonl_roundtrip() -> Result<(), VirError> {
        let s = sample();
        let line = s.to_json_line()?;
        let back = PairedSample::from_json_line(&line)?;
        assert_eq!(back, s);
        assert_eq!(
            back.text(),
            "Jelaskan gambar ini. Seekor kucing di atas kertas."
        );
        assert_eq!(back.count_kind(ModalKind::Image), 1);
        Ok(())
    }

    #[test]
    fn validasi_lisensi_dan_peran() -> Result<(), VirError> {
        let mut s = sample();
        s.validate(false)?;
        s.license = UNKNOWN_LICENSE.into();
        assert!(s.validate(false).is_err());
        s.validate(true)?;

        let mut bad = sample();
        bad.segments[1].role = SegmentRole::Audio;
        assert!(bad.validate(false).is_err());

        let mut missing = sample();
        missing.segments[1].asset = None;
        assert!(missing.validate(false).is_err());

        let mut orphan = PairedSample::new("x", "MIT", "s");
        orphan.push(Segment::text(SegmentRole::User, "teks"));
        orphan.push(Segment::text(SegmentRole::User, "lagi"));
        orphan.push(Segment::asset(
            SegmentRole::User,
            ModalAsset::audio("a.wav", 1_000, 16_000, 1),
        ));
        assert!(orphan.validate(false).is_err());
        Ok(())
    }

    #[test]
    fn waktu_dan_json_kosong() -> Result<(), VirError> {
        let mut t = Segment::text(SegmentRole::Time, "t");
        t.meta.insert("t".into(), "12.5".into());
        assert_eq!(t.time_s(), Some(12.5));
        assert!(t.validate().is_ok());
        assert_eq!(SegmentRole::Time.modality(), None);
        assert!(!SegmentRole::Time.has_asset());
        assert!(SegmentRole::System.modality().is_none());
        assert!(Segment::text(SegmentRole::System, "sys").validate().is_ok());
        Ok(())
    }
}
