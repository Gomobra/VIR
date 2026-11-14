//! Ruang ID global, token kode diskret, dan perluasan vocab (ADR-009 §5).
//!
//! Satu ruang ID berisi empat blok berurutan:
//!
//! ```text
//! [ 0 .. text_vocab )            teks (BPE + EOS + FIM)
//! [ text_vocab .. marker_end )   penanda <img>/</img>, <aud>/</aud>, <vid>/</vid>, <t=..>
//! [ marker_end .. total )        codebook per (kind, level), disejajarkan 64
//! ```
//!
//! Penanda adalah token teks biasa; kode modalitas memakai baris embedding
//! tersendiri per level, sehingga tiap `(kind, level)` harus punya rentang ID
//! sendiri (ADR-009 §5, alternatif 3 ditolak).

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use vir_common::VirError;

use crate::kind::ModalKind;

/// Ukuran codebook bawaan (12 bit, sama dengan SNAC 24 kHz di ADR-009 §3).
pub const DEFAULT_CODEBOOK: u32 = 4096;

/// Perataan awal rentang codebook agar hemat memori kernel.
pub const SPAN_ALIGN: u32 = 64;

/// Jumlah penanda waktu `<t=0>`..`<t=63>` untuk video (ADR-009 §4).
pub const TIME_MARKERS: u32 = 64;

/// Rentang ID setengah terbuka `[start, start + len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Span {
    pub start: u32,
    pub len: u32,
}

impl Span {
    pub const fn new(start: u32, len: u32) -> Self {
        Self { start, len }
    }

    pub const fn end(&self) -> u32 {
        self.start + self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn contains(&self, id: u32) -> bool {
        id >= self.start && id - self.start < self.len
    }

    pub fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end() && other.start < self.end()
    }

    /// Geser rentang sejauh `delta` (dipakai saat teks di-resize).
    pub const fn shift(&self, delta: u32) -> Span {
        Span::new(self.start + delta, self.len)
    }
}

/// Kunci satu codebook: modalitas + level (0 = paling kasar).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct CodebookKey {
    pub kind: ModalKind,
    pub level: u8,
}

impl CodebookKey {
    pub const fn new(kind: ModalKind, level: u8) -> Self {
        Self { kind, level }
    }

    pub const fn token(&self, code: u32) -> ModalToken {
        ModalToken::new(self.kind, self.level, code)
    }

    pub const fn is_text(&self) -> bool {
        matches!(self.kind, ModalKind::Text)
    }
}

impl fmt::Display for CodebookKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.kind, self.level)
    }
}

/// Satu kode diskret milik modalitas tertentu pada level tertentu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModalToken {
    pub kind: ModalKind,
    pub level: u8,
    pub code: u32,
}

impl ModalToken {
    pub const fn new(kind: ModalKind, level: u8, code: u32) -> Self {
        Self { kind, level, code }
    }

    /// Token teks (level 0, kode = ID BPE).
    pub const fn text(code: u32) -> Self {
        Self::new(ModalKind::Text, 0, code)
    }

    pub const fn key(&self) -> CodebookKey {
        CodebookKey::new(self.kind, self.level)
    }

    pub const fn is_text(&self) -> bool {
        matches!(self.kind, ModalKind::Text)
    }
}

/// Blok tempat sebuah ID berada.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Text,
    Marker(ModalKind),
    Codebook(CodebookKey),
}

/// Penanda modalitas: `open`, `close`, dan (khusus video) blok penanda waktu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Markers {
    pub open: u32,
    pub close: u32,
    /// ID `<t=0>`; `None` bila modalitas tidak punya penanda waktu.
    pub time_base: Option<u32>,
}

impl Markers {
    /// Rentang seluruh penanda modalitas ini (open, close, lalu waktu).
    pub const fn span(&self) -> Span {
        let len = match self.time_base {
            Some(_) => 2 + TIME_MARKERS,
            None => 2,
        };
        Span::new(self.open, len)
    }

    pub const fn time(&self, t: u32) -> Option<u32> {
        match self.time_base {
            Some(base) if t < TIME_MARKERS => Some(base + t),
            _ => None,
        }
    }
}

/// Tata letak vocab: satu ruang ID + peta codebook per `(kind, level)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VocabLayout {
    text_vocab: u32,
    markers: BTreeMap<ModalKind, Markers>,
    codebooks: BTreeMap<CodebookKey, Span>,
}

impl VocabLayout {
    /// Alokasi blok teks + penanda, belum ada codebook.
    pub fn new(text_vocab: u32) -> Result<Self, VirError> {
        if text_vocab == 0 {
            return Err(VirError::Config("text_vocab harus > 0".into()));
        }
        let mut markers = BTreeMap::new();
        let mut next = text_vocab;
        for kind in ModalKind::MARKED {
            let time_base = if kind == ModalKind::Video {
                Some(next + 2)
            } else {
                None
            };
            let m = Markers {
                open: next,
                close: next + 1,
                time_base,
            };
            next = m.span().end();
            markers.insert(kind, m);
        }
        Ok(Self {
            text_vocab,
            markers,
            codebooks: BTreeMap::new(),
        })
    }

    /// Tambah codebook berukuran `size` (dibulatkan ke `SPAN_ALIGN`).
    /// Codebook diletakkan setelah blok terakhir yang sudah ada.
    pub fn add_codebook(&mut self, key: CodebookKey, size: u32) -> Result<(), VirError> {
        if size == 0 {
            return Err(VirError::Config("ukuran codebook harus > 0".into()));
        }
        if key.is_text() {
            return Err(VirError::Config(
                "teks tidak punya codebook (pakai blok vocab teks)".into(),
            ));
        }
        if self.codebooks.contains_key(&key) {
            return Err(VirError::Config(format!("codebook {key} sudah ada")));
        }
        let start = self.next_free();
        let span = Span::new(start, align_up(size));
        self.codebooks.insert(key, span);
        Ok(())
    }

    /// ID berikutnya yang bebas (akhir blok tertinggi).
    pub fn next_free(&self) -> u32 {
        let mut next = self.text_vocab;
        for m in self.markers.values() {
            next = next.max(m.span().end());
        }
        for span in self.codebooks.values() {
            next = next.max(span.end());
        }
        next
    }

    pub const fn text_vocab(&self) -> u32 {
        self.text_vocab
    }

    /// Panjang blok penanda (0 bila belum dialokasikan — selalu > 0 setelah `new`).
    pub fn marker_len(&self) -> u32 {
        self.markers
            .values()
            .fold(0, |acc, m| acc.max(m.span().end()))
            .saturating_sub(self.text_vocab)
    }

    /// Total ID yang dialokasikan.
    pub fn total(&self) -> u32 {
        self.next_free()
    }

    pub fn marker(&self, kind: ModalKind) -> Option<&Markers> {
        self.markers.get(&kind)
    }

    pub fn codebook_span(&self, key: CodebookKey) -> Option<Span> {
        self.codebooks.get(&key).copied()
    }

    pub fn codebook_keys(&self) -> impl Iterator<Item = CodebookKey> + '_ {
        self.codebooks.keys().copied()
    }

    /// ID global untuk sebuah kode, atau `None` bila di luar codebook.
    pub fn encode(&self, key: CodebookKey, code: u32) -> Option<u32> {
        let span = self.codebooks.get(&key)?;
        if code >= span.len {
            return None;
        }
        Some(span.start + code)
    }

    /// Kebalikan `encode`: ID global → kode modalitas.
    pub fn decode(&self, id: u32) -> Option<ModalToken> {
        for (key, span) in &self.codebooks {
            if span.contains(id) {
                return Some(ModalToken::new(key.kind, key.level, id - span.start));
            }
        }
        None
    }

    /// Blok tempat sebuah ID berada; `None` bila melewati total vocab.
    pub fn region(&self, id: u32) -> Option<Region> {
        if id < self.text_vocab {
            return Some(Region::Text);
        }
        for (kind, m) in &self.markers {
            if m.span().contains(id) {
                return Some(Region::Marker(*kind));
            }
        }
        for (key, span) in &self.codebooks {
            if span.contains(id) {
                return Some(Region::Codebook(*key));
            }
        }
        None
    }

    /// Migrasi checkpoint saat BPE dilatih ulang (ADR-009 §5): teks boleh
    /// bertambah, semua blok bergeser `delta` dan isinya dipertahankan.
    pub fn remap_after_text_resize(&self, new_text_vocab: u32) -> Result<Self, VirError> {
        if new_text_vocab < self.text_vocab {
            return Err(VirError::Config(format!(
                "pengecilan vocab teks tidak didukung: {} -> {new_text_vocab}",
                self.text_vocab
            )));
        }
        let delta = new_text_vocab - self.text_vocab;
        let mut markers = BTreeMap::new();
        for (kind, m) in &self.markers {
            let time_base = m.time_base.map(|b| b + delta);
            markers.insert(
                *kind,
                Markers {
                    open: m.open + delta,
                    close: m.close + delta,
                    time_base,
                },
            );
        }
        let codebooks = self
            .codebooks
            .iter()
            .map(|(k, s)| (*k, s.shift(delta)))
            .collect();
        Ok(Self {
            text_vocab: new_text_vocab,
            markers,
            codebooks,
        })
    }
}

/// Bangun tata letak standar F6: teks + penanda + codebook gambar/audio.
///
/// `image_levels`/`audio_levels` adalah jumlah level (mis. SNAC = 3).
pub fn build_layout(
    text_vocab: u32,
    image_levels: u8,
    audio_levels: u8,
    codebook: u32,
) -> Result<VocabLayout, VirError> {
    let mut layout = VocabLayout::new(text_vocab)?;
    for level in 0..image_levels {
        layout.add_codebook(CodebookKey::new(ModalKind::Image, level), codebook)?;
    }
    for level in 0..audio_levels {
        layout.add_codebook(CodebookKey::new(ModalKind::Audio, level), codebook)?;
    }
    Ok(layout)
}

const fn align_up(v: u32) -> u32 {
    let a = SPAN_ALIGN as u64;
    ((v as u64).div_ceil(a) * a) as u32
}

