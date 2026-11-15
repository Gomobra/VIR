//! Token khusus: PAD, BOS, EOS, MASK, penanda FIM, cadangan modalitas (PRD F1.2).
//!
//! Tata letak id tokenizer byte-level:
//!
//! ```text
//! [0, 256)          satu token per nilai byte
//! [256, 263)        token khusus teks (PAD, BOS, EOS, MASK, FIM)
//! [263, 271)        cadangan modalitas (dipakai vir-modal pada F6.0)
//! [271, vocab)      hasil merge BPE
//! ```
//!
//! `encode` tidak pernah menghasilkan token khusus: teks dipotong per byte dan
//! token khusus hanya ditambahkan oleh pipeline data (`vir-data`), sehingga
//! `encode`/`decode` tetap lossless pada byte.

/// Jumlah token byte dasar.
pub const BYTE_VOCAB: u32 = 256;

/// Jumlah slot cadangan untuk penanda modalitas.
pub const MODAL_RESERVE: u32 = 8;

/// Penanda akhir kiri (padding).
pub const PAD: &str = "<pad>";
/// Penanda awal urutan.
pub const BOS: &str = "<bos>";
/// Penanda akhir urutan.
pub const EOS: &str = "<eos>";
/// Penanda posisi tersembunyi (latihan ter-mask).
pub const MASK: &str = "<mask>";
/// Penanda FIM: bagian awal (prefix).
pub const FIM_PREFIX: &str = "<fim_pre>";
/// Penanda FIM: bagian akhir (suffix).
pub const FIM_SUFFIX: &str = "<fim_suf>";
/// Penanda FIM: bagian tengah (middle).
pub const FIM_MIDDLE: &str = "<fim_mid>";

/// Seluruh token khusus teks, berurutan sesuai id.
pub const SPECIAL_TOKENS: &[&str] = &[PAD, BOS, EOS, MASK, FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE];

/// Id token khusus, diturunkan dari tata letak di atas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialIds {
    pub pad: u32,
    pub bos: u32,
    pub eos: u32,
    pub mask: u32,
    pub fim_prefix: u32,
    pub fim_suffix: u32,
    pub fim_middle: u32,
}

impl SpecialIds {
    /// Tata letak bawaan.
    pub const fn new() -> Self {
        Self {
            pad: BYTE_VOCAB,
            bos: BYTE_VOCAB + 1,
            eos: BYTE_VOCAB + 2,
            mask: BYTE_VOCAB + 3,
            fim_prefix: BYTE_VOCAB + 4,
            fim_suffix: BYTE_VOCAB + 5,
            fim_middle: BYTE_VOCAB + 6,
        }
    }

    /// Panjang blok dasar: byte + token khusus + cadangan modalitas.
    pub const fn base_len(&self) -> u32 {
        BYTE_VOCAB + SPECIAL_TOKENS.len() as u32 + MODAL_RESERVE
    }

    /// Id pertama slot cadangan modalitas.
    pub const fn first_reserved(&self) -> u32 {
        BYTE_VOCAB + SPECIAL_TOKENS.len() as u32
    }

    /// Id sebuah token khusus, atau `None` bila bukan token khusus.
    pub fn id_of(&self, token: &str) -> Option<u32> {
        let index = SPECIAL_TOKENS.iter().position(|t| *t == token)?;
        Some(BYTE_VOCAB + index as u32)
    }

    /// Nama token khusus untuk sebuah id, atau `None` bila bukan token khusus.
    pub fn name_of(&self, id: u32) -> Option<&'static str> {
        if !self.is_special(id) {
            return None;
        }
        SPECIAL_TOKENS.get((id - BYTE_VOCAB) as usize).copied()
    }

    /// `true` bila id adalah token khusus teks.
    pub const fn is_special(&self, id: u32) -> bool {
        id >= BYTE_VOCAB && id < BYTE_VOCAB + SPECIAL_TOKENS.len() as u32
    }

    /// `true` bila id adalah slot cadangan modalitas.
    pub const fn is_reserved(&self, id: u32) -> bool {
        id >= self.first_reserved() && id < self.base_len()
    }

    /// Rentang id cadangan modalitas `[first, base_len)`.
    pub const fn modal_range(&self) -> std::ops::Range<u32> {
        self.first_reserved()..self.base_len()
    }
}

impl Default for SpecialIds {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_khusus_berurutan_dari_256() {
        let ids = SpecialIds::new();
        assert_eq!(ids.pad, 256);
        assert_eq!(ids.bos, 257);
        assert_eq!(ids.eos, 258);
        assert_eq!(ids.mask, 259);
        assert_eq!(ids.fim_prefix, 260);
        assert_eq!(ids.fim_suffix, 261);
        assert_eq!(ids.fim_middle, 262);
        assert_eq!(ids.first_reserved(), 263);
        assert_eq!(ids.base_len(), 271);
    }

    #[test]
    fn nama_dan_id_saling_konsisten() {
        let ids = SpecialIds::new();
        for token in SPECIAL_TOKENS {
            let id = ids.id_of(token).unwrap_or(u32::MAX);
            assert_eq!(ids.name_of(id), Some(*token));
        }
        assert_eq!(ids.name_of(0), None);
        assert_eq!(ids.id_of("tidak-ada"), None);
    }

    #[test]
    fn cadangan_modalitas_tidak_tertukar_dengan_khusus() {
        let ids = SpecialIds::new();
        for id in ids.modal_range() {
            assert!(ids.is_reserved(id));
            assert!(!ids.is_special(id));
        }
        assert!(ids.is_special(ids.eos));
        assert!(!ids.is_reserved(ids.eos));
    }
}
