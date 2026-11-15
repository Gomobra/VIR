//! Tabel merge BPE: vocab, pasangan, dan peringkat (PRD F1.2 langkah 2–3).
//!
//! Id hasil merge selalu `base_len + rank`, sehingga peringkat dan id dapat
//! saling diverifikasi saat impor (`export.rs`). Vocab menyimpan byte mentah tiap
//! token; pasangan byte yang tidak unik tidak menghasilkan tabrakan karena id
//! token khusus berasal dari tata letak tetap, bukan dari pencarian byte.

use std::collections::HashMap;

use vir_common::VirError;

use crate::special::{MODAL_RESERVE, SPECIAL_TOKENS, SpecialIds};

/// Pasangan id yang digabung menjadi satu token.
pub type Pair = (u32, u32);

/// Penanda node kosong (bukan id token yang sah).
pub(crate) const NIL: u32 = u32::MAX;

/// Vocab dan daftar merge hasil pelatihan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeTable {
    vocab: Vec<Vec<u8>>,
    merges: Vec<(Pair, u32)>,
    rank_of: HashMap<Pair, u32>,
    base_len: u32,
}

impl Default for MergeTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MergeTable {
    /// Tabel dasar: 256 byte, token khusus, lalu cadangan modalitas.
    pub fn new() -> Self {
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for token in SPECIAL_TOKENS {
            vocab.push(token.as_bytes().to_vec());
        }
        for _ in 0..MODAL_RESERVE {
            vocab.push(Vec::new());
        }
        let base_len = vocab.len() as u32;
        Self {
            vocab,
            merges: Vec::new(),
            rank_of: HashMap::new(),
            base_len,
        }
    }

    /// Seluruh vocab, id = indeks.
    pub fn vocab(&self) -> &[Vec<u8>] {
        &self.vocab
    }

    /// Ukuran vocab termasuk token khusus.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Jumlah blok dasar (byte + khusus + cadangan).
    pub fn base_len(&self) -> u32 {
        self.base_len
    }

    /// Daftar merge berurutan menurut peringkat.
    pub fn merges(&self) -> &[(Pair, u32)] {
        &self.merges
    }

    /// Jumlah merge yang sudah dipelajari.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// Byte sebuah id, atau `None` bila di luar vocab.
    pub fn token(&self, id: u32) -> Option<&[u8]> {
        self.vocab.get(id as usize).map(|v| v.as_slice())
    }

    /// Peringkat merge sebuah pasangan, atau `None` bila belum digabung.
    pub fn rank(&self, pair: Pair) -> Option<u32> {
        self.rank_of.get(&pair).copied()
    }

    /// Id hasil merge dengan peringkat `rank`.
    pub fn id_for_rank(&self, rank: u32) -> Option<u32> {
        if rank as usize >= self.merges.len() {
            return None;
        }
        Some(self.base_len + rank)
    }

    /// Tambah satu merge; `None` bila pasangan sudah ada atau byte-nya tak lengkap.
    pub fn push_merge(&mut self, pair: Pair) -> Option<u32> {
        if self.rank_of.contains_key(&pair) {
            return None;
        }
        let mut bytes = self.token(pair.0)?.to_vec();
        bytes.extend_from_slice(self.token(pair.1)?);
        let rank = self.merges.len() as u32;
        let id = self.base_len + rank;
        self.vocab.push(bytes);
        self.rank_of.insert(pair, rank);
        self.merges.push((pair, id));
        Some(id)
    }

    /// Bangun tabel dari vocab yang sudah berurutan, tanpa merge.
    ///
    /// Berkas tiktoken pihak lain hanya menyimpan byte tiap token, bukan merge,
    /// sehingga tabel hasil impor bekerja pada tingkat byte (tetap lossless).
    /// Ekspor VIR menulis blok merge, jadi impornya tepat; lihat
    /// [`from_vocab_merges`](Self::from_vocab_merges).
    pub fn from_vocab(vocab: Vec<Vec<u8>>) -> Result<Self, VirError> {
        Self::validated(vocab, Vec::new())
    }

    /// Bangun tabel dari vocab plus daftar merge (impor ekspor VIR).
    ///
    /// Peringkat = posisi merge, dan token hasil merge harus persis byte kedua
    /// token sumbernya; pelanggaran berarti berkas tidak konsisten.
    pub fn from_vocab_merges(vocab: Vec<Vec<u8>>, merges: Vec<Pair>) -> Result<Self, VirError> {
        Self::validated(vocab, merges)
    }

    /// Validasi tata letak blok dasar lalu terapkan merge bila ada.
    fn validated(vocab: Vec<Vec<u8>>, merges: Vec<Pair>) -> Result<Self, VirError> {
        let specials = SpecialIds::new();
        let base_len = specials.base_len() as usize;
        if vocab.len() < base_len {
            let got = vocab.len();
            return Err(VirError::Internal(format!(
                "vocab tiktoken terlalu pendek: {got} < {base_len}"
            )));
        }
        for (i, byte) in vocab
            .iter()
            .enumerate()
            .take(crate::special::BYTE_VOCAB as usize)
        {
            if byte.len() != 1 || byte[0] != i as u8 {
                return Err(VirError::Internal(format!("token byte ke-{i} tidak valid")));
            }
        }
        for (k, token) in SPECIAL_TOKENS.iter().enumerate() {
            let got = &vocab[crate::special::BYTE_VOCAB as usize + k];
            if got.as_slice() != token.as_bytes() {
                return Err(VirError::Internal(format!(
                    "token khusus ke-{k} bukan {token}"
                )));
            }
        }
        let first = specials.first_reserved() as usize;
        for (id, slot) in vocab.iter().enumerate().skip(first).take(base_len - first) {
            if !slot.is_empty() {
                return Err(VirError::Internal(format!(
                    "slot cadangan {id} tidak kosong"
                )));
            }
        }
        let mut table = Self {
            vocab,
            merges: Vec::new(),
            rank_of: HashMap::new(),
            base_len: specials.base_len(),
        };
        table.apply_merges(merges)?;
        Ok(table)
    }

    /// Terapkan merge berurutan; peringkat `r` harus menghasilkan id `base_len + r`.
    ///
    /// Vocab impor sudah memuat token hasil merge, jadi ia dipisah dulu: tiap
    /// merge dibangun ulang lewat [`push_merge`](Self::push_merge) lalu byte-nya
    /// dibandingkan dengan token impor pada id yang sama. Sisa token impor tanpa
    /// merge (berkas tiktoken pihak lain) dipertahankan apa adanya.
    fn apply_merges(&mut self, merges: Vec<Pair>) -> Result<(), VirError> {
        let imported = self.vocab.split_off(self.base_len as usize);
        let mut rest = imported.into_iter();
        for (rank, pair) in merges.into_iter().enumerate() {
            let expected = self.base_len + rank as u32;
            let Some(existing) = rest.next() else {
                return Err(VirError::Internal(format!(
                    "vocab tanpa token merge peringkat {rank}"
                )));
            };
            if pair.0.max(pair.1) >= expected {
                return Err(VirError::Internal(format!(
                    "merge peringkat {rank} memakai id di luar vocab"
                )));
            }
            let Some(id) = self.push_merge(pair) else {
                return Err(VirError::Internal(format!(
                    "merge peringkat {rank} ganda atau tidak sah"
                )));
            };
            if id != expected || self.token(id) != Some(existing.as_slice()) {
                return Err(VirError::Internal(format!(
                    "merge peringkat {rank} tidak cocok dengan vocab"
                )));
            }
        }
        self.vocab.extend(rest);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabel_dasar_berukuran_271() {
        let table = MergeTable::new();
        assert_eq!(table.base_len(), 271);
        assert_eq!(table.vocab_size(), 271);
        assert_eq!(table.token(65), Some(b"A".as_slice()));
        assert_eq!(table.token(256), Some(b"<pad>".as_slice()));
        assert_eq!(table.token(270), Some([].as_slice()));
        assert_eq!(table.token(271), None);
    }

    #[test]
    fn merge_baru_mendapat_id_berurutan() {
        let mut table = MergeTable::new();
        assert_eq!(table.push_merge((97, 98)), Some(271));
        assert_eq!(table.rank((97, 98)), Some(0));
        assert_eq!(table.token(271), Some(b"ab".as_slice()));
        assert_eq!(table.push_merge((97, 98)), None);
        assert_eq!(table.push_merge((271, 99)), Some(272));
        assert_eq!(table.token(272), Some(b"abc".as_slice()));
    }
}
