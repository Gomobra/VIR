//! Tabel merge BPE: vocab, pasangan, dan peringkat (PRD F1.2 langkah 2–3).
//!
//! Id hasil merge selalu `base_len + rank`, sehingga peringkat dan id dapat
//! saling diverifikasi saat impor (`export.rs`). Vocab menyimpan byte mentah tiap
//! token; pasangan byte yang tidak unik tidak menghasilkan tabrakan karena id
//! token khusus berasal dari tata letak tetap, bukan dari pencarian byte.

use std::collections::HashMap;

use vir_common::VirError;

use crate::encoder::{Scratch, encode_chunk};
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

    /// Bangun tabel dari vocab lalu pulihkan merge dari peringkat.
    ///
    /// Pemulihan menjalankan encoder pada byte tiap token: merge terakhir yang
    /// dihasilkan adalah merge yang membentuk token itu, sehingga
    /// `from_vocab(export(t)) == t`.
    pub fn from_vocab(vocab: Vec<Vec<u8>>) -> Result<Self, VirError> {
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
        table.recover_merges()?;
        Ok(table)
    }

    /// Pulihkan merge satu per satu dari peringkat vocab.
    fn recover_merges(&mut self) -> Result<(), VirError> {
        let mut scratch = Scratch::new();
        let mut ids: Vec<u32> = Vec::new();
        let mut trace: Vec<Pair> = Vec::new();
        let merge_count = self.vocab.len() as u32 - self.base_len;
        for rank in 0..merge_count {
            let id = self.base_len + rank;
            let Some(bytes) = self.vocab.get(id as usize).cloned() else {
                return Err(VirError::Internal(format!("vocab tanpa id {id}")));
            };
            ids.clear();
            trace.clear();
            encode_chunk(self, &bytes, &mut scratch, &mut ids, &mut trace);
            let Some(pair) = trace.last().copied() else {
                return Err(VirError::Internal(format!(
                    "merge token {id} tidak dapat dipulihkan"
                )));
            };
            let Some(new_id) = self.push_merge(pair) else {
                return Err(VirError::Internal(format!("merge ganda pada token {id}")));
            };
            if new_id != id || self.token(id) != Some(bytes.as_slice()) {
                return Err(VirError::Internal(format!(
                    "merge token {id} tidak cocok dengan vocab"
                )));
            }
        }
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
