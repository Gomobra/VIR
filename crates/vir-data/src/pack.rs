//! Pengemasan blok panjang tetap: EOS pemisah + mask padding (PRD F1.3 langkah 3).
//!
//! Dua jalur: `pack_blocks` memotong aliran token datar menjadi blok berukuran
//! sama (dipakai stream dari shard), `pack_documents` mengemas dokumen yang sudah
//! dipisah dengan EOS di antara dokumen dan padding di akhir blok. Mask bernilai
//! `true` hanya untuk token nyata, sehingga loss bisa mengabaikan padding.

use vir_text::SpecialIds;

/// Ukuran blok bawaan.
pub const DEFAULT_BLOCK_SIZE: usize = 512;

/// Konfigurasi pengemasan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackConfig {
    pub block_size: usize,
    pub pad_id: u32,
    pub eos_id: u32,
    pub add_eos: bool,
}

impl Default for PackConfig {
    fn default() -> Self {
        let ids = SpecialIds::new();
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            pad_id: ids.pad,
            eos_id: ids.eos,
            add_eos: true,
        }
    }
}

/// Satu blok token dengan mask padding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedBlock {
    pub tokens: Vec<u32>,
    pub mask: Vec<bool>,
}

impl PackedBlock {
    /// Jumlah token (selalu sama dengan `cfg.block_size` kecuali blok terakhir).
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// `true` bila blok tidak memuat token.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Jumlah token nyata (mask `true`).
    pub fn real_tokens(&self) -> usize {
        self.mask.iter().filter(|keep| **keep).count()
    }

    /// Jumlah token padding.
    pub fn pad_tokens(&self) -> usize {
        self.len() - self.real_tokens()
    }
}

/// Potong aliran token datar menjadi blok berukuran `cfg.block_size`.
///
/// Blok terakhir diisi `pad_id` dengan mask `false`. EOS diperlakukan sebagai
/// token biasa (tanpa asumsi dokumen).
pub fn pack_blocks(tokens: &[u32], cfg: &PackConfig) -> Vec<PackedBlock> {
    let size = cfg.block_size.max(1);
    let mut out = Vec::with_capacity(tokens.len() / size + 1);
    let mut block = Vec::with_capacity(size);
    let mut mask = Vec::with_capacity(size);
    for token in tokens {
        block.push(*token);
        mask.push(true);
        if block.len() == size {
            out.push(take_block(&mut block, &mut mask));
        }
    }
    if !block.is_empty() {
        out.push(pad_block(block, mask, size, cfg.pad_id));
    }
    out
}

/// Kemas dokumen (tiap irisan = satu dokumen) ke blok tetap dengan EOS pemisah.
///
/// Dokumen yang lebih panjang dari satu blok dipecah; blok terakhir di-pad.
pub fn pack_documents<'a, I>(documents: I, cfg: &PackConfig) -> Vec<PackedBlock>
where
    I: IntoIterator<Item = &'a [u32]>,
{
    let size = cfg.block_size.max(1);
    let mut out = Vec::new();
    let mut block = Vec::with_capacity(size);
    let mut mask = Vec::with_capacity(size);
    for document in documents {
        push_document(document, cfg, size, &mut block, &mut mask, &mut out);
        if cfg.add_eos {
            push_token(
                cfg.eos_id, size, cfg.pad_id, &mut block, &mut mask, &mut out,
            );
        }
    }
    if !block.is_empty() {
        out.push(pad_block(block, mask, size, cfg.pad_id));
    }
    out
}

/// Versi lama: potong datar menjadi `Vec<Vec<u32>>` (tanpa mask).
pub fn pack_tokens(tokens: &[u32], block_size: usize) -> Vec<Vec<u32>> {
    let cfg = PackConfig {
        block_size: block_size.max(1),
        ..PackConfig::default()
    };
    pack_blocks(tokens, &cfg)
        .into_iter()
        .map(|block| block.tokens)
        .collect()
}

/// Tulis satu dokumen; pecah bila melebihi satu blok.
fn push_document(
    document: &[u32],
    cfg: &PackConfig,
    size: usize,
    block: &mut Vec<u32>,
    mask: &mut Vec<bool>,
    out: &mut Vec<PackedBlock>,
) {
    if document.is_empty() {
        return;
    }
    for piece in document.chunks(size) {
        if block.len() + piece.len() > size {
            out.push(pad_block(
                std::mem::take(block),
                std::mem::take(mask),
                size,
                cfg.pad_id,
            ));
        }
        block.extend_from_slice(piece);
        mask.extend(std::iter::repeat_n(true, piece.len()));
    }
}

/// Tambah satu token, tutup blok bila sudah penuh.
fn push_token(
    token: u32,
    size: usize,
    pad_id: u32,
    block: &mut Vec<u32>,
    mask: &mut Vec<bool>,
    out: &mut Vec<PackedBlock>,
) {
    if block.len() >= size {
        out.push(pad_block(
            std::mem::take(block),
            std::mem::take(mask),
            size,
            pad_id,
        ));
    }
    block.push(token);
    mask.push(true);
}

fn take_block(block: &mut Vec<u32>, mask: &mut Vec<bool>) -> PackedBlock {
    PackedBlock {
        tokens: std::mem::take(block),
        mask: std::mem::take(mask),
    }
}

fn pad_block(mut block: Vec<u32>, mut mask: Vec<bool>, size: usize, pad_id: u32) -> PackedBlock {
    while block.len() < size {
        block.push(pad_id);
        mask.push(false);
    }
    PackedBlock {
        tokens: block,
        mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blok_tepat_dan_pada() {
        let cfg = PackConfig {
            block_size: 4,
            ..PackConfig::default()
        };
        let blocks = pack_blocks(&[1, 2, 3, 4, 5], &cfg);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].tokens, vec![1, 2, 3, 4]);
        assert_eq!(blocks[0].real_tokens(), 4);
        assert_eq!(blocks[1].tokens.len(), 4);
        assert_eq!(blocks[1].real_tokens(), 1);
        assert_eq!(blocks[1].pad_tokens(), 3);
        assert_eq!(blocks[1].tokens[1], cfg.pad_id);
    }

    #[test]
    fn dokumen_dipisah_eos() {
        let cfg = PackConfig {
            block_size: 8,
            ..PackConfig::default()
        };
        let blocks = pack_documents([&[1, 2][..], &[3, 4][..]], &cfg);
        assert_eq!(blocks.len(), 1);
        let mut expected = vec![1, 2, cfg.eos_id, 3, 4, cfg.eos_id];
        expected.resize(8, cfg.pad_id);
        assert_eq!(blocks[0].tokens, expected);
        assert_eq!(blocks[0].real_tokens(), 6);
        let last = blocks[0].tokens.last().copied();
        assert_eq!(last, Some(cfg.pad_id));
        assert!(blocks.iter().all(|block| block.len() == 8));
    }

    #[test]
    fn dokumen_panjang_dipecah() {
        let cfg = PackConfig {
            block_size: 4,
            ..PackConfig::default()
        };
        let long: Vec<u32> = (0..10).collect();
        let blocks = pack_documents([long.as_slice()], &cfg);
        assert_eq!(blocks.len(), 3);
        assert!(blocks.iter().all(|block| block.len() == 4));
        assert_eq!(blocks[0].real_tokens(), 4);
        assert_eq!(blocks[1].real_tokens(), 4);
        assert_eq!(blocks[2].real_tokens(), 3);
    }

    #[test]
    fn pack_tokens_tanpa_mask() {
        let blocks = pack_tokens(&[1, 2, 3], 2);
        let pad = SpecialIds::new().pad;
        assert_eq!(blocks, vec![vec![1, 2], vec![3, pad]]);
    }
}
