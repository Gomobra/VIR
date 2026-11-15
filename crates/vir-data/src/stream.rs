//! Urutan deterministik + state resume (PRD F1.3 langkah 2).
//!
//! Urutan blok ditentukan seed: setiap epoch mengocok indeks blok dengan
//! Fisher–Yates memakai splitmix64, lalu blok dibaca berurutan. State resume
//! menyimpan seed, state RNG awal epoch, indeks posisi, dan jumlah blok yang sudah
//! diberikan, sehingga melanjutkan setelah mati listrik menghasilkan urutan yang
//! sama persis dengan tanpa interupsi.

use std::path::Path;

use vir_common::VirError;

use crate::pack::{PackConfig, pack_blocks};
use crate::shard::ShardReader;

/// PRNG splitmix64: state 64 bit, cukup untuk resume dan deterministik lintas
/// platform (tanpa dependensi `rand`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// PRNG dengan state awal `state`.
    pub const fn new(state: u64) -> Self {
        Self { state }
    }

    /// Nilai acak berikutnya.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Bilangan `[0, n)`; `0` bila `n <= 1`.
    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let x = self.next_u64();
        ((u128::from(x) * u128::from(n as u64)) >> 64) as usize
    }

    /// Bilangan `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        f64::from_bits((self.next_u64() >> 11) | 0x3FF0_0000_0000_0000) - 1.0
    }
}

/// State stream untuk resume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamState {
    /// Seed awal stream.
    pub seed: u64,
    /// State RNG pada awal epoch berjalan (agar urutan bisa diulang).
    pub epoch_rng: u64,
    /// Nomor epoch.
    pub epoch: u32,
    /// Posisi di dalam urutan epoch.
    pub index: u32,
    /// Jumlah blok yang sudah diberikan.
    pub produced: u64,
}

/// Konfigurasi stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamConfig {
    pub seed: u64,
    pub block_size: usize,
    pub shuffle: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            block_size: crate::pack::DEFAULT_BLOCK_SIZE,
            shuffle: true,
        }
    }
}

/// Iterator blok token deterministik.
#[derive(Debug, Clone)]
pub struct TokenStream {
    blocks: Vec<Vec<u32>>,
    order: Vec<u32>,
    cursor: usize,
    epoch: u32,
    epoch_rng: u64,
    rng: SplitMix64,
    produced: u64,
    cfg: StreamConfig,
}

impl TokenStream {
    /// Stream dari blok yang sudah dipotong (urutan bisa diacak).
    pub fn from_blocks(blocks: Vec<Vec<u32>>, cfg: StreamConfig) -> Self {
        let mut stream = Self {
            blocks,
            order: Vec::new(),
            cursor: 0,
            epoch: 0,
            epoch_rng: cfg.seed,
            rng: SplitMix64::new(cfg.seed),
            produced: 0,
            cfg,
        };
        stream.refill_order();
        stream
    }

    /// Stream dari token datar: dipotong ke blok `cfg.block_size`.
    pub fn from_tokens(tokens: Vec<u32>, cfg: StreamConfig) -> Self {
        let pack = PackConfig {
            block_size: cfg.block_size,
            ..PackConfig::default()
        };
        let blocks = pack_blocks(&tokens, &pack)
            .into_iter()
            .map(|block| block.tokens)
            .collect();
        Self::from_blocks(blocks, cfg)
    }

    /// Stream dari shard biner (blok shard dip-kemas ulang ke `cfg.block_size`).
    pub fn from_shard(path: impl AsRef<Path>, cfg: StreamConfig) -> Result<Self, VirError> {
        let mut reader = ShardReader::open(path)?;
        let tokens = reader.tokens()?;
        Ok(Self::from_tokens(tokens, cfg))
    }

    /// Blok berikutnya; `None` bila stream kosong.
    pub fn next_block(&mut self) -> Option<Vec<u32>> {
        let block = self.next_block_ref()?;
        Some(block.to_vec())
    }

    /// Blok berikutnya sebagai irisan.
    pub fn next_block_ref(&mut self) -> Option<&[u32]> {
        if self.blocks.is_empty() {
            return None;
        }
        if self.cursor >= self.order.len() {
            self.epoch += 1;
            self.refill_order();
        }
        let index = match self.order.get(self.cursor) {
            Some(index) => *index as usize,
            None => return None,
        };
        self.cursor += 1;
        self.produced += 1;
        self.blocks.get(index).map(|block| block.as_slice())
    }

    /// State saat ini untuk disimpan bersama checkpoint.
    pub fn state(&self) -> StreamState {
        StreamState {
            seed: self.cfg.seed,
            epoch_rng: self.epoch_rng,
            epoch: self.epoch,
            index: self.cursor as u32,
            produced: self.produced,
        }
    }

    /// Kocok ulang urutan epoch dengan RNG berjalan, lalu catat state epoch.
    fn refill_order(&mut self) {
        self.epoch_rng = self.rng.state;
        let mut order: Vec<u32> = (0..self.blocks.len() as u32).collect();
        if self.cfg.shuffle {
            for i in (1..order.len()).rev() {
                let j = self.rng.below(i + 1);
                order.swap(i, j);
            }
        }
        self.order = order;
        self.cursor = 0;
    }

    /// Pulihkan stream dari state; `false` bila state berasal dari stream lain.
    pub fn seek(&mut self, state: StreamState) -> bool {
        if state.seed != self.cfg.seed || state.index as usize > self.order.len() {
            return false;
        }
        self.rng = SplitMix64::new(state.epoch_rng);
        self.refill_order();
        self.cursor = state.index as usize;
        self.produced = state.produced;
        self.epoch = state.epoch;
        true
    }

    /// Jumlah blok yang sudah diberikan.
    pub fn produced(&self) -> u64 {
        self.produced
    }

    /// Jumlah blok unik dalam satu epoch.
    pub fn epoch_len(&self) -> usize {
        self.blocks.len()
    }
}

/// Versi ringkas dengan tanda tangan lama `vir-data`: token datar + seed.
impl TokenStream {
    /// Token datar, seed, dan ukuran blok; urutan diacak dengan seed.
    pub fn new(data: Vec<u32>, seed: u64, block_size: usize) -> Self {
        let cfg = StreamConfig {
            seed,
            block_size,
            shuffle: true,
        };
        Self::from_tokens(data, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks() -> Vec<Vec<u32>> {
        (0..8u32)
            .map(|i| (0..8u32).map(|j| i * 100 + j).collect())
            .collect()
    }

    fn take(stream: &mut TokenStream, n: usize) -> Vec<Vec<u32>> {
        (0..n).filter_map(|_| stream.next_block()).collect()
    }

    #[test]
    fn urutan_deterministik() {
        let cfg = StreamConfig {
            block_size: 8,
            ..StreamConfig::default()
        };
        let first = take(&mut TokenStream::from_blocks(blocks(), cfg), 20);
        let second = take(&mut TokenStream::from_blocks(blocks(), cfg), 20);
        assert_eq!(first, second);
        assert_eq!(first.len(), 20);
    }

    #[test]
    fn resume_melanjutkan_urutan_yang_sama() {
        let cfg = StreamConfig {
            block_size: 8,
            ..StreamConfig::default()
        };
        let mut stream = TokenStream::from_blocks(blocks(), cfg);
        let head = take(&mut stream, 5);
        let state = stream.state();
        let mut resumed = TokenStream::from_blocks(blocks(), cfg);
        assert!(resumed.seek(state));
        assert_eq!(resumed.produced(), 5);
        let tail = take(&mut resumed, 15);
        let all = take(&mut TokenStream::from_blocks(blocks(), cfg), 20);
        assert_eq!(head[..], all[..5]);
        assert_eq!(tail[..], all[5..]);
    }

    #[test]
    fn state_lintas_stream_lain_ditolak() {
        let mut stream = TokenStream::from_blocks(blocks(), StreamConfig::default());
        let _ = stream.next_block();
        let mut other = TokenStream::from_blocks(
            blocks(),
            StreamConfig {
                seed: 7,
                ..StreamConfig::default()
            },
        );
        assert!(!other.seek(stream.state()));
    }

    #[test]
    fn dari_token_dan_shard() {
        let tokens: Vec<u32> = (0..100).collect();
        let stream = TokenStream::from_tokens(
            tokens.clone(),
            StreamConfig {
                block_size: 16,
                ..StreamConfig::default()
            },
        );
        assert_eq!(stream.epoch_len(), 7);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("vir-data-stream-{pid}.shard"));
        crate::shard::write_shard(&path, &tokens).expect("tulis shard");
        let from_shard = TokenStream::from_shard(
            &path,
            StreamConfig {
                block_size: 16,
                ..StreamConfig::default()
            },
        )
        .expect("baca shard");
        assert_eq!(from_shard.epoch_len(), 7);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn konstruktor_ringkas_dan_rng() {
        let mut stream = TokenStream::new((0..40).collect(), 3, 8);
        let first = stream.next_block().unwrap_or_default();
        assert_eq!(first.len(), 8);
        assert_eq!(stream.produced(), 1);
        let mut rng = SplitMix64::new(1);
        assert_eq!(rng.below(1), 0);
        let value = rng.next_f64();
        assert!(value >= 0.0 && value < 1.0);
        assert_eq!(SplitMix64::new(5).next_u64(), SplitMix64::new(5).next_u64());
    }
}
