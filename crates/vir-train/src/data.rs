//! Sumber batch token dari shard (PRD F2.5.a, F1.7 langkah "DataLoader mini").
//!
//! `BatchSource` menghasilkan satu batch `[B, L]` token per panggilan. Urutan
//! berasal dari [`TokenStream`] vir-data: blok dipotong deterministik menurut
//! `seed`, diacak per epoch bila `shuffle`, dan state epoch/posisi bisa
//! disimpan (`state`/`seek`) sehingga resume menghasilkan batch yang sama.
//!
//! Batas tugas:
//!
//! ```text
//! shard (vir-data) → TokenStream (seed, resume) → batch [B, L] (Vec<i64>)
//! ```
//!
//! Blok dibaca lewat `TokenStream::from_shard` (seluruh token shard di RAM,
//! blok zstd didekompresi sekali). Aliran per blok dari disk (tanpa menampung
//! seluruh token) adalah F2.5.d; di sini batch wajib deterministik dan
//! resume-safe, dan keduanya sudah dipenuhi jalur ini.

use std::path::PathBuf;

use vir_common::VirError;
use vir_data::{StreamConfig, StreamState, TokenStream};

/// Sumber batch token deterministik dan resume-safe.
pub trait BatchSource {
    /// Batch token berikutnya: nilai baris-major `[B, L]` dan bentuk `[B, L]`.
    fn next_batch(&mut self) -> Result<(Vec<i64>, [usize; 2]), VirError>;
}

/// Asal blok token: shard di disk atau token di memori (dipakai tes).
#[derive(Debug, Clone)]
enum Origin {
    Shard(PathBuf),
    Tokens(Vec<u32>),
}

/// Batch token dari shard/token dengan seed dan posisi yang bisa disimpan.
#[derive(Debug, Clone)]
pub struct ShardBatchSource {
    origin: Origin,
    cfg: StreamConfig,
    stream: TokenStream,
    batch: usize,
    len: usize,
    produced: u64,
}

impl ShardBatchSource {
    /// Buka shard; token dipotong ulang menjadi blok sepanjang `len`.
    pub fn from_shard(
        path: impl Into<PathBuf>,
        batch: usize,
        len: usize,
        seed: u64,
        shuffle: bool,
    ) -> Result<Self, VirError> {
        let path = path.into();
        let len = len.max(1);
        let cfg = StreamConfig {
            seed,
            block_size: len,
            shuffle,
        };
        let stream = TokenStream::from_shard(&path, cfg)?;
        Ok(Self::origin(Origin::Shard(path), cfg, stream, batch, len))
    }

    /// Sumber dari token di memori (jalur yang sama, tanpa berkas).
    pub fn from_tokens(
        tokens: Vec<u32>,
        batch: usize,
        len: usize,
        seed: u64,
        shuffle: bool,
    ) -> Result<Self, VirError> {
        let len = len.max(1);
        let cfg = StreamConfig {
            seed,
            block_size: len,
            shuffle,
        };
        if tokens.is_empty() {
            return Err(VirError::Config("token sumber kosong".into()));
        }
        let stream = TokenStream::from_tokens(tokens.clone(), cfg);
        Ok(Self::origin(
            Origin::Tokens(tokens),
            cfg,
            stream,
            batch,
            len,
        ))
    }

    /// Rakit dari asal + stream yang sudah dibangun.
    fn origin(
        origin: Origin,
        cfg: StreamConfig,
        stream: TokenStream,
        batch: usize,
        len: usize,
    ) -> Self {
        Self {
            origin,
            cfg,
            stream,
            batch: batch.max(1),
            len,
            produced: 0,
        }
    }

    /// Ukuran batch `B`.
    pub fn batch_size(&self) -> usize {
        self.batch
    }

    /// Panjang urutan `L`.
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// Jumlah blok unik per epoch.
    pub fn epoch_len(&self) -> usize {
        self.stream.epoch_len()
    }

    /// Jumlah batch yang sudah diberikan.
    pub fn produced(&self) -> u64 {
        self.produced
    }

    /// State stream untuk checkpoint.
    pub fn state(&self) -> StreamState {
        self.stream.state()
    }

    /// Pulihkan state stream; `false` bila state berasal dari sumber lain.
    pub fn seek(&mut self, state: StreamState) -> bool {
        if !self.stream.seek(state) {
            return false;
        }
        self.produced = u64::from(state.produced) / self.batch as u64;
        true
    }

    /// Lompat ke posisi batch `step` (resume dari langkah trainer).
    ///
    /// Stream dibangun ulang dari asal dengan seed yang sama lalu `step` batch
    /// dilewati; karena urutan blok adalah fungsi deterministik dari seed,
    /// hasilnya identik dengan jalan tanpa interupsi.
    pub fn seek_step(&mut self, step: u64) -> Result<(), VirError> {
        if step == self.produced {
            return Ok(());
        }
        self.stream = self.rebuild()?;
        self.produced = 0;
        for _ in 0..step {
            self.take()?;
        }
        Ok(())
    }

    /// Bangun ulang stream dari asal (seed & konfigurasi sama).
    fn rebuild(&self) -> Result<TokenStream, VirError> {
        match &self.origin {
            Origin::Shard(path) => TokenStream::from_shard(path, self.cfg),
            Origin::Tokens(tokens) => Ok(TokenStream::from_tokens(tokens.clone(), self.cfg)),
        }
    }

    /// Ambil satu batch dan naikkan penghitung.
    fn take(&mut self) -> Result<(Vec<i64>, [usize; 2]), VirError> {
        let mut out = Vec::with_capacity(self.batch * self.len);
        for _ in 0..self.batch {
            let block = self.stream.next_block().ok_or_else(|| {
                VirError::Internal("stream token habis sebelum batch penuh".into())
            })?;
            for i in 0..self.len {
                out.push(block.get(i).copied().unwrap_or(0) as i64);
            }
        }
        self.produced += 1;
        Ok((out, [self.batch, self.len]))
    }
}

impl BatchSource for ShardBatchSource {
    fn next_batch(&mut self) -> Result<(Vec<i64>, [usize; 2]), VirError> {
        self.take()
    }
}
