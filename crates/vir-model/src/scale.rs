//! Operator skala level `Down_k`/`R_k` (PRD F1.4.b) untuk generator.
//!
//! Satu [`LevelScales`] memegang matriks turun/naik tiap level untuk satu
//! panjang konteks. Konteks `L` dipakai saat latihan dan satu blok; loop blok
//! (F2.3) membangun ulang karena prefix membuat konteks bertambah.

use burn::prelude::*;
use burn::tensor::Device;
use vir_vq::{Phi, ScaleOps};

/// `Down_k`/`R_k` untuk konteks `[B, ctx, d] → [B, ℓ_k, d]`.
#[derive(Debug, Clone)]
pub struct LevelScales {
    ops: Vec<ScaleOps>,
    levels: Vec<usize>,
    ctx_len: usize,
}

impl LevelScales {
    /// Operator turun/naik untuk setiap level pada konteks `ctx_len`.
    pub fn new(levels: &[usize], ctx_len: usize, device: &Device) -> Self {
        Self {
            ops: levels
                .iter()
                .map(|&len| ScaleOps::new(ctx_len, len, device))
                .collect(),
            levels: levels.to_vec(),
            ctx_len,
        }
    }

    /// Jumlah level.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Panjang konteks `ctx_len`.
    pub fn ctx_len(&self) -> usize {
        self.ctx_len
    }

    /// Panjang level `ℓ_k`.
    pub fn level_len(&self, k: usize) -> usize {
        self.levels[k]
    }

    /// Turun: `[B, ctx, d] → [B, ℓ_k, d]`.
    pub fn down(&self, k: usize, x: Tensor<3>) -> Tensor<3> {
        self.ops[k].down(x)
    }

    /// Naik: `[B, ℓ_k, d] → [B, ctx, d]`.
    pub fn up(&self, k: usize, e: Tensor<3>) -> Tensor<3> {
        self.ops[k].up(e)
    }

    /// `R_k(e) = up(e) + Phi_k(up(e))` (F1.4.b langkah 4).
    pub fn reconstruct(&self, k: usize, phi: &Phi, e: Tensor<3>) -> Tensor<3> {
        let up = self.up(k, e);
        up.clone() + phi.forward(up)
    }
}
