//! Kedalaman rekuren (PRD F2.1) dan pengendali halting (F2.2).
//!
//! Rumus F2.1 langkah 1 (arXiv 2502.05171 §3):
//!
//! ```text
//! e   = P(x)                       prelude
//! s_i = R(e, s_{i−1})  i = 1..r    inti, injeksi e lewat adapter A([s, e])
//! p   = C(s_r)                     coda
//! ```
//!
//! Kedalaman efektif `l_P + r·l_R + l_C`; jumlah parameter tidak bertambah saat
//! `r` naik (F2.1). Satu `r` per mikro-batch (locked-step) saat latihan, `r`
//! tetap saat inferensi (F2.1 langkah 3).

use burn::nn::Linear;
use burn::prelude::*;
use vir_nn::Block;

/// Prelude → `r` iterasi inti dengan injeksi `e` → coda.
///
/// `r = 0` diperlakukan sebagai satu iterasi: jalur rekuren tidak boleh kosong
/// (F2.1 langkah 3 menjamin `r ≥ 1`).
pub fn recurrent_forward(
    prelude: &[Block],
    core: &[Block],
    coda: &[Block],
    adapter: &Linear,
    e: Tensor<3>,
    r: usize,
) -> Tensor<3> {
    let mut s = prelude.iter().fold(e.clone(), |acc, block| block.forward(acc));
    for _ in 0..r.max(1) {
        // Injeksi ulang keluaran prelude di tiap iterasi (F2.1 langkah 1).
        let cat = Tensor::cat(vec![s.clone(), e.clone()], 2);
        let inj = adapter.forward(cat);
        s = core.iter().fold(s + inj, |acc, block| block.forward(acc));
    }
    coda.iter().fold(s, |acc, block| block.forward(acc))
}

/// Sampling `r ~ Poisson(exp(τ)) + 1`, `τ ~ N(log r̄ − σ²/2, σ)` (F2.1 langkah 3).
pub fn sample_r(mean_r: f32, sigma: f32, rng: &mut impl rand::Rng) -> usize {
    let mu = mean_r.max(1e-3).ln() - 0.5 * sigma * sigma;
    let tau = mu + sigma * rand_distr(rng);
    (poisson(tau.exp(), rng) + 1).max(1)
}

/// `N(0, 1)` lewat Box–Muller.
fn rand_distr(rng: &mut impl rand::Rng) -> f32 {
    let u1 = rand::Rng::gen_range(&mut *rng, 0.0f32..1.0).max(1e-7);
    let u2 = rand::Rng::gen_range(&mut *rng, 0.0f32..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// `Poisson(λ)` (Knuth) dengan batas iterasi agar selalu berakhir.
fn poisson(lambda: f32, rng: &mut impl rand::Rng) -> usize {
    let l = (-lambda.clamp(0.0, 1.0e6)).exp();
    let mut k = 0usize;
    let mut p = 1.0f32;
    for _ in 0..10_000 {
        p *= rand::Rng::gen_range(&mut *rng, 0.0f32..1.0);
        if p <= l {
            return k;
        }
        k += 1;
    }
    k
}

/// Pengendali halting (F2.2): putuskan kapan berhenti menambah iterasi/level.
pub trait Halter {
    /// `true` bila iterasi berikutnya sebaiknya dihentikan.
    fn should_halt(&self, entropy: f32) -> bool;
}

/// Halting tetap: tidak pernah keluar dini (F2.2 tahap 1).
#[derive(Debug, Clone, Copy)]
pub struct FixedHalter {
    /// Batas iterasi (dipakai pemanggil, bukan halter ini).
    pub max_loops: usize,
}

impl Halter for FixedHalter {
    fn should_halt(&self, _entropy: f32) -> bool {
        false
    }
}

/// Halting entropi: berhenti saat entropi di bawah ambang (F2.2 tahap 2).
#[derive(Debug, Clone, Copy)]
pub struct EntropyHalter {
    /// Ambang entropi (dikalibrasi, bukan dari makalah).
    pub threshold: f32,
}

impl Halter for EntropyHalter {
    fn should_halt(&self, entropy: f32) -> bool {
        entropy < self.threshold
    }
}

/// Entropi Shannon dari distribusi probabilitas.
pub fn entropy(probs: &[f32]) -> f32 {
    -probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}