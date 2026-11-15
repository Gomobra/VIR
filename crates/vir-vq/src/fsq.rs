//! Kuantizer alternatif: FSQ, RFSQ (skala + LayerNorm terbalik), iFSQ
//! (PRD F1.4.f).
//!
//! FSQ (arXiv 2309.15505): proyeksikan laten ke `n < 10` dimensi, lalu
//! bulatkan tiap dimensi ke himpunan nilai tetap. Tidak ada codebook yang
//! dipelajari, tidak ada commitment loss, dan tidak ada codebook collapse.
//!
//! ```text
//! half_l  = (L−1)(1+ε)/2,  offset = 0,5 bila L genap (selain itu 0)
//! shift   = atanh(offset / half_l)
//! bound(z)= tanh(z + shift)·half_l − offset
//! q       = round(bound(z))  dengan STE:  z + (round(z) − z).detach()
//! idx     = Σ_i (q_i + ⌊L_i/2⌋)·b_i ,  b_i = ∏_{j<i} L_j
//! ```
//!
//! Varian:
//!
//! - **RFSQ strategi 1** ([`Rfsq`]): skala terpelajar per tahap
//!   `q_k = FSQ_k(α_k·r)`, `r_k = r − q_k/α_k` (arXiv 2508.15860).
//! - **RFSQ strategi 2** ([`RfsqNorm`]): LayerNorm yang bisa dibalik pada
//!   residual, `r_k = r − LayerNorm⁻¹(q_k)` (simpan statistik normalisasi).
//! - **iFSQ** ([`FsqConfig::ifsq`]): `tanh` diganti `2σ(1,6x) − 1` agar
//!   aktivasi mendekati seragam (arXiv 2601.17124; temuan dari gambar, jadi
//!   diuji di VIR).

use crate::quantizer::{Quantized, Quantizer, QuantizerStats, histogram, vec_to_tensor};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use rand::RngCore;
use vir_common::VirError;

/// Aktivasi pembatas FSQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsqActivation {
    /// `tanh(z + shift)` — FSQ asli.
    Tanh,
    /// `2σ(1,6z) − 1` — iFSQ.
    Sigmoid,
}

/// Konfigurasi FSQ: level per dimensi + aktivasi.
#[derive(Debug, Clone)]
pub struct FsqConfig {
    /// Himpunan nilai per dimensi `L_i` (`L_i ≥ 5` disarankan makalah).
    pub levels: Vec<usize>,
    /// `ε` pada `half_l` (10⁻³ pada kode acuan).
    pub eps: f32,
    /// Aktivasi pembatas.
    pub activation: FsqActivation,
}

impl FsqConfig {
    /// FSQ dengan `tanh`; `|V| = ∏ L_i`.
    pub fn new(levels: Vec<usize>) -> Self {
        Self {
            levels,
            eps: 1e-3,
            activation: FsqActivation::Tanh,
        }
    }

    /// iFSQ: aktivasi sigmoid kembar (PRD F1.4.f langkah 3).
    pub fn ifsq(levels: Vec<usize>) -> Self {
        Self {
            activation: FsqActivation::Sigmoid,
            ..Self::new(levels)
        }
    }

    /// Ukuran codebook implisit `∏ L_i`.
    pub fn codebook_size(&self) -> usize {
        self.levels.iter().product()
    }

    /// Validasi konfigurasi.
    pub fn validate(&self) -> Result<(), VirError> {
        if self.levels.is_empty() {
            return Err(VirError::Config("levels FSQ tidak boleh kosong".into()));
        }
        if self.levels.len() > 10 {
            return Err(VirError::Config(
                "FSQ memakai <= 10 dimensi (PRD F1.4.f)".into(),
            ));
        }
        if let Some(bad) = self.levels.iter().find(|l| **l < 2) {
            return Err(VirError::Config(format!("L_i >= 2, diberi {bad}")));
        }
        Ok(())
    }
}

impl Default for FsqConfig {
    fn default() -> Self {
        // Kandidat `|V_k| = 4096` dari PRD F1.4.f langkah 2.
        Self::new(vec![7, 5, 5, 5, 5])
    }
}

/// FSQ: proyeksi linear `d → n`, pembulatan, proyeksi balik `n → d`.
#[derive(Module, Debug)]
pub struct Fsq {
    pub proj: Linear,
    pub unproj: Linear,
    #[module(skip)]
    pub levels: Vec<usize>,
    #[module(skip)]
    pub activation: FsqActivation,
    #[module(skip)]
    half_l: Tensor<1>,
    #[module(skip)]
    half_exact: Tensor<1>,
    #[module(skip)]
    shift: Tensor<1>,
    #[module(skip)]
    offset: Tensor<1>,
    #[module(skip)]
    basis: Tensor<1>,
    #[module(skip)]
    eps: f32,
    #[module(skip)]
    updates: u64,
}

impl Fsq {
    /// Rakit FSQ dari konfigurasi.
    pub fn new(cfg: &FsqConfig, dim: usize, device: &Device) -> Result<Self, VirError> {
        cfg.validate()?;
        let n = cfg.levels.len();
        let mut half_l = Vec::with_capacity(n);
        let mut half_exact = Vec::with_capacity(n);
        let mut shift = Vec::with_capacity(n);
        let mut offset = Vec::with_capacity(n);
        for l in &cfg.levels {
            let lf = *l as f32;
            let hl = (lf - 1.0) * (1.0 + cfg.eps) / 2.0;
            let off = if l % 2 == 0 { 0.5 } else { 0.0 };
            let sh = (off / hl).atanh();
            half_l.push(hl);
            half_exact.push((lf - 1.0) / 2.0);
            shift.push(sh);
            offset.push(off);
        }
        let mut basis = Vec::with_capacity(n);
        let mut acc = 1usize;
        for l in &cfg.levels {
            basis.push(acc as f32);
            acc *= l;
        }
        Ok(Self {
            proj: LinearConfig::new(dim, n).init(device),
            unproj: LinearConfig::new(n, dim).init(device),
            levels: cfg.levels.clone(),
            activation: cfg.activation,
            half_l: vec_to_tensor(&half_l, device),
            half_exact: vec_to_tensor(&half_exact, device),
            shift: vec_to_tensor(&shift, device),
            offset: vec_to_tensor(&offset, device),
            basis: vec_to_tensor(&basis, device),
            eps: cfg.eps,
            updates: 0,
        })
    }

    /// Jumlah pembaruan state yang diterima (untuk membedakan dari VQ yang
    /// benar-benar memperbarui codebook).
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Pembatasan ke `[-half-offset, half-offset]` (langkah 1).
    fn bound(&self, z: Tensor<2>) -> Tensor<2> {
        let shape = [1, self.levels.len()];
        let shifted = match self.activation {
            FsqActivation::Tanh => (z + self.shift.clone().reshape(shape)).tanh(),
            FsqActivation::Sigmoid => {
                // iFSQ: 2σ(1,6z) − 1 (PRD F1.4.f langkah 3).
                sigmoid(z.mul_scalar(1.6)).mul_scalar(2.0) - 1.0
            }
        };
        shifted * self.half_l.clone().reshape(shape) - self.offset.clone().reshape(shape)
    }

    /// Pembulatan dengan straight-through estimator (langkah 1).
    fn round_ste(&self, x: Tensor<2>) -> Tensor<2> {
        let rounded = x.clone().round();
        x.clone() + (rounded - x).detach()
    }

    /// Kode diskret `[N, n]` (float, nilai bulat) dari `u` `[N, d]`.
    pub fn quantize_lowdim(&self, u: Tensor<2>) -> Tensor<2> {
        self.round_ste(self.bound(self.proj.forward(u)))
    }

    /// Indeks campuran (`idx = Σ (q_i + ⌊L_i/2⌋)·b_i`) dari kode `[N, n]`.
    fn mixed_radix_index(&self, codes: Tensor<2>) -> Tensor<1, Int> {
        let shape = [1, self.levels.len()];
        let digits = codes.clone() + self.half_exact.clone().reshape(shape);
        let flat = digits.matmul(self.basis.clone().reshape([self.levels.len(), 1]));
        flat.reshape([codes.dims()[0]]).round().int()
    }

    /// Dekode indeks campuran kembali ke kode `[N, n]` (untuk `lookup`).
    fn decode_index(&self, indices: Tensor<1, Int>) -> Tensor<2> {
        let n = self.levels.len();
        let rows = indices.dims()[0];
        let values = indices.float().reshape([rows, 1]);
        let mut digits: Vec<Tensor<2>> = Vec::with_capacity(n);
        let mut rest = values;
        for l in &self.levels {
            let lf = *l as f32;
            let div = rest.clone().div_scalar(lf).floor();
            digits.push(rest.clone() - div.clone().mul_scalar(lf));
            rest = div;
        }
        Tensor::<2>::cat(digits, 1) - self.half_exact.clone().reshape([1, n])
    }

    /// Kuat: `n` dimensi kode dari `u`.
    fn codes_of(&self, u: Tensor<2>) -> (Tensor<1, Int>, Tensor<2>, Tensor<2>) {
        let codes = self.quantize_lowdim(u.clone());
        let idx = self.mixed_radix_index(codes.clone());
        let quantized = self.unproj.forward(codes.clone());
        (idx, codes, quantized)
    }
}

impl Quantizer for Fsq {
    fn num_codes(&self) -> usize {
        self.levels.iter().product()
    }

    fn dim(&self) -> usize {
        self.proj.weight.val().dims()[1]
    }

    fn quantize(&self, u: Tensor<2>) -> Quantized {
        let (idx, codes, quantized) = self.codes_of(u);
        Quantized {
            indices: idx,
            quantized,
            codes,
        }
    }

    /// FSQ tidak punya codebook terpelajar: tidak ada EMA, restart, atau loss
    /// komitmen (PRD F1.4.f). Yang berubah hanya penghitung pembaruan.
    fn update(&mut self, _indices: &Tensor<1, Int>, _u: &Tensor<2>, _rng: &mut dyn RngCore) {
        self.updates = self.updates.saturating_add(1);
    }

    fn lookup(&self, indices: Tensor<1, Int>) -> Tensor<2> {
        self.unproj.forward(self.decode_index(indices))
    }

    fn stats(&self, indices: &Tensor<1, Int>) -> QuantizerStats {
        QuantizerStats::from_counts(&histogram(indices, self.num_codes()))
    }
}

/// RFSQ strategi 1: skala terpelajar per tahap (arXiv 2508.15860).
#[derive(Module, Debug)]
pub struct Rfsq {
    pub inner: Fsq,
    /// `α` per tahap, diinisialisasi 1,0.
    pub alpha: Param<Tensor<1>>,
}

impl Rfsq {
    pub fn new(cfg: &FsqConfig, dim: usize, device: &Device) -> Result<Self, VirError> {
        Ok(Self {
            inner: Fsq::new(cfg, dim, device)?,
            alpha: Param::from_tensor(Tensor::ones([1], device)),
        })
    }
}

impl Quantizer for Rfsq {
    fn num_codes(&self) -> usize {
        self.inner.num_codes()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn quantize(&self, u: Tensor<2>) -> Quantized {
        // q = unproj(FSQ(proj(α·r))); r_q = q / α.
        let a = self.alpha.val().reshape([1, 1]);
        let q = self.inner.quantize(u * a.clone());
        let quantized = q.quantized / a;
        Quantized {
            indices: q.indices,
            quantized,
            codes: q.codes,
        }
    }

    fn update(&mut self, indices: &Tensor<1, Int>, u: &Tensor<2>, rng: &mut dyn RngCore) {
        self.inner.update(indices, u, rng);
    }

    fn lookup(&self, indices: Tensor<1, Int>) -> Tensor<2> {
        let a = self.alpha.val().reshape([1, 1]);
        self.inner.lookup(indices) / a
    }

    fn stats(&self, indices: &Tensor<1, Int>) -> QuantizerStats {
        self.inner.stats(indices)
    }
}

/// Statistik normalisasi untuk inversi (RFSQ strategi 2).
#[derive(Debug, Clone)]
pub struct Normalized {
    /// Input ter���ormalisasi, `[N, d]`.
    pub x: Tensor<2>,
    mean: Tensor<2>,
    rstd: Tensor<2>,
}

/// Normalisasi pada sumbu `d` sekaligus menyimpan statistik untuk inversi.
pub fn normalize_with_stats(x: Tensor<2>, eps: f32) -> Normalized {
    let [n, _d] = x.dims();
    let mean = x.clone().mean_dim(1).reshape([n, 1]);
    let centered = x - mean.clone();
    let var = (centered.clone() * centered.clone())
        .mean_dim(1)
        .reshape([n, 1]);
    let rstd = (var + eps).sqrt().recip();
    Normalized {
        x: centered.clone() * rstd.clone(),
        mean,
        rstd,
    }
}

/// Inversi normalisasi (`LayerNorm⁻¹`) memakai statistik tersimpan.
pub fn denormalize(q: Tensor<2>, stats: &Normalized) -> Tensor<2> {
    q / stats.rstd.clone() + stats.mean.clone()
}

/// RFSQ strategi 2: LayerNorm yang bisa dibalik pada residual.
#[derive(Module, Debug)]
pub struct RfsqNorm {
    pub inner: Fsq,
    #[module(skip)]
    pub eps: f32,
}

impl RfsqNorm {
    pub fn new(cfg: &FsqConfig, dim: usize, device: &Device) -> Result<Self, VirError> {
        Ok(Self {
            inner: Fsq::new(cfg, dim, device)?,
            eps: 1e-5,
        })
    }
}

impl Quantizer for RfsqNorm {
    fn num_codes(&self) -> usize {
        self.inner.num_codes()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn quantize(&self, u: Tensor<2>) -> Quantized {
        let stats = normalize_with_stats(u, self.eps);
        let q = self.inner.quantize(stats.x.clone());
        let quantized = denormalize(q.quantized, &stats);
        Quantized {
            indices: q.indices,
            quantized,
            codes: q.codes,
        }
    }

    fn update(&mut self, indices: &Tensor<1, Int>, u: &Tensor<2>, rng: &mut dyn RngCore) {
        self.inner.update(indices, u, rng);
    }

    fn lookup(&self, indices: Tensor<1, Int>) -> Tensor<2> {
        self.inner.lookup(indices)
    }

    fn stats(&self, indices: &Tensor<1, Int>) -> QuantizerStats {
        self.inner.stats(indices)
    }
}
