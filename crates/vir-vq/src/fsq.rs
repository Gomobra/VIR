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

use crate::quantizer::{histogram, vec_to_tensor, Quantized, Quantizer, QuantizerStats};
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
        self.inner.lookup(indices)
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
    let var = (centered.clone() * centered.clone()).mean_dim(1).reshape([n, 1]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn device() -> Device {
        Device::default()
    }

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(3)
    }

    /// Codebook implisit = hasil kali level.
    #[test]
    fn codebook_size_is_product_of_levels() {
        assert_eq!(FsqConfig::new(vec![7, 5, 5, 5, 5]).codebook_size(), 4375);
        assert_eq!(FsqConfig::new(vec![8, 5, 5, 5]).codebook_size(), 1000);
        assert_eq!(FsqConfig::new(vec![16, 16, 16]).codebook_size(), 4096);
        assert_eq!(FsqConfig::new(vec![8, 8, 6, 5]).codebook_size(), 1920);
    }

    /// Konfigurasi tidak valid harus ditolak.
    #[test]
    fn config_validation() {
        assert!(FsqConfig::new(vec![]).validate().is_err());
        assert!(FsqConfig::new(vec![1]).validate().is_err());
        assert!(FsqConfig::new(vec![5; 11]).validate().is_err());
        assert!(FsqConfig::new(vec![5, 5]).validate().is_ok());
    }

    /// Kode harus berada dalam rentang dan indeks di `[0, |V|)`.
    #[test]
    fn quantize_produces_valid_indices() {
        let d = device();
        let cfg = FsqConfig::new(vec![7, 5]);
        let fsq = Fsq::new(&cfg, 4, &d).expect("fsq");
        let u = Tensor::<2>::random([32, 4], Distribution::Normal(0.0, 1.0), &d);
        let q = fsq.quantize(u);
        assert_eq!(q.indices.dims(), [32]);
        let idx: Vec<i64> = q.indices.clone().into_data().try_to_vec::<i64>().unwrap_or_default();
        assert!(idx.iter().all(|i| *i >= 0 && *i < 35), "idx={idx:?}");

        let codes: Vec<f32> = q.codes.into_data().to_vec::<f32>().unwrap_or_default();
        // Dimensi 0: L=7 → bulat ∈ [-3, 3]; dimensi 1: L=5 → ∈ [-2, 2].
        for (i, v) in codes.iter().enumerate() {
            let bound = if i % 2 == 0 { 3.0 } else { 2.0 };
            assert!(v.abs() <= bound + 1e-4, "codes={codes:?}");
        }
    }

    /// `lookup(quantize(u).indices)` harus mengembalikan kode yang sama.
    #[test]
    fn lookup_inverts_quantize() {
        let d = device();
        let cfg = FsqConfig::new(vec![7, 5]);
        let fsq = Fsq::new(&cfg, 4, &d).expect("fsq");
        let u = Tensor::<2>::random([16, 4], Distribution::Normal(0.0, 1.0), &d);
        let q = fsq.quantize(u);
        let back = fsq.lookup(q.indices);
        let a: Vec<f32> = q.codes.into_data().to_vec::<f32>().unwrap_or_default();
        let b: Vec<f32> = back.into_data().to_vec::<f32>().unwrap_or_default();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "a={a:?} b={b:?}");
        }
    }

    /// `straight_through` FSQ sama dengan kode terkuantisasi dan gradiennya 1.
    #[test]
    fn straight_through_passes_gradient() {
        let d = Device::default().autodiff();
        let cfg = FsqConfig::new(vec![5, 5]);
        let fsq = Fsq::new(&cfg, 3, &d).expect("fsq");
        let u = Tensor::<2>::from_floats([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]], &d).require_grad();
        let q = fsq.quantize(u.clone());
        let st = fsq.straight_through(u.clone());
        let loss = (st.clone() * st.clone()).sum();
        let grads = loss.backward();
        let g = u
            .grad(&grads)
            .unwrap_or_else(|| Tensor::zeros([2, 3], &d));
        let gv: Vec<f32> = g.into_data().to_vec::<f32>().unwrap_or_default();
        let sv: Vec<f32> = st.into_data().to_vec::<f32>().unwrap_or_default();
        let qv: Vec<f32> = q.quantized.into_data().to_vec::<f32>().unwrap_or_default();
        for (a, b) in sv.iter().zip(qv.iter()) {
            assert!((a - b).abs() < 1e-5, "st={sv:?} q={qv:?}");
        }
        for v in &gv {
            assert!(v.abs() > 0.0, "gradien hilang: {gv:?}");
        }
    }

    /// FSQ tidak punya collapse: semua entri histogram kosong kecuali yang
    /// terpakai, dan utilisasi = used/|V|.
    #[test]
    fn stats_have_no_collapse_by_construction() {
        let d = device();
        let cfg = FsqConfig::new(vec![5, 5]);
        let fsq = Fsq::new(&cfg, 3, &d).expect("fsq");
        let u = Tensor::<2>::random([64, 3], Distribution::Normal(0.0, 1.0), &d);
        let q = fsq.quantize(u);
        let stats = fsq.stats(&q.indices);
        assert_eq!(stats.codes, 25);
        assert_eq!(stats.used, stats.counts.iter().filter(|c| **c > 0).count());
        assert!(stats.perplexity >= 1.0);
    }

    /// RFSQ: `lookup∘quantize` konsisten dan `alpha = 1` pada awal.
    #[test]
    fn rfsq_roundtrip_with_unit_scale() {
        let d = device();
        let cfg = FsqConfig::new(vec![5, 5, 5]);
        let q = Rfsq::new(&cfg, 4, &d).expect("rfsq");
        let u = Tensor::<2>::random([8, 4], Distribution::Normal(0.0, 1.0), &d);
        let quantized = q.quantize(u);
        let back = q.lookup(quantized.indices);
        assert_eq!(back.dims(), [8, 4]);
        let a: f32 = q.alpha.val().into_data().to_vec::<f32>().unwrap_or_default()[0];
        assert!((a - 1.0).abs() < 1e-6);
    }

    /// RFSQ dengan skala ≠ 1: `lookup` kode FSQ dibagi α harus konsisten.
    #[test]
    fn rfsq_scale_is_inverted() {
        let d = device();
        let cfg = FsqConfig::new(vec![5, 5]);
        let mut q = Rfsq::new(&cfg, 3, &d).expect("rfsq");
        let alpha = Tensor::<1>::from_floats([2.0], &d);
        crate::quantizer::replace_param(&mut q.alpha, alpha);

        let u = Tensor::<2>::from_floats([[0.3, 0.1, 0.2]], &d);
        let out = q.quantize(u);
        let back = q.lookup(out.indices);
        let a: Vec<f32> = out.quantized.into_data().to_vec::<f32>().unwrap_or_default();
        let b: Vec<f32> = back.into_data().to_vec::<f32>().unwrap_or_default();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "a={a:?} b={b:?}");
        }
    }

    /// LayerNorm + inversi harus mengembalikan input (invarian RFSQ strategi 2).
    #[test]
    fn denormalize_inverts_normalize() {
        let d = device();
        let x = Tensor::<2>::from_floats(
            [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 0.0, 2.0]],
            &d,
        );
        let stats = normalize_with_stats(x.clone(), 1e-5);
        let back = denormalize(stats.x.clone(), &stats);
        let a: Vec<f32> = x.into_data().to_vec::<f32>().unwrap_or_default();
        let b: Vec<f32> = back.into_data().to_vec::<f32>().unwrap_or_default();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "a={a:?} b={b:?}");
        }
        // Ter-Normalisasi: mean 0, simpangan 1.
        let mean: f32 = stats
            .x
            .clone()
            .mean_dim(1)
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default()[0];
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    /// iFSQ memakai aktivasi sigmoid, tetapi tetap menghasilkan indeks valid.
    #[test]
    fn ifsq_uses_sigmoid_and_valid_indices() {
        let d = device();
        let cfg = FsqConfig::ifsq(vec![16, 16, 16]);
        assert_eq!(cfg.activation, FsqActivation::Sigmoid);
        let fsq = Fsq::new(&cfg, 3, &d).expect("ifsq");
        assert_eq!(fsq.num_codes(), 4096);
        let u = Tensor::<2>::random([24, 3], Distribution::Normal(0.0, 1.0), &d);
        let q = fsq.quantize(u);
        let idx: Vec<i64> = q.indices.into_data().try_to_vec::<i64>().unwrap_or_default();
        assert!(idx.iter().all(|i| *i >= 0 && *i < 4096));
    }

    /// `update` FSQ tidak mengubah bobot proyeksi (tidak ada EMA).
    #[test]
    fn fsq_update_is_stateless() {
        let d = device();
        let cfg = FsqConfig::new(vec![5, 5]);
        let mut fsq = Fsq::new(&cfg, 3, &d).expect("fsq");
        let before: Vec<f32> = fsq
            .proj
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        let u = Tensor::<2>::random([8, 3], Distribution::Normal(0.0, 1.0), &d);
        let idx = fsq.quantize(u.clone()).indices;
        fsq.update(&idx, &u, &mut rng());
        assert_eq!(fsq.updates(), 1);
        let after: Vec<f32> = fsq
            .proj
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        assert_eq!(before, after);
    }
}