//! Codebook VQ: kuantisasi terdekat, EMA, restart kode mati, straight-through
//! (PRD F1.4.a, rumus R1).
//!
//! Empat bagian PRD F1.4.a ada di berkas ini:
//!
//! 1. **R1** — `z_q^i = argmin_j ‖z^i − e_j‖²` dihitung sebagai
//!    `argmax_j (z·e_j − ½‖e_j‖²)`; lihat [`Codebook::quantize_idx`] dan
//!    [`Codebook::quantize_idx_reference`].
//! 2. **EMA** — `N_j ← γN_j + (1−γ)n_j`, `m_j ← γm_j + (1−γ)Σz`,
//!    `e_j = m_j/N_j`; lihat [`Codebook::ema_update`].
//! 3. **Restart kode mati** — entri dengan hitungan EMA di bawah
//!    [`CodebookConfig::restart_threshold`] diganti sampel dari batch.
//! 4. **Straight-through** — `z + (z_q − z).detach()`; lihat
//!    [`Codebook::straight_through`] dan tes gradien terhadap diferensiasi
//!    hingga di modul tes.
//!
//! Statistik utilisasi dan perplexity (NCM Gambar 16) dihitung oleh
//! [`crate::quantizer::QuantizerStats`] lewat implementasi
//! [`crate::quantizer::Quantizer`] di bawah.

use crate::quantizer::{histogram, Quantized, Quantizer, QuantizerStats};
use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::{Distribution, IndexingUpdateOp};
use rand::{Rng, RngCore};
use vir_common::VirError;

/// Konfigurasi satu codebook (NCM Tabel 5).
#[derive(Debug, Clone)]
pub struct CodebookConfig {
    /// Ukuran codebook `|V_k|` (4096 untuk OWT, 8192 untuk LM1B).
    pub num_entries: usize,
    /// Dimensi laten `d` (384 pada NCM).
    pub dim: usize,
    /// EMA decay `γ` (0,95–0,99 pada NCM Tabel 5).
    pub ema_decay: f32,
    /// Bobot suku komitmen `β` (0,25 pada kode acuan NCM).
    pub commitment_cost: f32,
    /// Hitungan EMA di bawah nilai ini menandai kode mati.
    pub restart_threshold: f32,
    /// Pencegah pembagian nol pada `m_j / N_j`.
    pub eps: f32,
}

impl Default for CodebookConfig {
    fn default() -> Self {
        Self {
            num_entries: 4096,
            dim: 384,
            ema_decay: 0.99,
            commitment_cost: 0.25,
            restart_threshold: 1.0,
            eps: 1e-5,
        }
    }
}

/// Codebook dengan cache skor untuk kuantisasi batch besar.
#[derive(Module, Debug)]
pub struct Codebook {
    /// Bobot kode `e`, `[|V|, d]`. Diperbarui EMA, bukan oleh gradien.
    pub weight: Param<Tensor<2>>,
    #[module(skip)]
    ema_sum: Tensor<2>,
    #[module(skip)]
    ema_count: Tensor<1>,
    #[module(skip)]
    e_t: Tensor<2>,
    #[module(skip)]
    e_sq: Tensor<1>,
    #[module(skip)]
    pub num_entries: usize,
    #[module(skip)]
    pub dim: usize,
    pub ema_decay: f32,
    pub commitment_cost: f32,
    pub restart_threshold: f32,
    pub eps: f32,
}

impl Codebook {
    /// Codebook acak `N(0,1)` dengan buffer EMA mulai dari bobot awal
    /// (`count = 1`, `sum = e`), sehingga `e = m/N` konsisten sejak awal.
    pub fn new(cfg: &CodebookConfig, device: &Device) -> Self {
        let shape = [cfg.num_entries, cfg.dim];
        let w = Tensor::random(shape, Distribution::Normal(0.0, 1.0), device);
        let e_t = w.clone().transpose();
        let e_sq = (w.clone() * w.clone()).sum_dim(1).reshape([cfg.num_entries]);
        Self {
            e_t,
            e_sq,
            ema_sum: w.clone(),
            ema_count: Tensor::ones([cfg.num_entries], device),
            weight: Param::from_tensor(w),
            num_entries: cfg.num_entries,
            dim: cfg.dim,
            ema_decay: cfg.ema_decay,
            commitment_cost: cfg.commitment_cost,
            restart_threshold: cfg.restart_threshold,
            eps: cfg.eps,
        }
    }

    /// Segarkan cache skor setelah bobot berubah (mis. setelah EMA).
    ///
    /// Bobot dilepas dari graf autodiff: codebook dipelajari EMA, bukan gradien,
    /// jadi cache tidak perlu ikut membawa graf.
    pub fn refresh_cache(&mut self) {
        let w = self.weight.val().detach();
        let v = w.dims()[0];
        self.e_t = w.clone().transpose();
        self.e_sq = (w.clone() * w.clone()).sum_dim(1).reshape([v]);
    }

    /// Tetapkan bobot baru tanpa mengganti `ParamId` (agar optimizer dan
    /// pemuatan checkpoint tetap cocok).
    fn set_weight(&mut self, weight: Tensor<2>) {
        crate::quantizer::replace_param(&mut self.weight, weight);
        self.refresh_cache();
    }

    /// Isi codebook dari data `[num_entries * dim]` (baris demi baris).
    ///
    /// Dipakai untuk inisialisasi dari k-means dan untuk mengunci invarian pada
    /// tes (codebook yang merepresentasikan data secara tepat).
    pub fn set_entries(&mut self, rows: &[f32]) -> Result<(), VirError> {
        if rows.len() != self.num_entries * self.dim {
            return Err(VirError::Config(format!(
                "set_entries butuh {} nilai, diberi {}",
                self.num_entries * self.dim,
                rows.len()
            )));
        }
        let device = self.weight.val().device();
        self.set_weight(crate::quantizer::rows_to_tensor(
            rows,
            self.dim,
            &device,
        ));
        Ok(())
    }

    /// Argmin nearest-neighbor (chunked, cache e_t/e_sq).
    ///
    /// Dengan feature `fused` dan device CubeCL f32, memakai kernel terfusi
    /// (matmul + argmax dalam satu kernel, tanpa buffer `[N, V]`); selain itu
    /// memakai [`Self::quantize_idx_reference`].
    pub fn quantize_idx(&self, z: Tensor<2>) -> Tensor<1, Int> {
        #[cfg(feature = "fused")]
        {
            // bias = -0.5 * ||e_j||^2 sudah di-cache (e_sq), jadi kernel hanya
            // menerima satu tensor [V] tambahan.
            let bias = self.e_sq.clone().mul_scalar(-0.5);
            if let Some(idx) = crate::kernel::try_fused_argmin(&z, &self.weight.val(), &bias) {
                return idx;
            }
        }
        self.quantize_idx_reference(z)
    }

    /// Jalur matmul + argmax biasa (chunked, cache e_t/e_sq).
    ///
    /// Dipakai sebagai fallback dan sebagai pembanding benih (ground truth) untuk
    /// kernel terfusi pada device non-CubeCL.
    pub fn quantize_idx_reference(&self, z: Tensor<2>) -> Tensor<1, Int> {
        let [n, d] = z.dims();
        let v = self.e_t.dims()[1];
        let chunk = 8192usize;
        let mut parts: Vec<Tensor<1, Int>> = Vec::new();
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            let zc = z.clone().slice([start..end, 0..d]);
            let cross = zc.matmul(self.e_t.clone());
            let scores = cross - self.e_sq.clone().reshape([1, v]) * 0.5;
            let rows = end - start;
            parts.push(scores.argmax(1).reshape([rows]));
            start = end;
        }
        if parts.len() == 1 {
            parts.remove(0)
        } else {
            Tensor::cat(parts, 0)
        }
    }

    /// Indeks + vektor terkuantisasi (`[N]`, `[N, d]`).
    ///
    /// Vektor terkuantisasi dilepas dari graf autodiff (nilainya tetap dipakai
    /// straight-through, yang hanya meneruskan gradien ke `z`).
    pub fn quantize_pair(&self, z: Tensor<2>) -> (Tensor<1, Int>, Tensor<2>) {
        let idx = self.quantize_idx(z);
        let q = self.weight.val().detach().select(0, idx.clone());
        (idx, q)
    }

    /// Straight-through: `z + (z_q - z).detach()` (NCM §3.3).
    pub fn straight_through(&self, z: Tensor<2>) -> Tensor<2> {
        let (_, q) = self.quantize_pair(z.clone());
        z.clone() + (q - z).detach()
    }

    /// Pembaruan EMA + restart kode mati (F1.4.a langkah 2).
    ///
    /// `u` adalah batch `[N, d]` dan `indices` hasil [`Self::quantize_idx`].
    /// Kode mati diganti sampel acak dari `u` (NCM §3.3 "restart kode mati
    /// dengan sampel dari batch").
    pub fn ema_update(&mut self, u: &Tensor<2>, indices: &Tensor<1, Int>, rng: &mut dyn RngCore) {
        let v = self.num_entries;
        let onehot = indices.clone().float().one_hot::<2>(v);
        let batch_count = onehot.clone().sum_dim(0).reshape([v]);
        let batch_sum = onehot.transpose().matmul(u.clone().detach());
        let decay = self.ema_decay;
        let mut count = self.ema_count.clone().mul_scalar(decay)
            + batch_count.mul_scalar(1.0 - decay);
        let mut sum =
            self.ema_sum.clone().mul_scalar(decay) + batch_sum.mul_scalar(1.0 - decay);
        self.restart_dead(u, &mut count, &mut sum, rng);

        // `count` sudah dijaga `>= eps`; reshape agar bisa membagi `[V, d]`.
        let denom = count.clone().clamp_min(self.eps).reshape([v, 1]);
        let weight = sum.clone() / denom;
        self.ema_count = count;
        self.ema_sum = sum;
        self.set_weight(weight);
    }

    /// Ganti kode dengan hitungan EMA di bawah ambang memakai sampel batch.
    fn restart_dead(
        &self,
        u: &Tensor<2>,
        count: &mut Tensor<1>,
        sum: &mut Tensor<2>,
        rng: &mut dyn RngCore,
    ) {
        let rows = u.dims()[0];
        if rows == 0 {
            return;
        }
        let live = count.clone().to_data().try_to_vec::<f32>().unwrap_or_default();
        let dead: Vec<i64> = live
            .iter()
            .enumerate()
            .filter(|(_, c)| **c < self.restart_threshold)
            .map(|(i, _)| i as i64)
            .collect();
        if dead.is_empty() {
            return;
        }
        let device = u.device();
        let dead_idx = crate::quantizer::vec_to_int_tensor(&dead, &device);
        let picks_vec: Vec<i64> = (0..dead.len())
            .map(|_| Rng::gen_range(&mut *rng, 0..rows) as i64)
            .collect();
        let samples = u
            .clone()
            .select(0, crate::quantizer::vec_to_int_tensor(&picks_vec, &device));
        let ones = Tensor::ones([dead_idx.dims()[0]], &device);
        *count = count
            .clone()
            .select_assign(0, dead_idx.clone(), ones, IndexingUpdateOp::Assign);
        *sum = sum.clone().select_assign(0, dead_idx, samples, IndexingUpdateOp::Assign);
    }

    /// Jumlah kode mati menurut hitungan EMA saat ini.
    pub fn dead_codes(&self) -> usize {
        let live = self
            .ema_count
            .clone()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        live.iter()
            .filter(|c| **c < self.restart_threshold)
            .count()
    }

    /// Jumlah entri yang terpakai minimal sekali (tanpa batch baru).
    pub fn used_codes(&self) -> usize {
        let live = self
            .ema_count
            .clone()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        live.iter().filter(|c| **c >= 1.0).count()
    }
}

impl Quantizer for Codebook {
    fn num_codes(&self) -> usize {
        self.num_entries
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn quantize(&self, u: Tensor<2>) -> Quantized {
        let (idx, q) = self.quantize_pair(u);
        Quantized {
            indices: idx,
            quantized: q.clone(),
            codes: q,
        }
    }

    fn update(&mut self, indices: &Tensor<1, Int>, u: &Tensor<2>, rng: &mut dyn RngCore) {
        self.ema_update(u, indices, rng);
    }

    fn lookup(&self, indices: Tensor<1, Int>) -> Tensor<2> {
        self.weight.val().detach().select(0, indices)
    }

    fn stats(&self, indices: &Tensor<1, Int>) -> QuantizerStats {
        QuantizerStats::from_counts(&histogram(indices, self.num_entries))
    }
}

