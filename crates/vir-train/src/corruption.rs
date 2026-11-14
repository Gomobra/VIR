//! Corruption kode masukan per level (PRD F1.7 langkah 2, NCM §3.4 / F.3).
//!
//! Sebagian kode masukan level `k` diganti entri acak **seragam dari codebook
//! level yang sama**; target tetap kode bersih. Laju turun linear terhadap
//! level (level kasar lebih banyak dikorupsi) dengan rata-rata `mean_rate`;
//! sapuan laju rata-rata `{0; 0,25; 0,5; 0,75}` dipakai untuk ablasi.
//!
//! Jumlah posisi terkorupsi dibuat **tepat** (`round(rate · N)`), sehingga
//! laju sebenarnya bisa diuji mendekati target, bukan hanya diharapkan.

use crate::rng::Rng;
use burn::prelude::*;
use burn::tensor::Bool;
use serde::{Deserialize, Serialize};

/// Bentuk jadwal laju korupsi terhadap level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateShape {
    /// Laju sama untuk semua level.
    Flat,
    /// Turun linear: `rate_k = 2·mean·(1 − k/(K−1))`, rata-rata = `mean`.
    LinearDecay,
}

/// Laju per level untuk `levels` (panjang level, urut naik).
///
/// K=1 selalu mengembalikan `mean_rate`. Untuk [`RateShape::LinearDecay`],
/// `rate_k = clamp(2·mean·(1 − t), 0, 1)` dengan `t = k/(K−1)`; karena rata-rata
/// `1 − t` adalah 1/2, laju rata-rata hasilnya tepat `mean`.
pub fn level_rates(levels: &[usize], mean_rate: f32, shape: RateShape) -> Vec<f32> {
    let k = levels.len();
    if k == 0 {
        return Vec::new();
    }
    if k == 1 || matches!(shape, RateShape::Flat) {
        return vec![mean_rate.clamp(0.0, 1.0); k];
    }
    let denom = (k - 1) as f32;
    (0..k)
        .map(|i| {
            let t = i as f32 / denom;
            (2.0 * mean_rate * (1.0 - t)).clamp(0.0, 1.0)
        })
        .collect()
}

/// Satu batch korupsi untuk satu level: mask posisi dan kode pengganti.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorruptBatch {
    /// `true` bila posisi diganti (panjang `batch · ℓ_k`).
    pub mask: Vec<bool>,
    /// Kode pengganti (0 di posisi yang tidak terkorupsi).
    pub codes: Vec<i32>,
    /// Jumlah posisi terkorupsi pada batch ini.
    pub corrupted: usize,
    /// Jumlah posisi pada batch ini (`batch · ℓ_k`).
    pub positions: usize,
}

impl CorruptBatch {
    /// Laju sebenarnya pada batch ini (untuk log dan tes).
    pub fn measured_rate(&self) -> f32 {
        if self.positions == 0 {
            return 0.0;
        }
        self.corrupted as f32 / self.positions as f32
    }
}

/// Generator korupsi dengan state RNG (bisa di-checkpoint).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Corruption {
    /// Laju rata-rata yang diminta.
    pub mean_rate: f32,
    /// Bentuk jadwal.
    pub shape: RateShape,
    /// Laju per level hasil `level_rates`.
    pub rates: Vec<f32>,
    /// Laju sebenarnya pada batch terakhir tiap level (host, untuk log).
    pub last_rates: Vec<f32>,
    rng: Rng,
}

impl Corruption {
    /// Tanpa korupsi (mean 0) — dipakai sebagai pembanding pada ablasi.
    pub fn none(seed: u64, levels: &[usize]) -> Self {
        Self::new(seed, levels, 0.0, RateShape::Flat)
    }

    /// RNG baru. `seed` deterministik → resume identik.
    pub fn new(seed: u64, levels: &[usize], mean_rate: f32, shape: RateShape) -> Self {
        let rates = level_rates(levels, mean_rate, shape);
        let last_rates = vec![0.0; rates.len()];
        Self {
            mean_rate: mean_rate.clamp(0.0, 1.0),
            shape,
            rates,
            last_rates,
            rng: Rng::new(seed ^ 0x00C0_11A5),
        }
    }

    /// Laju per level.
    pub fn rates(&self) -> &[f32] {
        &self.rates
    }

    /// Laju sebenarnya batch terakhir per level.
    pub fn last_rates(&self) -> &[f32] {
        &self.last_rates
    }

    /// State RNG (untuk checkpoint).
    pub fn rng(&self) -> &Rng {
        &self.rng
    }

    /// Pulihkan state RNG (resume).
    pub fn set_rng(&mut self, rng: Rng) {
        self.rng = rng;
    }

    /// Bangun mask + kode pengganti untuk level `level` (`batch × ℓ_level`).
    pub fn next_batch(&mut self, level: usize, batch: usize, len: usize, codes: usize) -> CorruptBatch {
        let rate = self.rates.get(level).copied().unwrap_or(0.0);
        self.next_batch_rate(level, batch, len, codes, rate)
    }

    /// Seperti [`Self::next_batch`] tetapi dengan laju eksplisit (ablasi).
    pub fn next_batch_rate(
        &mut self,
        level: usize,
        batch: usize,
        len: usize,
        codes: usize,
        rate: f32,
    ) -> CorruptBatch {
        let positions = batch.saturating_mul(len);
        let count = ((rate.clamp(0.0, 1.0) * positions as f32).round() as usize).min(positions);
        let mut mask = vec![false; positions];
        let mut code = vec![0i32; positions];
        let picks = self.rng.sample_positions(positions, count);
        for p in picks {
            mask[p] = true;
            code[p] = self.rng.below(codes.max(1)) as i32;
        }
        if let Some(slot) = self.last_rates.get_mut(level) {
            *slot = if positions == 0 {
                0.0
            } else {
                count as f32 / positions as f32
            };
        }
        CorruptBatch {
            mask,
            codes: code,
            corrupted: count,
            positions,
        }
    }

    /// Terapkan korupsi ke indeks kode `[B, ℓ]`: posisi bermask diganti kode
    /// pengganti. Dipakai di jalur data sebelum forward generator.
    pub fn apply(&self, idx: &Tensor<2, Int>, c: &CorruptBatch) -> Tensor<2, Int> {
        let [b, l] = idx.dims();
        if c.mask.len() != b * l {
            return idx.clone();
        }
        let device = idx.device();
        let mask = Tensor::<2, Bool>::from_data(
            burn::tensor::TensorData::new(c.mask.clone(), [b, l]),
            &device,
        );
        let codes = Tensor::<2, Int>::from_data(
            burn::tensor::TensorData::new(c.codes.clone(), [b, l]),
            &device,
        );
        idx.clone().mask_where(mask, codes)
    }

    /// Terapkan korupsi ke beberapa level sekaligus (indeks per level).
    pub fn apply_levels(&self, idx: &[Tensor<2, Int>], batches: &[CorruptBatch]) -> Vec<Tensor<2, Int>> {
        idx.iter()
            .zip(batches.iter())
            .map(|(t, c)| self.apply(t, c))
            .collect()
    }

    /// Rata-rata laju sebenarnya seluruh level (untuk log/tes).
    pub fn mean_measured(&self) -> f32 {
        if self.last_rates.is_empty() {
            return 0.0;
        }
        self.last_rates.iter().sum::<f32>() / self.last_rates.len() as f32
    }
}

/// Uji asap korupsi: jumlah terkorupsi tepat dan rata-rata mendekati target.
pub fn corruption_smoke() -> Result<(), vir_common::VirError> {
    use vir_common::VirError;
    let levels = vec![8usize, 16, 32, 64, 128];
    let rates = level_rates(&levels, 0.5, RateShape::LinearDecay);
    let mean: f32 = rates.iter().sum::<f32>() / rates.len() as f32;
    if (mean - 0.5).abs() > 1e-5 {
        return Err(VirError::Internal(format!(
            "rata-rata laju {mean} != 0,5"
        )));
    }
    if rates[0] <= rates[4] {
        return Err(VirError::Internal("laju tidak turun terhadap level".into()));
    }
    let mut c = Corruption::new(3, &levels, 0.5, RateShape::LinearDecay);
    let mut acc = 0.0f32;
    for (k, &l) in levels.iter().enumerate() {
        let batch = c.next_batch(k, 4, l, 16);
        let expect = (rates[k] * (4 * l) as f32).round() as usize;
        if batch.corrupted != expect {
            return Err(VirError::Internal("jumlah terkorupsi tidak tepat".into()));
        }
        if batch.mask.len() != 4 * l {
            return Err(VirError::Internal("panjang mask salah".into()));
        }
        acc += batch.measured_rate();
    }
    if (acc / levels.len() as f32 - 0.5).abs() > 0.05 {
        return Err(VirError::Internal("laju sebenarnya jauh dari target".into()));
    }
    let clean = Corruption::none(5, &levels).next_batch(0, 2, 8, 16);
    if clean.corrupted != 0 {
        return Err(VirError::Internal("laju 0 seharusnya tanpa korupsi".into()));
    }
    Ok(())
}