//! Sampling level demi level (PRD F1.6 task 3: temperature, top-k, deterministik
//! dengan seed) dan sampling lanjutan (CFG, MBR).
//!
//! Sampling dikerjakan di host dengan RNG ber-seed (`vir_common::DeterministicRng`),
//! bukan dengan RNG device: hasilnya identik pada backend apa pun dan bisa
//! dihitung tangan pada vocab 2 kode (PRD F1.6 langkah 6). Biayanya satu
//! perpindahan host `B·ℓ_k·|V_k|` per level; kernel terfusi menyusul (F7.1).

use burn::prelude::*;
use vir_common::DeterministicRng;

/// Pemilih kode saat generasi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampler {
    /// Suhu logit; `<= 0` berarti greedy (argmax).
    pub temperature: f32,
    /// `top_k` > 0 membatasi sampling ke k logit terbesar.
    pub top_k: usize,
    /// Benih RNG (determinisme, F1.6 langkah 6).
    pub seed: u64,
}

impl Sampler {
    /// Greedy: argmax tiap posisi.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            seed: 0,
        }
    }

    /// Suhu `temperature`, `top_k`, dan benih eksplisit.
    pub fn new(temperature: f32, top_k: usize, seed: u64) -> Self {
        Self {
            temperature,
            top_k,
            seed,
        }
    }

    /// Benih per level: `seed + k·φ` agar derau antar level tidak berkorelasi.
    pub fn for_level(&self, level: usize) -> Self {
        Self {
            seed: self
                .seed
                .wrapping_add((level as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            ..*self
        }
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::greedy()
    }
}

/// Argmax satu baris.
pub fn argmax_row(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_value {
            best_value = v;
            best = i;
        }
    }
    best
}

/// Softmax satu baris dengan suhu; `temperature <= 0` memberi distribusi satu puncak.
pub fn softmax_row(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    if temperature <= 0.0 {
        let mut probs = vec![0.0f32; logits.len()];
        probs[argmax_row(logits)] = 1.0;
        return probs;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&z| ((z - max) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    } else {
        let uniform = 1.0 / logits.len() as f32;
        for p in probs.iter_mut() {
            *p = uniform;
        }
    }
    probs
}

/// `k` indeks dengan nilai terbesar (urut turun); `k = 0` memberi semua indeks.
pub fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if k > 0 {
        idx.truncate(k.min(values.len()));
    }
    idx
}

/// Sampel satu indeks dari distribusi (uniform dengan `rng`).
pub fn sample_index(probs: &[f32], rng: &mut impl rand::Rng) -> usize {
    let total: f32 = probs.iter().sum();
    if probs.is_empty() || total <= 0.0 {
        return 0;
    }
    let u: f32 = rand::Rng::gen_range(&mut *rng, 0.0..1.0) * total;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

/// Sampel kode dari logits `[B, ℓ, |V|]` → `[B, ℓ]` (deterministik pada seed).
///
/// Jatuh ke argmax bila data logits tidak dapat dibaca dari device.
pub fn sample_codes(logits: Tensor<3>, sampler: &Sampler) -> Tensor<2, Int> {
    let [b, l, v] = logits.dims();
    let device = logits.device();
    let values = logits.to_data().try_to_vec::<f32>().unwrap_or_default();
    if values.len() != b * l * v {
        return logits.argmax(2).reshape([b, l]);
    }
    let mut rng = DeterministicRng::new(sampler.seed);
    let mut out: Vec<i64> = Vec::with_capacity(b * l);
    for row in values.chunks_exact(v) {
        out.push(sample_row(row, sampler, &mut rng) as i64);
    }
    Tensor::<2, Int>::from_ints(burn::tensor::TensorData::new(out, [b, l]), &device)
}

/// Sampel satu baris logit.
fn sample_row(row: &[f32], sampler: &Sampler, rng: &mut DeterministicRng) -> usize {
    if sampler.temperature <= 0.0 {
        return argmax_row(row);
    }
    let probs = softmax_row(row, sampler.temperature);
    let keep = if sampler.top_k > 0 {
        top_k_indices(&probs, sampler.top_k)
    } else {
        (0..probs.len()).collect()
    };
    let mut filtered = vec![0.0f32; probs.len()];
    for i in keep {
        filtered[i] = probs[i];
    }
    sample_index(&filtered, rng.inner_mut())
}

/// CFG pada logit: `ℓ_uc + β(ℓ − ℓ_uc)` (F1.6 sampling lanjutan).
pub fn cfg_logit(uc: f32, c: f32, beta: f32) -> f32 {
    uc + beta * (c - uc)
}

/// MBR: pilih kandidat dengan kemiripan rata-rata tertinggi ke kandidat lain.
pub fn mbr_select<F>(candidates: &[Vec<u32>], sim: F) -> usize
where
    F: Fn(&[u32], &[u32]) -> f32,
{
    let k = candidates.len();
    if k <= 1 {
        return 0;
    }
    let mut best = 0usize;
    let mut best_score = f32::MIN;
    for i in 0..k {
        let mut acc = 0.0f32;
        for j in 0..k {
            if i == j {
                continue;
            }
            acc += sim(&candidates[i], &candidates[j]);
        }
        let score = acc / (k - 1) as f32;
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best
}
