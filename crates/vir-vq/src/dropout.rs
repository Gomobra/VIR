//! Nested dropout (PRD F1.4.c, rumus R3).
//!
//! Saat latihan hanya `k` level pertama yang dipakai (`k` acak), sehingga level
//! awal memikul informasi bermakna dan tokenizer bisa dipakai pada panjang
//! lebih pendek.
//!
//! Distribusi `k` yang dipakai (PRD F1.4.c langkah 2): dengan peluang `prob`
//! level halus dijatuhkan dan `k` diambil seragam dari `1..K-1`; selain itu
//! hierarki penuh (`k = K`) dipakai. Teks NCM tidak menuliskan distribusi
//! persisnya, jadi `prob` adalah parameter [`crate::multiscale`]/
//! [`vir_common::ModelConfig::nested_dropout_prob`].

use burn::prelude::*;
use rand::RngCore;

/// Peluang ablasi nested dropout yang dipinjau pada G1 (PRD F1.4.c langkah 3).
pub const ABLATION_PROBS: [f32; 4] = [0.0, 0.3, 0.5, 0.7];

/// Sampler cutoff: `k ∈ 1..=num_levels` (R3).
///
/// `prob <= 0` selalu memberi hierarki penuh; `prob >= 1` selalu memotong.
pub fn sample_cutoff(num_levels: usize, prob: f32, rng: &mut dyn RngCore) -> usize {
    if num_levels <= 1 {
        return num_levels;
    }
    let p = prob.clamp(0.0, 1.0);
    let draw = rng.next_u32() as f32 / u32::MAX as f32;
    let dropped = p > 0.0 && draw < p;
    if dropped && num_levels > 2 {
        // Seragam pada 1..num_levels-1 (inklusif).
        let span = (num_levels - 1) as u32;
        let pick = (rng.next_u64() % u64::from(span)) as usize;
        pick + 1
    } else if dropped {
        1
    } else {
        num_levels
    }
}

/// Versi generik dari [`sample_cutoff`] untuk pemanggil dengan RNG konkret.
pub fn nested_dropout_cutoff(num_levels: usize, prob: f32, rng: &mut impl RngCore) -> usize {
    sample_cutoff(num_levels, prob, rng)
}

/// Distribusi peluang cutoff `k = 1..=num_levels` (untuk laporan dan tes).
pub fn cutoff_distribution(prob: f32, num_levels: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; num_levels + 1];
    if num_levels == 0 {
        return out;
    }
    let p = prob.clamp(0.0, 1.0) as f64;
    out[num_levels] += 1.0 - p;
    if num_levels == 1 {
        out[1] = 1.0;
        return out;
    }
    let tail = p / (num_levels - 1) as f64;
    for k in 1..num_levels {
        out[k] += tail;
    }
    out
}

/// Nol-kan kontribusi level `>= cutoff` (masking R3).
///
/// `levels` adalah kontribusi `R_j(e_j(r_j))` per level, semuanya `[B, L, d]`.
pub fn mask_reconstruction(levels: &[Tensor<3>], cutoff: usize) -> Vec<Tensor<3>> {
    levels
        .iter()
        .enumerate()
        .map(|(k, z)| {
            if k < cutoff {
                z.clone()
            } else {
                z.zeros_like()
            }
        })
        .collect()
}

/// Jumlah kontribusi level; `None` bila daftar kosong.
pub fn sum_levels(levels: &[Tensor<3>]) -> Option<Tensor<3>> {
    let (first, rest) = levels.split_first()?;
    let mut acc = first.clone();
    for z in rest {
        acc = acc + z.clone();
    }
    Some(acc)
}

/// Rekonstruksi kumulatif `ẑ^(k) = Σ_{j<=k} R_j(e_j(r_j))` untuk semua `k`.
pub fn cumulative_sums(levels: &[Tensor<3>]) -> Vec<Tensor<3>> {
    let mut out = Vec::with_capacity(levels.len());
    let mut acc: Option<Tensor<3>> = None;
    for z in levels {
        acc = Some(match acc {
            Some(prev) => prev + z.clone(),
            None => z.clone(),
        });
        out.push(acc.clone().unwrap_or_else(|| z.zeros_like()));
    }
    out
}
