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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(11)
    }

    #[test]
    fn cutoff_stays_in_range() {
        let mut rng = rng();
        for _ in 0..200 {
            let k = nested_dropout_cutoff(12, 0.5, &mut rng);
            assert!((1..=12).contains(&k));
        }
    }

    /// `prob = 0` menjatuhkan hierarki penuh; `prob = 1` selalu memotong.
    #[test]
    fn cutoff_extremes_are_deterministic() {
        let mut rng = rng();
        assert_eq!(sample_cutoff(12, 0.0, &mut rng), 12);
        for _ in 0..50 {
            assert!(sample_cutoff(12, 1.0, &mut rng) < 12);
        }
        assert_eq!(sample_cutoff(1, 1.0, &mut rng), 1);
    }

    /// Distribusi cutoff harus berjumlah satu dan menaruh sisa massa di `K`.
    #[test]
    fn distribution_sums_to_one() {
        for prob in ABLATION_PROBS {
            let dist = cutoff_distribution(prob, 12);
            let total: f64 = dist.iter().sum();
            assert!((total - 1.0).abs() < 1e-9, "prob={prob} total={total}");
            assert!((dist[12] - (1.0 - f64::from(prob))).abs() < 1e-9);
        }
        let dist = cutoff_distribution(0.5, 1);
        assert!((dist[1] - 1.0).abs() < 1e-9);
    }

    /// Frekuensi empiris sampler harus mendekati distribusi teoretis.
    #[test]
    fn sampler_matches_distribution() {
        let mut rng = rng();
        let levels = 4;
        let prob = 0.5f32;
        let draws = 4000;
        let mut hits = vec![0usize; levels + 1];
        for _ in 0..draws {
            hits[sample_cutoff(levels, prob, &mut rng)] += 1;
        }
        let dist = cutoff_distribution(prob, levels);
        for k in 1..=levels {
            let expected = dist[k] * draws as f64;
            let got = hits[k] as f64;
            assert!(
                (got - expected).abs() < 6.0 * expected.sqrt() + 3.0,
                "k={k} got={got} expected={expected}"
            );
        }
    }

    /// Masking harus menghilangkan kontribusi level >= cutoff saja.
    #[test]
    fn mask_zeroes_tail_levels() {
        let device = Device::default();
        let levels: Vec<Tensor<3>> = (0..3)
            .map(|k| Tensor::<3>::full([1, 2, 1], (k + 1) as f32, &device))
            .collect();
        let masked = mask_reconstruction(&levels, 2);
        let keep: Vec<f32> = masked[0]
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        let drop: Vec<f32> = masked[2]
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        assert!(keep.iter().all(|v| *v == 1.0));
        assert!(drop.iter().all(|v| *v == 0.0));

        let summed = sum_levels(&masked).unwrap_or_else(|| Tensor::zeros([1, 2, 1], &device));
        let vals: Vec<f32> = summed.into_data().try_to_vec::<f32>().unwrap_or_default();
        assert!(vals.iter().all(|v| *v == 3.0), "got {vals:?}");
    }

    /// Rekonstruksi kumulatif = jumlah berjalan.
    #[test]
    fn cumulative_sums_are_running_totals() {
        let device = Device::default();
        let levels: Vec<Tensor<3>> = (0..3)
            .map(|k| Tensor::<3>::full([1, 1, 1], (k + 1) as f32, &device))
            .collect();
        let sums = cumulative_sums(&levels);
        let last: Vec<f32> = sums[2]
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        assert_eq!(last, vec![6.0]);
        assert!(sum_levels(&[]).is_none());
    }
}