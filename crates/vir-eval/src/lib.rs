//! Evaluasi: pass@k, rekonstruksi, RULER-style konteks panjang (F5.4),
//! dan metrik generatif Gen. PPL + entropi unigram (F4.5).

mod metrics;
mod ruler;

pub use metrics::{TokenDist, generative_ppl, unigram_entropy};
pub use ruler::{RulerCategory, RulerSample, gen_needle, gen_ruler, gen_ruler_needle};

// ---------------------------------------------------------------------------
// F4.2 / F4.5 — estimator pass@k tak bias
// ---------------------------------------------------------------------------

/// Estimator pass@k tak bias (Chen et al. 2021):
/// `1 - C(n-c, k) / C(n, k)`, dengan `c` benar dari `n` sampel.
pub fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    let c = c.min(n);
    if k == 0 {
        return 1.0;
    }
    if n - c < k {
        return 1.0;
    }
    let mut prod = 1.0f64;
    for i in (n - c + 1)..=n {
        prod *= 1.0 - (k as f64) / (i as f64);
    }
    1.0 - prod
}

pub fn eval_pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    pass_at_k(n, c, k)
}

/// Rekonstruksi eksak: persentase posisi yang cocok.
pub fn recon_accuracy(original: &[u32], reconstructed: &[u32]) -> f64 {
    if original.is_empty() {
        return 0.0;
    }
    let n = original.len().min(reconstructed.len());
    let mut ok = 0usize;
    for i in 0..n {
        if original[i] == reconstructed[i] {
            ok += 1;
        }
    }
    ok as f64 / original.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_at_k_delegates() {
        assert!((eval_pass_at_k(4, 1, 2) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn pass_at_k_hand_values() {
        // 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7
        assert!((pass_at_k(5, 2, 2) - 0.7).abs() < 1e-12);
        // semua benar -> 1
        assert!((pass_at_k(5, 5, 2) - 1.0).abs() < 1e-12);
        // tidak ada benar -> 0
        assert!((pass_at_k(5, 0, 1) - 0.0).abs() < 1e-12);
        // k > n -> 1 (semua kemungkinan tercakup)
        assert!((pass_at_k(3, 0, 5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn recon_accuracy_full() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        assert!((recon_accuracy(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn recon_accuracy_partial() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 9, 3, 4];
        assert!((recon_accuracy(&a, &b) - 0.75).abs() < 1e-9);
    }
}
