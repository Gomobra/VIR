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

