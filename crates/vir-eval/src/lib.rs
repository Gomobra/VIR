//! Evaluasi: pass@k, rekonstruksi, RULER-style konteks panjang (F5.4),
//! metrik generatif Gen. PPL + entropi unigram + ROUGE/BLEU (F4.5),
//! serta suite bench bertabel (F4.5/F5.4).

mod bench;
mod metrics;
mod ruler;

pub use bench::{
    DEFAULT_LENGTHS, DEFAULT_THRESHOLD, EvalConfig, LengthScore, MetricTable, RulerReport,
    Weighting, fmt_num, pass_at_k_curve, run_ruler_suite,
};
pub use metrics::{TokenDist, bleu, generative_ppl, rouge_l, rouge_n, unigram_entropy};
pub use ruler::{
    RulerCategory, RulerSample, effective_context_length, gen_needle, gen_ruler, gen_ruler_needle,
    match_score,
};

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

/// Rekonstruksi eksak: persentase posisi yang cocok, di bagi panjang minimum
/// (posisi yang tidak ada di kedua sisi tidak dihitung).
pub fn recon_accuracy(original: &[u32], reconstructed: &[u32]) -> f64 {
    let n = original.len().min(reconstructed.len());
    if n == 0 {
        return 0.0;
    }
    let mut ok = 0usize;
    for i in 0..n {
        if original[i] == reconstructed[i] {
            ok += 1;
        }
    }
    ok as f64 / n as f64
}
