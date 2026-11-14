//! `vir.eval`: metrik evaluasi murni (PRD F4.2/F5.4).
//!
//! Cermin dari crate `vir-eval` (workspace terpisah yang tidak dapat menjadi
//! dependensi build inti). Hanya metrik tanpa tensor yang terpapar di sini.

use std::collections::BTreeMap;

use pyo3::prelude::*;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pass_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(recon_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(unigram_entropy, m)?)?;
    Ok(())
}

/// Estimator pass@k tak bias (Chen et al. 2021): `1 − C(n−c, k)/C(n, k)`.
#[pyfunction]
fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let c = c.min(n);
    if n - c < k {
        return 1.0;
    }
    let mut prod = 1.0f64;
    for i in (n - c + 1)..=n {
        prod *= 1.0 - (k as f64) / (i as f64);
    }
    1.0 - prod
}

/// Akurasi rekonstruksi: posisi yang cocok terhadap panjang asli.
#[pyfunction]
fn recon_accuracy(original: Vec<u32>, reconstructed: Vec<u32>) -> f64 {
    if original.is_empty() {
        return 0.0;
    }
    let n = original.len().min(reconstructed.len());
    let ok = (0..n).filter(|&i| original[i] == reconstructed[i]).count();
    ok as f64 / original.len() as f64
}

/// Entropi unigram (bit) dari distribusi empiris token.
#[pyfunction]
fn unigram_entropy(tokens: Vec<u32>) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for token in &tokens {
        *counts.entry(*token).or_insert(0usize) += 1;
    }
    let total = tokens.len() as f64;
    let mut bits = 0.0;
    for count in counts.values() {
        let p = *count as f64 / total;
        bits -= p * p.log2();
    }
    bits
}
