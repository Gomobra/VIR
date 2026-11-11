//! Latihan (skeleton + loss berbobot level).

use vir_common::VirError;

/// Bobot level: w_k = ℓ_k^{-α} / Σ_j ℓ_j^{1-α}.
pub fn level_weight(levels: &[usize], alpha: f32) -> Vec<f32> {
    let mut ws: Vec<f32> = levels.iter().map(|&l| (l as f32).powf(-alpha)).collect();
    // normalisasi: w_k = ℓ_k^{-α} / Σ_j ℓ_j^{1-α}
    let denom: f32 = levels.iter().map(|&l| (l as f32).powf(1.0 - alpha)).sum();
    if denom > 0.0 {
        for (i, &l) in levels.iter().enumerate() {
            ws[i] = (l as f32).powf(-alpha) / denom;
        }
    }
    ws
}

/// pass@k estimator tak bias (Chen et al.): 1 - prod_{i=n-c+1}^{n}(1 - k/i).
pub fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if n == 0 {
        return 0.0;
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

pub fn train_smoke() -> Result<(), VirError> {
    let levels = vec![1usize, 13, 24, 36];
    let w = level_weight(&levels, 0.0);
    // α=0: tiap kode berbobot 1/N
    let sum: f32 = w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // pass@k sanity
    assert!((pass_at_k(4, 1, 2) - 0.5).abs() < 1e-9);
    assert!((pass_at_k(10, 3, 5) - 0.9166666667).abs() < 1e-6);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn train_smoke_ok() {
        assert!(train_smoke().is_ok());
    }
    #[test]
    fn pass_at_k_k1() {
        // k=1 menyusut menjadi c/n
        assert!((pass_at_k(5, 2, 1) - 0.4).abs() < 1e-9);
    }
}
