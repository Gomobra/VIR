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

/// Keuntungan GRPO: (r_i - mean) / std, dengan epsilon untuk std=0.
pub fn grpo_advantage(rewards: &[f32], eps: f32) -> Vec<f32> {
    if rewards.is_empty() {
        return vec![];
    }
    let n = rewards.len() as f32;
    let mean: f32 = rewards.iter().sum::<f32>() / n;
    let var: f32 = rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / n;
    let std = (var + eps).sqrt();
    rewards.iter().map(|&r| (r - mean) / std).collect()
}

pub fn train_smoke() -> Result<(), VirError> {
    let levels = vec![1usize, 13, 24, 36];
    let w = level_weight(&levels, 0.0);
    // α=0: w_k = 1/N, sehingga Σ_k w_k·ℓ_k = 1
    let weighted: f32 = w.iter().zip(levels.iter()).map(|(a, b)| a * (*b as f32)).sum();
    assert!((weighted - 1.0).abs() < 1e-5);
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
    #[test]
    fn grpo_advantage_basic() {
        let r = grpo_advantage(&[1.0, 1.0, 0.0, 0.0], 1e-8);
        // mean=0.5; std populasi ~0.5 -> keuntungan ±1
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[2] + 1.0).abs() < 1e-5);
    }
}
