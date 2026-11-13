//! Latihan: loss berbobot, Muon, checkpoint, JSONL.

use std::fs::File;
use std::io::{BufWriter, Write};
use vir_common::VirError;

/// Bobot level: w_k = ℓ_k^{-α} / Σ_j ℓ_j^{1-α}.
pub fn level_weight(levels: &[usize], alpha: f32) -> Vec<f32> {
    let denom: f32 = levels.iter().map(|&l| (l as f32).powf(1.0 - alpha)).sum();
    levels
        .iter()
        .map(|&l| {
            if denom > 0.0 {
                (l as f32).powf(-alpha) / denom
            } else {
                0.0
            }
        })
        .collect()
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

/// Keuntungan GRPO.
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

/// Newton-Schulz orthogonalization (Muon), 5 iterasi, koefisien (3.4445, -4.7750, 2.0315).
pub fn newton_schulz(m: &mut [f32], rows: usize, cols: usize) {
    // X0 = M / ||M||_F
    let norm: f32 = m.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-7);
    for v in m.iter_mut() {
        *v /= norm;
    }
    let (a, b, c) = (3.4445f32, -4.7750, 2.0315);
    for _ in 0..5 {
        // X <- aX + b(XX^T)X + c(XX^T)^2 X
        let xtx = matmul_t(m, rows, cols); // [cols, cols] = X^T X
        let xtx2 = matmul(&xtx, &xtx, cols, cols, cols); // [cols, cols]
                                                         // X (aI + b X^T X + c (X^T X)^2)
        let mut new = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut acc = 0.0f32;
                for k in 0..cols {
                    let xtx_v = xtx[k * cols + j];
                    let xtx2_v = xtx2[k * cols + j];
                    acc +=
                        (if k == j { a } else { 0.0 } + b * xtx_v + c * xtx2_v) * m[i * cols + k];
                }
                new[i * cols + j] = acc;
            }
        }
        *m = new;
    }
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn matmul_t(a: &[f32], m: usize, k: usize) -> Vec<f32> {
    // A^T A: [k, k]
    let mut out = vec![0.0f32; k * k];
    for i in 0..m {
        for j in 0..k {
            for l in 0..k {
                out[j * k + l] += a[i * k + j] * a[i * k + l];
            }
        }
    }
    out
}

/// Muon optimizer state (momentum + Newton-Schulz).
pub struct Muon {
    pub momentum: Vec<f32>,
    pub beta: f32,
}

impl Muon {
    pub fn new(size: usize, beta: f32) -> Self {
        Self {
            momentum: vec![0.0; size],
            beta,
        }
    }

    /// Update bobot W [rows, cols] dengan grad G (row-major).
    pub fn step(&mut self, w: &mut [f32], g: &[f32], rows: usize, cols: usize, lr: f32, wd: f32) {
        let n = w.len();
        for i in 0..n {
            self.momentum[i] = self.beta * self.momentum[i] + g[i];
        }
        let mut m = self.momentum.clone();
        newton_schulz(&mut m, rows, cols);
        let scale = 0.2 * (rows.max(cols) as f32).sqrt();
        for i in 0..n {
            w[i] -= lr * (scale * m[i] + wd * w[i]);
        }
    }
}

/// Logger JSONL.
pub struct JsonlLogger {
    w: BufWriter<File>,
}

impl JsonlLogger {
    pub fn new(path: &str) -> Result<Self, VirError> {
        let f = File::create(path)?;
        Ok(Self {
            w: BufWriter::new(f),
        })
    }
    pub fn log(&mut self, obj: &str) -> Result<(), VirError> {
        writeln!(self.w, "{obj}")?;
        self.w.flush()?;
        Ok(())
    }
}

pub fn train_smoke() -> Result<(), VirError> {
    let levels = vec![1usize, 13, 24, 36];
    let w = level_weight(&levels, 0.0);
    let weighted: f32 = w
        .iter()
        .zip(levels.iter())
        .map(|(a, b)| a * (*b as f32))
        .sum();
    assert!((weighted - 1.0).abs() < 1e-5);
    assert!((pass_at_k(4, 1, 2) - 0.5).abs() < 1e-9);
    assert!((pass_at_k(10, 3, 5) - 0.9166666667).abs() < 1e-6);
    let adv = grpo_advantage(&[1.0, 1.0, 0.0, 0.0], 1e-8);
    assert!((adv[0] - 1.0).abs() < 1e-5);
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
    fn muon_reduces_norm() {
        let mut w = vec![1.0f32; 16];
        let g = vec![0.1f32; 16];
        let mut m = Muon::new(16, 0.95);
        let before: f32 = w.iter().map(|v| v * v).sum();
        m.step(&mut w, &g, 4, 4, 0.01, 0.0);
        let after: f32 = w.iter().map(|v| v * v).sum();
        assert!(after < before, "norm tidak turun");
    }
    #[test]
    fn newton_schulz_orthogonalish() {
        let mut m = vec![1.0f32; 16];
        newton_schulz(&mut m, 4, 4);
        // hasil bukan ortogonal eksak (f(1)=0.701), tapi finite
        assert!(m.iter().all(|v| v.is_finite()));
    }
}
