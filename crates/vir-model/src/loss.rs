//! Loss R6 dan masukan teacher forcing (PRD F1.6 langkah 3, F1.7).
//!
//! `L_NCM = −Σ_k w_k Σ_i log p_θ(r_k^i | z̃^(k−1))`. CE dihitung per level dari
//! `CrossEntropyLoss` Burn (rerata atas `B·ℓ_k`); bobot level `w_k` datang dari
//! pemanggil (F1.7: `ℓ_k^{1−α} / Σ_j ℓ_j^{1−α}`).

use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::Device;

/// CE per level, sudah dikali bobot (untuk log dan gerbang).
pub fn level_losses(
    logits: &[Tensor<3>],
    targets: &[Tensor<2, Int>],
    weights: &[f32],
) -> Vec<Tensor<1>> {
    let n = logits.len().min(targets.len()).min(weights.len());
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let [b, l, v] = logits[k].dims();
        let ce = CrossEntropyLossConfig::new()
            .init(&logits[k].device())
            .forward(
                logits[k].clone().reshape([b * l, v]),
                targets[k].clone().reshape([b * l]),
            );
        out.push(ce * weights[k]);
    }
    out
}

/// Loss R6 total: `Σ_k w_k · CE_k` (nol bila daftar kosong).
pub fn ncm_loss(logits: &[Tensor<3>], targets: &[Tensor<2, Int>], weights: &[f32]) -> Tensor<1> {
    let mut total: Option<Tensor<1>> = None;
    for term in level_losses(logits, targets, weights) {
        total = Some(match total {
            Some(acc) => acc + term,
            None => term,
        });
    }
    total.unwrap_or_else(|| Tensor::zeros([1], &Device::default()))
}

/// Masukan teacher forcing (F1.6 langkah 3): level k melihat `ẑ^(k−1)`, jadi
/// daftar rekonstruksi kumulatif digeser satu dengan `ẑ^(0) = 0` di depan.
pub fn teacher_inputs(
    z_hat: &[Tensor<3>],
    batch: usize,
    len: usize,
    dim: usize,
    device: &Device,
) -> Vec<Tensor<3>> {
    let k = z_hat.len();
    let mut out = Vec::with_capacity(k);
    out.push(Tensor::<3>::zeros([batch, len, dim], device));
    for z in z_hat.iter().take(k.saturating_sub(1)) {
        out.push(z.clone());
    }
    out
}
