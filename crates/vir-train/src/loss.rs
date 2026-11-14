//! Loss berbobot level (R6) dan penolong CE yang mengabaikan padding
//! (PRD F1.7 langkah 1: `L_NCM = −Σ_k w_k Σ_i log p_θ(r_k^i | z̃^(k−1))`).
//!
//! Bobot: `w_k = ℓ_k^{−α} / Σ_j ℓ_j^{1−α}`. α=0 → bobot 1/N per kode
//! (setara maksimisasi ELBO, default NCM); α=1 → tiap level sama besar.
//! α=0,5 memberi 19% porsi loss ke level kasar, α=1,5 memberi 63%.
//!
//! CE dihitung dari `log_softmax` + `gather` (bukan `CrossEntropyLoss`) karena
//! loss per level butuh **rerata hanya pada posisi valid**: padding di akhir
//! baris tidak boleh masuk penyebut.

use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::Device;
use burn::tensor::activation::log_softmax;

/// Bobot level ternormalisasi: `w_k = ℓ_k^{1−α} / Σ_j ℓ_j^{1−α}`.
///
/// Bentuk PRD F1.7 langkah 1 adalah bobot **per kode**
/// `ℓ_k^{−α} / Σ_j ℓ_j^{1−α}` yang mengalikan jumlah (bukan rata-rata) CE per
/// level; karena di sini CE level dihitung sebagai rata-rata, bobot setara
/// adalah `ℓ_k` kali bentuk PRD, yaitu rumus di atas. Akibatnya bobot
/// berjumlah satu dan:
///
/// - α=0 → `w_k = ℓ_k / Σℓ_j` (tiap kode berbobot `1/N`, setara ELBO);
/// - α=1 → `w_k = 1/K` (tiap level sama besar).
pub fn level_weight(levels: &[usize], alpha: f32) -> Vec<f32> {
    let denom: f32 = levels.iter().map(|&l| (l as f32).powf(1.0 - alpha)).sum();
    levels
        .iter()
        .map(|&l| {
            if denom > 0.0 {
                (l as f32).powf(1.0 - alpha) / denom
            } else {
                0.0
            }
        })
        .collect()
}

/// Bobot per kode (bentuk PRD apa adanya): `ℓ_k^{−α} / Σ_j ℓ_j^{1−α}`.
pub fn code_weight(levels: &[usize], alpha: f32) -> Vec<f32> {
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

/// NLL per posisi `-log p(target)`: `[B, L, V]` → `[B*L]`.
pub fn nll_per_position(logits: Tensor<3>, target: Tensor<2, Int>) -> Tensor<1> {
    let [b, l, v] = logits.dims();
    let n = b * l;
    let logp = log_softmax(logits, 2).reshape([n, v]);
    let idx = target.reshape([n, 1]);
    logp.gather(1, idx).reshape([n]).neg()
}

/// CE rata-rata pada posisi valid. `mask` `[B, L]` bernilai 1,0 untuk token
/// nyata dan 0 untuk padding; `None` berarti semua posisi dipakai.
pub fn masked_level_loss(
    logits: Tensor<3>,
    target: Tensor<2, Int>,
    mask: Option<&Tensor<2>>,
) -> Tensor<1> {
    let nll = nll_per_position(logits, target);
    match mask {
        None => nll.mean(),
        Some(m) => {
            let [b, l] = m.dims();
            let w = m.clone().reshape([b * l]);
            nll.mul(w.clone()).sum() / w.sum().clamp_min(1.0)
        }
    }
}

/// CE rata-rata biasa (tanpa mask) — padanan `vir_model::ncm_loss`.
pub fn ncm_loss(logits: &[Tensor<3>], targets: &[Tensor<2, Int>], weights: &[f32]) -> Tensor<1> {
    let terms = level_loss_terms(logits, targets, weights, None);
    sum_terms(&terms)
}

/// Loss per level sudah dikali bobot: `w_k · CE_k` (mask opsional per level).
pub fn level_loss_terms(
    logits: &[Tensor<3>],
    targets: &[Tensor<2, Int>],
    weights: &[f32],
    masks: Option<&[Tensor<2>]>,
) -> Vec<Tensor<1>> {
    let n = logits.len().min(targets.len()).min(weights.len());
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mask = masks.and_then(|m| m.get(k));
        let ce = masked_level_loss(logits[k].clone(), targets[k].clone(), mask);
        out.push(ce * weights[k]);
    }
    out
}

/// Jumlah suku loss. Kosong → nol (dipakai saat belum ada level aktif).
pub fn sum_terms(terms: &[Tensor<1>]) -> Tensor<1> {
    let mut acc: Option<Tensor<1>> = None;
    for t in terms {
        acc = Some(match acc {
            Some(a) => a + t.clone(),
            None => t.clone(),
        });
    }
    acc.unwrap_or_else(|| Tensor::zeros([1], &Device::default()))
}

/// Rincian loss satu langkah: total (tensor) dan nilai per level (host).
#[derive(Debug, Clone)]
pub struct LossBreakdown {
    /// Loss total yang sudah dibawa ke backward.
    pub total: Tensor<1>,
    /// Suku per level, sudah berbobot.
    pub terms: Vec<Tensor<1>>,
    /// Jumlah token valid per level (untuk log dan normalisasi).
    pub tokens: Vec<f32>,
}

impl LossBreakdown {
    /// Susun dari suku per level dan jumlah token valid.
    pub fn new(terms: Vec<Tensor<1>>, tokens: Vec<f32>) -> Self {
        let total = sum_terms(&terms);
        Self {
            total,
            terms,
            tokens,
        }
    }

    /// Nilai suku per level sebagai `f32` (host).
    pub fn values(&self) -> Vec<f32> {
        self.terms.iter().map(|t| t.clone().into_scalar::<f32>()).collect()
    }

    /// Nilai total sebagai `f32` (host).
    pub fn total_value(&self) -> f32 {
        self.total.clone().into_scalar::<f32>()
    }

    /// Jumlah token valid seluruh level.
    pub fn total_tokens(&self) -> f32 {
        self.tokens.iter().sum()
    }
}

/// SFT: prefix-LM loss hanya pada target (mask prefix).
/// `logits` [B, T, V], `target` [B, T], `prefix_len` panjang prefix yang sama.
pub fn sft_loss(
    logits: Tensor<3>,
    target: Tensor<2, Int>,
    prefix_len: usize,
    device: &Device,
) -> Tensor<1> {
    let [b, t, v] = logits.dims();
    if prefix_len >= t {
        return Tensor::zeros([1], device);
    }
    let logits = logits.slice([0..b, prefix_len..t, 0..v]);
    let target = target.slice([0..b, prefix_len..t]);
    let [b2, t2, v2] = logits.dims();
    let lg = logits.reshape([b2 * t2, v2]);
    let tg = target.reshape([b2 * t2]);
    CrossEntropyLossConfig::new().init(device).forward(lg, tg)
}

/// pass@k estimator tak bias (Chen et al.): `1 - prod_{i=n-c+1}^{n}(1 - k/i)`.
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

/// Keuntungan GRPO (z-score berkelompok).
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

/// GRPO objective: PPO-clip + penalti KL (tanpa critic).
pub fn grpo_objective(logp: f32, ref_logp: f32, adv: f32, clip: f32, kl_coef: f32) -> f32 {
    let ratio = (logp - ref_logp).exp();
    let clipped = ratio.clamp(1.0 - clip, 1.0 + clip);
    let policy = ratio * adv;
    let policy_clipped = clipped * adv;
    let kl = logp - ref_logp;
    policy.min(policy_clipped) - kl_coef * kl
}

/// Uji asap loss: bobot berjumlah satu dan α=1 memberi bobot sama per level.
pub fn loss_smoke() -> Result<(), vir_common::VirError> {
    use vir_common::VirError;
    let levels = vec![1usize, 13, 24, 36];
    let w0 = level_weight(&levels, 0.0);
    let sum0: f32 = w0.iter().sum();
    if (sum0 - 1.0).abs() > 1e-5 {
        return Err(VirError::Internal("bobot α=0 tidak berjumlah satu".into()));
    }
    let w1 = level_weight(&levels, 1.0);
    for w in &w1 {
        if (*w - 0.25).abs() > 1e-5 {
            return Err(VirError::Internal("bobot α=1 tidak seragam".into()));
        }
    }
    let w15 = level_weight(&levels, 1.5);
    if w15[0] <= w0[0] {
        return Err(VirError::Internal(
            "α=1,5 harus menaikkan porsi level terpendek (kasar)".into(),
        ));
    }
    if w0[0] >= w0[3] {
        return Err(VirError::Internal(
            "α=0 harus memberi bobot lebih besar ke level terpanjang".into(),
        ));
    }
    let cw = code_weight(&levels, 0.0);
    let code_sum: f32 = cw.iter().sum();
    if (code_sum - 4.0 / 74.0).abs() > 1e-6 {
        return Err(VirError::Internal(format!(
            "bobot per kode α=0 {code_sum} != 4/74"
        )));
    }
    if (pass_at_k(4, 1, 2) - 0.5).abs() > 1e-9 {
        return Err(VirError::Internal("pass@k salah".into()));
    }
    let adv = grpo_advantage(&[1.0, 1.0, 0.0, 0.0], 1e-8);
    if (adv[0] - 1.0).abs() > 1e-5 {
        return Err(VirError::Internal("grpo_advantage salah".into()));
    }
    Ok(())
}