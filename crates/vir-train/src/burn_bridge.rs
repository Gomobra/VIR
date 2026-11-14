//! Jembatan ke optimizer Burn 0.22: Muon untuk matriks blok + AdamW untuk
//! embedding/norm/head (PRD F1.7 langkah 4 dan task "optimizer Muon + AdamW
//! hibrid").
//!
//! Dua jalur disediakan dan sengaja dipisah:
//!
//! 1. [`crate::optim`] — Muon/AdamW **manual** di atas `ParamStore` (buffer
//!    `f32`), dipakai trainer dan diuji numerik tanpa Burn. Jalur ini yang
//!    dipakai [`crate::trainer::Trainer`] karena checkpoint/resume-nya byte demi
//!    byte.
//! 2. Modul ini — [`ModuleOptimizer`] Burn dengan grup parameter: AdamW untuk
//!    semua parameter, Muon untuk parameter yang path-nya memuat predikat blok
//!    tersembunyi. Dipakai pemanggil yang sudah memegang model Burn (skrip
//!    Kaggle), sehingga `model = opt.step(lr, model, grads)` tetap berlaku.
//!
//! Skala Muon memakai `AdjustLrFn::MatchRmsAdamW` (`η·0,2·√max(A,B)`) agar LR
//! dan weight decay AdamW dapat dipakai ulang, sesuai PRD langkah 4.

use burn::module::ParamGroup;
use burn::optim::{AdamWConfig, AdjustLrFn, ModuleOptimizer, MuonConfig};

/// Setelan optimizer Burn.
#[derive(Debug, Clone, PartialEq)]
pub struct BurnHybridSettings {
    /// Predikat path parameter yang memakai Muon (mis. `core`, `block`).
    pub muon_predicate: String,
    /// β₁ AdamW.
    pub beta1: f32,
    /// β₂ AdamW.
    pub beta2: f32,
    /// Epsilon AdamW.
    pub eps: f32,
    /// Weight decay (dipakai kedua grup).
    pub weight_decay: f32,
    /// Iterasi Newton–Schulz Muon.
    pub ns_steps: usize,
}

impl Default for BurnHybridSettings {
    fn default() -> Self {
        Self {
            muon_predicate: "core".to_string(),
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            ns_steps: 5,
        }
    }
}

/// Bangun optimizer Burn hibrid.
///
/// Urutan grup penting: grup terakhir yang cocok menang, jadi AdamW dipasang
/// lebih dulu (pola `ParamGroup::all()` bawaan [`ModuleOptimizer`]) lalu Muon
/// untuk predikat blok tersembunyi. Pemanggil memakai
/// `model = opt.step(lr, model, GradientsParams::from_grads(loss.backward(), &model))`.
pub fn burn_optimizer(cfg: &BurnHybridSettings) -> ModuleOptimizer {
    let adamw = AdamWConfig::new()
        .with_beta_1(cfg.beta1)
        .with_beta_2(cfg.beta2)
        .with_epsilon(cfg.eps)
        .with_weight_decay(cfg.weight_decay)
        .init();
    let muon = MuonConfig::new()
        .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
        .with_ns_steps(cfg.ns_steps)
        .build();
    adamw.with_group(
        ParamGroup::from_predicate(cfg.muon_predicate.clone()),
        muon,
        None,
    )
}
