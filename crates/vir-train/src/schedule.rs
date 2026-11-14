//! Jadwal laju pembelajaran (PRD F1.7 langkah 3 dan task "dijadwal LR
//! (kosinus dan warmup-stable-decay) diablasi").
//!
//! Hiperparameter awal NCM Tabel 6: puncak `3e-4`, lantai `3e-5` (0,1×),
//! warmup 500–1000 langkah. Bentuk penurunan tidak tertulis di teks, jadi di
//! sini ada dua kandidat yang bisa dipilih per konfigurasi:
//!
//! | [`LrKind`] | Setelah warmup |
//! |---|---|
//! | [`LrKind::Cosine`] | kosinus setengah periode dari puncak ke lantai |
//! | [`LrKind::Wsd`] | stabil di puncak, lalu turun linear ke lantai |
//!
//! Semua fungsi memakai `f64` (tipe `LearningRate` di Burn) dan mengembalikan
//! nilai yang sama untuk `step` di luar rentang (dijepit).

use serde::{Deserialize, Serialize};

/// Bentuk kurva penurunan laju pembelajaran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LrKind {
    /// Kosinus setengah periode (default PRD).
    Cosine,
    /// Warmup–stabil–decay: rata dulu, baru turun linear.
    Wsd,
}

/// Jadwal: warmup linear → puncak → penurunan → lantai.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LrSchedule {
    /// Laju puncak (setelah warmup).
    pub peak: f64,
    /// Laju lantai (akhir pelatihan).
    pub floor: f64,
    /// Jumlah langkah warmup linear dari ~0 ke `peak`.
    pub warmup: u64,
    /// Total langkah yang direncanakan (untuk menentukan ujung kurva).
    pub total: u64,
    /// Bentuk penurunan.
    pub kind: LrKind,
    /// Untuk [`LrKind::Wsd`]: porsi akhir (0..1) yang dipakai menurun.
    pub decay_fraction: f64,
}

impl LrSchedule {
    /// Jadwal kosinus standar NCM.
    pub fn cosine(peak: f64, floor: f64, warmup: u64, total: u64) -> Self {
        Self {
            peak,
            floor,
            warmup,
            total,
            kind: LrKind::Cosine,
            decay_fraction: 0.2,
        }
    }

    /// Jadwal warmup–stabil–decay (diablason pada F1.7).
    pub fn wsd(peak: f64, floor: f64, warmup: u64, total: u64, decay_fraction: f64) -> Self {
        Self {
            peak,
            floor,
            warmup,
            total,
            kind: LrKind::Wsd,
            decay_fraction: decay_fraction.clamp(0.01, 1.0),
        }
    }

    /// Laju pada `step` (0-based, di luar rentang dijepit).
    pub fn lr(&self, step: u64) -> f64 {
        if self.warmup > 0 && step < self.warmup {
            let frac = (step as f64 + 1.0) / (self.warmup as f64);
            return self.peak * frac;
        }
        let span = self.total.saturating_sub(self.warmup);
        if span == 0 {
            return self.floor;
        }
        let t = (step.saturating_sub(self.warmup) as f64 / span as f64).clamp(0.0, 1.0);
        match self.kind {
            LrKind::Cosine => {
                let c = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                self.floor + (self.peak - self.floor) * c
            }
            LrKind::Wsd => {
                let stable = 1.0 - self.decay_fraction;
                if t <= stable {
                    self.peak
                } else {
                    let d = (t - stable) / self.decay_fraction.max(1e-6);
                    self.peak + (self.floor - self.peak) * d.clamp(0.0, 1.0)
                }
            }
        }
    }

    /// Laju Muon: grid terpisah (optimum Muon jauh lebih rendah dari AdamW),
    /// jadi `peak`/`floor` diskalakan dengan `scale` (default 0,1–0,5).
    pub fn lr_scaled(&self, step: u64, scale: f64) -> f64 {
        self.lr(step) * scale
    }

    /// Laju sebagai `f32` (untuk tensor dan log).
    pub fn lr_f32(&self, step: u64) -> f32 {
        self.lr(step) as f32
    }

    /// Cek konsistensi jadwal; dipakai `TrainConfig::validate`.
    pub fn validate(&self) -> Result<(), vir_common::VirError> {
        use vir_common::VirError;
        if self.peak <= 0.0 || self.floor < 0.0 {
            return Err(VirError::Config(
                "LR puncak harus > 0 dan lantai >= 0".into(),
            ));
        }
        if self.floor > self.peak {
            return Err(VirError::Config(
                "LR lantai tidak boleh melebihi puncak".into(),
            ));
        }
        if self.total == 0 {
            return Err(VirError::Config("total langkah LR harus > 0".into()));
        }
        if self.warmup > self.total {
            return Err(VirError::Config("warmup tidak boleh melebihi total".into()));
        }
        Ok(())
    }
}

/// Uji asap jadwal: monotonik naik saat warmup, lalu tidak pernah melebihi
/// puncak, dan ujungnya menyentuh lantai.
pub fn schedule_smoke() -> Result<(), vir_common::VirError> {
    use vir_common::VirError;
    let s = LrSchedule::cosine(3e-4, 3e-5, 10, 100);
    let bad = |m: &str| Err(VirError::Internal(m.to_string()));
    if s.lr(0) > s.lr(9) {
        return bad("warmup tidak naik");
    }
    if (s.lr(9) - 3e-4).abs() > 1e-9 {
        return bad("puncak tidak tercapai di akhir warmup");
    }
    for step in 0..=100 {
        if s.lr(step) > 3e-4 + 1e-9 {
            return bad("laju melebihi puncak");
        }
    }
    if (s.lr(100) - 3e-5).abs() > 1e-9 {
        return bad("lantai tidak tercapai");
    }
    let w = LrSchedule::wsd(3e-4, 3e-5, 10, 100, 0.2);
    if (w.lr(50) - 3e-4).abs() > 1e-9 {
        return bad("fase stabil WSD tidak rata");
    }
    if w.lr(100) > w.lr(90) {
        return bad("fase decay WSD tidak turun");
    }
    Ok(())
}
