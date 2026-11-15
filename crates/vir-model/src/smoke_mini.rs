//! Irisan vertikal mini (PRD F1.8, gerbang G1): BPE → VQ → generator →
//! decoder → sampling, dengan model yang **menghafal 8 sampel lalu
//! mereproduksinya persis**.
//!
//! Alur (eksekusi di [`crate::smoke_mini_impl`]):
//!
//! ```text
//! token → laten beku → encode VQ (F1.4.b) → kode target r_k
//!   → generator di-training (kepala kuadrat terkecil, F1.6)
//!   → generate level demi level dengan benih awal (Algoritma 2)
//!   → rekonstruksi ẑ^(K) → decode token (F1.4.d)
//! ```
//!
//! **Kenapa benih awal.** Level 0 generator bersyarat pada `ẑ^(−1) = 0`; tanpa
//! konteks awal, semua sampel punya masukan identik sehingga mustahil
//! dibedakan. Benih per (sampel, posisi) memberi identitas yang mengalir ke
//! tiap level lewat atensi (blok penuh, F1.5).
//!
//! **Kenapa kepala kuadrat terkecil.** `vir-model` tidak menarik feature
//! `train` Burn (optimizer di `vir-train`, F1.7). Gerbang mini menyelesaikan
//! `X·W = Y` langsung untuk kepala tiap level; bila fitur (baris) bebas linear,
//! argmax logit sama persis dengan kode target. Ini mengukur kemampuan
//! arsitektur + loss memuat data, bukan konvergensi optimizer.

use vir_common::{DeterministicRng, LevelScheduleKind, VirError, level_schedule};

use crate::smoke_mini_impl::{Raw, execute};

/// Konfigurasi irisan mini.
#[derive(Debug, Clone)]
pub struct MiniSmokeConfig {
    /// Sampel token (tiap baris satu sampel); dipangkas/diisi sampai `L`.
    pub samples: Vec<Vec<u32>>,
    /// Jadwal level `ℓ_k` (naik ketat, berakhir `L`).
    pub levels: Vec<usize>,
    /// Lebar laten/hidden `d`.
    pub dim: usize,
    /// Jumlah head.
    pub num_heads: usize,
    /// Ukuran codebook `|V_k|`.
    pub codebook_size: usize,
    /// Kernel `Phi` pada `R_k` (`<= 1` memakai `Phi` nol).
    pub phi_kernel: usize,
    /// Ukuran vocab token (untuk embedding beku dan decoder).
    pub vocab: usize,
    /// Panjang prefix (bagian awal konteks, `P`).
    pub prefix_len: usize,
    /// Jenis jadwal (untuk memeriksa Σℓ_k sesuai F1.1).
    pub schedule: LevelScheduleKind,
    /// Benih deterministik.
    pub seed: u64,
}

impl MiniSmokeConfig {
    /// Konfigurasi mini standar (L = 8, K = 3, d = 128, 8 sampel).
    ///
    /// Jadwal level diambil dari [`level_schedule`] (F1.1) agar Σℓ_k konsisten.
    pub fn standard(samples: Vec<Vec<u32>>) -> Self {
        let max_token = samples
            .iter()
            .flatten()
            .copied()
            .max()
            .map_or(0, |m| m as usize);
        Self {
            samples,
            levels: level_schedule(LevelScheduleKind::Linear, 8, 3),
            dim: 128,
            num_heads: 4,
            codebook_size: 256,
            phi_kernel: 1,
            vocab: (max_token + 1).max(2),
            prefix_len: 2,
            schedule: LevelScheduleKind::Linear,
            seed: 42,
        }
    }

    /// Tetapkan vocab token eksplisit.
    pub fn with_vocab(mut self, vocab: usize) -> Self {
        self.vocab = vocab.max(2);
        self
    }

    /// Tetapkan jadwal level eksplisit.
    pub fn with_levels(mut self, levels: Vec<usize>) -> Self {
        self.levels = levels;
        self
    }
}

/// Hasil irisan mini (metrik + gerbang).
#[derive(Debug, Clone)]
pub struct SmokeResult {
    /// Jadwal level yang dipakai.
    pub levels: Vec<usize>,
    /// `Σ ℓ_k`.
    pub total_positions: usize,
    /// CE per level sebelum latihan kepala.
    pub loss_before: Vec<f32>,
    /// CE per level sesudah latihan kepala.
    pub loss_after: Vec<f32>,
    /// Akurasi argmax per level sebelum/sesudah.
    pub acc_before: Vec<f32>,
    /// Akurasi argmax per level sesudah.
    pub acc_after: Vec<f32>,
    /// Galat rekonstruksi kumulatif VQ per level (informasional; VQ mini beku).
    pub recon_error: Vec<f32>,
    /// `true` bila kode hasil generate sama persis dengan kode target.
    pub exact_codes: bool,
    /// `true` bila token hasil dekode sama persis dengan token masukan.
    pub exact_tokens: bool,
    /// Token hasil dekode `ẑ` dari kode yang di-generate.
    pub generated_tokens: Vec<u32>,
    /// Posisi generator yang diproses (`Σ ℓ_k`).
    pub positions: usize,
    /// `true` bila semua gerbang lulus.
    pub ok: bool,
    /// Alasan kegagalan (kosong bila `ok`).
    pub failures: Vec<String>,
}

/// Jalankan irisan mini dan kembalikan metrik + hasil gerbang.
pub fn run_mini_smoke(cfg: &MiniSmokeConfig) -> Result<SmokeResult, VirError> {
    let raw = execute(cfg)?;
    Ok(SmokeResult::from_raw(raw, cfg))
}

/// Bangun sampel token deterministik (untuk CLI dan tes).
pub fn synth_samples(count: usize, len: usize, vocab: u32, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = DeterministicRng::new(seed);
    let hi = vocab.max(1);
    (0..count)
        .map(|_| {
            (0..len)
                .map(|_| rand::Rng::gen_range(rng.inner_mut(), 0..hi))
                .collect()
        })
        .collect()
}

impl SmokeResult {
    /// Susun hasil dan evaluasi gerbang G1.
    fn from_raw(raw: Raw, cfg: &MiniSmokeConfig) -> Self {
        let failures = gate_failures(&raw, cfg);
        let ok = failures.is_empty();
        SmokeResult {
            levels: raw.levels,
            total_positions: raw.total_positions,
            loss_before: raw.loss_before,
            loss_after: raw.loss_after,
            acc_before: raw.acc_before,
            acc_after: raw.acc_after,
            recon_error: raw.recon_error,
            exact_codes: raw.exact_codes,
            exact_tokens: raw.exact_tokens,
            generated_tokens: raw.generated_tokens,
            positions: raw.positions,
            ok,
            failures,
        }
    }
}

/// Kumpulkan alasan kegagalan gerbang G1 (kosong bila lulus).
fn gate_failures(raw: &Raw, cfg: &MiniSmokeConfig) -> Vec<String> {
    let mut failures = Vec::new();
    let expect = level_schedule(
        cfg.schedule,
        raw.levels.last().copied().unwrap_or(0),
        cfg.levels.len(),
    );
    if raw.levels != expect {
        failures.push(format!("jadwal {:?} != F1.1 {:?}", raw.levels, expect));
    }
    if raw.positions != raw.total_positions {
        failures.push(format!(
            "posisi generate {} != Σℓ_k {}",
            raw.positions, raw.total_positions
        ));
    }
    if !raw.exact_codes {
        failures.push("kode hasil generate != kode target".into());
    }
    if !raw.exact_tokens {
        failures.push("token hasil dekode != token masukan".into());
    }
    for k in 0..raw.levels.len() {
        if !(raw.loss_after[k] < raw.loss_before[k]) {
            failures.push(format!("loss level {k} tidak turun"));
        }
        if raw.acc_after[k] < MIN_ACC {
            failures.push(format!("akurasi level {k} < 1"));
        }
    }
    if raw.recon_error.iter().any(|v| !v.is_finite()) {
        failures.push("galat rekonstruksi VQ tidak hingga".into());
    }
    failures
}

/// Akurasi minimum per level untuk lulus gerbang.
const MIN_ACC: f32 = 0.999;
