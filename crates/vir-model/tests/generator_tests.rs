//! Tes nyata F1.6 (generator NCM: generate level demi level, determinisme
//! seed, Σℓ_k sesuai jadwal) untuk `vir-model`.
//!
//! Dijalankan job CI `vir-model` (`cargo test --manifest-path
//! crates/vir-model/Cargo.toml`). Memakai backend CPU default (flex).

use burn::prelude::*;
use vir_common::{LevelScheduleKind, level_schedule};
use vir_model::{GenerateOutput, Generator, GeneratorConfig, Sampler};
use vir_vq::indices_to_vec;

/// Generator mini untuk uji: `d=16`, 2 head, jadwal kecil.
fn mini_generator(levels: Vec<usize>) -> Generator {
    let device = Device::default();
    let mut cfg = GeneratorConfig::new(16, 2, levels);
    cfg.codebook_size = 16;
    cfg.phi_kernel = 1;
    cfg.prelude_blocks = 1;
    cfg.core_blocks = 1;
    cfg.coda_blocks = 1;
    cfg.num_loops = 1;
    Generator::new(&cfg, &device).expect("generator mini")
}

/// Baca semua kode per level sebagai `Vec<Vec<i64>>` (satu baris datar/level).
fn all_codes(out: &GenerateOutput) -> Vec<Vec<i64>> {
    out.codes
        .iter()
        .map(|t| {
            let [b, l] = t.dims();
            indices_to_vec(&t.clone().reshape([b * l]))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Generate level demi level
// ---------------------------------------------------------------------------

#[test]
fn generate_produces_one_tensor_per_level() {
    let levels = vec![2usize, 4, 8];
    let model = mini_generator(levels.clone());
    let out = model
        .generate(None, &Sampler::greedy())
        .expect("generate harus sukses");

    assert_eq!(out.codes.len(), levels.len(), "satu tensor kode per level");
    for (k, t) in out.codes.iter().enumerate() {
        let [b, l] = t.dims();
        assert_eq!(b, 1);
        assert_eq!(l, levels[k], "panjang level {k}");
    }
    assert_eq!(out.batch, 1);
    assert_eq!(out.prefix_len, 0);
    assert_eq!(out.ctx_len, levels.last().copied().unwrap_or(0));
    assert_eq!(out.z_hat.dims(), [1, levels[2], 16]);
}

#[test]
fn generate_positions_equal_sum_of_levels() {
    let levels = level_schedule(LevelScheduleKind::Linear, 8, 3); // [1,4,8]
    let model = mini_generator(levels.clone());
    let out = model
        .generate(None, &Sampler::greedy())
        .expect("generate harus sukses");

    let expected: usize = levels.iter().sum();
    assert_eq!(out.positions, expected, "Σℓ_k = {expected}");
}

#[test]
fn generate_is_deterministic_with_seed() {
    let model = mini_generator(vec![2, 4, 8]);
    // Suhu > 0 agar benih benar-benar memengaruhi hasil.
    let sampler = Sampler::new(0.9, 0, 7);

    let a = model.generate(None, &sampler).expect("a");
    let b = model.generate(None, &sampler).expect("b");
    assert_eq!(all_codes(&a), all_codes(&b), "seed sama → identik");

    // Benih berbeda menghasilkan kode berbeda (dengan suhu > 0).
    let other = Sampler::new(0.9, 0, 8);
    let c = model.generate(None, &other).expect("c");
    assert_ne!(all_codes(&a), all_codes(&c), "seed berbeda → hasil berbeda");
}

#[test]
fn generate_with_prefix_reports_prefix_len() {
    let device = Device::default();
    let levels = vec![2usize, 4, 8];
    let model = mini_generator(levels.clone());
    let prefix = Tensor::<3>::zeros([1, 3, 16], &device);

    let out = model
        .generate(Some(prefix), &Sampler::greedy())
        .expect("generate dengan prefix");

    assert_eq!(out.prefix_len, 3);
    assert_eq!(out.batch, 1);
    // Konteks = prefix + L.
    assert_eq!(out.ctx_len, 3 + levels[2]);
    assert_eq!(out.positions, levels.iter().sum::<usize>());
    // z_hat mencakup prefix + blok.
    assert_eq!(out.z_hat.dims(), [1, 3 + levels[2], 16]);
}

#[test]
fn generate_from_accepts_initial_context() {
    let device = Device::default();
    let levels = vec![2usize, 4, 8];
    let model = mini_generator(levels.clone());
    // Konteks awal [B, P+L, d] dengan P=2.
    let init = Tensor::<3>::zeros([2, 2 + 8, 16], &device);

    let out = model
        .generate_from(init, 2, &Sampler::greedy())
        .expect("generate_from harus sukses");

    assert_eq!(out.batch, 2);
    assert_eq!(out.prefix_len, 2);
    assert_eq!(out.codes.len(), levels.len());
    for t in &out.codes {
        assert_eq!(t.dims()[0], 2, "batch harus dipertahankan");
    }
}

#[test]
fn generate_from_rejects_short_context() {
    let device = Device::default();
    let levels = vec![2usize, 4, 8];
    let model = mini_generator(levels);
    let init = Tensor::<3>::zeros([1, 4, 16], &device); // ctx < L=8
    let err = model.generate_from(init, 0, &Sampler::greedy());
    assert!(err.is_err(), "konteks < L harus ditolak");
}

// ---------------------------------------------------------------------------
// Jadwal level konsisten dengan vir_common
// ---------------------------------------------------------------------------

#[test]
fn scheduled_config_matches_level_schedule() {
    for kind in [
        LevelScheduleKind::Linear,
        LevelScheduleKind::Quadratic,
        LevelScheduleKind::Sqrt,
        LevelScheduleKind::Log,
        LevelScheduleKind::Exp,
        LevelScheduleKind::Cosine,
    ] {
        let cfg = GeneratorConfig::scheduled(16, 2, kind, 16, 5);
        assert_eq!(cfg.levels, level_schedule(kind, 16, 5), "kind={kind:?}");
        assert_eq!(cfg.total_len(), 16);
        assert_eq!(cfg.num_levels(), 5);
    }
}

#[test]
fn config_validate_rejects_bad_schedules() {
    // Tidak naik ketat.
    let mut cfg = GeneratorConfig::new(16, 2, vec![2, 2, 4]);
    cfg.codebook_size = 8;
    assert!(cfg.validate().is_err());

    // dim tidak habis dibagi num_heads.
    let bad = GeneratorConfig::new(10, 3, vec![2, 4]);
    assert!(bad.validate().is_err());
}
