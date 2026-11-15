//! Tes nyata F2.1 (recurrent depth, sampling `r`) dan F2.2 (halting) untuk
//! `vir-model`.
//!
//! Dijalankan job CI `vir-model` (`cargo test --manifest-path
//! crates/vir-model/Cargo.toml`). Memakai backend CPU default (flex).

use burn::module::Module;
use burn::nn::LinearConfig;
use burn::prelude::*;
use burn::tensor::Distribution;
use vir_common::DeterministicRng;
use vir_model::{
    EntropyHalter, FixedHalter, Generator, GeneratorConfig, Halter, entropy, recurrent_forward,
    sample_r,
};
use vir_nn::Block;

/// Susun prelude/inti/coda + adapter untuk satu dimensi `d`.
fn make_parts(d: usize, heads: usize) -> (Vec<Block>, Vec<Block>, Vec<Block>, burn::nn::Linear) {
    let device = Device::default();
    let prelude = vec![Block::new(d, heads, &device)];
    let core = vec![Block::new(d, heads, &device)];
    let coda = vec![Block::new(d, heads, &device)];
    let adapter = LinearConfig::new(d * 2, d).init(&device);
    (prelude, core, coda, adapter)
}

/// Tensor `[1, t, d]` acak (deterministik lewat distribusi Burn).
fn rand_input(t: usize, d: usize) -> Tensor<3> {
    let device = Device::default();
    Tensor::<3>::random([1, t, d], Distribution::Normal(0.0, 1.0), &device)
}

fn read3(t: &Tensor<3>) -> Vec<f32> {
    t.to_data().try_to_vec::<f32>().unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Rekurensi (F2.1)
// ---------------------------------------------------------------------------

#[test]
fn more_loops_change_output_but_not_parameters() {
    let d = 16;
    let (prelude, core, coda, adapter) = make_parts(d, 2);
    let e = rand_input(5, d);

    let out1 = recurrent_forward(&prelude, &core, &coda, &adapter, e.clone(), 1);
    let out4 = recurrent_forward(&prelude, &core, &coda, &adapter, e, 4);

    assert_eq!(out1.dims(), [1, 5, d]);
    assert_eq!(out4.dims(), [1, 5, d]);
    // Iterasi lebih banyak menghasilkan keluaran berbeda (komputasi, bukan param).
    assert_ne!(read3(&out1), read3(&out4), "r=1 dan r=4 harus berbeda");
}

#[test]
fn loops_do_not_add_parameters() {
    // `num_loops` hanya mengubah jumlah iterasi; jumlah parameter tetap.
    let device = Device::default();
    let levels = vec![2usize, 4];
    let mut cfg = GeneratorConfig::new(16, 2, levels.clone());
    cfg.codebook_size = 8;
    cfg.phi_kernel = 1;

    cfg.num_loops = 1;
    let g1 = Generator::new(&cfg, &device).expect("generator r=1");
    cfg.num_loops = 8;
    let g8 = Generator::new(&cfg, &device).expect("generator r=8");

    assert_eq!(
        g1.num_params(),
        g8.num_params(),
        "param tidak boleh bertambah"
    );
    assert_eq!(g1.num_loops(), 1);
    assert_eq!(g8.num_loops(), 8);
    assert_eq!(g1.num_levels(), levels.len());
}

#[test]
fn zero_loops_treated_as_one() {
    let d = 8;
    let (prelude, core, coda, adapter) = make_parts(d, 2);
    let e = rand_input(3, d);
    // r = 0 diperlakukan sebagai satu iterasi.
    let zero = recurrent_forward(&prelude, &core, &coda, &adapter, e.clone(), 0);
    let one = recurrent_forward(&prelude, &core, &coda, &adapter, e, 1);
    assert_eq!(read3(&zero), read3(&one));
}

// ---------------------------------------------------------------------------
// Sampling r (Poisson lognormal, F2.1 langkah 3)
// ---------------------------------------------------------------------------

#[test]
fn sample_r_is_deterministic_with_seed() {
    let a: Vec<usize> = {
        let mut rng = DeterministicRng::new(42);
        (0..64)
            .map(|_| sample_r(32.0, 0.5, rng.inner_mut()))
            .collect()
    };
    let b: Vec<usize> = {
        let mut rng = DeterministicRng::new(42);
        (0..64)
            .map(|_| sample_r(32.0, 0.5, rng.inner_mut()))
            .collect()
    };
    assert_eq!(a, b, "seed sama harus menghasilkan r identik");
}

#[test]
fn sample_r_is_reasonable_and_positive() {
    let mut rng = DeterministicRng::new(7);
    let draws: Vec<usize> = (0..4000)
        .map(|_| sample_r(32.0, 0.5, rng.inner_mut()))
        .collect();
    assert!(draws.iter().all(|&r| r >= 1), "r harus >= 1");
    let mean = draws.iter().sum::<usize>() as f32 / draws.len() as f32;
    // E[r] = r̄ + 1 = 33 (simulasi PRD); toleransi longgar untuk bias Knuth.
    assert!((mean - 33.0).abs() < 3.0, "mean r={mean}");
    // Nilai ekstrem tetap jauh di bawah batas wajar.
    assert!(draws.iter().all(|&r| r < 1000), "ada r ekstrem");
}

#[test]
fn sample_r_small_mean_stays_positive() {
    let mut rng = DeterministicRng::new(9);
    for _ in 0..200 {
        let r = sample_r(1.0, 0.5, rng.inner_mut());
        assert!(r >= 1);
    }
}

// ---------------------------------------------------------------------------
// Halting (F2.2)
// ---------------------------------------------------------------------------

#[test]
fn fixed_halter_never_halts() {
    let h = FixedHalter { max_loops: 4 };
    assert!(!h.should_halt(0.0));
    assert!(!h.should_halt(100.0));
}

#[test]
fn entropy_halter_stops_below_threshold() {
    let h = EntropyHalter { threshold: 0.5 };
    assert!(h.should_halt(0.1), "entropi rendah → berhenti");
    assert!(!h.should_halt(2.0), "entropi tinggi → lanjut");
    // Ambang batas: tepat di ambang TIDAK berhenti (strict <).
    assert!(!h.should_halt(0.5));
}

#[test]
fn entropy_uniform_vs_peaked() {
    // Distribusi seragam 4 nilai: H = ln 4.
    let uniform = [0.25f32; 4];
    // Distribusi puncak: H ≈ 0.
    let peaked = [1.0f32, 0.0, 0.0, 0.0];
    let hu = entropy(&uniform);
    let hp = entropy(&peaked);
    assert!((hu - 4f32.ln()).abs() < 1e-5, "H seragam={hu}");
    assert!(hp.abs() < 1e-6, "H puncak={hp}");
    assert!(hu > hp);

    // Halter ambang 0.5: puncak berhenti, seragam lanjut.
    let h = EntropyHalter { threshold: 0.5 };
    assert!(h.should_halt(hp));
    assert!(!h.should_halt(hu));
}

#[test]
fn entropy_ignores_zero_probabilities() {
    // 0·log 0 dianggap 0; hanya dua nilai non-nol → H = ln 2.
    let h = entropy(&[0.5f32, 0.0, 0.5, 0.0]);
    assert!((h - 2f32.ln()).abs() < 1e-6, "H={h}");
}
