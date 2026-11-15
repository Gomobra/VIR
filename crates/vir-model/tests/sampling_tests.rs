//! Tes nyata F1.6 (sampling) dan F2.4 (CFG, MBR) untuk `vir-model`.
//!
//! Dijalankan job CI `vir-model` (`cargo test --manifest-path
//! crates/vir-model/Cargo.toml`). Sebagian fungsi murni (f32/`Vec`) diuji
//! langsung; `sample_codes` memakai tensor CPU default (backend flex).

use std::collections::HashSet;

use burn::prelude::*;
use burn::tensor::TensorData;
use vir_common::DeterministicRng;
use vir_model::{
    Sampler, argmax_row, cfg_logit, entropy, mbr_select, sample_codes, sample_index, softmax_row,
    top_k_indices,
};
use vir_vq::indices_to_vec;

/// Tensor `[b, l, v]` dari data host baris-major.
fn logits_tensor(data: &[f32], b: usize, l: usize, v: usize) -> Tensor<3> {
    let device = Device::default();
    Tensor::<3>::from_floats(TensorData::new(data.to_vec(), [b, l, v]), &device)
}

/// Baca indeks `[b, l]` menjadi `Vec<i64>`.
fn read_codes(t: &Tensor<2, Int>) -> Vec<i64> {
    let [b, l] = t.dims();
    indices_to_vec(&t.clone().reshape([b * l]))
}

// ---------------------------------------------------------------------------
// Softmax / argmax / top-k
// ---------------------------------------------------------------------------

#[test]
fn argmax_picks_largest() {
    assert_eq!(argmax_row(&[1.0, 5.0, 3.0]), 1);
    assert_eq!(argmax_row(&[-2.0, -1.0, -9.0]), 1);
    assert_eq!(argmax_row(&[7.0]), 0);
}

#[test]
fn softmax_zero_temperature_is_one_hot() {
    let probs = softmax_row(&[1.0, 5.0, 3.0], 0.0);
    assert_eq!(probs.len(), 3);
    assert_eq!(probs[1], 1.0);
    assert_eq!(probs[0], 0.0);
    assert_eq!(probs[2], 0.0);
}

#[test]
fn softmax_sums_to_one_and_finite() {
    let probs = softmax_row(&[0.5, -1.0, 2.0, 0.0], 0.7);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    assert!(probs.iter().all(|p| p.is_finite() && *p >= 0.0));
}

#[test]
fn top_k_indices_orders_and_truncates() {
    let values = [0.1, 0.9, 0.4, 0.7];
    assert_eq!(top_k_indices(&values, 1), vec![1]);
    assert_eq!(top_k_indices(&values, 2), vec![1, 3]);
    // k = 0 mengembalikan semua indeks (urut turun).
    assert_eq!(top_k_indices(&values, 0), vec![1, 3, 2, 0]);
    // k melebihi panjang tidak panik.
    assert_eq!(top_k_indices(&values, 99).len(), 4);
}

// ---------------------------------------------------------------------------
// Distribusi vs suhu
// ---------------------------------------------------------------------------

#[test]
fn higher_temperature_raises_entropy() {
    let logits = [0.5, -1.0, 2.0, 0.0];
    let cold = entropy(&softmax_row(&logits, 0.05));
    let warm = entropy(&softmax_row(&logits, 1.0));
    let hot = entropy(&softmax_row(&logits, 20.0));
    assert!(cold < warm, "cold={cold} warm={warm}");
    assert!(warm < hot, "warm={warm} hot={hot}");
    // Suhu sangat tinggi mendekati seragam: H → ln(4).
    assert!((hot - 4f32.ln()).abs() < 0.05, "hot={hot}");
}

#[test]
fn sample_index_respects_distribution_shape() {
    let probs = [0.0, 0.0, 1.0];
    let mut rng = DeterministicRng::new(1);
    for _ in 0..50 {
        assert_eq!(sample_index(&probs, rng.inner_mut()), 2);
    }
}

// ---------------------------------------------------------------------------
// Determinisme seed dan top-k = argmax
// ---------------------------------------------------------------------------

#[test]
fn sample_codes_is_deterministic_with_seed() {
    // 2 batch × 3 posisi × 4 kod, dua puncak dekat.
    let base = [0.2f32, 0.1, 0.9, 0.0];
    let data: Vec<f32> = (0..6).flat_map(|_| base.iter().copied()).collect();
    let logits = logits_tensor(&data, 2, 3, 4);
    let sampler = Sampler::new(0.9, 0, 7);

    let a = read_codes(&sample_codes(logits.clone(), &sampler));
    let b = read_codes(&sample_codes(logits, &sampler));
    assert_eq!(a, b, "seed sama harus identik");
    assert_eq!(a.len(), 6);
}

#[test]
fn sample_codes_top_k_one_equals_argmax() {
    let base = [0.2f32, 0.1, 0.9, 0.3];
    let data: Vec<f32> = (0..6).flat_map(|_| base.iter().copied()).collect();
    let logits = logits_tensor(&data, 2, 3, 4);

    // Greedy (temperature 0) dan top-k=1 sama-sama argmax (indeks 2).
    let greedy = read_codes(&sample_codes(logits.clone(), &Sampler::greedy()));
    let top1 = read_codes(&sample_codes(logits, &Sampler::new(1.0, 1, 99)));
    assert!(greedy.iter().all(|&c| c == 2), "greedy={greedy:?}");
    assert_eq!(greedy, top1, "top-k=1 harus sama dengan argmax");
}

#[test]
fn low_temperature_concentrates_on_argmax() {
    // Baris logit [0, 0, 1]: argmax = 2. Suhu rendah → hampir semua sampel = 2;
    // suhu tinggi → jauh lebih banyak nilai berbeda.
    let data: Vec<f32> = (0..300).flat_map(|_| [0.0f32, 0.0, 1.0]).collect();
    let logits = logits_tensor(&data, 1, 300, 3);

    let cold = read_codes(&sample_codes(logits.clone(), &Sampler::new(0.01, 0, 3)));
    let hot = read_codes(&sample_codes(logits, &Sampler::new(100.0, 0, 3)));

    let cold_argmax = cold.iter().filter(|&&c| c == 2).count();
    let cold_distinct = cold.iter().copied().collect::<HashSet<_>>().len();
    let hot_distinct = hot.iter().copied().collect::<HashSet<_>>().len();
    // Suhu rendah terkonsentrasi pada argmax; suhu tinggi menyebar.
    assert!(
        cold_argmax * 100 >= cold.len() * 99,
        "cold argmax={cold_argmax}"
    );
    assert!(cold_distinct <= 2, "cold distinct={cold_distinct}");
    assert!(hot_distinct > cold_distinct, "hot harus lebih acak");
}

// ---------------------------------------------------------------------------
// CFG (F2.4)
// ---------------------------------------------------------------------------

#[test]
fn cfg_logit_matches_formula() {
    // ℓ_cfg = ℓ_uc + β(ℓ − ℓ_uc).
    assert!((cfg_logit(0.0, 2.0, 1.5) - 3.0).abs() < 1e-6);
    // β = 0 → hanya unconditional.
    assert!((cfg_logit(1.0, 5.0, 0.0) - 1.0).abs() < 1e-6);
    // β = 1 → hanya conditional.
    assert!((cfg_logit(1.0, 5.0, 1.0) - 5.0).abs() < 1e-6);
    // β = 2 setara (1+λ)ℓ − λℓ_uc dengan λ = 1.
    assert!((cfg_logit(-1.0, 4.0, 2.0) - 9.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// MBR select (F2.4)
// ---------------------------------------------------------------------------

/// Kemiripan Hamming: `1 / (1 + jarak)` (bisa diganti pemanggil).
fn sim_hamming(a: &[u32], b: &[u32]) -> f32 {
    let dist: usize = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| usize::from(x != y))
        .sum();
    1.0 / (1.0 + dist as f32)
}

#[test]
fn mbr_selects_most_central_candidate() {
    // Kandidat tengah [0,1] paling mirip ke keduanya.
    let candidates = vec![vec![0u32, 0], vec![0, 1], vec![1, 1]];
    let idx = mbr_select(&candidates, sim_hamming);
    assert_eq!(idx, 1, "kandidat paling sentral harus dipilih");
}

#[test]
fn mbr_similarity_function_is_replaceable() {
    let candidates = vec![vec![0u32, 0], vec![0, 1], vec![1, 1]];
    // Fungsi kemiripan berbeda (hanya elemen pertama) memberi hasil berbeda.
    let by_first = mbr_select(&candidates, |a, b| if a[0] == b[0] { 1.0 } else { 0.0 });
    assert_eq!(by_first, 0, "kemiripan elemen pertama memilih kandidat 0");
    // Kandidat tunggal → indeks 0.
    assert_eq!(mbr_select(&[vec![1u32, 2]], sim_hamming), 0);
}
