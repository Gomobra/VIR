//! Tes nyata F1.5 / F1.5.a untuk `vir-nn`: RoPE (invarian translasi), mask
//! block-diagonal dan prefix-LM (uji kebocoran nyata), RMSNorm, SwiGLU, serta
//! attention (panjang variabel + padding). Dijalankan oleh job CI `vir-nn`
//! (`cargo test --manifest-path crates/vir-nn/Cargo.toml`).

use burn::prelude::*;
use burn::tensor::{Bool, Distribution, TensorData};
use vir_nn::{
    Block, RmsNorm, SwiGluMlp, apply_rope, block_diagonal_mask, prefix_lm_mask, rope_cos_sin,
};

/// Baca tensor `[b, t, d]` sebagai `Vec<f32>` baris-major.
fn read3(t: &Tensor<3>) -> Vec<f32> {
    t.to_data().try_to_vec::<f32>().expect("data f32")
}

/// Tensor `[b, t, d]` dari data host.
fn to_tensor(data: &[f32], b: usize, t: usize, d: usize, device: &Device) -> Tensor<3> {
    Tensor::<3>::from_floats(TensorData::new(data.to_vec(), [b, t, d]), device)
}

/// Tensor bool `[t, t]` dari matriks baris (true = boleh melihat).
fn bool_mask(rows: &[Vec<bool>], device: &Device) -> Tensor<2, Bool> {
    let t = rows.len();
    let flat: Vec<bool> = rows.iter().flatten().copied().collect();
    Tensor::<2, Bool>::from_bool(TensorData::new(flat, [t, t]), device)
}

/// Masukan yang tiap elemennya berbeda agar kebocoran antar-token terdeteksi.
fn pattern_input(b: usize, t: usize, d: usize) -> Vec<f32> {
    (0..b * t * d)
        .map(|i| ((i % 17) as f32) * 0.1 - 0.5)
        .collect()
}

/// `⟨R_p q, R_s k⟩` dihitung lewat `apply_rope` pada posisi `p` dan `s`.
fn rope_dot(q: &[f32], k: &[f32], p: usize, s: usize, d: usize, base: f32) -> f32 {
    let t = p.max(s) + 1;
    let device = Device::default();
    let mut qd = vec![0.0f32; t * d];
    let mut kd = vec![0.0f32; t * d];
    qd[p * d..p * d + d].copy_from_slice(q);
    kd[s * d..s * d + d].copy_from_slice(k);
    let qr = read3(&apply_rope(to_tensor(&qd, 1, t, d, &device), base));
    let kr = read3(&apply_rope(to_tensor(&kd, 1, t, d, &device), base));
    (0..d).map(|i| qr[p * d + i] * kr[s * d + i]).sum()
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

#[test]
fn rope_cos_sin_shape_and_duplicated_halves() {
    let (cos, sin) = rope_cos_sin(5, 8, 10_000.0);
    assert_eq!(cos.len(), 5 * 8);
    assert_eq!(sin.len(), 5 * 8);
    // posisi 0 → sudut 0 → cos = 1, sin = 0.
    for i in 0..8 {
        assert!((cos[i] - 1.0).abs() < 1e-6, "pos 0 cos[{i}]");
        assert!(sin[i].abs() < 1e-6, "pos 0 sin[{i}]");
    }
    // Paruh kedua menduplikasi paruh pertama (gaya GPT-NeoX).
    for p in 0..5 {
        for i in 0..4 {
            assert_eq!(cos[p * 8 + i], cos[p * 8 + i + 4]);
            assert_eq!(sin[p * 8 + i], sin[p * 8 + i + 4]);
        }
    }
}

#[test]
fn rope_dot_depends_only_on_position_difference() {
    let d = 8;
    let base = 10_000.0;
    let q: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.3).collect();
    let k: Vec<f32> = (0..d).map(|i| (d - i) as f32 * 0.2 - 0.1).collect();
    // f32 pada sudut besar (mis. 7 rad) menyisakan galat ~1e-6; toleransi 1e-4.
    let tol = 1e-4;

    // Selisih 0: titik harus sama untuk posisi berapa pun.
    let a = rope_dot(&q, &k, 0, 0, d, base);
    let b = rope_dot(&q, &k, 3, 3, d, base);
    let c = rope_dot(&q, &k, 7, 7, d, base);
    assert!((a - b).abs() < tol && (a - c).abs() < tol, "{a} {b} {c}");

    // Selisih 2.
    let d0 = rope_dot(&q, &k, 2, 0, d, base);
    let d1 = rope_dot(&q, &k, 5, 3, d, base);
    let d2 = rope_dot(&q, &k, 7, 5, d, base);
    assert!(
        (d0 - d1).abs() < tol && (d0 - d2).abs() < tol,
        "{d0} {d1} {d2}"
    );

    // Selisih 5.
    let e0 = rope_dot(&q, &k, 5, 0, d, base);
    let e1 = rope_dot(&q, &k, 7, 2, d, base);
    assert!((e0 - e1).abs() < tol, "{e0} {e1}");

    // Bukan konstan: selisih berbeda menghasilkan titik berbeda.
    assert!((a - d0).abs() > 1e-3, "invarian tak boleh konstan");
}

// ---------------------------------------------------------------------------
// Mask block-diagonal
// ---------------------------------------------------------------------------

#[test]
fn block_diagonal_mask_is_block_structure() {
    let lengths = [1usize, 3, 6, 8];
    let mask = block_diagonal_mask(&lengths);
    let total: usize = lengths.iter().sum();
    assert_eq!(mask.len(), total);
    assert!(mask.iter().all(|r| r.len() == total));
    let mut start = 0usize;
    for &len in &lengths {
        let end = start + len;
        for i in 0..total {
            for j in 0..total {
                let same = (start..end).contains(&i) && (start..end).contains(&j);
                assert_eq!(mask[i][j], same, "i={i} j={j}");
            }
        }
        start = end;
    }
}

#[test]
fn block_diagonal_forward_no_leakage() {
    let lengths = [1usize, 3, 6, 8];
    let total: usize = lengths.iter().sum();
    let device = Device::default();
    let mask = bool_mask(&block_diagonal_mask(&lengths), &device);
    let block = Block::new(8, 2, &device);
    let x = pattern_input(1, total, 8);
    let base =
        read3(&block.forward_masked(to_tensor(&x, 1, total, 8, &device), Some(mask.clone())));

    // Ganggu satu token di blok 1 (posisi 2, dim 3).
    let mut x2 = x.clone();
    x2[2 * 8 + 3] += 1.0;
    let after = read3(&block.forward_masked(to_tensor(&x2, 1, total, 8, &device), Some(mask)));

    // Blok 2 = posisi 4..10 harus identik bit-demi-bit.
    for pos in 4..10 {
        for dd in 0..8 {
            assert_eq!(
                base[pos * 8 + dd],
                after[pos * 8 + dd],
                "leak pos {pos} dim {dd}"
            );
        }
    }
    // Posisi yang diganggu memang berubah (uji ini bermakna).
    assert_ne!(base[2 * 8 + 3], after[2 * 8 + 3]);
}

// ---------------------------------------------------------------------------
// Mask prefix-LM
// ---------------------------------------------------------------------------

#[test]
fn prefix_lm_mask_structure() {
    let (prefix, target) = (4usize, 6);
    let total = prefix + target;
    let mask = prefix_lm_mask(prefix, target);
    for i in 0..prefix {
        for j in 0..prefix {
            assert!(mask[i][j], "prompt {i}->{j}");
        }
        for j in prefix..total {
            assert!(!mask[i][j], "prompt {i} tak boleh lihat target {j}");
        }
    }
    for i in prefix..total {
        for j in 0..=i {
            assert!(mask[i][j], "target {i}->{j}");
        }
        for j in (i + 1)..total {
            assert!(!mask[i][j], "target {i} tak boleh lihat masa depan {j}");
        }
    }
}

#[test]
fn prefix_lm_forward_no_prompt_leak() {
    let (prefix, target) = (4usize, 6);
    let total = prefix + target;
    let device = Device::default();
    let mask = bool_mask(&prefix_lm_mask(prefix, target), &device);
    let block = Block::new(8, 2, &device);
    let x = pattern_input(1, total, 8);
    let base =
        read3(&block.forward_masked(to_tensor(&x, 1, total, 8, &device), Some(mask.clone())));

    // Ubah token target pertama.
    let mut x2 = x.clone();
    x2[prefix * 8] += 1.0;
    let after = read3(&block.forward_masked(to_tensor(&x2, 1, total, 8, &device), Some(mask)));

    // Posisi prompt tidak boleh berubah.
    for pos in 0..prefix {
        for dd in 0..8 {
            assert_eq!(
                base[pos * 8 + dd],
                after[pos * 8 + dd],
                "prompt leak pos {pos} dim {dd}"
            );
        }
    }
    assert_ne!(base[prefix * 8], after[prefix * 8]);
}

// ---------------------------------------------------------------------------
// RMSNorm / SwiGLU
// ---------------------------------------------------------------------------

#[test]
fn rmsnorm_shape_and_finite() {
    let device = Device::default();
    let norm = RmsNorm::new(8, &device);
    let x = Tensor::<3>::random([2, 3, 8], Distribution::Normal(0.0, 1.0), &device);
    let y = norm.forward(x);
    assert_eq!(y.dims(), [2, 3, 8]);
    assert!(read3(&y).iter().all(|v| v.is_finite()));
}

#[test]
fn rmsnorm_constant_input_unit_scale() {
    let device = Device::default();
    let norm = RmsNorm::new(4, &device);
    let x = to_tensor(&[2.0f32; 4], 1, 1, 4, &device);
    for v in read3(&norm.forward(x)) {
        assert!((v - 1.0).abs() < 1e-4, "got {v}");
    }
}

#[test]
fn swiglu_shape_and_finite() {
    let device = Device::default();
    let mlp = SwiGluMlp::new(8, 16, &device);
    let x = Tensor::<3>::random([2, 3, 8], Distribution::Normal(0.0, 1.0), &device);
    let y = mlp.forward(x);
    assert_eq!(y.dims(), [2, 3, 8]);
    assert!(read3(&y).iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// Attention: panjang variabel + padding
// ---------------------------------------------------------------------------

#[test]
fn attention_variable_length_no_nan() {
    let device = Device::default();
    let block = Block::new(8, 2, &device);
    for t in [1usize, 2, 3, 5, 8] {
        let x = Tensor::<3>::random([1, t, 8], Distribution::Normal(0.0, 1.0), &device);
        let y = block.forward_masked(x, None);
        assert_eq!(y.dims(), [1, t, 8], "t={t}");
        assert!(read3(&y).iter().all(|v| v.is_finite()), "t={t}");
    }
}

#[test]
fn attention_padding_no_nan() {
    let device = Device::default();
    let block = Block::new(8, 2, &device);
    let (t, valid, d) = (8usize, 5usize, 8usize);
    let mut mask = vec![vec![false; t]; t];
    for i in 0..t {
        mask[i][i] = true; // tiap posisi melihat dirinya (cegah baris -inf)
    }
    for i in 0..valid {
        for j in 0..valid {
            mask[i][j] = true; // token valid saling melihat
        }
    }
    let mut data = pattern_input(1, t, d);
    for pos in valid..t {
        for dd in 0..d {
            data[pos * d + dd] = 0.0; // padding = nol
        }
    }
    let y = block.forward_masked(
        to_tensor(&data, 1, t, d, &device),
        Some(bool_mask(&mask, &device)),
    );
    assert_eq!(y.dims(), [1, t, d]);
    assert!(read3(&y).iter().all(|v| v.is_finite()));
}
