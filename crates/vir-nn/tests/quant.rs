//! Tes PTQ nyata F7.2 untuk `vir-nn`: int8/int4/fp8 (E4M3), per-tensor dan
//! per-blok, ekspor → impor (round-trip), serta tabel kualitas vs ukuran.
//! Dijalankan oleh job CI `vir-nn`
//! (`cargo test --manifest-path crates/vir-nn/Cargo.toml`).

use vir_nn::quant::{
    QuantGranularity, QuantScheme, dequantize_tensor, export_quantized, load_quantized, mse,
    quality_table, quantize_tensor,
};

/// PRNG deterministik (LCG) — data uji tanpa dependensi eksternal.
fn pseudo_random(n: usize) -> Vec<f32> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push((state >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0);
    }
    out
}

/// Direktori ekspor sementara unik per pemanggilan.
fn temp_quant_dir(tag: &str) -> String {
    let dir = std::env::temp_dir().join(format!("vir_nn_quant_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir.to_str().expect("path utf8").to_string()
}

#[test]
fn int8_dequant_close_and_smaller() {
    let x = pseudo_random(1024);
    let qt = quantize_tensor(&x, QuantScheme::Int8, QuantGranularity::PerTensor).expect("quant");
    let deq = dequantize_tensor(&qt).expect("dequant");
    let m = mse(&x, &deq);
    assert!(m < 1e-3, "MSE int8 terlalu besar: {m}");
    let qbytes = qt.packed.len() + qt.scales.len() * 4;
    assert!(qbytes < x.len() * 4, "int8 harus lebih kecil dari f32");
    assert_eq!(qbytes, x.len() + 4);
}

#[test]
fn int4_smaller_with_reasonable_mse() {
    let x = pseudo_random(1024);
    let q8 = quantize_tensor(&x, QuantScheme::Int8, QuantGranularity::PerTensor).expect("q8");
    let q4 = quantize_tensor(&x, QuantScheme::Int4, QuantGranularity::PerTensor).expect("q4");
    let deq = dequantize_tensor(&q4).expect("dequant");
    let m = mse(&x, &deq);
    assert!(m < 0.02, "MSE int4 terlalu besar: {m}");
    let b4 = q4.packed.len() + q4.scales.len() * 4;
    let b8 = q8.packed.len() + q8.scales.len() * 4;
    assert!(b4 < b8, "int4 ({b4}) harus lebih kecil dari int8 ({b8})");
    assert_eq!(b4, x.len() / 2 + 4, "int4: 2 nilai/byte + 1 skala");
}

#[test]
fn fp8_e4m3_finite_and_reasonable() {
    let x = pseudo_random(1024);
    let qt = quantize_tensor(&x, QuantScheme::Fp8, QuantGranularity::PerTensor).expect("quant");
    let deq = dequantize_tensor(&qt).expect("dequant");
    assert!(deq.iter().all(|v| v.is_finite()), "fp8 harus finite");
    let m = mse(&x, &deq);
    assert!(m < 0.01, "MSE fp8 terlalu besar: {m}");
    // fp8 E4M3 pada rentang [-1,1] dengan skala ~1/448: galat ~magnitudo
    // kuantisasi, harus lebih kecil dari int4.
    let m4 = {
        let q4 = quantize_tensor(&x, QuantScheme::Int4, QuantGranularity::PerTensor).unwrap();
        mse(&x, &dequantize_tensor(&q4).unwrap())
    };
    assert!(m < m4, "fp8 ({m}) harus lebih akurat dari int4 ({m4})");
}

#[test]
fn fp8_e4m3_exact_for_powers_of_two() {
    // Nilai pangkat-dua tepat representabel di E4M3; dengan puncak 4.0,
    // skala = 4/448 dan tiap x/scale = x·112 juga pangkat-dua → hasil
    // dekuantisasi harus sama persis dengan input (galat < 1e-6).
    let x = [4.0f32, 2.0, 1.0, 0.5, -2.0];
    let qt = quantize_tensor(&x, QuantScheme::Fp8, QuantGranularity::PerTensor).expect("quant");
    let deq = dequantize_tensor(&qt).expect("dequant");
    for (a, b) in x.iter().zip(deq.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[test]
fn fp8_e4m3_saturates_outliers() {
    // Nilai ekstrem harus saturasi ke 448·skala, bukan NaN.
    let x = [1e9f32, -1e9, 1.0];
    let qt = quantize_tensor(&x, QuantScheme::Fp8, QuantGranularity::PerTensor).expect("quant");
    let deq = dequantize_tensor(&qt).expect("dequant");
    assert!(deq.iter().all(|v| v.is_finite()));
    assert!((deq[0] - 448.0 * qt.scales[0]).abs() < 1e-3);
}

#[test]
fn per_block_scales_improve_accuracy() {
    // Data dengan skala berbeda per blok: per-blok harus jauh lebih akurat.
    let mut x = Vec::with_capacity(512);
    for blk in 0..16 {
        let amp = if blk % 2 == 0 { 1.0 } else { 0.01 };
        for i in 0..32 {
            x.push(amp * ((i as f32 / 32.0) * 2.0 - 1.0));
        }
    }
    let pt = quantize_tensor(&x, QuantScheme::Int4, QuantGranularity::PerTensor).expect("pt");
    let pb = quantize_tensor(
        &x,
        QuantScheme::Int4,
        QuantGranularity::PerBlock { block_size: 32 },
    )
    .expect("pb");
    let m_pt = mse(&x, &dequantize_tensor(&pt).expect("d1"));
    let m_pb = mse(&x, &dequantize_tensor(&pb).expect("d2"));
    assert!(
        m_pb < m_pt / 10.0,
        "per-blok ({m_pb}) harus jauh lebih akurat dari per-tensor ({m_pt})"
    );
    assert_eq!(pb.scales.len(), 16);
}

#[test]
fn export_import_roundtrip_identical() {
    let x = pseudo_random(768);
    for scheme in [QuantScheme::Int8, QuantScheme::Int4, QuantScheme::Fp8] {
        let qt = quantize_tensor(&x, scheme, QuantGranularity::PerTensor).expect("quant");
        let dir = temp_quant_dir("roundtrip");
        export_quantized(&qt, &dir, None).expect("export");
        let back = load_quantized(&dir).expect("import");
        let direct = dequantize_tensor(&qt).expect("dequant");
        assert_eq!(
            back, direct,
            "round-trip harus identik bit demi bit ({scheme:?})"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn quality_vs_size_table_printed() {
    // Tabel kualitas vs ukuran untuk blob 4096 nilai (gaya laporan F7.2).
    let x = pseudo_random(4096);
    let rows = quality_table(&x);
    println!("{:>10} {:>10} {:>10}", "skema", "bytes", "mse");
    let f32bytes = x.len() * 4;
    for (name, bytes, m) in &rows {
        println!("{name:>10} {bytes:>10} {m:>10.6}");
        assert!(bytes <= &f32bytes, "{name} tidak boleh melebihi f32");
    }
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].0, "float32");
    assert!(rows[1].2 < rows[2].2, "int8 harus lebih akurat dari int4");
}

#[test]
fn quantize_rejects_empty() {
    assert!(quantize_tensor(&[], QuantScheme::Int8, QuantGranularity::PerTensor).is_err());
}

/// Paritas dengan PTQ bawaan Burn (Flex): skala & hasil Q8S harus identik
/// dengan `Tensor::quantize_dynamic` pada skema Burn yang setara.
/// Diabaikan bawaan; jalankan `cargo test -- --ignored` di runner yang
/// memiliki backend Flex (default crate ini).
#[test]
#[ignore = "pembanding Burn (butuh backend Flex)"]
fn burn_parity_int8() {
    use burn::prelude::*;
    use burn::tensor::quantization::{QuantScheme as BurnScheme, QuantValue};

    let x = pseudo_random(1024);
    let ours = quantize_tensor(&x, QuantScheme::Int8, QuantGranularity::PerTensor).expect("ours");
    let ours_deq = dequantize_tensor(&ours).expect("deq");

    let device = Device::default();
    let t: Tensor<1> =
        Tensor::from_floats(burn::tensor::TensorData::new(x.clone(), [x.len()]), &device);
    let scheme = BurnScheme::default().with_value(QuantValue::Q8S);
    let burn_q = t.quantize_dynamic(&scheme);
    let burn_deq: Vec<f32> = burn_q
        .dequantize()
        .to_data()
        .try_to_vec::<f32>()
        .expect("burn deq");

    assert_eq!(ours.scales.len(), 1);
    let burn_mse = mse(&x, &burn_deq);
    let ours_mse = mse(&x, &ours_deq);
    assert!(
        (burn_mse - ours_mse).abs() < 1e-9,
        "paritas Burn: burn={burn_mse} ours={ours_mse}"
    );
}
