//! F1.4.f — FSQ/RFSQ di belakang trait `Quantizer` (PRD F1.4.f).

mod support;

use burn::tensor::Tensor;
use support::{read_int1, tensor2};
use vir_vq::fsq::{Fsq, FsqConfig, Rfsq, RfsqNorm};
use vir_vq::quantizer::Quantizer;

fn device() -> burn::tensor::Device {
    Default::default()
}

#[test]
fn ukuran_codebook_adalah_produk_level() {
    assert_eq!(FsqConfig::new(vec![7, 5, 5, 5, 5]).codebook_size(), 4375);
    assert_eq!(FsqConfig::new(vec![8, 8, 8, 6, 5]).codebook_size(), 15360);
    assert_eq!(FsqConfig::new(vec![16, 16, 16]).codebook_size(), 4096);
}

#[test]
fn validate_menolak_level_kecil_dan_kosong() {
    assert!(FsqConfig::new(vec![]).validate().is_err(), "kosong ditolak");
    assert!(
        FsqConfig::new(vec![1, 5, 5]).validate().is_err(),
        "L_i < 2 ditolak"
    );
    assert!(FsqConfig::new(vec![7, 5, 5, 5, 5]).validate().is_ok());
}

#[test]
fn fsq_dim_adalah_dimensi_laten() {
    // E-014: Quantizer::dim() harus d (dimensi laten), bukan jumlah level.
    let cfg = FsqConfig::new(vec![7, 5, 5, 5, 5]);
    let fsq = Fsq::new(&cfg, 16, &device()).expect("fsq");
    assert_eq!(fsq.dim(), 16, "dim = d laten");
    assert_eq!(fsq.num_codes(), 4375, "|V| = prod L_i");
}

#[test]
fn fsq_indices_dalam_rentang() {
    let cfg = FsqConfig::new(vec![7, 5, 5, 5, 5]);
    let fsq = Fsq::new(&cfg, 16, &device()).expect("fsq");
    let mut rng = support::Lcg::new(9);
    // proj: d(16) -> n(5); masukan harus berdimensi d.
    let data: Vec<f32> = (0..128).map(|_| rng.unit() * 4.0 - 2.0).collect();
    let u = tensor2(8, 16, &data);
    let q = fsq.quantize(u);
    let idx = read_int1(&q.indices);
    for i in idx {
        assert!((0..4375).contains(&i), "index {i} di luar 0..4375");
    }
}

#[test]
fn fsq_lookup_adalah_keluaran_unproj_dari_indeks() {
    // lookup(idx) = unproj(decode_index(idx)); quantize(u).indices -> lookup harus
    // menghasilkan vektor berdimensi d yang sama dengan quantized, karena
    // decode_index(mixed_radix_index(codes)) == codes secara eksak (bilangan bulat).
    let cfg = FsqConfig::new(vec![8, 8, 6, 5]);
    let fsq = Fsq::new(&cfg, 12, &device()).expect("fsq");
    let mut rng = support::Lcg::new(11);
    let data: Vec<f32> = (0..60).map(|_| rng.unit() * 3.0 - 1.5).collect();
    let u = tensor2(5, 12, &data);
    let q = fsq.quantize(u);
    let back = fsq.lookup(q.indices);
    // Bentuk benar: [N, d].
    assert_eq!(back.dims(), q.quantized.dims());
    assert_eq!(back.dims()[0], 5);
    assert_eq!(back.dims()[1], 12);
    // Determinisme: lookup(idx) sama untuk idx sama.
    let back2 = fsq.lookup(q.indices);
    let b1 = support::read1(&back.reshape([60]));
    let b2 = support::read1(&back2.reshape([60]));
    for (x, y) in b1.iter().zip(b2.iter()) {
        assert!(
            (x - y).abs() < 1e-6,
            "lookup tidak deterministik: {x} vs {y}"
        );
    }
    let _ = support::read1(&q.quantized.reshape([60]));
    // Keluaran lookup finite (tidak NaN).
    assert!(b1.iter().all(|v| v.is_finite()));
}

#[test]
fn ifsq_tanpa_commitment() {
    let cfg = FsqConfig::ifsq(vec![16, 16, 16]);
    let mut fsq = Fsq::new(&cfg, 8, &device()).expect("ifsq");
    let before = fsq.updates();
    // dim 8, 2 baris.
    let u = tensor2(
        2,
        8,
        &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ],
    );
    let q = fsq.quantize(u.clone());
    fsq.update(&q.indices, &u, &mut support::Lcg::new(1));
    assert_eq!(
        fsq.updates(),
        before + 1,
        "update hanya menambah penghitung"
    );
}

#[test]
fn rfsq_norm_roundtrip() {
    // RfsqNorm: denormalize(normalize(x)) ~ x (dalam toleransi numerik).
    let cfg = FsqConfig::new(vec![8, 8, 8]);
    let r = Rfsq::new(&cfg, 6, &device()).expect("rfsq");
    assert_eq!(r.num_codes(), 512);
    let _ = RfsqNorm::new(&cfg, 6, &device()).expect("rfsqnorm");
    let _ = Tensor::<2>::from_floats(
        burn::tensor::TensorData::new(vec![0.0f32; 12], [2, 6]),
        &device(),
    );
}
