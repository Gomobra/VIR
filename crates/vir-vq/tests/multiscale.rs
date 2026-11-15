//! F1.4.b — operator Down_k/R_k, rekonstruksi kumulatif (PRD F1.4.b).

mod support;

use burn::tensor::{Tensor, TensorData};
use support::read2;
use vir_vq::multiscale::ScaleOps;

fn device() -> burn::tensor::Device {
    Default::default()
}

fn tensor3(b: usize, l: usize, d: usize, data: &[f32]) -> Tensor<3> {
    Tensor::<3>::from_floats(TensorData::new(data.to_vec(), [b, l, d]), &device())
}

#[test]
fn scale_ops_identitas_ketika_panjang_sama() {
    let ops = ScaleOps::new(4, 4, &device());
    assert_eq!(ops.in_len(), 4);
    assert_eq!(ops.out_len(), 4);
    let x = tensor3(1, 4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let d = ops.down(x.clone());
    let up = ops.up(d);
    let orig = support::read1(&x.reshape([8]));
    let back = support::read1(&up.reshape([8]));
    for (a, b) in orig.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-4, "identitas down/up: {a} vs {b}");
    }
}

#[test]
fn down_mengecilkan_panjang() {
    let ops = ScaleOps::new(4, 2, &device());
    let x = tensor3(1, 4, 1, &[1.0, 2.0, 3.0, 4.0]);
    let d = ops.down(x);
    assert_eq!(d.dims()[1], 2, "down 4 -> 2");
}

#[test]
fn up_memperbesar_panjang() {
    let ops = ScaleOps::new(4, 2, &device());
    let x = tensor3(1, 2, 1, &[1.0, 3.0]);
    let u = ops.up(x);
    assert_eq!(u.dims()[1], 4, "up 2 -> 4");
}

#[test]
fn broadcast_dari_satu_posisi() {
    // E-013: in_len == 1 -> interp harus menghasilkan [1, ℓ] (bukan [1,1] yang
    // membuat ScaleOps::new(1, ℓ) panik di TensorData).
    let ops = ScaleOps::new(1, 4, &device());
    let x = tensor3(1, 1, 1, &[5.0]);
    let u = ops.up(x);
    let vals = support::read1(&u.reshape([4]));
    assert_eq!(vals.len(), 4);
    for v in &vals {
        assert!((v - 5.0).abs() < 1e-4, "broadcast 5 -> {v}");
    }
}

#[test]
fn down_average_pooling_numerik() {
    // L=4, ℓ=2: blok [0,1] dan [2,3]; rata-rata 1,2 -> 1.5; 3,4 -> 3.5.
    let ops = ScaleOps::new(4, 2, &device());
    let x = tensor3(1, 4, 1, &[1.0, 2.0, 3.0, 4.0]);
    let d = support::read1(&ops.down(x).reshape([2]));
    assert!((d[0] - 1.5).abs() < 1e-4, "blok0 {}", d[0]);
    assert!((d[1] - 3.5).abs() < 1e-4, "blok1 {}", d[1]);
}
