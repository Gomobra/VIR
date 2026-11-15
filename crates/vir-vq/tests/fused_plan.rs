//! F7.1 — sisi host kernel terfusi: rencana luncur dan cadangan argmin
//! (PRD F7.1). Semua CPU, tak butuh CUDA.

use burn::tensor::{Int, Tensor, TensorData};
use vir_vq::codebook::Codebook;
use vir_vq::fused_plan::{ROWS_PER_CUBE, naive_argmin, output_shape, plan_launch};

fn device() -> burn::tensor::Device {
    Default::default()
}

fn read_int(t: &Tensor<1, Int>) -> Vec<i32> {
    t.to_data().try_to_vec::<i32>().expect("i32")
}

fn cb_4x2() -> Codebook {
    let mut cb = Codebook::new(
        &vir_vq::codebook::CodebookConfig {
            num_entries: 4,
            dim: 2,
            ema_decay: 0.99,
            commitment_cost: 0.25,
            restart_threshold: 1.0,
            eps: 1e-5,
        },
        &device(),
    );
    cb.set_entries(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("set");
    cb
}

#[test]
fn plan_launch_membulatkan_ke_kelipatan_rows_per_cube() {
    let p = plan_launch(1);
    assert_eq!(p.rows_per_cube, ROWS_PER_CUBE);
    assert!(p.rows_per_launch >= ROWS_PER_CUBE);
    assert!(p.rows_per_launch % ROWS_PER_CUBE == 0);
    assert_eq!(p.launches, 1);
}

#[test]
fn plan_launch_banyak_baris_beberapa_peluncuran() {
    // rows = 3 * ROWS_PER_CUBE -> tepat 3 blok.
    let rows = 3 * ROWS_PER_CUBE;
    let p = plan_launch(rows);
    assert_eq!(p.rows_per_launch, rows);
    assert_eq!(p.launches, 1, "{rows} baris cukup satu peluncuran");
}

#[test]
fn output_shape_satu_indeks_per_baris() {
    let z = Tensor::<2>::from_floats(TensorData::new(vec![0.0f32; 6], [3, 2]), &device());
    let s = output_shape(&z.shape());
    assert_eq!(s.as_slice(), &[3]);
}

#[test]
fn naive_argmin_sama_dengan_quantize_reference() {
    let cb = cb_4x2();
    let z = Tensor::<2>::from_floats(
        TensorData::new(vec![0.1f32, 0.1, 0.9, 0.9, 0.4, 0.6], [3, 2]),
        &device(),
    );
    let naive = read_int(&naive_argmin(
        z.clone(),
        cb.weight.val(),
        Tensor::<1>::from_floats(TensorData::new(vec![0.0f32; 4], [4]), &device()),
    ));
    let reference = read_int(&cb.quantize_idx_reference(z));
    assert_eq!(naive, reference, "kernel setara matmul+argmax");
}

#[test]
fn naive_argmin_memilih_terdekat() {
    let cb = cb_4x2();
    // bias 0; [0.1,0.1]->0, [0.9,0.9]->3.
    let z = Tensor::<2>::from_floats(
        TensorData::new(vec![0.1f32, 0.1, 0.9, 0.9], [2, 2]),
        &device(),
    );
    let bias = neg_half_sq_norm(&cb);
    let got = read_int(&naive_argmin(z, cb.weight.val(), bias));
    assert_eq!(got, vec![0, 3]);
}

#[test]
fn is_cube_device_false_on_cpu() {
    // Pada build CPU (tanpa cuda) tidak ada device CubeCL.
    assert!(!vir_vq::fused_plan::is_cube_device(&device()));
}
