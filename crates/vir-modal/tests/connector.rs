//! Tes nyata F6.1/F6.2 untuk konektor beku → prefix: `Linear` (proyeksi) dan
//! `Connector` (MLP `d_in → h → d_out` dengan GELU). Dijalankan job CI
//! `vir-modal`; semua fungsi murni (`Vec<f32>`).

use vir_common::VirError;
use vir_modal::connector::{Connector, Linear, gelu};

fn is_err<T>(r: Result<T, VirError>) -> bool {
    r.is_err()
}

#[test]
fn linear_zeros_forward_is_bias_only() {
    let zeros = Linear::zeros(3, 2).expect("linear");
    assert_eq!((zeros.in_dim, zeros.out_dim), (3, 2));
    assert_eq!(zeros.w, vec![0.0; 6]);
    assert_eq!(zeros.b, vec![0.0; 2]);
    assert_eq!(
        zeros.forward(&[1.0, 2.0, 3.0]).expect("forward"),
        vec![0.0, 0.0]
    );
    assert!(is_err(Linear::zeros(0, 2)));
    assert!(is_err(Linear::zeros(2, 0)));
}

#[test]
fn linear_randomized_is_deterministic_and_scaled() {
    let a = Linear::randomized(42, 3, 2).expect("linear");
    let b = Linear::randomized(42, 3, 2).expect("linear");
    assert_eq!(a.w, b.w);
    let limit = (6.0f32 / 3.0).sqrt();
    assert_eq!(a.w.len(), 6);
    assert!(a.w.iter().all(|v| v.abs() <= limit));

    let out = a.forward(&[1.0, 2.0, 3.0]).expect("forward");
    assert_eq!(out.len(), 2);
    let want0 = a.b[0] + a.w[0] * 1.0 + a.w[1] * 2.0 + a.w[2] * 3.0;
    let want1 = a.b[1] + a.w[3] * 1.0 + a.w[4] * 2.0 + a.w[5] * 3.0;
    assert!((out[0] - want0).abs() < 1e-6);
    assert!((out[1] - want1).abs() < 1e-6);

    let c = Linear::randomized(43, 3, 2).expect("linear");
    assert_ne!(a.w, c.w);
}

#[test]
fn linear_forward_shapes_and_errors() {
    let lin = Linear::randomized(7, 2, 3).expect("linear");
    // 6 elemen / d_in 2 = 3 baris, tiap baris → d_out 3 = 9 keluaran.
    assert_eq!(lin.forward(&[0.0; 6]).expect("forward").len(), 9);
    assert!(lin.forward(&[0.0; 5]).is_err());
}

#[test]
fn connector_zero_weights_give_zero_prefix() {
    let conn = Connector::new(4, 8, 2).expect("connector");
    assert_eq!((conn.d_in(), conn.hidden(), conn.d_out()), (4, 8, 2));
    assert_eq!(conn.fc1.w.len(), 4 * 8);
    assert_eq!(conn.fc2.w.len(), 8 * 2);
    assert_eq!(conn.forward(&[1.0; 8]).expect("forward"), vec![0.0; 4]);
    assert!(is_err(Connector::new(0, 8, 2)));
    assert!(is_err(Connector::new(4, 0, 2)));
    assert!(is_err(Connector::new(4, 8, 0)));
}

#[test]
fn connector_forward_and_rows() {
    let conn = Connector::randomized(9, 8, 16, 4).expect("connector");
    let x = vec![0.3f32; 8 * 5];
    let out = conn.forward(&x).expect("forward");
    assert_eq!(out.len(), 5 * 4);
    assert!(out.iter().all(|v| v.is_finite()));

    let (rows, tokens) = conn.forward_rows(&x).expect("rows");
    assert_eq!(tokens, 5);
    assert_eq!(rows.len(), tokens * conn.d_out());

    let again = Connector::randomized(9, 8, 16, 4).expect("connector");
    assert_eq!(again.forward(&x).expect("forward"), out);
    let other = Connector::randomized(10, 8, 16, 4).expect("connector");
    assert_ne!(other.forward(&x).expect("forward"), out);
    assert!(is_err(conn.forward(&[0.0; 7])));
}

#[test]
fn connector_preserves_token_count() {
    let conn = Connector::randomized(1, 4, 6, 3).expect("connector");
    for n in [1usize, 2, 7] {
        let (rows, tokens) = conn.forward_rows(&vec![0.5f32; 4 * n]).expect("rows");
        assert_eq!(tokens, n);
        assert_eq!(rows.len(), n * 3);
        for i in 0..n {
            assert!(rows[i * 3..(i + 1) * 3].iter().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn gelu_matches_tanh_approximation() {
    assert_eq!(gelu(0.0), 0.0);
    assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-3);
    assert!((gelu(-1.0) + 0.158_655_3).abs() < 1e-3);
    assert!(gelu(10.0) > 9.9);
    assert!(gelu(-10.0) > -1e-4 && gelu(-10.0) < 1e-4);
    assert!(gelu(-3.0) < 0.0);
}
