//! Tes nyata Muon Newton–Schulz (PRD F1.7 langkah 4–5).
//!
//! PRD langkah 5: `f(1)=a+b+c=0,701`, jadi hasil NS **bukan** ortogonal eksak;
//! untuk 768×3072 nilai singular berada di [0,68; 1,13]. Karena itu tes memakai
//! toleransi, bukan kesamaan dengan matriks ortogonal. Nilai singular dihitung
//! dari Gram matriks pada dimensi terkecil lewat rotasi Jacobi (host, f64).

use vir_train::{
    MUON_SCALE, Muon, MuonParams, NS_COEFFS, NS_STEPS, ParamRole, ParamSlot, Rng, newton_schulz,
};

/// Matriks `a` (`m×k`) kali `b` (`k×n`), row-major.
fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Transpos `m×n` → `n×m`.
fn transpose(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

/// Gram pada dimensi terkecil: `X Xᵀ` bila `rows ≤ cols`, selain itu `XᵀX`.
fn gram_small(x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, usize) {
    let xt = transpose(x, rows, cols);
    if rows <= cols {
        (matmul(x, rows, cols, &xt, rows), rows)
    } else {
        (matmul(&xt, cols, rows, x, cols), cols)
    }
}

/// Satu rotasi Jacobi pada kolom/baris `p,q`.
fn jacobi_rotate(m: &mut [f64], n: usize, p: usize, q: usize, c: f64, s: f64) {
    for k in 0..n {
        let mkp = m[k * n + p];
        let mkq = m[k * n + q];
        m[k * n + p] = c * mkp - s * mkq;
        m[k * n + q] = s * mkp + c * mkq;
    }
    for k in 0..n {
        let mpk = m[p * n + k];
        let mqk = m[q * n + k];
        m[p * n + k] = c * mpk - s * mqk;
        m[q * n + k] = s * mpk + c * mqk;
    }
    m[p * n + q] = 0.0;
    m[q * n + p] = 0.0;
}

/// Nilai eigen matriks simetris `n×n` (Jacobi sederhana, f64).
fn jacobi_eigenvalues(a: &[f32], n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n * n {
        m[i] = a[i] as f64;
    }
    for _ in 0..64 {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[i * n + j].abs();
            }
        }
        if off < 1e-10 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-14 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                jacobi_rotate(&mut m, n, p, q, c, t * c);
            }
        }
    }
    (0..n).map(|i| m[i * n + i]).collect()
}

/// Gauss standar dari dua seragam (Box–Muller).
fn randn(rng: &mut Rng) -> f32 {
    let u1 = rng.next_f32().max(1e-7);
    let u2 = rng.next_f32();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Matriks Gaussian `rows×cols`.
fn random_matrix(rng: &mut Rng, rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|_| randn(rng)).collect()
}

/// Norma L2 tiap baris.
fn row_norms(x: &[f32], _rows: usize, cols: usize) -> Vec<f32> {
    x.chunks(cols)
        .map(|r| r.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

/// Nilai singular dari Gram dimensi kecil.
fn singular_values(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let (g, n) = gram_small(x, rows, cols);
    let mut vals: Vec<f64> = jacobi_eigenvalues(&g, n)
        .into_iter()
        .map(|v| v.max(0.0).sqrt())
        .collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    vals
}

#[test]
fn ns_coefficients_match_prd_constant() {
    let (a, b, c) = NS_COEFFS;
    assert!((a + b + c - 0.701).abs() < 1e-3, "a+b+c = {}", a + b + c);
    assert_eq!(NS_STEPS, 5, "PRD: lima iterasi Newton–Schulz");
}

#[test]
fn ns_tall_24x96_singular_values_are_near_one_but_not_exact() {
    let mut rng = Rng::new(5);
    let (rows, cols) = (24usize, 96usize);
    let mut x = random_matrix(&mut rng, rows, cols);
    newton_schulz(&mut x, rows, cols);

    let sig = singular_values(&x, rows, cols);
    assert_eq!(sig.len(), rows);
    for s in &sig {
        assert!(
            *s > 0.5 && *s < 1.3,
            "nilai singular {s} di luar [0,5; 1,3]"
        );
    }
    let mean = sig.iter().sum::<f64>() / sig.len() as f64;
    assert!(
        (0.6..1.15).contains(&mean),
        "rata-rata nilai singular {mean} di luar dugaan"
    );
    assert!(
        sig[0] < 0.99,
        "PRD f(1)=0,701: hasil NS tidak boleh ortogonal eksak"
    );
}

#[test]
fn ns_square_32x32_singular_values_are_near_one_but_not_exact() {
    let mut rng = Rng::new(9);
    let (rows, cols) = (32usize, 32usize);
    let mut x = random_matrix(&mut rng, rows, cols);
    newton_schulz(&mut x, rows, cols);

    let sig = singular_values(&x, rows, cols);
    assert_eq!(sig.len(), rows);
    for s in &sig {
        assert!(
            *s > 0.2 && *s < 1.3,
            "nilai singular {s} di luar [0,2; 1,3]"
        );
    }
    let mean = sig.iter().sum::<f64>() / sig.len() as f64;
    assert!((0.5..1.15).contains(&mean), "rata-rata {mean}");
    assert!(
        sig.iter().any(|s| (s - 1.0).abs() > 1e-3),
        "hasil NS terlalu mendekati ortogonal eksak"
    );
}

#[test]
fn ns_768x3072_rows_are_orthonormal_within_tolerance() {
    let mut rng = Rng::new(17);
    let (rows, cols) = (768usize, 3072usize);
    let mut x = random_matrix(&mut rng, rows, cols);
    newton_schulz(&mut x, rows, cols);

    let norms = row_norms(&x, rows, cols);
    assert!(norms.iter().all(|v| v.is_finite()), "ada norma non-finite");
    for (i, n) in norms.iter().enumerate() {
        assert!(*n > 0.4 && *n < 1.4, "norma baris {i} = {n}");
    }
    let mean = norms.iter().sum::<f32>() / norms.len() as f32;
    assert!(
        (0.6..1.2).contains(&mean),
        "rata-rata norma baris {mean} di luar [0,6; 1,2]"
    );
}

#[test]
fn ns_768x768_stays_finite_and_not_exactly_orthogonal() {
    let mut rng = Rng::new(23);
    let (rows, cols) = (768usize, 768usize);
    let mut x = random_matrix(&mut rng, rows, cols);
    newton_schulz(&mut x, rows, cols);

    let norms = row_norms(&x, rows, cols);
    assert!(norms.iter().all(|v| v.is_finite()), "ada norma non-finite");
    assert!(
        norms.iter().all(|v| *v < 1.5),
        "norma baris melebihi 1,5 (NS gagal menormalkan)"
    );
    let mean = norms.iter().sum::<f32>() / norms.len() as f32;
    assert!(
        mean < 0.98,
        "PRD: 768×768 acak tidak ortogonal eksak (mean {mean})"
    );
}

#[test]
fn muon_step_rms_matches_adamw_scale() {
    let mut slot = ParamSlot::new(
        "core.q.weight",
        32,
        16,
        ParamRole::Generator,
        vec![0.1; 32 * 16],
    );
    let mut seed = 99u64;
    for g in slot.grad.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *g = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
    }
    let before = slot.data.clone();
    let mut muon = Muon::new(1, &MuonParams::default());
    assert!(muon.step(0, &mut slot, 1e-3, 0.1), "Muon menolak slot");

    let delta: Vec<f32> = slot
        .data
        .iter()
        .zip(before.iter())
        .map(|(a, b)| a - b)
        .collect();
    let rms = (delta.iter().map(|d| d * d).sum::<f32>() / delta.len() as f32).sqrt();
    let expect = 1e-3 * MUON_SCALE;
    assert!(
        rms > expect * 0.55 && rms < expect * 1.7,
        "RMS update Muon {rms} di luar dugaan {expect}"
    );

    slot.frozen = true;
    assert!(
        !muon.step(0, &mut slot, 1e-3, 0.1),
        "slot beku harus dilewati"
    );
}
