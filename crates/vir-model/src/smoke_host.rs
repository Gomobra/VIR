//! Utilitas latihan sisi host untuk gerbang mini (PRD F1.8).
//!
//! `vir-model` tidak menarik feature `train` Burn (optimizer ada di
//! `vir-train`, F1.7), jadi gerbang mini melatih **kepala linear** tiap level di
//! host. Dua jalur:
//!
//! 1. [`fit_heads`] — Adam batch penuh (dipakai `full_smoke` lama).
//! 2. [`fit_heads_exact`] — kuadrat terkecil ber-ridge (dipakai irisan mini):
//!    menyelesaikan `X·W = Y` secara langsung, sehingga argmax logit sama
//!    persis dengan kode target (memorisasi 8 sampel, F1.8 gerbang G1).
//!
//! Konvensi bobot mengikuti Burn 0.22: `Linear.weight` `[d_in, d_out]`,
//! `logits = x·W`. Semua fungsi bebas `unwrap` (R3).

use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use vir_common::VirError;
use vir_vq::Decoder;
use vir_vq::quantizer::indices_to_vec;

use crate::generator::{Generator, replace_head};

/// Laju belajar Adam host.
const HEAD_LR: f32 = 0.02;
/// Ridge untuk kuadrat terkecil (menjaga `X·Xᵀ + λI` dapat dibalik).
const RIDGE: f32 = 1.0e-4;
/// Toleransi residual maksimum agar argmax dianggap tepat.
const RESIDUAL_TOL: f32 = 5.0e-2;

/// Akurasi argmax rata-rata atas semua level dan posisi.
pub fn accuracy(logits: &[Tensor<3>], targets: &[Tensor<2, Int>]) -> f32 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for (lg, target) in logits.iter().zip(targets.iter()) {
        let [b, l, _] = lg.dims();
        let pred = indices_to_vec(&lg.clone().argmax(2).reshape([b * l]));
        let want = indices_to_vec(&target.clone().reshape([b * l]));
        for (p, w) in pred.iter().zip(want.iter()) {
            total += 1;
            if p == w {
                hit += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hit as f32 / total as f32
    }
}

/// Token hasil dekode `ẑ` lewat decoder tokenizer.
pub fn decode_tokens(decoder: &Decoder, z: &Tensor<3>) -> Vec<u32> {
    let tokens = decoder.argmax_tokens(z.clone());
    let [b, l] = tokens.dims();
    indices_to_vec(&tokens.reshape([b * l]))
        .into_iter()
        .map(|v| v.max(0) as u32)
        .collect()
}

/// Adam mini di host (tanpa feature `train`).
#[derive(Debug, Clone)]
struct HostAdam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
    lr: f32,
}

impl HostAdam {
    fn new(len: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
            t: 0,
            lr,
        }
    }

    /// Satu langkah Adam (`β₁ = 0,9`, `β₂ = 0,999`).
    fn step(&mut self, w: &mut [f32], grad: &[f32]) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        self.t = self.t.saturating_add(1);
        let c1 = 1.0 - B1.powi(self.t as i32);
        let c2 = 1.0 - B2.powi(self.t as i32);
        for i in 0..w.len() {
            self.m[i] = B1 * self.m[i] + (1.0 - B1) * grad[i];
            self.v[i] = B2 * self.v[i] + (1.0 - B2) * grad[i] * grad[i];
            let mh = self.m[i] / c1;
            let vh = self.v[i] / c2;
            w[i] -= self.lr * mh / (vh.sqrt() + EPS);
        }
    }
}

/// Softmax in-place satu baris logit.
fn softmax_in_place(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Latih satu kepala (`logits = x·W`, `W` `[h, V]`) dengan Adam batch penuh.
fn fit_head(x: &[f32], targets: &[i64], w: &mut [f32], vocab: usize, steps: usize) -> f32 {
    let rows = targets.len();
    let hidden = if rows == 0 { 0 } else { x.len() / rows };
    if rows == 0 || hidden == 0 || vocab == 0 || w.len() != hidden * vocab {
        return f32::NAN;
    }
    let mut opt = HostAdam::new(w.len(), HEAD_LR);
    let mut probs = vec![0.0f32; vocab];
    let mut grad = vec![0.0f32; w.len()];
    let mut loss = 0.0f32;
    for _ in 0..steps {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        loss = 0.0;
        for row in 0..rows {
            let xr = &x[row * hidden..(row + 1) * hidden];
            for v in 0..vocab {
                let mut z = 0.0f32;
                for j in 0..hidden {
                    z += w[j * vocab + v] * xr[j];
                }
                probs[v] = z;
            }
            softmax_in_place(&mut probs);
            let target = targets[row];
            if target >= 0 && (target as usize) < vocab {
                loss -= probs[target as usize].max(1e-9).ln();
            }
            for v in 0..vocab {
                let p = probs[v] - if v as i64 == target { 1.0 } else { 0.0 };
                if p == 0.0 {
                    continue;
                }
                let scale = p / rows as f32;
                for j in 0..hidden {
                    grad[j * vocab + v] += scale * xr[j];
                }
            }
        }
        opt.step(w, &grad);
    }
    loss / rows as f32
}

/// Latih kepala semua level (Adam) dari fitur; kembalikan CE per level.
pub fn fit_heads_from_features(
    model: &mut Generator,
    features: &[Tensor<3>],
    targets: &[Tensor<2, Int>],
    steps: usize,
) -> Result<Vec<f32>, VirError> {
    let n = features.len().min(targets.len()).min(model.heads.len());
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let [b, l, h] = features[k].dims();
        let rows = b * l;
        let vocab = model.heads[k].weight.val().dims()[1];
        let x = read_f32(&features[k].clone().reshape([rows, h]));
        let want = read_i64(&targets[k].clone().reshape([rows]));
        if x.len() != rows * h || want.len() != rows {
            return Err(VirError::Internal(format!(
                "data kepala level {k} tidak dapat dibaca"
            )));
        }
        let device = model.heads[k].weight.val().device();
        let mut w = read_f32(&model.heads[k].weight.val());
        out.push(fit_head(&x, &want, &mut w, vocab, steps));
        replace_head(&mut model.heads[k], w, h, vocab, &device)?;
    }
    Ok(out)
}

/// Latih kepala dari masukan teacher (fitur dihitung dari `model.scales`).
pub fn fit_heads(
    model: &mut Generator,
    inputs: &[Tensor<3>],
    targets: &[Tensor<2, Int>],
    steps: usize,
) -> Result<Vec<f32>, VirError> {
    let features = model.forward_features(inputs, model.num_loops());
    fit_heads_from_features(model, &features, targets, steps)
}

/// Selesaikan `A·X = B` (Gauss–Jordan, pivot parsial). `A` `n×n`, `B` `n×m`.
fn solve_multi(a: &mut [f64], b: &mut [f64], n: usize, m: usize) -> bool {
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if a[piv * n + col].abs() < 1e-12 {
            return false;
        }
        if piv != col {
            for j in 0..n {
                a.swap(piv * n + j, col * n + j);
            }
            for j in 0..m {
                b.swap(piv * m + j, col * m + j);
            }
        }
        let d = a[col * n + col];
        for j in col..n {
            a[col * n + j] /= d;
        }
        for j in 0..m {
            b[col * m + j] /= d;
        }
        for r in 0..n {
            let f = a[r * n + col];
            if r == col || f == 0.0 {
                continue;
            }
            for j in col..n {
                a[r * n + j] -= f * a[col * n + j];
            }
            for j in 0..m {
                b[r * m + j] -= f * b[col * m + j];
            }
        }
    }
    true
}

/// Kuadrat terkecil ber-ridge: `W` `[h, V]` meminimumkan `‖X·W − Y‖² + λ‖W‖²`.
///
/// `x` `[n, h]`, `y` `[n, v]` (baris-major). Mengembalikan `W` `[h, v]`
/// (layout Burn). Diselesaikan dalam `f64` agar residual kecil; `None` bila
/// dimensi tidak konsisten atau sistem tidak dapat dibalik.
pub fn solve_lstsq(x: &[f32], y: &[f32], n: usize, h: usize, v: usize) -> Option<Vec<f32>> {
    if n == 0 || h == 0 || v == 0 || x.len() != n * h || y.len() != n * v {
        return None;
    }
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f64;
            for t in 0..h {
                s += f64::from(x[i * h + t]) * f64::from(x[j * h + t]);
            }
            a[i * n + j] = s + if i == j { f64::from(RIDGE) } else { 0.0 };
        }
    }
    let mut b = vec![0.0f64; n * v];
    for i in 0..n {
        for c in 0..v {
            b[i * v + c] = f64::from(y[i * v + c]);
        }
    }
    if !solve_multi(&mut a, &mut b, n, v) {
        return None;
    }
    let mut w = vec![0.0f64; h * v];
    for i in 0..n {
        for j in 0..h {
            let xv = f64::from(x[i * h + j]);
            if xv == 0.0 {
                continue;
            }
            for c in 0..v {
                w[j * v + c] += xv * b[i * v + c];
            }
        }
    }
    Some(w.into_iter().map(|v| v as f32).collect())
}

/// Residual maksimum `‖X·W − Y‖∞` (untuk memastikan argmax tepat).
fn max_residual(x: &[f32], w: &[f32], y: &[f32], n: usize, h: usize, v: usize) -> f32 {
    let mut worst = 0.0f32;
    for i in 0..n {
        for c in 0..v {
            let mut z = 0.0f32;
            for j in 0..h {
                z += x[i * h + j] * w[j * v + c];
            }
            worst = worst.max((z - y[i * v + c]).abs());
        }
    }
    worst
}

/// Kepala kuadrat terkecil dari fitur; CE per level setelah pemasangan.
///
/// Menolak bila residual argmax terlalu besar (fitur tidak berpangkat penuh),
/// sehingga kegagalan gerbang terlihat sebagai galat, bukan diam-diam.
pub fn fit_heads_exact(
    model: &mut Generator,
    features: &[Tensor<3>],
    targets: &[Tensor<2, Int>],
) -> Result<Vec<f32>, VirError> {
    let n = features.len().min(targets.len()).min(model.heads.len());
    let mut ce = Vec::with_capacity(n);
    for k in 0..n {
        let [b, l, h] = features[k].dims();
        let rows = b * l;
        let vocab = model.heads[k].weight.val().dims()[1];
        let x = read_f32(&features[k].clone().reshape([rows, h]));
        let want = read_i64(&targets[k].clone().reshape([rows]));
        if x.len() != rows * h || want.len() != rows {
            return Err(VirError::Internal(format!(
                "data kepala level {k} tidak dapat dibaca"
            )));
        }
        let y = one_hot(&want, vocab);
        let w = solve_lstsq(&x, &y, rows, h, vocab).ok_or_else(|| {
            VirError::Internal(format!("kepala level {k} tidak dapat diselesaikan"))
        })?;
        let resid = max_residual(&x, &w, &y, rows, h, vocab);
        if resid > RESIDUAL_TOL {
            return Err(VirError::Internal(format!(
                "kepala level {k} tidak dapat memorfisasi (residual {resid})"
            )));
        }
        let device = model.heads[k].weight.val().device();
        replace_head(&mut model.heads[k], w, h, vocab, &device)?;
        ce.push(ce_level(model, k, features, targets));
    }
    Ok(ce)
}

/// CE satu level (tanpa bobot) dari logits Burn.
fn ce_level(
    model: &Generator,
    k: usize,
    features: &[Tensor<3>],
    targets: &[Tensor<2, Int>],
) -> f32 {
    let logits = model.heads[k].forward(features[k].clone());
    let [b, l, v] = logits.dims();
    let ce = CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(
            logits.reshape([b * l, v]),
            targets[k].clone().reshape([b * l]),
        );
    ce.into_scalar::<f32>()
}

/// Matriks one-hot `[n, v]` dari indeks.
fn one_hot(targets: &[i64], v: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; targets.len() * v];
    for (i, &t) in targets.iter().enumerate() {
        if t >= 0 && (t as usize) < v {
            y[i * v + t as usize] = 1.0;
        }
    }
    y
}

/// Baca tensor sebagai `Vec<f32>` (kosong bila gagal).
fn read_f32(t: &Tensor<2>) -> Vec<f32> {
    t.to_data().try_to_vec::<f32>().unwrap_or_default()
}

/// Baca tensor indeks sebagai `Vec<i64>` (kosong bila gagal).
fn read_i64(t: &Tensor<1, Int>) -> Vec<i64> {
    t.to_data().try_to_vec::<i64>().unwrap_or_default()
}
