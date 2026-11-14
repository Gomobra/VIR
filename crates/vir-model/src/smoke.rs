//! Gerbang mini (PRD F1.8): tokenizer VQ multiscale → generator → kepala dilatih
//! di host sampai loss turun → generate deterministik → loop blok.
//!
//! Gerbang ini menggantikan uji asap lama yang hanya memeriksa `loss >= 0`.
//! Kepala tiap level dilatih dengan Adam di host (gradien softmax CE batch
//! penuh) lalu bobotnya ditulis kembali ke `Param`, sehingga gerbang mengukur
//! "model ini bisa menurunkan loss pada data kecil" tanpa feature `train` dan
//! tanpa optimizer Burn; trainer sebenarnya ada di `vir-train` (F1.7).
//!
//! Batas tugas yang diperiksa:
//!
//! ```text
//! encode VQ (F1.4.b) → teacher inputs (F1.6 langkah 3) → forward (F2.1)
//! → loss turun + akurasi naik → generate deterministik (F1.6 langkah 4)
//! → loop blok (F2.3)
//! ```

use burn::prelude::*;
use burn::tensor::{Device, Distribution, TensorData};
use vir_common::VirError;
use vir_vq::quantizer::indices_to_vec;
use vir_vq::{Codebook, CodebookConfig, Decoder, MultiscaleVq};

use crate::block_ar::{BlockArConfig, BlockMeta, block_ar};
use crate::generator::{Generator, GeneratorConfig, ncm_loss, teacher_inputs};
use crate::sampling::Sampler;

/// Level gerbang (naik ketat, berakhir `L = 16`).
const LEVELS: [usize; 4] = [2, 4, 8, 16];
/// Lebar model mini.
const DIM: usize = 32;
/// Jumlah head mini.
const HEADS: usize = 4;
/// Ukuran codebook mini.
const CODEBOOK: usize = 64;
/// Jumlah sampel (PRD: memorisasi 8 sampel).
const SAMPLES: usize = 8;
/// Panjang blok `L`.
const LEN: usize = 16;
/// Langkah Adam host untuk kepala.
const HEAD_STEPS: usize = 300;
/// Laju belajar Adam host.
const HEAD_LR: f32 = 0.02;
/// Faktor penurunan loss minimum yang lulus gerbang.
const MIN_DROP: f32 = 0.9;

/// Device uji asap: CUDA bila feature menyala, selain itu default.
fn smoke_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::default()
    }
}

/// Bobot level rata: `1/K` (F1.7 α = 1).
fn uniform_weights(levels: usize) -> Vec<f32> {
    if levels == 0 {
        return Vec::new();
    }
    vec![1.0 / levels as f32; levels]
}

/// Akurasi argmax rata-rata atas semua level dan posisi.
fn accuracy(logits: &[Tensor<3>], targets: &[Tensor<2, Int>]) -> f32 {
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

/// Token hasil dekode `ẑ^(K)` lewat decoder tokenizer.
fn decode_tokens(decoder: &Decoder, z: &Tensor<3>) -> Vec<u32> {
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

/// Latih satu kepala linear (`logits = x·Wᵀ`) dengan Adam batch penuh.
///
/// Mengembalikan CE rata-rata akhir. `x` `[N, h]` baris-major, `w` `[V, h]`.
fn fit_head(x: &[f32], targets: &[i64], w: &mut [f32], vocab: usize, steps: usize) -> f32 {
    let rows = targets.len();
    let hidden = if rows == 0 { 0 } else { x.len() / rows };
    if rows == 0 || hidden == 0 || vocab == 0 || w.len() != vocab * hidden {
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
                let wr = &w[v * hidden..(v + 1) * hidden];
                let mut z = 0.0f32;
                for j in 0..hidden {
                    z += wr[j] * xr[j];
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
                let gr = &mut grad[v * hidden..(v + 1) * hidden];
                for j in 0..hidden {
                    gr[j] += scale * xr[j];
                }
            }
        }
        opt.step(w, &grad);
    }
    loss / rows as f32
}

/// Latih kepala semua level di host; kembalikan CE per level (tanpa bobot).
fn fit_heads(
    model: &mut Generator,
    inputs: &[Tensor<3>],
    targets: &[Tensor<2, Int>],
    steps: usize,
) -> Result<Vec<f32>, VirError> {
    let features = model.forward_features(inputs, model.num_loops());
    let n = features.len().min(targets.len()).min(model.heads.len());
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let [b, l, h] = features[k].dims();
        let rows = b * l;
        let vocab = model.heads[k].weight.val().dims()[0];
        let x = features[k]
            .clone()
            .reshape([rows, h])
            .to_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        let want = targets[k]
            .clone()
            .reshape([rows])
            .to_data()
            .try_to_vec::<i64>()
            .unwrap_or_default();
        if x.len() != rows * h || want.len() != rows {
            return Err(VirError::Internal(format!(
                "data kepala level {k} tidak dapat dibaca"
            )));
        }
        let device = model.heads[k].weight.val().device();
        let mut w = model.heads[k]
            .weight
            .val()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        if w.len() != vocab * h {
            return Err(VirError::Internal(format!(
                "bobot kepala level {k} berukuran {}, butuh {}",
                w.len(),
                vocab * h
            )));
        }
        out.push(fit_head(&x, &want, &mut w, vocab, steps));
        crate::generator::replace_head(&mut model.heads[k], w, vocab, h, &device)?;
    }
    Ok(out)
}

/// Gerbang mini F1.8.
pub fn full_smoke() -> Result<(), VirError> {
    let device = smoke_device();
    let levels = LEVELS.to_vec();
    let weights = uniform_weights(levels.len());

    // 1. Tokenizer beku (F1.4.b): jadwal sama dengan generator.
    let cb = CodebookConfig {
        num_entries: CODEBOOK,
        dim: DIM,
        ..CodebookConfig::default()
    };
    let vq = MultiscaleVq::new(&levels, 3, |_| Codebook::new(&cb, &device), &device);
    let z = Tensor::<3>::random(
        [SAMPLES, LEN, DIM],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let encoded = vq.encode_full(z);
    let targets = encoded.indices.clone();

    // 2. Generator dengan codebook dan `Phi` tokenizer (F1.7 tahap 2).
    let mut model = Generator::new(&GeneratorConfig::new(DIM, HEADS, levels.clone()), &device)?;
    model.load_tokenizer(&vq)?;
    let inputs = teacher_inputs(&encoded.z_hat, SAMPLES, LEN, DIM, &device);

    // 3. Loss awal (kepala acak) lalu latih kepala di host.
    let before_logits = model.forward(&inputs);
    let loss_before = ncm_loss(&before_logits, &targets, &weights).into_scalar::<f32>();
    let acc_before = accuracy(&before_logits, &targets);
    let per_level = fit_heads(&mut model, &inputs, &targets, HEAD_STEPS)?;
    let after_logits = model.forward(&inputs);
    let loss_after = ncm_loss(&after_logits, &targets, &weights).into_scalar::<f32>();
    let host_loss: f32 = per_level
        .iter()
        .zip(weights.iter())
        .map(|(ce, w)| ce * w)
        .sum();
    let acc_after = accuracy(&after_logits, &targets);

    // 4. Gerbang: loss turun, akurasi naik, dan host cocok dengan jalur Burn.
    if !loss_after.is_finite() || !loss_before.is_finite() {
        return Err(VirError::Internal("loss gerbang tidak hingga".into()));
    }
    if !(loss_after < MIN_DROP * loss_before) {
        return Err(VirError::Internal(format!(
            "gerbang loss gagal: {loss_before} → {loss_after}"
        )));
    }
    if !(acc_after > acc_before) {
        return Err(VirError::Internal(format!(
            "gerbang akurasi gagal: {acc_before} → {acc_after}"
        )));
    }
    if (host_loss - loss_after).abs() > 0.05 * loss_before {
        return Err(VirError::Internal(format!(
            "loss host {host_loss} tidak cocok dengan Burn {loss_after}"
        )));
    }

    // 5. Determinisme generate pada seed sama (F1.6 langkah 6).
    let sampler = Sampler::new(0.9, 8, 7);
    let first = model.generate(None, &sampler)?;
    let second = model.generate(None, &sampler)?;
    for k in 0..first.codes.len() {
        if indices_to_vec(&first.codes[k]) != indices_to_vec(&second.codes[k]) {
            return Err(VirError::Internal(format!(
                "generate level {k} tidak deterministik"
            )));
        }
    }
    let expected_positions: usize = levels.iter().sum();
    if first.positions != expected_positions {
        return Err(VirError::Internal(format!(
            "posisi generate {} != Σℓ_k {expected_positions}",
            first.positions
        )));
    }

    // 6. Loop blok (F2.3): prefix bertambah, berhenti pada batas token.
    let decoder = Decoder::new(DIM, CODEBOOK, &device);
    let cfg = BlockArConfig {
        max_blocks: 3,
        max_tokens: 12,
        eos: u32::MAX,
    };
    let mut prefix: Option<Tensor<3>> = None;
    let (tokens, cost) = block_ar(
        &cfg,
        |prefix_len| {
            let out = model.generate(prefix.clone(), &sampler)?;
            prefix = Some(out.z_hat.clone());
            let meta = BlockMeta {
                positions: out.positions,
                decode_positions: out.ctx_len,
                next_prefix_len: prefix_len + LEN,
            };
            Ok((out.z_hat, meta))
        },
        |z: &Tensor<3>| Ok(decode_tokens(&decoder, z)),
    )?;
    if cost.blocks == 0 || tokens.is_empty() {
        return Err(VirError::Internal("loop blok tidak menghasilkan token".into()));
    }
    if tokens.len() > cfg.max_tokens {
        return Err(VirError::Internal(format!(
            "loop blok melewati batas token: {}",
            tokens.len()
        )));
    }
    Ok(())
}

/// Uji asap kecil: forward, generate, dan loss berjalan pada model mini.
pub fn model_smoke() -> Result<(), VirError> {
    let device = smoke_device();
    let levels = vec![2usize, 4, 8];
    let mut model = Generator::new(&GeneratorConfig::new(16, 2, levels.clone()), &device)?;
    let inputs: Vec<Tensor<3>> = (0..levels.len())
        .map(|_| Tensor::<3>::zeros([2, 8, 16], &device))
        .collect();
    let logits = model.forward(&inputs);
    if logits.len() != levels.len() {
        return Err(VirError::Internal(format!(
            "forward memberi {} level, butuh {}",
            logits.len(),
            levels.len()
        )));
    }
    let targets: Vec<Tensor<2, Int>> = (0..levels.len())
        .map(|_| Tensor::<2, Int>::zeros([2, 8], &device))
        .collect();
    let loss = ncm_loss(&logits, &targets, &uniform_weights(levels.len()));
    if !loss.into_scalar::<f32>().is_finite() {
        return Err(VirError::Internal("loss tidak hingga".into()));
    }
    let out = model.generate(None, &Sampler::greedy())?;
    if out.codes.len() != levels.len() {
        return Err(VirError::Internal(format!(
            "generate memberi {} level, butuh {}",
            out.codes.len(),
            levels.len()
        )));
    }
    Ok(())
}