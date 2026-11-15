//! Gerbang mini (PRD F1.8): tokenizer VQ multiscale → generator → kepala dilatih
//! di host sampai loss turun → generate deterministik → loop blok.
//!
//! Kepala tiap level dilatih di host ([`crate::smoke_host`]) lalu bobotnya
//! ditulis kembali ke `Param`, sehingga gerbang mengukur "model ini bisa
//! menurunkan loss pada data kecil" tanpa feature `train` dan tanpa optimizer
//! Burn; trainer sebenarnya ada di `vir-train` (F1.7).
//!
//! Batas tugas yang diperiksa:
//!
//! ```text
//! encode VQ (F1.4.b) → teacher inputs (F1.6 langkah 3) → forward (F2.1)
//! → loss turun + akurasi naik → generate deterministik (F1.6 langkah 4)
//! → loop blok (F2.3)
//! ```
//!
//! Untuk gerbang G1 yang menuntut reproduksi **persis** 8 sampel, pakai
//! [`crate::smoke_mini::run_mini_smoke`] (kepala kuadrat terkecil + prompt).

use burn::prelude::*;
use burn::tensor::{Device, Distribution};
use vir_common::VirError;
use vir_vq::quantizer::indices_to_vec;
use vir_vq::{Codebook, CodebookConfig, Decoder, MultiscaleVq};

use crate::block_ar::{BlockArConfig, BlockMeta, block_ar};
use crate::generator::{Generator, GeneratorConfig};
use crate::loss::{ncm_loss, teacher_inputs};
use crate::sampling::Sampler;
use crate::smoke_host::{accuracy, decode_tokens, fit_heads};

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
/// Faktor penurunan loss minimum yang lulus gerbang.
const MIN_DROP: f32 = 0.9;

/// Device uji asap: CUDA bila feature menyala, selain itu default.
pub(crate) fn smoke_device() -> Device {
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
pub(crate) fn uniform_weights(levels: usize) -> Vec<f32> {
    if levels == 0 {
        return Vec::new();
    }
    vec![1.0 / levels as f32; levels]
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
    let z = Tensor::<3>::random([SAMPLES, LEN, DIM], Distribution::Normal(0.0, 1.0), &device);
    let encoded = vq.encode_full(z);
    let targets = encoded.indices.clone();

    // 2. Generator dengan codebook dan `Phi` tokenizer (F1.7 tahap 2).
    let mut gcfg = GeneratorConfig::new(DIM, HEADS, levels.clone());
    gcfg.codebook_size = CODEBOOK;
    let mut model = Generator::new(&gcfg, &device)?;
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
    if !codes_equal(&first.codes, &second.codes) {
        return Err(VirError::Internal("generate tidak deterministik".into()));
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
        return Err(VirError::Internal(
            "loop blok tidak menghasilkan token".into(),
        ));
    }
    if tokens.len() > cfg.max_tokens {
        return Err(VirError::Internal(format!(
            "loop blok melewati batas token: {}",
            tokens.len()
        )));
    }
    Ok(())
}

/// Bandingkan dua daftar indeks kode level (bit demi bit).
pub(crate) fn codes_equal(a: &[Tensor<2, Int>], b: &[Tensor<2, Int>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        let [bx, lx] = x.dims();
        let [by, ly] = y.dims();
        let xv = indices_to_vec(&x.clone().reshape([bx * lx]));
        let yv = indices_to_vec(&y.clone().reshape([by * ly]));
        xv == yv
    })
}

/// Uji asap kecil: forward, generate, dan loss berjalan pada model mini.
pub fn model_smoke() -> Result<(), VirError> {
    let device = smoke_device();
    let levels = vec![2usize, 4, 8];
    let model = Generator::new(&GeneratorConfig::new(16, 2, levels.clone()), &device)?;
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
// Alias gerbang G1 di modul `smoke` (API yang dipakai CLI/bind-eng):
// `vir_model::smoke::run_mini_smoke`.
pub use crate::smoke_mini::{MiniSmokeConfig, SmokeResult, run_mini_smoke, synth_samples};
