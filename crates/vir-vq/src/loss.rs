//! Loss tokenizer R4 (PRD F1.4.d).
//!
//! ```text
//! L_tok = −(1/L) Σ_i log p_ψ(x^i | ẑ^(k)) + β·MSE(z, sg[ẑ^(k)]) + MSE(sg[z], ẑ^(k)])
//! ```
//!
//! Suku kedua adalah komitmen (menarik `z` ke kode), suku ketiga kuantisasi
//! laten (menarik kode ke `z`). `β` = [`crate::codebook::CodebookConfig::commitment_cost`]
//! (0,25 pada kode acuan NCM).
//!
//! Dengan FSQ ([`crate::fsq`]) kedua suku MSE dihilangkan: FSQ tidak butuh loss
//! tambahan (PRD F1.4.d langkah 2).

use crate::decoder::{chunked_cross_entropy, Decoder};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;

/// `MSE(z, sg[z_q])` — suku komitmen R4.
pub fn commitment_loss(z: Tensor<2>, z_q: Tensor<2>) -> Tensor<1> {
    let diff = z - z_q.detach();
    (diff.clone() * diff).mean()
}

/// `MSE(sg[z], z_q)` — suku kuantisasi laten R4.
pub fn latent_loss(z: Tensor<2>, z_q: Tensor<2>) -> Tensor<1> {
    let diff = z.detach() - z_q;
    (diff.clone() * diff).mean()
}

/// CE token dari logits `[B, L, V]` dan target `[B, L]` (rerata per token).
pub fn token_cross_entropy(logits: Tensor<3>, target: Tensor<2, Int>, device: &Device) -> Tensor<1> {
    let [b, t, v] = logits.dims();
    let flat = logits.reshape([b * t, v]);
    let tgt = target.reshape([b * t]);
    CrossEntropyLossConfig::new().init(device).forward(flat, tgt)
}

/// Loss R4 lengkap dari logits yang sudah dihitung (versi padat).
pub fn vq_loss(
    logits: Tensor<3>,
    target: Tensor<2, Int>,
    z: Tensor<2>,
    z_q: Tensor<2>,
    commitment_cost: f32,
    device: &Device,
) -> Tensor<1> {
    let ce = token_cross_entropy(logits, target, device);
    let commit = commitment_loss(z.clone(), z_q.clone()) * commitment_cost;
    let latent = latent_loss(z, z_q);
    ce + commit + latent
}

/// Loss R4 dengan CE per potongan (PRD F1.4.d langkah 3).
///
/// `z` adalah laten asli `[B, L, d]`, `z_hat` rekonstruksi kumulatif `[B, L, d]`
/// (sudah straight-through, jadi gradien mengalir ke encoder).
pub fn vq_loss_chunked(
    decoder: &Decoder,
    z: Tensor<3>,
    z_hat: Tensor<3>,
    target: Tensor<2, Int>,
    commitment_cost: f32,
    chunk: usize,
) -> Tensor<1> {
    let [b, t, d] = z.dims();
    let ce = chunked_cross_entropy(decoder, z_hat.clone(), target, chunk);
    let commit = commitment_loss(z.clone().reshape([b * t, d]), z_hat.clone().reshape([b * t, d]));
    let latent = latent_loss(z.reshape([b * t, d]), z_hat.reshape([b * t, d]));
    ce + commit * commitment_cost + latent
}

/// Loss tokenizer tanpa suku MSE (FSQ dan variannya, PRD F1.4.f).
pub fn fsq_loss(
    decoder: &Decoder,
    z_hat: Tensor<3>,
    target: Tensor<2, Int>,
    chunk: usize,
) -> Tensor<1> {
    chunked_cross_entropy(decoder, z_hat, target, chunk)
}

