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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::Decoder;
    use burn::tensor::Distribution;

    fn device() -> Device {
        Device::default()
    }

    /// Suku MSE harus nol saat rekonstruksi tepat, dan positif bila berbeda.
    #[test]
    fn mse_terms_behave() {
        let device = device();
        let z = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let zero = commitment_loss(z.clone(), z.clone());
        assert!(zero.into_data().try_to_vec::<f32>().unwrap_or_default()[0].abs() < 1e-6);
        let far = latent_loss(z.clone(), z.clone() + 2.0);
        let val = far.into_data().try_to_vec::<f32>().unwrap_or_default()[0];
        assert!((val - 4.0).abs() < 1e-5, "got {val}");
    }

    /// Gradien suku komitmen hanya mengalir ke `z` (bukan ke `z_q`).
    #[test]
    fn commitment_gradient_is_one_sided() {
        let device = Device::default().autodiff();
        let z = Tensor::<2>::from_floats([[2.0]], &device).require_grad();
        let z_q = Tensor::<2>::from_floats([[0.0]], &device).require_grad();
        let loss = commitment_loss(z.clone(), z_q.clone());
        let grads = loss.backward();
        let gz = z
            .grad(&grads)
            .unwrap_or_else(|| Tensor::zeros([1, 1], &device));
        let gq = z_q
            .grad(&grads)
            .unwrap_or_else(|| Tensor::zeros([1, 1], &device));
        let gz: f32 = gz.into_data().try_to_vec::<f32>().unwrap_or_default()[0];
        let gq: f32 = gq.into_data().try_to_vec::<f32>().unwrap_or_default()[0];
        // d/dz (z - sg[z_q])² = 2(z - z_q) = 4; gradien ke z_q harus nol.
        assert!((gz - 4.0).abs() < 1e-4, "gz={gz}");
        assert!(gq.abs() < 1e-6, "gq={gq}");
    }

    /// `vq_loss` = CE + β·commit + latent pada nilai yang mudah dihitung.
    #[test]
    fn vq_loss_composition() {
        let device = device();
        let decoder = Decoder::new(3, 4, &device);
        let z = Tensor::<3>::random([1, 2, 3], Distribution::Normal(0.0, 1.0), &device);
        let target = Tensor::<2, Int>::from_ints([[1i64, 2]], &device);
        let logits = decoder.forward(z.clone());
        let z_flat = z.clone().reshape([2, 3]);
        let loss = vq_loss(
            logits,
            target,
            z_flat.clone(),
            z_flat.clone(),
            0.25,
            &device,
        );
        // z == z_q → kedua suku MSE nol, sisa hanya CE (> 0).
        let val = loss.into_data().try_to_vec::<f32>().unwrap_or_default()[0];
        assert!(val > 0.0, "CE harus positif, got {val}");
    }

    /// `vq_loss_chunked` harus sama dengan `vq_loss` padat.
    #[test]
    fn chunked_loss_matches_dense() {
        let device = device();
        let decoder = Decoder::new(4, 6, &device);
        let z = Tensor::<3>::random([2, 5, 4], Distribution::Normal(0.0, 1.0), &device);
        let target = Tensor::<2, Int>::from_ints(
            [[0i64, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
            &device,
        );
        let z_hat = z.clone() * 0.5;
        let dense = vq_loss(
            decoder.forward(z_hat.clone()),
            target.clone(),
            z.clone().reshape([10, 4]),
            z_hat.clone().reshape([10, 4]),
            0.25,
            &device,
        );
        let chunked = vq_loss_chunked(&decoder, z, z_hat, target, 0.25, 2);
        let (a, b) = (
            dense.into_data().try_to_vec::<f32>().unwrap_or_default()[0],
            chunked.into_data().try_to_vec::<f32>().unwrap_or_default()[0],
        );
        assert!((a - b).abs() < 1e-4, "dense={a} chunked={b}");
    }
}