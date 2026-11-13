//! VQ-VAE multiscale (PRD F1.4): codebook EMA, residual multiscale, nested
//! dropout, decoder + loss R4, evaluasi rekonstruksi, dan kuantizer FSQ.
//!
//! Tata letak mengikuti urutan rakit PRD F1.4 (langkah 2) dan rumus R1–R4:
//!
//! | Modul | Isi |
//! |---|---|
//! | [`quantizer`] | trait [`Quantizer`] bersama + statistik utilisasi/perplexity |
//! | [`codebook`] | F1.4.a — kuantisasi terdekat (R1), EMA, restart kode mati, straight-through |
//! | [`multiscale`] | F1.4.b — `Down_k`/`R_k`, rekonstruksi kumulatif (R2) |
//! | [`dropout`] | F1.4.c — nested dropout (R3) + masking + ablasi peluang |
//! | [`decoder`] | F1.4.d — decoder kecil + CE per potongan |
//! | [`loss`] | F1.4.d — loss R4 (CE + komitmen + kuantisasi laten) |
//! | [`metrics`] | F1.4.e — akurasi per level, kategori kesalahan, JSON |
//! | [`fsq`] | F1.4.f — FSQ / RFSQ / iFSQ di belakang trait yang sama |
//! | [`kernel`] | F7.1 — kernel terfusi matmul + argmax (feature `fused`) |
//!
//! Catatan ADR (PRD F1.4.b langkah 4): `Down_k` = average pooling area tanpa
//! tumpang tindih, `R_k` = interpolasi linear + konvolusi residual `Phi`. NCM
//! hanya menyebut "operator rekonstruksi"; SNAC tidak dipakai (itu untuk audio,
//! F6.4).

pub mod codebook;
pub mod decoder;
pub mod dropout;
pub mod fsq;
#[cfg(feature = "fused")]
pub mod kernel;
pub mod loss;
pub mod metrics;
pub mod multiscale;
pub mod quantizer;

pub use codebook::{Codebook, CodebookConfig};
pub use decoder::{chunked_cross_entropy, Decoder, DecoderConfig};
pub use dropout::{
    cumulative_sums, cutoff_distribution, mask_reconstruction, nested_dropout_cutoff,
    sample_cutoff, sum_levels, ABLATION_PROBS,
};
pub use fsq::{Fsq, FsqActivation, FsqConfig, Normalized, Rfsq, RfsqNorm};
pub use loss::{commitment_loss, fsq_loss, latent_loss, token_cross_entropy, vq_loss, vq_loss_chunked};
pub use metrics::{
    classify_token, is_markup, is_number, is_username, is_word_boundary, level_accuracy,
    reports_json, tensor_accuracy, token_accuracy, ErrorCategory, ErrorReport, LevelReport,
};
pub use multiscale::{
    weighted_sum, EncodeOutput, LevelOutput, LevelVq, MultiscaleVq, Phi, ScaleOps,
};
pub use quantizer::{
    histogram, indices_to_vec, replace_param, Quantized, Quantizer, QuantizerStats,
};

use burn::prelude::*;
use burn::tensor::Device;
use vir_common::VirError;

/// Uji asap: codebook kecil + kuantisasi satu batch.
pub fn vq_smoke() -> Result<(), VirError> {
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(not(feature = "cuda"))]
    let device = Device::default();
    let cfg = CodebookConfig {
        num_entries: 8,
        dim: 4,
        ..CodebookConfig::default()
    };
    let cb = Codebook::new(&cfg, &device);
    let z: Tensor<2> = Tensor::ones([2, 4], &device);
    let (idx, q) = cb.quantize_pair(z);
    let _ = (idx, q);
    Ok(())
}

/// Uji asap FSQ: kuantisasi satu batch lewat trait yang sama.
pub fn fsq_smoke() -> Result<(), VirError> {
    let device = Device::default();
    let fsq = Fsq::new(&FsqConfig::default(), 8, &device)?;
    let z: Tensor<2> = Tensor::zeros([2, 8], &device);
    let q = fsq.quantize(z);
    let _ = q.indices;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_quantize_runs() {
        assert!(vq_smoke().is_ok());
    }

    #[test]
    fn fsq_smoke_runs() {
        assert!(fsq_smoke().is_ok());
    }

    /// Modul yang diekspor tetap terjangkau dari akar krat.
    #[test]
    fn public_api_is_wired() {
        assert_eq!(ABLATION_PROBS.len(), 4);
        assert_eq!(classify_token("42"), ErrorCategory::Number);
        let dist = cutoff_distribution(0.5, 12);
        assert_eq!(dist.len(), 13);
    }
}