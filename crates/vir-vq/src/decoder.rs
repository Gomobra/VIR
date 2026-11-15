//! Decoder kecil dan cross-entropy per potongan (PRD F1.4.d).
//!
//! NCM memakai 1 blok transformer (4 blok untuk LM1B) di atas rekonstruksi
//! kumulatif. Blok transformer-nya adalah tugas `vir-nn` (F1.5), jadi di sini
//! decoder-nya adalah satu blok MLP pre-norm + proyeksi ke vocab: bentuk
//! `L_tok = −(1/L) Σ log p_ψ(x^i | ẑ^(k))` yang sama, dengan biaya parameter
//! jauh di bawah blok transformer.
//!
//! Vokab besar (151.680 pada WMT14) membuat logits `[L, |vocab|]` mahal untuk
//! disimpan. [`chunked_cross_entropy`] memotong sumbu waktu: setiap potongan
//! menjalankan decoder lalu CE-nya dijumlahkan, jadi hanya satu potongan logits
//! yang pernah ada. Untuk memangkas graf juga aktifkan
//! `Device::gradient_checkpointing()`.

use burn::module::{Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;

/// Konfigurasi decoder (NCM Tabel 5: `d = 384`, satu blok).
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub dim: usize,
    pub vocab: usize,
    /// Lebar MLP dalam blok; `0` memakai `4·dim` (rasio FFN 4×).
    pub hidden: usize,
}

impl DecoderConfig {
    pub fn new(dim: usize, vocab: usize) -> Self {
        Self {
            dim,
            vocab,
            hidden: dim * 4,
        }
    }

    fn hidden_or_default(&self) -> usize {
        if self.hidden == 0 {
            self.dim * 4
        } else {
            self.hidden
        }
    }
}

/// Decoder: `ẑ^(k)` `[B, L, d]` → logit `[B, L, |vocab|]`.
#[derive(Module, Debug)]
pub struct Decoder {
    pub norm_gamma: Param<Tensor<1>>,
    pub norm_beta: Param<Tensor<1>>,
    pub fc1: Linear,
    pub fc2: Linear,
    pub head: Linear,
    #[module(skip)]
    pub eps: f32,
}

impl Decoder {
    /// Decoder dengan rasio FFN 4× (default NCM).
    pub fn new(dim: usize, vocab: usize, device: &Device) -> Self {
        Self::with_config(&DecoderConfig::new(dim, vocab), device)
    }

    /// Decoder dari konfigurasi eksplisit.
    pub fn with_config(cfg: &DecoderConfig, device: &Device) -> Self {
        let hidden = cfg.hidden_or_default();
        Self {
            norm_gamma: Param::from_tensor(Tensor::ones([cfg.dim], device)),
            norm_beta: Param::from_tensor(Tensor::zeros([cfg.dim], device)),
            fc1: LinearConfig::new(cfg.dim, hidden).init(device),
            fc2: LinearConfig::new(hidden, cfg.dim).init(device),
            head: LinearConfig::new(cfg.dim, cfg.vocab).init(device),
            eps: 1e-5,
        }
    }

    /// LayerNorm pada sumbu terakhir (sumbu `d`).
    fn norm(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, l, d] = x.dims();
        let mu = x.clone().mean_dim(2).reshape([b, l, 1]);
        let centered = x - mu;
        let var = (centered.clone() * centered.clone())
            .mean_dim(2)
            .reshape([b, l, 1]);
        let normed = centered / (var + self.eps).sqrt();
        normed * self.norm_gamma.val().reshape([1, 1, d]) + self.norm_beta.val().reshape([1, 1, d])
    }

    /// Satu blok MLP pre-norm + residual, lalu kepala softmax.
    pub fn forward(&self, z: Tensor<3>) -> Tensor<3> {
        let h = self.norm(z.clone());
        let h = gelu(self.fc1.forward(h));
        let h = self.fc2.forward(h);
        self.head.forward(z + h)
    }

    /// Token hasil argmax per posisi, `[B, L]`.
    pub fn argmax_tokens(&self, z: Tensor<3>) -> Tensor<2, Int> {
        let [b, l, _] = z.dims();
        self.forward(z).argmax(2).reshape([b, l])
    }

    /// Ukuran vocab (untuk pemanggil yang hanya punya decoder).
    pub fn vocab_size(&self) -> usize {
        self.head.weight.val().dims()[1]
    }
}

/// Cross-entropy rata-rata per potongan waktu, dijumlahkan.
///
/// Setara dengan CE penuh (rerata atas `B·L` token) tetapi memori logits hanya
/// sebesar satu potongan. `chunk <= 0` diperlakukan sebagai `L`.
pub fn chunked_cross_entropy(
    decoder: &Decoder,
    z: Tensor<3>,
    target: Tensor<2, Int>,
    chunk: usize,
) -> Tensor<1> {
    let [batch, len, dim] = z.dims();
    let device = z.device();
    let vocab = decoder.vocab_size();
    let size = if chunk == 0 { len } else { chunk.min(len) };
    let loss = CrossEntropyLossConfig::new().init(&device);

    let mut acc: Option<Tensor<1>> = None;
    let mut start = 0usize;
    while start < len {
        let end = (start + size).min(len);
        let rows = batch * (end - start);
        let zc = z.clone().slice([0..batch, start..end, 0..dim]);
        let tc = target.clone().slice([0..batch, start..end]).reshape([rows]);
        let logits = decoder.forward(zc).reshape([rows, vocab]);
        let mean = loss.forward(logits, tc);
        // Kali jumlah token di potongan ini, lalu bagi total token di akhir.
        let weighted = mean.mul_scalar(rows as f32);
        acc = Some(match acc {
            Some(prev) => prev + weighted,
            None => weighted,
        });
        start = end;
    }

    let denom = (batch * len).max(1) as f32;
    acc.map(|sum| sum / denom)
        .unwrap_or_else(|| Tensor::zeros([1], &device))
}
