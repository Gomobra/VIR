//! VQ-VAE multiscale (codebook, EMA, nested dropout, decoder, loss).

use burn::module::Module;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Device;
use vir_common::VirError;

#[derive(Debug, Clone)]
pub struct CodebookConfig {
    pub num_entries: usize,
    pub dim: usize,
    pub ema_decay: f32,
    pub commitment_cost: f32,
}

impl Default for CodebookConfig {
    fn default() -> Self {
        Self {
            num_entries: 4096,
            dim: 384,
            ema_decay: 0.99,
            commitment_cost: 0.25,
        }
    }
}

#[derive(Module, Debug)]
pub struct Codebook {
    pub weight: Param<Tensor<2>>,
    #[module(skip)]
    e_t: Tensor<2>,
    #[module(skip)]
    e_sq: Tensor<1>,
    pub ema_decay: f32,
    pub commitment_cost: f32,
}

impl Codebook {
    pub fn new(cfg: &CodebookConfig, device: &Device) -> Self {
        let w = Tensor::random(
            [cfg.num_entries, cfg.dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let v = cfg.num_entries;
        let e_t = w.clone().transpose();
        let e_sq = (w.clone() * w.clone()).sum_dim(1).reshape([v]);
        Self {
            weight: Param::from_tensor(w),
            e_t,
            e_sq,
            ema_decay: cfg.ema_decay,
            commitment_cost: cfg.commitment_cost,
        }
    }

    /// Refresh cache setelah weight berubah (mis. EMA update).
    pub fn refresh_cache(&mut self) {
        let w = self.weight.val();
        let v = w.dims()[0];
        self.e_t = w.clone().transpose();
        self.e_sq = (w.clone() * w.clone()).sum_dim(1).reshape([v]);
    }

    /// Argmin nearest-neighbor (chunked, cache e_t/e_sq).
    pub fn quantize_idx(&self, z: Tensor<2>) -> Tensor<1, Int> {
        let [n, d] = z.dims();
        let v = self.e_t.dims()[1];
        let chunk = 8192usize;
        let mut parts: Vec<Tensor<1, Int>> = Vec::new();
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            let zc = z.clone().slice([start..end, 0..d]);
            let cross = zc.matmul(self.e_t.clone());
            let scores = cross - self.e_sq.clone().reshape([1, v]) * 0.5;
            let rows = end - start;
            parts.push(scores.argmax(1).reshape([rows]));
            start = end;
        }
        if parts.len() == 1 {
            parts.remove(0)
        } else {
            Tensor::cat(parts, 0)
        }
    }

    /// Full quantize: indices + quantized vectors.
    pub fn quantize(&self, z: Tensor<2>) -> (Tensor<1, Int>, Tensor<2>) {
        let idx = self.quantize_idx(z);
        let q = self.weight.val().select(0, idx.clone());
        (idx, q)
    }

    /// Straight-through: z + (z_q - z).detach()
    pub fn straight_through(&self, z: Tensor<2>) -> Tensor<2> {
        let z_cl = z.clone();
        let (_, q) = self.quantize(z_cl);
        z.clone() + (q - z).detach()
    }
}

/// Decoder kecil: rekonstruksi -> logit vocab.
#[derive(Module, Debug)]
pub struct Decoder {
    pub fc: Linear,
}

impl Decoder {
    pub fn new(dim: usize, vocab: usize, device: &Device) -> Self {
        Self {
            fc: LinearConfig::new(dim, vocab).init(device),
        }
    }
    pub fn forward(&self, z: Tensor<3>) -> Tensor<3> {
        self.fc.forward(z)
    }
}

/// Loss tokenizer: CE + commitment + latent quantization.
pub fn vq_loss(
    logits: Tensor<3>,
    target: Tensor<2, Int>,
    z: Tensor<2>,
    z_q: Tensor<2>,
    commitment_cost: f32,
    device: &Device,
) -> Tensor<1> {
    let [b, t, v] = logits.dims();
    let logits = logits.reshape([b * t, v]);
    let target = target.reshape([b * t]);
    let ce = burn::nn::loss::CrossEntropyLossConfig::new()
        .init(device)
        .forward(logits, target);
    let z_d = z.clone().detach();
    let z_q_d = z_q.clone().detach();
    let d1 = z_d - z_q;
    let d2 = z - z_q_d;
    let commit = (d1.clone() * d1).mean();
    let latent = (d2.clone() * d2).mean();
    ce + commit * commitment_cost + latent
}

/// Nested dropout: acak cutoff k.
pub fn nested_dropout_cutoff(num_levels: usize, prob: f32, rng: &mut impl rand::Rng) -> usize {
    if num_levels <= 1 || prob <= 0.0 {
        return num_levels;
    }
    if rand::Rng::gen_range(&mut *rng, 0.0..1.0) < prob {
        rand::Rng::gen_range(&mut *rng, 1..num_levels)
    } else {
        num_levels
    }
}

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
    let (idx, q) = cb.quantize(z);
    let _ = (idx, q);
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
    fn nested_dropout_bounds() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let k = nested_dropout_cutoff(12, 0.5, &mut rng);
            assert!((1..=12).contains(&k));
        }
    }
}