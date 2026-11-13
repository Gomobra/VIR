//! Generator NCM: embedding kode per level, proyeksi rekonstruksi kumulatif,
//! prelude → recurrent core → coda, head per level, loss R6, generate.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Device;
use vir_common::VirError;
use vir_nn::Block;
use vir_vq::{Codebook, CodebookConfig};

pub struct GeneratorConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub num_levels: usize,
    pub num_loops: usize,
    pub codebook_size: usize,
}

#[derive(Module, Debug)]
pub struct Generator {
    /// Embedding kode per level (codebook beku).
    pub codebooks: Vec<Codebook>,
    /// Proyeksi rekonstruksi kumulatif d → hidden.
    pub proj: Linear,
    /// Prelude (1 blok).
    pub prelude: Block,
    /// Core (diulang).
    pub core: Block,
    /// Coda (1 blok).
    pub coda: Block,
    /// Head per level: hidden → |V_k|.
    pub heads: Vec<Linear>,
    pub dim: usize,
}

impl Generator {
    pub fn new(cfg: &GeneratorConfig, device: &Device) -> Self {
        let mut codebooks = Vec::with_capacity(cfg.num_levels);
        for _ in 0..cfg.num_levels {
            codebooks.push(Codebook::new(
                &CodebookConfig {
                    num_entries: cfg.codebook_size,
                    dim: cfg.dim,
                    ..CodebookConfig::default()
                },
                device,
            ));
        }
        let mut heads = Vec::with_capacity(cfg.num_levels);
        for _ in 0..cfg.num_levels {
            heads.push(LinearConfig::new(cfg.dim, cfg.codebook_size).init(device));
        }
        Self {
            codebooks,
            proj: LinearConfig::new(cfg.dim, cfg.dim).init(device),
            prelude: Block::new(cfg.dim, cfg.num_heads, device),
            core: Block::new(cfg.dim, cfg.num_heads, device),
            coda: Block::new(cfg.dim, cfg.num_heads, device),
            heads,
            dim: cfg.dim,
        }
    }

    /// Embedding kode level k → [B, ℓ_k, d].
    fn embed_level(&self, k: usize, idx: Tensor<2, Int>) -> Tensor<3> {
        let [b, l] = idx.dims();
        let e = self.codebooks[k].weight.val();
        let flat = e.select(0, idx.reshape([b * l]));
        flat.reshape([b, l, self.dim])
    }

    /// Bangun input level k: resize rekonstruksi kumulatif ẑ^(k-1) ke ℓ_k, proyeksi.
    pub fn level_input(&self, z_hat: Tensor<3>, target_len: usize) -> Tensor<3> {
        let [b, l, d] = z_hat.dims();
        if l == target_len {
            return self.proj.forward(z_hat);
        }
        // resize via average pooling (area) di torch; di sini pakai slice/reshape sederhana
        // untuk mini: ambil target_len posisi pertama (asumsi L >= target_len).
        let z = z_hat.slice([0..b, 0..target_len, 0..d]);
        self.proj.forward(z)
    }

    /// Forward training: prediksi semua level dari rekonstruksi kumulatif.
    /// `z_hats` = rekonstruksi kumulatif per level (K tensor [B, L, d]).
    pub fn forward(&self, z_hats: &[Tensor<3>]) -> Vec<Tensor<3>> {
        let mut out = Vec::with_capacity(z_hats.len());
        for (k, zh) in z_hats.iter().enumerate() {
            let [b, l, _] = zh.dims();
            let x = self.level_input(zh.clone(), l);
            let x = self.prelude.forward(x);
            let x = self.core.forward(x);
            let x = self.coda.forward(x);
            let logits = self.heads[k].forward(x);
            let _ = b;
            out.push(logits);
        }
        out
    }

    /// Generate level demi level (greedy): kembalikan indeks per level.
    pub fn generate(&self, z_hat: Tensor<3>, target_len: usize) -> Vec<Tensor<2, Int>> {
        let mut codes = Vec::new();
        let mut cur = z_hat;
        for k in 0..self.heads.len() {
            let x = self.level_input(cur.clone(), target_len);
            let x = self.prelude.forward(x);
            let x = self.core.forward(x);
            let x = self.coda.forward(x);
            let logits = self.heads[k].forward(x);
            let idx = logits.argmax(2); // [B, target_len]
            codes.push(idx.clone());
            // update rekonstruksi kumulatif: tambah embedding kode
            let [b, l] = idx.dims();
            let emb = self.embed_level(k, idx).reshape([b, l, self.dim]);
            cur = cur + emb;
        }
        codes
    }
}

/// Loss R6: L_NCM = -Σ_k w_k Σ_i log p_θ(r_k^i | z̃^(k-1)).
pub fn ncm_loss(logits: &[Tensor<3>], targets: &[Tensor<2, Int>], weights: &[f32]) -> Tensor<1> {
    let mut total: Option<Tensor<1>> = None;
    for (k, lg) in logits.iter().enumerate() {
        let [b, l, v] = lg.dims();
        let lg2 = lg.clone().reshape([b * l, v]);
        let t = targets[k].clone().reshape([b * l]);
        let ce = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&lg.device())
            .forward(lg2, t);
        let w = weights[k];
        let term = ce * w;
        total = Some(match total {
            Some(acc) => acc + term,
            None => term,
        });
    }
    total.unwrap_or_else(|| Tensor::zeros([1], &Device::default()))
}

pub fn model_smoke() -> Result<(), VirError> {
    let device = Device::default();
    let cfg = GeneratorConfig {
        dim: 64,
        num_heads: 4,
        num_blocks: 2,
        num_levels: 4,
        num_loops: 2,
        codebook_size: 32,
    };
    let g = Generator::new(&cfg, &device);
    let z: Tensor<3> = Tensor::zeros([2, 16, 64], &device);
    let z_hats = vec![z; 4];
    let logits = g.forward(&z_hats);
    assert_eq!(logits.len(), 4);
    let codes = g.generate(Tensor::zeros([2, 16, 64], &device), 16);
    assert_eq!(codes.len(), 4);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generator_forward_runs() {
        assert!(model_smoke().is_ok());
    }
}
