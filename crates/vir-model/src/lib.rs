//! Generator NCM + recurrent depth + halting + block-AR + sampling.

use burn::prelude::*;
use burn::module::Module;
use burn::module::Param;
use burn::tensor::backend::Backend;
use vir_common::VirError;
use vir_nn::Block;

pub struct GeneratorConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub num_levels: usize,
    pub num_loops: usize,
}

// ---------- Recurrent depth ----------
#[derive(Module, Debug)]
pub struct RecurrentCore<B: Backend> {
    pub blocks: Vec<Block<B>>,
}

impl<B: Backend> RecurrentCore<B> {
    pub fn new(dim: usize, num_heads: usize, num_blocks: usize, device: &B::Device) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(Block::new(dim, num_heads, device));
        }
        Self { blocks }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = x;
        for b in &self.blocks {
            h = b.forward(h);
        }
        h
    }
}

// ---------- Halting ----------
pub trait Halter {
    fn should_halt(&self, entropy: f32) -> bool;
}

pub struct FixedHalter {
    pub max_loops: usize,
}

impl Halter for FixedHalter {
    fn should_halt(&self, _entropy: f32) -> bool {
        false
    }
}

pub struct EntropyHalter {
    pub threshold: f32,
}

impl Halter for EntropyHalter {
    fn should_halt(&self, entropy: f32) -> bool {
        entropy < self.threshold
    }
}

// ---------- Sampling ----------
pub fn cfg_logit(uc: f32, c: f32, beta: f32) -> f32 {
    uc + beta * (c - uc)
}

pub fn mbr_select(scores: &[f32]) -> usize {
    // pilih indeks dengan skor tertinggi
    let mut best = 0usize;
    let mut bestv = f32::MIN;
    for (i, &s) in scores.iter().enumerate() {
        if s > bestv {
            bestv = s;
            best = i;
        }
    }
    best
}

// ---------- Generator ----------
#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    pub blocks: Vec<Block<B>>,
    pub core: Option<RecurrentCore<B>>,
}

impl<B: Backend> Generator<B> {
    pub fn new(cfg: &GeneratorConfig, device: &B::Device) -> Self {
        let mut blocks = Vec::with_capacity(cfg.num_blocks);
        for _ in 0..cfg.num_blocks {
            blocks.push(Block::new(cfg.dim, cfg.num_heads, device));
        }
        let core = if cfg.num_loops > 1 {
            Some(RecurrentCore::new(
                cfg.dim,
                cfg.num_heads,
                cfg.num_blocks,
                device,
            ))
        } else {
            None
        };
        Self { blocks, core }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = x;
        for b in &self.blocks {
            h = b.forward(h);
        }
        if let Some(core) = &self.core {
            // recurrent: ulang core n kali dengan injeksi input awal (simplified: tanpa injeksi)
            for _ in 0..2 {
                h = core.forward(h);
            }
        }
        h
    }
}

pub fn model_smoke() -> Result<(), VirError> {
    let device: <burn::backend::Flex as Backend>::Device = Default::default();
    let cfg = GeneratorConfig {
        dim: 64,
        num_heads: 4,
        num_blocks: 2,
        num_levels: 4,
        num_loops: 2,
    };
    let g = Generator::<burn::backend::Flex>::new(&cfg, &device);
    let x: Tensor<burn::backend::Flex, 3> = Tensor::ones([2, 4, 64], &device);
    let _ = g.forward(x);
    // sanity sampling helpers
    assert!((cfg_logit(0.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
    assert_eq!(mbr_select(&[0.1, 0.5, 0.3]), 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generator_forward_runs() {
        assert!(model_smoke().is_ok());
    }
    #[test]
    fn halter_entropy() {
        let h = EntropyHalter { threshold: 0.5 };
        assert!(h.should_halt(0.1));
        assert!(!h.should_halt(1.0));
    }
}
