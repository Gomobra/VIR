//! VQ-VAE multiscale (skeleton).

use burn::module::Module;
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Device;
use vir_common::VirError;

#[derive(Debug)]
pub struct CodebookConfig {
    pub num_entries: usize,
    pub dim: usize,
}

#[derive(Module, Debug)]
pub struct Codebook {
    pub weight: Param<Tensor<2>>,
}

impl Codebook {
    pub fn new(cfg: &CodebookConfig, device: &Device) -> Self {
        let w = Tensor::random(
            [cfg.num_entries, cfg.dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        Self {
            weight: Param::from_tensor(w),
        }
    }

    pub fn quantize(&self, z: Tensor<2>) -> (Tensor<1, Int>, Tensor<2>) {
        let z_sq = z.clone().powf(2.0).sum_dim(1);
        let e_sq = self.weight.val().clone().powf(2.0).sum_dim(1);
        let cross = z.clone().matmul(self.weight.val().clone().transpose());
        let dist = z_sq.unsqueeze_dim(1) - cross * 2.0 + e_sq.unsqueeze_dim(0);
        let idx = dist.argmin(1);
        let q = self.weight.val().clone().select(0, idx.clone());
        (idx, q)
    }
}

pub fn vq_smoke() -> Result<(), VirError> {
    let device = Device::default();
    let cfg = CodebookConfig {
        num_entries: 8,
        dim: 4,
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
}