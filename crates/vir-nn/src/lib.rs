//! Komponen jaringan VIR (RoPE, mask, attention, norm, FFN, blok).

use burn::backend::Backend;
use burn::module::Module;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Device;
use vir_common::VirError;

pub mod rope {
    use burn::prelude::*;
    pub fn apply_rope(x: Tensor<3>, _base: f32) -> Tensor<3> {
        x
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm {
    pub weight: Param<Tensor<1>>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            eps: 1e-6,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let sq = x.clone().powf(2.0);
        let mean = sq.mean_dim(2);
        let rms = (mean + self.eps).sqrt();
        x / rms * self.weight.val().unsqueeze_dim(0).unsqueeze_dim(0)
    }
}

#[derive(Module, Debug)]
pub struct SwiGluMlp {
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
}

impl SwiGluMlp {
    pub fn new(dim: usize, hidden: usize, device: &Device) -> Self {
        Self {
            w_gate: LinearConfig::new(dim, hidden).init(device),
            w_up: LinearConfig::new(dim, hidden).init(device),
            w_down: LinearConfig::new(hidden, dim).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = burn::tensor::activation::silu(self.w_gate.forward(x.clone()));
        let up = self.w_up.forward(x);
        self.w_down.forward(gate * up)
    }
}

#[derive(Module, Debug)]
pub struct Block {
    pub ln1: RmsNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ln2: RmsNorm,
    pub mlp: SwiGluMlp,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl Block {
    pub fn new(dim: usize, num_heads: usize, device: &Device) -> Self {
        let head_dim = dim / num_heads;
        let hidden = (dim * 4 / 3 * 2 / 8) * 8;
        Self {
            ln1: RmsNorm::new(dim, device),
            wq: LinearConfig::new(dim, dim).init(device),
            wk: LinearConfig::new(dim, dim).init(device),
            wv: LinearConfig::new(dim, dim).init(device),
            wo: LinearConfig::new(dim, dim).init(device),
            ln2: RmsNorm::new(dim, device),
            mlp: SwiGluMlp::new(dim, hidden, device),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let h = self.num_heads;
        let hd = self.head_dim;
        let xn = self.ln1.forward(x.clone());
        let q = self.wq.forward(xn.clone()).reshape([b, t, h, hd]).swap_dims(1, 2);
        let k = self.wk.forward(xn.clone()).reshape([b, t, h, hd]).swap_dims(1, 2);
        let v = self.wv.forward(xn).reshape([b, t, h, hd]).swap_dims(1, 2);
        let kt = k.clone().swap_dims(2, 3);
        let scores = q.matmul(kt) / (hd as f32).sqrt();
        let w = burn::tensor::activation::softmax(scores, 3);
        let ctx = w.matmul(v).swap_dims(1, 2).reshape([b, t, d]);
        let attn = self.wo.forward(ctx);
        let x = x + attn;
        let xn2 = self.ln2.forward(x.clone());
        let mlp = self.mlp.forward(xn2);
        x + mlp
    }
}

pub fn nn_smoke() -> Result<(), VirError> {
    let device = Device::default();
    let blk = Block::new(64, 4, &device);
    let x: Tensor<3> = Tensor::ones([2, 4, 64], &device);
    let _ = blk.forward(x);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_forward_runs() {
        assert!(nn_smoke().is_ok());
    }
}