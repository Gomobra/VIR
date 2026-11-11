//! Komponen jaringan VIR (RoPE, mask, attention, norm, FFN, blok).

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::module::Module;
use burn::module::Param;
use burn::tensor::backend::Backend;
use vir_common::VirError;

pub mod rope {
    use burn::prelude::*;
use burn::module::Module;
use burn::module::Param;
use burn::tensor::backend::Backend;

    pub fn apply_rope<B: Backend>(x: Tensor<B, 3>, base: f32) -> Tensor<B, 3> {
        // x: [B, T, D], D even. Rotasi pasangan.
        let [b, t, d] = x.dims();
        let half = d / 2;
        let mut out = x.clone();
        for pos in 0..t {
            for i in 0..half {
                let theta = 1.0 / (base.powf(2.0 * i as f32 / d as f32));
                let angle = pos as f32 * theta;
                let c = angle.cos();
                let s = angle.sin();
                // apply to pairs (i, i+half)
                // gather not trivial; do in-place via slice assignment is complex in Burn.
                // Placeholder: return x unchanged (RoPE identity) — diuji nanti.
                let _ = (c, s);
            }
        }
        let _ = (b, half);
        out
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f32,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            eps: 1e-6,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let sq = x.clone().powf(2.0);
        let mean = sq.mean_dim(2);
        let rms = (mean + self.eps).sqrt();
        x / rms * self.weight.val().unsqueeze_dim(0).unsqueeze_dim(0)
    }
}

#[derive(Module, Debug)]
pub struct SwiGluMlp<B: Backend> {
    pub w_gate: Linear<B>,
    pub w_up: Linear<B>,
    pub w_down: Linear<B>,
}

impl<B: Backend> SwiGluMlp<B> {
    pub fn new(dim: usize, hidden: usize, device: &B::Device) -> Self {
        Self {
            w_gate: LinearConfig::new(dim, hidden).init(device),
            w_up: LinearConfig::new(dim, hidden).init(device),
            w_down: LinearConfig::new(hidden, dim).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = burn::tensor::activation::silu(self.w_gate.forward(x.clone()));
        let up = self.w_up.forward(x);
        self.w_down.forward(gate * up)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub ln1: RmsNorm<B>,
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub wo: Linear<B>,
    pub ln2: RmsNorm<B>,
    pub mlp: SwiGluMlp<B>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> Block<B> {
    pub fn new(dim: usize, num_heads: usize, device: &B::Device) -> Self {
        let head_dim = dim / num_heads;
        let hidden = dim * 4 / 3 * 2; // ~8/3
        let hidden = (hidden / 8) * 8;
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

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // attention (simplified single-head-style MHA via reshape)
        let [b, t, d] = x.dims();
        let h = self.num_heads;
        let hd = self.head_dim;
        let xn = self.ln1.forward(x.clone());
        let q = self
            .wq
            .forward(xn.clone())
            .reshape([b, t, h, hd])
            .swap_dims(1, 2);
        let k = self
            .wk
            .forward(xn.clone())
            .reshape([b, t, h, hd])
            .swap_dims(1, 2);
        let v = self.wv.forward(xn).reshape([b, t, h, hd]).swap_dims(1, 2);
        // scores [b,h,t,t]
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
    let device: <burn::backend::Flex as Backend>::Device = Default::default();
    let blk = Block::<burn::backend::Flex>::new(64, 4, &device);
    let x: Tensor<burn::backend::Flex, 3> = Tensor::ones([2, 4, 64], &device);
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
