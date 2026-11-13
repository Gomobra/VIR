//! Komponen jaringan VIR (RoPE, mask, attention, norm, FFN, blok).

use burn::module::Module;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Device;
use vir_common::VirError;

/// RoPE gaya GPT-NeoX: pasangan (i, i+half), sudut pos·θ^(-2i/d).
pub fn rope_cos_sin(seq_len: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * head_dim);
    let mut sin = Vec::with_capacity(seq_len * head_dim);
    for pos in 0..seq_len {
        let mut row_cos = Vec::with_capacity(head_dim);
        let mut row_sin = Vec::with_capacity(head_dim);
        for i in 0..half {
            let inv = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * inv;
            row_cos.push(angle.cos());
            row_sin.push(angle.sin());
        }
        // duplikasi ke paruh kedua
        for i in 0..half {
            row_cos.push(row_cos[i]);
            row_sin.push(row_sin[i]);
        }
        cos.extend_from_slice(&row_cos);
        sin.extend_from_slice(&row_sin);
    }
    (cos, sin)
}

/// Terapkan RoPE ke x [B, T, D].
pub fn apply_rope(x: Tensor<3>, base: f32) -> Tensor<3> {
    let [b, t, d] = x.dims();
    let half = d / 2;
    let device = x.device();
    let (cos, sin) = rope_cos_sin(t, d, base);
    let cos = Tensor::<2>::from_floats(burn::tensor::TensorData::new(cos, [t, d]), &device)
        .unsqueeze_dim(0);
    let sin = Tensor::<2>::from_floats(burn::tensor::TensorData::new(sin, [t, d]), &device)
        .unsqueeze_dim(0);
    let x1 = x.clone().slice([0..b, 0..t, 0..half]);
    let x2 = x.clone().slice([0..b, 0..t, half..d]);
    let rotated = burn::tensor::Tensor::cat(vec![x2.clone().neg(), x1], 2);
    x * cos + rotated * sin
}

/// Mask block-diagonal: token level k hanya melihat blok level k.
/// `lengths` = panjang tiap level (jumlah = T). Return [T, T] bool (true = boleh).
pub fn block_diagonal_mask(lengths: &[usize]) -> Vec<Vec<bool>> {
    let total: usize = lengths.iter().sum();
    let mut mask = vec![vec![false; total]; total];
    let mut start = 0usize;
    for &len in lengths {
        let end = start + len;
        for i in start..end {
            for j in start..end {
                mask[i][j] = true;
            }
        }
        start = end;
    }
    mask
}

/// Mask prefix-LM: prompt (prefix_len) saling melihat; target melihat prompt + dirinya (kausal).
pub fn prefix_lm_mask(prefix_len: usize, target_len: usize) -> Vec<Vec<bool>> {
    let total = prefix_len + target_len;
    let mut mask = vec![vec![false; total]; total];
    // prompt: full
    for i in 0..prefix_len {
        for j in 0..prefix_len {
            mask[i][j] = true;
        }
    }
    // target: lihat prompt + dirinya (kausal)
    for i in prefix_len..total {
        for j in 0..prefix_len {
            mask[i][j] = true;
        }
        for j in prefix_len..=i {
            mask[i][j] = true;
        }
    }
    mask
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
        let sq = x.clone() * x.clone();
        let mean = sq.mean_dim(2);
        let rms = (mean + self.eps).sqrt();
        let d = self.weight.val().dims()[0];
        x / rms * self.weight.val().clone().reshape([1, 1, d])
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
        let q = apply_rope(self.wq.forward(xn.clone()), 10000.0)
            .reshape([b, t, h, hd])
            .swap_dims(1, 2);
        let k = apply_rope(self.wk.forward(xn.clone()), 10000.0)
            .reshape([b, t, h, hd])
            .swap_dims(1, 2);
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

    #[test]
    fn block_diagonal_isolates_levels() {
        let m = block_diagonal_mask(&[2, 3]);
        assert_eq!(m.len(), 5);
        assert!(m[0][0] && m[0][1]);
        assert!(!m[0][2], "level 0 tidak boleh lihat level 1");
        assert!(m[2][2] && !m[2][0]);
    }

    #[test]
    fn prefix_lm_prompt_isolated_from_target() {
        let m = prefix_lm_mask(2, 3);
        // prompt (0,1) tidak lihat target (2,3,4)
        assert!(!m[0][2] && !m[1][4]);
        // target lihat prompt + dirinya
        assert!(m[2][0] && m[2][1] && m[2][2]);
        assert!(!m[2][3], "kausal: pos 2 tidak lihat pos 3");
    }

    #[test]
    fn rope_preserves_norm() {
        let x = Tensor::<3>::random(
            [1, 4, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &Device::default(),
        );
        let y = apply_rope(x.clone(), 10000.0);
        // rotasi menjaga norma per posisi
        let nx = (x.clone() * x.clone()).sum_dim(2);
        let ny = (y.clone() * y.clone()).sum_dim(2);
        let diff: f32 = (nx - ny).abs().max().into_scalar();
        assert!(diff < 1e-3, "norm berubah: {diff}");
    }
}
