//! Komponen jaringan VIR (RoPE, mask, attention, norm, FFN, blok, PTQ F7.2).

pub mod export;
pub mod quant;

use burn::module::Module;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Bool;
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
    pub use_rope: bool,
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
            use_rope: true,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_masked(x, None)
    }

    /// Forward dengan mask opsional [T, T] bool (true = boleh lihat).
    pub fn forward_masked(&self, x: Tensor<3>, mask: Option<Tensor<2, Bool>>) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let h = self.num_heads;
        let hd = self.head_dim;
        let xn = self.ln1.forward(x.clone());
        let q = if self.use_rope {
            apply_rope(self.wq.forward(xn.clone()), 10000.0)
        } else {
            self.wq.forward(xn.clone())
        }
        .reshape([b, t, h, hd])
        .swap_dims(1, 2);
        let k = if self.use_rope {
            apply_rope(self.wk.forward(xn.clone()), 10000.0)
        } else {
            self.wk.forward(xn.clone())
        }
        .reshape([b, t, h, hd])
        .swap_dims(1, 2);
        let v = self.wv.forward(xn).reshape([b, t, h, hd]).swap_dims(1, 2);
        let kt = k.clone().swap_dims(2, 3);
        let mut scores = q.matmul(kt) / (hd as f32).sqrt();
        if let Some(m) = mask {
            // mask [T,T] -> [1,1,T,T]
            let m = m.reshape([1, 1, t, t]);
            let neg = m.clone().bool_not();
            scores = scores.mask_fill(neg, f32::NEG_INFINITY);
        }
        let w = burn::tensor::activation::softmax(scores, 3);
        let ctx = w.matmul(v).swap_dims(1, 2).reshape([b, t, d]);
        let attn = self.wo.forward(ctx);
        let x = x + attn;
        let xn2 = self.ln2.forward(x.clone());
        let mlp = self.mlp.forward(xn2);
        x + mlp
    }
}

/// PTQ config (F7.2): quant scheme (dipakai modul `quant`; enum ini
/// kompatibel dengan `quant::QuantScheme`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    Int8,
    Int4,
    Fp8,
}

pub fn quantize_scale(scheme: QuantScheme) -> f32 {
    match scheme {
        QuantScheme::Int8 => 127.0,
        QuantScheme::Int4 => 7.0,
        QuantScheme::Fp8 => 448.0,
    }
}

/// KDA (Kimi Linear) satu langkah: `S_t = (I − β k kᵀ)·Diag(α)·S_{t−1} + β k vᵀ`.
///
/// Oracle naif (F5.2): satu iterasi per token, `O(d_k²·d_v)` per langkah.
/// `s` `[d_k, d_v]`, `k`/`alpha` `[d_k]`, `v` `[d_v]`, `beta` skalar.
///
/// Catatan: versi awal modul ini hanya menerapkan suku diagonal `I − β k kᵀ`
/// (tanpa `−β k (α ⊙ k)ᵀ S`), jadi bukan rekurensi PRD F5.2 langkah 1.
pub fn kda_step(
    s: &[[f32; 8]; 8],
    k: &[f32; 8],
    v: &[f32; 8],
    alpha: &[f32; 8],
    beta: f32,
) -> [[f32; 8]; 8] {
    let mut out = [[0.0f32; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            // (Diag(α)S)_{ij} − β·k_i·(kᵀ·Diag(α)·S)_j + β·k_i·v_j
            let mut projected = 0.0f32;
            for l in 0..8 {
                projected += k[l] * alpha[l] * s[l][j];
            }
            out[i][j] = alpha[i] * s[i][j] - beta * k[i] * projected + beta * k[i] * v[j];
        }
    }
    out
}

/// KDA chunkwise (transformasi UT, F5.2): setara [`kda_step`] untuk satu chunk.
///
/// `k`/`alpha` `[L][d_k]`, `v` `[L][d_v]`, `beta` `[L]`; `s` state awal chunk.
/// Panjang efektif = minimum panjang keempat masukan (pemanggil memberi panjang sama).
/// Kedua jalur identik secara aljabar; pada regime stabil (`‖k‖ = 1`, `β ≤ 1`)
/// selisih numerik ≈ 1e-7 (lihat tes).
pub fn kda_step_chunkwise(
    s: &[[f32; 8]; 8],
    k: &[[f32; 8]],
    v: &[[f32; 8]],
    alpha: &[[f32; 8]],
    beta: &[f32],
) -> [[f32; 8]; 8] {
    let len = k.len().min(v.len()).min(alpha.len()).min(beta.len());
    if len == 0 {
        return *s;
    }
    let u = kda_chunk_update(s, k, v, alpha, beta, len);
    kda_chunk_state(s, k, alpha, beta, &u, len)
}

/// `U` dari `(I + M)·U = W` (forward substitution), dengan `M` strictly lower.
fn kda_chunk_update(
    s: &[[f32; 8]; 8],
    k: &[[f32; 8]],
    v: &[[f32; 8]],
    alpha: &[[f32; 8]],
    beta: &[f32],
    len: usize,
) -> Vec<[f32; 8]> {
    let mut cum = [1.0f32; 8]; // `cum[i] = Π_{r<=t} α_r[i]`
    let mut u = vec![[0.0f32; 8]; len];
    for t in 0..len {
        for i in 0..8 {
            cum[i] *= alpha[t][i];
        }
        // `W[t] = v_t − S_0ᵀ·(cum ⊙ k_t)`.
        for j in 0..8 {
            let mut corr = 0.0f32;
            for i in 0..8 {
                corr += s[i][j] * cum[i] * k[t][i];
            }
            u[t][j] = v[t][j] - corr;
        }
        // `g[i] = Π_{r=c+1..t−1} α_r[i]`; produk kosong = 1 saat `c = t−1`.
        let mut g = [1.0f32; 8];
        for c in (0..t).rev() {
            let mut m = 0.0f32;
            for i in 0..8 {
                m += k[c][i] * g[i] * alpha[t][i] * k[t][i];
            }
            m *= beta[c];
            for j in 0..8 {
                u[t][j] -= m * u[c][j];
            }
            for i in 0..8 {
                g[i] *= alpha[c][i];
            }
        }
    }
    u
}

/// `S_L = (Π_r α_r)·S_0 + Σ_t (Π_{r>t} α_r)·β_t·k_t·u_tᵀ`.
fn kda_chunk_state(
    s: &[[f32; 8]; 8],
    k: &[[f32; 8]],
    alpha: &[[f32; 8]],
    beta: &[f32],
    u: &[[f32; 8]],
    len: usize,
) -> [[f32; 8]; 8] {
    let mut suffix = [1.0f32; 8]; // `suffix = Π_{r>t} α_r`
    let mut out = [[0.0f32; 8]; 8];
    for t in (0..len).rev() {
        for i in 0..8 {
            for j in 0..8 {
                out[i][j] += suffix[i] * beta[t] * k[t][i] * u[t][j];
            }
            suffix[i] *= alpha[t][i];
        }
    }
    for i in 0..8 {
        for j in 0..8 {
            out[i][j] += suffix[i] * s[i][j];
        }
    }
    out
}

pub fn nn_smoke() -> Result<(), VirError> {
    let device = Device::default();
    let blk = Block::new(64, 4, &device);
    let x: Tensor<3> = Tensor::ones([2, 4, 64], &device);
    let _ = blk.forward(x);
    Ok(())
}
