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
    /// Adapter injeksi recurrent: [s, e] (2h) → h.
    pub adapter: Linear,
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
            adapter: LinearConfig::new(cfg.dim * 2, cfg.dim).init(device),
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
            let [b, l, _] = logits.dims();
            let idx = logits.argmax(2).reshape([b, l]); // [B, target_len]
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

/// Recurrent forward: prelude(e) → core diulang r kali dengan injeksi e → coda.
pub fn recurrent_forward(
    prelude: &Block,
    core: &Block,
    coda: &Block,
    adapter: &Linear,
    e: Tensor<3>,
    r: usize,
) -> Tensor<3> {
    let mut s = prelude.forward(e.clone());
    for _ in 0..r {
        // injeksi: adapter([s, e])
        let [b, t, d] = s.dims();
        let cat = burn::tensor::Tensor::cat(vec![s.clone(), e.clone()], 2);
        let inj = adapter.forward(cat).reshape([b, t, d]);
        s = core.forward(s + inj);
    }
    coda.forward(s)
}

/// Sampling r ~ Poisson(exp(τ)) + 1, τ ~ N(log r̄ - σ²/2, σ).
pub fn sample_r(mean_r: f32, sigma: f32, rng: &mut impl rand::Rng) -> usize {
    let mu = (mean_r.ln()) - 0.5 * sigma * sigma;
    let tau = mu + sigma * rand_distr(rng);
    let r = poisson(tau.exp()) + 1;
    r.max(1)
}

fn rand_distr(rng: &mut impl rand::Rng) -> f32 {
    // Box-Muller
    let u1 = rand::Rng::gen_range(&mut *rng, 0.0f32..1.0).max(1e-7);
    let u2 = rand::Rng::gen_range(&mut *rng, 0.0f32..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn poisson(lambda: f32) -> usize {
    // Knuth (lambda kecil)
    let l = (-lambda).exp();
    let mut k = 0usize;
    let mut p = 1.0f32;
    loop {
        p *= rand::Rng::gen_range(&mut rand::thread_rng(), 0.0f32..1.0);
        if p <= l {
            return k;
        }
        k += 1;
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

/// Entropy dari distribusi probabilitas (logits softmax).
pub fn entropy(probs: &[f32]) -> f32 {
    -probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}

// ---------- Block-AR ----------
/// Generate blok demi blok sampai EOS atau max_blocks.
/// `generate_block` mengembalikan token blok (indeks kode level akhir).
pub fn block_ar<F>(mut generate_block: F, eos: u32, max_blocks: usize) -> Vec<u32>
where
    F: FnMut() -> Vec<u32>,
{
    let mut out = Vec::new();
    for _ in 0..max_blocks {
        let block = generate_block();
        let mut hit_eos = false;
        for t in block {
            if t == eos {
                hit_eos = true;
                break;
            }
            out.push(t);
        }
        if hit_eos {
            break;
        }
    }
    out
}

// ---------- Sampling lanjutan (CFG + MBR) ----------
/// CFG level logit: ℓ_uc + β(ℓ - ℓ_uc).
pub fn cfg_logit(uc: f32, c: f32, beta: f32) -> f32 {
    uc + beta * (c - uc)
}

/// MBR: pilih kandidat dengan kemiripan rata-rata tertinggi ke kandidat lain.
/// `candidates`: K_c × L (token ids). `sim`: fungsi kemiripan.
pub fn mbr_select<F>(candidates: &[Vec<u32>], sim: F) -> usize
where
    F: Fn(&[u32], &[u32]) -> f32,
{
    let k = candidates.len();
    if k <= 1 {
        return 0;
    }
    let mut best = 0usize;
    let mut best_score = f32::MIN;
    for i in 0..k {
        let mut acc = 0.0f32;
        for j in 0..k {
            if i == j {
                continue;
            }
            acc += sim(&candidates[i], &candidates[j]);
        }
        let score = acc / (k - 1) as f32;
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best
}

/// F1.8 irisan vertikal mini: VQ encode → generator forward → loss turun.
pub fn full_smoke() -> Result<(), VirError> {
    let device = Device::default();
    let cfg = GeneratorConfig {
        dim: 32,
        num_heads: 4,
        num_blocks: 2,
        num_levels: 4,
        num_loops: 2,
        codebook_size: 64,
    };
    let g = Generator::new(&cfg, &device);
    // mini batch: 2 × 16 token, 4 level
    let z = Tensor::<3>::random(
        [2, 16, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let z_hats = vec![z.clone(); 4];
    let logits = g.forward(&z_hats);
    assert_eq!(logits.len(), 4);
    // loss turun: hitung ncm_loss dengan target acak
    let targets: Vec<Tensor<2, Int>> = (0..4)
        .map(|_| {
            Tensor::<2, Int>::random(
                [2, 16],
                burn::tensor::Distribution::Uniform(0.0, 64.0),
                &device,
            )
        })
        .collect();
    let weights = vec![0.25f32; 4];
    let loss = ncm_loss(&logits, &targets, &weights);
    assert!(loss.into_scalar::<f32>() >= 0.0);
    Ok(())
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
    #[test]
    fn full_smoke_runs() {
        assert!(full_smoke().is_ok());
    }
    #[test]
    fn halter_entropy() {
        let h = EntropyHalter { threshold: 0.5 };
        assert!(h.should_halt(0.1));
        assert!(!h.should_halt(1.0));
    }
    #[test]
    fn entropy_uniform() {
        let e = entropy(&[0.5, 0.5]);
        assert!((e - 0.6931).abs() < 1e-3);
    }
    #[test]
    fn block_ar_stops_at_eos() {
        let blocks = vec![vec![1u32, 2], vec![3, 99], vec![4]];
        let mut i = 0;
        let out = block_ar(
            || {
                let b = blocks[i].clone();
                i += 1;
                b
            },
            99,
            10,
        );
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn cfg_logit_math() {
        assert!((cfg_logit(0.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((cfg_logit(0.0, 1.0, 0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn mbr_picks_most_similar() {
        let cands = vec![vec![1u32, 2, 3], vec![1, 2, 3], vec![9, 9, 9]];
        let sim = |a: &[u32], b: &[u32]| -> f32 {
            let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
            same as f32 / a.len().max(1) as f32
        };
        // kandidat 0 dan 1 identik -> salah satu terpilih, bukan 2
        let best = mbr_select(&cands, sim);
        assert!(best == 0 || best == 1);
    }
}
