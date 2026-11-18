//! Generator NCM (PRD F1.6): masukan level dari rekonstruksi kumulatif yang
//! di-*down* (`Down_k`), inti rekuren (F2.1), head per level, dan `R_k` untuk
//! rekonstruksi kumulatif saat generasi.
//!
//! Faktor (PRD F1.6 langkah 1, NCM Pers. 4–5):
//!
//! ```text
//! p_θ(r_{1:K}) = Π_k p_θ(r_k | ẑ^(k−1))
//! p_θ(r_k^i | ẑ^(k−1)) = p_θ(r_k^i | z̃^(k−1))   (kode bebas bersyarat di level k)
//! ẑ^(k) = ẑ^(k−1) + R_k(e_k(r_k))               (F1.4.b rumus R2)
//! ```
//!
//! Masukan level k adalah `Down_k(ẑ^(k−1)) ∈ ℝ^{ℓ_k×d}` lalu proyeksi linear ke
//! lebar hidden (F1.6 langkah 2) — bukan pemotongan posisi: `Down_k` adalah
//! *average pooling* area yang menghormati panjang level (F1.4.b langkah 4).
//! Kode level k juga independen bersyarat, jadi satu lintasan menghasilkan
//! seluruh `ℓ_k` kode (F1.6 langkah 1).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Device, TensorData};
use vir_common::{LevelScheduleKind, VirError, level_schedule};
use vir_nn::Block;
use vir_vq::quantizer::replace_param;
use vir_vq::{Codebook, CodebookConfig, MultiscaleVq, Phi, Quantizer};

use crate::recurrent::recurrent_forward;
use crate::sampling::{Sampler, sample_codes};
use crate::scale::LevelScales;

/// Konfigurasi generator.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Lebar laten/hidden `d`.
    pub dim: usize,
    /// Jumlah head attention (F1.5: 64 per head pada skala NCM).
    pub num_heads: usize,
    /// Panjang level `ℓ_k` (naik ketat, berakhir `L`).
    pub levels: Vec<usize>,
    /// Blok prelude (F2.1: bentuk (1, 2, 1) pada skala kecil).
    pub prelude_blocks: usize,
    /// Blok inti yang diulang `r` kali.
    pub core_blocks: usize,
    /// Blok coda.
    pub coda_blocks: usize,
    /// Iterasi rekurensi saat inferensi (F2.1: tetap saat inferensi).
    pub num_loops: usize,
    /// Rata-rata iterasi saat latihan (F2.1 langkah 3).
    pub mean_loops: f32,
    /// Sigma log-normal untuk sampling `r`.
    pub loop_sigma: f32,
    /// Ukuran codebook `|V_k|` (4096 pada NCM Tabel 5).
    pub codebook_size: usize,
    /// Kernel konvolusi `Phi` pada `R_k` (`<= 1` memakai `Phi` nol).
    pub phi_kernel: usize,
}

impl GeneratorConfig {
    /// Konfigurasi dengan jadwal level eksplisit.
    pub fn new(dim: usize, num_heads: usize, levels: Vec<usize>) -> Self {
        Self {
            dim,
            num_heads,
            levels,
            prelude_blocks: 1,
            core_blocks: 2,
            coda_blocks: 1,
            num_loops: 2,
            mean_loops: 32.0,
            loop_sigma: 0.5,
            codebook_size: 4096,
            phi_kernel: 3,
        }
    }

    /// Jadwal level dari `vir_common` (F1.3) dengan `L = target_length`.
    pub fn scheduled(
        dim: usize,
        num_heads: usize,
        kind: LevelScheduleKind,
        target_length: usize,
        num_levels: usize,
    ) -> Self {
        Self::new(
            dim,
            num_heads,
            level_schedule(kind, target_length, num_levels),
        )
    }

    /// Jumlah level `K`.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Panjang konteks `L` (level terakhir).
    pub fn total_len(&self) -> usize {
        self.levels.last().copied().unwrap_or(0)
    }

    /// Validasi jadwal dan dimensi (R2).
    pub fn validate(&self) -> Result<(), VirError> {
        if self.dim == 0 || self.num_heads == 0 || self.dim % self.num_heads != 0 {
            return Err(VirError::Config(format!(
                "dim={} tidak habis dibagi num_heads={}",
                self.dim, self.num_heads
            )));
        }
        if self.levels.is_empty() {
            return Err(VirError::Config("jadwal level kosong".into()));
        }
        if self.levels[0] == 0 {
            return Err(VirError::Config("panjang level harus > 0".into()));
        }
        if self.levels.windows(2).any(|w| w[0] >= w[1]) {
            return Err(VirError::Config("jadwal level harus naik ketat".into()));
        }
        if self.codebook_size == 0 {
            return Err(VirError::Config("codebook_size harus > 0".into()));
        }
        if self.prelude_blocks == 0 || self.core_blocks == 0 || self.coda_blocks == 0 {
            return Err(VirError::Config("jumlah blok harus > 0".into()));
        }
        if self.num_loops == 0 {
            return Err(VirError::Config("num_loops harus > 0".into()));
        }
        Ok(())
    }
}

/// Hasil [`Generator::generate`] (F1.6 langkah 4).
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    /// Indeks kode per level, `[B, ℓ_k]`.
    pub codes: Vec<Tensor<2, Int>>,
    /// Rekonstruksi kumulatif `ẑ^(K)` pada konteks `prefix + L`.
    pub z_hat: Tensor<3>,
    /// Panjang konteks (prefix + blok).
    pub ctx_len: usize,
    /// Panjang prefix (0 bila tanpa prefix).
    pub prefix_len: usize,
    /// Ukuran batch.
    pub batch: usize,
    /// Posisi yang diproses (`Σ ℓ_k`, F1.6 langkah 5).
    pub positions: usize,
}

/// Generator NCM.
#[derive(Module, Debug)]
pub struct Generator {
    /// Embedding kode per level (bobot codebook tokenizer, beku).
    pub codebooks: Vec<Codebook>,
    /// `Phi_k` pada `R_k` (dari tokenizer, beku).
    pub phis: Vec<Phi>,
    /// Proyeksi `Down_k(ẑ)` d → hidden.
    pub proj: Linear,
    /// Prelude (F2.1).
    pub prelude: Vec<Block>,
    /// Inti yang diulang `r` kali (F2.1).
    pub core: Vec<Block>,
    /// Coda (F2.1).
    pub coda: Vec<Block>,
    /// Head per level: hidden → `|V_k|`.
    pub heads: Vec<Linear>,
    /// Adapter injeksi rekuren `[s, e]` (2h) → h (F2.1 langkah 1).
    pub adapter: Linear,
    #[module(skip)]
    scales: LevelScales,
    #[module(skip)]
    levels: Vec<usize>,
    #[module(skip)]
    dim: usize,
    #[module(skip)]
    num_loops: usize,
    #[module(skip)]
    mean_loops: f32,
    #[module(skip)]
    loop_sigma: f32,
}

impl Generator {
    /// Rakit generator dari konfigurasi.
    pub fn new(cfg: &GeneratorConfig, device: &Device) -> Result<Self, VirError> {
        cfg.validate()?;
        let codebooks = (0..cfg.num_levels())
            .map(|_| {
                Codebook::new(
                    &CodebookConfig {
                        num_entries: cfg.codebook_size,
                        dim: cfg.dim,
                        ..CodebookConfig::default()
                    },
                    device,
                )
            })
            .collect();
        let phis = (0..cfg.num_levels())
            .map(|_| {
                if cfg.phi_kernel > 1 {
                    Phi::new(cfg.dim, cfg.phi_kernel, device)
                } else {
                    Phi::zero(cfg.dim, 1, device)
                }
            })
            .collect();
        let heads = (0..cfg.num_levels())
            .map(|_| LinearConfig::new(cfg.dim, cfg.codebook_size).init(device))
            .collect();
        Ok(Self {
            codebooks,
            phis,
            proj: LinearConfig::new(cfg.dim, cfg.dim).init(device),
            prelude: blocks(cfg.prelude_blocks, cfg.dim, cfg.num_heads, device),
            core: blocks(cfg.core_blocks, cfg.dim, cfg.num_heads, device),
            coda: blocks(cfg.coda_blocks, cfg.dim, cfg.num_heads, device),
            heads,
            adapter: LinearConfig::new(cfg.dim * 2, cfg.dim).init(device),
            scales: LevelScales::new(&cfg.levels, cfg.total_len(), device),
            levels: cfg.levels.clone(),
            dim: cfg.dim,
            num_loops: cfg.num_loops,
            mean_loops: cfg.mean_loops,
            loop_sigma: cfg.loop_sigma,
        })
    }

    /// Jumlah level `K`.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Panjang level `ℓ_k`.
    pub fn level_len(&self, k: usize) -> usize {
        self.levels[k]
    }

    /// Panjang konteks `L`.
    pub fn total_len(&self) -> usize {
        self.levels.last().copied().unwrap_or(0)
    }

    /// Lebar model `d`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Iterasi rekurensi saat inferensi.
    pub fn num_loops(&self) -> usize {
        self.num_loops
    }

    /// Device parameter pertama.
    pub fn device(&self) -> Device {
        self.proj.weight.val().device()
    }

    /// Muat bobot codebook dan `Phi` dari tokenizer beku (F1.7 tahap 2).
    pub fn load_tokenizer(&mut self, vq: &MultiscaleVq<Codebook>) -> Result<(), VirError> {
        if vq.num_levels() != self.levels.len() {
            return Err(VirError::Config(format!(
                "tokenizer punya {} level, generator butuh {}",
                vq.num_levels(),
                self.levels.len()
            )));
        }
        if vq.total_len() != self.total_len() {
            return Err(VirError::Config(format!(
                "tokenizer L={}, generator L={}",
                vq.total_len(),
                self.total_len()
            )));
        }
        for k in 0..self.levels.len() {
            let level = &vq.levels[k];
            if level.scale.out_len() != self.levels[k] {
                return Err(VirError::Config(format!(
                    "level {k}: tokenizer ℓ={}, generator ℓ={}",
                    level.scale.out_len(),
                    self.levels[k]
                )));
            }
            if level.quantizer.dim() != self.dim {
                return Err(VirError::Config(format!(
                    "level {k}: tokenizer d={}, generator d={}",
                    level.quantizer.dim(),
                    self.dim
                )));
            }
            replace_param(
                &mut self.codebooks[k].weight,
                level.quantizer.weight.val().detach(),
            );
            replace_param(
                &mut self.phis[k].conv.weight,
                level.phi.conv.weight.val().detach(),
            );
        }
        Ok(())
    }

    /// Embedding kode level k: `[B, ℓ_k]` → `[B, ℓ_k, d]`.
    pub fn embed_codes(&self, k: usize, idx: Tensor<2, Int>) -> Tensor<3> {
        let [b, l] = idx.dims();
        let table = self.codebooks[k].weight.val();
        table
            .select(0, idx.reshape([b * l]))
            .reshape([b, l, self.dim])
    }

    /// Masukan level k: `proj(Down_k(ẑ^(k−1)))` (F1.6 langkah 2).
    pub fn level_input(&self, k: usize, z_hat: Tensor<3>) -> Tensor<3> {
        self.level_input_with(&self.scales, k, z_hat)
    }

    /// Sama dengan [`Self::level_input`] dengan operator skala eksplisit.
    pub fn level_input_with(&self, scales: &LevelScales, k: usize, z_hat: Tensor<3>) -> Tensor<3> {
        self.proj.forward(scales.down(k, z_hat))
    }

    /// Rekonstruksi `R_k(e_k(r_k))` satu level pada konteks `L`.
    pub fn reconstruct_level(&self, k: usize, codes: Tensor<3>) -> Tensor<3> {
        self.scales.reconstruct(k, &self.phis[k], codes)
    }

    /// Rekonstruksi kumulatif dari indeks semua level: `Σ_k R_k(e_k(r_k))`.
    pub fn reconstruct_codes(&self, codes: &[Tensor<2, Int>]) -> Result<Tensor<3>, VirError> {
        if codes.len() != self.levels.len() || codes.is_empty() {
            return Err(VirError::Config(format!(
                "butuh {} indeks level, diberi {}",
                self.levels.len(),
                codes.len()
            )));
        }
        let mut acc: Option<Tensor<3>> = None;
        for k in 0..self.levels.len() {
            let contribution = self.reconstruct_level(k, self.embed_codes(k, codes[k].clone()));
            acc = Some(match acc {
                Some(prev) => prev + contribution,
                None => contribution,
            });
        }
        acc.ok_or_else(|| VirError::Config("jadwal level kosong".into()))
    }

    /// Hidden state sebelum head tiap level (untuk probing dan gerbang mini).
    pub fn forward_features(&self, z_hats: &[Tensor<3>], r: usize) -> Vec<Tensor<3>> {
        self.forward_features_with(&self.scales, z_hats, r)
    }

    /// Sama dengan [`Self::forward_features`] dengan operator skala eksplisit.
    ///
    /// Irisan mini (F1.8) memakai konteks `prefix + L`, jadi `Down_k`/`R_k`
    /// dibangun untuk panjang konteks itu; operator yang sama dipakai saat
    /// latihan dan generasi agar fitur identik bit demi bit.
    pub fn forward_features_with(
        &self,
        scales: &LevelScales,
        z_hats: &[Tensor<3>],
        r: usize,
    ) -> Vec<Tensor<3>> {
        let n = z_hats.len().min(self.heads.len());
        if n == 0 {
            return Vec::new();
        }
        // B4: level-level independen (attention per-sequence, blok di-share).
        // Batch jadi `[B·n, L, d]` → satu lintasan blok, bukan `n` lintasan.
        // Numerik identik (matmul/LayerNorm/softmax per-baris), 5× lebih sedikit
        // luncuran kernel + matmul lebih besar (GPU lebih terisi).
        let inputs: Vec<Tensor<3>> = (0..n)
            .map(|k| self.level_input_with(scales, k, z_hats[k].clone()))
            .collect();
        let [b, l, d] = inputs[0].dims();
        let batched = Tensor::cat(inputs, 0);
        let out = recurrent_forward(
            &self.prelude,
            &self.core,
            &self.coda,
            &self.adapter,
            batched,
            r,
        );
        (0..n)
            .map(|k| out.clone().slice([k * b..(k + 1) * b, 0..l, 0..d]))
            .collect()
    }

    /// Operator skala untuk konteks `ctx_len` (mis. `prefix + L`).
    pub fn scales_for(&self, ctx_len: usize) -> LevelScales {
        LevelScales::new(&self.levels, ctx_len, &self.device())
    }

    /// Forward latihan (Algoritma 1): satu lintasan untuk semua level, `r` tetap.
    pub fn forward(&self, z_hats: &[Tensor<3>]) -> Vec<Tensor<3>> {
        self.forward_loops(z_hats, self.num_loops)
    }

    /// Forward dengan `r` iterasi rekurensi eksplisit (inferensi, F2.1).
    pub fn forward_loops(&self, z_hats: &[Tensor<3>], r: usize) -> Vec<Tensor<3>> {
        let n = z_hats.len().min(self.heads.len());
        let features = self.forward_features(z_hats, r);
        (0..n)
            .map(|k| self.heads[k].forward(features[k].clone()))
            .collect()
    }

    /// Forward dengan operator skala eksplisit (konteks `prefix + L`).
    pub fn forward_with(&self, scales: &LevelScales, z_hats: &[Tensor<3>]) -> Vec<Tensor<3>> {
        let features = self.forward_features_with(scales, z_hats, self.num_loops);
        (0..features.len())
            .map(|k| self.heads[k].forward(features[k].clone()))
            .collect()
    }

    /// Satu `r` per mikro-batch (locked-step, F2.1 langkah 3).
    pub fn forward_sampled(
        &self,
        z_hats: &[Tensor<3>],
        rng: &mut impl rand::Rng,
    ) -> (Vec<Tensor<3>>, usize) {
        let r = crate::recurrent::sample_r(self.mean_loops, self.loop_sigma, rng);
        (self.forward_loops(z_hats, r), r)
    }

    /// Algoritma 2 (F1.6 langkah 4): level demi level, `ℓ_k` kode per level,
    /// `ẑ^(k) = ẑ^(k−1) + R_k(e_k(r_k))`.
    pub fn generate(
        &self,
        prefix: Option<Tensor<3>>,
        sampler: &Sampler,
    ) -> Result<GenerateOutput, VirError> {
        let (cur, scales, batch, prefix_len) = self.init_context(prefix)?;
        self.run_levels(cur, scales, batch, prefix_len, sampler)
    }

    /// Algoritma 2 dengan konteks awal eksplisit (prompt + benih blok).
    ///
    /// Irisan mini (F1.8) memakai ini: level 0 bersyarat pada `ẑ^(−1) = 0`,
    /// jadi tanpa konteks awal semua sampel identik. Konteks `[B, P+L, d]`
    /// memberi identitas per (sampel, posisi) sehingga kepala dapat menghafal.
    pub fn generate_from(
        &self,
        init: Tensor<3>,
        prefix_len: usize,
        sampler: &Sampler,
    ) -> Result<GenerateOutput, VirError> {
        let [batch, ctx_len, dim] = init.dims();
        if dim != self.dim {
            return Err(VirError::Config(format!(
                "init d={dim}, butuh d={}",
                self.dim
            )));
        }
        if ctx_len < self.total_len() {
            return Err(VirError::Config(format!(
                "init ctx={ctx_len} < L={}",
                self.total_len()
            )));
        }
        let scales = LevelScales::new(&self.levels, ctx_len, &self.device());
        self.run_levels(init, scales, batch, prefix_len, sampler)
    }

    /// Loop Algoritma 2 di atas konteks dan operator skala yang sudah siap.
    fn run_levels(
        &self,
        mut cur: Tensor<3>,
        scales: LevelScales,
        batch: usize,
        prefix_len: usize,
        sampler: &Sampler,
    ) -> Result<GenerateOutput, VirError> {
        let mut codes = Vec::with_capacity(self.levels.len());
        let mut positions = 0usize;
        for k in 0..self.levels.len() {
            let x = self.level_input_with(&scales, k, cur.clone());
            let hidden = recurrent_forward(
                &self.prelude,
                &self.core,
                &self.coda,
                &self.adapter,
                x,
                self.num_loops,
            );
            let logits = self.heads[k].forward(hidden);
            let idx = sample_codes(logits, &sampler.for_level(k));
            let codes_k = self.embed_codes(k, idx.clone());
            cur = cur + scales.reconstruct(k, &self.phis[k], codes_k);
            positions += self.levels[k];
            codes.push(idx);
        }
        Ok(GenerateOutput {
            codes,
            ctx_len: scales.ctx_len(),
            z_hat: cur,
            prefix_len,
            batch,
            positions,
        })
    }

    /// Satu blok: `ẑ^(K)` pada konteks `prefix + L` (F2.3 langkah 3).
    pub fn generate_block(
        &self,
        prefix: Option<Tensor<3>>,
        sampler: &Sampler,
    ) -> Result<Tensor<3>, VirError> {
        Ok(self.generate(prefix, sampler)?.z_hat)
    }

    /// Siapkan konteks: prefix (beku) + blok nol, dan operator skala.
    fn init_context(
        &self,
        prefix: Option<Tensor<3>>,
    ) -> Result<(Tensor<3>, LevelScales, usize, usize), VirError> {
        let device = self.device();
        let prefix_len = prefix.as_ref().map_or(0, |p| p.dims()[1]);
        let batch = prefix.as_ref().map_or(1, |p| p.dims()[0]);
        if let Some(p) = &prefix {
            let [_, pl, pd] = p.dims();
            if pd != self.dim {
                return Err(VirError::Config(format!(
                    "prefix d={pd}, butuh d={}",
                    self.dim
                )));
            }
            if pl == 0 {
                return Err(VirError::Config(
                    "prefix harus [B, P, d] dengan P > 0".into(),
                ));
            }
        }
        let scales = if prefix_len == 0 {
            self.scales.clone()
        } else {
            LevelScales::new(&self.levels, self.total_len() + prefix_len, &device)
        };
        let block = Tensor::<3>::zeros([batch, self.total_len(), self.dim], &device);
        let cur = match prefix {
            Some(p) => Tensor::cat(vec![p, block], 1),
            None => block,
        };
        Ok((cur, scales, batch, prefix_len))
    }
}

/// Rakit `count` blok pre-norm (F1.5).
fn blocks(count: usize, dim: usize, num_heads: usize, device: &Device) -> Vec<Block> {
    (0..count)
        .map(|_| Block::new(dim, num_heads, device))
        .collect()
}

/// Tulis bobot kepala linear dari data host (`[d_in, d_out]`, baris-major).
///
/// Burn 0.22 menyimpan bobot `Linear` sebagai `[d_input, d_output]` dan
/// menghitung `y = x·W` (lihat `burn-nn` `linear.rs`). Bias ditulis nol agar
/// `logits = x·W` murni, sehingga pembelajaran host (Adam atau kuadrat terkecil)
/// dan jalur Burn menghasilkan angka yang sama. `ParamId` dipertahankan agar
/// optimizer dan checkpoint tetap cocok.
pub fn replace_head(
    head: &mut Linear,
    weight: Vec<f32>,
    d_in: usize,
    d_out: usize,
    device: &Device,
) -> Result<(), VirError> {
    if weight.len() != d_in * d_out {
        return Err(VirError::Config(format!(
            "bobot kepala butuh {} nilai, diberi {}",
            d_in * d_out,
            weight.len()
        )));
    }
    replace_param(
        &mut head.weight,
        Tensor::<2>::from_floats(TensorData::new(weight, [d_in, d_out]), device),
    );
    if let Some(bias) = head.bias.as_mut() {
        replace_param(bias, Tensor::<1>::zeros([d_out], device));
    }
    Ok(())
}
