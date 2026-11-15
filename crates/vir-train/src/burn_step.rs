//! `StepFn` nyata (PRD F1.7, F2.5.b): `VirStepFn` menghubungkan model VIR
//! ([`crate::model::VirTrainModel`]) ke [`Trainer`] tanpa mengubah trainer.
//!
//! Satu langkah:
//!
//! ```text
//! batch [B, L] dari BatchSource (posisi ctx.step)
//!   → store → model (params::load_store)
//!   → tahap Vq:  embed → VQ(+EMA) → decoder → CE + komitmen + laten (R4)
//!     tahap Gen: embed → VQ(≈) → kode target + masukan teacher (korupsi)
//!                → generator → L_NCM berbobot level (R6)
//!   → loss.backward() → gradien per parameter → slot ParamStore
//!   → codebook/EMA + salinan tokenizer → slot Tokenizer (ikut checkpoint)
//! ```
//!
//! Pembagian metrik `StepOut`:
//!
//! | medan | tahap Vq | tahap Gen |
//! |---|---|---|
//! | `loss_total` | R4 (CE + komitmen + laten) | `Σ_k w_k·CE_k` |
//! | `per_level` | MSE rekonstruksi `‖z − ẑ^(k)‖²` | CE berbobot per level |
//! | `utilization`/`perplexity` | codebook | codebook tokenizer (beku) |
//! | `recon` | akurasi token | `None` (gerbang hanya tahap 1) |

use burn::prelude::*;
use burn::tensor::{Device, TensorData};
use vir_common::VirError;
use vir_model::ncm_loss;
use vir_vq::{Codebook, MultiscaleVq, Quantizer, tensor_accuracy, vq_loss_chunked};

use crate::corruption::Corruption;
use crate::data::{BatchSource, ShardBatchSource};
use crate::model::{VirTrainConfig, VirTrainModel, train_device};
use crate::optim::ParamStore;
use crate::params;
use crate::trainer::{Stage, StepCtx, StepFn, StepOut};

/// Langkah latihan nyata di atas `VirTrainModel` + `BatchSource`.
pub struct VirStepFn {
    model: Option<VirTrainModel>,
    source: ShardBatchSource,
    cfg: VirTrainConfig,
    device: Device,
    weights: Vec<f32>,
    consumed: u64,
    tokenizer_loaded: bool,
    grad_slots: usize,
    copied_slots: usize,
}

impl VirStepFn {
    /// Rakit langkah + `ParamStore` awal dari bobot model.
    ///
    /// `stage` menentukan slot mana yang beku: pada tahap Vq generator dibekukan
    /// (dan sebaliknya); codebook + salinan tokenizer selalu beku terhadap
    /// optimizer karena diperbarui EMA/di-load dari tokenizer.
    pub fn new(
        cfg: VirTrainConfig,
        source: ShardBatchSource,
        device: Device,
        stage: Stage,
    ) -> Result<(Self, ParamStore), VirError> {
        cfg.validate()?;
        let len = source.seq_len();
        if len != cfg.total_len() {
            return Err(VirError::Config(format!(
                "L batch {len} != L model {}",
                cfg.total_len()
            )));
        }
        let device = train_device(device, cfg.seed);
        let model = VirTrainModel::new(&cfg, &device)?;
        let store = params::build_store(&model, stage)?;
        let weights = crate::loss::level_weight(&cfg.levels, cfg.alpha);
        let step = Self {
            model: Some(model),
            source,
            cfg,
            device,
            weights,
            consumed: 0,
            tokenizer_loaded: false,
            grad_slots: 0,
            copied_slots: 0,
        };
        Ok((step, store))
    }

    /// Konfigurasi model.
    pub fn config(&self) -> &VirTrainConfig {
        &self.cfg
    }

    /// Device aktif.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Sumber batch.
    pub fn source(&self) -> &ShardBatchSource {
        &self.source
    }

    /// Jumlah slot yang menerima gradien pada langkah terakhir.
    pub fn grad_slots(&self) -> usize {
        self.grad_slots
    }

    /// Jumlah slot `Tokenizer` yang disalin model → store pada langkah terakhir.
    pub fn copied_slots(&self) -> usize {
        self.copied_slots
    }

    /// SINKRON: posisi batch dan status beku store menurut tahap aktif.
    ///
    /// Salinan tokenizer di generator tidak dimuat di sini: ia harus menimpa
    /// nilai dari store (lihat [`Self::sync_tokenizer`]), bukan ditimpa olehnya.
    fn prepare(&mut self, step: u64, stage: Stage, store: &mut ParamStore) -> Result<(), VirError> {
        if step != self.consumed {
            self.source.seek_step(step)?;
        }
        params::sync_frozen(stage, store);
        Ok(())
    }

    /// Salin tokenizer VQ ke generator sekali per `VirStepFn` (F1.7 tahap 2).
    ///
    /// Dijalankan *setelah* `load_store` agar generator memakai codebook/Phi
    /// tokenizer hasil EMA, bukan bobot awal store; tanpa ini salinan tokenizer
    /// di generator tidak pernah berubah dan `copy_back` tak punya nilai baru.
    fn sync_tokenizer(&mut self, model: &mut VirTrainModel, stage: Stage) -> Result<(), VirError> {
        if stage != Stage::Generator || self.tokenizer_loaded {
            return Ok(());
        }
        let vq = &model.vq;
        let generator = &mut model.generator;
        generator.load_tokenizer(vq)?;
        self.tokenizer_loaded = true;
        Ok(())
    }

    /// Batch token berikutnya tervalidasi (L dan rentang vocab).
    fn next_tokens(&mut self) -> Result<(Vec<i64>, [usize; 2]), VirError> {
        let (tokens, shape) = self.source.next_batch()?;
        if shape[1] != self.cfg.total_len() {
            return Err(VirError::Internal(format!(
                "L batch {} != L model {}",
                shape[1],
                self.cfg.total_len()
            )));
        }
        for t in &tokens {
            if *t < 0 || *t as usize >= self.cfg.vocab {
                return Err(VirError::Config(format!(
                    "token {t} di luar vocab {}",
                    self.cfg.vocab
                )));
            }
        }
        Ok((tokens, shape))
    }

    /// Tahap 1: tokenizer VQ + decoder (R4).
    fn vq_forward(
        &mut self,
        model: &mut VirTrainModel,
        ctx: &mut StepCtx<'_>,
    ) -> Result<(Tensor<1>, StepOut), VirError> {
        let (tokens, shape) = self.next_tokens()?;
        let target = tokens_tensor(&tokens, shape, &self.device);
        let z = model.embed_tokens(&target);
        let enc = model.encode_with_ema(z.clone(), ctx.rng)?;
        let loss = vq_loss_chunked(
            &model.dec,
            z.clone(),
            enc.z_hat.clone(),
            target.clone(),
            self.cfg.commitment_cost,
            self.cfg.chunk,
        );
        let loss_value = loss.clone().into_scalar::<f32>();
        let pred = model.dec.argmax_tokens(enc.z_hat);
        let recon = tensor_accuracy(&pred, &target) as f32;
        let per_level = recon_errors(&z, &enc.levels);
        let (utilization, perplexity) = codebook_stats(&model.vq, &enc.indices);
        let level_tokens = level_tokens(&self.cfg, shape[0]);
        Ok((
            loss,
            StepOut {
                loss_total: loss_value,
                per_level,
                level_tokens,
                utilization,
                perplexity,
                recon: Some(recon),
                corruption_rate: None,
            },
        ))
    }

    /// Tahap 2: generator NCM dengan tokenizer beku (R6).
    fn gen_forward(
        &mut self,
        model: &VirTrainModel,
        ctx: &mut StepCtx<'_>,
    ) -> Result<(Tensor<1>, StepOut), VirError> {
        let (tokens, shape) = self.next_tokens()?;
        let target_tokens = tokens_tensor(&tokens, shape, &self.device);
        let z = model.embed_tokens(&target_tokens).detach();
        let enc = model.encode(z);
        let clean = enc.indices;
        let (corrupted, rate) = self.corrupt(&clean, ctx.corruption.as_deref_mut());
        let inputs = teacher_inputs(model, &corrupted, self.cfg.dim, &self.device);
        let logits = model.generator.forward(&inputs);
        let loss = ncm_loss(&logits, &clean, &self.weights);
        let mut per_level = Vec::with_capacity(logits.len());
        let mut perplexity = Vec::with_capacity(logits.len());
        for (k, logit) in logits.iter().enumerate() {
            let ce = level_ce(logit, &clean[k]);
            per_level.push(ce * self.weights[k]);
            perplexity.push(ce.exp());
        }
        let (utilization, _) = codebook_stats(&model.vq, &clean);
        let loss_total = loss.clone().into_scalar::<f32>();
        Ok((
            loss,
            StepOut {
                loss_total,
                per_level,
                level_tokens: level_tokens(&self.cfg, shape[0]),
                utilization,
                perplexity,
                recon: None,
                corruption_rate: rate,
            },
        ))
    }

    /// Terapkan korupsi kode masukan (PRD F1.7 langkah 2); target tetap bersih.
    fn corrupt(
        &self,
        clean: &[Tensor<2, Int>],
        corruption: Option<&mut Corruption>,
    ) -> (Vec<Tensor<2, Int>>, Option<f32>) {
        let Some(c) = corruption else {
            return (clean.to_vec(), None);
        };
        let batch = clean.first().map_or(0, |t| t.dims()[0]);
        let mut batches = Vec::with_capacity(clean.len());
        for k in 0..clean.len() {
            let len = self.cfg.level_len(k);
            let codes = self.cfg.codebook_size;
            batches.push(c.next_batch(k, batch, len, codes));
        }
        let out = c.apply_levels(clean, &batches);
        (out, Some(c.mean_measured()))
    }
}

impl StepFn for VirStepFn {
    fn compute(
        &mut self,
        mut ctx: StepCtx<'_>,
        store: &mut ParamStore,
    ) -> Result<StepOut, VirError> {
        let step = ctx.step;
        self.prepare(step, ctx.stage, store)?;
        let model = self.model.take().ok_or_else(missing)?;
        let mut model = params::load_store(model, store)?;
        self.sync_tokenizer(&mut model, ctx.stage)?;
        let outcome = self.step_once(&mut model, ctx.stage, store, &mut ctx);
        self.model = Some(model);
        let out = outcome?;
        self.consumed = step + 1;
        Ok(out)
    }
}

impl VirStepFn {
    /// Satu langkah model: forward → loss → gradien → slot store.
    fn step_once(
        &mut self,
        model: &mut VirTrainModel,
        stage: Stage,
        store: &mut ParamStore,
        ctx: &mut StepCtx<'_>,
    ) -> Result<StepOut, VirError> {
        let (loss, out) = match stage {
            Stage::Vq => self.vq_forward(model, ctx)?,
            Stage::Generator => self.gen_forward(model, ctx)?,
        };
        if !loss.is_tracked() {
            return Err(VirError::Internal(
                "loss tidak terlacak autodiff (device tanpa gradien?)".into(),
            ));
        }
        let grads = loss.backward();
        let flat = params::collect_grads(model, &grads);
        self.grad_slots = params::write_grads(store, &flat)?;
        self.copied_slots = params::copy_back(model, store)?;
        Ok(out)
    }
}

/// Kembalikan model yang hilang sebagai galat (bukan panic).
fn missing() -> VirError {
    VirError::Internal("VirStepFn tanpa model".into())
}

/// Tensor token `[B, L]` dari data host.
fn tokens_tensor(tokens: &[i64], shape: [usize; 2], device: &Device) -> Tensor<2, Int> {
    Tensor::<2, Int>::from_ints(TensorData::new(tokens.to_vec(), shape), device)
}

/// MSE rekonstruksi kumulatif per level: `‖z − ẑ^(k)‖²`.
fn recon_errors(z: &Tensor<3>, levels: &[Tensor<3>]) -> Vec<f32> {
    levels
        .iter()
        .map(|zh| {
            let d = zh.clone() - z.clone();
            (d.clone() * d).mean().into_scalar::<f32>()
        })
        .collect()
}

/// Jumlah posisi token per level (`B·ℓ_k`).
fn level_tokens(cfg: &VirTrainConfig, batch: usize) -> Vec<f32> {
    cfg.levels.iter().map(|l| (batch * l) as f32).collect()
}

/// Utilisasi dan perplexity codebook per level.
fn codebook_stats(vq: &MultiscaleVq<Codebook>, indices: &[Tensor<2, Int>]) -> (Vec<f32>, Vec<f32>) {
    let mut utilization = Vec::with_capacity(indices.len());
    let mut perplexity = Vec::with_capacity(indices.len());
    for (k, idx) in indices.iter().enumerate() {
        let [b, l] = idx.dims();
        let stats = vq.levels[k].quantizer.stats(&idx.clone().reshape([b * l]));
        utilization.push(stats.utilization);
        perplexity.push(stats.perplexity);
    }
    (utilization, perplexity)
}

/// CE satu level (tanpa bobot) dari logits dan target kode.
fn level_ce(logits: &Tensor<3>, target: &Tensor<2, Int>) -> f32 {
    let loss = vir_model::level_losses(
        std::slice::from_ref(logits),
        std::slice::from_ref(target),
        &[1.0],
    );
    loss.first()
        .map(|t| t.clone().into_scalar::<f32>())
        .unwrap_or(0.0)
}

/// Masukan teacher forcing (F1.6 langkah 3) dari kode (boleh terkorupsi).
///
/// `z̃^(k) = z̃^(k−1) + R_k(e_k(r̃_k))` dengan `z̃^(−1) = 0`; operator `R_k`
/// milik generator agar identik dengan loop `generate`.
///
/// Rekonstruksi kumulatif hidup pada **panjang konteks penuh** `L`: `R_k`
/// memetakan `[B, ℓ_k, d] → [B, L, d]` (naik/in interpolasi `LevelScales`), jadi
/// `ẑ^(k)` tidak boleh dialokasikan pada `ℓ_k` — menjumlahkannya dengan kontribusi
/// level berikutnya akan gagal bentuk (`ℓ_k != ℓ_{k+1}`).
fn teacher_inputs(
    model: &VirTrainModel,
    idx: &[Tensor<2, Int>],
    dim: usize,
    device: &Device,
) -> Vec<Tensor<3>> {
    let batch = idx.first().map_or(0, |t| t.dims()[0]);
    let ctx = model.total_len();
    let mut cur = Tensor::<3>::zeros([batch, ctx, dim], device);
    let mut out = Vec::with_capacity(idx.len());
    for (k, codes) in idx.iter().enumerate() {
        out.push(cur.clone());
        let e = model.generator.embed_codes(k, codes.clone());
        cur = cur + model.generator.reconstruct_level(k, e);
    }
    out
}
