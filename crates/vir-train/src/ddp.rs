//! Data-parallel multi-GPU (DDP-style) di atas `Trainer`/`ParamStore` (F2.5.c).
//!
//! `Trainer` memakai `ParamStore` host (f32) + `StepFn` manual (bukan
//! `Learner` Burn), jadi DDP ditulis di atas `StepFn` — tanpa mengubah
//! `Trainer`/optimizer:
//!
//! ```text
//! DdpStepFn {
//!   ranks: Vec<VirRank>,      // satu replika model per device
//!   reducer: Box<dyn GradAllReduce>,
//! }
//! VirRank { step: VirStepFn, store: ParamStore }  // grad lokal
//! ```
//!
//! Satu langkah:
//! 1. Tiap rank menyinkronkan bobot dari store bersama → store lokal.
//! 2. Tiap rank memproses **shard data berbeda** (batch dibagi world).
//! 3. Gradien lokal di-*all-reduce* (Mean) → store bersama.
//! 4. `Trainer` melakukan optimizer step sekali (host), lalu bobot baru
//!    disinkronkan ke semua replika pada langkah berikutnya.
//!
//! `world = 1` (tanpa GPU / 1 device) identik dengan `VirStepFn` biasa.

use burn::tensor::Device;
use vir_common::VirError;

use crate::burn_step::VirStepFn;
use crate::model::VirTrainConfig;
use crate::optim::ParamStore;
use crate::trainer::{Stage, StepCtx, StepFn, StepOut};

/// Kumpulan device untuk latihan data-parallel.
#[derive(Debug, Clone)]
pub struct DevicePool {
    devices: Vec<Device>,
}

impl DevicePool {
    /// Deteksi device CUDA; jatuh ke satu device default bila tak ada.
    ///
    /// `Device::enumerate(DeviceType::Cuda)` mengembalikan daftar kosong bila
    /// backend CUDA tidak dikompilasi (mis. CI CPU), sehingga `world = 1`.
    pub fn detect() -> Self {
        #[cfg(feature = "cuda")]
        {
            let devices = burn::tensor::Device::enumerate(burn::tensor::DeviceType::Cuda);
            if !devices.is_empty() {
                return Self {
                    devices: devices.into_vec(),
                };
            }
        }
        Self {
            devices: vec![Device::default()],
        }
    }

    /// Daftar device eksplisit (untuk tes / override).
    pub fn new(devices: Vec<Device>) -> Self {
        if devices.is_empty() {
            Self {
                devices: vec![Device::default()],
            }
        } else {
            Self { devices }
        }
    }

    /// Jumlah rank (world size).
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    /// Device per rank.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }
}

/// All-reduce gradien lintas rank.
pub trait GradAllReduce {
    /// Rata-rata elementwise gradien lintas rank (in-place).
    ///
    /// `grads[r]` = buffer grad pihak `r`; setelah dipanggil semua buffer
    /// berisi nilai mean.
    fn all_reduce_mean(&self, grads: &mut [Vec<f32>]) -> Result<(), VirError>;
}

/// All-reduce di host (portabel, tanpa NCCL): jumlahkan lalu bagi `world`.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostAllReduce;

impl GradAllReduce for HostAllReduce {
    fn all_reduce_mean(&self, grads: &mut [Vec<f32>]) -> Result<(), VirError> {
        let world = grads.len();
        if world == 0 {
            return Ok(());
        }
        if world == 1 {
            return Ok(());
        }
        let len = grads.iter().map(Vec::len).find(|l| *l != 0).unwrap_or(0);
        if grads.iter().any(|g| g.len() != len) {
            return Err(VirError::Config(
                "panjang grad tidak konsisten antar rank".into(),
            ));
        }
        let denom = world as f32;
        for i in 0..len {
            let mut sum = 0.0f32;
            for g in grads.iter() {
                sum += g[i];
            }
            let mean = sum / denom;
            for g in grads.iter_mut() {
                g[i] = mean;
            }
        }
        Ok(())
    }
}

/// Satu rank: replika model + store grad lokal.
struct VirRank {
    step: VirStepFn,
    store: ParamStore,
}

/// `StepFn` data-parallel: membungkus N rank di N device.
pub struct DdpStepFn {
    ranks: Vec<VirRank>,
    reducer: Box<dyn GradAllReduce>,
    world: usize,
}

impl DdpStepFn {
    /// Rakit DDP dari config, sumber data per rank, dan pool device.
    ///
    /// `sources` panjangnya harus sama dengan `pool.world_size()`; tiap rank
    /// memakai shard sendiri. Mengembalikan store bersama (yang di-step
    /// `Trainer`).
    pub fn new(
        cfg: VirTrainConfig,
        sources: Vec<crate::data::ShardBatchSource>,
        pool: &DevicePool,
        stage: Stage,
        reducer: Box<dyn GradAllReduce>,
    ) -> Result<(Self, ParamStore), VirError> {
        let world = pool.world_size();
        if sources.len() != world {
            return Err(VirError::Config(format!(
                "jumlah sumber {} != world {}",
                sources.len(),
                world
            )));
        }
        let mut ranks = Vec::with_capacity(world);
        for (source, device) in sources.into_iter().zip(pool.devices().iter()) {
            let (step, store) = VirStepFn::new(cfg.clone(), source, device.clone(), stage)?;
            ranks.push(VirRank { step, store });
        }
        // Store bersama = salinan store rank 0 (bobot awal identik semua rank).
        let shared = ranks
            .first()
            .map(|r| r.store.clone())
            .ok_or_else(|| VirError::Config("world 0".into()))?;
        Ok((
            Self {
                ranks,
                reducer,
                world,
            },
            shared,
        ))
    }

    /// Jumlah rank.
    pub fn world_size(&self) -> usize {
        self.world
    }
}

impl StepFn for DdpStepFn {
    fn compute(&mut self, ctx: StepCtx<'_>, store: &mut ParamStore) -> Result<StepOut, VirError> {
        if ctx.corruption.is_some() {
            return Err(VirError::Config(
                "DDP belum mendukung korupsi (set corruption=None)".into(),
            ));
        }
        let world = self.world;
        let mut losses = Vec::with_capacity(world);
        // 1. Tiap rank: sinkron bobot dari store bersama, lalu hitung grad lokal.
        for r in 0..world {
            sync_weights(store, &mut self.ranks[r].store)?;
            let mut rank_rng = ctx.rng.fork(r as u64);
            let rank_ctx = StepCtx {
                stage: ctx.stage,
                step: ctx.step,
                stage_step: ctx.stage_step,
                lr: ctx.lr,
                rng: &mut rank_rng,
                corruption: None,
            };
            let rank = &mut self.ranks[r];
            let out = rank.step.compute(rank_ctx, &mut rank.store)?;
            losses.push(out);
        }
        // 2. All-reduce gradien lintas rank → store bersama.
        all_reduce_into(store, &mut self.ranks, self.reducer.as_ref())?;
        // 3. Agregasi metrik (rata-rata).
        let n = losses.len() as f32;
        let agg = aggregate(&losses, n);
        Ok(agg)
    }
}

/// Salin bobot (slot `data`) dari store sumber ke store tujuan.
fn sync_weights(src: &ParamStore, dst: &mut ParamStore) -> Result<(), VirError> {
    if src.slots.len() != dst.slots.len() {
        return Err(VirError::Config(
            "jumlah slot store sumber/tujuan beda".into(),
        ));
    }
    for (s, d) in src.slots.iter().zip(dst.slots.iter_mut()) {
        if s.data.len() != d.data.len() {
            return Err(VirError::Config(format!(
                "panjang slot {} beda: {} vs {}",
                s.name,
                s.data.len(),
                d.data.len()
            )));
        }
        d.data.copy_from_slice(&s.data);
    }
    Ok(())
}

/// Kumpulkan grad lokal tiap rank, all-reduce, tulis ke store bersama.
fn all_reduce_into(
    store: &mut ParamStore,
    ranks: &mut [VirRank],
    reducer: &dyn GradAllReduce,
) -> Result<(), VirError> {
    let world = ranks.len();
    for slot_idx in 0..store.slots.len() {
        let mut per_rank: Vec<Vec<f32>> = Vec::with_capacity(world);
        for rank in ranks.iter() {
            per_rank.push(rank.store.slots[slot_idx].grad.clone());
        }
        reducer.all_reduce_mean(&mut per_rank)?;
        // Tulis grad rata-rata ke store bersama (slot yang sama).
        if let Some(mean) = per_rank.first() {
            store.slots[slot_idx].grad = mean.clone();
        }
    }
    Ok(())
}

/// Rata-rata metrik antar rank.
fn aggregate(losses: &[StepOut], n: f32) -> StepOut {
    let inv = if n > 0.0 { 1.0 / n } else { 0.0 };
    let mean = |f: fn(&StepOut) -> f32| losses.iter().map(f).sum::<f32>() * inv;
    let mean_vec = |f: fn(&StepOut) -> &Vec<f32>| -> Vec<f32> {
        let len = losses.first().map(|l| f(l).len()).unwrap_or(0);
        (0..len)
            .map(|i| losses.iter().map(|l| f(l)[i]).sum::<f32>() * inv)
            .collect()
    };
    StepOut {
        loss_total: mean(|l| l.loss_total),
        per_level: mean_vec(|l| &l.per_level),
        level_tokens: mean_vec(|l| &l.level_tokens),
        utilization: mean_vec(|l| &l.utilization),
        perplexity: mean_vec(|l| &l.perplexity),
        recon: losses.first().and_then(|l| l.recon),
        corruption_rate: None,
    }
}
