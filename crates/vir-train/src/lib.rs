//! Latihan dua tahap VIR (PRD F1.7): VQ-VAE (tahap 1) sampai gerbang
//! rekonstruksi lulus, lalu generator NCM dengan tokenizer beku (tahap 2).
//!
//! Tata letak:
//!
//! | Modul | Isi |
//! |---|---|
//! | [`loss`] | loss berbobot level (R6), CE dengan mask padding, pass@k, GRPO |
//! | [`corruption`] | korupsi kode masukan per level (NCM §3.4) + laju sebenarnya |
//! | [`optim`] | Muon (Newton–Schulz) + AdamW hibrid, clipping, state 8-bit |
//! | [`schedule`] | jadwal LR kosinus dan warmup–stabil–decay |
//! | [`trainer`] | trainer dua tahap + gerbang rekonstruksi + resume |
//! | [`checkpoint`] | checkpoint atomik (`state.json` + `tensors.bin`) |
//! | [`logging`] | log JSONL per langkah (loss total, per level, utilisasi) |
//! | [`metrics`] | utilisasi/perplexity codebook per level |
//! | [`rng`] | RNG xoshiro256++ yang bisa diserialisasi (resume identik) |
//! | [`burn_bridge`] | optimizer Burn: AdamW + grup Muon |
//!
//! Contoh alur (pemanggil menyediakan `StepFn` yang memegang model):
//!
//! ```text
//! let mut trainer = Trainer::new(config, store, "runs/exp1")?;
//! let out = trainer.run(&mut |ctx, store| { /* forward+backward, isi grad */ })?;
//! ```

pub mod burn_bridge;
pub mod checkpoint;
pub mod corruption;
pub mod logging;
pub mod loss;
pub mod metrics;
pub mod optim;
pub mod rng;
pub mod schedule;
pub mod trainer;

pub use checkpoint::{build as build_checkpoint, Checkpoint, CheckpointFile, Progress};
pub use corruption::{level_rates, Corruption, CorruptBatch, RateShape};
pub use logging::{EventRecord, JsonlLogger, StepRecord};
pub use loss::{
    code_weight, grpo_advantage, grpo_objective, level_loss_terms, level_weight,
    masked_level_loss, ncm_loss, nll_per_position, pass_at_k, sft_loss, LossBreakdown,
};
pub use metrics::{histogram, level_stats, perplexity, utilization, LevelStat};
pub use optim::{
    classify, clip_grad_norm, newton_schulz, newton_schulz_steps, AdamState, AdamW, Hybrid,
    MomState, Muon, MuonParams, OptimSettings, OptGroup, ParamRole, ParamSlot, ParamStore,
    MUON_SCALE, NS_COEFFS, NS_STEPS,
};
pub use rng::Rng;
pub use schedule::{LrKind, LrSchedule};
pub use trainer::{
    GateMetric, GateTracker, ReconGate, RunOutcome, Stage, StepCtx, StepFn, StepOut, TrainConfig,
    Trainer,
};

use vir_common::VirError;

/// Uji asap menyeluruh (dipakai `vir smoke`): RNG, loss, korupsi, Muon,
/// AdamW, jadwal, metrik, logger, dan checkpoint.
pub fn train_smoke() -> Result<(), VirError> {
    rng::rng_smoke()?;
    loss::loss_smoke()?;
    corruption::corruption_smoke()?;
    optim::muon_smoke()?;
    optim::adam_smoke()?;
    schedule::schedule_smoke()?;
    metrics::metrics_smoke()?;
    logging::logging_smoke()?;
    checkpoint::checkpoint_smoke()?;
    Ok(())
}

/// Uji asap mini: jalankan dua langkah nyata pada permukaan kuadratik,
/// checkpoint, resume, lalu selesaikan — hasil harus identik dengan satu jalan
/// tanpa henti (PRD F1.7: resume identik).
pub fn trainer_smoke() -> Result<(), VirError> {
    let dir_a = temp_dir("vir_train_trainer_smoke")?;
    let dir_b = temp_dir("vir_train_trainer_smoke_full")?;
    let dir_a = dir_a.as_str();
    let dir_b = dir_b.as_str();

    let config = TrainConfig {
        seed: 9,
        stage: Stage::Vq,
        steps: 6,
        log_every: 1,
        checkpoint_every: 2,
        gate: None,
        corruption: Some(Corruption::new(9, &[4, 8], 0.5, RateShape::LinearDecay)),
        ..TrainConfig::default()
    };

    let mut a = Trainer::new(config.clone(), smoke_store(), dir_a)?;
    let mut f = smoke_step;
    a.run_steps(2, &mut f)?;
    a.save()?;
    // Hentikan checkpoint otomatis agar titik resume tidak tertimpa.
    a.config.checkpoint_every = 0;
    let after_two = a.store.slots[0].data.clone();
    a.run_steps(4, &mut f)?;
    let full = a.store.slots[0].data.clone();

    let mut b = Trainer::new(config, smoke_store(), dir_b)?;
    b.run_steps(6, &mut f)?;
    if b.store.slots[0].data != full {
        return Err(VirError::Internal(
            "dua jalan tanpa resume berbeda (RNG tidak konsisten)".into(),
        ));
    }

    let mut c = Trainer::resume(dir_a)?;
    if c.step != 2 {
        return Err(VirError::Internal(format!("resume step {} != 2", c.step)));
    }
    if c.store.slots[0].data != after_two {
        return Err(VirError::Internal(
            "resume tidak memulihkan parameter secara identik".into(),
        ));
    }
    c.run_steps(4, &mut f)?;
    if c.store.slots[0].data != full {
        return Err(VirError::Internal(
            "resume menghasilkan langkah berikutnya yang berbeda".into(),
        ));
    }

    std::fs::remove_dir_all(dir_a)?;
    std::fs::remove_dir_all(dir_b)?;
    Ok(())
}

/// Direktori sementara bersih untuk smoke.
fn temp_dir(name: &str) -> Result<String, VirError> {
    let path = std::env::temp_dir().join(name);
    let dir = path
        .to_str()
        .ok_or_else(|| VirError::Internal("path smoke tidak valid".into()))?
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    Ok(dir)
}

/// Store mini: satu matriks dan satu bias (semua peran VQ).
fn smoke_store() -> ParamStore {
    let mut store = ParamStore::new();
    store.add(ParamSlot::new(
        "core.q.weight",
        2,
        2,
        ParamRole::Vq,
        vec![0.0; 4],
    ));
    store.add(ParamSlot::vector("norm.bias", ParamRole::Vq, vec![0.0, 0.0]));
    store
}

/// Langkah mini: `loss = ½ Σ (w − t)²`, gradien `w − t`.
fn smoke_step(mut ctx: StepCtx<'_>, store: &mut ParamStore) -> Result<StepOut, VirError> {
    let target = [1.0f32, -1.0, 0.5, -0.5];
    let mut loss = 0.0f32;
    let mut grads: Vec<Vec<f32>> = Vec::with_capacity(store.slots.len());
    for slot in store.slots.iter() {
        let mut g = Vec::with_capacity(slot.len());
        let mut acc = 0.0f32;
        for (k, w) in slot.data.iter().enumerate() {
            let t = target[k % target.len()];
            acc += 0.5 * (w - t) * (w - t);
            g.push(w - t);
        }
        loss += acc;
        grads.push(g);
    }
    for (i, g) in grads.iter().enumerate() {
        store.set_grad(i, g);
    }
    let corruption_rate = ctx
        .corruption
        .as_mut()
        .map(|c| c.next_batch(0, 1, 4, 8).measured_rate());
    Ok(StepOut {
        loss_total: loss,
        per_level: vec![loss],
        level_tokens: vec![4.0],
        utilization: vec![0.5],
        perplexity: vec![3.0],
        recon: Some(1.0 / (1.0 + loss)),
        corruption_rate,
    })
}
