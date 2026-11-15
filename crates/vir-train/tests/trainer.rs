//! Tes nyata trainer dua tahap (PRD F1.7): resume identik bit demi bit setelah
//! simpan, dan gerbang rekonstruksi membekukan VQ lalu berpindah ke tahap 2.

use vir_common::VirError;
use vir_train::{
    LrSchedule, ParamRole, ParamSlot, ParamStore, ReconGate, Stage, StepCtx, StepOut, TrainConfig,
    Trainer,
};

fn temp_dir(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("vir_train_{tag}_{}_{nanos}", std::process::id()));
    path.to_str().expect("path utf8").to_string()
}

fn test_config() -> TrainConfig {
    TrainConfig {
        seed: 9,
        stage: Stage::Vq,
        steps: 6,
        levels: vec![4, 8],
        alpha: 0.0,
        lr: LrSchedule::cosine(0.05, 0.005, 2, 6),
        grad_clip: 1.0,
        log_every: 1,
        checkpoint_every: 0,
        gate: None,
        corruption: None,
    }
}

fn test_store() -> ParamStore {
    let mut store = ParamStore::new();
    store.add(ParamSlot::new(
        "core.q.weight",
        2,
        2,
        ParamRole::Vq,
        vec![0.0; 4],
    ));
    store.add(ParamSlot::vector(
        "norm.weight",
        ParamRole::Vq,
        vec![0.0, 0.0],
    ));
    store
}

/// Langkah kuadratik `½ Σ (w − t)²`; RNG ikut dipakai agar state-nya diuji.
fn quadratic_step(mut ctx: StepCtx<'_>, store: &mut ParamStore) -> Result<StepOut, VirError> {
    let target = [1.0f32, -1.0, 0.5, -0.5];
    let jitter = 1.0 + 0.05 * (ctx.rng.next_f32() - 0.5);
    let mut loss = 0.0f32;
    let mut grads: Vec<Vec<f32>> = Vec::with_capacity(store.slots.len());
    for slot in &store.slots {
        let mut g = Vec::with_capacity(slot.len());
        for (k, w) in slot.data.iter().enumerate() {
            let t = target[k % target.len()];
            loss += 0.5 * (w - t) * (w - t);
            g.push((w - t) * jitter);
        }
        grads.push(g);
    }
    for (i, g) in grads.iter().enumerate() {
        store.set_grad(i, g);
    }
    let corruption = ctx
        .corruption
        .as_mut()
        .map(|c| c.next_batch(0, 1, 4, 8).measured_rate());
    Ok(StepOut {
        loss_total: loss,
        per_level: vec![loss],
        level_tokens: vec![4.0],
        utilization: vec![0.5],
        perplexity: vec![3.0],
        recon: None,
        corruption_rate: corruption,
    })
}

fn perfect_recon(_ctx: StepCtx<'_>, _store: &mut ParamStore) -> Result<StepOut, VirError> {
    Ok(StepOut {
        recon: Some(1.0),
        ..StepOut::default()
    })
}

#[test]
fn resume_reproduces_subsequent_steps_bitwise() {
    let dir_a = temp_dir("resume_a");
    let dir_b = temp_dir("resume_b");
    let config = test_config();
    let mut step = quadratic_step;

    let mut a = Trainer::new(config.clone(), test_store(), &dir_a).expect("trainer a");
    a.run_steps(2, &mut step).expect("jalan a 1");
    let after_two = a.store.slots[0].data.clone();
    let rng_at_save = a.rng.state();
    a.save().expect("simpan");
    a.run_steps(4, &mut step).expect("jalan a 2");
    let full = a.store.slots[0].data.clone();
    drop(a);

    let mut b = Trainer::new(config, test_store(), &dir_b).expect("trainer b");
    b.run_steps(6, &mut step).expect("jalan b");
    assert_eq!(
        b.store.slots[0].data, full,
        "dua jalan tanpa resume berbeda"
    );
    drop(b);

    let mut c = Trainer::resume(&dir_a).expect("resume");
    assert_eq!(c.step, 2, "langkah hasil resume salah");
    assert_eq!(c.rng.state(), rng_at_save, "state RNG tidak pulih");
    assert_eq!(
        c.store.slots[0].data, after_two,
        "parameter tidak pulih identik"
    );
    c.run_steps(4, &mut step).expect("jalan c");
    assert_eq!(
        c.store.slots[0].data, full,
        "langkah setelah resume berbeda dari jalan penuh"
    );

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn gate_transition_freezes_vq_and_resumes_in_stage_two() {
    let dir = temp_dir("gate_stage");
    let mut config = test_config();
    config.steps = 1;
    config.gate = Some(ReconGate::accuracy(0.5, 1));

    let mut trainer = Trainer::new(config, test_store(), &dir).expect("trainer");
    let mut recon = perfect_recon;
    let out = trainer.run_steps(1, &mut recon).expect("run");
    assert!(out.gate_passed, "gerbang tidak tercatat lulus");
    assert!(out.transitioned, "transisi tahap tidak terjadi");
    assert_eq!(trainer.stage, Stage::Generator);
    assert_eq!(trainer.transitions, 1);
    assert!(trainer.gate.passed);
    assert_eq!(
        trainer.store.freeze_role(ParamRole::Vq),
        0,
        "VQ harus sudah dibekukan saat transisi"
    );
    assert_eq!(trainer.store.slots.iter().filter(|s| s.frozen).count(), 2);

    let resumed = Trainer::resume(&dir).expect("resume");
    assert_eq!(resumed.stage, Stage::Generator);
    assert!(
        resumed.store.slots.iter().all(|s| s.frozen),
        "status beku tidak pulih"
    );

    drop(trainer);
    let _ = std::fs::remove_dir_all(&dir);
}
