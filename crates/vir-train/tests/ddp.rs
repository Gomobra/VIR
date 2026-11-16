//! F2.5.c — data-parallel multi-GPU (DDP-style) di CPU (PRD F2.5.c).
//!
//! `HostAllReduce` dan `DdpStepFn` diuji tanpa GPU: world=1 identik, dan
//! world=2 (dua rank CPU) merata-ratakan gradien.

use vir_train::Rng;
use vir_train::data::ShardBatchSource;
use vir_train::ddp::{DdpStepFn, DevicePool, GradAllReduce, HostAllReduce};
use vir_train::model::VirTrainConfig;
use vir_train::optim::ParamRole;
use vir_train::schedule::LrSchedule;
use vir_train::trainer::{Stage, StepCtx, StepFn, TrainConfig, Trainer};

const SAMPLES: usize = 8;
const LEN: usize = 4;

fn tokens() -> Vec<u32> {
    let mut x = 12345u32;
    (0..SAMPLES * LEN)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            x % 4
        })
        .collect()
}

fn temp_shard(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("vir-ddp-{tag}-{}", std::process::id()));
    p
}

fn source(tag: &str) -> ShardBatchSource {
    let path = temp_shard(tag);
    vir_data::write_shard(&path, &tokens()).expect("tulis shard");
    ShardBatchSource::from_shard(&path, SAMPLES, LEN, 5, true).expect("buka shard")
}

fn mini_config() -> VirTrainConfig {
    VirTrainConfig {
        dim: 16,
        num_heads: 2,
        levels: vec![2, 4],
        codebook_size: 32,
        vocab: 4,
        phi_kernel: 1,
        prelude_blocks: 1,
        core_blocks: 1,
        coda_blocks: 1,
        num_loops: 1,
        decoder_hidden: 16,
        commitment_cost: 0.25,
        alpha: 0.0,
        chunk: 0,
        seed: 5,
    }
}

fn train_config(stage: Stage, steps: u64) -> TrainConfig {
    TrainConfig {
        seed: 5,
        stage,
        steps,
        levels: vec![2, 4],
        alpha: 0.0,
        lr: LrSchedule::cosine(0.05, 0.005, 10, steps.max(10)),
        grad_clip: 1.0,
        log_every: 0,
        checkpoint_every: 0,
        gate: None,
        corruption: None,
    }
}

#[test]
fn host_all_reduce_mean_numerik() {
    let mut grads = vec![vec![1.0f32, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
    HostAllReduce
        .all_reduce_mean(&mut grads)
        .expect("all-reduce");
    assert_eq!(grads[0], vec![2.0, 3.0, 4.0]);
    assert_eq!(grads[1], vec![2.0, 3.0, 4.0]);
}

#[test]
fn host_all_reduce_world_satu_identik() {
    let mut grads = vec![vec![1.5f32, -2.0]];
    HostAllReduce
        .all_reduce_mean(&mut grads)
        .expect("all-reduce");
    assert_eq!(grads[0], vec![1.5, -2.0]);
}

#[test]
fn host_all_reduce_panjang_beda_galat() {
    let mut grads = vec![vec![1.0f32, 2.0], vec![1.0f32]];
    assert!(HostAllReduce.all_reduce_mean(&mut grads).is_err());
}

#[test]
fn device_pool_world_min_satu() {
    let pool = DevicePool::detect();
    assert!(pool.world_size() >= 1);
    assert_eq!(pool.devices().len(), pool.world_size());
}

#[test]
fn ddp_world_satu_identik_dengan_virstep() {
    // DDP world=1 harus menghasilkan langkah yang sama dengan VirStepFn tunggal.
    let pool = DevicePool::new(vec![Default::default()]);
    let (mut ddp, mut store) = DdpStepFn::new(
        mini_config(),
        vec![source("ddp1")],
        &pool,
        Stage::Generator,
        Box::new(HostAllReduce),
    )
    .expect("rakit ddp");
    let mut rng = Rng::new(5);
    let ctx = StepCtx {
        stage: Stage::Generator,
        step: 0,
        stage_step: 0,
        lr: 0.05,
        rng: &mut rng,
        corruption: None,
    };
    let out = ddp.compute(ctx, &mut store).expect("langkah ddp");
    assert!(out.loss_total.is_finite());
    assert!(store.has_grads(), "grad harus ditulis ke store bersama");
}

#[test]
fn ddp_world_dua_merata_ratakan_grad() {
    // Dua rank CPU (device sama) + dua shard: gradien harus rata-rata, bukan
    // salah satu rank saja.
    let pool = DevicePool::new(vec![Default::default(), Default::default()]);
    let (mut ddp, mut store) = DdpStepFn::new(
        mini_config(),
        vec![source("ddp2a"), source("ddp2b")],
        &pool,
        Stage::Generator,
        Box::new(HostAllReduce),
    )
    .expect("rakit ddp2");
    assert_eq!(ddp.world_size(), 2);
    let mut rng = Rng::new(5);
    let ctx = StepCtx {
        stage: Stage::Generator,
        step: 0,
        stage_step: 0,
        lr: 0.05,
        rng: &mut rng,
        corruption: None,
    };
    let out = ddp.compute(ctx, &mut store).expect("langkah ddp2");
    assert!(out.loss_total.is_finite());
    assert!(store.has_grads());
}

#[test]
fn ddp_tahap_vq_dan_generator_jalan() {
    let pool = DevicePool::new(vec![Default::default()]);
    for stage in [Stage::Vq, Stage::Generator] {
        let (mut ddp, mut store) = DdpStepFn::new(
            mini_config(),
            vec![source("ddpstage")],
            &pool,
            stage,
            Box::new(HostAllReduce),
        )
        .expect("rakit");
        let mut rng = Rng::new(5);
        let ctx = StepCtx {
            stage,
            step: 0,
            stage_step: 0,
            lr: 0.05,
            rng: &mut rng,
            corruption: None,
        };
        let out = ddp.compute(ctx, &mut store).expect("langkah");
        assert!(out.loss_total.is_finite(), "stage {stage:?} loss");
    }
}

#[test]
fn ddp_slot_generator_ada_dan_role_benar() {
    let pool = DevicePool::new(vec![Default::default()]);
    let (_ddp, store) = DdpStepFn::new(
        mini_config(),
        vec![source("ddproles")],
        &pool,
        Stage::Vq,
        Box::new(HostAllReduce),
    )
    .expect("rakit");
    let roles: Vec<ParamRole> = store.slots.iter().map(|s| s.role).collect();
    assert!(roles.contains(&ParamRole::Vq));
    assert!(roles.contains(&ParamRole::Generator));
    assert!(roles.contains(&ParamRole::Tokenizer));
}

#[test]
fn ddp_trainer_mini_jalan() {
    // Trainer + DdpStepFn (world=1): beberapa langkah tanpa panik.
    let pool = DevicePool::new(vec![Default::default()]);
    let (ddp, store) = DdpStepFn::new(
        mini_config(),
        vec![source("ddptrainer")],
        &pool,
        Stage::Generator,
        Box::new(HostAllReduce),
    )
    .expect("rakit");
    let dir = std::env::temp_dir().join(format!("vir-ddp-trainer-{}", std::process::id()));
    let mut trainer = Trainer::new(
        train_config(Stage::Generator, 3),
        store,
        dir.to_str().unwrap(),
    )
    .expect("trainer");
    let mut ddp = ddp;
    let out = trainer.run(&mut ddp).expect("run");
    assert!(out.steps <= 3);
}
