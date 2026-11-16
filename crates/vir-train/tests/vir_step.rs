//! Tes nyata `VirStepFn` + `Trainer` (PRD F2.5.b): model VIR mini benar-benar
//! belajar di CPU, gradien tersalin ke `ParamStore`, dan checkpoint→resume
//! identik.

use vir_train::Rng;
use vir_train::burn_step::VirStepFn;
use vir_train::data::ShardBatchSource;
use vir_train::model::VirTrainConfig;
use vir_train::optim::ParamRole;
use vir_train::schedule::LrSchedule;
use vir_train::trainer::{Stage, StepCtx, StepFn, TrainConfig, Trainer};

const SAMPLES: usize = 8;
const LEN: usize = 4;

/// Token sintetis deterministik (nilai < 4, bervariasi antar posisi).
fn tokens() -> Vec<u32> {
    let mut x = 12345u32;
    (0..SAMPLES * LEN)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 16) % 4
        })
        .collect()
}

fn temp_dir(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "vir_train_step_{tag}_{}_{nanos}",
        std::process::id()
    ));
    path.to_str().expect("path utf8").to_string()
}

fn temp_shard(tag: &str) -> String {
    format!("{}.shard", temp_dir(tag))
}

/// Model mini: `d=8`, 2 level `[2, 4]`, codebook 8, vocab 4.
fn mini_config() -> VirTrainConfig {
    VirTrainConfig {
        dim: 8,
        num_heads: 2,
        levels: vec![2, 4],
        codebook_size: 8,
        vocab: 4,
        phi_kernel: 1,
        prelude_blocks: 1,
        core_blocks: 1,
        coda_blocks: 1,
        num_loops: 1,
        decoder_hidden: 0,
        commitment_cost: 0.25,
        alpha: 0.0,
        chunk: 0,
        seed: 5,
    }
}

fn source(tag: &str) -> ShardBatchSource {
    let path = temp_shard(tag);
    vir_data::write_shard(&path, &tokens()).expect("tulis shard");
    ShardBatchSource::from_shard(&path, SAMPLES, LEN, 5, true).expect("buka shard")
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
fn store_parameter_cocok_dengan_peran_tahap() {
    let (step, store) = VirStepFn::new(
        mini_config(),
        source("roles"),
        Default::default(),
        Stage::Vq,
    )
    .expect("rakit");
    assert!(store.slots.len() > 10, "slot terlalu sedikit");
    let roles: Vec<ParamRole> = store.slots.iter().map(|s| s.role).collect();
    assert!(roles.contains(&ParamRole::Vq), "slot Vq hilang");
    assert!(
        roles.contains(&ParamRole::Generator),
        "slot generator hilang"
    );
    assert!(
        roles.contains(&ParamRole::Tokenizer),
        "slot tokenizer hilang"
    );
    // Tahap Vq: generator beku, Vq aktif, tokenizer beku.
    for slot in &store.slots {
        let expect = match slot.role {
            ParamRole::Vq => false,
            ParamRole::Generator | ParamRole::Tokenizer => true,
        };
        assert_eq!(slot.frozen, expect, "slot {} beku salah", slot.name);
    }
    assert_eq!(step.grad_slots(), 0);
}

#[test]
fn init_bobot_identik_walau_konstruksi_paralel() {
    // Regresi akar: RNG device global + init lazy membuat bobot awal bergantung
    // pada urutan konstruksi. Empat `VirStepFn` dibangun serentak; toko awal
    // semuanya harus sama (kalau tidak, resume tak mungkin identik).
    let handles: Vec<_> = ["init_p0", "init_p1", "init_p2", "init_p3"]
        .into_iter()
        .map(|tag| {
            std::thread::spawn(move || {
                let (_, store) =
                    VirStepFn::new(mini_config(), source(tag), Default::default(), Stage::Vq)
                        .expect("rakit");
                store
                    .slots
                    .iter()
                    .map(|s| s.data.clone())
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let mut handles = handles.into_iter();
    let base = handles.next().expect("satu handle").join().expect("join 0");
    for (i, handle) in handles.enumerate() {
        let other = handle.join().expect("join");
        assert_eq!(
            other,
            base,
            "toko awal konstruksi paralel ke-{} beda",
            i + 1
        );
    }
}

#[test]
fn gradien_tersalin_ke_slot_store() {
    let (mut step, mut store) = VirStepFn::new(
        mini_config(),
        source("grads"),
        Default::default(),
        Stage::Generator,
    )
    .expect("rakit");
    let mut rng = Rng::new(5);
    let ctx = StepCtx {
        stage: Stage::Generator,
        step: 0,
        stage_step: 0,
        lr: 0.05,
        rng: &mut rng,
        corruption: None,
    };
    let out = step.compute(ctx, &mut store).expect("langkah");
    assert!(out.loss_total.is_finite(), "loss tidak hingga");
    assert!(out.loss_total > 0.0, "loss harus positif");

    // Gradien setiap slot non-beku harus non-nol dan panjangnya cocok.
    let mut nonfrozen = 0;
    let mut nonzero = 0;
    for slot in &store.slots {
        if slot.frozen {
            assert!(
                slot.grad.iter().all(|g| *g == 0.0),
                "slot beku {} punya gradien",
                slot.name
            );
            continue;
        }
        nonfrozen += 1;
        assert_eq!(
            slot.grad.len(),
            slot.data.len(),
            "panjang grad {}",
            slot.name
        );
        if slot.grad.iter().any(|g| *g != 0.0) {
            nonzero += 1;
        }
    }
    assert_eq!(step.grad_slots(), nonfrozen);
    assert!(nonzero > 0, "tidak ada gradien non-nol");
    // Codebook (Tokenizer) ikut disalin model → store walau beku.
    assert!(step.copied_slots() > 0, "codebook tidak disalin ke store");
}

#[test]
fn tahap_vq_metrik_masuk_akal() {
    let (mut step, mut store) =
        VirStepFn::new(mini_config(), source("vq"), Default::default(), Stage::Vq).expect("rakit");
    for i in 0..3u64 {
        let mut rng = Rng::new(5 + i);
        let ctx = StepCtx {
            stage: Stage::Vq,
            step: i,
            stage_step: i,
            lr: 0.01,
            rng: &mut rng,
            corruption: None,
        };
        let out = step.compute(ctx, &mut store).expect("langkah vq");
        assert!(out.loss_total.is_finite(), "loss Vq tidak hingga");
        assert_eq!(out.per_level.len(), 2, "per level");
        assert_eq!(out.level_tokens, vec![16.0, 32.0]);
        assert_eq!(out.utilization.len(), 2);
        assert!((0.0..=1.0).contains(&out.utilization[0]));
        assert!(out.perplexity.iter().all(|p| p.is_finite()));
        let acc = out.recon.expect("akurasi rekon");
        assert!((0.0..=1.0).contains(&acc), "akurasi {acc}");
    }
}

#[test]
fn tahap_vq_menurunkan_loss() {
    let mut trainer = Trainer::new(
        train_config(Stage::Vq, 80),
        {
            let (_, store) = VirStepFn::new(
                mini_config(),
                source("vq_loss"),
                Default::default(),
                Stage::Vq,
            )
            .expect("rakit");
            store
        },
        &temp_dir("vq_loss_run"),
    )
    .expect("trainer");
    let (mut step, _) = VirStepFn::new(
        mini_config(),
        source("vq_loss"),
        Default::default(),
        Stage::Vq,
    )
    .expect("rakit");
    trainer.run_steps(1, &mut step).expect("langkah");
    let before = trainer.last_loss;
    trainer.run_steps(79, &mut step).expect("langkah");
    let after = trainer.last_loss;
    assert!(after < before, "loss Vq tidak turun: {before} → {after}");
}

#[test]
fn tahap_generator_menurunkan_loss() {
    let dir = temp_dir("gen_loss");
    let (mut step, store) = VirStepFn::new(
        mini_config(),
        source("gen_loss"),
        Default::default(),
        Stage::Generator,
    )
    .expect("rakit");
    let mut trainer =
        Trainer::new(train_config(Stage::Generator, 60), store, &dir).expect("trainer");
    trainer.run_steps(1, &mut step).expect("langkah");
    let before = trainer.last_loss;
    trainer.run_steps(59, &mut step).expect("langkah");
    let after = trainer.last_loss;
    assert!(
        after < 0.9 * before,
        "loss generator tidak turun cukup: {before} → {after}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resume_generator_identik() {
    let dir_a = temp_dir("resume_a");
    let dir_b = temp_dir("resume_b");
    let cfg = train_config(Stage::Generator, 4);

    // Jalan penuh tanpa henti.
    let (mut full_step, full_store) = VirStepFn::new(
        mini_config(),
        source("resume_full"),
        Default::default(),
        Stage::Generator,
    )
    .expect("rakit");
    let mut full = Trainer::new(cfg.clone(), full_store, &dir_b).expect("trainer");
    full.run_steps(4, &mut full_step).expect("langkah");
    let full_slots: Vec<Vec<f32>> = full.store.slots.iter().map(|s| s.data.clone()).collect();

    // Jalan terputus: 2 langkah → checkpoint → resume dengan langkah BARU.
    let (mut part_step, part_store) = VirStepFn::new(
        mini_config(),
        source("resume_part"),
        Default::default(),
        Stage::Generator,
    )
    .expect("rakit");
    let mut part = Trainer::new(cfg, part_store, &dir_a).expect("trainer");
    part.run_steps(2, &mut part_step).expect("langkah");
    part.save().expect("simpan");

    let mut resumed = Trainer::resume(&dir_a).expect("resume");
    assert_eq!(resumed.step, 2);
    // Langkah baru: model di-nilai ulang dari store, sumber di-seek ke langkah 2.
    let (mut fresh_step, _) = VirStepFn::new(
        mini_config(),
        source("resume_part"),
        Default::default(),
        Stage::Generator,
    )
    .expect("rakit");
    resumed.run_steps(2, &mut fresh_step).expect("langkah");

    let part_slots: Vec<Vec<f32>> = resumed.store.slots.iter().map(|s| s.data.clone()).collect();
    assert_eq!(part_slots.len(), full_slots.len());
    for (i, (a, b)) in part_slots.iter().zip(full_slots.iter()).enumerate() {
        assert_eq!(a, b, "slot {i} beda setelah resume");
    }
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}
