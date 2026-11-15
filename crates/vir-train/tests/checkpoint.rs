//! Tes nyata checkpoint atomik + resume identik (PRD F1.7 langkah 7): payload
//! parameter dan state optimizer harus pulih bit demi bit, penulisan harus
//! atomik (tanpa `.tmp`/`.old` yang tertinggal), dan payload rusak ditolak.

use vir_train::{
    Checkpoint, CheckpointFile, Corruption, GateMetric, GateTracker, Hybrid, LrSchedule,
    OptimSettings, ParamRole, ParamSlot, ParamStore, Progress, RateShape, Rng, TrainConfig,
    build_checkpoint,
};

fn temp_dir(tag: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("vir_train_{tag}_{}_{nanos}", std::process::id()));
    path.to_str().expect("path utf8").to_string()
}

fn sample_store() -> ParamStore {
    let mut store = ParamStore::new();
    store.add(ParamSlot::new(
        "core.q.weight",
        2,
        2,
        ParamRole::Vq,
        vec![0.5, -0.25, 1.0, 0.0],
    ));
    store.add(ParamSlot::vector(
        "norm.weight",
        ParamRole::Vq,
        vec![1.0, 2.0],
    ));
    store
}

fn sample_config() -> TrainConfig {
    TrainConfig {
        seed: 11,
        steps: 7,
        levels: vec![4, 8],
        lr: LrSchedule::cosine(0.1, 0.01, 1, 7),
        ..TrainConfig::default()
    }
}

fn sample_progress() -> Progress {
    Progress {
        stage: 1,
        step: 7,
        stage_step: 7,
        transitions: 0,
    }
}

/// Store + optimizer yang sudah satu langkah (state momentum/Adam non-nol).
fn stepped_state() -> (ParamStore, Hybrid) {
    let mut store = sample_store();
    store.set_grad(0, &[0.1; 4]);
    store.set_grad(1, &[0.2, -0.2]);
    let mut opt = Hybrid::new(&store, &OptimSettings::default());
    opt.step_single(&mut store, 0.01);
    (store, opt)
}

#[test]
fn save_is_atomic_and_restores_bits() {
    let dir = temp_dir("ckpt_atomic");
    let (store, opt) = stepped_state();
    let gate = GateTracker::new(GateMetric::TokenAccuracy);
    let corruption = Corruption::new(3, &[4, 8], 0.5, RateShape::LinearDecay);
    let file = build_checkpoint(
        &sample_config(),
        &sample_progress(),
        &Rng::new(5),
        &gate,
        Some(&corruption),
        &store,
        &opt,
    );
    file.save_atomic(&dir).expect("save");

    assert!(std::path::Path::new(&format!("{dir}/state.json")).exists());
    assert!(std::path::Path::new(&format!("{dir}/tensors.bin")).exists());
    assert!(
        !std::path::Path::new(&format!("{dir}.tmp")).exists(),
        "direktori sementara tertinggal"
    );
    assert!(
        !std::path::Path::new(&format!("{dir}.old")).exists(),
        "direktori lama tertinggal"
    );

    let loaded = CheckpointFile::load(&dir).expect("load");
    assert_eq!(
        loaded.checkpoint.version,
        vir_train::checkpoint::CHECKPOINT_VERSION
    );
    assert_eq!(loaded.checkpoint.step, 7);
    assert_eq!(loaded.checkpoint.params.len(), store.slots.len());
    assert_eq!(loaded.checkpoint.rng.state(), Rng::new(5).state());
    let rates = loaded
        .checkpoint
        .corruption
        .as_ref()
        .expect("korupsi ikut checkpoint")
        .rates()
        .to_vec();
    assert_eq!(rates, corruption.rates());

    let (store2, opt2) = loaded.restore().expect("restore");
    for (a, b) in store.slots.iter().zip(store2.slots.iter()) {
        assert_eq!(
            a.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "parameter slot {} tidak pulih bit demi bit",
            a.name
        );
        assert_eq!(
            (a.rows, a.cols, a.group, a.frozen),
            (b.rows, b.cols, b.group, b.frozen)
        );
    }
    for i in 0..opt.muon.states.len() {
        for k in 0..store.slots[i].len() {
            assert_eq!(
                opt.muon.states[i].value(k).to_bits(),
                opt2.muon.states[i].value(k).to_bits(),
                "momentum Muon slot {i} elemen {k}"
            );
        }
        assert_eq!(opt.adamw.states[i].t, opt2.adamw.states[i].t);
        assert_eq!(
            opt.adamw.states[i]
                .m
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            opt2.adamw.states[i]
                .m
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            opt.adamw.states[i]
                .v
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            opt2.adamw.states[i]
                .v
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn second_save_overwrites_without_leftovers() {
    let dir = temp_dir("ckpt_overwrite");
    let (mut store, opt) = stepped_state();
    let gate = GateTracker::new(GateMetric::ReconLoss);
    let config = sample_config();
    let progress = sample_progress();

    build_checkpoint(&config, &progress, &Rng::new(5), &gate, None, &store, &opt)
        .save_atomic(&dir)
        .expect("save pertama");

    for v in store.slots[0].data.iter_mut() {
        *v += 1.0;
    }
    build_checkpoint(&config, &progress, &Rng::new(6), &gate, None, &store, &opt)
        .save_atomic(&dir)
        .expect("save kedua");

    let loaded = CheckpointFile::load(&dir).expect("load");
    let (store2, _) = loaded.restore().expect("restore");
    assert_eq!(store.slots[0].data, store2.slots[0].data);
    assert!(!std::path::Path::new(&format!("{dir}.tmp")).exists());
    assert!(!std::path::Path::new(&format!("{dir}.old")).exists());
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn truncated_blob_is_rejected() {
    let dir = temp_dir("ckpt_truncated");
    let (store, opt) = stepped_state();
    let gate = GateTracker::new(GateMetric::TokenAccuracy);
    build_checkpoint(
        &sample_config(),
        &sample_progress(),
        &Rng::new(5),
        &gate,
        None,
        &store,
        &opt,
    )
    .save_atomic(&dir)
    .expect("save");

    let path = format!("{dir}/tensors.bin");
    let blob = std::fs::read(&path).expect("read blob");
    std::fs::write(&path, &blob[..blob.len() - 1]).expect("truncate");
    assert!(
        CheckpointFile::load(&dir).is_err(),
        "payload terpotong harus ditolak"
    );
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn unknown_checkpoint_version_is_rejected() {
    let dir = temp_dir("ckpt_version");
    let (store, opt) = stepped_state();
    let gate = GateTracker::new(GateMetric::TokenAccuracy);
    build_checkpoint(
        &sample_config(),
        &sample_progress(),
        &Rng::new(5),
        &gate,
        None,
        &store,
        &opt,
    )
    .save_atomic(&dir)
    .expect("save");

    let meta = std::fs::read_to_string(format!("{dir}/state.json")).expect("read meta");
    let bad = meta.replacen(vir_train::checkpoint::CHECKPOINT_VERSION, "vir-train-0", 1);
    std::fs::write(format!("{dir}/state.json"), bad).expect("write meta");
    assert!(
        CheckpointFile::load(&dir).is_err(),
        "versi tak dikenal harus ditolak"
    );
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn checkpoint_json_roundtrip_keeps_metadata() {
    let (store, opt) = stepped_state();
    let gate = GateTracker::new(GateMetric::TokenAccuracy);
    let file = build_checkpoint(
        &sample_config(),
        &sample_progress(),
        &Rng::new(5),
        &gate,
        None,
        &store,
        &opt,
    );
    let json = file.checkpoint.to_json().expect("to_json");
    let back = Checkpoint::from_json(&json).expect("from_json");
    assert_eq!(back.step, file.checkpoint.step);
    assert_eq!(back.blob_len, file.checkpoint.blob_len);
    assert_eq!(back.params.len(), file.checkpoint.params.len());
    assert_eq!(back.rng.state(), file.checkpoint.rng.state());
}
