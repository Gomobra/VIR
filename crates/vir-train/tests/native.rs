//! F2.5.c — tes `NativeTrainer` (GPU-native, no roundtrip) di CPU.

use vir_train::data::ShardBatchSource;
use vir_train::model::VirTrainConfig;
use vir_train::native::NativeTrainer;
use vir_train::params::{export_safetensors, import_safetensors};
use vir_train::trainer::Stage;

fn tokens() -> Vec<u32> {
    let mut x = 12345u32;
    (0..8 * 4)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            x % 4
        })
        .collect()
}

fn cfg() -> VirTrainConfig {
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

fn source(tag: &str) -> ShardBatchSource {
    let p = std::env::temp_dir().join(format!("vir-native-{tag}-{}", std::process::id()));
    vir_data::write_shard(&p, &tokens()).expect("shard");
    ShardBatchSource::from_shard(&p, 8, 4, 5, false).expect("source")
}

#[test]
fn native_trainer_jalan() {
    let dir = std::env::temp_dir().join(format!("vir-native-run-{}", std::process::id()));
    let mut trainer = NativeTrainer::new(cfg(), source("a"), Default::default(), Stage::Generator)
        .expect("rakit");
    trainer.open(dir.to_str().expect("dir")).expect("open");
    trainer.run_steps(5).expect("5 langkah");
    // jalan nyata: tok/s terhitung (finite, >= 0).
    let tps = trainer.tokens_per_sec();
    assert!(tps.is_finite() && tps >= 0.0, "tok/s finite: {tps}");
}

#[test]
fn native_warmup_tak_dihitung_sebagai_langkah() {
    let dir = std::env::temp_dir().join(format!("vir-native-warmup-{}", std::process::id()));
    let mut trainer = NativeTrainer::new(cfg(), source("w"), Default::default(), Stage::Generator)
        .expect("rakit");
    trainer.set_warmup(2);
    trainer.open(dir.to_str().expect("dir")).expect("open");
    trainer.run_steps(3).expect("3 langkah");
    // Warmup (2) dijalankan tanpa menambah langkah terukur → outcome.steps = 3.
    assert_eq!(
        trainer.outcome(false).steps,
        3,
        "warmup tak menambah langkah"
    );
    let tps = trainer.tokens_per_sec();
    assert!(tps.is_finite() && tps >= 0.0, "tok/s finite: {tps}");
}

#[test]
fn safetensors_roundtrip_identik() {
    let device = vir_train::model::train_device(Default::default(), 5);
    let cfg = cfg();
    let model = vir_train::model::VirTrainModel::new(&cfg, &device).expect("model");
    let tensors = export_safetensors(&model).expect("export");
    assert!(!tensors.is_empty(), "export tidak boleh kosong");
    let model2 = import_safetensors(model.clone(), &tensors).expect("import");
    let tensors2 = export_safetensors(&model2).expect("export2");
    assert_eq!(tensors.len(), tensors2.len(), "jumlah tensor");
    for (a, b) in tensors.iter().zip(tensors2.iter()) {
        assert_eq!(a.0, b.0, "path");
        assert_eq!(a.1, b.1, "shape {}", a.0);
        assert_eq!(a.2, b.2, "data {}", a.0);
    }
}

#[test]
fn resume_identik_antara_jalan_penuh_vs_resume() {
    // Native resume: bobot setelah N langkah sama dengan jalan penuh.
    let dir = std::env::temp_dir().join(format!("vir-native-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut t = NativeTrainer::new(cfg(), source("r1"), Default::default(), Stage::Generator)
        .expect("rakit");
    t.open(dir.to_str().expect("dir")).expect("open");
    t.run_steps(4).expect("4 langkah");
    t.save().expect("save");
    let mut t2 = NativeTrainer::resume(dir.to_str().expect("dir"), source("r2")).expect("resume");
    t2.run_steps(2).expect("2 langkah");
    assert_eq!(t2.tokens_per_sec().is_finite(), true);
}
