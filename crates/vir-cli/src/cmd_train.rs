//! `vir train` — latihan dua tahap nyata (PRD F1.7, F2.5).
//!
//! Menjalankan `vir-train` sungguhan: BPE-shard → `VirStepFn` (atau `DdpStepFn`
//! bila `--world > 1`) → `Trainer` (Muon+AdamW, checkpoint atomik, resume).
//! Backend Burn ada di `vir-cli` (dependency `vir-model`), jadi latihan nyata
//! dijalankan di sini — bukan placeholder.

use std::path::Path;

use serde_json::Value as Json;
use vir_train::{
    DdpStepFn, DevicePool, HostAllReduce, LrSchedule, ShardBatchSource, Stage, TrainConfig,
    Trainer, VirStepFn, VirTrainConfig,
};

use crate::error::CliError;

/// Konfigurasi model dari berkas JSON (opsional); medan yang absen pakai default.
#[derive(Debug, Clone, Default)]
struct FileModel {
    dim: Option<usize>,
    num_heads: Option<usize>,
    levels: Option<Vec<usize>>,
    codebook_size: Option<usize>,
    phi_kernel: Option<usize>,
    prelude_blocks: Option<usize>,
    core_blocks: Option<usize>,
    coda_blocks: Option<usize>,
    num_loops: Option<usize>,
    decoder_hidden: Option<usize>,
    commitment_cost: Option<f32>,
    alpha: Option<f32>,
    chunk: Option<usize>,
    seed: Option<u64>,
}

impl FileModel {
    fn from_json(text: &str) -> Result<Self, CliError> {
        let v: Json = serde_json::from_str(text)
            .map_err(|e| CliError::other(format!("config JSON tidak valid: {e}")))?;
        let get = |k: &str| v.get(k).cloned();
        let num = |k: &str| get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let f = |k: &str| get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
        Ok(Self {
            dim: num("dim"),
            num_heads: num("num_heads"),
            levels: get("levels")
                .and_then(|x| {
                    x.as_array().map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_u64().map(|x| x as usize))
                            .collect::<Vec<usize>>()
                    })
                })
                .filter(|l| !l.is_empty()),
            codebook_size: num("codebook_size"),
            phi_kernel: num("phi_kernel"),
            prelude_blocks: num("prelude_blocks"),
            core_blocks: num("core_blocks"),
            coda_blocks: num("coda_blocks"),
            num_loops: num("num_loops"),
            decoder_hidden: num("decoder_hidden"),
            commitment_cost: f("commitment_cost"),
            alpha: f("alpha"),
            chunk: num("chunk"),
            seed: get("seed").and_then(|x| x.as_u64()),
        })
    }
}

/// Bangun `VirTrainConfig` dari default + override berkas + vocab dari data.
fn build_config(file: &FileModel, vocab: usize) -> Result<VirTrainConfig, CliError> {
    let mut cfg = VirTrainConfig::default();
    if let Some(v) = file.dim {
        cfg.dim = v;
    }
    if let Some(v) = file.num_heads {
        cfg.num_heads = v;
    }
    if let Some(v) = &file.levels {
        cfg.levels = v.clone();
    }
    if let Some(v) = file.codebook_size {
        cfg.codebook_size = v;
    }
    cfg.vocab = vocab.max(2);
    if let Some(v) = file.phi_kernel {
        cfg.phi_kernel = v;
    }
    if let Some(v) = file.prelude_blocks {
        cfg.prelude_blocks = v;
    }
    if let Some(v) = file.core_blocks {
        cfg.core_blocks = v;
    }
    if let Some(v) = file.coda_blocks {
        cfg.coda_blocks = v;
    }
    if let Some(v) = file.num_loops {
        cfg.num_loops = v;
    }
    if let Some(v) = file.decoder_hidden {
        cfg.decoder_hidden = v;
    }
    if let Some(v) = file.commitment_cost {
        cfg.commitment_cost = v;
    }
    if let Some(v) = file.alpha {
        cfg.alpha = v;
    }
    if let Some(v) = file.chunk {
        cfg.chunk = v;
    }
    if let Some(v) = file.seed {
        cfg.seed = v;
    }
    cfg.validate().map_err(|e| CliError::other(e.to_string()))?;
    Ok(cfg)
}

/// Turunkan vocab dari shard (token tertinggi + 1).
fn vocab_from_shard(path: &str) -> Result<usize, CliError> {
    let mut reader = vir_data::ShardReader::open(path)
        .map_err(|e| CliError::other(format!("baca shard: {e}")))?;
    let mut max = 0u32;
    reader
        .for_each_block(|block: &[u32]| -> Result<(), vir_common::VirError> {
            for t in block {
                max = max.max(*t);
            }
            Ok(())
        })
        .map_err(|e| CliError::other(format!("pindai shard: {e}")))?;
    Ok((max as usize) + 1)
}

/// Argumen `vir train`.
#[derive(clap::Args)]
pub(crate) struct Args {
    /// Shard token biner `vir-data`
    #[arg(long)]
    pub data: String,
    /// Direktori keluaran run (checkpoint + log JSONL)
    #[arg(long)]
    pub out: String,
    /// Jumlah langkah latihan
    #[arg(long, default_value_t = 1000)]
    pub steps: u64,
    /// Lanjutkan dari direktori run sebelumnya
    #[arg(long)]
    pub resume: Option<String>,
    /// Berkas config model (JSON) — medan yang absen pakai default mini
    #[arg(long)]
    pub config: Option<String>,
    /// Ukuran vocab (bila tidak, diturunkan dari shard)
    #[arg(long)]
    pub vocab: Option<usize>,
    /// Ukuran batch
    #[arg(long, default_value_t = 8)]
    pub batch: usize,
    /// Panjang urutan L (harus = level terakhir)
    #[arg(long, default_value_t = 32)]
    pub seq_len: usize,
    /// Jumlah GPU/rank data-parallel (default 1 = single device)
    #[arg(long, default_value_t = 1)]
    pub world: usize,
    /// Pakai trainer GPU-native (tanpa roundtrip host↔GPU; hanya world=1)
    #[arg(long)]
    pub native: bool,
    /// Langkah warmup tak terukur sebelum pengukuran (W0; hanya --native)
    #[arg(long, default_value_t = 0)]
    pub warmup: u64,
    /// Presisi latih native: fp32 | bf16 (tensor core; hanya --native)
    #[arg(long, default_value = "fp32")]
    pub dtype: String,
    /// Tahap awal: vq | gen
    #[arg(long, default_value = "gen")]
    pub stage: String,
    /// Seed (bila config JSON tak ada)
    #[arg(long)]
    pub seed: Option<u64>,
    /// Cetak status sebagai JSON
    #[arg(long)]
    pub json: bool,
}

/// Jalankan latihan.
pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    if !Path::new(&args.data).exists() {
        return Err(CliError::other(format!(
            "shard tidak ditemukan: {}",
            args.data
        )));
    }
    std::fs::create_dir_all(&args.out).map_err(|e| CliError::other(format!("buat out: {e}")))?;
    if args.steps == 0 {
        return Err(CliError::other("steps harus > 0"));
    }
    let world = args.world.max(1);
    let stage = parse_stage(&args.stage)?;
    let file = match &args.config {
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .map_err(|e| CliError::other(format!("baca config {}: {e}", p)))?;
            FileModel::from_json(&text)?
        }
        None => FileModel::default(),
    };
    let vocab = match args.vocab {
        Some(v) => v,
        None => vocab_from_shard(&args.data)?,
    };
    let cfg = build_config(&file, vocab)?;
    if cfg.total_len() != args.seq_len {
        return Err(CliError::other(format!(
            "--seq-len {} != L model {} (samakan dengan config levels terakhir)",
            args.seq_len,
            cfg.total_len()
        )));
    }
    let tcfg = train_config(&cfg, stage, args.steps, args.seed);
    let dir = args.resume.clone().unwrap_or_else(|| args.out.clone());
    let outcome = train(
        &cfg,
        &args.data,
        &dir,
        world,
        &tcfg,
        args.batch,
        args.native,
        args.warmup,
        &args.dtype,
    )?;
    report(&outcome, vocab, world, args.json);
    Ok(())
}

/// Tahap dari argumen.
fn parse_stage(s: &str) -> Result<Stage, CliError> {
    match s.to_ascii_lowercase().as_str() {
        "vq" => Ok(Stage::Vq),
        "gen" | "generator" => Ok(Stage::Generator),
        other => Err(CliError::other(format!("tahap tak dikenal: {other}"))),
    }
}

/// `TrainConfig` dari model cfg + argumen.
fn train_config(cfg: &VirTrainConfig, stage: Stage, steps: u64, seed: Option<u64>) -> TrainConfig {
    TrainConfig {
        seed: seed.unwrap_or(cfg.seed),
        stage,
        steps,
        levels: cfg.levels.clone(),
        alpha: cfg.alpha,
        // Warmup proporsional (<= 10% langkah, maks 500) agar tak melebihi total.
        lr: LrSchedule::cosine(3e-4, 3e-5, (steps / 10).clamp(1, 500), steps.max(1)),
        grad_clip: 1.0,
        log_every: 10,
        checkpoint_every: 200.max(steps / 4),
        gate: None,
        corruption: None,
    }
}

/// Bangun sumber data (N replika untuk DDP) dan jalankan trainer.
#[allow(clippy::too_many_arguments)]
fn train(
    cfg: &VirTrainConfig,
    data: &str,
    dir: &str,
    world: usize,
    tcfg: &TrainConfig,
    batch: usize,
    native: bool,
    warmup: u64,
    dtype: &str,
) -> Result<vir_train::RunOutcome, CliError> {
    if native {
        return train_native(cfg, data, dir, world, tcfg, batch, warmup, dtype);
    }
    let stage = tcfg.stage;
    if world <= 1 {
        let source = ShardBatchSource::from_shard(data, batch, cfg.total_len(), cfg.seed, false)
            .map_err(|e| CliError::other(e.to_string()))?;
        let (mut step, store) = VirStepFn::new(cfg.clone(), source, Default::default(), stage)
            .map_err(|e| CliError::other(e.to_string()))?;
        let mut trainer =
            Trainer::new(tcfg.clone(), store, dir).map_err(|e| CliError::other(e.to_string()))?;
        trainer
            .run(&mut step)
            .map_err(|e| CliError::other(e.to_string()))
    } else {
        let pool = DevicePool::detect();
        let mut sources = Vec::with_capacity(world);
        for r in 0..world {
            sources.push(
                ShardBatchSource::from_shard(
                    data,
                    batch,
                    cfg.total_len(),
                    cfg.seed + r as u64,
                    r % 2 == 0,
                )
                .map_err(|e| CliError::other(e.to_string()))?,
            );
        }
        let (mut step, store) =
            DdpStepFn::new(cfg.clone(), sources, &pool, stage, Box::new(HostAllReduce))
                .map_err(|e| CliError::other(e.to_string()))?;
        let mut trainer =
            Trainer::new(tcfg.clone(), store, dir).map_err(|e| CliError::other(e.to_string()))?;
        trainer
            .run(&mut step)
            .map_err(|e| CliError::other(e.to_string()))
    }
}

/// Jalur GPU-native (tanpa roundtrip host↔GPU; hanya single device).
#[allow(clippy::too_many_arguments)]
fn train_native(
    cfg: &VirTrainConfig,
    data: &str,
    dir: &str,
    world: usize,
    tcfg: &TrainConfig,
    batch: usize,
    warmup: u64,
    dtype: &str,
) -> Result<vir_train::RunOutcome, CliError> {
    if world > 1 {
        return Err(CliError::other(
            "--native belum mendukung --world > 1 (DDP native belum ada)",
        ));
    }
    let bf16 = matches!(dtype.to_ascii_lowercase().as_str(), "bf16" | "bfloat16");
    let source = ShardBatchSource::from_shard(data, batch, cfg.total_len(), cfg.seed, false)
        .map_err(|e| CliError::other(e.to_string()))?;
    let mut trainer = vir_train::NativeTrainer::with_train_config_dtype(
        cfg.clone(),
        source,
        Default::default(),
        tcfg.clone(),
        bf16,
    )
    .map_err(|e| CliError::other(e.to_string()))?;
    trainer.set_warmup(warmup);
    trainer
        .open(dir)
        .map_err(|e| CliError::other(e.to_string()))?;
    trainer.run().map_err(|e| CliError::other(e.to_string()))?;
    Ok(trainer.outcome(trainer.gate_passed()))
}

/// Ringkasan hasil (JSON atau teks).
fn report(outcome: &vir_train::RunOutcome, vocab: usize, world: usize, json: bool) {
    if json {
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "steps": outcome.steps,
                "vocab_size": vocab,
                "world": world,
                "tok_s": outcome.tok_s,
                "fwd_ms": outcome.fwd_ms,
                "bwd_ms": outcome.bwd_ms,
                "opt_ms": outcome.opt_ms,
            })
        );
        return;
    }
    println!("train: steps={} vocab={vocab} world={world}", outcome.steps);
}
