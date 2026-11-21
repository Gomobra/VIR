//! `vir generate` — generasi NCM dari checkpoint (PRD F1.6).
//!
//! Non-autoregressive: satu lintasan menghasilkan seluruh blok (L token) secara
//! paralel — keunggulan arsitektur vs model AR (butuh L lintasan).

use std::path::Path;

use crate::error::CliError;

#[derive(clap::Args)]
pub(crate) struct Args {
    /// Direktori run (`native_state.json` + `model.safetensors`)
    #[arg(long)]
    pub ckpt: String,
    /// Suhu sampling
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    /// Jumlah sampel (batch)
    #[arg(long, default_value_t = 1)]
    pub batch: usize,
    /// Jumlah generasi warmup (tidak dihitung) sebelum pengukuran — memicu
    /// kompilasi kernel langkah-1 agar tok_s = steady-state (ADR-010).
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
    /// Cetak status sebagai JSON
    #[arg(long)]
    pub json: bool,
}

/// Jalankan generasi NCM.
pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    if !Path::new(&args.ckpt).exists() {
        return Err(CliError::other(format!(
            "checkpoint tidak ditemukan: {}",
            args.ckpt
        )));
    }
    if args.batch == 0 {
        return Err(CliError::other("batch harus > 0"));
    }
    if !(0.0..=2.0).contains(&args.temperature) {
        return Err(CliError::other("temperature di luar rentang [0, 2]"));
    }
    // 1. Muat checkpoint native (state + bobot).
    let state_path = format!("{}/{}", args.ckpt, vir_train::native::NATIVE_STATE_FILE);
    let state_text = std::fs::read_to_string(&state_path)
        .map_err(|e| CliError::other(format!("baca state {state_path}: {e}")))?;
    let state: vir_train::native::NativeCheckpoint = serde_json::from_str(&state_text)
        .map_err(|e| CliError::other(format!("state JSON rusak: {e}")))?;
    let model_path = format!("{}/{}", args.ckpt, vir_train::native::MODEL_FILE);
    let blob = std::fs::read(&model_path)
        .map_err(|e| CliError::other(format!("baca model {model_path}: {e}")))?;
    let tensors = vir_train::native::read_safetensors(&blob)
        .map_err(|e| CliError::other(format!("safetensors rusak: {e}")))?;

    // 2. Bangun model + muat bobot.
    let device = vir_train::model::train_device(Default::default(), state.model.seed);
    let model = vir_train::model::VirTrainModel::new(&state.model, &device)
        .map_err(|e| CliError::other(format!("bangun model: {e}")))?;
    let model = vir_train::params::import_safetensors(model, &tensors)
        .map_err(|e| CliError::other(format!("muat bobot: {e}")))?;

    // 3. Generasi NCM (satu lintasan paralel untuk seluruh blok).
    let sampler = vir_model::Sampler::new(args.temperature, 0, 42);
    // Warmup: jalankan `warmup` kali (memicu kompilasi kernel langkah-1) sebelum
    // mengukur, agar tok_s mencerminkan steady-state (ADR-010).
    for _ in 0..args.warmup {
        model
            .generator
            .generate(None, &sampler)
            .map_err(|e| CliError::other(format!("generate (warmup): {e}")))?;
    }
    let t0 = std::time::Instant::now();
    let out = model
        .generator
        .generate(None, &sampler)
        .map_err(|e| CliError::other(format!("generate: {e}")))?;
    let dt = t0.elapsed().as_secs_f64();
    let ntok = out.positions;
    let tok_s = if dt > 0.0 { ntok as f64 / dt } else { 0.0 };

    // 4. Laporan.
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "tokens": ntok,
                "dt_s": round3(dt),
                "tok_s": round1(tok_s),
            })
        );
    } else {
        println!("generate: {ntok} token, {dt:.2}s, {tok_s:.1} tok/s");
    }
    Ok(())
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}
