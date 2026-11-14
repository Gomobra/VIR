//! `vir train` — latihan dua tahap (PRD F1.7).
//!
//! Membutuhkan backend `train` (Burn). Build inti tidak memuatnya, jadi argumen
//! divalidasi lebih dulu lalu perintah gagal dengan galat yang jelas.

use std::path::Path;

use clap::Args;

use crate::backend::{BackendUnavailable, TRAIN_BACKEND};
use crate::CliError;

#[derive(Args)]
pub(crate) struct Args {
    /// Korpus token (shard biner vir-data)
    #[arg(long)]
    pub data: String,
    /// Direktori keluaran run (checkpoint + log JSONL)
    #[arg(long)]
    pub out: String,
    /// Jumlah langkah latihan
    #[arg(long, default_value_t = 1000)]
    pub steps: u64,
    /// Direktori run untuk resume
    #[arg(long)]
    pub resume: Option<String>,
    /// Konfigurasi model (JSON dari `vir.model`)
    #[arg(long)]
    pub config: Option<String>,
    /// Keluarkan status sebagai JSON
    #[arg(long)]
    pub json: bool,
}

pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    if !Path::new(&args.data).exists() {
        let msg = format!("korpus tidak ditemukan: {}", args.data);
        return Err(CliError::other(msg));
    }
    if !Path::new(&args.out).exists() {
        let msg = format!("direktori keluaran tidak ditemukan: {}", args.out);
        return Err(CliError::other(msg));
    }
    if args.steps == 0 {
        return Err(CliError::other("steps harus > 0"));
    }
    if let Some(dir) = &args.resume {
        if !Path::new(dir).exists() {
            let msg = format!("run resume tidak ditemukan: {dir}");
            return Err(CliError::other(msg));
        }
    }
    let missing = BackendUnavailable::new("train", TRAIN_BACKEND);
    if args.json {
        let payload = missing.json();
        println!("{payload}");
    }
    Err(CliError::Backend(missing))
}