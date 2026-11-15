//! `vir generate` — generasi dari checkpoint (PRD F1.6).
//!
//! Membutuhkan backend `model` (Burn). Build inti tidak memuatnya, jadi argumen
//! divalidasi lebih dulu lalu perintah gagal dengan galat yang jelas.

use std::path::Path;

use crate::backend::{BackendUnavailable, MODEL_BACKEND};
use crate::error::CliError;

#[derive(clap::Args)]
pub(crate) struct Args {
    /// Checkpoint model (`state.json` + `tensors.bin`)
    #[arg(long)]
    pub ckpt: String,
    /// Prompt awal
    #[arg(long, default_value = "")]
    pub prompt: String,
    /// Panjang maksimum keluaran (token)
    #[arg(long, default_value_t = 128)]
    pub max_tokens: usize,
    /// Suhu sampling
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    /// Keluarkan status sebagai JSON
    #[arg(long)]
    pub json: bool,
}

pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    if !Path::new(&args.ckpt).exists() {
        let msg = format!("checkpoint tidak ditemukan: {}", args.ckpt);
        return Err(CliError::other(msg));
    }
    if args.max_tokens == 0 {
        return Err(CliError::other("max-tokens harus > 0"));
    }
    if !(0.0..=2.0).contains(&args.temperature) {
        let msg = format!("temperature di luar rentang: {}", args.temperature);
        return Err(CliError::other(msg));
    }
    let missing = BackendUnavailable::new("generate", MODEL_BACKEND);
    if args.json {
        let payload = missing.json();
        println!("{payload}");
    }
    Err(CliError::Backend(missing))
}
