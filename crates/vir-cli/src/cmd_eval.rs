//! `vir eval` — metrik evaluasi (PRD F4.2/F5.4).
//!
//! Cermin dari crate `vir-eval` (workspace terpisah yang tidak dapat menjadi
//! dependensi build inti): hanya metrik tanpa tensor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::Subcommand;

use crate::error::CliError;

#[derive(clap::Args)]
pub(crate) struct Args {
    #[command(subcommand)]
    pub mode: Mode,
}

#[derive(Subcommand)]
pub(crate) enum Mode {
    /// Estimator pass@k tak bias
    PassAtK {
        /// Jumlah sampel per tugas
        #[arg(long)]
        n: usize,
        /// Jumlah sampel benar
        #[arg(long)]
        c: usize,
        /// Ukuran k
        #[arg(long)]
        k: usize,
        /// Keluarkan sebagai JSON
        #[arg(long)]
        json: bool,
    },
    /// Entropi unigram (bit) dari berkas token
    Entropy {
        /// Berkas token (bilangan bulat dipisah spasi)
        #[arg(long)]
        tokens: PathBuf,
        /// Keluarkan sebagai JSON
        #[arg(long)]
        json: bool,
    },
    /// Akurasi rekonstruksi posisi
    Recon {
        /// Token asli
        #[arg(long)]
        original: PathBuf,
        /// Token hasil rekonstruksi
        #[arg(long)]
        reconstructed: PathBuf,
        /// Keluarkan sebagai JSON
        #[arg(long)]
        json: bool,
    },
}

pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    match &args.mode {
        Mode::PassAtK { n, c, k, json } => {
            let value = pass_at_k(*n, *c, *k);
            if *json {
                let payload = serde_json::json!({
                    "metric": "pass_at_k",
                    "n": n,
                    "c": c,
                    "k": k,
                    "value": value,
                });
                println!("{payload}");
            } else {
                println!("pass_at_k(n={n}, c={c}, k={k}) = {value:.6}");
            }
            Ok(())
        }
        Mode::Entropy { tokens, json } => {
            let data = read_tokens(tokens)?;
            let count = data.len();
            let value = unigram_entropy(&data);
            if *json {
                let payload = serde_json::json!({
                    "metric": "unigram_entropy_bits",
                    "tokens": count,
                    "value": value,
                });
                println!("{payload}");
            } else {
                println!("unigram_entropy(tokens={count}) = {value:.6}");
            }
            Ok(())
        }
        Mode::Recon {
            original,
            reconstructed,
            json,
        } => {
            let a = read_tokens(original)?;
            let b = read_tokens(reconstructed)?;
            let value = recon_accuracy(&a, &b);
            let count = a.len();
            if *json {
                let payload = serde_json::json!({
                    "metric": "recon_accuracy",
                    "original_tokens": count,
                    "value": value,
                });
                println!("{payload}");
            } else {
                println!("recon_accuracy(original_tokens={count}) = {value:.6}");
            }
            Ok(())
        }
    }
}

/// Estimator pass@k tak bias (Chen et al. 2021): `1 − C(n−c, k)/C(n, k)`.
fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let c = c.min(n);
    if n - c < k {
        return 1.0;
    }
    let mut prod = 1.0f64;
    for i in (n - c + 1)..=n {
        prod *= 1.0 - (k as f64) / (i as f64);
    }
    1.0 - prod
}

/// Akurasi rekonstruksi: posisi yang cocok terhadap panjang asli.
fn recon_accuracy(original: &[u32], reconstructed: &[u32]) -> f64 {
    if original.is_empty() {
        return 0.0;
    }
    let n = original.len().min(reconstructed.len());
    let ok = (0..n).filter(|&i| original[i] == reconstructed[i]).count();
    ok as f64 / original.len() as f64
}

/// Entropi unigram (bit) dari distribusi empiris token.
fn unigram_entropy(tokens: &[u32]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
    for token in tokens {
        *counts.entry(*token).or_insert(0usize) += 1;
    }
    let total = tokens.len() as f64;
    let mut bits = 0.0;
    for count in counts.values() {
        let p = *count as f64 / total;
        bits -= p * p.log2();
    }
    bits
}

/// Baca token (bilangan bulat dipisah spasi/baris).
fn read_tokens(path: &Path) -> Result<Vec<u32>, CliError> {
    let text = std::fs::read_to_string(path).map_err(|e| CliError::other(e.to_string()))?;
    let mut out = Vec::new();
    for token in text.split_whitespace() {
        let value: u32 = token
            .parse()
            .map_err(|_| CliError::other(format!("token bukan bilangan bulat: {token}")))?;
        out.push(value);
    }
    Ok(out)
}
