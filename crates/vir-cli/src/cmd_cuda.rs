//! `vir cuda-train` — jembatan ke engine C++/CUDA (`vir-cuda-train`).
//!
//! Mengekspor token shard ke berkas datar (uint32) lalu memanggil binari C++/CUDA
//! yang melatih model dan menulis checkpoint. Binari C++ dibangun terpisah (nvcc)
//! dan dicari di `VIR_CUDA_BIN` atau `PATH`.
//!
//! Ini integrasi pertama (R7): plumbing nyata, bukan stub. Checkpoint format
//! sederhana (`model.bin` + `manifest.json`) — kompatibilitas penuh `vir generate`
//! (safetensors Burn) = langkah lanjutan.

use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::error::CliError;

/// Argumen `vir cuda-train`.
#[derive(clap::Args)]
pub(crate) struct Args {
    /// Shard token biner `vir-data`
    #[arg(long)]
    pub data: String,
    /// Direktori keluaran (checkpoint + log)
    #[arg(long)]
    pub out: String,
    /// Jumlah langkah latihan
    #[arg(long, default_value_t = 500)]
    pub steps: usize,
    /// Ukuran batch
    #[arg(long, default_value_t = 64)]
    pub batch: usize,
    /// Panjang urutan L
    #[arg(long, default_value_t = 32)]
    pub seq_len: usize,
    /// Jumlah blok transformer
    #[arg(long, default_value_t = 6)]
    pub n_blocks: usize,
    /// Lebar model
    #[arg(long, default_value_t = 64)]
    pub d_model: usize,
    /// Jumlah head attention
    #[arg(long, default_value_t = 4)]
    pub n_heads: usize,
    /// Ukuran vocab (bila tidak, diturunkan dari shard)
    #[arg(long)]
    pub vocab: Option<usize>,
    /// Laju belajar
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f32,
    /// Jumlah token yang diekspor (batas; default = steps*batch*seq_len)
    #[arg(long)]
    pub max_tokens: Option<usize>,
    /// Cetak status sebagai JSON
    #[arg(long)]
    pub json: bool,
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

/// Ekspor token shard ke berkas datar uint32 (little-endian).
fn export_tokens(data: &str, path: &str, max_tokens: usize) -> Result<usize, CliError> {
    let mut reader = vir_data::ShardReader::open(path)
        .map_err(|e| CliError::other(format!("baca shard: {e}")))?;
    let mut out = std::fs::File::create(path)
        .map_err(|e| CliError::other(format!("buat {path}: {e}")))?;
    let mut n = 0usize;
    reader
        .for_each_block(|block: &[u32]| -> Result<(), vir_common::VirError> {
            for t in block {
                if n >= max_tokens {
                    break;
                }
                out.write_all(&t.to_le_bytes())
                    .map_err(|e| vir_common::VirError::Internal(e.to_string()))?;
                n += 1;
            }
            Ok(())
        })
        .map_err(|e| CliError::other(format!("ekspor token: {e}")))?;
    Ok(n)
}

/// Jalankan `vir cuda-train`.
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
    let vocab = match args.vocab {
        Some(v) => v,
        None => vocab_from_shard(&args.data)?,
    };
    let max_tokens = args
        .max_tokens
        .unwrap_or_else(|| args.steps.saturating_mul(args.batch).saturating_mul(args.seq_len));
    let tokens_path = format!("{}/tokens.bin", args.out);
    let ntok = export_tokens(&args.data, &tokens_path, max_tokens)?;

    let bin = std::env::var("VIR_CUDA_BIN").unwrap_or_else(|_| "vir-cuda-train".to_string());
    let t0 = std::time::Instant::now();
    let status = Command::new(&bin)
        .arg("--tokens")
        .arg(&tokens_path)
        .arg("--out")
        .arg(&args.out)
        .arg("--vocab")
        .arg(vocab.to_string())
        .arg("--d-model")
        .arg(args.d_model.to_string())
        .arg("--n-heads")
        .arg(args.n_heads.to_string())
        .arg("--n-blocks")
        .arg(args.n_blocks.to_string())
        .arg("--seq-len")
        .arg(args.seq_len.to_string())
        .arg("--batch")
        .arg(args.batch.to_string())
        .arg("--steps")
        .arg(args.steps.to_string())
        .arg("--lr")
        .arg(args.lr.to_string())
        .status()
        .map_err(|e| CliError::other(format!("jalankan {bin}: {e} (apakah nvcc-built?)")))?;
    let dt = t0.elapsed().as_secs_f64();
    if !status.success() {
        return Err(CliError::other(format!(
            "{bin} keluar dengan kode {:?}",
            status.code()
        )));
    }
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "engine": "cuda",
                "tokens": ntok,
                "vocab_size": vocab,
                "steps": args.steps,
                "dt_s": (dt * 1000.0).round() / 1000.0,
            })
        );
    } else {
        println!(
            "cuda-train: {ntok} token, vocab={vocab}, steps={} (engine C++/CUDA, {:.2}s)",
            args.steps, dt
        );
    }
    Ok(())
}