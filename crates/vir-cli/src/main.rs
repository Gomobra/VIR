//! Titik masuk `vir` (PRD F3.6).
//!
//! Setiap subperintah hanya memanggil pustaka; tidak ada logika bisnis di sini.

mod cmd_eval;
mod cmd_generate;
mod cmd_make_shard;
mod cmd_mem;
mod cmd_rlm;
mod cmd_train;
mod error;
mod smoke;

use clap::{Parser, Subcommand};
use vir_common::build_info;

use crate::error::CliError;

#[derive(Parser)]
#[command(name = "vir", version, about = "VIR Vbeta CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Irisan vertikal ujung ke ujung (G1)
    Smoke {
        /// Keluarkan status sebagai JSON
        #[arg(long)]
        json: bool,
    },
    /// Antarmuka terminal penuh
    Tui,
    /// Operasi memori MFS
    Mem(cmd_mem::Args),
    /// RLM dengan LM nyata + MFS nyata
    Rlm(cmd_rlm::Args),
    /// Evaluasi (metrik)
    Eval(cmd_eval::Args),
    /// Latihan model
    Train(cmd_train::Args),
    /// Generasi dari checkpoint
    Generate(cmd_generate::Args),
    /// Bikin shard token sintetis (vir-data)
    MakeShard(cmd_make_shard::Args),
    /// Info build
    Version,
}

fn main() -> Result<(), CliError> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Smoke { json } => smoke::run(json).map_err(|e| CliError::other(e.to_string())),
        Commands::Tui => vir_tui::run().map_err(|e| CliError::other(e.to_string())),
        Commands::Mem(args) => cmd_mem::run(&args),
        Commands::Rlm(args) => cmd_rlm::run(&args),
        Commands::Eval(args) => cmd_eval::run(&args),
        Commands::Train(args) => cmd_train::run(&args),
        Commands::Generate(args) => cmd_generate::run(&args),
        Commands::MakeShard(args) => cmd_make_shard::run(&args),
        Commands::Version => {
            println!("{}", build_info());
            Ok(())
        }
    }
}
