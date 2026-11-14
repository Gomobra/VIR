//! `vir mem` — operasi memori MFS (PRD F3.3).

use clap::Args;
use vir_mfs::MemoryStore;

use crate::CliError;

#[derive(Args)]
pub(crate) struct Args {
    /// Path basis data (`:memory:` untuk toko sementara)
    #[arg(long)]
    pub path: String,
    /// Tulis versi baru dengan isi ini
    #[arg(long)]
    pub write: Option<String>,
    /// Baca satu versi berdasarkan id
    #[arg(long)]
    pub read: Option<i64>,
    /// Cari dengan FTS + vektor
    #[arg(long)]
    pub find: Option<String>,
}

pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    let store = MemoryStore::open(&args.path).map_err(CliError::other)?;
    if let Some(content) = &args.write {
        let version_id = store.write("/cli", content).map_err(CliError::other)?;
        println!("wrote version {version_id}");
    }
    if let Some(version_id) = args.read {
        let content = store.read(version_id).map_err(CliError::other)?;
        println!("{content}");
    }
    if let Some(query) = &args.find {
        let hits = store.find(query, 5).map_err(CliError::other)?;
        for (id, content, score) in hits {
            println!("{id}: {content} (score {score:.4})");
        }
    }
    Ok(())
}