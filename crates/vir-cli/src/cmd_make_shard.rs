//! `vir make-shard` — bikin shard token sintetis dari korpus (PRD F1.3, F2.5).
//!
//! Melatih BPE mini pada korpus sintetis (`vir-data`), men-tokenisasi, lalu
//! menulis shard biner `vir-data`. Dipakai notebook Kaggle untuk latihan nyata
//! (F2.5.e) tanpa perlu Python.

use std::path::Path;

use vir_text::BpeTokenizer;

use crate::error::CliError;

#[derive(clap::Args)]
pub(crate) struct Args {
    /// Berkas keluaran shard (mis. data/mini_shard.bin)
    #[arg(long)]
    pub out: String,
    /// Jumlah dokumen sintetis
    #[arg(long, default_value_t = 2000)]
    pub docs: usize,
    /// Ukuran vocab BPE
    #[arg(long, default_value_t = 512)]
    pub vocab: usize,
    /// Seed (bila diisi; determinisme tokenizer)
    #[arg(long)]
    pub seed: Option<u64>,
}

pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    let corpus = vir_data::synth_corpus(args.docs.max(8));
    let mut tok = BpeTokenizer::new();
    tok.train(&corpus, args.vocab.max(64));
    let vocab = tok.vocab_size();
    let mut tokens: Vec<u32> = Vec::new();
    for text in &corpus {
        tokens.extend(tok.encode(text));
    }
    if tokens.is_empty() {
        return Err(CliError::other("korpus menghasilkan 0 token".to_string()));
    }
    let path = Path::new(&args.out);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::other(format!("buat dir {}: {e}", parent.display())))?;
        }
    }
    vir_data::write_shard(path, &tokens)
        .map_err(|e| CliError::other(format!("tulis shard: {e}")))?;
    println!(
        "make-shard: out={} docs={} tokens={} vocab={}",
        args.out,
        corpus.len(),
        tokens.len(),
        vocab
    );
    Ok(())
}
