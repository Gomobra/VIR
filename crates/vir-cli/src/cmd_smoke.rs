//! `vir smoke` — uji asap tokenizer dan info build.

use clap::Args;
use vir_common::build_info;
use vir_data::synth_arithmetic;
use vir_text::BpeTokenizer;

use crate::CliError;

#[derive(Args)]
pub(crate) struct Args {
    /// Keluarkan hasil sebagai JSON
    #[arg(long)]
    pub json: bool,
}

pub(crate) fn run(args: &Args) -> Result<(), CliError> {
    let texts: Vec<String> = synth_arithmetic(50).into_iter().collect();
    let mut tok = BpeTokenizer::new();
    tok.train(&texts, 280);
    let sample = "1 + 2 = 3";
    let decoded = tok.decode(&tok.encode(sample));
    let ok = decoded == sample;
    let vocab_size = tok.vocab_size();
    if args.json {
        let payload = serde_json::json!({
            "status": if ok { "ok" } else { "fail" },
            "vocab_size": vocab_size,
            "roundtrip_ok": ok,
            "build": build_info(),
        });
        println!("{payload}");
    } else {
        println!("smoke: vocab_size={vocab_size} roundtrip_ok={ok}");
    }
    if !ok {
        return Err(CliError::other("roundtrip tokenizer gagal"));
    }
    Ok(())
}